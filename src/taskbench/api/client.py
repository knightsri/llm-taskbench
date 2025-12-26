"""
OpenRouter API client for LLM completions.

This module provides an async HTTP client for interacting with the OpenRouter API,
supporting both standard and JSON mode completions.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from taskbench.api.retry import retry_with_backoff, RateLimiter, with_rate_limit
from taskbench.core.models import CompletionResponse

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class RateLimitError(OpenRouterError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(OpenRouterError):
    """Raised when API authentication fails."""
    pass


class BadRequestError(OpenRouterError):
    """Raised when the request is malformed."""
    pass


class OpenRouterClient:
    """
    Async HTTP client for OpenRouter API.

    Handles authentication, request formatting, response parsing,
    and error handling for OpenRouter API calls.

    Example:
        ```python
        async with OpenRouterClient(api_key="your-key") as client:
            response = await client.complete(
                model="anthropic/claude-sonnet-4.5",
                prompt="Explain Python lists"
            )
            print(response.content)
        ```
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = None,
        limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API (default: official endpoint)
            timeout: Request timeout in seconds (default: from TASKBENCH_TIMEOUT env or 120s)
            limiter: Optional RateLimiter to guard outbound requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout or float(os.getenv("TASKBENCH_TIMEOUT", "120.0"))
        self.limiter = limiter

        # Initialize async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_default_headers()
        )

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/llm-taskbench",
            "X-Title": "LLM TaskBench",
            "Content-Type": "application/json"
        }

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _post(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        """
        Internal POST with retry and optional rate limiting.
        """
        if self.limiter:
            await self.limiter.acquire()
        return await self.client.post(path, json=payload)

    async def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Send a completion request to OpenRouter with usage tracking by default.

        Args:
            model: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate (default: 1000)
            temperature: Sampling temperature 0-1 (default: 0.7)
            json_mode: If True, request structured JSON response
            **kwargs: Additional parameters to pass to the API

        Returns:
            CompletionResponse with the model's output and metadata
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},  # inline usage accounting
            **kwargs
        }

        if json_mode:
            payload.setdefault("response_format", {"type": "json_object"})

        # Measure latency
        start_time = time.perf_counter()

        try:
            response = await self._post(f"{self.base_url}/chat/completions", payload)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid API key. Please check your OPENROUTER_API_KEY."
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    "Rate limit exceeded. Please retry after a delay."
                )
            elif response.status_code == 400:
                error_detail = response.json().get("error", {}).get("message", "Unknown error")
                raise BadRequestError(
                    f"Bad request: {error_detail}"
                )
            elif response.status_code >= 500:
                raise OpenRouterError(
                    f"Server error ({response.status_code}): {response.text}"
                )
            elif response.status_code != 200:
                raise OpenRouterError(
                    f"Unexpected error ({response.status_code}): {response.text}"
                )

            data = response.json()

            # Extract content
            content = data["choices"][0]["message"]["content"]

            # Extract token usage
            usage = data.get("usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            cost_inline = usage.get("cost")

            completion = CompletionResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                generation_id=data.get("id"),
                billed_cost_usd=cost_inline,
            )

            logger.info(
                f"Completion successful: model={model}, "
                f"tokens={total_tokens}, latency={latency_ms:.2f}ms"
            )

            return completion

        except httpx.TimeoutException as e:
            raise OpenRouterError(f"Request timeout after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise OpenRouterError(f"HTTP error: {str(e)}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise OpenRouterError(f"Failed to parse API response: {str(e)}") from e

    async def complete_with_json(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Request a completion in JSON mode.

        This method adds JSON formatting instructions to the prompt and
        validates that the response is valid JSON.

        Args:
            model: Model identifier
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with JSON content

        Raises:
            OpenRouterError: If response is not valid JSON
        """
        # Enhance prompt with JSON instructions
        json_prompt = (
            f"{prompt}\n\nRespond ONLY with valid JSON. "
            f"Do not include any explanatory text outside the JSON structure."
        )

        response = await self.complete(
            model=model,
            prompt=json_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
            **kwargs
        )

        # Validate JSON
        try:
            content = response.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()
            json.loads(content)
            response.content = content

            logger.info(f"JSON mode completion successful: model={model}")
            return response

        except json.JSONDecodeError as e:
            logger.error(f"JSON validation failed: {str(e)}")
            logger.error(f"Response content: {response.content[:500]}")
            raise OpenRouterError(
                f"Model did not return valid JSON: {str(e)}\n"
                f"Content: {response.content[:200]}..."
            ) from e

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.info("OpenRouter client closed")

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def get_generation_stats(self, generation_id: str) -> Dict[str, Any]:
        """
        Fetch detailed cost/token stats for a given generation ID.
        """
        response = await self.client.get(
            f"{self.base_url}/generation",
            params={"id": generation_id}
        )

        if response.status_code == 404:
            raise OpenRouterError(f"Generation ID not found: {generation_id}")
        if response.status_code != 200:
            raise OpenRouterError(
                f"Failed to fetch generation stats ({response.status_code}): {response.text}"
            )

        return response.json()

    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
