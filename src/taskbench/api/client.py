"""
OpenRouter API client for LLM TaskBench.

This module provides an async HTTP client for interacting with the OpenRouter API,
including token usage tracking, latency measurement, and error handling.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from taskbench.core.models import CompletionResponse

logger = logging.getLogger(__name__)


class OpenRouterAPIError(Exception):
    """Base exception for OpenRouter API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        """
        Initialize OpenRouter API error.

        Args:
            message: Error message
            status_code: HTTP status code if available
            response_body: Response body if available
        """
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class OpenRouterClient:
    """
    Async HTTP client for OpenRouter API.

    This client handles API communication with proper error handling,
    token usage tracking, and latency measurement.

    Example:
        >>> async with OpenRouterClient() as client:
        ...     response = await client.complete(
        ...         model="anthropic/claude-sonnet-4.5",
        ...         prompt="Explain quantum computing"
        ...     )
        ...     print(f"Response: {response.content}")
        ...     print(f"Tokens used: {response.total_tokens}")
        ...     print(f"Latency: {response.latency_ms}ms")
    """

    API_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 120.0  # 2 minutes

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        app_name: str = "LLM-TaskBench",
        site_url: str = "https://github.com/yourusername/llm-taskbench"
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var
            timeout: Request timeout in seconds (default: 120)
            app_name: Application name for OpenRouter headers
            site_url: Site URL for OpenRouter headers

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Please provide api_key parameter "
                "or set OPENROUTER_API_KEY environment variable."
            )

        self.timeout = timeout
        self.app_name = app_name
        self.site_url = site_url
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_name,
                    "Content-Type": "application/json"
                }
            )

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("OpenRouter client closed")

    def _handle_error_response(self, response: httpx.Response) -> None:
        """
        Handle API error responses.

        Args:
            response: HTTP response object

        Raises:
            OpenRouterAPIError: For various API error conditions
        """
        status_code = response.status_code
        try:
            error_body = response.text
            error_data = response.json() if response.text else {}
            error_message = error_data.get("error", {}).get("message", error_body)
        except Exception:
            error_body = response.text
            error_message = error_body

        # Map status codes to appropriate error messages
        if status_code == 401:
            raise OpenRouterAPIError(
                "Authentication failed. Please check your API key.",
                status_code=status_code,
                response_body=error_body
            )
        elif status_code == 403:
            raise OpenRouterAPIError(
                "Access forbidden. You may not have access to this model.",
                status_code=status_code,
                response_body=error_body
            )
        elif status_code == 429:
            raise OpenRouterAPIError(
                "Rate limit exceeded. Please retry after a delay.",
                status_code=status_code,
                response_body=error_body
            )
        elif status_code == 400:
            raise OpenRouterAPIError(
                f"Bad request: {error_message}",
                status_code=status_code,
                response_body=error_body
            )
        elif status_code in (500, 502, 503, 504):
            raise OpenRouterAPIError(
                f"Server error (HTTP {status_code}). This is a transient error, please retry.",
                status_code=status_code,
                response_body=error_body
            )
        else:
            raise OpenRouterAPIError(
                f"API request failed with status {status_code}: {error_message}",
                status_code=status_code,
                response_body=error_body
            )

    def _extract_token_usage(self, response_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract token usage from API response.

        Args:
            response_data: Response JSON data

        Returns:
            Dictionary with input_tokens, output_tokens, and total_tokens
        """
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }

    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Send a completion request to OpenRouter API.

        Args:
            model: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters to pass to the API

        Returns:
            CompletionResponse object with the model's response and metadata

        Raises:
            OpenRouterAPIError: If the API request fails

        Example:
            >>> client = OpenRouterClient()
            >>> response = await client.complete(
            ...     model="anthropic/claude-sonnet-4.5",
            ...     prompt="What is Python?",
            ...     temperature=0.5
            ... )
        """
        await self._ensure_client()

        # Build request payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add any additional parameters
        payload.update(kwargs)

        # Measure latency
        start_time = time.time()

        try:
            logger.debug(f"Sending completion request to {model}")
            response = await self._client.post(
                f"{self.API_BASE_URL}/chat/completions",
                json=payload
            )

            latency_ms = (time.time() - start_time) * 1000

            # Check for errors
            if response.status_code != 200:
                self._handle_error_response(response)

            # Parse response
            response_data = response.json()

            # Extract content
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract token usage
            token_usage = self._extract_token_usage(response_data)

            logger.info(
                f"Completion successful: model={model}, tokens={token_usage['total_tokens']}, "
                f"latency={latency_ms:.0f}ms"
            )

            return CompletionResponse(
                content=content,
                model=model,
                input_tokens=token_usage["input_tokens"],
                output_tokens=token_usage["output_tokens"],
                total_tokens=token_usage["total_tokens"],
                latency_ms=latency_ms
            )

        except OpenRouterAPIError:
            raise
        except httpx.TimeoutException as e:
            raise OpenRouterAPIError(f"Request timed out after {self.timeout} seconds") from e
        except httpx.HTTPError as e:
            raise OpenRouterAPIError(f"HTTP error occurred: {e}") from e
        except Exception as e:
            raise OpenRouterAPIError(f"Unexpected error: {e}") from e

    async def complete_with_json(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Send a completion request with JSON mode enabled.

        This method requests the model to return structured JSON output.
        Note: Not all models support JSON mode.

        Args:
            model: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters to pass to the API

        Returns:
            CompletionResponse object with the model's JSON response

        Raises:
            OpenRouterAPIError: If the API request fails

        Example:
            >>> response = await client.complete_with_json(
            ...     model="openai/gpt-4o",
            ...     prompt="Return a JSON object with name and age fields"
            ... )
        """
        # Add JSON mode parameter
        kwargs["response_format"] = {"type": "json_object"}

        logger.debug(f"Requesting JSON mode completion from {model}")
        return await self.complete(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
