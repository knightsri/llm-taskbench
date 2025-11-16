"""
Retry logic and rate limiting for LLM TaskBench API calls.

This module provides decorators and utilities for handling transient failures
and rate limiting when calling LLM APIs.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Set, Type, TypeVar

from taskbench.api.client import OpenRouterAPIError

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_status_codes: Optional[Set[int]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds exponential backoff retry logic to async functions.

    This decorator will retry the function on transient errors (rate limits,
    server errors) but not on client errors (authentication, bad requests).

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        retryable_status_codes: Set of HTTP status codes to retry on.
            If None, defaults to {429, 500, 502, 503, 504}

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=5, initial_delay=2.0)
        ... async def call_api():
        ...     # API call that might fail
        ...     return await client.complete(model="...", prompt="...")
        ...
        >>> result = await call_api()  # Will retry on transient errors
    """
    # Default retryable status codes (transient errors only)
    if retryable_status_codes is None:
        retryable_status_codes = {429, 500, 502, 503, 504}

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    # Try to execute the function
                    result = await func(*args, **kwargs)

                    # Log success if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt + 1}/{max_retries + 1}"
                        )

                    return result

                except OpenRouterAPIError as e:
                    last_exception = e

                    # Check if this is a retryable error
                    if e.status_code not in retryable_status_codes:
                        logger.warning(
                            f"Non-retryable error (status {e.status_code}) in {func.__name__}: {e}"
                        )
                        raise

                    # Don't retry if we've exhausted attempts
                    if attempt >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                            f"Last error: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    logger.warning(
                        f"Retryable error (status {e.status_code}) in {func.__name__} "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )

                    # Wait before retrying
                    await asyncio.sleep(delay)

                except asyncio.TimeoutError as e:
                    last_exception = e

                    # Treat timeouts as retryable
                    if attempt >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__} due to timeout"
                        )
                        raise

                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    logger.warning(
                        f"Timeout in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)

                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(
                        f"Unexpected error in {func.__name__}: {e}. Not retrying."
                    )
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic failed unexpectedly in {func.__name__}")

        return wrapper
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request rates.

    This class implements a token bucket algorithm to prevent burst requests
    and maintain a steady rate of API calls.

    Example:
        >>> limiter = RateLimiter(requests_per_minute=60)
        >>> async def make_requests():
        ...     for i in range(100):
        ...         await limiter.acquire()
        ...         result = await api_call()
        ...         print(f"Request {i} completed")
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            burst_size: Maximum burst size (default: same as requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute

        # Token bucket parameters
        self.tokens = float(self.burst_size)  # Start with full bucket
        self.max_tokens = float(self.burst_size)
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second
        self.last_refill = time.time()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"RateLimiter initialized: {requests_per_minute} req/min, "
            f"burst_size={self.burst_size}"
        )

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate new tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket, waiting if necessary.

        This method will block until the requested number of tokens is available.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Example:
            >>> limiter = RateLimiter(requests_per_minute=60)
            >>> await limiter.acquire()  # Wait for 1 token
            >>> # Make API call
        """
        if tokens > self.max_tokens:
            raise ValueError(
                f"Requested tokens ({tokens}) exceeds maximum burst size ({self.max_tokens})"
            )

        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                self._refill_tokens()

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"Acquired {tokens} token(s). Remaining: {self.tokens:.2f}")
                    return

                # Calculate how long to wait for enough tokens
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate

                logger.debug(
                    f"Insufficient tokens ({self.tokens:.2f}/{tokens}). "
                    f"Waiting {wait_time:.2f}s..."
                )

                # Release lock while waiting
                await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were acquired, False otherwise

        Example:
            >>> limiter = RateLimiter(requests_per_minute=60)
            >>> if await limiter.try_acquire():
            ...     # Make API call
            ...     pass
            ... else:
            ...     # Handle rate limit
            ...     print("Rate limited, skipping request")
        """
        async with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                logger.debug(f"Acquired {tokens} token(s). Remaining: {self.tokens:.2f}")
                return True

            logger.debug(f"Insufficient tokens ({self.tokens:.2f}/{tokens})")
            return False

    def get_available_tokens(self) -> float:
        """
        Get the current number of available tokens.

        Returns:
            Number of tokens currently available

        Example:
            >>> limiter = RateLimiter(requests_per_minute=60)
            >>> print(f"Available tokens: {limiter.get_available_tokens()}")
        """
        # Update tokens first
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        current_tokens = min(self.max_tokens, self.tokens + new_tokens)
        return current_tokens

    def reset(self) -> None:
        """
        Reset the rate limiter to full capacity.

        Example:
            >>> limiter = RateLimiter(requests_per_minute=60)
            >>> # ... make some requests ...
            >>> limiter.reset()  # Reset to full capacity
        """
        self.tokens = float(self.max_tokens)
        self.last_refill = time.time()
        logger.info("RateLimiter reset to full capacity")
