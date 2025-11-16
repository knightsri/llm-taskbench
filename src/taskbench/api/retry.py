"""
Retry logic and rate limiting for API calls.

This module provides decorators and utilities for handling transient errors
and rate limiting when making API requests.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Optional, Set, Type

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Tracks requests per minute and enforces rate limits by sleeping
    when the limit would be exceeded.

    Example:
        ```python
        limiter = RateLimiter(max_requests_per_minute=60)
        await limiter.acquire()  # Wait if rate limit would be exceeded
        # Make API request
        ```
    """

    def __init__(self, max_requests_per_minute: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Sleeps if making a request now would exceed the rate limit.
        """
        async with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            cutoff = now - 60.0
            self.requests = [t for t in self.requests if t > cutoff]

            # Check if we're at the limit
            if len(self.requests) >= self.max_requests:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                sleep_time = 60.0 - (now - oldest_request)

                if sleep_time > 0:
                    logger.info(
                        f"Rate limit reached ({len(self.requests)}/{self.max_requests}). "
                        f"Sleeping for {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    now = time.time()

            # Record this request
            self.requests.append(now)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[Set[Type[Exception]]] = None
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.

    Retries transient errors (5xx, timeouts, rate limits) but not
    permanent errors (4xx except 429).

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        retryable_exceptions: Set of exception types to retry (if None, uses defaults)
        non_retryable_exceptions: Set of exception types to never retry (if None, uses defaults)

    Returns:
        Decorated function with retry logic

    Example:
        ```python
        @retry_with_backoff(max_retries=3, base_delay=2.0)
        async def make_api_call():
            # ... API call code ...
        ```
    """
    # Import here to avoid circular dependencies
    from taskbench.api.client import (
        AuthenticationError,
        BadRequestError,
        OpenRouterError,
        RateLimitError,
    )

    # Default retryable exceptions (transient errors)
    if retryable_exceptions is None:
        retryable_exceptions = {
            RateLimitError,  # 429 - Rate limit
            OpenRouterError,  # Generic errors (including 5xx)
            asyncio.TimeoutError,
            ConnectionError,
        }

    # Default non-retryable exceptions (permanent errors)
    if non_retryable_exceptions is None:
        non_retryable_exceptions = {
            AuthenticationError,  # 401 - Invalid API key
            BadRequestError,  # 400 - Malformed request
            ValueError,  # Programming errors
            TypeError,
        }

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result

                except Exception as e:
                    last_exception = e

                    # Check if this is a non-retryable error
                    if any(isinstance(e, exc_type) for exc_type in non_retryable_exceptions):
                        logger.error(
                            f"Non-retryable error in {func.__name__}: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = any(isinstance(e, exc_type) for exc_type in retryable_exceptions)

                    if not is_retryable or attempt >= max_retries:
                        # Not retryable or out of retries
                        if attempt >= max_retries:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for {func.__name__}"
                            )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {type(e).__name__}: {str(e)}. "
                        f"Waiting {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def with_rate_limit(limiter: RateLimiter) -> Callable:
    """
    Decorator to enforce rate limiting on async functions.

    Args:
        limiter: RateLimiter instance to use

    Returns:
        Decorated function with rate limiting

    Example:
        ```python
        limiter = RateLimiter(max_requests_per_minute=60)

        @with_rate_limit(limiter)
        async def make_request():
            # ... API call code ...
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator
