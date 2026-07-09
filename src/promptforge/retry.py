"""Retry decorator with exponential backoff, tailored for flaky LLM API calls."""
from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts have been used up."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"Gave up after {attempts} attempts: {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: total number of attempts (including the first).
        base_delay: delay in seconds before the second attempt; doubles
            each subsequent attempt.
        max_delay: cap on the delay between attempts.
        jitter: if True, randomize each delay by +/-50% to avoid
            thundering-herd retries against a rate-limited API.
        exceptions: exception types that should trigger a retry. Anything
            else propagates immediately.
        sleep_fn: injectable sleep function, primarily for testing.

    Example:
        @with_retry(max_attempts=5, exceptions=(TimeoutError, ConnectionError))
        def call_llm(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad catch
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    sleep_fn(delay)
            assert last_error is not None
            raise RetryExhaustedError(max_attempts, last_error) from last_error

        return wrapper

    return decorator
