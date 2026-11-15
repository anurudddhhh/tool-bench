"""Rate limiting and exponential backoff handling for LLM API calls (F11b).

Provides proactive sliding-window rate limiting (RPM/TPM/RPD) and reactive
exponential backoff with jitter on HTTP 429 and transient provider errors.
Decoupled from schema-validation retry logic.
"""

from __future__ import annotations

import collections
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from toolbench.config import RateLimitConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Sliding-window rate limiter with exponential backoff on transient errors."""

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the rate limiter with configuration.

        Args:
            config: RateLimitConfig instance (defaults to default RateLimitConfig).
            time_fn: Clock function returning seconds (injected for testing).
            sleep_fn: Sleep function accepting seconds (injected for testing).
        """
        self.config = config or RateLimitConfig()
        self._time = time_fn
        self._sleep = sleep_fn

        # Sliding window of request timestamps in the last 60 seconds
        self._request_timestamps: collections.deque[float] = collections.deque()
        # Daily request counter
        self._daily_requests = 0
        self._day_start = self._time()

    def acquire(self) -> float:
        """Wait if necessary to comply with RPM/RPD limits, then record the call.

        Returns:
            Duration slept in seconds (0.0 if no throttle wait was needed).
        """
        now = self._time()

        # Reset daily window if 24 hours have elapsed
        if now - self._day_start >= 86400:
            self._daily_requests = 0
            self._day_start = now

        # Check RPD limit
        if self._daily_requests >= self.config.rpd:
            raise RuntimeError(
                f"Daily request quota exceeded ({self.config.rpd} requests/day). "
                "Halting evaluation to prevent account lockout."
            )

        slept_duration = 0.0

        # Purge timestamps older than 60 seconds
        cutoff = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] <= cutoff:
            self._request_timestamps.popleft()

        # If at or exceeding RPM capacity, wait until oldest request clears the window
        if len(self._request_timestamps) >= self.config.rpm:
            oldest = self._request_timestamps[0]
            wait_time = (oldest + 60.0) - now + 0.05  # small 50ms buffer
            if wait_time > 0:
                logger.debug(
                    "Rate limit reached (%d RPM). Throttling for %.2fs",
                    self.config.rpm,
                    wait_time,
                )
                self._sleep(wait_time)
                slept_duration += wait_time
                now = self._time()
                # Re-purge after sleeping
                cutoff = now - 60.0
                while self._request_timestamps and self._request_timestamps[0] <= cutoff:
                    self._request_timestamps.popleft()

        self._request_timestamps.append(now)
        self._daily_requests += 1
        return slept_duration

    def compute_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with full jitter for a given retry attempt.

        Formula:
            max_delay = min(max_backoff, backoff_factor * (2 ** attempt))
            jittered_delay = uniform(0.5 * max_delay, max_delay)

        Args:
            attempt: Zero-based retry attempt number (0, 1, 2...).

        Returns:
            Jittered wait duration in seconds.
        """
        base_delay = self.config.backoff_factor * (2**attempt)
        max_delay = min(float(self.config.max_backoff), base_delay)
        # Jitter between 50% and 100% of the calculated interval
        return random.uniform(0.5 * max_delay, max_delay)

    def execute_with_retry(
        self,
        func: Callable[[], T],
        is_retryable: Callable[[Exception], bool] | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> T:
        """Execute a function with proactive rate limiting and reactive backoff.

        Args:
            func: Zero-argument callable making the network/API request.
            is_retryable: Predicate returning True if exception is transient (429, etc.).
            on_retry: Optional callback invoked before sleeping on retry (attempt, exc, wait).

        Returns:
            Return value of func().

        Raises:
            Exception: If max retries are exhausted or non-retryable exception occurs.
        """
        attempt = 0
        while True:
            self.acquire()
            try:
                return func()
            except Exception as exc:
                if is_retryable is not None:
                    should_retry = is_retryable(exc)
                else:
                    should_retry = self.default_is_retryable(exc)

                if not should_retry or attempt >= self.config.max_retries:
                    logger.error(
                        "API request failed on attempt %d/%d (retryable=%s): %s",
                        attempt + 1,
                        self.config.max_retries,
                        should_retry,
                        exc,
                    )
                    raise

                wait_sec = self.compute_backoff(attempt)
                logger.warning(
                    "Rate limit or transient error encountered (attempt %d/%d). "
                    "Retrying in %.2fs. Error: %s",
                    attempt + 1,
                    self.config.max_retries,
                    wait_sec,
                    exc,
                )

                if on_retry:
                    on_retry(attempt, exc, wait_sec)

                self._sleep(wait_sec)
                attempt += 1

    @staticmethod
    def default_is_retryable(exc: Exception) -> bool:
        """Determine whether an exception represents a transient rate-limit or network error."""
        error_str = str(exc).lower()
        type_name = type(exc).__name__.lower()

        # HTTP status code indicators
        transient_indicators = ["429", "503", "504", "resource_exhausted", "quota"]
        if any(code in error_str for code in transient_indicators):
            return True

        # Exception type names from google-genai or httpx
        transient_types = ["ratelimit", "exhausted", "unavailable", "timeout"]
        if any(keyword in type_name for keyword in transient_types):
            return True

        # Common attributes
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (429, 503, 504):
            return True

        return False
