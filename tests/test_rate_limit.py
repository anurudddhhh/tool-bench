"""Unit tests for rate limiting and exponential backoff handling (F11b)."""

import pytest

from toolbench.config import RateLimitConfig
from toolbench.rate_limit import RateLimiter


class FakeClock:
    """Deterministic simulated clock for testing rate limiter without real-time delays."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time
        self.sleep_history: list[float] = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.sleep_history.append(seconds)
        self.current_time += seconds


def test_acquire_under_rpm_limit() -> None:
    """Verify that requests within RPM capacity do not induce throttle sleeps."""
    clock = FakeClock()
    config = RateLimitConfig(rpm=5, rpd=100)
    limiter = RateLimiter(config=config, time_fn=clock.time, sleep_fn=clock.sleep)

    # Make 5 requests (the limit)
    for _ in range(5):
        slept = limiter.acquire()
        assert slept == 0.0

    assert len(clock.sleep_history) == 0


def test_acquire_throttles_at_rpm_limit() -> None:
    """Verify that exceeding RPM capacity causes acquire() to pause execution."""
    clock = FakeClock()
    config = RateLimitConfig(rpm=3, rpd=100)
    limiter = RateLimiter(config=config, time_fn=clock.time, sleep_fn=clock.sleep)

    # Exhaust capacity of 3
    for _ in range(3):
        limiter.acquire()
        clock.current_time += 1.0  # requests at t=1000, 1001, 1002

    assert len(clock.sleep_history) == 0

    # 4th request at t=1003 must wait until t=1000 + 60.0 + 0.05 = 1060.05
    slept = limiter.acquire()
    assert slept > 50.0
    assert len(clock.sleep_history) == 1
    assert clock.current_time >= 1060.0


def test_rpd_limit_exceeded() -> None:
    """Verify that exceeding daily quota immediately halts execution with an error."""
    clock = FakeClock()
    config = RateLimitConfig(rpm=10, rpd=2)
    limiter = RateLimiter(config=config, time_fn=clock.time, sleep_fn=clock.sleep)

    limiter.acquire()
    limiter.acquire()

    with pytest.raises(RuntimeError) as exc_info:
        limiter.acquire()
    assert "Daily request quota exceeded" in str(exc_info.value)

    # Advance clock by 24 hours (crosses Midnight PT)
    clock.current_time += 86400.0
    # Must now succeed because daily count was reset
    slept = limiter.acquire()
    assert slept == 0.0


def test_compute_backoff_bounds() -> None:
    """Verify exponential backoff values grow and stay within jitter/cap bounds."""
    config = RateLimitConfig(backoff_factor=2.0, max_backoff=30.0)
    limiter = RateLimiter(config=config)

    # Attempt 0: base = 2.0 * 1 = 2.0 -> bounds [1.0, 2.0]
    for _ in range(20):
        val = limiter.compute_backoff(0)
        assert 1.0 <= val <= 2.0

    # Attempt 1: base = 2.0 * 2 = 4.0 -> bounds [2.0, 4.0]
    for _ in range(20):
        val = limiter.compute_backoff(1)
        assert 2.0 <= val <= 4.0

    # Large attempt: capped at max_backoff (30.0) -> bounds [15.0, 30.0]
    for _ in range(20):
        val = limiter.compute_backoff(10)
        assert 15.0 <= val <= 30.0


def test_default_is_retryable() -> None:
    """Verify classification of transient vs fatal exceptions."""
    assert RateLimiter.default_is_retryable(Exception("HTTP 429 Too Many Requests"))
    assert RateLimiter.default_is_retryable(Exception("503 Service Unavailable"))
    assert RateLimiter.default_is_retryable(Exception("Resource_Exhausted: quota exceeded"))

    # Non-retryable
    assert not RateLimiter.default_is_retryable(ValueError("Invalid argument"))
    assert not RateLimiter.default_is_retryable(TypeError("Expected dict"))
    assert not RateLimiter.default_is_retryable(Exception("401 Unauthorized"))


def test_execute_with_retry_success() -> None:
    """Verify standard execution without exceptions."""
    clock = FakeClock()
    limiter = RateLimiter(time_fn=clock.time, sleep_fn=clock.sleep)

    result = limiter.execute_with_retry(lambda: "success_result")
    assert result == "success_result"
    assert len(clock.sleep_history) == 0


def test_execute_with_retry_recovers_after_transient_failures() -> None:
    """Verify retry recovers after transient 429s and returns value."""
    clock = FakeClock()
    limiter = RateLimiter(time_fn=clock.time, sleep_fn=clock.sleep)

    attempts = 0
    retry_log: list[tuple[int, float]] = []

    def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("HTTP 429 Rate Limit Exceeded")
        return "recovered"

    def on_retry(att: int, exc: Exception, wait: float) -> None:
        retry_log.append((att, wait))

    result = limiter.execute_with_retry(flaky_call, on_retry=on_retry)
    assert result == "recovered"
    assert attempts == 3
    assert len(retry_log) == 2  # Retried twice before succeeding
    assert len(clock.sleep_history) == 2


def test_execute_with_retry_exhaustion_raises() -> None:
    """Verify that failing past max_retries raises the exception."""
    clock = FakeClock()
    config = RateLimitConfig(max_retries=3)
    limiter = RateLimiter(config=config, time_fn=clock.time, sleep_fn=clock.sleep)

    def always_fails() -> None:
        raise RuntimeError("HTTP 429 Rate Limit Exceeded")

    with pytest.raises(RuntimeError) as exc_info:
        limiter.execute_with_retry(always_fails)

    assert "HTTP 429 Rate Limit Exceeded" in str(exc_info.value)
    # Slept for attempts 0, 1, 2 (3 retries total)
    assert len(clock.sleep_history) == 3


def test_execute_with_retry_non_retryable_raises_immediately() -> None:
    """Verify non-retryable exceptions abort immediately without retries."""
    clock = FakeClock()
    limiter = RateLimiter(time_fn=clock.time, sleep_fn=clock.sleep)

    def fatal_error() -> None:
        raise ValueError("Non-retryable parameter error")

    with pytest.raises(ValueError):
        limiter.execute_with_retry(fatal_error)

    # Zero retries / zero sleeps
    assert len(clock.sleep_history) == 0
