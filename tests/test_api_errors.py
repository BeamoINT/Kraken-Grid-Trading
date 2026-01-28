"""
Tests for Kraken API error handling.

Tests cover:
- Error classification
- Retry logic
- Backoff calculation
- Circuit breaker
"""

import time
import pytest
from unittest.mock import Mock, patch

from src.api.kraken_errors import (
    KrakenAPIError,
    RateLimitError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
    ErrorCategory,
    ErrorSeverity,
    RetryStrategy,
    RetryConfig,
    calculate_backoff,
    with_retry,
    with_async_retry,
    CircuitBreaker,
    ErrorAggregator,
    classify_and_raise,
)


class TestKrakenAPIError:
    """Tests for KrakenAPIError base class."""

    def test_create_from_single_error(self):
        """Test creating error from single string."""
        error = KrakenAPIError("EAPI:Rate limit exceeded")
        assert "EAPI:Rate limit exceeded" in error.errors
        assert error.error_info is not None

    def test_create_from_error_list(self):
        """Test creating error from list of strings."""
        error = KrakenAPIError(["EAPI:Invalid key", "EGeneral:Invalid arguments"])
        assert len(error.errors) == 2

    def test_rate_limit_classification(self):
        """Test rate limit error is classified correctly."""
        error = KrakenAPIError(["EAPI:Rate limit exceeded"])
        assert error.error_info.category == ErrorCategory.RATE_LIMIT
        assert error.error_info.severity == ErrorSeverity.MEDIUM
        assert error.error_info.retry_strategy == RetryStrategy.RATE_LIMIT_WAIT
        assert error.is_recoverable

    def test_auth_error_classification(self):
        """Test authentication error is classified correctly."""
        error = KrakenAPIError(["EAPI:Invalid key"])
        assert error.error_info.category == ErrorCategory.AUTHENTICATION
        assert error.error_info.severity == ErrorSeverity.CRITICAL
        assert not error.is_recoverable
        assert not error.should_retry

    def test_insufficient_funds_classification(self):
        """Test insufficient funds error is classified correctly."""
        error = KrakenAPIError(["EOrder:Insufficient funds"])
        assert error.error_info.category == ErrorCategory.INSUFFICIENT_FUNDS
        assert not error.should_retry

    def test_unknown_error_classification(self):
        """Test unknown error gets default classification."""
        error = KrakenAPIError(["ESomething:Unknown error type"])
        assert error.error_info.category == ErrorCategory.UNKNOWN
        assert error.error_info.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF


class TestClassifyAndRaise:
    """Tests for classify_and_raise function."""

    def test_raises_rate_limit_error(self):
        """Test raises RateLimitError for rate limit."""
        with pytest.raises(RateLimitError):
            classify_and_raise(["EAPI:Rate limit exceeded"])

    def test_raises_auth_error(self):
        """Test raises AuthenticationError for auth issues."""
        with pytest.raises(AuthenticationError):
            classify_and_raise(["EAPI:Invalid key"])

    def test_raises_insufficient_funds(self):
        """Test raises InsufficientFundsError."""
        with pytest.raises(InsufficientFundsError):
            classify_and_raise(["EOrder:Insufficient funds"])

    def test_raises_order_error(self):
        """Test raises OrderError for order issues."""
        with pytest.raises(OrderError):
            classify_and_raise(["EOrder:Unknown order"])

    def test_raises_generic_for_unknown(self):
        """Test raises generic KrakenAPIError for unknown."""
        with pytest.raises(KrakenAPIError):
            classify_and_raise(["EUnknown:Something"])


class TestCalculateBackoff:
    """Tests for backoff calculation."""

    def test_no_retry_returns_zero(self):
        """Test NO_RETRY strategy returns 0."""
        config = RetryConfig()
        delay = calculate_backoff(0, RetryStrategy.NO_RETRY, config)
        assert delay == 0.0

    def test_immediate_returns_small_delay(self):
        """Test IMMEDIATE returns small delay."""
        config = RetryConfig()
        delay = calculate_backoff(0, RetryStrategy.IMMEDIATE, config)
        assert delay == 0.1

    def test_linear_backoff(self):
        """Test LINEAR_BACKOFF returns constant delay."""
        config = RetryConfig(base_delay=2.0, jitter=False)
        delay1 = calculate_backoff(0, RetryStrategy.LINEAR_BACKOFF, config)
        delay2 = calculate_backoff(5, RetryStrategy.LINEAR_BACKOFF, config)
        assert delay1 == delay2 == 2.0

    def test_exponential_backoff(self):
        """Test EXPONENTIAL_BACKOFF increases exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        delay0 = calculate_backoff(0, RetryStrategy.EXPONENTIAL_BACKOFF, config)
        delay1 = calculate_backoff(1, RetryStrategy.EXPONENTIAL_BACKOFF, config)
        delay2 = calculate_backoff(2, RetryStrategy.EXPONENTIAL_BACKOFF, config)

        assert delay0 == 1.0   # 1.0 * 2^0
        assert delay1 == 2.0   # 1.0 * 2^1
        assert delay2 == 4.0   # 1.0 * 2^2

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=5.0, jitter=False)
        delay = calculate_backoff(10, RetryStrategy.EXPONENTIAL_BACKOFF, config)
        assert delay == 5.0

    def test_rate_limit_uses_retry_after(self):
        """Test RATE_LIMIT_WAIT uses retry_after when provided."""
        config = RetryConfig(base_delay=1.0, jitter=False)
        delay = calculate_backoff(
            0,
            RetryStrategy.RATE_LIMIT_WAIT,
            config,
            retry_after=10.0,
        )
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds variation to delay."""
        config = RetryConfig(base_delay=1.0, jitter=True)

        delays = [
            calculate_backoff(0, RetryStrategy.LINEAR_BACKOFF, config)
            for _ in range(100)
        ]

        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # But should all be between base and base + 25%
        assert all(1.0 <= d <= 1.25 for d in delays)


class TestWithRetry:
    """Tests for synchronous retry decorator."""

    def test_success_on_first_try(self):
        """Test function succeeds without retry."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_recoverable_error(self):
        """Test function retries on recoverable error."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError(["EAPI:Rate limit exceeded"])
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_unrecoverable_error(self):
        """Test function does not retry unrecoverable errors."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        def always_fail_auth():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError(["EAPI:Invalid key"])

        with pytest.raises(AuthenticationError):
            always_fail_auth()

        assert call_count == 1  # No retry

    def test_max_retries_exhausted(self):
        """Test error raised after max retries."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError(["EAPI:Rate limit exceeded"])

        with pytest.raises(RateLimitError):
            always_fail()

        assert call_count == 4  # Initial + 3 retries


class TestWithAsyncRetry:
    """Tests for async retry decorator."""

    @pytest.mark.asyncio
    async def test_async_success_on_first_try(self):
        """Test async function succeeds without retry."""
        call_count = 0

        @with_async_retry(RetryConfig(max_retries=3))
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_recoverable_error(self):
        """Test async function retries on recoverable error."""
        call_count = 0

        @with_async_retry(RetryConfig(max_retries=3, base_delay=0.01))
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError(["EAPI:Rate limit exceeded"])
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert not breaker.is_open

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        assert not breaker.is_open

        breaker.record_failure()
        assert not breaker.is_open

        breaker.record_failure()
        assert breaker.is_open

    def test_success_does_not_increment_failures(self):
        """Test success doesn't affect failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        assert not breaker.is_open

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after reset timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.1,
        )

        breaker.record_failure()
        assert breaker.is_open

        time.sleep(0.15)
        assert not breaker.is_open  # Now half-open

    def test_closes_after_successful_half_open_calls(self):
        """Test circuit closes after successful half-open calls."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.01,
            half_open_max_calls=2,
        )

        breaker.record_failure()
        time.sleep(0.02)

        # Half-open - successes should close it
        breaker.record_success()
        breaker.record_success()

        assert breaker._state == "closed"

    def test_reopens_on_half_open_failure(self):
        """Test circuit reopens on failure during half-open."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.01,
        )

        breaker.record_failure()
        time.sleep(0.02)

        # Check is_open to trigger half-open transition
        assert not breaker.is_open

        # Failure during half-open reopens
        breaker.record_failure()
        assert breaker._state == "open"

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker(failure_threshold=1)

        breaker.record_failure()
        assert breaker.is_open

        breaker.reset()
        assert not breaker.is_open


class TestErrorAggregator:
    """Tests for ErrorAggregator class."""

    def test_record_error(self):
        """Test recording errors."""
        agg = ErrorAggregator(window_size=100)
        error = RateLimitError(["EAPI:Rate limit exceeded"])

        agg.record_error(error)
        stats = agg.get_stats(time_window=60.0)

        assert stats["total"] == 1
        assert stats["by_category"]["rate_limit"] == 1

    def test_window_trimming(self):
        """Test that old errors are trimmed."""
        agg = ErrorAggregator(window_size=5)

        for _ in range(10):
            agg.record_error(RateLimitError(["EAPI:Rate limit exceeded"]))

        assert len(agg._errors) == 5

    def test_get_stats_time_window(self):
        """Test stats respect time window."""
        agg = ErrorAggregator()

        # Record old error
        error = RateLimitError(["EAPI:Rate limit exceeded"])
        agg.record_error(error)
        agg._errors[-1]["timestamp"] = time.time() - 1000  # Old

        # Record new error
        agg.record_error(error)

        # Only new error should appear in stats
        stats = agg.get_stats(time_window=60.0)
        assert stats["total"] == 1

    def test_should_alert_high_rate(self):
        """Test alert triggers on high error rate."""
        agg = ErrorAggregator()

        # Add many errors quickly
        for _ in range(15):
            agg.record_error(RateLimitError(["EAPI:Rate limit exceeded"]))

        assert agg.should_alert()

    def test_should_alert_critical_error(self):
        """Test alert triggers on critical error."""
        agg = ErrorAggregator()

        agg.record_error(AuthenticationError(["EAPI:Invalid key"]))
        assert agg.should_alert()

    def test_no_alert_for_normal_operation(self):
        """Test no alert for normal error rates."""
        agg = ErrorAggregator()

        agg.record_error(RateLimitError(["EAPI:Rate limit exceeded"]))
        assert not agg.should_alert()
