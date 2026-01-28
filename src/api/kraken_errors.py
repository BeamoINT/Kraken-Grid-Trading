"""
Kraken API Error Handling.

Comprehensive error handling for Kraken API responses including:
- Error classification and severity
- Retry logic with exponential backoff
- Rate limit handling
- Error recovery strategies

Kraken Error Codes Reference:
- EAPI:Rate limit exceeded - Rate limit hit
- EAPI:Invalid key - Invalid API key
- EAPI:Invalid signature - Signature mismatch
- EAPI:Invalid nonce - Nonce too low
- EOrder:* - Order-related errors
- EGeneral:* - General errors
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Optional, List, Dict, Any, Callable, TypeVar, Union

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = auto()       # Informational, can continue
    MEDIUM = auto()    # Warning, may need attention
    HIGH = auto()      # Error, operation failed
    CRITICAL = auto()  # Critical, halt trading


class ErrorCategory(Enum):
    """Categories of Kraken API errors."""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    ORDER = "order"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    MARKET_CLOSED = "market_closed"
    INVALID_PARAMETER = "invalid_parameter"
    NETWORK = "network"
    TIMEOUT = "timeout"
    SERVER = "server"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    NO_RETRY = auto()           # Do not retry
    IMMEDIATE = auto()          # Retry immediately
    LINEAR_BACKOFF = auto()     # Constant wait time
    EXPONENTIAL_BACKOFF = auto()  # Exponential wait
    RATE_LIMIT_WAIT = auto()    # Wait for rate limit reset


@dataclass
class ErrorInfo:
    """Detailed information about an error."""
    code: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    retry_strategy: RetryStrategy
    retry_after: Optional[float] = None  # Suggested wait time
    recoverable: bool = True
    details: Optional[Dict[str, Any]] = None


# Error code mappings
ERROR_MAPPINGS: Dict[str, ErrorInfo] = {
    # Rate limiting
    "EAPI:Rate limit exceeded": ErrorInfo(
        code="EAPI:Rate limit exceeded",
        message="API rate limit exceeded",
        category=ErrorCategory.RATE_LIMIT,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.RATE_LIMIT_WAIT,
        retry_after=5.0,
        recoverable=True,
    ),

    # Authentication errors
    "EAPI:Invalid key": ErrorInfo(
        code="EAPI:Invalid key",
        message="Invalid API key",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.CRITICAL,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EAPI:Invalid signature": ErrorInfo(
        code="EAPI:Invalid signature",
        message="Invalid API signature",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.CRITICAL,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EAPI:Invalid nonce": ErrorInfo(
        code="EAPI:Invalid nonce",
        message="Nonce value is too low",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.IMMEDIATE,
        recoverable=True,
    ),

    # Permission errors
    "EAPI:Permission denied": ErrorInfo(
        code="EAPI:Permission denied",
        message="API key lacks required permissions",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.HIGH,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),

    # Order errors
    "EOrder:Insufficient funds": ErrorInfo(
        code="EOrder:Insufficient funds",
        message="Insufficient funds for order",
        category=ErrorCategory.INSUFFICIENT_FUNDS,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EOrder:Minimum order size": ErrorInfo(
        code="EOrder:Minimum order size",
        message="Order below minimum size",
        category=ErrorCategory.ORDER,
        severity=ErrorSeverity.LOW,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EOrder:Orders limit exceeded": ErrorInfo(
        code="EOrder:Orders limit exceeded",
        message="Too many open orders",
        category=ErrorCategory.ORDER,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.LINEAR_BACKOFF,
        retry_after=60.0,
        recoverable=True,
    ),
    "EOrder:Rate limit exceeded": ErrorInfo(
        code="EOrder:Rate limit exceeded",
        message="Order rate limit exceeded",
        category=ErrorCategory.RATE_LIMIT,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retry_after=1.0,
        recoverable=True,
    ),
    "EOrder:Unknown order": ErrorInfo(
        code="EOrder:Unknown order",
        message="Order not found",
        category=ErrorCategory.ORDER,
        severity=ErrorSeverity.LOW,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EOrder:Trading agreement required": ErrorInfo(
        code="EOrder:Trading agreement required",
        message="Must accept trading agreement",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.HIGH,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),

    # General errors
    "EGeneral:Invalid arguments": ErrorInfo(
        code="EGeneral:Invalid arguments",
        message="Invalid request parameters",
        category=ErrorCategory.INVALID_PARAMETER,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=False,
    ),
    "EGeneral:Internal error": ErrorInfo(
        code="EGeneral:Internal error",
        message="Kraken internal server error",
        category=ErrorCategory.SERVER,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retry_after=5.0,
        recoverable=True,
    ),
    "EGeneral:Temporary lockout": ErrorInfo(
        code="EGeneral:Temporary lockout",
        message="Account temporarily locked",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.HIGH,
        retry_strategy=RetryStrategy.LINEAR_BACKOFF,
        retry_after=900.0,  # 15 minutes
        recoverable=True,
    ),
    "EService:Unavailable": ErrorInfo(
        code="EService:Unavailable",
        message="Kraken service unavailable",
        category=ErrorCategory.SERVER,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retry_after=30.0,
        recoverable=True,
    ),
    "EService:Market in cancel_only mode": ErrorInfo(
        code="EService:Market in cancel_only mode",
        message="Market is in cancel-only mode",
        category=ErrorCategory.MARKET_CLOSED,
        severity=ErrorSeverity.HIGH,
        retry_strategy=RetryStrategy.LINEAR_BACKOFF,
        retry_after=300.0,
        recoverable=True,
    ),
    "EService:Market in post_only mode": ErrorInfo(
        code="EService:Market in post_only mode",
        message="Market is in post-only mode",
        category=ErrorCategory.MARKET_CLOSED,
        severity=ErrorSeverity.MEDIUM,
        retry_strategy=RetryStrategy.NO_RETRY,
        recoverable=True,
    ),
}


class KrakenAPIError(Exception):
    """Base exception for Kraken API errors."""

    def __init__(
        self,
        errors: Union[str, List[str]],
        response: Optional[Dict[str, Any]] = None,
        error_info: Optional[ErrorInfo] = None,
    ):
        if isinstance(errors, str):
            errors = [errors]

        self.errors = errors
        self.response = response
        self.error_info = error_info or self._classify_error(errors[0] if errors else "Unknown")

        super().__init__(f"Kraken API error: {errors}")

    def _classify_error(self, error_code: str) -> ErrorInfo:
        """Classify an error code into ErrorInfo."""
        # Check for exact match
        if error_code in ERROR_MAPPINGS:
            return ERROR_MAPPINGS[error_code]

        # Check for prefix match
        for code, info in ERROR_MAPPINGS.items():
            if error_code.startswith(code.split(":")[0] + ":"):
                return ErrorInfo(
                    code=error_code,
                    message=error_code,
                    category=info.category,
                    severity=info.severity,
                    retry_strategy=info.retry_strategy,
                    retry_after=info.retry_after,
                    recoverable=info.recoverable,
                )

        # Default unknown error
        return ErrorInfo(
            code=error_code,
            message=f"Unknown error: {error_code}",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            retry_after=5.0,
            recoverable=True,
        )

    @property
    def is_recoverable(self) -> bool:
        """Check if error is recoverable."""
        return self.error_info.recoverable

    @property
    def should_retry(self) -> bool:
        """Check if request should be retried."""
        return self.error_info.retry_strategy != RetryStrategy.NO_RETRY

    @property
    def category(self) -> ErrorCategory:
        """Get error category."""
        return self.error_info.category


class RateLimitError(KrakenAPIError):
    """Specific exception for rate limit errors."""
    pass


class AuthenticationError(KrakenAPIError):
    """Specific exception for authentication errors."""
    pass


class InsufficientFundsError(KrakenAPIError):
    """Specific exception for insufficient funds."""
    pass


class OrderError(KrakenAPIError):
    """Specific exception for order-related errors."""
    pass


class NetworkError(KrakenAPIError):
    """Specific exception for network errors."""
    pass


def classify_and_raise(errors: List[str], response: Optional[Dict] = None) -> None:
    """
    Classify errors and raise appropriate exception type.

    Args:
        errors: List of error strings from API
        response: Full API response

    Raises:
        Appropriate KrakenAPIError subclass
    """
    if not errors:
        return

    primary_error = errors[0]
    error_info = ERROR_MAPPINGS.get(primary_error)

    if error_info:
        if error_info.category == ErrorCategory.RATE_LIMIT:
            raise RateLimitError(errors, response, error_info)
        elif error_info.category == ErrorCategory.AUTHENTICATION:
            raise AuthenticationError(errors, response, error_info)
        elif error_info.category == ErrorCategory.INSUFFICIENT_FUNDS:
            raise InsufficientFundsError(errors, response, error_info)
        elif error_info.category == ErrorCategory.ORDER:
            raise OrderError(errors, response, error_info)

    raise KrakenAPIError(errors, response)


T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


def calculate_backoff(
    attempt: int,
    strategy: RetryStrategy,
    config: RetryConfig,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate backoff time for retry.

    Args:
        attempt: Current attempt number (0-indexed)
        strategy: Retry strategy to use
        config: Retry configuration
        retry_after: Explicit wait time from error

    Returns:
        Seconds to wait before retry
    """
    import random

    if strategy == RetryStrategy.NO_RETRY:
        return 0.0

    if strategy == RetryStrategy.RATE_LIMIT_WAIT and retry_after:
        delay = retry_after
    elif strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.base_delay
    elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.base_delay * (config.exponential_base ** attempt)
    elif strategy == RetryStrategy.IMMEDIATE:
        delay = 0.1
    else:
        delay = config.base_delay

    # Apply max delay cap
    delay = min(delay, config.max_delay)

    # Add jitter (up to 25% of delay)
    if config.jitter and delay > 0:
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable:
    """
    Decorator for sync functions with automatic retry on recoverable errors.

    Usage:
        @with_retry(RetryConfig(max_retries=5))
        def api_call():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except KrakenAPIError as e:
                    last_error = e

                    if not e.should_retry or attempt >= config.max_retries:
                        raise

                    backoff = calculate_backoff(
                        attempt,
                        e.error_info.retry_strategy,
                        config,
                        e.error_info.retry_after,
                    )

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {e.error_info.code}, "
                        f"waiting {backoff:.1f}s"
                    )
                    time.sleep(backoff)

                except Exception as e:
                    # Network/timeout errors
                    last_error = e

                    if attempt >= config.max_retries:
                        raise

                    backoff = calculate_backoff(
                        attempt,
                        RetryStrategy.EXPONENTIAL_BACKOFF,
                        config,
                    )

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {type(e).__name__}: {e}, "
                        f"waiting {backoff:.1f}s"
                    )
                    time.sleep(backoff)

            raise last_error or Exception("Max retries exceeded")

        return wrapper
    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
) -> Callable:
    """
    Decorator for async functions with automatic retry on recoverable errors.

    Usage:
        @with_async_retry(RetryConfig(max_retries=5))
        async def api_call():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except KrakenAPIError as e:
                    last_error = e

                    if not e.should_retry or attempt >= config.max_retries:
                        raise

                    backoff = calculate_backoff(
                        attempt,
                        e.error_info.retry_strategy,
                        config,
                        e.error_info.retry_after,
                    )

                    logger.warning(
                        f"Async retry {attempt + 1}/{config.max_retries} for {e.error_info.code}, "
                        f"waiting {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)

                except Exception as e:
                    last_error = e

                    if attempt >= config.max_retries:
                        raise

                    backoff = calculate_backoff(
                        attempt,
                        RetryStrategy.EXPONENTIAL_BACKOFF,
                        config,
                    )

                    logger.warning(
                        f"Async retry {attempt + 1}/{config.max_retries} for {type(e).__name__}: {e}, "
                        f"waiting {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)

            raise last_error or Exception("Max retries exceeded")

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            reset_timeout: Seconds before attempting reset
            half_open_max_calls: Max test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "closed"
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._state == "open":
            # Check if we should transition to half-open
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == "half_open":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "half_open":
            self._state = "open"
            logger.warning("Circuit breaker re-opened after half-open failure")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._failure_count = 0
        self._state = "closed"
        self._half_open_calls = 0


class ErrorAggregator:
    """
    Aggregates errors for monitoring and alerting.

    Tracks error patterns to detect systemic issues.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize error aggregator.

        Args:
            window_size: Number of recent errors to track
        """
        self.window_size = window_size
        self._errors: List[Dict[str, Any]] = []

    def record_error(self, error: KrakenAPIError) -> None:
        """Record an error occurrence."""
        self._errors.append({
            "timestamp": time.time(),
            "code": error.error_info.code,
            "category": error.error_info.category.value,
            "severity": error.error_info.severity.name,
        })

        # Trim to window size
        if len(self._errors) > self.window_size:
            self._errors = self._errors[-self.window_size:]

    def get_stats(self, time_window: float = 300.0) -> Dict[str, Any]:
        """
        Get error statistics for recent window.

        Args:
            time_window: Seconds to look back (default 5 min)

        Returns:
            Dict with error statistics
        """
        cutoff = time.time() - time_window
        recent = [e for e in self._errors if e["timestamp"] >= cutoff]

        if not recent:
            return {"total": 0, "by_category": {}, "by_severity": {}}

        # Count by category and severity
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for error in recent:
            cat = error["category"]
            sev = error["severity"]
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total": len(recent),
            "by_category": by_category,
            "by_severity": by_severity,
            "rate_per_minute": len(recent) / (time_window / 60),
        }

    def should_alert(self) -> bool:
        """Check if error rate warrants an alert."""
        stats = self.get_stats(time_window=60.0)

        # Alert if more than 10 errors per minute
        if stats["rate_per_minute"] > 10:
            return True

        # Alert if any critical errors
        if stats["by_severity"].get("CRITICAL", 0) > 0:
            return True

        # Alert if many auth errors (possible key issue)
        if stats["by_category"].get("authentication", 0) > 3:
            return True

        return False
