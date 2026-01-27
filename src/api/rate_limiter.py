"""
Rate limiting utilities for Kraken API.

Implements token bucket algorithm for respecting Kraken's rate limits:
- Starter tier: 15 max counter, 0.33/sec decay
- Intermediate tier: 20 max counter, 0.5/sec decay
- Pro tier: 20 max counter, 1.0/sec decay

Usage:
    limiter = RateLimiter(max_counter=20, decay_rate=0.5)
    await limiter.acquire()  # Blocks until token available
    response = await make_api_call()
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Callable, Awaitable, Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiter configuration."""

    max_counter: int = 20  # Maximum counter value
    decay_rate: float = 0.5  # Counter decay per second
    min_delay: float = 0.1  # Minimum delay between calls (safety margin)
    burst_size: int = 1  # Maximum burst capacity


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    The Kraken API uses a "counter" system:
    - Each API call adds to a counter
    - Counter decays over time
    - If counter exceeds max, calls are rejected

    This limiter tracks the virtual counter and waits
    when necessary to avoid rate limit errors.
    """

    def __init__(
        self,
        max_counter: int = 20,
        decay_rate: float = 0.5,
        min_delay: float = 0.1,
        buffer: float = 0.8,
    ):
        """
        Initialize rate limiter.

        Args:
            max_counter: Maximum counter value before rejection
            decay_rate: Counter decay per second
            min_delay: Minimum delay between calls
            buffer: Use this fraction of capacity (0.8 = 80%)
        """
        self._max_counter = max_counter * buffer  # Apply safety buffer
        self._decay_rate = decay_rate
        self._min_delay = min_delay

        self._counter = 0.0
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> float:
        """
        Acquire permission to make an API call.

        Blocks until a token is available. Returns the actual
        wait time if any waiting was required.

        Args:
            cost: Counter cost of the API call (default 1)

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update

            # Apply decay
            self._counter = max(
                0.0,
                self._counter - (elapsed * self._decay_rate)
            )
            self._last_update = now

            # Check if we need to wait
            wait_time = 0.0
            if self._counter + cost > self._max_counter:
                # Calculate wait time for counter to decay enough
                excess = (self._counter + cost) - self._max_counter
                wait_time = excess / self._decay_rate

                logger.debug(
                    f"Rate limiting: counter={self._counter:.2f}, "
                    f"cost={cost}, waiting {wait_time:.2f}s"
                )

                await asyncio.sleep(wait_time)

                # Update after waiting
                self._counter = max(
                    0.0,
                    self._counter - (wait_time * self._decay_rate)
                )
                self._last_update = time.monotonic()

            # Add cost to counter
            self._counter += cost

            # Enforce minimum delay
            if self._min_delay > 0:
                await asyncio.sleep(self._min_delay)

            return wait_time

    @property
    def current_counter(self) -> float:
        """Get current counter value (for monitoring)."""
        elapsed = time.monotonic() - self._last_update
        return max(0.0, self._counter - (elapsed * self._decay_rate))

    @property
    def available_capacity(self) -> float:
        """Get remaining capacity before rate limiting kicks in."""
        return self._max_counter - self.current_counter

    def reset(self) -> None:
        """Reset the rate limiter (e.g., after long pause)."""
        self._counter = 0.0
        self._last_update = time.monotonic()


class SyncRateLimiter:
    """
    Synchronous rate limiter for non-async code.

    Same algorithm as RateLimiter but uses time.sleep()
    instead of asyncio.sleep().
    """

    def __init__(
        self,
        max_counter: int = 20,
        decay_rate: float = 0.5,
        min_delay: float = 0.1,
        buffer: float = 0.8,
    ):
        """
        Initialize synchronous rate limiter.

        Args:
            max_counter: Maximum counter value before rejection
            decay_rate: Counter decay per second
            min_delay: Minimum delay between calls
            buffer: Use this fraction of capacity (0.8 = 80%)
        """
        self._max_counter = max_counter * buffer
        self._decay_rate = decay_rate
        self._min_delay = min_delay

        self._counter = 0.0
        self._last_update = time.monotonic()
        # Note: This is not thread-safe. Use threading.Lock if needed.

    def acquire(self, cost: int = 1) -> float:
        """
        Acquire permission to make an API call.

        Blocks until a token is available.

        Args:
            cost: Counter cost of the API call

        Returns:
            Wait time in seconds
        """
        now = time.monotonic()
        elapsed = now - self._last_update

        # Apply decay
        self._counter = max(
            0.0,
            self._counter - (elapsed * self._decay_rate)
        )
        self._last_update = now

        # Check if we need to wait
        wait_time = 0.0
        if self._counter + cost > self._max_counter:
            excess = (self._counter + cost) - self._max_counter
            wait_time = excess / self._decay_rate

            logger.debug(
                f"Rate limiting: counter={self._counter:.2f}, "
                f"cost={cost}, waiting {wait_time:.2f}s"
            )

            time.sleep(wait_time)

            self._counter = max(
                0.0,
                self._counter - (wait_time * self._decay_rate)
            )
            self._last_update = time.monotonic()

        self._counter += cost

        if self._min_delay > 0:
            time.sleep(self._min_delay)

        return wait_time

    @property
    def current_counter(self) -> float:
        """Get current counter value."""
        elapsed = time.monotonic() - self._last_update
        return max(0.0, self._counter - (elapsed * self._decay_rate))


def rate_limited(
    limiter: Optional[RateLimiter] = None,
    cost: int = 1,
) -> Callable:
    """
    Decorator to rate-limit an async function.

    Usage:
        limiter = RateLimiter()

        @rate_limited(limiter, cost=2)
        async def api_call():
            ...

    Args:
        limiter: RateLimiter instance (creates default if None)
        cost: Counter cost for this call

    Returns:
        Decorated function
    """
    _limiter = limiter or RateLimiter()

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            await _limiter.acquire(cost)
            return await func(*args, **kwargs)
        return wrapper

    return decorator
