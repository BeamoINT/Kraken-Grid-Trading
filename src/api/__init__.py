"""
Kraken API Module.

Provides clients and utilities for interacting with Kraken exchange:
- Authentication (HMAC-SHA512)
- Public REST API (historical data, ticker, etc.)
- Private REST API (balance, orders, trades)
- WebSocket (real-time data, order updates)
- Order management (grid order lifecycle)
- Error handling (retry logic, circuit breaker)

Usage:
    # Public API (no auth needed)
    from src.api import KrakenPublicClient

    client = KrakenPublicClient()
    trades, last_id = client.get_trades("XBTUSD")

    # Private API (requires credentials)
    from src.api import KrakenPrivateClient, OrderSide, OrderType

    client = KrakenPrivateClient.from_env()  # Uses KRAKEN_API_KEY, KRAKEN_API_SECRET
    balance = client.get_balance()
    result = client.add_order(
        pair="XBTUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        volume=0.001,
        price=50000,
    )

    # WebSocket (real-time data)
    from src.api import KrakenWebSocketClient

    async with KrakenWebSocketClient() as ws:
        await ws.subscribe_ticker(["BTC/USD"])
        async for msg in ws:
            print(msg)

    # Order management (grid trading)
    from src.api import OrderManager, GridOrderType

    manager = OrderManager(client)
    order = manager.create_grid_order(
        level=5,
        price=Decimal("50000"),
        side=GridOrderType.BUY,
        volume=Decimal("0.001"),
    )
    manager.submit_order(order)
"""

# Authentication
from .auth import (
    KrakenAuth,
    KrakenCredentials,
    NonceManager,
    AsyncNonceManager,
    load_credentials_from_env,
    load_credentials_from_file,
)

# Rate limiting
from .rate_limiter import (
    RateLimiter,
    SyncRateLimiter,
    RateLimitConfig,
    rate_limited,
)

# Error handling
from .kraken_errors import (
    # Base errors
    KrakenAPIError,
    RateLimitError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
    NetworkError,
    # Error info
    ErrorInfo,
    ErrorCategory,
    ErrorSeverity,
    # Retry utilities
    RetryConfig,
    RetryStrategy,
    with_retry,
    with_async_retry,
    calculate_backoff,
    # Circuit breaker
    CircuitBreaker,
    ErrorAggregator,
)

# Private REST API
from .kraken_private import (
    KrakenPrivateClient,
    # Data classes
    Balance,
    TradeBalance,
    OrderInfo,
    TradeInfo,
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
)

# Order management
from .order_manager import (
    OrderManager,
    OrderManagerConfig,
    # Order types
    GridOrder,
    GridOrderType,
    OrderState,
    # Position tracking
    GridPosition,
)

# WebSocket
from .websocket_client import (
    KrakenWebSocketClient,
    PublicWSClient,
    PrivateWSClient,
    WebSocketConfig,
    # Subscription types
    Subscription,
    SubscriptionType,
    ConnectionState,
    # Message parsers
    parse_ticker_message,
    parse_ohlc_message,
    parse_trade_message,
)


__all__ = [
    # Authentication
    "KrakenAuth",
    "KrakenCredentials",
    "NonceManager",
    "AsyncNonceManager",
    "load_credentials_from_env",
    "load_credentials_from_file",
    # Rate limiting
    "RateLimiter",
    "SyncRateLimiter",
    "RateLimitConfig",
    "rate_limited",
    # Errors
    "KrakenAPIError",
    "RateLimitError",
    "AuthenticationError",
    "InsufficientFundsError",
    "OrderError",
    "NetworkError",
    "ErrorInfo",
    "ErrorCategory",
    "ErrorSeverity",
    "RetryConfig",
    "RetryStrategy",
    "with_retry",
    "with_async_retry",
    "calculate_backoff",
    "CircuitBreaker",
    "ErrorAggregator",
    # Private client
    "KrakenPrivateClient",
    "Balance",
    "TradeBalance",
    "OrderInfo",
    "TradeInfo",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Order management
    "OrderManager",
    "OrderManagerConfig",
    "GridOrder",
    "GridOrderType",
    "OrderState",
    "GridPosition",
    # WebSocket
    "KrakenWebSocketClient",
    "PublicWSClient",
    "PrivateWSClient",
    "WebSocketConfig",
    "Subscription",
    "SubscriptionType",
    "ConnectionState",
    "parse_ticker_message",
    "parse_ohlc_message",
    "parse_trade_message",
]
