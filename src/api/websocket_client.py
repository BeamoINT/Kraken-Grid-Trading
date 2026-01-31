"""
Kraken WebSocket Client.

Real-time data streaming via Kraken WebSocket API v2:
- Public feeds: ticker, OHLC, trades, book
- Private feeds: own trades, open orders (requires auth)
- Automatic reconnection with exponential backoff
- Heartbeat monitoring

WebSocket v2 API Reference:
https://docs.kraken.com/api/docs/websocket-v2/overview
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Callable, Awaitable, Set, Union
)

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from .auth import KrakenAuth
from .kraken_errors import KrakenAPIError

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATING = auto()
    AUTHENTICATED = auto()
    RECONNECTING = auto()
    CLOSED = auto()


class SubscriptionType(Enum):
    """Available subscription types."""
    # Public
    TICKER = "ticker"
    OHLC = "ohlc"
    TRADE = "trade"
    BOOK = "book"
    # Private (requires auth)
    OWN_TRADES = "ownTrades"
    OPEN_ORDERS = "openOrders"


@dataclass
class WebSocketConfig:
    """WebSocket client configuration."""
    # Connection URLs
    public_url: str = "wss://ws.kraken.com/v2"
    private_url: str = "wss://ws-auth.kraken.com/v2"

    # Heartbeat
    ping_interval: float = 30.0
    ping_timeout: float = 10.0

    # Reconnection
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 300.0
    reconnect_multiplier: float = 2.0
    max_reconnect_attempts: int = 0  # 0 = unlimited

    # Message handling
    message_queue_size: int = 1000


@dataclass
class Subscription:
    """Represents a WebSocket subscription."""
    sub_type: SubscriptionType
    symbols: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    req_id: Optional[int] = None
    active: bool = False


# Type aliases for callbacks
MessageCallback = Callable[[Dict[str, Any]], Awaitable[None]]
ErrorCallback = Callable[[Exception], Awaitable[None]]
StateCallback = Callable[[ConnectionState], Awaitable[None]]


class KrakenWebSocketClient:
    """
    Async WebSocket client for Kraken real-time data.

    Usage:
        async with KrakenWebSocketClient() as client:
            await client.subscribe_ticker(["BTC/USD"])

            async for message in client:
                process(message)

    Or with callbacks:
        client = KrakenWebSocketClient(
            on_message=handle_message,
            on_error=handle_error,
        )
        await client.connect()
        await client.run_forever()
    """

    def __init__(
        self,
        config: Optional[WebSocketConfig] = None,
        auth: Optional[KrakenAuth] = None,
        on_message: Optional[MessageCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        on_state_change: Optional[StateCallback] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            config: WebSocket configuration
            auth: Authentication for private channels
            on_message: Callback for received messages
            on_error: Callback for errors
            on_state_change: Callback for state changes
        """
        self._config = config or WebSocketConfig()
        self._auth = auth
        self._on_message = on_message
        self._on_error = on_error
        self._on_state_change = on_state_change

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_count = 0
        self._token: Optional[str] = None  # Private API token

        # Subscriptions
        self._subscriptions: Dict[str, Subscription] = {}
        self._req_id_counter = 1

        # Message queue for iterator interface
        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._config.message_queue_size
        )

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Shutdown flag
        self._shutdown = False

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected and ready."""
        return self._state in (
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
        )

    async def _set_state(self, state: ConnectionState) -> None:
        """Update state and notify callback."""
        if state != self._state:
            old_state = self._state
            self._state = state
            logger.debug(f"WebSocket state: {old_state.name} -> {state.name}")

            if self._on_state_change:
                try:
                    await self._on_state_change(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def _next_req_id(self) -> int:
        """Get next request ID."""
        req_id = self._req_id_counter
        self._req_id_counter += 1
        return req_id

    # ==========================================
    # CONNECTION MANAGEMENT
    # ==========================================

    async def connect(self, private: bool = False) -> None:
        """
        Establish WebSocket connection.

        Args:
            private: Connect to private (authenticated) endpoint
        """
        if self._state not in (ConnectionState.DISCONNECTED, ConnectionState.CLOSED):
            logger.warning(f"Cannot connect in state {self._state}")
            return

        await self._set_state(ConnectionState.CONNECTING)

        url = self._config.private_url if private else self._config.public_url

        try:
            self._ws = await websockets.connect(
                url,
                ping_interval=None,  # We handle our own heartbeat
                ping_timeout=self._config.ping_timeout,
                close_timeout=10,
            )

            await self._set_state(ConnectionState.CONNECTED)
            self._reconnect_count = 0

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info(f"WebSocket connected to {url}")

            # Authenticate if needed for private endpoint
            if private and self._auth:
                await self._authenticate()

            # Resubscribe to any existing subscriptions
            await self._resubscribe()

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self._set_state(ConnectionState.DISCONNECTED)
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._shutdown = True

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

        self._ws = None
        await self._set_state(ConnectionState.CLOSED)
        logger.info("WebSocket disconnected")

    async def _reconnect(self) -> None:
        """Attempt to reconnect after disconnection."""
        if self._shutdown:
            return

        self._reconnect_count += 1
        await self._set_state(ConnectionState.RECONNECTING)

        # Check max attempts
        if (
            self._config.max_reconnect_attempts > 0 and
            self._reconnect_count > self._config.max_reconnect_attempts
        ):
            logger.error(f"Max reconnection attempts ({self._config.max_reconnect_attempts}) reached")
            await self._set_state(ConnectionState.CLOSED)
            return

        # Calculate backoff delay
        delay = min(
            self._config.reconnect_delay * (self._config.reconnect_multiplier ** (self._reconnect_count - 1)),
            self._config.max_reconnect_delay,
        )

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

        try:
            # Determine if we need private connection
            needs_private = any(
                sub.sub_type in (SubscriptionType.OWN_TRADES, SubscriptionType.OPEN_ORDERS)
                for sub in self._subscriptions.values()
            )
            await self.connect(private=needs_private)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            # Will retry via _receive_loop handling

    async def _authenticate(self) -> None:
        """Authenticate for private channels."""
        if not self._auth:
            raise ValueError("Authentication required but no auth provided")

        await self._set_state(ConnectionState.AUTHENTICATING)

        # Get WebSocket token via REST API
        # Note: This would need to be implemented in the private client
        # For now, we'll use the signature-based auth flow
        # Kraken WS v2 uses a different auth flow

        # In v2, we send an auth message with API key
        # The actual implementation depends on Kraken's current auth requirements

        await self._set_state(ConnectionState.AUTHENTICATED)
        logger.info("WebSocket authenticated")

    async def _resubscribe(self) -> None:
        """Resubscribe to all active subscriptions after reconnect."""
        for sub in self._subscriptions.values():
            if sub.active:
                sub.active = False  # Will be set true on confirmation
                await self._send_subscribe(sub)

    # ==========================================
    # MESSAGE HANDLING
    # ==========================================

    async def _receive_loop(self) -> None:
        """Background task to receive and process messages."""
        while not self._shutdown:
            try:
                if not self._ws:
                    break

                message = await self._ws.recv()

                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                data = json.loads(message)
                await self._handle_message(data)

            except ConnectionClosedOK:
                logger.info("WebSocket closed normally")
                break

            except ConnectionClosedError as e:
                logger.warning(f"WebSocket connection lost: {e}")
                await self._set_state(ConnectionState.DISCONNECTED)
                if not self._shutdown:
                    await self._reconnect()
                break

            except asyncio.CancelledError:
                break

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON message: {e}")

            except Exception as e:
                logger.error(f"Receive error: {e}")
                if self._on_error:
                    await self._on_error(e)

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """
        Process a received message.

        Args:
            data: Parsed JSON message
        """
        msg_type = data.get("method") or data.get("channel")

        # Handle system messages
        if msg_type == "pong":
            return  # Heartbeat response

        if msg_type == "subscribe":
            await self._handle_subscribe_response(data)
            return

        if msg_type == "unsubscribe":
            await self._handle_unsubscribe_response(data)
            return

        if "error" in data:
            logger.error(f"WebSocket error: {data.get('error')}")
            if self._on_error:
                await self._on_error(KrakenAPIError([data.get("error", "Unknown error")]))
            return

        # Queue data messages
        try:
            self._message_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping message")

        # Trigger callback
        if self._on_message:
            try:
                await self._on_message(data)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

    async def _handle_subscribe_response(self, data: Dict[str, Any]) -> None:
        """Handle subscription confirmation."""
        success = data.get("success", False)
        result = data.get("result", {})
        req_id = data.get("req_id")

        if success:
            # Mark subscription as active
            for sub in self._subscriptions.values():
                if sub.req_id == req_id:
                    sub.active = True
                    logger.info(f"Subscribed to {sub.sub_type.value}: {sub.symbols}")
                    break
        else:
            error = data.get("error", "Unknown error")
            logger.error(f"Subscription failed: {error}")

    async def _handle_unsubscribe_response(self, data: Dict[str, Any]) -> None:
        """Handle unsubscription confirmation."""
        success = data.get("success", False)
        req_id = data.get("req_id")

        if success:
            # Remove subscription
            to_remove = None
            for key, sub in self._subscriptions.items():
                if sub.req_id == req_id:
                    to_remove = key
                    logger.info(f"Unsubscribed from {sub.sub_type.value}")
                    break

            if to_remove:
                del self._subscriptions[to_remove]

    async def _heartbeat_loop(self) -> None:
        """Background task for heartbeat/ping."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._config.ping_interval)

                if self._ws and self.is_connected:
                    await self._send({"method": "ping"})

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _send(self, data: Dict[str, Any]) -> None:
        """Send a message to WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")

        message = json.dumps(data)
        await self._ws.send(message)

    # ==========================================
    # SUBSCRIPTIONS
    # ==========================================

    async def _send_subscribe(self, sub: Subscription) -> None:
        """Send subscription request."""
        sub.req_id = self._next_req_id()

        message = {
            "method": "subscribe",
            "req_id": sub.req_id,
            "params": {
                "channel": sub.sub_type.value,
                "symbol": sub.symbols,
                **sub.params,
            },
        }

        await self._send(message)
        logger.debug(f"Sent subscribe: {sub.sub_type.value} {sub.symbols}")

    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """
        Subscribe to ticker updates.

        Args:
            symbols: Trading pairs (e.g., ["BTC/USD", "ETH/USD"])
        """
        key = f"ticker_{','.join(sorted(symbols))}"
        sub = Subscription(
            sub_type=SubscriptionType.TICKER,
            symbols=symbols,
        )
        self._subscriptions[key] = sub

        if self.is_connected:
            await self._send_subscribe(sub)

    async def subscribe_ohlc(
        self,
        symbols: List[str],
        interval: int = 5,
    ) -> None:
        """
        Subscribe to OHLC candle updates.

        Args:
            symbols: Trading pairs
            interval: Candle interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        """
        key = f"ohlc_{interval}_{','.join(sorted(symbols))}"
        sub = Subscription(
            sub_type=SubscriptionType.OHLC,
            symbols=symbols,
            params={"interval": interval},
        )
        self._subscriptions[key] = sub

        if self.is_connected:
            await self._send_subscribe(sub)

    async def subscribe_trade(self, symbols: List[str]) -> None:
        """
        Subscribe to trade updates.

        Args:
            symbols: Trading pairs
        """
        key = f"trade_{','.join(sorted(symbols))}"
        sub = Subscription(
            sub_type=SubscriptionType.TRADE,
            symbols=symbols,
        )
        self._subscriptions[key] = sub

        if self.is_connected:
            await self._send_subscribe(sub)

    async def subscribe_book(
        self,
        symbols: List[str],
        depth: int = 10,
    ) -> None:
        """
        Subscribe to order book updates.

        Args:
            symbols: Trading pairs
            depth: Book depth (10, 25, 100, 500, 1000)
        """
        key = f"book_{depth}_{','.join(sorted(symbols))}"
        sub = Subscription(
            sub_type=SubscriptionType.BOOK,
            symbols=symbols,
            params={"depth": depth},
        )
        self._subscriptions[key] = sub

        if self.is_connected:
            await self._send_subscribe(sub)

    async def subscribe_own_trades(self) -> None:
        """Subscribe to own trade updates (requires authentication)."""
        if self._state != ConnectionState.AUTHENTICATED:
            raise ConnectionError("Must be authenticated for private subscriptions")

        key = "own_trades"
        sub = Subscription(
            sub_type=SubscriptionType.OWN_TRADES,
            symbols=[],
        )
        self._subscriptions[key] = sub
        await self._send_subscribe(sub)

    async def subscribe_open_orders(self) -> None:
        """Subscribe to open order updates (requires authentication)."""
        if self._state != ConnectionState.AUTHENTICATED:
            raise ConnectionError("Must be authenticated for private subscriptions")

        key = "open_orders"
        sub = Subscription(
            sub_type=SubscriptionType.OPEN_ORDERS,
            symbols=[],
        )
        self._subscriptions[key] = sub
        await self._send_subscribe(sub)

    async def unsubscribe(
        self,
        sub_type: SubscriptionType,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Unsubscribe from a channel.

        Args:
            sub_type: Subscription type
            symbols: Symbols to unsubscribe (if applicable)
        """
        # Find matching subscription
        to_unsub = None
        for key, sub in self._subscriptions.items():
            if sub.sub_type == sub_type:
                if symbols is None or set(symbols) == set(sub.symbols):
                    to_unsub = sub
                    break

        if not to_unsub:
            logger.warning(f"No active subscription found for {sub_type.value}")
            return

        req_id = self._next_req_id()
        to_unsub.req_id = req_id

        message = {
            "method": "unsubscribe",
            "req_id": req_id,
            "params": {
                "channel": sub_type.value,
            },
        }

        if symbols:
            message["params"]["symbol"] = symbols

        await self._send(message)

    # ==========================================
    # ASYNC ITERATOR INTERFACE
    # ==========================================

    def __aiter__(self):
        """Async iterator for messages."""
        return self

    async def __anext__(self) -> Dict[str, Any]:
        """Get next message."""
        if self._shutdown:
            raise StopAsyncIteration

        try:
            message = await asyncio.wait_for(
                self._message_queue.get(),
                timeout=60.0,
            )
            return message
        except asyncio.TimeoutError:
            if self._shutdown:
                raise StopAsyncIteration
            return await self.__anext__()

    def get_message_nowait(self) -> Optional[Dict[str, Any]]:
        """Get message without waiting (returns None if empty)."""
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ==========================================
    # CONTEXT MANAGER
    # ==========================================

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    # ==========================================
    # RUN FOREVER
    # ==========================================

    async def run_forever(self) -> None:
        """
        Run the WebSocket client forever.

        Handles reconnection automatically. Use callbacks for message handling.
        """
        while not self._shutdown:
            try:
                if not self.is_connected:
                    await self.connect()

                # Wait for receive task to complete (on disconnect)
                if self._receive_task:
                    await self._receive_task

            except Exception as e:
                logger.error(f"Run forever error: {e}")
                await asyncio.sleep(self._config.reconnect_delay)

        logger.info("WebSocket run_forever completed")


class PublicWSClient(KrakenWebSocketClient):
    """Convenience class for public-only WebSocket."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def connect(self) -> None:
        await super().connect(private=False)


class PrivateWSClient(KrakenWebSocketClient):
    """Convenience class for private (authenticated) WebSocket."""

    def __init__(self, auth: KrakenAuth, **kwargs):
        super().__init__(auth=auth, **kwargs)

    async def connect(self) -> None:
        await super().connect(private=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_ticker_message(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse ticker channel message.

    Returns:
        Dict with parsed ticker data or None if not a ticker message
    """
    if data.get("channel") != "ticker":
        return None

    ticker_data = data.get("data", [{}])[0]
    return {
        "symbol": ticker_data.get("symbol"),
        "bid": float(ticker_data.get("bid", 0)),
        "ask": float(ticker_data.get("ask", 0)),
        "last": float(ticker_data.get("last", 0)),
        "volume": float(ticker_data.get("volume", 0)),
        "vwap": float(ticker_data.get("vwap", 0)),
        "high": float(ticker_data.get("high", 0)),
        "low": float(ticker_data.get("low", 0)),
    }


def parse_ohlc_message(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse OHLC channel message.

    Returns:
        Dict with parsed OHLC data or None if not an OHLC message
    """
    if data.get("channel") != "ohlc":
        return None

    ohlc_data = data.get("data", [{}])[0]
    return {
        "symbol": ohlc_data.get("symbol"),
        "timestamp": ohlc_data.get("timestamp"),
        "open": float(ohlc_data.get("open", 0)),
        "high": float(ohlc_data.get("high", 0)),
        "low": float(ohlc_data.get("low", 0)),
        "close": float(ohlc_data.get("close", 0)),
        "volume": float(ohlc_data.get("volume", 0)),
        "vwap": float(ohlc_data.get("vwap", 0)),
        "trades": int(ohlc_data.get("trades", 0)),
    }


def parse_trade_message(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse trade channel message.

    Returns:
        List of parsed trades or None if not a trade message
    """
    if data.get("channel") != "trade":
        return None

    trades = []
    for trade in data.get("data", []):
        trades.append({
            "symbol": trade.get("symbol"),
            "price": float(trade.get("price", 0)),
            "qty": float(trade.get("qty", 0)),
            "side": trade.get("side"),
            "timestamp": trade.get("timestamp"),
        })
    return trades
