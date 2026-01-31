"""
Kraken Private REST API Client.

Authenticated client for Kraken private endpoints:
- Account balance and trade balance
- Open/closed orders
- Trade history
- Order placement/modification/cancellation

All private endpoints require HMAC-SHA512 authentication.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .auth import KrakenAuth, KrakenCredentials, load_credentials_from_env
from .rate_limiter import SyncRateLimiter
from .kraken_errors import (
    KrakenAPIError,
    RateLimitError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
    classify_and_raise,
    with_retry,
    RetryConfig,
)

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types supported by Kraken."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"
    SETTLE_POSITION = "settle-position"


class OrderStatus(Enum):
    """Order status values."""
    PENDING = "pending"      # Order pending book entry
    OPEN = "open"            # Open order
    CLOSED = "closed"        # Closed order
    CANCELED = "canceled"    # Canceled order
    EXPIRED = "expired"      # Expired order


@dataclass
class Balance:
    """Account balance for a single asset."""
    asset: str
    balance: Decimal
    hold: Decimal = Decimal("0")  # Amount on hold for orders

    @property
    def available(self) -> Decimal:
        """Get available balance (total - hold)."""
        return self.balance - self.hold


@dataclass
class TradeBalance:
    """Trading account balance summary."""
    equivalent_balance: Decimal  # Total balance in base currency
    trade_balance: Decimal       # Balance available for trading
    margin_amount: Decimal       # Margin amount for positions
    unrealized_pnl: Decimal      # Unrealized P&L from open positions
    cost_basis: Decimal          # Cost basis of open positions
    floating_valuation: Decimal  # Floating valuation of open positions
    equity: Decimal              # Trade balance + unrealized P&L
    free_margin: Decimal         # Equity - margin_amount
    margin_level: Optional[Decimal] = None  # Margin level (if positions open)


@dataclass
class OrderInfo:
    """Information about an order."""
    order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    price: Optional[Decimal]
    volume: Decimal
    volume_executed: Decimal
    cost: Decimal
    fee: Decimal
    status: OrderStatus
    open_time: float
    close_time: Optional[float] = None
    stop_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    description: str = ""
    trades: List[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN)

    @property
    def remaining_volume(self) -> Decimal:
        """Get remaining volume to fill."""
        return self.volume - self.volume_executed


@dataclass
class TradeInfo:
    """Information about an executed trade."""
    trade_id: str
    order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    price: Decimal
    volume: Decimal
    cost: Decimal
    fee: Decimal
    timestamp: float
    position_status: str = ""
    misc: str = ""


class KrakenPrivateClient:
    """
    Client for Kraken private (authenticated) REST API.

    Usage:
        # From environment variables
        client = KrakenPrivateClient.from_env()

        # With explicit credentials
        client = KrakenPrivateClient(
            api_key="...",
            api_secret="...",
        )

        # Get balance
        balance = client.get_balance()

        # Place order
        order = client.add_order(
            pair="XBTUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=0.001,
            price=50000,
        )
    """

    BASE_URL = "https://api.kraken.com"

    # Private endpoint paths
    BALANCE_PATH = "/0/private/Balance"
    TRADE_BALANCE_PATH = "/0/private/TradeBalance"
    OPEN_ORDERS_PATH = "/0/private/OpenOrders"
    CLOSED_ORDERS_PATH = "/0/private/ClosedOrders"
    QUERY_ORDERS_PATH = "/0/private/QueryOrders"
    TRADES_HISTORY_PATH = "/0/private/TradesHistory"
    QUERY_TRADES_PATH = "/0/private/QueryTrades"
    ADD_ORDER_PATH = "/0/private/AddOrder"
    CANCEL_ORDER_PATH = "/0/private/CancelOrder"
    CANCEL_ALL_PATH = "/0/private/CancelAll"
    EDIT_ORDER_PATH = "/0/private/EditOrder"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        timeout: int = 30,
        max_retries: int = 3,
        paper_trading: bool = False,
    ):
        """
        Initialize Kraken private client.

        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret (base64 encoded)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            paper_trading: If True, validate orders but don't execute
        """
        self._auth = KrakenAuth(api_key, api_secret)
        self._timeout = timeout
        self._max_retries = max_retries
        self._paper_trading = paper_trading

        # Private endpoints share rate limit with public (same counter)
        # But private calls count more (typically 2x)
        self._rate_limiter = SyncRateLimiter(
            max_counter=20,
            decay_rate=0.5,
            min_delay=0.5,  # More conservative for private
        )

        # Configure session with retries
        self._session = requests.Session()
        retry_strategy = Retry(
            total=0,  # We handle retries ourselves
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)

        logger.info(
            f"Initialized Kraken private client "
            f"(paper_trading={paper_trading})"
        )

    @classmethod
    def from_env(cls, **kwargs) -> "KrakenPrivateClient":
        """
        Create client from environment variables.

        Expects KRAKEN_API_KEY and KRAKEN_API_SECRET.
        """
        credentials = load_credentials_from_env()
        return cls(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            **kwargs,
        )

    @classmethod
    def from_credentials(
        cls,
        credentials: KrakenCredentials,
        **kwargs,
    ) -> "KrakenPrivateClient":
        """Create client from credentials object."""
        return cls(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            **kwargs,
        )

    def _make_request(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        cost: int = 2,  # Private calls typically cost 2
    ) -> Dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            path: API endpoint path
            data: Request data (optional)
            cost: Rate limit cost

        Returns:
            API result dictionary

        Raises:
            KrakenAPIError: On API error
        """
        self._rate_limiter.acquire(cost)

        if data is None:
            data = {}

        # Get signed headers and data with nonce
        headers, signed_data = self._auth.get_signed_data(path, data)

        url = f"{self.BASE_URL}{path}"

        try:
            response = self._session.post(
                url,
                data=signed_data,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Check for API errors
            if result.get("error"):
                classify_and_raise(result["error"], result)

            return result.get("result", {})

        except requests.exceptions.Timeout:
            raise KrakenAPIError(["Request timeout"])
        except requests.exceptions.ConnectionError as e:
            raise KrakenAPIError([f"Connection error: {e}"])

    # ==========================================
    # BALANCE METHODS
    # ==========================================

    @with_retry(RetryConfig(max_retries=3))
    def get_balance(self) -> Dict[str, Balance]:
        """
        Get account balance for all assets.

        Returns:
            Dict mapping asset to Balance
        """
        result = self._make_request(self.BALANCE_PATH)

        balances = {}
        for asset, value in result.items():
            balances[asset] = Balance(
                asset=asset,
                balance=Decimal(str(value)),
            )

        logger.debug(f"Got balance for {len(balances)} assets")
        return balances

    @with_retry(RetryConfig(max_retries=3))
    def get_trade_balance(self, asset: str = "ZUSD") -> TradeBalance:
        """
        Get trading balance summary.

        Args:
            asset: Base asset for balance calculation (default USD)

        Returns:
            TradeBalance summary
        """
        result = self._make_request(
            self.TRADE_BALANCE_PATH,
            data={"asset": asset},
        )

        return TradeBalance(
            equivalent_balance=Decimal(result.get("eb", "0")),
            trade_balance=Decimal(result.get("tb", "0")),
            margin_amount=Decimal(result.get("m", "0")),
            unrealized_pnl=Decimal(result.get("n", "0")),
            cost_basis=Decimal(result.get("c", "0")),
            floating_valuation=Decimal(result.get("v", "0")),
            equity=Decimal(result.get("e", "0")),
            free_margin=Decimal(result.get("mf", "0")),
            margin_level=Decimal(result["ml"]) if result.get("ml") else None,
        )

    # ==========================================
    # ORDER QUERY METHODS
    # ==========================================

    @with_retry(RetryConfig(max_retries=3))
    def get_open_orders(
        self,
        trades: bool = False,
        userref: Optional[int] = None,
    ) -> Dict[str, OrderInfo]:
        """
        Get all open orders.

        Args:
            trades: Include related trade IDs
            userref: Filter by user reference ID

        Returns:
            Dict mapping order ID to OrderInfo
        """
        data: Dict[str, Any] = {"trades": trades}
        if userref is not None:
            data["userref"] = userref

        result = self._make_request(self.OPEN_ORDERS_PATH, data)

        orders = {}
        for order_id, order_data in result.get("open", {}).items():
            orders[order_id] = self._parse_order(order_id, order_data)

        logger.debug(f"Got {len(orders)} open orders")
        return orders

    @with_retry(RetryConfig(max_retries=3))
    def get_closed_orders(
        self,
        trades: bool = False,
        start: Optional[int] = None,
        end: Optional[int] = None,
        offset: int = 0,
        closetime: str = "both",
    ) -> Dict[str, OrderInfo]:
        """
        Get closed orders.

        Args:
            trades: Include related trade IDs
            start: Start timestamp
            end: End timestamp
            offset: Result offset for pagination
            closetime: "open", "close", or "both"

        Returns:
            Dict mapping order ID to OrderInfo
        """
        data: Dict[str, Any] = {
            "trades": trades,
            "ofs": offset,
            "closetime": closetime,
        }
        if start is not None:
            data["start"] = start
        if end is not None:
            data["end"] = end

        result = self._make_request(self.CLOSED_ORDERS_PATH, data)

        orders = {}
        for order_id, order_data in result.get("closed", {}).items():
            orders[order_id] = self._parse_order(order_id, order_data)

        logger.debug(f"Got {len(orders)} closed orders")
        return orders

    @with_retry(RetryConfig(max_retries=3))
    def query_orders(
        self,
        order_ids: List[str],
        trades: bool = False,
    ) -> Dict[str, OrderInfo]:
        """
        Query specific orders by ID.

        Args:
            order_ids: List of order transaction IDs
            trades: Include related trade IDs

        Returns:
            Dict mapping order ID to OrderInfo
        """
        data = {
            "txid": ",".join(order_ids),
            "trades": trades,
        }

        result = self._make_request(self.QUERY_ORDERS_PATH, data)

        orders = {}
        for order_id, order_data in result.items():
            orders[order_id] = self._parse_order(order_id, order_data)

        return orders

    def _parse_order(self, order_id: str, data: Dict) -> OrderInfo:
        """Parse order data from API response."""
        descr = data.get("descr", {})

        # Parse status
        status_str = data.get("status", "open")
        try:
            status = OrderStatus(status_str)
        except ValueError:
            status = OrderStatus.OPEN

        # Parse side
        side_str = descr.get("type", "buy")
        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        # Parse order type
        order_type_str = descr.get("ordertype", "limit")
        try:
            order_type = OrderType(order_type_str)
        except ValueError:
            order_type = OrderType.LIMIT

        return OrderInfo(
            order_id=order_id,
            pair=descr.get("pair", ""),
            side=side,
            order_type=order_type,
            price=Decimal(descr.get("price", "0")) if descr.get("price") else None,
            volume=Decimal(data.get("vol", "0")),
            volume_executed=Decimal(data.get("vol_exec", "0")),
            cost=Decimal(data.get("cost", "0")),
            fee=Decimal(data.get("fee", "0")),
            status=status,
            open_time=float(data.get("opentm", 0)),
            close_time=float(data.get("closetm", 0)) if data.get("closetm") else None,
            stop_price=Decimal(data.get("stopprice", "0")) if data.get("stopprice") else None,
            limit_price=Decimal(data.get("limitprice", "0")) if data.get("limitprice") else None,
            description=descr.get("order", ""),
            trades=data.get("trades", []),
        )

    # ==========================================
    # TRADE HISTORY METHODS
    # ==========================================

    @with_retry(RetryConfig(max_retries=3))
    def get_trades_history(
        self,
        trade_type: str = "all",
        start: Optional[int] = None,
        end: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, TradeInfo]:
        """
        Get trade history.

        Args:
            trade_type: "all", "any position", "closed position", etc.
            start: Start timestamp
            end: End timestamp
            offset: Result offset for pagination

        Returns:
            Dict mapping trade ID to TradeInfo
        """
        data: Dict[str, Any] = {
            "type": trade_type,
            "ofs": offset,
        }
        if start is not None:
            data["start"] = start
        if end is not None:
            data["end"] = end

        result = self._make_request(self.TRADES_HISTORY_PATH, data)

        trades = {}
        for trade_id, trade_data in result.get("trades", {}).items():
            trades[trade_id] = self._parse_trade(trade_id, trade_data)

        logger.debug(f"Got {len(trades)} historical trades")
        return trades

    @with_retry(RetryConfig(max_retries=3))
    def query_trades(
        self,
        trade_ids: List[str],
    ) -> Dict[str, TradeInfo]:
        """
        Query specific trades by ID.

        Args:
            trade_ids: List of trade IDs

        Returns:
            Dict mapping trade ID to TradeInfo
        """
        data = {"txid": ",".join(trade_ids)}
        result = self._make_request(self.QUERY_TRADES_PATH, data)

        trades = {}
        for trade_id, trade_data in result.items():
            trades[trade_id] = self._parse_trade(trade_id, trade_data)

        return trades

    def _parse_trade(self, trade_id: str, data: Dict) -> TradeInfo:
        """Parse trade data from API response."""
        side_str = data.get("type", "buy")
        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        order_type_str = data.get("ordertype", "limit")
        try:
            order_type = OrderType(order_type_str)
        except ValueError:
            order_type = OrderType.LIMIT

        return TradeInfo(
            trade_id=trade_id,
            order_id=data.get("ordertxid", ""),
            pair=data.get("pair", ""),
            side=side,
            order_type=order_type,
            price=Decimal(str(data.get("price", "0"))),
            volume=Decimal(str(data.get("vol", "0"))),
            cost=Decimal(str(data.get("cost", "0"))),
            fee=Decimal(str(data.get("fee", "0"))),
            timestamp=float(data.get("time", 0)),
            position_status=data.get("posstatus", ""),
            misc=data.get("misc", ""),
        )

    # ==========================================
    # ORDER MANAGEMENT METHODS
    # ==========================================

    def add_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        volume: Union[Decimal, float, str],
        price: Optional[Union[Decimal, float, str]] = None,
        price2: Optional[Union[Decimal, float, str]] = None,
        leverage: Optional[str] = None,
        reduce_only: bool = False,
        start_time: Optional[str] = None,
        expire_time: Optional[str] = None,
        userref: Optional[int] = None,
        validate: Optional[bool] = None,
        close_order_type: Optional[OrderType] = None,
        close_price: Optional[Union[Decimal, float, str]] = None,
        close_price2: Optional[Union[Decimal, float, str]] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            side: Buy or sell
            order_type: Order type (market, limit, etc.)
            volume: Order volume in base currency
            price: Price for limit orders
            price2: Secondary price (stop-loss limit, take-profit limit)
            leverage: Leverage amount ("none", "2", "3", etc.)
            reduce_only: Only reduce existing position
            start_time: Scheduled start time
            expire_time: Order expiration time
            userref: User reference ID for tracking
            validate: Validate only, don't place order
            close_order_type: Conditional close order type
            close_price: Conditional close price
            close_price2: Conditional close secondary price
            time_in_force: GTC, IOC, GTD

        Returns:
            Dict with "descr" (description) and "txid" (order ID list)

        Raises:
            InsufficientFundsError: If not enough funds
            OrderError: For order-related errors
        """
        # Force validate mode if paper trading
        if validate is None:
            validate = self._paper_trading

        data: Dict[str, Any] = {
            "pair": pair,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": str(volume),
        }

        if price is not None:
            data["price"] = str(price)
        if price2 is not None:
            data["price2"] = str(price2)
        if leverage is not None:
            data["leverage"] = leverage
        if reduce_only:
            data["reduce_only"] = "true"
        if start_time is not None:
            data["starttm"] = start_time
        if expire_time is not None:
            data["expiretm"] = expire_time
        if userref is not None:
            data["userref"] = userref
        if validate:
            data["validate"] = "true"
        if time_in_force is not None:
            data["timeinforce"] = time_in_force

        # Conditional close
        if close_order_type is not None:
            data["close[ordertype]"] = close_order_type.value
            if close_price is not None:
                data["close[price]"] = str(close_price)
            if close_price2 is not None:
                data["close[price2]"] = str(close_price2)

        result = self._make_request(self.ADD_ORDER_PATH, data)

        if validate:
            logger.info(f"Order validated: {result.get('descr', {}).get('order', '')}")
            # Generate a fake txid for paper trading to enable order tracking
            if self._paper_trading:
                import uuid
                fake_txid = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
                result["txid"] = [fake_txid]
                logger.debug(f"Paper trading: generated fake txid {fake_txid}")
        else:
            txids = result.get("txid", [])
            logger.info(f"Order placed: {txids}, {result.get('descr', {}).get('order', '')}")

        return result

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Order transaction ID

        Returns:
            Dict with count of canceled orders

        Raises:
            OrderError: If order not found or cannot be canceled
        """
        result = self._make_request(
            self.CANCEL_ORDER_PATH,
            data={"txid": order_id},
        )

        count = result.get("count", 0)
        logger.info(f"Canceled order {order_id}, count={count}")
        return result

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Dict with count of canceled orders
        """
        result = self._make_request(self.CANCEL_ALL_PATH)

        count = result.get("count", 0)
        logger.info(f"Canceled all orders, count={count}")
        return result

    def edit_order(
        self,
        order_id: str,
        pair: str,
        volume: Optional[Union[Decimal, float, str]] = None,
        price: Optional[Union[Decimal, float, str]] = None,
        price2: Optional[Union[Decimal, float, str]] = None,
        userref: Optional[int] = None,
        validate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Edit an existing order.

        Note: Not all order parameters can be edited. Check Kraken docs.

        Args:
            order_id: Order transaction ID to edit
            pair: Trading pair (required for edit)
            volume: New volume
            price: New price
            price2: New secondary price
            userref: New user reference ID
            validate: Validate only

        Returns:
            Dict with new order info
        """
        if validate is None:
            validate = self._paper_trading

        data: Dict[str, Any] = {
            "txid": order_id,
            "pair": pair,
        }

        if volume is not None:
            data["volume"] = str(volume)
        if price is not None:
            data["price"] = str(price)
        if price2 is not None:
            data["price2"] = str(price2)
        if userref is not None:
            data["userref"] = userref
        if validate:
            data["validate"] = "true"

        result = self._make_request(self.EDIT_ORDER_PATH, data)

        if validate:
            logger.info(f"Order edit validated: {order_id}")
        else:
            logger.info(f"Order edited: {order_id} -> {result.get('txid', '')}")

        return result

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
