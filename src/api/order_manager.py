"""
Order Manager for Grid Trading.

High-level order management that handles:
- Grid order placement and tracking
- Order state synchronization
- Partial fills and replacements
- Position management

This sits above the low-level API client and provides
trading-specific logic for grid management.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime
import threading
import time

from .kraken_private import (
    KrakenPrivateClient,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderInfo,
    Balance,
)
from .kraken_errors import (
    KrakenAPIError,
    InsufficientFundsError,
    OrderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class GridOrderType(Enum):
    """Types of grid orders."""
    BUY = auto()
    SELL = auto()


class OrderState(Enum):
    """Order lifecycle states."""
    PENDING_SUBMIT = auto()    # Queued for submission
    SUBMITTED = auto()         # Submitted to exchange
    OPEN = auto()              # Confirmed open on exchange
    PARTIALLY_FILLED = auto()  # Some volume filled
    FILLED = auto()            # Fully filled
    PENDING_CANCEL = auto()    # Queued for cancellation
    CANCELED = auto()          # Confirmed canceled
    FAILED = auto()            # Submission failed


@dataclass
class GridOrder:
    """Represents a single grid order."""
    grid_id: str              # Internal grid identifier
    level: int                # Grid level (0 = lowest)
    price: Decimal            # Target price
    side: GridOrderType       # Buy or sell
    volume: Decimal           # Order volume

    # Exchange tracking
    order_id: Optional[str] = None  # Kraken order ID
    state: OrderState = OrderState.PENDING_SUBMIT

    # Fill tracking
    filled_volume: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    fees: Decimal = Decimal("0")

    # Timing
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    filled_at: Optional[float] = None

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def remaining_volume(self) -> Decimal:
        """Get unfilled volume."""
        return self.volume - self.filled_volume

    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.state in (
            OrderState.SUBMITTED,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
        )

    @property
    def is_complete(self) -> bool:
        """Check if order lifecycle is complete."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.FAILED,
        )


@dataclass
class GridPosition:
    """Tracks current position from grid trades."""
    pair: str
    base_asset: str
    quote_asset: str

    # Position
    quantity: Decimal = Decimal("0")  # Positive = long, negative = short
    avg_entry_price: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Trade counts
    buy_count: int = 0
    sell_count: int = 0

    def update_from_fill(
        self,
        side: GridOrderType,
        volume: Decimal,
        price: Decimal,
        fee: Decimal,
    ) -> Decimal:
        """
        Update position from an order fill.

        Args:
            side: Buy or sell
            volume: Filled volume
            price: Fill price
            fee: Fee paid

        Returns:
            Realized P&L from this fill (if closing position)
        """
        realized = Decimal("0")
        cost = volume * price

        if side == GridOrderType.BUY:
            self.buy_count += 1

            # Check if closing short position
            if self.quantity < 0:
                close_volume = min(volume, abs(self.quantity))
                close_cost = close_volume * price
                original_cost = close_volume * self.avg_entry_price
                realized = original_cost - close_cost  # Short P&L

                # Update position
                self.quantity += close_volume
                self.total_cost -= original_cost

                # Remaining volume opens new long
                remaining = volume - close_volume
                if remaining > 0:
                    self.quantity += remaining
                    self.total_cost += remaining * price
            else:
                # Adding to long position
                self.quantity += volume
                self.total_cost += cost

                # Update average entry
                if self.quantity > 0:
                    self.avg_entry_price = self.total_cost / self.quantity

        else:  # SELL
            self.sell_count += 1

            # Check if closing long position
            if self.quantity > 0:
                close_volume = min(volume, self.quantity)
                close_cost = close_volume * price
                original_cost = close_volume * self.avg_entry_price
                realized = close_cost - original_cost  # Long P&L

                # Update position
                self.quantity -= close_volume
                self.total_cost -= original_cost

                # Remaining volume opens new short
                remaining = volume - close_volume
                if remaining > 0:
                    self.quantity -= remaining
                    self.total_cost += remaining * price
            else:
                # Adding to short position
                self.quantity -= volume
                self.total_cost += cost

                # Update average entry
                if self.quantity < 0:
                    self.avg_entry_price = self.total_cost / abs(self.quantity)

        self.total_fees += fee
        self.realized_pnl += realized

        return realized

    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current price."""
        if self.quantity == 0:
            self.unrealized_pnl = Decimal("0")
        elif self.quantity > 0:
            # Long position
            current_value = self.quantity * current_price
            self.unrealized_pnl = current_value - self.total_cost
        else:
            # Short position
            current_value = abs(self.quantity) * current_price
            self.unrealized_pnl = self.total_cost - current_value


@dataclass
class OrderManagerConfig:
    """Configuration for order manager."""
    pair: str = "XBTUSD"
    base_asset: str = "XBT"
    quote_asset: str = "USD"

    # Precision
    price_decimals: int = 1     # XBTUSD price precision
    volume_decimals: int = 8    # XBT volume precision

    # Order settings
    order_type: OrderType = OrderType.LIMIT
    time_in_force: str = "GTC"  # Good till canceled

    # Retry settings
    max_submit_retries: int = 3
    max_cancel_retries: int = 5

    # Sync settings
    sync_interval: float = 5.0   # Seconds between syncs
    stale_order_threshold: float = 300.0  # 5 min before considering order stale

    # User reference base (for tracking our orders)
    userref_base: int = 1000000


class OrderManager:
    """
    Manages grid orders lifecycle.

    Responsibilities:
    - Submit orders to exchange
    - Track order states
    - Handle fills and partial fills
    - Maintain position
    - Sync with exchange state
    """

    def __init__(
        self,
        client: KrakenPrivateClient,
        config: Optional[OrderManagerConfig] = None,
        on_fill: Optional[Callable[[GridOrder, Decimal, Decimal], None]] = None,
    ):
        """
        Initialize order manager.

        Args:
            client: Kraken private API client
            config: Order manager configuration
            on_fill: Callback when order is filled (order, volume, price)
        """
        self._client = client
        self._config = config or OrderManagerConfig()
        self._on_fill = on_fill

        # Order tracking
        self._orders: Dict[str, GridOrder] = {}  # grid_id -> GridOrder
        self._order_id_map: Dict[str, str] = {}  # kraken_id -> grid_id

        # Position tracking
        self._position = GridPosition(
            pair=self._config.pair,
            base_asset=self._config.base_asset,
            quote_asset=self._config.quote_asset,
        )

        # User reference counter for this session
        self._userref_counter = self._config.userref_base

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Initialized OrderManager for {self._config.pair}")

    def _next_userref(self) -> int:
        """Get next user reference ID."""
        with self._lock:
            ref = self._userref_counter
            self._userref_counter += 1
            return ref

    def _round_price(self, price: Decimal) -> Decimal:
        """Round price to exchange precision."""
        return price.quantize(
            Decimal(10) ** -self._config.price_decimals,
            rounding=ROUND_DOWN,
        )

    def _round_volume(self, volume: Decimal) -> Decimal:
        """Round volume to exchange precision."""
        return volume.quantize(
            Decimal(10) ** -self._config.volume_decimals,
            rounding=ROUND_DOWN,
        )

    # ==========================================
    # ORDER SUBMISSION
    # ==========================================

    def create_grid_order(
        self,
        level: int,
        price: Decimal,
        side: GridOrderType,
        volume: Decimal,
    ) -> GridOrder:
        """
        Create a new grid order (not yet submitted).

        Args:
            level: Grid level
            price: Target price
            side: Buy or sell
            volume: Order volume

        Returns:
            GridOrder in PENDING_SUBMIT state
        """
        grid_id = f"grid_{level}_{side.name.lower()}_{int(time.time() * 1000)}"

        order = GridOrder(
            grid_id=grid_id,
            level=level,
            price=self._round_price(price),
            side=side,
            volume=self._round_volume(volume),
        )

        with self._lock:
            self._orders[grid_id] = order

        logger.debug(f"Created grid order: {grid_id} {side.name} {volume} @ {price}")
        return order

    def submit_order(self, order: GridOrder) -> bool:
        """
        Submit an order to the exchange.

        Args:
            order: GridOrder to submit

        Returns:
            True if submission successful
        """
        if order.state != OrderState.PENDING_SUBMIT:
            logger.warning(f"Cannot submit order {order.grid_id} in state {order.state}")
            return False

        api_side = OrderSide.BUY if order.side == GridOrderType.BUY else OrderSide.SELL
        userref = self._next_userref()

        for attempt in range(self._config.max_submit_retries):
            try:
                result = self._client.add_order(
                    pair=self._config.pair,
                    side=api_side,
                    order_type=self._config.order_type,
                    volume=order.volume,
                    price=order.price,
                    userref=userref,
                    time_in_force=self._config.time_in_force,
                )

                # Get order ID from result
                txids = result.get("txid", [])
                if txids:
                    order.order_id = txids[0]
                    order.state = OrderState.SUBMITTED
                    order.submitted_at = time.time()

                    with self._lock:
                        self._order_id_map[order.order_id] = order.grid_id

                    logger.info(
                        f"Submitted order {order.grid_id} -> {order.order_id}: "
                        f"{order.side.name} {order.volume} @ {order.price}"
                    )
                    return True

                logger.error(f"No order ID returned for {order.grid_id}")
                return False

            except InsufficientFundsError as e:
                order.state = OrderState.FAILED
                order.last_error = str(e)
                logger.error(f"Insufficient funds for {order.grid_id}: {e}")
                return False

            except RateLimitError as e:
                logger.warning(f"Rate limited on submit, attempt {attempt + 1}")
                time.sleep(e.error_info.retry_after or 5.0)

            except KrakenAPIError as e:
                order.error_count += 1
                order.last_error = str(e)
                logger.warning(f"Submit failed for {order.grid_id}: {e}")

                if not e.should_retry or attempt >= self._config.max_submit_retries - 1:
                    order.state = OrderState.FAILED
                    return False

                time.sleep(2 ** attempt)  # Exponential backoff

        order.state = OrderState.FAILED
        return False

    def submit_orders_batch(self, orders: List[GridOrder]) -> Tuple[int, int]:
        """
        Submit multiple orders with rate limiting.

        Args:
            orders: List of orders to submit

        Returns:
            Tuple of (success_count, failure_count)
        """
        success = 0
        failure = 0

        for order in orders:
            if self.submit_order(order):
                success += 1
            else:
                failure += 1

        logger.info(f"Batch submit: {success} succeeded, {failure} failed")
        return success, failure

    # ==========================================
    # ORDER CANCELLATION
    # ==========================================

    def cancel_order(self, grid_id: str) -> bool:
        """
        Cancel an order by grid ID.

        Args:
            grid_id: Internal grid order ID

        Returns:
            True if cancellation successful
        """
        with self._lock:
            order = self._orders.get(grid_id)

        if not order:
            logger.warning(f"Order not found: {grid_id}")
            return False

        if not order.order_id:
            # Never submitted, just mark as canceled
            order.state = OrderState.CANCELED
            return True

        if not order.is_active:
            logger.debug(f"Order {grid_id} not active, cannot cancel")
            return False

        order.state = OrderState.PENDING_CANCEL

        for attempt in range(self._config.max_cancel_retries):
            try:
                self._client.cancel_order(order.order_id)
                order.state = OrderState.CANCELED
                logger.info(f"Canceled order {grid_id} ({order.order_id})")
                return True

            except OrderError as e:
                if "Unknown order" in str(e):
                    # Order may have been filled
                    order.state = OrderState.CANCELED
                    return True
                order.last_error = str(e)
                logger.warning(f"Cancel failed for {grid_id}: {e}")

            except RateLimitError:
                time.sleep(5.0)

            except KrakenAPIError as e:
                order.error_count += 1
                order.last_error = str(e)

                if not e.should_retry:
                    break

                time.sleep(2 ** attempt)

        logger.error(f"Failed to cancel order {grid_id} after {self._config.max_cancel_retries} attempts")
        return False

    def cancel_all_grid_orders(self) -> Tuple[int, int]:
        """
        Cancel all active grid orders.

        Returns:
            Tuple of (success_count, failure_count)
        """
        active_orders = [
            order for order in self._orders.values()
            if order.is_active
        ]

        success = 0
        failure = 0

        for order in active_orders:
            if self.cancel_order(order.grid_id):
                success += 1
            else:
                failure += 1

        logger.info(f"Cancel all: {success} succeeded, {failure} failed")
        return success, failure

    # ==========================================
    # ORDER SYNCHRONIZATION
    # ==========================================

    def sync_orders(self) -> Dict[str, OrderState]:
        """
        Synchronize order states with exchange.

        Fetches current order status from Kraken and updates local state.

        Returns:
            Dict mapping grid_id to new state for changed orders
        """
        changes: Dict[str, OrderState] = {}

        # Get order IDs to check
        order_ids = [
            order.order_id
            for order in self._orders.values()
            if order.order_id and order.is_active
        ]

        if not order_ids:
            return changes

        try:
            # Query orders from exchange
            exchange_orders = self._client.query_orders(order_ids)

            for kraken_id, order_info in exchange_orders.items():
                grid_id = self._order_id_map.get(kraken_id)
                if not grid_id:
                    continue

                order = self._orders.get(grid_id)
                if not order:
                    continue

                # Update from exchange state
                old_state = order.state
                new_state = self._update_from_exchange(order, order_info)

                if new_state != old_state:
                    changes[grid_id] = new_state
                    logger.debug(f"Order {grid_id} state change: {old_state} -> {new_state}")

        except KrakenAPIError as e:
            logger.warning(f"Sync failed: {e}")

        return changes

    def _update_from_exchange(
        self,
        order: GridOrder,
        info: OrderInfo,
    ) -> OrderState:
        """
        Update grid order from exchange order info.

        Args:
            order: GridOrder to update
            info: OrderInfo from exchange

        Returns:
            New order state
        """
        # Check for fills
        if info.volume_executed > order.filled_volume:
            fill_volume = info.volume_executed - order.filled_volume
            fill_price = info.cost / info.volume_executed if info.volume_executed > 0 else order.price

            # Update order
            order.filled_volume = info.volume_executed
            order.avg_fill_price = fill_price
            order.fees = info.fee

            # Update position
            realized = self._position.update_from_fill(
                side=order.side,
                volume=fill_volume,
                price=fill_price,
                fee=info.fee - order.fees if order.fees else info.fee,
            )

            # Trigger callback
            if self._on_fill:
                self._on_fill(order, fill_volume, fill_price)

            logger.info(
                f"Fill: {order.grid_id} {fill_volume} @ {fill_price}, "
                f"realized P&L: {realized}"
            )

        # Update state based on exchange status
        if info.status == OrderStatus.CLOSED:
            if info.volume_executed >= order.volume:
                order.state = OrderState.FILLED
                order.filled_at = time.time()
            else:
                order.state = OrderState.CANCELED
        elif info.status == OrderStatus.CANCELED:
            order.state = OrderState.CANCELED
        elif info.status == OrderStatus.EXPIRED:
            order.state = OrderState.CANCELED
        elif info.volume_executed > 0:
            order.state = OrderState.PARTIALLY_FILLED
        elif info.status == OrderStatus.OPEN:
            order.state = OrderState.OPEN

        return order.state

    # ==========================================
    # POSITION MANAGEMENT
    # ==========================================

    @property
    def position(self) -> GridPosition:
        """Get current position."""
        return self._position

    def update_position_mark(self, current_price: Decimal) -> None:
        """Update position unrealized P&L with current price."""
        self._position.update_unrealized_pnl(current_price)

    def get_position_value(self, current_price: Decimal) -> Decimal:
        """Get total position value at current price."""
        return abs(self._position.quantity) * current_price

    def get_open_order_exposure(self) -> Tuple[Decimal, Decimal]:
        """
        Get total exposure from open orders.

        Returns:
            Tuple of (buy_exposure, sell_exposure) in quote currency
        """
        buy_exposure = Decimal("0")
        sell_exposure = Decimal("0")

        for order in self._orders.values():
            if not order.is_active:
                continue

            exposure = order.remaining_volume * order.price

            if order.side == GridOrderType.BUY:
                buy_exposure += exposure
            else:
                sell_exposure += exposure

        return buy_exposure, sell_exposure

    # ==========================================
    # ORDER QUERIES
    # ==========================================

    def get_order(self, grid_id: str) -> Optional[GridOrder]:
        """Get order by grid ID."""
        return self._orders.get(grid_id)

    def get_order_by_exchange_id(self, order_id: str) -> Optional[GridOrder]:
        """Get order by exchange order ID."""
        grid_id = self._order_id_map.get(order_id)
        if grid_id:
            return self._orders.get(grid_id)
        return None

    def get_active_orders(self) -> List[GridOrder]:
        """Get all active orders."""
        return [order for order in self._orders.values() if order.is_active]

    def get_orders_by_level(self, level: int) -> List[GridOrder]:
        """Get all orders at a grid level."""
        return [order for order in self._orders.values() if order.level == level]

    def get_orders_by_side(self, side: GridOrderType) -> List[GridOrder]:
        """Get all orders of a specific side."""
        return [order for order in self._orders.values() if order.side == side]

    def get_filled_orders(self) -> List[GridOrder]:
        """Get all filled orders."""
        return [
            order for order in self._orders.values()
            if order.state == OrderState.FILLED
        ]

    # ==========================================
    # STATISTICS
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        orders = list(self._orders.values())

        active = [o for o in orders if o.is_active]
        filled = [o for o in orders if o.state == OrderState.FILLED]
        failed = [o for o in orders if o.state == OrderState.FAILED]

        total_filled_volume = sum(o.filled_volume for o in orders)
        total_fees = sum(o.fees for o in orders)

        return {
            "total_orders": len(orders),
            "active_orders": len(active),
            "filled_orders": len(filled),
            "failed_orders": len(failed),
            "position": {
                "quantity": float(self._position.quantity),
                "avg_entry": float(self._position.avg_entry_price),
                "realized_pnl": float(self._position.realized_pnl),
                "unrealized_pnl": float(self._position.unrealized_pnl),
                "total_fees": float(self._position.total_fees),
            },
            "totals": {
                "filled_volume": float(total_filled_volume),
                "fees": float(total_fees),
                "buy_count": self._position.buy_count,
                "sell_count": self._position.sell_count,
            },
        }

    def clear_completed_orders(self) -> int:
        """
        Remove completed orders from tracking.

        Returns:
            Number of orders removed
        """
        completed_ids = [
            grid_id for grid_id, order in self._orders.items()
            if order.is_complete
        ]

        with self._lock:
            for grid_id in completed_ids:
                order = self._orders.pop(grid_id, None)
                if order and order.order_id:
                    self._order_id_map.pop(order.order_id, None)

        logger.debug(f"Cleared {len(completed_ids)} completed orders")
        return len(completed_ids)
