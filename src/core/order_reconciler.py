"""
Order Reconciler for Crash Recovery.

Reconciles local order state with exchange state on startup.
Handles scenarios where:
- Orders were submitted but confirmation was lost
- Fills occurred while bot was down
- Orders expired or were canceled externally

Usage:
    reconciler = OrderReconciler(rest_client, state_manager)
    result = await reconciler.reconcile()
    print(f"Matched: {result.matched_orders}, Filled during downtime: {result.filled_during_downtime}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from src.api.kraken_private import (
    KrakenPrivateClient,
    OrderInfo,
    TradeInfo,
    OrderStatus,
    OrderSide,
)
from src.api.order_manager import (
    GridOrder,
    GridOrderType,
    OrderState,
    GridPosition,
)
from src.core.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of order reconciliation."""

    # Counts
    matched_orders: int = 0
    orphaned_local: int = 0
    orphaned_exchange: int = 0
    filled_during_downtime: int = 0
    partially_filled: int = 0

    # Details
    matched_order_ids: List[str] = field(default_factory=list)
    orphaned_local_ids: List[str] = field(default_factory=list)
    orphaned_exchange_ids: List[str] = field(default_factory=list)
    filled_order_ids: List[str] = field(default_factory=list)
    partially_filled_ids: List[str] = field(default_factory=list)

    # Position updates
    position_fills: List[Dict[str, Any]] = field(default_factory=list)

    # Actions taken
    actions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"ReconciliationResult("
            f"matched={self.matched_orders}, "
            f"orphaned_local={self.orphaned_local}, "
            f"orphaned_exchange={self.orphaned_exchange}, "
            f"filled_during_downtime={self.filled_during_downtime}, "
            f"partially_filled={self.partially_filled})"
        )


class OrderReconciler:
    """
    Reconciles local order state with exchange state.

    This is critical for crash recovery - when the bot restarts after an
    unexpected shutdown, the local database may not reflect the current
    state on the exchange. This class:

    1. Fetches all open orders from exchange
    2. Fetches recent trades since last known fill
    3. Matches local orders to exchange orders
    4. Updates local state for fills that occurred during downtime
    5. Marks orders as canceled if they disappeared from exchange

    The reconciler is conservative - it never places new orders or cancels
    existing ones, only updates local state to match exchange reality.
    """

    def __init__(
        self,
        rest_client: KrakenPrivateClient,
        state_manager: StateManager,
        trading_pair: str = "XBTUSD",
    ):
        """
        Initialize reconciler.

        Args:
            rest_client: Kraken REST API client
            state_manager: State manager for database access
            trading_pair: Trading pair to filter trades
        """
        self._client = rest_client
        self._state_manager = state_manager
        self._trading_pair = trading_pair

    def reconcile(self) -> ReconciliationResult:
        """
        Perform full reconciliation.

        Returns:
            ReconciliationResult with details of changes made
        """
        result = ReconciliationResult()

        logger.info("Starting order reconciliation with exchange...")

        # 1. Get local active orders
        local_orders = self._state_manager.get_active_orders()
        logger.info(f"Found {len(local_orders)} local active orders")

        if not local_orders:
            logger.info("No local orders to reconcile")
            return result

        # 2. Get open orders from exchange
        try:
            exchange_orders = self._client.get_open_orders(trades=True)
            logger.info(f"Found {len(exchange_orders)} orders on exchange")
        except Exception as e:
            logger.error(f"Failed to fetch open orders from exchange: {e}")
            raise

        # 3. Get last fill timestamp for trade query
        last_fill_time = self._get_last_fill_timestamp()
        logger.info(f"Last known fill timestamp: {last_fill_time}")

        # 4. Get trades since last fill
        try:
            # Query trades from a bit before last fill to ensure we don't miss any
            start_time = int(last_fill_time - 3600) if last_fill_time else None
            recent_trades = self._client.get_trades_history(start=start_time)
            logger.info(f"Found {len(recent_trades)} trades since last fill")
        except Exception as e:
            logger.error(f"Failed to fetch trade history: {e}")
            recent_trades = {}

        # 5. Build lookup maps
        exchange_order_map = {oid: info for oid, info in exchange_orders.items()}
        trades_by_order = self._group_trades_by_order(recent_trades)

        # 6. Reconcile each local order
        for local_order in local_orders:
            self._reconcile_order(
                local_order,
                exchange_order_map,
                trades_by_order,
                result,
            )

        # 7. Check for orphaned exchange orders (exist on exchange but not locally)
        local_exchange_ids = {
            o.get("exchange_id")
            for o in local_orders
            if o.get("exchange_id")
        }
        for exchange_id, order_info in exchange_orders.items():
            if exchange_id not in local_exchange_ids:
                # This is an orphaned exchange order
                if self._is_our_pair(order_info):
                    result.orphaned_exchange += 1
                    result.orphaned_exchange_ids.append(exchange_id)
                    result.actions.append(
                        f"Found orphaned exchange order: {exchange_id}"
                    )
                    logger.warning(
                        f"Found orphaned order on exchange: {exchange_id} "
                        f"({order_info.side.value} {order_info.volume} @ {order_info.price})"
                    )

        logger.info(f"Reconciliation complete: {result}")
        return result

    def _reconcile_order(
        self,
        local_order: Dict[str, Any],
        exchange_orders: Dict[str, OrderInfo],
        trades_by_order: Dict[str, List[TradeInfo]],
        result: ReconciliationResult,
    ) -> None:
        """
        Reconcile a single local order with exchange state.

        Args:
            local_order: Local order from database
            exchange_orders: Map of exchange order ID to OrderInfo
            trades_by_order: Map of exchange order ID to list of TradeInfo
            result: Result object to update
        """
        grid_id = local_order.get("grid_id")
        exchange_id = local_order.get("exchange_id")
        local_state = local_order.get("state")

        logger.debug(f"Reconciling order {grid_id} (exchange_id={exchange_id}, state={local_state})")

        # Case 1: Order has exchange ID and is still on exchange
        if exchange_id and exchange_id in exchange_orders:
            exchange_info = exchange_orders[exchange_id]
            self._sync_order_with_exchange(grid_id, local_order, exchange_info, result)
            return

        # Case 2: Order has exchange ID but not on exchange (filled or canceled)
        if exchange_id:
            # Check if it was filled
            fills = trades_by_order.get(exchange_id, [])
            if fills:
                self._process_missed_fills(grid_id, local_order, fills, result)
            else:
                # Order was canceled or expired
                self._mark_order_canceled(grid_id, local_order, result)
            return

        # Case 3: Order was never submitted (no exchange ID)
        if local_state in ("PENDING_SUBMIT", "SUBMITTED"):
            # This order was never confirmed on exchange
            self._mark_order_failed(grid_id, local_order, result)

    def _sync_order_with_exchange(
        self,
        grid_id: str,
        local_order: Dict[str, Any],
        exchange_info: OrderInfo,
        result: ReconciliationResult,
    ) -> None:
        """Sync local order state with exchange state."""
        result.matched_orders += 1
        result.matched_order_ids.append(grid_id)

        # Check for partial fills
        if exchange_info.volume_executed > 0:
            local_filled = Decimal(local_order.get("filled_volume", "0") or "0")
            new_fills = exchange_info.volume_executed - local_filled

            if new_fills > 0:
                result.partially_filled += 1
                result.partially_filled_ids.append(grid_id)

                # Record position update
                result.position_fills.append({
                    "grid_id": grid_id,
                    "side": local_order.get("side"),
                    "volume": str(new_fills),
                    "price": str(exchange_info.price) if exchange_info.price else local_order.get("price"),
                    "fee": str(exchange_info.fee),
                })

                result.actions.append(
                    f"Updated partial fill for {grid_id}: +{new_fills} filled"
                )

        # Update order state if needed
        if exchange_info.status == OrderStatus.OPEN:
            if exchange_info.volume_executed > 0:
                new_state = "PARTIALLY_FILLED"
            else:
                new_state = "OPEN"
        else:
            new_state = exchange_info.status.value.upper()

        if local_order.get("state") != new_state:
            self._state_manager.update_order_state(grid_id, new_state)
            result.actions.append(f"Updated {grid_id} state: {local_order.get('state')} -> {new_state}")

    def _process_missed_fills(
        self,
        grid_id: str,
        local_order: Dict[str, Any],
        fills: List[TradeInfo],
        result: ReconciliationResult,
    ) -> None:
        """Process fills that occurred while bot was down."""
        result.filled_during_downtime += 1
        result.filled_order_ids.append(grid_id)

        total_volume = sum(fill.volume for fill in fills)
        total_cost = sum(fill.cost for fill in fills)
        total_fee = sum(fill.fee for fill in fills)
        avg_price = total_cost / total_volume if total_volume > 0 else Decimal("0")

        # Record position update
        result.position_fills.append({
            "grid_id": grid_id,
            "side": local_order.get("side"),
            "volume": str(total_volume),
            "price": str(avg_price),
            "fee": str(total_fee),
        })

        # Update order state
        self._state_manager.update_order_state(grid_id, "FILLED")

        # Save fills to database
        for fill in fills:
            self._state_manager.save_fill(
                order_id=grid_id,
                fill_volume=fill.volume,
                fill_price=fill.price,
                fee=fill.fee,
                timestamp=datetime.fromtimestamp(fill.timestamp),
            )

        result.actions.append(
            f"Processed missed fills for {grid_id}: {total_volume} @ {avg_price}"
        )
        logger.info(f"Order {grid_id} was filled during downtime: {total_volume} @ {avg_price}")

    def _mark_order_canceled(
        self,
        grid_id: str,
        local_order: Dict[str, Any],
        result: ReconciliationResult,
    ) -> None:
        """Mark order as canceled (not found on exchange and no fills)."""
        result.orphaned_local += 1
        result.orphaned_local_ids.append(grid_id)

        self._state_manager.update_order_state(grid_id, "CANCELED")
        result.actions.append(f"Marked {grid_id} as CANCELED (not found on exchange)")
        logger.info(f"Order {grid_id} not found on exchange, marked as CANCELED")

    def _mark_order_failed(
        self,
        grid_id: str,
        local_order: Dict[str, Any],
        result: ReconciliationResult,
    ) -> None:
        """Mark order as failed (never made it to exchange)."""
        result.orphaned_local += 1
        result.orphaned_local_ids.append(grid_id)

        self._state_manager.update_order_state(grid_id, "FAILED")
        result.actions.append(f"Marked {grid_id} as FAILED (never submitted)")
        logger.info(f"Order {grid_id} was never submitted, marked as FAILED")

    def _get_last_fill_timestamp(self) -> Optional[float]:
        """Get timestamp of last known fill from database."""
        fills = self._state_manager.get_fill_history(limit=1)
        if fills:
            timestamp_str = fills[0].get("timestamp")
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    return dt.timestamp()
                except ValueError:
                    pass
        return None

    def _group_trades_by_order(
        self,
        trades: Dict[str, TradeInfo],
    ) -> Dict[str, List[TradeInfo]]:
        """Group trades by their order ID."""
        by_order: Dict[str, List[TradeInfo]] = {}
        for trade_id, trade_info in trades.items():
            order_id = trade_info.order_id
            if order_id not in by_order:
                by_order[order_id] = []
            by_order[order_id].append(trade_info)
        return by_order

    def _is_our_pair(self, order_info: OrderInfo) -> bool:
        """Check if order is for our trading pair."""
        # Kraken pair names can vary (XBTUSD, XBT/USD, XXBTZUSD)
        pair = order_info.pair.upper().replace("/", "")
        target = self._trading_pair.upper().replace("/", "")
        return pair == target or pair.replace("X", "").replace("Z", "") == target


def get_fill_history_with_limit(
    state_manager: StateManager,
    limit: int = 1,
) -> List[Dict[str, Any]]:
    """
    Helper to get fill history with a limit.

    The StateManager doesn't have a limit parameter on get_fill_history,
    so we use this helper.
    """
    # Get all fills ordered by timestamp DESC, take first N
    fills = state_manager.get_fill_history()
    return fills[:limit]
