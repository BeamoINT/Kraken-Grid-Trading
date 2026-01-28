"""
Tests for Order Manager.

Tests cover:
- Grid order creation
- Order state management
- Position tracking
- P&L calculation
"""

from decimal import Decimal
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.api.order_manager import (
    OrderManager,
    OrderManagerConfig,
    GridOrder,
    GridOrderType,
    OrderState,
    GridPosition,
)
from src.api.kraken_private import (
    KrakenPrivateClient,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderInfo,
)


class TestGridOrder:
    """Tests for GridOrder dataclass."""

    def test_create_grid_order(self):
        """Test creating a grid order."""
        order = GridOrder(
            grid_id="grid_1_buy_12345",
            level=1,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        assert order.grid_id == "grid_1_buy_12345"
        assert order.level == 1
        assert order.price == Decimal("50000")
        assert order.side == GridOrderType.BUY
        assert order.state == OrderState.PENDING_SUBMIT

    def test_remaining_volume(self):
        """Test remaining volume calculation."""
        order = GridOrder(
            grid_id="test",
            level=0,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
            filled_volume=Decimal("0.004"),
        )

        assert order.remaining_volume == Decimal("0.006")

    def test_is_active_states(self):
        """Test is_active for different states."""
        order = GridOrder(
            grid_id="test",
            level=0,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )

        # PENDING_SUBMIT is not active
        assert not order.is_active

        # SUBMITTED is active
        order.state = OrderState.SUBMITTED
        assert order.is_active

        # OPEN is active
        order.state = OrderState.OPEN
        assert order.is_active

        # PARTIALLY_FILLED is active
        order.state = OrderState.PARTIALLY_FILLED
        assert order.is_active

        # FILLED is not active
        order.state = OrderState.FILLED
        assert not order.is_active

        # CANCELED is not active
        order.state = OrderState.CANCELED
        assert not order.is_active

    def test_is_complete(self):
        """Test is_complete for different states."""
        order = GridOrder(
            grid_id="test",
            level=0,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )

        # PENDING_SUBMIT is not complete
        assert not order.is_complete

        # FILLED is complete
        order.state = OrderState.FILLED
        assert order.is_complete

        # CANCELED is complete
        order.state = OrderState.CANCELED
        assert order.is_complete

        # FAILED is complete
        order.state = OrderState.FAILED
        assert order.is_complete


class TestGridPosition:
    """Tests for GridPosition class."""

    def test_initial_position(self):
        """Test initial position is flat."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
        )

        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == Decimal("0")
        assert pos.buy_count == 0
        assert pos.sell_count == 0

    def test_buy_opens_long(self):
        """Test buy from flat opens long position."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
        )

        realized = pos.update_from_fill(
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
            price=Decimal("50000"),
            fee=Decimal("0.50"),
        )

        assert pos.quantity == Decimal("0.01")
        assert pos.total_cost == Decimal("500")  # 0.01 * 50000
        assert pos.avg_entry_price == Decimal("50000")
        assert pos.buy_count == 1
        assert realized == Decimal("0")  # No realized P&L on open

    def test_sell_closes_long_with_profit(self):
        """Test sell closes long position with profit."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0.01"),
            total_cost=Decimal("500"),
            avg_entry_price=Decimal("50000"),
        )

        realized = pos.update_from_fill(
            side=GridOrderType.SELL,
            volume=Decimal("0.01"),
            price=Decimal("51000"),  # Sold at higher price
            fee=Decimal("0.50"),
        )

        assert pos.quantity == Decimal("0")
        assert realized == Decimal("10")  # (51000 - 50000) * 0.01

    def test_sell_closes_long_with_loss(self):
        """Test sell closes long position with loss."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0.01"),
            total_cost=Decimal("500"),
            avg_entry_price=Decimal("50000"),
        )

        realized = pos.update_from_fill(
            side=GridOrderType.SELL,
            volume=Decimal("0.01"),
            price=Decimal("49000"),  # Sold at lower price
            fee=Decimal("0.50"),
        )

        assert pos.quantity == Decimal("0")
        assert realized == Decimal("-10")  # (49000 - 50000) * 0.01

    def test_partial_close(self):
        """Test partial close of position."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0.02"),
            total_cost=Decimal("1000"),
            avg_entry_price=Decimal("50000"),
        )

        realized = pos.update_from_fill(
            side=GridOrderType.SELL,
            volume=Decimal("0.01"),  # Half position
            price=Decimal("52000"),
            fee=Decimal("0.50"),
        )

        assert pos.quantity == Decimal("0.01")  # Half remaining
        assert pos.total_cost == Decimal("500")  # Half cost remaining
        assert realized == Decimal("20")  # Profit on closed portion

    def test_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0.01"),
            total_cost=Decimal("500"),
        )

        pos.update_unrealized_pnl(Decimal("52000"))

        # Position value: 0.01 * 52000 = 520
        # Cost: 500
        # Unrealized: 520 - 500 = 20
        assert pos.unrealized_pnl == Decimal("20")

    def test_unrealized_pnl_flat(self):
        """Test unrealized P&L is zero when flat."""
        pos = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
        )

        pos.update_unrealized_pnl(Decimal("50000"))
        assert pos.unrealized_pnl == Decimal("0")


class TestOrderManager:
    """Tests for OrderManager class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Kraken client."""
        client = Mock(spec=KrakenPrivateClient)
        client.add_order = MagicMock(return_value={
            "descr": {"order": "buy 0.001 XBTUSD @ limit 50000"},
            "txid": ["ORDER-ABC-123"],
        })
        client.cancel_order = MagicMock(return_value={"count": 1})
        client.query_orders = MagicMock(return_value={})
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create order manager with mock client."""
        return OrderManager(mock_client)

    def test_create_grid_order(self, manager):
        """Test creating a grid order."""
        order = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        assert order.level == 5
        assert order.price == Decimal("50000.0")  # Rounded
        assert order.side == GridOrderType.BUY
        assert order.state == OrderState.PENDING_SUBMIT

    def test_submit_order_success(self, manager, mock_client):
        """Test successful order submission."""
        order = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        result = manager.submit_order(order)

        assert result is True
        assert order.order_id == "ORDER-ABC-123"
        assert order.state == OrderState.SUBMITTED
        mock_client.add_order.assert_called_once()

    def test_submit_order_insufficient_funds(self, manager, mock_client):
        """Test order submission with insufficient funds."""
        from src.api.kraken_errors import InsufficientFundsError

        mock_client.add_order.side_effect = InsufficientFundsError(
            ["EOrder:Insufficient funds"]
        )

        order = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        result = manager.submit_order(order)

        assert result is False
        assert order.state == OrderState.FAILED

    def test_cancel_order_success(self, manager, mock_client):
        """Test successful order cancellation."""
        order = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        # Submit first
        manager.submit_order(order)
        order.state = OrderState.OPEN  # Simulate confirmation

        # Cancel
        result = manager.cancel_order(order.grid_id)

        assert result is True
        assert order.state == OrderState.CANCELED

    def test_cancel_never_submitted_order(self, manager):
        """Test canceling an order that was never submitted."""
        order = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        result = manager.cancel_order(order.grid_id)

        assert result is True
        assert order.state == OrderState.CANCELED

    def test_get_active_orders(self, manager, mock_client):
        """Test getting active orders."""
        order1 = manager.create_grid_order(
            level=1,
            price=Decimal("49000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )
        order2 = manager.create_grid_order(
            level=2,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        manager.submit_order(order1)
        order1.state = OrderState.OPEN

        # order2 still pending
        active = manager.get_active_orders()

        assert len(active) == 1
        assert active[0].grid_id == order1.grid_id

    def test_get_orders_by_level(self, manager):
        """Test filtering orders by level."""
        order1 = manager.create_grid_order(
            level=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )
        order2 = manager.create_grid_order(
            level=5,
            price=Decimal("51000"),
            side=GridOrderType.SELL,
            volume=Decimal("0.001"),
        )
        order3 = manager.create_grid_order(
            level=6,
            price=Decimal("52000"),
            side=GridOrderType.SELL,
            volume=Decimal("0.001"),
        )

        orders = manager.get_orders_by_level(5)

        assert len(orders) == 2
        assert all(o.level == 5 for o in orders)

    def test_get_open_order_exposure(self, manager, mock_client):
        """Test calculating open order exposure."""
        buy_order = manager.create_grid_order(
            level=1,
            price=Decimal("49000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )
        sell_order = manager.create_grid_order(
            level=2,
            price=Decimal("51000"),
            side=GridOrderType.SELL,
            volume=Decimal("0.01"),
        )

        # Submit and make active
        manager.submit_order(buy_order)
        manager.submit_order(sell_order)
        buy_order.state = OrderState.OPEN
        sell_order.state = OrderState.OPEN

        buy_exp, sell_exp = manager.get_open_order_exposure()

        assert buy_exp == Decimal("490")   # 0.01 * 49000
        assert sell_exp == Decimal("510")  # 0.01 * 51000

    def test_get_stats(self, manager, mock_client):
        """Test getting manager statistics."""
        order = manager.create_grid_order(
            level=1,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )
        manager.submit_order(order)
        order.state = OrderState.OPEN

        stats = manager.get_stats()

        assert stats["total_orders"] == 1
        assert stats["active_orders"] == 1
        assert stats["filled_orders"] == 0
        assert "position" in stats
        assert "totals" in stats

    def test_clear_completed_orders(self, manager, mock_client):
        """Test clearing completed orders."""
        order1 = manager.create_grid_order(
            level=1,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )
        order2 = manager.create_grid_order(
            level=2,
            price=Decimal("51000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )

        manager.submit_order(order1)
        manager.submit_order(order2)

        order1.state = OrderState.FILLED
        order2.state = OrderState.OPEN

        cleared = manager.clear_completed_orders()

        assert cleared == 1
        assert order1.grid_id not in manager._orders
        assert order2.grid_id in manager._orders

    def test_fill_callback(self, mock_client):
        """Test fill callback is triggered."""
        fills = []

        def on_fill(order, volume, price):
            fills.append((order.grid_id, volume, price))

        manager = OrderManager(mock_client, on_fill=on_fill)

        order = manager.create_grid_order(
            level=1,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.01"),
        )
        manager.submit_order(order)

        # Simulate fill via _update_from_exchange
        order_info = OrderInfo(
            order_id=order.order_id,
            pair="XBTUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
            volume=Decimal("0.01"),
            volume_executed=Decimal("0.01"),
            cost=Decimal("500"),
            fee=Decimal("0.50"),
            status=OrderStatus.CLOSED,
            open_time=1234567890.0,
        )

        manager._update_from_exchange(order, order_info)

        assert len(fills) == 1
        assert fills[0][0] == order.grid_id
        assert fills[0][1] == Decimal("0.01")
