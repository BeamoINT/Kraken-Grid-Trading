"""
Tests for GridExecutor module.

Tests:
- Grid deployment
- Order submission
- Fill handling
- Stop-loss execution
- Pause/resume
- Snapshot/restore
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import time

from src.grid import (
    GridExecutor,
    GridStrategy,
    GridCalculator,
    GridState,
    GridLevel,
    GridParameters,
    ExecutionResult,
    GridSnapshot,
    RegimeAdaptation,
)
from src.api import (
    GridOrder,
    GridOrderType,
    OrderState,
    GridPosition,
)
from src.regime import MarketRegime
from config.settings import GridConfig, RiskConfig, GridSpacing


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful result creation."""
        result = ExecutionResult(
            success=True,
            orders_submitted=5,
            orders_canceled=2,
        )

        assert result.success is True
        assert result.total_actions == 7
        assert "SUCCESS" in str(result)

    def test_failed_result(self):
        """Test failed result creation."""
        result = ExecutionResult(
            success=False,
            orders_failed=3,
            errors=["Order rejected", "Insufficient funds"],
        )

        assert result.success is False
        assert result.orders_failed == 3
        assert "FAILED" in str(result)


class TestGridExecutor:
    """Tests for GridExecutor class."""

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock OrderManager."""
        manager = Mock()
        manager.position = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
        )
        manager.get_active_orders.return_value = []
        manager.get_filled_orders.return_value = []
        manager.cancel_all_grid_orders.return_value = (0, 0)

        def create_order(level, price, side, volume):
            return GridOrder(
                grid_id=f"grid_{level}_{side.name}_{int(time.time() * 1000)}",
                level=level,
                price=price,
                side=side,
                volume=volume,
            )

        manager.create_grid_order.side_effect = create_order
        manager.submit_order.return_value = True
        manager.cancel_order.return_value = True

        return manager

    @pytest.fixture
    def mock_strategy(self):
        """Create mock GridStrategy."""
        strategy = Mock(spec=GridStrategy)
        strategy.compute_stop_loss_price.return_value = Decimal("40000")
        return strategy

    @pytest.fixture
    def risk_config(self):
        """Create risk config."""
        return RiskConfig(
            max_position_percent=70.0,
            max_open_orders=20,
        )

    @pytest.fixture
    def executor(self, mock_order_manager, mock_strategy, risk_config):
        """Create executor instance."""
        return GridExecutor(
            order_manager=mock_order_manager,
            strategy=mock_strategy,
            risk_config=risk_config,
            capital=Decimal("400"),
        )

    @pytest.fixture
    def sample_grid_state(self):
        """Create sample grid state."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(1, Decimal("48000"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(2, Decimal("48500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(3, Decimal("51500"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(4, Decimal("52000"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(5, Decimal("52500"), GridOrderType.SELL, Decimal("0.001")),
        ]

        return GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("150"),
            total_sell_exposure=Decimal("150"),
        )

    def test_initial_state(self, executor):
        """Test initial executor state."""
        assert executor.current_grid is None
        assert executor.is_trading is False
        assert executor.is_paused is False
        assert executor.stop_loss_price is None

    def test_deploy_grid_success(self, executor, sample_grid_state, mock_order_manager):
        """Test successful grid deployment."""
        adaptation = RegimeAdaptation()

        result = executor.deploy_grid(
            sample_grid_state, adaptation,
            MarketRegime.RANGING, 0.85
        )

        assert result.success is True
        assert result.orders_submitted == 6  # All 6 levels
        assert executor.is_trading is True
        assert executor.current_grid is not None

    def test_deploy_grid_sets_stop_loss(self, executor, sample_grid_state, mock_strategy):
        """Test that deployment sets stop-loss."""
        adaptation = RegimeAdaptation()

        executor.deploy_grid(
            sample_grid_state, adaptation,
            MarketRegime.RANGING, 0.85
        )

        mock_strategy.compute_stop_loss_price.assert_called_once()
        assert executor.stop_loss_price == Decimal("40000")

    def test_deploy_grid_validation_failure(self, executor, risk_config):
        """Test deployment fails with over-exposure."""
        # Create grid with too much exposure
        levels = [
            GridLevel(0, Decimal("50000"), GridOrderType.BUY, Decimal("0.01")),
        ]
        grid_state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("50000"),
            lower_bound=Decimal("50000"),
            total_buy_exposure=Decimal("500"),  # Exceeds 70% of $400
            total_sell_exposure=Decimal("0"),
        )

        result = executor.deploy_grid(
            grid_state, RegimeAdaptation(),
            MarketRegime.RANGING, 0.85
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "exceeds" in result.errors[0].lower()

    def test_deploy_grid_respects_adaptation(self, executor, mock_order_manager):
        """Test deployment respects level activation in adaptation."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(1, Decimal("52500"), GridOrderType.SELL, Decimal("0.001")),
        ]
        grid_state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("50"),
            total_sell_exposure=Decimal("50"),
        )

        # Disable buys
        adaptation = RegimeAdaptation(buy_levels_active=False)

        result = executor.deploy_grid(
            grid_state, adaptation,
            MarketRegime.TRENDING_DOWN, 0.85
        )

        # Only 1 sell order should be submitted
        assert result.orders_submitted == 1

    def test_update_grid(self, executor, sample_grid_state, mock_order_manager):
        """Test grid update with diff."""
        adaptation = RegimeAdaptation()

        # First deploy
        executor.deploy_grid(sample_grid_state, adaptation)

        # Create updated grid with one level changed
        new_levels = list(sample_grid_state.levels)
        new_levels[0] = GridLevel(0, Decimal("47000"), GridOrderType.BUY, Decimal("0.001"))  # Changed price

        new_grid = GridState(
            levels=new_levels,
            center_price=Decimal("49000"),  # Shifted
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47000"),
            total_buy_exposure=Decimal("150"),
            total_sell_exposure=Decimal("150"),
        )

        result = executor.update_grid(new_grid, adaptation)

        assert result.success is True

    def test_cancel_all_orders(self, executor, sample_grid_state, mock_order_manager):
        """Test cancel all orders."""
        # Deploy first
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())

        mock_order_manager.cancel_all_grid_orders.return_value = (6, 0)

        result = executor.cancel_all_orders()

        assert result.success is True
        assert result.orders_canceled == 6

    def test_pause_trading(self, executor, sample_grid_state, mock_order_manager):
        """Test pause trading."""
        # Deploy first
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())

        mock_order_manager.cancel_all_grid_orders.return_value = (6, 0)

        result = executor.pause_trading("breakout_regime")

        assert result.success is True
        assert executor.is_paused is True
        assert executor.pause_reason == "breakout_regime"
        assert executor.is_trading is False

    def test_resume_trading(self, executor, sample_grid_state, mock_order_manager):
        """Test resume trading after pause."""
        # Deploy and pause
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())
        mock_order_manager.cancel_all_grid_orders.return_value = (6, 0)
        executor.pause_trading("test")

        # Resume
        result = executor.resume_trading()

        assert result.success is True
        assert executor.is_paused is False
        assert executor.is_trading is True

    def test_check_stop_loss_not_triggered(self, executor, sample_grid_state):
        """Test stop-loss not triggered when price is above threshold."""
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())

        triggered = executor.check_stop_loss(Decimal("45000"))  # Above $40000 stop

        assert triggered is False

    def test_check_stop_loss_triggered(self, executor, sample_grid_state, mock_order_manager):
        """Test stop-loss triggers when price drops below threshold."""
        callback_called = []

        def on_stop_loss(price):
            callback_called.append(price)

        executor._on_stop_loss = on_stop_loss
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())

        mock_order_manager.cancel_all_grid_orders.return_value = (6, 0)

        triggered = executor.check_stop_loss(Decimal("39000"))  # Below $40000 stop

        assert triggered is True
        assert len(callback_called) == 1
        assert callback_called[0] == Decimal("39000")

    def test_handle_fill(self, executor, sample_grid_state):
        """Test fill handling."""
        fill_callback_called = []

        def on_fill(order, volume, price):
            fill_callback_called.append((order, volume, price))

        executor._external_on_fill = on_fill
        executor.deploy_grid(sample_grid_state, RegimeAdaptation())

        order = GridOrder(
            grid_id="test_order",
            level=0,
            price=Decimal("47500"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
            state=OrderState.FILLED,
        )

        executor.handle_fill(order, Decimal("0.001"), Decimal("47500"))

        assert len(fill_callback_called) == 1

    def test_get_snapshot(self, executor, sample_grid_state):
        """Test snapshot creation."""
        executor.deploy_grid(
            sample_grid_state, RegimeAdaptation(),
            MarketRegime.RANGING, 0.85
        )

        snapshot = executor.get_snapshot()

        assert snapshot.grid_state is not None
        assert snapshot.regime == MarketRegime.RANGING
        assert snapshot.confidence == 0.85
        assert snapshot.stop_loss_price == Decimal("40000")

    def test_restore_from_snapshot(self, executor, sample_grid_state, mock_order_manager):
        """Test restore from snapshot."""
        mock_order_manager.sync_orders.return_value = {}

        snapshot = GridSnapshot(
            grid_state=sample_grid_state,
            active_order_ids=["order1", "order2"],
            position_quantity=Decimal("0.001"),
            position_avg_price=Decimal("50000"),
            realized_pnl=Decimal("10"),
            regime=MarketRegime.TRENDING_UP,
            confidence=0.9,
            stop_loss_price=Decimal("42000"),
            is_paused=False,
            pause_reason="",
        )

        result = executor.restore_from_snapshot(snapshot)

        assert result.success is True
        assert executor.current_grid == sample_grid_state
        assert executor.stop_loss_price == Decimal("42000")

    def test_get_stats(self, executor, sample_grid_state):
        """Test statistics retrieval."""
        executor.deploy_grid(
            sample_grid_state, RegimeAdaptation(),
            MarketRegime.RANGING, 0.85
        )

        stats = executor.get_stats()

        assert stats["is_trading"] is True
        assert stats["is_paused"] is False
        assert stats["current_regime"] == "RANGING"
        assert stats["confidence"] == 0.85


class TestGridExecutorOrderSubmission:
    """Tests for order submission logic."""

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager with submission tracking."""
        manager = Mock()
        manager.position = GridPosition("XBTUSD", "XBT", "USD")
        manager.get_active_orders.return_value = []
        manager.cancel_all_grid_orders.return_value = (0, 0)

        submitted_orders = []

        def create_order(level, price, side, volume):
            order = GridOrder(
                grid_id=f"grid_{level}",
                level=level,
                price=price,
                side=side,
                volume=volume,
            )
            return order

        def submit_order(order):
            submitted_orders.append(order)
            return True

        manager.create_grid_order.side_effect = create_order
        manager.submit_order.side_effect = submit_order
        manager.submitted_orders = submitted_orders

        return manager

    def test_orders_submitted_for_active_levels(self, mock_order_manager):
        """Test orders are submitted only for active levels."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001"), is_active=True),
            GridLevel(1, Decimal("48000"), GridOrderType.BUY, Decimal("0.001"), is_active=False),  # Inactive
            GridLevel(2, Decimal("52000"), GridOrderType.SELL, Decimal("0.001"), is_active=True),
        ]
        grid_state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52000"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("100"),
            total_sell_exposure=Decimal("50"),
        )

        strategy = Mock(spec=GridStrategy)
        strategy.compute_stop_loss_price.return_value = Decimal("40000")

        executor = GridExecutor(
            mock_order_manager, strategy, RiskConfig(), Decimal("400")
        )

        result = executor.deploy_grid(
            grid_state, RegimeAdaptation(),
            MarketRegime.RANGING, 0.85
        )

        # Only 2 orders should be submitted (level 1 is inactive)
        assert result.orders_submitted == 2
        assert len(mock_order_manager.submitted_orders) == 2
