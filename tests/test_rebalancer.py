"""
Tests for Rebalancer module.

Tests:
- Price drift detection
- Position drift detection
- Regime change triggers
- Fill imbalance detection
- Minimum interval enforcement
- Urgency calculation
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock
import time

from src.grid import (
    Rebalancer,
    DriftMetrics,
    RebalanceReason,
    RebalanceDecision,
    GridState,
    GridLevel,
)
from src.api import GridOrderType, GridPosition, OrderManager
from src.regime import MarketRegime


class TestDriftMetrics:
    """Tests for DriftMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating drift metrics."""
        metrics = DriftMetrics(
            price_drift_percent=5.0,
            position_drift_percent=30.0,
            fill_imbalance=2.0,
            time_since_last_rebalance=600.0,
            levels_filled=3,
            levels_remaining=7,
            current_price=Decimal("52500"),
            grid_center=Decimal("50000"),
        )

        assert metrics.price_drift_percent == 5.0
        assert metrics.position_drift_percent == 30.0
        assert metrics.fill_ratio == 0.3  # 3/10

    def test_is_price_outside_grid(self):
        """Test price outside grid detection."""
        # Price within grid (5% drift, not outside)
        metrics = DriftMetrics(
            price_drift_percent=5.0,
            position_drift_percent=0.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=0.0,
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("52500"),
            grid_center=Decimal("50000"),
        )
        assert metrics.is_price_outside_grid is False

        # Price outside grid (>100% drift)
        metrics_outside = DriftMetrics(
            price_drift_percent=150.0,
            position_drift_percent=0.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=0.0,
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("75000"),
            grid_center=Decimal("50000"),
        )
        assert metrics_outside.is_price_outside_grid is True


class TestRebalanceDecision:
    """Tests for RebalanceDecision dataclass."""

    def test_no_rebalance_decision(self):
        """Test no rebalance decision."""
        decision = RebalanceDecision(
            should_rebalance=False,
            details="All good",
        )

        assert decision.should_rebalance is False
        assert "no rebalance" in str(decision).lower()

    def test_rebalance_decision(self):
        """Test rebalance decision."""
        decision = RebalanceDecision(
            should_rebalance=True,
            reason=RebalanceReason.PRICE_DRIFT,
            urgency=0.8,
            suggested_action="full_rebuild",
            details="Price moved 15%",
        )

        assert decision.should_rebalance is True
        assert decision.reason == RebalanceReason.PRICE_DRIFT
        assert "REBALANCE" in str(decision)


class TestRebalancer:
    """Tests for Rebalancer class."""

    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer with default settings."""
        rb = Rebalancer(
            rebalance_threshold=0.1,  # 10%
            position_threshold=0.5,  # 50%
            imbalance_threshold=3.0,  # 3:1
            min_rebalance_interval=300.0,  # 5 min
            max_capital=Decimal("400"),
        )
        # Set last rebalance time to past so interval is met
        rb._last_rebalance_time = time.time() - 600
        return rb

    @pytest.fixture
    def sample_grid_state(self):
        """Create sample grid state."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(9, Decimal("52500"), GridOrderType.SELL, Decimal("0.001")),
        ]
        return GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("100"),
            total_sell_exposure=Decimal("100"),
        )

    @pytest.fixture
    def mock_position(self):
        """Create mock position."""
        return GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0"),
            buy_count=0,
            sell_count=0,
        )

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        manager = Mock(spec=OrderManager)
        manager.get_active_orders.return_value = []
        manager.get_filled_orders.return_value = []
        return manager

    def test_compute_drift_metrics(
        self, rebalancer, sample_grid_state, mock_position, mock_order_manager
    ):
        """Test drift metrics computation."""
        current_price = Decimal("52000")  # 4% above center

        metrics = rebalancer.compute_drift_metrics(
            sample_grid_state, current_price, mock_position, mock_order_manager
        )

        assert metrics.price_drift_percent == pytest.approx(4.0, rel=0.01)
        assert metrics.current_price == current_price
        assert metrics.grid_center == Decimal("50000")

    def test_should_rebalance_price_drift(self, rebalancer, sample_grid_state):
        """Test rebalance triggers on price drift."""
        # 15% price drift, above 10% threshold
        metrics = DriftMetrics(
            price_drift_percent=15.0,
            position_drift_percent=0.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=600.0,
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("57500"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics, MarketRegime.RANGING, MarketRegime.RANGING
        )

        assert decision.should_rebalance is True
        assert decision.reason == RebalanceReason.PRICE_DRIFT

    def test_should_rebalance_position_drift(self, rebalancer):
        """Test rebalance triggers on position drift."""
        # 60% position drift, above 50% threshold
        metrics = DriftMetrics(
            price_drift_percent=5.0,  # Below threshold
            position_drift_percent=60.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=600.0,
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("52500"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics, MarketRegime.RANGING, MarketRegime.RANGING
        )

        assert decision.should_rebalance is True
        assert decision.reason == RebalanceReason.POSITION_DRIFT

    def test_should_rebalance_regime_change(self, rebalancer):
        """Test rebalance triggers on regime change."""
        metrics = DriftMetrics(
            price_drift_percent=2.0,  # Below threshold
            position_drift_percent=10.0,  # Below threshold
            fill_imbalance=1.0,  # Balanced
            time_since_last_rebalance=600.0,
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("51000"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics,
            MarketRegime.TRENDING_UP,  # Current
            MarketRegime.RANGING,  # Previous (different)
        )

        assert decision.should_rebalance is True
        assert decision.reason == RebalanceReason.REGIME_CHANGE

    def test_should_rebalance_fill_imbalance(self, rebalancer):
        """Test rebalance triggers on fill imbalance."""
        # 4:1 buy/sell imbalance, above 3:1 threshold
        metrics = DriftMetrics(
            price_drift_percent=2.0,
            position_drift_percent=10.0,
            fill_imbalance=4.0,
            time_since_last_rebalance=600.0,
            levels_filled=5,
            levels_remaining=5,
            current_price=Decimal("51000"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics, MarketRegime.RANGING, MarketRegime.RANGING
        )

        assert decision.should_rebalance is True
        assert decision.reason == RebalanceReason.FILL_IMBALANCE

    def test_should_not_rebalance_below_thresholds(self, rebalancer):
        """Test no rebalance when all metrics below thresholds."""
        metrics = DriftMetrics(
            price_drift_percent=5.0,  # Below 10%
            position_drift_percent=20.0,  # Below 50%
            fill_imbalance=1.5,  # Below 3:1
            time_since_last_rebalance=600.0,
            levels_filled=2,
            levels_remaining=8,
            current_price=Decimal("52500"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics, MarketRegime.RANGING, MarketRegime.RANGING
        )

        assert decision.should_rebalance is False

    def test_min_interval_respected(self, rebalancer):
        """Test minimum interval is respected."""
        # Set last rebalance to very recent
        rebalancer._last_rebalance_time = time.time() - 60  # 1 min ago

        metrics = DriftMetrics(
            price_drift_percent=50.0,  # Way above threshold
            position_drift_percent=0.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=60.0,  # Only 1 min
            levels_filled=0,
            levels_remaining=10,
            current_price=Decimal("75000"),
            grid_center=Decimal("50000"),
        )

        decision = rebalancer.should_rebalance(
            metrics, MarketRegime.RANGING, MarketRegime.RANGING
        )

        # Should not rebalance because min interval not met
        assert decision.should_rebalance is False
        assert "interval" in decision.details.lower()

    def test_calculate_urgency(self, rebalancer):
        """Test urgency calculation."""
        # Low drift = low urgency
        low_metrics = DriftMetrics(
            price_drift_percent=5.0,
            position_drift_percent=10.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=600.0,
            levels_filled=1,
            levels_remaining=9,
            current_price=Decimal("52500"),
            grid_center=Decimal("50000"),
        )
        low_urgency = rebalancer.calculate_rebalance_urgency(low_metrics)

        # High drift = high urgency
        high_metrics = DriftMetrics(
            price_drift_percent=20.0,
            position_drift_percent=80.0,
            fill_imbalance=1.0,
            time_since_last_rebalance=600.0,
            levels_filled=8,
            levels_remaining=2,
            current_price=Decimal("60000"),
            grid_center=Decimal("50000"),
        )
        high_urgency = rebalancer.calculate_rebalance_urgency(high_metrics)

        assert high_urgency > low_urgency
        assert 0 <= low_urgency <= 1
        assert 0 <= high_urgency <= 1

    def test_get_suggested_new_center_neutral(
        self, rebalancer, sample_grid_state, mock_position
    ):
        """Test suggested center with no position."""
        current_price = Decimal("52000")

        new_center = rebalancer.get_suggested_new_center(
            sample_grid_state, current_price, mock_position
        )

        # With no position, should be close to current price
        assert new_center == current_price

    def test_get_suggested_new_center_long(
        self, rebalancer, sample_grid_state
    ):
        """Test suggested center shifts up with long position."""
        position = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("0.01"),  # Long
            avg_entry_price=Decimal("49000"),
        )
        current_price = Decimal("50000")

        new_center = rebalancer.get_suggested_new_center(
            sample_grid_state, current_price, position
        )

        # Long position should shift center up
        assert new_center > current_price

    def test_get_suggested_new_center_short(
        self, rebalancer, sample_grid_state
    ):
        """Test suggested center shifts down with short position."""
        position = GridPosition(
            pair="XBTUSD",
            base_asset="XBT",
            quote_asset="USD",
            quantity=Decimal("-0.01"),  # Short
            avg_entry_price=Decimal("51000"),
        )
        current_price = Decimal("50000")

        new_center = rebalancer.get_suggested_new_center(
            sample_grid_state, current_price, position
        )

        # Short position should shift center down
        assert new_center < current_price

    def test_mark_rebalanced(self, rebalancer):
        """Test rebalance marking."""
        assert rebalancer.rebalance_count == 0

        rebalancer.mark_rebalanced()

        assert rebalancer.rebalance_count == 1
        assert rebalancer.last_rebalance_time > 0

    def test_reset_stats(self, rebalancer):
        """Test stats reset."""
        rebalancer.mark_rebalanced()
        rebalancer.mark_rebalanced()

        assert rebalancer.rebalance_count == 2

        rebalancer.reset_stats()

        assert rebalancer.rebalance_count == 0
        assert rebalancer.last_rebalance_time == 0


class TestRebalancerRegimeUrgency:
    """Tests for regime change urgency calculation."""

    @pytest.fixture
    def rebalancer(self):
        rb = Rebalancer()
        rb._last_rebalance_time = time.time() - 600
        return rb

    def test_high_urgency_transitions(self, rebalancer):
        """Test high urgency regime transitions."""
        # RANGING -> BREAKOUT is high urgency
        urgency = rebalancer._calculate_regime_change_urgency(
            MarketRegime.BREAKOUT, MarketRegime.RANGING
        )
        assert urgency == 0.9

        # TRENDING_UP -> TRENDING_DOWN is high urgency
        urgency = rebalancer._calculate_regime_change_urgency(
            MarketRegime.TRENDING_DOWN, MarketRegime.TRENDING_UP
        )
        assert urgency == 0.9

    def test_low_urgency_transitions(self, rebalancer):
        """Test low urgency regime transitions (recovery)."""
        # BREAKOUT -> RANGING is recovery (low urgency)
        urgency = rebalancer._calculate_regime_change_urgency(
            MarketRegime.RANGING, MarketRegime.BREAKOUT
        )
        assert urgency == 0.3

        # HIGH_VOLATILITY -> RANGING is recovery
        urgency = rebalancer._calculate_regime_change_urgency(
            MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY
        )
        assert urgency == 0.3
