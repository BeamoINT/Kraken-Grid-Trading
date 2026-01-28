"""
Tests for GridStrategy module.

Tests regime-based grid adaptations:
- RANGING: Tighter spacing
- TRENDING_UP: Shift up, wider spacing
- TRENDING_DOWN: Pause buys
- HIGH_VOLATILITY: Wide spacing
- BREAKOUT: Pause trading
"""

import pytest
from decimal import Decimal

from src.grid import (
    GridCalculator,
    GridStrategy,
    GridParameters,
    GridState,
    GridLevel,
    RegimeAdaptation,
)
from src.regime import MarketRegime
from src.api import GridOrderType
from config.settings import GridConfig, RiskConfig, GridSpacing


class TestRegimeAdaptation:
    """Tests for RegimeAdaptation dataclass."""

    def test_default_adaptation(self):
        """Test default adaptation values."""
        adaptation = RegimeAdaptation()

        assert adaptation.spacing_multiplier == 1.0
        assert adaptation.shift_percent == 0.0
        assert adaptation.buy_levels_active is True
        assert adaptation.sell_levels_active is True
        assert adaptation.pause_trading is False
        assert adaptation.stop_loss_tightening == 0.0

    def test_custom_adaptation(self):
        """Test custom adaptation values."""
        adaptation = RegimeAdaptation(
            spacing_multiplier=1.5,
            shift_percent=2.0,
            buy_levels_active=False,
            pause_trading=True,
        )

        assert adaptation.spacing_multiplier == 1.5
        assert adaptation.shift_percent == 2.0
        assert adaptation.buy_levels_active is False
        assert adaptation.pause_trading is True


class TestGridStrategy:
    """Tests for GridStrategy class."""

    @pytest.fixture
    def grid_config(self):
        """Create grid config."""
        return GridConfig(
            num_levels=10,
            range_percent=5.0,
            order_size_quote=40.0,
            spacing=GridSpacing.EQUAL,
        )

    @pytest.fixture
    def risk_config(self):
        """Create risk config."""
        return RiskConfig(
            max_position_percent=70.0,
            min_confidence=0.6,
            stop_loss_percent=15.0,
        )

    @pytest.fixture
    def calculator(self, grid_config, risk_config):
        """Create calculator."""
        return GridCalculator(grid_config, risk_config)

    @pytest.fixture
    def strategy(self, grid_config, risk_config, calculator):
        """Create strategy instance."""
        return GridStrategy(grid_config, risk_config, calculator)

    def test_ranging_adaptation(self, strategy):
        """Test RANGING regime produces tighter spacing."""
        adaptation = strategy.get_adaptation(
            MarketRegime.RANGING, confidence=0.85
        )

        assert adaptation.spacing_multiplier == 0.8  # Tighter
        assert adaptation.shift_percent == 0.0
        assert adaptation.buy_levels_active is True
        assert adaptation.sell_levels_active is True
        assert adaptation.pause_trading is False

    def test_trending_up_adaptation(self, strategy):
        """Test TRENDING_UP regime shifts grid up."""
        adaptation = strategy.get_adaptation(
            MarketRegime.TRENDING_UP, confidence=0.85
        )

        assert adaptation.spacing_multiplier == 1.2  # Wider
        assert adaptation.shift_percent == 2.0  # Shift up
        assert adaptation.buy_levels_active is True
        assert adaptation.sell_levels_active is True
        assert adaptation.pause_trading is False

    def test_trending_down_adaptation(self, strategy):
        """Test TRENDING_DOWN regime pauses buys."""
        adaptation = strategy.get_adaptation(
            MarketRegime.TRENDING_DOWN, confidence=0.85
        )

        assert adaptation.buy_levels_active is False  # Buys paused
        assert adaptation.sell_levels_active is True
        assert adaptation.stop_loss_tightening == 0.5  # Tighter stop

    def test_high_volatility_adaptation(self, strategy):
        """Test HIGH_VOLATILITY regime widens spacing."""
        adaptation = strategy.get_adaptation(
            MarketRegime.HIGH_VOLATILITY, confidence=0.85
        )

        assert adaptation.spacing_multiplier == 1.5  # Much wider
        assert adaptation.buy_levels_active is True
        assert adaptation.sell_levels_active is True

    def test_breakout_adaptation(self, strategy):
        """Test BREAKOUT regime pauses trading."""
        adaptation = strategy.get_adaptation(
            MarketRegime.BREAKOUT, confidence=0.85
        )

        assert adaptation.pause_trading is True
        assert adaptation.buy_levels_active is False
        assert adaptation.sell_levels_active is False

    def test_low_confidence_falls_back_to_ranging(self, strategy):
        """Test low confidence falls back to RANGING behavior."""
        # Even with BREAKOUT regime, low confidence should use RANGING
        adaptation = strategy.get_adaptation(
            MarketRegime.BREAKOUT, confidence=0.4  # Below 0.6 threshold
        )

        # Should get RANGING adaptation (default)
        ranging_adaptation = GridStrategy.REGIME_ADAPTATIONS[MarketRegime.RANGING]
        assert adaptation == ranging_adaptation

    def test_adapt_parameters_shift(self, strategy):
        """Test parameter adaptation applies shift."""
        base_params = GridParameters(
            center_price=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=10,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
        )

        adapted = strategy.adapt_parameters(
            base_params, MarketRegime.TRENDING_UP, confidence=0.85
        )

        # TRENDING_UP has +2% shift
        expected_center = Decimal("50000") * Decimal("1.02")
        assert adapted.center_price == expected_center

        # And 1.2x spacing
        assert adapted.range_percent == 5.0 * 1.2

    def test_adapt_parameters_spacing_multiplier(self, strategy):
        """Test parameter adaptation applies spacing multiplier."""
        base_params = GridParameters(
            center_price=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=10,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
        )

        adapted = strategy.adapt_parameters(
            base_params, MarketRegime.RANGING, confidence=0.85
        )

        # RANGING has 0.8x spacing
        assert adapted.range_percent == 5.0 * 0.8

    def test_compute_adapted_grid(self, strategy):
        """Test full adapted grid computation."""
        grid_state, adaptation = strategy.compute_adapted_grid(
            current_price=Decimal("50000"),
            atr=Decimal("500"),
            regime=MarketRegime.RANGING,
            confidence=0.85,
        )

        assert grid_state is not None
        assert len(grid_state.levels) > 0
        assert adaptation == GridStrategy.REGIME_ADAPTATIONS[MarketRegime.RANGING]

    def test_compute_adapted_grid_paused(self, strategy):
        """Test adapted grid returns empty when paused."""
        grid_state, adaptation = strategy.compute_adapted_grid(
            current_price=Decimal("50000"),
            atr=Decimal("500"),
            regime=MarketRegime.BREAKOUT,
            confidence=0.85,
        )

        # BREAKOUT pauses trading, should return empty grid
        assert len(grid_state.levels) == 0

    def test_should_pause_trading_breakout(self, strategy):
        """Test should_pause_trading for BREAKOUT."""
        should_pause, reason = strategy.should_pause_trading(
            MarketRegime.BREAKOUT, confidence=0.85
        )

        assert should_pause is True
        assert "breakout" in reason.lower()

    def test_should_pause_trading_very_low_confidence(self, strategy):
        """Test should_pause_trading for very low confidence."""
        # Below half the threshold (0.6 * 0.5 = 0.3)
        should_pause, reason = strategy.should_pause_trading(
            MarketRegime.RANGING, confidence=0.2
        )

        assert should_pause is True
        assert "confidence" in reason.lower()

    def test_should_not_pause_normal(self, strategy):
        """Test should_pause_trading returns False for normal conditions."""
        should_pause, reason = strategy.should_pause_trading(
            MarketRegime.RANGING, confidence=0.85
        )

        assert should_pause is False
        assert reason == ""

    def test_compute_stop_loss_price(self, strategy):
        """Test stop-loss price calculation."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(9, Decimal("52500"), GridOrderType.SELL, Decimal("0.001")),
        ]
        grid_state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("100"),
            total_sell_exposure=Decimal("100"),
        )

        adaptation = RegimeAdaptation()  # No tightening

        stop_loss = strategy.compute_stop_loss_price(
            grid_state, adaptation, base_stop_loss_percent=15.0
        )

        # 15% below $47500 = $47500 * 0.85 = $40375
        expected = Decimal("47500") * Decimal("0.85")
        assert stop_loss == expected

    def test_compute_stop_loss_with_tightening(self, strategy):
        """Test stop-loss price with tightening."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
        ]
        grid_state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("100"),
            total_sell_exposure=Decimal("0"),
        )

        adaptation = RegimeAdaptation(stop_loss_tightening=0.5)  # 50% tighter

        stop_loss = strategy.compute_stop_loss_price(
            grid_state, adaptation, base_stop_loss_percent=15.0
        )

        # With 50% tightening: adjusted = 15% * (1 - 0.5*0.5) = 15% * 0.75 = 11.25%
        # Stop-loss = $47500 * (1 - 0.1125) = $42156.25
        expected = Decimal("47500") * Decimal("0.8875")
        assert abs(stop_loss - expected) < Decimal("1")

    def test_regime_change_tracking(self, strategy):
        """Test that regime changes are tracked."""
        assert strategy.last_regime is None
        assert strategy.regime_change_count == 0

        strategy.compute_adapted_grid(
            Decimal("50000"), Decimal("500"),
            MarketRegime.RANGING, 0.85
        )

        assert strategy.last_regime == MarketRegime.RANGING
        assert strategy.regime_change_count == 0

        strategy.compute_adapted_grid(
            Decimal("50000"), Decimal("500"),
            MarketRegime.TRENDING_UP, 0.85
        )

        assert strategy.last_regime == MarketRegime.TRENDING_UP
        assert strategy.regime_change_count == 1


class TestGridStrategyLevelActivation:
    """Tests for level activation based on regime."""

    @pytest.fixture
    def strategy(self):
        config = GridConfig()
        risk_config = RiskConfig()
        calculator = GridCalculator(config, risk_config)
        return GridStrategy(config, risk_config, calculator)

    def test_trending_down_disables_buy_levels(self, strategy):
        """Test TRENDING_DOWN disables buy levels."""
        grid_state, adaptation = strategy.compute_adapted_grid(
            current_price=Decimal("50000"),
            atr=Decimal("500"),
            regime=MarketRegime.TRENDING_DOWN,
            confidence=0.85,
        )

        # Check that buy levels are inactive
        for level in grid_state.levels:
            if level.side == GridOrderType.BUY:
                assert level.is_active is False
            else:
                assert level.is_active is True

    def test_ranging_all_levels_active(self, strategy):
        """Test RANGING keeps all levels active."""
        grid_state, adaptation = strategy.compute_adapted_grid(
            current_price=Decimal("50000"),
            atr=Decimal("500"),
            regime=MarketRegime.RANGING,
            confidence=0.85,
        )

        # All levels should be active
        for level in grid_state.levels:
            assert level.is_active is True
