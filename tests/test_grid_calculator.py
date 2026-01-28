"""
Tests for GridCalculator module.

Tests grid level calculation with:
- Equal spacing
- Geometric spacing
- ATR-based spacing
- Volume calculations
- Risk rule validation
"""

import pytest
from decimal import Decimal

from src.grid import (
    GridLevel,
    GridParameters,
    GridState,
    GridCalculator,
)
from src.api import GridOrderType
from config.settings import GridConfig, RiskConfig, GridSpacing


class TestGridLevel:
    """Tests for GridLevel dataclass."""

    def test_grid_level_creation(self):
        """Test creating a valid grid level."""
        level = GridLevel(
            index=5,
            price=Decimal("50000"),
            side=GridOrderType.BUY,
            volume=Decimal("0.001"),
        )

        assert level.index == 5
        assert level.price == Decimal("50000")
        assert level.side == GridOrderType.BUY
        assert level.volume == Decimal("0.001")
        assert level.is_active is True

    def test_grid_level_invalid_price(self):
        """Test that negative price raises error."""
        with pytest.raises(ValueError, match="Price must be positive"):
            GridLevel(
                index=0,
                price=Decimal("-100"),
                side=GridOrderType.BUY,
                volume=Decimal("0.001"),
            )

    def test_grid_level_invalid_volume(self):
        """Test that negative volume raises error."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            GridLevel(
                index=0,
                price=Decimal("50000"),
                side=GridOrderType.BUY,
                volume=Decimal("-0.001"),
            )


class TestGridParameters:
    """Tests for GridParameters dataclass."""

    def test_parameters_creation(self):
        """Test creating valid parameters."""
        params = GridParameters(
            center_price=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=10,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
        )

        assert params.center_price == Decimal("50000")
        assert params.num_levels == 10

    def test_parameters_invalid_center(self):
        """Test that non-positive center price raises error."""
        with pytest.raises(ValueError, match="Center price must be positive"):
            GridParameters(
                center_price=Decimal("0"),
                atr=Decimal("500"),
                num_levels=10,
                range_percent=5.0,
                spacing=GridSpacing.EQUAL,
                order_size_quote=Decimal("40"),
                capital=Decimal("400"),
            )

    def test_parameters_invalid_levels(self):
        """Test that too few levels raises error."""
        with pytest.raises(ValueError, match="Need at least 2 levels"):
            GridParameters(
                center_price=Decimal("50000"),
                atr=Decimal("500"),
                num_levels=1,
                range_percent=5.0,
                spacing=GridSpacing.EQUAL,
                order_size_quote=Decimal("40"),
                capital=Decimal("400"),
            )


class TestGridState:
    """Tests for GridState dataclass."""

    def test_grid_state_properties(self):
        """Test GridState computed properties."""
        levels = [
            GridLevel(0, Decimal("47500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(1, Decimal("48000"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(2, Decimal("48500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(3, Decimal("49000"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(4, Decimal("49500"), GridOrderType.BUY, Decimal("0.001")),
            GridLevel(5, Decimal("50500"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(6, Decimal("51000"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(7, Decimal("51500"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(8, Decimal("52000"), GridOrderType.SELL, Decimal("0.001")),
            GridLevel(9, Decimal("52500"), GridOrderType.SELL, Decimal("0.001")),
        ]

        state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("200"),
            total_sell_exposure=Decimal("200"),
        )

        assert state.num_levels == 10
        assert state.num_buy_levels == 5
        assert state.num_sell_levels == 5
        assert len(state.active_levels) == 10

    def test_grid_state_range_percent(self):
        """Test range_percent calculation."""
        state = GridState(
            levels=[],
            center_price=Decimal("50000"),
            upper_bound=Decimal("52500"),
            lower_bound=Decimal("47500"),
            total_buy_exposure=Decimal("0"),
            total_sell_exposure=Decimal("0"),
        )

        # Range is 52500 - 47500 = 5000, which is 10% of 50000
        assert state.range_percent == pytest.approx(10.0, rel=0.01)


class TestGridCalculator:
    """Tests for GridCalculator class."""

    @pytest.fixture
    def config(self):
        """Create default grid config."""
        return GridConfig(
            num_levels=10,
            range_percent=5.0,
            order_size_quote=40.0,
            spacing=GridSpacing.EQUAL,
        )

    @pytest.fixture
    def risk_config(self):
        """Create default risk config."""
        return RiskConfig(
            max_position_percent=70.0,
            max_open_orders=20,
            order_risk_percent=2.0,
        )

    @pytest.fixture
    def calculator(self, config, risk_config):
        """Create calculator instance."""
        return GridCalculator(config, risk_config)

    def test_compute_level_prices_equal(self, calculator):
        """Test equal spacing produces correct prices."""
        prices = calculator.compute_level_prices_equal(
            center=Decimal("50000"),
            range_percent=10.0,  # +/- 5% = 10% total
            num_levels=5,
        )

        assert len(prices) == 5

        # With 10% range around 50000, range is 47500 to 52500
        # 5 levels: 47500, 48750, 50000, 51250, 52500
        assert prices[0] == Decimal("47500")
        assert prices[2] == Decimal("50000")  # Center
        assert prices[4] == Decimal("52500")

        # Check equal spacing
        step = prices[1] - prices[0]
        for i in range(1, len(prices)):
            assert prices[i] - prices[i - 1] == step

    def test_compute_level_prices_geometric(self, calculator):
        """Test geometric spacing produces percentage-based gaps."""
        prices = calculator.compute_level_prices_geometric(
            center=Decimal("50000"),
            range_percent=10.0,
            num_levels=5,
        )

        assert len(prices) == 5

        # Geometric spacing means each step is same percentage
        # Ratios between adjacent prices should be approximately equal
        ratios = [prices[i] / prices[i - 1] for i in range(1, len(prices))]

        # All ratios should be approximately equal
        for i in range(1, len(ratios)):
            assert ratios[i] == pytest.approx(float(ratios[0]), rel=0.01)

    def test_compute_level_prices_atr(self, calculator):
        """Test ATR-based spacing scales with volatility."""
        # Low ATR = tight spacing
        prices_low = calculator.compute_level_prices_atr(
            center=Decimal("50000"),
            atr=Decimal("100"),
            num_levels=5,
            atr_multiplier=0.5,
        )

        # High ATR = wide spacing
        prices_high = calculator.compute_level_prices_atr(
            center=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=5,
            atr_multiplier=0.5,
        )

        # High ATR should have wider range
        range_low = prices_low[-1] - prices_low[0]
        range_high = prices_high[-1] - prices_high[0]

        assert range_high > range_low

    def test_compute_level_prices_atr_zero(self, calculator):
        """Test ATR-based spacing with zero ATR falls back to equal."""
        prices = calculator.compute_level_prices_atr(
            center=Decimal("50000"),
            atr=Decimal("0"),
            num_levels=5,
            atr_multiplier=0.5,
        )

        # Should still return valid prices
        assert len(prices) == 5
        assert prices[2] == Decimal("50000")  # Center

    def test_compute_volume_for_level(self, calculator):
        """Test volume calculation with risk cap."""
        # Normal case: order size fits within risk limit
        volume = calculator.compute_volume_for_level(
            price=Decimal("50000"),
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
            max_risk_percent=2.0,
        )

        # $40 / $50000 = 0.0008 BTC
        assert volume == Decimal("0.0008")

    def test_compute_volume_capped_by_risk(self, calculator):
        """Test volume is capped at 2% risk per level."""
        # High order size that would exceed risk limit
        volume = calculator.compute_volume_for_level(
            price=Decimal("50000"),
            order_size_quote=Decimal("100"),  # Would be 0.002 BTC
            capital=Decimal("400"),
            max_risk_percent=2.0,  # Max $8 risk = 0.00016 BTC
        )

        # Should be capped at risk limit
        # 2% of $400 = $8, $8 / $50000 = 0.00016 BTC
        assert volume == Decimal("0.00016")

    def test_adjust_for_min_order_size_valid(self, calculator):
        """Test valid order size passes through."""
        volume = calculator.adjust_for_min_order_size(
            volume=Decimal("0.001"),
            price=Decimal("50000"),
            min_order_usd=Decimal("5"),
        )

        # 0.001 * 50000 = $50, which is > $5 minimum
        assert volume == Decimal("0.001")

    def test_adjust_for_min_order_size_too_small(self, calculator):
        """Test order below minimum returns None."""
        volume = calculator.adjust_for_min_order_size(
            volume=Decimal("0.00001"),  # 0.00001 * 50000 = $0.50
            price=Decimal("50000"),
            min_order_usd=Decimal("5"),
        )

        assert volume is None

    def test_compute_levels_full_grid(self, calculator):
        """Test computing a full grid."""
        params = GridParameters(
            center_price=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=10,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
        )

        grid_state = calculator.compute_levels(params)

        # Should have levels (might be fewer if some are too small)
        assert grid_state.num_levels > 0

        # Check center is approximately correct
        assert grid_state.center_price == Decimal("50000")

        # Check bounds are within expected range
        expected_lower = Decimal("50000") * Decimal("0.975")  # -2.5%
        expected_upper = Decimal("50000") * Decimal("1.025")  # +2.5%

        assert grid_state.lower_bound >= expected_lower - Decimal("100")
        assert grid_state.upper_bound <= expected_upper + Decimal("100")

        # Check buy/sell distribution
        buy_levels = grid_state.get_levels_by_side(GridOrderType.BUY)
        sell_levels = grid_state.get_levels_by_side(GridOrderType.SELL)

        # Should have roughly equal buys and sells
        assert len(buy_levels) > 0
        assert len(sell_levels) > 0

    def test_validate_grid_exposure_valid(self, calculator):
        """Test validation passes for valid grid."""
        levels = [
            GridLevel(0, Decimal("50000"), GridOrderType.BUY, Decimal("0.001")),
        ]
        state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("50000"),
            lower_bound=Decimal("50000"),
            total_buy_exposure=Decimal("50"),  # $50, well under 70% of $400
            total_sell_exposure=Decimal("0"),
        )

        errors = calculator.validate_grid_exposure(state, Decimal("400"))
        assert len(errors) == 0

    def test_validate_grid_exposure_exceeds_max(self, calculator):
        """Test validation fails when exposure exceeds max."""
        levels = [
            GridLevel(0, Decimal("50000"), GridOrderType.BUY, Decimal("0.01")),
        ]
        state = GridState(
            levels=levels,
            center_price=Decimal("50000"),
            upper_bound=Decimal("50000"),
            lower_bound=Decimal("50000"),
            total_buy_exposure=Decimal("300"),  # $300, exceeds 70% of $400 = $280
            total_sell_exposure=Decimal("0"),
        )

        errors = calculator.validate_grid_exposure(state, Decimal("400"))
        assert len(errors) > 0
        assert "exceeds max" in errors[0].lower()

    def test_price_rounding(self, calculator):
        """Test that prices are rounded to 1 decimal for XBTUSD."""
        params = GridParameters(
            center_price=Decimal("50000.123456"),
            atr=Decimal("500"),
            num_levels=4,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
            price_decimals=1,
        )

        grid_state = calculator.compute_levels(params)

        # All prices should have at most 1 decimal place
        for level in grid_state.levels:
            price_str = str(level.price)
            if "." in price_str:
                decimals = len(price_str.split(".")[1])
                assert decimals <= 1


class TestGridCalculatorEdgeCases:
    """Edge case tests for GridCalculator."""

    @pytest.fixture
    def calculator(self):
        config = GridConfig()
        risk_config = RiskConfig()
        return GridCalculator(config, risk_config)

    def test_single_level_equal(self, calculator):
        """Test single level returns center."""
        prices = calculator.compute_level_prices_equal(
            center=Decimal("50000"),
            range_percent=10.0,
            num_levels=1,
        )

        assert len(prices) == 1
        assert prices[0] == Decimal("50000")

    def test_single_level_geometric(self, calculator):
        """Test single level geometric returns center."""
        prices = calculator.compute_level_prices_geometric(
            center=Decimal("50000"),
            range_percent=10.0,
            num_levels=1,
        )

        assert len(prices) == 1
        assert prices[0] == Decimal("50000")

    def test_very_small_price(self, calculator):
        """Test with very small price (penny stock)."""
        prices = calculator.compute_level_prices_equal(
            center=Decimal("0.01"),
            range_percent=10.0,
            num_levels=5,
        )

        assert len(prices) == 5
        assert all(p > 0 for p in prices)

    def test_very_large_price(self, calculator):
        """Test with very large price."""
        prices = calculator.compute_level_prices_equal(
            center=Decimal("1000000"),
            range_percent=10.0,
            num_levels=5,
        )

        assert len(prices) == 5
        assert prices[2] == Decimal("1000000")
