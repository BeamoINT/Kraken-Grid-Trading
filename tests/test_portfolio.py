"""
Tests for Portfolio module.

Tests:
- Initial state (equity = capital, HWM = capital)
- HWM increases with profit
- Drawdown calculation from HWM
- Max drawdown tracking
- Exposure calculation
- Persistence (to_dict/from_dict)
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.core import (
    Portfolio,
    EquitySnapshot,
    DrawdownState,
    EquitySource,
)


class TestPortfolioInitialization:
    """Tests for Portfolio initialization."""

    def test_initial_state(self):
        """Test initial portfolio state."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        assert portfolio.total_equity == Decimal("400")
        assert portfolio.high_water_mark == Decimal("400")
        assert portfolio.initial_capital == Decimal("400")
        assert portfolio.available_balance == Decimal("400")
        assert portfolio.current_drawdown == Decimal("0")
        assert portfolio.current_drawdown_percent == 0.0

    def test_initial_hwm_override(self):
        """Test initializing with pre-existing HWM."""
        portfolio = Portfolio(
            initial_capital=Decimal("400"),
            high_water_mark=Decimal("450"),
        )

        assert portfolio.high_water_mark == Decimal("450")
        # Drawdown from HWM to current equity
        assert portfolio.current_drawdown == Decimal("50")
        assert portfolio.current_drawdown_percent == pytest.approx(11.11, rel=0.01)


class TestHighWaterMark:
    """Tests for high-water mark tracking."""

    def test_hwm_increases_with_profit(self):
        """Test HWM increases when equity exceeds it."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Simulate profit
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()

        assert portfolio.high_water_mark == Decimal("450")

    def test_hwm_unchanged_on_loss(self):
        """Test HWM unchanged when equity decreases."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Set HWM higher
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()

        # Simulate loss
        portfolio._total_equity = Decimal("420")
        portfolio._update_high_water_mark()

        # HWM should still be 450
        assert portfolio.high_water_mark == Decimal("450")

    def test_hwm_reset(self):
        """Test HWM reset functionality."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()

        # Reset to current equity
        portfolio._total_equity = Decimal("380")
        portfolio.reset_high_water_mark()

        assert portfolio.high_water_mark == Decimal("380")

    def test_hwm_reset_with_value(self):
        """Test HWM reset with specific value."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        portfolio.reset_high_water_mark(Decimal("500"))

        assert portfolio.high_water_mark == Decimal("500")


class TestDrawdownCalculation:
    """Tests for drawdown calculation."""

    def test_no_drawdown_at_peak(self):
        """Test zero drawdown when at HWM."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()

        assert portfolio.current_drawdown == Decimal("0")
        assert portfolio.current_drawdown_percent == 0.0

    def test_drawdown_from_hwm(self):
        """Test drawdown calculation from HWM."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Set HWM
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()

        # Lose 20%
        portfolio._total_equity = Decimal("400")

        assert portfolio.current_drawdown == Decimal("100")
        assert portfolio.current_drawdown_percent == 20.0

    def test_max_drawdown_tracking(self):
        """Test max drawdown is tracked."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Peak at 500
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()

        # Drop to 400 (20% DD)
        portfolio._total_equity = Decimal("400")
        portfolio._update_high_water_mark()

        # Recover to 420
        portfolio._total_equity = Decimal("420")
        portfolio._update_high_water_mark()

        # Max DD should still be recorded
        drawdown_state = portfolio.get_drawdown_state()
        assert drawdown_state.max_drawdown_percent == 20.0

    def test_drawdown_state_dataclass(self):
        """Test DrawdownState contains all fields."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("400")
        portfolio._update_high_water_mark()

        state = portfolio.get_drawdown_state()

        assert state.high_water_mark == Decimal("450")
        assert state.current_drawdown == Decimal("50")
        assert state.current_drawdown_percent == pytest.approx(11.11, rel=0.01)
        assert isinstance(state.hwm_timestamp, datetime)
        assert isinstance(state.max_drawdown_timestamp, datetime)


class TestUpdateFromPosition:
    """Tests for update_from_position method."""

    def test_update_long_position(self):
        """Test updating from a long position."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        portfolio.update_from_position(
            position_quantity=Decimal("0.01"),
            avg_entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            realized_pnl=Decimal("0"),
            open_order_exposure=Decimal("100"),
        )

        # Position value = 0.01 * 51000 = 510
        assert portfolio.position_value == Decimal("510")
        # Unrealized P&L = 510 - (0.01 * 50000) = 510 - 500 = 10
        assert portfolio.unrealized_pnl == Decimal("10")
        # Total equity = 400 + 0 + 10 = 410
        assert portfolio.total_equity == Decimal("410")

    def test_update_short_position(self):
        """Test updating from a short position."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        portfolio.update_from_position(
            position_quantity=Decimal("-0.01"),
            avg_entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            realized_pnl=Decimal("0"),
            open_order_exposure=Decimal("0"),
        )

        # Position value = |-0.01| * 49000 = 490
        assert portfolio.position_value == Decimal("490")
        # Short P&L = cost - current = 500 - 490 = 10
        assert portfolio.unrealized_pnl == Decimal("10")
        # Equity = 400 + 0 + 10 = 410
        assert portfolio.total_equity == Decimal("410")

    def test_update_no_position(self):
        """Test updating with no position."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        portfolio.update_from_position(
            position_quantity=Decimal("0"),
            avg_entry_price=Decimal("0"),
            current_price=Decimal("50000"),
            realized_pnl=Decimal("25"),
            open_order_exposure=Decimal("50"),
        )

        assert portfolio.position_value == Decimal("0")
        assert portfolio.unrealized_pnl == Decimal("0")
        assert portfolio.realized_pnl == Decimal("25")
        # Equity = 400 + 25 + 0 = 425
        assert portfolio.total_equity == Decimal("425")

    def test_hwm_updates_on_profit(self):
        """Test HWM updates when profit increases equity."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        portfolio.update_from_position(
            position_quantity=Decimal("0.01"),
            avg_entry_price=Decimal("40000"),
            current_price=Decimal("50000"),
            realized_pnl=Decimal("0"),
            open_order_exposure=Decimal("0"),
        )

        # Unrealized P&L = 500 - 400 = 100
        # Equity = 400 + 0 + 100 = 500
        assert portfolio.total_equity == Decimal("500")
        assert portfolio.high_water_mark == Decimal("500")


class TestExposureCalculation:
    """Tests for exposure calculation."""

    def test_total_exposure(self):
        """Test total exposure calculation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._position_value = Decimal("200")
        portfolio._open_order_exposure = Decimal("80")

        assert portfolio.get_total_exposure() == Decimal("280")

    def test_exposure_percent(self):
        """Test exposure percentage calculation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._position_value = Decimal("200")
        portfolio._open_order_exposure = Decimal("80")

        # 280 / 400 = 70%
        assert portfolio.get_exposure_percent() == 70.0

    def test_exposure_percent_zero_equity(self):
        """Test exposure percent with zero equity returns 100%."""
        portfolio = Portfolio(initial_capital=Decimal("0"))

        assert portfolio.get_exposure_percent() == 100.0


class TestEquitySnapshot:
    """Tests for EquitySnapshot functionality."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._position_value = Decimal("100")
        portfolio._unrealized_pnl = Decimal("10")

        snapshot = portfolio.get_latest_snapshot()

        assert snapshot.total_equity == Decimal("400")
        assert snapshot.position_value == Decimal("100")
        assert snapshot.unrealized_pnl == Decimal("10")
        assert snapshot.source == EquitySource.CALCULATED

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snapshot = EquitySnapshot(
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            total_equity=Decimal("400"),
            available_balance=Decimal("300"),
            position_value=Decimal("100"),
            open_order_exposure=Decimal("50"),
            unrealized_pnl=Decimal("10"),
            realized_pnl=Decimal("5"),
            source=EquitySource.API_BALANCE,
        )

        data = snapshot.to_dict()

        assert data["total_equity"] == "400"
        assert data["available_balance"] == "300"
        assert data["source"] == "api_balance"

    def test_equity_history(self):
        """Test equity history tracking."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Generate some snapshots
        for i in range(5):
            portfolio._total_equity = Decimal(str(400 + i * 10))
            snapshot = portfolio._create_snapshot()
            portfolio._record_snapshot(snapshot)

        history = portfolio.get_equity_history()
        assert len(history) == 5
        assert history[-1].total_equity == Decimal("440")


class TestPersistence:
    """Tests for Portfolio persistence."""

    def test_to_dict(self):
        """Test serialization to dict."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()
        portfolio._realized_pnl = Decimal("50")

        data = portfolio.to_dict()

        assert data["initial_capital"] == "400"
        assert data["total_equity"] == "450"
        assert data["high_water_mark"] == "450"
        assert data["realized_pnl"] == "50"

    def test_from_dict(self):
        """Test restoration from dict."""
        original = Portfolio(initial_capital=Decimal("400"))
        original._total_equity = Decimal("450")
        original._update_high_water_mark()
        original._realized_pnl = Decimal("50")
        original._max_drawdown_percent = 5.0

        data = original.to_dict()
        restored = Portfolio.from_dict(data)

        assert restored.initial_capital == Decimal("400")
        assert restored.total_equity == Decimal("450")
        assert restored.high_water_mark == Decimal("450")
        assert restored.realized_pnl == Decimal("50")
        assert restored._max_drawdown_percent == 5.0

    def test_round_trip_persistence(self):
        """Test full round-trip persistence."""
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Make some changes
        portfolio.update_from_position(
            position_quantity=Decimal("0.01"),
            avg_entry_price=Decimal("40000"),
            current_price=Decimal("45000"),
            realized_pnl=Decimal("20"),
            open_order_exposure=Decimal("50"),
        )

        # Round-trip
        data = portfolio.to_dict()
        restored = Portfolio.from_dict(data)

        # Key fields should match
        assert restored.total_equity == portfolio.total_equity
        assert restored.high_water_mark == portfolio.high_water_mark
        assert restored.realized_pnl == portfolio.realized_pnl
        assert restored._max_drawdown_percent == portfolio._max_drawdown_percent


class TestTotalReturn:
    """Tests for total return calculation."""

    def test_positive_return(self):
        """Test positive return calculation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("480")

        # (480 - 400) / 400 * 100 = 20%
        assert portfolio.total_return_percent == 20.0

    def test_negative_return(self):
        """Test negative return calculation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("360")

        # (360 - 400) / 400 * 100 = -10%
        assert portfolio.total_return_percent == -10.0

    def test_zero_capital_return(self):
        """Test return with zero capital."""
        portfolio = Portfolio(initial_capital=Decimal("0"))

        assert portfolio.total_return_percent == 0.0


class TestSummary:
    """Tests for portfolio summary."""

    def test_get_summary(self):
        """Test summary generation."""
        portfolio = Portfolio(initial_capital=Decimal("400"))
        portfolio._total_equity = Decimal("450")
        portfolio._position_value = Decimal("100")
        portfolio._unrealized_pnl = Decimal("50")
        portfolio._update_high_water_mark()

        summary = portfolio.get_summary()

        assert summary["initial_capital"] == 400.0
        assert summary["total_equity"] == 450.0
        assert summary["position_value"] == 100.0
        assert summary["unrealized_pnl"] == 50.0
        assert summary["high_water_mark"] == 450.0
        assert summary["current_drawdown_percent"] == 0.0
        assert "exposure_percent" in summary
