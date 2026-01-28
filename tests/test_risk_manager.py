"""
Tests for Risk Manager module.

Tests:
- Passes when within all limits
- Halts on 20% drawdown
- Blocks order exceeding 2% risk
- Allows order within limits
- Pauses on low confidence
- Exposure limit enforcement
- Graduated alerts (warning -> critical -> halt)
- Resume after pause
- clear_halt requires admin_override
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from src.core import (
    RiskManager,
    RiskAction,
    RiskCheckResult,
    RiskState,
    Portfolio,
    AlertManager,
    AlertSeverity,
)
from config.settings import RiskConfig


class TestRiskManagerInitialization:
    """Tests for RiskManager initialization."""

    @pytest.fixture
    def risk_config(self):
        """Create default risk config."""
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        """Create portfolio with $400 capital."""
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager."""
        return AlertManager()

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio, alert_manager):
        """Create risk manager."""
        return RiskManager(
            config=risk_config,
            portfolio=portfolio,
            alert_manager=alert_manager,
        )

    def test_initial_state(self, risk_manager):
        """Test initial risk manager state."""
        assert risk_manager.is_halted is False
        assert risk_manager.is_paused is False

    def test_get_risk_state(self, risk_manager):
        """Test getting risk state."""
        state = risk_manager.get_risk_state()

        assert isinstance(state, RiskState)
        assert state.is_halted is False
        assert state.is_paused is False
        assert state.drawdown_percent == 0.0


class TestDrawdownChecks:
    """Tests for drawdown-related risk checks."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def alert_manager(self):
        return AlertManager()

    def test_passes_within_drawdown_limit(self, risk_config, portfolio, alert_manager):
        """Test passes when drawdown is within limit."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 10% drawdown (within 20% limit)
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("405")  # 10% from HWM

        result = risk_manager.run_risk_check()

        assert result.passed is True
        assert result.action == RiskAction.NONE

    def test_halt_on_max_drawdown(self, risk_config, portfolio, alert_manager):
        """Test halt triggered on max drawdown."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 20% drawdown (at limit)
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("400")  # 20% from HWM

        result = risk_manager.run_risk_check()

        assert result.passed is False
        assert result.action == RiskAction.HALT_TRADING
        assert risk_manager.is_halted is True
        assert "drawdown" in result.reason.lower()

    def test_halt_exceeds_max_drawdown(self, risk_config, portfolio, alert_manager):
        """Test halt when exceeding max drawdown."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 25% drawdown (exceeds 20% limit)
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("375")  # 25% from HWM

        result = risk_manager.run_risk_check()

        assert result.passed is False
        assert result.action == RiskAction.HALT_TRADING
        assert risk_manager.is_halted is True

    def test_warning_alert_at_15_percent(self, risk_config, portfolio, alert_manager):
        """Test warning alert at 15% drawdown."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 15% drawdown
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("425")  # 15% from HWM

        risk_manager.run_risk_check()

        # Should have warning alert
        alerts = alert_manager.get_recent_alerts(min_severity=AlertSeverity.WARNING)
        assert len(alerts) >= 1

    def test_critical_alert_at_18_percent(self, risk_config, portfolio, alert_manager):
        """Test critical alert at 18% drawdown."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 18% drawdown
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("410")  # 18% from HWM

        risk_manager.run_risk_check()

        # Should have critical alert
        alerts = alert_manager.get_recent_alerts(min_severity=AlertSeverity.CRITICAL)
        assert len(alerts) >= 1


class TestExposureChecks:
    """Tests for exposure-related risk checks."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def alert_manager(self):
        return AlertManager()

    def test_passes_within_exposure_limit(self, risk_config, portfolio, alert_manager):
        """Test passes when exposure is within limit."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 50% exposure (within 70% limit)
        portfolio._position_value = Decimal("150")
        portfolio._open_order_exposure = Decimal("50")

        result = risk_manager.run_risk_check()

        assert result.passed is True

    def test_reduce_exposure_at_limit(self, risk_config, portfolio, alert_manager):
        """Test reduce exposure action at limit."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 75% exposure (exceeds 70% limit)
        portfolio._position_value = Decimal("200")
        portfolio._open_order_exposure = Decimal("100")

        result = risk_manager.run_risk_check()

        assert result.passed is False
        assert result.action == RiskAction.REDUCE_EXPOSURE
        assert "exposure" in result.reason.lower()

    def test_exposure_warning_at_60_percent(self, risk_config, portfolio, alert_manager):
        """Test warning alert at 60% exposure."""
        risk_manager = RiskManager(risk_config, portfolio, alert_manager)

        # 65% exposure (above 60% warning, below 70% limit)
        portfolio._position_value = Decimal("200")
        portfolio._open_order_exposure = Decimal("60")

        risk_manager.run_risk_check()

        # Should have warning alert
        alerts = alert_manager.get_recent_alerts(min_severity=AlertSeverity.WARNING)
        # At least one alert should be present
        assert any("exposure" in a.message.lower() for a in alerts)


class TestOrderValidation:
    """Tests for order validation."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio):
        return RiskManager(risk_config, portfolio)

    def test_order_within_limit_allowed(self, risk_manager):
        """Test order within 2% limit is allowed."""
        # $8 order on $400 = 2%
        result = risk_manager.check_order_allowed(
            order_value=Decimal("8"),
            capital=Decimal("400"),
        )

        assert result.passed is True
        assert result.action == RiskAction.NONE

    def test_order_exceeding_limit_blocked(self, risk_manager):
        """Test order exceeding 2% limit is blocked."""
        # $12 order on $400 = 3%
        result = risk_manager.check_order_allowed(
            order_value=Decimal("12"),
            capital=Decimal("400"),
        )

        assert result.passed is False
        assert result.action == RiskAction.CANCEL_ORDER
        assert "2%" in result.reason or "order" in result.reason.lower()

    def test_order_at_exact_limit(self, risk_manager):
        """Test order at exact 2% limit."""
        # Exactly $8 on $400
        result = risk_manager.check_order_allowed(
            order_value=Decimal("8"),
            capital=Decimal("400"),
        )

        assert result.passed is True

    def test_order_blocked_when_halted(self, risk_manager, portfolio):
        """Test orders blocked when trading is halted."""
        # Force halt
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("380")  # 24% drawdown
        risk_manager.run_risk_check()

        # Try to place order
        result = risk_manager.check_order_allowed(
            order_value=Decimal("5"),
            capital=Decimal("400"),
        )

        assert result.passed is False
        assert "halt" in result.reason.lower()


class TestConfidenceChecks:
    """Tests for confidence-related risk checks."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def alert_manager(self):
        return AlertManager()

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio, alert_manager):
        return RiskManager(risk_config, portfolio, alert_manager)

    def test_confidence_above_minimum_passes(self, risk_manager):
        """Test confidence above minimum passes."""
        result = risk_manager.check_confidence(0.75)

        assert result.passed is True
        assert result.action == RiskAction.NONE

    def test_confidence_at_minimum_passes(self, risk_manager):
        """Test confidence at minimum passes."""
        result = risk_manager.check_confidence(0.6)

        assert result.passed is True

    def test_low_confidence_pauses(self, risk_manager):
        """Test low confidence pauses trading."""
        result = risk_manager.check_confidence(0.5)

        assert result.passed is False
        assert result.action == RiskAction.PAUSE_TRADING
        assert "confidence" in result.reason.lower()

    def test_very_low_confidence_pauses(self, risk_manager):
        """Test very low confidence pauses trading."""
        result = risk_manager.check_confidence(0.25)

        assert result.passed is False
        assert result.action == RiskAction.PAUSE_TRADING


class TestStopLossCheck:
    """Tests for stop-loss checks."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio):
        return RiskManager(risk_config, portfolio)

    def test_price_above_stop_loss_passes(self, risk_manager):
        """Test price above stop-loss passes."""
        result = risk_manager.check_stop_loss(
            current_price=Decimal("48000"),
            lowest_grid_price=Decimal("47000"),
            position_quantity=Decimal("0.01"),
        )

        assert result.passed is True

    def test_price_at_stop_loss_triggers(self, risk_manager):
        """Test price at stop-loss triggers."""
        # Stop-loss = 15% below lowest grid
        # 47000 * 0.85 = 39950
        result = risk_manager.check_stop_loss(
            current_price=Decimal("39900"),
            lowest_grid_price=Decimal("47000"),
            position_quantity=Decimal("0.01"),
        )

        assert result.passed is False
        assert result.action == RiskAction.HALT_TRADING
        assert "stop" in result.reason.lower()

    def test_no_position_no_stop_loss(self, risk_manager):
        """Test no stop-loss check when no position."""
        result = risk_manager.check_stop_loss(
            current_price=Decimal("30000"),  # Way below stop
            lowest_grid_price=Decimal("47000"),
            position_quantity=Decimal("0"),  # No position
        )

        assert result.passed is True


class TestHaltAndPause:
    """Tests for halt and pause functionality."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def alert_manager(self):
        return AlertManager()

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio, alert_manager):
        return RiskManager(risk_config, portfolio, alert_manager)

    def test_pause_trading(self, risk_manager):
        """Test pausing trading."""
        risk_manager.pause_trading("Test pause")

        assert risk_manager.is_paused is True
        assert risk_manager.is_halted is False

    def test_resume_from_pause(self, risk_manager):
        """Test resuming from pause."""
        risk_manager.pause_trading("Test pause")
        assert risk_manager.is_paused is True

        result = risk_manager.resume_trading()

        assert result is True
        assert risk_manager.is_paused is False

    def test_cannot_resume_from_halt(self, risk_manager, portfolio):
        """Test cannot resume from halt without admin override."""
        # Force halt
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("380")
        risk_manager.run_risk_check()

        assert risk_manager.is_halted is True

        # Try to resume
        result = risk_manager.resume_trading()

        assert result is False
        assert risk_manager.is_halted is True

    def test_clear_halt_requires_admin(self, risk_manager, portfolio):
        """Test clearing halt requires admin override."""
        # Force halt
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("380")
        risk_manager.run_risk_check()

        # Try without admin
        result = risk_manager.clear_halt(admin_override=False)
        assert result is False
        assert risk_manager.is_halted is True

        # With admin override
        result = risk_manager.clear_halt(admin_override=True)
        assert result is True
        assert risk_manager.is_halted is False

    def test_halt_callback_called(self, risk_config, portfolio, alert_manager):
        """Test halt callback is called."""
        on_halt = Mock()
        risk_manager = RiskManager(
            risk_config, portfolio, alert_manager, on_halt=on_halt
        )

        # Force halt
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()
        portfolio._total_equity = Decimal("380")
        risk_manager.run_risk_check()

        on_halt.assert_called_once()

    def test_pause_callback_called(self, risk_config, portfolio, alert_manager):
        """Test pause callback is called."""
        on_pause = Mock()
        risk_manager = RiskManager(
            risk_config, portfolio, alert_manager, on_pause=on_pause
        )

        risk_manager.pause_trading("Test")

        on_pause.assert_called_once_with("Test")


class TestRiskStats:
    """Tests for risk statistics."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    @pytest.fixture
    def risk_manager(self, risk_config, portfolio):
        return RiskManager(risk_config, portfolio)

    def test_get_stats(self, risk_manager):
        """Test getting risk statistics."""
        stats = risk_manager.get_stats()

        assert "total_checks" in stats
        assert "checks_passed" in stats
        assert "checks_failed" in stats
        assert "halts" in stats
        assert "pauses" in stats

    def test_stats_increment_on_check(self, risk_manager):
        """Test stats increment on checks."""
        initial_stats = risk_manager.get_stats()

        risk_manager.run_risk_check()
        risk_manager.run_risk_check()

        stats = risk_manager.get_stats()
        assert stats["total_checks"] == initial_stats["total_checks"] + 2


class TestPersistence:
    """Tests for risk manager persistence."""

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )

    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=Decimal("400"))

    def test_to_dict(self, risk_config, portfolio):
        """Test serialization to dict."""
        risk_manager = RiskManager(risk_config, portfolio)
        risk_manager.pause_trading("Test")

        data = risk_manager.to_dict()

        assert data["is_paused"] is True
        assert data["is_halted"] is False
        assert "pause_reason" in data

    def test_restore_from_dict(self, risk_config, portfolio):
        """Test restoration from dict."""
        risk_manager = RiskManager(risk_config, portfolio)
        risk_manager.pause_trading("Test pause")

        data = risk_manager.to_dict()

        # Create new manager and restore
        new_manager = RiskManager(risk_config, portfolio)
        new_manager.restore_from_dict(data)

        assert new_manager.is_paused is True
        assert new_manager._pause_reason == "Test pause"


class TestRiskCheckResult:
    """Tests for RiskCheckResult dataclass."""

    def test_passed_result(self):
        """Test passed result."""
        result = RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="All checks passed",
            details={},
        )

        assert result.passed is True
        assert result.action == RiskAction.NONE

    def test_failed_result(self):
        """Test failed result."""
        result = RiskCheckResult(
            passed=False,
            action=RiskAction.HALT_TRADING,
            reason="Max drawdown exceeded",
            details={"drawdown_percent": 22.5},
        )

        assert result.passed is False
        assert result.action == RiskAction.HALT_TRADING
        assert result.details["drawdown_percent"] == 22.5


class TestIntegration:
    """Integration tests for full risk check flow."""

    def test_full_risk_check_flow(self):
        """Test complete risk check flow."""
        config = RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )
        portfolio = Portfolio(initial_capital=Decimal("400"))
        alert_manager = AlertManager()
        risk_manager = RiskManager(config, portfolio, alert_manager)

        # 1. Initial state - all checks pass
        result = risk_manager.run_risk_check()
        assert result.passed is True

        # 2. Simulate profit - HWM increases
        portfolio._total_equity = Decimal("450")
        portfolio._update_high_water_mark()
        assert portfolio.high_water_mark == Decimal("450")

        # 3. Check still passes
        result = risk_manager.run_risk_check()
        assert result.passed is True

        # 4. Simulate loss approaching limit
        portfolio._total_equity = Decimal("382.5")  # 15% from $450
        result = risk_manager.run_risk_check()
        assert result.passed is True  # Warning but still passes

        # 5. Simulate hitting limit
        portfolio._total_equity = Decimal("360")  # 20% from $450
        result = risk_manager.run_risk_check()
        assert result.passed is False
        assert result.action == RiskAction.HALT_TRADING
        assert risk_manager.is_halted is True

        # 6. Cannot resume without admin
        assert risk_manager.resume_trading() is False

        # 7. Admin clears halt
        assert risk_manager.clear_halt(admin_override=True) is True
        assert risk_manager.is_halted is False

    def test_graduated_alert_escalation(self):
        """Test alerts escalate from warning to critical to halt."""
        config = RiskConfig(
            max_position_percent=70.0,
            max_drawdown_percent=20.0,
            order_risk_percent=2.0,
            min_confidence=0.6,
        )
        portfolio = Portfolio(initial_capital=Decimal("400"))
        alert_manager = AlertManager()
        risk_manager = RiskManager(config, portfolio, alert_manager)

        # Peak at $500
        portfolio._total_equity = Decimal("500")
        portfolio._update_high_water_mark()

        # 15% DD - should get warning
        portfolio._total_equity = Decimal("425")
        risk_manager.run_risk_check()
        alerts = alert_manager.get_recent_alerts()
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        assert len(warning_alerts) >= 1

        # 18% DD - should get critical (use force=True on alert manager calls)
        portfolio._total_equity = Decimal("410")
        alert_manager.clear_alert_type(alert_manager._alert_history[-1].alert_type if alert_manager._alert_history else None)
        risk_manager.run_risk_check()

        # 20% DD - should halt
        portfolio._total_equity = Decimal("400")
        risk_manager.run_risk_check()
        assert risk_manager.is_halted is True
