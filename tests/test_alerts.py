"""
Tests for Alert System module.

Tests:
- Alert creation
- Deduplication (same alert not repeated within window)
- Emergency alerts never deduplicated
- Handler routing by severity
- Alert acknowledgment
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core import (
    AlertManager,
    Alert,
    AlertType,
    AlertSeverity,
    AlertHandler,
    LoggingAlertHandler,
    CallbackAlertHandler,
    create_drawdown_alert,
    create_exposure_alert,
    create_confidence_alert,
)


class TestAlertCreation:
    """Tests for Alert creation."""

    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.DRAWDOWN_WARNING,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            details={"value": 15.0},
        )

        assert alert.alert_id == "test_1"
        assert alert.alert_type == AlertType.DRAWDOWN_WARNING
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
        assert alert.details["value"] == 15.0
        assert alert.acknowledged is False

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.EXPOSURE_HIGH,
            severity=AlertSeverity.CRITICAL,
            message="High exposure",
            details={"exposure_percent": 65.0},
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
        )

        data = alert.to_dict()

        assert data["alert_id"] == "test_1"
        assert data["alert_type"] == "EXPOSURE_HIGH"
        assert data["severity"] == "critical"
        assert data["message"] == "High exposure"
        assert data["acknowledged"] is False

    def test_alert_str(self):
        """Test alert string representation."""
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.DRAWDOWN_HALT,
            severity=AlertSeverity.EMERGENCY,
            message="Trading halted",
            details={},
        )

        s = str(alert)
        assert "EMERGENCY" in s
        assert "DRAWDOWN_HALT" in s


class TestAlertManager:
    """Tests for AlertManager."""

    def test_create_alert(self):
        """Test alert creation through manager."""
        manager = AlertManager()

        alert = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Drawdown at 15%",
            {"drawdown_percent": 15.0},
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DRAWDOWN_WARNING
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_history(self):
        """Test alert history tracking."""
        manager = AlertManager()

        manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Alert 1",
            force=True,
        )
        manager.create_alert(
            AlertType.EXPOSURE_HIGH,
            AlertSeverity.WARNING,
            "Alert 2",
            force=True,
        )

        history = manager.get_recent_alerts()
        assert len(history) == 2

    def test_get_alert_by_id(self):
        """Test retrieving alert by ID."""
        manager = AlertManager()

        created = manager.create_alert(
            AlertType.LOW_CONFIDENCE,
            AlertSeverity.WARNING,
            "Low confidence",
            force=True,
        )

        found = manager.get_alert_by_id(created.alert_id)
        assert found is not None
        assert found.alert_id == created.alert_id

    def test_get_stats(self):
        """Test alert statistics."""
        manager = AlertManager()

        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "W1", force=True)
        manager.create_alert(AlertType.DRAWDOWN_CRITICAL, AlertSeverity.CRITICAL, "C1", force=True)
        manager.create_alert(AlertType.ORDER_FAILED, AlertSeverity.WARNING, "W2", force=True)

        stats = manager.get_stats()

        assert stats["total_alerts"] == 3
        assert stats["by_severity"]["warning"] == 2
        assert stats["by_severity"]["critical"] == 1


class TestAlertDeduplication:
    """Tests for alert deduplication."""

    def test_info_deduplication(self):
        """Test INFO alerts deduplicated within 5 min window."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.TRADING_RESUMED,
            AlertSeverity.INFO,
            "Trading resumed",
        )
        alert2 = manager.create_alert(
            AlertType.TRADING_RESUMED,
            AlertSeverity.INFO,
            "Trading resumed",
        )

        # Second should be deduplicated
        assert alert1 is not None
        assert alert2 is None

    def test_warning_deduplication(self):
        """Test WARNING alerts deduplicated within 2 min window."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Drawdown warning",
        )
        alert2 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Drawdown warning",
        )

        assert alert1 is not None
        assert alert2 is None

    def test_emergency_never_deduplicated(self):
        """Test EMERGENCY alerts are never deduplicated."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.DRAWDOWN_HALT,
            AlertSeverity.EMERGENCY,
            "HALT",
        )
        alert2 = manager.create_alert(
            AlertType.DRAWDOWN_HALT,
            AlertSeverity.EMERGENCY,
            "HALT",
        )

        # Both should be created
        assert alert1 is not None
        assert alert2 is not None

    def test_force_bypasses_deduplication(self):
        """Test force=True bypasses deduplication."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Warning",
        )
        alert2 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Warning",
            force=True,
        )

        assert alert1 is not None
        assert alert2 is not None

    def test_different_types_not_deduplicated(self):
        """Test different alert types are not deduplicated."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Drawdown",
        )
        alert2 = manager.create_alert(
            AlertType.EXPOSURE_HIGH,
            AlertSeverity.WARNING,
            "Exposure",
        )

        assert alert1 is not None
        assert alert2 is not None

    def test_clear_alert_type(self):
        """Test clearing alert type allows immediate resend."""
        manager = AlertManager()

        alert1 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Warning",
        )

        # Clear dedup
        manager.clear_alert_type(AlertType.DRAWDOWN_WARNING)

        alert2 = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Warning",
        )

        assert alert1 is not None
        assert alert2 is not None


class TestAlertHandlers:
    """Tests for alert handlers."""

    def test_logging_handler(self):
        """Test LoggingAlertHandler."""
        handler = LoggingAlertHandler(min_severity=AlertSeverity.INFO)

        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.DRAWDOWN_WARNING,
            severity=AlertSeverity.WARNING,
            message="Test",
            details={},
        )

        # Should not raise
        result = handler.handle(alert)
        assert result is True

    def test_callback_handler(self):
        """Test CallbackAlertHandler."""
        callback = Mock()
        handler = CallbackAlertHandler(callback, min_severity=AlertSeverity.WARNING)

        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.DRAWDOWN_CRITICAL,
            severity=AlertSeverity.CRITICAL,
            message="Critical",
            details={},
        )

        result = handler.handle(alert)

        assert result is True
        callback.assert_called_once_with(alert)

    def test_callback_handler_exception(self):
        """Test CallbackAlertHandler handles exceptions."""
        callback = Mock(side_effect=Exception("Callback failed"))
        handler = CallbackAlertHandler(callback)

        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.ORDER_FAILED,
            severity=AlertSeverity.WARNING,
            message="Failed",
            details={},
        )

        result = handler.handle(alert)
        assert result is False

    def test_handler_routing_by_severity(self):
        """Test handlers only receive alerts at or above their min severity."""
        manager = AlertManager()

        info_callback = Mock()
        warning_callback = Mock()
        critical_callback = Mock()

        manager.add_handler(CallbackAlertHandler(info_callback, AlertSeverity.INFO))
        manager.add_handler(CallbackAlertHandler(warning_callback, AlertSeverity.WARNING))
        manager.add_handler(CallbackAlertHandler(critical_callback, AlertSeverity.CRITICAL))

        # Create INFO alert
        manager.create_alert(AlertType.TRADING_RESUMED, AlertSeverity.INFO, "Info", force=True)

        # Only INFO handler should be called
        assert info_callback.call_count == 1
        assert warning_callback.call_count == 0
        assert critical_callback.call_count == 0

        # Create WARNING alert
        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "Warning", force=True)

        # INFO and WARNING handlers called
        assert info_callback.call_count == 2
        assert warning_callback.call_count == 1
        assert critical_callback.call_count == 0

        # Create CRITICAL alert
        manager.create_alert(AlertType.DRAWDOWN_CRITICAL, AlertSeverity.CRITICAL, "Critical", force=True)

        # All handlers called
        assert info_callback.call_count == 3
        assert warning_callback.call_count == 2
        assert critical_callback.call_count == 1

    def test_remove_handler(self):
        """Test removing handler."""
        manager = AlertManager()
        callback = Mock()
        handler = CallbackAlertHandler(callback)

        manager.add_handler(handler)
        manager.create_alert(AlertType.ORDER_FAILED, AlertSeverity.WARNING, "Test", force=True)
        assert callback.call_count == 1

        manager.remove_handler(handler)
        manager.create_alert(AlertType.ORDER_FAILED, AlertSeverity.WARNING, "Test2", force=True)
        assert callback.call_count == 1  # Not called again


class TestAlertAcknowledgment:
    """Tests for alert acknowledgment."""

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        manager = AlertManager()

        alert = manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Warning",
            force=True,
        )

        result = manager.acknowledge_alert(alert.alert_id)
        assert result is True

        # Verify acknowledged
        found = manager.get_alert_by_id(alert.alert_id)
        assert found.acknowledged is True
        assert found.acknowledged_at is not None

    def test_acknowledge_nonexistent(self):
        """Test acknowledging nonexistent alert."""
        manager = AlertManager()

        result = manager.acknowledge_alert("nonexistent_id")
        assert result is False

    def test_get_unacknowledged(self):
        """Test getting unacknowledged alerts."""
        manager = AlertManager()

        alert1 = manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "W1", force=True)
        alert2 = manager.create_alert(AlertType.EXPOSURE_HIGH, AlertSeverity.WARNING, "W2", force=True)

        # Acknowledge one
        manager.acknowledge_alert(alert1.alert_id)

        unacked = manager.get_unacknowledged()
        assert len(unacked) == 1
        assert unacked[0].alert_id == alert2.alert_id

    def test_get_unacknowledged_by_severity(self):
        """Test getting unacknowledged alerts filtered by severity."""
        manager = AlertManager()

        manager.create_alert(AlertType.TRADING_RESUMED, AlertSeverity.INFO, "Info", force=True)
        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "Warning", force=True)
        manager.create_alert(AlertType.DRAWDOWN_CRITICAL, AlertSeverity.CRITICAL, "Critical", force=True)

        # Get WARNING and above
        unacked = manager.get_unacknowledged(min_severity=AlertSeverity.WARNING)
        assert len(unacked) == 2

        # Get CRITICAL only
        unacked = manager.get_unacknowledged(min_severity=AlertSeverity.CRITICAL)
        assert len(unacked) == 1


class TestAlertFiltering:
    """Tests for alert filtering."""

    def test_filter_by_type(self):
        """Test filtering alerts by type."""
        manager = AlertManager()

        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "DD1", force=True)
        manager.create_alert(AlertType.EXPOSURE_HIGH, AlertSeverity.WARNING, "EXP1", force=True)
        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "DD2", force=True)

        filtered = manager.get_recent_alerts(alert_type=AlertType.DRAWDOWN_WARNING)
        assert len(filtered) == 2

    def test_filter_by_severity(self):
        """Test filtering alerts by severity."""
        manager = AlertManager()

        manager.create_alert(AlertType.TRADING_RESUMED, AlertSeverity.INFO, "Info", force=True)
        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "Warning", force=True)
        manager.create_alert(AlertType.DRAWDOWN_HALT, AlertSeverity.EMERGENCY, "Emergency", force=True)

        filtered = manager.get_recent_alerts(min_severity=AlertSeverity.WARNING)
        assert len(filtered) == 2

    def test_filter_by_time(self):
        """Test filtering alerts by time."""
        manager = AlertManager()

        # Create alert
        manager.create_alert(AlertType.DRAWDOWN_WARNING, AlertSeverity.WARNING, "Old", force=True)

        # Filter from future time
        future = datetime.utcnow() + timedelta(hours=1)
        filtered = manager.get_recent_alerts(since=future)
        assert len(filtered) == 0


class TestConvenienceFunctions:
    """Tests for convenience alert functions."""

    def test_create_drawdown_alert_warning(self):
        """Test drawdown warning alert creation."""
        manager = AlertManager()

        alert = create_drawdown_alert(manager, 15.5)

        assert alert is not None
        assert alert.alert_type == AlertType.DRAWDOWN_WARNING
        assert alert.severity == AlertSeverity.WARNING

    def test_create_drawdown_alert_critical(self):
        """Test drawdown critical alert creation."""
        manager = AlertManager()

        alert = create_drawdown_alert(manager, 18.5)

        assert alert is not None
        assert alert.alert_type == AlertType.DRAWDOWN_CRITICAL
        assert alert.severity == AlertSeverity.CRITICAL

    def test_create_drawdown_alert_halt(self):
        """Test drawdown halt alert creation."""
        manager = AlertManager()

        alert = create_drawdown_alert(manager, 20.5)

        assert alert is not None
        assert alert.alert_type == AlertType.DRAWDOWN_HALT
        assert alert.severity == AlertSeverity.EMERGENCY

    def test_create_drawdown_alert_below_threshold(self):
        """Test no alert below warning threshold."""
        manager = AlertManager()

        alert = create_drawdown_alert(manager, 10.0)

        assert alert is None

    def test_create_exposure_alert_warning(self):
        """Test exposure warning alert creation."""
        manager = AlertManager()

        alert = create_exposure_alert(manager, 65.0)

        assert alert is not None
        assert alert.alert_type == AlertType.EXPOSURE_HIGH
        assert alert.severity == AlertSeverity.WARNING

    def test_create_exposure_alert_exceeded(self):
        """Test exposure exceeded alert creation."""
        manager = AlertManager()

        alert = create_exposure_alert(manager, 75.0)

        assert alert is not None
        assert alert.alert_type == AlertType.EXPOSURE_EXCEEDED
        assert alert.severity == AlertSeverity.CRITICAL

    def test_create_confidence_alert_low(self):
        """Test low confidence alert creation."""
        manager = AlertManager()

        alert = create_confidence_alert(manager, 0.55, min_confidence=0.6)

        assert alert is not None
        assert alert.alert_type == AlertType.LOW_CONFIDENCE
        assert alert.severity == AlertSeverity.WARNING

    def test_create_confidence_alert_very_low(self):
        """Test very low confidence alert creation."""
        manager = AlertManager()

        alert = create_confidence_alert(manager, 0.25, min_confidence=0.6)

        assert alert is not None
        assert alert.alert_type == AlertType.VERY_LOW_CONFIDENCE
        assert alert.severity == AlertSeverity.CRITICAL


class TestAlertHistoryLimit:
    """Tests for alert history size limit."""

    def test_history_limit_enforced(self):
        """Test alert history doesn't exceed max size."""
        manager = AlertManager(max_history=10)

        # Create more alerts than limit
        for i in range(15):
            manager.create_alert(
                AlertType.ORDER_FAILED,
                AlertSeverity.WARNING,
                f"Alert {i}",
                force=True,
            )

        history = manager.get_recent_alerts()
        assert len(history) == 10

        # Should have latest alerts
        assert "Alert 14" in history[-1].message
