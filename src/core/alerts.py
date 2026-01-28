"""
Alert System for Risk Events.

Provides:
- Alert types and severity levels
- Alert creation and routing
- Alert handlers (logging, webhooks, etc.)
- Alert history and deduplication
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"  # Informational
    WARNING = "warning"  # Needs attention
    CRITICAL = "critical"  # Immediate action required
    EMERGENCY = "emergency"  # Trading halted


class AlertType(Enum):
    """Types of risk alerts."""

    # Drawdown alerts
    DRAWDOWN_WARNING = auto()  # Approaching max drawdown
    DRAWDOWN_CRITICAL = auto()  # Near halt threshold
    DRAWDOWN_HALT = auto()  # Max drawdown exceeded - halt
    DRAWDOWN_RECOVERED = auto()  # Drawdown recovered

    # Exposure alerts
    EXPOSURE_HIGH = auto()  # Exposure approaching limit
    EXPOSURE_EXCEEDED = auto()  # Exposure limit exceeded

    # Stop-loss alerts
    STOP_LOSS_APPROACHING = auto()  # Price nearing stop-loss
    STOP_LOSS_TRIGGERED = auto()  # Stop-loss executed

    # Confidence alerts
    LOW_CONFIDENCE = auto()  # Model confidence low
    VERY_LOW_CONFIDENCE = auto()  # Model confidence very low

    # Order/Position alerts
    ORDER_FAILED = auto()  # Order submission failed
    POSITION_LARGE = auto()  # Position size unusual
    SYNC_FAILED = auto()  # Exchange sync failed

    # System alerts
    API_ERROR = auto()  # API connectivity issue
    RATE_LIMITED = auto()  # Rate limiting triggered

    # Recovery alerts
    TRADING_PAUSED = auto()  # Trading paused
    TRADING_RESUMED = auto()  # Trading resumed
    TRADING_HALTED = auto()  # Trading halted

    # Grid alerts
    REBALANCE_NEEDED = auto()  # Grid needs rebalancing
    REGIME_CHANGE = auto()  # Market regime changed


@dataclass
class Alert:
    """A single alert instance."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.alert_type.name}: {self.message}"
        )


class AlertHandler(ABC):
    """Base class for alert handlers."""

    @abstractmethod
    def handle(self, alert: Alert) -> bool:
        """
        Handle an alert.

        Args:
            alert: Alert to handle

        Returns:
            True if handled successfully
        """
        pass

    @property
    @abstractmethod
    def min_severity(self) -> AlertSeverity:
        """Minimum severity this handler processes."""
        pass


class LoggingAlertHandler(AlertHandler):
    """Handler that logs alerts."""

    def __init__(self, min_severity: AlertSeverity = AlertSeverity.INFO):
        self._min_severity = min_severity

    @property
    def min_severity(self) -> AlertSeverity:
        return self._min_severity

    def handle(self, alert: Alert) -> bool:
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }[alert.severity]

        log_method(
            f"[ALERT] {alert.alert_type.name}: {alert.message} | "
            f"Details: {alert.details}"
        )
        return True


class CallbackAlertHandler(AlertHandler):
    """Handler that calls a callback function."""

    def __init__(
        self,
        callback: Callable[[Alert], None],
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        self._callback = callback
        self._min_severity = min_severity

    @property
    def min_severity(self) -> AlertSeverity:
        return self._min_severity

    def handle(self, alert: Alert) -> bool:
        try:
            self._callback(alert)
            return True
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False


class AlertManager:
    """
    Central alert management system.

    Handles:
    - Alert creation and routing
    - Deduplication (don't spam same alert)
    - Handler registration
    - Alert history

    Usage:
        alert_mgr = AlertManager()
        alert_mgr.add_handler(LoggingAlertHandler())
        alert_mgr.add_handler(
            CallbackAlertHandler(my_callback, AlertSeverity.CRITICAL)
        )

        # Create alert
        alert_mgr.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            "Drawdown at 15%",
            {"drawdown_percent": 15.0, "threshold": 20.0}
        )
    """

    # Deduplication windows by severity
    DEDUP_WINDOWS = {
        AlertSeverity.INFO: timedelta(minutes=5),
        AlertSeverity.WARNING: timedelta(minutes=2),
        AlertSeverity.CRITICAL: timedelta(seconds=30),
        AlertSeverity.EMERGENCY: timedelta(seconds=0),  # Always send
    }

    # Severity ordering for comparison
    SEVERITY_ORDER = [
        AlertSeverity.INFO,
        AlertSeverity.WARNING,
        AlertSeverity.CRITICAL,
        AlertSeverity.EMERGENCY,
    ]

    def __init__(self, max_history: int = 1000):
        self._handlers: List[AlertHandler] = []
        self._alert_history: List[Alert] = []
        self._last_alert_times: Dict[AlertType, datetime] = {}
        self._alert_counter = 0
        self._max_history = max_history

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)
        logger.debug(f"Added alert handler: {handler.__class__.__name__}")

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Alert]:
        """
        Create and dispatch an alert.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Human-readable message
            details: Additional context
            force: Bypass deduplication

        Returns:
            Alert if created, None if deduplicated
        """
        # Check deduplication
        if not force and not self._should_send(alert_type, severity):
            logger.debug(f"Alert deduplicated: {alert_type.name}")
            return None

        # Create alert
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_counter}_{int(datetime.utcnow().timestamp())}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
        )

        # Record
        self._last_alert_times[alert_type] = datetime.utcnow()
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]

        # Dispatch to handlers
        self._dispatch(alert)

        return alert

    def _should_send(self, alert_type: AlertType, severity: AlertSeverity) -> bool:
        """Check if alert should be sent (deduplication)."""
        last_time = self._last_alert_times.get(alert_type)
        if last_time is None:
            return True

        window = self.DEDUP_WINDOWS[severity]
        return datetime.utcnow() - last_time > window

    def _dispatch(self, alert: Alert) -> None:
        """Dispatch alert to all applicable handlers."""
        alert_severity_idx = self.SEVERITY_ORDER.index(alert.severity)

        for handler in self._handlers:
            handler_min_idx = self.SEVERITY_ORDER.index(handler.min_severity)
            if alert_severity_idx >= handler_min_idx:
                try:
                    handler.handle(alert)
                except Exception as e:
                    logger.error(f"Handler {handler.__class__.__name__} failed: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False

    def get_unacknowledged(
        self,
        min_severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get unacknowledged alerts."""
        alerts = [a for a in self._alert_history if not a.acknowledged]
        if min_severity:
            min_idx = self.SEVERITY_ORDER.index(min_severity)
            alerts = [
                a
                for a in alerts
                if self.SEVERITY_ORDER.index(a.severity) >= min_idx
            ]
        return alerts

    def get_recent_alerts(
        self,
        since: Optional[datetime] = None,
        alert_type: Optional[AlertType] = None,
        min_severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get recent alerts with optional filters."""
        alerts = self._alert_history
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if min_severity:
            min_idx = self.SEVERITY_ORDER.index(min_severity)
            alerts = [
                a
                for a in alerts
                if self.SEVERITY_ORDER.index(a.severity) >= min_idx
            ]
        return alerts

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert by ID."""
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                return alert
        return None

    def clear_alert_type(self, alert_type: AlertType) -> None:
        """Clear last alert time for a type (allows immediate resend)."""
        if alert_type in self._last_alert_times:
            del self._last_alert_times[alert_type]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        severity_counts = {s.value: 0 for s in AlertSeverity}
        for alert in self._alert_history:
            severity_counts[alert.severity.value] += 1

        return {
            "total_alerts": len(self._alert_history),
            "unacknowledged": len(self.get_unacknowledged()),
            "by_severity": severity_counts,
            "handler_count": len(self._handlers),
        }


# === Convenience Functions ===


def create_drawdown_alert(
    alert_manager: AlertManager,
    drawdown_percent: float,
    warning_threshold: float = 15.0,
    critical_threshold: float = 18.0,
    halt_threshold: float = 20.0,
) -> Optional[Alert]:
    """
    Create appropriate drawdown alert based on level.

    Args:
        alert_manager: AlertManager instance
        drawdown_percent: Current drawdown percentage
        warning_threshold: Warning level (default 15%)
        critical_threshold: Critical level (default 18%)
        halt_threshold: Halt level (default 20%)

    Returns:
        Created alert or None
    """
    if drawdown_percent >= halt_threshold:
        return alert_manager.create_alert(
            AlertType.DRAWDOWN_HALT,
            AlertSeverity.EMERGENCY,
            f"MAX DRAWDOWN EXCEEDED: {drawdown_percent:.2f}% - TRADING HALTED",
            {
                "drawdown_percent": drawdown_percent,
                "threshold": halt_threshold,
                "action": "halt_trading",
            },
            force=True,
        )
    elif drawdown_percent >= critical_threshold:
        return alert_manager.create_alert(
            AlertType.DRAWDOWN_CRITICAL,
            AlertSeverity.CRITICAL,
            f"Drawdown approaching limit: {drawdown_percent:.2f}%",
            {
                "drawdown_percent": drawdown_percent,
                "threshold": halt_threshold,
                "remaining": halt_threshold - drawdown_percent,
            },
        )
    elif drawdown_percent >= warning_threshold:
        return alert_manager.create_alert(
            AlertType.DRAWDOWN_WARNING,
            AlertSeverity.WARNING,
            f"Drawdown elevated: {drawdown_percent:.2f}%",
            {
                "drawdown_percent": drawdown_percent,
                "threshold": halt_threshold,
            },
        )
    return None


def create_exposure_alert(
    alert_manager: AlertManager,
    exposure_percent: float,
    warning_threshold: float = 60.0,
    limit_threshold: float = 70.0,
) -> Optional[Alert]:
    """
    Create appropriate exposure alert based on level.

    Args:
        alert_manager: AlertManager instance
        exposure_percent: Current exposure percentage
        warning_threshold: Warning level (default 60%)
        limit_threshold: Limit level (default 70%)

    Returns:
        Created alert or None
    """
    if exposure_percent >= limit_threshold:
        return alert_manager.create_alert(
            AlertType.EXPOSURE_EXCEEDED,
            AlertSeverity.CRITICAL,
            f"Exposure limit exceeded: {exposure_percent:.2f}%",
            {
                "exposure_percent": exposure_percent,
                "limit": limit_threshold,
            },
        )
    elif exposure_percent >= warning_threshold:
        return alert_manager.create_alert(
            AlertType.EXPOSURE_HIGH,
            AlertSeverity.WARNING,
            f"Exposure elevated: {exposure_percent:.2f}%",
            {
                "exposure_percent": exposure_percent,
                "limit": limit_threshold,
            },
        )
    return None


def create_confidence_alert(
    alert_manager: AlertManager,
    confidence: float,
    min_confidence: float = 0.6,
) -> Optional[Alert]:
    """
    Create appropriate confidence alert based on level.

    Args:
        alert_manager: AlertManager instance
        confidence: Current model confidence (0-1)
        min_confidence: Minimum required confidence

    Returns:
        Created alert or None
    """
    if confidence < min_confidence * 0.5:
        return alert_manager.create_alert(
            AlertType.VERY_LOW_CONFIDENCE,
            AlertSeverity.CRITICAL,
            f"Very low model confidence: {confidence:.2%}",
            {
                "confidence": confidence,
                "min_required": min_confidence,
            },
        )
    elif confidence < min_confidence:
        return alert_manager.create_alert(
            AlertType.LOW_CONFIDENCE,
            AlertSeverity.WARNING,
            f"Low model confidence: {confidence:.2%}",
            {
                "confidence": confidence,
                "min_required": min_confidence,
            },
        )
    return None
