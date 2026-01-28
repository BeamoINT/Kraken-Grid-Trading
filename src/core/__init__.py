"""
Core Risk Management Module.

Provides:
- Portfolio state tracking with high-water mark drawdown
- Alert system for risk events
- Central risk enforcement coordinating all components
"""

from .portfolio import (
    Portfolio,
    EquitySnapshot,
    DrawdownState,
    EquitySource,
)
from .alerts import (
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
from .risk_manager import (
    RiskManager,
    RiskAction,
    RiskCheckResult,
    RiskState,
)

__all__ = [
    # Portfolio
    "Portfolio",
    "EquitySnapshot",
    "DrawdownState",
    "EquitySource",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertHandler",
    "LoggingAlertHandler",
    "CallbackAlertHandler",
    "create_drawdown_alert",
    "create_exposure_alert",
    "create_confidence_alert",
    # Risk Manager
    "RiskManager",
    "RiskAction",
    "RiskCheckResult",
    "RiskState",
]
