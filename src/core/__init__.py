"""
Core Trading Module.

Provides:
- Portfolio state tracking with high-water mark drawdown
- Alert system for risk events
- Central risk enforcement coordinating all components
- Orchestrator for main loop coordination
- State management for persistence and recovery
- Health monitoring
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
from .state_manager import (
    StateManager,
    BotState,
)
from .health_check import (
    HealthChecker,
    HealthStatus,
    SystemHealth,
    HealthLevel,
)
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
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
    # State Manager
    "StateManager",
    "BotState",
    # Health Check
    "HealthChecker",
    "HealthStatus",
    "SystemHealth",
    "HealthLevel",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
]
