"""
Health Check System for Bot Monitoring.

Provides:
- API connectivity checks
- WebSocket status monitoring
- Data freshness validation
- Metrics collection
- Health status reporting
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api import KrakenPrivateClient, KrakenWebSocketClient
    from .state_manager import StateManager

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthStatus:
    """Health check result for a single component."""

    component: str
    healthy: bool
    level: HealthLevel
    message: str
    latency_ms: Optional[float] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "component": self.component,
            "healthy": self.healthy,
            "level": self.level.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_success": (
                self.last_success.isoformat() if self.last_success else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Complete system health status."""

    timestamp: datetime
    overall_healthy: bool
    overall_level: HealthLevel
    components: Dict[str, HealthStatus]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_healthy": self.overall_healthy,
            "overall_level": self.overall_level.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "metrics": self.metrics,
        }


class HealthChecker:
    """
    Monitors system health and provides metrics.

    Checks:
    - API connectivity (REST)
    - WebSocket status
    - Data freshness
    - Database connectivity

    Usage:
        checker = HealthChecker(
            rest_client=client,
            websocket=ws,
            state_manager=state_mgr,
        )

        # Run all checks
        health = await checker.run_all_checks()
        if not health.overall_healthy:
            # Handle unhealthy state

        # Get metrics
        metrics = checker.get_metrics()
    """

    # Health thresholds
    API_LATENCY_WARNING_MS = 500.0
    API_LATENCY_CRITICAL_MS = 2000.0
    DATA_FRESHNESS_WARNING_SEC = 30.0
    DATA_FRESHNESS_CRITICAL_SEC = 120.0
    WS_DISCONNECT_WARNING_SEC = 30.0
    WS_DISCONNECT_CRITICAL_SEC = 60.0

    def __init__(
        self,
        rest_client: Optional["KrakenPrivateClient"] = None,
        websocket: Optional["KrakenWebSocketClient"] = None,
        state_manager: Optional["StateManager"] = None,
        stale_data_threshold: float = 120.0,
        on_alert: Optional[Callable[[HealthStatus], None]] = None,
    ):
        """
        Initialize health checker.

        Args:
            rest_client: Kraken REST API client
            websocket: Kraken WebSocket client
            state_manager: State manager for database checks
            stale_data_threshold: Seconds before data is considered stale
            on_alert: Callback for health alerts
        """
        self._rest_client = rest_client
        self._websocket = websocket
        self._state_manager = state_manager
        self._stale_threshold = stale_data_threshold
        self._on_alert = on_alert

        # Tracking
        self._last_price_update: Optional[datetime] = None
        self._last_ohlc_update: Optional[datetime] = None
        self._last_api_success: Optional[datetime] = None
        self._api_consecutive_failures = 0
        self._ws_consecutive_failures = 0

        # Metrics
        self._startup_time = datetime.utcnow()
        self._total_api_calls = 0
        self._total_api_errors = 0
        self._total_ws_messages = 0
        self._total_ws_reconnects = 0
        self._latency_samples: List[float] = []
        self._max_latency_samples = 100

        # Current state
        self._current_price: Optional[float] = None
        self._current_regime: Optional[str] = None
        self._current_drawdown: float = 0.0

    # === Health Checks ===

    async def check_api_connectivity(self) -> HealthStatus:
        """
        Check REST API connectivity.

        Attempts a lightweight API call and measures latency.

        Returns:
            HealthStatus for API component
        """
        if self._rest_client is None:
            return HealthStatus(
                component="api",
                healthy=False,
                level=HealthLevel.UNKNOWN,
                message="REST client not configured",
            )

        start_time = time.time()
        try:
            # Use get_system_status or similar lightweight call
            # For now, simulate with a balance check
            # Run synchronous method in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._rest_client.get_balance)

            latency_ms = (time.time() - start_time) * 1000
            self._record_latency("api", latency_ms)
            self._total_api_calls += 1
            self._last_api_success = datetime.utcnow()
            self._api_consecutive_failures = 0

            # Determine health level based on latency
            if latency_ms > self.API_LATENCY_CRITICAL_MS:
                level = HealthLevel.WARNING
                message = f"API responding slowly ({latency_ms:.0f}ms)"
            elif latency_ms > self.API_LATENCY_WARNING_MS:
                level = HealthLevel.WARNING
                message = f"API latency elevated ({latency_ms:.0f}ms)"
            else:
                level = HealthLevel.HEALTHY
                message = f"API responding normally ({latency_ms:.0f}ms)"

            return HealthStatus(
                component="api",
                healthy=True,
                level=level,
                message=message,
                latency_ms=latency_ms,
                last_success=self._last_api_success,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._total_api_errors += 1
            self._api_consecutive_failures += 1

            status = HealthStatus(
                component="api",
                healthy=False,
                level=HealthLevel.CRITICAL,
                message=f"API error: {str(e)[:100]}",
                latency_ms=latency_ms,
                last_success=self._last_api_success,
                consecutive_failures=self._api_consecutive_failures,
            )

            if self._on_alert:
                self._on_alert(status)

            return status

    async def check_websocket_status(self) -> HealthStatus:
        """
        Check WebSocket connection status.

        Returns:
            HealthStatus for WebSocket component
        """
        if self._websocket is None:
            return HealthStatus(
                component="websocket",
                healthy=False,
                level=HealthLevel.UNKNOWN,
                message="WebSocket client not configured",
            )

        try:
            is_connected = self._websocket.is_connected
            state = str(self._websocket.state.value) if hasattr(self._websocket, 'state') else "unknown"

            if is_connected:
                self._ws_consecutive_failures = 0
                return HealthStatus(
                    component="websocket",
                    healthy=True,
                    level=HealthLevel.HEALTHY,
                    message=f"WebSocket connected (state: {state})",
                    details={"state": state},
                )
            else:
                self._ws_consecutive_failures += 1

                # Check how long disconnected
                if self._last_price_update:
                    disconnect_time = (
                        datetime.utcnow() - self._last_price_update
                    ).total_seconds()
                else:
                    disconnect_time = float("inf")

                if disconnect_time > self.WS_DISCONNECT_CRITICAL_SEC:
                    level = HealthLevel.CRITICAL
                elif disconnect_time > self.WS_DISCONNECT_WARNING_SEC:
                    level = HealthLevel.WARNING
                else:
                    level = HealthLevel.WARNING

                status = HealthStatus(
                    component="websocket",
                    healthy=False,
                    level=level,
                    message=f"WebSocket disconnected (state: {state})",
                    consecutive_failures=self._ws_consecutive_failures,
                    details={
                        "state": state,
                        "disconnect_seconds": disconnect_time,
                    },
                )

                if level == HealthLevel.CRITICAL and self._on_alert:
                    self._on_alert(status)

                return status

        except Exception as e:
            return HealthStatus(
                component="websocket",
                healthy=False,
                level=HealthLevel.CRITICAL,
                message=f"WebSocket check error: {str(e)[:100]}",
            )

    async def check_data_freshness(self) -> HealthStatus:
        """
        Check if we're receiving fresh market data.

        Returns:
            HealthStatus for data freshness
        """
        if self._last_price_update is None:
            return HealthStatus(
                component="data_freshness",
                healthy=False,
                level=HealthLevel.WARNING,
                message="No price data received yet",
            )

        age_seconds = (datetime.utcnow() - self._last_price_update).total_seconds()

        if age_seconds > self.DATA_FRESHNESS_CRITICAL_SEC:
            status = HealthStatus(
                component="data_freshness",
                healthy=False,
                level=HealthLevel.CRITICAL,
                message=f"Data stale ({age_seconds:.0f}s old)",
                details={"age_seconds": age_seconds},
            )
            if self._on_alert:
                self._on_alert(status)
            return status

        elif age_seconds > self.DATA_FRESHNESS_WARNING_SEC:
            return HealthStatus(
                component="data_freshness",
                healthy=True,
                level=HealthLevel.WARNING,
                message=f"Data slightly stale ({age_seconds:.0f}s old)",
                details={"age_seconds": age_seconds},
            )

        else:
            return HealthStatus(
                component="data_freshness",
                healthy=True,
                level=HealthLevel.HEALTHY,
                message=f"Data fresh ({age_seconds:.1f}s old)",
                details={"age_seconds": age_seconds},
            )

    async def check_database(self) -> HealthStatus:
        """
        Check database connectivity.

        Returns:
            HealthStatus for database component
        """
        if self._state_manager is None:
            return HealthStatus(
                component="database",
                healthy=False,
                level=HealthLevel.UNKNOWN,
                message="State manager not configured",
            )

        start_time = time.time()
        try:
            # Attempt a simple query
            stats = self._state_manager.get_stats()
            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                component="database",
                healthy=True,
                level=HealthLevel.HEALTHY,
                message=f"Database accessible ({latency_ms:.0f}ms)",
                latency_ms=latency_ms,
                details=stats,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                component="database",
                healthy=False,
                level=HealthLevel.CRITICAL,
                message=f"Database error: {str(e)[:100]}",
                latency_ms=latency_ms,
            )

    async def run_all_checks(self) -> SystemHealth:
        """
        Run all health checks.

        Returns:
            SystemHealth with overall status and component details
        """
        components: Dict[str, HealthStatus] = {}

        # Run checks
        components["api"] = await self.check_api_connectivity()
        components["websocket"] = await self.check_websocket_status()
        components["data_freshness"] = await self.check_data_freshness()
        components["database"] = await self.check_database()

        # Determine overall health
        all_healthy = all(c.healthy for c in components.values())

        # Overall level is worst of all components
        levels = [c.level for c in components.values()]
        if HealthLevel.CRITICAL in levels:
            overall_level = HealthLevel.CRITICAL
        elif HealthLevel.WARNING in levels:
            overall_level = HealthLevel.WARNING
        elif HealthLevel.UNKNOWN in levels:
            overall_level = HealthLevel.UNKNOWN
        else:
            overall_level = HealthLevel.HEALTHY

        return SystemHealth(
            timestamp=datetime.utcnow(),
            overall_healthy=all_healthy,
            overall_level=overall_level,
            components=components,
            metrics=self.get_metrics(),
        )

    # === Metrics ===

    def record_price_update(self, price: float) -> None:
        """Record a price update for freshness tracking."""
        self._last_price_update = datetime.utcnow()
        self._current_price = price
        self._total_ws_messages += 1

    def record_ohlc_update(self) -> None:
        """Record an OHLC update."""
        self._last_ohlc_update = datetime.utcnow()
        self._total_ws_messages += 1

    def record_ws_reconnect(self) -> None:
        """Record a WebSocket reconnection."""
        self._total_ws_reconnects += 1

    def record_regime(self, regime: str, drawdown: float) -> None:
        """Record current regime and drawdown."""
        self._current_regime = regime
        self._current_drawdown = drawdown

    def _record_latency(self, operation: str, latency_ms: float) -> None:
        """Record operation latency."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_latency_samples:
            self._latency_samples = self._latency_samples[-self._max_latency_samples:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary of metrics
        """
        uptime = (datetime.utcnow() - self._startup_time).total_seconds()

        # Calculate average latency
        avg_latency = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples
            else 0
        )

        return {
            "uptime_seconds": uptime,
            "total_api_calls": self._total_api_calls,
            "total_api_errors": self._total_api_errors,
            "api_error_rate": (
                self._total_api_errors / self._total_api_calls
                if self._total_api_calls > 0
                else 0
            ),
            "total_ws_messages": self._total_ws_messages,
            "total_ws_reconnects": self._total_ws_reconnects,
            "avg_api_latency_ms": avg_latency,
            "last_price_update": (
                self._last_price_update.isoformat()
                if self._last_price_update
                else None
            ),
            "last_ohlc_update": (
                self._last_ohlc_update.isoformat()
                if self._last_ohlc_update
                else None
            ),
            "current_price": self._current_price,
            "current_regime": self._current_regime,
            "current_drawdown_pct": self._current_drawdown,
        }

    def get_metrics_prometheus(self) -> str:
        """
        Get metrics in Prometheus format.

        Returns:
            Metrics string in Prometheus exposition format
        """
        metrics = self.get_metrics()
        lines = []

        lines.append(f"# HELP trading_bot_uptime_seconds Bot uptime in seconds")
        lines.append(f"# TYPE trading_bot_uptime_seconds gauge")
        lines.append(f"trading_bot_uptime_seconds {metrics['uptime_seconds']:.2f}")

        lines.append(f"# HELP trading_bot_api_calls_total Total API calls")
        lines.append(f"# TYPE trading_bot_api_calls_total counter")
        lines.append(f"trading_bot_api_calls_total {metrics['total_api_calls']}")

        lines.append(f"# HELP trading_bot_api_errors_total Total API errors")
        lines.append(f"# TYPE trading_bot_api_errors_total counter")
        lines.append(f"trading_bot_api_errors_total {metrics['total_api_errors']}")

        lines.append(f"# HELP trading_bot_ws_messages_total Total WebSocket messages")
        lines.append(f"# TYPE trading_bot_ws_messages_total counter")
        lines.append(f"trading_bot_ws_messages_total {metrics['total_ws_messages']}")

        lines.append(f"# HELP trading_bot_ws_reconnects_total Total WebSocket reconnects")
        lines.append(f"# TYPE trading_bot_ws_reconnects_total counter")
        lines.append(f"trading_bot_ws_reconnects_total {metrics['total_ws_reconnects']}")

        lines.append(f"# HELP trading_bot_api_latency_ms Average API latency")
        lines.append(f"# TYPE trading_bot_api_latency_ms gauge")
        lines.append(f"trading_bot_api_latency_ms {metrics['avg_api_latency_ms']:.2f}")

        if metrics['current_price']:
            lines.append(f"# HELP trading_bot_price Current price")
            lines.append(f"# TYPE trading_bot_price gauge")
            lines.append(f"trading_bot_price {metrics['current_price']}")

        lines.append(f"# HELP trading_bot_drawdown_pct Current drawdown percentage")
        lines.append(f"# TYPE trading_bot_drawdown_pct gauge")
        lines.append(f"trading_bot_drawdown_pct {metrics['current_drawdown_pct']:.2f}")

        return "\n".join(lines)

    # === Alert Callback ===

    def set_alert_callback(
        self,
        callback: Callable[[HealthStatus], None],
    ) -> None:
        """Set callback for health alerts."""
        self._on_alert = callback
