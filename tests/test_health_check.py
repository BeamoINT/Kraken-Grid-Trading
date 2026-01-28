"""
Tests for Health Check module.

Tests:
- Individual health checks
- Overall health status
- Metrics collection
- Alert callbacks
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.core import (
    HealthChecker,
    HealthStatus,
    SystemHealth,
    HealthLevel,
)


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy_status(self):
        """Test creating healthy status."""
        status = HealthStatus(
            component="api",
            healthy=True,
            level=HealthLevel.HEALTHY,
            message="API responding normally",
            latency_ms=150.0,
        )

        assert status.healthy is True
        assert status.level == HealthLevel.HEALTHY
        assert status.latency_ms == 150.0

    def test_unhealthy_status(self):
        """Test creating unhealthy status."""
        status = HealthStatus(
            component="websocket",
            healthy=False,
            level=HealthLevel.CRITICAL,
            message="WebSocket disconnected",
            consecutive_failures=5,
        )

        assert status.healthy is False
        assert status.level == HealthLevel.CRITICAL
        assert status.consecutive_failures == 5

    def test_status_to_dict(self):
        """Test serialization to dict."""
        ts = datetime(2025, 1, 15, 12, 0, 0)
        status = HealthStatus(
            component="database",
            healthy=True,
            level=HealthLevel.HEALTHY,
            message="Database OK",
            latency_ms=25.0,
            last_success=ts,
        )

        data = status.to_dict()

        assert data["component"] == "database"
        assert data["healthy"] is True
        assert data["level"] == "healthy"
        assert data["latency_ms"] == 25.0
        assert data["last_success"] == ts.isoformat()


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_overall_healthy(self):
        """Test overall healthy system."""
        components = {
            "api": HealthStatus(
                component="api",
                healthy=True,
                level=HealthLevel.HEALTHY,
                message="OK",
            ),
            "websocket": HealthStatus(
                component="websocket",
                healthy=True,
                level=HealthLevel.HEALTHY,
                message="OK",
            ),
        }

        health = SystemHealth(
            timestamp=datetime.utcnow(),
            overall_healthy=True,
            overall_level=HealthLevel.HEALTHY,
            components=components,
            metrics={},
        )

        assert health.overall_healthy is True
        assert health.overall_level == HealthLevel.HEALTHY

    def test_overall_unhealthy(self):
        """Test overall unhealthy system."""
        components = {
            "api": HealthStatus(
                component="api",
                healthy=True,
                level=HealthLevel.HEALTHY,
                message="OK",
            ),
            "websocket": HealthStatus(
                component="websocket",
                healthy=False,
                level=HealthLevel.CRITICAL,
                message="Disconnected",
            ),
        }

        health = SystemHealth(
            timestamp=datetime.utcnow(),
            overall_healthy=False,
            overall_level=HealthLevel.CRITICAL,
            components=components,
            metrics={},
        )

        assert health.overall_healthy is False
        assert health.overall_level == HealthLevel.CRITICAL

    def test_system_health_to_dict(self):
        """Test serialization to dict."""
        health = SystemHealth(
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            overall_healthy=True,
            overall_level=HealthLevel.WARNING,
            components={
                "api": HealthStatus(
                    component="api",
                    healthy=True,
                    level=HealthLevel.WARNING,
                    message="High latency",
                ),
            },
            metrics={"uptime": 3600},
        )

        data = health.to_dict()

        assert data["overall_healthy"] is True
        assert data["overall_level"] == "warning"
        assert "api" in data["components"]
        assert data["metrics"]["uptime"] == 3600


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker without clients."""
        return HealthChecker()

    @pytest.fixture
    def mock_rest_client(self):
        """Create mock REST client."""
        client = Mock()
        client.get_balance = AsyncMock(return_value={"USD": Mock()})
        return client

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket client."""
        ws = Mock()
        ws.is_connected = True
        ws.state = Mock()
        ws.state.value = "connected"
        return ws

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        mgr = Mock()
        mgr.get_stats = Mock(return_value={"total_orders": 10})
        return mgr

    def test_no_clients_configured(self, health_checker):
        """Test health check without configured clients."""
        # Should not raise
        assert health_checker._rest_client is None
        assert health_checker._websocket is None

    @pytest.mark.asyncio
    async def test_api_check_no_client(self, health_checker):
        """Test API check without client."""
        status = await health_checker.check_api_connectivity()

        assert status.healthy is False
        assert status.level == HealthLevel.UNKNOWN
        assert "not configured" in status.message.lower()

    @pytest.mark.asyncio
    async def test_api_check_success(self, mock_rest_client):
        """Test successful API check."""
        checker = HealthChecker(rest_client=mock_rest_client)

        status = await checker.check_api_connectivity()

        assert status.healthy is True
        assert status.latency_ms is not None
        mock_rest_client.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_check_failure(self, mock_rest_client):
        """Test API check failure."""
        mock_rest_client.get_balance = AsyncMock(
            side_effect=Exception("Connection error")
        )
        checker = HealthChecker(rest_client=mock_rest_client)

        status = await checker.check_api_connectivity()

        assert status.healthy is False
        assert status.level == HealthLevel.CRITICAL
        assert "error" in status.message.lower()

    @pytest.mark.asyncio
    async def test_websocket_check_connected(self, mock_websocket):
        """Test WebSocket check when connected."""
        checker = HealthChecker(websocket=mock_websocket)

        status = await checker.check_websocket_status()

        assert status.healthy is True
        assert status.level == HealthLevel.HEALTHY

    @pytest.mark.asyncio
    async def test_websocket_check_disconnected(self, mock_websocket):
        """Test WebSocket check when disconnected."""
        mock_websocket.is_connected = False
        checker = HealthChecker(websocket=mock_websocket)

        # Record a price update in the past
        checker._last_price_update = datetime.utcnow() - timedelta(seconds=90)

        status = await checker.check_websocket_status()

        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_data_freshness_no_data(self, health_checker):
        """Test data freshness with no data."""
        status = await health_checker.check_data_freshness()

        assert status.healthy is False
        assert status.level == HealthLevel.WARNING
        assert "no" in status.message.lower() or "not" in status.message.lower()

    @pytest.mark.asyncio
    async def test_data_freshness_fresh(self, health_checker):
        """Test data freshness with fresh data."""
        health_checker.record_price_update(50000.0)

        status = await health_checker.check_data_freshness()

        assert status.healthy is True
        assert status.level == HealthLevel.HEALTHY

    @pytest.mark.asyncio
    async def test_data_freshness_stale(self, health_checker):
        """Test data freshness with stale data."""
        health_checker._last_price_update = datetime.utcnow() - timedelta(seconds=150)

        status = await health_checker.check_data_freshness()

        assert status.healthy is False
        assert status.level == HealthLevel.CRITICAL
        assert "stale" in status.message.lower()

    @pytest.mark.asyncio
    async def test_database_check_success(self, mock_state_manager):
        """Test database check success."""
        checker = HealthChecker(state_manager=mock_state_manager)

        status = await checker.check_database()

        assert status.healthy is True
        assert status.latency_ms is not None
        mock_state_manager.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_check_failure(self, mock_state_manager):
        """Test database check failure."""
        mock_state_manager.get_stats = Mock(side_effect=Exception("DB error"))
        checker = HealthChecker(state_manager=mock_state_manager)

        status = await checker.check_database()

        assert status.healthy is False
        assert status.level == HealthLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_run_all_checks(
        self, mock_rest_client, mock_websocket, mock_state_manager
    ):
        """Test running all health checks."""
        checker = HealthChecker(
            rest_client=mock_rest_client,
            websocket=mock_websocket,
            state_manager=mock_state_manager,
        )
        checker.record_price_update(50000.0)

        health = await checker.run_all_checks()

        assert health.overall_healthy is True
        assert "api" in health.components
        assert "websocket" in health.components
        assert "database" in health.components
        assert "data_freshness" in health.components


class TestMetrics:
    """Tests for metrics collection."""

    @pytest.fixture
    def health_checker(self):
        return HealthChecker()

    def test_record_price_update(self, health_checker):
        """Test recording price update."""
        health_checker.record_price_update(51000.0)

        assert health_checker._last_price_update is not None
        assert health_checker._current_price == 51000.0
        assert health_checker._total_ws_messages == 1

    def test_record_ohlc_update(self, health_checker):
        """Test recording OHLC update."""
        health_checker.record_ohlc_update()

        assert health_checker._last_ohlc_update is not None
        assert health_checker._total_ws_messages == 1

    def test_record_ws_reconnect(self, health_checker):
        """Test recording WebSocket reconnect."""
        health_checker.record_ws_reconnect()
        health_checker.record_ws_reconnect()

        assert health_checker._total_ws_reconnects == 2

    def test_record_regime(self, health_checker):
        """Test recording regime."""
        health_checker.record_regime("TRENDING_UP", 5.5)

        assert health_checker._current_regime == "TRENDING_UP"
        assert health_checker._current_drawdown == 5.5

    def test_get_metrics(self, health_checker):
        """Test getting metrics."""
        health_checker.record_price_update(50000.0)
        health_checker.record_regime("RANGING", 2.5)

        metrics = health_checker.get_metrics()

        assert "uptime_seconds" in metrics
        assert metrics["uptime_seconds"] >= 0
        assert metrics["total_ws_messages"] == 1
        assert metrics["current_price"] == 50000.0
        assert metrics["current_regime"] == "RANGING"
        assert metrics["current_drawdown_pct"] == 2.5

    def test_metrics_prometheus_format(self, health_checker):
        """Test Prometheus format metrics."""
        health_checker.record_price_update(52000.0)

        prom = health_checker.get_metrics_prometheus()

        assert "trading_bot_uptime_seconds" in prom
        assert "trading_bot_price 52000" in prom
        assert "TYPE" in prom  # Has type annotations


class TestAlertCallback:
    """Tests for alert callbacks."""

    @pytest.fixture
    def health_checker(self):
        return HealthChecker()

    def test_set_alert_callback(self, health_checker):
        """Test setting alert callback."""
        callback = Mock()
        health_checker.set_alert_callback(callback)

        assert health_checker._on_alert == callback

    @pytest.mark.asyncio
    async def test_alert_triggered_on_critical(self):
        """Test alert triggered on critical health."""
        callback = Mock()
        mock_client = Mock()
        mock_client.get_balance = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        checker = HealthChecker(
            rest_client=mock_client,
            on_alert=callback,
        )

        await checker.check_api_connectivity()

        callback.assert_called_once()
        call_arg = callback.call_args[0][0]
        assert call_arg.level == HealthLevel.CRITICAL


class TestHealthThresholds:
    """Tests for health threshold behavior."""

    @pytest.mark.asyncio
    async def test_api_latency_warning(self):
        """Test API latency warning threshold."""
        mock_client = Mock()

        # Simulate slow response
        async def slow_balance():
            import asyncio
            await asyncio.sleep(0.6)  # 600ms
            return {}

        mock_client.get_balance = slow_balance
        checker = HealthChecker(rest_client=mock_client)

        status = await checker.check_api_connectivity()

        assert status.healthy is True  # Still healthy
        assert status.level == HealthLevel.WARNING
        assert status.latency_ms > 500

    @pytest.mark.asyncio
    async def test_data_freshness_warning_threshold(self):
        """Test data freshness warning threshold."""
        checker = HealthChecker()
        checker._last_price_update = datetime.utcnow() - timedelta(seconds=60)

        status = await checker.check_data_freshness()

        assert status.healthy is True  # Still healthy
        assert status.level == HealthLevel.WARNING

    @pytest.mark.asyncio
    async def test_data_freshness_critical_threshold(self):
        """Test data freshness critical threshold."""
        checker = HealthChecker(stale_data_threshold=120.0)
        checker._last_price_update = datetime.utcnow() - timedelta(seconds=150)

        status = await checker.check_data_freshness()

        assert status.healthy is False
        assert status.level == HealthLevel.CRITICAL
