"""
Tests for Orchestrator module.

Tests:
- Initialization
- Lifecycle management
- State transitions
- Event handling
"""

import pytest
import asyncio
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.core import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
)
from config.settings import BotConfig, GridConfig, RiskConfig


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.feature_compute_interval == 60.0
        assert config.regime_predict_interval == 300.0
        assert config.risk_check_interval == 10.0
        assert config.exchange_sync_interval == 60.0
        assert config.health_check_interval == 60.0
        assert config.state_persist_interval == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            risk_check_interval=5.0,
            state_persist_interval=15.0,
        )

        assert config.risk_check_interval == 5.0
        assert config.state_persist_interval == 15.0


class TestOrchestratorState:
    """Tests for OrchestratorState enum."""

    def test_states_exist(self):
        """Test all expected states exist."""
        assert OrchestratorState.CREATED
        assert OrchestratorState.INITIALIZING
        assert OrchestratorState.RUNNING
        assert OrchestratorState.PAUSED
        assert OrchestratorState.SHUTTING_DOWN
        assert OrchestratorState.STOPPED


class TestOrchestratorCreation:
    """Tests for Orchestrator creation."""

    @pytest.fixture
    def mock_config(self):
        """Create mock bot configuration."""
        config = BotConfig()
        config.paper_trading = True
        config.grid = GridConfig(num_levels=5, order_size_quote=20.0)
        config.risk = RiskConfig()
        return config

    def test_creation(self, mock_config):
        """Test orchestrator creation."""
        orchestrator = Orchestrator(mock_config)

        assert orchestrator.state == OrchestratorState.CREATED
        assert orchestrator._config == mock_config

    def test_creation_with_custom_config(self, mock_config):
        """Test orchestrator creation with custom timing config."""
        orch_config = OrchestratorConfig(risk_check_interval=5.0)
        orchestrator = Orchestrator(mock_config, orch_config)

        assert orchestrator._orch_config.risk_check_interval == 5.0


class TestOrchestratorLifecycle:
    """Tests for Orchestrator lifecycle."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def mock_config(self, temp_db):
        """Create mock configuration."""
        config = BotConfig()
        config.paper_trading = True
        config.database.path = temp_db
        config.grid = GridConfig(num_levels=5, order_size_quote=20.0)
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create orchestrator with mocked dependencies."""
        return Orchestrator(mock_config)

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test initialization creates all components."""
        with patch("src.core.orchestrator.KrakenPrivateClient") as mock_client, \
             patch("src.core.orchestrator.KrakenWebSocketClient") as mock_ws:

            mock_client.from_env.return_value = Mock()
            mock_ws.return_value = Mock()

            await orchestrator.initialize()

            assert orchestrator._state_manager is not None
            assert orchestrator._portfolio is not None
            assert orchestrator._alert_manager is not None
            assert orchestrator._risk_manager is not None
            assert orchestrator._grid_calculator is not None
            assert orchestrator._grid_strategy is not None
            assert orchestrator._health_checker is not None

    @pytest.mark.asyncio
    async def test_cannot_start_without_initialize(self, orchestrator):
        """Test cannot start without initializing first."""
        with pytest.raises(RuntimeError):
            await orchestrator.start()

    @pytest.mark.asyncio
    async def test_stop_from_created(self, orchestrator):
        """Test stopping from CREATED state."""
        # Should not raise
        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.STOPPED


class TestOrchestratorStateTransitions:
    """Tests for state transitions."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    def test_initial_state(self, mock_config):
        """Test initial state is CREATED."""
        orchestrator = Orchestrator(mock_config)
        assert orchestrator.state == OrchestratorState.CREATED

    @pytest.mark.asyncio
    async def test_pause_resume(self, mock_config):
        """Test pause and resume transitions."""
        orchestrator = Orchestrator(mock_config)

        # Mock components
        orchestrator._grid_executor = Mock()
        orchestrator._grid_executor.pause_trading = Mock()
        orchestrator._grid_executor.resume_trading = Mock()

        orchestrator._risk_manager = Mock()
        orchestrator._risk_manager.pause_trading = Mock()
        orchestrator._risk_manager.resume_trading = Mock(return_value=True)

        orchestrator._alert_manager = Mock()
        orchestrator._alert_manager.create_alert = Mock()

        # Set to RUNNING state
        orchestrator._state = OrchestratorState.RUNNING

        # Pause
        await orchestrator.pause("Test pause")
        assert orchestrator.state == OrchestratorState.PAUSED

        # Resume
        await orchestrator.resume()
        assert orchestrator.state == OrchestratorState.RUNNING


class TestOrchestratorEventHandlers:
    """Tests for event handlers."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        return Orchestrator(mock_config)

    def test_on_halt_callback(self, orchestrator):
        """Test halt callback sets paused state."""
        orchestrator._state = OrchestratorState.RUNNING
        orchestrator._on_halt("Max drawdown exceeded")

        assert orchestrator.state == OrchestratorState.PAUSED

    def test_on_pause_callback(self, orchestrator):
        """Test pause callback from running state."""
        orchestrator._state = OrchestratorState.RUNNING
        orchestrator._on_pause("Low confidence")

        assert orchestrator.state == OrchestratorState.PAUSED

    def test_on_stop_loss(self, orchestrator):
        """Test stop-loss callback."""
        orchestrator._alert_manager = Mock()
        orchestrator._alert_manager.create_alert = Mock()
        orchestrator._risk_manager = Mock()
        orchestrator._risk_manager._halt_trading = Mock()

        orchestrator._on_stop_loss(Decimal("45000"))

        orchestrator._alert_manager.create_alert.assert_called_once()
        orchestrator._risk_manager._halt_trading.assert_called_once()


class TestOrchestratorMessageHandling:
    """Tests for WebSocket message handling."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        orch = Orchestrator(mock_config)
        orch._health_checker = Mock()
        orch._health_checker.record_price_update = Mock()
        orch._health_checker.record_ohlc_update = Mock()
        orch._order_manager = Mock()
        orch._order_manager.update_position_mark = Mock()
        return orch

    @pytest.mark.asyncio
    async def test_handle_ticker(self, orchestrator):
        """Test handling ticker message."""
        message = {
            "channel": "ticker",
            "data": [{"last": "50000.00"}],
        }

        await orchestrator._handle_ticker(message)

        assert orchestrator._current_price == Decimal("50000.00")
        orchestrator._health_checker.record_price_update.assert_called()

    @pytest.mark.asyncio
    async def test_handle_ohlc(self, orchestrator):
        """Test handling OHLC message."""
        message = {
            "channel": "ohlc",
            "data": [{
                "timestamp": "2025-01-15T12:00:00",
                "open": "50000",
                "high": "50500",
                "low": "49500",
                "close": "50200",
                "volume": "100",
            }],
        }

        await orchestrator._handle_ohlc(message)

        assert len(orchestrator._ohlc_buffer) == 1
        orchestrator._health_checker.record_ohlc_update.assert_called()

    @pytest.mark.asyncio
    async def test_ohlc_buffer_limit(self, orchestrator):
        """Test OHLC buffer size limit."""
        orchestrator._ohlc_buffer_size = 5

        for i in range(10):
            message = {
                "channel": "ohlc",
                "data": [{
                    "timestamp": f"2025-01-15T12:{i:02d}:00",
                    "open": str(50000 + i),
                    "high": str(50500 + i),
                    "low": str(49500 + i),
                    "close": str(50200 + i),
                    "volume": "100",
                }],
            }
            await orchestrator._handle_ohlc(message)

        assert len(orchestrator._ohlc_buffer) == 5


class TestOrchestratorStats:
    """Tests for orchestrator statistics."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        return Orchestrator(mock_config)

    def test_get_state(self, orchestrator):
        """Test getting orchestrator state."""
        orchestrator._current_price = Decimal("51000")
        orchestrator._current_confidence = 0.85

        state = orchestrator.get_state()

        assert state["state"] == "CREATED"
        assert state["current_price"] == "51000"
        assert state["current_confidence"] == 0.85

    def test_get_stats(self, orchestrator):
        """Test getting comprehensive stats."""
        stats = orchestrator.get_stats()

        assert "state" in stats
        assert stats["state"] == "CREATED"


class TestOrchestratorRegimeHandling:
    """Tests for regime change handling."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        orch = Orchestrator(mock_config)
        orch._alert_manager = Mock()
        orch._alert_manager.create_alert = Mock()
        orch._grid_strategy = Mock()
        orch._grid_executor = Mock()
        orch._portfolio = Mock()
        orch._portfolio.total_equity = Decimal("400")
        return orch

    @pytest.mark.asyncio
    async def test_regime_change_creates_alert(self, orchestrator):
        """Test regime change creates alert."""
        from src.regime import MarketRegime

        await orchestrator._handle_regime_change(
            MarketRegime.RANGING,
            MarketRegime.TRENDING_UP,
            0.85,
        )

        orchestrator._alert_manager.create_alert.assert_called_once()


class TestOrchestratorRiskHandling:
    """Tests for risk action handling."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        return Orchestrator(mock_config)

    @pytest.mark.asyncio
    async def test_handle_halt_action(self, orchestrator):
        """Test handling HALT_TRADING action."""
        from src.core import RiskAction, RiskCheckResult

        result = RiskCheckResult(
            passed=False,
            action=RiskAction.HALT_TRADING,
            reason="Max drawdown exceeded",
            details={},
        )

        await orchestrator._handle_risk_action(result)

        assert orchestrator.state == OrchestratorState.PAUSED

    @pytest.mark.asyncio
    async def test_handle_pause_action(self, orchestrator):
        """Test handling PAUSE_TRADING action."""
        from src.core import RiskAction, RiskCheckResult

        orchestrator._state = OrchestratorState.RUNNING
        orchestrator._grid_executor = Mock()
        orchestrator._grid_executor.pause_trading = Mock()
        orchestrator._risk_manager = Mock()
        orchestrator._risk_manager.pause_trading = Mock()
        orchestrator._alert_manager = Mock()
        orchestrator._alert_manager.create_alert = Mock()

        result = RiskCheckResult(
            passed=False,
            action=RiskAction.PAUSE_TRADING,
            reason="Low confidence",
            details={},
        )

        await orchestrator._handle_risk_action(result)

        assert orchestrator.state == OrchestratorState.PAUSED


class TestOrchestratorStatePersistence:
    """Tests for state persistence."""

    @pytest.fixture
    def mock_config(self):
        config = BotConfig()
        config.paper_trading = True
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        orch = Orchestrator(mock_config)
        orch._state_manager = Mock()
        orch._state_manager.save_state = Mock()
        orch._state_manager.save_metrics_snapshot = Mock()
        orch._grid_executor = Mock()
        orch._grid_executor.current_grid = None
        orch._risk_manager = Mock()
        orch._risk_manager.to_dict = Mock(return_value={})
        orch._risk_manager._pause_reason = ""
        orch._order_manager = Mock()
        orch._order_manager.position = Mock()
        orch._order_manager.position.__dict__ = {}
        orch._portfolio = Mock()
        orch._portfolio.total_equity = Decimal("400")
        orch._portfolio.current_drawdown_percent = 2.5
        orch._order_manager.get_active_orders = Mock(return_value=[])
        orch._order_manager.position.quantity = Decimal("0.001")
        return orch

    @pytest.mark.asyncio
    async def test_save_state(self, orchestrator):
        """Test saving state."""
        orchestrator._session_id = "session_test"

        await orchestrator._save_state()

        orchestrator._state_manager.save_state.assert_called_once()


class TestOrchestratorIntegration:
    """Integration tests for orchestrator."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.mark.asyncio
    async def test_full_lifecycle_mocked(self, temp_db):
        """Test full lifecycle with mocked external dependencies."""
        config = BotConfig()
        config.paper_trading = True
        config.database.path = temp_db
        config.grid = GridConfig(num_levels=5, order_size_quote=20.0)

        orchestrator = Orchestrator(config)

        with patch("src.core.orchestrator.KrakenPrivateClient") as mock_client, \
             patch("src.core.orchestrator.KrakenWebSocketClient") as mock_ws:

            # Setup mocks
            mock_client_instance = Mock()
            mock_client.from_env.return_value = mock_client_instance

            mock_ws_instance = AsyncMock()
            mock_ws_instance.connect = AsyncMock()
            mock_ws_instance.disconnect = AsyncMock()
            mock_ws_instance.subscribe_ticker = AsyncMock()
            mock_ws_instance.subscribe_ohlc = AsyncMock()
            mock_ws_instance.is_connected = True
            mock_ws_instance.__aiter__ = Mock(return_value=iter([]))
            mock_ws.return_value = mock_ws_instance

            # Initialize
            await orchestrator.initialize()
            assert orchestrator._state_manager is not None

            # Stop (without starting - simplified test)
            await orchestrator.stop()
            assert orchestrator.state == OrchestratorState.STOPPED
