"""
Orchestrator - Central Coordinator for the Trading Bot.

Provides:
- Component initialization and wiring
- Async main loop coordination
- Event routing (price updates, fills, regime changes)
- Graceful startup and shutdown
- State management and recovery
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from config.settings import BotConfig
from src.api import (
    KrakenPrivateClient,
    KrakenWebSocketClient,
    OrderManager,
    OrderManagerConfig,
    GridOrder,
    GridOrderType,
)
from src.grid import (
    GridCalculator,
    GridStrategy,
    GridExecutor,
    Rebalancer,
    GridParameters,
)
from src.regime import MarketRegime
from .portfolio import Portfolio
from .alerts import AlertManager, AlertType, AlertSeverity, LoggingAlertHandler
from .risk_manager import RiskManager, RiskAction
from .state_manager import StateManager, BotState
from .health_check import HealthChecker, HealthLevel
from src.data.data_sync import DataSyncService, DataSyncConfig

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""

    CREATED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class OrchestratorConfig:
    """Timing intervals for the main loop."""

    feature_compute_interval: float = 60.0  # 1 min
    regime_predict_interval: float = 300.0  # 5 min
    risk_check_interval: float = 10.0  # 10 sec
    exchange_sync_interval: float = 60.0  # 1 min
    health_check_interval: float = 60.0  # 1 min
    state_persist_interval: float = 30.0  # 30 sec
    rebalance_check_interval: float = 60.0  # 1 min
    data_sync_interval: float = 3600.0  # 1 hour - sync historical data
    shutdown_timeout: float = 10.0  # 10 sec

    # Recovery settings
    max_state_age_seconds: int = 86400  # 24 hours (default)
    skip_state_restore: bool = False  # Skip state restoration if True
    always_reconcile: bool = True  # Always reconcile with exchange on startup

    # Data sync settings
    data_sync_enabled: bool = True  # Enable periodic data sync
    data_sync_timeframes: List[str] = field(default_factory=lambda: ["1h", "4h"])


class Orchestrator:
    """
    Central coordinator for the trading bot.

    Responsibilities:
    - Initialize all components in correct dependency order
    - Run async main loop coordinating data flow
    - Handle order fill callbacks
    - Manage graceful shutdown
    - Support paper trading mode

    Usage:
        config = BotConfig(...)
        orchestrator = Orchestrator(config)

        await orchestrator.initialize()
        await orchestrator.start()

        # Bot runs until stopped
        await orchestrator.stop()
    """

    def __init__(
        self,
        config: BotConfig,
        orchestrator_config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Bot configuration
            orchestrator_config: Timing configuration (uses defaults if None)
        """
        self._config = config
        self._orch_config = orchestrator_config or OrchestratorConfig()
        self._state = OrchestratorState.CREATED

        # Components (initialized in initialize())
        self._state_manager: Optional[StateManager] = None
        self._rest_client: Optional[KrakenPrivateClient] = None
        self._websocket: Optional[KrakenWebSocketClient] = None
        self._portfolio: Optional[Portfolio] = None
        self._alert_manager: Optional[AlertManager] = None
        self._risk_manager: Optional[RiskManager] = None
        self._grid_calculator: Optional[GridCalculator] = None
        self._grid_strategy: Optional[GridStrategy] = None
        self._order_manager: Optional[OrderManager] = None
        self._grid_executor: Optional[GridExecutor] = None
        self._rebalancer: Optional[Rebalancer] = None
        self._health_checker: Optional[HealthChecker] = None
        self._data_sync_service: Optional[DataSyncService] = None

        # ML components (optional)
        self._classifier = None
        self._scaler = None
        self._feature_buffer: List[Dict[str, Any]] = []

        # Tasks
        self._tasks: List[asyncio.Task] = []

        # Runtime state
        self._current_price: Optional[Decimal] = None
        self._current_regime: MarketRegime = MarketRegime.RANGING
        self._previous_regime: MarketRegime = MarketRegime.RANGING
        self._current_confidence: float = 0.5
        self._session_id: Optional[str] = None
        self._shutdown_event = asyncio.Event()

        # OHLC buffer for feature computation
        self._ohlc_buffer: List[Dict[str, Any]] = []
        self._ohlc_buffer_size = 200  # Keep last 200 candles

        logger.info(
            f"Orchestrator created (paper_trading={config.paper_trading})"
        )

    @property
    def state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self._state

    @property
    def state_manager(self) -> Optional[StateManager]:
        """Get state manager (for external access)."""
        return self._state_manager

    # === Lifecycle Methods ===

    async def initialize(self) -> None:
        """
        Initialize all components in dependency order.

        Raises:
            Exception: If initialization fails
        """
        self._state = OrchestratorState.INITIALIZING
        logger.info("Initializing orchestrator...")

        try:
            # 1. State Manager
            self._state_manager = StateManager(self._config.database.path)
            logger.info("StateManager initialized")

            # 2. REST Client
            self._rest_client = KrakenPrivateClient.from_env()
            if self._config.paper_trading:
                self._rest_client._paper_trading = True
            logger.info("REST client initialized")

            # 3. WebSocket Client
            self._websocket = KrakenWebSocketClient(
                config=self._config.websocket,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_state_change=self._on_ws_state_change,
            )
            logger.info("WebSocket client initialized")

            # 4. Portfolio
            initial_capital = Decimal(str(self._config.grid.total_capital_required))
            self._portfolio = Portfolio(initial_capital=initial_capital)
            logger.info(f"Portfolio initialized with ${initial_capital}")

            # 5. Alert Manager
            self._alert_manager = AlertManager()
            self._alert_manager.add_handler(LoggingAlertHandler())
            logger.info("AlertManager initialized")

            # 6. Risk Manager
            self._risk_manager = RiskManager(
                config=self._config.risk,
                portfolio=self._portfolio,
                alert_manager=self._alert_manager,
                on_halt=self._on_halt,
                on_pause=self._on_pause,
            )
            logger.info("RiskManager initialized")

            # 7. Try to load ML model (optional)
            await self._load_ml_model()

            # 8. Grid Calculator
            self._grid_calculator = GridCalculator(
                config=self._config.grid,
                risk_config=self._config.risk,
            )
            logger.info("GridCalculator initialized")

            # 9. Grid Strategy
            self._grid_strategy = GridStrategy(
                grid_config=self._config.grid,
                risk_config=self._config.risk,
                calculator=self._grid_calculator,
            )
            logger.info("GridStrategy initialized")

            # 10. Order Manager
            order_config = OrderManagerConfig(
                pair=self._config.trading.pair,
                base_asset=self._config.trading.base_currency,
                quote_asset=self._config.trading.quote_currency,
            )
            self._order_manager = OrderManager(
                client=self._rest_client,
                config=order_config,
                on_fill=self._on_order_fill_internal,
            )
            logger.info("OrderManager initialized")

            # 11. Grid Executor
            self._grid_executor = GridExecutor(
                order_manager=self._order_manager,
                strategy=self._grid_strategy,
                risk_config=self._config.risk,
                capital=initial_capital,
                on_fill=self._on_grid_fill,
                on_stop_loss=self._on_stop_loss,
            )
            logger.info("GridExecutor initialized")

            # 12. Rebalancer
            self._rebalancer = Rebalancer(
                rebalance_threshold=self._config.grid.rebalance_threshold,
                max_capital=initial_capital,
            )
            logger.info("Rebalancer initialized")

            # 13. Health Checker
            self._health_checker = HealthChecker(
                rest_client=self._rest_client,
                websocket=self._websocket,
                state_manager=self._state_manager,
            )
            logger.info("HealthChecker initialized")

            # 14. Data Sync Service (optional - keeps historical data updated)
            if self._orch_config.data_sync_enabled:
                data_sync_config = DataSyncConfig(
                    trade_sync_interval=self._orch_config.data_sync_interval,
                    pairs=[self._config.trading.pair],
                    timeframes=self._orch_config.data_sync_timeframes,
                    raw_data_path=Path("data/raw"),
                    ohlcv_path=Path("data/ohlcv"),
                )
                self._data_sync_service = DataSyncService(config=data_sync_config)
                logger.info("DataSyncService initialized")

            # 15. Restore state if exists
            await self._restore_state()

            logger.info("Orchestrator initialization complete")

        except Exception as e:
            self._state = OrchestratorState.ERROR
            logger.exception(f"Initialization failed: {e}")
            raise

    async def _load_ml_model(self) -> None:
        """Load ML model from registry if available."""
        try:
            from src.models import ModelRegistry

            registry = ModelRegistry(registry_path=self._config.storage.models_path)
            result = registry.get_production_model()

            if result:
                self._classifier, metadata, self._scaler = result
                logger.info(
                    f"Loaded production model: {metadata.model_id}"
                )
            else:
                logger.warning(
                    "No production model found, using rule-based regime detection"
                )
        except ImportError:
            logger.warning("ML models not available, using rule-based regime detection")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}, using rule-based regime")

    async def _restore_state(self) -> None:
        """Restore state from persistence if available."""
        # Check if state restoration should be skipped
        if self._orch_config.skip_state_restore:
            logger.info("State restoration skipped (--no-resume flag)")
            return

        # Check for pending operations (WAL recovery)
        await self._recover_pending_operations()

        if not self._state_manager.has_state():
            logger.info("No saved state to restore")
            return

        # Use configurable state age limit
        state_age = self._state_manager.get_state_age_seconds()
        max_age = self._orch_config.max_state_age_seconds
        if state_age and state_age > max_age:
            logger.warning(
                f"Saved state is {state_age/3600:.1f} hours old "
                f"(max: {max_age/3600:.1f}h), starting fresh"
            )
            return

        try:
            saved_state = self._state_manager.load_state()
            if saved_state is None:
                return

            # Restore risk state
            if saved_state.risk_state and self._risk_manager:
                self._risk_manager.restore_from_dict(saved_state.risk_state)
                logger.info("Restored risk state")

            # Restore regime
            if saved_state.last_regime:
                try:
                    self._current_regime = MarketRegime[saved_state.last_regime]
                    self._current_confidence = saved_state.last_confidence
                    logger.info(
                        f"Restored regime: {self._current_regime.name} "
                        f"(confidence: {self._current_confidence:.2f})"
                    )
                except KeyError:
                    pass

            # Restore grid state
            if saved_state.grid_snapshot and self._grid_executor:
                # Grid restoration requires syncing with exchange first
                logger.info("Grid state found, will restore after exchange sync")

            logger.info("State restoration complete")

        except Exception as e:
            logger.warning(f"Failed to restore state: {e}, starting fresh")

    async def _recover_pending_operations(self) -> None:
        """Recover incomplete operations from Write-Ahead Log."""
        pending_ops = self._state_manager.get_pending_operations()
        if not pending_ops:
            return

        logger.warning(f"Found {len(pending_ops)} pending operations from previous run")

        for op in pending_ops:
            op_id = op.get("id")
            op_type = op.get("operation_type")
            op_data = op.get("operation_data", {})
            grid_id = op.get("grid_id")

            logger.info(f"Recovering operation {op_id}: {op_type} for {grid_id}")

            try:
                if op_type == "SUBMIT_ORDER":
                    # Check if order exists on exchange
                    # For now, mark as failed - reconciliation will handle actual state
                    self._state_manager.complete_operation(op_id, success=False)
                    logger.info(f"Marked pending SUBMIT_ORDER {op_id} as failed (will reconcile)")

                elif op_type == "CANCEL_ORDER":
                    # Check if order still exists on exchange
                    # For now, mark as completed - order likely canceled or gone
                    self._state_manager.complete_operation(op_id, success=True)
                    logger.info(f"Marked pending CANCEL_ORDER {op_id} as completed")

                else:
                    # Unknown operation type, mark as failed
                    self._state_manager.complete_operation(op_id, success=False)
                    logger.warning(f"Unknown operation type {op_type}, marked as failed")

            except Exception as e:
                logger.error(f"Error recovering operation {op_id}: {e}")
                self._state_manager.complete_operation(op_id, success=False)

    async def start(self) -> None:
        """
        Start the orchestrator main loop.

        Creates and starts async tasks for:
        - WebSocket connection and message handling
        - Periodic feature computation
        - Periodic regime prediction
        - Periodic risk checks
        - Periodic exchange sync
        - Periodic health checks
        - Periodic state persistence
        """
        if self._state != OrchestratorState.INITIALIZING:
            raise RuntimeError(
                f"Cannot start from state {self._state.name}, must be INITIALIZING"
            )

        logger.info("Starting orchestrator...")
        self._shutdown_event.clear()

        # Start session
        self._session_id = self._state_manager.start_session(
            initial_capital=self._portfolio.total_equity if self._portfolio else None
        )
        logger.info(f"Started session: {self._session_id}")

        # Connect WebSocket
        await self._websocket.connect()
        logger.info("WebSocket connected")

        # Subscribe to market data
        symbol = self._config.trading.ws_symbol
        await self._websocket.subscribe_ticker([symbol])
        await self._websocket.subscribe_ohlc([symbol], interval=5)
        logger.info(f"Subscribed to {symbol} ticker and OHLC")

        # Sync with exchange
        await self._sync_with_exchange()

        # Get initial price
        await self._fetch_initial_price()

        # Deploy initial grid if not paused
        if not self._risk_manager.is_halted and not self._risk_manager.is_paused:
            await self._deploy_initial_grid()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._websocket_task(), name="websocket"),
            asyncio.create_task(self._risk_check_task(), name="risk_check"),
            asyncio.create_task(self._exchange_sync_task(), name="exchange_sync"),
            asyncio.create_task(self._health_check_task(), name="health_check"),
            asyncio.create_task(self._state_persist_task(), name="state_persist"),
            asyncio.create_task(self._regime_predict_task(), name="regime_predict"),
            asyncio.create_task(self._rebalance_check_task(), name="rebalance_check"),
        ]

        # Add data sync task if enabled
        if self._data_sync_service is not None:
            self._tasks.append(
                asyncio.create_task(self._data_sync_task(), name="data_sync")
            )

        self._state = OrchestratorState.RUNNING
        logger.info("Orchestrator started")

    async def stop(self, cancel_orders: bool = True, emergency: bool = False) -> None:
        """
        Gracefully stop the orchestrator.

        Args:
            cancel_orders: Whether to cancel all open orders
            emergency: If True, skip order cancellation (for second signal)
        """
        if self._state in (OrchestratorState.STOPPED, OrchestratorState.SHUTTING_DOWN):
            return

        if emergency:
            logger.warning("Emergency shutdown - saving state only")
        else:
            logger.info("Stopping orchestrator...")

        self._state = OrchestratorState.SHUTTING_DOWN
        self._shutdown_event.set()

        # Save state FIRST before any cleanup (critical for crash recovery)
        try:
            await self._save_state()
            logger.info("State saved before shutdown")
        except Exception as e:
            logger.error(f"Failed to save state during shutdown: {e}")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=self._orch_config.shutdown_timeout,
            )
            if pending:
                logger.warning(f"{len(pending)} tasks did not complete within timeout")

        # Cancel orders if requested and not emergency
        if cancel_orders and not emergency and self._grid_executor:
            logger.info("Canceling all orders...")
            # Retry logic for order cancellation
            for attempt in range(3):
                try:
                    result = self._grid_executor.cancel_all_orders()
                    logger.info(f"Canceled orders: {result}")
                    break
                except Exception as e:
                    logger.error(f"Cancel attempt {attempt + 1}/3 failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)

        # Save final state after order cancellation
        try:
            await self._save_state()
        except Exception as e:
            logger.error(f"Failed to save final state: {e}")

        # End session
        if self._session_id and self._state_manager:
            try:
                stats = self.get_stats()
                self._state_manager.end_session(self._session_id, stats)
            except Exception as e:
                logger.error(f"Failed to end session: {e}")

        # Disconnect WebSocket
        if self._websocket:
            try:
                await self._websocket.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect WebSocket: {e}")

        self._state = OrchestratorState.STOPPED
        logger.info("Orchestrator stopped")

    async def pause(self, reason: str) -> None:
        """Pause trading (cancel orders, keep state)."""
        if self._state != OrchestratorState.RUNNING:
            return

        logger.info(f"Pausing trading: {reason}")
        self._state = OrchestratorState.PAUSED

        if self._grid_executor:
            self._grid_executor.pause_trading(reason)

        if self._risk_manager:
            self._risk_manager.pause_trading(reason)

        # Create alert
        if self._alert_manager:
            self._alert_manager.create_alert(
                AlertType.TRADING_PAUSED,
                AlertSeverity.WARNING,
                f"Trading paused: {reason}",
                {"reason": reason},
            )

    async def resume(self) -> None:
        """Resume trading after pause."""
        if self._state != OrchestratorState.PAUSED:
            return

        if self._risk_manager and not self._risk_manager.resume_trading():
            logger.warning("Cannot resume - risk manager blocked")
            return

        logger.info("Resuming trading")
        self._state = OrchestratorState.RUNNING

        if self._grid_executor:
            self._grid_executor.resume_trading()

        # Create alert
        if self._alert_manager:
            self._alert_manager.create_alert(
                AlertType.TRADING_RESUMED,
                AlertSeverity.INFO,
                "Trading resumed",
                {},
            )

    # === Main Loop Tasks ===

    async def _websocket_task(self) -> None:
        """Handle WebSocket messages."""
        logger.info("WebSocket task started")
        msg_count = 0

        try:
            async for message in self._websocket:
                msg_count += 1
                logger.debug(f"Received WS message #{msg_count}: channel={message.get('channel')}")

                if self._shutdown_event.is_set():
                    break

                try:
                    await self._process_ws_message(message)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except asyncio.CancelledError:
            logger.info("WebSocket task cancelled")
        except Exception as e:
            logger.exception(f"WebSocket task error: {e}")

    async def _process_ws_message(self, message: Dict[str, Any]) -> None:
        """Process a WebSocket message."""
        channel = message.get("channel")

        if channel == "ticker":
            await self._handle_ticker(message)
        elif channel == "ohlc":
            await self._handle_ohlc(message)
        elif channel == "ownTrades":
            await self._handle_own_trades(message)

    async def _handle_ticker(self, message: Dict[str, Any]) -> None:
        """Handle ticker message."""
        try:
            data = message.get("data", [])
            if not data:
                return

            ticker = data[0] if isinstance(data, list) else data
            price_str = ticker.get("last", ticker.get("c", [None])[0])

            if price_str:
                self._current_price = Decimal(str(price_str))
                logger.info(f"Price update: ${self._current_price}")

                # Update health checker
                if self._health_checker:
                    self._health_checker.record_price_update(float(self._current_price))

                # Update position mark price
                if self._order_manager:
                    self._order_manager.update_position_mark(self._current_price)

        except Exception as e:
            logger.warning(f"Error handling ticker: {e}")

    async def _handle_ohlc(self, message: Dict[str, Any]) -> None:
        """Handle OHLC candle message."""
        try:
            data = message.get("data", [])
            if not data:
                return

            candle = data[0] if isinstance(data, list) else data

            # Add to buffer
            self._ohlc_buffer.append({
                "timestamp": candle.get("timestamp"),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0)),
            })

            # Keep buffer size limited
            if len(self._ohlc_buffer) > self._ohlc_buffer_size:
                self._ohlc_buffer = self._ohlc_buffer[-self._ohlc_buffer_size:]

            if self._health_checker:
                self._health_checker.record_ohlc_update()

        except Exception as e:
            logger.debug(f"Error handling OHLC: {e}")

    async def _handle_own_trades(self, message: Dict[str, Any]) -> None:
        """Handle own trades message (fill notification)."""
        try:
            data = message.get("data", [])
            for trade in data:
                order_id = trade.get("ordertxid")
                volume = Decimal(str(trade.get("vol", 0)))
                price = Decimal(str(trade.get("price", 0)))

                if order_id and self._order_manager:
                    order = self._order_manager.get_order_by_exchange_id(order_id)
                    if order:
                        logger.info(
                            f"Fill received: order={order.grid_id}, "
                            f"volume={volume}, price={price}"
                        )

        except Exception as e:
            logger.error(f"Error handling own trades: {e}")

    async def _regime_predict_task(self) -> None:
        """Periodic regime prediction."""
        logger.info("Regime predict task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.regime_predict_interval)

                if self._shutdown_event.is_set():
                    break

                try:
                    await self._predict_regime()
                except Exception as e:
                    logger.error(f"Regime prediction error: {e}")

        except asyncio.CancelledError:
            logger.info("Regime predict task cancelled")

    async def _predict_regime(self) -> None:
        """Predict market regime from current features."""
        if self._classifier is None:
            # Use rule-based fallback
            self._current_regime = MarketRegime.RANGING
            self._current_confidence = 0.5
            return

        if len(self._ohlc_buffer) < 50:
            logger.debug("Not enough OHLC data for prediction")
            return

        try:
            # Build DataFrame from OHLC buffer
            df = pd.DataFrame(self._ohlc_buffer)

            # Compute features
            from src.features import FeaturePipeline
            pipeline = FeaturePipeline()
            features_df = pipeline.compute_all_features(df)

            if features_df.empty:
                return

            # Get latest row
            latest = features_df.iloc[-1:].copy()

            # Scale features
            if self._scaler:
                feature_cols = [c for c in latest.columns if c not in [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ]]
                latest[feature_cols] = self._scaler.transform(latest[feature_cols])

            # Predict
            predictions, confidences, is_confident = self._classifier.predict_with_confidence(
                latest[feature_cols],
                min_confidence=self._config.risk.min_confidence,
            )

            new_regime = MarketRegime(predictions[0])
            confidence = confidences[0]

            # Check for regime change
            if new_regime != self._current_regime:
                self._previous_regime = self._current_regime
                self._current_regime = new_regime
                self._current_confidence = confidence

                await self._handle_regime_change(
                    self._previous_regime,
                    new_regime,
                    confidence,
                )
            else:
                self._current_confidence = confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")

    async def _risk_check_task(self) -> None:
        """Periodic risk checks."""
        logger.info("Risk check task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.risk_check_interval)

                if self._shutdown_event.is_set():
                    break

                try:
                    await self._check_risk()
                except Exception as e:
                    logger.error(f"Risk check error: {e}")

        except asyncio.CancelledError:
            logger.info("Risk check task cancelled")

    async def _check_risk(self) -> None:
        """Run risk checks."""
        if not self._risk_manager or not self._order_manager:
            return

        # Update portfolio from position
        position = self._order_manager.position
        if self._current_price and self._portfolio:
            buy_exp, sell_exp = self._order_manager.get_open_order_exposure()
            self._portfolio.update_from_position(
                position_quantity=position.quantity,
                avg_entry_price=position.avg_entry_price,
                current_price=self._current_price,
                realized_pnl=position.realized_pnl,
                open_order_exposure=buy_exp + sell_exp,
            )

        # Run risk check
        result = self._risk_manager.run_risk_check(
            current_price=self._current_price,
            position=position,
            order_manager=self._order_manager,
        )

        if not result.passed:
            await self._handle_risk_action(result)

        # Check stop-loss
        if self._grid_executor and self._current_price:
            if self._grid_executor.check_stop_loss(self._current_price):
                logger.warning("Stop-loss triggered!")
                # GridExecutor handles the callback

        # Update health checker
        if self._health_checker and self._portfolio:
            self._health_checker.record_regime(
                self._current_regime.name,
                self._portfolio.current_drawdown_percent,
            )

    async def _exchange_sync_task(self) -> None:
        """Periodic exchange sync."""
        logger.info("Exchange sync task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.exchange_sync_interval)

                if self._shutdown_event.is_set():
                    break

                try:
                    await self._sync_with_exchange()
                except Exception as e:
                    logger.error(f"Exchange sync error: {e}")

        except asyncio.CancelledError:
            logger.info("Exchange sync task cancelled")

    async def _sync_with_exchange(self) -> None:
        """Sync local state with exchange."""
        if not self._order_manager:
            return

        # Sync orders (paper trading simulates order fills locally)
        if not self._config.paper_trading:
            state_changes = self._order_manager.sync_orders()
            if state_changes:
                logger.debug(f"Order state changes: {state_changes}")

        # Sync balance - skip in paper trading mode to use simulated balance
        if self._config.paper_trading:
            logger.debug("Paper trading: using simulated balance (skipping API sync)")
            return

        if self._rest_client and self._portfolio:
            try:
                balance = self._rest_client.get_balance()
                trade_balance = self._rest_client.get_trade_balance()
                self._portfolio.sync_from_api(balance, trade_balance)
            except Exception as e:
                logger.warning(f"Balance sync failed: {e}")

    async def _health_check_task(self) -> None:
        """Periodic health checks."""
        logger.info("Health check task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.health_check_interval)

                if self._shutdown_event.is_set():
                    break

                try:
                    health = await self._health_checker.run_all_checks()

                    if not health.overall_healthy:
                        logger.warning(
                            f"System unhealthy: {health.overall_level.value}"
                        )

                        # Pause if critical
                        if health.overall_level == HealthLevel.CRITICAL:
                            await self.pause("Health check critical")

                except Exception as e:
                    logger.error(f"Health check error: {e}")

        except asyncio.CancelledError:
            logger.info("Health check task cancelled")

    async def _state_persist_task(self) -> None:
        """Periodic state persistence."""
        logger.info("State persist task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.state_persist_interval)

                if self._shutdown_event.is_set():
                    break

                try:
                    await self._save_state()
                except Exception as e:
                    logger.error(f"State persist error: {e}")

        except asyncio.CancelledError:
            logger.info("State persist task cancelled")

    async def _rebalance_check_task(self) -> None:
        """Periodic rebalance checks."""
        logger.info("Rebalance check task started")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self._orch_config.rebalance_check_interval)

                if self._shutdown_event.is_set():
                    break

                if self._state != OrchestratorState.RUNNING:
                    continue

                try:
                    await self._check_rebalance()
                except Exception as e:
                    logger.error(f"Rebalance check error: {e}")

        except asyncio.CancelledError:
            logger.info("Rebalance check task cancelled")

    async def _check_rebalance(self) -> None:
        """Check if grid rebalancing is needed."""
        if not all([
            self._rebalancer,
            self._grid_executor,
            self._order_manager,
            self._current_price,
        ]):
            return

        current_grid = self._grid_executor.current_grid
        if current_grid is None:
            return

        # Compute drift metrics
        metrics = self._rebalancer.compute_drift_metrics(
            current_grid,
            self._current_price,
            self._order_manager.position,
            self._order_manager,
        )

        # Check if rebalance needed
        decision = self._rebalancer.should_rebalance(
            metrics,
            self._current_regime,
            self._previous_regime,
        )

        if decision.should_rebalance:
            logger.info(
                f"Rebalancing triggered: {decision.reason.name if decision.reason else 'unknown'} "
                f"(urgency: {decision.urgency:.2f})"
            )

            # Get new center price
            new_center = self._rebalancer.get_suggested_new_center(
                current_grid,
                self._current_price,
                self._order_manager.position,
            )

            await self._update_grid(self._current_regime, self._current_confidence)
            self._rebalancer.mark_rebalanced()

    async def _data_sync_task(self) -> None:
        """
        Periodic data synchronization task.

        Downloads new trades and updates OHLCV candles at configured intervals.
        This keeps historical data fresh during live trading for:
        - Feature computation with up-to-date lookback windows
        - Regime model retraining (if needed)
        - Analytics and reporting
        """
        if self._data_sync_service is None:
            return

        logger.info(
            f"Data sync task started (interval: {self._orch_config.data_sync_interval}s)"
        )

        try:
            await self._data_sync_service.start()

            while not self._shutdown_event.is_set():
                # Wait for sync interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._orch_config.data_sync_interval,
                    )
                    # If we get here, shutdown was signaled
                    break
                except asyncio.TimeoutError:
                    pass  # Normal timeout, proceed with sync

                if self._shutdown_event.is_set():
                    break

                # Skip sync if not running
                if self._state != OrchestratorState.RUNNING:
                    continue

                # Perform sync for each configured pair
                for pair in self._data_sync_service.config.pairs:
                    if self._shutdown_event.is_set():
                        break

                    try:
                        result = await self._data_sync_service.sync_once(pair)
                        if result.get("errors"):
                            for error in result["errors"]:
                                logger.warning(f"Data sync issue: {error}")
                    except Exception as e:
                        logger.error(f"Data sync error for {pair}: {e}")

        except asyncio.CancelledError:
            logger.info("Data sync task cancelled")
        finally:
            if self._data_sync_service:
                await self._data_sync_service.stop()

    # === Event Handlers ===

    async def _on_ws_message(self, message: Dict[str, Any]) -> None:
        """WebSocket message callback."""
        # Messages are handled in _websocket_task via async iterator
        pass

    async def _on_ws_error(self, error: Exception) -> None:
        """WebSocket error callback."""
        logger.error(f"WebSocket error: {error}")

    async def _on_ws_state_change(self, state: Any) -> None:
        """WebSocket state change callback."""
        logger.info(f"WebSocket state changed: {state}")
        if self._health_checker:
            self._health_checker.record_ws_reconnect()

    def _on_order_fill_internal(
        self,
        order: GridOrder,
        volume: Decimal,
        price: Decimal,
    ) -> None:
        """Internal order fill callback from OrderManager."""
        logger.info(
            f"Order filled: {order.grid_id}, "
            f"volume={volume}, price={price}"
        )

        # Save fill to database
        if self._state_manager:
            self._state_manager.save_fill(
                order_id=order.grid_id,
                fill_volume=volume,
                fill_price=price,
                fee=order.fees,
                timestamp=datetime.utcnow(),
            )

    def _on_grid_fill(
        self,
        order: GridOrder,
        volume: Decimal,
        price: Decimal,
    ) -> None:
        """Grid fill callback from GridExecutor."""
        # GridExecutor handles placing counter orders
        pass

    def _on_stop_loss(self, trigger_price: Decimal) -> None:
        """Stop-loss trigger callback."""
        logger.warning(f"Stop-loss triggered at {trigger_price}")

        if self._alert_manager:
            self._alert_manager.create_alert(
                AlertType.STOP_LOSS_TRIGGERED,
                AlertSeverity.EMERGENCY,
                f"Stop-loss triggered at ${trigger_price}",
                {"trigger_price": str(trigger_price)},
                force=True,
            )

        # Halt trading
        if self._risk_manager:
            self._risk_manager._halt_trading(
                f"Stop-loss triggered at {trigger_price}"
            )

    def _on_halt(self, reason: str) -> None:
        """Halt callback from RiskManager."""
        logger.critical(f"Trading HALTED: {reason}")
        self._state = OrchestratorState.PAUSED

    def _on_pause(self, reason: str) -> None:
        """Pause callback from RiskManager."""
        logger.warning(f"Trading paused: {reason}")
        if self._state == OrchestratorState.RUNNING:
            self._state = OrchestratorState.PAUSED

    async def _handle_regime_change(
        self,
        old_regime: MarketRegime,
        new_regime: MarketRegime,
        confidence: float,
    ) -> None:
        """Handle regime change - update grid."""
        logger.info(
            f"Regime changed: {old_regime.name} -> {new_regime.name} "
            f"(confidence: {confidence:.2f})"
        )

        if self._alert_manager:
            self._alert_manager.create_alert(
                AlertType.REGIME_CHANGE,
                AlertSeverity.INFO,
                f"Regime: {old_regime.name} -> {new_regime.name}",
                {
                    "old_regime": old_regime.name,
                    "new_regime": new_regime.name,
                    "confidence": confidence,
                },
            )

        # Update grid for new regime
        await self._update_grid(new_regime, confidence)

    async def _handle_risk_action(self, result) -> None:
        """Handle risk check result."""
        if result.action == RiskAction.HALT_TRADING:
            logger.critical(f"HALT triggered: {result.reason}")
            self._state = OrchestratorState.PAUSED
        elif result.action == RiskAction.PAUSE_TRADING:
            logger.warning(f"PAUSE triggered: {result.reason}")
            await self.pause(result.reason)
        elif result.action == RiskAction.REDUCE_EXPOSURE:
            logger.warning(f"Reduce exposure: {result.reason}")
            # GridExecutor handles this

    # === Grid Management ===

    async def _fetch_initial_price(self) -> None:
        """Fetch initial price from exchange."""
        if not self._rest_client:
            return

        try:
            ticker = self._rest_client.get_ticker(self._config.trading.pair)
            if ticker:
                # Extract last price from ticker
                pair_data = list(ticker.values())[0] if ticker else {}
                last_price = pair_data.get("c", [None])[0]
                if last_price:
                    self._current_price = Decimal(str(last_price))
                    logger.info(f"Initial price: ${self._current_price}")
        except Exception as e:
            logger.warning(f"Failed to fetch initial price: {e}")

    async def _deploy_initial_grid(self) -> None:
        """Deploy grid for first time after initialization."""
        if not all([
            self._grid_strategy,
            self._grid_executor,
            self._current_price,
        ]):
            logger.warning("Cannot deploy grid - missing components or price")
            return

        # Check if should pause for regime
        should_pause, reason = self._grid_strategy.should_pause_trading(
            self._current_regime,
            self._current_confidence,
        )

        if should_pause:
            logger.info(f"Not deploying grid: {reason}")
            await self.pause(reason)
            return

        # Compute grid
        try:
            grid_state, adaptation = self._grid_strategy.compute_adapted_grid(
                current_price=self._current_price,
                atr=self._current_price * Decimal("0.02"),  # Estimate 2% ATR
                regime=self._current_regime,
                confidence=self._current_confidence,
                capital=self._portfolio.total_equity if self._portfolio else Decimal("400"),
            )

            # Deploy
            result = self._grid_executor.deploy_grid(
                grid_state,
                adaptation,
                self._current_regime,
                self._current_confidence,
            )

            logger.info(
                f"Initial grid deployed: {result.orders_submitted} orders submitted"
            )

        except Exception as e:
            logger.error(f"Failed to deploy initial grid: {e}")

    async def _update_grid(
        self,
        regime: MarketRegime,
        confidence: float,
        force: bool = False,
    ) -> None:
        """Update grid based on regime."""
        if not all([
            self._grid_strategy,
            self._grid_executor,
            self._current_price,
        ]):
            return

        try:
            # Compute new grid
            grid_state, adaptation = self._grid_strategy.compute_adapted_grid(
                current_price=self._current_price,
                atr=self._current_price * Decimal("0.02"),  # Estimate
                regime=regime,
                confidence=confidence,
                capital=self._portfolio.total_equity if self._portfolio else Decimal("400"),
            )

            # Update
            result = self._grid_executor.update_grid(
                grid_state,
                adaptation,
                regime,
                confidence,
            )

            logger.info(
                f"Grid updated: submitted={result.orders_submitted}, "
                f"canceled={result.orders_canceled}"
            )

        except Exception as e:
            logger.error(f"Failed to update grid: {e}")

    # === State Management ===

    async def _save_state(self) -> None:
        """Save current state."""
        if not self._state_manager:
            return

        state = BotState(
            timestamp=datetime.utcnow(),
            grid_snapshot=(
                self._grid_executor.get_snapshot().__dict__
                if self._grid_executor and self._grid_executor.current_grid
                else None
            ),
            risk_state=(
                self._risk_manager.to_dict()
                if self._risk_manager
                else None
            ),
            position=(
                self._order_manager.position.__dict__
                if self._order_manager
                else None
            ),
            last_regime=self._current_regime.name,
            last_confidence=self._current_confidence,
            is_trading=self._state == OrchestratorState.RUNNING,
            is_paused=self._state == OrchestratorState.PAUSED,
            pause_reason=(
                self._risk_manager._pause_reason
                if self._risk_manager
                else ""
            ),
        )

        self._state_manager.save_state(state)

        # Save metrics snapshot
        if self._session_id and self._portfolio and self._order_manager:
            self._state_manager.save_metrics_snapshot(
                session_id=self._session_id,
                equity=self._portfolio.total_equity,
                drawdown_pct=self._portfolio.current_drawdown_percent,
                regime=self._current_regime.name,
                confidence=self._current_confidence,
                active_orders=len(self._order_manager.get_active_orders()),
                position_qty=self._order_manager.position.quantity,
            )

    def get_state(self) -> Dict[str, Any]:
        """Get current orchestrator state for monitoring."""
        return {
            "state": self._state.name,
            "current_price": str(self._current_price) if self._current_price else None,
            "current_regime": self._current_regime.name,
            "current_confidence": self._current_confidence,
            "session_id": self._session_id,
            "is_halted": self._risk_manager.is_halted if self._risk_manager else False,
            "is_paused": self._risk_manager.is_paused if self._risk_manager else False,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "state": self._state.name,
            "session_id": self._session_id,
            "current_price": str(self._current_price) if self._current_price else None,
            "current_regime": self._current_regime.name,
        }

        if self._portfolio:
            stats["portfolio"] = self._portfolio.get_summary()

        if self._order_manager:
            stats["orders"] = self._order_manager.get_stats()

        if self._risk_manager:
            stats["risk"] = self._risk_manager.get_stats()

        if self._health_checker:
            stats["health"] = self._health_checker.get_metrics()

        if self._rebalancer:
            stats["rebalance_count"] = self._rebalancer.rebalance_count

        return stats
