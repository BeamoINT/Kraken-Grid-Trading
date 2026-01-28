"""
Grid Executor for managing grid order lifecycle.

Handles:
- Grid deployment and updates
- Order submission via OrderManager
- Fill handling and position tracking
- Stop-loss monitoring
- Grid state persistence and recovery
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.api import (
    OrderManager,
    GridOrder,
    GridOrderType,
    OrderState,
    GridPosition,
)
from src.regime import MarketRegime
from config.settings import RiskConfig

from .grid_calculator import GridLevel, GridState
from .grid_strategy import GridStrategy, RegimeAdaptation

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of an execution action."""

    success: bool
    orders_submitted: int = 0
    orders_canceled: int = 0
    orders_failed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        """Total number of order actions taken."""
        return self.orders_submitted + self.orders_canceled

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"ExecutionResult({status}: "
            f"submitted={self.orders_submitted}, "
            f"canceled={self.orders_canceled}, "
            f"failed={self.orders_failed})"
        )


@dataclass
class GridSnapshot:
    """Snapshot of grid state for persistence/recovery."""

    grid_state: GridState
    active_order_ids: List[str]  # List of grid_ids
    position_quantity: Decimal
    position_avg_price: Decimal
    realized_pnl: Decimal
    regime: MarketRegime
    confidence: float
    stop_loss_price: Decimal
    is_paused: bool
    pause_reason: str
    timestamp: float = field(default_factory=time.time)


class GridExecutor:
    """
    Executes grid orders and manages order lifecycle.

    Coordinates with OrderManager to:
    - Submit orders to create/update grid
    - Cancel orders on regime change
    - Handle fill callbacks
    - Maintain grid consistency
    - Track stop-loss conditions

    Example:
        executor = GridExecutor(
            order_manager=order_manager,
            strategy=strategy,
            risk_config=risk_config,
            on_fill=my_fill_handler,
            on_stop_loss=my_stop_loss_handler,
        )

        grid_state, adaptation = strategy.compute_adapted_grid(...)
        result = executor.deploy_grid(grid_state, adaptation)

        if executor.check_stop_loss(current_price):
            handle_stop_loss()
    """

    def __init__(
        self,
        order_manager: OrderManager,
        strategy: GridStrategy,
        risk_config: RiskConfig,
        capital: Decimal = Decimal("400"),
        on_fill: Optional[Callable[[GridOrder, Decimal, Decimal], None]] = None,
        on_stop_loss: Optional[Callable[[Decimal], None]] = None,
    ):
        """
        Initialize grid executor.

        Args:
            order_manager: OrderManager for order operations
            strategy: GridStrategy for computing adaptations
            risk_config: Risk management configuration
            capital: Total trading capital
            on_fill: Callback when order fills (order, volume, price)
            on_stop_loss: Callback when stop-loss triggers (trigger_price)
        """
        self._order_manager = order_manager
        self._strategy = strategy
        self._risk_config = risk_config
        self._capital = capital
        self._external_on_fill = on_fill
        self._on_stop_loss = on_stop_loss

        # State
        self._current_grid: Optional[GridState] = None
        self._current_adaptation: Optional[RegimeAdaptation] = None
        self._stop_loss_price: Optional[Decimal] = None
        self._is_paused = False
        self._pause_reason = ""
        self._current_regime: Optional[MarketRegime] = None
        self._current_confidence: float = 0.0

        # Track which levels have orders
        self._level_order_map: Dict[int, str] = {}  # level_index -> grid_id

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._total_fills = 0
        self._total_submitted = 0
        self._total_canceled = 0
        self._stop_loss_triggered = False

    @property
    def current_grid(self) -> Optional[GridState]:
        """Get current grid state."""
        return self._current_grid

    @property
    def is_trading(self) -> bool:
        """Check if actively trading (grid deployed and not paused)."""
        return self._current_grid is not None and not self._is_paused

    @property
    def is_paused(self) -> bool:
        """Check if trading is paused."""
        return self._is_paused

    @property
    def pause_reason(self) -> str:
        """Get reason for pause."""
        return self._pause_reason

    @property
    def stop_loss_price(self) -> Optional[Decimal]:
        """Get current stop-loss price."""
        return self._stop_loss_price

    @property
    def position(self) -> GridPosition:
        """Get current position from OrderManager."""
        return self._order_manager.position

    def deploy_grid(
        self,
        grid_state: GridState,
        adaptation: RegimeAdaptation,
        regime: MarketRegime = MarketRegime.RANGING,
        confidence: float = 1.0,
    ) -> ExecutionResult:
        """
        Deploy a new grid, submitting all orders.

        Cancels any existing orders first, then submits orders
        for all active grid levels.

        Args:
            grid_state: Grid state to deploy
            adaptation: Regime adaptation applied
            regime: Current market regime
            confidence: Model confidence

        Returns:
            ExecutionResult with success/failure details
        """
        with self._lock:
            logger.info(
                f"Deploying grid: {len(grid_state.levels)} levels, "
                f"center={grid_state.center_price}"
            )

            errors = []

            # Validate against risk rules
            validation_errors = self._validate_against_risk_rules(
                grid_state, self._capital
            )
            if validation_errors:
                logger.error(f"Grid validation failed: {validation_errors}")
                return ExecutionResult(
                    success=False,
                    errors=validation_errors,
                )

            # Cancel existing orders if any
            cancel_result = self._cancel_all_internal()
            if cancel_result.orders_failed > 0:
                errors.extend(cancel_result.errors)

            # Calculate stop-loss price
            self._stop_loss_price = self._strategy.compute_stop_loss_price(
                grid_state, adaptation
            )
            logger.info(f"Stop-loss set at {self._stop_loss_price}")

            # Submit orders for active levels
            submitted, failed = self._submit_level_orders(
                grid_state.active_levels, adaptation
            )

            # Update state
            self._current_grid = grid_state
            self._current_adaptation = adaptation
            self._current_regime = regime
            self._current_confidence = confidence
            self._is_paused = False
            self._pause_reason = ""
            self._stop_loss_triggered = False

            self._total_submitted += submitted

            result = ExecutionResult(
                success=failed == 0,
                orders_submitted=submitted,
                orders_canceled=cancel_result.orders_canceled,
                orders_failed=failed,
                errors=errors,
            )

            logger.info(f"Grid deployment: {result}")
            return result

    def update_grid(
        self,
        new_grid_state: GridState,
        new_adaptation: RegimeAdaptation,
        regime: MarketRegime = MarketRegime.RANGING,
        confidence: float = 1.0,
    ) -> ExecutionResult:
        """
        Update existing grid to new state.

        Computes diff between current and new grid, canceling
        changed orders and submitting new ones.

        Args:
            new_grid_state: New grid state
            new_adaptation: New regime adaptation
            regime: Current market regime
            confidence: Model confidence

        Returns:
            ExecutionResult with success/failure details
        """
        with self._lock:
            if self._current_grid is None:
                # No existing grid, do full deployment
                return self.deploy_grid(new_grid_state, new_adaptation, regime, confidence)

            logger.info(
                f"Updating grid: {len(new_grid_state.levels)} levels, "
                f"center={new_grid_state.center_price}"
            )

            errors = []

            # Validate new grid
            validation_errors = self._validate_against_risk_rules(
                new_grid_state, self._capital
            )
            if validation_errors:
                return ExecutionResult(success=False, errors=validation_errors)

            # Compute what needs to change
            orders_to_cancel: List[str] = []
            levels_to_add: List[GridLevel] = []

            # Get current level indices with orders
            current_level_indices = set(self._level_order_map.keys())
            new_level_indices = {level.index for level in new_grid_state.active_levels}

            # Levels to remove (in current but not in new)
            for level_idx in current_level_indices - new_level_indices:
                if level_idx in self._level_order_map:
                    orders_to_cancel.append(self._level_order_map[level_idx])

            # Levels to add (in new but not in current)
            for level in new_grid_state.active_levels:
                if level.index not in current_level_indices:
                    levels_to_add.append(level)

            # Check for price/volume changes on existing levels
            for level in new_grid_state.active_levels:
                if level.index in current_level_indices:
                    current_level = self._current_grid.get_level_by_index(level.index)
                    if current_level and (
                        level.price != current_level.price or
                        level.volume != current_level.volume or
                        level.side != current_level.side
                    ):
                        # Level changed - cancel old order and create new
                        if level.index in self._level_order_map:
                            orders_to_cancel.append(self._level_order_map[level.index])
                        levels_to_add.append(level)

            # Cancel orders
            canceled = 0
            cancel_failed = 0
            for grid_id in orders_to_cancel:
                if self._order_manager.cancel_order(grid_id):
                    canceled += 1
                    # Remove from level map
                    for lvl_idx, gid in list(self._level_order_map.items()):
                        if gid == grid_id:
                            del self._level_order_map[lvl_idx]
                            break
                else:
                    cancel_failed += 1
                    errors.append(f"Failed to cancel order {grid_id}")

            # Submit new orders
            submitted, submit_failed = self._submit_level_orders(
                levels_to_add, new_adaptation
            )

            # Update stop-loss
            self._stop_loss_price = self._strategy.compute_stop_loss_price(
                new_grid_state, new_adaptation
            )

            # Update state
            self._current_grid = new_grid_state
            self._current_adaptation = new_adaptation
            self._current_regime = regime
            self._current_confidence = confidence

            self._total_submitted += submitted
            self._total_canceled += canceled

            result = ExecutionResult(
                success=(cancel_failed + submit_failed) == 0,
                orders_submitted=submitted,
                orders_canceled=canceled,
                orders_failed=cancel_failed + submit_failed,
                errors=errors,
            )

            logger.info(f"Grid update: {result}")
            return result

    def cancel_all_orders(self) -> ExecutionResult:
        """
        Cancel all grid orders (emergency stop or rebalance).

        Returns:
            ExecutionResult with cancellation details
        """
        with self._lock:
            return self._cancel_all_internal()

    def _cancel_all_internal(self) -> ExecutionResult:
        """Internal cancel all without lock."""
        logger.info("Canceling all grid orders")

        success, failed = self._order_manager.cancel_all_grid_orders()
        self._level_order_map.clear()
        self._total_canceled += success

        errors = []
        if failed > 0:
            errors.append(f"Failed to cancel {failed} orders")

        return ExecutionResult(
            success=failed == 0,
            orders_canceled=success,
            orders_failed=failed,
            errors=errors,
        )

    def pause_trading(self, reason: str) -> ExecutionResult:
        """
        Pause trading - cancel all orders but keep state.

        Called on BREAKOUT regime or low confidence.

        Args:
            reason: Reason for pausing

        Returns:
            ExecutionResult with cancellation details
        """
        with self._lock:
            logger.warning(f"Pausing trading: {reason}")

            self._is_paused = True
            self._pause_reason = reason

            return self._cancel_all_internal()

    def resume_trading(self) -> ExecutionResult:
        """
        Resume trading after pause by redeploying grid.

        Returns:
            ExecutionResult with deployment details
        """
        with self._lock:
            if not self._is_paused:
                return ExecutionResult(
                    success=True,
                    errors=["Trading was not paused"],
                )

            if self._current_grid is None or self._current_adaptation is None:
                return ExecutionResult(
                    success=False,
                    errors=["No grid state to resume"],
                )

            logger.info("Resuming trading")
            self._is_paused = False
            self._pause_reason = ""

            # Redeploy the grid
            submitted, failed = self._submit_level_orders(
                self._current_grid.active_levels,
                self._current_adaptation,
            )

            self._total_submitted += submitted

            return ExecutionResult(
                success=failed == 0,
                orders_submitted=submitted,
                orders_failed=failed,
            )

    def check_stop_loss(self, current_price: Decimal) -> bool:
        """
        Check if stop-loss triggered.

        Args:
            current_price: Current market price

        Returns:
            True if stop-loss hit (and triggers callback)
        """
        if self._stop_loss_price is None:
            return False

        if self._stop_loss_triggered:
            return True  # Already triggered

        if current_price <= self._stop_loss_price:
            logger.warning(
                f"STOP-LOSS TRIGGERED: price={current_price} <= "
                f"stop_loss={self._stop_loss_price}"
            )
            self._stop_loss_triggered = True

            # Cancel all orders
            self.cancel_all_orders()

            # Trigger callback
            if self._on_stop_loss:
                try:
                    self._on_stop_loss(current_price)
                except Exception as e:
                    logger.error(f"Error in stop-loss callback: {e}")

            return True

        return False

    def handle_fill(
        self,
        order: GridOrder,
        fill_volume: Decimal,
        fill_price: Decimal,
    ) -> None:
        """
        Handle order fill notification.

        Called when OrderManager detects a fill. Updates internal
        state and triggers external callback.

        Args:
            order: The filled order
            fill_volume: Volume that was filled
            fill_price: Price at which fill occurred
        """
        with self._lock:
            self._total_fills += 1

            logger.info(
                f"Fill: level={order.level}, side={order.side.name}, "
                f"volume={fill_volume}, price={fill_price}"
            )

            # If order is fully filled, remove from level map
            if order.state == OrderState.FILLED:
                for level_idx, grid_id in list(self._level_order_map.items()):
                    if grid_id == order.grid_id:
                        del self._level_order_map[level_idx]
                        break

            # Trigger external callback
            if self._external_on_fill:
                try:
                    self._external_on_fill(order, fill_volume, fill_price)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")

    def get_snapshot(self) -> GridSnapshot:
        """
        Get complete snapshot for persistence.

        Returns:
            GridSnapshot with all state for recovery
        """
        with self._lock:
            position = self._order_manager.position

            return GridSnapshot(
                grid_state=self._current_grid or GridState(
                    levels=[],
                    center_price=Decimal("0"),
                    upper_bound=Decimal("0"),
                    lower_bound=Decimal("0"),
                    total_buy_exposure=Decimal("0"),
                    total_sell_exposure=Decimal("0"),
                ),
                active_order_ids=list(self._level_order_map.values()),
                position_quantity=position.quantity,
                position_avg_price=position.avg_entry_price,
                realized_pnl=position.realized_pnl,
                regime=self._current_regime or MarketRegime.RANGING,
                confidence=self._current_confidence,
                stop_loss_price=self._stop_loss_price or Decimal("0"),
                is_paused=self._is_paused,
                pause_reason=self._pause_reason,
            )

    def restore_from_snapshot(
        self,
        snapshot: GridSnapshot,
    ) -> ExecutionResult:
        """
        Restore grid state from snapshot (crash recovery).

        Syncs with exchange to get current order states,
        then reconciles with snapshot.

        Args:
            snapshot: Previously saved snapshot

        Returns:
            ExecutionResult with restoration details
        """
        with self._lock:
            logger.info("Restoring from snapshot")

            # Restore state
            self._current_grid = snapshot.grid_state
            self._stop_loss_price = snapshot.stop_loss_price
            self._is_paused = snapshot.is_paused
            self._pause_reason = snapshot.pause_reason
            self._current_regime = snapshot.regime
            self._current_confidence = snapshot.confidence

            # Sync with exchange to get current order states
            try:
                changed = self._order_manager.sync_orders()
                logger.info(f"Synced {len(changed)} orders with exchange")
            except Exception as e:
                logger.error(f"Failed to sync orders: {e}")
                return ExecutionResult(
                    success=False,
                    errors=[f"Failed to sync with exchange: {e}"],
                )

            # Rebuild level_order_map from active orders
            self._level_order_map.clear()
            for order in self._order_manager.get_active_orders():
                self._level_order_map[order.level] = order.grid_id

            return ExecutionResult(
                success=True,
                orders_submitted=0,
                orders_canceled=0,
            )

    def _submit_level_orders(
        self,
        levels: List[GridLevel],
        adaptation: RegimeAdaptation,
    ) -> Tuple[int, int]:
        """
        Submit orders for specified levels.

        Respects adaptation.buy_levels_active/sell_levels_active.

        Args:
            levels: Levels to submit orders for
            adaptation: Current regime adaptation

        Returns:
            Tuple of (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0

        for level in levels:
            # Check if this side is active
            if level.side == GridOrderType.BUY and not adaptation.buy_levels_active:
                logger.debug(f"Skipping buy level {level.index} (buys disabled)")
                continue
            if level.side == GridOrderType.SELL and not adaptation.sell_levels_active:
                logger.debug(f"Skipping sell level {level.index} (sells disabled)")
                continue

            if not level.is_active:
                logger.debug(f"Skipping inactive level {level.index}")
                continue

            try:
                # Create order via OrderManager
                order = self._order_manager.create_grid_order(
                    level=level.index,
                    price=level.price,
                    side=level.side,
                    volume=level.volume,
                )

                # Submit order
                if self._order_manager.submit_order(order):
                    success_count += 1
                    self._level_order_map[level.index] = order.grid_id
                    logger.debug(
                        f"Submitted order: level={level.index}, "
                        f"side={level.side.name}, price={level.price}"
                    )
                else:
                    failure_count += 1
                    logger.warning(f"Failed to submit order for level {level.index}")

            except Exception as e:
                failure_count += 1
                logger.error(f"Error submitting order for level {level.index}: {e}")

        return success_count, failure_count

    def _validate_against_risk_rules(
        self,
        grid_state: GridState,
        capital: Decimal,
    ) -> List[str]:
        """
        Validate grid against risk rules.

        Checks:
        - Max exposure (70%)
        - Max open orders

        Args:
            grid_state: Grid state to validate
            capital: Total capital

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check total exposure
        total_exposure = grid_state.total_buy_exposure + grid_state.total_sell_exposure
        max_exposure = capital * Decimal(str(self._risk_config.max_position_percent)) / Decimal("100")

        if total_exposure > max_exposure:
            errors.append(
                f"Total exposure ${total_exposure:.2f} exceeds "
                f"max ${max_exposure:.2f} ({self._risk_config.max_position_percent}%)"
            )

        # Check number of orders
        num_active = len(grid_state.active_levels)
        if num_active > self._risk_config.max_open_orders:
            errors.append(
                f"Number of levels {num_active} exceeds "
                f"max orders {self._risk_config.max_open_orders}"
            )

        return errors

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        with self._lock:
            active_orders = self._order_manager.get_active_orders()
            position = self._order_manager.position

            return {
                "is_trading": self.is_trading,
                "is_paused": self._is_paused,
                "pause_reason": self._pause_reason,
                "current_regime": self._current_regime.name if self._current_regime else None,
                "confidence": self._current_confidence,
                "grid_levels": len(self._current_grid.levels) if self._current_grid else 0,
                "active_orders": len(active_orders),
                "level_order_map_size": len(self._level_order_map),
                "stop_loss_price": float(self._stop_loss_price) if self._stop_loss_price else None,
                "stop_loss_triggered": self._stop_loss_triggered,
                "total_fills": self._total_fills,
                "total_submitted": self._total_submitted,
                "total_canceled": self._total_canceled,
                "position_quantity": float(position.quantity),
                "realized_pnl": float(position.realized_pnl),
                "unrealized_pnl": float(position.unrealized_pnl),
            }
