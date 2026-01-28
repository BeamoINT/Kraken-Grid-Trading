"""
Central Risk Manager.

Enforces all risk rules:
- Max 70% capital in positions
- Max 2% risk per order
- Stop-loss 15% below lowest grid
- Halt at 20% drawdown (high-water mark tracked)
- Pause if confidence < 60%

Coordinates Portfolio and AlertManager for comprehensive risk control.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from config.settings import RiskConfig

from .portfolio import Portfolio, EquitySnapshot
from .alerts import (
    AlertManager,
    AlertType,
    AlertSeverity,
    LoggingAlertHandler,
    CallbackAlertHandler,
    create_drawdown_alert,
    create_exposure_alert,
    create_confidence_alert,
)

if TYPE_CHECKING:
    from src.api import OrderManager, GridPosition, KrakenPrivateClient

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Actions the risk manager can take."""

    NONE = auto()  # No action needed
    REDUCE_EXPOSURE = auto()  # Reduce position/order size
    PAUSE_BUYS = auto()  # Stop placing buy orders
    PAUSE_TRADING = auto()  # Pause all trading
    HALT_TRADING = auto()  # Emergency halt - close all
    CANCEL_ORDER = auto()  # Cancel specific order


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    action: RiskAction
    reason: str
    details: Dict[str, Any]

    @property
    def should_halt(self) -> bool:
        return self.action == RiskAction.HALT_TRADING

    @property
    def should_pause(self) -> bool:
        return self.action in (RiskAction.PAUSE_TRADING, RiskAction.HALT_TRADING)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"RiskCheck({status}): {self.reason}"


@dataclass
class RiskState:
    """Complete risk state snapshot."""

    timestamp: datetime
    # Drawdown
    drawdown_percent: float
    high_water_mark: Decimal
    max_drawdown_percent: float
    # Exposure
    total_exposure: Decimal
    exposure_percent: float
    position_value: Decimal
    open_order_exposure: Decimal
    # Flags
    is_halted: bool
    is_paused: bool
    halt_reason: str
    pause_reason: str
    # Alerts
    active_alerts: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "drawdown_percent": self.drawdown_percent,
            "high_water_mark": str(self.high_water_mark),
            "max_drawdown_percent": self.max_drawdown_percent,
            "total_exposure": str(self.total_exposure),
            "exposure_percent": self.exposure_percent,
            "position_value": str(self.position_value),
            "open_order_exposure": str(self.open_order_exposure),
            "is_halted": self.is_halted,
            "is_paused": self.is_paused,
            "halt_reason": self.halt_reason,
            "pause_reason": self.pause_reason,
            "active_alerts": self.active_alerts,
        }


class RiskManager:
    """
    Central risk management and enforcement.

    Key Responsibilities:
    - Enforce max drawdown rule (20% from HWM -> halt)
    - Enforce max exposure rule (70% of capital)
    - Enforce per-order risk limit (2%)
    - Coordinate with portfolio tracking
    - Generate alerts for risk events
    - Provide risk state for decision making

    Usage:
        portfolio = Portfolio(initial_capital=Decimal("400"))
        risk_manager = RiskManager(
            config=risk_config,
            portfolio=portfolio,
        )

        # Check before placing order
        result = risk_manager.check_order_allowed(order_value, capital)
        if not result.passed:
            handle_risk_violation(result)

        # Periodic risk check
        result = risk_manager.run_risk_check(current_price)
        if result.should_halt:
            halt_trading(result.reason)

        # Update from market data
        risk_manager.update_from_position(position, order_manager, price)
    """

    # Threshold percentages for graduated alerts
    DRAWDOWN_WARNING_PCT = 15.0
    DRAWDOWN_CRITICAL_PCT = 18.0
    EXPOSURE_WARNING_PCT = 60.0
    CONFIDENCE_WARNING = 0.7

    def __init__(
        self,
        config: RiskConfig,
        portfolio: Portfolio,
        alert_manager: Optional[AlertManager] = None,
        on_halt: Optional[Callable[[str], None]] = None,
        on_pause: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration
            portfolio: Portfolio tracker
            alert_manager: Alert manager (created if not provided)
            on_halt: Callback when trading halted
            on_pause: Callback when trading paused
        """
        self._config = config
        self._portfolio = portfolio
        self._alert_manager = alert_manager or AlertManager()
        self._on_halt = on_halt
        self._on_pause = on_pause

        # State
        self._is_halted = False
        self._is_paused = False
        self._halt_reason = ""
        self._pause_reason = ""
        self._current_price: Optional[Decimal] = None
        self._last_check_time: Optional[datetime] = None

        # Thread safety
        self._lock = threading.Lock()

        # Setup default handlers
        self._setup_default_handlers()

        logger.info(
            f"RiskManager initialized: max_drawdown={config.max_drawdown_percent}%, "
            f"max_exposure={config.max_position_percent}%, "
            f"order_risk={config.order_risk_percent}%"
        )

    def _setup_default_handlers(self) -> None:
        """Setup default alert handlers."""
        self._alert_manager.add_handler(LoggingAlertHandler())

        # Add halt callback handler
        if self._on_halt:

            def halt_callback(alert):
                if alert.alert_type == AlertType.DRAWDOWN_HALT:
                    self._on_halt(alert.message)

            self._alert_manager.add_handler(
                CallbackAlertHandler(halt_callback, AlertSeverity.EMERGENCY)
            )

    # === Core Risk Checks ===

    def run_risk_check(
        self,
        current_price: Optional[Decimal] = None,
        position: Optional["GridPosition"] = None,
        order_manager: Optional["OrderManager"] = None,
    ) -> RiskCheckResult:
        """
        Run comprehensive risk check.

        Should be called periodically (e.g., every price update).

        Args:
            current_price: Current market price
            position: Current position
            order_manager: Order manager for exposure calculation

        Returns:
            RiskCheckResult with action to take
        """
        with self._lock:
            self._last_check_time = datetime.utcnow()

            if current_price:
                self._current_price = current_price

            # Update portfolio if we have position data
            if position and order_manager and self._current_price:
                buy_exp, sell_exp = order_manager.get_open_order_exposure()
                self._portfolio.update_from_position(
                    position_quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=self._current_price,
                    realized_pnl=position.realized_pnl,
                    open_order_exposure=buy_exp + sell_exp,
                )

            # Already halted?
            if self._is_halted:
                return RiskCheckResult(
                    passed=False,
                    action=RiskAction.HALT_TRADING,
                    reason=f"Already halted: {self._halt_reason}",
                    details={"halt_reason": self._halt_reason},
                )

            # Check drawdown (CRITICAL - highest priority)
            drawdown_result = self._check_drawdown()
            if not drawdown_result.passed:
                return drawdown_result

            # Check exposure
            exposure_result = self._check_exposure()
            if not exposure_result.passed:
                return exposure_result

            return RiskCheckResult(
                passed=True,
                action=RiskAction.NONE,
                reason="All risk checks passed",
                details={
                    "drawdown_percent": self._portfolio.current_drawdown_percent,
                    "exposure_percent": self._portfolio.get_exposure_percent(),
                },
            )

    def _check_drawdown(self) -> RiskCheckResult:
        """Check drawdown against limits."""
        drawdown_pct = self._portfolio.current_drawdown_percent
        max_allowed = self._config.max_drawdown_percent

        # Create graduated alerts
        create_drawdown_alert(
            self._alert_manager,
            drawdown_pct,
            warning_threshold=self.DRAWDOWN_WARNING_PCT,
            critical_threshold=self.DRAWDOWN_CRITICAL_PCT,
            halt_threshold=max_allowed,
        )

        # Check for halt
        if drawdown_pct >= max_allowed:
            self._halt_trading(f"Max drawdown exceeded: {drawdown_pct:.2f}%")
            return RiskCheckResult(
                passed=False,
                action=RiskAction.HALT_TRADING,
                reason=f"Drawdown {drawdown_pct:.2f}% >= {max_allowed}%",
                details={
                    "drawdown_percent": drawdown_pct,
                    "max_allowed": max_allowed,
                    "high_water_mark": float(self._portfolio.high_water_mark),
                    "current_equity": float(self._portfolio.total_equity),
                },
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="Drawdown within limits",
            details={"drawdown_percent": drawdown_pct},
        )

    def _check_exposure(self) -> RiskCheckResult:
        """Check exposure against limits."""
        exposure_pct = self._portfolio.get_exposure_percent()
        max_allowed = self._config.max_position_percent

        # Create alerts
        create_exposure_alert(
            self._alert_manager,
            exposure_pct,
            warning_threshold=self.EXPOSURE_WARNING_PCT,
            limit_threshold=max_allowed,
        )

        if exposure_pct >= max_allowed:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.REDUCE_EXPOSURE,
                reason=f"Exposure {exposure_pct:.2f}% >= {max_allowed}%",
                details={
                    "exposure_percent": exposure_pct,
                    "max_allowed": max_allowed,
                },
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="Exposure within limits",
            details={"exposure_percent": exposure_pct},
        )

    # === Order Validation ===

    def check_order_allowed(
        self,
        order_value: Decimal,
        capital: Optional[Decimal] = None,
    ) -> RiskCheckResult:
        """
        Check if a new order is allowed by risk rules.

        Args:
            order_value: Value of proposed order
            capital: Total capital (uses portfolio equity if not provided)

        Returns:
            RiskCheckResult indicating if order is allowed
        """
        if capital is None:
            capital = self._portfolio.total_equity

        if self._is_halted:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.HALT_TRADING,
                reason=f"Trading halted: {self._halt_reason}",
                details={},
            )

        if self._is_paused:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.PAUSE_TRADING,
                reason=f"Trading paused: {self._pause_reason}",
                details={},
            )

        # Check order risk limit (2%)
        max_order_risk = (
            capital * Decimal(str(self._config.order_risk_percent)) / Decimal("100")
        )
        if order_value > max_order_risk:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.CANCEL_ORDER,
                reason=f"Order value ${order_value:.2f} exceeds {self._config.order_risk_percent}% limit",
                details={
                    "order_value": float(order_value),
                    "max_allowed": float(max_order_risk),
                    "risk_percent": self._config.order_risk_percent,
                },
            )

        # Check if adding this order would exceed exposure limit
        current_exposure = self._portfolio.get_total_exposure()
        new_total = current_exposure + order_value
        max_exposure = (
            self._portfolio.total_equity
            * Decimal(str(self._config.max_position_percent))
            / Decimal("100")
        )

        if new_total > max_exposure:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.CANCEL_ORDER,
                reason="Order would exceed exposure limit",
                details={
                    "current_exposure": float(current_exposure),
                    "order_value": float(order_value),
                    "would_be": float(new_total),
                    "max_allowed": float(max_exposure),
                },
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="Order allowed",
            details={},
        )

    def check_confidence(self, confidence: float) -> RiskCheckResult:
        """
        Check if model confidence allows trading.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            RiskCheckResult indicating if trading allowed
        """
        min_conf = self._config.min_confidence

        # Create alerts
        create_confidence_alert(self._alert_manager, confidence, min_conf)

        if confidence < min_conf * 0.5:  # Very low
            return RiskCheckResult(
                passed=False,
                action=RiskAction.PAUSE_TRADING,
                reason=f"Confidence {confidence:.2%} < {min_conf * 0.5:.2%}",
                details={"confidence": confidence},
            )
        elif confidence < min_conf:
            return RiskCheckResult(
                passed=False,
                action=RiskAction.PAUSE_BUYS,
                reason=f"Confidence {confidence:.2%} < {min_conf:.2%}",
                details={"confidence": confidence},
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="Confidence acceptable",
            details={"confidence": confidence},
        )

    def check_stop_loss(
        self,
        current_price: Decimal,
        stop_loss_price: Decimal,
    ) -> RiskCheckResult:
        """
        Check if stop-loss has been triggered.

        Args:
            current_price: Current market price
            stop_loss_price: Stop-loss trigger price

        Returns:
            RiskCheckResult indicating if stop-loss triggered
        """
        if current_price <= stop_loss_price:
            self._alert_manager.create_alert(
                AlertType.STOP_LOSS_TRIGGERED,
                AlertSeverity.EMERGENCY,
                f"Stop-loss triggered: price ${current_price} <= ${stop_loss_price}",
                {
                    "current_price": float(current_price),
                    "stop_loss_price": float(stop_loss_price),
                },
                force=True,
            )
            return RiskCheckResult(
                passed=False,
                action=RiskAction.HALT_TRADING,
                reason=f"Stop-loss triggered at ${current_price}",
                details={
                    "current_price": float(current_price),
                    "stop_loss_price": float(stop_loss_price),
                },
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            reason="Price above stop-loss",
            details={
                "current_price": float(current_price),
                "stop_loss_price": float(stop_loss_price),
                "buffer": float(current_price - stop_loss_price),
            },
        )

    def get_max_order_size(self, price: Decimal) -> Decimal:
        """
        Calculate maximum allowed order size.

        Args:
            price: Order price

        Returns:
            Maximum order volume
        """
        capital = self._portfolio.total_equity
        max_order_value = (
            capital * Decimal(str(self._config.order_risk_percent)) / Decimal("100")
        )
        return max_order_value / price if price > 0 else Decimal("0")

    # === Halt/Pause Management ===

    def _halt_trading(self, reason: str) -> None:
        """Halt all trading (emergency)."""
        if not self._is_halted:
            self._is_halted = True
            self._halt_reason = reason
            logger.critical(f"TRADING HALTED: {reason}")

            self._alert_manager.create_alert(
                AlertType.TRADING_HALTED,
                AlertSeverity.EMERGENCY,
                f"Trading halted: {reason}",
                {"reason": reason},
                force=True,
            )

            if self._on_halt:
                try:
                    self._on_halt(reason)
                except Exception as e:
                    logger.error(f"Halt callback failed: {e}")

    def pause_trading(self, reason: str) -> None:
        """Pause trading (recoverable)."""
        with self._lock:
            if not self._is_paused and not self._is_halted:
                self._is_paused = True
                self._pause_reason = reason

                self._alert_manager.create_alert(
                    AlertType.TRADING_PAUSED,
                    AlertSeverity.WARNING,
                    f"Trading paused: {reason}",
                    {"reason": reason},
                )

                logger.warning(f"Trading paused: {reason}")

                if self._on_pause:
                    try:
                        self._on_pause(reason)
                    except Exception as e:
                        logger.error(f"Pause callback failed: {e}")

    def resume_trading(self) -> bool:
        """
        Resume trading after pause.

        Returns:
            True if resumed, False if still halted
        """
        with self._lock:
            if self._is_halted:
                logger.warning("Cannot resume: trading is halted")
                return False

            if self._is_paused:
                # Check if safe to resume
                drawdown_pct = self._portfolio.current_drawdown_percent
                if drawdown_pct >= self._config.max_drawdown_percent * 0.9:
                    logger.warning(
                        f"Cannot resume: drawdown still at {drawdown_pct:.2f}%"
                    )
                    return False

                self._is_paused = False
                self._pause_reason = ""

                self._alert_manager.create_alert(
                    AlertType.TRADING_RESUMED,
                    AlertSeverity.INFO,
                    "Trading resumed",
                    {"drawdown_percent": drawdown_pct},
                )

                logger.info("Trading resumed")
                return True

            return True

    def clear_halt(self, admin_override: bool = False) -> bool:
        """
        Clear halt state (requires manual intervention).

        Args:
            admin_override: Explicit admin authorization

        Returns:
            True if cleared
        """
        if not admin_override:
            logger.error("Cannot clear halt without admin_override=True")
            return False

        with self._lock:
            self._is_halted = False
            self._halt_reason = ""
            self._is_paused = False
            self._pause_reason = ""

            logger.warning("HALT CLEARED by admin override")

            self._alert_manager.create_alert(
                AlertType.TRADING_RESUMED,
                AlertSeverity.INFO,
                "Trading halt cleared by admin",
                {"admin_override": True},
            )

            return True

    # === State Access ===

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def is_paused(self) -> bool:
        return self._is_paused or self._is_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def pause_reason(self) -> str:
        return self._halt_reason if self._is_halted else self._pause_reason

    @property
    def portfolio(self) -> Portfolio:
        """Get portfolio reference."""
        return self._portfolio

    @property
    def alert_manager(self) -> AlertManager:
        """Get alert manager reference."""
        return self._alert_manager

    def get_risk_state(self) -> RiskState:
        """Get complete risk state snapshot."""
        drawdown_state = self._portfolio.get_drawdown_state()

        return RiskState(
            timestamp=datetime.utcnow(),
            drawdown_percent=drawdown_state.current_drawdown_percent,
            high_water_mark=drawdown_state.high_water_mark,
            max_drawdown_percent=drawdown_state.max_drawdown_percent,
            total_exposure=self._portfolio.get_total_exposure(),
            exposure_percent=self._portfolio.get_exposure_percent(),
            position_value=self._portfolio.position_value,
            open_order_exposure=self._portfolio.open_order_exposure,
            is_halted=self._is_halted,
            is_paused=self._is_paused,
            halt_reason=self._halt_reason,
            pause_reason=self._pause_reason,
            active_alerts=len(self._alert_manager.get_unacknowledged()),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        state = self.get_risk_state()
        return {
            "is_halted": state.is_halted,
            "is_paused": state.is_paused,
            "halt_reason": state.halt_reason,
            "pause_reason": state.pause_reason,
            "drawdown_percent": state.drawdown_percent,
            "max_drawdown_percent": state.max_drawdown_percent,
            "high_water_mark": float(state.high_water_mark),
            "exposure_percent": state.exposure_percent,
            "total_exposure": float(state.total_exposure),
            "active_alerts": state.active_alerts,
            "last_check": (
                self._last_check_time.isoformat() if self._last_check_time else None
            ),
        }

    # === API Sync ===

    def sync_from_api(
        self,
        client: "KrakenPrivateClient",
    ) -> EquitySnapshot:
        """
        Sync portfolio from Kraken API.

        Args:
            client: Kraken private client

        Returns:
            Updated equity snapshot
        """
        try:
            balance = client.get_balance()
            trade_balance = client.get_trade_balance()

            snapshot = self._portfolio.sync_from_api(balance, trade_balance)

            # Run risk check after sync
            self.run_risk_check()

            return snapshot

        except Exception as e:
            self._alert_manager.create_alert(
                AlertType.SYNC_FAILED,
                AlertSeverity.WARNING,
                f"API sync failed: {e}",
                {"error": str(e)},
            )
            raise

    # === Persistence ===

    def to_dict(self) -> Dict[str, Any]:
        """Serialize risk manager state."""
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "is_paused": self._is_paused,
            "pause_reason": self._pause_reason,
            "portfolio": self._portfolio.to_dict(),
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self._is_halted = data.get("is_halted", False)
        self._halt_reason = data.get("halt_reason", "")
        self._is_paused = data.get("is_paused", False)
        self._pause_reason = data.get("pause_reason", "")

        if "portfolio" in data:
            self._portfolio = Portfolio.from_dict(data["portfolio"])

        logger.info(
            f"RiskManager restored: halted={self._is_halted}, paused={self._is_paused}"
        )
