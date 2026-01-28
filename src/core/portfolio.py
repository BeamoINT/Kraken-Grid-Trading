"""
Portfolio State and Exposure Tracking.

Maintains:
- Current equity and balance sync with Kraken
- High-water mark for drawdown calculation
- Position and order exposure aggregation
- P&L tracking (realized and unrealized)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api import Balance, TradeBalance

logger = logging.getLogger(__name__)


class EquitySource(Enum):
    """Source of equity value."""

    API_BALANCE = "api_balance"  # From Kraken balance API
    CALCULATED = "calculated"  # From position + orders
    ESTIMATED = "estimated"  # From last known + unrealized P&L


@dataclass
class EquitySnapshot:
    """Point-in-time equity snapshot."""

    timestamp: datetime
    total_equity: Decimal
    available_balance: Decimal
    position_value: Decimal
    open_order_exposure: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    source: EquitySource

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_equity": str(self.total_equity),
            "available_balance": str(self.available_balance),
            "position_value": str(self.position_value),
            "open_order_exposure": str(self.open_order_exposure),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "source": self.source.value,
        }


@dataclass
class DrawdownState:
    """Tracks high-water mark and drawdown."""

    high_water_mark: Decimal
    hwm_timestamp: datetime
    current_drawdown: Decimal
    current_drawdown_percent: float
    max_drawdown_seen: Decimal
    max_drawdown_percent: float
    max_drawdown_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "high_water_mark": str(self.high_water_mark),
            "hwm_timestamp": self.hwm_timestamp.isoformat(),
            "current_drawdown": str(self.current_drawdown),
            "current_drawdown_percent": self.current_drawdown_percent,
            "max_drawdown_seen": str(self.max_drawdown_seen),
            "max_drawdown_percent": self.max_drawdown_percent,
            "max_drawdown_timestamp": self.max_drawdown_timestamp.isoformat(),
        }


class Portfolio:
    """
    Portfolio state manager with high-water mark drawdown tracking.

    Key Responsibilities:
    - Sync with Kraken balance API
    - Track high-water mark for drawdown calculation
    - Aggregate exposure from positions and orders
    - Provide equity snapshots

    Usage:
        portfolio = Portfolio(initial_capital=Decimal("400"))

        # Update from position changes
        portfolio.update_from_position(
            position_quantity=Decimal("0.01"),
            avg_entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            realized_pnl=Decimal("10"),
            open_order_exposure=Decimal("100"),
        )

        # Get drawdown state
        drawdown = portfolio.get_drawdown_state()
        if drawdown.current_drawdown_percent > 20.0:
            # Halt trading
            pass
    """

    def __init__(
        self,
        initial_capital: Decimal,
        high_water_mark: Optional[Decimal] = None,
    ):
        """
        Initialize portfolio tracker.

        Args:
            initial_capital: Starting capital
            high_water_mark: Optional pre-existing HWM (for recovery)
        """
        self._initial_capital = initial_capital
        self._high_water_mark = high_water_mark or initial_capital
        self._hwm_timestamp = datetime.utcnow()

        # Current state
        self._total_equity = initial_capital
        self._available_balance = initial_capital
        self._position_value = Decimal("0")
        self._open_order_exposure = Decimal("0")
        self._unrealized_pnl = Decimal("0")
        self._realized_pnl = Decimal("0")

        # Tracking
        self._max_drawdown_seen = Decimal("0")
        self._max_drawdown_percent: float = 0.0
        self._max_drawdown_timestamp = datetime.utcnow()
        self._last_sync_time: Optional[datetime] = None
        self._equity_source = EquitySource.CALCULATED

        # History (for metrics/debugging)
        self._equity_history: List[EquitySnapshot] = []
        self._max_history_size = 1000

        logger.info(
            f"Portfolio initialized: capital=${initial_capital}, "
            f"HWM=${self._high_water_mark}"
        )

    # === Public Properties ===

    @property
    def total_equity(self) -> Decimal:
        """Current total portfolio equity."""
        return self._total_equity

    @property
    def high_water_mark(self) -> Decimal:
        """Peak equity value achieved."""
        return self._high_water_mark

    @property
    def available_balance(self) -> Decimal:
        """Balance available for new orders."""
        return self._available_balance

    @property
    def current_drawdown(self) -> Decimal:
        """Current drawdown from HWM."""
        if self._high_water_mark <= 0:
            return Decimal("0")
        return self._high_water_mark - self._total_equity

    @property
    def current_drawdown_percent(self) -> float:
        """Current drawdown as percentage of HWM."""
        if self._high_water_mark <= 0:
            return 0.0
        return float(self.current_drawdown / self._high_water_mark * 100)

    @property
    def initial_capital(self) -> Decimal:
        """Starting capital."""
        return self._initial_capital

    @property
    def total_return_percent(self) -> float:
        """Total return from initial capital."""
        if self._initial_capital <= 0:
            return 0.0
        return float(
            (self._total_equity - self._initial_capital) / self._initial_capital * 100
        )

    @property
    def position_value(self) -> Decimal:
        """Current position value."""
        return self._position_value

    @property
    def open_order_exposure(self) -> Decimal:
        """Capital locked in open orders."""
        return self._open_order_exposure

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized P&L from open positions."""
        return self._unrealized_pnl

    @property
    def realized_pnl(self) -> Decimal:
        """Realized P&L from closed positions."""
        return self._realized_pnl

    # === Sync Methods ===

    def sync_from_api(
        self,
        balance: Dict[str, "Balance"],
        trade_balance: "TradeBalance",
        quote_asset: str = "ZUSD",
    ) -> EquitySnapshot:
        """
        Sync portfolio state from Kraken API.

        Args:
            balance: Asset balances from get_balance()
            trade_balance: Trade balance from get_trade_balance()
            quote_asset: Quote currency asset code

        Returns:
            EquitySnapshot after sync
        """
        # Extract equity from trade balance
        self._total_equity = trade_balance.equity
        self._available_balance = trade_balance.free_margin
        self._unrealized_pnl = trade_balance.unrealized_pnl

        # Calculate position value from trade balance
        self._position_value = trade_balance.cost_basis + trade_balance.floating_valuation

        self._equity_source = EquitySource.API_BALANCE
        self._last_sync_time = datetime.utcnow()

        # Update high-water mark if new peak
        self._update_high_water_mark()

        snapshot = self._create_snapshot()
        self._record_snapshot(snapshot)

        logger.info(
            f"Portfolio synced: equity=${self._total_equity:.2f}, "
            f"HWM=${self._high_water_mark:.2f}, "
            f"drawdown={self.current_drawdown_percent:.2f}%"
        )

        return snapshot

    def update_from_position(
        self,
        position_quantity: Decimal,
        avg_entry_price: Decimal,
        current_price: Decimal,
        realized_pnl: Decimal,
        open_order_exposure: Decimal,
    ) -> None:
        """
        Update portfolio from position and order state.

        Called when we can't sync with API but have local state.

        Args:
            position_quantity: Current position size (signed)
            avg_entry_price: Average entry price
            current_price: Current market price
            realized_pnl: Total realized P&L
            open_order_exposure: Capital locked in open orders
        """
        # Calculate position value
        self._position_value = abs(position_quantity) * current_price

        # Calculate unrealized P&L
        if position_quantity != 0 and avg_entry_price > 0:
            cost_basis = abs(position_quantity) * avg_entry_price
            if position_quantity > 0:  # Long
                self._unrealized_pnl = self._position_value - cost_basis
            else:  # Short
                self._unrealized_pnl = cost_basis - self._position_value
        else:
            self._unrealized_pnl = Decimal("0")

        self._realized_pnl = realized_pnl
        self._open_order_exposure = open_order_exposure

        # Calculate total equity
        # Equity = Initial Capital + Realized P&L + Unrealized P&L
        self._total_equity = (
            self._initial_capital + self._realized_pnl + self._unrealized_pnl
        )

        # Available = Equity - Position Value - Order Exposure
        self._available_balance = (
            self._total_equity - self._position_value - self._open_order_exposure
        )
        if self._available_balance < 0:
            self._available_balance = Decimal("0")

        self._equity_source = EquitySource.CALCULATED

        # Update high-water mark
        self._update_high_water_mark()

        logger.debug(
            f"Portfolio updated: equity=${self._total_equity:.2f}, "
            f"position=${self._position_value:.2f}, "
            f"unrealized_pnl=${self._unrealized_pnl:.2f}"
        )

    # === Drawdown Tracking ===

    def _update_high_water_mark(self) -> bool:
        """
        Update high-water mark if current equity is a new peak.

        Returns:
            True if HWM was updated
        """
        if self._total_equity > self._high_water_mark:
            old_hwm = self._high_water_mark
            self._high_water_mark = self._total_equity
            self._hwm_timestamp = datetime.utcnow()
            logger.info(
                f"New high-water mark: ${old_hwm:.2f} -> ${self._high_water_mark:.2f}"
            )
            return True

        # Track max drawdown
        current_dd = self.current_drawdown
        current_dd_pct = self.current_drawdown_percent

        if current_dd > self._max_drawdown_seen:
            self._max_drawdown_seen = current_dd
            self._max_drawdown_percent = current_dd_pct
            self._max_drawdown_timestamp = datetime.utcnow()
            logger.warning(
                f"New max drawdown: ${current_dd:.2f} ({current_dd_pct:.2f}%)"
            )

        return False

    def get_drawdown_state(self) -> DrawdownState:
        """Get current drawdown tracking state."""
        return DrawdownState(
            high_water_mark=self._high_water_mark,
            hwm_timestamp=self._hwm_timestamp,
            current_drawdown=self.current_drawdown,
            current_drawdown_percent=self.current_drawdown_percent,
            max_drawdown_seen=self._max_drawdown_seen,
            max_drawdown_percent=self._max_drawdown_percent,
            max_drawdown_timestamp=self._max_drawdown_timestamp,
        )

    def reset_high_water_mark(self, new_hwm: Optional[Decimal] = None) -> None:
        """
        Reset high-water mark (e.g., after capital injection).

        Args:
            new_hwm: New HWM value, defaults to current equity
        """
        self._high_water_mark = new_hwm or self._total_equity
        self._hwm_timestamp = datetime.utcnow()
        self._max_drawdown_seen = Decimal("0")
        self._max_drawdown_percent = 0.0
        logger.info(f"HWM reset to ${self._high_water_mark:.2f}")

    # === Exposure Calculation ===

    def get_total_exposure(self) -> Decimal:
        """Get total capital at risk (position + orders)."""
        return self._position_value + self._open_order_exposure

    def get_exposure_percent(self) -> float:
        """Get exposure as percentage of equity."""
        if self._total_equity <= 0:
            return 100.0
        return float(self.get_total_exposure() / self._total_equity * 100)

    # === Snapshot Management ===

    def _create_snapshot(self) -> EquitySnapshot:
        """Create current equity snapshot."""
        return EquitySnapshot(
            timestamp=datetime.utcnow(),
            total_equity=self._total_equity,
            available_balance=self._available_balance,
            position_value=self._position_value,
            open_order_exposure=self._open_order_exposure,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self._realized_pnl,
            source=self._equity_source,
        )

    def _record_snapshot(self, snapshot: EquitySnapshot) -> None:
        """Record snapshot in history."""
        self._equity_history.append(snapshot)
        if len(self._equity_history) > self._max_history_size:
            self._equity_history = self._equity_history[-self._max_history_size :]

    def get_latest_snapshot(self) -> EquitySnapshot:
        """Get most recent equity snapshot."""
        return self._create_snapshot()

    def get_equity_history(self, limit: int = 100) -> List[EquitySnapshot]:
        """Get recent equity history."""
        return self._equity_history[-limit:]

    # === Summary ===

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for logging/display."""
        return {
            "initial_capital": float(self._initial_capital),
            "total_equity": float(self._total_equity),
            "available_balance": float(self._available_balance),
            "position_value": float(self._position_value),
            "open_order_exposure": float(self._open_order_exposure),
            "unrealized_pnl": float(self._unrealized_pnl),
            "realized_pnl": float(self._realized_pnl),
            "total_return_percent": self.total_return_percent,
            "high_water_mark": float(self._high_water_mark),
            "current_drawdown": float(self.current_drawdown),
            "current_drawdown_percent": self.current_drawdown_percent,
            "max_drawdown_percent": self._max_drawdown_percent,
            "exposure_percent": self.get_exposure_percent(),
        }

    # === Persistence ===

    def to_dict(self) -> Dict[str, Any]:
        """Serialize portfolio state for persistence."""
        return {
            "initial_capital": str(self._initial_capital),
            "total_equity": str(self._total_equity),
            "high_water_mark": str(self._high_water_mark),
            "hwm_timestamp": self._hwm_timestamp.isoformat(),
            "available_balance": str(self._available_balance),
            "position_value": str(self._position_value),
            "open_order_exposure": str(self._open_order_exposure),
            "unrealized_pnl": str(self._unrealized_pnl),
            "realized_pnl": str(self._realized_pnl),
            "max_drawdown_seen": str(self._max_drawdown_seen),
            "max_drawdown_percent": self._max_drawdown_percent,
            "max_drawdown_timestamp": self._max_drawdown_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        """Restore portfolio from persisted state."""
        portfolio = cls(
            initial_capital=Decimal(data["initial_capital"]),
            high_water_mark=Decimal(data["high_water_mark"]),
        )
        portfolio._total_equity = Decimal(data["total_equity"])
        portfolio._hwm_timestamp = datetime.fromisoformat(data["hwm_timestamp"])
        portfolio._available_balance = Decimal(data["available_balance"])
        portfolio._position_value = Decimal(data["position_value"])
        portfolio._open_order_exposure = Decimal(data["open_order_exposure"])
        portfolio._unrealized_pnl = Decimal(data["unrealized_pnl"])
        portfolio._realized_pnl = Decimal(data["realized_pnl"])
        portfolio._max_drawdown_seen = Decimal(data["max_drawdown_seen"])
        portfolio._max_drawdown_percent = data["max_drawdown_percent"]
        portfolio._max_drawdown_timestamp = datetime.fromisoformat(
            data["max_drawdown_timestamp"]
        )
        return portfolio
