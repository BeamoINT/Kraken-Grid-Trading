"""
Rebalancer for detecting grid drift and triggering rebalancing.

Monitors:
- Price drift from grid center
- Position drift (accumulating long/short)
- Fill imbalance (too many buys vs sells)
- Regime changes
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional

from src.api import OrderManager, GridOrderType, GridPosition
from src.regime import MarketRegime

from .grid_calculator import GridState

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Reasons for triggering rebalance."""

    PRICE_DRIFT = auto()  # Price moved significantly from center
    POSITION_DRIFT = auto()  # Position too long/short
    REGIME_CHANGE = auto()  # Market regime changed
    FILL_IMBALANCE = auto()  # Too many fills on one side
    MANUAL = auto()  # User-requested
    SCHEDULED = auto()  # Time-based


@dataclass
class DriftMetrics:
    """Metrics for grid drift detection."""

    price_drift_percent: float  # % price moved from grid center
    position_drift_percent: float  # % of max position reached
    fill_imbalance: float  # Ratio of buys to sells (1.0 = balanced)
    time_since_last_rebalance: float  # Seconds since last rebalance
    levels_filled: int  # How many levels have been filled
    levels_remaining: int  # Active levels remaining
    current_price: Decimal  # Current market price
    grid_center: Decimal  # Grid center price

    @property
    def is_price_outside_grid(self) -> bool:
        """Check if price is outside grid range."""
        return abs(self.price_drift_percent) > 100.0

    @property
    def fill_ratio(self) -> float:
        """Get fill ratio (filled / total)."""
        total = self.levels_filled + self.levels_remaining
        if total == 0:
            return 0.0
        return self.levels_filled / total


@dataclass
class RebalanceDecision:
    """Decision from rebalancer."""

    should_rebalance: bool
    reason: Optional[RebalanceReason] = None
    urgency: float = 0.0  # 0-1, how urgent (affects batching)
    suggested_action: str = ""  # "full_rebuild", "shift", "partial"
    details: str = ""  # Human-readable details

    def __str__(self) -> str:
        if not self.should_rebalance:
            return "RebalanceDecision(no rebalance needed)"
        return (
            f"RebalanceDecision(REBALANCE: reason={self.reason.name}, "
            f"urgency={self.urgency:.2f}, action={self.suggested_action})"
        )


class Rebalancer:
    """
    Detects grid drift and triggers rebalancing.

    Monitors multiple drift signals and decides when the grid
    needs to be recalculated and redeployed.

    Triggers:
        PRICE_DRIFT: Price moves beyond rebalance_threshold from center
        POSITION_DRIFT: Position exceeds position_threshold of max
        FILL_IMBALANCE: Buy/sell fills ratio exceeds imbalance_threshold
        REGIME_CHANGE: Market regime changes

    Example:
        rebalancer = Rebalancer(rebalance_threshold=0.1)

        metrics = rebalancer.compute_drift_metrics(
            grid_state, current_price, position, order_manager
        )
        decision = rebalancer.should_rebalance(
            metrics, current_regime, previous_regime
        )

        if decision.should_rebalance:
            new_center = rebalancer.get_suggested_new_center(...)
            # Trigger grid recalculation
    """

    def __init__(
        self,
        rebalance_threshold: float = 0.1,  # 10% price drift triggers rebalance
        position_threshold: float = 0.5,  # 50% of max position
        imbalance_threshold: float = 3.0,  # 3:1 buy/sell ratio
        min_rebalance_interval: float = 300.0,  # 5 min minimum between rebalances
        max_capital: Decimal = Decimal("400"),  # For position drift calculation
    ):
        """
        Initialize rebalancer.

        Args:
            rebalance_threshold: Price drift threshold (0.1 = 10%)
            position_threshold: Position drift threshold (0.5 = 50% of max)
            imbalance_threshold: Fill imbalance threshold (3.0 = 3:1 ratio)
            min_rebalance_interval: Minimum seconds between rebalances
            max_capital: Maximum capital for position calculations
        """
        self._rebalance_threshold = rebalance_threshold
        self._position_threshold = position_threshold
        self._imbalance_threshold = imbalance_threshold
        self._min_rebalance_interval = min_rebalance_interval
        self._max_capital = max_capital

        self._last_rebalance_time: float = 0.0
        self._rebalance_count = 0

    def compute_drift_metrics(
        self,
        grid_state: GridState,
        current_price: Decimal,
        position: GridPosition,
        order_manager: OrderManager,
    ) -> DriftMetrics:
        """
        Calculate current drift metrics.

        Args:
            grid_state: Current grid state
            current_price: Current market price
            position: Current position
            order_manager: Order manager for fill counts

        Returns:
            DriftMetrics with all measurements
        """
        # Calculate price drift
        if grid_state.center_price > 0:
            price_drift = (
                (current_price - grid_state.center_price) /
                grid_state.center_price * 100
            )
        else:
            price_drift = 0.0

        # Calculate position drift (as % of max capital)
        position_value = abs(position.quantity * current_price)
        if self._max_capital > 0:
            position_drift = float(position_value / self._max_capital * 100)
        else:
            position_drift = 0.0

        # Calculate fill imbalance
        buy_count = position.buy_count
        sell_count = position.sell_count

        if sell_count > 0:
            fill_imbalance = buy_count / sell_count
        elif buy_count > 0:
            fill_imbalance = float("inf")
        else:
            fill_imbalance = 1.0  # Balanced if no fills

        # Count levels
        active_orders = order_manager.get_active_orders()
        filled_orders = order_manager.get_filled_orders()

        levels_filled = len(filled_orders)
        levels_remaining = len(active_orders)

        # Time since last rebalance
        time_since = time.time() - self._last_rebalance_time

        return DriftMetrics(
            price_drift_percent=float(price_drift),
            position_drift_percent=position_drift,
            fill_imbalance=fill_imbalance,
            time_since_last_rebalance=time_since,
            levels_filled=levels_filled,
            levels_remaining=levels_remaining,
            current_price=current_price,
            grid_center=grid_state.center_price,
        )

    def should_rebalance(
        self,
        metrics: DriftMetrics,
        current_regime: MarketRegime,
        previous_regime: Optional[MarketRegime],
    ) -> RebalanceDecision:
        """
        Determine if rebalancing is needed.

        Checks all triggers and returns decision with highest
        urgency if multiple triggers fire.

        Args:
            metrics: Current drift metrics
            current_regime: Current market regime
            previous_regime: Previous market regime (None if first check)

        Returns:
            RebalanceDecision with recommendation
        """
        # Check minimum interval
        if metrics.time_since_last_rebalance < self._min_rebalance_interval:
            return RebalanceDecision(
                should_rebalance=False,
                details=f"Min interval not met ({metrics.time_since_last_rebalance:.0f}s < {self._min_rebalance_interval}s)",
            )

        decisions = []

        # Check regime change
        if previous_regime is not None and current_regime != previous_regime:
            urgency = self._calculate_regime_change_urgency(
                current_regime, previous_regime
            )
            decisions.append(
                RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.REGIME_CHANGE,
                    urgency=urgency,
                    suggested_action="full_rebuild",
                    details=f"Regime changed: {previous_regime.name} -> {current_regime.name}",
                )
            )

        # Check price drift
        price_drift_abs = abs(metrics.price_drift_percent) / 100
        if price_drift_abs > self._rebalance_threshold:
            urgency = min(price_drift_abs / (self._rebalance_threshold * 2), 1.0)
            action = "full_rebuild" if metrics.is_price_outside_grid else "shift"
            decisions.append(
                RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.PRICE_DRIFT,
                    urgency=urgency,
                    suggested_action=action,
                    details=f"Price drift {metrics.price_drift_percent:.2f}% exceeds threshold {self._rebalance_threshold*100:.0f}%",
                )
            )

        # Check position drift
        if metrics.position_drift_percent / 100 > self._position_threshold:
            urgency = min(
                (metrics.position_drift_percent / 100) / (self._position_threshold * 1.5),
                1.0
            )
            decisions.append(
                RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.POSITION_DRIFT,
                    urgency=urgency,
                    suggested_action="partial",
                    details=f"Position {metrics.position_drift_percent:.1f}% of max exceeds threshold",
                )
            )

        # Check fill imbalance
        if metrics.fill_imbalance > self._imbalance_threshold:
            urgency = min(
                metrics.fill_imbalance / (self._imbalance_threshold * 2), 1.0
            )
            decisions.append(
                RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.FILL_IMBALANCE,
                    urgency=urgency,
                    suggested_action="shift",
                    details=f"Fill imbalance {metrics.fill_imbalance:.1f}:1 (buys:sells)",
                )
            )
        elif metrics.fill_imbalance < 1.0 / self._imbalance_threshold:
            urgency = min(
                (1.0 / metrics.fill_imbalance) / (self._imbalance_threshold * 2), 1.0
            )
            decisions.append(
                RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.FILL_IMBALANCE,
                    urgency=urgency,
                    suggested_action="shift",
                    details=f"Fill imbalance 1:{1.0/metrics.fill_imbalance:.1f} (buys:sells)",
                )
            )

        # Return highest urgency decision
        if not decisions:
            return RebalanceDecision(
                should_rebalance=False,
                details="No rebalance triggers met",
            )

        # Sort by urgency and return highest
        decisions.sort(key=lambda d: d.urgency, reverse=True)
        return decisions[0]

    def calculate_rebalance_urgency(self, metrics: DriftMetrics) -> float:
        """
        Calculate overall urgency score (0-1).

        Higher urgency = faster rebalance needed.
        Used to prioritize rebalancing during high activity.

        Args:
            metrics: Current drift metrics

        Returns:
            Urgency score from 0 to 1
        """
        urgency = 0.0

        # Price drift contribution (max 0.4)
        price_drift_norm = abs(metrics.price_drift_percent) / 100 / self._rebalance_threshold
        urgency += min(price_drift_norm * 0.4, 0.4)

        # Position drift contribution (max 0.3)
        position_drift_norm = metrics.position_drift_percent / 100 / self._position_threshold
        urgency += min(position_drift_norm * 0.3, 0.3)

        # Fill ratio contribution (max 0.2)
        fill_ratio_contribution = metrics.fill_ratio * 0.2
        urgency += fill_ratio_contribution

        # Price outside grid is urgent (add 0.1)
        if metrics.is_price_outside_grid:
            urgency += 0.1

        return min(urgency, 1.0)

    def get_suggested_new_center(
        self,
        grid_state: GridState,
        current_price: Decimal,
        position: GridPosition,
    ) -> Decimal:
        """
        Calculate optimal new grid center.

        Considers:
        - Current price
        - Position bias (favor reducing exposure)
        - Recent fill prices

        Args:
            grid_state: Current grid state
            current_price: Current market price
            position: Current position

        Returns:
            Suggested new center price
        """
        # Start with current price as base
        new_center = current_price

        # Adjust based on position
        # If we're long, shift grid up slightly to encourage sells
        # If we're short, shift grid down slightly to encourage buys
        if position.quantity != 0 and position.avg_entry_price > 0:
            position_value = position.quantity * current_price

            # Calculate shift as percentage of position value
            # Max shift: 1% of center price
            max_shift = current_price * Decimal("0.01")

            if position.quantity > 0:
                # Long position - shift up to sell
                shift = min(
                    abs(position_value) / self._max_capital * max_shift,
                    max_shift
                )
                new_center = current_price + shift
                logger.debug(
                    f"Long position: shifting center up by {shift} "
                    f"to {new_center}"
                )
            else:
                # Short position - shift down to buy
                shift = min(
                    abs(position_value) / self._max_capital * max_shift,
                    max_shift
                )
                new_center = current_price - shift
                logger.debug(
                    f"Short position: shifting center down by {shift} "
                    f"to {new_center}"
                )

        logger.info(
            f"Suggested new center: {new_center} "
            f"(current price: {current_price}, "
            f"old center: {grid_state.center_price})"
        )

        return new_center

    def mark_rebalanced(self) -> None:
        """Mark that a rebalance has occurred."""
        self._last_rebalance_time = time.time()
        self._rebalance_count += 1
        logger.info(f"Rebalance #{self._rebalance_count} completed")

    def _calculate_regime_change_urgency(
        self,
        current: MarketRegime,
        previous: MarketRegime,
    ) -> float:
        """
        Calculate urgency for regime change.

        Some regime transitions are more urgent than others.

        Args:
            current: New regime
            previous: Old regime

        Returns:
            Urgency score 0-1
        """
        # High urgency transitions
        high_urgency = {
            (MarketRegime.RANGING, MarketRegime.BREAKOUT),
            (MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT),
            (MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT),
            (MarketRegime.RANGING, MarketRegime.TRENDING_DOWN),
            (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN),
        }

        if (previous, current) in high_urgency:
            return 0.9

        # Medium urgency transitions
        medium_urgency = {
            (MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY),
            (MarketRegime.TRENDING_UP, MarketRegime.HIGH_VOLATILITY),
            (MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY),
            (MarketRegime.RANGING, MarketRegime.TRENDING_UP),
        }

        if (previous, current) in medium_urgency:
            return 0.6

        # Recovery transitions (low urgency)
        low_urgency = {
            (MarketRegime.BREAKOUT, MarketRegime.RANGING),
            (MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGING),
            (MarketRegime.TRENDING_DOWN, MarketRegime.RANGING),
        }

        if (previous, current) in low_urgency:
            return 0.3

        # Default
        return 0.5

    @property
    def last_rebalance_time(self) -> float:
        """Get timestamp of last rebalance."""
        return self._last_rebalance_time

    @property
    def rebalance_count(self) -> int:
        """Get total number of rebalances."""
        return self._rebalance_count

    @property
    def rebalance_threshold(self) -> float:
        """Get rebalance threshold."""
        return self._rebalance_threshold

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._last_rebalance_time = 0.0
        self._rebalance_count = 0
