"""
Grid Strategy for adapting grid behavior based on market regime.

Provides regime-specific adaptations:
- RANGING: Tighter spacing for mean reversion
- TRENDING_UP: Shift grid up, favor sells
- TRENDING_DOWN: Pause buys, tighter stop-loss
- HIGH_VOLATILITY: Widen spacing
- BREAKOUT: Pause trading entirely
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Tuple, Optional

from src.regime import MarketRegime
from config.settings import GridConfig, RiskConfig, GridSpacing

from .grid_calculator import (
    GridCalculator,
    GridParameters,
    GridState,
    GridLevel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeAdaptation:
    """
    Adaptations to apply for a specific market regime.

    Attributes:
        spacing_multiplier: Multiply base spacing (>1 = wider, <1 = tighter)
        shift_percent: Shift grid center (+ = up, - = down)
        buy_levels_active: Enable/disable buy orders
        sell_levels_active: Enable/disable sell orders
        pause_trading: Completely pause all activity
        stop_loss_tightening: Reduce stop-loss distance (0-1, 0=no change, 1=very tight)
    """

    spacing_multiplier: float = 1.0
    shift_percent: float = 0.0
    buy_levels_active: bool = True
    sell_levels_active: bool = True
    pause_trading: bool = False
    stop_loss_tightening: float = 0.0


class GridStrategy:
    """
    Adapts grid behavior based on detected market regime.

    The strategy modifies grid parameters based on the predicted
    market regime from the ML model. Each regime has specific
    adaptations to optimize grid performance.

    Regime Behaviors:
        RANGING: Normal grid, tighter spacing (0.8x) for mean reversion
        TRENDING_UP: Shift grid up 2%, wider spacing (1.2x)
        TRENDING_DOWN: Pause buy orders, tighter stop-loss
        HIGH_VOLATILITY: Widen spacing (1.5x) to avoid whipsaws
        BREAKOUT: Pause trading entirely until regime stabilizes

    Example:
        strategy = GridStrategy(grid_config, risk_config, calculator)
        grid_state, adaptation = strategy.compute_adapted_grid(
            current_price=Decimal("50000"),
            atr=Decimal("500"),
            regime=MarketRegime.RANGING,
            confidence=0.85,
        )
    """

    # Class-level regime adaptation configurations
    REGIME_ADAPTATIONS: Dict[MarketRegime, RegimeAdaptation] = {
        MarketRegime.RANGING: RegimeAdaptation(
            spacing_multiplier=0.8,  # Tighter for mean reversion
            shift_percent=0.0,
            buy_levels_active=True,
            sell_levels_active=True,
            pause_trading=False,
            stop_loss_tightening=0.0,
        ),
        MarketRegime.TRENDING_UP: RegimeAdaptation(
            spacing_multiplier=1.2,  # Wider to catch bigger moves
            shift_percent=2.0,  # Shift grid up
            buy_levels_active=True,  # Still buy but at higher prices
            sell_levels_active=True,
            pause_trading=False,
            stop_loss_tightening=0.0,
        ),
        MarketRegime.TRENDING_DOWN: RegimeAdaptation(
            spacing_multiplier=1.0,
            shift_percent=-1.0,  # Slight downward shift
            buy_levels_active=False,  # PAUSE buys - don't catch falling knife
            sell_levels_active=True,
            pause_trading=False,
            stop_loss_tightening=0.5,  # Tighter stop-loss
        ),
        MarketRegime.HIGH_VOLATILITY: RegimeAdaptation(
            spacing_multiplier=1.5,  # Much wider to avoid whipsaws
            shift_percent=0.0,
            buy_levels_active=True,
            sell_levels_active=True,
            pause_trading=False,
            stop_loss_tightening=0.0,
        ),
        MarketRegime.BREAKOUT: RegimeAdaptation(
            spacing_multiplier=1.0,
            shift_percent=0.0,
            buy_levels_active=False,  # Pause all
            sell_levels_active=False,  # Pause all
            pause_trading=True,  # Complete pause
            stop_loss_tightening=0.0,
        ),
    }

    # Default adaptation when regime is uncertain
    DEFAULT_ADAPTATION = REGIME_ADAPTATIONS[MarketRegime.RANGING]

    def __init__(
        self,
        grid_config: GridConfig,
        risk_config: RiskConfig,
        calculator: GridCalculator,
    ):
        """
        Initialize grid strategy.

        Args:
            grid_config: Grid configuration parameters
            risk_config: Risk management parameters
            calculator: Grid calculator instance
        """
        self._grid_config = grid_config
        self._risk_config = risk_config
        self._calculator = calculator
        self._last_regime: Optional[MarketRegime] = None
        self._regime_change_count = 0

    def get_adaptation(
        self,
        regime: MarketRegime,
        confidence: float,
        min_confidence: Optional[float] = None,
    ) -> RegimeAdaptation:
        """
        Get regime adaptation, falling back to RANGING if confidence too low.

        If confidence is below the minimum threshold, we don't trust
        the regime prediction and fall back to conservative RANGING behavior.

        Args:
            regime: Predicted market regime
            confidence: Model confidence (0-1)
            min_confidence: Minimum confidence to act on prediction
                           (defaults to risk_config.min_confidence)

        Returns:
            RegimeAdaptation for the given regime
        """
        if min_confidence is None:
            min_confidence = self._risk_config.min_confidence

        if confidence < min_confidence:
            logger.info(
                f"Confidence {confidence:.2%} below threshold {min_confidence:.2%}, "
                f"falling back to RANGING behavior"
            )
            return self.DEFAULT_ADAPTATION

        adaptation = self.REGIME_ADAPTATIONS.get(regime, self.DEFAULT_ADAPTATION)
        logger.debug(
            f"Regime {regime.name}: spacing={adaptation.spacing_multiplier}x, "
            f"shift={adaptation.shift_percent}%, buys={adaptation.buy_levels_active}, "
            f"sells={adaptation.sell_levels_active}, pause={adaptation.pause_trading}"
        )
        return adaptation

    def adapt_parameters(
        self,
        base_params: GridParameters,
        regime: MarketRegime,
        confidence: float,
    ) -> GridParameters:
        """
        Modify grid parameters based on regime.

        Args:
            base_params: Original grid parameters
            regime: Predicted market regime
            confidence: Model confidence (0-1)

        Returns:
            New GridParameters with adapted values
        """
        adaptation = self.get_adaptation(regime, confidence)

        # Apply spacing multiplier to range_percent
        adapted_range = base_params.range_percent * adaptation.spacing_multiplier

        # Apply shift to center price
        shift_factor = Decimal("1") + Decimal(str(adaptation.shift_percent)) / Decimal("100")
        adapted_center = base_params.center_price * shift_factor

        return GridParameters(
            center_price=adapted_center,
            atr=base_params.atr,
            num_levels=base_params.num_levels,
            range_percent=adapted_range,
            spacing=base_params.spacing,
            order_size_quote=base_params.order_size_quote,
            capital=base_params.capital,
            price_decimals=base_params.price_decimals,
            volume_decimals=base_params.volume_decimals,
        )

    def compute_adapted_grid(
        self,
        current_price: Decimal,
        atr: Decimal,
        regime: MarketRegime,
        confidence: float,
        capital: Optional[Decimal] = None,
    ) -> Tuple[GridState, RegimeAdaptation]:
        """
        Full pipeline: adapt parameters and compute grid.

        Args:
            current_price: Current market price
            atr: Current ATR value
            regime: Predicted market regime
            confidence: Model confidence (0-1)
            capital: Total capital (defaults to grid_config total)

        Returns:
            Tuple of (GridState, applied RegimeAdaptation)
        """
        if capital is None:
            capital = Decimal(str(self._grid_config.total_capital_required))

        # Get adaptation
        adaptation = self.get_adaptation(regime, confidence)

        # Track regime changes
        if self._last_regime is not None and self._last_regime != regime:
            self._regime_change_count += 1
            logger.info(
                f"Regime changed: {self._last_regime.name} -> {regime.name} "
                f"(change #{self._regime_change_count})"
            )
        self._last_regime = regime

        # Check if trading should be paused
        should_pause, reason = self.should_pause_trading(regime, confidence)
        if should_pause:
            logger.warning(f"Trading paused: {reason}")
            # Return empty grid state when paused
            return self._create_paused_grid_state(current_price), adaptation

        # Create base parameters
        base_params = GridParameters(
            center_price=current_price,
            atr=atr,
            num_levels=self._grid_config.num_levels,
            range_percent=self._grid_config.range_percent,
            spacing=self._grid_config.spacing,
            order_size_quote=Decimal(str(self._grid_config.order_size_quote)),
            capital=capital,
        )

        # Adapt parameters based on regime
        adapted_params = self.adapt_parameters(base_params, regime, confidence)

        # Compute grid levels
        grid_state = self._calculator.compute_levels(adapted_params)

        # Apply level activation based on adaptation
        grid_state = self._apply_level_activation(grid_state, adaptation)

        logger.info(
            f"Adapted grid for {regime.name} (conf={confidence:.2%}): "
            f"{len(grid_state.active_levels)} active levels, "
            f"center={adapted_params.center_price}, "
            f"range={adapted_params.range_percent:.2f}%"
        )

        return grid_state, adaptation

    def should_pause_trading(
        self,
        regime: MarketRegime,
        confidence: float,
    ) -> Tuple[bool, str]:
        """
        Determine if trading should be paused.

        Args:
            regime: Predicted market regime
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_pause, reason)
        """
        adaptation = self.get_adaptation(regime, confidence)

        if adaptation.pause_trading:
            return True, f"regime_{regime.name.lower()}"

        # Also pause if confidence is very low (below half the threshold)
        if confidence < self._risk_config.min_confidence * 0.5:
            return True, "very_low_confidence"

        return False, ""

    def compute_stop_loss_price(
        self,
        grid_state: GridState,
        adaptation: RegimeAdaptation,
        base_stop_loss_percent: Optional[float] = None,
    ) -> Decimal:
        """
        Compute stop-loss price with regime adjustment.

        Stop-loss is calculated as a percentage below the lowest grid level,
        with the percentage tightened based on the regime adaptation.

        Args:
            grid_state: Current grid state
            adaptation: Applied regime adaptation
            base_stop_loss_percent: Base stop-loss percentage
                                   (defaults to risk_config.stop_loss_percent)

        Returns:
            Stop-loss price
        """
        if base_stop_loss_percent is None:
            base_stop_loss_percent = self._risk_config.stop_loss_percent

        # Apply tightening from adaptation
        # tightening=0 means no change, tightening=1 means very tight (50% of base)
        tightening_factor = 1.0 - (adaptation.stop_loss_tightening * 0.5)
        adjusted_percent = base_stop_loss_percent * tightening_factor

        # Calculate stop-loss price
        stop_loss_factor = Decimal("1") - Decimal(str(adjusted_percent)) / Decimal("100")
        stop_loss_price = grid_state.lower_bound * stop_loss_factor

        logger.debug(
            f"Stop-loss: base={base_stop_loss_percent}%, "
            f"adjusted={adjusted_percent:.2f}%, "
            f"price={stop_loss_price}"
        )

        return stop_loss_price

    def _apply_level_activation(
        self,
        grid_state: GridState,
        adaptation: RegimeAdaptation,
    ) -> GridState:
        """
        Apply level activation based on adaptation.

        Deactivates buy or sell levels based on the regime adaptation.

        Args:
            grid_state: Grid state with all levels
            adaptation: Regime adaptation to apply

        Returns:
            Updated grid state with correct level activation
        """
        from src.api import GridOrderType

        for level in grid_state.levels:
            if level.side == GridOrderType.BUY and not adaptation.buy_levels_active:
                level.is_active = False
            elif level.side == GridOrderType.SELL and not adaptation.sell_levels_active:
                level.is_active = False

        return grid_state

    def _create_paused_grid_state(self, current_price: Decimal) -> GridState:
        """Create an empty grid state for paused trading."""
        return GridState(
            levels=[],
            center_price=current_price,
            upper_bound=current_price,
            lower_bound=current_price,
            total_buy_exposure=Decimal("0"),
            total_sell_exposure=Decimal("0"),
        )

    @property
    def last_regime(self) -> Optional[MarketRegime]:
        """Get the last processed regime."""
        return self._last_regime

    @property
    def regime_change_count(self) -> int:
        """Get the number of regime changes observed."""
        return self._regime_change_count
