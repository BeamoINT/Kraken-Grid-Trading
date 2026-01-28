"""
Grid Calculator for computing grid levels and volumes.

Calculates grid price levels based on:
- Center price (current market price)
- Volatility (ATR) for dynamic spacing
- Spacing strategy (equal or geometric)
- Risk parameters for volume sizing
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional
import time

from src.api import GridOrderType
from config.settings import GridConfig, RiskConfig, GridSpacing

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """Represents a single grid level."""

    index: int  # 0 = lowest, num_levels-1 = highest
    price: Decimal  # Target price for this level
    side: GridOrderType  # BUY (below center) or SELL (above center)
    volume: Decimal  # Order volume in base currency
    is_active: bool = True  # Whether orders should be placed at this level

    def __post_init__(self):
        """Validate grid level."""
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")


@dataclass
class GridParameters:
    """Input parameters for grid calculation."""

    center_price: Decimal  # Current market price (grid center)
    atr: Decimal  # Current ATR for volatility-based spacing
    num_levels: int  # Total number of grid levels
    range_percent: float  # Total range as % of center price
    spacing: GridSpacing  # EQUAL or GEOMETRIC
    order_size_quote: Decimal  # Order size in quote currency (USD)
    capital: Decimal  # Total capital for risk calculations
    price_decimals: int = 1  # Price precision (1 for XBTUSD)
    volume_decimals: int = 8  # Volume precision (8 for XBT)

    def __post_init__(self):
        """Validate parameters."""
        if self.center_price <= 0:
            raise ValueError(f"Center price must be positive, got {self.center_price}")
        if self.num_levels < 2:
            raise ValueError(f"Need at least 2 levels, got {self.num_levels}")
        if self.range_percent <= 0:
            raise ValueError(f"Range percent must be positive, got {self.range_percent}")


@dataclass
class GridState:
    """Complete state of the grid."""

    levels: List[GridLevel]
    center_price: Decimal
    upper_bound: Decimal  # Highest grid price
    lower_bound: Decimal  # Lowest grid price
    total_buy_exposure: Decimal  # Sum of all buy order values (volume * price)
    total_sell_exposure: Decimal  # Sum of all sell order values
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def num_levels(self) -> int:
        """Get total number of levels."""
        return len(self.levels)

    @property
    def num_buy_levels(self) -> int:
        """Get number of buy levels."""
        return sum(1 for level in self.levels if level.side == GridOrderType.BUY)

    @property
    def num_sell_levels(self) -> int:
        """Get number of sell levels."""
        return sum(1 for level in self.levels if level.side == GridOrderType.SELL)

    @property
    def active_levels(self) -> List[GridLevel]:
        """Get only active levels."""
        return [level for level in self.levels if level.is_active]

    @property
    def range_percent(self) -> float:
        """Calculate actual range as percentage of center."""
        if self.center_price == 0:
            return 0.0
        range_value = self.upper_bound - self.lower_bound
        return float(range_value / self.center_price * 100)

    def get_level_by_index(self, index: int) -> Optional[GridLevel]:
        """Get level by index."""
        for level in self.levels:
            if level.index == index:
                return level
        return None

    def get_levels_by_side(self, side: GridOrderType) -> List[GridLevel]:
        """Get all levels for a given side."""
        return [level for level in self.levels if level.side == side]


class GridCalculator:
    """
    Calculates grid levels and volumes.

    Supports:
    - Equal spacing (linear distribution)
    - Geometric spacing (percentage-based distribution)
    - ATR-based dynamic spacing
    - Volume sizing based on risk rules

    Example:
        calculator = GridCalculator(grid_config, risk_config)
        params = GridParameters(
            center_price=Decimal("50000"),
            atr=Decimal("500"),
            num_levels=10,
            range_percent=5.0,
            spacing=GridSpacing.EQUAL,
            order_size_quote=Decimal("40"),
            capital=Decimal("400"),
        )
        grid_state = calculator.compute_levels(params)
    """

    # Kraken minimum order size in USD
    MIN_ORDER_USD = Decimal("5")

    def __init__(
        self,
        config: GridConfig,
        risk_config: RiskConfig,
    ):
        """
        Initialize grid calculator.

        Args:
            config: Grid configuration parameters
            risk_config: Risk management parameters
        """
        self._config = config
        self._risk_config = risk_config

    def compute_levels(self, params: GridParameters) -> GridState:
        """
        Compute all grid levels based on parameters.

        Args:
            params: Grid calculation parameters

        Returns:
            GridState with all level details
        """
        # Calculate price levels based on spacing strategy
        if params.spacing == GridSpacing.EQUAL:
            prices = self.compute_level_prices_equal(
                params.center_price,
                params.range_percent,
                params.num_levels,
            )
        else:
            prices = self.compute_level_prices_geometric(
                params.center_price,
                params.range_percent,
                params.num_levels,
            )

        # Round prices to exchange precision
        prices = [
            self._round_price(price, params.price_decimals) for price in prices
        ]

        # Determine center index (levels below are buys, above are sells)
        center_index = params.num_levels // 2

        # Create grid levels
        levels: List[GridLevel] = []
        total_buy_exposure = Decimal("0")
        total_sell_exposure = Decimal("0")

        for i, price in enumerate(prices):
            # Determine side: below center = BUY, above center = SELL
            side = GridOrderType.BUY if i < center_index else GridOrderType.SELL

            # Compute volume for this level
            volume = self.compute_volume_for_level(
                price,
                params.order_size_quote,
                params.capital,
                self._risk_config.order_risk_percent,
            )

            # Round volume to exchange precision
            volume = self._round_volume(volume, params.volume_decimals)

            # Check minimum order size
            adjusted_volume = self.adjust_for_min_order_size(
                volume, price, self.MIN_ORDER_USD
            )

            if adjusted_volume is None:
                logger.warning(
                    f"Level {i} volume too small, skipping: "
                    f"{volume} @ {price} = ${volume * price}"
                )
                continue

            level = GridLevel(
                index=i,
                price=price,
                side=side,
                volume=adjusted_volume,
                is_active=True,
            )
            levels.append(level)

            # Track exposure
            exposure = adjusted_volume * price
            if side == GridOrderType.BUY:
                total_buy_exposure += exposure
            else:
                total_sell_exposure += exposure

        if not levels:
            raise ValueError("No valid grid levels could be created")

        # Get bounds
        lower_bound = min(level.price for level in levels)
        upper_bound = max(level.price for level in levels)

        grid_state = GridState(
            levels=levels,
            center_price=params.center_price,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            total_buy_exposure=total_buy_exposure,
            total_sell_exposure=total_sell_exposure,
        )

        logger.info(
            f"Computed grid: {len(levels)} levels, "
            f"center={params.center_price}, "
            f"range={lower_bound}-{upper_bound} ({grid_state.range_percent:.2f}%), "
            f"buy_exposure=${total_buy_exposure:.2f}, "
            f"sell_exposure=${total_sell_exposure:.2f}"
        )

        return grid_state

    def compute_level_prices_equal(
        self,
        center: Decimal,
        range_percent: float,
        num_levels: int,
    ) -> List[Decimal]:
        """
        Calculate prices with equal (linear) spacing.

        Distributes levels evenly across the range.

        Args:
            center: Center price
            range_percent: Total range as percentage of center
            num_levels: Number of levels to create

        Returns:
            List of prices from lowest to highest
        """
        # Calculate total range
        half_range = center * Decimal(str(range_percent)) / Decimal("100") / Decimal("2")

        lower_bound = center - half_range
        upper_bound = center + half_range

        if num_levels == 1:
            return [center]

        # Calculate step size
        step = (upper_bound - lower_bound) / Decimal(str(num_levels - 1))

        prices = []
        for i in range(num_levels):
            price = lower_bound + step * Decimal(str(i))
            prices.append(price)

        return prices

    def compute_level_prices_geometric(
        self,
        center: Decimal,
        range_percent: float,
        num_levels: int,
    ) -> List[Decimal]:
        """
        Calculate prices with geometric (percentage-based) spacing.

        Each level is separated by the same percentage, creating
        exponential spacing that's useful for volatile markets.

        Args:
            center: Center price
            range_percent: Total range as percentage of center
            num_levels: Number of levels to create

        Returns:
            List of prices from lowest to highest
        """
        if num_levels == 1:
            return [center]

        # Calculate the ratio between adjacent levels
        # If total range is R%, then each step should be (1+R/100)^(1/n)
        half_levels = num_levels // 2
        if half_levels == 0:
            half_levels = 1

        # Calculate ratio for each step
        total_ratio = Decimal("1") + Decimal(str(range_percent)) / Decimal("100")
        step_ratio = total_ratio ** (Decimal("1") / Decimal(str(half_levels)))

        prices = []

        # Generate prices below center (divide by ratio)
        price = center
        lower_prices = []
        for _ in range(half_levels):
            price = price / step_ratio
            lower_prices.append(price)
        lower_prices.reverse()
        prices.extend(lower_prices)

        # Add center
        prices.append(center)

        # Generate prices above center (multiply by ratio)
        price = center
        for _ in range(num_levels - half_levels - 1):
            price = price * step_ratio
            prices.append(price)

        return prices

    def compute_level_prices_atr(
        self,
        center: Decimal,
        atr: Decimal,
        num_levels: int,
        atr_multiplier: float = 0.5,
    ) -> List[Decimal]:
        """
        Calculate prices based on ATR multiples.

        Each level is separated by a multiple of ATR, providing
        volatility-adjusted spacing.

        Args:
            center: Center price
            atr: Current Average True Range
            num_levels: Number of levels to create
            atr_multiplier: Multiplier for ATR (default 0.5 = half ATR per level)

        Returns:
            List of prices from lowest to highest
        """
        if num_levels == 1:
            return [center]

        if atr <= 0:
            logger.warning("ATR is zero or negative, falling back to equal spacing")
            # Calculate reasonable range_percent based on typical volatility
            return self.compute_level_prices_equal(center, 5.0, num_levels)

        # Calculate step size as multiple of ATR
        step = atr * Decimal(str(atr_multiplier))

        prices = []
        half_levels = num_levels // 2

        # Generate prices below center
        for i in range(half_levels, 0, -1):
            price = center - step * Decimal(str(i))
            if price > 0:
                prices.append(price)

        # Add center
        prices.append(center)

        # Generate prices above center
        for i in range(1, num_levels - half_levels):
            price = center + step * Decimal(str(i))
            prices.append(price)

        return prices

    def compute_volume_for_level(
        self,
        price: Decimal,
        order_size_quote: Decimal,
        capital: Decimal,
        max_risk_percent: float,
    ) -> Decimal:
        """
        Compute volume ensuring max risk per level.

        Volume is capped to limit loss to max_risk_percent of capital
        if the position moves against us.

        Args:
            price: Level price
            order_size_quote: Desired order size in quote currency (USD)
            capital: Total capital for risk calculation
            max_risk_percent: Maximum risk percentage per level

        Returns:
            Volume in base currency
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Calculate base volume from order size
        base_volume = order_size_quote / price

        # Calculate max volume based on risk limit
        # If we risk max_risk_percent of capital per level
        max_risk_amount = capital * Decimal(str(max_risk_percent)) / Decimal("100")
        max_volume = max_risk_amount / price

        # Use the smaller of the two
        volume = min(base_volume, max_volume)

        logger.debug(
            f"Volume calculation: price={price}, "
            f"base_volume={base_volume:.8f}, "
            f"max_volume={max_volume:.8f}, "
            f"final={volume:.8f}"
        )

        return volume

    def adjust_for_min_order_size(
        self,
        volume: Decimal,
        price: Decimal,
        min_order_usd: Decimal,
    ) -> Optional[Decimal]:
        """
        Ensure volume meets Kraken minimum order size.

        Args:
            volume: Calculated volume
            price: Level price
            min_order_usd: Minimum order size in USD

        Returns:
            Adjusted volume or None if order would be too small
        """
        order_value = volume * price

        if order_value < min_order_usd:
            # Order too small - return None to skip this level
            return None

        return volume

    def _round_price(self, price: Decimal, decimals: int) -> Decimal:
        """Round price to exchange precision."""
        quantize_str = "0." + "0" * decimals if decimals > 0 else "1"
        return price.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _round_volume(self, volume: Decimal, decimals: int) -> Decimal:
        """Round volume to exchange precision (round down to be safe)."""
        quantize_str = "0." + "0" * decimals if decimals > 0 else "1"
        return volume.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def validate_grid_exposure(
        self,
        grid_state: GridState,
        capital: Decimal,
    ) -> List[str]:
        """
        Validate grid against risk rules.

        Args:
            grid_state: Grid state to validate
            capital: Total capital

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

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
