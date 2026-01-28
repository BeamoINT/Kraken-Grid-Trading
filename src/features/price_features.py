"""
Price-based features for market analysis.

Computes features derived from price data:
- Returns (log returns, simple returns)
- Momentum indicators
- Moving averages and crossovers
- Price position relative to historical ranges
- Gap analysis
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns.

    Log returns are preferred over simple returns because:
    - Additive over time: log(P_t/P_0) = sum of log returns
    - More normally distributed for shorter timeframes
    - Better for statistical analysis

    Args:
        close: Close prices
        periods: Number of periods for return calculation (default: 1)

    Returns:
        Series of log returns
    """
    return np.log(close / close.shift(periods))


def simple_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate simple percentage returns.

    Args:
        close: Close prices
        periods: Number of periods for return calculation (default: 1)

    Returns:
        Series of simple returns (as decimals, not percentages)
    """
    return close.pct_change(periods=periods)


def cumulative_returns(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate cumulative returns over a rolling window.

    Args:
        close: Close prices
        window: Lookback window (default: 20)

    Returns:
        Series of cumulative returns
    """
    return (close / close.shift(window)) - 1


def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        close: Close prices
        period: SMA period

    Returns:
        Series of SMA values
    """
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    EMA gives more weight to recent prices, making it more
    responsive to new information than SMA.

    Args:
        close: Close prices
        period: EMA period (span)

    Returns:
        Series of EMA values
    """
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def wma(close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Weighted Moving Average.

    Linearly weighted: most recent data has highest weight.

    Args:
        close: Close prices
        period: WMA period

    Returns:
        Series of WMA values
    """
    weights = np.arange(1, period + 1)
    return close.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def hull_ma(close: pd.Series, period: int = 16) -> pd.Series:
    """
    Calculate Hull Moving Average.

    HMA reduces lag while maintaining smoothness:
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    Args:
        close: Close prices
        period: HMA period (default: 16)

    Returns:
        Series of HMA values
    """
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    wma_half = wma(close, half_period)
    wma_full = wma(close, period)

    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_period)


def price_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate price momentum (Rate of Change).

    Momentum = (Current Price - Price n periods ago) / Price n periods ago

    Args:
        close: Close prices
        period: Lookback period (default: 10)

    Returns:
        Series of momentum values
    """
    return (close - close.shift(period)) / close.shift(period)


def price_acceleration(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate price acceleration (second derivative of price).

    Measures the rate of change of momentum.

    Args:
        close: Close prices
        period: Lookback period (default: 10)

    Returns:
        Series of acceleration values
    """
    momentum = price_momentum(close, period)
    return momentum - momentum.shift(period)


def ma_crossover_signal(
    close: pd.Series,
    fast_period: int = 10,
    slow_period: int = 20,
) -> pd.Series:
    """
    Calculate moving average crossover signal.

    Signal values:
    - 1: Fast MA above Slow MA (bullish)
    - -1: Fast MA below Slow MA (bearish)
    - 0: Transition period (NaN handling)

    Args:
        close: Close prices
        fast_period: Fast MA period (default: 10)
        slow_period: Slow MA period (default: 20)

    Returns:
        Series of crossover signals
    """
    fast_ma = ema(close, fast_period)
    slow_ma = ema(close, slow_period)

    signal = pd.Series(
        np.where(fast_ma > slow_ma, 1, -1),
        index=close.index
    )

    # Set to 0 where we don't have enough data
    signal = signal.where(slow_ma.notna(), 0)

    return signal


def ma_distance(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate distance from price to moving average.

    Positive: Price above MA
    Negative: Price below MA

    Normalized by MA for comparability across price levels.

    Args:
        close: Close prices
        period: MA period (default: 20)

    Returns:
        Series of normalized distances
    """
    ma = sma(close, period)
    return (close - ma) / ma


def multi_ma_distances(
    close: pd.Series,
    periods: List[int] = None,
) -> pd.DataFrame:
    """
    Calculate distances from multiple moving averages.

    Args:
        close: Close prices
        periods: List of MA periods (default: [14, 50, 200])

    Returns:
        DataFrame with distance columns for each period
    """
    if periods is None:
        periods = [14, 50, 200]

    result = pd.DataFrame(index=close.index)

    for period in periods:
        result[f'ma_dist_{period}'] = ma_distance(close, period)

    return result


def price_position_in_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate price position within its historical range.

    Value of 1.0: At the high of the range
    Value of 0.0: At the low of the range
    Value of 0.5: At the middle of the range

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 20)

    Returns:
        Series of position values (0 to 1)
    """
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()

    position = (close - lowest) / (highest - lowest)
    return position.replace([np.inf, -np.inf], np.nan)


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Calculate classical pivot points.

    Pivot Point (P) = (High + Low + Close) / 3
    Support 1 (S1) = (2 * P) - High
    Support 2 (S2) = P - (High - Low)
    Resistance 1 (R1) = (2 * P) - Low
    Resistance 2 (R2) = P + (High - Low)

    Note: Uses previous candle's HLC for current pivot levels.

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        DataFrame with P, S1, S2, R1, R2 columns
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3

    s1 = (2 * pivot) - prev_high
    s2 = pivot - (prev_high - prev_low)
    r1 = (2 * pivot) - prev_low
    r2 = pivot + (prev_high - prev_low)

    return pd.DataFrame({
        'pivot': pivot,
        'support_1': s1,
        'support_2': s2,
        'resistance_1': r1,
        'resistance_2': r2,
    }, index=close.index)


def price_to_pivot_distance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Calculate normalized distance from current price to pivot levels.

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        DataFrame with distances to each pivot level
    """
    pivots = pivot_points(high, low, close)

    result = pd.DataFrame(index=close.index)
    for col in pivots.columns:
        result[f'dist_to_{col}'] = (close - pivots[col]) / close

    return result


def gap_analysis(
    open_price: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Calculate gap metrics between sessions.

    Gap = Current Open - Previous Close

    Args:
        open_price: Open prices
        close: Close prices

    Returns:
        DataFrame with gap metrics
    """
    prev_close = close.shift(1)

    gap = open_price - prev_close
    gap_percent = gap / prev_close

    # Gap direction
    gap_direction = pd.Series(
        np.where(gap > 0, 1, np.where(gap < 0, -1, 0)),
        index=close.index
    )

    # Gap filled within same candle
    gap_filled = pd.Series(
        np.where(
            (gap > 0) & (close <= prev_close),  # Gap up filled
            1,
            np.where(
                (gap < 0) & (close >= prev_close),  # Gap down filled
                1,
                0
            )
        ),
        index=close.index
    )

    return pd.DataFrame({
        'gap': gap,
        'gap_percent': gap_percent,
        'gap_direction': gap_direction,
        'gap_filled': gap_filled,
    }, index=close.index)


def higher_highs_lower_lows(
    high: pd.Series,
    low: pd.Series,
    period: int = 5,
) -> pd.DataFrame:
    """
    Detect higher highs and lower lows patterns.

    Trend identification:
    - Higher highs + higher lows: Uptrend
    - Lower highs + lower lows: Downtrend
    - Mixed: Consolidation/transition

    Args:
        high: High prices
        low: Low prices
        period: Lookback for comparison (default: 5)

    Returns:
        DataFrame with HH/LL flags and counts
    """
    prev_high = high.shift(period)
    prev_low = low.shift(period)

    # Higher high: current high > previous high
    higher_high = (high > prev_high).astype(int)
    lower_low = (low < prev_low).astype(int)
    higher_low = (low > prev_low).astype(int)
    lower_high = (high < prev_high).astype(int)

    # Count consecutive HH or LL
    hh_count = higher_high.rolling(window=4, min_periods=1).sum()
    ll_count = lower_low.rolling(window=4, min_periods=1).sum()

    # Trend score: positive = uptrend, negative = downtrend
    trend_score = (higher_high + higher_low - lower_high - lower_low).rolling(
        window=period, min_periods=1
    ).sum()

    return pd.DataFrame({
        'higher_high': higher_high,
        'lower_low': lower_low,
        'higher_low': higher_low,
        'lower_high': lower_high,
        'hh_count': hh_count,
        'll_count': ll_count,
        'trend_score': trend_score,
    }, index=high.index)


def swing_points(
    high: pd.Series,
    low: pd.Series,
    left_bars: int = 5,
    right_bars: int = 5,
) -> pd.DataFrame:
    """
    Identify swing highs and swing lows.

    Swing High: Highest point with lower highs on both sides
    Swing Low: Lowest point with higher lows on both sides

    Args:
        high: High prices
        low: Low prices
        left_bars: Bars to the left for confirmation (default: 5)
        right_bars: Bars to the right for confirmation (default: 5)

    Returns:
        DataFrame with swing_high and swing_low flags
    """
    total_bars = left_bars + right_bars + 1

    def is_swing_high(window):
        if len(window) < total_bars:
            return 0
        middle_idx = left_bars
        middle_val = window[middle_idx]
        left_max = max(window[:left_bars])
        right_max = max(window[left_bars + 1:])
        return 1 if middle_val > left_max and middle_val > right_max else 0

    def is_swing_low(window):
        if len(window) < total_bars:
            return 0
        middle_idx = left_bars
        middle_val = window[middle_idx]
        left_min = min(window[:left_bars])
        right_min = min(window[left_bars + 1:])
        return 1 if middle_val < left_min and middle_val < right_min else 0

    swing_high = high.rolling(window=total_bars, min_periods=total_bars).apply(
        is_swing_high, raw=True
    ).shift(-right_bars)  # Align with the actual swing point

    swing_low = low.rolling(window=total_bars, min_periods=total_bars).apply(
        is_swing_low, raw=True
    ).shift(-right_bars)

    return pd.DataFrame({
        'swing_high': swing_high.fillna(0).astype(int),
        'swing_low': swing_low.fillna(0).astype(int),
    }, index=high.index)


def compute_price_features(
    df: pd.DataFrame,
    short_window: int = 14,
    medium_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """
    Compute all price-based features for an OHLCV DataFrame.

    Args:
        df: DataFrame with open, high, low, close columns
        short_window: Short-term lookback (default: 14)
        medium_window: Medium-term lookback (default: 50)
        long_window: Long-term lookback (default: 200)

    Returns:
        DataFrame with all price features
    """
    features = pd.DataFrame(index=df.index)

    # Returns
    features['log_return'] = log_returns(df['close'])
    features['log_return_5'] = log_returns(df['close'], periods=5)
    features['log_return_20'] = log_returns(df['close'], periods=20)

    # Cumulative returns
    features['cum_return_short'] = cumulative_returns(df['close'], short_window)
    features['cum_return_medium'] = cumulative_returns(df['close'], medium_window)

    # Moving average distances
    features['ma_dist_short'] = ma_distance(df['close'], short_window)
    features['ma_dist_medium'] = ma_distance(df['close'], medium_window)
    features['ma_dist_long'] = ma_distance(df['close'], long_window)

    # Moving averages (for reference, can be dropped later)
    features['sma_short'] = sma(df['close'], short_window)
    features['sma_medium'] = sma(df['close'], medium_window)
    features['sma_long'] = sma(df['close'], long_window)
    features['ema_short'] = ema(df['close'], short_window)
    features['ema_medium'] = ema(df['close'], medium_window)

    # MA crossover signals
    features['ma_cross_short_medium'] = ma_crossover_signal(
        df['close'], short_window, medium_window
    )
    features['ma_cross_medium_long'] = ma_crossover_signal(
        df['close'], medium_window, long_window
    )

    # Momentum
    features['momentum_short'] = price_momentum(df['close'], short_window)
    features['momentum_medium'] = price_momentum(df['close'], medium_window)
    features['acceleration'] = price_acceleration(df['close'], short_window)

    # Price position in range
    features['price_position'] = price_position_in_range(
        df['high'], df['low'], df['close'], medium_window
    )

    # Higher highs / lower lows trend analysis
    hh_ll = higher_highs_lower_lows(df['high'], df['low'])
    features['hh_count'] = hh_ll['hh_count']
    features['ll_count'] = hh_ll['ll_count']
    features['trend_score'] = hh_ll['trend_score']

    # Candle body analysis
    features['candle_body'] = (df['close'] - df['open']) / df['open']
    features['candle_range'] = (df['high'] - df['low']) / df['low']
    features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

    return features
