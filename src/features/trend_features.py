"""
Trend-based features for market analysis.

Computes features related to market trends:
- ADX-based trend strength
- Moving average trends
- Trend direction and duration
- Trend consistency and quality
- Breakout detection
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from .indicators import (
    adx,
    macd,
    rsi,
    cci,
    ichimoku_cloud,
    donchian_channels,
)
from .price_features import sma, ema


def trend_strength(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate trend strength using ADX.

    ADX values:
    - 0-20: No trend / weak
    - 20-40: Developing trend
    - 40-60: Strong trend
    - 60+: Very strong trend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        Series of ADX values
    """
    adx_values, _, _ = adx(high, low, close, period)
    return adx_values


def trend_direction(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Determine trend direction using +DI/-DI.

    Returns:
    -  1: Uptrend (+DI > -DI)
    - -1: Downtrend (-DI > +DI)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        Series of direction values
    """
    _, plus_di, minus_di = adx(high, low, close, period)

    direction = pd.Series(
        np.where(plus_di > minus_di, 1, -1),
        index=close.index
    )

    return direction


def di_difference(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate +DI/-DI difference for trend bias.

    Positive: Bullish bias
    Negative: Bearish bias
    Magnitude indicates strength of directional bias.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        Series of DI difference values
    """
    _, plus_di, minus_di = adx(high, low, close, period)
    return plus_di - minus_di


def adx_slope(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    slope_period: int = 5,
) -> pd.Series:
    """
    Calculate ADX slope (rate of change).

    Positive slope: Trend strengthening
    Negative slope: Trend weakening

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)
        slope_period: Slope calculation period (default: 5)

    Returns:
        Series of ADX slope values
    """
    adx_values = trend_strength(high, low, close, period)
    return (adx_values - adx_values.shift(slope_period)) / slope_period


def ma_trend(
    close: pd.Series,
    short_period: int = 10,
    medium_period: int = 20,
    long_period: int = 50,
) -> pd.DataFrame:
    """
    Calculate MA-based trend indicators.

    Multiple timeframe analysis using moving averages.

    Args:
        close: Close prices
        short_period: Short MA period (default: 10)
        medium_period: Medium MA period (default: 20)
        long_period: Long MA period (default: 50)

    Returns:
        DataFrame with MA trend features
    """
    ma_short = ema(close, short_period)
    ma_medium = ema(close, medium_period)
    ma_long = ema(close, long_period)

    # Price relative to MAs
    price_vs_short = (close > ma_short).astype(int)
    price_vs_medium = (close > ma_medium).astype(int)
    price_vs_long = (close > ma_long).astype(int)

    # MA alignment (bullish: short > medium > long)
    ma_alignment = (
        (ma_short > ma_medium).astype(int) +
        (ma_medium > ma_long).astype(int) +
        (ma_short > ma_long).astype(int)
    )

    # Trend score: sum of position indicators (-3 to 3)
    trend_score = (
        price_vs_short.map({0: -1, 1: 1}) +
        price_vs_medium.map({0: -1, 1: 1}) +
        price_vs_long.map({0: -1, 1: 1})
    )

    # MA slopes
    short_slope = ma_short.diff(5) / ma_short.shift(5)
    medium_slope = ma_medium.diff(5) / ma_medium.shift(5)
    long_slope = ma_long.diff(5) / ma_long.shift(5)

    return pd.DataFrame({
        'price_above_short_ma': price_vs_short,
        'price_above_medium_ma': price_vs_medium,
        'price_above_long_ma': price_vs_long,
        'ma_alignment': ma_alignment,
        'trend_score': trend_score,
        'short_ma_slope': short_slope,
        'medium_ma_slope': medium_slope,
        'long_ma_slope': long_slope,
    }, index=close.index)


def golden_death_cross(
    close: pd.Series,
    short_period: int = 50,
    long_period: int = 200,
) -> pd.DataFrame:
    """
    Detect Golden Cross and Death Cross patterns.

    Golden Cross: Short MA crosses above Long MA (bullish)
    Death Cross: Short MA crosses below Long MA (bearish)

    Args:
        close: Close prices
        short_period: Short MA period (default: 50)
        long_period: Long MA period (default: 200)

    Returns:
        DataFrame with cross signals and state
    """
    ma_short = sma(close, short_period)
    ma_long = sma(close, long_period)

    # Current state
    short_above_long = ma_short > ma_long
    prev_short_above_long = short_above_long.shift(1).fillna(False)

    # Cross events (fillna to handle NaN values at start)
    golden_cross = (~prev_short_above_long) & short_above_long.fillna(False)
    death_cross = prev_short_above_long & (~short_above_long.fillna(False))

    # Days since last cross
    def days_since_cross(crosses: pd.Series) -> pd.Series:
        cross_idx = crosses[crosses].index
        result = pd.Series(np.nan, index=crosses.index)

        for i, idx in enumerate(crosses.index):
            prior_crosses = cross_idx[cross_idx < idx]
            if len(prior_crosses) > 0:
                result.loc[idx] = (idx - prior_crosses[-1]).days
            else:
                result.loc[idx] = np.nan

        return result

    return pd.DataFrame({
        'short_above_long': short_above_long.astype(int),
        'golden_cross': golden_cross.astype(int),
        'death_cross': death_cross.astype(int),
        'ma_spread': (ma_short - ma_long) / ma_long,
    }, index=close.index)


def macd_trend(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate MACD-based trend features.

    Args:
        close: Close prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        DataFrame with MACD trend features
    """
    macd_line, signal_line, histogram = macd(close, fast_period, slow_period, signal_period)

    # MACD above/below signal
    macd_above_signal = (macd_line > signal_line).astype(int)

    # MACD above/below zero
    macd_above_zero = (macd_line > 0).astype(int)

    # Histogram direction
    hist_rising = (histogram > histogram.shift(1)).astype(int)

    # Signal line crossover (fillna to handle NaN values)
    prev_above = (macd_line.shift(1) > signal_line.shift(1)).fillna(False)
    curr_above = (macd_line > signal_line).fillna(False)
    bullish_cross = (~prev_above) & curr_above
    bearish_cross = prev_above & (~curr_above)

    # Zero line crossover (fillna to handle NaN values)
    prev_above_zero = (macd_line.shift(1) > 0).fillna(False)
    curr_above_zero = (macd_line > 0).fillna(False)
    zero_bullish_cross = (~prev_above_zero) & curr_above_zero
    zero_bearish_cross = prev_above_zero & (~curr_above_zero)

    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
        'macd_above_signal': macd_above_signal,
        'macd_above_zero': macd_above_zero,
        'macd_hist_rising': hist_rising,
        'macd_bullish_cross': bullish_cross.astype(int),
        'macd_bearish_cross': bearish_cross.astype(int),
        'macd_zero_bullish': zero_bullish_cross.astype(int),
        'macd_zero_bearish': zero_bearish_cross.astype(int),
    }, index=close.index)


def rsi_trend(
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calculate RSI-based trend features.

    Args:
        close: Close prices
        period: RSI period (default: 14)

    Returns:
        DataFrame with RSI trend features
    """
    rsi_values = rsi(close, period)

    # RSI zones
    oversold = (rsi_values < 30).astype(int)
    overbought = (rsi_values > 70).astype(int)
    neutral_bullish = ((rsi_values >= 50) & (rsi_values <= 70)).astype(int)
    neutral_bearish = ((rsi_values >= 30) & (rsi_values < 50)).astype(int)

    # RSI direction
    rsi_rising = (rsi_values > rsi_values.shift(1)).astype(int)

    # RSI divergence signals
    # Bullish divergence: price lower low, RSI higher low
    price_lower = close < close.rolling(window=5).min().shift(1)
    rsi_higher = rsi_values > rsi_values.rolling(window=5).min().shift(1)
    bullish_div = (price_lower & rsi_higher).astype(int)

    # Bearish divergence: price higher high, RSI lower high
    price_higher = close > close.rolling(window=5).max().shift(1)
    rsi_lower = rsi_values < rsi_values.rolling(window=5).max().shift(1)
    bearish_div = (price_higher & rsi_lower).astype(int)

    return pd.DataFrame({
        'rsi': rsi_values,
        'rsi_oversold': oversold,
        'rsi_overbought': overbought,
        'rsi_neutral_bullish': neutral_bullish,
        'rsi_neutral_bearish': neutral_bearish,
        'rsi_rising': rsi_rising,
        'rsi_bullish_div': bullish_div,
        'rsi_bearish_div': bearish_div,
    }, index=close.index)


def cci_trend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculate CCI-based trend features.

    CCI Interpretation:
    - CCI > 100: Strong uptrend
    - CCI < -100: Strong downtrend
    - CCI between -100 and 100: Ranging

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CCI period (default: 20)

    Returns:
        DataFrame with CCI trend features
    """
    cci_values = cci(high, low, close, period)

    # CCI zones
    strong_bullish = (cci_values > 100).astype(int)
    strong_bearish = (cci_values < -100).astype(int)
    neutral = ((cci_values >= -100) & (cci_values <= 100)).astype(int)

    # CCI direction
    cci_rising = (cci_values > cci_values.shift(1)).astype(int)

    # Zero line crosses (fillna to handle NaN values)
    prev_above_zero = (cci_values.shift(1) > 0).fillna(False)
    curr_above_zero = (cci_values > 0).fillna(False)
    bullish_cross = (~prev_above_zero) & curr_above_zero
    bearish_cross = prev_above_zero & (~curr_above_zero)

    return pd.DataFrame({
        'cci': cci_values,
        'cci_strong_bullish': strong_bullish,
        'cci_strong_bearish': strong_bearish,
        'cci_neutral': neutral,
        'cci_rising': cci_rising,
        'cci_bullish_cross': bullish_cross.astype(int),
        'cci_bearish_cross': bearish_cross.astype(int),
    }, index=close.index)


def trend_consistency(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate trend consistency score.

    Measures how consistently price moves in one direction.
    Range: 0 to 1
    1.0: Perfect trend (all moves in same direction)
    0.0: No trend (equal up and down moves)

    Args:
        close: Close prices
        period: Lookback period (default: 20)

    Returns:
        Series of consistency values
    """
    changes = close.diff()
    up_moves = (changes > 0).astype(int)
    down_moves = (changes < 0).astype(int)

    up_count = up_moves.rolling(window=period, min_periods=period).sum()
    down_count = down_moves.rolling(window=period, min_periods=period).sum()

    # Consistency = |up - down| / (up + down)
    total = up_count + down_count
    consistency = (up_count - down_count).abs() / total

    return consistency.replace([np.inf, -np.inf], np.nan)


def trend_duration(
    close: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculate trend duration metrics.

    Counts consecutive closes above/below moving average.

    Args:
        close: Close prices
        period: MA period for reference (default: 20)

    Returns:
        DataFrame with duration metrics
    """
    ma = sma(close, period)
    above_ma = (close > ma).fillna(False)

    # Count consecutive periods above/below MA
    def count_consecutive(series: pd.Series) -> pd.Series:
        """Count consecutive True values."""
        series = series.fillna(False)  # Handle NaN values
        cumsum = series.cumsum()
        reset_points = cumsum.where(~series).ffill().fillna(0)
        return cumsum - reset_points

    days_above = count_consecutive(above_ma)
    days_below = count_consecutive(~above_ma)

    # Current trend duration (positive = above MA, negative = below MA)
    duration = days_above.where(above_ma, -days_below)

    return pd.DataFrame({
        'days_above_ma': days_above,
        'days_below_ma': days_below,
        'trend_duration': duration,
    }, index=close.index)


def donchian_trend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculate Donchian Channel trend features.

    Breakout trading strategy based on highest highs and lowest lows.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Channel period (default: 20)

    Returns:
        DataFrame with Donchian trend features
    """
    upper, middle, lower = donchian_channels(high, low, period)

    # Position within channel
    channel_position = (close - lower) / (upper - lower)

    # Breakout signals
    new_high = close >= upper
    new_low = close <= lower

    # Channel width (volatility indicator)
    channel_width = (upper - lower) / middle

    return pd.DataFrame({
        'donchian_upper': upper,
        'donchian_middle': middle,
        'donchian_lower': lower,
        'channel_position': channel_position.replace([np.inf, -np.inf], np.nan),
        'new_high': new_high.astype(int),
        'new_low': new_low.astype(int),
        'channel_width': channel_width,
    }, index=close.index)


def ichimoku_trend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud trend features.

    Comprehensive trend indicator from Japanese analysis.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Conversion line period (default: 9)
        kijun_period: Base line period (default: 26)
        senkou_b_period: Leading span B period (default: 52)

    Returns:
        DataFrame with Ichimoku trend features
    """
    ichimoku = ichimoku_cloud(high, low, close, tenkan_period, kijun_period, senkou_b_period)

    tenkan = ichimoku['tenkan']
    kijun = ichimoku['kijun']
    senkou_a = ichimoku['senkou_a']
    senkou_b = ichimoku['senkou_b']

    # Price relative to cloud
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    above_cloud = (close > cloud_top).astype(int)
    below_cloud = (close < cloud_bottom).astype(int)
    in_cloud = ((close >= cloud_bottom) & (close <= cloud_top)).astype(int)

    # Cloud color (bullish: senkou_a > senkou_b)
    bullish_cloud = (senkou_a > senkou_b).astype(int)

    # TK cross (tenkan crosses kijun)
    tk_bullish = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).astype(int)
    tk_bearish = ((tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))).astype(int)

    # Price relative to kijun
    price_above_kijun = (close > kijun).astype(int)

    # Cloud thickness (future support/resistance strength)
    cloud_thickness = (cloud_top - cloud_bottom) / close

    return pd.DataFrame({
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'above_cloud': above_cloud,
        'below_cloud': below_cloud,
        'in_cloud': in_cloud,
        'bullish_cloud': bullish_cloud,
        'tk_bullish': tk_bullish,
        'tk_bearish': tk_bearish,
        'price_above_kijun': price_above_kijun,
        'cloud_thickness': cloud_thickness,
    }, index=close.index)


def compute_trend_features(
    df: pd.DataFrame,
    adx_period: int = 14,
    short_ma: int = 14,
    medium_ma: int = 50,
    long_ma: int = 200,
) -> pd.DataFrame:
    """
    Compute all trend-based features for an OHLCV DataFrame.

    Args:
        df: DataFrame with open, high, low, close columns
        adx_period: ADX calculation period (default: 14)
        short_ma: Short MA period (default: 14)
        medium_ma: Medium MA period (default: 50)
        long_ma: Long MA period (default: 200)

    Returns:
        DataFrame with all trend features
    """
    features = pd.DataFrame(index=df.index)

    # ADX-based features
    adx_values, plus_di, minus_di = adx(df['high'], df['low'], df['close'], adx_period)
    features['adx'] = adx_values
    features['plus_di'] = plus_di
    features['minus_di'] = minus_di
    features['di_difference'] = plus_di - minus_di
    features['trend_direction'] = trend_direction(df['high'], df['low'], df['close'], adx_period)
    features['adx_slope'] = adx_slope(df['high'], df['low'], df['close'], adx_period)

    # MA-based trend features
    ma_features = ma_trend(df['close'], short_ma, medium_ma, long_ma)
    for col in ma_features.columns:
        features[col] = ma_features[col]

    # Golden/Death cross
    gdc = golden_death_cross(df['close'], 50, 200)
    features['short_above_long'] = gdc['short_above_long']
    features['golden_cross'] = gdc['golden_cross']
    features['death_cross'] = gdc['death_cross']
    features['ma_spread'] = gdc['ma_spread']

    # MACD features
    macd_feat = macd_trend(df['close'])
    for col in ['macd', 'macd_signal', 'macd_histogram', 'macd_above_signal',
                'macd_above_zero', 'macd_hist_rising']:
        features[col] = macd_feat[col]

    # RSI features
    rsi_feat = rsi_trend(df['close'])
    for col in ['rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_rising']:
        features[col] = rsi_feat[col]

    # CCI features
    cci_feat = cci_trend(df['high'], df['low'], df['close'])
    features['cci'] = cci_feat['cci']
    features['cci_strong_bullish'] = cci_feat['cci_strong_bullish']
    features['cci_strong_bearish'] = cci_feat['cci_strong_bearish']

    # Trend consistency and duration
    features['trend_consistency'] = trend_consistency(df['close'], medium_ma)
    duration = trend_duration(df['close'], medium_ma)
    features['trend_duration'] = duration['trend_duration']

    # Donchian channel features
    donchian = donchian_trend(df['high'], df['low'], df['close'])
    features['channel_position'] = donchian['channel_position']
    features['new_high'] = donchian['new_high']
    features['new_low'] = donchian['new_low']

    return features
