"""
Core technical indicators for market analysis.

Implements the fundamental indicators used for feature engineering:
- ATR (Average True Range) - Volatility measurement
- ADX (Average Directional Index) - Trend strength
- Bollinger Bands - Volatility bands
- RSI (Relative Strength Index) - Momentum oscillator
- MACD (Moving Average Convergence Divergence) - Trend momentum
- Stochastic Oscillator - Momentum indicator
- CCI (Commodity Channel Index) - Mean reversion indicator

All functions are designed to work with pandas DataFrames containing OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Calculate True Range - the foundation for ATR.

    True Range is the maximum of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Series of True Range values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 14)

    Returns:
        Series of ATR values
    """
    tr = true_range(high, low, close)

    # Use exponential moving average (Wilder's smoothing)
    # Wilder's smoothing factor = 1/period
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def directional_movement(
    high: pd.Series,
    low: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Positive and Negative Directional Movement (+DM, -DM).

    +DM: Current High - Previous High (if positive and > |Current Low - Previous Low|)
    -DM: Previous Low - Current Low (if positive and > Current High - Previous High)

    Args:
        high: High prices
        low: Low prices

    Returns:
        Tuple of (+DM, -DM) Series
    """
    high_diff = high.diff()
    low_diff = -low.diff()  # Inverted because we want (prev_low - current_low)

    # +DM: high moved up more than low moved down
    plus_dm = pd.Series(np.where(
        (high_diff > low_diff) & (high_diff > 0),
        high_diff,
        0.0
    ), index=high.index)

    # -DM: low moved down more than high moved up
    minus_dm = pd.Series(np.where(
        (low_diff > high_diff) & (low_diff > 0),
        low_diff,
        0.0
    ), index=low.index)

    return plus_dm, minus_dm


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI.

    ADX measures trend strength on a scale of 0-100:
    - ADX < 20: Weak trend or ranging
    - ADX 20-40: Developing trend
    - ADX 40-60: Strong trend
    - ADX > 60: Very strong trend

    +DI > -DI: Uptrend
    -DI > +DI: Downtrend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 14)

    Returns:
        Tuple of (ADX, +DI, -DI) Series
    """
    # Calculate True Range
    tr = true_range(high, low, close)

    # Calculate Directional Movement
    plus_dm, minus_dm = directional_movement(high, low)

    # Smooth with Wilder's EMA
    atr_smooth = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth

    # Handle division by zero
    plus_di = plus_di.replace([np.inf, -np.inf], 0).fillna(0)
    minus_di = minus_di.replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate DX (Directional Movement Index)
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    dx = 100 * di_diff / di_sum
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)

    # ADX is smoothed DX
    adx_values = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return adx_values, plus_di, minus_di


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of:
    - Middle Band: SMA of close prices
    - Upper Band: Middle + (std_dev * standard deviation)
    - Lower Band: Middle - (std_dev * standard deviation)

    Args:
        close: Close prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band) Series
    """
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return upper, middle, lower


def bollinger_bandwidth(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """
    Calculate Bollinger Bandwidth.

    Bandwidth = (Upper Band - Lower Band) / Middle Band

    Useful for detecting volatility squeezes (low values)
    and breakouts (expanding values).

    Args:
        close: Close prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Series of bandwidth values
    """
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    bandwidth = (upper - lower) / middle
    return bandwidth


def bollinger_percent_b(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """
    Calculate Bollinger %B indicator.

    %B = (Close - Lower Band) / (Upper Band - Lower Band)

    Interpretation:
    - %B > 1: Price above upper band (overbought)
    - %B < 0: Price below lower band (oversold)
    - %B = 0.5: Price at middle band

    Args:
        close: Close prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Series of %B values
    """
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    percent_b = (close - lower) / (upper - lower)
    return percent_b.replace([np.inf, -np.inf], np.nan)


def rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes:
    - RSI > 70: Overbought
    - RSI < 30: Oversold

    Uses Wilder's smoothing method.

    Args:
        close: Close prices
        period: Lookback period (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Use Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    # Handle edge cases
    rsi_values = rsi_values.replace([np.inf, -np.inf], np.nan)

    return rsi_values


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD consists of:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line

    Trading signals:
    - MACD crosses above signal: Bullish
    - MACD crosses below signal: Bearish
    - Histogram growing: Trend strengthening

    Args:
        close: Close prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (MACD Line, Signal Line, Histogram) Series
    """
    fast_ema = close.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).

    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA of %K (signal line)

    Interpretation:
    - %K > 80: Overbought
    - %K < 20: Oversold
    - %K crosses above %D: Bullish
    - %K crosses below %D: Bearish

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)

    Returns:
        Tuple of (%K, %D) Series
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k_values = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_values = k_values.replace([np.inf, -np.inf], np.nan)

    d_values = k_values.rolling(window=d_period, min_periods=d_period).mean()

    return k_values, d_values


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).

    CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
    Typical Price = (High + Low + Close) / 3

    Interpretation:
    - CCI > 100: Overbought / Strong uptrend
    - CCI < -100: Oversold / Strong downtrend
    - CCI between -100 and 100: Ranging

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 20)

    Returns:
        Series of CCI values
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period, min_periods=period).mean()

    # Mean deviation (not standard deviation)
    mean_dev = typical_price.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(),
        raw=True
    )

    cci_values = (typical_price - sma) / (0.015 * mean_dev)
    return cci_values.replace([np.inf, -np.inf], np.nan)


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Williams %R indicator.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Interpretation:
    - %R > -20: Overbought
    - %R < -80: Oversold

    Note: Williams %R is similar to Stochastic %K but inverted and uses -100 to 0 scale.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 14)

    Returns:
        Series of Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.replace([np.inf, -np.inf], np.nan)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).

    MFI is a volume-weighted RSI:
    - Typical Price = (High + Low + Close) / 3
    - Money Flow = Typical Price * Volume
    - Positive/Negative flow based on typical price direction

    Interpretation:
    - MFI > 80: Overbought
    - MFI < 20: Oversold

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        period: Lookback period (default: 14)

    Returns:
        Series of MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    # Determine positive/negative flow
    tp_change = typical_price.diff()

    positive_flow = money_flow.where(tp_change > 0, 0.0)
    negative_flow = money_flow.where(tp_change < 0, 0.0)

    # Rolling sums
    positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()

    # Money Flow Ratio and Index
    mf_ratio = positive_mf / negative_mf
    mfi_values = 100 - (100 / (1 + mf_ratio))

    return mfi_values.replace([np.inf, -np.inf], np.nan)


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.

    Similar to Bollinger Bands but uses ATR instead of standard deviation:
    - Middle: EMA of close
    - Upper: Middle + (ATR * multiplier)
    - Lower: Middle - (ATR * multiplier)

    Often used with Bollinger Bands for "squeeze" detection.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period for middle line (default: 20)
        atr_period: ATR period (default: 10)
        atr_multiplier: ATR multiplier for bands (default: 2.0)

    Returns:
        Tuple of (Upper, Middle, Lower) Series
    """
    middle = close.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean()
    atr_values = atr(high, low, close, atr_period)

    upper = middle + (atr_multiplier * atr_values)
    lower = middle - (atr_multiplier * atr_values)

    return upper, middle, lower


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels.

    Based on highest high and lowest low over a period:
    - Upper: Highest high
    - Lower: Lowest low
    - Middle: (Upper + Lower) / 2

    Used in turtle trading and breakout strategies.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default: 20)

    Returns:
        Tuple of (Upper, Middle, Lower) Series
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2

    return upper, middle, lower


def ichimoku_cloud(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict:
    """
    Calculate Ichimoku Cloud components.

    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A: (Tenkan + Kijun) / 2, shifted forward
    - Senkou Span B: (52-period high + 52-period low) / 2, shifted forward
    - Chikou Span: Current close, shifted backward

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen period (default: 9)
        kijun_period: Kijun-sen period (default: 26)
        senkou_b_period: Senkou Span B period (default: 52)

    Returns:
        Dict with all Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A) - shifted forward by kijun_period
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B) - shifted forward by kijun_period
    senkou_b_high = high.rolling(window=senkou_b_period, min_periods=senkou_b_period).max()
    senkou_b_low = low.rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
    senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span) - shifted backward by kijun_period
    chikou = close.shift(-kijun_period)

    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou,
    }
