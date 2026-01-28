"""
Volume-based features for market analysis.

Computes features derived from trading volume:
- On-Balance Volume (OBV)
- Volume-weighted metrics
- Buy/Sell pressure indicators
- Volume anomaly detection
- Accumulation/Distribution indicators
"""

import numpy as np
import pandas as pd
from typing import Optional


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV uses volume flow to predict changes in stock price.
    Volume is added on up days and subtracted on down days.

    Rising OBV indicates buying pressure.
    Falling OBV indicates selling pressure.

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        Series of OBV values
    """
    price_change = close.diff()

    # +volume for up days, -volume for down days, 0 for unchanged
    signed_volume = pd.Series(
        np.where(price_change > 0, volume,
                 np.where(price_change < 0, -volume, 0)),
        index=close.index
    )

    return signed_volume.cumsum()


def obv_ema(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate EMA of On-Balance Volume.

    Smoothed OBV for trend identification.

    Args:
        close: Close prices
        volume: Trading volume
        period: EMA period (default: 20)

    Returns:
        Series of OBV EMA values
    """
    obv = on_balance_volume(close, volume)
    return obv.ewm(span=period, min_periods=period, adjust=False).mean()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average of volume.

    Args:
        volume: Trading volume
        period: SMA period (default: 20)

    Returns:
        Series of volume SMA values
    """
    return volume.rolling(window=period, min_periods=period).mean()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate volume ratio (current volume / average volume).

    Values > 1: Above average volume
    Values < 1: Below average volume

    Useful for detecting unusual trading activity.

    Args:
        volume: Trading volume
        period: Average period (default: 20)

    Returns:
        Series of volume ratio values
    """
    avg_volume = volume_sma(volume, period)
    ratio = volume / avg_volume
    return ratio.replace([np.inf, -np.inf], np.nan)


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate relative volume (z-score of volume).

    Normalized volume for comparing across different periods.

    Args:
        volume: Trading volume
        period: Lookback period (default: 20)

    Returns:
        Series of z-score normalized volume
    """
    vol_mean = volume.rolling(window=period, min_periods=period).mean()
    vol_std = volume.rolling(window=period, min_periods=period).std()

    z_score = (volume - vol_mean) / vol_std
    return z_score.replace([np.inf, -np.inf], np.nan)


def volume_trend(volume: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate volume trend (percentage change over period).

    Positive: Volume increasing
    Negative: Volume decreasing

    Args:
        volume: Trading volume
        period: Trend period (default: 10)

    Returns:
        Series of volume trend values
    """
    return (volume - volume.shift(period)) / volume.shift(period)


def buy_sell_ratio(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """
    Calculate buy/sell volume ratio.

    Values > 1: More buying pressure
    Values < 1: More selling pressure
    Values = 1: Balanced

    Args:
        buy_volume: Taker buy volume
        sell_volume: Taker sell volume

    Returns:
        Series of buy/sell ratio values
    """
    # Avoid division by zero
    ratio = buy_volume / sell_volume.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def net_volume(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """
    Calculate net volume (buy - sell).

    Positive: Net buying
    Negative: Net selling

    Args:
        buy_volume: Taker buy volume
        sell_volume: Taker sell volume

    Returns:
        Series of net volume values
    """
    return buy_volume - sell_volume


def volume_delta_ratio(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    total_volume: pd.Series,
) -> pd.Series:
    """
    Calculate volume delta as ratio of total volume.

    (Buy - Sell) / Total

    Range: -1 to 1
    Positive: Net buying
    Negative: Net selling

    Args:
        buy_volume: Taker buy volume
        sell_volume: Taker sell volume
        total_volume: Total volume

    Returns:
        Series of delta ratio values
    """
    delta = buy_volume - sell_volume
    ratio = delta / total_volume
    return ratio.replace([np.inf, -np.inf], np.nan)


def cumulative_volume_delta(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Cumulative Volume Delta (CVD) over a rolling period.

    Running sum of (buy - sell) volume.
    Useful for identifying divergences with price.

    Args:
        buy_volume: Taker buy volume
        sell_volume: Taker sell volume
        period: Rolling window (default: 20)

    Returns:
        Series of CVD values
    """
    delta = buy_volume - sell_volume
    return delta.rolling(window=period, min_periods=1).sum()


def vwap_deviation(
    close: pd.Series,
    vwap: pd.Series,
) -> pd.Series:
    """
    Calculate deviation from VWAP.

    Price above VWAP: Bullish (buying at premium)
    Price below VWAP: Bearish (buying at discount)

    Args:
        close: Close prices
        vwap: Volume-weighted average price

    Returns:
        Series of VWAP deviation values (as percentage)
    """
    return (close - vwap) / vwap


def vwap_bands(
    close: pd.Series,
    vwap: pd.Series,
    std_dev: float = 1.0,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculate VWAP bands (standard deviation bands around VWAP).

    Similar to Bollinger Bands but centered on VWAP.

    Args:
        close: Close prices
        vwap: Volume-weighted average price
        std_dev: Standard deviation multiplier (default: 1.0)
        period: Lookback for std calculation (default: 20)

    Returns:
        DataFrame with upper, lower bands and %B
    """
    deviation = close - vwap
    rolling_std = deviation.rolling(window=period, min_periods=period).std()

    upper = vwap + (std_dev * rolling_std)
    lower = vwap - (std_dev * rolling_std)

    # Percent B (position within bands)
    percent_b = (close - lower) / (upper - lower)

    return pd.DataFrame({
        'vwap_upper': upper,
        'vwap_lower': lower,
        'vwap_percent_b': percent_b.replace([np.inf, -np.inf], np.nan),
    }, index=close.index)


def accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line.

    A/D = Cumulative sum of Money Flow Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = MFM * Volume

    Rising A/D: Accumulation (buying)
    Falling A/D: Distribution (selling)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume

    Returns:
        Series of A/D Line values
    """
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

    # Money Flow Volume
    mfv = mfm * volume

    # Cumulative A/D Line
    return mfv.cumsum()


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Chaikin Money Flow (CMF).

    CMF measures buying and selling pressure over a period.
    Range: -1 to 1

    CMF > 0: Buying pressure (accumulation)
    CMF < 0: Selling pressure (distribution)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        period: Lookback period (default: 20)

    Returns:
        Series of CMF values
    """
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

    # Money Flow Volume
    mfv = mfm * volume

    # CMF = Sum(MFV) / Sum(Volume)
    cmf = mfv.rolling(window=period, min_periods=period).sum() / \
          volume.rolling(window=period, min_periods=period).sum()

    return cmf.replace([np.inf, -np.inf], np.nan)


def ease_of_movement(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Ease of Movement (EMV).

    Measures the relationship between price change and volume.
    High EMV: Price moves easily with low volume
    Low EMV: Price requires high volume to move

    Args:
        high: High prices
        low: Low prices
        volume: Trading volume
        period: Smoothing period (default: 14)

    Returns:
        Series of smoothed EMV values
    """
    distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / 1e8) / (high - low)  # Scaled for reasonable values

    emv = distance / box_ratio
    emv = emv.replace([np.inf, -np.inf], 0)

    # Smooth the EMV
    return emv.rolling(window=period, min_periods=period).mean()


def force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """
    Calculate Force Index.

    Force Index = (Current Close - Previous Close) * Volume

    Measures the force (strength) of bulls and bears.
    Positive: Bullish force
    Negative: Bearish force

    Args:
        close: Close prices
        volume: Trading volume
        period: EMA smoothing period (default: 13)

    Returns:
        Series of smoothed Force Index values
    """
    force = close.diff() * volume
    return force.ewm(span=period, min_periods=period, adjust=False).mean()


def negative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Negative Volume Index (NVI).

    NVI only changes on days when volume decreases.
    Theory: Smart money acts on low volume days.

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        Series of NVI values
    """
    nvi = pd.Series(1000.0, index=close.index)  # Start at 1000
    returns = close.pct_change()
    volume_decreased = volume < volume.shift(1)

    for i in range(1, len(nvi)):
        if volume_decreased.iloc[i]:
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + returns.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]

    return nvi


def positive_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Positive Volume Index (PVI).

    PVI only changes on days when volume increases.
    Theory: Crowd follows price on high volume days.

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        Series of PVI values
    """
    pvi = pd.Series(1000.0, index=close.index)  # Start at 1000
    returns = close.pct_change()
    volume_increased = volume > volume.shift(1)

    for i in range(1, len(pvi)):
        if volume_increased.iloc[i]:
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + returns.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]

    return pvi


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).

    VPT = Previous VPT + Volume * (Close Change % )

    Similar to OBV but uses percentage change instead of direction.

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        Series of VPT values
    """
    pct_change = close.pct_change()
    vpt = (volume * pct_change).cumsum()
    return vpt


def klinger_volume_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
) -> pd.DataFrame:
    """
    Calculate Klinger Volume Oscillator.

    Combines price, volume, and direction to measure money flow.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        fast_period: Fast EMA period (default: 34)
        slow_period: Slow EMA period (default: 55)
        signal_period: Signal line period (default: 13)

    Returns:
        DataFrame with KVO and signal line
    """
    # Trend direction
    hlc3 = (high + low + close) / 3
    trend = pd.Series(
        np.where(hlc3 > hlc3.shift(1), 1, -1),
        index=close.index
    )

    # dm = High - Low
    dm = high - low

    # Cumulative volume
    cv = pd.Series(0.0, index=close.index)

    for i in range(1, len(cv)):
        if trend.iloc[i] == trend.iloc[i-1]:
            cv.iloc[i] = cv.iloc[i-1] + volume.iloc[i]
        else:
            cv.iloc[i] = volume.iloc[i]

    # Volume force
    vf = volume * abs(2 * (dm / hlc3) - 1) * trend * 100

    # KVO
    kvo = vf.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean() - \
          vf.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()

    # Signal line
    signal = kvo.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()

    return pd.DataFrame({
        'kvo': kvo,
        'kvo_signal': signal,
        'kvo_histogram': kvo - signal,
    }, index=close.index)


def compute_volume_features(
    df: pd.DataFrame,
    short_window: int = 14,
    medium_window: int = 50,
) -> pd.DataFrame:
    """
    Compute all volume-based features for an OHLCV DataFrame.

    Expects DataFrame with columns:
    - open, high, low, close, volume
    - buy_volume, sell_volume (if available)
    - vwap (if available)

    Args:
        df: DataFrame with OHLCV data
        short_window: Short-term lookback (default: 14)
        medium_window: Medium-term lookback (default: 50)

    Returns:
        DataFrame with all volume features
    """
    features = pd.DataFrame(index=df.index)

    # Basic volume metrics
    features['volume_sma_short'] = volume_sma(df['volume'], short_window)
    features['volume_sma_medium'] = volume_sma(df['volume'], medium_window)
    features['volume_ratio'] = volume_ratio(df['volume'], medium_window)
    features['relative_volume'] = relative_volume(df['volume'], medium_window)
    features['volume_trend'] = volume_trend(df['volume'], short_window)

    # On-Balance Volume
    features['obv'] = on_balance_volume(df['close'], df['volume'])
    features['obv_ema'] = obv_ema(df['close'], df['volume'], medium_window)

    # A/D and CMF
    features['ad_line'] = accumulation_distribution(
        df['high'], df['low'], df['close'], df['volume']
    )
    features['cmf'] = chaikin_money_flow(
        df['high'], df['low'], df['close'], df['volume'], medium_window
    )

    # Force Index
    features['force_index'] = force_index(df['close'], df['volume'], short_window)

    # Volume Price Trend
    features['vpt'] = volume_price_trend(df['close'], df['volume'])

    # Ease of Movement
    features['emv'] = ease_of_movement(df['high'], df['low'], df['volume'], short_window)

    # Buy/Sell volume features (if available)
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        features['buy_sell_ratio'] = buy_sell_ratio(df['buy_volume'], df['sell_volume'])
        features['net_volume'] = net_volume(df['buy_volume'], df['sell_volume'])
        features['volume_delta_ratio'] = volume_delta_ratio(
            df['buy_volume'], df['sell_volume'], df['volume']
        )
        features['cvd'] = cumulative_volume_delta(
            df['buy_volume'], df['sell_volume'], medium_window
        )

    # VWAP features (if available)
    if 'vwap' in df.columns:
        features['vwap_deviation'] = vwap_deviation(df['close'], df['vwap'])
        vwap_band_features = vwap_bands(df['close'], df['vwap'], period=medium_window)
        features['vwap_percent_b'] = vwap_band_features['vwap_percent_b']

    # Klinger Oscillator
    kvo = klinger_volume_oscillator(
        df['high'], df['low'], df['close'], df['volume']
    )
    features['kvo'] = kvo['kvo']
    features['kvo_signal'] = kvo['kvo_signal']

    return features
