"""
Volatility-based features for market analysis.

Computes features related to market volatility:
- ATR-based volatility measures
- Bollinger Band metrics
- Historical volatility
- Volatility percentiles and regimes
- Volatility clustering
"""

import numpy as np
import pandas as pd
from typing import Tuple

from .indicators import (
    atr,
    true_range,
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_percent_b,
    keltner_channels,
)


def normalized_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Normalized ATR (NATR).

    NATR = (ATR / Close) * 100

    Expresses ATR as a percentage of price, allowing comparison
    across different price levels and assets.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        Series of NATR values (as percentage)
    """
    atr_values = atr(high, low, close, period)
    return (atr_values / close) * 100


def atr_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int = 7,
    long_period: int = 21,
) -> pd.Series:
    """
    Calculate ATR ratio (short ATR / long ATR).

    Values > 1: Volatility expanding
    Values < 1: Volatility contracting

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        short_period: Short ATR period (default: 7)
        long_period: Long ATR period (default: 21)

    Returns:
        Series of ATR ratio values
    """
    atr_short = atr(high, low, close, short_period)
    atr_long = atr(high, low, close, long_period)

    ratio = atr_short / atr_long
    return ratio.replace([np.inf, -np.inf], np.nan)


def atr_percentile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    lookback: int = 100,
) -> pd.Series:
    """
    Calculate ATR percentile rank.

    Indicates how current volatility compares to recent history.

    Values near 100: Very high volatility
    Values near 0: Very low volatility

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_period: ATR period (default: 14)
        lookback: Lookback for percentile calculation (default: 100)

    Returns:
        Series of percentile values (0-100)
    """
    atr_values = atr(high, low, close, atr_period)

    def rolling_percentile(x):
        if len(x) < lookback:
            return np.nan
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100

    return atr_values.rolling(window=lookback, min_periods=lookback).apply(
        rolling_percentile, raw=False
    )


def historical_volatility(
    close: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pd.Series:
    """
    Calculate Historical Volatility (realized volatility).

    Based on standard deviation of log returns.

    Args:
        close: Close prices
        period: Lookback period (default: 20)
        annualize: Whether to annualize (default: True)
        trading_periods: Trading periods per year (default: 252)

    Returns:
        Series of historical volatility values
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=period, min_periods=period).std()

    if annualize:
        vol = vol * np.sqrt(trading_periods)

    return vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pd.Series:
    """
    Calculate Parkinson Volatility.

    Uses high-low range instead of close prices.
    More efficient estimator when there are no gaps.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default: 20)
        annualize: Whether to annualize (default: True)
        trading_periods: Trading periods per year (default: 252)

    Returns:
        Series of Parkinson volatility values
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))

    vol = np.sqrt(factor * (log_hl ** 2).rolling(window=period, min_periods=period).mean())

    if annualize:
        vol = vol * np.sqrt(trading_periods)

    return vol


def garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pd.Series:
    """
    Calculate Garman-Klass Volatility.

    Uses OHLC data for a more efficient volatility estimate.
    Accounts for overnight gaps.

    Args:
        open_price: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 20)
        annualize: Whether to annualize (default: True)
        trading_periods: Trading periods per year (default: 252)

    Returns:
        Series of Garman-Klass volatility values
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_price) ** 2

    # Garman-Klass formula
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    vol = np.sqrt(gk.rolling(window=period, min_periods=period).mean())

    if annualize:
        vol = vol * np.sqrt(trading_periods)

    return vol


def yang_zhang_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pd.Series:
    """
    Calculate Yang-Zhang Volatility.

    Combines overnight and intraday volatility.
    Most efficient when there are gaps (crypto markets).

    Args:
        open_price: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 20)
        annualize: Whether to annualize (default: True)
        trading_periods: Trading periods per year (default: 252)

    Returns:
        Series of Yang-Zhang volatility values
    """
    # Overnight return (close to open)
    log_oc = np.log(open_price / close.shift(1))

    # Open to close return
    log_oco = np.log(close / open_price)

    # Rogers-Satchell component
    log_ho = np.log(high / open_price)
    log_lo = np.log(low / open_price)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)

    rs = log_ho * log_hc + log_lo * log_lc

    # Variance components
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    var_overnight = log_oc.rolling(window=period, min_periods=period).var()
    var_open_close = log_oco.rolling(window=period, min_periods=period).var()
    var_rs = rs.rolling(window=period, min_periods=period).mean()

    # Yang-Zhang variance
    yz_var = var_overnight + k * var_open_close + (1 - k) * var_rs

    vol = np.sqrt(yz_var)

    if annualize:
        vol = vol * np.sqrt(trading_periods)

    return vol


def volatility_percentile(
    close: pd.Series,
    vol_period: int = 20,
    lookback: int = 100,
) -> pd.Series:
    """
    Calculate historical volatility percentile.

    Args:
        close: Close prices
        vol_period: Volatility calculation period (default: 20)
        lookback: Lookback for percentile (default: 100)

    Returns:
        Series of volatility percentile values (0-100)
    """
    vol = historical_volatility(close, vol_period, annualize=False)

    def rolling_percentile(x):
        if len(x) < lookback:
            return np.nan
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100

    return vol.rolling(window=lookback, min_periods=lookback).apply(
        rolling_percentile, raw=False
    )


def volatility_regime(
    close: pd.Series,
    vol_period: int = 20,
    lookback: int = 100,
    low_threshold: float = 25.0,
    high_threshold: float = 75.0,
) -> pd.Series:
    """
    Classify volatility into regimes.

    Returns:
    - -1: Low volatility
    -  0: Normal volatility
    -  1: High volatility

    Args:
        close: Close prices
        vol_period: Volatility period (default: 20)
        lookback: Lookback for percentile (default: 100)
        low_threshold: Low vol threshold (default: 25th percentile)
        high_threshold: High vol threshold (default: 75th percentile)

    Returns:
        Series of regime values (-1, 0, 1)
    """
    vol_pct = volatility_percentile(close, vol_period, lookback)

    regime = pd.Series(0, index=close.index)
    regime = regime.where(vol_pct >= low_threshold, -1)
    regime = regime.where(vol_pct <= high_threshold, 1)

    return regime


def volatility_clustering(
    close: pd.Series,
    period: int = 20,
    threshold: float = 1.5,
) -> pd.Series:
    """
    Detect volatility clustering.

    Returns the ratio of current volatility to average volatility
    in consecutive high-volatility periods.

    Args:
        close: Close prices
        period: Volatility period (default: 20)
        threshold: Threshold for high volatility (default: 1.5x average)

    Returns:
        Series indicating cluster intensity
    """
    vol = historical_volatility(close, period, annualize=False)
    avg_vol = vol.rolling(window=period * 5, min_periods=period).mean()

    vol_ratio = vol / avg_vol
    is_high_vol = vol_ratio > threshold

    # Count consecutive high volatility periods
    cluster = is_high_vol.astype(int)
    cluster_cumsum = cluster.cumsum()
    reset_points = cluster_cumsum.where(~is_high_vol).ffill().fillna(0)
    consecutive = cluster_cumsum - reset_points

    return consecutive


def bollinger_squeeze(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_period: int = 10,
    kc_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Detect Bollinger Band / Keltner Channel squeeze.

    A squeeze occurs when Bollinger Bands move inside Keltner Channels,
    indicating low volatility that often precedes a breakout.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Bollinger Bands std dev (default: 2.0)
        kc_period: Keltner Channel EMA period (default: 20)
        kc_atr_period: Keltner Channel ATR period (default: 10)
        kc_multiplier: Keltner Channel multiplier (default: 1.5)

    Returns:
        DataFrame with squeeze signals and momentum
    """
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, bb_period, bb_std)

    # Keltner Channels
    kc_upper, kc_middle, kc_lower = keltner_channels(
        high, low, close, kc_period, kc_atr_period, kc_multiplier
    )

    # Squeeze: BB inside KC
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Momentum (using MACD-like calculation)
    highest = high.rolling(window=bb_period, min_periods=bb_period).max()
    lowest = low.rolling(window=bb_period, min_periods=bb_period).min()
    midline = (highest + lowest) / 2 + close.ewm(span=bb_period, min_periods=bb_period, adjust=False).mean()
    midline = midline / 2

    momentum = close - midline

    # Squeeze momentum direction
    squeeze_momentum = momentum.diff()

    return pd.DataFrame({
        'squeeze_on': squeeze_on.astype(int),
        'squeeze_momentum': momentum,
        'squeeze_direction': np.sign(squeeze_momentum).fillna(0).astype(int),
    }, index=close.index)


def intraday_volatility_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate intraday volatility ratio.

    Ratio of true range to close-to-close change.
    High values: Intraday volatility not reflected in close prices
    Low values: Close prices capture most of the movement

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Smoothing period (default: 14)

    Returns:
        Series of ratio values
    """
    tr = true_range(high, low, close)
    close_change = close.diff().abs()

    # Avoid division by zero
    close_change = close_change.replace(0, np.nan)

    ratio = tr / close_change
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # Smooth the ratio
    return ratio.rolling(window=period, min_periods=period).mean()


def volatility_breakout(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Detect volatility breakouts.

    Breakout occurs when price moves more than ATR * multiplier.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_period: ATR period (default: 14)
        multiplier: ATR multiplier for breakout (default: 2.0)

    Returns:
        DataFrame with breakout signals
    """
    atr_values = atr(high, low, close, atr_period)
    threshold = atr_values * multiplier

    # High breakout: price moved up more than threshold
    price_change = close.diff()
    breakout_up = price_change > threshold
    breakout_down = price_change < -threshold

    # Magnitude of breakout (how many ATRs)
    breakout_magnitude = (price_change.abs() / atr_values).replace([np.inf, -np.inf], 0)

    return pd.DataFrame({
        'vol_breakout_up': breakout_up.astype(int),
        'vol_breakout_down': breakout_down.astype(int),
        'breakout_magnitude': breakout_magnitude,
    }, index=close.index)


def compute_volatility_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    bb_period: int = 20,
    vol_period: int = 20,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Compute all volatility-based features for an OHLCV DataFrame.

    Args:
        df: DataFrame with open, high, low, close columns
        atr_period: ATR period (default: 14)
        bb_period: Bollinger Bands period (default: 20)
        vol_period: Historical volatility period (default: 20)
        lookback: Lookback for percentiles (default: 100)

    Returns:
        DataFrame with all volatility features
    """
    features = pd.DataFrame(index=df.index)

    # ATR-based features
    features['atr'] = atr(df['high'], df['low'], df['close'], atr_period)
    features['natr'] = normalized_atr(df['high'], df['low'], df['close'], atr_period)
    features['atr_ratio'] = atr_ratio(df['high'], df['low'], df['close'])
    features['atr_percentile'] = atr_percentile(
        df['high'], df['low'], df['close'], atr_period, lookback
    )

    # Historical volatility
    features['hist_vol'] = historical_volatility(df['close'], vol_period)
    features['parkinson_vol'] = parkinson_volatility(df['high'], df['low'], vol_period)
    features['gk_vol'] = garman_klass_volatility(
        df['open'], df['high'], df['low'], df['close'], vol_period
    )
    features['vol_percentile'] = volatility_percentile(df['close'], vol_period, lookback)
    features['vol_regime'] = volatility_regime(df['close'], vol_period, lookback)

    # Bollinger Band features
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'], bb_period)
    features['bb_bandwidth'] = bollinger_bandwidth(df['close'], bb_period)
    features['bb_percent_b'] = bollinger_percent_b(df['close'], bb_period)
    features['bb_upper_dist'] = (df['close'] - bb_upper) / df['close']
    features['bb_lower_dist'] = (df['close'] - bb_lower) / df['close']

    # Squeeze detection
    squeeze = bollinger_squeeze(df['close'], df['high'], df['low'])
    features['squeeze_on'] = squeeze['squeeze_on']
    features['squeeze_momentum'] = squeeze['squeeze_momentum']

    # Volatility clustering
    features['vol_clustering'] = volatility_clustering(df['close'], vol_period)

    # Intraday volatility
    features['intraday_vol_ratio'] = intraday_volatility_ratio(
        df['high'], df['low'], df['close'], atr_period
    )

    # Breakout detection
    breakout = volatility_breakout(df['high'], df['low'], df['close'], atr_period)
    features['vol_breakout_up'] = breakout['vol_breakout_up']
    features['vol_breakout_down'] = breakout['vol_breakout_down']
    features['breakout_magnitude'] = breakout['breakout_magnitude']

    return features
