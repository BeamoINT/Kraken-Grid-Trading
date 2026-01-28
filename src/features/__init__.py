"""
Feature Engineering Module.

Provides technical indicators and derived features for market analysis
and machine learning model training.

Modules:
- indicators: Core technical indicators (ATR, ADX, Bollinger Bands, RSI, MACD, etc.)
- price_features: Price-based features (returns, momentum, moving averages)
- volume_features: Volume-based features (OBV, buy/sell pressure, VWAP)
- volatility_features: Volatility features (ATR-based, historical vol, squeeze)
- trend_features: Trend features (ADX-based, MA crossovers, MACD signals)
- feature_pipeline: Main pipeline for computing all features

Usage:
    from src.features import FeaturePipeline

    pipeline = FeaturePipeline(ohlcv_path, features_path)
    features = pipeline.compute_features("XBTUSD", ["5m", "1h"])

    # Or use individual indicator functions
    from src.features.indicators import atr, adx, rsi
    atr_values = atr(high, low, close, period=14)
"""

from .feature_pipeline import FeaturePipeline

from .indicators import (
    # Volatility
    true_range,
    atr,
    # Trend
    directional_movement,
    adx,
    # Bollinger Bands
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_percent_b,
    # Momentum
    rsi,
    macd,
    stochastic_oscillator,
    cci,
    williams_r,
    mfi,
    # Channels
    keltner_channels,
    donchian_channels,
    # Ichimoku
    ichimoku_cloud,
)

from .price_features import (
    log_returns,
    simple_returns,
    cumulative_returns,
    sma,
    ema,
    wma,
    hull_ma,
    price_momentum,
    price_acceleration,
    ma_crossover_signal,
    ma_distance,
    multi_ma_distances,
    price_position_in_range,
    pivot_points,
    gap_analysis,
    higher_highs_lower_lows,
    swing_points,
    compute_price_features,
)

from .volume_features import (
    on_balance_volume,
    obv_ema,
    volume_sma,
    volume_ratio,
    relative_volume,
    volume_trend,
    buy_sell_ratio,
    net_volume,
    volume_delta_ratio,
    cumulative_volume_delta,
    vwap_deviation,
    vwap_bands,
    accumulation_distribution,
    chaikin_money_flow,
    ease_of_movement,
    force_index,
    volume_price_trend,
    klinger_volume_oscillator,
    compute_volume_features,
)

from .volatility_features import (
    normalized_atr,
    atr_ratio,
    atr_percentile,
    historical_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
    volatility_percentile,
    volatility_regime,
    volatility_clustering,
    bollinger_squeeze,
    intraday_volatility_ratio,
    volatility_breakout,
    compute_volatility_features,
)

from .trend_features import (
    trend_strength,
    trend_direction,
    di_difference,
    adx_slope,
    ma_trend,
    golden_death_cross,
    macd_trend,
    rsi_trend,
    cci_trend,
    trend_consistency,
    trend_duration,
    donchian_trend,
    ichimoku_trend,
    compute_trend_features,
)

__all__ = [
    # Main pipeline
    "FeaturePipeline",
    # Core indicators
    "true_range",
    "atr",
    "directional_movement",
    "adx",
    "bollinger_bands",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "rsi",
    "macd",
    "stochastic_oscillator",
    "cci",
    "williams_r",
    "mfi",
    "keltner_channels",
    "donchian_channels",
    "ichimoku_cloud",
    # Price features
    "log_returns",
    "simple_returns",
    "cumulative_returns",
    "sma",
    "ema",
    "wma",
    "hull_ma",
    "price_momentum",
    "price_acceleration",
    "ma_crossover_signal",
    "ma_distance",
    "multi_ma_distances",
    "price_position_in_range",
    "pivot_points",
    "gap_analysis",
    "higher_highs_lower_lows",
    "swing_points",
    "compute_price_features",
    # Volume features
    "on_balance_volume",
    "obv_ema",
    "volume_sma",
    "volume_ratio",
    "relative_volume",
    "volume_trend",
    "buy_sell_ratio",
    "net_volume",
    "volume_delta_ratio",
    "cumulative_volume_delta",
    "vwap_deviation",
    "vwap_bands",
    "accumulation_distribution",
    "chaikin_money_flow",
    "ease_of_movement",
    "force_index",
    "volume_price_trend",
    "klinger_volume_oscillator",
    "compute_volume_features",
    # Volatility features
    "normalized_atr",
    "atr_ratio",
    "atr_percentile",
    "historical_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "yang_zhang_volatility",
    "volatility_percentile",
    "volatility_regime",
    "volatility_clustering",
    "bollinger_squeeze",
    "intraday_volatility_ratio",
    "volatility_breakout",
    "compute_volatility_features",
    # Trend features
    "trend_strength",
    "trend_direction",
    "di_difference",
    "adx_slope",
    "ma_trend",
    "golden_death_cross",
    "macd_trend",
    "rsi_trend",
    "cci_trend",
    "trend_consistency",
    "trend_duration",
    "donchian_trend",
    "ichimoku_trend",
    "compute_trend_features",
]
