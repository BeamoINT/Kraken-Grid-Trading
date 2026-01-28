"""
Market Regime Detection and Labeling Module.

Provides tools for classifying market conditions into distinct regimes
for adaptive trading strategies.

Regime Types:
- RANGING: Low trend strength, price oscillating in range
- TRENDING_UP: Strong uptrend with directional momentum
- TRENDING_DOWN: Strong downtrend with directional momentum
- HIGH_VOLATILITY: Elevated volatility without clear direction
- BREAKOUT: Sharp price movement with volume spike

Usage:
    from src.regime import RegimeLabeler, MarketRegime

    labeler = RegimeLabeler(adx_trending_threshold=25.0)
    labeled_df = labeler.label_regimes(features_df)

    # Or use the pipeline
    from src.regime import RegimeLabelPipeline

    pipeline = RegimeLabelPipeline(features_path, labels_path)
    results = pipeline.label_and_save("XBTUSD", ["5m", "1h"])
"""

from .regime_labeler import (
    MarketRegime,
    RegimeLabeler,
    RegimeLabelPipeline,
)

__all__ = [
    "MarketRegime",
    "RegimeLabeler",
    "RegimeLabelPipeline",
]
