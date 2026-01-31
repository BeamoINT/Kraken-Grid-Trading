#!/usr/bin/env python3
"""
Feature Computation CLI.

Computes technical indicators and derived features from OHLCV data.
Also handles regime labeling for ML training.

Usage:
    # Compute features for all timeframes
    python scripts/compute_features.py --pairs XBTUSD

    # Compute for specific timeframes
    python scripts/compute_features.py --pairs XBTUSD --timeframes 5m 1h

    # Also label regimes
    python scripts/compute_features.py --pairs XBTUSD --label-regimes

    # Show summary
    python scripts/compute_features.py --pairs XBTUSD --summary
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_labeler import RegimeLabelPipeline, RegimeLabeler, MarketRegime
from src.regime.outcome_labeler import OutcomeBasedLabeler, OutcomeBasedLabelPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_features(
    pairs: List[str],
    timeframes: Optional[List[str]],
    ohlcv_path: Path,
    features_path: Path,
    short_window: int = 14,
    medium_window: int = 50,
    long_window: int = 200,
) -> dict:
    """
    Compute features for trading pairs.

    Args:
        pairs: List of trading pairs
        timeframes: List of timeframes (None for all)
        ohlcv_path: Path to OHLCV data
        features_path: Path for feature output
        short_window: Short lookback window
        medium_window: Medium lookback window
        long_window: Long lookback window

    Returns:
        Dict with computation results
    """
    pipeline = FeaturePipeline(
        ohlcv_path=ohlcv_path,
        features_path=features_path,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
    )

    results = {}

    for pair in pairs:
        logger.info(f"Computing features for {pair}...")

        try:
            pair_results = pipeline.compute_features(
                pair=pair,
                timeframes=timeframes,
                save=True,
            )

            results[pair] = {
                "status": "success",
                "timeframes": {
                    tf: {
                        "rows": len(df),
                        "features": len(df.columns),
                    }
                    for tf, df in pair_results.items()
                },
            }

            logger.info(f"  Completed {pair}:")
            for tf, df in pair_results.items():
                logger.info(f"    {tf}: {len(df)} rows, {len(df.columns)} features")

        except Exception as e:
            logger.error(f"Error computing features for {pair}: {e}")
            results[pair] = {"status": "error", "error": str(e)}

    return results


def label_regimes(
    pairs: List[str],
    timeframes: Optional[List[str]],
    features_path: Path,
    labels_path: Path,
    ohlcv_path: Path,
    labeling_mode: str = "outcome",
    adx_threshold: float = 25.0,
    high_vol_percentile: float = 80.0,
    outcome_lookahead: int = 20,
    outcome_trend_threshold: float = 0.02,
) -> dict:
    """
    Label features with market regimes.

    Args:
        pairs: List of trading pairs
        timeframes: List of timeframes (None for all)
        features_path: Path to computed features
        labels_path: Path for labeled data output
        ohlcv_path: Path to OHLCV data (for close prices in outcome mode)
        labeling_mode: "outcome" (default, for ML training) or "indicator" (for inference)
        adx_threshold: ADX threshold for trending (indicator mode)
        high_vol_percentile: Percentile for high volatility
        outcome_lookahead: Periods to look ahead (outcome mode)
        outcome_trend_threshold: Return threshold for trending (outcome mode)

    Returns:
        Dict with labeling results
    """
    results = {}

    if labeling_mode == "outcome":
        logger.info("Using OUTCOME-based labeling (recommended for ML training)")
        labeler = OutcomeBasedLabeler(
            lookahead=outcome_lookahead,
            trend_threshold=outcome_trend_threshold,
            vol_percentile=high_vol_percentile,
        )
        pipeline = OutcomeBasedLabelPipeline(
            features_path=features_path,
            labels_path=labels_path,
            labeler=labeler,
        )

        for pair in pairs:
            logger.info(f"Labeling regimes for {pair}...")
            results[pair] = {}

            # Get available timeframes
            if timeframes is None:
                available_tfs = []
                for tf_dir in features_path.iterdir():
                    if tf_dir.is_dir() and (tf_dir / pair / "features.parquet").exists():
                        available_tfs.append(tf_dir.name)
                tfs = available_tfs
            else:
                tfs = timeframes

            for tf in tfs:
                try:
                    tf_result = pipeline.label_and_save(
                        pair=pair,
                        timeframe=tf,
                        ohlcv_path=ohlcv_path,
                    )
                    results[pair][tf] = tf_result

                    if "error" in tf_result:
                        logger.error(f"  {tf}: {tf_result['error']}")
                    else:
                        stats = tf_result.get("stats", {})
                        logger.info(f"  {tf}: {tf_result['rows']} rows (dropped {tf_result.get('dropped_nan', 0)} NaN)")
                        logger.info(f"    Balanced: {stats.get('is_balanced', 'N/A')}")
                        for regime in MarketRegime:
                            regime_stats = stats.get(regime.name, {})
                            pct = regime_stats.get("percentage", 0)
                            logger.info(f"    {regime.name}: {pct:.1f}%")

                except Exception as e:
                    logger.error(f"  {tf}: Error - {e}")
                    results[pair][tf] = {"error": str(e)}

    else:
        # Indicator-based labeling (original method)
        logger.warning("Using INDICATOR-based labeling. This causes target leakage for ML training!")
        labeler = RegimeLabeler(
            adx_trending_threshold=adx_threshold,
            high_vol_percentile=high_vol_percentile,
            mode="indicator",
        )

        pipeline = RegimeLabelPipeline(
            features_path=features_path,
            labels_path=labels_path,
            labeler=labeler,
        )

        for pair in pairs:
            logger.info(f"Labeling regimes for {pair}...")

            try:
                pair_results = pipeline.label_and_save(
                    pair=pair,
                    timeframes=timeframes,
                )

                results[pair] = pair_results

                for tf, tf_result in pair_results.items():
                    if "error" in tf_result:
                        logger.error(f"  {tf}: {tf_result['error']}")
                    else:
                        validation = tf_result.get("validation", {})
                        logger.info(f"  {tf}: {tf_result['rows']} rows")
                        logger.info(f"    Class balance: {validation.get('is_balanced', 'N/A')}")
                        logger.info(f"    Avg regime duration: {validation.get('avg_regime_duration', 'N/A'):.1f} candles")

                        # Log regime distribution
                        stats = tf_result.get("stats", {})
                        for regime_name, regime_stats in stats.items():
                            pct = regime_stats.get("percentage", 0)
                            logger.info(f"    {regime_name}: {pct:.1f}%")

            except Exception as e:
                logger.error(f"Error labeling regimes for {pair}: {e}")
                results[pair] = {"error": str(e)}

    return results


def show_summary(
    pairs: List[str],
    features_path: Path,
    labels_path: Path,
) -> None:
    """
    Display summary of computed features and labels.

    Args:
        pairs: List of trading pairs
        features_path: Path to features
        labels_path: Path to labeled data
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    # Features summary
    print("\n📊 Computed Features:")
    print("-" * 40)

    for pair in pairs:
        print(f"\n{pair}:")

        # Check each timeframe
        for tf_dir in sorted(features_path.iterdir()):
            if not tf_dir.is_dir():
                continue

            tf = tf_dir.name
            feature_file = tf_dir / pair / "features.parquet"

            if feature_file.exists():
                import pyarrow.parquet as pq
                metadata = pq.read_metadata(feature_file)
                size_mb = feature_file.stat().st_size / (1024 * 1024)
                print(f"  {tf}: {metadata.num_rows:,} rows, "
                      f"{metadata.num_columns} cols, {size_mb:.1f} MB")
            else:
                print(f"  {tf}: Not computed")

    # Labels summary
    print("\n🏷️ Regime Labels:")
    print("-" * 40)

    for pair in pairs:
        print(f"\n{pair}:")

        for tf_dir in sorted(labels_path.iterdir()):
            if not tf_dir.is_dir():
                continue

            tf = tf_dir.name
            label_file = tf_dir / pair / "labeled.parquet"

            if label_file.exists():
                import pandas as pd
                import pyarrow.parquet as pq

                df = pq.read_table(label_file).to_pandas()

                # Regime distribution
                regime_counts = df["regime"].value_counts()
                total = len(df)

                print(f"  {tf} ({total:,} rows):")
                for regime in MarketRegime:
                    count = regime_counts.get(regime.value, 0)
                    pct = count / total * 100 if total > 0 else 0
                    bar = "█" * int(pct / 5)
                    print(f"    {regime.name:15} {count:6,} ({pct:5.1f}%) {bar}")
            else:
                print(f"  {tf}: Not labeled")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute technical features from OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute features for BTC/USD
    python scripts/compute_features.py --pairs XBTUSD

    # Compute for specific timeframes with regime labeling
    python scripts/compute_features.py --pairs XBTUSD --timeframes 5m 1h --label-regimes

    # Show summary
    python scripts/compute_features.py --pairs XBTUSD --summary
        """,
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["XBTUSD"],
        help="Trading pairs to process (default: XBTUSD)",
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Timeframes to compute (default: all available)",
    )

    parser.add_argument(
        "--ohlcv-path",
        type=Path,
        default=Path("data/ohlcv"),
        help="Path to OHLCV data (default: data/ohlcv)",
    )

    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/features"),
        help="Path for feature output (default: data/features)",
    )

    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/labels"),
        help="Path for labeled data output (default: data/labels)",
    )

    parser.add_argument(
        "--short-window",
        type=int,
        default=14,
        help="Short-term lookback window (default: 14)",
    )

    parser.add_argument(
        "--medium-window",
        type=int,
        default=50,
        help="Medium-term lookback window (default: 50)",
    )

    parser.add_argument(
        "--long-window",
        type=int,
        default=200,
        help="Long-term lookback window (default: 200)",
    )

    parser.add_argument(
        "--label-regimes",
        action="store_true",
        help="Also label data with market regimes",
    )

    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=25.0,
        help="ADX threshold for trending regime (default: 25)",
    )

    parser.add_argument(
        "--high-vol-percentile",
        type=float,
        default=80.0,
        help="Percentile for high volatility regime (default: 80)",
    )

    parser.add_argument(
        "--labeling-mode",
        type=str,
        choices=["outcome", "indicator"],
        default="outcome",
        help="Labeling mode: 'outcome' (default, uses future returns - correct for ML) "
             "or 'indicator' (uses current indicators - causes target leakage!)",
    )

    parser.add_argument(
        "--outcome-lookahead",
        type=int,
        default=20,
        help="Periods to look ahead for outcome-based labeling (default: 20)",
    )

    parser.add_argument(
        "--outcome-trend-threshold",
        type=float,
        default=0.02,
        help="Return threshold for trending in outcome mode (default: 0.02 = 2%%)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of computed features and labels",
    )

    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature computation (use with --label-regimes)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show summary and exit
    if args.summary:
        show_summary(
            pairs=args.pairs,
            features_path=args.features_path,
            labels_path=args.labels_path,
        )
        return 0

    # Compute features
    if not args.skip_features:
        logger.info("=" * 60)
        logger.info("COMPUTING FEATURES")
        logger.info("=" * 60)

        results = compute_features(
            pairs=args.pairs,
            timeframes=args.timeframes,
            ohlcv_path=args.ohlcv_path,
            features_path=args.features_path,
            short_window=args.short_window,
            medium_window=args.medium_window,
            long_window=args.long_window,
        )

        # Check for errors
        errors = [p for p, r in results.items() if r.get("status") == "error"]
        if errors:
            logger.error(f"Errors in pairs: {errors}")

    # Label regimes
    if args.label_regimes:
        logger.info("")
        logger.info("=" * 60)
        logger.info("LABELING REGIMES")
        logger.info("=" * 60)

        label_results = label_regimes(
            pairs=args.pairs,
            timeframes=args.timeframes,
            features_path=args.features_path,
            labels_path=args.labels_path,
            ohlcv_path=args.ohlcv_path,
            labeling_mode=args.labeling_mode,
            adx_threshold=args.adx_threshold,
            high_vol_percentile=args.high_vol_percentile,
            outcome_lookahead=args.outcome_lookahead,
            outcome_trend_threshold=args.outcome_trend_threshold,
        )

    logger.info("")
    logger.info("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
