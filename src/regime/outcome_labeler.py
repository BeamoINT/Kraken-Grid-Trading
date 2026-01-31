"""
Outcome-Based Regime Labeler.

Labels market regimes based on FUTURE price returns rather than current
indicator values. This avoids target leakage where labels are derived
from the same features used for prediction.

The key insight is that regime labels should represent "what the market
will do" not "what indicators currently show". This creates a legitimate
prediction task for the ML model.

Example:
    labeler = OutcomeBasedLabeler(lookahead=20, trend_threshold=0.02)
    labels = labeler.label(ohlcv_df)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .regime_labeler import MarketRegime

logger = logging.getLogger(__name__)


class OutcomeBasedLabeler:
    """
    Labels market regimes based on future price behavior.

    Instead of using current indicators (ADX, DI, etc.) to define regimes,
    this labeler looks at what the market ACTUALLY DID in the future:

    - TRENDING_UP: Future return > threshold (price went up)
    - TRENDING_DOWN: Future return < -threshold (price went down)
    - HIGH_VOLATILITY: Future realized volatility in top percentile
    - BREAKOUT: Very large future move (2x threshold)
    - RANGING: Default - small future moves

    This creates a legitimate prediction task: "Given current features,
    predict what the market will do next."
    """

    def __init__(
        self,
        lookahead: int = 20,
        trend_threshold: float = 0.02,
        vol_percentile: float = 80.0,
        breakout_multiplier: float = 2.0,
    ):
        """
        Initialize the outcome-based labeler.

        Args:
            lookahead: Number of periods to look ahead for returns (default: 20)
            trend_threshold: Return threshold for trending (default: 0.02 = 2%)
            vol_percentile: Percentile for high volatility (default: 80)
            breakout_multiplier: Multiplier of trend_threshold for breakout (default: 2x)
        """
        self.lookahead = lookahead
        self.trend_threshold = trend_threshold
        self.vol_percentile = vol_percentile
        self.breakout_multiplier = breakout_multiplier

    def label(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.Series:
        """
        Label each row with a market regime based on future outcomes.

        Args:
            df: DataFrame with OHLCV data
            close_col: Name of close price column

        Returns:
            Series of MarketRegime labels (int)
        """
        if close_col not in df.columns:
            raise ValueError(f"Column '{close_col}' not found in DataFrame")

        close = df[close_col]

        # Calculate future returns (what the price actually did)
        future_return = close.shift(-self.lookahead) / close - 1

        # Calculate future realized volatility
        returns = close.pct_change()
        future_vol = returns.rolling(window=self.lookahead).std().shift(-self.lookahead)

        # Calculate volatility threshold from data
        vol_threshold = future_vol.quantile(self.vol_percentile / 100.0)

        # Initialize all as NaN (will be set to RANGING where future data exists)
        labels = pd.Series(np.nan, index=df.index)

        # Only label rows where we have future data
        valid_mask = future_return.notna()

        # Initialize valid rows as RANGING (default)
        labels[valid_mask] = MarketRegime.RANGING

        # Label based on future outcomes (order matters - later overrides earlier)

        # HIGH_VOLATILITY: Future volatility is high
        labels[(future_vol > vol_threshold) & valid_mask] = MarketRegime.HIGH_VOLATILITY

        # TRENDING_UP: Price goes up significantly
        labels[(future_return > self.trend_threshold) & valid_mask] = MarketRegime.TRENDING_UP

        # TRENDING_DOWN: Price goes down significantly
        labels[(future_return < -self.trend_threshold) & valid_mask] = MarketRegime.TRENDING_DOWN

        # BREAKOUT: Very large move in either direction
        breakout_threshold = self.trend_threshold * self.breakout_multiplier
        labels[(future_return.abs() > breakout_threshold) & valid_mask] = MarketRegime.BREAKOUT

        # Return as nullable integer (keeps NaN for rows without future data)
        return labels

    def label_with_confidence(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Label with additional confidence score based on outcome magnitude.

        Args:
            df: DataFrame with OHLCV data
            close_col: Name of close price column

        Returns:
            DataFrame with 'regime' and 'regime_confidence' columns
        """
        close = df[close_col]
        future_return = close.shift(-self.lookahead) / close - 1

        labels = self.label(df, close_col)

        # Calculate confidence based on how far past the threshold
        confidence = pd.Series(0.5, index=df.index)  # Default

        # For trending, confidence = how much past threshold
        trending_up = labels == MarketRegime.TRENDING_UP
        trending_down = labels == MarketRegime.TRENDING_DOWN

        # Normalize confidence to 0.5-1.0 range
        confidence[trending_up] = np.clip(
            0.5 + (future_return[trending_up] - self.trend_threshold) / self.trend_threshold * 0.5,
            0.5, 1.0
        )
        confidence[trending_down] = np.clip(
            0.5 + (abs(future_return[trending_down]) - self.trend_threshold) / self.trend_threshold * 0.5,
            0.5, 1.0
        )

        # Breakouts have high confidence
        breakout = labels == MarketRegime.BREAKOUT
        confidence[breakout] = 0.9

        result = pd.DataFrame({
            "regime": labels,
            "regime_confidence": confidence,
        }, index=df.index)

        return result

    def get_label_stats(
        self,
        labels: pd.Series,
    ) -> dict:
        """
        Calculate statistics about label distribution.

        Args:
            labels: Series of regime labels

        Returns:
            Dict with statistics
        """
        # Drop NaN from lookahead
        valid_labels = labels.dropna()
        total = len(valid_labels)

        if total == 0:
            return {"error": "No valid labels"}

        stats = {
            "total_samples": total,
            "nan_samples": len(labels) - total,
        }

        for regime in MarketRegime:
            count = (valid_labels == regime.value).sum()
            stats[regime.name] = {
                "count": int(count),
                "percentage": count / total * 100,
            }

        # Check balance
        percentages = [stats[r.name]["percentage"] for r in MarketRegime]
        stats["min_class_pct"] = min(percentages)
        stats["max_class_pct"] = max(percentages)
        stats["is_balanced"] = min(percentages) > 5.0

        return stats

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> dict:
        """
        Validate that labels don't have leakage from features.

        This checks that labels are computed from future data only,
        not from any of the feature columns.

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names

        Returns:
            Dict with validation results
        """
        if "regime" not in df.columns:
            return {"error": "No regime column found"}

        validation = {
            "passed": True,
            "issues": [],
        }

        # Check that regime labels use future data (shift causes NaN at end)
        last_labels = df["regime"].tail(self.lookahead)
        nan_count = last_labels.isna().sum()

        if nan_count < self.lookahead:
            validation["passed"] = False
            validation["issues"].append(
                f"Expected {self.lookahead} NaN labels at end from lookahead, found {nan_count}"
            )

        # Check correlation between labels and individual features
        # High correlation might indicate leakage (though not always)
        suspicious_features = []
        for col in feature_cols:
            if col in df.columns:
                corr = df[col].corr(df["regime"])
                if abs(corr) > 0.9:  # Very high correlation
                    suspicious_features.append((col, corr))

        if suspicious_features:
            validation["warning"] = f"Features with >0.9 correlation to labels: {suspicious_features}"

        return validation


class OutcomeBasedLabelPipeline:
    """
    Pipeline for outcome-based labeling of feature files.
    """

    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        labeler: Optional[OutcomeBasedLabeler] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            features_path: Path to computed features
            labels_path: Path for labeled data output
            labeler: OutcomeBasedLabeler instance (default: create new)
        """
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.labeler = labeler or OutcomeBasedLabeler()

        self.labels_path.mkdir(parents=True, exist_ok=True)

    def label_and_save(
        self,
        pair: str,
        timeframe: str,
        ohlcv_path: Optional[Path] = None,
    ) -> dict:
        """
        Label features with outcome-based regimes and save.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            ohlcv_path: Path to OHLCV data (for close prices)

        Returns:
            Dict with results
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Load features
        feature_file = self.features_path / timeframe / pair / "features.parquet"
        if not feature_file.exists():
            return {"error": f"Features not found: {feature_file}"}

        features_df = pq.read_table(feature_file).to_pandas()

        # Need close price for labeling
        if "close" not in features_df.columns:
            # Try loading from OHLCV if available
            if ohlcv_path:
                ohlcv_file = Path(ohlcv_path) / timeframe / pair / "ohlcv.parquet"
                if ohlcv_file.exists():
                    ohlcv_df = pq.read_table(ohlcv_file).to_pandas()
                    features_df["close"] = ohlcv_df["close"].values
                else:
                    return {"error": "Close price not in features and OHLCV not found"}
            else:
                return {"error": "Close price not in features, provide ohlcv_path"}

        # Label with outcomes
        logger.info(f"Labeling {pair} {timeframe} with outcome-based approach...")
        labels = self.labeler.label(features_df)

        # Add to dataframe
        features_df["regime"] = labels
        features_df["regime_name"] = labels.map(lambda x: MarketRegime(x).name if pd.notna(x) else None)

        # Drop rows with NaN labels (from lookahead)
        valid_mask = features_df["regime"].notna()
        labeled_df = features_df[valid_mask].copy()

        # Get stats
        stats = self.labeler.get_label_stats(labels)

        # Save
        output_dir = self.labels_path / timeframe / pair
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "labeled.parquet"

        table = pa.Table.from_pandas(labeled_df, preserve_index=False)
        pq.write_table(table, output_file, compression="snappy")

        logger.info(f"Saved {len(labeled_df)} labeled rows to {output_file}")

        return {
            "rows": len(labeled_df),
            "dropped_nan": (~valid_mask).sum(),
            "stats": stats,
            "output_file": str(output_file),
        }
