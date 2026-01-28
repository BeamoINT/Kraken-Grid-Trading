"""
Market Regime Labeler.

Labels historical data with market regimes for supervised learning.
Regimes are determined by a combination of trend, volatility, and
momentum indicators.

Regime Types:
- RANGING: Low trend strength, price oscillating in range
- TRENDING_UP: Strong uptrend with directional momentum
- TRENDING_DOWN: Strong downtrend with directional momentum
- HIGH_VOLATILITY: Elevated volatility without clear direction
- BREAKOUT: Sharp price movement with volume spike
"""

import logging
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Market regime classifications."""
    RANGING = 0
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    HIGH_VOLATILITY = 3
    BREAKOUT = 4


class RegimeLabeler:
    """
    Labels market data with regime classifications.

    Uses a rules-based approach to create training labels:
    1. Check volatility level (percentile)
    2. Check trend strength (ADX)
    3. Check trend direction (+DI/-DI)
    4. Check for breakouts (price/volume)

    The labels can then be used for supervised ML training.
    """

    def __init__(
        self,
        # ADX thresholds
        adx_trending_threshold: float = 25.0,
        adx_strong_trend_threshold: float = 40.0,
        # Volatility thresholds
        high_vol_percentile: float = 80.0,
        low_vol_percentile: float = 20.0,
        # Breakout detection
        breakout_vol_multiplier: float = 2.0,
        breakout_atr_multiplier: float = 1.5,
        # Forward look for performance labeling
        forward_look: int = 5,
    ):
        """
        Initialize the regime labeler.

        Args:
            adx_trending_threshold: ADX level for trending market (default: 25)
            adx_strong_trend_threshold: ADX for strong trend (default: 40)
            high_vol_percentile: Percentile for high volatility (default: 80)
            low_vol_percentile: Percentile for low volatility (default: 20)
            breakout_vol_multiplier: Volume multiple for breakout (default: 2x)
            breakout_atr_multiplier: ATR multiple for breakout (default: 1.5x)
            forward_look: Candles to look forward for labeling (default: 5)
        """
        self.adx_trending = adx_trending_threshold
        self.adx_strong = adx_strong_trend_threshold
        self.high_vol_pct = high_vol_percentile
        self.low_vol_pct = low_vol_percentile
        self.breakout_vol_mult = breakout_vol_multiplier
        self.breakout_atr_mult = breakout_atr_multiplier
        self.forward_look = forward_look

    def _check_breakout(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Detect breakout conditions.

        Breakout = High volume + Large price move + Bollinger breach

        Args:
            df: DataFrame with feature columns

        Returns:
            Series of breakout flags (boolean)
        """
        # Volume spike
        volume_col = "vol_volume_ratio" if "vol_volume_ratio" in df.columns else None
        if volume_col:
            volume_spike = df[volume_col] > self.breakout_vol_mult
        else:
            volume_spike = pd.Series(False, index=df.index)

        # Large ATR move
        atr_col = "volat_breakout_magnitude" if "volat_breakout_magnitude" in df.columns else None
        if atr_col:
            large_move = df[atr_col] > self.breakout_atr_mult
        else:
            large_move = pd.Series(False, index=df.index)

        # Bollinger breach
        bb_col = "volat_bb_percent_b" if "volat_bb_percent_b" in df.columns else None
        if bb_col:
            bb_breach = (df[bb_col] > 1.0) | (df[bb_col] < 0.0)
        else:
            bb_breach = pd.Series(False, index=df.index)

        # Breakout if volume spike AND (large move OR BB breach)
        breakout = volume_spike & (large_move | bb_breach)

        return breakout

    def _check_high_volatility(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Detect high volatility conditions.

        Args:
            df: DataFrame with feature columns

        Returns:
            Series of high vol flags (boolean)
        """
        vol_pct_col = "volat_vol_percentile" if "volat_vol_percentile" in df.columns else None

        if vol_pct_col:
            return df[vol_pct_col] > self.high_vol_pct
        else:
            return pd.Series(False, index=df.index)

    def _check_trending(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect trending conditions using ADX.

        Args:
            df: DataFrame with feature columns

        Returns:
            Tuple of (is_trending, trend_direction) Series
        """
        adx_col = "trend_adx" if "trend_adx" in df.columns else None
        di_diff_col = "trend_di_difference" if "trend_di_difference" in df.columns else None

        if adx_col:
            is_trending = df[adx_col] > self.adx_trending
        else:
            is_trending = pd.Series(False, index=df.index)

        if di_diff_col:
            # Positive = uptrend, Negative = downtrend
            trend_direction = np.sign(df[di_diff_col])
        else:
            trend_direction = pd.Series(0, index=df.index)

        return is_trending, trend_direction

    def label_regimes(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label each row with a market regime.

        Priority order:
        1. BREAKOUT (highest priority - unusual market condition)
        2. HIGH_VOLATILITY (elevated risk environment)
        3. TRENDING_UP/DOWN (directional market)
        4. RANGING (default - no strong trend)

        Args:
            df: DataFrame with computed features

        Returns:
            DataFrame with 'regime' column added
        """
        result = df.copy()

        # Initialize as RANGING (default)
        regime = pd.Series(MarketRegime.RANGING, index=df.index)

        # Check conditions
        is_breakout = self._check_breakout(df)
        is_high_vol = self._check_high_volatility(df)
        is_trending, trend_dir = self._check_trending(df)

        # Apply labels in priority order (lowest priority first)

        # Trending (check direction)
        regime = regime.where(
            ~(is_trending & (trend_dir > 0)),
            MarketRegime.TRENDING_UP
        )
        regime = regime.where(
            ~(is_trending & (trend_dir < 0)),
            MarketRegime.TRENDING_DOWN
        )

        # High volatility (overrides trending if very high)
        regime = regime.where(
            ~is_high_vol,
            MarketRegime.HIGH_VOLATILITY
        )

        # Breakout (highest priority)
        regime = regime.where(
            ~is_breakout,
            MarketRegime.BREAKOUT
        )

        result["regime"] = regime.astype(int)
        result["regime_name"] = regime.map(lambda x: MarketRegime(x).name)

        return result

    def label_forward_returns(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Add forward return labels for performance analysis.

        Useful for understanding how each regime performs.

        Args:
            df: DataFrame with regime labels
            close_col: Name of close price column

        Returns:
            DataFrame with forward return columns
        """
        result = df.copy()

        if close_col not in df.columns:
            logger.warning(f"Close column '{close_col}' not found, skipping forward returns")
            return result

        close = df[close_col]

        # Forward returns
        for period in [1, 5, 10, 20]:
            result[f"fwd_return_{period}"] = (
                close.shift(-period) / close - 1
            )

        # Forward volatility
        log_returns = np.log(close / close.shift(1))
        result["fwd_volatility"] = log_returns.shift(-1).rolling(
            window=self.forward_look
        ).std() * np.sqrt(252)

        # Forward max drawdown
        def calc_max_drawdown(series):
            if len(series) < 2:
                return np.nan
            cummax = series.cummax()
            drawdown = (series - cummax) / cummax
            return drawdown.min()

        result["fwd_max_drawdown"] = close.rolling(
            window=self.forward_look
        ).apply(calc_max_drawdown, raw=False).shift(-self.forward_look)

        return result

    def get_regime_stats(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for each regime.

        Args:
            df: DataFrame with regime labels

        Returns:
            Dict with regime statistics
        """
        if "regime" not in df.columns:
            raise ValueError("DataFrame must have 'regime' column")

        stats = {}

        for regime in MarketRegime:
            mask = df["regime"] == regime.value
            regime_df = df[mask]

            regime_stats = {
                "count": len(regime_df),
                "percentage": len(regime_df) / len(df) * 100 if len(df) > 0 else 0,
            }

            # Forward return stats if available
            for period in [1, 5, 10]:
                ret_col = f"fwd_return_{period}"
                if ret_col in df.columns:
                    returns = regime_df[ret_col].dropna()
                    if len(returns) > 0:
                        regime_stats[f"mean_return_{period}"] = returns.mean()
                        regime_stats[f"std_return_{period}"] = returns.std()
                        regime_stats[f"sharpe_{period}"] = (
                            returns.mean() / returns.std()
                            if returns.std() > 0 else 0
                        )

            stats[regime.name] = regime_stats

        return stats

    def validate_labels(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Validate regime labels for quality.

        Checks for:
        - Class balance
        - Transition patterns
        - Regime duration

        Args:
            df: DataFrame with regime labels

        Returns:
            Dict with validation results
        """
        if "regime" not in df.columns:
            raise ValueError("DataFrame must have 'regime' column")

        validation = {}

        # Class distribution
        regime_counts = df["regime"].value_counts()
        total = len(df)

        class_balance = {}
        for regime in MarketRegime:
            count = regime_counts.get(regime.value, 0)
            class_balance[regime.name] = {
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0,
            }
        validation["class_balance"] = class_balance

        # Check for severe imbalance
        percentages = [v["percentage"] for v in class_balance.values()]
        validation["min_class_pct"] = min(percentages)
        validation["max_class_pct"] = max(percentages)
        validation["is_balanced"] = min(percentages) > 5.0  # At least 5% each class

        # Regime transitions
        regime_changes = df["regime"].diff().fillna(0) != 0
        transition_count = regime_changes.sum()
        validation["transition_count"] = int(transition_count)
        validation["avg_regime_duration"] = len(df) / max(transition_count, 1)

        # Check for too frequent transitions
        validation["transitions_too_frequent"] = validation["avg_regime_duration"] < 3

        return validation


class RegimeLabelPipeline:
    """
    Pipeline for labeling features with regimes and saving results.
    """

    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        labeler: Optional[RegimeLabeler] = None,
    ):
        """
        Initialize the labeling pipeline.

        Args:
            features_path: Path to computed features
            labels_path: Path for storing labeled data
            labeler: RegimeLabeler instance (default: create new)
        """
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.labeler = labeler or RegimeLabeler()

        self.labels_path.mkdir(parents=True, exist_ok=True)

    def load_features(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Load computed features."""
        file_path = self.features_path / timeframe / pair / "features.parquet"

        if not file_path.exists():
            raise ValueError(f"No features found for {pair} at {timeframe}")

        return pq.read_table(file_path).to_pandas()

    def label_and_save(
        self,
        pair: str,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Label features with regimes and save.

        Args:
            pair: Trading pair
            timeframes: List of timeframes (default: all available)

        Returns:
            Dict with labeling results per timeframe
        """
        if timeframes is None:
            timeframes = []
            for tf_dir in self.features_path.iterdir():
                if tf_dir.is_dir():
                    pair_dir = tf_dir / pair
                    if pair_dir.exists():
                        timeframes.append(tf_dir.name)

        results = {}

        for tf in timeframes:
            logger.info(f"Labeling {pair} at {tf}...")

            try:
                # Load features
                features = self.load_features(pair, tf)

                # Label regimes
                labeled = self.labeler.label_regimes(features)

                # Add forward returns
                labeled = self.labeler.label_forward_returns(labeled)

                # Validate
                validation = self.labeler.validate_labels(labeled)

                # Get stats
                stats = self.labeler.get_regime_stats(labeled)

                # Save
                output_dir = self.labels_path / tf / pair
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "labeled.parquet"

                table = pa.Table.from_pandas(labeled, preserve_index=False)
                pq.write_table(table, output_file, compression="snappy")

                results[tf] = {
                    "rows": len(labeled),
                    "validation": validation,
                    "stats": stats,
                    "output_file": str(output_file),
                }

                logger.info(f"  Saved {len(labeled)} labeled rows to {output_file}")

            except Exception as e:
                logger.error(f"Error labeling {tf}: {e}")
                results[tf] = {"error": str(e)}

        return results

    def load_labeled_data(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Load labeled data."""
        file_path = self.labels_path / timeframe / pair / "labeled.parquet"

        if not file_path.exists():
            raise ValueError(f"No labeled data for {pair} at {timeframe}")

        return pq.read_table(file_path).to_pandas()

    def get_training_data(
        self,
        pair: str,
        timeframe: str,
        feature_cols: Optional[List[str]] = None,
        dropna: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Get data formatted for ML training.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            feature_cols: List of feature columns (default: auto-detect)
            dropna: Drop rows with NaN (default: True)

        Returns:
            Tuple of (X features, y labels, timestamps)
        """
        df = self.load_labeled_data(pair, timeframe)

        # Identify feature columns
        if feature_cols is None:
            exclude_cols = [
                "timestamp", "open", "high", "low", "close", "volume",
                "regime", "regime_name",
            ]
            exclude_cols.extend([c for c in df.columns if c.startswith("fwd_")])
            feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df["regime"].copy()
        timestamps = df["timestamp"].copy()

        # Handle NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        if dropna:
            valid_mask = ~X.isna().any(axis=1)
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)
            timestamps = timestamps[valid_mask].reset_index(drop=True)

        return X, y, timestamps
