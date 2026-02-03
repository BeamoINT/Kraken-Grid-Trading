"""
Feature Engineering Pipeline.

Orchestrates the computation of all features from OHLCV data:
- Price features
- Volume features
- Volatility features
- Trend features

Handles multi-timeframe feature computation and storage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .price_features import compute_price_features
from .volume_features import compute_volume_features
from .volatility_features import compute_volatility_features
from .trend_features import compute_trend_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for OHLCV data.

    Computes technical indicators and derived features across
    multiple timeframes for use in ML models.

    Usage:
        pipeline = FeaturePipeline(ohlcv_path, features_path)
        pipeline.compute_features("XBTUSD", ["5m", "1h"])
    """

    # Feature schema for Parquet storage
    FEATURE_SCHEMA = pa.schema([
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("feature_name", pa.string()),
        ("value", pa.float64()),
    ])

    def __init__(
        self,
        ohlcv_path: Path,
        features_path: Path,
        short_window: int = 14,
        medium_window: int = 50,
        long_window: int = 200,
    ):
        """
        Initialize the feature pipeline.

        Args:
            ohlcv_path: Path to OHLCV data directory
            features_path: Path for storing computed features
            short_window: Short-term lookback window (default: 14)
            medium_window: Medium-term lookback window (default: 50)
            long_window: Long-term lookback window (default: 200)
        """
        self.ohlcv_path = Path(ohlcv_path)
        self.features_path = Path(features_path)
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

        self.features_path.mkdir(parents=True, exist_ok=True)

    def _load_ohlcv(
        self,
        pair: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for a pair and timeframe."""
        file_path = self.ohlcv_path / timeframe / pair / "ohlcv.parquet"

        if not file_path.exists():
            raise ValueError(f"No OHLCV data for {pair} at {timeframe}")

        df = pq.read_table(file_path).to_pandas()

        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]

        return df.sort_values("timestamp").reset_index(drop=True)

    def compute_all_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all features for an OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all computed features
        """
        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Computing features for {len(df)} candles...")

        # Collect all feature DataFrames for single concat (avoids fragmentation)
        feature_dfs = []

        # Base OHLCV columns
        base_df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        feature_dfs.append(base_df)

        # Price features
        logger.debug("Computing price features...")
        price_feats = compute_price_features(
            df,
            short_window=self.short_window,
            medium_window=self.medium_window,
            long_window=self.long_window,
        )
        feature_dfs.append(price_feats.add_prefix("price_"))

        # Volume features
        logger.debug("Computing volume features...")
        volume_feats = compute_volume_features(
            df,
            short_window=self.short_window,
            medium_window=self.medium_window,
        )
        feature_dfs.append(volume_feats.add_prefix("vol_"))

        # Volatility features
        logger.debug("Computing volatility features...")
        volatility_feats = compute_volatility_features(
            df,
            atr_period=self.short_window,
            bb_period=20,
            vol_period=20,
            lookback=100,
        )
        feature_dfs.append(volatility_feats.add_prefix("volat_"))

        # Trend features
        logger.debug("Computing trend features...")
        trend_feats = compute_trend_features(
            df,
            adx_period=self.short_window,
            short_ma=self.short_window,
            medium_ma=self.medium_window,
            long_ma=self.long_window,
        )
        feature_dfs.append(trend_feats.add_prefix("trend_"))

        # Single concat avoids DataFrame fragmentation
        features = pd.concat(feature_dfs, axis=1)

        # Add lagged features for key indicators
        logger.debug("Computing lagged features...")
        features = self._add_lagged_features(features)

        logger.info(f"Computed {len(features.columns)} features")
        return features

    def _add_lagged_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = None,
    ) -> pd.DataFrame:
        """
        Add lagged versions of key features.

        Lags help the model capture temporal patterns.

        Args:
            df: DataFrame with features
            lags: List of lag periods (default: [1, 2, 3, 5])

        Returns:
            DataFrame with lagged features added
        """
        if lags is None:
            lags = [1, 2, 3, 5]

        # Key features to lag
        key_features = [
            "price_log_return",
            "price_momentum_short",
            "vol_volume_ratio",
            "vol_buy_sell_ratio",
            "volat_natr",
            "trend_adx",
            "trend_di_difference",
            "trend_rsi",
            "trend_macd_histogram",
        ]

        # Collect all lagged columns (avoids DataFrame fragmentation)
        lagged_cols = {}
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    lagged_cols[f"{feature}_lag{lag}"] = df[feature].shift(lag)

        # Single concat of all lagged features
        if lagged_cols:
            lagged_df = pd.DataFrame(lagged_cols, index=df.index)
            df = pd.concat([df, lagged_df], axis=1)

        return df

    def compute_features(
        self,
        pair: str,
        timeframes: Optional[List[str]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute features for a trading pair across timeframes.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            timeframes: List of timeframes (default: all available)
            start: Start timestamp (optional)
            end: End timestamp (optional)
            save: Whether to save to Parquet (default: True)

        Returns:
            Dict mapping timeframe to feature DataFrame
        """
        if timeframes is None:
            # Find available timeframes
            timeframes = []
            for tf_dir in self.ohlcv_path.iterdir():
                if tf_dir.is_dir():
                    pair_dir = tf_dir / pair
                    if pair_dir.exists():
                        timeframes.append(tf_dir.name)

        if not timeframes:
            raise ValueError(f"No OHLCV data found for {pair}")

        logger.info(f"Computing features for {pair} at timeframes: {timeframes}")

        results = {}
        for tf in timeframes:
            logger.info(f"Processing {tf}...")

            try:
                # Load OHLCV data
                ohlcv = self._load_ohlcv(pair, tf, start, end)

                if len(ohlcv) < self.long_window + 10:
                    logger.warning(
                        f"Insufficient data for {tf}: {len(ohlcv)} candles, "
                        f"need at least {self.long_window + 10}"
                    )
                    continue

                # Compute features
                features = self.compute_all_features(ohlcv)

                # Save if requested
                if save:
                    self._save_features(features, pair, tf)

                results[tf] = features
                logger.info(f"  {tf}: {len(features)} rows, {len(features.columns)} features")

            except Exception as e:
                logger.error(f"Error processing {tf}: {e}")
                raise

        return results

    def _save_features(
        self,
        features: pd.DataFrame,
        pair: str,
        timeframe: str,
    ) -> None:
        """Save features to Parquet."""
        output_dir = self.features_path / timeframe / pair
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "features.parquet"

        # Convert timestamp to proper format
        features = features.copy()
        if not pd.api.types.is_datetime64_any_dtype(features["timestamp"]):
            features["timestamp"] = pd.to_datetime(features["timestamp"])

        # Ensure timezone aware
        if features["timestamp"].dt.tz is None:
            features["timestamp"] = features["timestamp"].dt.tz_localize("UTC")

        # Write to Parquet
        table = pa.Table.from_pandas(features, preserve_index=False)
        pq.write_table(table, output_file, compression="snappy")

        logger.info(f"Saved features to {output_file}")

    def load_features(
        self,
        pair: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Load computed features for a pair and timeframe.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            start: Start timestamp (optional)
            end: End timestamp (optional)

        Returns:
            DataFrame with features
        """
        file_path = self.features_path / timeframe / pair / "features.parquet"

        if not file_path.exists():
            raise ValueError(f"No features found for {pair} at {timeframe}")

        df = pq.read_table(file_path).to_pandas()

        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_feature_names(
        self,
        include_ohlcv: bool = False,
        include_lags: bool = True,
    ) -> List[str]:
        """
        Get list of computed feature names.

        Args:
            include_ohlcv: Include base OHLCV columns (default: False)
            include_lags: Include lagged features (default: True)

        Returns:
            List of feature column names
        """
        # Create a small sample to get column names
        sample_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="5min"),
            "open": np.random.randn(300).cumsum() + 100,
            "high": np.random.randn(300).cumsum() + 101,
            "low": np.random.randn(300).cumsum() + 99,
            "close": np.random.randn(300).cumsum() + 100,
            "volume": np.abs(np.random.randn(300)) * 1000,
            "buy_volume": np.abs(np.random.randn(300)) * 500,
            "sell_volume": np.abs(np.random.randn(300)) * 500,
            "vwap": np.random.randn(300).cumsum() + 100,
        })

        features = self.compute_all_features(sample_df)
        columns = list(features.columns)

        # Filter based on options
        if not include_ohlcv:
            ohlcv_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            columns = [c for c in columns if c not in ohlcv_cols]

        if not include_lags:
            columns = [c for c in columns if "_lag" not in c]

        return columns

    def get_summary(self, pair: str) -> Dict[str, Any]:
        """
        Get summary of computed features for a pair.

        Returns:
            Dict with feature counts per timeframe
        """
        summary = {}

        for tf_dir in self.features_path.iterdir():
            if not tf_dir.is_dir():
                continue

            pair_file = tf_dir / pair / "features.parquet"
            if pair_file.exists():
                metadata = pq.read_metadata(pair_file)
                summary[tf_dir.name] = {
                    "rows": metadata.num_rows,
                    "columns": metadata.num_columns,
                    "file_size_mb": pair_file.stat().st_size / (1024 * 1024),
                }
            else:
                summary[tf_dir.name] = {
                    "rows": 0,
                    "columns": 0,
                    "file_size_mb": 0,
                }

        return summary

    def get_ml_features(
        self,
        pair: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Get features formatted for ML training.

        Drops OHLCV columns, handles NaN values.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            start: Start timestamp (optional)
            end: End timestamp (optional)
            dropna: Whether to drop rows with NaN (default: True)

        Returns:
            DataFrame ready for ML training
        """
        features = self.load_features(pair, timeframe, start, end)

        # Keep timestamp for reference but don't use as feature
        timestamp = features["timestamp"]

        # Drop OHLCV columns (not features)
        drop_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        feature_cols = [c for c in features.columns if c not in drop_cols]
        ml_features = features[feature_cols]

        # Handle infinity values
        ml_features = ml_features.replace([np.inf, -np.inf], np.nan)

        if dropna:
            # Find rows with NaN
            valid_mask = ~ml_features.isna().any(axis=1)
            ml_features = ml_features[valid_mask].reset_index(drop=True)
            timestamp = timestamp[valid_mask].reset_index(drop=True)

        # Add timestamp back for reference
        ml_features.insert(0, "timestamp", timestamp)

        return ml_features

    def compute_multi_timeframe_features(
        self,
        pair: str,
        base_timeframe: str = "5m",
        higher_timeframes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute features with multi-timeframe information.

        Adds features from higher timeframes to provide context.

        Args:
            pair: Trading pair
            base_timeframe: Base timeframe for features
            higher_timeframes: Higher timeframes to include (default: ["1h", "4h"])

        Returns:
            DataFrame with multi-timeframe features
        """
        if higher_timeframes is None:
            higher_timeframes = ["1h", "4h"]

        # Load base timeframe features
        base_features = self.load_features(pair, base_timeframe)

        # Key features to include from higher timeframes
        htf_features = [
            "trend_adx",
            "trend_di_difference",
            "trend_rsi",
            "volat_natr",
            "price_ma_dist_short",
        ]

        for htf in higher_timeframes:
            try:
                htf_data = self.load_features(pair, htf)

                # Resample to match base timeframe
                for feature in htf_features:
                    if feature in htf_data.columns:
                        # Forward fill higher timeframe data
                        htf_series = htf_data.set_index("timestamp")[feature]
                        htf_series = htf_series.reindex(
                            base_features["timestamp"],
                            method="ffill"
                        )
                        base_features[f"{feature}_{htf}"] = htf_series.values

            except ValueError:
                logger.warning(f"Could not load {htf} features for multi-timeframe")
                continue

        return base_features
