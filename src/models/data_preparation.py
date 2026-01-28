"""
Data Preparation for ML Model Training.

Handles loading labeled data and preparing it for model training:
- Time-series aware train/validation/test splitting
- Feature selection and preprocessing
- Class balancing strategies
- Data validation and quality checks
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/val/test data splits."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    timestamps_train: pd.Series
    timestamps_val: pd.Series
    timestamps_test: pd.Series
    feature_names: List[str]
    scaler: Optional[Any] = None

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get class distribution for each split."""
        return {
            "train": dict(self.y_train.value_counts().sort_index()),
            "val": dict(self.y_val.value_counts().sort_index()),
            "test": dict(self.y_test.value_counts().sort_index()),
        }


class DataPreparation:
    """
    Prepares labeled data for ML model training.

    Key features:
    - Chronological splitting (no data leakage)
    - Feature scaling with train-only fitting
    - Class weight computation for imbalanced data
    - Feature selection utilities
    """

    # Columns to exclude from features
    EXCLUDE_COLS = [
        "timestamp", "open", "high", "low", "close", "volume",
        "regime", "regime_name", "buy_volume", "sell_volume", "vwap",
        "trade_count",
    ]

    # Columns that start with these prefixes are excluded
    EXCLUDE_PREFIXES = ["fwd_"]  # Forward-looking columns

    def __init__(
        self,
        labels_path: Path,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        scale_features: bool = True,
        scaler_type: str = "robust",  # "standard" or "robust"
    ):
        """
        Initialize data preparation.

        Args:
            labels_path: Path to labeled data directory
            train_split: Fraction for training (default: 0.7)
            val_split: Fraction for validation (default: 0.15)
            test_split: Fraction for testing (default: 0.15)
            scale_features: Whether to scale features (default: True)
            scaler_type: Type of scaler - "standard" or "robust"
        """
        self.labels_path = Path(labels_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.scale_features = scale_features
        self.scaler_type = scaler_type

        # Validate splits
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Splits must sum to 1.0, got {total}")

    def load_labeled_data(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Load labeled data for a pair and timeframe.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            timeframe: Timeframe (e.g., "5m")

        Returns:
            DataFrame with features and labels
        """
        file_path = self.labels_path / timeframe / pair / "labeled.parquet"

        if not file_path.exists():
            raise ValueError(f"No labeled data found at {file_path}")

        df = pq.read_table(file_path).to_pandas()
        logger.info(f"Loaded {len(df)} rows from {file_path}")

        return df

    def _identify_feature_columns(
        self,
        df: pd.DataFrame,
        additional_exclude: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Identify feature columns from DataFrame.

        Args:
            df: DataFrame with all columns
            additional_exclude: Additional columns to exclude

        Returns:
            List of feature column names
        """
        exclude = set(self.EXCLUDE_COLS)
        if additional_exclude:
            exclude.update(additional_exclude)

        feature_cols = []
        for col in df.columns:
            # Skip excluded columns
            if col in exclude:
                continue

            # Skip columns with excluded prefixes
            if any(col.startswith(prefix) for prefix in self.EXCLUDE_PREFIXES):
                continue

            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        return sorted(feature_cols)

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        strategy: str = "drop",
    ) -> pd.DataFrame:
        """
        Handle missing values in features.

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
            strategy: How to handle - "drop", "fill_median", "fill_zero"

        Returns:
            DataFrame with missing values handled
        """
        # Replace infinities with NaN
        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Count missing
        missing_counts = df[feature_cols].isna().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values across features")

            # Log columns with most missing
            worst_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
            if len(worst_cols) > 0:
                logger.debug(f"Columns with missing values:\n{worst_cols.head(10)}")

        if strategy == "drop":
            before = len(df)
            df = df.dropna(subset=feature_cols)
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with missing values")

        elif strategy == "fill_median":
            for col in feature_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        elif strategy == "fill_zero":
            df[feature_cols] = df[feature_cols].fillna(0)

        return df

    def prepare_data(
        self,
        pair: str,
        timeframe: str,
        feature_cols: Optional[List[str]] = None,
        min_samples: int = 1000,
        missing_strategy: str = "drop",
    ) -> DataSplit:
        """
        Prepare data for model training with chronological splits.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            feature_cols: Specific feature columns (None = auto-detect)
            min_samples: Minimum samples required
            missing_strategy: How to handle missing values

        Returns:
            DataSplit object with train/val/test data
        """
        # Load data
        df = self.load_labeled_data(pair, timeframe)

        # Ensure sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Identify feature columns
        if feature_cols is None:
            feature_cols = self._identify_feature_columns(df)

        logger.info(f"Using {len(feature_cols)} feature columns")

        # Handle missing values
        df = self._handle_missing_values(df, feature_cols, missing_strategy)

        if len(df) < min_samples:
            raise ValueError(
                f"Insufficient data: {len(df)} samples, need {min_samples}"
            )

        # Extract features and labels
        X = df[feature_cols].copy()
        y = df["regime"].copy()
        timestamps = df["timestamp"].copy()

        # Chronological split indices
        n = len(df)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        # Split data
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        ts_train = timestamps.iloc[:train_end]

        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        ts_val = timestamps.iloc[train_end:val_end]

        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        ts_test = timestamps.iloc[val_end:]

        logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Scale features (fit only on training data)
        scaler = None
        if self.scale_features:
            if self.scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            # Fit on training data only
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=feature_cols,
                index=X_train.index,
            )

            # Transform val and test
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=feature_cols,
                index=X_val.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_cols,
                index=X_test.index,
            )

            X_train = X_train_scaled
            X_val = X_val_scaled
            X_test = X_test_scaled

        return DataSplit(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_val=X_val.reset_index(drop=True),
            y_val=y_val.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            timestamps_train=ts_train.reset_index(drop=True),
            timestamps_val=ts_val.reset_index(drop=True),
            timestamps_test=ts_test.reset_index(drop=True),
            feature_names=feature_cols,
            scaler=scaler,
        )

    def compute_class_weights(
        self,
        y: pd.Series,
        strategy: str = "balanced",
    ) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Args:
            y: Target labels
            strategy: Weight strategy - "balanced" or "sqrt_balanced"

        Returns:
            Dict mapping class label to weight
        """
        classes = np.unique(y)

        if strategy == "balanced":
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y,
            )
        elif strategy == "sqrt_balanced":
            # Less aggressive balancing
            balanced_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y,
            )
            weights = np.sqrt(balanced_weights)
        else:
            weights = np.ones(len(classes))

        return dict(zip(classes, weights))

    def get_sample_weights(
        self,
        y: pd.Series,
        class_weights: Dict[int, float],
    ) -> np.ndarray:
        """
        Convert class weights to per-sample weights.

        Args:
            y: Target labels
            class_weights: Dict of class weights

        Returns:
            Array of sample weights
        """
        return np.array([class_weights[label] for label in y])

    def validate_data_quality(
        self,
        data_split: DataSplit,
    ) -> Dict[str, Any]:
        """
        Validate data quality and return report.

        Args:
            data_split: Prepared data split

        Returns:
            Dict with validation results
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check for sufficient samples
        if data_split.train_size < 500:
            report["warnings"].append(
                f"Small training set: {data_split.train_size} samples"
            )

        # Check class distribution
        class_dist = data_split.get_class_distribution()

        for split_name, dist in class_dist.items():
            total = sum(dist.values())
            for class_id, count in dist.items():
                pct = count / total * 100
                if pct < 5:
                    report["warnings"].append(
                        f"Class {class_id} has only {pct:.1f}% in {split_name}"
                    )

        # Check for constant features
        constant_features = []
        for col in data_split.feature_names:
            if data_split.X_train[col].std() == 0:
                constant_features.append(col)

        if constant_features:
            report["warnings"].append(
                f"Constant features found: {constant_features[:5]}..."
            )

        # Check for highly correlated features
        corr_matrix = data_split.X_train.corr().abs()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j])
                    )

        if high_corr_pairs:
            report["warnings"].append(
                f"Highly correlated feature pairs: {len(high_corr_pairs)}"
            )

        # Check timestamp continuity
        train_ts = data_split.timestamps_train
        val_ts = data_split.timestamps_val
        test_ts = data_split.timestamps_test

        if train_ts.max() >= val_ts.min():
            report["errors"].append("Data leakage: train overlaps with val")
            report["valid"] = False

        if val_ts.max() >= test_ts.min():
            report["errors"].append("Data leakage: val overlaps with test")
            report["valid"] = False

        return report

    def select_features_by_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> List[str]:
        """
        Select features based on importance scores.

        Args:
            feature_importance: Dict of feature -> importance
            top_n: Select top N features (optional)
            min_importance: Minimum importance threshold

        Returns:
            List of selected feature names
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter by minimum importance
        filtered = [(f, imp) for f, imp in sorted_features if imp >= min_importance]

        # Take top N if specified
        if top_n is not None:
            filtered = filtered[:top_n]

        return [f for f, _ in filtered]

    def create_temporal_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        gap: int = 0,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create temporal cross-validation folds.

        Unlike standard K-fold, this respects time ordering:
        - Each fold uses earlier data for training, later for validation
        - Optional gap between train and val to prevent leakage

        Args:
            df: DataFrame with data
            n_folds: Number of folds
            gap: Number of samples to skip between train and val

        Returns:
            List of (train_indices, val_indices) tuples
        """
        n = len(df)
        fold_size = n // (n_folds + 1)

        folds = []
        for i in range(n_folds):
            # Training: from start to fold boundary
            train_end = fold_size * (i + 1)
            train_idx = np.arange(0, train_end)

            # Validation: after gap, one fold size
            val_start = train_end + gap
            val_end = min(val_start + fold_size, n)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) > 0:
                folds.append((train_idx, val_idx))

        return folds
