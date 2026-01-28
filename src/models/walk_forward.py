"""
Walk-Forward Validation for Time-Series ML.

Implements walk-forward validation which is essential for
financial time-series to avoid look-ahead bias:
- Train on historical data
- Validate on future data
- Roll forward and repeat

This provides a more realistic estimate of model performance
than standard cross-validation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Container for a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    train_score: Optional[float] = None
    test_score: Optional[float] = None
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation results."""
    folds: List[WalkForwardFold]
    mean_train_score: float
    mean_test_score: float
    std_test_score: float
    all_predictions: np.ndarray
    all_actuals: np.ndarray
    all_timestamps: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_folds": len(self.folds),
            "mean_train_score": self.mean_train_score,
            "mean_test_score": self.mean_test_score,
            "std_test_score": self.std_test_score,
            "fold_details": [
                {
                    "fold_id": f.fold_id,
                    "train_period": f"{f.train_start} to {f.train_end}",
                    "test_period": f"{f.test_start} to {f.test_end}",
                    "train_size": f.train_size,
                    "test_size": f.test_size,
                    "train_score": f.train_score,
                    "test_score": f.test_score,
                }
                for f in self.folds
            ],
        }


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series.

    Strategy:
    1. Start with initial training window
    2. Train model on training window
    3. Test on next test window
    4. Expand training window (or roll forward)
    5. Repeat until end of data

    Two modes:
    - Expanding: Training window grows each fold
    - Rolling: Training window slides forward (fixed size)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[float] = None,
        test_size: float = 0.1,
        gap: int = 0,
        expanding: bool = True,
        min_train_size: int = 1000,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of forward splits
            train_size: Initial training fraction (None = auto)
            test_size: Test fraction for each fold
            gap: Samples to skip between train and test
            expanding: If True, expand training window; else roll
            min_train_size: Minimum training samples
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
        self.min_train_size = min_train_size

    def _calculate_splits(
        self,
        n_samples: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate train/test boundaries for each fold.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        test_samples = int(n_samples * self.test_size)

        if self.train_size is not None:
            initial_train = int(n_samples * self.train_size)
        else:
            # Auto-calculate initial training size
            # Reserve enough for all test folds
            total_test = test_samples * self.n_splits + self.gap * self.n_splits
            initial_train = max(self.min_train_size, n_samples - total_test)

        splits = []

        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window: train from start
                train_start = 0
                train_end = initial_train + i * test_samples
            else:
                # Rolling window: fixed training size
                train_start = i * test_samples
                train_end = train_start + initial_train

            # Test window
            test_start = train_end + self.gap
            test_end = min(test_start + test_samples, n_samples)

            # Skip if not enough data
            if train_end > n_samples or test_start >= n_samples:
                break

            if test_end - test_start < 10:  # Minimum test size
                break

            splits.append((train_start, train_end, test_start, test_end))

        return splits

    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        score_func: Optional[Callable] = None,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation.

        Args:
            model: Sklearn-compatible model (will be cloned)
            X: Feature DataFrame
            y: Target Series
            timestamps: Timestamp Series for tracking
            sample_weight: Optional sample weights
            score_func: Custom scoring function (default: accuracy)
            verbose: Print progress

        Returns:
            WalkForwardResult with all folds and metrics
        """
        n_samples = len(X)
        splits = self._calculate_splits(n_samples)

        if len(splits) == 0:
            raise ValueError(
                f"Could not create any splits with {n_samples} samples"
            )

        if verbose:
            logger.info(f"Walk-forward validation with {len(splits)} folds")

        folds = []
        all_predictions = []
        all_actuals = []
        all_timestamps = []
        all_probabilities = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(splits):
            if verbose:
                logger.info(
                    f"  Fold {fold_id + 1}: Train [{train_start}:{train_end}], "
                    f"Test [{test_start}:{test_end}]"
                )

            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            ts_test = timestamps.iloc[test_start:test_end]

            # Get sample weights for training
            train_weights = None
            if sample_weight is not None:
                train_weights = sample_weight[train_start:train_end]

            # Clone and train model
            fold_model = clone(model)

            try:
                if train_weights is not None:
                    fold_model.fit(X_train, y_train, sample_weight=train_weights)
                else:
                    fold_model.fit(X_train, y_train)
            except TypeError:
                # Some models don't support sample_weight
                fold_model.fit(X_train, y_train)

            # Predict
            predictions = fold_model.predict(X_test)

            # Get probabilities if available
            probabilities = None
            if hasattr(fold_model, "predict_proba"):
                probabilities = fold_model.predict_proba(X_test)
                all_probabilities.append(probabilities)

            # Score
            if score_func:
                train_score = score_func(y_train, fold_model.predict(X_train))
                test_score = score_func(y_test, predictions)
            else:
                train_score = fold_model.score(X_train, y_train)
                test_score = fold_model.score(X_test, y_test)

            # Create fold result
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=timestamps.iloc[train_start],
                train_end=timestamps.iloc[train_end - 1],
                test_start=timestamps.iloc[test_start],
                test_end=timestamps.iloc[test_end - 1],
                train_size=train_end - train_start,
                test_size=test_end - test_start,
                train_score=train_score,
                test_score=test_score,
                predictions=predictions,
                actuals=y_test.values,
                probabilities=probabilities,
            )
            folds.append(fold)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_timestamps.extend(ts_test.values)

            if verbose:
                logger.info(
                    f"    Train score: {train_score:.4f}, Test score: {test_score:.4f}"
                )

        # Aggregate results
        train_scores = [f.train_score for f in folds if f.train_score is not None]
        test_scores = [f.test_score for f in folds if f.test_score is not None]

        result = WalkForwardResult(
            folds=folds,
            mean_train_score=np.mean(train_scores),
            mean_test_score=np.mean(test_scores),
            std_test_score=np.std(test_scores),
            all_predictions=np.array(all_predictions),
            all_actuals=np.array(all_actuals),
            all_timestamps=pd.Series(all_timestamps),
        )

        if verbose:
            logger.info(
                f"Walk-forward complete - Mean test score: {result.mean_test_score:.4f} "
                f"(+/- {result.std_test_score:.4f})"
            )

        return result


class PurgedWalkForward:
    """
    Purged walk-forward validation.

    Adds a purge period between train and test to prevent
    information leakage from overlapping features (e.g., rolling windows).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_gap: int = 5,
        test_size: float = 0.1,
    ):
        """
        Initialize purged walk-forward validator.

        Args:
            n_splits: Number of folds
            purge_gap: Samples to purge before test set
            embargo_gap: Samples to embargo after test set
            test_size: Fraction of data for testing per fold
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.test_size = test_size

    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Perform purged walk-forward validation.

        Similar to regular walk-forward but with purge/embargo gaps.
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        total_gap = self.purge_gap + self.embargo_gap

        folds = []
        all_predictions = []
        all_actuals = []
        all_timestamps = []

        for fold_id in range(self.n_splits):
            # Calculate boundaries
            test_start = (fold_id + 1) * test_samples + self.purge_gap
            test_end = min(test_start + test_samples, n_samples)

            if test_end > n_samples - self.embargo_gap:
                break

            # Training excludes purge zone before test and embargo after
            train_end = test_start - self.purge_gap

            if train_end < 100:  # Minimum training size
                continue

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            ts_test = timestamps.iloc[test_start:test_end]

            if verbose:
                logger.info(
                    f"Fold {fold_id + 1}: Train [0:{train_end}], "
                    f"Purge [{train_end}:{test_start}], "
                    f"Test [{test_start}:{test_end}]"
                )

            # Train and predict
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            predictions = fold_model.predict(X_test)

            train_score = fold_model.score(X_train, y_train)
            test_score = fold_model.score(X_test, y_test)

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=timestamps.iloc[0],
                train_end=timestamps.iloc[train_end - 1],
                test_start=timestamps.iloc[test_start],
                test_end=timestamps.iloc[test_end - 1],
                train_size=train_end,
                test_size=test_end - test_start,
                train_score=train_score,
                test_score=test_score,
                predictions=predictions,
                actuals=y_test.values,
            )
            folds.append(fold)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_timestamps.extend(ts_test.values)

        # Aggregate
        train_scores = [f.train_score for f in folds]
        test_scores = [f.test_score for f in folds]

        return WalkForwardResult(
            folds=folds,
            mean_train_score=np.mean(train_scores),
            mean_test_score=np.mean(test_scores),
            std_test_score=np.std(test_scores),
            all_predictions=np.array(all_predictions),
            all_actuals=np.array(all_actuals),
            all_timestamps=pd.Series(all_timestamps),
        )


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    More sophisticated approach that:
    1. Creates multiple test sets
    2. Ensures no leakage through purging
    3. Provides better coverage of data

    Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 10,
    ):
        """
        Initialize combinatorial CV.

        Args:
            n_splits: Number of data groups
            n_test_splits: Number of groups to use for testing
            purge_gap: Gap between train and test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap

    def get_splits(
        self,
        n_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Returns:
            List of (train_indices, test_indices) tuples
        """
        from itertools import combinations

        # Divide data into groups
        group_size = n_samples // self.n_splits
        groups = [
            np.arange(i * group_size, min((i + 1) * group_size, n_samples))
            for i in range(self.n_splits)
        ]

        splits = []

        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[g] for g in test_groups])

            # Training uses remaining groups with purge
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_idx = np.concatenate([groups[g] for g in train_groups])

            # Apply purge: remove samples near test boundaries
            min_test = test_idx.min()
            max_test = test_idx.max()

            purge_mask = (
                (train_idx < min_test - self.purge_gap) |
                (train_idx > max_test + self.purge_gap)
            )
            train_idx = train_idx[purge_mask]

            if len(train_idx) > 100:  # Minimum training size
                splits.append((train_idx, test_idx))

        return splits

    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform combinatorial purged CV.

        Returns:
            Dict with scores and predictions
        """
        splits = self.get_splits(len(X))

        if verbose:
            logger.info(f"Combinatorial CV with {len(splits)} splits")

        scores = []
        all_predictions = []
        all_actuals = []
        all_test_indices = []

        for i, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            predictions = fold_model.predict(X_test)
            score = fold_model.score(X_test, y_test)
            scores.append(score)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_test_indices.extend(test_idx)

            if verbose and (i + 1) % 5 == 0:
                logger.info(f"  Completed {i + 1}/{len(splits)} splits")

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "all_scores": scores,
            "predictions": np.array(all_predictions),
            "actuals": np.array(all_actuals),
            "test_indices": np.array(all_test_indices),
        }
