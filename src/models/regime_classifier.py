"""
Regime Classifier using XGBoost.

Classifies market conditions into regimes for adaptive grid trading:
- RANGING: Normal grid operation
- TRENDING_UP: Shift grid up, reduce buys
- TRENDING_DOWN: Pause buys, tighten stop-loss
- HIGH_VOLATILITY: Widen grid spacing
- BREAKOUT: Pause trading entirely

Supports hyperparameter tuning and ensemble methods.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import ParameterGrid

from .data_preparation import DataSplit

logger = logging.getLogger(__name__)


@dataclass
class XGBoostParams:
    """XGBoost hyperparameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    use_label_encoder: bool = False
    eval_metric: str = "mlogloss"
    early_stopping_rounds: Optional[int] = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XGBoost."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "use_label_encoder": self.use_label_encoder,
            "eval_metric": self.eval_metric,
        }


@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    params: Dict[str, Any]
    train_score: float
    val_score: float
    feature_importance: Dict[str, float]
    training_history: Optional[Dict[str, List[float]]] = None
    best_iteration: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "params": self.params,
            "train_score": self.train_score,
            "val_score": self.val_score,
            "feature_importance": self.feature_importance,
            "best_iteration": self.best_iteration,
        }


class RegimeClassifier:
    """
    XGBoost-based regime classifier.

    Handles:
    - Model training with early stopping
    - Hyperparameter tuning
    - Feature importance analysis
    - Prediction with confidence scores
    """

    def __init__(
        self,
        params: Optional[XGBoostParams] = None,
        num_classes: int = 5,
    ):
        """
        Initialize the classifier.

        Args:
            params: XGBoost hyperparameters
            num_classes: Number of regime classes (default: 5)
        """
        self.params = params or XGBoostParams()
        self.num_classes = num_classes
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.class_weights: Optional[Dict[int, float]] = None
        self._training_result: Optional[TrainingResult] = None

    def _create_model(
        self,
        params: Optional[XGBoostParams] = None,
    ) -> xgb.XGBClassifier:
        """Create XGBoost classifier instance."""
        p = params or self.params
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_classes,
            **p.to_dict(),
        )

    def train(
        self,
        data_split: DataSplit,
        class_weights: Optional[Dict[int, float]] = None,
        early_stopping: bool = True,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Train the classifier.

        Args:
            data_split: Prepared data split
            class_weights: Optional class weights for imbalanced data
            early_stopping: Use early stopping on validation set
            verbose: Print training progress

        Returns:
            TrainingResult with model and metrics
        """
        self.feature_names = data_split.feature_names
        self.class_weights = class_weights

        # Create model
        self.model = self._create_model()

        # Prepare sample weights if class weights provided
        sample_weight = None
        if class_weights:
            sample_weight = np.array([
                class_weights.get(label, 1.0)
                for label in data_split.y_train
            ])

        # Prepare eval set for early stopping
        eval_set = [(data_split.X_val, data_split.y_val)]

        # Train with early stopping
        if early_stopping and self.params.early_stopping_rounds:
            self.model.fit(
                data_split.X_train,
                data_split.y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                verbose=verbose,
            )
            best_iteration = self.model.best_iteration
        else:
            self.model.fit(
                data_split.X_train,
                data_split.y_train,
                sample_weight=sample_weight,
                verbose=verbose,
            )
            best_iteration = None

        # Calculate scores
        train_score = self.model.score(data_split.X_train, data_split.y_train)
        val_score = self.model.score(data_split.X_val, data_split.y_val)

        # Get feature importance
        importance = self._get_feature_importance()

        # Get training history
        history = None
        if hasattr(self.model, "evals_result"):
            history = self.model.evals_result()

        self._training_result = TrainingResult(
            model=self.model,
            params=self.params.to_dict(),
            train_score=train_score,
            val_score=val_score,
            feature_importance=importance,
            training_history=history,
            best_iteration=best_iteration,
        )

        logger.info(
            f"Training complete - Train: {train_score:.4f}, Val: {val_score:.4f}"
        )

        return self._training_result

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or self.feature_names is None:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict regime labels.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted regime labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)

    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        min_confidence: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence filtering.

        Args:
            X: Feature matrix
            min_confidence: Minimum probability for confident prediction

        Returns:
            Tuple of (predictions, confidences, is_confident)
        """
        proba = self.predict_proba(X)
        predictions = proba.argmax(axis=1)
        confidences = proba.max(axis=1)
        is_confident = confidences >= min_confidence

        return predictions, confidences, is_confident

    def tune_hyperparameters(
        self,
        data_split: DataSplit,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_iter: Optional[int] = None,
        class_weights: Optional[Dict[int, float]] = None,
        verbose: bool = True,
    ) -> Tuple[XGBoostParams, Dict[str, Any]]:
        """
        Tune hyperparameters using grid search.

        Args:
            data_split: Prepared data split
            param_grid: Parameter grid to search
            n_iter: Max iterations for random search (None = full grid)
            class_weights: Class weights for imbalanced data
            verbose: Print progress

        Returns:
            Tuple of (best_params, search_results)
        """
        if param_grid is None:
            param_grid = {
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [50, 100, 200],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            }

        # Generate parameter combinations
        all_params = list(ParameterGrid(param_grid))

        # Limit if n_iter specified
        if n_iter and len(all_params) > n_iter:
            np.random.shuffle(all_params)
            all_params = all_params[:n_iter]

        logger.info(f"Tuning {len(all_params)} parameter combinations...")

        # Prepare sample weights
        sample_weight = None
        if class_weights:
            sample_weight = np.array([
                class_weights.get(label, 1.0)
                for label in data_split.y_train
            ])

        results = []
        best_score = -np.inf
        best_params = None

        for i, params in enumerate(all_params):
            if verbose and i % 10 == 0:
                logger.info(f"  Trying combination {i+1}/{len(all_params)}...")

            # Create params object
            xgb_params = XGBoostParams(**{
                **self.params.to_dict(),
                **params,
            })

            # Train model
            model = self._create_model(xgb_params)

            try:
                model.fit(
                    data_split.X_train,
                    data_split.y_train,
                    sample_weight=sample_weight,
                    eval_set=[(data_split.X_val, data_split.y_val)],
                    verbose=False,
                )

                val_score = model.score(data_split.X_val, data_split.y_val)

                results.append({
                    "params": params,
                    "val_score": val_score,
                    "best_iteration": getattr(model, "best_iteration", None),
                })

                if val_score > best_score:
                    best_score = val_score
                    best_params = xgb_params

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue

        logger.info(f"Best validation score: {best_score:.4f}")

        search_results = {
            "all_results": results,
            "best_score": best_score,
            "best_params": best_params.to_dict() if best_params else None,
        }

        return best_params, search_results

    def get_top_features(
        self,
        n: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self._training_result is None:
            raise ValueError("Model not trained yet")

        importance = self._training_result.feature_importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_features[:n]

    def save(self, path: Path) -> None:
        """
        Save model and metadata.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            "params": self.params.to_dict(),
            "num_classes": self.num_classes,
            "feature_names": self.feature_names,
            "class_weights": self.class_weights,
        }

        if self._training_result:
            metadata["training_result"] = self._training_result.to_dict()

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path) -> "RegimeClassifier":
        """
        Load model from directory.

        Args:
            path: Directory containing saved model

        Returns:
            Loaded RegimeClassifier
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create classifier
        params = XGBoostParams(**metadata["params"])
        classifier = cls(params=params, num_classes=metadata["num_classes"])
        classifier.feature_names = metadata["feature_names"]
        classifier.class_weights = metadata.get("class_weights")

        # Load model
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            classifier.model = pickle.load(f)

        logger.info(f"Loaded model from {path}")
        return classifier


class EnsembleRegimeClassifier:
    """
    Ensemble classifier combining multiple models.

    Uses voting or averaging to combine predictions from:
    - XGBoost
    - Random Forest
    - Additional models as needed
    """

    def __init__(
        self,
        num_classes: int = 5,
        voting: str = "soft",  # "hard" or "soft"
    ):
        """
        Initialize ensemble classifier.

        Args:
            num_classes: Number of regime classes
            voting: Voting strategy - "hard" or "soft"
        """
        self.num_classes = num_classes
        self.voting = voting
        self.models: Dict[str, Any] = {}
        self.ensemble: Optional[VotingClassifier] = None
        self.feature_names: Optional[List[str]] = None

    def add_xgboost(
        self,
        params: Optional[XGBoostParams] = None,
    ) -> None:
        """Add XGBoost to ensemble."""
        p = params or XGBoostParams()
        self.models["xgboost"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_classes,
            **p.to_dict(),
        )

    def add_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        """Add Random Forest to ensemble."""
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(
        self,
        data_split: DataSplit,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        Train ensemble on data.

        Args:
            data_split: Prepared data split
            class_weights: Optional class weights

        Returns:
            Dict of model scores
        """
        if not self.models:
            # Add default models
            self.add_xgboost()
            self.add_random_forest()

        self.feature_names = data_split.feature_names

        # Create ensemble
        estimators = list(self.models.items())
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            n_jobs=-1,
        )

        # Train
        sample_weight = None
        if class_weights:
            sample_weight = np.array([
                class_weights.get(label, 1.0)
                for label in data_split.y_train
            ])

        self.ensemble.fit(
            data_split.X_train,
            data_split.y_train,
            sample_weight=sample_weight,
        )

        # Get individual model scores
        scores = {
            "ensemble": self.ensemble.score(data_split.X_val, data_split.y_val),
        }

        for name, model in self.models.items():
            if hasattr(model, "score"):
                scores[name] = model.score(data_split.X_val, data_split.y_val)

        logger.info(f"Ensemble scores: {scores}")
        return scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict regime labels."""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained yet")
        return self.ensemble.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict regime probabilities."""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained yet")
        return self.ensemble.predict_proba(X)
