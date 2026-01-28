"""
Model Evaluation for Regime Classification.

Provides comprehensive evaluation metrics and visualizations:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Per-class performance
- Feature importance analysis
- Regime transition analysis
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
)

from src.regime.regime_labeler import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of true instances
    predicted_count: int  # Number of predictions


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    log_loss_value: Optional[float]
    class_metrics: List[ClassMetrics]
    confusion_matrix: np.ndarray
    report: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "log_loss": self.log_loss_value,
            "class_metrics": {
                cm.class_name: {
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "f1": cm.f1,
                    "support": cm.support,
                }
                for cm in self.class_metrics
            },
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "MODEL EVALUATION SUMMARY",
            "=" * 50,
            f"Accuracy:         {self.accuracy:.4f}",
            f"Macro F1:         {self.macro_f1:.4f}",
            f"Weighted F1:      {self.weighted_f1:.4f}",
            "",
            "Per-class performance:",
        ]

        for cm in self.class_metrics:
            lines.append(
                f"  {cm.class_name:15} P={cm.precision:.3f} R={cm.recall:.3f} "
                f"F1={cm.f1:.3f} (n={cm.support})"
            )

        lines.append("=" * 50)
        return "\n".join(lines)


class ModelEvaluator:
    """
    Comprehensive model evaluation for regime classification.

    Provides:
    - Standard classification metrics
    - Per-regime performance analysis
    - Confusion matrix with regime names
    - Trading-relevant metrics
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            class_names: Names for each class (default: MarketRegime names)
        """
        if class_names is None:
            self.class_names = [r.name for r in MarketRegime]
        else:
            self.class_names = class_names

        self.num_classes = len(self.class_names)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            EvaluationResult with all metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Log loss if probabilities provided
        log_loss_value = None
        if y_proba is not None:
            try:
                log_loss_value = log_loss(y_true, y_proba)
            except Exception:
                pass

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        # Per-class metrics
        class_metrics = []
        for i in range(self.num_classes):
            # True positives, predictions, and support for this class
            true_mask = y_true == i
            pred_mask = y_pred == i

            support = true_mask.sum()
            predicted_count = pred_mask.sum()

            if support > 0:
                precision = precision_score(
                    y_true == i, y_pred == i, zero_division=0
                )
                recall = recall_score(
                    y_true == i, y_pred == i, zero_division=0
                )
                f1 = f1_score(
                    y_true == i, y_pred == i, zero_division=0
                )
            else:
                precision = recall = f1 = 0.0

            class_metrics.append(ClassMetrics(
                class_id=i,
                class_name=self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                precision=precision,
                recall=recall,
                f1=f1,
                support=int(support),
                predicted_count=int(predicted_count),
            ))

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names[:self.num_classes],
            zero_division=0,
        )

        return EvaluationResult(
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            log_loss_value=log_loss_value,
            class_metrics=class_metrics,
            confusion_matrix=cm,
            report=report,
        )

    def evaluate_with_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        thresholds: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate at different confidence thresholds.

        Shows accuracy vs coverage trade-off.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidence: Prediction confidence scores
            thresholds: Confidence thresholds to evaluate

        Returns:
            Dict with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        results = {}

        for thresh in thresholds:
            mask = confidence >= thresh
            coverage = mask.mean()

            if mask.sum() > 0:
                accuracy = accuracy_score(y_true[mask], y_pred[mask])
                f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            else:
                accuracy = 0.0
                f1 = 0.0

            results[thresh] = {
                "coverage": coverage,
                "accuracy": accuracy,
                "f1": f1,
                "n_samples": int(mask.sum()),
            }

        return results

    def confusion_matrix_df(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Create confusion matrix as DataFrame.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalize by row (true class)

        Returns:
            DataFrame with confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        if normalize:
            cm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums != 0)

        df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in self.class_names],
            columns=[f"Pred_{name}" for name in self.class_names],
        )

        return df

    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        timestamps: pd.Series,
        n_examples: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Features
            timestamps: Timestamps
            n_examples: Number of examples per error type

        Returns:
            Dict with error analysis
        """
        errors = y_true != y_pred
        error_idx = np.where(errors)[0]

        analysis = {
            "total_errors": len(error_idx),
            "error_rate": len(error_idx) / len(y_true),
            "error_distribution": {},
            "examples": [],
        }

        # Error distribution by (true, pred) pair
        for idx in error_idx:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            key = f"{true_class} -> {pred_class}"

            if key not in analysis["error_distribution"]:
                analysis["error_distribution"][key] = 0
            analysis["error_distribution"][key] += 1

        # Sort by frequency
        analysis["error_distribution"] = dict(
            sorted(
                analysis["error_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Get example errors
        sample_idx = np.random.choice(
            error_idx,
            size=min(n_examples, len(error_idx)),
            replace=False,
        )

        for idx in sample_idx:
            analysis["examples"].append({
                "index": int(idx),
                "timestamp": str(timestamps.iloc[idx]),
                "true": self.class_names[y_true[idx]],
                "predicted": self.class_names[y_pred[idx]],
            })

        return analysis

    def regime_transition_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze regime transition predictions.

        Checks how well the model predicts regime changes.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dict with transition analysis
        """
        # Find actual regime changes
        true_changes = np.where(np.diff(y_true) != 0)[0] + 1
        pred_changes = np.where(np.diff(y_pred) != 0)[0] + 1

        # Transition detection metrics
        # True positive: predicted change within window of actual change
        window = 2  # Allow +/- 2 samples

        true_positives = 0
        for tc in true_changes:
            if any(abs(pc - tc) <= window for pc in pred_changes):
                true_positives += 1

        false_negatives = len(true_changes) - true_positives
        false_positives = len(pred_changes) - true_positives

        # Calculate metrics
        precision = true_positives / max(len(pred_changes), 1)
        recall = true_positives / max(len(true_changes), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        return {
            "true_transitions": len(true_changes),
            "predicted_transitions": len(pred_changes),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "transition_precision": precision,
            "transition_recall": recall,
            "transition_f1": f1,
        }


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for the regime classifier.
    """

    def __init__(
        self,
        feature_names: List[str],
    ):
        """
        Initialize analyzer.

        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names

    def analyze_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Analyze feature importance.

        Args:
            importance_dict: Feature -> importance mapping
            top_n: Number of top features to return

        Returns:
            DataFrame with importance analysis
        """
        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance_dict.items()
        ])

        df = df.sort_values("importance", ascending=False)
        df["cumulative_importance"] = df["importance"].cumsum()
        df["rank"] = range(1, len(df) + 1)

        return df.head(top_n)

    def group_by_category(
        self,
        importance_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Group feature importance by category.

        Categories are determined by feature name prefixes
        (price_, vol_, volat_, trend_).

        Args:
            importance_dict: Feature -> importance mapping

        Returns:
            Dict of category -> total importance
        """
        categories = {
            "price": 0.0,
            "volume": 0.0,
            "volatility": 0.0,
            "trend": 0.0,
            "other": 0.0,
        }

        for feature, importance in importance_dict.items():
            if feature.startswith("price_"):
                categories["price"] += importance
            elif feature.startswith("vol_"):
                categories["volume"] += importance
            elif feature.startswith("volat_"):
                categories["volatility"] += importance
            elif feature.startswith("trend_"):
                categories["trend"] += importance
            else:
                categories["other"] += importance

        return categories

    def calculate_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Calculate permutation importance.

        More reliable than model-based importance but slower.

        Args:
            model: Trained model
            X: Features
            y: Labels
            n_repeats: Number of permutations per feature
            random_state: Random seed

        Returns:
            Dict of feature -> importance
        """
        np.random.seed(random_state)

        base_score = model.score(X, y)
        importance = {}

        for col in X.columns:
            scores = []

            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                score = model.score(X_permuted, y)
                scores.append(base_score - score)

            importance[col] = np.mean(scores)

        return importance


class TradingMetrics:
    """
    Trading-specific evaluation metrics.

    Evaluates model predictions in the context of
    trading performance, not just classification accuracy.
    """

    def __init__(
        self,
        regime_actions: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize trading metrics.

        Args:
            regime_actions: Mapping of regime -> action
        """
        self.regime_actions = regime_actions or {
            0: "GRID_NORMAL",      # RANGING
            1: "GRID_BULLISH",     # TRENDING_UP
            2: "GRID_PAUSE",       # TRENDING_DOWN
            3: "GRID_WIDE",        # HIGH_VOLATILITY
            4: "GRID_HALT",        # BREAKOUT
        }

    def calculate_action_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate accuracy at the action level.

        Some regime confusions are worse than others:
        - Confusing TRENDING_UP with TRENDING_DOWN is critical
        - Confusing RANGING with HIGH_VOLATILITY is less severe

        Args:
            y_true: True regimes
            y_pred: Predicted regimes

        Returns:
            Dict with action-level metrics
        """
        # Define action groups
        groups = {
            "trade": [0, 1],       # RANGING, TRENDING_UP
            "cautious": [2, 3],    # TRENDING_DOWN, HIGH_VOLATILITY
            "halt": [4],           # BREAKOUT
        }

        def get_group(regime):
            for group_name, regimes in groups.items():
                if regime in regimes:
                    return group_name
            return "unknown"

        true_groups = np.array([get_group(r) for r in y_true])
        pred_groups = np.array([get_group(r) for r in y_pred])

        group_accuracy = (true_groups == pred_groups).mean()

        # Critical errors: opposite direction
        critical_errors = (
            ((y_true == 1) & (y_pred == 2)) |  # UP predicted as DOWN
            ((y_true == 2) & (y_pred == 1))     # DOWN predicted as UP
        ).sum()

        return {
            "regime_accuracy": (y_true == y_pred).mean(),
            "group_accuracy": group_accuracy,
            "critical_errors": int(critical_errors),
            "critical_error_rate": critical_errors / len(y_true),
        }

    def simulate_trading_decisions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Simulate trading decisions based on predictions.

        Args:
            y_true: True regimes
            y_pred: Predicted regimes
            returns: Actual forward returns (optional)

        Returns:
            Dict with simulated trading metrics
        """
        results = {
            "total_periods": len(y_true),
            "decisions": {},
        }

        # Count decisions by predicted regime
        for regime_id in range(5):
            regime_name = MarketRegime(regime_id).name
            mask = y_pred == regime_id
            results["decisions"][regime_name] = {
                "count": int(mask.sum()),
                "percentage": float(mask.mean() * 100),
            }

            if returns is not None and mask.sum() > 0:
                results["decisions"][regime_name]["mean_return"] = float(
                    returns[mask].mean()
                )

        # Correct decisions
        correct = y_true == y_pred
        results["correct_decisions"] = int(correct.sum())
        results["correct_rate"] = float(correct.mean())

        return results
