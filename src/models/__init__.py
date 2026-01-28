"""
ML Models for Regime Classification.

Provides machine learning models for classifying market regimes:
- XGBoost-based regime classifier
- Walk-forward validation
- Model evaluation and metrics
- Model versioning and registry

Usage:
    from src.models import (
        DataPreparation,
        RegimeClassifier,
        ModelEvaluator,
        ModelRegistry,
    )

    # Prepare data
    data_prep = DataPreparation(labels_path)
    data_split = data_prep.prepare_data("XBTUSD", "5m")

    # Train model
    classifier = RegimeClassifier()
    result = classifier.train(data_split)

    # Evaluate
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate(y_true, y_pred)

    # Save to registry
    registry = ModelRegistry(models_path)
    model_id = registry.save_model(classifier, ...)
"""

from .data_preparation import (
    DataPreparation,
    DataSplit,
)

from .regime_classifier import (
    RegimeClassifier,
    EnsembleRegimeClassifier,
    XGBoostParams,
    TrainingResult,
)

from .walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
    WalkForwardFold,
    PurgedWalkForward,
    CombinatorialPurgedCV,
)

from .model_evaluation import (
    ModelEvaluator,
    EvaluationResult,
    ClassMetrics,
    FeatureImportanceAnalyzer,
    TradingMetrics,
)

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
)

__all__ = [
    # Data preparation
    "DataPreparation",
    "DataSplit",
    # Classifiers
    "RegimeClassifier",
    "EnsembleRegimeClassifier",
    "XGBoostParams",
    "TrainingResult",
    # Walk-forward validation
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardFold",
    "PurgedWalkForward",
    "CombinatorialPurgedCV",
    # Evaluation
    "ModelEvaluator",
    "EvaluationResult",
    "ClassMetrics",
    "FeatureImportanceAnalyzer",
    "TradingMetrics",
    # Registry
    "ModelRegistry",
    "ModelMetadata",
]
