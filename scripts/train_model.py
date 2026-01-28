#!/usr/bin/env python3
"""
Model Training CLI.

Trains XGBoost regime classifier on labeled data.

Usage:
    # Train with default settings
    python scripts/train_model.py --pair XBTUSD --timeframe 5m

    # Train with hyperparameter tuning
    python scripts/train_model.py --pair XBTUSD --timeframe 5m --tune

    # Train with walk-forward validation
    python scripts/train_model.py --pair XBTUSD --timeframe 5m --walk-forward

    # List trained models
    python scripts/train_model.py --list-models

    # Set production model
    python scripts/train_model.py --set-production MODEL_ID
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_preparation import DataPreparation, DataSplit
from src.models.regime_classifier import RegimeClassifier, XGBoostParams
from src.models.walk_forward import WalkForwardValidator
from src.models.model_evaluation import ModelEvaluator, FeatureImportanceAnalyzer
from src.models.model_registry import ModelRegistry
from src.regime.regime_labeler import MarketRegime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(
    pair: str,
    timeframe: str,
    labels_path: Path,
    models_path: Path,
    tune: bool = False,
    walk_forward: bool = False,
    n_folds: int = 5,
    verbose: bool = True,
) -> str:
    """
    Train a regime classifier.

    Args:
        pair: Trading pair
        timeframe: Timeframe
        labels_path: Path to labeled data
        models_path: Path for model storage
        tune: Whether to tune hyperparameters
        walk_forward: Whether to use walk-forward validation
        n_folds: Number of folds for walk-forward
        verbose: Print progress

    Returns:
        Model ID
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING REGIME CLASSIFIER")
    logger.info(f"Pair: {pair}, Timeframe: {timeframe}")
    logger.info("=" * 60)

    # Prepare data
    logger.info("\n1. Preparing data...")
    data_prep = DataPreparation(
        labels_path=labels_path,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        scale_features=True,
        scaler_type="robust",
    )

    data_split = data_prep.prepare_data(
        pair=pair,
        timeframe=timeframe,
        min_samples=500,
    )

    # Validate data
    validation = data_prep.validate_data_quality(data_split)
    if not validation["valid"]:
        logger.error(f"Data validation failed: {validation['errors']}")
        raise ValueError("Data validation failed")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(f"  {warning}")

    # Log class distribution
    class_dist = data_split.get_class_distribution()
    logger.info("\nClass distribution:")
    for split_name, dist in class_dist.items():
        total = sum(dist.values())
        logger.info(f"  {split_name}:")
        for class_id, count in sorted(dist.items()):
            regime_name = MarketRegime(class_id).name
            pct = count / total * 100
            logger.info(f"    {regime_name}: {count} ({pct:.1f}%)")

    # Compute class weights
    class_weights = data_prep.compute_class_weights(
        data_split.y_train,
        strategy="balanced",
    )
    logger.info(f"\nClass weights: {class_weights}")

    # Initialize classifier
    params = XGBoostParams(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=20,
    )

    classifier = RegimeClassifier(params=params)

    # Hyperparameter tuning
    if tune:
        logger.info("\n2. Tuning hyperparameters...")
        best_params, search_results = classifier.tune_hyperparameters(
            data_split=data_split,
            class_weights=class_weights,
            n_iter=30,  # Limit iterations
            verbose=verbose,
        )

        if best_params:
            classifier = RegimeClassifier(params=best_params)
            logger.info(f"Best params: {best_params.to_dict()}")
        else:
            logger.warning("Tuning failed, using default params")

    # Train model
    logger.info("\n3. Training model...")
    training_result = classifier.train(
        data_split=data_split,
        class_weights=class_weights,
        early_stopping=True,
        verbose=verbose,
    )

    logger.info(f"Training accuracy: {training_result.train_score:.4f}")
    logger.info(f"Validation accuracy: {training_result.val_score:.4f}")

    # Walk-forward validation
    wf_score = None
    if walk_forward:
        logger.info("\n4. Walk-forward validation...")

        # Combine train+val for walk-forward
        import pandas as pd
        X_combined = pd.concat([data_split.X_train, data_split.X_val])
        y_combined = pd.concat([data_split.y_train, data_split.y_val])
        ts_combined = pd.concat([data_split.timestamps_train, data_split.timestamps_val])

        # Get sample weights
        sample_weights = data_prep.get_sample_weights(y_combined, class_weights)

        # Create validator
        wf_validator = WalkForwardValidator(
            n_splits=n_folds,
            test_size=0.15,
            expanding=True,
        )

        # Create fresh model for walk-forward
        wf_model = classifier._create_model()

        wf_result = wf_validator.validate(
            model=wf_model,
            X=X_combined,
            y=y_combined,
            timestamps=ts_combined,
            sample_weight=sample_weights,
            verbose=verbose,
        )

        wf_score = wf_result.mean_test_score
        logger.info(
            f"Walk-forward score: {wf_score:.4f} (+/- {wf_result.std_test_score:.4f})"
        )

    # Evaluate on test set
    logger.info("\n5. Evaluating on test set...")
    evaluator = ModelEvaluator()

    y_pred = classifier.predict(data_split.X_test)
    y_proba = classifier.predict_proba(data_split.X_test)

    evaluation = evaluator.evaluate(
        y_true=data_split.y_test.values,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    logger.info(evaluation.summary())

    # Feature importance
    logger.info("\n6. Feature importance (top 15):")
    top_features = classifier.get_top_features(n=15)
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"  {i:2}. {feature}: {importance:.4f}")

    # Group importance by category
    importance_analyzer = FeatureImportanceAnalyzer(data_split.feature_names)
    category_importance = importance_analyzer.group_by_category(
        training_result.feature_importance
    )
    logger.info("\nImportance by category:")
    for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {category}: {importance:.4f}")

    # Save model
    logger.info("\n7. Saving model...")
    registry = ModelRegistry(models_path)

    model_id = registry.save_model(
        classifier=classifier,
        pair=pair,
        timeframe=timeframe,
        train_samples=data_split.train_size,
        val_samples=data_split.val_size,
        test_samples=data_split.test_size,
        evaluation=evaluation,
        scaler=data_split.scaler,
        description=f"XGBoost regime classifier for {pair} {timeframe}",
        walk_forward_score=wf_score,
    )

    logger.info(f"Saved model: {model_id}")
    logger.info("=" * 60)

    return model_id


def list_models(
    models_path: Path,
    pair: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> None:
    """List all trained models."""
    registry = ModelRegistry(models_path)
    models = registry.list_models(pair=pair, timeframe=timeframe)

    if not models:
        logger.info("No models found")
        return

    print("\n" + "=" * 80)
    print("TRAINED MODELS")
    print("=" * 80)

    for model in models:
        prod_marker = " [PRODUCTION]" if model.is_production else ""
        print(f"\n{model.model_id}{prod_marker}")
        print(f"  Pair: {model.pair}, Timeframe: {model.timeframe}")
        print(f"  Created: {model.created_at}")
        print(f"  Test Accuracy: {model.test_accuracy:.4f}, Macro F1: {model.macro_f1:.4f}")
        if model.walk_forward_score:
            print(f"  Walk-Forward: {model.walk_forward_score:.4f}")
        print(f"  Features: {model.feature_count}, Samples: {model.train_samples}/{model.val_samples}/{model.test_samples}")

    print("\n" + "=" * 80)


def set_production(
    models_path: Path,
    model_id: str,
) -> None:
    """Set a model as production."""
    registry = ModelRegistry(models_path)
    registry.set_production_model(model_id)
    logger.info(f"Set {model_id} as production model")


def compare_models(
    models_path: Path,
    model_ids: list,
) -> None:
    """Compare multiple models."""
    registry = ModelRegistry(models_path)
    comparison = registry.compare_models(model_ids)

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train regime classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train model
    python scripts/train_model.py --pair XBTUSD --timeframe 5m

    # Train with tuning and walk-forward
    python scripts/train_model.py --pair XBTUSD --timeframe 5m --tune --walk-forward

    # List models
    python scripts/train_model.py --list-models

    # Set production model
    python scripts/train_model.py --set-production MODEL_ID
        """,
    )

    # Training arguments
    parser.add_argument(
        "--pair",
        type=str,
        default="XBTUSD",
        help="Trading pair (default: XBTUSD)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="Timeframe (default: 5m)",
    )

    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/labels"),
        help="Path to labeled data (default: data/labels)",
    )

    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path("data/models"),
        help="Path for model storage (default: data/models)",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune hyperparameters",
    )

    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for walk-forward (default: 5)",
    )

    # Management arguments
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all trained models",
    )

    parser.add_argument(
        "--set-production",
        type=str,
        metavar="MODEL_ID",
        help="Set a model as production",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="MODEL_ID",
        help="Compare multiple models",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle commands
    if args.list_models:
        list_models(
            models_path=args.models_path,
            pair=args.pair if args.pair != "XBTUSD" else None,
            timeframe=args.timeframe if args.timeframe != "5m" else None,
        )
        return 0

    if args.set_production:
        set_production(args.models_path, args.set_production)
        return 0

    if args.compare:
        compare_models(args.models_path, args.compare)
        return 0

    # Train model
    try:
        model_id = train_model(
            pair=args.pair,
            timeframe=args.timeframe,
            labels_path=args.labels_path,
            models_path=args.models_path,
            tune=args.tune,
            walk_forward=args.walk_forward,
            n_folds=args.n_folds,
            verbose=args.verbose,
        )

        logger.info(f"\nTraining complete! Model ID: {model_id}")
        logger.info("Use --set-production to deploy this model")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
