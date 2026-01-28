"""
Model Registry for Regime Classifier.

Handles model versioning, storage, and loading:
- Save trained models with metadata
- Version control with timestamps
- Model comparison and selection
- Production model management
"""

import json
import logging
import pickle
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .regime_classifier import RegimeClassifier, XGBoostParams
from .model_evaluation import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    version: str
    created_at: str
    pair: str
    timeframe: str
    model_type: str

    # Training info
    train_samples: int
    val_samples: int
    test_samples: int
    feature_count: int

    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    macro_f1: float

    # Configuration
    params: Dict[str, Any]
    feature_names: List[str]
    class_names: List[str]

    # Optional
    description: str = ""
    is_production: bool = False
    walk_forward_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Registry for managing trained models.

    Features:
    - Store models with versioned naming
    - Track model metadata and performance
    - Compare model versions
    - Manage production model
    """

    MODELS_SUBDIR = "versions"
    PRODUCTION_LINK = "production"
    METADATA_FILE = "metadata.json"
    MODEL_FILE = "model.pkl"
    SCALER_FILE = "scaler.pkl"
    EVALUATION_FILE = "evaluation.json"

    def __init__(
        self,
        registry_path: Path,
    ):
        """
        Initialize model registry.

        Args:
            registry_path: Base path for model storage
        """
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / self.MODELS_SUBDIR
        self.models_path.mkdir(parents=True, exist_ok=True)

    def _generate_model_id(
        self,
        pair: str,
        timeframe: str,
    ) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{pair}_{timeframe}_{timestamp}"

    def _get_model_path(self, model_id: str) -> Path:
        """Get path for a model."""
        return self.models_path / model_id

    def save_model(
        self,
        classifier: RegimeClassifier,
        pair: str,
        timeframe: str,
        train_samples: int,
        val_samples: int,
        test_samples: int,
        evaluation: EvaluationResult,
        scaler: Optional[Any] = None,
        description: str = "",
        walk_forward_score: Optional[float] = None,
    ) -> str:
        """
        Save a trained model to the registry.

        Args:
            classifier: Trained RegimeClassifier
            pair: Trading pair
            timeframe: Timeframe
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
            evaluation: Evaluation results
            scaler: Feature scaler (optional)
            description: Model description
            walk_forward_score: Walk-forward validation score

        Returns:
            Model ID
        """
        # Generate ID and create directory
        model_id = self._generate_model_id(pair, timeframe)
        model_path = self._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version="1.0",
            created_at=datetime.utcnow().isoformat(),
            pair=pair,
            timeframe=timeframe,
            model_type="xgboost",
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            feature_count=len(classifier.feature_names or []),
            train_accuracy=classifier._training_result.train_score if classifier._training_result else 0.0,
            val_accuracy=classifier._training_result.val_score if classifier._training_result else 0.0,
            test_accuracy=evaluation.accuracy,
            macro_f1=evaluation.macro_f1,
            params=classifier.params.to_dict(),
            feature_names=classifier.feature_names or [],
            class_names=[cm.class_name for cm in evaluation.class_metrics],
            description=description,
            walk_forward_score=walk_forward_score,
        )

        # Save metadata
        metadata_path = model_path / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Save model
        model_file = model_path / self.MODEL_FILE
        with open(model_file, "wb") as f:
            pickle.dump(classifier.model, f)

        # Save scaler if provided
        if scaler is not None:
            scaler_file = model_path / self.SCALER_FILE
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)

        # Save evaluation
        eval_file = model_path / self.EVALUATION_FILE
        with open(eval_file, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        logger.info(f"Saved model {model_id} to {model_path}")
        return model_id

    def load_model(
        self,
        model_id: str,
    ) -> tuple:
        """
        Load a model from the registry.

        Args:
            model_id: Model ID to load

        Returns:
            Tuple of (classifier, metadata, scaler)
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")

        # Load metadata
        metadata_path = model_path / self.METADATA_FILE
        with open(metadata_path, "r") as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        # Create classifier with params
        params = XGBoostParams(**metadata.params)
        classifier = RegimeClassifier(params=params)
        classifier.feature_names = metadata.feature_names

        # Load model
        model_file = model_path / self.MODEL_FILE
        with open(model_file, "rb") as f:
            classifier.model = pickle.load(f)

        # Load scaler if exists
        scaler = None
        scaler_file = model_path / self.SCALER_FILE
        if scaler_file.exists():
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

        logger.info(f"Loaded model {model_id}")
        return classifier, metadata, scaler

    def list_models(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """
        List all models in registry.

        Args:
            pair: Filter by pair (optional)
            timeframe: Filter by timeframe (optional)

        Returns:
            List of model metadata
        """
        models = []

        for model_dir in self.models_path.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_path = model_dir / self.METADATA_FILE
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = ModelMetadata.from_dict(json.load(f))

                # Apply filters
                if pair and metadata.pair != pair:
                    continue
                if timeframe and metadata.timeframe != timeframe:
                    continue

                models.append(metadata)

            except Exception as e:
                logger.warning(f"Failed to load metadata from {model_dir}: {e}")

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def get_best_model(
        self,
        pair: str,
        timeframe: str,
        metric: str = "macro_f1",
    ) -> Optional[ModelMetadata]:
        """
        Get the best model for a pair/timeframe.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            metric: Metric to compare (default: macro_f1)

        Returns:
            Best model metadata or None
        """
        models = self.list_models(pair=pair, timeframe=timeframe)

        if not models:
            return None

        # Sort by metric
        if metric == "macro_f1":
            models.sort(key=lambda m: m.macro_f1, reverse=True)
        elif metric == "test_accuracy":
            models.sort(key=lambda m: m.test_accuracy, reverse=True)
        elif metric == "walk_forward_score":
            models.sort(key=lambda m: m.walk_forward_score or 0, reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return models[0]

    def set_production_model(
        self,
        model_id: str,
    ) -> None:
        """
        Set a model as the production model.

        Args:
            model_id: Model ID to set as production
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")

        # Update metadata
        metadata_path = model_path / self.METADATA_FILE
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata["is_production"] = True

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Remove production flag from other models
        for other_dir in self.models_path.iterdir():
            if other_dir.name == model_id or not other_dir.is_dir():
                continue

            other_metadata_path = other_dir / self.METADATA_FILE
            if other_metadata_path.exists():
                with open(other_metadata_path, "r") as f:
                    other_metadata = json.load(f)

                if other_metadata.get("is_production"):
                    other_metadata["is_production"] = False
                    with open(other_metadata_path, "w") as f:
                        json.dump(other_metadata, f, indent=2)

        # Create/update production symlink
        production_link = self.registry_path / self.PRODUCTION_LINK
        if production_link.exists():
            production_link.unlink()

        production_link.symlink_to(model_path)

        logger.info(f"Set {model_id} as production model")

    def get_production_model(self) -> Optional[tuple]:
        """
        Get the current production model.

        Returns:
            Tuple of (classifier, metadata, scaler) or None
        """
        production_link = self.registry_path / self.PRODUCTION_LINK

        if not production_link.exists():
            # Try to find model marked as production
            models = self.list_models()
            for model in models:
                if model.is_production:
                    return self.load_model(model.model_id)
            return None

        model_id = production_link.resolve().name
        return self.load_model(model_id)

    def compare_models(
        self,
        model_ids: List[str],
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            DataFrame with comparison
        """
        rows = []

        for model_id in model_ids:
            try:
                _, metadata, _ = self.load_model(model_id)
                rows.append({
                    "model_id": model_id,
                    "created_at": metadata.created_at,
                    "train_accuracy": metadata.train_accuracy,
                    "val_accuracy": metadata.val_accuracy,
                    "test_accuracy": metadata.test_accuracy,
                    "macro_f1": metadata.macro_f1,
                    "walk_forward": metadata.walk_forward_score,
                    "features": metadata.feature_count,
                    "is_production": metadata.is_production,
                })
            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")

        return pd.DataFrame(rows)

    def delete_model(
        self,
        model_id: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID to delete
            force: Force delete even if production

        Returns:
            True if deleted
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            logger.warning(f"Model {model_id} not found")
            return False

        # Check if production
        metadata_path = model_path / self.METADATA_FILE
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if metadata.get("is_production") and not force:
            raise ValueError(
                f"Cannot delete production model {model_id}. Use force=True."
            )

        # Delete
        shutil.rmtree(model_path)

        # Remove production link if pointing to this model
        production_link = self.registry_path / self.PRODUCTION_LINK
        if production_link.exists() and production_link.resolve().name == model_id:
            production_link.unlink()

        logger.info(f"Deleted model {model_id}")
        return True

    def cleanup_old_models(
        self,
        keep_n: int = 5,
        keep_production: bool = True,
    ) -> int:
        """
        Clean up old models, keeping the most recent.

        Args:
            keep_n: Number of models to keep per pair/timeframe
            keep_production: Keep production models

        Returns:
            Number of models deleted
        """
        models = self.list_models()

        # Group by pair/timeframe
        groups: Dict[str, List[ModelMetadata]] = {}
        for model in models:
            key = f"{model.pair}_{model.timeframe}"
            if key not in groups:
                groups[key] = []
            groups[key].append(model)

        deleted = 0

        for group_models in groups.values():
            # Sort by creation (newest first)
            group_models.sort(key=lambda m: m.created_at, reverse=True)

            # Delete old ones
            for model in group_models[keep_n:]:
                if keep_production and model.is_production:
                    continue

                try:
                    self.delete_model(model.model_id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {model.model_id}: {e}")

        return deleted

    def export_model(
        self,
        model_id: str,
        export_path: Path,
    ) -> None:
        """
        Export a model to a standalone directory.

        Args:
            model_id: Model ID to export
            export_path: Destination path
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy all files
        shutil.copytree(model_path, export_path, dirs_exist_ok=True)

        logger.info(f"Exported {model_id} to {export_path}")

    def import_model(
        self,
        import_path: Path,
    ) -> str:
        """
        Import a model from external directory.

        Args:
            import_path: Path to model directory

        Returns:
            Imported model ID
        """
        import_path = Path(import_path)

        if not import_path.exists():
            raise ValueError(f"Import path {import_path} not found")

        # Load and validate metadata
        metadata_path = import_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise ValueError("No metadata.json found in import path")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Generate new ID to avoid conflicts
        model_id = self._generate_model_id(
            metadata.get("pair", "unknown"),
            metadata.get("timeframe", "unknown"),
        )

        # Update metadata with new ID
        metadata["model_id"] = model_id
        metadata["is_production"] = False

        # Copy to registry
        model_path = self._get_model_path(model_id)
        shutil.copytree(import_path, model_path)

        # Save updated metadata
        with open(model_path / self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Imported model as {model_id}")
        return model_id
