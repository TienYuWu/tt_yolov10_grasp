"""Model manager for tracking trained models."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ModelManager:
    """Manager for trained model registry."""

    def __init__(self, registry_dir: Optional[Path] = None):
        """Initialize model manager.

        Args:
            registry_dir: Directory for registry file (defaults to ~/.smart_label)
        """
        if registry_dir is None:
            registry_dir = Path.home() / ".smart_label"

        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.registry_dir / "model_registry.json"

        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model registry: {e}")
                self.registry = {'models': []}
        else:
            self.registry = {'models': []}

    def _save_registry(self) -> None:
        """Save model registry to file."""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Failed to save model registry: {e}")

    def register_model(self, model_info: Dict) -> str:
        """Register a new trained model.

        Args:
            model_info: Model information dictionary
                Required keys: name, model_path, training_dir
                Optional keys: metrics, config

        Returns:
            Model ID

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = ['name', 'model_path', 'training_dir']
        for key in required_keys:
            if key not in model_info:
                raise ValueError(f"Missing required key: {key}")

        # Generate unique model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build model record
        model_record = {
            'id': model_id,
            'name': model_info['name'],
            'model_path': str(model_info['model_path']),
            'training_dir': str(model_info['training_dir']),
            'created_at': datetime.now().isoformat(),
            'metrics': model_info.get('metrics', {}),
            'config': model_info.get('config', {})
        }

        # Add to registry
        self.registry['models'].append(model_record)
        self._save_registry()

        return model_id

    def list_models(self) -> List[Dict]:
        """List all registered models.

        Returns:
            List of model dictionaries
        """
        return self.registry['models'].copy()

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model information by ID.

        Args:
            model_id: Model ID

        Returns:
            Model dictionary or None if not found
        """
        for model in self.registry['models']:
            if model['id'] == model_id:
                return model.copy()
        return None

    def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """Delete a model from registry and optionally delete files.

        Args:
            model_id: Model ID to delete
            delete_files: Whether to delete model files and training dir

        Returns:
            True if model was deleted, False if not found
        """
        model = self.get_model(model_id)
        if not model:
            return False

        if delete_files:
            # Delete model file
            model_path = Path(model['model_path'])
            if model_path.exists():
                try:
                    model_path.unlink()
                    print(f"Deleted model file: {model_path}")
                except Exception as e:
                    print(f"Warning: Failed to delete model file: {e}")

            # Delete training directory
            training_dir = Path(model['training_dir'])
            if training_dir.exists():
                try:
                    shutil.rmtree(training_dir)
                    print(f"Deleted training directory: {training_dir}")
                except Exception as e:
                    print(f"Warning: Failed to delete training directory: {e}")

        # Remove from registry
        self.registry['models'] = [
            m for m in self.registry['models'] if m['id'] != model_id
        ]
        self._save_registry()

        return True

    def update_model_metrics(self, model_id: str, metrics: Dict) -> bool:
        """Update model metrics.

        Args:
            model_id: Model ID
            metrics: Metrics dictionary to update

        Returns:
            True if updated, False if model not found
        """
        for model in self.registry['models']:
            if model['id'] == model_id:
                model['metrics'].update(metrics)
                self._save_registry()
                return True
        return False
