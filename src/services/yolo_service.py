"""YOLO service for training and evaluation."""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class YOLOService:
    """Service for YOLO training operations."""

    def __init__(self):
        """Initialize YOLO service."""
        pass

    def generate_dataset_yaml(
        self,
        train_images: List[Path],
        val_images: List[Path],
        output_dir: Path
    ) -> Path:
        """Generate YOLO OBB dataset YAML configuration.

        Args:
            train_images: List of training image paths
            val_images: List of validation image paths
            output_dir: Output directory (dataset root)

        Returns:
            Path to generated YAML file

        Raises:
            ValueError: If no images provided or output_dir invalid
        """
        if not train_images:
            raise ValueError("No training images provided")

        if not output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")

        # Detect actual image directory name (could be 'images', 'rgb', etc.)
        import shutil
        images_dir = None
        for possible_name in ['rgb', 'images', 'img']:
            candidate = output_dir / possible_name
            if candidate.exists() and candidate.is_dir():
                images_dir = candidate
                break

        if not images_dir:
            raise ValueError(f"找不到圖片目錄（嘗試過: rgb, images, img）在 {output_dir}")

        # Ensure labels directory exists (YOLO OBB expects labels in 'labels' dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        # Copy OBB labels from obb_labels (or labels_obb) to labels directory
        # YOLO OBB expects labels to be in a 'labels' directory that mirrors images
        for possible_obb_dir in ['labels_obb', 'obb_labels']:
            obb_labels_dir = output_dir / possible_obb_dir
            if obb_labels_dir.exists():
                for obb_file in obb_labels_dir.glob("*.txt"):
                    target_file = labels_dir / obb_file.name
                    shutil.copy2(obb_file, target_file)
                break

        # Get image directory name relative to output_dir
        images_dir_name = images_dir.name

        # Create train.txt with ABSOLUTE paths
        # YOLO requires absolute paths when using .txt files
        train_txt = output_dir / "train.txt"
        with open(train_txt, 'w') as f:
            for img in train_images:
                # Write absolute path to each image
                f.write(f"{img.absolute()}\n")

        # Create val.txt with absolute paths
        val_txt = output_dir / "val.txt"
        with open(val_txt, 'w') as f:
            for img in val_images:
                # Write absolute path to each image
                f.write(f"{img.absolute()}\n")

        # Generate YAML configuration
        # YOLO will look for labels by replacing 'images' with 'labels' in the path
        # and .jpg/.png with .txt
        yaml_content = {
            'path': str(output_dir.absolute()),
            'train': str(train_txt.absolute()),
            'val': str(val_txt.absolute()),
            'names': {
                0: 'object'
            }
        }

        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        return yaml_path

    def parse_training_results(self, results_dir: Path) -> Dict[str, Any]:
        """Parse training results from YOLO output directory.

        Args:
            results_dir: Training results directory

        Returns:
            Dictionary containing training results
        """
        results = {
            'best_model': None,
            'last_model': None,
            'results_csv': None,
            'confusion_matrix': None,
            'pr_curve': None
        }

        if not results_dir.exists():
            return results

        # Find model files
        weights_dir = results_dir / 'weights'
        if weights_dir.exists():
            best_pt = weights_dir / 'best.pt'
            last_pt = weights_dir / 'last.pt'
            if best_pt.exists():
                results['best_model'] = str(best_pt)
            if last_pt.exists():
                results['last_model'] = str(last_pt)

        # Find result files
        results_csv = results_dir / 'results.csv'
        if results_csv.exists():
            results['results_csv'] = str(results_csv)

        confusion_matrix = results_dir / 'confusion_matrix.png'
        if confusion_matrix.exists():
            results['confusion_matrix'] = str(confusion_matrix)

        pr_curve = results_dir / 'PR_curve.png'
        if pr_curve.exists():
            results['pr_curve'] = str(pr_curve)

        return results

    def parse_evaluation_results(self, results) -> Dict[str, float]:
        """Parse evaluation results from YOLO validator.

        Args:
            results: YOLO validation results object

        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

        try:
            # Try to extract box metrics (for OBB)
            if hasattr(results, 'box') and results.box is not None:
                metrics['mAP50'] = float(results.box.map50)
                metrics['mAP50-95'] = float(results.box.map)
                metrics['precision'] = float(results.box.mp)
                metrics['recall'] = float(results.box.mr)
            # Try OBB metrics
            elif hasattr(results, 'obb') and results.obb is not None:
                metrics['mAP50'] = float(results.obb.map50)
                metrics['mAP50-95'] = float(results.obb.map)
                metrics['precision'] = float(results.obb.mp)
                metrics['recall'] = float(results.obb.mr)
        except Exception as e:
            print(f"Warning: Failed to parse evaluation results: {e}")

        return metrics
