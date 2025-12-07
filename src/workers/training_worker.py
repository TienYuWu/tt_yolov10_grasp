"""Training worker for YOLO OBB model."""

from pathlib import Path
from typing import Dict, Any

import torch
from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    """Worker thread for YOLO training."""

    # Signals
    epoch_started = Signal(int, int)  # current_epoch, total_epochs
    epoch_completed = Signal(int, dict)  # epoch, metrics: {box_loss, lr, ...}
    training_completed = Signal(dict)  # final_metrics: {best_model, save_dir}
    training_failed = Signal(str)  # error_message
    evaluation_completed = Signal(dict)  # eval_results: {mAP50, mAP, ...}
    log_message = Signal(str)  # Log message for display

    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        project: str = "runs/obb",
        name: str = "train",
        augmentation_config: Dict[str, Any] = None
    ):
        """Initialize training worker.

        Args:
            model_path: Path to YOLO model file
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size
            project: Project directory for results
            name: Run name
            augmentation_config: Dict of YOLO augmentation parameters (optional)
                If None or empty, YOLO defaults will be used.
                Supported parameters: degrees, translate, scale, shear, perspective,
                flipud, fliplr, hsv_h, hsv_s, hsv_v, mosaic, mixup, copy_paste, erasing
        """
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.project = project
        self.name = name
        self.augmentation_config = augmentation_config or {}
        self.model = None

    def run(self):
        """Execute training."""
        try:
            # Import YOLO
            try:
                from ultralytics import YOLO
            except ImportError:
                self.training_failed.emit(
                    "未安裝 ultralytics 套件\n\n請執行: pip install ultralytics"
                )
                return

            self.log_message.emit("載入模型...")

            # Load model
            self.model = YOLO(self.model_path)

            self.log_message.emit(f"模型已載入: {self.model_path}")

            # Detect device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log_message.emit(f"使用裝置: {device}")

            # Training arguments
            train_args = {
                'data': self.data_yaml,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': 640,
                'amp': True,
                'device': device,
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.1,
                'weight_decay': 0.0005,
                'project': self.project,
                'name': self.name,
                'save': True,
                'save_period': 10,
                'workers': 4,
                'patience': 100,
                'plots': True,
                'val': True,
                'verbose': True
            }

            # Merge augmentation config (will override defaults if provided)
            if self.augmentation_config:
                train_args.update(self.augmentation_config)
                self.log_message.emit(
                    f"使用自訂資料強化參數 ({len(self.augmentation_config)} 項)"
                )
            else:
                self.log_message.emit("使用 YOLO 預設資料強化參數")

            self.log_message.emit("開始訓練...")

            # Add custom callback for progress tracking
            def on_train_epoch_end(trainer):
                """Callback for epoch end."""
                epoch = trainer.epoch + 1  # YOLO epochs are 0-indexed

                # Extract metrics
                metrics = {}

                # Method 1: Try to get averaged loss from loss_items (recommended)
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        # loss_items contains averaged losses for the epoch
                        loss_vals = trainer.loss_items
                        if hasattr(loss_vals, 'cpu'):
                            loss_vals = loss_vals.cpu().numpy()
                        # Sum all loss components
                        total = float(sum(loss_vals))
                        if total < 100:  # Sanity check - loss should typically be < 100
                            metrics['total_loss'] = total
                    except Exception as e:
                        pass

                # Method 2: Try from label_loss_items (averaged loss components)
                if 'total_loss' not in metrics and hasattr(trainer, 'label_loss_items'):
                    try:
                        loss_items = trainer.label_loss_items()
                        if loss_items:
                            metrics['total_loss'] = sum(float(x) for x in loss_items if x is not None)
                    except:
                        pass

                # Method 3: Try from metrics/results_dict
                if 'total_loss' not in metrics:
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        try:
                            # Get results_dict from metrics
                            if hasattr(trainer.metrics, 'results_dict'):
                                results = trainer.metrics.results_dict
                            elif isinstance(trainer.metrics, dict):
                                results = trainer.metrics
                            else:
                                results = {}

                            # Try various loss keys
                            for key in ['train/loss', 'metrics/loss', 'loss']:
                                if key in results:
                                    metrics['total_loss'] = float(results[key])
                                    break
                        except:
                            pass

                # Method 4: Try from tloss attribute (training loss)
                if 'total_loss' not in metrics and hasattr(trainer, 'tloss'):
                    try:
                        if hasattr(trainer.tloss, 'item'):
                            metrics['total_loss'] = float(trainer.tloss.item())
                        else:
                            metrics['total_loss'] = float(trainer.tloss)
                    except:
                        pass

                # Debug: Print what we found
                if 'total_loss' in metrics:
                    self.log_message.emit(f"[DEBUG] Extracted loss: {metrics['total_loss']:.4f}")

                # If still no total_loss, set to 0
                if 'total_loss' not in metrics:
                    metrics['total_loss'] = 0.0
                    self.log_message.emit("[DEBUG] No loss found, using 0.0")

                # Emit progress
                self.epoch_completed.emit(epoch, metrics)
                self.log_message.emit(
                    f"[Epoch {epoch}/{self.epochs}] "
                    f"total_loss: {metrics.get('total_loss', 0):.4f}"
                )

            # Register callback using YOLO's callback manager
            self.model.add_callback('on_train_epoch_end', on_train_epoch_end)

            # Start training
            results = self.model.train(**train_args)

            # Parse training results
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(self.project) / self.name
            best_model = save_dir / 'weights' / 'best.pt'

            training_results = {
                'best_model': str(best_model) if best_model.exists() else None,
                'save_dir': str(save_dir),
                'success': True
            }

            self.training_completed.emit(training_results)
            self.log_message.emit("✅ 訓練完成！")

            # Perform evaluation
            self.log_message.emit("開始評估模型...")

            try:
                eval_results = self.model.val(data=self.data_yaml)

                # Parse evaluation results
                eval_metrics = self._parse_eval_results(eval_results)
                eval_metrics['results_dir'] = str(save_dir)

                self.evaluation_completed.emit(eval_metrics)
                self.log_message.emit(
                    f"評估完成 - mAP50: {eval_metrics.get('mAP50', 0):.4f}"
                )

            except Exception as e:
                self.log_message.emit(f"⚠️  評估失敗: {str(e)}")
                # Still emit empty results
                self.evaluation_completed.emit({
                    'mAP50': 0.0,
                    'mAP50-95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'results_dir': str(save_dir)
                })

        except Exception as e:
            error_msg = f"訓練失敗: {str(e)}\n\n"

            if 'out of memory' in str(e).lower():
                error_msg += "建議：\n"
                error_msg += "1. 降低 Batch Size\n"
                error_msg += "2. 減小圖片尺寸\n"
                error_msg += "3. 使用較小的模型"

            self.training_failed.emit(error_msg)

    def _parse_eval_results(self, results) -> Dict[str, float]:
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
            # Try results_dict
            elif hasattr(results, 'results_dict'):
                rd = results.results_dict
                metrics['mAP50'] = float(rd.get('metrics/mAP50(B)', 0))
                metrics['mAP50-95'] = float(rd.get('metrics/mAP50-95(B)', 0))
                metrics['precision'] = float(rd.get('metrics/precision(B)', 0))
                metrics['recall'] = float(rd.get('metrics/recall(B)', 0))
        except Exception as e:
            print(f"Warning: Failed to parse evaluation results: {e}")

        return metrics
