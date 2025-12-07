"""Training tab for YOLO OBB model training."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QLineEdit, QSpinBox, QTextEdit, QSplitter,
    QFileDialog, QMessageBox, QComboBox, QTabWidget
)
from PySide6.QtCore import Qt, Slot

from ..core import ServiceContainer
from ..services import YOLOService, ModelManager
from ..workers import TrainingWorker
from ..config import AppConfig
from .dataset_split_dialog import DatasetSplitDialog
from .plot_widget import PlotWidget
from .eval_widget import EvalWidget
from .model_manager_widget import ModelManagerWidget


class TrainingTab(QWidget):
    """Training tab for YOLO OBB training."""

    def __init__(self, config: AppConfig, service_container: ServiceContainer, parent=None):
        """Initialize training tab.

        Args:
            config: Application configuration
            service_container: Service container
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        self.service_container = service_container
        self.yolo_service: YOLOService = service_container.get('yolo_service')
        self.model_manager: ModelManager = service_container.get('model_manager')

        # Training state
        self.training_worker: Optional[TrainingWorker] = None
        self.dataset_yaml: Optional[Path] = None
        self.training_in_progress = False

        # Augmentation configuration (empty dict = YOLO defaults)
        self.augmentation_config = {}

        self._build_ui()

    def _build_ui(self):
        """Build the training tab UI."""
        main_layout = QHBoxLayout()

        # Left panel: Configuration and control
        left_panel = self._create_control_panel()

        # Right panel: Progress and results
        right_panel = self._create_results_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_control_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Dataset configuration
        dataset_group = self._create_dataset_group()
        layout.addWidget(dataset_group)

        # Training parameters
        params_group = self._create_params_group()
        layout.addWidget(params_group)

        # Control buttons
        control_group = self._create_control_group()
        layout.addWidget(control_group)

        # Model manager
        model_manager_group = self._create_model_manager_section()
        layout.addWidget(model_manager_group, 1)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _create_dataset_group(self) -> QGroupBox:
        """Create dataset configuration group."""
        group = QGroupBox("資料集配置")
        layout = QVBoxLayout()

        # Output directory selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("資料集目錄:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setPlaceholderText("選擇標註輸出目錄...")
        dir_layout.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("📁 瀏覽")
        browse_btn.clicked.connect(self._on_browse_output_dir)
        dir_layout.addWidget(browse_btn)

        layout.addLayout(dir_layout)

        # Dataset split button
        split_btn = QPushButton("📊 配置訓練/驗證集分割")
        split_btn.clicked.connect(self._on_configure_split)
        layout.addWidget(split_btn)

        # Dataset info
        self.dataset_info_label = QLabel("（尚未配置資料集）")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)

        group.setLayout(layout)
        return group

    def _create_params_group(self) -> QGroupBox:
        """Create training parameters group."""
        group = QGroupBox("訓練參數")
        layout = QVBoxLayout()

        # Model info (fixed model path)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))

        # Fixed model path
        self.model_path = Path(__file__).parent.parent / "yolov10n-obb.pt"

        model_info_label = QLabel("yolov10n-obb.pt")
        model_info_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(model_info_label)

        # Show if model exists
        if self.model_path.exists():
            status_label = QLabel("✅")
            status_label.setToolTip(f"模型路徑: {self.model_path}")
        else:
            status_label = QLabel("❌")
            status_label.setToolTip(f"找不到模型: {self.model_path}")
            status_label.setStyleSheet("color: red;")
        model_layout.addWidget(status_label)

        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("訓練輪數:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        layout.addLayout(epochs_layout)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        layout.addLayout(batch_layout)

        # Augmentation settings button
        aug_btn_layout = QHBoxLayout()
        self.aug_btn = QPushButton("⚙️ 資料強化設定")
        self.aug_btn.setToolTip("設定訓練時的資料強化參數 (旋轉、翻轉、色彩調整等)")
        self.aug_btn.clicked.connect(self._on_augmentation_settings)
        self.aug_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-weight: bold;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        aug_btn_layout.addWidget(self.aug_btn)
        aug_btn_layout.addStretch()
        layout.addLayout(aug_btn_layout)

        # Augmentation status label
        self.aug_status_label = QLabel("（使用 YOLO 預設強化參數）")
        self.aug_status_label.setWordWrap(True)
        self.aug_status_label.setStyleSheet("color: #6c757d; font-size: 9pt;")
        layout.addWidget(self.aug_status_label)

        group.setLayout(layout)
        return group

    def _create_control_group(self) -> QGroupBox:
        """Create training control group."""
        group = QGroupBox("訓練控制")
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ 開始訓練")
        self.start_btn.clicked.connect(self._on_start_training)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ 停止訓練")
        self.stop_btn.clicked.connect(self._on_stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Status
        self.status_label = QLabel("狀態: 就緒")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        group.setLayout(layout)
        return group

    def _create_model_manager_section(self) -> QGroupBox:
        """Create model manager section."""
        group = QGroupBox("模型管理")
        layout = QVBoxLayout()

        self.model_manager_widget = ModelManagerWidget(self.model_manager, self)
        layout.addWidget(self.model_manager_widget)

        group.setLayout(layout)
        return group

    def _create_results_panel(self) -> QWidget:
        """Create right results panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Tab widget for different views
        self.results_tabs = QTabWidget()

        # Training log tab
        log_widget = self._create_log_widget()
        self.results_tabs.addTab(log_widget, "📝 訓練日誌")

        # Training curves tab
        curves_widget = self._create_curves_widget()
        self.results_tabs.addTab(curves_widget, "📈 訓練曲線")

        # Evaluation results tab
        eval_widget = self._create_eval_widget()
        self.results_tabs.addTab(eval_widget, "📊 評估結果")

        layout.addWidget(self.results_tabs)

        panel.setLayout(layout)
        return panel

    def _create_log_widget(self) -> QWidget:
        """Create training log widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.log_text)

        # Clear log button
        clear_btn = QPushButton("🗑️ 清除日誌")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        widget.setLayout(layout)
        return widget

    def _create_curves_widget(self) -> QWidget:
        """Create training curves widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.loss_plot = PlotWidget("Training Loss", self)
        self.loss_plot.set_ylabel("Loss")
        layout.addWidget(self.loss_plot)

        widget.setLayout(layout)
        return widget

    def _create_eval_widget(self) -> QWidget:
        """Create evaluation results widget."""
        self.eval_widget = EvalWidget(self)
        return self.eval_widget

    # Slots for UI events

    @Slot()
    def _on_augmentation_settings(self):
        """Open augmentation settings dialog."""
        from .augmentation_dialog import AugmentationDialog

        dialog = AugmentationDialog(self, initial_config=self.augmentation_config)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.augmentation_config = dialog.get_config()

            # Count active parameters (non-zero values)
            active_count = sum(1 for v in self.augmentation_config.values() if v > 0)

            # Update status label
            self.aug_status_label.setText(f"✅ 已設定自訂強化參數 ({active_count} 項啟用)")
            self.aug_status_label.setStyleSheet("color: #28a745; font-weight: bold; font-size: 9pt;")

            self._log_message(f"📋 資料強化設定已更新 ({active_count} 項啟用)")

    @Slot()
    def _on_browse_output_dir(self):
        """Browse for output directory."""
        # Try to get default from config
        default_dir = str(self.config.get_output_dir()) if self.config.get_output_dir() else ""

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇標註輸出目錄",
            default_dir
        )

        if dir_path:
            self.output_dir_edit.setText(dir_path)
            self._check_dataset_ready()

    @Slot()
    def _on_configure_split(self):
        """Configure train/val split."""
        output_dir_str = self.output_dir_edit.text()

        if not output_dir_str:
            QMessageBox.warning(
                self,
                "警告",
                "請先選擇資料集目錄"
            )
            return

        output_dir = Path(output_dir_str)

        if not output_dir.exists():
            QMessageBox.warning(
                self,
                "警告",
                f"目錄不存在: {output_dir}"
            )
            return

        # Show split dialog
        dialog = DatasetSplitDialog(output_dir, self)
        if dialog.exec():
            # Generate dataset YAML
            try:
                train_images, val_images = dialog.get_split_info()

                self.dataset_yaml = self.yolo_service.generate_dataset_yaml(
                    train_images,
                    val_images,
                    output_dir
                )

                self.dataset_info_label.setText(
                    f"✅ 資料集已配置\n"
                    f"訓練集: {len(train_images)} 張\n"
                    f"驗證集: {len(val_images)} 張\n"
                    f"YAML: {self.dataset_yaml.name}"
                )

                self._check_dataset_ready()

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    f"生成 YAML 失敗: {e}"
                )

    def _check_dataset_ready(self):
        """Check if dataset is ready for training."""
        if self.dataset_yaml and self.dataset_yaml.exists():
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(False)

    @Slot()
    def _on_start_training(self):
        """Start training."""
        if self.training_in_progress:
            QMessageBox.warning(self, "警告", "訓練已在進行中")
            return

        if not self.dataset_yaml or not self.dataset_yaml.exists():
            QMessageBox.warning(
                self,
                "警告",
                "請先配置資料集分割"
            )
            return

        # Check if model exists
        if not self.model_path.exists():
            QMessageBox.critical(
                self,
                "錯誤",
                f"找不到模型檔案:\n{self.model_path}\n\n請確認模型檔案存在"
            )
            return

        # Get training parameters
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()

        # Clear previous results
        self.log_text.clear()
        self.loss_plot.clear()
        self.eval_widget.clear()

        # Create training worker
        self.training_worker = TrainingWorker(
            model_path=str(self.model_path),
            data_yaml=str(self.dataset_yaml),
            epochs=epochs,
            batch_size=batch_size,
            project="runs/obb",
            name="train",
            augmentation_config=self.augmentation_config  # Pass augmentation settings
        )

        # Connect signals
        self.training_worker.log_message.connect(self._on_log_message)
        self.training_worker.epoch_completed.connect(self._on_epoch_completed)
        self.training_worker.training_completed.connect(self._on_training_completed)
        self.training_worker.training_failed.connect(self._on_training_failed)
        self.training_worker.evaluation_completed.connect(self._on_evaluation_completed)

        # Update UI state
        self.training_in_progress = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("狀態: 訓練中...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")

        # Start training
        self.training_worker.start()

        self._log_message(f"開始訓練 - 模型: yolov10n-obb.pt, Epochs: {epochs}, Batch: {batch_size}")

    @Slot()
    def _on_stop_training(self):
        """Stop training."""
        if not self.training_in_progress or not self.training_worker:
            return

        reply = QMessageBox.question(
            self,
            "確認停止",
            "確定要停止訓練嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.training_worker.terminate()
            self.training_worker.wait()  # Wait for thread to finish

            # Reset training state
            self.training_in_progress = False
            self.training_worker = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("狀態: 訓練已停止")
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")

            self._log_message("=" * 50)
            self._log_message("⚠️  訓練已被使用者中止")
            self._log_message("=" * 50)

    @Slot(str)
    def _on_log_message(self, message: str):
        """Handle log message from worker."""
        self._log_message(message)

    @Slot(int, dict)
    def _on_epoch_completed(self, epoch: int, metrics: dict):
        """Handle epoch completion."""
        # Update training curves
        self.loss_plot.add_data_point(epoch, metrics)

        # Switch to curves tab automatically
        self.results_tabs.setCurrentIndex(1)

    @Slot(dict)
    def _on_training_completed(self, results: dict):
        """Handle training completion."""
        self.training_in_progress = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("狀態: 訓練完成 ✅")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")

        self._log_message("=" * 50)
        self._log_message("訓練完成！")
        if results.get('best_model'):
            self._log_message(f"最佳模型: {results['best_model']}")
        if results.get('save_dir'):
            self._log_message(f"結果目錄: {results['save_dir']}")
        self._log_message("=" * 50)

    @Slot(str)
    def _on_training_failed(self, error_message: str):
        """Handle training failure."""
        self.training_in_progress = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("狀態: 訓練失敗 ❌")
        self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")

        self._log_message("=" * 50)
        self._log_message(f"❌ 訓練失敗: {error_message}")
        self._log_message("=" * 50)

        QMessageBox.critical(
            self,
            "訓練失敗",
            error_message
        )

    @Slot(dict)
    def _on_evaluation_completed(self, eval_results: dict):
        """Handle evaluation completion."""
        # Update evaluation widget
        self.eval_widget.update_metrics(eval_results)

        # Update plots if results_dir available
        if 'results_dir' in eval_results:
            results_dir = Path(eval_results['results_dir'])
            self.eval_widget.update_plots(results_dir)

        # Switch to evaluation tab
        self.results_tabs.setCurrentIndex(2)

        # Register model with ModelManager
        if 'results_dir' in eval_results:
            try:
                results_dir = Path(eval_results['results_dir'])
                best_model = results_dir / 'weights' / 'best.pt'

                if best_model.exists():
                    # Generate model name from timestamp
                    from datetime import datetime
                    model_name = f"obb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    self.model_manager.register_model(
                        name=model_name,
                        model_path=str(best_model),
                        metrics=eval_results
                    )

                    self._log_message(f"✅ 模型已註冊: {model_name}")

                    # Refresh model manager widget
                    self.model_manager_widget._refresh_models()

            except Exception as e:
                self._log_message(f"⚠️  模型註冊失敗: {e}")

    def _log_message(self, message: str):
        """Append message to log.

        Args:
            message: Log message
        """
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
