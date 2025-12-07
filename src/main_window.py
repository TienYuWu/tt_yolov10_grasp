"""Main window - Tab container for Smart Label application."""

import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar
from PySide6.QtCore import Signal
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QPushButton, QGroupBox, QRadioButton, QCheckBox,
    QListWidget, QComboBox, QWidget
)

from .core import ServiceContainer
from .services import SAMService, YOLOService, ModelManager
from .ui import AnnotationTab, TrainingTab, DetectionTab
from .config import AppConfig


class MainWindow(QMainWindow):
    """Main application window with tabs."""

    def __init__(self, config: AppConfig, predictor):
        """Initialize main window.

        Args:
            config: Application configuration
            predictor: SAM predictor instance
        """
        super().__init__()
        self.setWindowTitle("Smart Label - 智能標註與訓練工具")
        self.config = config

        # Initialize Service Container
        self.service_container = ServiceContainer()
        self.service_container.register('sam_service', SAMService(predictor))
        self.service_container.register('yolo_service', YOLOService())
        self.service_container.register('model_manager', ModelManager())

        # Create Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Annotation Tab
        self.annotation_tab = AnnotationTab(config, self.service_container, self)
        self.tabs.addTab(self.annotation_tab, "📝 標記")

        # Training Tab
        self.training_tab = TrainingTab(config, self.service_container, self)
        self.tabs.addTab(self.training_tab, "🎯 訓練")

        # Detection Tab
        model_path = self._load_model_path(config)
        self.detection_tab = DetectionTab(config, model_path, self)
        self.tabs.addTab(self.detection_tab, "🔍 檢測")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Set initial size
        self.resize(config.window_width, config.window_height)

        # Align UI text to left for better readability
        self._align_ui_text_left()

    def _load_model_path(self, config: AppConfig) -> str:
        """Load model path from config file or use default.

        Args:
            config: Application configuration

        Returns:
            Model file path (str)
        """
        try:
            # Try to load from config file
            if hasattr(config, 'output_dir') and config.output_dir:
                config_file = Path(config.output_dir) / "model_config.json"
            else:
                config_file = Path.cwd() / "model_config.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    model_path = config_data.get('model_path')
                    if model_path and Path(model_path).exists():
                        print(f"Loaded model path from config: {model_path}")
                        return str(model_path)
        except Exception as e:
            print(f"Failed to load model config: {e}")

        # Default fallback: look for best.pt in project directory
        default_path = Path.cwd() / "best.pt"
        if default_path.exists():
            print(f"Using default model path: {default_path}")
            return str(default_path)

        # Final fallback: original hardcoded path
        fallback_path = r"C:\Users\NCKU_CSIE_RL_TIEN\Desktop\tt_yolov10_grasp\best.pt"
        print(f"Using fallback model path: {fallback_path}")
        return fallback_path

    def update_status(self, message: str):
        """Update status bar message.

        Args:
            message: Status message to display
        """
        self.status_bar.showMessage(message)

    def _align_ui_text_left(self):
        """Recursively set common widget text alignment to left.

        This enforces left alignment for labels, buttons, groupbox titles,
        combo boxes and list items so the UI text appears left-aligned.
        """
        # Align QLabel
        for lbl in self.findChildren(QLabel):
            try:
                lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass

        # Left-align QPushButton text via stylesheet
        for btn in self.findChildren(QPushButton):
            try:
                # keep existing style but enforce left text alignment
                btn.setStyleSheet((btn.styleSheet() or "") + "\ntext-align: left;")
            except Exception:
                pass

        # GroupBox title alignment
        for gb in self.findChildren(QGroupBox):
            try:
                gb.setAlignment(Qt.AlignLeft)
            except Exception:
                pass

        # Radio/Check buttons
        for rb in self.findChildren(QRadioButton):
            try:
                rb.setStyleSheet((rb.styleSheet() or "") + "\ntext-align: left;")
            except Exception:
                pass
        for cb in self.findChildren(QCheckBox):
            try:
                cb.setStyleSheet((cb.styleSheet() or "") + "\ntext-align: left;")
            except Exception:
                pass

        # Combo boxes: left text
        for combo in self.findChildren(QComboBox):
            try:
                combo.setStyleSheet((combo.styleSheet() or "") + "\ntext-align: left;")
            except Exception:
                pass

        # QListWidget items: set item alignment to left
        for lst in self.findChildren(QListWidget):
            try:
                for i in range(lst.count()):
                    item = lst.item(i)
                    if item is not None:
                        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
