"""Widget for displaying evaluation results."""

from pathlib import Path
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class EvalWidget(QWidget):
    """Widget for displaying YOLO evaluation results."""

    def __init__(self, parent=None):
        """Initialize evaluation widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()

        # Metrics section
        metrics_group = self._create_metrics_section()
        layout.addWidget(metrics_group)

        # Plots section
        plots_group = self._create_plots_section()
        layout.addWidget(plots_group, 1)

        self.setLayout(layout)

    def _create_metrics_section(self) -> QGroupBox:
        """Create metrics display section."""
        group = QGroupBox("評估指標")
        layout = QGridLayout()

        # Create metric labels
        metrics = [
            ("mAP50", "mAP@0.5"),
            ("mAP50-95", "mAP@0.5:0.95"),
            ("precision", "Precision"),
            ("recall", "Recall")
        ]

        self.metric_labels = {}

        for i, (key, display_name) in enumerate(metrics):
            # Metric name
            name_label = QLabel(f"{display_name}:")
            name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(name_label, i, 0)

            # Metric value
            value_label = QLabel("---")
            value_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            layout.addWidget(value_label, i, 1)

            self.metric_labels[key] = value_label

        group.setLayout(layout)
        return group

    def _create_plots_section(self) -> QGroupBox:
        """Create plots display section."""
        group = QGroupBox("評估圖表")
        layout = QVBoxLayout()

        # Confusion matrix
        cm_layout = QVBoxLayout()
        cm_label = QLabel("混淆矩陣:")
        cm_layout.addWidget(cm_label)

        self.cm_image_label = QLabel("（尚無數據）")
        self.cm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cm_image_label.setMinimumHeight(200)
        self.cm_image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        cm_layout.addWidget(self.cm_image_label)

        layout.addLayout(cm_layout)

        # PR curve
        pr_layout = QVBoxLayout()
        pr_label = QLabel("Precision-Recall 曲線:")
        pr_layout.addWidget(pr_label)

        self.pr_image_label = QLabel("（尚無數據）")
        self.pr_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pr_image_label.setMinimumHeight(200)
        self.pr_image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        pr_layout.addWidget(self.pr_image_label)

        layout.addLayout(pr_layout)

        group.setLayout(layout)
        return group

    def update_metrics(self, metrics: Dict[str, float]):
        """Update displayed metrics.

        Args:
            metrics: Dictionary of metric name to value
        """
        for key, value_label in self.metric_labels.items():
            if key in metrics:
                value = metrics[key]
                value_label.setText(f"{value:.4f}")

                # Color code based on value
                if value >= 0.8:
                    color = "#28a745"  # Green
                elif value >= 0.6:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red

                value_label.setStyleSheet(
                    f"font-weight: bold; font-size: 14px; color: {color};"
                )
            else:
                value_label.setText("---")

    def update_plots(self, results_dir: Path):
        """Update plots from results directory.

        Args:
            results_dir: Directory containing result plots
        """
        # Load confusion matrix
        cm_path = results_dir / "confusion_matrix.png"
        if cm_path.exists():
            pixmap = QPixmap(str(cm_path))
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.cm_image_label.setPixmap(scaled_pixmap)
        else:
            self.cm_image_label.setText("（找不到混淆矩陣圖）")

        # Load PR curve
        pr_path = results_dir / "PR_curve.png"
        if pr_path.exists():
            pixmap = QPixmap(str(pr_path))
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.pr_image_label.setPixmap(scaled_pixmap)
        else:
            self.pr_image_label.setText("（找不到 PR 曲線圖）")

    def clear(self):
        """Clear all displayed results."""
        # Reset metrics
        for value_label in self.metric_labels.values():
            value_label.setText("---")
            value_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Reset plots
        self.cm_image_label.clear()
        self.cm_image_label.setText("（尚無數據）")

        self.pr_image_label.clear()
        self.pr_image_label.setText("（尚無數據）")
