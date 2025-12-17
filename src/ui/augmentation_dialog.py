#!/usr/bin/env python3
"""Data augmentation configuration dialog.

This module provides a comprehensive UI for configuring YOLO data augmentation
parameters during training. Supports 14 augmentation parameters across geometric,
color, and advanced transform categories.

Vendor-ready code for industry-academia cooperation delivery.

Features:
1. 3-tab interface (Geometric / Color / Advanced)
2. Real-time slider controls with value display
3. Predefined presets (Light / Medium / Heavy)
4. Custom preset save/load functionality
5. Direct integration with YOLO training parameters

YOLO Augmentation Parameters:
- Geometric: degrees, translate, scale, shear, perspective, flipud, fliplr
- Color: hsv_h, hsv_s, hsv_v
- Advanced: mosaic, mixup, copy_paste, erasing
"""

import json
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QGroupBox, QPushButton, QDialogButtonBox,
    QTabWidget, QWidget, QInputDialog, QFileDialog, QMessageBox
)


class AugmentationDialog(QDialog):
    """Data augmentation configuration dialog with preset support.

    Provides interactive UI for configuring all YOLO augmentation parameters.
    Users can adjust parameters via sliders, apply presets, or save/load
    custom configurations.

    Usage:
        dialog = AugmentationDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            # Pass config to training worker
    """

    # Predefined presets for different augmentation strengths
    PRESETS = {
        'light': {
            # Geometric transforms (light)
            'degrees': 5.0,         # Rotation ±5°
            'translate': 0.05,      # Translation ±5%
            'scale': 0.3,           # Scale ±30%
            'shear': 0.0,           # No shear
            'perspective': 0.0,     # No perspective
            'flipud': 0.0,          # No vertical flip
            'fliplr': 0.5,          # 50% horizontal flip
            # Color transforms (light)
            'hsv_h': 0.01,          # Hue ±1%
            'hsv_s': 0.5,           # Saturation ±50%
            'hsv_v': 0.3,           # Value ±30%
            # Advanced (light)
            'mosaic': 0.5,          # 50% mosaic
            'mixup': 0.0,           # No mixup
            'copy_paste': 0.0,      # No copy-paste
            'erasing': 0.2          # 20% random erasing
        },
        'medium': {
            # Geometric transforms (medium)
            'degrees': 15.0,        # Rotation ±15°
            'translate': 0.1,       # Translation ±10%
            'scale': 0.5,           # Scale ±50%
            'shear': 0.0,           # No shear
            'perspective': 0.0,     # No perspective
            'flipud': 0.0,          # No vertical flip
            'fliplr': 0.5,          # 50% horizontal flip
            # Color transforms (medium)
            'hsv_h': 0.015,         # Hue ±1.5%
            'hsv_s': 0.7,           # Saturation ±70%
            'hsv_v': 0.4,           # Value ±40%
            # Advanced (medium)
            'mosaic': 1.0,          # 100% mosaic
            'mixup': 0.0,           # No mixup
            'copy_paste': 0.0,      # No copy-paste
            'erasing': 0.4          # 40% random erasing
        },
        'heavy': {
            # Geometric transforms (heavy)
            'degrees': 30.0,        # Rotation ±30°
            'translate': 0.2,       # Translation ±20%
            'scale': 0.7,           # Scale ±70%
            'shear': 10.0,          # Shear ±10°
            'perspective': 0.0005,  # Perspective transform
            'flipud': 0.2,          # 20% vertical flip
            'fliplr': 0.5,          # 50% horizontal flip
            # Color transforms (heavy)
            'hsv_h': 0.03,          # Hue ±3%
            'hsv_s': 0.8,           # Saturation ±80%
            'hsv_v': 0.5,           # Value ±50%
            # Advanced (heavy)
            'mosaic': 1.0,          # 100% mosaic
            'mixup': 0.3,           # 30% mixup
            'copy_paste': 0.1,      # 10% copy-paste
            'erasing': 0.5          # 50% random erasing
        }
    }

    def __init__(self, parent=None, initial_config: Dict = None):
        """Initialize augmentation dialog.

        Args:
            parent: Parent widget
            initial_config: Initial configuration dict (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("資料強化設定")
        self.setMinimumSize(700, 750)

        # Current configuration (default to medium preset)
        if initial_config:
            self.config = initial_config.copy()
        else:
            self.config = self.PRESETS['medium'].copy()

        # Slider widgets storage {key: {'slider': QSlider, 'label': QLabel, 'step': float, 'suffix': str}}
        self.sliders = {}

        self._build_ui()

    def _build_ui(self):
        """Build augmentation dialog UI."""
        layout = QVBoxLayout()

        # Tab widget for parameter categories
        tabs = QTabWidget()

        # Geometric transforms tab
        geo_tab = self._create_geometric_tab()
        tabs.addTab(geo_tab, "🔄 幾何變換")

        # Color transforms tab
        color_tab = self._create_color_tab()
        tabs.addTab(color_tab, "🎨 色彩變換")

        # Advanced tab
        adv_tab = self._create_advanced_tab()
        tabs.addTab(adv_tab, "🖼️ 進階強化")

        layout.addWidget(tabs)

        # Preset buttons
        preset_group = self._create_preset_group()
        layout.addWidget(preset_group)

        # Save/Load buttons
        io_layout = QHBoxLayout()

        save_btn = QPushButton("💾 儲存自訂預設")
        save_btn.setToolTip("將目前設定儲存為自訂預設檔案")
        save_btn.clicked.connect(self._on_save_preset)
        io_layout.addWidget(save_btn)

        load_btn = QPushButton("📂 載入預設")
        load_btn.setToolTip("從檔案載入自訂預設")
        load_btn.clicked.connect(self._on_load_preset)
        io_layout.addWidget(load_btn)

        io_layout.addStretch()
        layout.addLayout(io_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _create_geometric_tab(self) -> QWidget:
        """Create geometric transforms tab.

        Returns:
            Widget containing geometric transform sliders
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # Rotation
        layout.addWidget(self._create_slider_row(
            "旋轉角度 (degrees)", "degrees", 0.0, 45.0, 1.0, "°"
        ))

        # Translation
        layout.addWidget(self._create_slider_row(
            "平移比例 (translate)", "translate", 0.0, 0.3, 0.01
        ))

        # Scale
        layout.addWidget(self._create_slider_row(
            "縮放比例 (scale)", "scale", 0.0, 1.0, 0.01
        ))

        # Shear
        layout.addWidget(self._create_slider_row(
            "剪切角度 (shear)", "shear", 0.0, 20.0, 1.0, "°"
        ))

        # Perspective
        layout.addWidget(self._create_slider_row(
            "透視變換 (perspective)", "perspective", 0.0, 0.001, 0.0001
        ))

        # Vertical flip
        layout.addWidget(self._create_slider_row(
            "垂直翻轉機率 (flipud)", "flipud", 0.0, 1.0, 0.05
        ))

        # Horizontal flip
        layout.addWidget(self._create_slider_row(
            "水平翻轉機率 (fliplr)", "fliplr", 0.0, 1.0, 0.05
        ))

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_color_tab(self) -> QWidget:
        """Create color transforms tab.

        Returns:
            Widget containing color transform sliders
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # HSV Hue
        layout.addWidget(self._create_slider_row(
            "色相調整 (hsv_h)", "hsv_h", 0.0, 0.1, 0.005
        ))

        # HSV Saturation
        layout.addWidget(self._create_slider_row(
            "飽和度調整 (hsv_s)", "hsv_s", 0.0, 1.0, 0.05
        ))

        # HSV Value
        layout.addWidget(self._create_slider_row(
            "亮度調整 (hsv_v)", "hsv_v", 0.0, 1.0, 0.05
        ))

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced augmentation tab.

        Returns:
            Widget containing advanced augmentation sliders
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # Mosaic
        layout.addWidget(self._create_slider_row(
            "Mosaic 拼接機率 (mosaic)", "mosaic", 0.0, 1.0, 0.05
        ))

        # MixUp
        layout.addWidget(self._create_slider_row(
            "MixUp 混合機率 (mixup)", "mixup", 0.0, 1.0, 0.05
        ))

        # Copy-Paste
        layout.addWidget(self._create_slider_row(
            "Copy-Paste 機率 (copy_paste)", "copy_paste", 0.0, 1.0, 0.05
        ))

        # Random Erasing
        layout.addWidget(self._create_slider_row(
            "Random Erasing 機率 (erasing)", "erasing", 0.0, 1.0, 0.05
        ))

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_slider_row(self, label: str, key: str,
                          min_val: float, max_val: float,
                          step: float, suffix: str = "") -> QWidget:
        """Create slider row with label and value display.

        Args:
            label: Display label for parameter
            key: Config dictionary key
            min_val: Minimum slider value
            max_val: Maximum slider value
            step: Slider step size
            suffix: Value suffix (e.g., "°" for degrees)

        Returns:
            Widget containing slider row
        """
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        # Label
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(220)
        layout.addWidget(label_widget)

        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(self.config.get(key, 0) / step))
        slider.setMinimumWidth(280)

        # Connect slider
        slider.valueChanged.connect(
            lambda v, k=key, s=step: self._on_slider_changed(k, v * s)
        )
        layout.addWidget(slider)

        # Value label
        value_label = QLabel(f"{self.config.get(key, 0):.3f}{suffix}")
        value_label.setMinimumWidth(90)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(value_label)

        # Store references
        self.sliders[key] = {
            'slider': slider,
            'label': value_label,
            'step': step,
            'suffix': suffix
        }

        widget.setLayout(layout)
        return widget

    def _create_preset_group(self) -> QGroupBox:
        """Create preset buttons group.

        Returns:
            Group box containing preset buttons
        """
        group = QGroupBox("📋 快速預設組合")
        layout = QHBoxLayout()

        for preset_name, display_name in [
            ('light', '輕度強化'),
            ('medium', '中度強化 (建議)'),
            ('heavy', '重度強化')
        ]:
            btn = QPushButton(display_name)
            btn.setToolTip(f"套用 {display_name} 預設參數")
            btn.clicked.connect(lambda checked, p=preset_name: self._apply_preset(p))
            layout.addWidget(btn)

        group.setLayout(layout)
        return group

    @Slot(str, float)
    def _on_slider_changed(self, key: str, value: float):
        """Handle slider value change.

        Args:
            key: Config dictionary key
            value: New slider value
        """
        self.config[key] = value

        # Update label
        suffix = self.sliders[key].get('suffix', '')
        self.sliders[key]['label'].setText(f"{value:.3f}{suffix}")

    @Slot(str)
    def _apply_preset(self, preset_name: str):
        """Apply predefined preset.

        Args:
            preset_name: Name of preset to apply ('light', 'medium', 'heavy')
        """
        if preset_name not in self.PRESETS:
            return

        # Update config
        self.config = self.PRESETS[preset_name].copy()

        # Update all sliders
        for key, value in self.config.items():
            if key in self.sliders:
                step = self.sliders[key]['step']
                suffix = self.sliders[key].get('suffix', '')

                # Update slider position
                self.sliders[key]['slider'].blockSignals(True)  # Prevent signal spam
                self.sliders[key]['slider'].setValue(int(value / step))
                self.sliders[key]['slider'].blockSignals(False)

                # Update label
                self.sliders[key]['label'].setText(f"{value:.3f}{suffix}")

    @Slot()
    def _on_save_preset(self):
        """Save current config as custom preset."""
        name, ok = QInputDialog.getText(
            self, "儲存預設", "請輸入預設名稱:"
        )

        if ok and name:
            presets_dir = Path.home() / ".tt_yolov10_pilegrasp" / "augmentation_presets"
            presets_dir.mkdir(parents=True, exist_ok=True)

            preset_file = presets_dir / f"{name}.json"

            try:
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

                QMessageBox.information(
                    self,
                    "儲存成功",
                    f"預設已儲存:\n{preset_file}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "儲存失敗",
                    f"無法儲存預設:\n{str(e)}"
                )

    @Slot()
    def _on_load_preset(self):
        """Load custom preset from file."""
        presets_dir = Path.home() / ".tt_yolov10_pilegrasp" / "augmentation_presets"
        presets_dir.mkdir(parents=True, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "載入預設",
            str(presets_dir),
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # Validate and merge with current config
                valid_keys = set(self.config.keys())
                for key, value in loaded_config.items():
                    if key in valid_keys:
                        self.config[key] = value

                # Update all sliders
                for key, value in self.config.items():
                    if key in self.sliders:
                        step = self.sliders[key]['step']
                        suffix = self.sliders[key].get('suffix', '')

                        # Update slider position
                        self.sliders[key]['slider'].blockSignals(True)
                        self.sliders[key]['slider'].setValue(int(value / step))
                        self.sliders[key]['slider'].blockSignals(False)

                        # Update label
                        self.sliders[key]['label'].setText(f"{value:.3f}{suffix}")

                QMessageBox.information(
                    self,
                    "載入成功",
                    f"預設已載入:\n{Path(file_path).name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "載入失敗",
                    f"無法載入預設:\n{str(e)}"
                )

    def get_config(self) -> Dict:
        """Get current augmentation configuration.

        Returns:
            Dictionary of augmentation parameters
        """
        return self.config.copy()


# ===================================================================
# Standalone Test
# ===================================================================

if __name__ == "__main__":
    """Test augmentation dialog."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    dialog = AugmentationDialog()

    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_config()
        print("\n=== Augmentation Configuration ===")
        for key, value in config.items():
            print(f"{key:15s}: {value:.4f}")
    else:
        print("Dialog cancelled")

    sys.exit(0)
