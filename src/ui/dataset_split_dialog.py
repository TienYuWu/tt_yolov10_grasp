"""Dialog for dataset split configuration."""

from pathlib import Path
from typing import List, Tuple, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QGroupBox, QListWidget, QMessageBox, QProgressBar,
    QListWidgetItem
)
from PySide6.QtCore import Qt

import random


class DatasetSplitDialog(QDialog):
    """Dialog for splitting dataset into train/val sets."""

    def __init__(self, output_dir: Path, parent=None):
        """Initialize dataset split dialog.

        Args:
            output_dir: Output directory containing annotated images
            parent: Parent widget
        """
        super().__init__(parent)
        self.output_dir = output_dir
        self.all_images: List[Path] = []
        self.train_images: List[Path] = []
        self.val_images: List[Path] = []

        self.setWindowTitle("資料集分割配置")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        self._build_ui()
        self._load_images()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout()

        # Info section
        info_group = self._create_info_section()
        layout.addWidget(info_group)

        # Split ratio section
        ratio_group = self._create_ratio_section()
        layout.addWidget(ratio_group)

        # Preview section
        preview_group = self._create_preview_section()
        layout.addWidget(preview_group, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.split_btn = QPushButton("執行分割")
        self.split_btn.clicked.connect(self._on_split)
        self.split_btn.setEnabled(False)
        button_layout.addWidget(self.split_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_info_section(self) -> QGroupBox:
        """Create info section showing dataset directory."""
        group = QGroupBox("資料集資訊")
        layout = QVBoxLayout()

        self.dir_label = QLabel(f"資料夾: {self.output_dir}")
        self.dir_label.setWordWrap(True)
        layout.addWidget(self.dir_label)

        self.count_label = QLabel("圖片數量: 0")
        layout.addWidget(self.count_label)

        group.setLayout(layout)
        return group

    def _create_ratio_section(self) -> QGroupBox:
        """Create ratio configuration section."""
        group = QGroupBox("分割比例")
        layout = QHBoxLayout()

        # Train percentage
        layout.addWidget(QLabel("訓練集:"))
        self.train_percent_spin = QSpinBox()
        self.train_percent_spin.setRange(10, 90)
        self.train_percent_spin.setValue(80)
        self.train_percent_spin.setSuffix("%")
        self.train_percent_spin.valueChanged.connect(self._on_ratio_changed)
        layout.addWidget(self.train_percent_spin)

        layout.addWidget(QLabel("驗證集:"))
        self.val_percent_label = QLabel("20%")
        layout.addWidget(self.val_percent_label)

        layout.addStretch()

        # Shuffle button
        shuffle_btn = QPushButton("隨機打亂")
        shuffle_btn.clicked.connect(self._on_shuffle)
        layout.addWidget(shuffle_btn)

        group.setLayout(layout)
        return group

    def _create_preview_section(self) -> QGroupBox:
        """Create preview section showing split results."""
        group = QGroupBox("分割預覽")
        layout = QHBoxLayout()

        # Train list
        train_layout = QVBoxLayout()
        self.train_count_label = QLabel("訓練集 (0)")
        train_layout.addWidget(self.train_count_label)

        self.train_list = QListWidget()
        train_layout.addWidget(self.train_list)

        layout.addLayout(train_layout)

        # Val list
        val_layout = QVBoxLayout()
        self.val_count_label = QLabel("驗證集 (0)")
        val_layout.addWidget(self.val_count_label)

        self.val_list = QListWidget()
        val_layout.addWidget(self.val_list)

        layout.addLayout(val_layout)

        group.setLayout(layout)
        return group

    def _load_images(self):
        """Load annotated images from output directory."""
        # Look for images directory (try multiple possible names)
        images_dir = None
        for possible_name in ['rgb', 'images', 'img']:
            candidate = self.output_dir / possible_name
            if candidate.exists() and candidate.is_dir():
                images_dir = candidate
                break

        if not images_dir:
            QMessageBox.warning(
                self,
                "警告",
                f"找不到圖片目錄（嘗試過: rgb, images, img）\n在: {self.output_dir}\n\n請確認已經儲存標註"
            )
            return

        # Find all images
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.all_images = []

        for ext in supported_exts:
            self.all_images.extend(images_dir.glob(f"*{ext}"))

        self.all_images.sort()

        # Update UI
        self.count_label.setText(f"圖片數量: {len(self.all_images)}")

        if len(self.all_images) == 0:
            QMessageBox.warning(
                self,
                "警告",
                "沒有找到任何圖片\n\n請先完成標註並儲存"
            )
            return

        # Enable split button
        self.split_btn.setEnabled(True)

        # Perform initial split
        self._perform_split()

    def _on_ratio_changed(self, value: int):
        """Handle ratio change."""
        val_percent = 100 - value
        self.val_percent_label.setText(f"{val_percent}%")
        self._perform_split()

    def _on_shuffle(self):
        """Shuffle and re-split dataset."""
        random.shuffle(self.all_images)
        self._perform_split()

    def _perform_split(self):
        """Perform dataset split based on current ratio."""
        if not self.all_images:
            return

        train_percent = self.train_percent_spin.value()
        train_count = int(len(self.all_images) * train_percent / 100)

        self.train_images = self.all_images[:train_count]
        self.val_images = self.all_images[train_count:]

        # Update lists
        self.train_list.clear()
        for img in self.train_images:
            item = QListWidgetItem(img.name)
            try:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
            self.train_list.addItem(item)

        self.val_list.clear()
        for img in self.val_images:
            item = QListWidgetItem(img.name)
            try:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
            self.val_list.addItem(item)

        # Update counts
        self.train_count_label.setText(f"訓練集 ({len(self.train_images)})")
        self.val_count_label.setText(f"驗證集 ({len(self.val_images)})")

    def _on_split(self):
        """Execute the split and generate train/val files."""
        if not self.train_images or not self.val_images:
            QMessageBox.warning(
                self,
                "警告",
                "訓練集或驗證集為空，無法執行分割"
            )
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(2)
            self.progress_bar.setValue(0)

            # Create train.txt
            train_txt = self.output_dir / "train.txt"
            with open(train_txt, 'w') as f:
                for img in self.train_images:
                    # Write relative path from output_dir
                    rel_path = img.relative_to(self.output_dir)
                    f.write(f"{rel_path}\n")

            self.progress_bar.setValue(1)

            # Create val.txt
            val_txt = self.output_dir / "val.txt"
            with open(val_txt, 'w') as f:
                for img in self.val_images:
                    rel_path = img.relative_to(self.output_dir)
                    f.write(f"{rel_path}\n")

            self.progress_bar.setValue(2)

            QMessageBox.information(
                self,
                "成功",
                f"資料集分割完成！\n\n"
                f"訓練集: {len(self.train_images)} 張\n"
                f"驗證集: {len(self.val_images)} 張\n\n"
                f"已生成:\n"
                f"- {train_txt.name}\n"
                f"- {val_txt.name}"
            )

            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "錯誤",
                f"分割失敗: {e}"
            )
        finally:
            self.progress_bar.setVisible(False)

    def get_split_info(self) -> Tuple[List[Path], List[Path]]:
        """Get the split results.

        Returns:
            Tuple of (train_images, val_images)
        """
        return self.train_images, self.val_images
