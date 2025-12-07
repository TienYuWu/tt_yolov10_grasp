"""Widget for managing trained models."""

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QListWidget, QListWidgetItem, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal, Slot

from ..services import ModelManager


class ModelManagerWidget(QWidget):
    """Widget for managing trained YOLO models."""

    model_selected = Signal(str)  # Emitted when a model is selected (model_path)

    def __init__(self, model_manager: ModelManager, parent=None):
        """Initialize model manager widget.

        Args:
            model_manager: ModelManager service instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.model_manager = model_manager

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("📦 已訓練模型")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Models list
        self.models_list = QListWidget()
        self.models_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.models_list.customContextMenuRequested.connect(self._on_context_menu)
        self.models_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        layout.addWidget(self.models_list)

        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 重新整理")
        refresh_btn.clicked.connect(self._refresh_models)
        button_layout.addWidget(refresh_btn)

        delete_btn = QPushButton("🗑️ 刪除")
        delete_btn.clicked.connect(self._on_delete_model)
        button_layout.addWidget(delete_btn)

        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Info section
        info_group = self._create_info_section()
        layout.addWidget(info_group)

        self.setLayout(layout)

    def _create_info_section(self) -> QGroupBox:
        """Create model info section."""
        group = QGroupBox("模型資訊")
        layout = QVBoxLayout()

        self.info_label = QLabel("（未選擇模型）")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        group.setLayout(layout)
        return group

    def _refresh_models(self):
        """Refresh the models list."""
        self.models_list.clear()

        models = self.model_manager.list_models()

        if not models:
            self.info_label.setText("（尚無已訓練模型）")
            return

        for model_info in models:
            name = model_info['name']
            created_at = model_info.get('created_at', 'Unknown')

            item_text = f"{name}\n{created_at}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, model_info)

            try:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
            self.models_list.addItem(item)

    @Slot(QListWidgetItem)
    def _on_model_double_clicked(self, item: QListWidgetItem):
        """Handle model double click."""
        model_info = item.data(Qt.ItemDataRole.UserRole)
        if model_info:
            model_path = model_info['model_path']
            self.model_selected.emit(model_path)
            self._show_model_info(model_info)

    @Slot()
    def _on_delete_model(self):
        """Delete selected model."""
        current_item = self.models_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "請先選擇要刪除的模型")
            return

        model_info = current_item.data(Qt.ItemDataRole.UserRole)
        if not model_info:
            return

        model_name = model_info['name']

        reply = QMessageBox.question(
            self,
            "確認刪除",
            f"確定要刪除模型 '{model_name}' 嗎？\n\n此操作無法復原！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.model_manager.delete_model(model_name)
                QMessageBox.information(
                    self,
                    "成功",
                    f"模型 '{model_name}' 已刪除"
                )
                self._refresh_models()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    f"刪除失敗: {e}"
                )

    @Slot()
    def _on_context_menu(self, pos):
        """Show context menu."""
        item = self.models_list.itemAt(pos)
        if not item:
            return

        model_info = item.data(Qt.ItemDataRole.UserRole)
        if not model_info:
            return

        menu = QMenu(self)

        # Show info action
        info_action = menu.addAction("📋 查看詳細資訊")
        info_action.triggered.connect(lambda: self._show_model_info(model_info))

        # Load action
        load_action = menu.addAction("📂 載入模型")
        load_action.triggered.connect(
            lambda: self.model_selected.emit(model_info['model_path'])
        )

        menu.addSeparator()

        # Delete action
        delete_action = menu.addAction("🗑️ 刪除")
        delete_action.triggered.connect(self._on_delete_model)

        menu.exec(self.models_list.mapToGlobal(pos))

    def _show_model_info(self, model_info: Dict):
        """Show model information.

        Args:
            model_info: Model information dictionary
        """
        info_text = f"模型名稱: {model_info['name']}\n"
        info_text += f"創建時間: {model_info.get('created_at', 'Unknown')}\n"
        info_text += f"模型路徑: {model_info['model_path']}\n"

        if 'metrics' in model_info:
            metrics = model_info['metrics']
            info_text += "\n評估指標:\n"
            info_text += f"  mAP50: {metrics.get('mAP50', 0):.4f}\n"
            info_text += f"  mAP50-95: {metrics.get('mAP50-95', 0):.4f}\n"
            info_text += f"  Precision: {metrics.get('precision', 0):.4f}\n"
            info_text += f"  Recall: {metrics.get('recall', 0):.4f}\n"

        self.info_label.setText(info_text)

    def get_selected_model(self) -> Optional[str]:
        """Get the currently selected model path.

        Returns:
            Model path or None if no selection
        """
        current_item = self.models_list.currentItem()
        if not current_item:
            return None

        model_info = current_item.data(Qt.ItemDataRole.UserRole)
        if model_info:
            return model_info['model_path']

        return None
