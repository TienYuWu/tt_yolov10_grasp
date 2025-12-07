"""Annotation tab for Smart Label application."""

import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox, QListWidget, QComboBox,
    QSplitter, QRadioButton, QButtonGroup, QSpinBox, QSlider,
    QListWidgetItem, QMenu
)
from PySide6.QtCore import Qt, Slot, QPoint
from PySide6.QtGui import QColor

from ..canvas import ImageCanvas
from ..core import ServiceContainer
from ..services import SAMService
from ..workers import BatchProcessWorker
from ..config import AppConfig, SUPPORTED_EXTENSIONS
from ..utils import (
    save_mask_as_png,
    masks_to_yolo_format,
    compute_mask_geometry,
    bbox_to_yolo_line,
    obb_to_yolo_line,
    load_image,
    numpy_to_qpixmap,
    load_saved_annotations,
)
from .batch_dialog import BatchProgressDialog


class AnnotationTab(QWidget):
    """Annotation tab with ROI and batch processing."""

    def __init__(self, config: AppConfig, service_container: ServiceContainer, parent=None):
        """Initialize annotation tab.

        Args:
            config: Application configuration
            service_container: Service container with SAM service
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        self.service_container = service_container
        self.sam_service: SAMService = service_container.get('sam_service')

        # Current state
        self.current_image_path: Optional[Path] = None
        self.image_list: List[Path] = []
        self.current_index = 0
        self.current_masks: List[Dict] = []

        # Batch processing state
        self.batch_worker: Optional[BatchProcessWorker] = None
        self.batch_results: Dict[int, List[Dict]] = {}  # {image_index: masks}
        self.batch_in_progress = False

        # Editing state
        self._selected_mask_index: Optional[int] = None  # Currently selected mask index
        self._processing_mask_click = False  # Prevent re-entrant handling
        self._original_draw_mask: Optional[np.ndarray] = None  # Track edits for prompts
        self.current_image_rgb: Optional[np.ndarray] = None  # For SAM predictor

        self._build_ui()
        self._connect_signals()

        # Load ROI config if available
        if self.config.image_dir:
            roi = self.config.load_roi_config(self.config.image_dir)
            if roi:
                self.config.roi = roi

    def _build_ui(self):
        """Build the annotation tab UI."""
        main_layout = QHBoxLayout()

        # Left panel: Canvas
        left_panel = self._create_canvas_panel()

        # Right panel: Controls
        right_panel = self._create_control_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)  # Canvas takes more space
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_canvas_panel(self) -> QWidget:
        """Create canvas panel.

        Returns:
            Canvas panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout()

        # Canvas
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas)

        # Canvas toolbar
        toolbar = self._create_canvas_toolbar()
        layout.addWidget(toolbar)

        panel.setLayout(layout)
        return panel

    def _create_canvas_toolbar(self) -> QWidget:
        """Create canvas toolbar with editing tools.

        Returns:
            Toolbar widget
        """
        toolbar = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # First row: Drawing tools
        tools_row = QHBoxLayout()

        # Tool selection (SAM / 畫筆 / 橡皮擦)
        tool_label = QLabel("🎨 工具:")
        tools_row.addWidget(tool_label)

        self.tool_group = QButtonGroup()

        self.sam_mode_radio = QRadioButton("SAM")
        self.sam_mode_radio.setChecked(True)
        self.sam_mode_radio.setToolTip("點擊添加前景/背景點")
        self.tool_group.addButton(self.sam_mode_radio, 0)
        tools_row.addWidget(self.sam_mode_radio)

        self.brush_radio = QRadioButton("✏️ 畫筆")
        self.brush_radio.setToolTip("手動繪製遮罩")
        self.tool_group.addButton(self.brush_radio, 1)
        tools_row.addWidget(self.brush_radio)

        self.eraser_radio = QRadioButton("🧹 橡皮擦")
        self.eraser_radio.setToolTip("擦除遮罩")
        self.tool_group.addButton(self.eraser_radio, 2)
        tools_row.addWidget(self.eraser_radio)

        self.tool_group.buttonClicked.connect(self._on_tool_changed)

        tools_row.addSpacing(10)

        # Brush size control
        brush_size_label = QLabel("筆刷:")
        tools_row.addWidget(brush_size_label)

        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 100)
        self.brush_size_spinbox.setValue(10)
        self.brush_size_spinbox.setSuffix(" px")
        self.brush_size_spinbox.setMaximumWidth(80)
        self.brush_size_spinbox.valueChanged.connect(self._on_brush_size_changed)
        tools_row.addWidget(self.brush_size_spinbox)

        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 100)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.setMaximumWidth(100)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        tools_row.addWidget(self.brush_size_slider)

        tools_row.addSpacing(10)

        # ROI mode toggle
        self.roi_mode_btn = QPushButton("設定 ROI")
        self.roi_mode_btn.setCheckable(True)
        self.roi_mode_btn.clicked.connect(self._on_roi_mode_toggled)
        tools_row.addWidget(self.roi_mode_btn)

        # Clear ROI
        self.clear_roi_btn = QPushButton("清除 ROI")
        self.clear_roi_btn.clicked.connect(self._on_clear_roi)
        self.clear_roi_btn.setEnabled(False)
        tools_row.addWidget(self.clear_roi_btn)

        tools_row.addStretch()
        layout.addLayout(tools_row)

        # Second row: Controls
        controls_row = QHBoxLayout()

        # SAM controls
        self.run_sam_btn = QPushButton("🔍 執行 SAM")
        self.run_sam_btn.clicked.connect(self._on_run_sam)
        self.run_sam_btn.setEnabled(False)
        controls_row.addWidget(self.run_sam_btn)

        self.clear_points_btn = QPushButton("🗑️ 清除點")
        self.clear_points_btn.clicked.connect(self._on_clear_points)
        self.clear_points_btn.setEnabled(False)
        controls_row.addWidget(self.clear_points_btn)

        self.apply_drawing_btn = QPushButton("✅ 套用繪製")
        self.apply_drawing_btn.clicked.connect(self._on_apply_drawing)
        self.apply_drawing_btn.setToolTip("將繪製的遮罩添加為新實例")
        controls_row.addWidget(self.apply_drawing_btn)

        self.clear_mask_btn = QPushButton("🗑️ 清除遮罩")
        self.clear_mask_btn.clicked.connect(self._on_clear_current_mask)
        self.clear_mask_btn.setEnabled(False)
        controls_row.addWidget(self.clear_mask_btn)

        controls_row.addSpacing(10)

        # BBox/OBB toggle
        self.show_bbox_btn = QPushButton("📦 BBox")
        self.show_bbox_btn.setCheckable(True)
        self.show_bbox_btn.toggled.connect(self._on_toggle_bbox)
        self.show_bbox_btn.setToolTip("顯示方形框")
        controls_row.addWidget(self.show_bbox_btn)

        self.show_obb_btn = QPushButton("🧭 OBB")
        self.show_obb_btn.setCheckable(True)
        self.show_obb_btn.toggled.connect(self._on_toggle_obb)
        self.show_obb_btn.setToolTip("顯示旋轉框")
        controls_row.addWidget(self.show_obb_btn)

        controls_row.addSpacing(10)

        # Image navigation
        self.prev_btn = QPushButton("◀ 上一張")
        self.prev_btn.clicked.connect(self._on_prev_image)
        self.prev_btn.setEnabled(False)
        controls_row.addWidget(self.prev_btn)

        self.image_info_label = QLabel("無圖片")
        controls_row.addWidget(self.image_info_label)

        self.next_btn = QPushButton("下一張 ▶")
        self.next_btn.clicked.connect(self._on_next_image)
        self.next_btn.setEnabled(False)
        controls_row.addWidget(self.next_btn)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        toolbar.setLayout(layout)
        return toolbar

    def _create_control_panel(self) -> QWidget:
        """Create control panel.

        Returns:
            Control panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout()

        # Image folder group
        folder_group = self._create_folder_group()
        layout.addWidget(folder_group)

        # Image list
        list_group = self._create_image_list_group()
        layout.addWidget(list_group)

        # Masks list (current image masks)
        masks_group = self._create_masks_list_group()
        layout.addWidget(masks_group)

        # ROI status
        roi_group = self._create_roi_status_group()
        layout.addWidget(roi_group)

        # Batch processing group
        batch_group = self._create_batch_group()
        layout.addWidget(batch_group)

        # Class assignment group
        class_group = self._create_class_group()
        layout.addWidget(class_group)

        # Save/Export group
        save_group = self._create_save_group()
        layout.addWidget(save_group)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _create_folder_group(self) -> QGroupBox:
        """Create folder selection group.

        Returns:
            Folder group widget
        """
        group = QGroupBox("圖片資料夾")
        layout = QVBoxLayout()

        # Folder path label
        self.folder_label = QLabel("未選擇")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        # Select folder button
        select_btn = QPushButton("選擇資料夾")
        select_btn.clicked.connect(self._on_select_folder)
        layout.addWidget(select_btn)

        group.setLayout(layout)
        return group

    def _create_image_list_group(self) -> QGroupBox:
        """Create image list group.

        Returns:
            Image list group widget
        """
        group = QGroupBox("圖片列表")
        layout = QVBoxLayout()

        # Image list widget
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self._on_image_selected)
        layout.addWidget(self.image_list_widget)

        # Image count label
        self.image_count_label = QLabel("0 張圖片")
        layout.addWidget(self.image_count_label)

        group.setLayout(layout)
        return group

    def _create_masks_list_group(self) -> QGroupBox:
        """Create masks list group for current image.

        Returns:
            Masks list group widget
        """
        group = QGroupBox("當前圖片遮罩")
        layout = QVBoxLayout()

        # Masks list widget
        self.masks_list = QListWidget()
        self.masks_list.setMaximumHeight(150)
        self.masks_list.setStyleSheet("""
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        self.masks_list.itemClicked.connect(self._on_mask_clicked)
        self.masks_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.masks_list.customContextMenuRequested.connect(self._on_mask_context_menu)
        layout.addWidget(self.masks_list)

        group.setLayout(layout)
        return group

    def _create_roi_status_group(self) -> QGroupBox:
        """Create ROI status group.

        Returns:
            ROI status group widget
        """
        group = QGroupBox("ROI 狀態")
        layout = QVBoxLayout()

        # ROI status label
        self.roi_status_label = QLabel("未設定 ROI")
        self.roi_status_label.setWordWrap(True)
        layout.addWidget(self.roi_status_label)

        # Save ROI button
        self.save_roi_btn = QPushButton("儲存 ROI 設定")
        self.save_roi_btn.clicked.connect(self._on_save_roi)
        self.save_roi_btn.setEnabled(False)
        layout.addWidget(self.save_roi_btn)

        group.setLayout(layout)
        return group

    def _create_batch_group(self) -> QGroupBox:
        """Create batch processing group.

        Returns:
            Batch group widget
        """
        group = QGroupBox("批次處理")
        layout = QVBoxLayout()

        # Batch mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("處理範圍："))
        self.batch_mode_combo = QComboBox()
        self.batch_mode_combo.addItems(["所有圖片", "選中的圖片"])
        mode_layout.addWidget(self.batch_mode_combo)
        layout.addLayout(mode_layout)

        # Start batch button
        self.start_batch_btn = QPushButton("開始批次標註")
        self.start_batch_btn.clicked.connect(self._on_start_batch)
        self.start_batch_btn.setEnabled(False)
        layout.addWidget(self.start_batch_btn)

        # Batch status
        self.batch_status_label = QLabel("準備就緒")
        self.batch_status_label.setWordWrap(True)
        layout.addWidget(self.batch_status_label)

        group.setLayout(layout)
        return group

    def _create_class_group(self) -> QGroupBox:
        """Create class assignment group.

        Returns:
            Class group widget
        """
        group = QGroupBox("類別標註")
        layout = QVBoxLayout()

        # Class selection
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("選擇類別："))
        self.class_combo = QComboBox()
        self.class_combo.addItems(list(self.config.ontology.keys()))
        class_layout.addWidget(self.class_combo)
        layout.addLayout(class_layout)

        # Assign class button
        self.assign_class_btn = QPushButton("指定類別給所有遮罩")
        self.assign_class_btn.clicked.connect(self._on_assign_class)
        self.assign_class_btn.setEnabled(False)
        layout.addWidget(self.assign_class_btn)

        group.setLayout(layout)
        return group

    def _create_save_group(self) -> QGroupBox:
        """Create save/export group.

        Returns:
            Save group widget
        """
        group = QGroupBox("儲存與匯出")
        layout = QVBoxLayout()

        # Save current image
        self.save_current_btn = QPushButton("儲存當前圖片")
        self.save_current_btn.clicked.connect(self._on_save_current)
        self.save_current_btn.setEnabled(False)
        layout.addWidget(self.save_current_btn)

        # Export all
        self.export_all_btn = QPushButton("匯出所有標註")
        self.export_all_btn.clicked.connect(self._on_export_all)
        self.export_all_btn.setEnabled(False)
        layout.addWidget(self.export_all_btn)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect canvas signals."""
        # Connect SAM request signal (if canvas has this signal)
        if hasattr(self.canvas, 'request_sam_prediction'):
            self.canvas.request_sam_prediction.connect(self._on_run_sam)

    # =============================================================================
    # Slots for UI events
    # =============================================================================

    @Slot()
    def _on_select_folder(self):
        """Handle folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇圖片資料夾",
            str(self.config.image_dir) if self.config.image_dir else ""
        )

        if not folder:
            return

        folder_path = Path(folder)
        self.config.image_dir = folder_path

        # Scan images
        self._scan_images()

        # Update UI
        self.folder_label.setText(str(folder_path))

        # Try to load ROI config
        roi = self.config.load_roi_config(folder_path)
        if roi:
            self.config.roi = roi
            self.canvas.set_roi(roi)
            self._update_roi_status()

    def _scan_images(self):
        """Scan images from selected folder."""
        if not self.config.image_dir:
            return

        self.image_list.clear()
        self.image_list_widget.clear()

        # Find all supported images
        for ext in SUPPORTED_EXTENSIONS:
            self.image_list.extend(self.config.image_dir.glob(f"*{ext}"))

        self.image_list.sort()

        # Populate list widget (ensure left alignment for items)
        for img in self.image_list:
            item = QListWidgetItem(img.name)
            try:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
            self.image_list_widget.addItem(item)

        # Update count
        self.image_count_label.setText(f"{len(self.image_list)} 張圖片")

        # Enable batch button if images exist and ROI is set
        self._update_batch_button_state()

        # Load first image if available
        if self.image_list:
            self.current_index = 0
            self._load_image(0)

    @Slot()
    def _on_image_selected(self):
        """Handle image selection from list."""
        selected_items = self.image_list_widget.selectedItems()
        if not selected_items:
            return

        selected_name = selected_items[0].text()

        # Find index
        for i, img_path in enumerate(self.image_list):
            if img_path.name == selected_name:
                self.current_index = i
                self._load_image(i)
                break

    def _load_image(self, index: int):
        """Load image at index.

        Args:
            index: Image index in list
        """
        if not (0 <= index < len(self.image_list)):
            return

        img_path = self.image_list[index]
        self.current_image_path = img_path
        self.current_index = index

        try:
            # Load image and convert to pixmap
            image_bgr, image_rgb = load_image(img_path)
            self.current_image_rgb = image_rgb  # Save for SAM predictor
            # Convert BGR to RGB for display
            pixmap = numpy_to_qpixmap(image_rgb)

            # Set image in canvas
            self.canvas.set_image(pixmap)

            # Set image for SAM predictor
            if self.sam_service and self.sam_service.predictor:
                self.sam_service.predictor.set_image(image_rgb)

            # Load masks in priority order:
            # 1. Batch results (in-memory, highest priority)
            # 2. Saved annotations (from disk)
            # 3. No masks (empty)
            masks_loaded = False

            if index in self.batch_results:
                # Priority 1: Load from batch results
                self.current_masks = self.batch_results[index]
                masks_loaded = True
            else:
                # Priority 2: Try to load from saved annotations
                output_dir = self.config.get_output_dir()
                if output_dir:
                    saved_masks = load_saved_annotations(img_path, output_dir)
                    if saved_masks:
                        self.current_masks = saved_masks
                        masks_loaded = True

            if masks_loaded:
                # Display masks on canvas
                self._display_masks_on_canvas()
            else:
                # No masks available
                self.current_masks = []
                self.canvas.clear_masks()

            # Update masks display
            self._update_masks_display()

            # Update navigation
            self.prev_btn.setEnabled(index > 0)
            self.next_btn.setEnabled(index < len(self.image_list) - 1)
            self.image_info_label.setText(f"{index + 1}/{len(self.image_list)}")

            # Update save button
            self.save_current_btn.setEnabled(len(self.current_masks) > 0)
            self.assign_class_btn.setEnabled(len(self.current_masks) > 0)
            self.clear_mask_btn.setEnabled(len(self.current_masks) > 0)

            # Enable SAM buttons if predictor is available
            has_predictor = bool(self.sam_service and self.sam_service.predictor)
            self.run_sam_btn.setEnabled(has_predictor)
            self.clear_points_btn.setEnabled(has_predictor)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法載入圖片: {e}")

    def _display_masks_on_canvas(self):
        """Display current masks on canvas with different colors."""
        self.canvas.clear_masks()

        if not self.current_masks:
            return

        # Define colors for different masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        # Display each mask with a different color
        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data.get('mask')
            if mask is None:
                continue

            color = colors[i % len(colors)]
            mask_id = f"mask_{i}"

            # Add mask overlay to canvas
            self.canvas.add_mask_overlay(mask, mask_id, color, alpha=100)

    @Slot()
    def _on_prev_image(self):
        """Load previous image."""
        if self.current_index > 0:
            self._load_image(self.current_index - 1)

    @Slot()
    def _on_next_image(self):
        """Load next image."""
        if self.current_index < len(self.image_list) - 1:
            self._load_image(self.current_index + 1)

    @Slot(bool)
    def _on_roi_mode_toggled(self, checked: bool):
        """Toggle ROI mode.

        Args:
            checked: Whether ROI mode is enabled
        """
        self.canvas.set_roi_mode(checked)

        # Update ROI status when exiting ROI mode
        if not checked:
            self._update_roi_status()

    @Slot()
    def _on_clear_roi(self):
        """Clear ROI."""
        reply = QMessageBox.question(
            self,
            "確認",
            "確定要清除 ROI 嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.clear_roi()
            self.config.roi = None
            self._update_roi_status()
            self._update_batch_button_state()

    @Slot()
    def _on_save_roi(self):
        """Save ROI configuration."""
        roi = self.canvas.get_roi()

        if not roi:
            QMessageBox.warning(self, "警告", "尚未設定 ROI")
            return

        if not self.config.image_dir:
            QMessageBox.warning(self, "警告", "尚未選擇圖片資料夾")
            return

        self.config.roi = roi
        self.config.save_roi_config(self.config.image_dir)

        QMessageBox.information(self, "成功", "ROI 設定已儲存")
        self._update_roi_status()
        self._update_batch_button_state()

    def _update_roi_status(self):
        """Update ROI status display."""
        roi = self.canvas.get_roi()

        if roi:
            status = f"已設定 ROI\n"
            status += f"位置: ({roi['x']}, {roi['y']})\n"
            status += f"大小: {roi['width']} × {roi['height']}"
            self.roi_status_label.setText(status)
            self.clear_roi_btn.setEnabled(True)
            self.save_roi_btn.setEnabled(True)
        else:
            self.roi_status_label.setText("未設定 ROI")
            self.clear_roi_btn.setEnabled(False)
            self.save_roi_btn.setEnabled(False)

    def _update_batch_button_state(self):
        """Update batch processing button state."""
        has_images = len(self.image_list) > 0
        has_roi = self.canvas.get_roi() is not None

        self.start_batch_btn.setEnabled(has_images and has_roi and not self.batch_in_progress)

    @Slot()
    def _on_start_batch(self):
        """Start batch processing."""
        roi = self.canvas.get_roi()

        if not roi:
            QMessageBox.warning(self, "警告", "請先設定 ROI")
            return

        if not self.image_list:
            QMessageBox.warning(self, "警告", "沒有可處理的圖片")
            return

        # Determine which images to process
        batch_mode = self.batch_mode_combo.currentText()

        if batch_mode == "選中的圖片":
            selected_items = self.image_list_widget.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "警告", "請先選擇要處理的圖片")
                return

            selected_names = [item.text() for item in selected_items]
            images_to_process = [
                img for img in self.image_list if img.name in selected_names
            ]
        else:
            images_to_process = self.image_list.copy()

        if not images_to_process:
            return

        # Create progress dialog
        progress_dialog = BatchProgressDialog(len(images_to_process), self)

        # Create worker
        self.batch_worker = BatchProcessWorker(
            images_to_process,
            roi,
            self.sam_service,
            self.config
        )

        # Connect signals
        self.batch_worker.progress_updated.connect(progress_dialog.update_progress)
        self.batch_worker.image_processed.connect(progress_dialog.on_image_processed)
        self.batch_worker.image_processed.connect(self._on_batch_image_processed)
        self.batch_worker.batch_completed.connect(self._on_batch_completed)
        self.batch_worker.batch_completed.connect(progress_dialog.close)
        progress_dialog.canceled.connect(self.batch_worker.cancel)

        # Update state
        self.batch_in_progress = True
        self._update_batch_button_state()
        self.batch_status_label.setText("批次處理中...")

        # Start worker
        self.batch_worker.start()
        progress_dialog.exec()

    @Slot(int, list, bool, str)
    def _on_batch_image_processed(self, index: int, masks: list, success: bool, image_name: str):
        """Handle individual image processing completion.

        Args:
            index: Image index in batch
            masks: Generated masks
            success: Whether processing succeeded
            image_name: Image name
        """
        if success and masks:
            # Find the actual index in full image list
            for i, img_path in enumerate(self.image_list):
                if img_path.name == image_name:
                    self.batch_results[i] = masks
                    break

    @Slot(dict)
    def _on_batch_completed(self, stats: dict):
        """Handle batch processing completion.

        Args:
            stats: Processing statistics
        """
        self.batch_in_progress = False
        self._update_batch_button_state()

        # Show results
        message = f"批次處理完成！\n\n"
        message += f"成功：{stats['success']} 張\n"
        message += f"失敗：{stats['failed']} 張\n"
        message += f"總計：{stats['total']} 張\n\n"
        message += "請逐張確認並修改標註結果"

        if stats['failed_images']:
            message += "\n\n失敗的圖片：\n"
            for failed in stats['failed_images'][:5]:  # Show first 5
                message += f"- {failed['name']}: {failed['error']}\n"

        QMessageBox.information(self, "批次處理完成", message)

        self.batch_status_label.setText(
            f"已完成 - 成功 {stats['success']} / 失敗 {stats['failed']}"
        )

        # Enable export if we have results
        if self.batch_results:
            self.export_all_btn.setEnabled(True)

        # Reload current image to show masks if available
        if self.current_index in self.batch_results:
            self._load_image(self.current_index)

    @Slot()
    def _on_assign_class(self):
        """Assign class to all masks of current image."""
        if not self.current_masks:
            QMessageBox.warning(self, "警告", "當前圖片沒有遮罩")
            return

        selected_class = self.class_combo.currentText()
        class_id = list(self.config.ontology.keys()).index(selected_class)

        # Update all masks
        for mask in self.current_masks:
            mask['class_name'] = selected_class
            mask['class_id'] = class_id

        # Update masks display
        self._update_masks_display()

        QMessageBox.information(
            self,
            "成功",
            f"已將所有遮罩設為類別：{selected_class}"
        )

    @Slot()
    def _on_save_current(self):
        """Save current image annotations."""
        if not self.current_masks:
            QMessageBox.warning(self, "警告", "當前圖片沒有標註")
            return

        if self.current_image_path is None:
            QMessageBox.warning(self, "警告", "沒有載入圖片")
            return

        try:
            # Ensure output directories exist
            self.config.ensure_output_dirs()

            image_path = self.current_image_path
            image_name = image_path.stem

            # Load image to get dimensions
            image_bgr, image_rgb = load_image(image_path)
            height, width = image_rgb.shape[:2]

            # Save individual mask PNGs and collect metadata
            mask_info = []
            yolo_labels = []
            bbox_labels = []
            obb_labels = []

            for i, mask_data in enumerate(self.current_masks):
                mask = mask_data.get("mask")
                if mask is None:
                    continue

                class_name = mask_data.get("class_name", "未分類")
                class_id = mask_data.get("class_id", 0)
                mask_id = f"mask_{i}"

                # Save mask as PNG
                mask_filename = f"{image_name}_{mask_id}.png"
                mask_path = self.config.get_masks_dir() / mask_filename
                save_mask_as_png(mask, mask_path)

                # Convert to YOLO format
                yolo_line = masks_to_yolo_format(mask, class_id, width, height)
                if yolo_line:
                    yolo_labels.append(yolo_line)

                # Compute geometry
                geometry = compute_mask_geometry(mask)

                bbox_line = None
                obb_line = None
                if geometry:
                    bbox = geometry.get("bbox")
                    if bbox:
                        bbox_line = bbox_to_yolo_line(bbox, class_id, width, height)
                        if bbox_line:
                            bbox_labels.append(bbox_line)

                    obb = geometry.get("obb")
                    if obb and obb.get("points"):
                        obb_line = obb_to_yolo_line(obb["points"], class_id, width, height)
                        if obb_line:
                            obb_labels.append(obb_line)

                # Store mask info
                info_entry = {
                    "mask_id": mask_id,
                    "mask_file": mask_filename,
                    "class_name": class_name,
                    "class_id": class_id,
                }

                if geometry:
                    if geometry.get("bbox"):
                        info_entry["bbox"] = geometry["bbox"]
                    if geometry.get("bbox_xywh"):
                        info_entry["bbox_xywh"] = geometry["bbox_xywh"]
                    if geometry.get("bbox_xyxy"):
                        info_entry["bbox_xyxy"] = geometry["bbox_xyxy"]
                    if geometry.get("obb"):
                        info_entry["obb"] = geometry["obb"]

                mask_info.append(info_entry)

            # Save YOLO labels
            yolo_path = self.config.get_labels_dir() / f"{image_name}.txt"
            with open(yolo_path, "w") as f:
                f.write("\n".join(yolo_labels))

            bbox_path = self.config.get_bbox_labels_dir() / f"{image_name}.txt"
            with open(bbox_path, "w") as f:
                f.write("\n".join(bbox_labels))

            obb_path = self.config.get_obb_labels_dir() / f"{image_name}.txt"
            with open(obb_path, "w") as f:
                f.write("\n".join(obb_labels))

            # Copy original image
            output_image_path = self.config.get_images_dir() / image_path.name
            shutil.copy2(image_path, output_image_path)

            # Save metadata JSON
            metadata = {
                "image_file": image_path.name,
                "image_width": width,
                "image_height": height,
                "num_instances": len(self.current_masks),
                "instances": mask_info,
            }

            metadata_path = self.config.get_metadata_dir() / f"{image_name}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            num_instances = len(self.current_masks)
            output_dir = self.config.get_output_dir()
            QMessageBox.information(
                self,
                "成功",
                f"標註已儲存至:\n{output_dir}\n\n"
                f"實例數量: {num_instances}\n"
                f"YOLO 分割標籤: {len(yolo_labels)} 行\n"
                f"BBox 標籤: {len(bbox_labels)} 行\n"
                f"OBB 標籤: {len(obb_labels)} 行"
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存失敗: {e}")

    @Slot()
    def _on_export_all(self):
        """Export all annotations."""
        if not self.batch_results:
            QMessageBox.warning(self, "警告", "沒有可匯出的標註")
            return

        # Confirm export
        output_dir = self.config.get_output_dir()
        reply = QMessageBox.question(
            self,
            "確認匯出",
            f"確定要匯出 {len(self.batch_results)} 張已標註的圖片嗎？\n\n"
            f"匯出目錄：{output_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Ensure output directories exist
            self.config.ensure_output_dirs()

            success_count = 0
            failed_count = 0
            failed_images = []

            # Process each image with batch results
            for image_index, masks in self.batch_results.items():
                if image_index >= len(self.image_list):
                    continue

                image_path = self.image_list[image_index]
                image_name = image_path.stem

                try:
                    # Load image to get dimensions
                    image_bgr, image_rgb = load_image(image_path)
                    height, width = image_rgb.shape[:2]

                    # Save individual mask PNGs and collect metadata
                    mask_info = []
                    yolo_labels = []
                    bbox_labels = []
                    obb_labels = []

                    for i, mask_data in enumerate(masks):
                        mask = mask_data.get("mask")
                        if mask is None:
                            continue

                        class_name = mask_data.get("class_name", "未分類")
                        class_id = mask_data.get("class_id", 0)
                        mask_id = f"mask_{i}"

                        # Save mask as PNG
                        mask_filename = f"{image_name}_{mask_id}.png"
                        mask_path = self.config.get_masks_dir() / mask_filename
                        save_mask_as_png(mask, mask_path)

                        # Convert to YOLO format
                        yolo_line = masks_to_yolo_format(mask, class_id, width, height)
                        if yolo_line:
                            yolo_labels.append(yolo_line)

                        # Compute geometry
                        geometry = compute_mask_geometry(mask)

                        if geometry:
                            bbox = geometry.get("bbox")
                            if bbox:
                                bbox_line = bbox_to_yolo_line(bbox, class_id, width, height)
                                if bbox_line:
                                    bbox_labels.append(bbox_line)

                            obb = geometry.get("obb")
                            if obb and obb.get("points"):
                                obb_line = obb_to_yolo_line(obb["points"], class_id, width, height)
                                if obb_line:
                                    obb_labels.append(obb_line)

                        # Store mask info
                        info_entry = {
                            "mask_id": mask_id,
                            "mask_file": mask_filename,
                            "class_name": class_name,
                            "class_id": class_id,
                        }

                        if geometry:
                            if geometry.get("bbox"):
                                info_entry["bbox"] = geometry["bbox"]
                            if geometry.get("bbox_xywh"):
                                info_entry["bbox_xywh"] = geometry["bbox_xywh"]
                            if geometry.get("bbox_xyxy"):
                                info_entry["bbox_xyxy"] = geometry["bbox_xyxy"]
                            if geometry.get("obb"):
                                info_entry["obb"] = geometry["obb"]

                        mask_info.append(info_entry)

                    # Save YOLO labels
                    yolo_path = self.config.get_labels_dir() / f"{image_name}.txt"
                    with open(yolo_path, "w") as f:
                        f.write("\n".join(yolo_labels))

                    bbox_path = self.config.get_bbox_labels_dir() / f"{image_name}.txt"
                    with open(bbox_path, "w") as f:
                        f.write("\n".join(bbox_labels))

                    obb_path = self.config.get_obb_labels_dir() / f"{image_name}.txt"
                    with open(obb_path, "w") as f:
                        f.write("\n".join(obb_labels))

                    # Copy original image
                    output_image_path = self.config.get_images_dir() / image_path.name
                    shutil.copy2(image_path, output_image_path)

                    # Save metadata JSON
                    metadata = {
                        "image_file": image_path.name,
                        "image_width": width,
                        "image_height": height,
                        "num_instances": len(masks),
                        "instances": mask_info,
                    }

                    metadata_path = self.config.get_metadata_dir() / f"{image_name}.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    success_count += 1

                except Exception as e:
                    failed_count += 1
                    failed_images.append(f"{image_name}: {str(e)}")
                    print(f"❌ 匯出失敗 - {image_name}: {e}")

            # Show results
            message = f"批次匯出完成！\n\n"
            message += f"成功：{success_count} 張\n"
            message += f"失敗：{failed_count} 張\n"
            message += f"匯出路徑：{output_dir}"

            if failed_images:
                message += "\n\n失敗的圖片：\n"
                for failed in failed_images[:5]:  # Show first 5
                    message += f"- {failed}\n"
                if len(failed_images) > 5:
                    message += f"...還有 {len(failed_images) - 5} 張"

            QMessageBox.information(self, "匯出完成", message)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"匯出過程發生錯誤：{e}")

    # =============================================================================
    # 1. Tool Change and Brush Size Handlers
    # =============================================================================

    @Slot()
    def _on_tool_changed(self, button):
        """Handle drawing tool change."""
        tool_id = self.tool_group.id(button)

        if tool_id == 0:  # SAM mode
            self.canvas.set_drawing_mode(False)
            if hasattr(self.canvas, 'set_eraser_mode'):
                self.canvas.set_eraser_mode(False)
            # Clear drawing buffer
            if hasattr(self.canvas, '_current_draw_mask'):
                self.canvas._current_draw_mask = None
            self._original_draw_mask = None
            self.canvas.remove_mask_overlay("drawing")
        elif tool_id == 1:  # Brush
            if self._selected_mask_index is None:
                QMessageBox.warning(
                    self, "提示",
                    "請先在遮罩列表中點擊選擇要編輯的遮罩"
                )
                self.sam_mode_radio.setChecked(True)
                return
            self._load_mask_for_editing()
            self.canvas.set_drawing_mode(True)
            if hasattr(self.canvas, 'set_eraser_mode'):
                self.canvas.set_eraser_mode(False)
        elif tool_id == 2:  # Eraser
            if self._selected_mask_index is None:
                QMessageBox.warning(
                    self, "提示",
                    "請先在遮罩列表中點擊選擇要編輯的遮罩"
                )
                self.sam_mode_radio.setChecked(True)
                return
            self._load_mask_for_editing()
            self.canvas.set_drawing_mode(True)
            if hasattr(self.canvas, 'set_eraser_mode'):
                self.canvas.set_eraser_mode(True)

    @Slot(int)
    def _on_brush_size_changed(self, value: int):
        """Handle brush size change."""
        self.brush_size_spinbox.blockSignals(True)
        self.brush_size_slider.blockSignals(True)

        self.brush_size_spinbox.setValue(value)
        self.brush_size_slider.setValue(value)

        self.brush_size_spinbox.blockSignals(False)
        self.brush_size_slider.blockSignals(False)

        if hasattr(self.canvas, 'set_brush_size'):
            self.canvas.set_brush_size(value)

    # =============================================================================
    # 2. Mask Management Methods
    # =============================================================================

    def _update_masks_display(self):
        """Update the masks list display."""
        self.masks_list.blockSignals(True)
        try:
            self.masks_list.clear()

            if not self.current_masks:
                empty_item = QListWidgetItem("（無標註）")
                try:
                    empty_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                except Exception:
                    pass
                self.masks_list.addItem(empty_item)
                return

            grouped_masks = {}
            for i, mask_data in enumerate(self.current_masks):
                class_name = mask_data.get("class_name", "未分類")
                grouped_masks.setdefault(class_name, []).append((i, mask_data))

            for class_name, masks in sorted(grouped_masks.items()):
                header_item = QListWidgetItem(f"📦 {class_name} ({len(masks)} 個)")
                header_item.setData(Qt.ItemDataRole.UserRole, None)
                header_item.setForeground(QColor(0, 100, 200))
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                try:
                    header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                except Exception:
                    pass
                self.masks_list.addItem(header_item)

                for idx, (mask_index, mask_data) in enumerate(masks):
                    is_auto = mask_data.get('auto_generated', False)
                    icon = "🤖" if is_auto else "✏️"
                    item_text = f"  {icon} {class_name} - mask_{mask_index}"

                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, mask_index)
                    try:
                        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    except Exception:
                        pass
                    self.masks_list.addItem(item)
        finally:
            self.masks_list.blockSignals(False)

    @Slot()
    def _on_mask_clicked(self, item: QListWidgetItem):
        """Handle mask item clicked."""
        if self._processing_mask_click:
            return

        mask_index = item.data(Qt.ItemDataRole.UserRole)
        if mask_index is None:
            return

        self._processing_mask_click = True
        try:
            if self._selected_mask_index == mask_index:
                self._selected_mask_index = None
                self._display_masks_on_canvas()

                if hasattr(self.canvas, '_drawing_mode') and self.canvas._drawing_mode:
                    if hasattr(self.canvas, '_current_draw_mask'):
                        self.canvas._current_draw_mask = None
                    self._original_draw_mask = None
                    self.canvas.remove_mask_overlay("drawing")
                    self.sam_mode_radio.setChecked(True)
            else:
                self._selected_mask_index = mask_index
                self._highlight_selected_mask()

                if hasattr(self.canvas, '_drawing_mode') and self.canvas._drawing_mode:
                    self._load_mask_for_editing()
        finally:
            self._processing_mask_click = False

    @Slot(QPoint)
    def _on_mask_context_menu(self, pos: QPoint):
        """Handle right-click context menu on mask."""
        item = self.masks_list.itemAt(pos)
        if item is None:
            return

        mask_index = item.data(Qt.ItemDataRole.UserRole)
        if mask_index is None:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("🗑️ 刪除此遮罩")
        action = menu.exec(self.masks_list.mapToGlobal(pos))

        if action == delete_action:
            self._delete_mask(mask_index)

    def _delete_mask(self, mask_index: int):
        """Delete a specific mask."""
        if mask_index >= len(self.current_masks):
            return

        reply = QMessageBox.question(
            self,
            "確認刪除",
            f"確定要刪除遮罩 'mask_{mask_index}' 嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.current_masks[mask_index]
            self.canvas.remove_mask_overlay(f"mask_{mask_index}")

            if self._selected_mask_index == mask_index:
                self._selected_mask_index = None

            self._update_masks_display()
            self._display_masks_on_canvas()

    def _highlight_selected_mask(self):
        """Highlight the selected mask."""
        if self._selected_mask_index is None:
            return

        self.canvas.clear_masks()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data.get('mask')
            if mask is None:
                continue

            color = colors[i % len(colors)]
            mask_id = f"mask_{i}"

            if i == self._selected_mask_index:
                self.canvas.add_mask_overlay(mask, mask_id, (255, 255, 0), alpha=180)
            else:
                self.canvas.add_mask_overlay(mask, mask_id, color, alpha=100)

    def _load_mask_for_editing(self):
        """Load the selected mask into canvas drawing buffer."""
        if self._selected_mask_index is None or self._selected_mask_index >= len(self.current_masks):
            return

        mask_data = self.current_masks[self._selected_mask_index]
        mask = mask_data.get("mask")
        if mask is None:
            return

        if hasattr(self.canvas, '_current_draw_mask'):
            self.canvas._current_draw_mask = mask.copy()
        self._original_draw_mask = mask.copy()
        self.canvas.add_mask_overlay(mask, "drawing", (255, 144, 0), alpha=120)

    # =============================================================================
    # 3. SAM Manual Annotation Methods
    # =============================================================================

    @Slot()
    def _on_run_sam(self):
        """Run SAM prediction based on current points."""
        if not self.sam_service or not self.sam_service.predictor:
            QMessageBox.warning(self, "警告", "SAM 模型未載入")
            return

        if self.current_image_rgb is None:
            return

        if not hasattr(self.canvas, 'get_points_and_labels'):
            QMessageBox.warning(self, "警告", "Canvas 不支援點標註")
            return

        points, labels = self.canvas.get_points_and_labels()
        if len(points) == 0:
            QMessageBox.warning(self, "提示", "請先添加標註點")
            return

        try:
            masks, scores, logits = self.sam_service.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

            selected_class = self.class_combo.currentText()
            class_id = 0
            if selected_class in self.config.ontology:
                class_id = list(self.config.ontology.keys()).index(selected_class)

            self.current_masks.append({
                "mask": best_mask,
                "class_name": selected_class,
                "class_id": class_id,
                "auto_generated": False,
                "confidence": float(scores[best_idx])
            })

            self._display_masks_on_canvas()
            self._update_masks_display()

            if hasattr(self.canvas, 'clear_points'):
                self.canvas.clear_points()

            self.save_current_btn.setEnabled(True)
            self.clear_mask_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"SAM 預測失敗: {e}")

    @Slot()
    def _on_clear_points(self):
        """Clear all annotation points."""
        if hasattr(self.canvas, 'clear_points'):
            self.canvas.clear_points()

    @Slot()
    def _on_apply_drawing(self):
        """Apply the currently drawn mask."""
        draw_mask = None
        if hasattr(self.canvas, 'get_current_draw_mask'):
            draw_mask = self.canvas.get_current_draw_mask()

        if draw_mask is None or not draw_mask.any():
            QMessageBox.warning(self, "警告", "沒有繪製內容")
            return

        selected_class = self.class_combo.currentText()
        class_id = 0
        if selected_class in self.config.ontology:
            class_id = list(self.config.ontology.keys()).index(selected_class)

        if self._selected_mask_index is not None and self._selected_mask_index < len(self.current_masks):
            self.current_masks[self._selected_mask_index]["mask"] = draw_mask.copy()
        else:
            self.current_masks.append({
                "mask": draw_mask.copy(),
                "class_name": selected_class,
                "class_id": class_id,
                "auto_generated": False
            })

        if hasattr(self.canvas, '_current_draw_mask'):
            self.canvas._current_draw_mask = None
        self._original_draw_mask = None
        self.canvas.remove_mask_overlay("drawing")

        self._selected_mask_index = None
        self.sam_mode_radio.setChecked(True)

        self._display_masks_on_canvas()
        self._update_masks_display()

        self.save_current_btn.setEnabled(True)
        self.clear_mask_btn.setEnabled(True)

    @Slot()
    def _on_clear_current_mask(self):
        """Clear current mask or all masks."""
        if not self.current_masks:
            return

        num_masks = len(self.current_masks)

        if num_masks > 1:
            reply = QMessageBox.question(
                self,
                "清除遮罩",
                f"當前有 {num_masks} 個遮罩\n\n是否清除所有遮罩？\n（否則只清除最後一個）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                self.canvas.clear_masks()
                self.current_masks.clear()
            else:
                del self.current_masks[-1]
        else:
            self.canvas.clear_masks()
            self.current_masks.clear()

        self._display_masks_on_canvas()
        self._update_masks_display()

        self.save_current_btn.setEnabled(len(self.current_masks) > 0)
        self.clear_mask_btn.setEnabled(len(self.current_masks) > 0)

    # =============================================================================
    # 4. BBox/OBB Display Methods
    # =============================================================================

    @Slot(bool)
    def _on_toggle_bbox(self, checked: bool):
        """Handle toggle for displaying bounding boxes."""
        if checked:
            self._show_bbox_overlays()
        else:
            if hasattr(self.canvas, 'clear_bbox_overlays'):
                self.canvas.clear_bbox_overlays()

    @Slot(bool)
    def _on_toggle_obb(self, checked: bool):
        """Handle toggle for displaying oriented bounding boxes."""
        if checked:
            self._show_obb_overlays()
        else:
            if hasattr(self.canvas, 'clear_obb_overlays'):
                self.canvas.clear_obb_overlays()

    def _show_bbox_overlays(self):
        """Show bounding boxes for current masks."""
        if not self.current_masks or not hasattr(self.canvas, 'show_bbox_overlays'):
            return

        bbox_map = {}
        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data.get('mask')
            if mask is None:
                continue

            geometry = compute_mask_geometry(mask)
            if geometry and geometry.get("bbox"):
                bbox_map[f"mask_{i}"] = geometry["bbox"]

        if bbox_map:
            self.canvas.show_bbox_overlays(bbox_map)

    def _show_obb_overlays(self):
        """Show oriented bounding boxes for current masks."""
        if not self.current_masks or not hasattr(self.canvas, 'show_obb_overlays'):
            return

        obb_map = {}
        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data.get('mask')
            if mask is None:
                continue

            geometry = compute_mask_geometry(mask)
            if geometry and geometry.get("obb") and geometry["obb"].get("points"):
                obb_map[f"mask_{i}"] = geometry["obb"]["points"]

        if obb_map:
            self.canvas.show_obb_overlays(obb_map)
