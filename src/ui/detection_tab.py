"""Detection Tab - YOLO OBB Detection with 6D Pose Estimation

Supports both static images and RealSense camera live feed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox, QRadioButton,
    QButtonGroup, QCheckBox, QSplitter, QLineEdit, QFrame, QScrollArea,
    QSlider, QSpinBox, QDoubleSpinBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView
)


class NoWheelSpinBox(QSpinBox):
    """SpinBox that ignores mouse wheel to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores mouse wheel to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()

from .detection_canvas import DetectionCanvas
from .widgets import OutputConsole
from ..config import AppConfig
from ..services.detection_service import DetectionService
from ..workers.detection_worker import DetectionWorker


class DetectionTab(QWidget):
    """Detection tab for YOLO OBB + 6D pose estimation."""

    def __init__(self, config: AppConfig, model_path: str, parent=None):
        """Initialize detection tab.

        Args:
            config: Application configuration
            model_path: Path to YOLO OBB model weights
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        self.model_path = model_path

        # Services and workers
        self.detection_service: Optional[DetectionService] = None
        self.detection_worker: Optional[DetectionWorker] = None
        self.camera_adapter = None  # Will be initialized when needed

        # 3D visualization removed

        # Current state
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[Dict] = None
        self.current_annotated: Optional[np.ndarray] = None
        self.is_camera_active = False

        # Camera intrinsics for image mode
        self.custom_intrinsics = self._load_intrinsics_from_config()

        # Camera intrinsics for camera mode (read from RealSense factory calibration)
        self.camera_intrinsics = None

        # Detection parameters (tunable at runtime)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45

        # Start/Stop toggle state
        self.is_detecting = False

        self._build_ui()
        self._connect_signals()
        self._initialize_service()

    def _build_ui(self):
        """Build the detection tab UI with a two-pane layout.

        Layout structure:
        - Horizontal Splitter: Canvas (left) + Controls (right)
        - Output console kept hidden for logging only (no bottom pane)
        """
        main_layout = QVBoxLayout()

        # Hidden output console (kept for logging but not shown in UI)
        self.output_console = self._create_output_console()
        self.output_console.setVisible(False)

        # ========== TOP SECTION: Canvas + Control Panel ==========
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel: Canvas (70%)
        left_panel = self._create_canvas_panel()
        left_panel.setMinimumWidth(520)

        # Right panel: Controls (30%)
        right_panel = self._create_control_panel()
        right_panel.setMinimumWidth(320)

        # Inner horizontal splitter
        horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)
        horizontal_splitter.addWidget(left_panel)
        horizontal_splitter.addWidget(right_panel)
        horizontal_splitter.setStretchFactor(0, 7)  # 70% for canvas
        horizontal_splitter.setStretchFactor(1, 3)  # 30% for controls
        horizontal_splitter.setChildrenCollapsible(False)

        top_layout.addWidget(horizontal_splitter)
        top_widget.setLayout(top_layout)

        main_layout.addWidget(top_widget)
        self.setLayout(main_layout)

    def _create_canvas_panel(self) -> QWidget:
        """Create canvas panel with detection visualization.

        Returns:
            Canvas panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout()

        # Canvas
        self.canvas = DetectionCanvas()
        layout.addWidget(self.canvas)

        # Canvas toolbar (zoom controls)
        toolbar = self._create_canvas_toolbar()
        layout.addWidget(toolbar)

        panel.setLayout(layout)
        return panel

    def _create_canvas_toolbar(self) -> QWidget:
        """Create canvas toolbar with zoom controls.

        Returns:
            Toolbar widget
        """
        toolbar = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Zoom controls
        zoom_label = QLabel("🔍 縮放:")
        layout.addWidget(zoom_label)

        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("縮小 (-)")
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        layout.addWidget(zoom_out_btn)

        fit_btn = QPushButton("適應視窗")
        fit_btn.clicked.connect(self.canvas.reset_zoom)
        layout.addWidget(fit_btn)

        layout.addStretch()

        toolbar.setLayout(layout)
        return toolbar

    def _create_control_panel(self) -> QWidget:
        """Create control panel with reorganized workflow.

        Workflow: Setup → Action → Visualization → Storage

        Returns:
            Control panel widget
        """
        panel = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== Pinned: Setup (Collapsible with internal scroll) ==========
        setup_group = self._create_setup_group()
        main_layout.addWidget(setup_group)

        # ========== Pinned: Action ==========
        action_label = QLabel("▶️ Action")
        action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ec9b0;")
        main_layout.addWidget(action_label)

        action_button_widget = self._create_action_button()
        main_layout.addWidget(action_button_widget, 0)

        # ========== Pinned: Results ==========
        results_label = QLabel("📄 Results")
        results_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #c586c0;")
        main_layout.addWidget(results_label)

        results_group = self._create_results_group()
        main_layout.addWidget(results_group)

        # ========== Pinned: Storage ==========
        storage_label = QLabel("💾 Storage")
        storage_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #dcdcaa;")
        main_layout.addWidget(storage_label)

        save_group = self._create_save_group()
        main_layout.addWidget(save_group)

        # ========== Pinned: Status ==========
        status_group = self._create_status_group()
        main_layout.addWidget(status_group)

        panel.setLayout(main_layout)
        return panel

    def _create_setup_group(self) -> QGroupBox:
        """Create collapsible Setup group with scrollable content.

        Contains: Input Source, Model Settings, Pose Mode, Intrinsics, Visualization

        Returns:
            Setup group box
        """
        group = QGroupBox("⚙️ Setup")
        group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        group.setCheckable(True)
        group.setChecked(True)  # Expanded by default
        main_layout = QVBoxLayout()

        # Create internal scroll area for all setup items
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        scroll_area.setMaximumHeight(400)  # Limit height to allow scrolling

        scroll_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # 1. Input source
        input_group = self._create_input_source_group()
        layout.addWidget(input_group)

        # 2. Model settings (now includes parameters)
        model_group = self._create_model_settings_group()
        layout.addWidget(model_group)

        # 3. Pose mode
        pose_group = self._create_pose_mode_group()
        layout.addWidget(pose_group)

        # 4. Camera intrinsics (inline editable)
        intrinsics_group = self._create_intrinsics_group()
        layout.addWidget(intrinsics_group)

        # Flexible space
        layout.addStretch()

        scroll_widget.setLayout(layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        group.setLayout(main_layout)
        return group

    def _create_input_source_group(self) -> QGroupBox:
        """Create input source selection group."""
        group = QGroupBox("📂 輸入來源")
        layout = QVBoxLayout()

        # Mode selection
        self.input_mode_group = QButtonGroup()

        self.image_mode_radio = QRadioButton("圖片檔案")
        self.image_mode_radio.setChecked(True)
        self.input_mode_group.addButton(self.image_mode_radio, 0)
        layout.addWidget(self.image_mode_radio)

        # Image file row
        image_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("選擇圖片...")
        image_row.addWidget(self.image_path_edit)

        self.browse_btn = QPushButton("瀏覽")
        self.browse_btn.clicked.connect(self._on_browse_image)
        image_row.addWidget(self.browse_btn)

        layout.addLayout(image_row)

        # Camera mode
        self.camera_mode_radio = QRadioButton("RealSense 相機")
        self.input_mode_group.addButton(self.camera_mode_radio, 1)
        layout.addWidget(self.camera_mode_radio)

        # Camera controls
        camera_row = QHBoxLayout()
        self.start_camera_btn = QPushButton("啟動相機")
        self.start_camera_btn.clicked.connect(self._on_start_camera)
        camera_row.addWidget(self.start_camera_btn)

        self.stop_camera_btn = QPushButton("停止")
        self.stop_camera_btn.clicked.connect(self._on_stop_camera)
        self.stop_camera_btn.setEnabled(False)
        camera_row.addWidget(self.stop_camera_btn)

        layout.addLayout(camera_row)

        group.setLayout(layout)
        return group

    def _create_pose_mode_group(self) -> QGroupBox:
        """Create pose estimation mode selection group.

        Returns:
            Pose mode group box
        """
        group = QGroupBox("📐 姿態估計模式")
        layout = QVBoxLayout()

        self.pose_mode_group = QButtonGroup()

        self.simple_mode_radio = QRadioButton("Simple (直上直下)")
        self.simple_mode_radio.setChecked(True)
        self.simple_mode_radio.setToolTip("僅 Z 軸旋轉,適合垂直抓取")
        self.pose_mode_group.addButton(self.simple_mode_radio, 0)
        layout.addWidget(self.simple_mode_radio)

        self.full_mode_radio = QRadioButton("Full (完整 6D 姿態)")
        self.full_mode_radio.setToolTip("完整 6 自由度姿態,含表面法向量")
        self.pose_mode_group.addButton(self.full_mode_radio, 1)
        layout.addWidget(self.full_mode_radio)

        group.setLayout(layout)
        return group

    def _create_model_settings_group(self) -> QGroupBox:
        """Create model settings group including path and detection parameters.

        Returns:
            Model settings group box
        """
        group = QGroupBox("🤖 模型設定")
        layout = QVBoxLayout()

        # Model path button
        model_btn = QPushButton("模型路徑設定")
        model_btn.clicked.connect(self._on_model_settings)
        model_btn.setToolTip("設定 YOLO OBB 模型檔案路徑")
        layout.addWidget(model_btn)

        # Model status label
        self.model_status_label = QLabel("模型: 未載入")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: gray; }")
        layout.addWidget(self.model_status_label)

        # Separator
        layout.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addSpacing(5)

        # Detection parameters
        params_title = QLabel("🎚️ 檢測參數")
        params_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(params_title)

        # Confidence threshold slider
        conf_label = QLabel("信心度閖值 (Confidence)")
        layout.addWidget(conf_label)

        conf_slider_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        conf_slider_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(40)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        conf_slider_layout.addWidget(self.confidence_label)
        layout.addLayout(conf_slider_layout)

        # Separator
        layout.addSpacing(10)

        # IOU threshold slider
        iou_label = QLabel("IoU 閖值")
        layout.addWidget(iou_label)

        iou_slider_layout = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(0)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        iou_slider_layout.addWidget(self.iou_slider)

        self.iou_label = QLabel("0.45")
        self.iou_label.setMinimumWidth(40)
        self.iou_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        iou_slider_layout.addWidget(self.iou_label)
        layout.addLayout(iou_slider_layout)

        group.setLayout(layout)
        return group

    def _create_intrinsics_group(self) -> QGroupBox:
        """Create camera intrinsics inline editor group with apply confirmation."""
        group = QGroupBox("📷 相機內部參數")
        group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; }")
        layout = QVBoxLayout()

        # Resolution (stacked, minimal left/right splitting)
        layout.addWidget(QLabel("解析度"))
        res_row = QHBoxLayout()
        self.intrinsics_width = NoWheelSpinBox()
        self.intrinsics_width.setRange(320, 3840)
        self.intrinsics_width.setValue(self.custom_intrinsics['width'])
        self.intrinsics_width.setToolTip("影像寬度 (pixels)")
        self.intrinsics_width.valueChanged.connect(self._on_intrinsics_changed)
        res_row.addWidget(self.intrinsics_width)
        res_row.addWidget(QLabel("×"))
        self.intrinsics_height = NoWheelSpinBox()
        self.intrinsics_height.setRange(240, 2160)
        self.intrinsics_height.setValue(self.custom_intrinsics['height'])
        self.intrinsics_height.setToolTip("影像高度 (pixels)")
        self.intrinsics_height.valueChanged.connect(self._on_intrinsics_changed)
        res_row.addWidget(self.intrinsics_height)
        layout.addLayout(res_row)

        layout.addWidget(QLabel("fx"))
        self.intrinsics_fx = NoWheelDoubleSpinBox()
        self.intrinsics_fx.setRange(1.0, 5000.0)
        self.intrinsics_fx.setDecimals(2)
        self.intrinsics_fx.setValue(self.custom_intrinsics['fx'])
        self.intrinsics_fx.setToolTip("焦距 X (pixels)")
        self.intrinsics_fx.valueChanged.connect(self._on_intrinsics_changed)
        layout.addWidget(self.intrinsics_fx)

        layout.addWidget(QLabel("fy"))
        self.intrinsics_fy = NoWheelDoubleSpinBox()
        self.intrinsics_fy.setRange(1.0, 5000.0)
        self.intrinsics_fy.setDecimals(2)
        self.intrinsics_fy.setValue(self.custom_intrinsics['fy'])
        self.intrinsics_fy.setToolTip("焦距 Y (pixels)")
        self.intrinsics_fy.valueChanged.connect(self._on_intrinsics_changed)
        layout.addWidget(self.intrinsics_fy)

        layout.addWidget(QLabel("cx"))
        self.intrinsics_cx = NoWheelDoubleSpinBox()
        self.intrinsics_cx.setRange(0.0, 3840.0)
        self.intrinsics_cx.setDecimals(2)
        self.intrinsics_cx.setValue(self.custom_intrinsics['cx'])
        self.intrinsics_cx.setToolTip("主點 X (pixels)")
        self.intrinsics_cx.valueChanged.connect(self._on_intrinsics_changed)
        layout.addWidget(self.intrinsics_cx)

        layout.addWidget(QLabel("cy"))
        self.intrinsics_cy = NoWheelDoubleSpinBox()
        self.intrinsics_cy.setRange(0.0, 2160.0)
        self.intrinsics_cy.setDecimals(2)
        self.intrinsics_cy.setValue(self.custom_intrinsics['cy'])
        self.intrinsics_cy.setToolTip("主點 Y (pixels)")
        self.intrinsics_cy.valueChanged.connect(self._on_intrinsics_changed)
        layout.addWidget(self.intrinsics_cy)

        apply_btn = QPushButton("修改內參")
        apply_btn.setToolTip("點擊後確認並套用目前欄位值，並儲存設定")
        apply_btn.clicked.connect(self._on_intrinsics_apply_clicked)
        layout.addWidget(apply_btn)

        group.setLayout(layout)
        return group

    # Removed unused _create_parameters_group (duplicate of model settings)

    # Removed unused _create_visualization_group and 3D visualizer toggle

    def _create_results_group(self) -> QGroupBox:
        """Create detection results table group."""
        group = QGroupBox("檢測結果 (最新)")
        layout = QVBoxLayout()

        # Table
        # Columns: ID, Confidence, PosX/PosY/PosZ in meters, Roll/Pitch/Yaw in radians
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            "ID", "Conf", "PosX (m)", "PosY (m)", "PosZ (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"
        ])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(self.results_table.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(self.results_table.SelectionMode.SingleSelection)
        layout.addWidget(self.results_table)

        group.setLayout(layout)
        return group

    def _create_save_group(self) -> QGroupBox:
        """Create save options group.

        Returns:
            Save group box
        """
        group = QGroupBox("💾 儲存選項")
        layout = QVBoxLayout()

        # Auto-save checkbox
        self.auto_save_json_checkbox = QCheckBox("自動儲存 JSON (用於機器手臂)")
        self.auto_save_json_checkbox.setChecked(True)  # Default: enabled
        layout.addWidget(self.auto_save_json_checkbox)

        # Add separator
        layout.addSpacing(10)

        self.save_json_btn = QPushButton("手動儲存 JSON")
        self.save_json_btn.clicked.connect(self._on_save_json)
        self.save_json_btn.setEnabled(False)
        layout.addWidget(self.save_json_btn)

        self.save_txt_btn = QPushButton("儲存 TXT")
        self.save_txt_btn.clicked.connect(self._on_save_txt)
        self.save_txt_btn.setEnabled(False)
        layout.addWidget(self.save_txt_btn)

        self.save_image_btn = QPushButton("儲存圖片")
        self.save_image_btn.clicked.connect(self._on_save_image)
        self.save_image_btn.setEnabled(False)
        layout.addWidget(self.save_image_btn)

        group.setLayout(layout)
        return group

    def _create_status_group(self) -> QGroupBox:
        """Create status display group.

        Returns:
            Status group box
        """
        group = QGroupBox("📊 狀態")
        layout = QVBoxLayout()

        # FPS label
        self.fps_label = QLabel("FPS: --")
        layout.addWidget(self.fps_label)

        # Detection count label
        self.count_label = QLabel("檢測數: 0")
        layout.addWidget(self.count_label)

        # Processing time label
        self.time_label = QLabel("處理時間: -- ms")
        layout.addWidget(self.time_label)

        # Status message label
        self.status_label = QLabel("就緒")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        layout.addWidget(self.status_label)

        # Remove per-detection summary from status per request

        group.setLayout(layout)
        return group

    def _create_output_console(self) -> OutputConsole:
        """Create output console for logging detection results.

        Returns:
            OutputConsole widget
        """
        console = OutputConsole()
        console.setMinimumHeight(150)
        console.log_info("Detection Tab initialized. Ready to start.")
        return console

    # Removed unused _create_separator helper

    def _create_action_button(self) -> QWidget:
        """Create emphasized action button section.

        Returns:
            Action button container widget
        """
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Main execution button
        self.run_detection_btn = QPushButton("🚀 開始檢測")
        self.run_detection_btn.setMinimumHeight(50)
        self.run_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 2px solid #005a9e;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1084d8;
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #808080;
                border-color: #2d2d30;
            }
        """)
        self.run_detection_btn.clicked.connect(self._on_detect)
        layout.addWidget(self.run_detection_btn)

        # Single toggle button: text changes between Start/Stop

        container.setLayout(layout)
        return container

    def _on_stop_detection(self):
        """Stop detection worker if running (used by toggle)."""
        try:
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker.wait(1500)
                self.detection_worker = None
            self.is_detecting = False
            self.run_detection_btn.setEnabled(True)
            self.run_detection_btn.setText("🚀 開始檢測")
            self.fps_label.setText("FPS: --")
            self.status_label.setText("檢測已停止")
            self.output_console.log_warning("Detection stopped by user.")
        except Exception as e:
            self.output_console.log_error(f"Error stopping detection: {e}")

    def _connect_signals(self):
        """Connect UI signals to slots."""
        # Mode change handlers
        self.input_mode_group.buttonClicked.connect(self._on_input_mode_changed)
        self.pose_mode_group.buttonClicked.connect(self._on_pose_mode_changed)

    def _load_intrinsics_from_config(self) -> Dict:
        """Load custom intrinsics from config file.

        Returns:
            Dict with camera intrinsics, or default values
        """
        try:
            import json
            from pathlib import Path

            # Get config directory
            if hasattr(self.config, 'output_dir') and self.config.output_dir:
                config_file = Path(self.config.output_dir) / "camera_intrinsics.json"
            else:
                # Fallback to current directory
                config_file = Path.cwd() / "camera_intrinsics.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    intrinsics = json.load(f)
                    return intrinsics
        except Exception as e:
            print(f"Failed to load intrinsics config: {e}")

        # Default: User's test environment values (1280x720)
        return {
            'width': 1280,
            'height': 720,
            'fx': 924.07073975,
            'fy': 921.46142578,
            'cx': 643.87634277,
            'cy': 346.78930664
        }

    def _save_intrinsics_to_config(self, intrinsics: Dict):
        """Save custom intrinsics to config file.

        Args:
            intrinsics: Camera intrinsics dict to save
        """
        try:
            import json
            from pathlib import Path

            # Get config directory
            if hasattr(self.config, 'output_dir') and self.config.output_dir:
                config_file = Path(self.config.output_dir) / "camera_intrinsics.json"
            else:
                # Fallback to current directory
                config_file = Path.cwd() / "camera_intrinsics.json"

            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(intrinsics, f, indent=2)

        except Exception as e:
            print(f"Failed to save intrinsics config: {e}")

    def _load_model_path_from_config(self) -> Optional[str]:
        """Load model path from config file.

        Returns:
            Model path string, or None if not found
        """
        try:
            import json
            from pathlib import Path

            # Get config directory
            if hasattr(self.config, 'output_dir') and self.config.output_dir:
                config_file = Path(self.config.output_dir) / "model_config.json"
            else:
                # Fallback to current directory
                config_file = Path.cwd() / "model_config.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    model_path = config_data.get('model_path')
                    if model_path and Path(model_path).exists():
                        return model_path
        except Exception as e:
            print(f"Failed to load model config: {e}")

        return None

    def _save_model_path_to_config(self, model_path: str):
        """Save model path to config file.

        Args:
            model_path: Model file path to save
        """
        try:
            import json
            from pathlib import Path

            # Get config directory
            if hasattr(self.config, 'output_dir') and self.config.output_dir:
                config_file = Path(self.config.output_dir) / "model_config.json"
            else:
                # Fallback to current directory
                config_file = Path.cwd() / "model_config.json"

            config_file.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'model_path': str(model_path),
                'last_updated': datetime.now().isoformat()
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            print(f"Saved model path to config: {model_path}")

        except Exception as e:
            print(f"Failed to save model config: {e}")

    def _initialize_service(self):
        """Initialize detection service."""
        try:
            pose_mode = "simple" if self.simple_mode_radio.isChecked() else "full"

            self.detection_service = DetectionService(
                model_path=self.model_path,
                pose_mode=pose_mode,
                conf_threshold=0.15,
                iou_threshold=0.3,
                device="cuda" if self.config.device == "cuda" else "cpu"
            )

            self.status_label.setText("✅ 檢測服務已就緒")
            self.status_label.setStyleSheet("QLabel { color: green; }")

            # Update model status label
            if self.model_path and Path(self.model_path).exists():
                model_name = Path(self.model_path).name
                self.model_status_label.setText(f"模型: {model_name}")
                self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: green; }")
            else:
                self.model_status_label.setText("模型: 未設定")
                self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: orange; }")

            # Sync intrinsics UI with current values
            self._reset_intrinsics_ui_to_custom()

        except Exception as e:
            QMessageBox.critical(
                self,
                "初始化失敗",
                f"無法初始化檢測服務:\n{e}"
            )
            self.status_label.setText(f"❌ 初始化失敗: {e}")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            self.model_status_label.setText("模型: 載入失敗")
            self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: red; }")

    def _update_intrinsics_status(self):
        """Sync UI to custom intrinsics (no separate status label)."""
        self._reset_intrinsics_ui_to_custom()

    def _update_intrinsics_status_camera_mode(self):
        """No-op status updater (UI already shows camera intrinsics when set)."""
        return

    def _update_intrinsics_ui_from_camera(self, intrinsics: Dict):
        """Update intrinsics UI fields from camera factory calibration (read-only display)."""
        # Block signals to prevent triggering save
        self.intrinsics_width.blockSignals(True)
        self.intrinsics_height.blockSignals(True)
        self.intrinsics_fx.blockSignals(True)
        self.intrinsics_fy.blockSignals(True)
        self.intrinsics_cx.blockSignals(True)
        self.intrinsics_cy.blockSignals(True)

        self.intrinsics_width.setValue(intrinsics['width'])
        self.intrinsics_height.setValue(intrinsics['height'])
        self.intrinsics_fx.setValue(intrinsics['fx'])
        self.intrinsics_fy.setValue(intrinsics['fy'])
        self.intrinsics_cx.setValue(intrinsics['cx'])
        self.intrinsics_cy.setValue(intrinsics['cy'])

        # Unblock signals
        self.intrinsics_width.blockSignals(False)
        self.intrinsics_height.blockSignals(False)
        self.intrinsics_fx.blockSignals(False)
        self.intrinsics_fy.blockSignals(False)
        self.intrinsics_cx.blockSignals(False)
        self.intrinsics_cy.blockSignals(False)

        # Status handled via UI values only

    def _reset_intrinsics_ui_to_custom(self):
        """Reset intrinsics UI to custom settings (after camera stop)."""
        self._update_intrinsics_ui_from_camera(self.custom_intrinsics)

    @Slot()
    def _on_browse_image(self):
        """Handle browse image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            str(self.config.image_dir) if self.config.image_dir else "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )

        if file_path:
            self.current_image_path = Path(file_path)
            self.image_path_edit.setText(str(self.current_image_path))

            # Log image selection
            self.output_console.log_info(f"Image loaded: {self.current_image_path.name}")

            # Load and display image
            self._load_and_display_image(self.current_image_path)

    @Slot()
    def _on_start_camera(self):
        """Handle start camera button click."""
        try:
            # Log camera connection attempt
            self.output_console.log_info("Connecting to RealSense camera...")

            # Import RealSense service
            from ..services.realsense_service import RealSenseService

            self.status_label.setText("正在啟動相機...")

            # Initialize camera service if not exists
            if self.camera_adapter is None:
                self.camera_adapter = RealSenseService()
                # Start the camera now
                self.camera_adapter.start()
            elif not self.camera_adapter.is_started:
                # Resume existing camera adapter if stopped
                self.camera_adapter.start()

            # Read and store factory calibration from RealSense
            try:
                rs_intrinsics = self.camera_adapter.get_camera_intrinsics()
                self.camera_intrinsics = rs_intrinsics
                self.output_console.log_result(
                    f"廠牌校正內參已載入: {rs_intrinsics['width']}x{rs_intrinsics['height']}, "
                    f"fx={rs_intrinsics['fx']:.2f}, fy={rs_intrinsics['fy']:.2f}"
                )
                # Update UI to show factory intrinsics (read-only view)
                self._update_intrinsics_ui_from_camera(rs_intrinsics)
            except Exception as e:
                self.output_console.log_warning(f"無法讀取 RealSense 廠商內參: {e}")
                self.camera_intrinsics = self.custom_intrinsics

            # Update UI
            self.is_camera_active = True
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.browse_btn.setEnabled(False)
            self.camera_mode_radio.setChecked(True)

            # Log successful connection
            self.output_console.log_result("RealSense camera connected successfully. Ready to start detection.")

            # Update intrinsics display for camera mode
            self._update_intrinsics_status_camera_mode()

            # Note: Detection must be started separately via the "開始檢測" button

        except Exception as e:
            # Log error
            self.output_console.log_error(f"Failed to connect camera: {str(e)}")

            QMessageBox.critical(
                self,
                "相機啟動失敗",
                f"無法啟動 RealSense 相機:\n{e}"
            )
            self.status_label.setText(f"❌ 相機啟動失敗: {e}")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            self.is_camera_active = False

    @Slot()
    def _on_stop_camera(self):
        """Handle stop camera button click."""
        try:
            # Log camera stop
            self.output_console.log_info("Stopping camera...")

            # Stop worker first
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker.wait()
                self.detection_worker = None

            # Stop camera safely
            if self.camera_adapter and self.camera_adapter.is_started:
                self.camera_adapter.stop()

            # Update UI
            self.is_camera_active = False
            self.camera_intrinsics = None  # Clear factory calibration
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.browse_btn.setEnabled(True)

            self.status_label.setText("相機已停止")
            self.fps_label.setText("FPS: --")

            # Reset intrinsics UI and display to custom settings
            self._reset_intrinsics_ui_to_custom()

            # Log successful stop
            self.output_console.log_result("Camera stopped.")

        except Exception as e:
            # Log error
            self.output_console.log_error(f"Error stopping camera: {str(e)}")

            QMessageBox.warning(
                self,
                "停止相機",
                f"停止相機時發生錯誤:\n{e}"
            )
            # Ensure UI is updated even if stop failed
            self.is_camera_active = False
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)

    def shutdown(self):
        """Gracefully stop detection worker and camera before closing tab."""
        try:
            # Stop detection worker
            if self.detection_worker:
                try:
                    self.detection_worker.stop()
                    # give it a short time to finish
                    self.detection_worker.wait(1500)
                except Exception:
                    pass
                finally:
                    self.detection_worker = None

            # Stop camera if active
            if self.camera_adapter:
                try:
                    if hasattr(self.camera_adapter, 'is_started'):
                        if self.camera_adapter.is_started and hasattr(self.camera_adapter, 'stop'):
                            self.camera_adapter.stop()
                    elif hasattr(self.camera_adapter, 'stop'):
                        self.camera_adapter.stop()
                except Exception:
                    pass

            self.is_camera_active = False
        except Exception:
            # Swallow all exceptions during shutdown to ensure smooth exit
            pass

    @Slot(int)
    def _on_confidence_changed(self, value: int):
        """Handle confidence threshold slider change.

        Args:
            value: Slider value (0-100)
        """
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"{self.confidence_threshold:.2f}")

    @Slot(int)
    def _on_iou_changed(self, value: int):
        """Handle IOU threshold slider change.

        Args:
            value: Slider value (0-100)
        """
        self.iou_threshold = value / 100.0
        self.iou_label.setText(f"{self.iou_threshold:.2f}")

    @Slot()
    def _on_intrinsics_changed(self):
        """Handle camera intrinsics inline edit change."""
        # Update custom_intrinsics from UI fields
        self.custom_intrinsics = {
            'width': self.intrinsics_width.value(),
            'height': self.intrinsics_height.value(),
            'fx': self.intrinsics_fx.value(),
            'fy': self.intrinsics_fy.value(),
            'cx': self.intrinsics_cx.value(),
            'cy': self.intrinsics_cy.value()
        }

    @Slot()
    def _on_intrinsics_apply_clicked(self):
        """Confirm and apply intrinsics edits (saves to config)."""
        reply = QMessageBox.question(
            self,
            "套用內參",
            "確定要套用目前的內參設定並儲存嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._save_intrinsics_to_config(self.custom_intrinsics)
            QMessageBox.information(self, "內參已更新", "已套用並儲存內參設定。")

    @Slot()
    def _on_detect(self):
        """Handle detect button click (single toggle Start/Stop)."""
        # If currently detecting, stop worker
        if self.is_detecting:
            try:
                if self.detection_worker:
                    self.output_console.log_info("Stopping detection...")
                    self.detection_worker.stop()
                    self.detection_worker.wait(1500)
                    self.detection_worker = None
                self.is_detecting = False
                self.run_detection_btn.setText("🚀 開始檢測")
                self.status_label.setText("檢測已停止")
                self.fps_label.setText("FPS: --")
            except Exception as e:
                self.output_console.log_error(f"Error stopping detection: {e}")
            return

        # Not detecting: start detection based on mode
        if self.camera_mode_radio.isChecked():
            self.output_console.log_info("Starting camera detection...")
            if not self.is_camera_active:
                error_msg = "請先啟動相機 (按下 '啟動相機')"
                self.output_console.log_error(error_msg)
                QMessageBox.warning(self, "相機未啟動", error_msg)
                return
            self._start_camera_detection()
        else:
            # Image mode - single detection
            if not self.current_image_path:
                error_msg = "請先選擇要檢測的圖片"
                self.output_console.log_error(error_msg)
                QMessageBox.warning(self, "無圖片", error_msg)
                return

            self.output_console.log_info(f"Starting detection for image: {self.current_image_path.name}")
            self._detect_static_image()

    @Slot(int)
    def _on_input_mode_changed(self, button_id: int):
        """Handle input mode change.

        Args:
            button_id: Button ID (0=image, 1=camera)
        """
        if button_id == 0:  # Image mode
            if self.is_camera_active:
                self._on_stop_camera()
        # Camera mode change handled by start button

    @Slot(int)
    def _on_pose_mode_changed(self, button):
        """Handle pose mode change.

        Args:
            button: QAbstractButton that was clicked
        """
        button_id = self.pose_mode_group.id(button)
        if self.detection_service:
            new_mode = "simple" if button_id == 0 else "full"
            self.detection_service.set_pose_mode(new_mode)
            self.status_label.setText(f"姿態模式: {new_mode}")
            print(f"✅ Pose mode changed to: {new_mode}")

    @Slot()
    def _on_model_settings(self):
        """Open model path selection dialog and reload model."""
        # Browse for model file
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 YOLO OBB 模型檔案",
            str(Path(self.model_path).parent) if self.model_path else "",
            "PyTorch Models (*.pt *.pth);;All Files (*.*)"
        )

        if not model_path:
            return

        # Validate file exists
        if not Path(model_path).exists():
            QMessageBox.warning(
                self,
                "檔案不存在",
                f"選擇的模型檔案不存在:\n{model_path}"
            )
            return

        # Check if detection is running
        if self.is_camera_active or (self.detection_worker and self.detection_worker.isRunning()):
            QMessageBox.warning(
                self,
                "無法切換模型",
                "請先停止檢測再切換模型"
            )
            return

        # Update model path
        self.model_path = model_path
        self._save_model_path_to_config(model_path)

        # Reinitialize detection service with new model
        try:
            self._initialize_service()

            # Update status label
            model_name = Path(model_path).name
            self.model_status_label.setText(f"模型: {model_name}")
            self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: green; }")

            QMessageBox.information(
                self,
                "模型已載入",
                f"成功載入模型:\n{model_name}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "載入失敗",
                f"無法載入模型:\n{e}"
            )
            self.model_status_label.setText("模型: 載入失敗")
            self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: red; }")


    def _auto_save_json(self, result: Dict):
        """Auto-save detection results as JSON to output directory.

        Args:
            result: Detection result dictionary
        """
        try:
            # Check if output_dir is configured
            if not hasattr(self.config, 'output_dir') or self.config.output_dir is None:
                # Use current directory as fallback
                output_dir = Path.cwd() / "detections" / "json"
            else:
                output_dir = Path(self.config.output_dir) / "detections" / "json"

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"detection_{timestamp}.json"

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Full save path
            save_path = output_dir / filename

            # Save JSON
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update status
            self.status_label.setText(f"✅ 自動儲存: {filename}")

            # Log to console
            self.output_console.log_result(f"Auto-saved JSON: {filename}")
            print(f"Auto-saved JSON to: {save_path}")

        except Exception as e:
            error_msg = f"自動儲存 JSON 失敗: {e}"
            self.status_label.setText(f"❌ {error_msg}")

            # Log error to console
            self.output_console.log_error(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()

    @Slot()
    def _on_save_json(self):
        """Save detection results as JSON (manual save with file dialog)."""
        if not self.current_result:
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detection_{timestamp}.json"

        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 JSON",
            str(self.config.output_dir / "detections" / "json" / filename),
            "JSON Files (*.json)"
        )

        if save_path:
            try:
                # Ensure directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                # Save JSON
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, indent=2, ensure_ascii=False)

                self.status_label.setText(f"✅ 已儲存: {Path(save_path).name}")
                QMessageBox.information(
                    self,
                    "儲存成功",
                    f"檢測結果已儲存至:\n{save_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "儲存失敗",
                    f"儲存 JSON 時發生錯誤:\n{e}"
                )

    @Slot()
    def _on_save_txt(self):
        """Save detection results as TXT."""
        if not self.current_result:
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detection_{timestamp}.txt"

        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 TXT",
            str(self.config.output_dir / "detections" / "txt" / filename),
            "Text Files (*.txt)"
        )

        if save_path:
            try:
                # Ensure directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                # Format TXT content
                lines = self._format_txt_output(self.current_result)

                # Save TXT
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                self.status_label.setText(f"✅ 已儲存: {Path(save_path).name}")
                QMessageBox.information(
                    self,
                    "儲存成功",
                    f"檢測結果已儲存至:\n{save_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "儲存失敗",
                    f"儲存 TXT 時發生錯誤:\n{e}"
                )

    @Slot()
    def _on_save_image(self):
        """Save annotated image."""
        if self.current_annotated is None:
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detection_{timestamp}.jpg"

        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存圖片",
            str(self.config.output_dir / "detections" / "images" / filename),
            "JPEG (*.jpg);;PNG (*.png)"
        )

        if save_path:
            try:
                # Ensure directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(self.current_annotated, cv2.COLOR_RGB2BGR)

                # Save image
                cv2.imwrite(str(save_path), image_bgr)

                self.status_label.setText(f"✅ 已儲存: {Path(save_path).name}")
                QMessageBox.information(
                    self,
                    "儲存成功",
                    f"檢測圖片已儲存至:\n{save_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "儲存失敗",
                    f"儲存圖片時發生錯誤:\n{e}"
                )

    def _load_and_display_image(self, image_path: Path):
        """Load and display an image.

        Args:
            image_path: Path to image file
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display on canvas
            self.canvas.set_image_from_numpy(image_rgb)

            self.status_label.setText(f"已載入: {image_path.name}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "載入失敗",
                f"無法載入圖片:\n{e}"
            )

    def _detect_static_image(self):
        """Run detection on current static image."""
        try:
            # Toggle to detecting state
            self.is_detecting = True
            self.run_detection_btn.setEnabled(True)
            self.run_detection_btn.setText("⏹ 停止")
            self.status_label.setText("正在檢測...")

            # Start worker
            self.detection_worker = DetectionWorker(
                detection_service=self.detection_service,
                mode='image',
                image_path=self.current_image_path,
                custom_intrinsics=self.custom_intrinsics,
                show_obb=True
            )

            # Connect signals
            self.detection_worker.detection_completed.connect(self._on_detection_completed)
            self.detection_worker.performance_updated.connect(self._on_performance_updated)
            self.detection_worker.detection_failed.connect(self._on_detection_failed)
            self.detection_worker.status_message.connect(self._on_status_message)

            # Start worker
            self.detection_worker.start()

        except Exception as e:
            self.is_detecting = False
            self.run_detection_btn.setEnabled(True)
            self.run_detection_btn.setText("🚀 開始檢測")
            QMessageBox.critical(
                self,
                "檢測失敗",
                f"啟動檢測時發生錯誤:\n{e}"
            )

    def _start_camera_detection(self):
        """Start camera detection worker."""
        try:
            # Use factory calibration if available, otherwise fall back to custom
            intrinsics_to_use = self.camera_intrinsics if self.camera_intrinsics else self.custom_intrinsics

            # Start worker
            self.detection_worker = DetectionWorker(
                detection_service=self.detection_service,
                mode='camera',
                camera_adapter=self.camera_adapter,
                target_fps=30.0,
                custom_intrinsics=intrinsics_to_use,
                show_obb=True
            )

            # Connect signals
            self.detection_worker.detection_completed.connect(self._on_detection_completed)
            self.detection_worker.performance_updated.connect(self._on_performance_updated)
            self.detection_worker.detection_failed.connect(self._on_detection_failed)
            self.detection_worker.status_message.connect(self._on_status_message)

            # Start worker
            self.detection_worker.start()

            # Toggle to detecting state
            self.is_detecting = True
            self.run_detection_btn.setEnabled(True)
            self.run_detection_btn.setText("⏹ 停止")

        except Exception as e:
            self._on_stop_camera()
            QMessageBox.critical(
                self,
                "相機檢測失敗",
                f"啟動相機檢測時發生錯誤:\n{e}"
            )

    @Slot(dict, np.ndarray)
    def _on_detection_completed(self, result: Dict, annotated: np.ndarray):
        """Handle detection completion.

        Args:
            result: Detection result dictionary
            annotated: Annotated RGB image
        """
        # Store results
        self.current_result = result
        self.current_annotated = annotated

        # Display annotated image
        self.canvas.set_image_from_numpy(annotated)

        # Update status
        count = result['metadata']['detection_count']
        self.count_label.setText(f"檢測數: {count}")

        # Update table with latest results
        self._populate_results_table(result)

        # Log detection completion
        self.output_console.log_result(f"Detection completed. Found {count} object(s).")

        # Log individual detections
        if count > 0 and 'detections' in result:
            for idx, det in enumerate(result['detections'], 1):
                class_name = det.get('class_name', 'Unknown')
                confidence = det.get('confidence', 0.0)
                self.output_console.log_result(
                    f"  [{idx}] {class_name} (Confidence: {confidence:.2%})"
                )

        # Enable save buttons if we have detections
        has_detections = count > 0
        self.save_json_btn.setEnabled(has_detections)
        self.save_txt_btn.setEnabled(has_detections)
        self.save_image_btn.setEnabled(True)  # Always enable for annotated image

        # Auto-save JSON if enabled and has detections
        if self.auto_save_json_checkbox.isChecked() and has_detections:
            self._auto_save_json(result)

        # If image mode completes, reset Start/Stop toggle
        if result['metadata'].get('source_type') == 'image' or not self.is_camera_active:
            self.is_detecting = False
            self.run_detection_btn.setEnabled(True)
            self.run_detection_btn.setText("🚀 開始檢測")

    @Slot(float, float)
    def _on_performance_updated(self, fps: float, processing_time_ms: float):
        """Handle performance update.

        Args:
            fps: Frames per second
            processing_time_ms: Processing time in milliseconds
        """
        if fps > 0:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.time_label.setText(f"處理時間: {processing_time_ms:.1f} ms")

    @Slot(str)
    def _on_detection_failed(self, error_msg: str):
        """Handle detection failure.

        Args:
            error_msg: Error message
        """
        self.status_label.setText(f"❌ 檢測失敗: {error_msg}")
        self.status_label.setStyleSheet("QLabel { color: red; }")
        self.is_detecting = False
        self.run_detection_btn.setEnabled(True)
        self.run_detection_btn.setText("🚀 開始檢測")

        # Log error to console
        self.output_console.log_error(f"Detection failed: {error_msg}")

        QMessageBox.critical(
            self,
            "檢測失敗",
            f"檢測過程發生錯誤:\n{error_msg}"
        )

    def _populate_results_table(self, result: Dict):
        """Populate the results table with the latest detections."""
        try:
            detections = result.get('detections', [])
            self.results_table.setRowCount(len(detections))

            for row, det in enumerate(detections):
                pose = det.get('pose', {})
                pos = pose.get('position', {})
                rot = pose.get('rotation_euler', {})

                det_id = str(det.get('detection_id', ''))
                conf = det.get('confidence', 0.0)

                # Positions are provided in meters; display as meters
                pos_x = pos.get('x', 0.0)
                pos_y = pos.get('y', 0.0)
                pos_z = pos.get('z', 0.0)

                # Rotation angles (already corrected in Simple mode by detection service)
                roll = rot.get('roll_rad', 0.0)
                pitch = rot.get('pitch_rad', 0.0)
                yaw = rot.get('yaw_rad', 0.0)

                values = [
                    det_id,
                    f"{conf:.2f}",
                    f"{pos_x:.3f}",
                    f"{pos_y:.3f}",
                    f"{pos_z:.3f}",
                    f"{roll:.3f}",
                    f"{pitch:.3f}",
                    f"{yaw:.3f}",
                ]

                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    self.results_table.setItem(row, col, item)

            # If no detections, clear table
            if len(detections) == 0:
                self.results_table.setRowCount(0)
        except Exception as e:
            self.output_console.log_warning(f"無法更新結果表格: {e}")

    @Slot(str)
    def _on_status_message(self, message: str):
        """Handle status message.

        Args:
            message: Status message
        """
        self.status_label.setText(message)

    # Removed unused _update_last_detection_summary

    def _format_txt_output(self, result: Dict) -> list:
        """Format detection result as TXT lines.

        Args:
            result: Detection result dictionary

        Returns:
            List of text lines
        """
        lines = []

        # Header
        lines.append("# Detection Results")
        lines.append(f"# Timestamp: {result['metadata']['timestamp']}")

        source_type = result['metadata']['source_type']
        source_path = result['metadata'].get('source_path', 'N/A')
        lines.append(f"# Source: {source_type} | {source_path}")

        lines.append(f"# Pose Mode: {result['metadata']['pose_mode']}")
        lines.append(f"# Detection Count: {result['metadata']['detection_count']}")
        lines.append("")

        # Data format
        lines.append("# Format: id conf x y z roll pitch yaw obb_cx obb_cy obb_rot")

        # Data lines
        for detection in result['detections']:
            det_id = detection['detection_id']
            conf = detection['confidence']
            pos = detection['pose']['position']
            rot = detection['pose']['rotation_euler']
            obb = detection['obb']

            line = (
                f"{det_id} {conf:.2f} "
                f"{pos['x']:.3f} {pos['y']:.3f} {pos['z']:.3f} "
                f"{rot['roll_rad']:.3f} {rot['pitch_rad']:.3f} {rot['yaw_rad']:.3f} "
                f"{obb['center'][0]:.1f} {obb['center'][1]:.1f} {obb['rotation_rad']:.3f}"
            )
            lines.append(line)

        return lines

    def cleanup(self):
        """Cleanup resources when tab is closed."""
        # Stop camera if active
        if self.is_camera_active:
            self._on_stop_camera()

        # Stop worker if running
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker.wait()
