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
    QButtonGroup, QCheckBox, QSplitter, QLineEdit, QFrame
)

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

        # 3D visualization (optional feature)
        self.visualizer_3d = None  # Initialized on demand when checkbox is checked

        # Current state
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[Dict] = None
        self.current_annotated: Optional[np.ndarray] = None
        self.is_camera_active = False

        # Camera intrinsics for image mode
        self.custom_intrinsics = self._load_intrinsics_from_config()

        self._build_ui()
        self._connect_signals()
        self._initialize_service()

    def _build_ui(self):
        """Build the detection tab UI with vertical split layout.

        Layout structure:
        - Vertical Splitter (outer):
          - Top (70%): Horizontal Splitter (Canvas + Controls)
          - Bottom (30%): Output Console
        """
        main_layout = QVBoxLayout()

        # ========== OUTER VERTICAL SPLITTER ==========
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)

        # ========== TOP SECTION: Canvas + Control Panel ==========
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel: Canvas (70%)
        left_panel = self._create_canvas_panel()

        # Right panel: Controls (30%)
        right_panel = self._create_control_panel()

        # Inner horizontal splitter
        horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)
        horizontal_splitter.addWidget(left_panel)
        horizontal_splitter.addWidget(right_panel)
        horizontal_splitter.setStretchFactor(0, 7)  # 70% for canvas
        horizontal_splitter.setStretchFactor(1, 3)  # 30% for controls

        top_layout.addWidget(horizontal_splitter)
        top_widget.setLayout(top_layout)

        # ========== BOTTOM SECTION: Output Console ==========
        self.output_console = self._create_output_console()

        # Add both sections to vertical splitter
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(self.output_console)

        # Set vertical splitter proportions (70% top, 30% bottom)
        vertical_splitter.setStretchFactor(0, 7)
        vertical_splitter.setStretchFactor(1, 3)

        # Prevent full collapse of either section
        vertical_splitter.setChildrenCollapsible(False)

        main_layout.addWidget(vertical_splitter)
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
        layout = QVBoxLayout()

        # ========== GROUP 1: SETUP (設定區) ==========
        setup_label = QLabel("⚙️ SETUP")
        setup_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0078d4;")
        layout.addWidget(setup_label)

        # Input source group
        input_group = self._create_input_source_group()
        layout.addWidget(input_group)

        # Model settings group
        model_group = self._create_model_settings_group()
        layout.addWidget(model_group)

        # Pose mode group
        pose_group = self._create_pose_mode_group()
        layout.addWidget(pose_group)

        # Camera intrinsics group
        intrinsics_group = self._create_intrinsics_group()
        layout.addWidget(intrinsics_group)

        # Separator
        layout.addWidget(self._create_separator())

        # ========== GROUP 2: ACTION (動作區，強調) ==========
        action_label = QLabel("▶️ ACTION")
        action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ec9b0;")
        layout.addWidget(action_label)

        action_button_widget = self._create_action_button()
        layout.addWidget(action_button_widget)

        # Separator
        layout.addWidget(self._create_separator())

        # ========== GROUP 3: VISUALIZATION (視覺化) ==========
        viz_label = QLabel("👁️ VISUALIZATION")
        viz_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ce9178;")
        layout.addWidget(viz_label)

        vis_group = self._create_visualization_group()
        layout.addWidget(vis_group)

        # Separator
        layout.addWidget(self._create_separator())

        # ========== GROUP 4: STORAGE (儲存) ==========
        storage_label = QLabel("💾 STORAGE")
        storage_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #dcdcaa;")
        layout.addWidget(storage_label)

        save_group = self._create_save_group()
        layout.addWidget(save_group)

        # Flexible space to push content to top
        layout.addStretch()

        # ========== STATUS INDICATOR (底部) ==========
        # 保留精簡的狀態指示器在控制面板底部
        status_group = self._create_status_group()
        layout.addWidget(status_group)

        panel.setLayout(layout)
        return panel

    def _create_input_source_group(self) -> QGroupBox:
        """Create input source selection group.

        Returns:
            Input source group box
        """
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
        """Create model path settings group.

        Returns:
            Model settings group box
        """
        group = QGroupBox("🤖 模型設定")
        layout = QVBoxLayout()

        # Settings button
        model_btn = QPushButton("模型路徑設定")
        model_btn.clicked.connect(self._on_model_settings)
        model_btn.setToolTip("設定 YOLO OBB 模型檔案路徑")
        layout.addWidget(model_btn)

        # Status label
        self.model_status_label = QLabel("模型: 未載入")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("QLabel { font-size: 10px; color: gray; }")
        layout.addWidget(self.model_status_label)

        group.setLayout(layout)
        return group

    def _create_intrinsics_group(self) -> QGroupBox:
        """Create camera intrinsics settings group.

        Returns:
            Camera intrinsics group box
        """
        group = QGroupBox("相機內部參數")
        layout = QVBoxLayout()

        # Settings button
        intrinsics_btn = QPushButton("相機內參設定")
        intrinsics_btn.clicked.connect(self._on_intrinsics_settings)
        intrinsics_btn.setToolTip("設定 Image mode 使用的相機內部參數")
        layout.addWidget(intrinsics_btn)

        # Status label
        self.intrinsics_status_label = QLabel("內參: 使用預設值")
        self.intrinsics_status_label.setWordWrap(True)
        self.intrinsics_status_label.setStyleSheet("QLabel { font-size: 10px; color: gray; }")
        layout.addWidget(self.intrinsics_status_label)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group.

        Returns:
            Visualization group box
        """
        group = QGroupBox("👁️ 可視化選項")
        layout = QVBoxLayout()

        self.show_obb_check = QCheckBox("顯示 OBB 框")
        self.show_obb_check.setChecked(True)
        layout.addWidget(self.show_obb_check)

        self.show_axes_check = QCheckBox("顯示座標軸")
        self.show_axes_check.setChecked(True)
        layout.addWidget(self.show_axes_check)

        self.show_pose_text_check = QCheckBox("顯示姿態文字")
        self.show_pose_text_check.setChecked(True)
        layout.addWidget(self.show_pose_text_check)

        self.show_depth_check = QCheckBox("顯示深度圖")
        self.show_depth_check.setChecked(False)
        layout.addWidget(self.show_depth_check)

        self.show_3d_vis_check = QCheckBox("顯示 3D 姿態視窗")
        self.show_3d_vis_check.setChecked(False)
        self.show_3d_vis_check.setToolTip("開啟 Open3D 3D 可視化視窗 (顯示點雲、PCA 平面、法向量)")
        self.show_3d_vis_check.stateChanged.connect(self._on_3d_vis_toggled)
        layout.addWidget(self.show_3d_vis_check)

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

    def _create_separator(self) -> QWidget:
        """Create a horizontal separator line.

        Returns:
            Separator widget
        """
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #3e3e42;")
        return line

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

        # Optional: Stop button
        self.stop_detection_btn = QPushButton("⏹ 停止")
        self.stop_detection_btn.setMinimumHeight(35)
        self.stop_detection_btn.setEnabled(False)
        self.stop_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #4d4d4d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #505050;
            }
            QPushButton:pressed {
                background-color: #3e3e42;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #808080;
                border-color: #2d2d30;
            }
        """)
        self.stop_detection_btn.clicked.connect(self._on_stop_detection)
        layout.addWidget(self.stop_detection_btn)

        container.setLayout(layout)
        return container

    def _on_stop_detection(self):
        """Handle stop detection button click."""
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.output_console.log_warning("Detection stopped by user.")

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

            # Initialize camera service
            if self.camera_adapter is None:
                self.camera_adapter = RealSenseService()

            # Update UI
            self.is_camera_active = True
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.browse_btn.setEnabled(False)
            self.camera_mode_radio.setChecked(True)

            # Log successful connection
            self.output_console.log_result("RealSense camera connected successfully.")

            # Start detection worker in camera mode
            self._start_camera_detection()

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

    @Slot()
    def _on_stop_camera(self):
        """Handle stop camera button click."""
        try:
            # Log camera stop
            self.output_console.log_info("Stopping camera...")

            # Stop worker
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker.wait()
                self.detection_worker = None

            # Stop camera
            if self.camera_adapter:
                if hasattr(self.camera_adapter, 'stop'):
                    self.camera_adapter.stop()

            # Update UI
            self.is_camera_active = False
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.browse_btn.setEnabled(True)

            self.status_label.setText("相機已停止")
            self.fps_label.setText("FPS: --")

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

    @Slot()
    def _on_detect(self):
        """Handle detect button click."""
        if self.camera_mode_radio.isChecked():
            # Camera mode - start continuous detection
            self.output_console.log_info("Starting camera detection...")
            if not self.is_camera_active:
                self._on_start_camera()
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
    def _on_pose_mode_changed(self, button_id: int):
        """Handle pose mode change.

        Args:
            button_id: Button ID (0=simple, 1=full)
        """
        if self.detection_service:
            new_mode = "simple" if button_id == 0 else "full"
            self.detection_service.set_pose_mode(new_mode)
            self.status_label.setText(f"姿態模式: {new_mode}")

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

    def _on_intrinsics_settings(self):
        """Open intrinsics settings dialog."""
        from .intrinsics_dialog import IntrinsicsDialog

        # Pass RealSense service if available (camera mode)
        realsense = None
        if self.camera_mode_radio.isChecked() and hasattr(self, 'realsense_service'):
            realsense = self.realsense_service

        dialog = IntrinsicsDialog(
            parent=self,
            initial_intrinsics=self.custom_intrinsics,
            realsense_service=realsense
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_intrinsics = dialog.get_intrinsics()
            self._save_intrinsics_to_config(self.custom_intrinsics)

            # Update status label
            self.intrinsics_status_label.setText(
                f"內參: {self.custom_intrinsics['width']}x{self.custom_intrinsics['height']}, "
                f"fx={self.custom_intrinsics['fx']:.1f}"
            )

            QMessageBox.information(
                self,
                "設定已儲存",
                "相機內部參數已更新並儲存"
            )

    @Slot(int)
    def _on_3d_vis_toggled(self, state: int):
        """Handle 3D visualization toggle.

        Args:
            state: Checkbox state (0=unchecked, 2=checked)
        """
        if state == Qt.CheckState.Checked.value:
            # Initialize 3D visualizer on demand
            if self.visualizer_3d is None:
                try:
                    from .visualizer_3d_widget import Visualizer3DWidget
                    self.visualizer_3d = Visualizer3DWidget(
                        window_name="3D Pose Visualization",
                        width=1280,
                        height=720
                    )
                    self.status_label.setText("✅ 3D 可視化已啟動")
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "3D 可視化啟動失敗",
                        f"無法啟動 Open3D 可視化:\n{e}\n\n"
                        "請確認已安裝 open3d 套件"
                    )
                    self.show_3d_vis_check.setChecked(False)
                    self.status_label.setText(f"❌ 3D 可視化失敗: {e}")
        else:
            # Close 3D visualizer
            if self.visualizer_3d:
                try:
                    self.visualizer_3d.close()
                    self.visualizer_3d = None
                    self.status_label.setText("3D 可視化已關閉")
                except Exception as e:
                    print(f"Warning: Error closing 3D visualizer: {e}")

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
            self.run_detection_btn.setEnabled(False)
            self.status_label.setText("正在檢測...")

            # Start worker
            self.detection_worker = DetectionWorker(
                detection_service=self.detection_service,
                mode='image',
                image_path=self.current_image_path,
                custom_intrinsics=self.custom_intrinsics
            )

            # Connect signals
            self.detection_worker.detection_completed.connect(self._on_detection_completed)
            self.detection_worker.performance_updated.connect(self._on_performance_updated)
            self.detection_worker.detection_failed.connect(self._on_detection_failed)
            self.detection_worker.status_message.connect(self._on_status_message)

            # Start worker
            self.detection_worker.start()

        except Exception as e:
            self.run_detection_btn.setEnabled(True)
            QMessageBox.critical(
                self,
                "檢測失敗",
                f"啟動檢測時發生錯誤:\n{e}"
            )

    def _start_camera_detection(self):
        """Start camera detection worker."""
        try:
            # Start worker
            self.detection_worker = DetectionWorker(
                detection_service=self.detection_service,
                mode='camera',
                camera_adapter=self.camera_adapter,
                target_fps=30.0
            )

            # Connect signals
            self.detection_worker.detection_completed.connect(self._on_detection_completed)
            self.detection_worker.performance_updated.connect(self._on_performance_updated)
            self.detection_worker.detection_failed.connect(self._on_detection_failed)
            self.detection_worker.status_message.connect(self._on_status_message)

            # Start worker
            self.detection_worker.start()

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

        # Re-enable detect button for image mode
        if not self.is_camera_active:
            self.run_detection_btn.setEnabled(True)

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
        self.run_detection_btn.setEnabled(True)

        # Log error to console
        self.output_console.log_error(f"Detection failed: {error_msg}")

        QMessageBox.critical(
            self,
            "檢測失敗",
            f"檢測過程發生錯誤:\n{error_msg}"
        )

    @Slot(str)
    def _on_status_message(self, message: str):
        """Handle status message.

        Args:
            message: Status message
        """
        self.status_label.setText(message)

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
