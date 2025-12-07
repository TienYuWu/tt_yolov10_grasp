"""Camera Intrinsics Configuration Dialog

Provides GUI for configuring camera intrinsic parameters for pose estimation.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
    QLabel, QDialogButtonBox, QMessageBox
)
from PySide6.QtCore import Qt, Slot
from typing import Dict, Optional


class IntrinsicsDialog(QDialog):
    """相機內部參數設定對話框"""

    def __init__(
        self,
        parent=None,
        initial_intrinsics: Dict = None,
        realsense_service=None
    ):
        """Initialize intrinsics dialog.

        Args:
            parent: Parent widget
            initial_intrinsics: Current intrinsics dict (fx, fy, cx, cy, width, height)
            realsense_service: RealSenseService instance (optional, for auto-loading)
        """
        super().__init__(parent)
        self.setWindowTitle("相機內部參數設定")
        self.setMinimumWidth(500)

        self.realsense_service = realsense_service
        self.intrinsics = initial_intrinsics or {}

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        """Build UI layout."""
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "相機內部參數影響 3D 姿態估計的準確度。\n"
            "Camera mode 會自動使用 RealSense 原廠校正值。\n"
            "Image mode 使用此處設定的自訂值。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Intrinsics form
        form_group = QGroupBox("內部參數")
        form_layout = QFormLayout()

        # Resolution (read-only, for reference)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 4096)
        self.width_spin.setEnabled(False)  # Read-only
        form_layout.addRow("寬度 (Width):", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 4096)
        self.height_spin.setEnabled(False)  # Read-only
        form_layout.addRow("高度 (Height):", self.height_spin)

        # Focal lengths
        self.fx_spin = QDoubleSpinBox()
        self.fx_spin.setRange(1.0, 5000.0)
        self.fx_spin.setDecimals(8)
        self.fx_spin.setSingleStep(1.0)
        form_layout.addRow("焦距 fx (pixels):", self.fx_spin)

        self.fy_spin = QDoubleSpinBox()
        self.fy_spin.setRange(1.0, 5000.0)
        self.fy_spin.setDecimals(8)
        self.fy_spin.setSingleStep(1.0)
        form_layout.addRow("焦距 fy (pixels):", self.fy_spin)

        # Principal points
        self.cx_spin = QDoubleSpinBox()
        self.cx_spin.setRange(0.0, 4096.0)
        self.cx_spin.setDecimals(8)
        self.cx_spin.setSingleStep(1.0)
        form_layout.addRow("主點 cx (pixels):", self.cx_spin)

        self.cy_spin = QDoubleSpinBox()
        self.cy_spin.setRange(0.0, 4096.0)
        self.cy_spin.setDecimals(8)
        self.cy_spin.setSingleStep(1.0)
        form_layout.addRow("主點 cy (pixels):", self.cy_spin)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.load_rs_btn = QPushButton("從 RealSense 載入")
        self.load_rs_btn.clicked.connect(self._load_from_realsense)
        self.load_rs_btn.setEnabled(self.realsense_service is not None)
        btn_layout.addWidget(self.load_rs_btn)

        reset_btn = QPushButton("重設為預設值")
        reset_btn.clicked.connect(self._reset_to_defaults)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _load_values(self):
        """Load current intrinsics values into spinboxes."""
        self.width_spin.setValue(self.intrinsics.get('width', 1280))
        self.height_spin.setValue(self.intrinsics.get('height', 720))
        self.fx_spin.setValue(self.intrinsics.get('fx', 924.07073975))
        self.fy_spin.setValue(self.intrinsics.get('fy', 921.46142578))
        self.cx_spin.setValue(self.intrinsics.get('cx', 643.87634277))
        self.cy_spin.setValue(self.intrinsics.get('cy', 346.78930664))

    @Slot()
    def _load_from_realsense(self):
        """Load intrinsics from connected RealSense camera."""
        if not self.realsense_service:
            QMessageBox.warning(
                self,
                "未連接相機",
                "請先連接 RealSense 相機"
            )
            return

        try:
            # Get intrinsics from RealSense
            rs_intrinsics = self.realsense_service.get_camera_intrinsics()

            self.width_spin.setValue(rs_intrinsics['width'])
            self.height_spin.setValue(rs_intrinsics['height'])
            self.fx_spin.setValue(rs_intrinsics['fx'])
            self.fy_spin.setValue(rs_intrinsics['fy'])
            self.cx_spin.setValue(rs_intrinsics['cx'])
            self.cy_spin.setValue(rs_intrinsics['cy'])

            QMessageBox.information(
                self,
                "載入成功",
                f"已從 RealSense 相機載入內部參數\n"
                f"解析度: {rs_intrinsics['width']}x{rs_intrinsics['height']}\n"
                f"fx={rs_intrinsics['fx']:.2f}, fy={rs_intrinsics['fy']:.2f}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "載入失敗",
                f"無法從 RealSense 載入內部參數:\n{e}"
            )

    @Slot()
    def _reset_to_defaults(self):
        """Reset to user's typical values (1280x720)."""
        # User's test environment defaults
        self.width_spin.setValue(1280)
        self.height_spin.setValue(720)
        self.fx_spin.setValue(924.07073975)
        self.fy_spin.setValue(921.46142578)
        self.cx_spin.setValue(643.87634277)
        self.cy_spin.setValue(346.78930664)

    def get_intrinsics(self) -> Dict:
        """Get current intrinsics values.

        Returns:
            Dict with keys: width, height, fx, fy, cx, cy
        """
        return {
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'fx': self.fx_spin.value(),
            'fy': self.fy_spin.value(),
            'cx': self.cx_spin.value(),
            'cy': self.cy_spin.value()
        }
