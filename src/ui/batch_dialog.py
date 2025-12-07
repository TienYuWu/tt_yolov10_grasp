"""Batch processing progress dialog."""

from PySide6.QtWidgets import QProgressDialog, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt


class BatchProgressDialog(QProgressDialog):
    """Progress dialog for batch SAM processing."""

    def __init__(self, total_images, parent=None):
        """Initialize batch progress dialog.

        Args:
            total_images: Total number of images to process
            parent: Parent widget
        """
        super().__init__(
            "準備批次處理...",
            "取消",
            0,
            total_images,
            parent
        )
        self.setWindowTitle("批次標註進度")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(450)
        self.setMinimumHeight(150)

        # Create status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLabel(self.status_label)

        # Statistics
        self.success_count = 0
        self.failed_count = 0
        self.total_count = total_images

        # Update initial status
        self._update_status_text(0, "")

    def update_progress(self, current: int, total: int, image_name: str):
        """Update progress bar and status text.

        Args:
            current: Current image index (1-based)
            total: Total number of images
            image_name: Current image name
        """
        self.setValue(current)
        self._update_status_text(current, image_name)

    def on_image_processed(self, index: int, masks: list, success: bool, image_name: str):
        """Handle image processing completion.

        Args:
            index: Image index
            masks: List of generated masks
            success: Whether processing succeeded
            image_name: Image name
        """
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1

        # Update status immediately
        self._update_status_text(index + 1, image_name)

    def _update_status_text(self, current: int, image_name: str):
        """Update status label text.

        Args:
            current: Current progress count
            image_name: Current image name
        """
        if current == 0:
            status_text = "準備批次處理..."
        else:
            status_text = f"處理中：{image_name}\n"
            status_text += f"進度：{current}/{self.total_count}\n"
            status_text += f"✅ 成功：{self.success_count}  ❌ 失敗：{self.failed_count}"

        self.status_label.setText(status_text)

    def get_statistics(self):
        """Get processing statistics.

        Returns:
            Dictionary with success, failed, and total counts
        """
        return {
            'success': self.success_count,
            'failed': self.failed_count,
            'total': self.total_count
        }
