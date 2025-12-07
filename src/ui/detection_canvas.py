"""Detection Canvas - Display widget for detection results

Simple canvas for displaying annotated detection results with zoom/pan.
"""

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from ..utils import numpy_to_qpixmap


class DetectionCanvas(QGraphicsView):
    """Canvas for displaying detection results with zoom/pan controls."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize detection canvas.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Background
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

        # Image layer
        self._image_item = QGraphicsPixmapItem()
        self.scene().addItem(self._image_item)

        # Zoom state
        self._zoom = 0
        self._zoom_factor = 1.15
        self._min_zoom = -10
        self._max_zoom = 10

    def clear(self):
        """Clear the canvas."""
        self._image_item.setPixmap(QPixmap())
        self.scene().setSceneRect(0, 0, 0, 0)
        self._zoom = 0
        self.resetTransform()

    def set_image(self, pixmap: QPixmap):
        """Set the image to display.

        Args:
            pixmap: QPixmap image to display
        """
        self._image_item.setPixmap(pixmap)
        self.scene().setSceneRect(pixmap.rect())
        self.fit_in_view()

    def set_image_from_numpy(self, image: np.ndarray):
        """Set image from numpy array.

        Args:
            image: RGB image as numpy array (H x W x 3 uint8).
                   Must be in RGB format, NOT BGR.
                   If you have a BGR image from cv2.imread(), convert it first using:
                   rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        """
        pixmap = numpy_to_qpixmap(image)
        self.set_image(pixmap)

    def fit_in_view(self):
        """Fit the image in view."""
        if not self._image_item.pixmap().isNull():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 0

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom.

        Args:
            event: Wheel event
        """
        if not self._image_item.pixmap().isNull():
            # Get wheel delta
            delta = event.angleDelta().y()

            if delta > 0:
                # Zoom in
                if self._zoom < self._max_zoom:
                    self.scale(self._zoom_factor, self._zoom_factor)
                    self._zoom += 1
            else:
                # Zoom out
                if self._zoom > self._min_zoom:
                    self.scale(1.0 / self._zoom_factor, 1.0 / self._zoom_factor)
                    self._zoom -= 1

            event.accept()
        else:
            event.ignore()

    def zoom_in(self):
        """Zoom in programmatically."""
        if self._zoom < self._max_zoom:
            self.scale(self._zoom_factor, self._zoom_factor)
            self._zoom += 1

    def zoom_out(self):
        """Zoom out programmatically."""
        if self._zoom > self._min_zoom:
            self.scale(1.0 / self._zoom_factor, 1.0 / self._zoom_factor)
            self._zoom -= 1

    def reset_zoom(self):
        """Reset zoom to fit view."""
        self.fit_in_view()

    def get_current_pixmap(self) -> QPixmap:
        """Get the current displayed pixmap.

        Returns:
            Current QPixmap
        """
        return self._image_item.pixmap()
