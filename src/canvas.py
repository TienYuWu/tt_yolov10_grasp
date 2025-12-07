"""Interactive canvas widget for Smart Label application."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QWheelEvent, QMouseEvent, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from .utils import numpy_to_qpixmap


class ImageCanvas(QGraphicsView):
    """Interactive canvas for displaying images, masks, and annotation points."""

    # Signals
    point_added = Signal(float, float, int)  # x, y, label (1=positive, 0=negative)
    request_sam_prediction = Signal()  # Request SAM prediction

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the canvas.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Image layers
        self._image_item = QGraphicsPixmapItem()
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setZValue(1)

        self.scene().addItem(self._image_item)
        self.scene().addItem(self._mask_item)

        # Annotation data
        self._point_items: List[Tuple[QGraphicsEllipseItem, int]] = []
        self._points: List[Tuple[float, float]] = []  # Scene coordinates
        self._labels: List[int] = []  # 1 for positive, 0 for negative

        # Multiple masks support
        self._mask_overlays: Dict[str, QGraphicsPixmapItem] = {}  # mask_id -> item
        self._bbox_overlays: Dict[str, QGraphicsRectItem] = {}
        self._obb_overlays: Dict[str, QGraphicsPolygonItem] = {}

        # Drawing tools
        self._drawing_mode = False
        self._eraser_mode = False
        self._brush_size = 10
        self._is_drawing = False
        self._last_draw_pos: Optional[QPointF] = None
        self._current_draw_mask: Optional[np.ndarray] = None

        # ROI (Region of Interest) for batch processing
        self._roi_mode = False
        self._roi_rect: Optional[QGraphicsRectItem] = None
        self._roi_handles: List[QGraphicsRectItem] = []
        self._roi_data: Optional[Dict[str, int]] = None
        self._roi_dragging = False
        self._roi_resizing = False
        self._roi_resize_handle = -1
        self._roi_drag_start: Optional[QPointF] = None

        # Zoom
        self._zoom = 0
        self._zoom_factor = 1.15

        # Double click detection for auto SAM
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._on_single_click)
        self._pending_click_pos = None
        self._pending_click_label = None
        self._double_click_interval = 300  # ms

    def clear(self) -> None:
        """Clear all content from canvas."""
        self.clear_masks()
        self.clear_points()
        self._image_item.setPixmap(QPixmap())
        self.scene().setSceneRect(0, 0, 0, 0)

    def set_image(self, pixmap: QPixmap) -> None:
        """Set the base image.

        Args:
            pixmap: Image to display
        """
        self._image_item.setPixmap(pixmap)
        self._image_item.setOffset(0, 0)
        rect = pixmap.rect()
        self.scene().setSceneRect(rect)
        self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 0
        self.clear_points()
        self.clear_masks()

    def add_mask_overlay(
        self,
        mask: np.ndarray,
        mask_id: str = "default",
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: int = 120
    ) -> None:
        """Add a mask overlay with specific ID and color.

        Args:
            mask: Binary mask array
            mask_id: Unique identifier for this mask
            color: RGB color tuple
            alpha: Transparency (0-255)
        """
        # Create colored overlay
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        overlay[..., 3] = (mask.astype(np.uint8) * alpha)

        pixmap = numpy_to_qpixmap(overlay)

        # Remove existing mask with same ID
        if mask_id in self._mask_overlays:
            self.scene().removeItem(self._mask_overlays[mask_id])

        # Add new mask
        item = QGraphicsPixmapItem(pixmap)
        item.setZValue(1)
        self.scene().addItem(item)
        self._mask_overlays[mask_id] = item

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set the main segmentation mask overlay (for backward compatibility).

        Args:
            mask: Binary mask array or None to clear
        """
        if mask is None:
            self.clear_mask()
            return

        self.add_mask_overlay(mask, "main", (255, 0, 0), 120)

    def remove_mask_overlay(self, mask_id: str) -> None:
        """Remove a specific mask overlay.

        Args:
            mask_id: ID of mask to remove
        """
        if mask_id in self._mask_overlays:
            self.scene().removeItem(self._mask_overlays[mask_id])
            del self._mask_overlays[mask_id]

    def clear_mask(self) -> None:
        """Clear the main mask overlay."""
        self._mask_item.setPixmap(QPixmap())
        self.remove_mask_overlay("main")

    def clear_masks(self) -> None:
        """Clear all mask overlays."""
        for item in self._mask_overlays.values():
            self.scene().removeItem(item)
        self._mask_overlays.clear()
        self._mask_item.setPixmap(QPixmap())
        self.clear_bbox_overlays()
        self.clear_obb_overlays()

    def clear_bbox_overlays(self) -> None:
        """Remove all bounding box overlays."""
        for item in self._bbox_overlays.values():
            self.scene().removeItem(item)
        self._bbox_overlays.clear()

    def clear_obb_overlays(self) -> None:
        """Remove all oriented bounding box overlays."""
        for item in self._obb_overlays.values():
            self.scene().removeItem(item)
        self._obb_overlays.clear()

    def show_bbox_overlays(
        self,
        bboxes: Dict[str, Dict[str, float]],
        color: Tuple[int, int, int] = (255, 165, 0)
    ) -> None:
        """Display axis-aligned bounding boxes for masks."""
        self.clear_bbox_overlays()

        pen = QPen(QColor(*color))
        pen.setWidth(2)

        for mask_id, bbox in bboxes.items():
            try:
                x = float(bbox.get("x"))
                y = float(bbox.get("y"))
                width = float(bbox.get("width"))
                height = float(bbox.get("height"))
            except (AttributeError, TypeError, ValueError):
                continue

            if width <= 0 or height <= 0:
                continue

            item = self.scene().addRect(x, y, width, height, pen)
            item.setZValue(2.5)
            self._bbox_overlays[mask_id] = item

    def show_obb_overlays(
        self,
        obbs: Dict[str, List[List[float]]],
        color: Tuple[int, int, int] = (0, 200, 255)
    ) -> None:
        """Display oriented bounding boxes for masks."""
        self.clear_obb_overlays()

        pen = QPen(QColor(*color))
        pen.setWidth(2)

        for mask_id, points in obbs.items():
            if not points or len(points) != 4:
                continue

            try:
                polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
            except (TypeError, ValueError):
                continue

            item = self.scene().addPolygon(polygon, pen)
            item.setZValue(2.6)
            self._obb_overlays[mask_id] = item

    def add_point(self, x: float, y: float, label: int) -> None:
        """Add an annotation point.

        Args:
            x: X coordinate in scene
            y: Y coordinate in scene
            label: 1 for positive (foreground), 0 for negative (background)
        """
        self._points.append((x, y))
        self._labels.append(label)

        # Visual representation
        radius = 5
        color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
        brush = QBrush(color)
        pen = QPen(Qt.GlobalColor.white, 2)

        ellipse = self.scene().addEllipse(
            x - radius, y - radius, radius * 2, radius * 2,
            pen, brush
        )
        ellipse.setZValue(2)
        self._point_items.append((ellipse, label))

    def clear_points(self) -> None:
        """Clear all annotation points."""
        for item, _ in self._point_items:
            self.scene().removeItem(item)
        self._point_items.clear()
        self._points.clear()
        self._labels.clear()

    def get_points_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get annotation points and labels as numpy arrays.

        Returns:
            Tuple of (points array [N, 2], labels array [N])
        """
        if not self._points:
            return np.array([]), np.array([])

        points = np.array(self._points, dtype=np.float32)
        labels = np.array(self._labels, dtype=np.int32)
        return points, labels

    def set_drawing_mode(self, enabled: bool) -> None:
        """Enable or disable drawing mode.

        Args:
            enabled: True to enable drawing mode
        """
        self._drawing_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._is_drawing = False
            self._last_draw_pos = None

    def set_eraser_mode(self, enabled: bool) -> None:
        """Enable or disable eraser mode.

        Args:
            enabled: True to enable eraser mode
        """
        self._eraser_mode = enabled

    def set_brush_size(self, size: int) -> None:
        """Set brush size.

        Args:
            size: Brush size in pixels
        """
        self._brush_size = size

    def get_current_draw_mask(self) -> Optional[np.ndarray]:
        """Get the currently drawn mask.

        Returns:
            Current drawing mask or None
        """
        return self._current_draw_mask

    def _on_single_click(self) -> None:
        """Handle single click (after double-click timeout)."""
        if self._pending_click_pos and self._pending_click_label is not None:
            # Add point without auto-prediction
            self.add_point(
                self._pending_click_pos.x(),
                self._pending_click_pos.y(),
                self._pending_click_label
            )
            self.point_added.emit(
                self._pending_click_pos.x(),
                self._pending_click_pos.y(),
                self._pending_click_label
            )
            # Don't auto-trigger SAM on single click

        self._pending_click_pos = None
        self._pending_click_label = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events for adding points or drawing.

        Args:
            event: Mouse event
        """
        scene_pos = self.mapToScene(event.pos())

        # ROI mode handling
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on handle
            handle_idx = self._get_handle_at_pos(scene_pos)
            if handle_idx >= 0:
                self._roi_resizing = True
                self._roi_resize_handle = handle_idx
                self._roi_drag_start = scene_pos
                return

            # Check if clicking inside ROI
            if self._roi_rect and self._roi_rect.contains(scene_pos):
                self._roi_dragging = True
                self._roi_drag_start = scene_pos
                return

            # Start drawing new ROI
            self.clear_roi()
            pen = QPen(QColor(255, 165, 0), 3)
            brush = QBrush(QColor(255, 165, 0, 50))
            self._roi_rect = self.scene().addRect(
                scene_pos.x(), scene_pos.y(), 0, 0, pen, brush
            )
            self._roi_rect.setZValue(3)
            self._roi_drag_start = scene_pos
            self._roi_dragging = True
            return

        if self._drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            # Start drawing
            self._is_drawing = True
            scene_pos = self.mapToScene(event.pos())
            self._last_draw_pos = scene_pos

            # Initialize draw mask if needed
            if self._current_draw_mask is None and self._image_item.pixmap():
                pixmap = self._image_item.pixmap()
                height = pixmap.height()
                width = pixmap.width()
                self._current_draw_mask = np.zeros((height, width), dtype=bool)

            # Draw at current position
            self._draw_at_position(scene_pos)

        elif not self._drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            # Check if this is a double click
            scene_pos = self.mapToScene(event.pos())

            if self._click_timer.isActive():
                # This is a double click!
                self._click_timer.stop()
                self._pending_click_pos = None
                self._pending_click_label = None

                # Add point and trigger SAM immediately
                self.add_point(scene_pos.x(), scene_pos.y(), 1)
                self.point_added.emit(scene_pos.x(), scene_pos.y(), 1)
                self.request_sam_prediction.emit()
            else:
                # First click, wait to see if double click
                self._pending_click_pos = scene_pos
                self._pending_click_label = 1
                self._click_timer.start(self._double_click_interval)

        elif not self._drawing_mode and event.button() == Qt.MouseButton.RightButton:
            # Right click: add negative point (no double-click needed)
            scene_pos = self.mapToScene(event.pos())
            self.add_point(scene_pos.x(), scene_pos.y(), 0)
            self.point_added.emit(scene_pos.x(), scene_pos.y(), 0)
            # Auto-trigger SAM on right click
            self.request_sam_prediction.emit()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events for drawing.

        Args:
            event: Mouse event
        """
        scene_pos = self.mapToScene(event.pos())

        # ROI mode handling
        if self._roi_mode:
            if self._roi_dragging and self._roi_rect:
                if self._roi_resizing:
                    # Handle resizing
                    self._resize_roi(scene_pos)
                else:
                    # Handle moving or initial drawing
                    if self._roi_drag_start:
                        if self._roi_data:  # Moving existing ROI
                            dx = scene_pos.x() - self._roi_drag_start.x()
                            dy = scene_pos.y() - self._roi_drag_start.y()
                            new_x = self._roi_data['x'] + dx
                            new_y = self._roi_data['y'] + dy
                            self._roi_rect.setRect(
                                new_x, new_y,
                                self._roi_data['width'],
                                self._roi_data['height']
                            )
                            self._roi_drag_start = scene_pos
                        else:  # Drawing new ROI
                            start = self._roi_drag_start
                            width = scene_pos.x() - start.x()
                            height = scene_pos.y() - start.y()

                            # Normalize rectangle
                            x = start.x() if width >= 0 else scene_pos.x()
                            y = start.y() if height >= 0 else scene_pos.y()
                            w = abs(width)
                            h = abs(height)

                            self._roi_rect.setRect(x, y, w, h)
            return

        if self._is_drawing and self._drawing_mode:
            # Draw line from last position to current
            if self._last_draw_pos is not None:
                self._draw_line(self._last_draw_pos, scene_pos)

            self._last_draw_pos = scene_pos
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events.

        Args:
            event: Mouse event
        """
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._roi_dragging or self._roi_resizing:
                # Finalize ROI
                self._update_roi_from_rect()
                self._create_roi_handles()
                self._roi_dragging = False
                self._roi_resizing = False
                self._roi_resize_handle = -1
                self._roi_drag_start = None
            return

        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._last_draw_pos = None

            # Update mask overlay
            if self._current_draw_mask is not None:
                mask_id = "drawing"
                color = (255, 0, 0) if not self._eraser_mode else (128, 128, 128)
                self.add_mask_overlay(self._current_draw_mask, mask_id, color, alpha=100)
        else:
            super().mouseReleaseEvent(event)

    def _draw_at_position(self, pos: QPointF) -> None:
        """Draw at a specific position.

        Args:
            pos: Scene position
        """
        if self._current_draw_mask is None:
            return

        x, y = int(pos.x()), int(pos.y())
        height, width = self._current_draw_mask.shape
        radius = self._brush_size // 2

        # Draw circle
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        if self._eraser_mode:
                            self._current_draw_mask[py, px] = False
                        else:
                            self._current_draw_mask[py, px] = True

    def _draw_line(self, start: QPointF, end: QPointF) -> None:
        """Draw a line between two points.

        Args:
            start: Start position
            end: End position
        """
        # Bresenham's line algorithm with brush
        x0, y0 = int(start.x()), int(start.y())
        x1, y1 = int(end.x()), int(end.y())

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self._draw_at_position(QPointF(x0, y0))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel events for zooming.

        Args:
            event: Wheel event
        """
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = self._zoom_factor
            self._zoom += 1
        else:
            # Zoom out
            factor = 1 / self._zoom_factor
            self._zoom -= 1

        # Limit zoom range
        if self._zoom > 10:
            self._zoom = 10
            return
        elif self._zoom < -10:
            self._zoom = -10
            return

        self.scale(factor, factor)

    # ========== ROI Methods ==========

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable or disable ROI drawing mode.

        Args:
            enabled: True to enable ROI mode
        """
        self._roi_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            # Disable other modes
            self.set_drawing_mode(False)
        else:
            if not self._drawing_mode:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def get_roi(self) -> Optional[Dict[str, int]]:
        """Get current ROI coordinates.

        Returns:
            ROI dict {x, y, width, height} or None
        """
        return self._roi_data.copy() if self._roi_data else None

    def set_roi(self, roi: Dict[str, int]) -> None:
        """Set ROI from coordinates.

        Args:
            roi: ROI dictionary {x, y, width, height}
        """
        self.clear_roi()

        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

        # Create ROI rectangle
        pen = QPen(QColor(255, 165, 0))  # Orange
        pen.setWidth(3)
        brush = QBrush(QColor(255, 165, 0, 50))  # Semi-transparent orange

        self._roi_rect = self.scene().addRect(x, y, w, h, pen, brush)
        self._roi_rect.setZValue(3)

        self._roi_data = roi.copy()
        self._create_roi_handles()

    def clear_roi(self) -> None:
        """Clear current ROI."""
        if self._roi_rect:
            self.scene().removeItem(self._roi_rect)
            self._roi_rect = None

        for handle in self._roi_handles:
            self.scene().removeItem(handle)
        self._roi_handles.clear()

        self._roi_data = None

    def _create_roi_handles(self) -> None:
        """Create resize handles for ROI."""
        if not self._roi_rect or not self._roi_data:
            return

        # Clear existing handles
        for handle in self._roi_handles:
            self.scene().removeItem(handle)
        self._roi_handles.clear()

        x, y, w, h = (self._roi_data['x'], self._roi_data['y'],
                      self._roi_data['width'], self._roi_data['height'])

        handle_size = 10
        handle_positions = [
            (x - handle_size/2, y - handle_size/2),  # Top-left
            (x + w/2 - handle_size/2, y - handle_size/2),  # Top-center
            (x + w - handle_size/2, y - handle_size/2),  # Top-right
            (x + w - handle_size/2, y + h/2 - handle_size/2),  # Right-center
            (x + w - handle_size/2, y + h - handle_size/2),  # Bottom-right
            (x + w/2 - handle_size/2, y + h - handle_size/2),  # Bottom-center
            (x - handle_size/2, y + h - handle_size/2),  # Bottom-left
            (x - handle_size/2, y + h/2 - handle_size/2),  # Left-center
        ]

        pen = QPen(QColor(255, 165, 0))
        pen.setWidth(2)
        brush = QBrush(QColor(255, 255, 255))

        for pos in handle_positions:
            handle = self.scene().addRect(
                pos[0], pos[1], handle_size, handle_size, pen, brush
            )
            handle.setZValue(4)
            self._roi_handles.append(handle)

    def _get_handle_at_pos(self, scene_pos: QPointF) -> int:
        """Get handle index at position, or -1 if none.

        Args:
            scene_pos: Scene position

        Returns:
            Handle index (0-7) or -1
        """
        for i, handle in enumerate(self._roi_handles):
            rect = handle.rect()
            if rect.contains(scene_pos):
                return i
        return -1

    def _update_roi_from_rect(self) -> None:
        """Update ROI data from rectangle."""
        if self._roi_rect:
            rect = self._roi_rect.rect()
            self._roi_data = {
                'x': int(rect.x()),
                'y': int(rect.y()),
                'width': int(rect.width()),
                'height': int(rect.height())
            }

    def _resize_roi(self, scene_pos: QPointF) -> None:
        """Resize ROI based on handle dragging.

        Args:
            scene_pos: Current mouse position
        """
        if not self._roi_data or self._roi_resize_handle < 0:
            return

        x, y, w, h = (self._roi_data['x'], self._roi_data['y'],
                      self._roi_data['width'], self._roi_data['height'])

        handle = self._roi_resize_handle

        # Calculate new dimensions based on handle
        if handle == 0:  # Top-left
            new_x = scene_pos.x()
            new_y = scene_pos.y()
            w = x + w - new_x
            h = y + h - new_y
            x, y = new_x, new_y
        elif handle == 1:  # Top-center
            new_y = scene_pos.y()
            h = y + h - new_y
            y = new_y
        elif handle == 2:  # Top-right
            new_y = scene_pos.y()
            w = scene_pos.x() - x
            h = y + h - new_y
            y = new_y
        elif handle == 3:  # Right-center
            w = scene_pos.x() - x
        elif handle == 4:  # Bottom-right
            w = scene_pos.x() - x
            h = scene_pos.y() - y
        elif handle == 5:  # Bottom-center
            h = scene_pos.y() - y
        elif handle == 6:  # Bottom-left
            new_x = scene_pos.x()
            w = x + w - new_x
            h = scene_pos.y() - y
            x = new_x
        elif handle == 7:  # Left-center
            new_x = scene_pos.x()
            w = x + w - new_x
            x = new_x

        # Ensure minimum size
        if w < 10:
            w = 10
        if h < 10:
            h = 10

        self._roi_rect.setRect(x, y, w, h)
