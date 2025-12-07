"""Visualization Utilities - OBB and Pose Drawing Functions

This module provides visualization utilities for drawing oriented bounding boxes,
pose information, and depth maps on images.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_obb_box(
    image: np.ndarray,
    center: Tuple[float, float],
    width: float,
    height: float,
    rotation_rad: float,
    corners: Optional[List[Tuple[float, float]]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    draw_corners: bool = True,
    draw_center: bool = True
) -> np.ndarray:
    """Draw oriented bounding box on image.

    Args:
        image: Input image in RGB format (H x W x 3 uint8). Will be copied, not modified in-place.
        center: OBB center (cx, cy) in pixel coordinates
        width: OBB width in pixels
        height: OBB height in pixels
        rotation_rad: Rotation angle in radians
        corners: Optional pre-computed corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        color: Box color in RGB format (R, G, B) - default green (0, 255, 0)
        thickness: Line thickness in pixels
        draw_corners: Whether to draw corner circles
        draw_center: Whether to draw center circle

    Returns:
        Image with OBB drawn in RGB format

    Example:
        >>> img = cv2.cvtColor(cv2.imread('image.jpg'), cv2.COLOR_BGR2RGB)  # RGB format
        >>> result = draw_obb_box(img, (640, 360), 120, 80, 0.5, color=(0, 255, 0))
    """
    output = image.copy()

    # Use provided corners or calculate from center/width/height/rotation
    if corners is None:
        # Calculate corners from parameters
        cx, cy = center
        cos_angle = np.cos(rotation_rad)
        sin_angle = np.sin(rotation_rad)

        # Half dimensions
        hw = width / 2.0
        hh = height / 2.0

        # Calculate corner offsets
        corners_local = np.array([
            [-hw, -hh],  # Top-left
            [hw, -hh],   # Top-right
            [hw, hh],    # Bottom-right
            [-hw, hh]    # Bottom-left
        ])

        # Rotation matrix
        R = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        # Rotate and translate corners
        corners_rotated = corners_local @ R.T
        corners_world = corners_rotated + np.array([cx, cy])
        corners = corners_world.tolist()

    # Convert to integer pixel coordinates
    pts = np.array(corners, dtype=np.int32)

    # Draw box edges
    cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)

    # Draw corners as circles
    if draw_corners:
        for corner in pts:
            cv2.circle(output, tuple(corner), 4, color, -1)

    # Draw center
    if draw_center:
        center_px = (int(center[0]), int(center[1]))
        cv2.circle(output, center_px, 6, color, -1)
        cv2.circle(output, center_px, 6, (255, 255, 255), 2)

    return output


def draw_pose_info_text(
    image: np.ndarray,
    position: Dict[str, float],
    rotation_euler: Dict[str, float],
    detection_id: int,
    confidence: float,
    anchor_point: Tuple[int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Draw pose information text overlay on image.

    Args:
        image: Input image (will be copied)
        position: Position dict with keys 'x', 'y', 'z' in meters
        rotation_euler: Rotation dict with keys 'roll_rad', 'pitch_rad', 'yaw_rad'
        detection_id: Detection ID number
        confidence: Detection confidence score [0, 1]
        anchor_point: Text anchor position (x, y) in pixels
        font_scale: Font size scaling factor
        thickness: Text thickness
        bg_color: Background color (R, G, B)
        text_color: Text color (R, G, B)

    Returns:
        Image with text overlay

    Example:
        >>> position = {'x': 0.123, 'y': -0.045, 'z': 0.678}
        >>> rotation = {'roll_rad': 0.0, 'pitch_rad': 0.0, 'yaw_rad': 0.523}
        >>> result = draw_pose_info_text(img, position, rotation, 0, 0.95, (50, 50))
    """
    output = image.copy()

    # Format text lines
    lines = [
        f"ID: {detection_id} | Conf: {confidence:.2f}",
        f"Pos: ({position['x']:.3f}, {position['y']:.3f}, {position['z']:.3f})m",
        f"Rot: R={rotation_euler['roll_rad']:.3f}, P={rotation_euler['pitch_rad']:.3f}, Y={rotation_euler['yaw_rad']:.3f}"
    ]

    # Calculate text sizes for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25
    max_width = 0

    text_sizes = []
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        text_sizes.append((tw, th, baseline))
        max_width = max(max_width, tw)

    # Draw background rectangle
    padding = 8
    bg_top_left = (anchor_point[0] - padding, anchor_point[1] - padding)
    bg_bottom_right = (
        anchor_point[0] + max_width + padding,
        anchor_point[1] + len(lines) * line_height + padding
    )
    cv2.rectangle(output, bg_top_left, bg_bottom_right, bg_color, -1)
    cv2.rectangle(output, bg_top_left, bg_bottom_right, text_color, 1)

    # Draw text lines
    y_offset = anchor_point[1] + 5
    for i, line in enumerate(lines):
        tw, th, baseline = text_sizes[i]
        y_pos = y_offset + th + baseline
        cv2.putText(
            output,
            line,
            (anchor_point[0], y_pos),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        y_offset += line_height

    return output


def draw_depth_colormap(
    depth_image: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 2.0,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Convert depth image to color visualization.

    Args:
        depth_image: Depth image in meters (H x W float32)
        min_depth: Minimum depth for color mapping (meters)
        max_depth: Maximum depth for color mapping (meters)
        colormap: OpenCV colormap constant (default: COLORMAP_JET)

    Returns:
        Color-mapped depth image (H x W x 3 uint8)

    Example:
        >>> depth = np.random.rand(480, 640) * 2.0  # Random depth 0-2m
        >>> colored_depth = draw_depth_colormap(depth, 0.0, 2.0)
    """
    # Clip depth values
    depth_clipped = np.clip(depth_image, min_depth, max_depth)

    # Normalize to [0, 255]
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Mark invalid depths (0 or NaN) as black
    invalid_mask = (depth_image <= 0) | np.isnan(depth_image)
    depth_colored[invalid_mask] = [0, 0, 0]

    return depth_colored


def overlay_depth_on_rgb(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    alpha: float = 0.3,
    min_depth: float = 0.0,
    max_depth: float = 2.0
) -> np.ndarray:
    """Overlay depth colormap on RGB image with transparency.

    Args:
        rgb_image: RGB color image (H x W x 3 uint8)
        depth_image: Depth image in meters (H x W float32)
        alpha: Depth overlay opacity [0, 1] - default 0.3 (30% depth, 70% RGB)
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping

    Returns:
        Blended image (H x W x 3 uint8)

    Example:
        >>> rgb = cv2.imread('color.jpg')
        >>> depth = np.load('depth.npy')
        >>> blended = overlay_depth_on_rgb(rgb, depth, alpha=0.4)
    """
    # Convert depth to colormap
    depth_colored = draw_depth_colormap(depth_image, min_depth, max_depth)

    # Ensure same dimensions
    if rgb_image.shape[:2] != depth_colored.shape[:2]:
        depth_colored = cv2.resize(depth_colored, (rgb_image.shape[1], rgb_image.shape[0]))

    # Blend images
    blended = cv2.addWeighted(rgb_image, 1 - alpha, depth_colored, alpha, 0)

    return blended


def draw_fps_overlay(
    image: np.ndarray,
    fps: float,
    processing_time_ms: Optional[float] = None,
    position: str = 'top-left',
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """Draw FPS and processing time overlay on image.

    Args:
        image: Input image (will be copied)
        fps: Frames per second
        processing_time_ms: Optional processing time in milliseconds
        position: Text position - 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        font_scale: Font size scaling factor
        thickness: Text thickness

    Returns:
        Image with FPS overlay

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> result = draw_fps_overlay(img, 30.5, processing_time_ms=33.2)
    """
    output = image.copy()
    h, w = output.shape[:2]

    # Format text
    if processing_time_ms is not None:
        text = f"FPS: {fps:.1f} | {processing_time_ms:.1f}ms"
    else:
        text = f"FPS: {fps:.1f}"

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Determine position
    padding = 10
    if position == 'top-left':
        x, y = padding, th + padding
    elif position == 'top-right':
        x, y = w - tw - padding, th + padding
    elif position == 'bottom-left':
        x, y = padding, h - padding
    elif position == 'bottom-right':
        x, y = w - tw - padding, h - padding
    else:
        x, y = padding, th + padding  # Default to top-left

    # Draw background
    bg_top_left = (x - 5, y - th - 5)
    bg_bottom_right = (x + tw + 5, y + baseline + 5)
    cv2.rectangle(output, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
    cv2.rectangle(output, bg_top_left, bg_bottom_right, (255, 255, 255), 1)

    # Draw text
    cv2.putText(
        output,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),  # Green text
        thickness,
        cv2.LINE_AA
    )

    return output


def draw_detection_count(
    image: np.ndarray,
    count: int,
    position: str = 'top-right',
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """Draw detection count overlay on image.

    Args:
        image: Input image (will be copied)
        count: Number of detections
        position: Text position - 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        font_scale: Font size scaling factor
        thickness: Text thickness

    Returns:
        Image with detection count overlay

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> result = draw_detection_count(img, 5, position='top-right')
    """
    output = image.copy()
    h, w = output.shape[:2]

    # Format text
    text = f"Detections: {count}"

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Determine position
    padding = 10
    if position == 'top-left':
        x, y = padding, th + padding
    elif position == 'top-right':
        x, y = w - tw - padding, th + padding
    elif position == 'bottom-left':
        x, y = padding, h - padding
    elif position == 'bottom-right':
        x, y = w - tw - padding, h - padding
    else:
        x, y = w - tw - padding, th + padding  # Default to top-right

    # Draw background
    bg_top_left = (x - 5, y - th - 5)
    bg_bottom_right = (x + tw + 5, y + baseline + 5)
    cv2.rectangle(output, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
    cv2.rectangle(output, bg_top_left, bg_bottom_right, (255, 255, 255), 1)

    # Draw text
    cv2.putText(
        output,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 255),  # Cyan text (RGB format)
        thickness,
        cv2.LINE_AA
    )

    return output


def create_side_by_side_view(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 2.0
) -> np.ndarray:
    """Create side-by-side RGB and depth visualization.

    Args:
        rgb_image: RGB color image (H x W x 3)
        depth_image: Depth image in meters (H x W)
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping

    Returns:
        Side-by-side image (H x 2W x 3)

    Example:
        >>> rgb = cv2.imread('color.jpg')
        >>> depth = np.load('depth.npy')
        >>> side_by_side = create_side_by_side_view(rgb, depth)
    """
    # Convert depth to colormap
    depth_colored = draw_depth_colormap(depth_image, min_depth, max_depth)

    # Ensure same height
    if rgb_image.shape[0] != depth_colored.shape[0]:
        depth_colored = cv2.resize(depth_colored, (rgb_image.shape[1], rgb_image.shape[0]))

    # Concatenate horizontally
    side_by_side = np.hstack([rgb_image, depth_colored])

    # Add labels
    cv2.putText(side_by_side, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(side_by_side, "Depth", (rgb_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return side_by_side
