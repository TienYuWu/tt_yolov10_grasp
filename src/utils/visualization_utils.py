"""Visualization Utilities - OBB and Pose Drawing Functions

This module provides visualization utilities for drawing oriented bounding boxes,
pose information, and depth maps on images.
"""

from typing import List, Optional, Tuple

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
    thickness: int = 1,
    draw_corners: bool = True,
    draw_center: bool = False,
    draw_cross: bool = True,
    cross_color: Tuple[int, int, int] = (255, 0, 0)
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

    # Draw corners as circles (optional)
    if draw_corners:
        for corner in pts:
            cv2.circle(output, tuple(corner), 4, color, -1)

    # Draw oriented cross aligned to OBB axes
    if draw_cross:
        cx, cy = center
        cos_angle = np.cos(rotation_rad)
        sin_angle = np.sin(rotation_rad)

        # Axis unit vectors (x: width direction, y: height direction)
        axis_x = np.array([cos_angle, sin_angle])
        axis_y = np.array([-sin_angle, cos_angle])

        # Cross arm lengths (30% of width/height each side)
        arm_x = 0.3 * width / 2.0
        arm_y = 0.3 * height / 2.0

        pts_cross = [
            (int(cx - axis_x[0] * arm_x), int(cy - axis_x[1] * arm_x)),
            (int(cx + axis_x[0] * arm_x), int(cy + axis_x[1] * arm_x)),
            (int(cx - axis_y[0] * arm_y), int(cy - axis_y[1] * arm_y)),
            (int(cx + axis_y[0] * arm_y), int(cy + axis_y[1] * arm_y)),
        ]

        cv2.line(output, pts_cross[0], pts_cross[1], cross_color, thickness, cv2.LINE_AA)
        cv2.line(output, pts_cross[2], pts_cross[3], cross_color, thickness, cv2.LINE_AA)

    return output