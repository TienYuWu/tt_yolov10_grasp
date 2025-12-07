"""Pose Utilities - 6D Pose Transformation and Projection Functions

This module provides utilities for 6D pose transformations, coordinate frame
conversions, and 3D-to-2D projection for visualization.
"""

from typing import Dict, Tuple

import cv2
import numpy as np


def transform_matrix_to_euler(T: np.ndarray) -> Tuple[float, float, float]:
    """Convert 4x4 transformation matrix to Euler angles (radians).

    Uses the rotation matrix extraction and converts to Roll-Pitch-Yaw (XYZ).

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians

    Example:
        >>> T = np.eye(4)
        >>> roll, pitch, yaw = transform_matrix_to_euler(T)
        >>> print(f"Euler angles: {roll:.3f}, {pitch:.3f}, {yaw:.3f}")
    """
    # Extract rotation matrix (top-left 3x3)
    R = T[:3, :3]

    # Calculate pitch (rotation around Y-axis)
    # Clamp to avoid numerical issues with arcsin
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)

    # Check for gimbal lock (pitch near ±90°)
    threshold = 0.99999
    if np.abs(sin_pitch) < threshold:
        # Normal case - calculate roll and yaw
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock - set roll to 0 and calculate yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return roll, pitch, yaw


def euler_to_transform_matrix(
    position: np.ndarray,
    euler: np.ndarray
) -> np.ndarray:
    """Build 4x4 transformation matrix from position and Euler angles.

    Args:
        position: 3D position [x, y, z] in meters
        euler: Euler angles [roll, pitch, yaw] in radians

    Returns:
        4x4 homogeneous transformation matrix

    Example:
        >>> position = np.array([0.1, -0.05, 0.5])
        >>> euler = np.array([0.0, 0.0, np.pi/4])
        >>> T = euler_to_transform_matrix(position, euler)
    """
    roll, pitch, yaw = euler

    # Rotation matrices for each axis
    # Roll (rotation around X-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Pitch (rotation around Y-axis)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Yaw (rotation around Z-axis)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = R_z * R_y * R_x (ZYX order)
    R = R_z @ R_y @ R_x

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position

    return T


def project_3d_to_2d(
    point_3d: np.ndarray,
    camera_intrinsics: Dict[str, float]
) -> Tuple[float, float]:
    """Project 3D point to 2D image coordinates using camera intrinsics.

    Args:
        point_3d: 3D point [x, y, z] in camera frame (meters)
        camera_intrinsics: Dictionary with keys 'fx', 'fy', 'cx', 'cy'

    Returns:
        Tuple of (u, v) pixel coordinates

    Example:
        >>> point_3d = np.array([0.1, -0.05, 0.5])
        >>> intrinsics = {'fx': 900, 'fy': 900, 'cx': 640, 'cy': 360}
        >>> u, v = project_3d_to_2d(point_3d, intrinsics)
    """
    x, y, z = point_3d
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Perspective projection
    # u = fx * (x / z) + cx
    # v = fy * (y / z) + cy
    if z > 0:
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
    else:
        # Point behind camera - return center point
        u = cx
        v = cy

    return u, v


def draw_coordinate_axes(
    image: np.ndarray,
    T: np.ndarray,
    intrinsics: Dict[str, float],
    length: float = 0.05
) -> np.ndarray:
    """Draw 3D coordinate axes (X=Red, Y=Green, Z=Blue) on image.

    Projects the coordinate frame defined by transformation matrix T onto
    the image plane and draws colored arrows.

    Args:
        image: Input image (will be copied, not modified in-place)
        T: 4x4 transformation matrix defining the coordinate frame
        intrinsics: Camera intrinsics dict with 'fx', 'fy', 'cx', 'cy'
        length: Length of each axis arrow in meters (default: 0.05m = 5cm)

    Returns:
        Image with coordinate axes drawn

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> T = np.eye(4)
        >>> T[:3, 3] = [0.1, 0.0, 0.5]  # Position at (10cm, 0, 50cm)
        >>> intrinsics = {'fx': 900, 'fy': 900, 'cx': 640, 'cy': 360}
        >>> result = draw_coordinate_axes(img, T, intrinsics, length=0.1)
    """
    output = image.copy()

    # Extract position and rotation from transformation matrix
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    # Calculate end points of axes
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length

    # Project all points to 2D
    origin_2d = project_3d_to_2d(origin, intrinsics)
    x_end_2d = project_3d_to_2d(x_end, intrinsics)
    y_end_2d = project_3d_to_2d(y_end, intrinsics)
    z_end_2d = project_3d_to_2d(z_end, intrinsics)

    # Convert to integer pixel coordinates
    origin_px = (int(origin_2d[0]), int(origin_2d[1]))
    x_end_px = (int(x_end_2d[0]), int(x_end_2d[1]))
    y_end_px = (int(y_end_2d[0]), int(y_end_2d[1]))
    z_end_px = (int(z_end_2d[0]), int(z_end_2d[1]))

    # Draw axes as arrows (RGB format)
    # X-axis = Red
    cv2.arrowedLine(output, origin_px, x_end_px, (255, 0, 0), 2, tipLength=0.3)

    # Y-axis = Green
    cv2.arrowedLine(output, origin_px, y_end_px, (0, 255, 0), 2, tipLength=0.3)

    # Z-axis = Blue
    cv2.arrowedLine(output, origin_px, z_end_px, (0, 0, 255), 2, tipLength=0.3)

    # Draw small circle at origin for visibility
    cv2.circle(output, origin_px, 5, (255, 255, 255), -1)
    cv2.circle(output, origin_px, 5, (0, 0, 0), 2)

    return output


def get_rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Calculate rotation matrix that rotates vec_from to vec_to.

    Uses Rodrigues' rotation formula to compute the rotation matrix.

    Args:
        vec_from: Source unit vector (will be normalized)
        vec_to: Target unit vector (will be normalized)

    Returns:
        3x3 rotation matrix

    Example:
        >>> vec_from = np.array([1, 0, 0])
        >>> vec_to = np.array([0, 1, 0])
        >>> R = get_rotation_matrix_from_vectors(vec_from, vec_to)
    """
    # Normalize vectors
    a = vec_from / np.linalg.norm(vec_from)
    b = vec_to / np.linalg.norm(vec_to)

    # Cross product
    v = np.cross(a, b)
    c = np.dot(a, b)

    # Handle parallel vectors
    if np.allclose(v, 0):
        if c > 0:
            # Same direction - return identity
            return np.eye(3)
        else:
            # Opposite direction - return 180° rotation around perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, a) * a
            perp = perp / np.linalg.norm(perp)
            return 2 * np.outer(perp, perp) - np.eye(3)

    # Skew-symmetric cross-product matrix
    v_x = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rodrigues' formula
    R = np.eye(3) + v_x + v_x @ v_x * (1 / (1 + c))

    return R


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-pi, pi]

    Example:
        >>> angle = normalize_angle(3.5 * np.pi)
        >>> print(f"{angle:.3f} radians")
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
