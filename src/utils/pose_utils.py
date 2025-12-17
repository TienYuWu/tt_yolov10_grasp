"""Pose Utilities - 6D Pose Transformation and Projection Functions

This module provides utilities for 6D pose transformations, coordinate frame
conversions, and 3D-to-2D projection for visualization.
"""

from typing import Tuple

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
