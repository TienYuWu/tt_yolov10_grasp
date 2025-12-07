"""Detection Service - YOLO OBB Detection with 6D Pose Estimation

This service integrates YOLO OBB detection with 6D pose estimation,
supporting both simple (vertical grasp) and full (6D pose) modes.

Adapted from ROS2 yolo_worker_node.py for pure Python Windows environment.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO


class DetectionService:
    """YOLO OBB Detection + 6D Pose Estimation Service"""

    def __init__(
        self,
        model_path: str,
        pose_mode: str = "simple",
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.3,
        device: str = "cuda"
    ):
        """Initialize detection service.

        Args:
            model_path: Path to YOLO OBB model weights (.pt file)
            pose_mode: Pose estimation mode - 'simple' or 'full'
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device for inference - 'cuda' or 'cpu'
        """
        self.model_path = Path(model_path)
        self.pose_mode = pose_mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load YOLO model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        print(f"✅ Loaded YOLO model from: {self.model_path}")
        print(f"📐 Pose mode: {self.pose_mode}")
        print(f"🖥️ Device: {self.device}")

    def detect_and_estimate_pose(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_intrinsics: Dict[str, float]
    ) -> Dict:
        """Execute detection and pose estimation pipeline.

        Args:
            rgb_image: RGB image (H x W x 3 uint8)
            depth_image: Depth image in meters (H x W float32)
            camera_intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy', 'width', 'height'

        Returns:
            Results dictionary with structure matching JSON spec:
            {
                'metadata': {...},
                'camera_intrinsics': {...},
                'detections': [...]
            }
        """
        start_time = datetime.now()

        try:
            # YOLO detection
            with torch.no_grad():
                results = self.model(
                    rgb_image,
                    device=self.device,
                    imgsz=640,
                    verbose=False,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Extract OBB boxes
            obb_data = self._get_obb_boxes_data(results)
            print(f"🎯 Detected {len(obb_data)} objects")

            if not obb_data:
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                return self._create_empty_result(
                    rgb_image, camera_intrinsics, processing_time_ms
                )

            # Calculate poses for each detection
            detections = []
            for idx, box in enumerate(obb_data):
                pose_matrix = self._calculate_pose_for_box(
                    box, depth_image, camera_intrinsics
                )

                if pose_matrix is not None:
                    detection_data = self._create_detection_dict(
                        detection_id=idx,
                        box=box,
                        pose_matrix=pose_matrix
                    )
                    detections.append(detection_data)

            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Create full result dictionary
            result = {
                'metadata': {
                    'timestamp': start_time.isoformat(),
                    'source_type': 'unknown',  # Will be set by caller
                    'source_path': None,
                    'pose_mode': self.pose_mode,
                    'model_path': str(self.model_path),
                    'processing_time_ms': processing_time_ms,
                    'detection_count': len(detections)
                },
                'camera_intrinsics': camera_intrinsics,
                'detections': detections
            }

            return result

        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result(
                rgb_image, camera_intrinsics, 0.0, error=str(e)
            )

    def _get_obb_boxes_data(self, results) -> List[Dict]:
        """Extract OBB box data from YOLO results.

        Args:
            results: YOLO results object

        Returns:
            List of OBB dictionaries with keys:
            'center', 'width', 'height', 'rotation', 'corners', 'confidence'
        """
        obb_data = []

        if results and len(results) > 0 and results[0].obb:
            for box in results[0].obb:
                if hasattr(box, 'xywhr'):
                    data = box.data.cpu().numpy()[0]
                    x, y, w, h, r = data[0:5]
                    confidence = float(data[5]) if len(data) > 5 else 1.0

                    # Calculate corners
                    cos_r, sin_r = np.cos(r), np.sin(r)
                    corners_local = np.array([
                        [-w/2, -h/2],
                        [w/2, -h/2],
                        [w/2, h/2],
                        [-w/2, h/2]
                    ])

                    corners = np.array([
                        [cos_r * c[0] - sin_r * c[1] + x,
                         sin_r * c[0] + cos_r * c[1] + y]
                        for c in corners_local
                    ])

                    obb_data.append({
                        'center': (float(x), float(y)),
                        'width': float(w),
                        'height': float(h),
                        'rotation': float(r),
                        'corners': corners.tolist(),
                        'confidence': confidence
                    })

        return obb_data

    def _calculate_pose_for_box(
        self,
        box: Dict,
        depth_image: np.ndarray,
        camera_info: Dict
    ) -> Optional[np.ndarray]:
        """Calculate 6D pose for a detected box.

        Supports two modes:
        - 'simple': Median depth + Z-axis rotation only (vertical grasp)
        - 'full': Complete 6D pose with surface normal estimation

        Args:
            box: OBB dictionary
            depth_image: Depth image in meters (H x W)
            camera_info: Camera intrinsics dict

        Returns:
            4x4 transformation matrix or None if failed
        """
        try:
            # Extract parameters
            center = np.array(box['center'], dtype=np.float32)
            width, height, rotation = box['width'], box['height'], box['rotation']
            fx, fy, cx, cy = camera_info['fx'], camera_info['fy'], camera_info['cx'], camera_info['cy']

            # Get median depth at OBB center (5x5 window)
            h, w = depth_image.shape[:2]
            center_u, center_v = int(round(center[0])), int(round(center[1]))

            # Extract 5x5 window
            window_size = 5
            half_window = window_size // 2
            u_start = max(0, center_u - half_window)
            u_end = min(w, center_u + half_window + 1)
            v_start = max(0, center_v - half_window)
            v_end = min(h, center_v + half_window + 1)

            depth_window = depth_image[v_start:v_end, u_start:u_end]
            valid_depths = depth_window[depth_window > 0]

            if len(valid_depths) == 0:
                print(f"⚠️ No valid depth at OBB center")
                return None

            median_depth = float(np.median(valid_depths))
            print(f"📏 OBB center median depth: {median_depth:.4f}m ({len(valid_depths)}/{window_size*window_size} valid)")

            # ========================================
            # Mode branching: Simple vs Full
            # ========================================

            if self.pose_mode == 'simple':
                return self._calculate_simple_pose(
                    center_u, center_v, median_depth, rotation, fx, fy, cx, cy
                )
            else:
                return self._calculate_full_pose(
                    center, width, height, rotation, depth_image, fx, fy, cx, cy
                )

        except Exception as e:
            print(f"❌ Pose calculation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_simple_pose(
        self,
        center_u: int,
        center_v: int,
        median_depth: float,
        rotation: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> np.ndarray:
        """Calculate simple pose: median depth + Z-axis rotation.

        Args:
            center_u, center_v: OBB center pixel coordinates
            median_depth: Median depth at center in meters
            rotation: OBB rotation angle in radians
            fx, fy, cx, cy: Camera intrinsics

        Returns:
            4x4 transformation matrix
        """
        print("📍 Using Simple mode: median depth + Z-axis rotation")

        # Calculate 3D position using median depth
        z = median_depth
        x = (center_u - cx) * z / fx
        y = (center_v - cy) * z / fy

        # Build Z-axis rotation (yaw)
        # Rotate 90° to align gripper with short axis
        yaw = rotation + np.pi / 2

        # Build rotation matrix (Roll=0, Pitch=0, only Yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ], dtype=np.float64)

        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_z
        T[:3, 3] = [x, y, z]

        yaw_deg = np.degrees(yaw)
        print(f"✅ Simple pose: x={x:.4f}, y={y:.4f}, z={z:.4f}, yaw={yaw:.4f}rad ({yaw_deg:.1f}°)")

        return T

    def _calculate_full_pose(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        rotation: float,
        depth_image: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> Optional[np.ndarray]:
        """Calculate full 6D pose with surface normal estimation.

        Uses PCA on 3D point cloud to find surface normal.

        Args:
            center: OBB center (u, v)
            width, height: OBB dimensions
            rotation: OBB rotation angle
            depth_image: Depth image in meters
            fx, fy, cx, cy: Camera intrinsics

        Returns:
            4x4 transformation matrix or None
        """
        print("📍 Using Full mode: complete 6D pose with surface normal")

        # 2D rotation matrix
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R_2d = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        # Sampling window (33% of OBB size)
        half_w, half_h = width * 0.33, height * 0.33
        num_x = max(int(half_w * 2), 5)
        num_y = max(int(half_h * 2), 5)

        # Create local grid
        local_x = np.linspace(-half_w, half_w, num_x)
        local_y = np.linspace(-half_h, half_h, num_y)
        local_grid_x, local_grid_y = np.meshgrid(local_x, local_y)
        local_points = np.stack([local_grid_x.flatten(), local_grid_y.flatten()], axis=1)

        # Rotate and translate to image coordinates
        image_points = (R_2d @ local_points.T).T + center

        # Reproject to 3D
        h, w = depth_image.shape[:2]
        points_3d = []
        for pt in image_points:
            u, v = int(round(pt[0])), int(round(pt[1]))
            if 0 <= u < w and 0 <= v < h:
                depth = depth_image[v, u]
                if depth > 0:
                    z = depth
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points_3d.append([x, y, z])

        if len(points_3d) < 10:
            print(f"⚠️ Too few valid points: {len(points_3d)}")
            return None

        points_3d = np.array(points_3d)

        # ========== PCA to find surface normal ==========
        centroid = np.mean(points_3d, axis=0)
        centered = points_3d - centroid
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalues: large → small = major axis → minor axis → normal
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract axes
        axis_major = eigenvectors[:, 0].real  # Long axis (OBB long edge)
        axis_minor = eigenvectors[:, 1].real  # Short axis (OBB short edge)
        normal = eigenvectors[:, 2].real      # Surface normal (perpendicular)

        # Ensure normal points toward camera (negative Z in camera frame)
        if normal[2] > 0:
            normal = -normal
            axis_minor = -axis_minor  # Flip one axis to maintain right-hand rule

        print(f"📐 Eigenvalues: {eigenvalues}")
        print(f"📐 Normal vector: {normal} (toward camera)")

        # ========== Build gripper coordinate frame ==========
        # Gripper Z-axis = surface normal (approach direction)
        # Gripper X-axis = minor axis (gripper opening, aligned with short edge)
        # Gripper Y-axis = determined by right-hand rule

        gripper_z = normal / np.linalg.norm(normal)
        gripper_x = axis_minor / np.linalg.norm(axis_minor)

        # Ensure X and Z are orthogonal (Gram-Schmidt)
        gripper_x = gripper_x - np.dot(gripper_x, gripper_z) * gripper_z
        gripper_x = gripper_x / np.linalg.norm(gripper_x)

        # Y = Z × X (right-hand rule)
        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y = gripper_y / np.linalg.norm(gripper_y)

        # Build rotation matrix [X | Y | Z] as columns
        R_gripper = np.column_stack([gripper_x, gripper_y, gripper_z])

        # Ensure valid rotation matrix (det = +1)
        if np.linalg.det(R_gripper) < 0:
            R_gripper[:, 1] = -R_gripper[:, 1]  # Flip Y axis

        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_gripper
        T[:3, 3] = centroid

        # Calculate Euler angles for logging
        euler = R.from_matrix(R_gripper).as_euler('xyz', degrees=True)
        print(f"✅ Full pose: position={centroid}, euler(xyz)={euler}°")

        return T

    def _create_detection_dict(
        self,
        detection_id: int,
        box: Dict,
        pose_matrix: np.ndarray
    ) -> Dict:
        """Create detection dictionary matching JSON spec.

        Args:
            detection_id: Detection ID
            box: OBB dictionary
            pose_matrix: 4x4 transformation matrix

        Returns:
            Detection dictionary
        """
        # Extract position
        position = pose_matrix[:3, 3]

        # Extract rotation matrix and convert to Euler angles
        R_matrix = pose_matrix[:3, :3]
        euler = R.from_matrix(R_matrix).as_euler('xyz')  # Radians

        # Extract axes
        x_axis = R_matrix[:, 0]
        y_axis = R_matrix[:, 1]
        z_axis = R_matrix[:, 2]

        return {
            'detection_id': detection_id,
            'confidence': box['confidence'],
            'obb': {
                'center': box['center'],
                'width': box['width'],
                'height': box['height'],
                'rotation_rad': box['rotation'],
                'corners': box['corners']
            },
            'pose': {
                'transform_matrix': pose_matrix.tolist(),
                'position': {
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                },
                'rotation_euler': {
                    'roll_rad': float(euler[0]),
                    'pitch_rad': float(euler[1]),
                    'yaw_rad': float(euler[2])
                },
                'axes': {
                    'x_axis': x_axis.tolist(),
                    'y_axis': y_axis.tolist(),
                    'z_axis': z_axis.tolist()
                }
            }
        }

    def _create_empty_result(
        self,
        rgb_image: np.ndarray,
        camera_intrinsics: Dict,
        processing_time_ms: float,
        error: Optional[str] = None
    ) -> Dict:
        """Create empty result dictionary.

        Args:
            rgb_image: RGB image (for dimensions)
            camera_intrinsics: Camera intrinsics
            processing_time_ms: Processing time
            error: Optional error message

        Returns:
            Empty result dictionary
        """
        result = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_type': 'unknown',
                'source_path': None,
                'pose_mode': self.pose_mode,
                'model_path': str(self.model_path),
                'processing_time_ms': processing_time_ms,
                'detection_count': 0
            },
            'camera_intrinsics': camera_intrinsics,
            'detections': []
        }

        if error:
            result['metadata']['error'] = error

        return result

    def set_pose_mode(self, mode: str):
        """Change pose estimation mode.

        Args:
            mode: 'simple' or 'full'
        """
        if mode not in ['simple', 'full']:
            raise ValueError(f"Invalid pose mode: {mode}. Must be 'simple' or 'full'")

        self.pose_mode = mode
        print(f"📐 Pose mode changed to: {self.pose_mode}")
