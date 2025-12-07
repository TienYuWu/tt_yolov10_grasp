"""Detection Worker - Background thread for YOLO detection and pose estimation

Supports both static image and camera live feed modes.

Vendor-ready code for industry-academia cooperation delivery.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..services.detection_service import DetectionService
from ..services.realsense_service import RealSenseService


class DetectionWorker(QThread):
    """Worker thread for detection and pose estimation."""

    # Signals
    detection_completed = Signal(dict, np.ndarray)  # result_dict, annotated_image
    performance_updated = Signal(float, float)  # fps, processing_time_ms
    detection_failed = Signal(str)  # error_message
    status_message = Signal(str)  # status text

    def __init__(
        self,
        detection_service: DetectionService,
        mode: str = 'image',  # 'image' or 'camera'
        camera_adapter: Optional[RealSenseService] = None,
        image_path: Optional[Path] = None,
        target_fps: float = 30.0,
        custom_intrinsics: Optional[Dict] = None
    ):
        """Initialize detection worker.

        Args:
            detection_service: DetectionService instance
            mode: Operation mode - 'image' for static, 'camera' for live feed
            camera_adapter: RealSenseService instance (required for camera mode)
            image_path: Path to image file (required for image mode)
            target_fps: Target FPS for camera mode (default: 30)
            custom_intrinsics: Custom camera intrinsics for image mode (optional)
                Dict with keys: width, height, fx, fy, cx, cy
        """
        super().__init__()
        self.detection_service = detection_service
        self.mode = mode
        self.camera_adapter = camera_adapter
        self.image_path = image_path
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
        self.custom_intrinsics = custom_intrinsics

        self.running = False
        self.paused = False

        # Performance tracking
        self.fps_history = []
        self.max_history_len = 30  # Average over 30 frames

    def run(self):
        """Execute detection worker."""
        self.running = True

        if self.mode == 'image':
            self._run_image_mode()
        elif self.mode == 'camera':
            self._run_camera_mode()
        else:
            self.detection_failed.emit(f"Invalid mode: {self.mode}")

    def _run_image_mode(self):
        """Run detection on a single static image."""
        try:
            if not self.image_path or not self.image_path.exists():
                self.detection_failed.emit(f"Image not found: {self.image_path}")
                return

            self.status_message.emit(f"Loading image: {self.image_path.name}")

            # Load image (OpenCV always loads as BGR)
            bgr_image = cv2.imread(str(self.image_path))
            if bgr_image is None:
                self.detection_failed.emit(f"Failed to load image: {self.image_path}")
                return

            # Convert BGR to RGB for consistency with detection service
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Load depth image from paired .npz file if available
            depth_image = self._load_depth_image(self.image_path, rgb_image.shape[:2])

            # Use custom intrinsics from GUI or defaults
            if self.custom_intrinsics:
                camera_intrinsics = self.custom_intrinsics.copy()
                self.status_message.emit(
                    f"Using custom intrinsics: "
                    f"{camera_intrinsics['width']}x{camera_intrinsics['height']}, "
                    f"fx={camera_intrinsics['fx']:.1f}"
                )
            else:
                # Fallback: Use scaled defaults based on image size
                h, w = rgb_image.shape[:2]
                scale_w = w / 1280.0
                scale_h = h / 720.0

                camera_intrinsics = {
                    'width': w,
                    'height': h,
                    'fx': 924.07073975 * scale_w,
                    'fy': 921.46142578 * scale_h,
                    'cx': 643.87634277 * scale_w,
                    'cy': 346.78930664 * scale_h
                }
                self.status_message.emit(
                    f"No custom intrinsics, using scaled defaults for {w}x{h}"
                )

            self.status_message.emit("Running detection...")

            # Run detection
            result = self.detection_service.detect_and_estimate_pose(
                rgb_image, depth_image, camera_intrinsics
            )

            # Update metadata
            result['metadata']['source_type'] = 'image'
            result['metadata']['source_path'] = str(self.image_path)

            # Create annotated image
            annotated = self._create_annotated_image(rgb_image, result)

            # Emit results
            self.detection_completed.emit(result, annotated)
            self.performance_updated.emit(
                0.0,  # No FPS for static image
                result['metadata']['processing_time_ms']
            )
            self.status_message.emit(
                f"Detection complete: {result['metadata']['detection_count']} objects found"
            )

        except Exception as e:
            self.detection_failed.emit(f"Detection error: {e}")
            import traceback
            traceback.print_exc()

    def _load_depth_image(self, rgb_path: Path, rgb_shape: tuple) -> np.ndarray:
        """Load depth image from paired .npz file.

        Args:
            rgb_path: Path to RGB image file
            rgb_shape: (height, width) of RGB image

        Returns:
            Depth image as float32 array in meters.
            Returns zero array if depth file not found.
        """
        try:
            # Derive depth path from RGB path
            # Pattern: rgb_TIMESTAMP.png -> depth_TIMESTAMP.npz
            rgb_name = rgb_path.stem  # e.g., "rgb_20251204_061558_208361"

            if rgb_name.startswith('rgb_'):
                depth_name = 'depth_' + rgb_name[4:]  # Replace "rgb_" with "depth_"
                depth_path = rgb_path.parent / f"{depth_name}.npz"

                if depth_path.exists():
                    self.status_message.emit(f"Loading depth: {depth_path.name}")

                    # Load .npz file
                    data = np.load(str(depth_path))

                    # Try common key names
                    depth_array = None
                    for key in ['depth', 'arr_0', 'depth_image']:
                        if key in data:
                            depth_array = data[key]
                            break

                    if depth_array is None:
                        # If no known key, use first array
                        depth_array = data[list(data.keys())[0]]

                    # Convert to float32 meters if needed
                    if depth_array.dtype == np.uint16:
                        # RealSense format: uint16 millimeters -> float32 meters
                        depth_image = depth_array.astype(np.float32) / 1000.0
                    elif depth_array.dtype == np.float32 or depth_array.dtype == np.float64:
                        # Already in float, assume meters
                        depth_image = depth_array.astype(np.float32)
                    else:
                        # Unknown format, try converting anyway
                        depth_image = depth_array.astype(np.float32)

                    # Verify shape matches RGB
                    if depth_image.shape[:2] != rgb_shape:
                        self.status_message.emit(
                            f"⚠️ Depth shape {depth_image.shape} != RGB shape {rgb_shape}, using dummy"
                        )
                        return np.zeros(rgb_shape, dtype=np.float32)

                    self.status_message.emit(
                        f"✅ Loaded depth: {depth_array.dtype} -> float32, "
                        f"range [{depth_image.min():.3f}, {depth_image.max():.3f}] m"
                    )
                    return depth_image
                else:
                    self.status_message.emit(
                        f"⚠️ No depth file found at: {depth_path.name}, using dummy depth"
                    )
            else:
                self.status_message.emit(
                    "⚠️ RGB filename doesn't match pattern 'rgb_*.png', using dummy depth"
                )

        except Exception as e:
            self.status_message.emit(f"⚠️ Error loading depth: {e}, using dummy depth")
            import traceback
            traceback.print_exc()

        # Fallback to dummy depth
        return np.zeros(rgb_shape, dtype=np.float32)

    def _run_camera_mode(self):
        """Run detection on camera live feed."""
        try:
            if not self.camera_adapter:
                self.detection_failed.emit("Camera adapter not provided")
                return

            self.status_message.emit("Starting camera feed...")

            # Start camera
            if hasattr(self.camera_adapter, 'start'):
                self.camera_adapter.start()

            frame_count = 0
            last_time = time.time()

            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                frame_start = time.time()

                try:
                    # Get frame from camera
                    rgb_image, depth_image = self.camera_adapter.get_frame()

                    if rgb_image is None or depth_image is None:
                        self.status_message.emit("Waiting for camera frames...")
                        time.sleep(0.1)
                        continue

                    # Get camera intrinsics
                    camera_intrinsics = self.camera_adapter.get_intrinsics()

                    # Run detection
                    result = self.detection_service.detect_and_estimate_pose(
                        rgb_image, depth_image, camera_intrinsics
                    )

                    # Update metadata
                    result['metadata']['source_type'] = 'camera'
                    result['metadata']['source_path'] = None

                    # Create annotated image
                    annotated = self._create_annotated_image(rgb_image, result)

                    # Emit results
                    self.detection_completed.emit(result, annotated)

                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_time

                    if elapsed >= 1.0:  # Update FPS every second
                        fps = frame_count / elapsed
                        self.fps_history.append(fps)
                        if len(self.fps_history) > self.max_history_len:
                            self.fps_history.pop(0)

                        avg_fps = sum(self.fps_history) / len(self.fps_history)
                        self.performance_updated.emit(
                            avg_fps,
                            result['metadata']['processing_time_ms']
                        )

                        frame_count = 0
                        last_time = current_time

                    # Frame rate limiting
                    frame_elapsed = time.time() - frame_start
                    if frame_elapsed < self.frame_time:
                        time.sleep(self.frame_time - frame_elapsed)

                except Exception as e:
                    self.status_message.emit(f"Frame processing error: {e}")
                    time.sleep(0.1)

            # Stop camera
            if hasattr(self.camera_adapter, 'stop'):
                self.camera_adapter.stop()

            self.status_message.emit("Camera stopped")

        except Exception as e:
            self.detection_failed.emit(f"Camera mode error: {e}")
            import traceback
            traceback.print_exc()

    def _create_annotated_image(
        self,
        rgb_image: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """Create annotated image with OBB boxes and pose information.

        Args:
            rgb_image: RGB image
            result: Detection result dictionary

        Returns:
            Annotated RGB image
        """
        from ..utils.visualization_utils import (
            draw_obb_box,
            draw_pose_info_text,
            draw_fps_overlay,
            draw_detection_count
        )
        from ..utils.pose_utils import draw_coordinate_axes

        annotated = rgb_image.copy()
        camera_intrinsics = result['camera_intrinsics']

        # Draw each detection
        for detection in result['detections']:
            obb = detection['obb']
            pose = detection['pose']
            confidence = detection['confidence']
            det_id = detection['detection_id']

            # Draw OBB box (RGB format: Green)
            annotated = draw_obb_box(
                annotated,
                center=obb['center'],
                width=obb['width'],
                height=obb['height'],
                rotation_rad=obb['rotation_rad'],
                corners=obb['corners'],
                color=(0, 255, 0),  # RGB: Green
                thickness=2
            )

            # Draw coordinate axes
            T = np.array(pose['transform_matrix'])
            annotated = draw_coordinate_axes(
                annotated,
                T=T,
                intrinsics=camera_intrinsics,
                length=0.05  # 5cm axes
            )

            # Draw pose info text
            text_anchor = (int(obb['center'][0]) + 20, int(obb['center'][1]) - 20)
            annotated = draw_pose_info_text(
                annotated,
                position=pose['position'],
                rotation_euler=pose['rotation_euler'],
                detection_id=det_id,
                confidence=confidence,
                anchor_point=text_anchor
            )

        # Draw detection count
        annotated = draw_detection_count(
            annotated,
            count=result['metadata']['detection_count'],
            position='top-right'
        )

        return annotated

    def stop(self):
        """Stop the worker."""
        self.running = False
        self.status_message.emit("Stopping worker...")

    def pause(self):
        """Pause camera feed (camera mode only)."""
        self.paused = True
        self.status_message.emit("Paused")

    def resume(self):
        """Resume camera feed (camera mode only)."""
        self.paused = False
        self.status_message.emit("Resumed")

    def set_target_fps(self, fps: float):
        """Change target FPS for camera mode.

        Args:
            fps: Target frames per second
        """
        self.target_fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0.0
        self.status_message.emit(f"Target FPS set to: {fps:.1f}")
