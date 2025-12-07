#!/usr/bin/env python3
"""
RealSense Service - Intel RealSense camera integration for detection system

This module provides a unified interface for RealSense cameras, wrapping the
pyrealsense2 SDK with compatibility methods for the detection worker.

Vendor-ready code for industry-academia cooperation delivery.

Features:
1. RealSense pipeline management (RGB + Depth streams)
2. Aligned RGBD frame acquisition
3. Camera intrinsics extraction
4. Temporal filtering for depth noise reduction
5. Compatibility wrappers for detection worker integration

Output formats:
- color_image: np.ndarray (H, W, 3) BGR or RGB uint8
- depth_image: np.ndarray (H, W) uint16 (units: millimeters)
- camera_intrinsics: dict {width, height, fx, fy, cx, cy}
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Dict


class RealSenseService:
    """
    Intel RealSense camera service with unified interface.

    Provides both original interface (get_rgbd_frame) and compatibility
    wrappers (get_frame, get_intrinsics) for detection worker integration.

    Usage:
        # Context manager (recommended)
        with RealSenseService() as camera:
            rgb, depth = camera.get_frame()
            intrinsics = camera.get_intrinsics()

        # Manual management
        camera = RealSenseService()
        rgb, depth = camera.get_frame()
        camera.stop()
    """

    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 enable_temporal_filter: bool = True):
        """
        Initialize RealSense camera service.

        Args:
            width: Image width in pixels (default: 1280)
            height: Image height in pixels (default: 720)
            fps: Frame rate (default: 30)
            enable_temporal_filter: Enable depth noise reduction (default: True)

        Raises:
            RuntimeError: If camera initialization fails
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Configure RGB and depth streams
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        print(f"正在啟動 RealSense 相機 ({width}x{height} @ {fps}fps)...")
        self.profile = self.pipeline.start(config)
        print("RealSense 相機已啟動")

        # Align depth to color frame
        self.align = rs.align(rs.stream.color)

        # Temporal filter for depth noise reduction
        self.temporal_filter = rs.temporal_filter() if enable_temporal_filter else None

        # Extract depth scale (default: 0.001, meaning 1mm = 1 unit)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # Extract camera intrinsics
        depth_profile = self.profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        print(f"深度縮放比例: {self.depth_scale}")
        print(f"相機內參: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
              f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")

    # ================================================================
    # Original Interface (from realsense_adapter.py)
    # ================================================================

    def get_camera_intrinsics(self) -> Dict:
        """
        Get camera intrinsics in universal format.

        Returns:
            dict: Camera intrinsics with keys:
                - 'width': int - Image width
                - 'height': int - Image height
                - 'fx': float - Focal length X
                - 'fy': float - Focal length Y
                - 'cx': float - Principal point X
                - 'cy': float - Principal point Y
        """
        return {
            'width': self.intrinsics.width,
            'height': self.intrinsics.height,
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'cx': self.intrinsics.ppx,
            'cy': self.intrinsics.ppy
        }

    def get_rgbd_frame(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get aligned RGBD frame in original format.

        Returns:
            Tuple containing:
                - color_image: np.ndarray (H, W, 3) BGR uint8
                - depth_image: np.ndarray (H, W) uint16 (millimeters)
                - metadata: dict with:
                    - 'timestamp': float - Frame timestamp (milliseconds)
                    - 'frame_number': int - Frame sequence number
                    - 'depth_scale': float - Depth scale factor

        Raises:
            RuntimeError: If unable to acquire valid RGBD frame
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("無法獲取有效的 RGBD 幀")

        # Apply temporal filter if enabled
        if self.temporal_filter:
            depth_frame = self.temporal_filter.process(depth_frame)

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Package metadata
        metadata = {
            'timestamp': frames.get_timestamp(),
            'frame_number': frames.get_frame_number(),
            'depth_scale': self.depth_scale
        }

        return color_image, depth_image, metadata

    # ================================================================
    # Compatibility Interface (for detection_worker.py)
    # ================================================================

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get RGB and depth frame in detection worker compatible format.

        This is a compatibility wrapper that converts BGR to RGB and
        discards metadata for simpler detection worker integration.

        Returns:
            Tuple containing:
                - rgb_image: np.ndarray (H, W, 3) RGB uint8
                - depth_image: np.ndarray (H, W) uint16 (millimeters)

        Raises:
            RuntimeError: If unable to acquire valid frame

        Note:
            This method calls get_rgbd_frame() internally and converts
            color space from BGR (RealSense native) to RGB (standard).
        """
        # Get RGBD frame using original interface
        color_bgr, depth, _ = self.get_rgbd_frame()

        # Convert BGR to RGB for compatibility
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        return rgb, depth

    def get_intrinsics(self) -> Dict:
        """
        Get camera intrinsics (compatibility wrapper).

        Returns:
            dict: Camera intrinsics (same as get_camera_intrinsics())

        Note:
            This is an alias for get_camera_intrinsics() to match
            the detection worker's expected interface.
        """
        return self.get_camera_intrinsics()

    # ================================================================
    # Lifecycle Management
    # ================================================================

    def stop(self):
        """Stop RealSense pipeline and release resources."""
        print("正在停止 RealSense 相機...")
        self.pipeline.stop()
        print("RealSense 相機已停止")

    def __enter__(self):
        """Context manager entry (supports 'with' statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (automatic cleanup)."""
        self.stop()


# ===================================================================
# Standalone Test
# ===================================================================

if __name__ == "__main__":
    """
    Test RealSense service with both interfaces.

    Press 'q' to quit, 's' to save screenshot.
    """
    import cv2

    # Test with context manager
    with RealSenseService(width=1280, height=720, fps=30) as camera:
        print("\n=== 相機內參 ===")
        intrinsics = camera.get_intrinsics()
        for key, value in intrinsics.items():
            print(f"{key}: {value}")

        print("\n按 'q' 退出，按 's' 截圖")
        print("測試兩種介面: get_rgbd_frame() 和 get_frame()\n")

        frame_count = 0
        try:
            while True:
                # Test compatibility interface (RGB)
                rgb, depth = camera.get_frame()

                # Depth visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Convert RGB back to BGR for display
                display_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Side-by-side display
                images = np.hstack((display_image, depth_colormap))
                cv2.imshow('RealSense Service - RGB | Depth', images)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'rgb_{frame_count}.png', display_image)
                    cv2.imwrite(f'depth_{frame_count}.png', depth)
                    print(f"已儲存 rgb_{frame_count}.png 和 depth_{frame_count}.png")
                    frame_count += 1

        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，正在退出...")

        finally:
            cv2.destroyAllWindows()

    print("測試完成")
