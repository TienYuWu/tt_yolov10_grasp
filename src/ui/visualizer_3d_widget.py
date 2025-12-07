#!/usr/bin/env python3
"""
3D Pose Visualizer Widget - Open3D-based 3D visualization for detection system

This module provides a unified 3D visualization interface for pose estimation results,
wrapping Open3D with a clean API suitable for Qt integration.

Vendor-ready code for industry-academia cooperation delivery.

Features:
1. Point cloud display with color mapping
2. PCA plane mesh visualization (red)
3. Surface normal arrow visualization (blue)
4. Coordinate frame display (RGB = XYZ)
5. OBB region highlighting (green points)

Visual effect matches try_realsense.py reference implementation.
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Optional


class Visualizer3DWidget:
    """
    3D pose visualization widget using Open3D.

    Manages an Open3D visualizer window to display:
    - Point clouds (with OBB region color coding)
    - PCA plane meshes (red)
    - Surface normal arrows (blue)
    - Coordinate frames (optional, RGB = XYZ)

    Usage:
        # Create visualizer
        vis = Visualizer3DWidget(window_name="3D Pose Visualization")

        # Update visualization
        vis.update(point_cloud, poses, show_planes=True, show_normals=True)

        # Reset view
        vis.reset_view()

        # Close
        vis.close()
    """

    def __init__(self,
                 window_name: str = "3D Pose Visualization",
                 width: int = 1280,
                 height: int = 720,
                 point_size: float = 2.0):
        """
        Initialize 3D visualization widget.

        Args:
            window_name: Visualizer window title
            width: Window width in pixels (default: 1280)
            height: Window height in pixels (default: 720)
            point_size: Point cloud point size (default: 2.0)
        """
        self.window_name = window_name
        self.width = width
        self.height = height

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        # Initialize geometry objects
        self.point_cloud = o3d.geometry.PointCloud()
        self.pca_plane_meshes = []  # List of PCA plane meshes
        self.normal_arrows = []     # List of normal vector arrows
        self.coordinate_frames = [] # List of coordinate frames
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0])

        # Add initial geometries
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.world_frame)

        # Configure rendering options
        render_option = self.vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray([0, 0, 0])  # Black background

        # View control
        self.view_control = self.vis.get_view_control()
        self.initial_view_params = None

    def update(self,
               point_cloud: o3d.geometry.PointCloud,
               poses: List[Dict],
               show_planes: bool = True,
               show_normals: bool = True,
               show_coordinate_frames: bool = False):
        """
        Update visualization content.

        Args:
            point_cloud: Open3D point cloud (pre-colored, [0,0,0] points filtered)
            poses: Pose list from detection results (each pose is a dict with 'pca_result')
            show_planes: Display PCA plane meshes (default: True)
            show_normals: Display surface normal arrows (default: True)
            show_coordinate_frames: Display RGB coordinate axes (default: False)

        Note:
            The point_cloud should be pre-processed:
            - Colored (green for OBB regions, default otherwise)
            - Filtered ([0,0,0] invalid points removed)
        """
        # Remove old geometries
        for mesh in self.pca_plane_meshes:
            self.vis.remove_geometry(mesh, reset_bounding_box=False)
        for arrow in self.normal_arrows:
            self.vis.remove_geometry(arrow, reset_bounding_box=False)
        for frame in self.coordinate_frames:
            self.vis.remove_geometry(frame, reset_bounding_box=False)

        self.pca_plane_meshes.clear()
        self.normal_arrows.clear()
        self.coordinate_frames.clear()

        # Update point cloud
        self.point_cloud.points = point_cloud.points
        self.point_cloud.colors = point_cloud.colors
        self.vis.update_geometry(self.point_cloud)

        # Create visualization objects for each pose
        for pose in poses:
            pca_result = pose.get('pca_result')
            if pca_result is None:
                continue

            # 1. Create red PCA plane mesh
            if show_planes:
                plane_mesh = self._create_pca_plane_mesh(pca_result, color=[1, 0, 0])
                if plane_mesh is not None:
                    self.pca_plane_meshes.append(plane_mesh)
                    self.vis.add_geometry(plane_mesh, reset_bounding_box=False)

            # 2. Create blue surface normal arrow
            if show_normals:
                arrow = self._create_normal_arrow(pca_result, color=[0, 0, 1], scale=0.05)
                if arrow is not None:
                    self.normal_arrows.append(arrow)
                    self.vis.add_geometry(arrow, reset_bounding_box=False)

            # 3. Create RGB coordinate frame
            if show_coordinate_frames:
                center = pose.get('center')
                transform_matrix = pose.get('transform_matrix')
                if center is not None and transform_matrix is not None:
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=center)
                    # Rotate coordinate frame to align with pose
                    R = np.array(transform_matrix)[:3, :3]
                    coord_frame.rotate(R, center=center)
                    self.coordinate_frames.append(coord_frame)
                    self.vis.add_geometry(coord_frame, reset_bounding_box=False)

        # Render
        self.vis.poll_events()
        self.vis.update_renderer()

    def color_obb_points(self,
                        point_cloud: o3d.geometry.PointCloud,
                        poses: List[Dict],
                        color: List[float] = [0, 1, 0]) -> o3d.geometry.PointCloud:
        """
        Color points within OBB regions.

        Args:
            point_cloud: Original point cloud
            poses: Pose list (each containing '_linear_indices' and '_valid_mask')
            color: RGB color for OBB points [0-1] (default: green [0, 1, 0])

        Returns:
            Colored point cloud

        Note:
            Poses must contain internal keys '_linear_indices' and '_valid_mask'
            from the pose estimation process.
        """
        # Copy point cloud
        colored_pcd = o3d.geometry.PointCloud(point_cloud)
        full_colors = np.asarray(colored_pcd.colors)

        # Color points for each pose
        for pose in poses:
            if '_linear_indices' in pose and '_valid_mask' in pose:
                linear_indices = pose['_linear_indices']
                valid_mask = pose['_valid_mask']
                valid_linear_indices = linear_indices[valid_mask]
                full_colors[valid_linear_indices] = color

        colored_pcd.colors = o3d.utility.Vector3dVector(full_colors)

        return colored_pcd

    def filter_zero_points(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Filter out [0,0,0] invalid points.

        Args:
            point_cloud: Input point cloud

        Returns:
            Filtered point cloud (without [0,0,0] points)
        """
        points = np.asarray(point_cloud.points)
        valid_mask = np.any(points != 0, axis=1)
        valid_indices = np.where(valid_mask)[0]
        return point_cloud.select_by_index(valid_indices)

    def reset_view(self):
        """Reset camera view to initial state."""
        if self.initial_view_params is None:
            # Save current view as initial
            self.initial_view_params = self.view_control.convert_to_pinhole_camera_parameters()
        else:
            # Restore initial view
            self.view_control.convert_from_pinhole_camera_parameters(self.initial_view_params)

    def close(self):
        """Close visualizer window and release resources."""
        self.vis.destroy_window()

    # ===================================================================
    # Private Methods (geometry creation)
    # ===================================================================

    def _create_pca_plane_mesh(self, pca_result: Dict, color: List[float]) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Create PCA plane mesh (rectangular surface).

        Args:
            pca_result: PCA result dict with keys:
                - 'center': 3D point
                - 'axis1': First principal axis (major edge)
                - 'axis2': Second principal axis (minor edge)
                - 'extent1': Extent along axis1
                - 'extent2': Extent along axis2
                - 'normal': Surface normal
            color: Mesh color RGB [0-1] (default: red [1, 0, 0])

        Returns:
            TriangleMesh object or None if pca_result is invalid
        """
        if pca_result is None:
            return None

        center = pca_result['center']
        axis1 = pca_result['axis1']
        axis2 = pca_result['axis2']
        extent1 = pca_result['extent1']
        extent2 = pca_result['extent2']

        # Calculate 4 vertices of plane rectangle
        v0 = center - axis1 * extent1 / 2 - axis2 * extent2 / 2
        v1 = center + axis1 * extent1 / 2 - axis2 * extent2 / 2
        v2 = center + axis1 * extent1 / 2 + axis2 * extent2 / 2
        v3 = center - axis1 * extent1 / 2 + axis2 * extent2 / 2

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([v0, v1, v2, v3])

        # Define two triangles (vertex order determines normal direction)
        mesh.triangles = o3d.utility.Vector3iVector([
            [0, 1, 2],  # Triangle 1
            [0, 2, 3]   # Triangle 2
        ])

        # Set color
        mesh.paint_uniform_color(color)

        # Compute vertex normals
        mesh.compute_vertex_normals()

        return mesh

    def _create_normal_arrow(self,
                            pca_result: Dict,
                            color: List[float],
                            scale: float) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Create surface normal arrow (cylinder + cone).

        Args:
            pca_result: PCA result dict with keys:
                - 'center': 3D point
                - 'normal': Surface normal vector
            color: Arrow color RGB [0-1] (default: blue [0, 0, 1])
            scale: Arrow length in meters (default: 0.05)

        Returns:
            TriangleMesh arrow object or None if pca_result is invalid

        Note:
            Arrow is composed of:
            - Cylinder (70% of scale, 2% radius)
            - Cone tip (30% of scale, 4% radius)
        """
        if pca_result is None:
            return None

        center = pca_result['center']
        normal = pca_result['normal']

        # Arrow endpoint
        arrow_end = center + normal * scale

        # Create cylinder (arrow shaft)
        cylinder_height = scale * 0.7
        cylinder_radius = scale * 0.02
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cylinder_radius,
            height=cylinder_height
        )

        # Create cone (arrow head)
        cone_height = scale * 0.3
        cone_radius = scale * 0.04
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_radius,
            height=cone_height
        )

        # Move cone to cylinder top
        cone.translate([0, 0, cylinder_height / 2 + cone_height / 2])

        # Merge cylinder and cone
        arrow = cylinder + cone

        # Calculate rotation matrix (align Z-axis to normal)
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:  # Avoid division by zero
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

            # Rodrigues rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
            arrow.rotate(R, center=[0, 0, 0])

        # Move to correct position
        arrow.translate(center)

        # Set color
        arrow.paint_uniform_color(color)
        arrow.compute_vertex_normals()

        return arrow
