"""Utility modules for Smart Label application."""

# Lazy imports to avoid cv2/torch dependency errors at import time
# Functions will be imported when first accessed

import sys

# Module cache to prevent infinite recursion
_module_cache = {}

def __getattr__(name):
    """Lazy import utility functions for backward compatibility."""
    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # List of functions from image_utils
    image_utils_funcs = [
        'numpy_to_qpixmap',
        'detect_device',
        'build_sam_predictor',
        'download_checkpoint',
        'load_image',
        'create_colored_mask',
        'save_mask_as_png',
        'masks_to_yolo_format',
        'compute_mask_geometry',
        'obb_to_yolo_line',
        'bbox_to_yolo_line',
        'derive_default_output_dir',
        'load_saved_annotations',
    ]

    if name in image_utils_funcs:
        # Import module using importlib to avoid recursion
        import importlib
        image_utils = importlib.import_module('.image_utils', package='src.utils')
        attr = getattr(image_utils, name)
        _module_cache[name] = attr
        return attr

    # Submodules
    if name == 'pose_utils':
        import importlib
        module = importlib.import_module('.pose_utils', package='src.utils')
        _module_cache[name] = module
        return module

    if name == 'visualization_utils':
        import importlib
        module = importlib.import_module('.visualization_utils', package='src.utils')
        _module_cache[name] = module
        return module

    if name == 'image_utils':
        import importlib
        module = importlib.import_module('.image_utils', package='src.utils')
        _module_cache[name] = module
        return module

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Modules
    'pose_utils',
    'visualization_utils',
    'image_utils',
    # Functions (backward compatibility)
    'numpy_to_qpixmap',
    'detect_device',
    'build_sam_predictor',
    'download_checkpoint',
    'load_image',
    'create_colored_mask',
    'save_mask_as_png',
    'masks_to_yolo_format',
    'compute_mask_geometry',
    'obb_to_yolo_line',
    'bbox_to_yolo_line',
    'derive_default_output_dir',
    'load_saved_annotations',
]
