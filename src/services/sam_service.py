"""SAM service for batch processing and automatic mask generation."""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator
else:
    try:
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        SamPredictor = None
        SamAutomaticMaskGenerator = None
        sam_model_registry = None


class SAMService:
    """Service for SAM model operations."""

    def __init__(self, predictor: Optional["SamPredictor"] = None):
        """Initialize SAM service.

        Args:
            predictor: SAM predictor instance (can be None if SAM not available)
        """
        self.predictor = predictor
        self._mask_generator: Optional["SamAutomaticMaskGenerator"] = None

    def is_available(self) -> bool:
        """Check if SAM is available.

        Returns:
            True if SAM predictor is loaded
        """
        return self.predictor is not None

    def _get_mask_generator(self) -> Optional["SamAutomaticMaskGenerator"]:
        """Get or create automatic mask generator.

        Returns:
            SamAutomaticMaskGenerator instance or None
        """
        if not self.is_available():
            return None

        if self._mask_generator is None:
            if SamAutomaticMaskGenerator is None:
                return None

            # Use predictor's model for mask generation
            self._mask_generator = SamAutomaticMaskGenerator(
                model=self.predictor.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

        return self._mask_generator

    def process_image_with_roi(
        self,
        image_rgb: np.ndarray,
        roi: Dict[str, int]
    ) -> List[Dict[str, np.ndarray]]:
        """Process image within ROI using automatic mask generation.

        Args:
            image_rgb: RGB image array
            roi: ROI dictionary {x, y, width, height}

        Returns:
            List of mask dictionaries, each containing 'mask' array

        Raises:
            RuntimeError: If SAM is not available or processing fails
        """
        if not self.is_available():
            raise RuntimeError("SAM model not available")

        mask_generator = self._get_mask_generator()
        if mask_generator is None:
            raise RuntimeError("Failed to create mask generator")

        # Extract ROI region
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

        # Ensure ROI is within image bounds
        img_h, img_w = image_rgb.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return []

        # Crop ROI region
        roi_image = image_rgb[y:y+h, x:x+w].copy()

        # Generate masks on ROI
        try:
            masks = mask_generator.generate(roi_image)
        except Exception as e:
            raise RuntimeError(f"SAM mask generation failed: {e}")

        # Convert masks to full image coordinates
        result_masks = []
        for mask_data in masks:
            # Create full-size mask
            full_mask = np.zeros((img_h, img_w), dtype=bool)

            # Place ROI mask in full image
            roi_mask = mask_data['segmentation']
            full_mask[y:y+h, x:x+w] = roi_mask

            result_masks.append({
                'mask': full_mask,
                'area': int(mask_data.get('area', 0)),
                'bbox': mask_data.get('bbox'),  # Original bbox in ROI coordinates
                'predicted_iou': float(mask_data.get('predicted_iou', 0)),
                'stability_score': float(mask_data.get('stability_score', 0))
            })

        # Sort by area (largest first)
        result_masks.sort(key=lambda x: x['area'], reverse=True)

        return result_masks

    def batch_process(
        self,
        image_paths: List[Path],
        roi: Dict[str, int],
        progress_callback=None
    ) -> List[Dict]:
        """Batch process multiple images with ROI.

        Args:
            image_paths: List of image file paths
            roi: ROI dictionary {x, y, width, height}
            progress_callback: Optional callback(current, total, image_name)

        Returns:
            List of results, each containing {path, masks, success, error}
        """
        if not self.is_available():
            raise RuntimeError("SAM model not available")

        from ..utils import load_image

        results = []
        total = len(image_paths)

        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, total, img_path.name)

            result = {
                'path': img_path,
                'masks': [],
                'success': False,
                'error': None
            }

            try:
                # Load image
                _, image_rgb = load_image(img_path)

                # Process with ROI
                masks = self.process_image_with_roi(image_rgb, roi)

                result['masks'] = masks
                result['success'] = True

            except Exception as e:
                result['error'] = str(e)

            results.append(result)

        return results
