"""Batch processing worker for SAM automatic annotation."""

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QThread, Signal

from ..services.sam_service import SAMService
from ..config import AppConfig
from ..utils import load_image


class BatchProcessWorker(QThread):
    """Worker thread for batch processing images with SAM."""

    # Signals
    progress_updated = Signal(int, int, str)  # current, total, image_name
    image_processed = Signal(int, list, bool, str)  # index, masks, success, image_name
    batch_completed = Signal(dict)  # stats: {success, failed, total, failed_images}
    error_occurred = Signal(int, str, str)  # index, image_name, error_msg

    def __init__(
        self,
        image_paths: List[Path],
        roi: Dict[str, int],
        sam_service: SAMService,
        config: AppConfig
    ):
        """Initialize batch processing worker.

        Args:
            image_paths: List of image paths to process
            roi: ROI dictionary {x, y, width, height}
            sam_service: SAM service instance
            config: Application configuration
        """
        super().__init__()
        self.image_paths = image_paths
        self.roi = roi
        self.sam_service = sam_service
        self.config = config
        self.cancelled = False

    def run(self):
        """Execute batch processing."""
        stats = {
            'success': 0,
            'failed': 0,
            'total': len(self.image_paths),
            'failed_images': []
        }

        for i, img_path in enumerate(self.image_paths):
            if self.cancelled:
                break

            # Update progress
            self.progress_updated.emit(i + 1, stats['total'], img_path.name)

            try:
                # Load image
                _, image_rgb = load_image(img_path)

                # Process with ROI
                masks = self.sam_service.process_image_with_roi(image_rgb, self.roi)

                # Convert to project format
                formatted_masks = self._format_masks(masks)

                # Send success signal
                self.image_processed.emit(i, formatted_masks, True, img_path.name)
                stats['success'] += 1

            except Exception as e:
                error_msg = str(e)

                # Handle specific errors
                if 'out of memory' in error_msg.lower():
                    error_msg = "CUDA 記憶體不足"
                elif 'No masks found' in error_msg:
                    error_msg = "ROI 區域內未找到物體"

                # Record failure
                stats['failed_images'].append({
                    'name': img_path.name,
                    'error': error_msg
                })

                # Send error signal
                self.error_occurred.emit(i, img_path.name, error_msg)
                self.image_processed.emit(i, [], False, img_path.name)
                stats['failed'] += 1

        # Batch completed
        self.batch_completed.emit(stats)

    def cancel(self):
        """Cancel batch processing."""
        self.cancelled = True

    def _format_masks(self, masks: List[Dict]) -> List[Dict]:
        """Format SAM masks to project format.

        Args:
            masks: List of SAM mask dictionaries

        Returns:
            List of formatted mask dictionaries
        """
        formatted = []

        for mask_data in masks:
            formatted.append({
                'mask': mask_data['mask'],
                'class_name': 'object',  # Default class name
                'class_id': 0,
                'auto_generated': True,
                'confidence': mask_data.get('stability_score', 1.0)
            })

        return formatted
