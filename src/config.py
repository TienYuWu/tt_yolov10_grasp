"""Configuration module for Smart Label application."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 支援的圖片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# SAM 模型資訊
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# 預設的本體論（可以在 UI 中修改）
DEFAULT_ONTOLOGY = {
    "object": "object",
}


@dataclass
class AppConfig:
    """Application configuration."""

    # 圖片資料夾
    image_dir: Optional[Path] = None

    # 輸出資料夾
    output_dir: Optional[Path] = None
    output_dir_provided: bool = False

    # SAM 模型設定
    model_type: str = "vit_b"
    checkpoint_path: Optional[Path] = None
    device: str = "cuda"  # 或 "cpu"

    # 本體論設定
    ontology: Dict[str, str] = field(default_factory=lambda: DEFAULT_ONTOLOGY.copy())

    # UI 設定
    window_width: int = 1400
    window_height: int = 800
    point_radius: int = 5

    # ROI (Region of Interest) 設定
    roi: Optional[Dict[str, int]] = None

    # 資料夾結構
    masks_subdir: str = "masks"
    images_subdir: str = "images"
    metadata_subdir: str = "metadata"
    labels_subdir: str = "labels"
    bbox_labels_subdir: str = "labels_bbox"
    obb_labels_subdir: str = "labels_obb"
    detections_subdir: str = "detections"

    def get_output_dir(self) -> Optional[Path]:
        """Get output directory, auto-generated if not provided.

        Returns:
            Output directory path, or None if image_dir is not set
        """
        # If explicitly provided, use it
        if self.output_dir_provided and self.output_dir:
            return self.output_dir

        # Otherwise, generate based on image_dir
        if self.image_dir:
            # Create "{image_dir_name}_annotated" folder at same level
            parent_dir = self.image_dir.parent
            folder_name = self.image_dir.name
            return parent_dir / f"{folder_name}_annotated"

        return None

    def get_masks_dir(self) -> Path:
        """Get masks output directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.masks_subdir if output_dir else Path()

    def get_images_dir(self) -> Path:
        """Get images output directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.images_subdir if output_dir else Path()

    def get_metadata_dir(self) -> Path:
        """Get metadata output directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.metadata_subdir if output_dir else Path()

    def get_labels_dir(self) -> Path:
        """Get labels output directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.labels_subdir if output_dir else Path()

    def get_bbox_labels_dir(self) -> Path:
        """Get axis-aligned bounding box labels directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.bbox_labels_subdir if output_dir else Path()

    def get_obb_labels_dir(self) -> Path:
        """Get oriented bounding box labels directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.obb_labels_subdir if output_dir else Path()

    def get_detections_dir(self) -> Path:
        """Get detections output directory."""
        output_dir = self.get_output_dir()
        return output_dir / self.detections_subdir if output_dir else Path()

    def get_detections_json_dir(self) -> Path:
        """Get detections JSON output directory."""
        return self.get_detections_dir() / "json"

    def get_detections_txt_dir(self) -> Path:
        """Get detections TXT output directory."""
        return self.get_detections_dir() / "txt"

    def get_detections_images_dir(self) -> Path:
        """Get detections images output directory."""
        return self.get_detections_dir() / "images"

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        output_dir = self.get_output_dir()
        if not output_dir:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self.get_masks_dir().mkdir(parents=True, exist_ok=True)
        self.get_images_dir().mkdir(parents=True, exist_ok=True)
        self.get_metadata_dir().mkdir(parents=True, exist_ok=True)
        self.get_labels_dir().mkdir(parents=True, exist_ok=True)
        self.get_bbox_labels_dir().mkdir(parents=True, exist_ok=True)
        self.get_obb_labels_dir().mkdir(parents=True, exist_ok=True)
        self.get_detections_dir().mkdir(parents=True, exist_ok=True)
        self.get_detections_json_dir().mkdir(parents=True, exist_ok=True)
        self.get_detections_txt_dir().mkdir(parents=True, exist_ok=True)
        self.get_detections_images_dir().mkdir(parents=True, exist_ok=True)

    def save_roi_config(self, project_dir: Path) -> None:
        """Save ROI configuration to project directory.

        Args:
            project_dir: Project directory path
        """
        if not self.roi:
            return

        config_dir = project_dir / ".smart_label"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / "roi_config.json"

        config_data = {
            "roi": self.roi,
            "created_at": datetime.now().isoformat()
        }

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save ROI config: {e}")

    def load_roi_config(self, project_dir: Path) -> Optional[Dict[str, int]]:
        """Load ROI configuration from project directory.

        Args:
            project_dir: Project directory path

        Returns:
            ROI dict or None if not found
        """
        config_path = project_dir / ".smart_label" / "roi_config.json"

        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                return config_data.get('roi')
        except Exception as e:
            print(f"Warning: Failed to load ROI config: {e}")
            return None
