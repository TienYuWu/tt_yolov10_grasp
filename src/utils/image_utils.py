"""Utility functions for Smart Label application."""

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PySide6.QtGui import QImage, QPixmap
from segment_anything import SamPredictor, sam_model_registry

from ..config import SAM_MODEL_URLS


def detect_device() -> str:
    """Detect the best available device (cuda/cpu).

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  CUDA 不可用，使用 CPU")
    return device


def download_checkpoint(model_type: str, save_dir: Path) -> Path:
    """Download SAM checkpoint if not exists.

    Args:
        model_type: Model type (vit_h, vit_l, vit_b)
        save_dir: Directory to save checkpoint

    Returns:
        Path to checkpoint file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"sam_{model_type}.pth"

    if checkpoint_path.exists():
        print(f"✅ 模型已存在: {checkpoint_path}")
        return checkpoint_path

    url = SAM_MODEL_URLS.get(model_type)
    if not url:
        raise ValueError(f"不支援的模型類型: {model_type}")

    print(f"⏬ 下載模型 {model_type} 從 {url}...")
    urllib.request.urlretrieve(url, checkpoint_path)
    print(f"✅ 模型已下載: {checkpoint_path}")

    return checkpoint_path


def build_sam_predictor(
    model_type: str,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None
) -> SamPredictor:
    """Build SAM predictor.

    Args:
        model_type: Model type (vit_h, vit_l, vit_b)
        checkpoint_path: Path to checkpoint, downloads if None
        device: Device to use, auto-detects if None

    Returns:
        SAM predictor instance
    """
    if device is None:
        device = detect_device()

    if checkpoint_path is None:
        checkpoint_path = download_checkpoint(model_type, Path.home() / ".cache" / "sam")
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型檔案不存在: {checkpoint_path}")

    print(f"🔨 建立 SAM 模型 ({model_type}) 在 {device}...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("✅ SAM 模型已就緒")

    return predictor


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert numpy array to QPixmap.

    Args:
        image: Image array in RGB or RGBA format (NOT BGR).
               All input images must be converted to RGB before calling this function.
               - For OpenCV images loaded with cv2.imread(): use cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               - For RealSense frames: use realsense_service.get_frame() which returns RGB

    Returns:
        QPixmap instance
    """
    if image.ndim == 2:
        # Grayscale
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif image.shape[2] == 3:
        # RGB (no conversion needed since our images are already RGB)
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    elif image.shape[2] == 4:
        # RGBA (no conversion needed)
        height, width = image.shape[:2]
        bytes_per_line = 4 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    else:
        raise ValueError(f"不支援的圖片通道數: {image.shape[2]}")

    return QPixmap.fromImage(q_image)


def derive_default_output_dir(image_dir: Path) -> Path:
    """Derive default output directory from image directory.

    Args:
        image_dir: Input image directory

    Returns:
        Default output directory path
    """
    parent = image_dir.parent
    folder_name = image_dir.name
    return parent / f"{folder_name}_smart_labels"


def load_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load image in both BGR and RGB formats.

    Args:
        path: Path to image file

    Returns:
        Tuple of (BGR image, RGB image)
    """
    # Try direct cv2.imread first (fast path)
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    # cv2.imread may return None for various reasons on Windows (e.g. non-ASCII path,
    # unsupported codec, corrupted file). Fall back to reading bytes and using
    # cv2.imdecode which is more robust with Unicode paths.
    if bgr is None:
        try:
            data = path.read_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            bgr = None

    if bgr is None:
        raise ValueError(f"無法讀取圖片: {path}")

    # Normalize to common outputs:
    # - return `bgr` as loaded (can be HxW (gray), HxWx3 (BGR) or HxWx4 (BGRA))
    # - return `rgb` as HxWx3 (RGB) which is expected by downstream SAM predictor
    if bgr.ndim == 2:
        # Grayscale -> convert to 3-channel BGR for consistency
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    if bgr.shape[2] == 4:
        # BGRA -> produce rgb without alpha for predictor
        bgr_no_alpha = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(bgr_no_alpha, cv2.COLOR_BGR2RGB)
    else:
        # 3-channel BGR
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return bgr, rgb


def create_colored_mask(mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: int = 120) -> np.ndarray:
    """Create colored overlay from binary mask.

    Args:
        mask: Binary mask array
        color: RGB color tuple
        alpha: Transparency (0-255)

    Returns:
        RGBA overlay array
    """
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = (mask.astype(np.uint8) * alpha)
    return overlay


def save_mask_as_png(mask: np.ndarray, output_path: Path) -> None:
    """Save binary mask as PNG file.

    Args:
        mask: Binary mask array
        output_path: Output file path
    """
    mask_uint8 = (mask * 255).astype(np.uint8)

    # cv2.imwrite may fail on Windows for non-ASCII paths in some builds.
    # Use imencode + write as a more robust approach that supports Unicode paths.
    try:
        ext = '.png'
        success, encoded = cv2.imencode(ext, mask_uint8)
        if not success:
            raise IOError("cv2.imencode 無法編碼影像")
        # Write bytes using Python file I/O which handles Unicode paths well
        with output_path.open('wb') as f:
            f.write(encoded.tobytes())
    except Exception:
        # Fallback to cv2.imwrite as a last resort
        cv2.imwrite(str(output_path), mask_uint8)


def masks_to_yolo_format(
    masks: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int
) -> str:
    """Convert masks to YOLO segmentation format.

    Args:
        masks: Binary mask array
        class_id: Class ID for the object
        image_width: Image width
        image_height: Image height

    Returns:
        YOLO format string
    """
    contours, _ = cv2.findContours(
        masks.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # 歸一化座標
        normalized = []
        for point in contour:
            x, y = point[0]
            norm_x = x / image_width
            norm_y = y / image_height
            normalized.extend([norm_x, norm_y])

        # YOLO 格式: class_id x1 y1 x2 y2 ...
        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
        lines.append(line)

    return "\n".join(lines)


def compute_mask_geometry(mask: np.ndarray) -> Optional[Dict[str, object]]:
    """Compute bounding box and oriented bounding box geometry from a mask."""
    if mask is None:
        return None

    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() == 1:
        mask_uint8 = mask_uint8 * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.size == 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rect)

    geometry = {
        "bbox": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
        },
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
        "obb": {
            "points": [[float(pt[0]), float(pt[1])] for pt in box_points],
            "center": [float(rect[0][0]), float(rect[0][1])],
            "size": [float(rect[1][0]), float(rect[1][1])],
            "angle": float(rect[2]),
        },
    }

    return geometry


def _extract_xywh(bbox: object) -> Optional[Tuple[float, float, float, float]]:
    """Internal helper to normalize bbox representations to xywh."""
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        keys = {k.lower(): v for k, v in bbox.items()}
        x = keys.get("x") if "x" in keys else keys.get("left")
        y = keys.get("y") if "y" in keys else keys.get("top")
        w = keys.get("width") if "width" in keys else keys.get("w")
        h = keys.get("height") if "height" in keys else keys.get("h")
        if x is None or y is None or w is None or h is None:
            return None
        return float(x), float(y), float(w), float(h)

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = bbox
        return float(x), float(y), float(w), float(h)

    return None


def bbox_to_yolo_line(
    bbox: object,
    class_id: int,
    image_width: int,
    image_height: int
) -> Optional[str]:
    """Convert a bounding box to YOLO detection format."""
    xywh = _extract_xywh(bbox)
    if xywh is None:
        return None

    x, y, w, h = xywh
    if image_width <= 0 or image_height <= 0 or w <= 0 or h <= 0:
        return None

    center_x = (x + w / 2.0) / image_width
    center_y = (y + h / 2.0) / image_height
    norm_w = w / image_width
    norm_h = h / image_height

    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def obb_to_yolo_line(
    points: Optional[List[List[float]]],
    class_id: int,
    image_width: int,
    image_height: int
) -> Optional[str]:
    """Convert oriented bounding box points to a YOLO-style polygon line."""
    if not points or len(points) != 4:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    normalized: List[float] = []
    for point in points:
        if len(point) != 2:
            return None
        x, y = point
        normalized.append(x / image_width)
        normalized.append(y / image_height)

    coords = " ".join(f"{value:.6f}" for value in normalized)
    return f"{class_id} {coords}"


def load_saved_annotations(image_path: Path, output_dir: Path) -> Optional[List[Dict]]:
    """Load saved annotations for an image.

    Args:
        image_path: Path to the image file
        output_dir: Output directory containing saved annotations

    Returns:
        List of mask dictionaries, or None if no annotations found
    """
    if not output_dir:
        return None

    image_name = image_path.stem

    # Check if metadata file exists
    metadata_path = output_dir / "metadata" / f"{image_name}.json"
    if not metadata_path.exists():
        return None

    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        masks = []
        masks_dir = output_dir / "masks"

        # Load each mask
        for instance in metadata.get('instances', []):
            mask_file = instance.get('mask_file')
            if not mask_file:
                continue

            mask_path = masks_dir / mask_file
            if not mask_path.exists():
                continue

            # Load mask PNG (grayscale)
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue

            # Convert to boolean mask
            mask = mask_img > 0

            # Create mask dictionary
            mask_data = {
                'mask': mask,
                'class_name': instance.get('class_name', '未分類'),
                'class_id': instance.get('class_id', 0),
                'auto_generated': False,  # Loaded from file, treated as manual
            }

            # Add geometry info if available
            if 'bbox' in instance:
                mask_data['bbox'] = instance['bbox']
            if 'obb' in instance:
                mask_data['obb'] = instance['obb']

            masks.append(mask_data)

        return masks if masks else None

    except Exception as e:
        print(f"Warning: Failed to load annotations for {image_name}: {e}")
        return None
