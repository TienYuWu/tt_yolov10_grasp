"""Smart Label - 智能標註工具

手動 SAM 微調的桌面應用程式。
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import AppConfig
from src.main_window import MainWindow
from src.utils import build_sam_predictor, detect_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Smart Label - SAM 智能標註工具"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="預設載入的圖片資料夾"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="標註輸出資料夾"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="SAM 權重檔路徑（未提供時自動下載）"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 模型類型（預設 vit_b）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="執行裝置（預設自動偵測）"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU 閾值（預設 0.5）"
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> AppConfig:
    """Create application configuration from arguments.

    Args:
        args: Command line arguments

    Returns:
        Application configuration
    """
    config = AppConfig()

    if args.image_dir:
        config.image_dir = Path(args.image_dir).expanduser().resolve()

    if args.output_dir:
        config.output_dir = Path(args.output_dir).expanduser().resolve()
        config.output_dir_provided = True

    if args.checkpoint:
        config.checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    config.model_type = args.model_type
    config.device = args.device if args.device else detect_device()

    return config


def main(argv: Optional[list] = None) -> int:
    """Main entry point for Smart Label application.

    Args:
        argv: Command line arguments (for testing)

    Returns:
        Exit code
    """
    args = parse_args() if argv is None else parse_args()
    config = create_config_from_args(args)

    print("=" * 60)
    print("Smart Label - 智能標註工具")
    print("=" * 60)
    print(f"裝置: {config.device}")
    print(f"SAM 模型: {config.model_type}")
    print("=" * 60)

    # Build SAM predictor
    predictor = None
    try:
        predictor = build_sam_predictor(
            model_type=config.model_type,
            checkpoint_path=config.checkpoint_path,
            device=config.device
        )
    except Exception as e:
        print(f"⚠️  警告: 無法載入 SAM 模型: {e}")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Smart Label")
    app.setOrganizationName("Smart Label")

    # Create and show main window
    try:
        window = MainWindow(config, predictor)
        window.resize(config.window_width, config.window_height)
        window.show()

        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "錯誤", f"應用程式啟動失敗:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
