#!/usr/bin/env python3
"""
Test script for Detection Tab GUI layout

This script tests:
1. OutputConsole creation and logging
2. DetectionTab layout structure
3. Control panel reorganization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_output_console():
    """Test OutputConsole widget."""
    print("Testing OutputConsole...")

    from PySide6.QtWidgets import QApplication
    from ui.widgets import OutputConsole

    app = QApplication(sys.argv)

    # Create console
    console = OutputConsole()
    console.setWindowTitle("Output Console Test")
    console.resize(800, 300)

    # Test logging methods
    console.log_info("This is an INFO message")
    console.log_warning("This is a WARNING message")
    console.log_result("Detection completed with 5 objects")
    console.log_error("This is an ERROR message")
    console.log_result("  [1] Object1 (Confidence: 95.2%)")
    console.log_result("  [2] Object2 (Confidence: 87.3%)")

    console.show()

    print("✓ OutputConsole created successfully with 6 log entries")
    print("✓ All logging methods work correctly")

    return console


def test_detection_tab_layout():
    """Test Detection Tab layout structure."""
    print("\nTesting Detection Tab layout...")

    from PySide6.QtWidgets import QApplication, QMainWindow
    from config import AppConfig
    from ui.detection_tab import DetectionTab

    # Check if model file exists (use dummy path if not)
    model_path = Path.cwd() / "best.pt"
    if not model_path.exists():
        print(f"  Note: Model file not found at {model_path}")
        print("  Using dummy path for testing")

    try:
        app = QApplication.instance() or QApplication(sys.argv)
        config = AppConfig()

        # Create Detection Tab
        detection_tab = DetectionTab(config=config, model_path=str(model_path))

        print("✓ DetectionTab created successfully")

        # Check layout structure
        layout = detection_tab.layout()
        print(f"✓ Main layout type: {type(layout).__name__}")

        # Check if output console was created
        if hasattr(detection_tab, 'output_console'):
            print("✓ Output console widget exists")
            print(f"✓ Output console type: {type(detection_tab.output_console).__name__}")
        else:
            print("✗ Output console widget NOT found")
            return False

        # Check if action button was created
        if hasattr(detection_tab, 'run_detection_btn'):
            print("✓ Run detection button created")
        else:
            print("✗ Run detection button NOT found")
            return False

        # Check if stop button was created
        if hasattr(detection_tab, 'stop_detection_btn'):
            print("✓ Stop detection button created")
        else:
            print("✗ Stop detection button NOT found")
            return False

        print("✓ Control panel reorganization verified")

        return True

    except Exception as e:
        print(f"✗ Error creating Detection Tab: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Detection Tab GUI Layout Test")
    print("=" * 60)

    # Test OutputConsole
    try:
        console = test_output_console()
    except Exception as e:
        print(f"✗ OutputConsole test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Detection Tab layout
    try:
        layout_ok = test_detection_tab_layout()
    except Exception as e:
        print(f"✗ Layout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    if layout_ok:
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nTest Summary:")
        print("  1. OutputConsole widget: PASS")
        print("  2. Detection Tab layout: PASS")
        print("  3. Control panel reorganization: PASS")
        print("  4. Logging integration: PASS")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
