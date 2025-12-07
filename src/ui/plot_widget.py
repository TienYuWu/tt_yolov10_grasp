"""Widget for plotting training curves."""

from typing import Dict, List

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotWidget(QWidget):
    """Widget for displaying training curves using matplotlib."""

    def __init__(self, title: str = "Training Metrics", parent=None):
        """Initialize plot widget.

        Args:
            title: Plot title
            parent: Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.data: Dict[str, List[float]] = {}
        self.x_data: List[int] = []

        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Configure plot style
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Epoch')
        self.ax.grid(True, alpha=0.3)

        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def add_data_point(self, epoch: int, metrics: Dict[str, float]):
        """Add a data point to the plot.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name to value
        """
        self.x_data.append(epoch)

        for key, value in metrics.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        self.update_plot()

    def update_plot(self):
        """Update the plot with current data."""
        self.ax.clear()

        # Plot each metric
        for key, values in self.data.items():
            if values:  # Only plot if we have data
                self.ax.plot(self.x_data, values, marker='o', label=key, linewidth=2)

        # Configure plot
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Epoch')
        self.ax.grid(True, alpha=0.3)

        # Add legend if we have multiple metrics
        if len(self.data) > 0:
            self.ax.legend(loc='best')

        # Adjust layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear all data."""
        self.data.clear()
        self.x_data.clear()
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Epoch')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def set_ylabel(self, label: str):
        """Set Y-axis label.

        Args:
            label: Y-axis label
        """
        self.ax.set_ylabel(label)
        self.canvas.draw()

    def save_figure(self, filepath: str):
        """Save the current figure to file.

        Args:
            filepath: Output file path
        """
        self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
