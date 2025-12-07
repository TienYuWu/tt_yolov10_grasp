"""UI components for Smart Label application."""

from .batch_dialog import BatchProgressDialog
from .annotation_tab import AnnotationTab
from .training_tab import TrainingTab
from .detection_tab import DetectionTab
from .dataset_split_dialog import DatasetSplitDialog
from .plot_widget import PlotWidget
from .eval_widget import EvalWidget
from .model_manager_widget import ModelManagerWidget
from .visualizer_3d_widget import Visualizer3DWidget
from .augmentation_dialog import AugmentationDialog
from .intrinsics_dialog import IntrinsicsDialog

__all__ = [
    'BatchProgressDialog',
    'AnnotationTab',
    'TrainingTab',
    'DetectionTab',
    'DatasetSplitDialog',
    'PlotWidget',
    'EvalWidget',
    'ModelManagerWidget',
    'Visualizer3DWidget',
    'AugmentationDialog',
    'IntrinsicsDialog',
]
