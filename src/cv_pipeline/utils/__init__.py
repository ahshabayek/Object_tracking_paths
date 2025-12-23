"""Utility modules for CV Pipeline.

This package contains utility functions for:
- Visualization (drawing detections, tracks, lanes, paths)
- Metrics computation (mAP, MOTA, IDF1, F1)
- Camera Motion Compensation (CMC)
- MLFlow experiment tracking
"""

from cv_pipeline.utils.cmc import (
    ECCCMC,
    CameraMotionCompensator,
    FeatureCMC,
    OpticalFlowCMC,
    SparseOptFlowCMC,
)
from cv_pipeline.utils.metrics import (
    MetricsAccumulator,
    TrackingFrame,
    compute_ap,
    compute_detection_metrics,
    compute_iou,
    compute_lane_metrics,
    compute_tracking_metrics,
)
from cv_pipeline.utils.mlflow_utils import (
    ExperimentTracker,
    is_mlflow_available,
    log_detection_metrics,
    log_dict_as_artifact,
    log_lane_detection_metrics,
    log_metrics_safe,
    log_params_safe,
    log_pipeline_run,
    log_tracking_metrics,
    mlflow_run,
    mlflow_track,
)
from cv_pipeline.utils.visualization import (
    Visualizer,
    draw_detections,
    draw_lanes,
    draw_path,
    draw_scene,
    draw_tracks,
)
from cv_pipeline.utils.weights import (
    WeightsManager,
    get_checkpoint_info,
    get_model_size,
    get_pretrained_weights,
    load_weights,
    save_checkpoint,
)

__all__ = [
    # Visualization
    "draw_detections",
    "draw_tracks",
    "draw_lanes",
    "draw_path",
    "draw_scene",
    "Visualizer",
    # Camera Motion Compensation
    "CameraMotionCompensator",
    "ECCCMC",
    "FeatureCMC",
    "OpticalFlowCMC",
    "SparseOptFlowCMC",
    # Metrics
    "compute_detection_metrics",
    "compute_tracking_metrics",
    "compute_lane_metrics",
    "compute_iou",
    "compute_ap",
    "TrackingFrame",
    "MetricsAccumulator",
    # MLFlow
    "is_mlflow_available",
    "mlflow_run",
    "mlflow_track",
    "log_params_safe",
    "log_metrics_safe",
    "log_dict_as_artifact",
    "log_detection_metrics",
    "log_tracking_metrics",
    "log_lane_detection_metrics",
    "log_pipeline_run",
    "ExperimentTracker",
    # Weights Management
    "WeightsManager",
    "get_pretrained_weights",
    "load_weights",
    "save_checkpoint",
    "get_checkpoint_info",
    "get_model_size",
]
