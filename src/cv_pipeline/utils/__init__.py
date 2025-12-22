"""Utility modules for CV Pipeline.

This package contains utility functions for:
- Visualization (drawing detections, tracks, lanes, paths)
- Metrics computation (mAP, MOTA, IDF1, F1)
- Camera Motion Compensation (CMC)
"""

from cv_pipeline.utils.cmc import (
    ECCCMC,
    CameraMotionCompensator,
    FeatureCMC,
    OpticalFlowCMC,
    SparseOptFlowCMC,
)
from cv_pipeline.utils.visualization import (
    Visualizer,
    draw_detections,
    draw_lanes,
    draw_path,
    draw_scene,
    draw_tracks,
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
]
