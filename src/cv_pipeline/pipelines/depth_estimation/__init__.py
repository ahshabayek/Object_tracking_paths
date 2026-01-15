"""Depth Estimation Pipeline.

This module provides monocular depth estimation for converting 2D detections
to pseudo-3D using state-of-the-art depth models.

Supported Models:
    - ZoeDepth: Zero-shot depth estimation (best accuracy)
    - Metric3D: Metric depth with camera intrinsics
    - DepthAnything: Fast depth estimation
    - MiDaS: Classic depth estimation baseline

Why Depth Estimation?
    With a single camera, we cannot directly measure distance. Depth estimation
    models predict per-pixel depth from a single image, enabling:
    - Pseudo-3D object localization
    - Distance estimation to detected objects
    - Bridge between 2D detection and 3D perception

Accuracy Comparison:
    | Model         | Abs Rel | RMSE  | Speed  |
    |---------------|---------|-------|--------|
    | ZoeDepth      | 0.075   | 2.51  | Medium |
    | Metric3D      | 0.083   | 2.68  | Medium |
    | DepthAnything | 0.099   | 3.02  | Fast   |
    | MiDaS         | 0.110   | 3.45  | Fast   |
"""

from .nodes import (
    DepthConfig,
    # Factory
    DepthEstimatorFactory,
    # Dataclasses
    DepthResult,
    compute_depth_metrics,
    depth_to_pointcloud,
    estimate_depth,
    lift_detections_to_3d,
    # Node functions
    load_depth_model,
)

__all__ = [
    "DepthResult",
    "DepthConfig",
    "DepthEstimatorFactory",
    "load_depth_model",
    "estimate_depth",
    "depth_to_pointcloud",
    "lift_detections_to_3d",
    "compute_depth_metrics",
]
