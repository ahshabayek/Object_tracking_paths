"""BEV (Bird's Eye View) Perception Pipeline.

This module provides 3D object detection from multi-camera inputs using
state-of-the-art BEV (Bird's Eye View) perception models.

Supported Models:
    - SparseBEV: Current SOTA (67.5% NDS on nuScenes), real-time capable
    - BEVFormer: Classic BEV transformer baseline
    - Sparse4D: Unified detection + tracking
    - StreamPETR: Streaming perception with temporal modeling

Why BEV Perception?
    Unlike 2D detection which outputs pixel coordinates, BEV perception
    outputs 3D positions in real-world meters. This is essential for:
    - Autonomous driving path planning
    - Collision avoidance (need exact distances)
    - Multi-camera fusion (360° awareness)

Architecture Comparison:
    | Model      | NDS (nuScenes) | FPS  | Multi-Camera | Temporal |
    |------------|----------------|------|--------------|----------|
    | SparseBEV  | 67.5%          | 23.5 | Yes          | Yes      |
    | BEVFormer  | 56.9%          | 5-10 | Yes          | Yes      |
    | Sparse4D   | 65.0%          | 20+  | Yes          | Yes      |
    | StreamPETR | 62.0%          | 30+  | Yes          | Yes      |

Input Requirements:
    - Multi-camera images (typically 6 cameras for 360° coverage)
    - Camera intrinsic matrices (focal length, principal point)
    - Camera extrinsic matrices (position, orientation relative to ego)
    - Ego-motion (for temporal fusion)
"""

from .nodes import (
    BEVConfig,
    # Factory
    BEVPerceptionFactory,
    BEVResult,
    # Dataclasses
    CameraConfig,
    Detection3D,
    compute_bev_metrics,
    extract_3d_detections,
    # Node functions
    load_bev_model,
    run_bev_inference,
)
from .sparse4d_v3 import (
    Anchor4D,
    Sparse4DConfig,
    Sparse4DHead,
    Sparse4DModel,
    Sparse4DResult,
    Sparse4DTracker,
    create_sparse4d_tracker,
    load_sparse4d_model,
    run_sparse4d_tracking,
)

__all__ = [
    # Core BEV
    "CameraConfig",
    "BEVConfig",
    "Detection3D",
    "BEVResult",
    "BEVPerceptionFactory",
    "load_bev_model",
    "run_bev_inference",
    "extract_3d_detections",
    "compute_bev_metrics",
    # Sparse4D v3
    "Anchor4D",
    "Sparse4DConfig",
    "Sparse4DResult",
    "Sparse4DTracker",
    "Sparse4DHead",
    "Sparse4DModel",
    "create_sparse4d_tracker",
    "run_sparse4d_tracking",
    "load_sparse4d_model",
]
