"""3D Object Tracking Pipeline.

This module provides 3D multi-object tracking for autonomous driving.
Tracks objects in world coordinates (meters) using 3D detections.

Supported Trackers:
    - AB3DMOT: Baseline 3D MOT with Kalman filter (IROS 2020)
    - SimpleTrack: Improved baseline (ICCV 2021)
    - CenterPoint Tracking: CenterPoint's tracking head (CVPR 2021)
    - OC-SORT 3D: Observation-centric 3D tracking (CVPR 2023)

Example Usage:
    from cv_pipeline.pipelines.tracking_3d.nodes import (
        create_tracker,
        track_objects_3d,
        TrackingResult,
    )

    # Create tracker
    tracker = create_tracker({"tracker": "ab3dmot"})

    # Track objects
    detections = [
        {"center": [10, 2, 0], "size": [2, 4.5, 1.5], "rotation": 0.1, "confidence": 0.9}
    ]
    result = track_objects_3d(tracker, detections, {})
    print(f"Tracks: {result.num_tracks}")
"""

from .nodes import (
    AB3DMOTTracker,
    KalmanFilter3D,
    SimpleTrack3D,
    Track3D,
    Tracker3DFactory,
    TrackingResult,
    TrackState,
    compute_3d_iou,
    compute_bev_iou,
    compute_center_distance,
    compute_tracking_metrics,
    create_tracker,
    track_objects_3d,
)

__all__ = [
    # Data classes
    "TrackState",
    "Track3D",
    "TrackingResult",
    # Kalman filter
    "KalmanFilter3D",
    # Trackers
    "AB3DMOTTracker",
    "SimpleTrack3D",
    "Tracker3DFactory",
    # Utilities
    "compute_3d_iou",
    "compute_bev_iou",
    "compute_center_distance",
    # Node functions
    "create_tracker",
    "track_objects_3d",
    "compute_tracking_metrics",
]
