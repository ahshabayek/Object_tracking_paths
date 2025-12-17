"""Multi-Object Tracking Pipeline.

This module implements MOT using various trackers:
- BoT-SORT (Bag of Tricks for SORT)
- ByteTrack
- OC-SORT
- DeepSORT
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    initialize_tracker,
    run_tracking,
    extract_trajectories,
    compute_tracking_metrics,
    log_tracking_to_mlflow,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the multi-object tracking pipeline.
    
    Returns:
        A Kedro Pipeline object for object tracking.
    """
    return pipeline(
        [
            node(
                func=initialize_tracker,
                inputs=["params:tracking"],
                outputs="tracker",
                name="initialize_tracker",
                tags=["tracking", "init"],
            ),
            node(
                func=run_tracking,
                inputs=["tracker", "detection_results", "preprocessed_frames", "params:tracking"],
                outputs="tracked_objects",
                name="run_tracking",
                tags=["tracking", "inference"],
            ),
            node(
                func=extract_trajectories,
                inputs=["tracked_objects", "params:tracking"],
                outputs="track_trajectories",
                name="extract_trajectories",
                tags=["tracking", "trajectories"],
            ),
            node(
                func=compute_tracking_metrics,
                inputs=["tracked_objects", "params:tracking"],
                outputs="tracking_metrics",
                name="compute_tracking_metrics",
                tags=["tracking", "metrics"],
            ),
            node(
                func=log_tracking_to_mlflow,
                inputs=["tracking_metrics", "params:tracking"],
                outputs=None,
                name="log_tracking_to_mlflow",
                tags=["tracking", "mlflow"],
            ),
        ],
        namespace="tracking",
        tags=["tracking"],
    )
