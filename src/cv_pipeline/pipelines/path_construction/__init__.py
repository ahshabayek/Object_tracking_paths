"""Path Construction Pipeline.

This module implements path/trajectory construction by fusing:
- Lane detection results
- Object tracking results
- Ego vehicle state
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    fuse_perception_data,
    construct_drivable_path,
    compute_path_metrics,
    generate_trajectory,
    log_path_to_mlflow,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the path construction pipeline.
    
    Returns:
        A Kedro Pipeline object for path construction.
    """
    return pipeline(
        [
            node(
                func=fuse_perception_data,
                inputs=["lane_detections", "tracked_objects", "params:lane.path_construction"],
                outputs="fused_scene",
                name="fuse_perception_data",
                tags=["path", "fusion"],
            ),
            node(
                func=construct_drivable_path,
                inputs=["fused_scene", "lane_detections", "params:lane.path_construction"],
                outputs="drivable_path",
                name="construct_drivable_path",
                tags=["path", "construction"],
            ),
            node(
                func=generate_trajectory,
                inputs=["drivable_path", "params:lane.driving_path"],
                outputs="constructed_path",
                name="generate_trajectory",
                tags=["path", "trajectory"],
            ),
            node(
                func=compute_path_metrics,
                inputs=["constructed_path", "params:lane.path_construction"],
                outputs="path_metrics",
                name="compute_path_metrics",
                tags=["path", "metrics"],
            ),
            node(
                func=log_path_to_mlflow,
                inputs=["path_metrics"],
                outputs=None,
                name="log_path_to_mlflow",
                tags=["path", "mlflow"],
            ),
        ],
        namespace="path_construction",
        tags=["path_construction"],
    )
