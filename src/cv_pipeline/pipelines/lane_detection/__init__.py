"""Lane Detection Pipeline.

This module implements lane detection using:
- CLRerNet (SOTA - WACV 2024)
- CLRNet (CVPR 2022)
- LaneATT
- UFLD (Ultra-Fast Lane Detection)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_lane_model,
    run_lane_detection,
    fit_lane_curves,
    compute_lane_metrics,
    log_lane_to_mlflow,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the lane detection pipeline.
    
    Returns:
        A Kedro Pipeline object for lane detection.
    """
    return pipeline(
        [
            node(
                func=load_lane_model,
                inputs=["params:lane"],
                outputs="lane_model",
                name="load_lane_model",
                tags=["lane", "model"],
            ),
            node(
                func=run_lane_detection,
                inputs=["lane_model", "preprocessed_frames", "params:lane"],
                outputs="raw_lanes",
                name="run_lane_detection",
                tags=["lane", "inference"],
            ),
            node(
                func=fit_lane_curves,
                inputs=["raw_lanes", "params:lane"],
                outputs="lane_detections",
                name="fit_lane_curves",
                tags=["lane", "curves"],
            ),
            node(
                func=compute_lane_metrics,
                inputs=["lane_detections", "params:lane"],
                outputs="lane_metrics",
                name="compute_lane_metrics",
                tags=["lane", "metrics"],
            ),
            node(
                func=log_lane_to_mlflow,
                inputs=["lane_metrics", "params:lane"],
                outputs=None,
                name="log_lane_to_mlflow",
                tags=["lane", "mlflow"],
            ),
        ],
        namespace="lane_detection",
        tags=["lane_detection"],
    )
