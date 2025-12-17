"""Object Detection Pipeline.

This module implements the object detection pipeline supporting multiple
state-of-the-art models: RF-DETR, RT-DETR, YOLOv11/v12.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_detection_model,
    run_detection_inference,
    post_process_detections,
    filter_detections_by_class,
    compute_detection_metrics,
    log_detection_to_mlflow,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the object detection pipeline.
    
    Returns:
        A Kedro Pipeline object for object detection.
    """
    return pipeline(
        [
            node(
                func=load_detection_model,
                inputs=["params:detection"],
                outputs="detection_model",
                name="load_detection_model",
                tags=["detection", "model"],
            ),
            node(
                func=run_detection_inference,
                inputs=["detection_model", "detection_batch", "params:detection"],
                outputs="raw_detections",
                name="run_detection_inference",
                tags=["detection", "inference"],
            ),
            node(
                func=post_process_detections,
                inputs=["raw_detections", "params:detection"],
                outputs="processed_detections",
                name="post_process_detections",
                tags=["detection", "postprocess"],
            ),
            node(
                func=filter_detections_by_class,
                inputs=["processed_detections", "params:detection.target_classes"],
                outputs="detection_results",
                name="filter_detections_by_class",
                tags=["detection", "filter"],
            ),
            node(
                func=compute_detection_metrics,
                inputs=["detection_results", "params:detection"],
                outputs="detection_metrics",
                name="compute_detection_metrics",
                tags=["detection", "metrics"],
            ),
            node(
                func=log_detection_to_mlflow,
                inputs=["detection_metrics", "params:detection"],
                outputs=None,
                name="log_detection_to_mlflow",
                tags=["detection", "mlflow"],
            ),
        ],
        namespace="object_detection",
        tags=["object_detection"],
    )
