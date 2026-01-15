"""Kedro Pipeline Registry.

This module provides the central registry for all pipelines in the CV pipeline project.

Pipeline Data Flow:
    raw_video -> data_processing -> preprocessed_frames, detection_batch
                                          |                    |
                                          v                    v
                                   lane_detection      object_detection
                                          |                    |
                                          v                    v
                                   lane_detections     detection_results
                                          |                    |
                                          +--------------------+
                                                    |
                                                    v
                                               tracking
                                                    |
                                                    v
                                            tracked_objects
                                                    |
                                                    v
                                          path_construction
                                                    |
                                                    v
                                           constructed_path
"""

from typing import Dict

from kedro.pipeline import Pipeline, node, pipeline

from cv_pipeline.pipelines.data_processing import create_pipeline as create_data_pipeline
from cv_pipeline.pipelines.lane_detection import create_pipeline as create_lane_pipeline
from cv_pipeline.pipelines.object_detection import create_pipeline as create_detection_pipeline
from cv_pipeline.pipelines.path_construction import create_pipeline as create_path_pipeline
from cv_pipeline.pipelines.tracking import create_pipeline as create_tracking_pipeline


def create_inference_pipeline() -> Pipeline:
    """Create the full inference pipeline with proper data flow.

    This pipeline chains all components together:
    1. Data Processing: Load and preprocess video frames
    2. Object Detection: Detect objects in frames
    3. Tracking: Track detected objects across frames
    4. Lane Detection: Detect lane markings
    5. Path Construction: Construct drivable path

    Returns:
        A Kedro Pipeline for full inference.
    """
    # Create individual pipelines without namespaces for proper data flow
    # We'll use tags instead of namespaces for filtering

    from cv_pipeline.pipelines.data_processing.nodes import (
        apply_augmentations,
        create_batches,
        extract_frame_metadata,
        load_video_frames,
        preprocess_frames,
    )
    from cv_pipeline.pipelines.lane_detection.nodes import (
        compute_lane_metrics,
        fit_lane_curves,
        load_lane_model,
        log_lane_to_mlflow,
        run_lane_detection,
    )
    from cv_pipeline.pipelines.object_detection.nodes import (
        compute_detection_metrics,
        filter_detections_by_class,
        load_detection_model,
        log_detection_to_mlflow,
        post_process_detections,
        run_detection_inference,
    )
    from cv_pipeline.pipelines.path_construction.nodes import (
        compute_path_metrics,
        construct_drivable_path,
        fuse_perception_data,
        generate_trajectory,
        log_path_to_mlflow,
    )
    from cv_pipeline.pipelines.tracking.nodes import (
        compute_tracking_metrics,
        extract_trajectories,
        initialize_tracker,
        log_tracking_to_mlflow,
        run_tracking,
    )

    return pipeline(
        [
            # ========== DATA PROCESSING ==========
            node(
                func=load_video_frames,
                inputs=["raw_video", "params:data_processing"],
                outputs="raw_frames",
                name="load_video_frames",
                tags=["data_processing", "inference"],
            ),
            node(
                func=extract_frame_metadata,
                inputs="raw_frames",
                outputs="frame_metadata",
                name="extract_frame_metadata",
                tags=["data_processing", "inference"],
            ),
            node(
                func=preprocess_frames,
                inputs=["raw_frames", "params:data_processing"],
                outputs="preprocessed_frames",
                name="preprocess_frames",
                tags=["data_processing", "inference"],
            ),
            node(
                func=apply_augmentations,
                inputs=["preprocessed_frames", "params:data_processing.augmentation"],
                outputs="augmented_frames",
                name="apply_augmentations",
                tags=["data_processing", "inference"],
            ),
            node(
                func=create_batches,
                inputs=["augmented_frames", "params:data_processing.batch_size"],
                outputs="detection_batch",
                name="create_batches",
                tags=["data_processing", "inference"],
            ),
            # ========== OBJECT DETECTION ==========
            node(
                func=load_detection_model,
                inputs=["params:detection"],
                outputs="detection_model",
                name="load_detection_model",
                tags=["object_detection", "inference"],
            ),
            node(
                func=run_detection_inference,
                inputs=["detection_model", "detection_batch", "params:detection"],
                outputs="raw_detections",
                name="run_detection_inference",
                tags=["object_detection", "inference"],
            ),
            node(
                func=post_process_detections,
                inputs=["raw_detections", "params:detection"],
                outputs="processed_detections",
                name="post_process_detections",
                tags=["object_detection", "inference"],
            ),
            node(
                func=filter_detections_by_class,
                inputs=["processed_detections", "params:detection.target_classes"],
                outputs="detection_results",
                name="filter_detections_by_class",
                tags=["object_detection", "inference"],
            ),
            node(
                func=compute_detection_metrics,
                inputs=["detection_results", "params:detection"],
                outputs="detection_metrics",
                name="compute_detection_metrics",
                tags=["object_detection", "metrics"],
            ),
            node(
                func=log_detection_to_mlflow,
                inputs=["detection_metrics", "params:detection"],
                outputs=None,
                name="log_detection_to_mlflow",
                tags=["object_detection", "mlflow"],
            ),
            # ========== TRACKING ==========
            node(
                func=initialize_tracker,
                inputs=["params:tracking"],
                outputs="tracker",
                name="initialize_tracker",
                tags=["tracking", "inference"],
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
                tags=["tracking", "inference"],
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
            # ========== LANE DETECTION ==========
            node(
                func=load_lane_model,
                inputs=["params:lane"],
                outputs="lane_model",
                name="load_lane_model",
                tags=["lane_detection", "inference"],
            ),
            node(
                func=run_lane_detection,
                inputs=["lane_model", "preprocessed_frames", "params:lane"],
                outputs="raw_lanes",
                name="run_lane_detection",
                tags=["lane_detection", "inference"],
            ),
            node(
                func=fit_lane_curves,
                inputs=["raw_lanes", "params:lane"],
                outputs="lane_detections",
                name="fit_lane_curves",
                tags=["lane_detection", "inference"],
            ),
            node(
                func=compute_lane_metrics,
                inputs=["lane_detections", "params:lane"],
                outputs="lane_metrics",
                name="compute_lane_metrics",
                tags=["lane_detection", "metrics"],
            ),
            node(
                func=log_lane_to_mlflow,
                inputs=["lane_metrics", "params:lane"],
                outputs=None,
                name="log_lane_to_mlflow",
                tags=["lane_detection", "mlflow"],
            ),
            # ========== PATH CONSTRUCTION ==========
            node(
                func=fuse_perception_data,
                inputs=["lane_detections", "tracked_objects", "params:lane.path_construction"],
                outputs="fused_scene",
                name="fuse_perception_data",
                tags=["path_construction", "inference"],
            ),
            node(
                func=construct_drivable_path,
                inputs=["fused_scene", "lane_detections", "params:lane.path_construction"],
                outputs="drivable_path",
                name="construct_drivable_path",
                tags=["path_construction", "inference"],
            ),
            node(
                func=generate_trajectory,
                inputs=["drivable_path", "params:lane.driving_path"],
                outputs="constructed_path",
                name="generate_trajectory",
                tags=["path_construction", "inference"],
            ),
            node(
                func=compute_path_metrics,
                inputs=["constructed_path", "params:lane.path_construction"],
                outputs="path_metrics",
                name="compute_path_metrics",
                tags=["path_construction", "metrics"],
            ),
            node(
                func=log_path_to_mlflow,
                inputs=["path_metrics"],
                outputs=None,
                name="log_path_to_mlflow",
                tags=["path_construction", "mlflow"],
            ),
        ],
        tags=["inference"],
    )


def create_detection_only_pipeline() -> Pipeline:
    """Create pipeline for detection only (no tracking/lanes).

    Returns:
        Pipeline for object detection inference.
    """
    inference_pipeline = create_inference_pipeline()
    return inference_pipeline.only_nodes_with_tags("data_processing", "object_detection")


def create_tracking_only_pipeline() -> Pipeline:
    """Create pipeline for detection + tracking (no lanes).

    Returns:
        Pipeline for detection and tracking inference.
    """
    inference_pipeline = create_inference_pipeline()
    return inference_pipeline.only_nodes_with_tags(
        "data_processing", "object_detection", "tracking"
    )


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        A dictionary mapping pipeline names to Pipeline objects.

    Available Pipelines:
        - data_processing: Load and preprocess video/images
        - object_detection: Detect objects in frames
        - tracking: Track objects across frames
        - lane_detection: Detect lane markings
        - path_construction: Construct drivable path
        - inference: Full inference pipeline (recommended)
        - detection_only: Data processing + detection
        - tracking_only: Data processing + detection + tracking
        - __default__: Full inference pipeline

    Usage:
        kedro run                          # Run full inference
        kedro run --pipeline=inference     # Same as above
        kedro run --pipeline=detection_only
        kedro run --pipeline=tracking_only
        kedro run --tags=object_detection  # Run only detection nodes
    """
    # Individual namespaced pipelines (for running separately)
    data_processing_pipeline = create_data_pipeline()
    object_detection_pipeline = create_detection_pipeline()
    tracking_pipeline = create_tracking_pipeline()
    lane_detection_pipeline = create_lane_pipeline()
    path_construction_pipeline = create_path_pipeline()

    # Master inference pipeline with proper data flow
    inference_pipeline = create_inference_pipeline()

    # Subset pipelines
    detection_only = create_detection_only_pipeline()
    tracking_only = create_tracking_only_pipeline()

    return {
        # Individual pipelines (namespaced, for development/testing)
        "data_processing": data_processing_pipeline,
        "object_detection": object_detection_pipeline,
        "tracking": tracking_pipeline,
        "lane_detection": lane_detection_pipeline,
        "path_construction": path_construction_pipeline,
        # Master inference pipeline (recommended for production)
        "inference": inference_pipeline,
        # Subset pipelines
        "detection_only": detection_only,
        "tracking_only": tracking_only,
        # Default pipeline
        "__default__": inference_pipeline,
    }
