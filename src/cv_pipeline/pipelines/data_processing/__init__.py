"""Data Processing Pipeline.

This module handles video/image ingestion, preprocessing, and augmentation.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_video_frames,
    load_image_batch,
    preprocess_frames,
    apply_augmentations,
    create_batches,
    extract_frame_metadata,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline.
    
    Returns:
        A Kedro Pipeline object for data processing.
    """
    return pipeline(
        [
            node(
                func=load_video_frames,
                inputs=["raw_video", "params:data_processing"],
                outputs="raw_frames",
                name="load_video_frames",
                tags=["data", "video"],
            ),
            node(
                func=extract_frame_metadata,
                inputs="raw_frames",
                outputs="frame_metadata",
                name="extract_frame_metadata",
                tags=["data", "metadata"],
            ),
            node(
                func=preprocess_frames,
                inputs=["raw_frames", "params:data_processing"],
                outputs="preprocessed_frames",
                name="preprocess_frames",
                tags=["data", "preprocessing"],
            ),
            node(
                func=apply_augmentations,
                inputs=["preprocessed_frames", "params:data_processing.augmentation"],
                outputs="augmented_frames",
                name="apply_augmentations",
                tags=["data", "augmentation"],
            ),
            node(
                func=create_batches,
                inputs=["augmented_frames", "params:data_processing.batch_size"],
                outputs="detection_batch",
                name="create_batches",
                tags=["data", "batching"],
            ),
        ],
        namespace="data_processing",
        tags=["data_processing"],
    )
