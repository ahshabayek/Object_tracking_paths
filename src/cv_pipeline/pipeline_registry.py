"""Kedro Pipeline Registry.

This module provides the central registry for all pipelines in the CV pipeline project.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from cv_pipeline.pipelines.data_processing import create_pipeline as create_data_pipeline
from cv_pipeline.pipelines.object_detection import create_pipeline as create_detection_pipeline
from cv_pipeline.pipelines.tracking import create_pipeline as create_tracking_pipeline
from cv_pipeline.pipelines.lane_detection import create_pipeline as create_lane_pipeline
from cv_pipeline.pipelines.path_construction import create_pipeline as create_path_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines.
    
    Returns:
        A dictionary mapping pipeline names to Pipeline objects.
    """
    # Individual pipelines
    data_processing_pipeline = create_data_pipeline()
    object_detection_pipeline = create_detection_pipeline()
    tracking_pipeline = create_tracking_pipeline()
    lane_detection_pipeline = create_lane_pipeline()
    path_construction_pipeline = create_path_pipeline()
    
    # Combined pipelines
    full_detection_pipeline = (
        data_processing_pipeline 
        + object_detection_pipeline
    )
    
    full_tracking_pipeline = (
        data_processing_pipeline 
        + object_detection_pipeline 
        + tracking_pipeline
    )
    
    full_perception_pipeline = (
        data_processing_pipeline
        + object_detection_pipeline
        + tracking_pipeline
        + lane_detection_pipeline
        + path_construction_pipeline
    )
    
    return {
        # Individual pipelines
        "data_processing": data_processing_pipeline,
        "object_detection": object_detection_pipeline,
        "tracking": tracking_pipeline,
        "lane_detection": lane_detection_pipeline,
        "path_construction": path_construction_pipeline,
        
        # Combined pipelines
        "full_detection": full_detection_pipeline,
        "full_tracking": full_tracking_pipeline,
        "full_perception": full_perception_pipeline,
        
        # Default pipeline
        "__default__": full_perception_pipeline,
    }
