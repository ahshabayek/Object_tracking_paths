"""End-to-End Autonomous Driving Pipeline.

This module provides end-to-end autonomous driving models that unify
perception, prediction, and planning in a single differentiable framework.

Supported Models:
    - UniAD: CVPR 2023 Best Paper - Unified perception to planning
    - VAD: Vectorized Scene Representation (ICCV 2023)
    - BEVPlanner: BEV-based motion planning

Key Advantages of End-to-End:
    1. No information loss between perception stages
    2. Joint optimization for final planning objective
    3. Shared representations reduce redundancy
    4. Gradient flow enables task-level feedback

Example Usage:
    from cv_pipeline.pipelines.end_to_end.nodes import (
        load_end_to_end_model,
        run_end_to_end_inference,
        UniADOutput,
    )

    # Load model
    model = load_end_to_end_model({
        "model": "uniad",
        "device": "cuda:0",
    })

    # Run inference
    output = run_end_to_end_inference(model, images, {})
    print(f"Plan: {output.plan.trajectory}")
"""

from .nodes import (
    BEVEncoder,
    Detection3D,
    EndToEndFactory,
    MapElement,
    MotionForecast,
    MotionForecaster,
    OccupancyGrid,
    PlanningHead,
    PlanningOutput,
    TaskType,
    TrackQuery,
    TrackResult,
    UniADConfig,
    UniADModel,
    UniADOutput,
    VADModel,
    compute_end_to_end_metrics,
    load_end_to_end_model,
    run_end_to_end_inference,
)

__all__ = [
    # Enums
    "TaskType",
    # Data classes
    "Detection3D",
    "TrackResult",
    "MapElement",
    "MotionForecast",
    "OccupancyGrid",
    "PlanningOutput",
    "UniADConfig",
    "UniADOutput",
    # Model components
    "BEVEncoder",
    "TrackQuery",
    "MotionForecaster",
    "PlanningHead",
    # Models
    "UniADModel",
    "VADModel",
    "EndToEndFactory",
    # Node functions
    "load_end_to_end_model",
    "run_end_to_end_inference",
    "compute_end_to_end_metrics",
]
