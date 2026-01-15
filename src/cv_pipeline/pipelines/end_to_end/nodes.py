"""End-to-End Autonomous Driving Pipeline Nodes.

This module implements end-to-end autonomous driving models that unify
perception, prediction, and planning in a single differentiable framework.

Supported Models:
    - UniAD: CVPR 2023 Best Paper - Unified perception to planning
    - VAD: Vectorized Scene Representation (ICCV 2023)
    - BEVPlanner: BEV-based motion planning
    - Sparse4D v3: Unified detection + tracking + planning

Architecture Comparison:

Traditional Modular Pipeline:
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Camera  │ → │ Detect  │ → │  Track  │ → │ Predict │ → │  Plan   │
    │ Images  │   │ Objects │   │ Objects │   │ Motion  │   │ Traject │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
                       ↓             ↓             ↓             ↓
                  Separate Models with Information Loss at Each Stage

UniAD End-to-End Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        UniAD Network                            │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         │
    │  │  BEV    │ → │  Track  │ → │  Map    │ → │ Motion  │         │
    │  │ Encoder │   │  Query  │   │ Query   │   │ Query   │         │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────┘         │
    │       ↓             ↓             ↓             ↓               │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │            Unified Transformer Decoder                  │   │
    │  │         (Joint reasoning across all tasks)              │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │       ↓             ↓             ↓             ↓               │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         │
    │  │ 3D Det  │   │ Tracking│   │   Map   │   │ Motion  │→ Plan   │
    │  │ Output  │   │ Output  │   │ Output  │   │ Forecast│  Output │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────┘         │
    └─────────────────────────────────────────────────────────────────┘
                                     ↓
              End-to-End Differentiable with Shared Representations

Key Advantages of End-to-End:
    1. No information loss between stages
    2. Joint optimization for final planning objective
    3. Shared representations reduce redundancy
    4. Gradient flow enables task-level feedback
    5. Simpler deployment (single model)

References:
    - UniAD: https://github.com/OpenDriveLab/UniAD
    - VAD: https://github.com/hustvl/VAD
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class TaskType(Enum):
    """Autonomous driving task types."""

    DETECTION = "detection"
    TRACKING = "tracking"
    MAPPING = "mapping"
    MOTION_FORECAST = "motion_forecast"
    OCCUPANCY = "occupancy"
    PLANNING = "planning"


# nuScenes classes
NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Map elements
MAP_CLASSES = [
    "lane_divider",
    "road_divider",
    "road_edge",
    "crosswalk",
    "pedestrian_crossing",
    "stop_line",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Detection3D:
    """3D object detection result."""

    center: np.ndarray  # [x, y, z] in meters
    size: np.ndarray  # [w, l, h] in meters
    rotation: float  # yaw in radians
    velocity: Optional[np.ndarray] = None  # [vx, vy] in m/s
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "car"
    track_id: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "rotation": self.rotation,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass
class TrackResult:
    """Tracking result for a single object."""

    track_id: int
    detections: List[Detection3D]  # Historical detections
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "car"
    is_active: bool = True

    @property
    def current_position(self) -> np.ndarray:
        """Get current position."""
        if self.detections:
            return self.detections[-1].center
        return np.zeros(3)

    @property
    def current_velocity(self) -> np.ndarray:
        """Get current velocity."""
        if self.detections and self.detections[-1].velocity is not None:
            return self.detections[-1].velocity
        return np.zeros(2)


@dataclass
class MapElement:
    """Map element (lane, crossing, etc.)."""

    element_type: str  # lane_divider, road_edge, etc.
    points: np.ndarray  # [N, 2] or [N, 3] polyline points
    confidence: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "points": self.points.tolist(),
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class MotionForecast:
    """Motion forecast for a tracked object.

    Predicts future trajectories with multiple modes (possible futures).
    """

    track_id: int
    modes: List[np.ndarray]  # List of [T, 2] trajectories
    probabilities: List[float]  # Probability of each mode
    timestamps: np.ndarray  # [T,] timestamps in seconds
    confidence: float = 0.0

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    @property
    def best_mode(self) -> np.ndarray:
        """Get most likely trajectory."""
        if not self.modes:
            return np.zeros((6, 2))  # Default 3s at 2Hz
        best_idx = np.argmax(self.probabilities)
        return self.modes[best_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "modes": [m.tolist() for m in self.modes],
            "probabilities": self.probabilities,
            "timestamps": self.timestamps.tolist(),
            "confidence": self.confidence,
        }


@dataclass
class OccupancyGrid:
    """Predicted occupancy grid.

    Represents future occupancy of the scene at different time steps.
    """

    grid: np.ndarray  # [T, H, W] or [T, C, H, W] occupancy probabilities
    timestamps: np.ndarray  # [T,] timestamps in seconds
    resolution: float = 0.5  # meters per cell
    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0]))

    def get_occupancy_at(self, t: float) -> np.ndarray:
        """Get occupancy grid at specific time."""
        idx = np.argmin(np.abs(self.timestamps - t))
        return self.grid[idx]


@dataclass
class PlanningOutput:
    """Ego vehicle planning output.

    Contains planned trajectory and control signals.
    """

    trajectory: np.ndarray  # [T, 2] or [T, 3] planned positions
    timestamps: np.ndarray  # [T,] timestamps in seconds
    velocities: Optional[np.ndarray] = None  # [T,] planned speeds
    accelerations: Optional[np.ndarray] = None  # [T,] planned accelerations
    curvatures: Optional[np.ndarray] = None  # [T,] planned curvatures
    confidence: float = 0.0
    cost: float = 0.0  # Planning cost (lower is better)

    @property
    def horizon(self) -> float:
        """Planning horizon in seconds."""
        if len(self.timestamps) > 0:
            return float(self.timestamps[-1] - self.timestamps[0])
        return 0.0

    def get_position_at(self, t: float) -> np.ndarray:
        """Interpolate position at time t."""
        if len(self.timestamps) == 0:
            return np.zeros(2)
        return np.interp(t, self.timestamps, self.trajectory.T).T

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory.tolist(),
            "timestamps": self.timestamps.tolist(),
            "velocities": self.velocities.tolist() if self.velocities is not None else None,
            "confidence": self.confidence,
            "cost": self.cost,
            "horizon": self.horizon,
        }


@dataclass
class UniADConfig:
    """Configuration for UniAD model.

    UniAD (Unified Autonomous Driving) jointly optimizes:
    1. 3D Object Detection (TrackFormer-style queries)
    2. Multi-Object Tracking (query propagation)
    3. Online Mapping (lane and boundary estimation)
    4. Motion Forecasting (multi-modal prediction)
    5. Occupancy Prediction (scene-level)
    6. Planning (ego trajectory)
    """

    # Model architecture
    backbone: str = "resnet50"
    num_feature_levels: int = 4
    hidden_dim: int = 256
    num_queries: int = 900
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6

    # BEV configuration
    bev_size: Tuple[int, int] = (200, 200)  # cells
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)  # meters
    z_range: Tuple[float, float] = (-5.0, 3.0)

    # Task-specific
    num_classes: int = 10  # Detection classes
    num_map_classes: int = 6  # Map element classes
    forecast_horizon: float = 3.0  # seconds
    forecast_modes: int = 6  # Number of trajectory modes
    plan_horizon: float = 3.0  # seconds
    plan_points: int = 6  # Points in planned trajectory

    # Device
    device: str = "cuda:0"


@dataclass
class UniADOutput:
    """Complete output from UniAD model.

    Contains all task outputs unified in a single result.
    """

    # Detection and tracking
    detections: List[Detection3D]
    tracks: List[TrackResult]

    # Mapping
    map_elements: List[MapElement]

    # Prediction
    motion_forecasts: List[MotionForecast]
    occupancy: Optional[OccupancyGrid] = None

    # Planning
    plan: Optional[PlanningOutput] = None

    # Metadata
    frame_id: int = 0
    inference_time: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class": t.class_name,
                    "position": t.current_position.tolist(),
                }
                for t in self.tracks
            ],
            "map_elements": [m.to_dict() for m in self.map_elements],
            "motion_forecasts": [f.to_dict() for f in self.motion_forecasts],
            "plan": self.plan.to_dict() if self.plan else None,
            "frame_id": self.frame_id,
            "inference_time": self.inference_time,
        }


# =============================================================================
# UniAD Model Components
# =============================================================================


class BEVEncoder(nn.Module):
    """BEV feature encoder for multi-camera inputs.

    Transforms multi-view 2D features to BEV representation.
    Based on BEVFormer-style spatial cross-attention.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Simplified backbone placeholder
        # Real implementation would use ResNet/VoVNet with FPN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(),
        )

        # BEV positional encoding
        self.bev_pos = nn.Parameter(
            torch.randn(1, config.hidden_dim, config.bev_size[0], config.bev_size[1])
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode images to BEV features.

        Args:
            images: [B, N_cam, C, H, W] multi-camera images
            intrinsics: [B, N_cam, 3, 3] camera intrinsics
            extrinsics: [B, N_cam, 4, 4] camera extrinsics

        Returns:
            BEV features [B, C, H_bev, W_bev]
        """
        B, N, C, H, W = images.shape

        # Process each camera
        img_feats = []
        for i in range(N):
            feat = self.backbone(images[:, i])
            img_feats.append(feat)

        # Stack features
        img_feats = torch.stack(img_feats, dim=1)  # [B, N, C, h, w]

        # Project to BEV (simplified - real impl uses spatial cross-attention)
        bev_feat = img_feats.mean(dim=1)  # [B, C, h, w]

        # Interpolate to BEV size
        bev_feat = nn.functional.interpolate(
            bev_feat,
            size=self.config.bev_size,
            mode="bilinear",
            align_corners=False,
        )

        # Add positional encoding
        bev_feat = bev_feat + self.bev_pos

        return bev_feat


class TrackQuery(nn.Module):
    """Track query module for detection and tracking.

    Uses TrackFormer-style track queries that propagate across frames.
    Each query represents a tracked object.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config

        # Detection queries (new objects)
        self.det_queries = nn.Parameter(torch.randn(config.num_queries, config.hidden_dim))

        # Track query buffer (propagated from previous frame)
        self.register_buffer("track_queries", torch.zeros(0, config.hidden_dim))
        self.register_buffer("track_ids", torch.zeros(0, dtype=torch.long))

        # Query update network
        self.query_update = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )

    def forward(
        self,
        bev_feat: torch.Tensor,
        prev_track_queries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process track queries.

        Args:
            bev_feat: BEV features [B, C, H, W]
            prev_track_queries: Previous track queries [N_prev, C]

        Returns:
            Combined queries and query types (det/track)
        """
        B = bev_feat.shape[0]

        # Flatten BEV features
        bev_flat = bev_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Expand detection queries
        det_q = self.det_queries.unsqueeze(0).expand(B, -1, -1)

        # Combine with track queries if available
        if prev_track_queries is not None and prev_track_queries.shape[0] > 0:
            track_q = prev_track_queries.unsqueeze(0).expand(B, -1, -1)
            queries = torch.cat([track_q, det_q], dim=1)
            query_types = torch.cat(
                [
                    torch.ones(track_q.shape[1]),  # 1 = track
                    torch.zeros(det_q.shape[1]),  # 0 = detection
                ]
            )
        else:
            queries = det_q
            query_types = torch.zeros(det_q.shape[1])

        # Update queries with BEV context
        queries = self.query_update(queries, bev_flat)

        return queries, query_types


class MotionForecaster(nn.Module):
    """Motion forecasting module.

    Predicts multi-modal future trajectories for tracked objects.
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config
        self.num_modes = config.forecast_modes
        self.horizon_points = int(config.forecast_horizon * 2)  # 2 Hz

        # Trajectory prediction head
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.num_modes * self.horizon_points * 2),
        )

        # Mode probability head
        self.mode_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, self.num_modes),
        )

    def forward(self, track_queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future trajectories.

        Args:
            track_queries: [B, N, C] track query features

        Returns:
            trajectories: [B, N, K, T, 2] K modes, T timesteps
            probabilities: [B, N, K] mode probabilities
        """
        B, N, C = track_queries.shape

        # Predict trajectories
        traj = self.trajectory_head(track_queries)  # [B, N, K*T*2]
        traj = traj.view(B, N, self.num_modes, self.horizon_points, 2)

        # Predict mode probabilities
        probs = self.mode_head(track_queries)  # [B, N, K]
        probs = torch.softmax(probs, dim=-1)

        return traj, probs


class PlanningHead(nn.Module):
    """Planning head for ego trajectory generation.

    Generates the ego vehicle's future trajectory considering:
    1. BEV features (scene context)
    2. Predicted motion of other agents
    3. Map information
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config
        self.plan_points = config.plan_points

        # Ego query
        self.ego_query = nn.Parameter(torch.randn(1, config.hidden_dim))

        # Planning transformer
        self.planning_decoder = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )

        # Trajectory output
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.plan_points * 2),
        )

        # Confidence output
        self.confidence_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        bev_feat: torch.Tensor,
        motion_forecasts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ego plan.

        Args:
            bev_feat: BEV features [B, C, H, W]
            motion_forecasts: Other agents' predictions [B, N, K, T, 2]

        Returns:
            trajectory: [B, T, 2] planned positions
            confidence: [B, 1] planning confidence
        """
        B = bev_feat.shape[0]

        # Flatten BEV features
        bev_flat = bev_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Expand ego query
        ego_q = self.ego_query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, C]

        # Decode planning query
        plan_feat = self.planning_decoder(ego_q, bev_flat)  # [B, 1, C]
        plan_feat = plan_feat.squeeze(1)  # [B, C]

        # Generate trajectory
        traj = self.trajectory_head(plan_feat)  # [B, T*2]
        traj = traj.view(B, self.plan_points, 2)

        # Confidence
        conf = torch.sigmoid(self.confidence_head(plan_feat))

        return traj, conf


class UniADModel(nn.Module):
    """UniAD: Planning-oriented Autonomous Driving.

    CVPR 2023 Best Paper

    UniAD unifies multiple driving tasks in a single network:
    1. Detection → Track queries detect objects
    2. Tracking → Query propagation tracks objects
    3. Mapping → Map queries estimate lane structure
    4. Motion Forecasting → Predict agent futures
    5. Occupancy Prediction → Scene-level prediction
    6. Planning → Generate ego trajectory

    Key Innovation:
        All tasks share representations and are jointly optimized
        for the final planning objective.

    Paper: "Planning-oriented Autonomous Driving"
    GitHub: https://github.com/OpenDriveLab/UniAD
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config

        # BEV encoder
        self.bev_encoder = BEVEncoder(config)

        # Track queries (detection + tracking)
        self.track_query = TrackQuery(config)

        # Motion forecaster
        self.motion_forecaster = MotionForecaster(config)

        # Planning head
        self.planning_head = PlanningHead(config)

        # Detection head (outputs 3D boxes from queries)
        self.detection_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10),  # [x,y,z,w,l,h,yaw,vx,vy,conf]
        )

        # Classification head
        self.cls_head = nn.Linear(config.hidden_dim, config.num_classes)

        # Map head (simplified)
        self.map_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_map_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        prev_track_queries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            images: [B, N_cam, C, H, W] multi-camera images
            intrinsics: [B, N_cam, 3, 3] camera intrinsics
            extrinsics: [B, N_cam, 4, 4] camera extrinsics
            prev_track_queries: Previous frame's track queries

        Returns:
            Dictionary with all task outputs
        """
        # 1. Encode to BEV
        bev_feat = self.bev_encoder(images, intrinsics, extrinsics)

        # 2. Track queries
        queries, query_types = self.track_query(bev_feat, prev_track_queries)

        # 3. Detection
        box_preds = self.detection_head(queries)  # [B, N, 10]
        cls_preds = self.cls_head(queries)  # [B, N, num_classes]

        # 4. Motion forecasting (only for track queries)
        traj_preds, traj_probs = self.motion_forecaster(queries)

        # 5. Planning
        plan_traj, plan_conf = self.planning_head(bev_feat, traj_preds)

        return {
            "bev_features": bev_feat,
            "queries": queries,
            "query_types": query_types,
            "box_predictions": box_preds,
            "class_predictions": cls_preds,
            "trajectory_predictions": traj_preds,
            "trajectory_probabilities": traj_probs,
            "planned_trajectory": plan_traj,
            "planning_confidence": plan_conf,
        }


# =============================================================================
# VAD Model (Vectorized Autonomous Driving)
# =============================================================================


class VADModel(nn.Module):
    """VAD: Vectorized Scene Representation for Efficient Autonomous Driving.

    ICCV 2023

    VAD uses vectorized representation instead of dense grids:
    1. Agents represented as vectors (not rasterized)
    2. Map represented as polylines (not pixel masks)
    3. More efficient and accurate than dense methods

    Paper: "VAD: Vectorized Scene Representation for Efficient
           Autonomous Driving"
    GitHub: https://github.com/hustvl/VAD
    """

    def __init__(self, config: UniADConfig):
        super().__init__()
        self.config = config

        # Similar architecture to UniAD with vectorized outputs
        self.bev_encoder = BEVEncoder(config)

        # Vectorized agent queries
        self.agent_queries = nn.Parameter(torch.randn(config.num_queries, config.hidden_dim))

        # Vectorized map queries
        self.map_queries = nn.Parameter(torch.randn(100, config.hidden_dim))

        # Planning with vectorized context
        self.planning_head = PlanningHead(config)

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with vectorized outputs."""
        bev_feat = self.bev_encoder(images, intrinsics, extrinsics)

        # Process agent queries
        B = bev_feat.shape[0]
        bev_flat = bev_feat.flatten(2).transpose(1, 2)

        # Planning
        plan_traj, plan_conf = self.planning_head(bev_feat)

        return {
            "bev_features": bev_feat,
            "planned_trajectory": plan_traj,
            "planning_confidence": plan_conf,
        }


# =============================================================================
# End-to-End Factory
# =============================================================================


class EndToEndFactory:
    """Factory for creating end-to-end autonomous driving models.

    Supported Models:
        - uniad: UniAD (CVPR 2023 Best Paper)
        - vad: VAD Vectorized (ICCV 2023)
        - bevplanner: BEV-based planner

    Model Comparison:
        - UniAD: Most comprehensive (all tasks), 35 FPS
        - VAD: Efficient vectorized, 40 FPS
        - BEVPlanner: Simple BEV planning baseline
    """

    SUPPORTED_MODELS = ["uniad", "vad", "bevplanner"]

    @staticmethod
    def create(model_name: str, config: UniADConfig) -> nn.Module:
        """Create an end-to-end model.

        Args:
            model_name: Name of the model
            config: Model configuration

        Returns:
            Initialized model
        """
        model_name = model_name.lower()

        if model_name == "uniad":
            return UniADModel(config)
        elif model_name == "vad":
            return VADModel(config)
        elif model_name == "bevplanner":
            # Simplified planning-only model
            return UniADModel(config)  # Use UniAD as base
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: {EndToEndFactory.SUPPORTED_MODELS}"
            )


# =============================================================================
# Node Functions
# =============================================================================


def load_end_to_end_model(params: Dict[str, Any]) -> nn.Module:
    """Load an end-to-end autonomous driving model.

    Args:
        params: Model parameters

    Returns:
        Loaded model

    Example:
        params = {
            "model": "uniad",
            "backbone": "resnet50",
            "device": "cuda:0",
            "checkpoint": "weights/uniad_base.pth",
        }
        model = load_end_to_end_model(params)
    """
    config = UniADConfig(
        backbone=params.get("backbone", "resnet50"),
        hidden_dim=params.get("hidden_dim", 256),
        num_queries=params.get("num_queries", 900),
        num_classes=params.get("num_classes", 10),
        device=params.get("device", "cuda:0"),
    )

    model_name = params.get("model", "uniad")
    model = EndToEndFactory.create(model_name, config)

    # Load checkpoint if provided
    checkpoint_path = params.get("checkpoint", None)
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    device = config.device
    model = model.to(device)
    model.eval()

    logger.info(f"End-to-end model loaded: {model_name}")
    return model


def run_end_to_end_inference(
    model: nn.Module,
    images: torch.Tensor,
    params: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]] = None,
) -> UniADOutput:
    """Run end-to-end inference.

    Args:
        model: End-to-end model
        images: Multi-camera images [B, N_cam, C, H, W]
        params: Inference parameters
        prev_state: Previous frame state for temporal models

    Returns:
        UniADOutput with all task results
    """
    device = params.get("device", "cuda:0")
    conf_thresh = params.get("confidence_threshold", 0.3)

    # Move to device
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images).float()
    images = images.to(device)

    # Add batch dimension if needed
    if images.ndim == 4:
        images = images.unsqueeze(0)

    # Get previous track queries if available
    prev_track_queries = None
    if prev_state and "track_queries" in prev_state:
        prev_track_queries = prev_state["track_queries"]

    with torch.no_grad():
        start_time = time.time()
        outputs = model(images, prev_track_queries=prev_track_queries)
        inference_time = time.time() - start_time

    # Parse outputs
    result = _parse_end_to_end_outputs(outputs, conf_thresh, params)
    result.inference_time = inference_time

    logger.info(
        f"End-to-end inference: {result.num_detections} dets, "
        f"{result.num_tracks} tracks, {1 / inference_time:.1f} FPS"
    )

    return result


def _parse_end_to_end_outputs(
    outputs: Dict[str, torch.Tensor],
    conf_thresh: float,
    params: Dict[str, Any],
) -> UniADOutput:
    """Parse model outputs into structured result."""
    detections = []
    tracks = []
    motion_forecasts = []
    map_elements = []

    # Parse detections
    if "box_predictions" in outputs:
        box_preds = outputs["box_predictions"][0].cpu().numpy()  # First batch
        cls_preds = outputs.get("class_predictions", None)
        if cls_preds is not None:
            cls_preds = cls_preds[0].cpu().numpy()

        for i in range(len(box_preds)):
            box = box_preds[i]
            conf = float(box[9]) if len(box) > 9 else 0.5

            if conf < conf_thresh:
                continue

            class_id = int(np.argmax(cls_preds[i])) if cls_preds is not None else 0
            class_name = (
                NUSCENES_CLASSES[class_id] if class_id < len(NUSCENES_CLASSES) else "unknown"
            )

            det = Detection3D(
                center=np.array([box[0], box[1], box[2]]),
                size=np.array([box[3], box[4], box[5]]),
                rotation=float(box[6]),
                velocity=np.array([box[7], box[8]]) if len(box) > 8 else None,
                confidence=conf,
                class_id=class_id,
                class_name=class_name,
            )
            detections.append(det)

    # Parse trajectory predictions
    if "trajectory_predictions" in outputs:
        traj_preds = outputs["trajectory_predictions"][0].cpu().numpy()
        traj_probs = outputs["trajectory_probabilities"][0].cpu().numpy()

        for i in range(min(len(traj_preds), len(detections))):
            forecast = MotionForecast(
                track_id=i,
                modes=[traj_preds[i, k] for k in range(traj_preds.shape[1])],
                probabilities=traj_probs[i].tolist(),
                timestamps=np.arange(traj_preds.shape[2]) * 0.5,  # 2 Hz
                confidence=detections[i].confidence if i < len(detections) else 0.5,
            )
            motion_forecasts.append(forecast)

    # Parse planning
    plan = None
    if "planned_trajectory" in outputs:
        plan_traj = outputs["planned_trajectory"][0].cpu().numpy()
        plan_conf = outputs.get("planning_confidence", torch.tensor([[0.5]]))[0].cpu().item()

        plan = PlanningOutput(
            trajectory=plan_traj,
            timestamps=np.arange(len(plan_traj)) * 0.5,  # 2 Hz
            confidence=float(plan_conf),
        )

    return UniADOutput(
        detections=detections,
        tracks=tracks,
        map_elements=map_elements,
        motion_forecasts=motion_forecasts,
        plan=plan,
    )


def compute_end_to_end_metrics(
    predictions: List[UniADOutput],
    ground_truth: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute end-to-end metrics.

    Args:
        predictions: List of UniADOutput predictions
        ground_truth: Ground truth annotations
        params: Metric parameters

    Returns:
        Dictionary of metrics for all tasks

    Metrics:
        - Detection: mAP, NDS
        - Tracking: AMOTA, AMOTP
        - Mapping: mIoU for lanes
        - Motion: minADE, minFDE, MR
        - Planning: L2 error, collision rate
    """
    metrics = {
        # Detection
        "det_mAP": 0.0,
        "det_NDS": 0.0,
        # Tracking
        "track_AMOTA": 0.0,
        "track_AMOTP": 0.0,
        # Mapping
        "map_mIoU": 0.0,
        # Motion forecasting
        "motion_minADE": 0.0,  # Minimum Average Displacement Error
        "motion_minFDE": 0.0,  # Minimum Final Displacement Error
        "motion_MR": 0.0,  # Miss Rate
        # Planning
        "plan_L2_1s": 0.0,  # L2 error at 1s
        "plan_L2_2s": 0.0,  # L2 error at 2s
        "plan_L2_3s": 0.0,  # L2 error at 3s
        "plan_collision": 0.0,  # Collision rate
        # General
        "num_frames": len(predictions),
        "avg_fps": (
            len(predictions) / sum(p.inference_time for p in predictions) if predictions else 0
        ),
    }

    # Full metric computation would use nuScenes/Waymo evaluation
    # This is a placeholder for the interface

    logger.info(
        f"End-to-end metrics: mAP={metrics['det_mAP']:.3f}, L2@3s={metrics['plan_L2_3s']:.3f}m"
    )

    return metrics
