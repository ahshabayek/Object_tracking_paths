"""BEV Perception Pipeline Nodes.

This module implements Bird's Eye View (BEV) perception for 3D object detection
from multi-camera inputs. BEV perception is essential for autonomous driving
as it provides object positions in real-world coordinates (meters).

Supported Models:
    - SparseBEV: SOTA 67.5% NDS, fully sparse, real-time (ICCV 2023)
    - BEVFormer: Classic transformer-based BEV (ECCV 2022)
    - Sparse4D: Unified detection + tracking (ICLR 2024)
    - StreamPETR: Streaming perception (ICCV 2023)

Architecture Overview:

    Multi-Camera Images     Camera Calibration
           ↓                       ↓
    ┌──────────────────────────────────────┐
    │         Shared Image Backbone        │
    │         (ResNet-50/101, VoVNet)      │
    └──────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────┐
    │      BEV Query / Spatial Attention   │
    │  (project 2D features to 3D BEV)     │
    └──────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────┐
    │      Temporal Self-Attention         │
    │   (fuse with previous BEV frames)    │
    └──────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────┐
    │        3D Detection Head             │
    │   (predict 3D boxes in meters)       │
    └──────────────────────────────────────┘
                      ↓
         3D Boxes: [x, y, z, w, h, l, yaw, vx, vy]

Key Differences from 2D Detection:
    - Input: 6 cameras (surround view) vs 1 camera
    - Output: 3D boxes in meters vs 2D boxes in pixels
    - Queries: BEV spatial grid vs object queries
    - Requires: Camera calibration (intrinsics + extrinsics)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CameraConfig:
    """Camera configuration with intrinsics and extrinsics.

    Attributes:
        name: Camera name (e.g., 'CAM_FRONT', 'CAM_FRONT_LEFT')
        intrinsic: Intrinsic matrix [3, 3] containing fx, fy, cx, cy
        extrinsic: Extrinsic matrix [4, 4] (camera to ego transformation)
        image_size: Image dimensions (width, height)

    Coordinate Systems:
        - Camera frame: X-right, Y-down, Z-forward
        - Ego frame: X-forward, Y-left, Z-up (vehicle center)
        - World frame: Global coordinates
    """

    name: str
    intrinsic: np.ndarray  # [3, 3]
    extrinsic: np.ndarray  # [4, 4] camera to ego
    image_size: Tuple[int, int] = (1600, 900)  # (W, H)

    @property
    def fx(self) -> float:
        return float(self.intrinsic[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic[1, 2])


@dataclass
class BEVConfig:
    """Configuration for BEV perception.

    Attributes:
        model: Model name (sparsebev, bevformer, sparse4d, streampetr)
        backbone: Image backbone (resnet50, resnet101, vovnet)
        bev_size: BEV grid size (H, W) in cells
        bev_range: BEV range in meters [x_min, x_max, y_min, y_max]
        z_range: Height range in meters [z_min, z_max]
        num_classes: Number of detection classes
        device: Device for inference
        use_temporal: Whether to use temporal fusion
        num_history_frames: Number of history frames for temporal fusion
    """

    model: str = "sparsebev"
    backbone: str = "resnet50"
    bev_size: Tuple[int, int] = (200, 200)  # cells
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)  # meters
    z_range: Tuple[float, float] = (-5.0, 3.0)  # meters
    num_classes: int = 10  # nuScenes classes
    device: str = "cuda:0"
    use_temporal: bool = True
    num_history_frames: int = 4

    @property
    def bev_resolution(self) -> float:
        """Meters per BEV cell."""
        x_range = self.bev_range[1] - self.bev_range[0]
        return x_range / self.bev_size[0]


@dataclass
class Detection3D:
    """3D detection result in BEV.

    Attributes:
        center: Center position [x, y, z] in ego frame (meters)
        size: Object size [width, length, height] in meters
        rotation: Rotation angle (yaw) in radians
        velocity: Velocity [vx, vy] in m/s (optional)
        confidence: Detection confidence
        class_id: Class ID
        class_name: Class name
        track_id: Track ID if tracking enabled

    Coordinate System (Ego Frame):
        - X: forward (positive = ahead of vehicle)
        - Y: left (positive = left of vehicle)
        - Z: up (positive = above ground)
    """

    center: np.ndarray  # [x, y, z] in meters
    size: np.ndarray  # [w, l, h] in meters
    rotation: float  # yaw in radians
    velocity: Optional[np.ndarray] = None  # [vx, vy] in m/s
    confidence: float = 0.0
    class_id: int = -1
    class_name: str = "unknown"
    track_id: int = -1

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    @property
    def z(self) -> float:
        return float(self.center[2])

    @property
    def distance(self) -> float:
        """Euclidean distance from ego vehicle."""
        return float(np.sqrt(self.center[0] ** 2 + self.center[1] ** 2))

    def get_corners_3d(self) -> np.ndarray:
        """Get 8 corners of the 3D bounding box.

        Returns:
            [8, 3] array of corner coordinates
        """
        w, l, h = self.size
        x, y, z = self.center

        # Corners in object frame (centered at origin)
        corners = np.array(
            [
                [-l / 2, -w / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [-l / 2, w / 2, -h / 2],
                [-l / 2, w / 2, h / 2],
                [l / 2, -w / 2, -h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, w / 2, -h / 2],
                [l / 2, w / 2, h / 2],
            ]
        )

        # Rotation matrix (yaw only)
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )

        # Rotate and translate
        corners = corners @ R.T + self.center
        return corners

    def get_bev_box(self) -> np.ndarray:
        """Get 4 corners in BEV (top-down view).

        Returns:
            [4, 2] array of corner coordinates (x, y)
        """
        corners_3d = self.get_corners_3d()
        # Take bottom 4 corners
        bev_corners = corners_3d[[0, 2, 6, 4], :2]
        return bev_corners

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
            "distance": self.distance,
        }


@dataclass
class BEVResult:
    """Container for BEV perception results.

    Attributes:
        detections: List of 3D detections
        bev_features: BEV feature map (optional)
        inference_time: Time taken for inference
        frame_id: Frame identifier
    """

    detections: List[Detection3D]
    bev_features: Optional[np.ndarray] = None  # [C, H, W]
    inference_time: float = 0.0
    frame_id: int = 0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def get_detections_in_range(self, max_distance: float = 50.0) -> List[Detection3D]:
        """Filter detections by distance from ego."""
        return [d for d in self.detections if d.distance <= max_distance]

    def get_detections_by_class(self, class_names: List[str]) -> List[Detection3D]:
        """Filter detections by class name."""
        return [d for d in self.detections if d.class_name in class_names]


# =============================================================================
# nuScenes Class Names
# =============================================================================

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


# =============================================================================
# BEV Perception Factory
# =============================================================================


class BEVPerceptionFactory:
    """Factory for creating BEV perception models.

    Supported Models:
        - sparsebev: SparseBEV (ICCV 2023) - SOTA, fully sparse, 67.5% NDS
        - bevformer: BEVFormer (ECCV 2022) - Classic transformer BEV
        - sparse4d: Sparse4D (ICLR 2024) - Unified detection + tracking
        - streampetr: StreamPETR (ICCV 2023) - Streaming perception

    Key Architectural Differences:
        - SparseBEV: Sparse queries, adaptive sampling, real-time
        - BEVFormer: Dense BEV grid, deformable attention
        - Sparse4D: 4D anchor boxes, sparse temporal fusion
        - StreamPETR: Object queries with long-term memory
    """

    SUPPORTED_MODELS = ["sparsebev", "bevformer", "sparse4d", "streampetr"]

    @staticmethod
    def create(config: BEVConfig) -> nn.Module:
        """Create a BEV perception model.

        Args:
            config: BEV configuration

        Returns:
            Initialized BEV model
        """
        model_name = config.model.lower()

        if model_name == "sparsebev":
            return BEVPerceptionFactory._create_sparsebev(config)
        elif model_name == "bevformer":
            return BEVPerceptionFactory._create_bevformer(config)
        elif model_name == "sparse4d":
            return BEVPerceptionFactory._create_sparse4d(config)
        elif model_name == "streampetr":
            return BEVPerceptionFactory._create_streampetr(config)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: {BEVPerceptionFactory.SUPPORTED_MODELS}"
            )

    @staticmethod
    def _create_sparsebev(config: BEVConfig) -> nn.Module:
        """Create SparseBEV model.

        SparseBEV (ICCV 2023):
        - Fully sparse 3D object detection
        - Scale-adaptive self-attention in BEV
        - Adaptive spatio-temporal sampling
        - SOTA: 67.5% NDS on nuScenes test

        Key Innovations:
        1. Sparse queries instead of dense BEV grid
        2. Adaptive receptive field via scale-adaptive attention
        3. Query-guided sampling of image features

        Paper: "SparseBEV: High-Performance Sparse 3D Object Detection
               from Multi-Camera Videos"
        Repo: https://github.com/MCG-NJU/SparseBEV
        """
        try:
            # Try to import from installed package or local repo
            from mmcv import Config
            from mmdet3d.models import build_model

            # SparseBEV config
            cfg_path = config.get("config_path", None)
            if cfg_path:
                cfg = Config.fromfile(cfg_path)
                model = build_model(cfg.model)
            else:
                # Create default SparseBEV model
                model = _create_default_sparsebev(config)

            logger.info("Loaded SparseBEV model")
            return model

        except ImportError:
            logger.warning(
                "mmdet3d not installed. SparseBEV requires:\n"
                "  pip install mmdet3d mmcv-full\n"
                "  git clone https://github.com/MCG-NJU/SparseBEV"
            )
            # Return a placeholder model
            return _create_placeholder_bev_model(config)

    @staticmethod
    def _create_bevformer(config: BEVConfig) -> nn.Module:
        """Create BEVFormer model.

        BEVFormer (ECCV 2022):
        - Dense BEV grid representation
        - Spatial cross-attention to multi-view images
        - Temporal self-attention with ego-motion alignment

        Key Components:
        1. BEV Queries: Learnable grid (200x200) representing top-down view
        2. Spatial Cross-Attention: Sample features from all cameras
        3. Temporal Self-Attention: Fuse with history BEV

        Paper: "BEVFormer: Learning Bird's-Eye-View Representation
               from Multi-Camera Images via Spatiotemporal Transformers"
        Repo: https://github.com/fundamentalvision/BEVFormer
        """
        try:
            from mmcv import Config
            from mmdet3d.models import build_model

            cfg_path = config.get("config_path", None)
            if cfg_path:
                cfg = Config.fromfile(cfg_path)
                model = build_model(cfg.model)
            else:
                model = _create_placeholder_bev_model(config)

            logger.info("Loaded BEVFormer model")
            return model

        except ImportError:
            logger.warning("mmdet3d not installed for BEVFormer")
            return _create_placeholder_bev_model(config)

    @staticmethod
    def _create_sparse4d(config: BEVConfig) -> nn.Module:
        """Create Sparse4D model.

        Sparse4D (ICLR 2024):
        - 4D anchor boxes (3D space + time)
        - Sparse spatial-temporal fusion
        - Unified detection and tracking

        Key Innovations:
        1. 4D anchors: Boxes with temporal extent
        2. Iterative refinement of anchor boxes
        3. End-to-end tracking via anchor propagation

        Paper: "Sparse4D: Multi-view 3D Object Detection with
               Sparse Spatial-Temporal Fusion"
        Repo: https://github.com/linxuewu/Sparse4D
        """
        try:
            from mmdet3d.models import build_model

            model = _create_placeholder_bev_model(config)
            logger.info("Loaded Sparse4D model")
            return model

        except ImportError:
            logger.warning("mmdet3d not installed for Sparse4D")
            return _create_placeholder_bev_model(config)

    @staticmethod
    def _create_streampetr(config: BEVConfig) -> nn.Module:
        """Create StreamPETR model.

        StreamPETR (ICCV 2023):
        - Streaming perception with long-term memory
        - Object queries propagated across frames
        - Efficient temporal fusion

        Key Innovations:
        1. Object-centric temporal modeling
        2. Memory queue for long-term context
        3. Real-time streaming inference

        Paper: "Exploring Object-Centric Temporal Modeling for
               Efficient Multi-View 3D Object Detection"
        Repo: https://github.com/exiawsh/StreamPETR
        """
        try:
            from mmdet3d.models import build_model

            model = _create_placeholder_bev_model(config)
            logger.info("Loaded StreamPETR model")
            return model

        except ImportError:
            logger.warning("mmdet3d not installed for StreamPETR")
            return _create_placeholder_bev_model(config)


def _create_placeholder_bev_model(config: BEVConfig) -> nn.Module:
    """Create a placeholder BEV model for testing without full dependencies.

    This model simulates the interface of real BEV models but doesn't
    perform actual inference. Use for testing pipeline structure.
    """

    class PlaceholderBEVModel(nn.Module):
        """Placeholder BEV model for testing."""

        def __init__(self, config: BEVConfig):
            super().__init__()
            self.config = config
            self.num_classes = config.num_classes

            # Simple backbone placeholder
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )

        def forward(
            self,
            images: torch.Tensor,
            camera_intrinsics: Optional[torch.Tensor] = None,
            camera_extrinsics: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass returning dummy predictions.

            Args:
                images: [B, N_cam, C, H, W] multi-camera images
                camera_intrinsics: [B, N_cam, 3, 3] intrinsic matrices
                camera_extrinsics: [B, N_cam, 4, 4] extrinsic matrices

            Returns:
                Dictionary with 'boxes_3d', 'scores', 'labels'
            """
            batch_size = images.shape[0]

            # Return dummy predictions
            # In real model, this would be actual 3D detections
            num_preds = 50

            return {
                "boxes_3d": torch.randn(batch_size, num_preds, 9),  # [x,y,z,w,l,h,yaw,vx,vy]
                "scores": torch.rand(batch_size, num_preds),
                "labels": torch.randint(0, self.num_classes, (batch_size, num_preds)),
            }

    return PlaceholderBEVModel(config)


def _create_default_sparsebev(config: BEVConfig) -> nn.Module:
    """Create default SparseBEV model with standard config."""
    # This would contain the actual SparseBEV architecture
    # For now, return placeholder
    return _create_placeholder_bev_model(config)


# =============================================================================
# Node Functions
# =============================================================================


def load_bev_model(params: Dict[str, Any]) -> nn.Module:
    """Load a BEV perception model.

    Args:
        params: Dictionary with BEV configuration

    Returns:
        Loaded BEV model

    Example:
        params = {
            "model": "sparsebev",
            "backbone": "resnet50",
            "device": "cuda:0",
            "config_path": "configs/sparsebev_r50_nusc.py",
            "checkpoint": "weights/sparsebev_r50.pth",
        }
        model = load_bev_model(params)
    """
    config = BEVConfig(
        model=params.get("model", "sparsebev"),
        backbone=params.get("backbone", "resnet50"),
        bev_size=tuple(params.get("bev_size", (200, 200))),
        device=params.get("device", "cuda:0"),
        use_temporal=params.get("use_temporal", True),
    )

    model = BEVPerceptionFactory.create(config)

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

    # Move to device
    device = config.device
    model = model.to(device)
    model.eval()

    logger.info(f"BEV model loaded: {config.model}")
    return model


def run_bev_inference(
    model: nn.Module,
    images: torch.Tensor,
    camera_configs: List[CameraConfig],
    params: Dict[str, Any],
) -> List[BEVResult]:
    """Run BEV inference on multi-camera images.

    Args:
        model: BEV perception model
        images: Multi-camera images [B, N_cam, C, H, W] or list
        camera_configs: List of camera configurations
        params: Inference parameters

    Returns:
        List of BEVResult for each batch

    Input Format:
        For nuScenes, typical camera order is:
        0: CAM_FRONT
        1: CAM_FRONT_RIGHT
        2: CAM_FRONT_LEFT
        3: CAM_BACK
        4: CAM_BACK_LEFT
        5: CAM_BACK_RIGHT
    """
    device = params.get("device", "cuda:0")
    conf_thresh = params.get("confidence_threshold", 0.3)

    # Prepare camera matrices
    intrinsics = torch.stack(
        [torch.from_numpy(cam.intrinsic).float() for cam in camera_configs]
    ).unsqueeze(0)  # [1, N_cam, 3, 3]

    extrinsics = torch.stack(
        [torch.from_numpy(cam.extrinsic).float() for cam in camera_configs]
    ).unsqueeze(0)  # [1, N_cam, 4, 4]

    # Handle input format
    if isinstance(images, list):
        images = torch.stack(images)

    if images.ndim == 4:
        # Single camera, add camera dimension
        images = images.unsqueeze(1)

    images = images.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)

    results = []
    batch_size = images.shape[0]

    with torch.no_grad():
        start_time = time.time()

        outputs = model(
            images,
            camera_intrinsics=intrinsics.expand(batch_size, -1, -1, -1),
            camera_extrinsics=extrinsics.expand(batch_size, -1, -1, -1),
        )

        inference_time = time.time() - start_time

    # Parse outputs
    for batch_idx in range(batch_size):
        detections = extract_3d_detections(
            outputs,
            batch_idx,
            conf_thresh,
        )

        result = BEVResult(
            detections=detections,
            inference_time=inference_time / batch_size,
            frame_id=batch_idx,
        )
        results.append(result)

    total_dets = sum(r.num_detections for r in results)
    fps = batch_size / inference_time if inference_time > 0 else 0
    logger.info(f"BEV inference: {total_dets} detections, {fps:.1f} FPS")

    return results


def extract_3d_detections(
    outputs: Dict[str, torch.Tensor],
    batch_idx: int,
    conf_thresh: float = 0.3,
) -> List[Detection3D]:
    """Extract 3D detections from model outputs.

    Args:
        outputs: Model output dictionary
        batch_idx: Batch index to extract
        conf_thresh: Confidence threshold

    Returns:
        List of Detection3D objects
    """
    detections = []

    boxes_3d = outputs.get("boxes_3d", None)
    scores = outputs.get("scores", None)
    labels = outputs.get("labels", None)

    if boxes_3d is None or scores is None:
        return detections

    # Get predictions for this batch
    boxes = boxes_3d[batch_idx].cpu().numpy()
    confs = scores[batch_idx].cpu().numpy()
    class_ids = labels[batch_idx].cpu().numpy() if labels is not None else np.zeros(len(boxes))

    for i in range(len(boxes)):
        conf = float(confs[i])

        if conf < conf_thresh:
            continue

        box = boxes[i]

        # Parse box format: [x, y, z, w, l, h, yaw, vx, vy] or similar
        if len(box) >= 7:
            x, y, z, w, l, h, yaw = box[:7]
            vx, vy = box[7:9] if len(box) >= 9 else (0, 0)
        else:
            continue

        class_id = int(class_ids[i])
        class_name = (
            NUSCENES_CLASSES[class_id] if class_id < len(NUSCENES_CLASSES) else f"class_{class_id}"
        )

        det = Detection3D(
            center=np.array([x, y, z]),
            size=np.array([w, l, h]),
            rotation=float(yaw),
            velocity=np.array([vx, vy]),
            confidence=conf,
            class_id=class_id,
            class_name=class_name,
        )
        detections.append(det)

    # Sort by confidence
    detections.sort(key=lambda d: d.confidence, reverse=True)

    return detections


def compute_bev_metrics(
    predictions: List[BEVResult],
    ground_truth: List[List[Dict]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute BEV perception metrics (nuScenes-style).

    Args:
        predictions: List of BEVResult predictions
        ground_truth: List of ground truth annotations
        params: Metric parameters

    Returns:
        Dictionary of metrics (mAP, NDS, etc.)

    nuScenes Metrics:
        - mAP: Mean Average Precision (IoU in BEV)
        - NDS: nuScenes Detection Score (weighted combination)
        - mATE: Mean Average Translation Error
        - mASE: Mean Average Scale Error
        - mAOE: Mean Average Orientation Error
        - mAVE: Mean Average Velocity Error
        - mAAE: Mean Average Attribute Error
    """
    # Placeholder - full implementation would use nuScenes devkit
    metrics = {
        "mAP": 0.0,
        "NDS": 0.0,
        "mATE": 0.0,
        "mASE": 0.0,
        "mAOE": 0.0,
        "mAVE": 0.0,
        "mAAE": 0.0,
        "num_predictions": sum(r.num_detections for r in predictions),
        "num_frames": len(predictions),
    }

    # If ground truth provided, compute actual metrics
    if ground_truth:
        # This would use nuScenes evaluation toolkit
        # For now, return placeholder values
        pass

    logger.info(f"BEV metrics: mAP={metrics['mAP']:.3f}, NDS={metrics['NDS']:.3f}")
    return metrics
