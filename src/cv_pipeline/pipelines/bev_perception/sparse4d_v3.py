"""Sparse4D v3: Unified 3D Detection and Tracking.

This module implements Sparse4D v3, which unifies 3D object detection and
tracking in a single end-to-end framework using 4D anchor propagation.

Sparse4D v3 Key Innovations:
    1. 4D Anchor Boxes: 3D boxes with temporal extent (position + time)
    2. Sparse Temporal Fusion: Efficient cross-frame feature aggregation
    3. End-to-End Tracking: Track IDs from anchor propagation, no post-processing
    4. Iterative Refinement: Progressive anchor box updates

Architecture:

    Frame t-1                           Frame t
    ┌─────────────┐                    ┌─────────────┐
    │  4D Anchors │ ──── Propagate ──→ │  4D Anchors │
    │  (N boxes)  │     (warp by ego)  │  (N boxes)  │
    └─────────────┘                    └─────────────┘
           ↓                                  ↓
    ┌─────────────┐                    ┌─────────────┐
    │   Feature   │                    │   Feature   │
    │  Sampling   │                    │  Sampling   │
    └─────────────┘                    └─────────────┘
           ↓                                  ↓
    ┌─────────────────────────────────────────────────┐
    │         Sparse Temporal Cross-Attention         │
    │     (fuse features across time for each box)    │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │              Iterative Box Refinement           │
    │         (update box params: x,y,z,w,l,h,θ)      │
    └─────────────────────────────────────────────────┘
                            ↓
              3D Boxes with Track IDs

Tracking via Anchor Propagation:
    - Each anchor maintains identity across frames
    - New anchors initialized for new objects
    - Dead anchors removed based on confidence
    - No Hungarian matching needed (implicit association)

Performance (nuScenes val):
    - 71.9% NDS, 67.0% mAP
    - 20+ FPS on single GPU

References:
    Paper: "Sparse4D v3: Unified Detection and Tracking"
    GitHub: https://github.com/linxuewu/Sparse4D
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Anchor4D:
    """4D Anchor box with temporal extent.

    Represents a tracked object as a 4D anchor (3D box + time).
    The anchor is propagated across frames to maintain object identity.

    Attributes:
        position: Center [x, y, z] in ego frame (meters)
        size: Box dimensions [w, l, h] (meters)
        rotation: Yaw angle (radians)
        velocity: Velocity [vx, vy] (m/s)
        track_id: Unique track identifier
        confidence: Detection confidence
        class_id: Object class
        age: Number of frames since creation
        features: Learned anchor features
    """

    position: np.ndarray  # [x, y, z]
    size: np.ndarray  # [w, l, h]
    rotation: float  # yaw
    velocity: np.ndarray  # [vx, vy]
    track_id: int
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "car"
    age: int = 0
    features: Optional[np.ndarray] = None  # [C,] anchor features

    def propagate(self, ego_motion: np.ndarray, dt: float = 0.1) -> "Anchor4D":
        """Propagate anchor to next frame using ego motion.

        Args:
            ego_motion: [4, 4] ego transformation matrix (t-1 to t)
            dt: Time step in seconds

        Returns:
            Propagated anchor
        """
        # Apply velocity to position
        new_pos = self.position.copy()
        new_pos[0] += self.velocity[0] * dt
        new_pos[1] += self.velocity[1] * dt

        # Transform by ego motion (compensate for ego movement)
        pos_homo = np.append(new_pos, 1.0)
        new_pos = (ego_motion @ pos_homo)[:3]

        # Transform rotation by ego motion
        ego_yaw = np.arctan2(ego_motion[1, 0], ego_motion[0, 0])
        new_rot = self.rotation - ego_yaw

        return Anchor4D(
            position=new_pos,
            size=self.size.copy(),
            rotation=new_rot,
            velocity=self.velocity.copy(),
            track_id=self.track_id,
            confidence=self.confidence * 0.9,  # Decay confidence
            class_id=self.class_id,
            class_name=self.class_name,
            age=self.age + 1,
            features=self.features,
        )

    def update(self, detection: Dict[str, Any]) -> "Anchor4D":
        """Update anchor with new detection.

        Args:
            detection: Detection dict with center, size, rotation, etc.

        Returns:
            Updated anchor
        """
        # Compute velocity from position change
        new_pos = np.array(detection["center"])
        velocity = (new_pos[:2] - self.position[:2]) / 0.1  # Assume 10 Hz

        return Anchor4D(
            position=new_pos,
            size=np.array(detection["size"]),
            rotation=detection.get("rotation", self.rotation),
            velocity=velocity,
            track_id=self.track_id,
            confidence=detection.get("confidence", 0.5),
            class_id=detection.get("class_id", self.class_id),
            class_name=detection.get("class_name", self.class_name),
            age=self.age,
            features=self.features,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.position.tolist(),
            "size": self.size.tolist(),
            "rotation": self.rotation,
            "velocity": self.velocity.tolist(),
            "track_id": self.track_id,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "age": self.age,
        }

    @property
    def is_valid(self) -> bool:
        """Check if anchor is still valid (high enough confidence)."""
        return self.confidence > 0.1 and self.age < 10


@dataclass
class Sparse4DConfig:
    """Configuration for Sparse4D v3 model.

    Attributes:
        num_anchors: Number of 4D anchors (max tracked objects)
        hidden_dim: Feature dimension
        num_refinement_stages: Iterative refinement stages
        num_history_frames: Temporal context frames
        confidence_threshold: Detection threshold
        anchor_range: 3D range for anchor initialization
    """

    num_anchors: int = 900
    hidden_dim: int = 256
    num_refinement_stages: int = 6
    num_history_frames: int = 4
    confidence_threshold: float = 0.3
    anchor_range: Tuple[float, float, float, float, float, float] = (
        -51.2,
        51.2,  # x
        -51.2,
        51.2,  # y
        -5.0,
        3.0,  # z
    )
    device: str = "cuda:0"


@dataclass
class Sparse4DResult:
    """Result from Sparse4D v3 inference.

    Contains both detections and tracks (unified).
    """

    anchors: List[Anchor4D]  # Active 4D anchors (tracked objects)
    inference_time: float = 0.0
    frame_id: int = 0

    @property
    def num_objects(self) -> int:
        return len(self.anchors)

    @property
    def detections(self) -> List[Dict[str, Any]]:
        """Get detections in standard format."""
        return [a.to_dict() for a in self.anchors if a.confidence > 0.3]

    @property
    def tracks(self) -> List[Dict[str, Any]]:
        """Get tracks in standard format (same as detections with track_id)."""
        return self.detections


# =============================================================================
# Sparse4D v3 Tracker
# =============================================================================


class Sparse4DTracker:
    """Sparse4D v3 Unified Detection and Tracking.

    This class implements the Sparse4D v3 tracking paradigm where
    detection and tracking are unified through 4D anchor propagation.

    Key Differences from Traditional Tracking:
        1. No post-hoc association (Hungarian matching)
        2. Anchors maintain identity across frames
        3. End-to-end learned feature propagation
        4. Velocity and motion implicitly modeled

    Usage:
        tracker = Sparse4DTracker(config)
        for frame in frames:
            result = tracker.update(detections, ego_motion)
            # result.anchors contains tracked objects with IDs
    """

    def __init__(self, config: Sparse4DConfig):
        """Initialize Sparse4D tracker.

        Args:
            config: Sparse4D configuration
        """
        self.config = config
        self.anchors: List[Anchor4D] = []
        self.next_track_id = 1
        self.frame_count = 0

        # Anchor initialization grid
        self._init_anchor_grid()

    def _init_anchor_grid(self):
        """Initialize anchor grid for new object detection."""
        # Create initial anchor positions spanning the detection range
        x_range = self.config.anchor_range[:2]
        y_range = self.config.anchor_range[2:4]

        # Grid of initial positions
        xs = np.linspace(x_range[0], x_range[1], 10)
        ys = np.linspace(y_range[0], y_range[1], 10)

        self.init_positions = []
        for x in xs:
            for y in ys:
                self.init_positions.append(np.array([x, y, 0.0]))

    def reset(self):
        """Reset tracker state."""
        self.anchors = []
        self.next_track_id = 1
        self.frame_count = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        ego_motion: Optional[np.ndarray] = None,
    ) -> Sparse4DResult:
        """Update tracks with new detections.

        In Sparse4D v3, this simulates the anchor propagation and update
        that happens inside the neural network.

        Args:
            detections: List of detection dicts
            ego_motion: [4, 4] ego transformation matrix

        Returns:
            Sparse4DResult with updated anchors
        """
        start_time = time.time()
        self.frame_count += 1

        # Default ego motion (identity)
        if ego_motion is None:
            ego_motion = np.eye(4)

        # Step 1: Propagate existing anchors
        propagated_anchors = []
        for anchor in self.anchors:
            prop_anchor = anchor.propagate(ego_motion)
            if prop_anchor.is_valid:
                propagated_anchors.append(prop_anchor)

        # Step 2: Match detections to anchors (simplified - real Sparse4D uses learned matching)
        matched_anchor_ids = set()
        matched_det_ids = set()

        for det_idx, det in enumerate(detections):
            det_pos = np.array(det["center"])

            best_anchor_idx = None
            best_dist = float("inf")

            for anc_idx, anchor in enumerate(propagated_anchors):
                if anc_idx in matched_anchor_ids:
                    continue

                dist = np.linalg.norm(det_pos[:2] - anchor.position[:2])
                if dist < best_dist and dist < 2.0:  # 2m threshold
                    best_dist = dist
                    best_anchor_idx = anc_idx

            if best_anchor_idx is not None:
                # Update matched anchor
                propagated_anchors[best_anchor_idx] = propagated_anchors[best_anchor_idx].update(
                    det
                )
                matched_anchor_ids.add(best_anchor_idx)
                matched_det_ids.add(det_idx)

        # Step 3: Create new anchors for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_ids:
                continue

            if det.get("confidence", 0) < self.config.confidence_threshold:
                continue

            new_anchor = Anchor4D(
                position=np.array(det["center"]),
                size=np.array(det["size"]),
                rotation=det.get("rotation", 0.0),
                velocity=np.array(det.get("velocity", [0.0, 0.0])),
                track_id=self.next_track_id,
                confidence=det.get("confidence", 0.5),
                class_id=det.get("class_id", 0),
                class_name=det.get("class_name", "car"),
                age=0,
            )
            propagated_anchors.append(new_anchor)
            self.next_track_id += 1

        # Step 4: Filter low confidence anchors
        self.anchors = [a for a in propagated_anchors if a.is_valid]

        inference_time = time.time() - start_time

        return Sparse4DResult(
            anchors=self.anchors,
            inference_time=inference_time,
            frame_id=self.frame_count,
        )


# =============================================================================
# Sparse4D v3 Neural Network (Simplified)
# =============================================================================


class Sparse4DHead(nn.Module):
    """Sparse4D v3 detection and tracking head.

    This implements the core Sparse4D architecture for unified
    detection and tracking through anchor propagation.
    """

    def __init__(self, config: Sparse4DConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Anchor embedding
        self.anchor_embed = nn.Parameter(torch.randn(config.num_anchors, hidden_dim))

        # Feature sampling network
        self.sampling_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8 * 3),  # 8 sampling points, xyz
        )

        # Temporal fusion
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        # Iterative refinement
        self.refine_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(config.num_refinement_stages)
            ]
        )

        # Output heads
        self.box_head = nn.Linear(hidden_dim, 10)  # x,y,z,w,l,h,yaw,vx,vy,conf
        self.cls_head = nn.Linear(hidden_dim, 10)  # 10 classes

    def forward(
        self,
        bev_features: torch.Tensor,
        prev_anchors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            bev_features: [B, C, H, W] BEV features
            prev_anchors: [B, N, C] previous anchor features

        Returns:
            Dictionary with boxes, scores, and updated anchors
        """
        B = bev_features.shape[0]

        # Initialize or propagate anchors
        if prev_anchors is not None:
            anchors = prev_anchors
        else:
            anchors = self.anchor_embed.unsqueeze(0).expand(B, -1, -1)

        # Flatten BEV features
        bev_flat = bev_features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Temporal fusion with BEV context
        anchors, _ = self.temporal_attn(anchors, bev_flat, bev_flat)

        # Iterative refinement
        for refine_layer in self.refine_layers:
            anchors = anchors + refine_layer(anchors)

        # Predict boxes and classes
        box_preds = self.box_head(anchors)  # [B, N, 10]
        cls_preds = self.cls_head(anchors)  # [B, N, 10]

        return {
            "boxes": box_preds,
            "classes": cls_preds,
            "anchor_features": anchors,
        }


class Sparse4DModel(nn.Module):
    """Complete Sparse4D v3 model for detection and tracking.

    Combines BEV encoding with Sparse4D head for end-to-end
    3D detection and tracking.
    """

    def __init__(self, config: Sparse4DConfig):
        super().__init__()
        self.config = config

        # Simple BEV encoder (placeholder)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(),
        )

        # Sparse4D head
        self.head = Sparse4DHead(config)

        # Track anchor features across frames
        self.register_buffer("prev_anchors", None)

    def forward(
        self,
        images: torch.Tensor,
        reset_tracking: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with tracking.

        Args:
            images: [B, N_cam, C, H, W] multi-camera images
            reset_tracking: Whether to reset track IDs

        Returns:
            Dictionary with detections and track features
        """
        B, N, C, H, W = images.shape

        # Process first camera (simplified - real uses all cameras)
        img_feat = self.bev_encoder(images[:, 0])

        # Get previous anchors
        prev_anchors = None if reset_tracking else self.prev_anchors

        # Run detection + tracking head
        outputs = self.head(img_feat, prev_anchors)

        # Store anchors for next frame
        self.prev_anchors = outputs["anchor_features"].detach()

        return outputs


# =============================================================================
# Node Functions
# =============================================================================


def create_sparse4d_tracker(params: Dict[str, Any]) -> Sparse4DTracker:
    """Create a Sparse4D v3 tracker.

    Args:
        params: Configuration parameters

    Returns:
        Sparse4DTracker instance

    Example:
        tracker = create_sparse4d_tracker({
            "num_anchors": 900,
            "confidence_threshold": 0.3,
        })
    """
    config = Sparse4DConfig(
        num_anchors=params.get("num_anchors", 900),
        hidden_dim=params.get("hidden_dim", 256),
        confidence_threshold=params.get("confidence_threshold", 0.3),
        device=params.get("device", "cuda:0"),
    )

    tracker = Sparse4DTracker(config)
    logger.info(f"Created Sparse4D v3 tracker with {config.num_anchors} anchors")
    return tracker


def run_sparse4d_tracking(
    tracker: Sparse4DTracker,
    detections: List[Dict[str, Any]],
    ego_motion: Optional[np.ndarray] = None,
) -> Sparse4DResult:
    """Run Sparse4D v3 tracking update.

    Args:
        tracker: Sparse4D tracker instance
        detections: List of 3D detections
        ego_motion: Optional ego motion transformation

    Returns:
        Sparse4DResult with tracked objects
    """
    result = tracker.update(detections, ego_motion)

    logger.debug(
        f"Sparse4D tracking: {len(detections)} dets -> "
        f"{result.num_objects} tracks, {result.inference_time * 1000:.1f}ms"
    )

    return result


def load_sparse4d_model(params: Dict[str, Any]) -> Sparse4DModel:
    """Load Sparse4D v3 neural network model.

    Args:
        params: Model parameters

    Returns:
        Loaded Sparse4D model

    Example:
        model = load_sparse4d_model({
            "checkpoint": "weights/sparse4d_v3.pth",
            "device": "cuda:0",
        })
    """
    config = Sparse4DConfig(
        num_anchors=params.get("num_anchors", 900),
        hidden_dim=params.get("hidden_dim", 256),
        device=params.get("device", "cuda:0"),
    )

    model = Sparse4DModel(config)

    # Load checkpoint if provided
    checkpoint_path = params.get("checkpoint", None)
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded Sparse4D checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    device = config.device
    model = model.to(device)
    model.eval()

    logger.info("Sparse4D v3 model loaded")
    return model
