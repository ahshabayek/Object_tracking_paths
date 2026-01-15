"""3D Object Tracking Pipeline Nodes.

This module implements 3D multi-object tracking (MOT) for autonomous driving.
3D tracking operates in world coordinates (meters) and tracks objects across
frames using 3D detections from BEV perception or lifted 2D detections.

Supported Trackers:
    - AB3DMOT: Baseline 3D MOT with Kalman filter (IROS 2020)
    - CenterPoint Tracking: CenterPoint's tracking head (CVPR 2021)
    - SimpleTrack: Simple and effective 3D MOT (ICCV 2021)
    - OC-SORT 3D: Observation-centric 3D tracking (CVPR 2023)

Tracking Pipeline Overview:

    3D Detections (t)       Previous Tracks (t-1)
           ↓                        ↓
    ┌──────────────────────────────────────┐
    │        Motion Prediction             │
    │      (Kalman Filter / Velocity)      │
    └──────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────┐
    │         Data Association             │
    │  (3D IoU / Center Distance / GIoU)   │
    └──────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────┐
    │         Track Management             │
    │   (Update / Create / Delete tracks)  │
    └──────────────────────────────────────┘
                      ↓
              Updated Tracks (t)

Key Differences from 2D Tracking:
    - State space: 3D position, velocity, orientation (not pixels)
    - Association metric: 3D IoU or center distance (not 2D IoU)
    - Motion model: Constant velocity in 3D (not 2D Kalman)
    - Requires: Accurate depth information for good tracking

References:
    - AB3DMOT: https://github.com/xinshuoweng/AB3DMOT
    - SimpleTrack: https://github.com/tusen-ai/SimpleTrack
    - CenterPoint: https://github.com/tianweiy/CenterPoint
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrackState:
    """State vector for 3D object tracking.

    The state vector follows the constant velocity model:
        [x, y, z, yaw, l, w, h, vx, vy, vz]

    Coordinate System (Ego Frame):
        - x: forward (meters)
        - y: left (meters)
        - z: up (meters)
        - yaw: rotation around z-axis (radians)
        - l, w, h: length, width, height (meters)
        - vx, vy, vz: velocities (m/s)
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    length: float = 4.0
    width: float = 2.0
    height: float = 1.5
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to state vector."""
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.yaw,
                self.length,
                self.width,
                self.height,
                self.vx,
                self.vy,
                self.vz,
            ]
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "TrackState":
        """Create from state vector."""
        return cls(
            x=vec[0],
            y=vec[1],
            z=vec[2],
            yaw=vec[3],
            length=vec[4],
            width=vec[5],
            height=vec[6],
            vx=vec[7] if len(vec) > 7 else 0.0,
            vy=vec[8] if len(vec) > 8 else 0.0,
            vz=vec[9] if len(vec) > 9 else 0.0,
        )

    @classmethod
    def from_detection(cls, det: Dict[str, Any]) -> "TrackState":
        """Create from detection dictionary.

        Expected format:
            center: [x, y, z]
            size: [w, l, h] or [l, w, h]
            rotation: yaw in radians
            velocity: [vx, vy] (optional)
        """
        center = det["center"]
        size = det["size"]
        return cls(
            x=center[0],
            y=center[1],
            z=center[2],
            yaw=det.get("rotation", 0.0),
            length=size[1] if len(size) > 1 else size[0],
            width=size[0],
            height=size[2] if len(size) > 2 else 1.5,
            vx=det.get("velocity", [0, 0])[0] if det.get("velocity") else 0.0,
            vy=det.get("velocity", [0, 0])[1] if det.get("velocity") else 0.0,
        )

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.length, self.height])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


@dataclass
class Track3D:
    """3D object track.

    Attributes:
        track_id: Unique track identifier
        state: Current track state (position, size, velocity)
        class_id: Object class ID
        class_name: Object class name
        confidence: Detection confidence
        age: Number of frames since track creation
        hits: Number of frames with detections
        time_since_update: Frames since last detection
        history: List of historical states
    """

    track_id: int
    state: TrackState
    class_id: int = 0
    class_name: str = "car"
    confidence: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: List[TrackState] = field(default_factory=list)

    # Kalman filter state (internal)
    kf_state: Optional[np.ndarray] = None
    kf_covariance: Optional[np.ndarray] = None

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if it has enough hits."""
        return self.hits >= 3

    @property
    def is_dead(self) -> bool:
        """Track is dead if not updated for too long."""
        return self.time_since_update > 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "center": self.state.center.tolist(),
            "size": self.state.size.tolist(),
            "rotation": self.state.yaw,
            "velocity": self.state.velocity[:2].tolist(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
        }


@dataclass
class TrackingResult:
    """Container for 3D tracking results.

    Attributes:
        tracks: List of active tracks
        frame_id: Frame identifier
        inference_time: Tracking time in seconds
    """

    tracks: List[Track3D]
    frame_id: int = 0
    inference_time: float = 0.0

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    @property
    def confirmed_tracks(self) -> List[Track3D]:
        return [t for t in self.tracks if t.is_confirmed]


# =============================================================================
# Kalman Filter for 3D Tracking
# =============================================================================


class KalmanFilter3D:
    """Kalman filter for 3D object tracking.

    State Vector (10D):
        [x, y, z, yaw, l, w, h, vx, vy, vz]

    Motion Model:
        - Constant velocity model
        - Position updated by velocity: x' = x + vx * dt
        - Size assumed constant
        - Yaw updated with angular velocity (optional)

    This is the core of AB3DMOT and similar trackers.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float = 0.1,
    ):
        """Initialize Kalman filter.

        Args:
            initial_state: Initial state vector [10,]
            dt: Time step between frames (seconds)
        """
        self.dt = dt
        self.dim_x = 10  # State dimension
        self.dim_z = 7  # Measurement dimension [x, y, z, yaw, l, w, h]

        # State vector
        self.x = np.zeros(self.dim_x)
        self.x[:7] = initial_state[:7]
        if len(initial_state) > 7:
            self.x[7:10] = initial_state[7:10]

        # State covariance
        self.P = np.eye(self.dim_x)
        # High initial uncertainty for velocities
        self.P[7:10, 7:10] *= 100

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 7] = dt  # x += vx * dt
        self.F[1, 8] = dt  # y += vy * dt
        self.F[2, 9] = dt  # z += vz * dt

        # Measurement matrix (observe position and size, not velocity)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:7, :7] = np.eye(7)

        # Process noise (motion uncertainty)
        self.Q = np.eye(self.dim_x)
        self.Q[:3, :3] *= 0.1  # Position
        self.Q[3, 3] *= 0.1  # Yaw
        self.Q[4:7, 4:7] *= 0.01  # Size (nearly constant)
        self.Q[7:10, 7:10] *= 1.0  # Velocity

        # Measurement noise
        self.R = np.eye(self.dim_z)
        self.R[:3, :3] *= 0.5  # Position measurement noise
        self.R[3, 3] *= 0.2  # Yaw
        self.R[4:7, 4:7] *= 0.1  # Size

    def predict(self) -> np.ndarray:
        """Predict next state.

        Returns:
            Predicted state vector
        """
        # State prediction
        self.x = self.F @ self.x

        # Normalize yaw to [-pi, pi]
        self.x[3] = self._normalize_angle(self.x[3])

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement.

        Args:
            measurement: Measurement vector [x, y, z, yaw, l, w, h]

        Returns:
            Updated state vector
        """
        # Handle yaw angle discontinuity
        measurement = measurement.copy()
        measurement[3] = self._normalize_angle(measurement[3])

        # Adjust measurement yaw to be close to predicted yaw
        yaw_diff = measurement[3] - self.x[3]
        if yaw_diff > np.pi:
            measurement[3] -= 2 * np.pi
        elif yaw_diff < -np.pi:
            measurement[3] += 2 * np.pi

        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[3] = self._normalize_angle(self.x[3])

        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()


# =============================================================================
# 3D IoU Computation
# =============================================================================


def compute_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute 3D IoU between two boxes.

    Args:
        box1: First box [x, y, z, l, w, h, yaw]
        box2: Second box [x, y, z, l, w, h, yaw]

    Returns:
        3D IoU value in [0, 1]

    Note:
        This is a simplified 3D IoU that assumes axis-aligned boxes
        for computational efficiency. For rotated boxes, a more
        complex polygon intersection is needed.
    """
    # Extract parameters
    x1, y1, z1, l1, w1, h1, yaw1 = box1[:7]
    x2, y2, z2, l2, w2, h2, yaw2 = box2[:7]

    # Compute axis-aligned bounding boxes (AABB)
    # This is an approximation - proper 3D IoU requires rotated box intersection
    min1 = np.array([x1 - l1 / 2, y1 - w1 / 2, z1 - h1 / 2])
    max1 = np.array([x1 + l1 / 2, y1 + w1 / 2, z1 + h1 / 2])
    min2 = np.array([x2 - l2 / 2, y2 - w2 / 2, z2 - h2 / 2])
    max2 = np.array([x2 + l2 / 2, y2 + w2 / 2, z2 + h2 / 2])

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    # Union
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)


def compute_bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute BEV (Bird's Eye View) IoU between two boxes.

    Args:
        box1: First box [x, y, z, l, w, h, yaw]
        box2: Second box [x, y, z, l, w, h, yaw]

    Returns:
        BEV IoU value in [0, 1]

    Note:
        BEV IoU only considers x-y plane (top-down view).
        More suitable for driving scenarios where height overlap is less important.
    """
    x1, y1, _, l1, w1, _, _ = box1[:7]
    x2, y2, _, l2, w2, _, _ = box2[:7]

    # Axis-aligned BEV boxes
    min1 = np.array([x1 - l1 / 2, y1 - w1 / 2])
    max1 = np.array([x1 + l1 / 2, y1 + w1 / 2])
    min2 = np.array([x2 - l2 / 2, y2 - w2 / 2])
    max2 = np.array([x2 + l2 / 2, y2 + w2 / 2])

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_area = np.prod(inter_size)

    # Union
    area1 = l1 * w1
    area2 = l2 * w2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def compute_center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Euclidean center distance between two boxes.

    Args:
        box1: First box [x, y, z, ...]
        box2: Second box [x, y, z, ...]

    Returns:
        Euclidean distance in meters
    """
    center1 = box1[:3]
    center2 = box2[:3]
    return float(np.linalg.norm(center1 - center2))


# =============================================================================
# AB3DMOT Tracker
# =============================================================================


class AB3DMOTTracker:
    """AB3DMOT: 3D Multi-Object Tracking Baseline (IROS 2020).

    A simple and effective baseline for 3D multi-object tracking.
    Uses Kalman filter for motion prediction and 3D IoU for data association.

    Key Features:
        - Constant velocity Kalman filter
        - 3D IoU-based association (Hungarian algorithm)
        - Track birth/death management
        - No appearance features (pure motion)

    Paper: "3D Multi-Object Tracking: A Baseline and New Evaluation Metrics"
    GitHub: https://github.com/xinshuoweng/AB3DMOT

    Why AB3DMOT is a Strong Baseline:
        1. Simple: Only uses 3D geometry, no learning
        2. Fast: No neural network inference for tracking
        3. Effective: Competitive results on nuScenes/KITTI
        4. Interpretable: Clear failure modes
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.01,
        association_metric: str = "iou_3d",
        dt: float = 0.1,
    ):
        """Initialize AB3DMOT tracker.

        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            association_metric: 'iou_3d', 'iou_bev', or 'center_distance'
            dt: Time step between frames (seconds)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.association_metric = association_metric
        self.dt = dt

        self.tracks: List[Track3D] = []
        self.track_count = 0
        self.frame_count = 0

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Track3D]:
        """Update tracks with new detections.

        Args:
            detections: List of detection dictionaries with:
                - center: [x, y, z]
                - size: [w, l, h]
                - rotation: yaw in radians
                - confidence: detection score
                - class_id: object class
                - velocity: [vx, vy] (optional)

        Returns:
            List of updated tracks
        """
        self.frame_count += 1

        # Step 1: Predict all existing tracks
        for track in self.tracks:
            if track.kf_state is not None:
                kf = self._get_kalman_filter(track)
                predicted_state = kf.predict()
                track.state = TrackState.from_vector(predicted_state)
                track.kf_state = kf.state
                track.kf_covariance = kf.covariance

        # Step 2: Associate detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # Step 3: Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = detections[det_idx]

            # Create measurement from detection
            state = TrackState.from_detection(det)
            measurement = np.array(
                [
                    state.x,
                    state.y,
                    state.z,
                    state.yaw,
                    state.length,
                    state.width,
                    state.height,
                ]
            )

            # Update Kalman filter
            kf = self._get_kalman_filter(track)
            updated_state = kf.update(measurement)
            track.state = TrackState.from_vector(updated_state)
            track.kf_state = kf.state
            track.kf_covariance = kf.covariance

            track.confidence = det.get("confidence", track.confidence)
            track.hits += 1
            track.time_since_update = 0
            track.history.append(track.state)

        # Step 4: Create new tracks from unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det)

        # Step 5: Update unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.time_since_update += 1

        # Step 6: Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead]

        # Step 7: Update age for all tracks
        for track in self.tracks:
            track.age += 1

        return self.tracks

    def _associate(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using Hungarian algorithm.

        Args:
            detections: List of detection dictionaries

        Returns:
            - matched: List of (track_idx, det_idx) tuples
            - unmatched_dets: List of unmatched detection indices
            - unmatched_tracks: List of unmatched track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Build cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets))

        for t_idx, track in enumerate(self.tracks):
            track_box = self._track_to_box(track)
            for d_idx, det in enumerate(detections):
                det_box = self._det_to_box(det)

                if self.association_metric == "iou_3d":
                    iou = compute_3d_iou(track_box, det_box)
                    cost_matrix[t_idx, d_idx] = 1 - iou
                elif self.association_metric == "iou_bev":
                    iou = compute_bev_iou(track_box, det_box)
                    cost_matrix[t_idx, d_idx] = 1 - iou
                else:  # center_distance
                    dist = compute_center_distance(track_box, det_box)
                    cost_matrix[t_idx, d_idx] = dist

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(num_tracks))

        for row, col in zip(row_indices, col_indices):
            cost = cost_matrix[row, col]

            # Check if association is valid
            if self.association_metric in ["iou_3d", "iou_bev"]:
                if 1 - cost >= self.iou_threshold:
                    matched.append((row, col))
                    unmatched_dets.remove(col)
                    unmatched_tracks.remove(row)
            else:  # center_distance
                if cost <= 2.0:  # 2 meters threshold
                    matched.append((row, col))
                    unmatched_dets.remove(col)
                    unmatched_tracks.remove(row)

        return matched, unmatched_dets, unmatched_tracks

    def _create_track(self, det: Dict[str, Any]) -> Track3D:
        """Create a new track from detection."""
        self.track_count += 1

        state = TrackState.from_detection(det)

        track = Track3D(
            track_id=self.track_count,
            state=state,
            class_id=det.get("class_id", 0),
            class_name=det.get("class_name", "car"),
            confidence=det.get("confidence", 0.5),
            age=1,
            hits=1,
            time_since_update=0,
            history=[state],
        )

        # Initialize Kalman filter
        initial_state = state.to_vector()
        kf = KalmanFilter3D(initial_state, dt=self.dt)
        track.kf_state = kf.state
        track.kf_covariance = kf.covariance

        self.tracks.append(track)
        return track

    def _get_kalman_filter(self, track: Track3D) -> KalmanFilter3D:
        """Reconstruct Kalman filter from track state."""
        kf = KalmanFilter3D(track.kf_state, dt=self.dt)
        kf.x = track.kf_state.copy()
        kf.P = track.kf_covariance.copy()
        return kf

    @staticmethod
    def _track_to_box(track: Track3D) -> np.ndarray:
        """Convert track to box format [x, y, z, l, w, h, yaw]."""
        s = track.state
        return np.array([s.x, s.y, s.z, s.length, s.width, s.height, s.yaw])

    @staticmethod
    def _det_to_box(det: Dict[str, Any]) -> np.ndarray:
        """Convert detection to box format [x, y, z, l, w, h, yaw]."""
        center = det["center"]
        size = det["size"]
        yaw = det.get("rotation", 0.0)
        return np.array(
            [
                center[0],
                center[1],
                center[2],
                size[1] if len(size) > 1 else size[0],  # length
                size[0],  # width
                size[2] if len(size) > 2 else 1.5,  # height
                yaw,
            ]
        )


# =============================================================================
# SimpleTrack 3D Tracker
# =============================================================================


class SimpleTrack3D:
    """SimpleTrack: A Simple Baseline for 3D MOT (ICCV 2021).

    An improved baseline over AB3DMOT with better association strategies.

    Key Improvements over AB3DMOT:
        1. Two-stage association (high/low confidence)
        2. Velocity-based motion prediction
        3. Better track management

    Paper: "SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking"
    GitHub: https://github.com/tusen-ai/SimpleTrack
    """

    def __init__(
        self,
        high_score_threshold: float = 0.6,
        low_score_threshold: float = 0.1,
        max_age: int = 5,
        min_hits: int = 3,
        dt: float = 0.1,
    ):
        """Initialize SimpleTrack.

        Args:
            high_score_threshold: Threshold for first-stage association
            low_score_threshold: Threshold for second-stage association
            max_age: Maximum frames without update
            min_hits: Minimum hits to confirm track
            dt: Time step between frames
        """
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt

        self.tracks: List[Track3D] = []
        self.track_count = 0
        self.frame_count = 0

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Track3D]:
        """Update tracks with new detections.

        Two-stage association:
            1. First match high-confidence detections
            2. Then match remaining tracks with low-confidence detections
        """
        self.frame_count += 1

        # Predict all tracks
        for track in self.tracks:
            self._predict_track(track)

        # Split detections by confidence
        high_dets = [d for d in detections if d.get("confidence", 0) >= self.high_score_threshold]
        low_dets = [d for d in detections if d.get("confidence", 0) < self.high_score_threshold]
        low_dets = [d for d in low_dets if d.get("confidence", 0) >= self.low_score_threshold]

        # First stage: match with high-confidence detections
        matched_first, unmatched_dets_first, unmatched_tracks = self._associate(
            high_dets, self.tracks
        )

        # Update matched tracks
        matched_tracks = set()
        for track_idx, det_idx in matched_first:
            self._update_track(self.tracks[track_idx], high_dets[det_idx])
            matched_tracks.add(track_idx)

        # Second stage: match remaining tracks with low-confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        matched_second, unmatched_dets_second, still_unmatched = self._associate(
            low_dets, remaining_tracks
        )

        for track_list_idx, det_idx in matched_second:
            track_idx = unmatched_tracks[track_list_idx]
            self._update_track(self.tracks[track_idx], low_dets[det_idx])
            matched_tracks.add(track_idx)

        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets_first:
            self._create_track(high_dets[det_idx])

        # Update unmatched tracks
        for track_idx in range(len(self.tracks)):
            if track_idx not in matched_tracks:
                self.tracks[track_idx].time_since_update += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead]

        # Update age
        for track in self.tracks:
            track.age += 1

        return self.tracks

    def _predict_track(self, track: Track3D):
        """Predict track state using velocity."""
        s = track.state
        s.x += s.vx * self.dt
        s.y += s.vy * self.dt
        s.z += s.vz * self.dt

    def _update_track(self, track: Track3D, det: Dict[str, Any]):
        """Update track with detection."""
        new_state = TrackState.from_detection(det)

        # Update velocity estimate
        dt = self.dt
        track.state.vx = (new_state.x - track.state.x) / dt if dt > 0 else 0
        track.state.vy = (new_state.y - track.state.y) / dt if dt > 0 else 0

        # Update position and size
        track.state.x = new_state.x
        track.state.y = new_state.y
        track.state.z = new_state.z
        track.state.yaw = new_state.yaw
        track.state.length = new_state.length
        track.state.width = new_state.width
        track.state.height = new_state.height

        track.confidence = det.get("confidence", track.confidence)
        track.hits += 1
        track.time_since_update = 0
        track.history.append(track.state)

    def _create_track(self, det: Dict[str, Any]) -> Track3D:
        """Create a new track from detection."""
        self.track_count += 1

        state = TrackState.from_detection(det)
        track = Track3D(
            track_id=self.track_count,
            state=state,
            class_id=det.get("class_id", 0),
            class_name=det.get("class_name", "car"),
            confidence=det.get("confidence", 0.5),
            age=1,
            hits=1,
            time_since_update=0,
            history=[state],
        )

        self.tracks.append(track)
        return track

    def _associate(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Track3D],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using 3D IoU."""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))

        # Build IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t_idx, track in enumerate(tracks):
            track_box = np.array(
                [
                    track.state.x,
                    track.state.y,
                    track.state.z,
                    track.state.length,
                    track.state.width,
                    track.state.height,
                    track.state.yaw,
                ]
            )
            for d_idx, det in enumerate(detections):
                det_box = np.array(
                    [
                        det["center"][0],
                        det["center"][1],
                        det["center"][2],
                        det["size"][1] if len(det["size"]) > 1 else det["size"][0],
                        det["size"][0],
                        det["size"][2] if len(det["size"]) > 2 else 1.5,
                        det.get("rotation", 0.0),
                    ]
                )
                iou_matrix[t_idx, d_idx] = compute_3d_iou(track_box, det_box)

        # Hungarian algorithm (minimize negative IoU)
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= 0.01:
                matched.append((row, col))
                if col in unmatched_dets:
                    unmatched_dets.remove(col)
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)

        return matched, unmatched_dets, unmatched_tracks


# =============================================================================
# Tracker Factory
# =============================================================================


class Tracker3DFactory:
    """Factory for creating 3D object trackers.

    Supported Trackers:
        - ab3dmot: AB3DMOT (IROS 2020) - Kalman + 3D IoU
        - simpletrack: SimpleTrack (ICCV 2021) - Two-stage association
        - centerpoint: CenterPoint tracking (CVPR 2021)
        - ocsort3d: OC-SORT 3D (CVPR 2023)

    Performance Comparison (nuScenes val):
        - AB3DMOT: ~65% AMOTA
        - SimpleTrack: ~70% AMOTA
        - CenterPoint: ~66% AMOTA
    """

    SUPPORTED_TRACKERS = ["ab3dmot", "simpletrack", "centerpoint", "ocsort3d"]

    @staticmethod
    def create(
        tracker_name: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Create a 3D tracker.

        Args:
            tracker_name: Name of tracker
            params: Tracker parameters

        Returns:
            Tracker instance
        """
        params = params or {}
        tracker_name = tracker_name.lower()

        if tracker_name == "ab3dmot":
            return AB3DMOTTracker(
                max_age=params.get("max_age", 5),
                min_hits=params.get("min_hits", 3),
                iou_threshold=params.get("iou_threshold", 0.01),
                association_metric=params.get("association_metric", "iou_3d"),
                dt=params.get("dt", 0.1),
            )
        elif tracker_name == "simpletrack":
            return SimpleTrack3D(
                high_score_threshold=params.get("high_score_threshold", 0.6),
                low_score_threshold=params.get("low_score_threshold", 0.1),
                max_age=params.get("max_age", 5),
                min_hits=params.get("min_hits", 3),
                dt=params.get("dt", 0.1),
            )
        elif tracker_name in ["centerpoint", "ocsort3d"]:
            # These would require additional implementation
            logger.warning(
                f"{tracker_name} uses simplified implementation. "
                "For full version, install official packages."
            )
            return AB3DMOTTracker(
                max_age=params.get("max_age", 5),
                min_hits=params.get("min_hits", 3),
                iou_threshold=params.get("iou_threshold", 0.01),
                dt=params.get("dt", 0.1),
            )
        else:
            raise ValueError(
                f"Unknown tracker: {tracker_name}. Supported: {Tracker3DFactory.SUPPORTED_TRACKERS}"
            )


# =============================================================================
# Node Functions
# =============================================================================


def create_tracker(params: Dict[str, Any]):
    """Create a 3D object tracker.

    Args:
        params: Tracker configuration

    Returns:
        Tracker instance

    Example:
        params = {
            "tracker": "ab3dmot",
            "max_age": 5,
            "min_hits": 3,
            "iou_threshold": 0.01,
            "dt": 0.1,
        }
        tracker = create_tracker(params)
    """
    tracker_name = params.get("tracker", "ab3dmot")
    tracker = Tracker3DFactory.create(tracker_name, params)
    logger.info(f"Created 3D tracker: {tracker_name}")
    return tracker


def track_objects_3d(
    tracker,
    detections: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> TrackingResult:
    """Track objects in 3D.

    Args:
        tracker: 3D tracker instance
        detections: List of 3D detections
        params: Tracking parameters

    Returns:
        TrackingResult with updated tracks

    Detection Format:
        {
            "center": [x, y, z],  # meters
            "size": [w, l, h],    # meters
            "rotation": yaw,      # radians
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "car",
            "velocity": [vx, vy], # optional, m/s
        }
    """
    start_time = time.time()

    # Convert Detection3D objects to dicts if needed
    det_dicts = []
    for det in detections:
        if hasattr(det, "to_dict"):
            det_dicts.append(det.to_dict())
        elif isinstance(det, dict):
            det_dicts.append(det)
        else:
            # Assume it has center, size, rotation attributes
            det_dicts.append(
                {
                    "center": det.center.tolist() if hasattr(det.center, "tolist") else det.center,
                    "size": det.size.tolist() if hasattr(det.size, "tolist") else det.size,
                    "rotation": det.rotation,
                    "confidence": getattr(det, "confidence", 0.5),
                    "class_id": getattr(det, "class_id", 0),
                    "class_name": getattr(det, "class_name", "car"),
                    "velocity": (
                        det.velocity.tolist()
                        if hasattr(det, "velocity") and det.velocity is not None
                        else [0, 0]
                    ),
                }
            )

    # Update tracker
    tracks = tracker.update(det_dicts)

    inference_time = time.time() - start_time

    result = TrackingResult(
        tracks=tracks,
        frame_id=tracker.frame_count,
        inference_time=inference_time,
    )

    num_confirmed = len(result.confirmed_tracks)
    logger.debug(
        f"Tracking: {len(detections)} dets -> {result.num_tracks} tracks "
        f"({num_confirmed} confirmed), {inference_time * 1000:.1f}ms"
    )

    return result


def compute_tracking_metrics(
    predictions: List[TrackingResult],
    ground_truth: List[List[Dict]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute 3D tracking metrics (nuScenes-style).

    Args:
        predictions: List of tracking results
        ground_truth: List of ground truth annotations per frame
        params: Metric parameters

    Returns:
        Dictionary of metrics

    nuScenes Tracking Metrics:
        - AMOTA: Average Multi-Object Tracking Accuracy
        - AMOTP: Average Multi-Object Tracking Precision
        - RECALL: Recall at various thresholds
        - MOTAR: MOTA at various recall levels
        - IDS: Number of ID switches
        - FRAG: Number of fragmentations
    """
    metrics = {
        "AMOTA": 0.0,
        "AMOTP": 0.0,
        "RECALL": 0.0,
        "MOTAR": 0.0,
        "IDS": 0,
        "FRAG": 0,
        "num_frames": len(predictions),
        "total_tracks": sum(r.num_tracks for r in predictions),
        "avg_tracks_per_frame": (
            sum(r.num_tracks for r in predictions) / len(predictions) if predictions else 0
        ),
    }

    # Full metric computation would use nuscenes-devkit
    # This is a placeholder for the interface

    if ground_truth:
        # Compute actual metrics
        pass

    logger.info(
        f"Tracking metrics: AMOTA={metrics['AMOTA']:.3f}, "
        f"Avg tracks/frame={metrics['avg_tracks_per_frame']:.1f}"
    )

    return metrics
