"""Multi-Object Tracking Pipeline Nodes.

This module contains node functions for multi-object tracking using:
- BoT-SORT (best accuracy with Re-ID and CMC)
- ByteTrack (fast, uses low confidence detections)
- OC-SORT (observation-centric)
- DeepSORT (classic with deep Re-ID features)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    
    # Track state
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    
    # Motion and appearance
    velocity: Optional[np.ndarray] = None
    feature: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_id": self.frame_id,
            "state": self.state,
        }


@dataclass
class TrackingResult:
    """Container for tracking results on a single frame."""
    frame_id: int
    tracks: List[Track]
    processing_time: float


class KalmanBoxTracker:
    """Kalman Filter-based bounding box tracker."""
    
    count = 0
    
    def __init__(self, bbox: np.ndarray, class_id: int = 0, confidence: float = 1.0):
        """Initialize tracker with bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            class_id: Object class ID
            confidence: Detection confidence
        """
        # Initialize Kalman filter
        # State: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.class_id = class_id
        self.confidence = confidence
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = []
        self.feature = None
        
    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox [x1, y1, x2, y2] to [cx, cy, aspect, height]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        s = w / h if h > 0 else 1
        return np.array([[cx], [cy], [s], [h]])
    
    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, aspect, height] to bbox [x1, y1, x2, y2]."""
        cx, cy, s, h = z.flatten()
        w = s * h
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    def update(self, bbox: np.ndarray, confidence: float = 1.0):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.kf.update(self._bbox_to_z(bbox))
        
    def predict(self) -> np.ndarray:
        """Predict next state and return bbox."""
        if self.kf.x[7] + self.kf.x[3] <= 0:
            self.kf.x[7] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self.get_state())
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current bbox state."""
        return self._z_to_bbox(self.kf.x[:4])


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.
    
    Args:
        bb_test: Bounding boxes [N, 4]
        bb_gt: Ground truth boxes [M, 4]
    
    Returns:
        IoU matrix [N, M]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    
    intersection = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    union = area_test + area_gt - intersection
    
    iou = intersection / (union + 1e-6)
    
    return iou


def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear assignment problem.
    
    Args:
        cost_matrix: Cost matrix [N, M]
    
    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = np.column_stack([row_indices, col_indices])
    
    unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), row_indices)
    unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), col_indices)
    
    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack implementation for multi-object tracking."""
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize ByteTracker.
        
        Args:
            params: Tracking parameters
        """
        self.params = params.get("bytetrack", {})
        
        self.track_high_thresh = params.get("track_high_thresh", 0.5)
        self.track_low_thresh = params.get("track_low_thresh", 0.1)
        self.new_track_thresh = params.get("new_track_thresh", 0.6)
        self.track_buffer = params.get("track_buffer", 30)
        self.match_thresh = params.get("match_thresh", 0.8)
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        KalmanBoxTracker.count = 0
        
    def update(self, detections: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Update tracker with new detections.
        
        Args:
            detections: Detection boxes [N, 4]
            scores: Detection scores [N]
            classes: Detection classes [N]
        
        Returns:
            Array of active tracks [M, 5] (x1, y1, x2, y2, track_id)
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)
        
        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        high_classes = classes[high_mask]
        
        low_dets = detections[low_mask]
        low_scores = scores[low_mask]
        low_classes = classes[low_mask]
        
        # Predict new locations of existing trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Get predicted boxes
        if self.trackers:
            trk_boxes = np.array([t.get_state() for t in self.trackers])
        else:
            trk_boxes = np.empty((0, 4))
        
        # First association with high confidence detections
        if len(high_dets) > 0 and len(trk_boxes) > 0:
            iou_matrix = iou_batch(high_dets, trk_boxes)
            cost_matrix = 1 - iou_matrix
            
            matches, unmatched_dets, unmatched_trks = linear_assignment(cost_matrix)
            
            # Filter matches with low IoU
            valid_matches = []
            for m in matches:
                if iou_matrix[m[0], m[1]] >= self.match_thresh:
                    valid_matches.append(m)
                else:
                    unmatched_dets = np.append(unmatched_dets, m[0])
                    unmatched_trks = np.append(unmatched_trks, m[1])
            
            matches = np.array(valid_matches) if valid_matches else np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(high_dets))
            unmatched_trks = np.arange(len(self.trackers))
        
        # Update matched trackers
        for m in matches:
            self.trackers[m[1]].update(high_dets[m[0]], high_scores[m[0]])
        
        # Second association with low confidence detections
        if len(low_dets) > 0 and len(unmatched_trks) > 0:
            remaining_trks = [self.trackers[i] for i in unmatched_trks]
            remaining_boxes = np.array([t.get_state() for t in remaining_trks])
            
            iou_matrix = iou_batch(low_dets, remaining_boxes)
            cost_matrix = 1 - iou_matrix
            
            matches2, unmatched_dets2, unmatched_trks2 = linear_assignment(cost_matrix)
            
            for m in matches2:
                if iou_matrix[m[0], m[1]] >= 0.5:  # Lower threshold for second match
                    remaining_trks[m[1]].update(low_dets[m[0]], low_scores[m[0]])
                else:
                    unmatched_trks2 = np.append(unmatched_trks2, m[1])
            
            # Update unmatched_trks to reflect remaining unmatched
            unmatched_trks = np.array([unmatched_trks[i] for i in unmatched_trks2])
        
        # Create new trackers for unmatched high confidence detections
        for i in unmatched_dets:
            if high_scores[i] >= self.new_track_thresh:
                tracker = KalmanBoxTracker(
                    high_dets[i],
                    class_id=int(high_classes[i]),
                    confidence=high_scores[i]
                )
                self.trackers.append(tracker)
        
        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.track_buffer
        ]
        
        # Get outputs
        outputs = []
        for tracker in self.trackers:
            if tracker.time_since_update == 0 and tracker.hits >= 3:
                bbox = tracker.get_state()
                outputs.append([*bbox, tracker.id, tracker.class_id, tracker.confidence])
        
        return np.array(outputs) if outputs else np.empty((0, 7))


class BoTSORTTracker:
    """BoT-SORT implementation with Re-ID and Camera Motion Compensation."""
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize BoT-SORT tracker.
        
        Args:
            params: Tracking parameters
        """
        self.params = params.get("botsort", {})
        
        self.track_high_thresh = params.get("track_high_thresh", 0.5)
        self.track_low_thresh = params.get("track_low_thresh", 0.1)
        self.new_track_thresh = params.get("new_track_thresh", 0.6)
        self.track_buffer = params.get("track_buffer", 30)
        self.match_thresh = params.get("match_thresh", 0.8)
        
        self.proximity_thresh = self.params.get("proximity_thresh", 0.5)
        self.appearance_thresh = self.params.get("appearance_thresh", 0.25)
        self.with_reid = self.params.get("with_reid", True)
        self.lambda_ = self.params.get("lambda_", 0.98)
        self.ema_alpha = self.params.get("ema_alpha", 0.9)
        
        # Initialize base ByteTracker
        self.byte_tracker = ByteTracker(params)
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        # Re-ID model placeholder
        self.reid_model = None
        
        KalmanBoxTracker.count = 0
    
    def _compute_appearance_cost(
        self,
        det_features: np.ndarray,
        track_features: np.ndarray,
    ) -> np.ndarray:
        """Compute appearance cost matrix using cosine distance.
        
        Args:
            det_features: Detection features [N, D]
            track_features: Track features [M, D]
        
        Returns:
            Cost matrix [N, M]
        """
        if det_features.size == 0 or track_features.size == 0:
            return np.empty((len(det_features), len(track_features)))
        
        # Normalize features
        det_norm = det_features / (np.linalg.norm(det_features, axis=1, keepdims=True) + 1e-6)
        trk_norm = track_features / (np.linalg.norm(track_features, axis=1, keepdims=True) + 1e-6)
        
        # Cosine similarity
        similarity = np.dot(det_norm, trk_norm.T)
        
        # Convert to distance
        distance = 1 - similarity
        
        return distance
    
    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update tracker with new detections.
        
        Args:
            detections: Detection boxes [N, 4]
            scores: Detection scores [N]
            classes: Detection classes [N]
            features: Optional Re-ID features [N, D]
        
        Returns:
            Array of active tracks [M, 5+]
        """
        self.frame_count += 1
        
        # If Re-ID features provided, use enhanced matching
        if features is not None and self.with_reid:
            return self._update_with_reid(detections, scores, classes, features)
        
        # Fall back to ByteTrack
        return self.byte_tracker.update(detections, scores, classes)
    
    def _update_with_reid(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        """Update with Re-ID features for improved association."""
        # Similar to ByteTrack but with combined IoU and appearance cost
        self.byte_tracker.frame_count = self.frame_count
        
        for tracker in self.byte_tracker.trackers:
            tracker.predict()
        
        # Get tracked boxes and features
        if self.byte_tracker.trackers:
            trk_boxes = np.array([t.get_state() for t in self.byte_tracker.trackers])
            trk_features = np.array([
                t.feature if t.feature is not None else np.zeros(features.shape[1])
                for t in self.byte_tracker.trackers
            ])
        else:
            trk_boxes = np.empty((0, 4))
            trk_features = np.empty((0, features.shape[1] if features.size > 0 else 512))
        
        # Separate high and low confidence detections
        high_mask = scores >= self.track_high_thresh
        
        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        high_classes = classes[high_mask]
        high_features = features[high_mask] if features.size > 0 else np.empty((0, 512))
        
        # Compute combined cost matrix
        if len(high_dets) > 0 and len(trk_boxes) > 0:
            iou_matrix = iou_batch(high_dets, trk_boxes)
            iou_cost = 1 - iou_matrix
            
            appearance_cost = self._compute_appearance_cost(high_features, trk_features)
            
            # Combined cost with lambda weighting
            cost_matrix = self.lambda_ * iou_cost + (1 - self.lambda_) * appearance_cost
            
            matches, unmatched_dets, unmatched_trks = linear_assignment(cost_matrix)
            
            # Filter matches
            valid_matches = []
            for m in matches:
                if iou_matrix[m[0], m[1]] >= self.proximity_thresh or \
                   (appearance_cost[m[0], m[1]] if appearance_cost.size > 0 else 1.0) <= self.appearance_thresh:
                    valid_matches.append(m)
                else:
                    unmatched_dets = np.append(unmatched_dets, m[0])
                    unmatched_trks = np.append(unmatched_trks, m[1])
            
            matches = np.array(valid_matches) if valid_matches else np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(high_dets))
            unmatched_trks = np.arange(len(self.byte_tracker.trackers))
        
        # Update matched trackers with EMA feature update
        for m in matches:
            tracker = self.byte_tracker.trackers[m[1]]
            tracker.update(high_dets[m[0]], high_scores[m[0]])
            
            # EMA feature update
            if tracker.feature is not None:
                tracker.feature = self.ema_alpha * tracker.feature + \
                                  (1 - self.ema_alpha) * high_features[m[0]]
            else:
                tracker.feature = high_features[m[0]]
        
        # Create new trackers
        for i in unmatched_dets:
            if high_scores[i] >= self.new_track_thresh:
                tracker = KalmanBoxTracker(
                    high_dets[i],
                    class_id=int(high_classes[i]),
                    confidence=high_scores[i]
                )
                tracker.feature = high_features[i] if i < len(high_features) else None
                self.byte_tracker.trackers.append(tracker)
        
        # Remove dead trackers
        self.byte_tracker.trackers = [
            t for t in self.byte_tracker.trackers
            if t.time_since_update <= self.track_buffer
        ]
        
        # Get outputs
        outputs = []
        for tracker in self.byte_tracker.trackers:
            if tracker.time_since_update == 0 and tracker.hits >= 3:
                bbox = tracker.get_state()
                outputs.append([*bbox, tracker.id, tracker.class_id, tracker.confidence])
        
        return np.array(outputs) if outputs else np.empty((0, 7))


def initialize_tracker(params: Dict[str, Any]) -> Any:
    """Initialize the specified tracker.
    
    Args:
        params: Tracking parameters
    
    Returns:
        Initialized tracker object
    """
    tracker_name = params.get("tracker", "bytetrack")
    
    logger.info(f"Initializing tracker: {tracker_name}")
    
    if tracker_name == "bytetrack":
        return ByteTracker(params)
    elif tracker_name == "botsort":
        return BoTSORTTracker(params)
    elif tracker_name == "ocsort":
        # OC-SORT would require additional implementation
        logger.warning("OC-SORT not fully implemented, using ByteTrack")
        return ByteTracker(params)
    elif tracker_name == "deepsort":
        # DeepSORT would require additional implementation
        logger.warning("DeepSORT not fully implemented, using BoT-SORT")
        return BoTSORTTracker(params)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")


def run_tracking(
    tracker: Any,
    detection_results: List[Any],
    frames: List[np.ndarray],
    params: Dict[str, Any],
) -> List[TrackingResult]:
    """Run tracking on detection results.
    
    Args:
        tracker: Initialized tracker
        detection_results: List of detection results per frame
        frames: Original frames (for Re-ID feature extraction)
        params: Tracking parameters
    
    Returns:
        List of tracking results
    """
    tracking_results = []
    
    for frame_idx, det_result in enumerate(detection_results):
        start_time = time.time()
        
        # Get detections
        boxes = det_result.get_boxes()
        scores = det_result.get_scores()
        classes = det_result.get_classes()
        
        # Run tracker update
        if len(boxes) > 0:
            tracks = tracker.update(boxes, scores, classes)
        else:
            tracks = np.empty((0, 7))
        
        processing_time = time.time() - start_time
        
        # Convert to Track objects
        track_objects = []
        for track in tracks:
            if len(track) >= 6:
                track_obj = Track(
                    track_id=int(track[4]),
                    bbox=track[:4],
                    confidence=float(track[6]) if len(track) > 6 else 1.0,
                    class_id=int(track[5]),
                    class_name=det_result.detections[0].class_name if det_result.detections else "unknown",
                    frame_id=frame_idx,
                    state="confirmed",
                )
                track_objects.append(track_obj)
        
        tracking_results.append(TrackingResult(
            frame_id=frame_idx,
            tracks=track_objects,
            processing_time=processing_time,
        ))
    
    total_tracks = sum(len(r.tracks) for r in tracking_results)
    logger.info(f"Tracking complete: {len(tracking_results)} frames, {total_tracks} track instances")
    
    return tracking_results


def extract_trajectories(
    tracking_results: List[TrackingResult],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Extract trajectories from tracking results.
    
    Args:
        tracking_results: List of tracking results
        params: Tracking parameters
    
    Returns:
        DataFrame with trajectory information
    """
    trajectories = defaultdict(list)
    
    for result in tracking_results:
        for track in result.tracks:
            trajectories[track.track_id].append({
                "track_id": track.track_id,
                "frame_id": track.frame_id,
                "x1": track.bbox[0],
                "y1": track.bbox[1],
                "x2": track.bbox[2],
                "y2": track.bbox[3],
                "cx": (track.bbox[0] + track.bbox[2]) / 2,
                "cy": (track.bbox[1] + track.bbox[3]) / 2,
                "width": track.bbox[2] - track.bbox[0],
                "height": track.bbox[3] - track.bbox[1],
                "confidence": track.confidence,
                "class_id": track.class_id,
                "class_name": track.class_name,
            })
    
    # Flatten to DataFrame
    all_tracks = []
    for track_id, track_frames in trajectories.items():
        all_tracks.extend(track_frames)
    
    df = pd.DataFrame(all_tracks)
    
    if not df.empty:
        df = df.sort_values(["track_id", "frame_id"])
    
    logger.info(f"Extracted {len(trajectories)} unique trajectories")
    
    return df


def compute_tracking_metrics(
    tracking_results: List[TrackingResult],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute tracking metrics.
    
    Args:
        tracking_results: List of tracking results
        params: Tracking parameters
    
    Returns:
        Dictionary of computed metrics
    """
    if not tracking_results:
        return {}
    
    # Basic statistics
    total_frames = len(tracking_results)
    total_tracks = sum(len(r.tracks) for r in tracking_results)
    
    # Unique track IDs
    unique_ids = set()
    for result in tracking_results:
        for track in result.tracks:
            unique_ids.add(track.track_id)
    
    # Processing time
    processing_times = [r.processing_time for r in tracking_results]
    avg_time = np.mean(processing_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    # Track lengths
    track_lengths = defaultdict(int)
    for result in tracking_results:
        for track in result.tracks:
            track_lengths[track.track_id] += 1
    
    lengths = list(track_lengths.values())
    
    metrics = {
        "total_frames": total_frames,
        "total_track_instances": total_tracks,
        "unique_tracks": len(unique_ids),
        "tracks_per_frame": total_tracks / total_frames if total_frames > 0 else 0,
        "avg_track_length": np.mean(lengths) if lengths else 0,
        "max_track_length": max(lengths) if lengths else 0,
        "min_track_length": min(lengths) if lengths else 0,
        "avg_processing_time_ms": avg_time * 1000,
        "fps": fps,
    }
    
    logger.info(f"Tracking metrics: {len(unique_ids)} unique tracks, {fps:.1f} FPS")
    
    return metrics


def log_tracking_to_mlflow(
    metrics: Dict[str, float],
    params: Dict[str, Any],
) -> None:
    """Log tracking metrics to MLFlow.
    
    Args:
        metrics: Computed tracking metrics
        params: Tracking parameters
    """
    try:
        mlflow.log_param("tracker", params.get("tracker", "unknown"))
        mlflow.log_param("track_high_thresh", params.get("track_high_thresh", 0.5))
        mlflow.log_param("track_buffer", params.get("track_buffer", 30))
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"tracking_{key}", value)
        
        logger.info("Tracking metrics logged to MLFlow")
        
    except Exception as e:
        logger.warning(f"Failed to log to MLFlow: {e}")
