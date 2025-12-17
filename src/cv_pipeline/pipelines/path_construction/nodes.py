"""Path Construction Pipeline Nodes.

This module contains node functions for constructing drivable paths
by fusing lane detection and object tracking results.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """A single waypoint on the planned path."""
    x: float
    y: float
    heading: float = 0.0  # radians
    curvature: float = 0.0
    velocity: float = 0.0
    timestamp: float = 0.0


@dataclass
class DrivablePath:
    """Represents a drivable path."""
    frame_id: int
    waypoints: List[Waypoint]
    left_boundary: Optional[np.ndarray] = None
    right_boundary: Optional[np.ndarray] = None
    center_line: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class FusedScene:
    """Fused perception scene combining lanes and objects."""
    frame_id: int
    ego_lanes: Dict[str, np.ndarray]  # ego_left, ego_right
    adjacent_lanes: Dict[str, np.ndarray]  # adjacent_left, adjacent_right
    tracked_objects: List[Dict[str, Any]]
    drivable_area: Optional[np.ndarray] = None


def fuse_perception_data(
    lane_results: List[Dict[str, Any]],
    tracking_results: List[Any],
    params: Dict[str, Any],
) -> List[FusedScene]:
    """Fuse lane detection and tracking results.
    
    Args:
        lane_results: Lane detection results per frame
        tracking_results: Object tracking results per frame
        params: Path construction parameters
    
    Returns:
        List of fused scene representations
    """
    fused_scenes = []
    
    # Align by frame ID
    lane_by_frame = {r["frame_id"]: r for r in lane_results}
    tracking_by_frame = {}
    
    for track_result in tracking_results:
        frame_id = track_result.frame_id
        if frame_id not in tracking_by_frame:
            tracking_by_frame[frame_id] = []
        tracking_by_frame[frame_id].extend([t.to_dict() for t in track_result.tracks])
    
    # Get all frame IDs
    all_frames = sorted(set(lane_by_frame.keys()) | set(tracking_by_frame.keys()))
    
    for frame_id in all_frames:
        # Extract ego lanes
        ego_lanes = {}
        adjacent_lanes = {}
        
        if frame_id in lane_by_frame:
            lane_data = lane_by_frame[frame_id]
            for lane in lane_data.get("lanes", []):
                lane_type = lane.get("lane_type", "unknown")
                points = np.array(lane.get("points", []))
                
                if len(points) > 0:
                    if lane_type in ["ego_left", "ego_right"]:
                        ego_lanes[lane_type] = points
                    elif lane_type in ["adjacent_left", "adjacent_right"]:
                        adjacent_lanes[lane_type] = points
        
        # Get tracked objects
        objects = tracking_by_frame.get(frame_id, [])
        
        fused_scenes.append(FusedScene(
            frame_id=frame_id,
            ego_lanes=ego_lanes,
            adjacent_lanes=adjacent_lanes,
            tracked_objects=objects,
        ))
    
    logger.info(f"Fused perception data for {len(fused_scenes)} frames")
    
    return fused_scenes


def construct_drivable_path(
    fused_scenes: List[FusedScene],
    lane_results: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[DrivablePath]:
    """Construct drivable paths from fused scene data.
    
    Args:
        fused_scenes: Fused perception scenes
        lane_results: Original lane detection results
        params: Path construction parameters
    
    Returns:
        List of drivable paths per frame
    """
    smoothing_method = params.get("fitting_method", "bezier")
    lookahead = params.get("lookahead_distance", 30.0)
    path_resolution = params.get("path_resolution", 0.1)
    
    drivable_paths = []
    
    for scene in fused_scenes:
        # Compute center line from ego lanes
        center_line = _compute_center_line(
            scene.ego_lanes.get("ego_left"),
            scene.ego_lanes.get("ego_right"),
        )
        
        # Generate waypoints along center line
        if center_line is not None and len(center_line) > 1:
            waypoints = _generate_waypoints_from_line(
                center_line,
                resolution=path_resolution,
                lookahead=lookahead,
            )
        else:
            waypoints = []
        
        # Apply safety constraints from tracked objects
        if scene.tracked_objects:
            waypoints = _apply_object_avoidance(
                waypoints,
                scene.tracked_objects,
                safety_margin=params.get("safety", {}).get("safety_margin", 2.0),
            )
        
        # Smooth the path
        if waypoints and smoothing_method == "spline":
            waypoints = _smooth_path_spline(waypoints)
        elif waypoints and smoothing_method == "bezier":
            waypoints = _smooth_path_bezier(waypoints)
        
        drivable_paths.append(DrivablePath(
            frame_id=scene.frame_id,
            waypoints=waypoints,
            left_boundary=scene.ego_lanes.get("ego_left"),
            right_boundary=scene.ego_lanes.get("ego_right"),
            center_line=center_line,
        ))
    
    logger.info(f"Constructed drivable paths for {len(drivable_paths)} frames")
    
    return drivable_paths


def _compute_center_line(
    left_lane: Optional[np.ndarray],
    right_lane: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Compute center line from left and right lane boundaries.
    
    Args:
        left_lane: Left lane points [N, 2]
        right_lane: Right lane points [M, 2]
    
    Returns:
        Center line points [K, 2]
    """
    if left_lane is None and right_lane is None:
        return None
    
    if left_lane is None:
        # Only right lane available - offset inward
        return right_lane - np.array([100, 0])  # Rough offset
    
    if right_lane is None:
        # Only left lane available - offset inward
        return left_lane + np.array([100, 0])  # Rough offset
    
    # Both lanes available - compute average
    # Resample to same number of points
    n_points = max(len(left_lane), len(right_lane))
    
    left_resampled = _resample_curve(left_lane, n_points)
    right_resampled = _resample_curve(right_lane, n_points)
    
    # Average the points
    center = (left_resampled + right_resampled) / 2
    
    return center


def _resample_curve(
    points: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Resample curve to have exactly n_points.
    
    Args:
        points: Original curve points [N, 2]
        n_points: Target number of points
    
    Returns:
        Resampled curve [n_points, 2]
    """
    if len(points) == n_points:
        return points
    
    # Compute cumulative distance
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    total_distance = distances[-1]
    if total_distance == 0:
        return np.tile(points[0], (n_points, 1))
    
    # Sample at uniform distance intervals
    target_distances = np.linspace(0, total_distance, n_points)
    
    resampled = np.zeros((n_points, 2))
    for i, target_dist in enumerate(target_distances):
        # Find surrounding points
        idx = np.searchsorted(distances, target_dist)
        idx = min(idx, len(distances) - 1)
        
        if idx == 0:
            resampled[i] = points[0]
        else:
            # Linear interpolation
            t = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1] + 1e-6)
            resampled[i] = (1 - t) * points[idx-1] + t * points[idx]
    
    return resampled


def _generate_waypoints_from_line(
    line: np.ndarray,
    resolution: float = 0.1,
    lookahead: float = 30.0,
) -> List[Waypoint]:
    """Generate waypoints from a line.
    
    Args:
        line: Line points [N, 2]
        resolution: Distance between waypoints (meters)
        lookahead: Maximum distance to generate (meters)
    
    Returns:
        List of waypoints
    """
    if len(line) < 2:
        return []
    
    waypoints = []
    
    # Compute cumulative distance
    total_dist = 0
    for i in range(len(line)):
        if i > 0:
            total_dist += np.linalg.norm(line[i] - line[i-1])
        
        if total_dist > lookahead:
            break
        
        # Compute heading
        if i < len(line) - 1:
            direction = line[i+1] - line[i]
            heading = np.arctan2(direction[1], direction[0])
        elif i > 0:
            direction = line[i] - line[i-1]
            heading = np.arctan2(direction[1], direction[0])
        else:
            heading = 0.0
        
        # Compute curvature (simple finite difference)
        curvature = 0.0
        if i > 0 and i < len(line) - 1:
            v1 = line[i] - line[i-1]
            v2 = line[i+1] - line[i]
            angle_change = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            dist = np.linalg.norm(v1) + np.linalg.norm(v2)
            if dist > 0:
                curvature = 2 * angle_change / dist
        
        waypoints.append(Waypoint(
            x=float(line[i, 0]),
            y=float(line[i, 1]),
            heading=float(heading),
            curvature=float(curvature),
        ))
    
    return waypoints


def _apply_object_avoidance(
    waypoints: List[Waypoint],
    objects: List[Dict[str, Any]],
    safety_margin: float = 2.0,
) -> List[Waypoint]:
    """Apply object avoidance to path.
    
    Args:
        waypoints: Original waypoints
        objects: Tracked objects
        safety_margin: Safety distance in meters
    
    Returns:
        Adjusted waypoints
    """
    if not waypoints or not objects:
        return waypoints
    
    # Simple approach: flag waypoints that are too close to objects
    adjusted = []
    
    for wp in waypoints:
        wp_pos = np.array([wp.x, wp.y])
        
        min_dist = float('inf')
        for obj in objects:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            obj_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            ])
            dist = np.linalg.norm(wp_pos - obj_center)
            min_dist = min(min_dist, dist)
        
        # Reduce velocity near objects
        if min_dist < safety_margin * 2:
            wp.velocity = max(0, wp.velocity * (min_dist / (safety_margin * 2)))
        
        adjusted.append(wp)
    
    return adjusted


def _smooth_path_spline(
    waypoints: List[Waypoint],
    smoothing_factor: float = 0.5,
) -> List[Waypoint]:
    """Smooth path using cubic spline.
    
    Args:
        waypoints: Original waypoints
        smoothing_factor: Spline smoothing factor
    
    Returns:
        Smoothed waypoints
    """
    if len(waypoints) < 4:
        return waypoints
    
    try:
        points = np.array([[wp.x, wp.y] for wp in waypoints])
        
        # Fit spline
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing_factor)
        
        # Evaluate at original parameter values
        new_points = splev(u, tck)
        
        smoothed = []
        for i in range(len(waypoints)):
            wp = waypoints[i]
            wp.x = new_points[0][i]
            wp.y = new_points[1][i]
            smoothed.append(wp)
        
        return smoothed
        
    except Exception as e:
        logger.warning(f"Spline smoothing failed: {e}")
        return waypoints


def _smooth_path_bezier(
    waypoints: List[Waypoint],
    order: int = 3,
) -> List[Waypoint]:
    """Smooth path using Bezier curve.
    
    Args:
        waypoints: Original waypoints
        order: Bezier curve order
    
    Returns:
        Smoothed waypoints
    """
    if len(waypoints) < order + 1:
        return waypoints
    
    try:
        points = np.array([[wp.x, wp.y] for wp in waypoints])
        
        # Select control points uniformly
        indices = np.linspace(0, len(points) - 1, order + 1).astype(int)
        control_points = points[indices]
        
        # Evaluate Bezier curve at original number of points
        n_points = len(waypoints)
        t_values = np.linspace(0, 1, n_points)
        
        smoothed_points = _evaluate_bezier(control_points, t_values)
        
        smoothed = []
        for i, wp in enumerate(waypoints):
            wp.x = smoothed_points[i, 0]
            wp.y = smoothed_points[i, 1]
            smoothed.append(wp)
        
        return smoothed
        
    except Exception as e:
        logger.warning(f"Bezier smoothing failed: {e}")
        return waypoints


def _evaluate_bezier(
    control_points: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """Evaluate Bezier curve at given parameter values.
    
    Args:
        control_points: Control points [n+1, 2]
        t_values: Parameter values [M]
    
    Returns:
        Evaluated points [M, 2]
    """
    n = len(control_points) - 1
    result = np.zeros((len(t_values), 2))
    
    for i, t in enumerate(t_values):
        point = np.zeros(2)
        for j in range(n + 1):
            # Bernstein polynomial
            binom = np.math.factorial(n) / (np.math.factorial(j) * np.math.factorial(n - j))
            bernstein = binom * (t ** j) * ((1 - t) ** (n - j))
            point += bernstein * control_points[j]
        result[i] = point
    
    return result


def generate_trajectory(
    drivable_paths: List[DrivablePath],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate final trajectory from drivable paths.
    
    Args:
        drivable_paths: List of drivable paths
        params: Driving path parameters
    
    Returns:
        List of trajectory data per frame
    """
    output_format = params.get("output", {}).get("format", "waypoints")
    num_waypoints = params.get("output", {}).get("num_waypoints", 50)
    
    trajectories = []
    
    for path in drivable_paths:
        trajectory = {
            "frame_id": path.frame_id,
            "confidence": path.confidence,
            "waypoints": [],
        }
        
        # Resample waypoints if needed
        if len(path.waypoints) > num_waypoints:
            # Downsample
            indices = np.linspace(0, len(path.waypoints) - 1, num_waypoints).astype(int)
            selected_waypoints = [path.waypoints[i] for i in indices]
        else:
            selected_waypoints = path.waypoints
        
        for wp in selected_waypoints:
            trajectory["waypoints"].append({
                "x": wp.x,
                "y": wp.y,
                "heading": wp.heading,
                "curvature": wp.curvature,
                "velocity": wp.velocity,
            })
        
        # Add boundaries if available
        if path.left_boundary is not None:
            trajectory["left_boundary"] = path.left_boundary.tolist()
        if path.right_boundary is not None:
            trajectory["right_boundary"] = path.right_boundary.tolist()
        if path.center_line is not None:
            trajectory["center_line"] = path.center_line.tolist()
        
        trajectories.append(trajectory)
    
    logger.info(f"Generated trajectories for {len(trajectories)} frames")
    
    return trajectories


def compute_path_metrics(
    trajectories: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute path quality metrics.
    
    Args:
        trajectories: Generated trajectories
        params: Path construction parameters
    
    Returns:
        Dictionary of computed metrics
    """
    if not trajectories:
        return {}
    
    # Path statistics
    path_lengths = []
    max_curvatures = []
    avg_curvatures = []
    
    for traj in trajectories:
        waypoints = traj.get("waypoints", [])
        if len(waypoints) < 2:
            continue
        
        # Compute path length
        length = 0
        curvatures = []
        for i in range(1, len(waypoints)):
            dx = waypoints[i]["x"] - waypoints[i-1]["x"]
            dy = waypoints[i]["y"] - waypoints[i-1]["y"]
            length += np.sqrt(dx**2 + dy**2)
            curvatures.append(abs(waypoints[i].get("curvature", 0)))
        
        path_lengths.append(length)
        if curvatures:
            max_curvatures.append(max(curvatures))
            avg_curvatures.append(np.mean(curvatures))
    
    metrics = {
        "total_frames": len(trajectories),
        "frames_with_path": len(path_lengths),
        "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
        "max_path_length": max(path_lengths) if path_lengths else 0,
        "avg_curvature": np.mean(avg_curvatures) if avg_curvatures else 0,
        "max_curvature": max(max_curvatures) if max_curvatures else 0,
        "path_coverage_ratio": len(path_lengths) / len(trajectories) if trajectories else 0,
    }
    
    logger.info(f"Path metrics: {metrics['frames_with_path']} frames with valid paths")
    
    return metrics


def log_path_to_mlflow(metrics: Dict[str, float]) -> None:
    """Log path construction metrics to MLFlow.
    
    Args:
        metrics: Computed path metrics
    """
    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"path_{key}", value)
        
        logger.info("Path metrics logged to MLFlow")
        
    except Exception as e:
        logger.warning(f"Failed to log to MLFlow: {e}")
