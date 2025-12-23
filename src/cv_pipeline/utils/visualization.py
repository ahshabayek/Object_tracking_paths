"""Visualization utilities for CV Pipeline.

This module provides functions for drawing:
- Object detections with bounding boxes and labels
- Multi-object tracks with trajectories and IDs
- Lane markings with fitted curves
- Drivable paths with waypoints

All functions work with BGR images (OpenCV format) and modify images in-place.
"""

import colorsys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Default color palette (BGR format)
COLORS = [
    (255, 0, 0),  # Blue
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Orange
    (255, 128, 0),  # Light Blue
    (0, 128, 255),  # Light Orange
    (128, 255, 0),  # Light Green
    (255, 0, 128),  # Purple
    (0, 255, 128),  # Lime
]

# Lane type colors (BGR)
LANE_COLORS = {
    "ego_left": (0, 255, 0),  # Green
    "ego_right": (0, 255, 0),  # Green
    "adjacent_left": (0, 165, 255),  # Orange
    "adjacent_right": (0, 165, 255),  # Orange
    "unknown": (128, 128, 128),  # Gray
}

# Class-specific colors for common objects (BGR)
CLASS_COLORS = {
    "person": (0, 0, 255),  # Red
    "bicycle": (255, 165, 0),  # Orange
    "car": (255, 0, 0),  # Blue
    "motorcycle": (255, 0, 255),  # Magenta
    "bus": (0, 255, 255),  # Yellow
    "truck": (128, 0, 128),  # Purple
    "traffic light": (0, 255, 0),  # Green
    "stop sign": (0, 0, 255),  # Red
}


def get_color_for_id(
    track_id: int, palette: Optional[List[Tuple[int, int, int]]] = None
) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID.

    Args:
        track_id: Track identifier.
        palette: Optional color palette. Uses default if None.

    Returns:
        BGR color tuple.
    """
    if palette is None:
        palette = COLORS
    return palette[track_id % len(palette)]


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors.

    Args:
        n: Number of colors to generate.

    Returns:
        List of BGR color tuples.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def draw_detections(
    image: np.ndarray,
    detections: List[Any],
    show_labels: bool = True,
    show_confidence: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw detection bounding boxes on an image.

    Args:
        image: Input image (H, W, C) in BGR format.
        detections: List of Detection objects or dicts with 'bbox', 'class_name', 'confidence'.
        show_labels: Whether to show class labels.
        show_confidence: Whether to show confidence scores.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for labels.
        alpha: Transparency for filled rectangles (0-1).

    Returns:
        Image with detections drawn (modifies in-place and returns).
    """
    overlay = image.copy()

    for det in detections:
        # Extract detection info
        if hasattr(det, "bbox"):
            bbox = det.bbox
            class_name = det.class_name
            confidence = det.confidence
            class_id = det.class_id
        elif isinstance(det, dict):
            bbox = np.array(det["bbox"])
            class_name = det.get("class_name", "object")
            confidence = det.get("confidence", 1.0)
            class_id = det.get("class_id", 0)
        else:
            continue

        # Get color
        color = CLASS_COLORS.get(class_name, get_color_for_id(class_id))

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw semi-transparent fill
        if alpha > 0:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Draw label
        if show_labels or show_confidence:
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            label = " ".join(label_parts)

            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Draw label background
            label_y1 = max(y1 - label_h - 10, 0)
            label_y2 = y1
            cv2.rectangle(image, (x1, label_y1), (x1 + label_w + 4, label_y2), color, -1)

            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Blend overlay
    if alpha > 0:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


def draw_tracks(
    image: np.ndarray,
    tracks: List[Any],
    trajectories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    show_ids: bool = True,
    show_trails: bool = True,
    trail_length: int = 30,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw tracked objects with trajectories on an image.

    Args:
        image: Input image (H, W, C) in BGR format.
        tracks: List of Track objects or dicts with 'track_id', 'bbox', 'class_name'.
        trajectories: Optional dict mapping track_id to list of (cx, cy) center points.
        show_ids: Whether to show track IDs.
        show_trails: Whether to show trajectory trails.
        trail_length: Maximum number of points in trail.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for labels.

    Returns:
        Image with tracks drawn (modifies in-place and returns).
    """
    # Build trajectory history if not provided
    if trajectories is None:
        trajectories = {}

    for track in tracks:
        # Extract track info
        if hasattr(track, "track_id"):
            track_id = track.track_id
            bbox = track.bbox
            class_name = getattr(track, "class_name", "object")
            confidence = getattr(track, "confidence", 1.0)
        elif isinstance(track, dict):
            track_id = track["track_id"]
            bbox = np.array(track["bbox"])
            class_name = track.get("class_name", "object")
            confidence = track.get("confidence", 1.0)
        else:
            continue

        # Get consistent color for this track
        color = get_color_for_id(track_id)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Calculate center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Update trajectory
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))

        # Limit trail length
        if len(trajectories[track_id]) > trail_length:
            trajectories[track_id] = trajectories[track_id][-trail_length:]

        # Draw trail
        if show_trails and len(trajectories[track_id]) > 1:
            points = trajectories[track_id]
            for i in range(1, len(points)):
                # Fade trail based on age
                alpha = i / len(points)
                trail_color = tuple(int(c * alpha) for c in color)
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(image, pt1, pt2, trail_color, max(1, thickness - 1))

        # Draw center point
        cv2.circle(image, (cx, cy), 4, color, -1)

        # Draw track ID
        if show_ids:
            label = f"ID:{track_id}"

            # Calculate label position (above box)
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )

            label_x = x1
            label_y = max(y1 - 8, label_h + 4)

            # Draw label background
            cv2.rectangle(
                image,
                (label_x, label_y - label_h - 4),
                (label_x + label_w + 4, label_y + 4),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (label_x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return image


def draw_lanes(
    image: np.ndarray,
    lanes: List[Any],
    show_points: bool = True,
    show_type: bool = True,
    thickness: int = 3,
    point_radius: int = 4,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw lane markings on an image.

    Args:
        image: Input image (H, W, C) in BGR format.
        lanes: List of Lane objects or dicts with 'points', 'lane_type', 'confidence'.
        show_points: Whether to draw individual lane points.
        show_type: Whether to show lane type labels.
        thickness: Line thickness for lanes.
        point_radius: Radius for lane points.
        font_scale: Font scale for labels.

    Returns:
        Image with lanes drawn (modifies in-place and returns).
    """
    for lane in lanes:
        # Extract lane info
        if hasattr(lane, "points"):
            # Lane object with LanePoint list
            points = np.array([[p.x, p.y] for p in lane.points])
            lane_type = getattr(lane, "lane_type", "unknown")
            confidence = getattr(lane, "confidence", 1.0)
        elif isinstance(lane, dict):
            points = np.array(lane.get("points", []))
            lane_type = lane.get("lane_type", "unknown")
            confidence = lane.get("confidence", 1.0)
        else:
            continue

        if len(points) < 2:
            continue

        # Get color for lane type
        color = LANE_COLORS.get(lane_type, LANE_COLORS["unknown"])

        # Draw lane line
        points_int = points.astype(np.int32)
        cv2.polylines(image, [points_int], False, color, thickness)

        # Draw lane points
        if show_points:
            for pt in points_int:
                cv2.circle(image, tuple(pt), point_radius, color, -1)

        # Draw lane type label
        if show_type and len(points_int) > 0:
            # Position label at the top of the lane
            label_pt = points_int[0]
            label = f"{lane_type} ({confidence:.2f})"

            cv2.putText(
                image,
                label,
                (int(label_pt[0]), int(label_pt[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    return image


def draw_path(
    image: np.ndarray,
    path: Any,
    show_waypoints: bool = True,
    show_boundaries: bool = True,
    show_heading: bool = False,
    path_color: Tuple[int, int, int] = (0, 255, 0),
    boundary_color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    waypoint_radius: int = 5,
) -> np.ndarray:
    """Draw drivable path with waypoints and boundaries on an image.

    Args:
        image: Input image (H, W, C) in BGR format.
        path: DrivablePath object or dict with 'waypoints', 'left_boundary', 'right_boundary'.
        show_waypoints: Whether to draw waypoint markers.
        show_boundaries: Whether to draw lane boundaries.
        show_heading: Whether to draw heading arrows at waypoints.
        path_color: Color for the path line.
        boundary_color: Color for lane boundaries.
        thickness: Line thickness.
        waypoint_radius: Radius for waypoint markers.

    Returns:
        Image with path drawn (modifies in-place and returns).
    """
    # Extract path info
    if hasattr(path, "waypoints"):
        waypoints = path.waypoints
        left_boundary = getattr(path, "left_boundary", None)
        right_boundary = getattr(path, "right_boundary", None)
        center_line = getattr(path, "center_line", None)
    elif isinstance(path, dict):
        waypoints = path.get("waypoints", [])
        left_boundary = path.get("left_boundary")
        right_boundary = path.get("right_boundary")
        center_line = path.get("center_line")
    else:
        return image

    # Draw boundaries
    if show_boundaries:
        if left_boundary is not None:
            pts = np.array(left_boundary, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(image, [pts], False, boundary_color, thickness)

        if right_boundary is not None:
            pts = np.array(right_boundary, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(image, [pts], False, boundary_color, thickness)

    # Draw center line / path
    if center_line is not None:
        pts = np.array(center_line, dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(image, [pts], False, path_color, thickness + 1)

    # Draw waypoints
    if waypoints:
        waypoint_pts = []
        for wp in waypoints:
            if hasattr(wp, "x"):
                x, y = wp.x, wp.y
                heading = getattr(wp, "heading", 0)
            elif isinstance(wp, dict):
                x, y = wp["x"], wp["y"]
                heading = wp.get("heading", 0)
            else:
                continue

            waypoint_pts.append((int(x), int(y), heading))

        # Draw path line through waypoints
        if len(waypoint_pts) > 1:
            pts = np.array([(p[0], p[1]) for p in waypoint_pts], dtype=np.int32)
            cv2.polylines(image, [pts], False, path_color, thickness)

        # Draw waypoint markers
        if show_waypoints:
            for i, (x, y, heading) in enumerate(waypoint_pts):
                # Outer circle
                cv2.circle(image, (x, y), waypoint_radius, path_color, -1)
                # Inner circle
                cv2.circle(image, (x, y), waypoint_radius - 2, (255, 255, 255), -1)

                # Draw heading arrow
                if show_heading:
                    arrow_len = 20
                    end_x = int(x + arrow_len * np.cos(heading))
                    end_y = int(y + arrow_len * np.sin(heading))
                    cv2.arrowedLine(image, (x, y), (end_x, end_y), path_color, 2)

    return image


def draw_scene(
    image: np.ndarray,
    detections: Optional[List[Any]] = None,
    tracks: Optional[List[Any]] = None,
    lanes: Optional[List[Any]] = None,
    path: Optional[Any] = None,
    trajectories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw complete scene visualization with all elements.

    Args:
        image: Input image (H, W, C) in BGR format.
        detections: Optional list of detections.
        tracks: Optional list of tracks.
        lanes: Optional list of lanes.
        path: Optional drivable path.
        trajectories: Optional trajectory history for tracks.
        config: Optional configuration dict for visualization parameters.

    Returns:
        Image with all elements drawn.
    """
    config = config or {}

    # Draw in order: lanes (bottom) -> path -> tracks -> detections (top)
    if lanes is not None:
        draw_lanes(
            image,
            lanes,
            show_points=config.get("show_lane_points", True),
            show_type=config.get("show_lane_type", True),
            thickness=config.get("lane_thickness", 3),
        )

    if path is not None:
        draw_path(
            image,
            path,
            show_waypoints=config.get("show_waypoints", True),
            show_boundaries=config.get("show_boundaries", True),
            show_heading=config.get("show_heading", False),
            thickness=config.get("path_thickness", 2),
        )

    if tracks is not None:
        draw_tracks(
            image,
            tracks,
            trajectories=trajectories,
            show_ids=config.get("show_track_ids", True),
            show_trails=config.get("show_trails", True),
            trail_length=config.get("trail_length", 30),
            thickness=config.get("track_thickness", 2),
        )

    if detections is not None:
        draw_detections(
            image,
            detections,
            show_labels=config.get("show_labels", True),
            show_confidence=config.get("show_confidence", True),
            thickness=config.get("detection_thickness", 2),
            alpha=config.get("detection_alpha", 0.3),
        )

    return image


class Visualizer:
    """Stateful visualizer for video sequences.

    Maintains trajectory history across frames for smooth trail visualization.

    Example:
        visualizer = Visualizer(config={'trail_length': 50})

        for frame, tracks in zip(frames, tracking_results):
            annotated = visualizer.draw_frame(
                frame,
                tracks=tracks.tracks,
                lanes=lane_results,
            )
            output_frames.append(annotated)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Visualizer.

        Args:
            config: Visualization configuration parameters.
        """
        self.config = config or {}
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self.frame_count = 0

        # Configuration defaults
        self.trail_length = self.config.get("trail_length", 30)
        self.show_frame_info = self.config.get("show_frame_info", True)
        self.show_fps = self.config.get("show_fps", True)

    def reset(self) -> None:
        """Reset visualizer state."""
        self.trajectories.clear()
        self.frame_count = 0

    def update_trajectories(self, tracks: List[Any]) -> None:
        """Update trajectory history with new tracks.

        Args:
            tracks: List of current tracks.
        """
        current_ids = set()

        for track in tracks:
            if hasattr(track, "track_id"):
                track_id = track.track_id
                bbox = track.bbox
            elif isinstance(track, dict):
                track_id = track["track_id"]
                bbox = track["bbox"]
            else:
                continue

            current_ids.add(track_id)

            # Calculate center
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Update trajectory
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((cx, cy))

            # Limit trail length
            if len(self.trajectories[track_id]) > self.trail_length:
                self.trajectories[track_id] = self.trajectories[track_id][-self.trail_length :]

        # Clean up old trajectories (tracks that have been lost)
        max_age = self.config.get("trajectory_max_age", 30)
        stale_ids = []
        for track_id in self.trajectories:
            if track_id not in current_ids:
                # Track not seen this frame - will be cleaned up after max_age frames
                if len(self.trajectories[track_id]) > 0:
                    # Mark as stale by removing oldest point
                    if self.frame_count % max_age == 0:
                        stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.trajectories[track_id]

    def draw_frame(
        self,
        image: np.ndarray,
        detections: Optional[List[Any]] = None,
        tracks: Optional[List[Any]] = None,
        lanes: Optional[List[Any]] = None,
        path: Optional[Any] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Draw a single frame with all visualizations.

        Args:
            image: Input frame.
            detections: Optional detections for this frame.
            tracks: Optional tracks for this frame.
            lanes: Optional lane detections.
            path: Optional drivable path.
            fps: Optional FPS value to display.

        Returns:
            Annotated frame.
        """
        self.frame_count += 1

        # Update trajectories if tracks provided
        if tracks is not None:
            self.update_trajectories(tracks)

        # Draw scene elements
        output = draw_scene(
            image.copy(),
            detections=detections,
            tracks=tracks,
            lanes=lanes,
            path=path,
            trajectories=self.trajectories,
            config=self.config,
        )

        # Draw frame info overlay
        if self.show_frame_info:
            info_y = 30

            # Frame number
            cv2.putText(
                output,
                f"Frame: {self.frame_count}",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            info_y += 25

            # FPS
            if self.show_fps and fps is not None:
                cv2.putText(
                    output,
                    f"FPS: {fps:.1f}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                info_y += 25

            # Track count
            if tracks is not None:
                cv2.putText(
                    output,
                    f"Tracks: {len(tracks)}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                info_y += 25

            # Detection count
            if detections is not None:
                cv2.putText(
                    output,
                    f"Detections: {len(detections)}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return output

    def create_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        """Create video from annotated frames.

        Args:
            frames: List of annotated frames.
            output_path: Output video file path.
            fps: Output frame rate.
            codec: Video codec.
        """
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
