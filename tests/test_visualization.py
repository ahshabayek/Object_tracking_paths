"""Unit tests for visualization utilities."""

import numpy as np
import pytest


class TestDrawDetections:
    """Tests for draw_detections function."""

    def test_empty_detections(self):
        """Test with empty detection list."""
        from cv_pipeline.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(image, [])

        assert result.shape == image.shape
        assert np.array_equal(result, image)

    def test_single_detection_dict(self):
        """Test with single detection as dict."""
        from cv_pipeline.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                "bbox": [100, 100, 200, 200],
                "class_name": "car",
                "confidence": 0.95,
                "class_id": 2,
            }
        ]

        result = draw_detections(image, detections)

        assert result.shape == image.shape
        # Image should be modified (not all zeros)
        assert not np.array_equal(result, np.zeros_like(result))

    def test_multiple_detections(self):
        """Test with multiple detections."""
        from cv_pipeline.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [50, 50, 150, 150], "class_name": "person", "confidence": 0.9, "class_id": 0},
            {"bbox": [200, 100, 350, 250], "class_name": "car", "confidence": 0.85, "class_id": 2},
            {
                "bbox": [400, 200, 500, 400],
                "class_name": "truck",
                "confidence": 0.75,
                "class_id": 7,
            },
        ]

        result = draw_detections(image, detections)

        assert result.shape == image.shape

    def test_no_labels(self):
        """Test with show_labels=False."""
        from cv_pipeline.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [100, 100, 200, 200], "class_name": "car", "confidence": 0.95, "class_id": 2}
        ]

        result = draw_detections(image, detections, show_labels=False, show_confidence=False)

        assert result.shape == image.shape


class TestDrawTracks:
    """Tests for draw_tracks function."""

    def test_empty_tracks(self):
        """Test with empty track list."""
        from cv_pipeline.utils.visualization import draw_tracks

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_tracks(image, [])

        assert result.shape == image.shape

    def test_single_track(self):
        """Test with single track."""
        from cv_pipeline.utils.visualization import draw_tracks

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [
            {
                "track_id": 1,
                "bbox": [100, 100, 200, 200],
                "class_name": "car",
                "confidence": 0.95,
            }
        ]

        result = draw_tracks(image, tracks)

        assert result.shape == image.shape
        assert not np.array_equal(result, np.zeros_like(result))

    def test_with_trajectory(self):
        """Test with trajectory history."""
        from cv_pipeline.utils.visualization import draw_tracks

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [{"track_id": 1, "bbox": [150, 150, 250, 250], "class_name": "car"}]
        trajectories = {1: [(100, 100), (125, 125), (150, 150), (175, 175), (200, 200)]}

        result = draw_tracks(image, tracks, trajectories=trajectories, show_trails=True)

        assert result.shape == image.shape


class TestDrawLanes:
    """Tests for draw_lanes function."""

    def test_empty_lanes(self):
        """Test with empty lane list."""
        from cv_pipeline.utils.visualization import draw_lanes

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_lanes(image, [])

        assert result.shape == image.shape

    def test_single_lane(self):
        """Test with single lane."""
        from cv_pipeline.utils.visualization import draw_lanes

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        lanes = [
            {
                "points": [[100, 400], [120, 300], [140, 200], [160, 100]],
                "lane_type": "ego_left",
                "confidence": 0.9,
            }
        ]

        result = draw_lanes(image, lanes)

        assert result.shape == image.shape

    def test_multiple_lane_types(self):
        """Test with different lane types."""
        from cv_pipeline.utils.visualization import draw_lanes

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        lanes = [
            {"points": [[100, 400], [120, 200]], "lane_type": "ego_left", "confidence": 0.9},
            {"points": [[200, 400], [220, 200]], "lane_type": "ego_right", "confidence": 0.85},
            {"points": [[50, 400], [70, 200]], "lane_type": "adjacent_left", "confidence": 0.7},
        ]

        result = draw_lanes(image, lanes)

        assert result.shape == image.shape


class TestDrawPath:
    """Tests for draw_path function."""

    def test_empty_path(self):
        """Test with empty path."""
        from cv_pipeline.utils.visualization import draw_path

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_path(image, {"waypoints": []})

        assert result.shape == image.shape

    def test_path_with_waypoints(self):
        """Test with waypoints."""
        from cv_pipeline.utils.visualization import draw_path

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        path = {
            "waypoints": [
                {"x": 320, "y": 400, "heading": 0},
                {"x": 320, "y": 350, "heading": 0},
                {"x": 325, "y": 300, "heading": 0.1},
                {"x": 330, "y": 250, "heading": 0.15},
            ]
        }

        result = draw_path(image, path, show_waypoints=True)

        assert result.shape == image.shape

    def test_path_with_boundaries(self):
        """Test with lane boundaries."""
        from cv_pipeline.utils.visualization import draw_path

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        path = {
            "waypoints": [{"x": 320, "y": 400}, {"x": 320, "y": 200}],
            "left_boundary": [[280, 400], [280, 200]],
            "right_boundary": [[360, 400], [360, 200]],
        }

        result = draw_path(image, path, show_boundaries=True)

        assert result.shape == image.shape


class TestDrawScene:
    """Tests for draw_scene function."""

    def test_all_elements(self):
        """Test drawing all scene elements."""
        from cv_pipeline.utils.visualization import draw_scene

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = [
            {"bbox": [100, 100, 200, 200], "class_name": "car", "confidence": 0.9, "class_id": 2}
        ]
        tracks = [{"track_id": 1, "bbox": [100, 100, 200, 200], "class_name": "car"}]
        lanes = [{"points": [[100, 400], [100, 200]], "lane_type": "ego_left", "confidence": 0.8}]
        path = {"waypoints": [{"x": 320, "y": 400}, {"x": 320, "y": 200}]}

        result = draw_scene(
            image,
            detections=detections,
            tracks=tracks,
            lanes=lanes,
            path=path,
        )

        assert result.shape == image.shape

    def test_partial_elements(self):
        """Test with only some elements."""
        from cv_pipeline.utils.visualization import draw_scene

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_scene(
            image,
            tracks=[{"track_id": 1, "bbox": [100, 100, 200, 200], "class_name": "car"}],
        )

        assert result.shape == image.shape


class TestVisualizer:
    """Tests for Visualizer class."""

    def test_init(self):
        """Test visualizer initialization."""
        from cv_pipeline.utils.visualization import Visualizer

        vis = Visualizer(config={"trail_length": 50, "show_fps": True})

        assert vis.trail_length == 50
        assert vis.show_fps is True
        assert vis.frame_count == 0

    def test_reset(self):
        """Test reset functionality."""
        from cv_pipeline.utils.visualization import Visualizer

        vis = Visualizer()
        vis.frame_count = 100
        vis.trajectories = {1: [(0, 0), (1, 1)]}

        vis.reset()

        assert vis.frame_count == 0
        assert len(vis.trajectories) == 0

    def test_update_trajectories(self):
        """Test trajectory update."""
        from cv_pipeline.utils.visualization import Visualizer

        vis = Visualizer(config={"trail_length": 5})

        tracks = [{"track_id": 1, "bbox": [100, 100, 200, 200]}]

        # Update multiple times
        for i in range(10):
            tracks[0]["bbox"] = [100 + i * 10, 100, 200 + i * 10, 200]
            vis.update_trajectories(tracks)

        # Trail should be limited to trail_length
        assert len(vis.trajectories[1]) <= 5

    def test_draw_frame(self):
        """Test drawing a frame."""
        from cv_pipeline.utils.visualization import Visualizer

        vis = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tracks = [{"track_id": 1, "bbox": [100, 100, 200, 200], "class_name": "car"}]

        result = vis.draw_frame(image, tracks=tracks, fps=30.0)

        assert result.shape == image.shape
        assert vis.frame_count == 1


class TestColorUtilities:
    """Tests for color utility functions."""

    def test_get_color_for_id(self):
        """Test consistent color for ID."""
        from cv_pipeline.utils.visualization import get_color_for_id

        color1 = get_color_for_id(5)
        color2 = get_color_for_id(5)

        assert color1 == color2
        assert len(color1) == 3
        assert all(0 <= c <= 255 for c in color1)

    def test_generate_distinct_colors(self):
        """Test distinct color generation."""
        from cv_pipeline.utils.visualization import generate_distinct_colors

        colors = generate_distinct_colors(10)

        assert len(colors) == 10
        # All colors should be different
        assert len(set(colors)) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
