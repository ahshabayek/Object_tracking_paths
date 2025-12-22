"""Pytest configuration and fixtures for CV Pipeline tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_image():
    """Create a sample BGR image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Create sample detection data for testing."""
    return [
        {"bbox": [100, 100, 200, 200], "class_name": "car", "confidence": 0.95, "class_id": 2},
        {"bbox": [300, 150, 400, 300], "class_name": "person", "confidence": 0.87, "class_id": 0},
        {"bbox": [50, 50, 150, 200], "class_name": "truck", "confidence": 0.72, "class_id": 7},
    ]


@pytest.fixture
def sample_tracks():
    """Create sample tracking data for testing."""
    return [
        {"track_id": 1, "bbox": [100, 100, 200, 200], "class_name": "car", "confidence": 0.95},
        {"track_id": 2, "bbox": [300, 150, 400, 300], "class_name": "person", "confidence": 0.87},
        {"track_id": 3, "bbox": [50, 50, 150, 200], "class_name": "truck", "confidence": 0.72},
    ]


@pytest.fixture
def sample_lanes():
    """Create sample lane data for testing."""
    return [
        {
            "points": [[200, 400], [210, 300], [220, 200], [230, 100]],
            "lane_type": "ego_left",
            "confidence": 0.9,
        },
        {
            "points": [[400, 400], [390, 300], [380, 200], [370, 100]],
            "lane_type": "ego_right",
            "confidence": 0.85,
        },
    ]


@pytest.fixture
def sample_path():
    """Create sample path data for testing."""
    return {
        "waypoints": [
            {"x": 320, "y": 400, "heading": 0, "curvature": 0},
            {"x": 320, "y": 350, "heading": 0.05, "curvature": 0.01},
            {"x": 325, "y": 300, "heading": 0.1, "curvature": 0.02},
            {"x": 330, "y": 250, "heading": 0.15, "curvature": 0.02},
            {"x": 335, "y": 200, "heading": 0.2, "curvature": 0.01},
        ],
        "left_boundary": [[280, 400], [285, 300], [290, 200]],
        "right_boundary": [[360, 400], [365, 300], [370, 200]],
    }


@pytest.fixture
def sample_bboxes():
    """Create sample bounding boxes as numpy array."""
    return np.array(
        [
            [100, 100, 200, 200],
            [300, 150, 400, 300],
            [50, 50, 150, 200],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_detection_predictions():
    """Create sample detection predictions for metrics testing."""
    return [
        {"boxes": [[100, 100, 200, 200]], "scores": [0.95], "classes": [0]},
        {
            "boxes": [[50, 50, 150, 150], [200, 200, 300, 300]],
            "scores": [0.8, 0.7],
            "classes": [0, 1],
        },
    ]


@pytest.fixture
def sample_detection_ground_truth():
    """Create sample detection ground truth for metrics testing."""
    return [
        {"boxes": [[100, 100, 200, 200]], "classes": [0]},
        {"boxes": [[50, 50, 150, 150], [200, 200, 300, 300]], "classes": [0, 1]},
    ]
