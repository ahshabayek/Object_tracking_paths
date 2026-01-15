"""Tests for 3D perception modules.

Tests cover:
    - Depth estimation (ZoeDepth, MiDaS, etc.)
    - BEV perception (SparseBEV, BEVFormer, Sparse4D)
    - 3D tracking (AB3DMOT, SimpleTrack)
    - End-to-end (UniAD, VAD)
    - Sparse4D v3 unified tracking
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add source directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


# =============================================================================
# Helper functions
# =============================================================================


def load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from file path, bypassing package imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Load modules with mocked dependencies
# =============================================================================


@pytest.fixture(scope="module")
def depth_estimation_module():
    """Load depth estimation module."""
    module_path = src_dir / "cv_pipeline" / "pipelines" / "depth_estimation" / "nodes.py"
    return load_module_from_file("depth_estimation_nodes", module_path)


@pytest.fixture(scope="module")
def bev_perception_module():
    """Load BEV perception module."""
    module_path = src_dir / "cv_pipeline" / "pipelines" / "bev_perception" / "nodes.py"
    return load_module_from_file("bev_perception_nodes", module_path)


@pytest.fixture(scope="module")
def tracking_3d_module():
    """Load 3D tracking module."""
    module_path = src_dir / "cv_pipeline" / "pipelines" / "tracking_3d" / "nodes.py"
    return load_module_from_file("tracking_3d_nodes", module_path)


@pytest.fixture(scope="module")
def end_to_end_module():
    """Load end-to-end module."""
    module_path = src_dir / "cv_pipeline" / "pipelines" / "end_to_end" / "nodes.py"
    return load_module_from_file("end_to_end_nodes", module_path)


@pytest.fixture(scope="module")
def sparse4d_module():
    """Load Sparse4D v3 module."""
    module_path = src_dir / "cv_pipeline" / "pipelines" / "bev_perception" / "sparse4d_v3.py"
    return load_module_from_file("sparse4d_v3", module_path)


# =============================================================================
# Depth Estimation Tests
# =============================================================================


class TestDepthEstimation:
    """Tests for depth estimation module."""

    def test_depth_result_dataclass(self, depth_estimation_module):
        """Test DepthResult dataclass."""
        DepthResult = depth_estimation_module.DepthResult

        depth_map = np.random.rand(480, 640).astype(np.float32) * 50
        result = DepthResult(depth_map=depth_map, inference_time=0.05)

        assert result.depth_map.shape == (480, 640)
        assert result.inference_time == 0.05
        assert result.confidence is None

    def test_depth_result_with_confidence(self, depth_estimation_module):
        """Test DepthResult with confidence map."""
        DepthResult = depth_estimation_module.DepthResult

        depth_map = np.random.rand(480, 640).astype(np.float32) * 50
        confidence = np.random.rand(480, 640).astype(np.float32)
        result = DepthResult(depth_map=depth_map, confidence=confidence)

        assert result.confidence is not None
        assert result.confidence.shape == (480, 640)

    def test_depth_config_dataclass(self, depth_estimation_module):
        """Test DepthConfig dataclass."""
        DepthConfig = depth_estimation_module.DepthConfig

        config = DepthConfig(model="zoedepth", variant="nk", max_depth=80.0)

        assert config.model == "zoedepth"
        assert config.variant == "nk"
        assert config.max_depth == 80.0
        assert config.min_depth == 0.1

    def test_detection_3d_dataclass(self, depth_estimation_module):
        """Test Detection3D dataclass from depth module."""
        Detection3D = depth_estimation_module.Detection3D

        # This Detection3D uses different fields (bbox_2d, position_3d, depth)
        det = Detection3D(
            bbox_2d=np.array([100, 200, 200, 300]),
            position_3d=np.array([10.0, 2.0, 15.0]),
            depth=15.0,
            confidence=0.9,
            class_id=0,
            class_name="car",
        )

        assert det.depth == 15.0
        assert det.confidence == 0.9
        assert det.class_name == "car"
        assert det.position_3d[2] == 15.0

    def test_depth_estimator_factory_supported_models(self, depth_estimation_module):
        """Test DepthEstimatorFactory has correct supported models."""
        DepthEstimatorFactory = depth_estimation_module.DepthEstimatorFactory

        expected = ["zoedepth", "metric3d", "depth_anything", "midas"]
        assert DepthEstimatorFactory.SUPPORTED_MODELS == expected

    def test_depth_result_get_depth_at(self, depth_estimation_module):
        """Test DepthResult.get_depth_at method."""
        DepthResult = depth_estimation_module.DepthResult

        # Create depth map with known values
        depth_map = np.zeros((480, 640), dtype=np.float32)
        depth_map[240, 320] = 10.0  # Center pixel
        result = DepthResult(depth_map=depth_map)

        assert result.get_depth_at(320, 240) == 10.0
        assert result.get_depth_at(0, 0) == 0.0

    def test_depth_result_get_depth_in_box(self, depth_estimation_module):
        """Test DepthResult.get_depth_in_box method."""
        DepthResult = depth_estimation_module.DepthResult

        # Create depth map with uniform depth in a region
        depth_map = np.ones((480, 640), dtype=np.float32) * 15.0
        result = DepthResult(depth_map=depth_map)

        bbox = np.array([100, 100, 200, 200])
        depth = result.get_depth_in_box(bbox, method="center")
        assert depth == pytest.approx(15.0)

        depth = result.get_depth_in_box(bbox, method="median")
        assert depth == pytest.approx(15.0)


# =============================================================================
# BEV Perception Tests
# =============================================================================


class TestBEVPerception:
    """Tests for BEV perception module."""

    def test_camera_config_dataclass(self, bev_perception_module):
        """Test CameraConfig dataclass."""
        CameraConfig = bev_perception_module.CameraConfig

        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1000  # fx
        intrinsic[1, 1] = 1000  # fy
        intrinsic[0, 2] = 800  # cx
        intrinsic[1, 2] = 450  # cy

        extrinsic = np.eye(4)

        config = CameraConfig(
            name="CAM_FRONT",
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_size=(1600, 900),
        )

        assert config.name == "CAM_FRONT"
        assert config.fx == 1000
        assert config.fy == 1000
        assert config.cx == 800
        assert config.cy == 450

    def test_bev_config_dataclass(self, bev_perception_module):
        """Test BEVConfig dataclass."""
        BEVConfig = bev_perception_module.BEVConfig

        config = BEVConfig(
            model="sparsebev",
            backbone="resnet50",
            bev_size=(200, 200),
        )

        assert config.model == "sparsebev"
        assert config.backbone == "resnet50"
        assert config.bev_size == (200, 200)
        assert config.bev_resolution == pytest.approx(0.512)  # 102.4 / 200

    def test_detection_3d_dataclass(self, bev_perception_module):
        """Test Detection3D dataclass for BEV."""
        Detection3D = bev_perception_module.Detection3D

        det = Detection3D(
            center=np.array([15.0, 3.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.5,
            velocity=np.array([5.0, 0.5]),
            confidence=0.85,
            class_id=0,
            class_name="car",
        )

        assert det.x == pytest.approx(15.0)
        assert det.y == pytest.approx(3.0)
        assert det.z == pytest.approx(0.5)
        assert det.distance == pytest.approx(np.sqrt(225 + 9))

        # Test corners
        corners = det.get_corners_3d()
        assert corners.shape == (8, 3)

        # Test BEV box
        bev_box = det.get_bev_box()
        assert bev_box.shape == (4, 2)

    def test_bev_result_dataclass(self, bev_perception_module):
        """Test BEVResult dataclass."""
        Detection3D = bev_perception_module.Detection3D
        BEVResult = bev_perception_module.BEVResult

        dets = [
            Detection3D(
                center=np.array([10.0, 2.0, 0.5]),
                size=np.array([2.0, 4.5, 1.5]),
                rotation=0.0,
                confidence=0.9,
                class_name="car",
            ),
            Detection3D(
                center=np.array([60.0, 10.0, 0.5]),
                size=np.array([2.0, 4.5, 1.5]),
                rotation=0.0,
                confidence=0.8,
                class_name="car",
            ),
        ]

        result = BEVResult(detections=dets, inference_time=0.05)

        assert result.num_detections == 2
        assert result.inference_time == 0.05

        # Test filtering by distance
        close_dets = result.get_detections_in_range(50.0)
        assert len(close_dets) == 1  # Only first detection is within 50m

    def test_bev_factory_supported_models(self, bev_perception_module):
        """Test BEVPerceptionFactory supported models."""
        BEVPerceptionFactory = bev_perception_module.BEVPerceptionFactory

        expected = ["sparsebev", "bevformer", "sparse4d", "streampetr"]
        assert BEVPerceptionFactory.SUPPORTED_MODELS == expected

    def test_nuscenes_classes(self, bev_perception_module):
        """Test nuScenes class list."""
        classes = bev_perception_module.NUSCENES_CLASSES

        assert "car" in classes
        assert "pedestrian" in classes
        assert "bicycle" in classes
        assert len(classes) == 10


# =============================================================================
# 3D Tracking Tests
# =============================================================================


class TestTracking3D:
    """Tests for 3D tracking module."""

    def test_track_state_dataclass(self, tracking_3d_module):
        """Test TrackState dataclass."""
        TrackState = tracking_3d_module.TrackState

        state = TrackState(x=10.0, y=2.0, z=0.5, yaw=0.1, length=4.5, width=2.0, height=1.5)

        assert state.x == 10.0
        assert state.y == 2.0
        assert state.z == 0.5

        vec = state.to_vector()
        assert len(vec) == 10

    def test_track_state_from_detection(self, tracking_3d_module):
        """Test TrackState.from_detection."""
        TrackState = tracking_3d_module.TrackState

        det = {
            "center": [10.0, 2.0, 0.5],
            "size": [2.0, 4.5, 1.5],
            "rotation": 0.1,
            "velocity": [5.0, 0.5],
        }

        state = TrackState.from_detection(det)

        assert state.x == 10.0
        assert state.y == 2.0
        assert state.z == 0.5
        assert state.vx == 5.0

    def test_track_3d_dataclass(self, tracking_3d_module):
        """Test Track3D dataclass."""
        Track3D = tracking_3d_module.Track3D
        TrackState = tracking_3d_module.TrackState

        state = TrackState(x=10.0, y=2.0, z=0.5)
        track = Track3D(
            track_id=1,
            state=state,
            class_id=0,
            class_name="car",
            confidence=0.9,
            hits=5,
            time_since_update=0,
        )

        assert track.track_id == 1
        assert track.is_confirmed  # hits >= 3
        assert not track.is_dead  # time_since_update <= 5

    def test_kalman_filter_3d(self, tracking_3d_module):
        """Test KalmanFilter3D prediction and update."""
        KalmanFilter3D = tracking_3d_module.KalmanFilter3D

        initial_state = np.array([10.0, 2.0, 0.5, 0.1, 4.5, 2.0, 1.5, 5.0, 0.5, 0.0])
        kf = KalmanFilter3D(initial_state, dt=0.1)

        # Predict
        predicted = kf.predict()
        assert len(predicted) == 10
        # Position should move by velocity * dt
        assert predicted[0] == pytest.approx(10.0 + 5.0 * 0.1, rel=0.1)

        # Update with measurement
        measurement = np.array([10.6, 2.1, 0.5, 0.1, 4.5, 2.0, 1.5])
        updated = kf.update(measurement)
        assert len(updated) == 10

    def test_compute_3d_iou(self, tracking_3d_module):
        """Test 3D IoU computation."""
        compute_3d_iou = tracking_3d_module.compute_3d_iou

        # Same box should have IoU = 1
        box = np.array([10.0, 2.0, 0.5, 4.5, 2.0, 1.5, 0.0])
        iou = compute_3d_iou(box, box)
        assert iou == pytest.approx(1.0)

        # Non-overlapping boxes
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0])
        box2 = np.array([10.0, 10.0, 0.0, 2.0, 2.0, 2.0, 0.0])
        iou = compute_3d_iou(box1, box2)
        assert iou == pytest.approx(0.0)

    def test_compute_bev_iou(self, tracking_3d_module):
        """Test BEV IoU computation."""
        compute_bev_iou = tracking_3d_module.compute_bev_iou

        # Same box
        box = np.array([10.0, 2.0, 0.5, 4.5, 2.0, 1.5, 0.0])
        iou = compute_bev_iou(box, box)
        assert iou == pytest.approx(1.0)

    def test_ab3dmot_tracker_init(self, tracking_3d_module):
        """Test AB3DMOT tracker initialization."""
        AB3DMOTTracker = tracking_3d_module.AB3DMOTTracker

        tracker = AB3DMOTTracker(max_age=5, min_hits=3, iou_threshold=0.01)

        assert tracker.max_age == 5
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0

    def test_ab3dmot_tracker_update(self, tracking_3d_module):
        """Test AB3DMOT tracker update."""
        AB3DMOTTracker = tracking_3d_module.AB3DMOTTracker

        tracker = AB3DMOTTracker()

        # First detection
        dets = [
            {
                "center": [10.0, 2.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "car",
            }
        ]

        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

        # Second frame with same detection
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].hits == 2

    def test_ab3dmot_tracker_multiple_objects(self, tracking_3d_module):
        """Test AB3DMOT with multiple objects."""
        AB3DMOTTracker = tracking_3d_module.AB3DMOTTracker

        tracker = AB3DMOTTracker()

        dets = [
            {
                "center": [10.0, 2.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.9,
            },
            {
                "center": [20.0, -3.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.8,
            },
        ]

        tracks = tracker.update(dets)
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_simpletrack_3d(self, tracking_3d_module):
        """Test SimpleTrack3D tracker."""
        SimpleTrack3D = tracking_3d_module.SimpleTrack3D

        tracker = SimpleTrack3D(high_score_threshold=0.6, low_score_threshold=0.1)

        dets = [
            {
                "center": [10.0, 2.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.9,
            },
        ]

        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_tracker_factory(self, tracking_3d_module):
        """Test Tracker3DFactory."""
        Tracker3DFactory = tracking_3d_module.Tracker3DFactory

        # Create AB3DMOT
        tracker = Tracker3DFactory.create("ab3dmot", {"max_age": 5})
        assert tracker is not None

        # Create SimpleTrack
        tracker = Tracker3DFactory.create("simpletrack", {})
        assert tracker is not None

        # Invalid tracker
        with pytest.raises(ValueError):
            Tracker3DFactory.create("invalid_tracker", {})

    def test_tracking_result_dataclass(self, tracking_3d_module):
        """Test TrackingResult dataclass."""
        TrackingResult = tracking_3d_module.TrackingResult
        Track3D = tracking_3d_module.Track3D
        TrackState = tracking_3d_module.TrackState

        tracks = [
            Track3D(track_id=1, state=TrackState(), hits=5),
            Track3D(track_id=2, state=TrackState(), hits=2),
        ]

        result = TrackingResult(tracks=tracks, frame_id=10, inference_time=0.01)

        assert result.num_tracks == 2
        assert len(result.confirmed_tracks) == 1  # Only first has hits >= 3


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """Tests for end-to-end autonomous driving module."""

    def test_detection_3d_dataclass(self, end_to_end_module):
        """Test Detection3D dataclass."""
        Detection3D = end_to_end_module.Detection3D

        det = Detection3D(
            center=np.array([10.0, 2.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.1,
            velocity=np.array([5.0, 0.5]),
            confidence=0.9,
            class_name="car",
            track_id=1,
        )

        assert det.track_id == 1
        d = det.to_dict()
        assert "center" in d
        assert "track_id" in d

    def test_motion_forecast_dataclass(self, end_to_end_module):
        """Test MotionForecast dataclass."""
        MotionForecast = end_to_end_module.MotionForecast

        modes = [
            np.array([[1, 0], [2, 0], [3, 0]]),
            np.array([[1, 1], [2, 2], [3, 3]]),
        ]
        probs = [0.7, 0.3]

        forecast = MotionForecast(
            track_id=1,
            modes=modes,
            probabilities=probs,
            timestamps=np.array([0.5, 1.0, 1.5]),
        )

        assert forecast.num_modes == 2
        assert forecast.best_mode is not None
        assert len(forecast.best_mode) == 3

    def test_planning_output_dataclass(self, end_to_end_module):
        """Test PlanningOutput dataclass."""
        PlanningOutput = end_to_end_module.PlanningOutput

        trajectory = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        timestamps = np.array([0, 0.5, 1.0, 1.5])

        plan = PlanningOutput(
            trajectory=trajectory,
            timestamps=timestamps,
            confidence=0.9,
        )

        assert plan.horizon == pytest.approx(1.5)
        # Just test that trajectory is stored correctly
        assert plan.trajectory.shape == (4, 2)
        assert plan.confidence == 0.9

    def test_uniad_config_dataclass(self, end_to_end_module):
        """Test UniADConfig dataclass."""
        UniADConfig = end_to_end_module.UniADConfig

        config = UniADConfig(
            backbone="resnet50",
            num_queries=900,
            forecast_horizon=3.0,
        )

        assert config.backbone == "resnet50"
        assert config.num_queries == 900
        assert config.forecast_horizon == 3.0

    def test_uniad_output_dataclass(self, end_to_end_module):
        """Test UniADOutput dataclass."""
        UniADOutput = end_to_end_module.UniADOutput
        Detection3D = end_to_end_module.Detection3D

        dets = [
            Detection3D(
                center=np.array([10.0, 2.0, 0.5]),
                size=np.array([2.0, 4.5, 1.5]),
                rotation=0.0,
                confidence=0.9,
            )
        ]

        output = UniADOutput(
            detections=dets,
            tracks=[],
            map_elements=[],
            motion_forecasts=[],
            frame_id=1,
        )

        assert output.num_detections == 1
        assert output.num_tracks == 0

    def test_end_to_end_factory(self, end_to_end_module):
        """Test EndToEndFactory."""
        EndToEndFactory = end_to_end_module.EndToEndFactory

        expected = ["uniad", "vad", "bevplanner"]
        assert EndToEndFactory.SUPPORTED_MODELS == expected

    def test_task_type_enum(self, end_to_end_module):
        """Test TaskType enum."""
        TaskType = end_to_end_module.TaskType

        assert TaskType.DETECTION.value == "detection"
        assert TaskType.PLANNING.value == "planning"


# =============================================================================
# Sparse4D v3 Tests
# =============================================================================


class TestSparse4Dv3:
    """Tests for Sparse4D v3 unified detection and tracking."""

    def test_anchor_4d_dataclass(self, sparse4d_module):
        """Test Anchor4D dataclass."""
        Anchor4D = sparse4d_module.Anchor4D

        anchor = Anchor4D(
            position=np.array([10.0, 2.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.1,
            velocity=np.array([5.0, 0.5]),
            track_id=1,
            confidence=0.9,
            class_name="car",
        )

        assert anchor.track_id == 1
        assert anchor.is_valid  # confidence > 0.1, age < 10

        d = anchor.to_dict()
        assert "center" in d
        assert "track_id" in d

    def test_anchor_4d_propagate(self, sparse4d_module):
        """Test Anchor4D propagation."""
        Anchor4D = sparse4d_module.Anchor4D

        anchor = Anchor4D(
            position=np.array([10.0, 2.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.0,
            velocity=np.array([5.0, 0.0]),  # Moving forward at 5 m/s
            track_id=1,
            confidence=0.9,
        )

        # Propagate with identity ego motion
        ego_motion = np.eye(4)
        new_anchor = anchor.propagate(ego_motion, dt=0.1)

        # Position should move by velocity * dt
        assert new_anchor.position[0] == pytest.approx(10.5, rel=0.1)  # 10 + 5*0.1
        assert new_anchor.track_id == 1  # ID preserved
        assert new_anchor.age == 1  # Age incremented

    def test_sparse4d_config_dataclass(self, sparse4d_module):
        """Test Sparse4DConfig dataclass."""
        Sparse4DConfig = sparse4d_module.Sparse4DConfig

        config = Sparse4DConfig(
            num_anchors=900,
            hidden_dim=256,
            confidence_threshold=0.3,
        )

        assert config.num_anchors == 900
        assert config.hidden_dim == 256

    def test_sparse4d_result_dataclass(self, sparse4d_module):
        """Test Sparse4DResult dataclass."""
        Sparse4DResult = sparse4d_module.Sparse4DResult
        Anchor4D = sparse4d_module.Anchor4D

        anchors = [
            Anchor4D(
                position=np.array([10.0, 2.0, 0.5]),
                size=np.array([2.0, 4.5, 1.5]),
                rotation=0.0,
                velocity=np.array([0.0, 0.0]),
                track_id=1,
                confidence=0.9,
            )
        ]

        result = Sparse4DResult(anchors=anchors, frame_id=1)

        assert result.num_objects == 1
        assert len(result.detections) == 1
        assert len(result.tracks) == 1

    def test_sparse4d_tracker_init(self, sparse4d_module):
        """Test Sparse4DTracker initialization."""
        Sparse4DTracker = sparse4d_module.Sparse4DTracker
        Sparse4DConfig = sparse4d_module.Sparse4DConfig

        config = Sparse4DConfig(num_anchors=100)
        tracker = Sparse4DTracker(config)

        assert len(tracker.anchors) == 0
        assert tracker.next_track_id == 1

    def test_sparse4d_tracker_update(self, sparse4d_module):
        """Test Sparse4DTracker update."""
        Sparse4DTracker = sparse4d_module.Sparse4DTracker
        Sparse4DConfig = sparse4d_module.Sparse4DConfig

        config = Sparse4DConfig(confidence_threshold=0.3)
        tracker = Sparse4DTracker(config)

        # First detection
        dets = [
            {
                "center": [10.0, 2.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.9,
                "class_name": "car",
            }
        ]

        result = tracker.update(dets)
        assert result.num_objects == 1
        assert result.anchors[0].track_id == 1

        # Second frame - object moved slightly
        dets = [
            {
                "center": [10.5, 2.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": 0.0,
                "confidence": 0.9,
                "class_name": "car",
            }
        ]

        result = tracker.update(dets)
        assert result.num_objects == 1
        assert result.anchors[0].track_id == 1  # Same track ID

    def test_sparse4d_tracker_multiple_objects(self, sparse4d_module):
        """Test Sparse4DTracker with multiple objects."""
        Sparse4DTracker = sparse4d_module.Sparse4DTracker
        Sparse4DConfig = sparse4d_module.Sparse4DConfig

        config = Sparse4DConfig()
        tracker = Sparse4DTracker(config)

        dets = [
            {"center": [10.0, 2.0, 0.5], "size": [2.0, 4.5, 1.5], "confidence": 0.9},
            {"center": [25.0, -5.0, 0.5], "size": [2.0, 4.5, 1.5], "confidence": 0.8},
        ]

        result = tracker.update(dets)
        assert result.num_objects == 2

        # Track IDs should be different
        ids = [a.track_id for a in result.anchors]
        assert len(set(ids)) == 2

    def test_sparse4d_tracker_reset(self, sparse4d_module):
        """Test Sparse4DTracker reset."""
        Sparse4DTracker = sparse4d_module.Sparse4DTracker
        Sparse4DConfig = sparse4d_module.Sparse4DConfig

        tracker = Sparse4DTracker(Sparse4DConfig())

        dets = [{"center": [10.0, 2.0, 0.5], "size": [2.0, 4.5, 1.5], "confidence": 0.9}]
        tracker.update(dets)
        assert len(tracker.anchors) == 1

        tracker.reset()
        assert len(tracker.anchors) == 0
        assert tracker.next_track_id == 1

    def test_create_sparse4d_tracker_function(self, sparse4d_module):
        """Test create_sparse4d_tracker node function."""
        create_sparse4d_tracker = sparse4d_module.create_sparse4d_tracker

        tracker = create_sparse4d_tracker({"num_anchors": 500, "confidence_threshold": 0.4})
        assert tracker is not None
        assert tracker.config.num_anchors == 500


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests across modules."""

    def test_depth_to_3d_tracking_pipeline(self, depth_estimation_module, tracking_3d_module):
        """Test pipeline from depth estimation to 3D tracking."""
        Detection3D = depth_estimation_module.Detection3D
        AB3DMOTTracker = tracking_3d_module.AB3DMOTTracker

        # Simulated 3D detections from depth lifting
        # Note: depth_estimation.Detection3D uses position_3d, not center
        det_3d = Detection3D(
            bbox_2d=np.array([100, 200, 200, 300]),
            position_3d=np.array([10.0, 2.0, 15.0]),
            depth=15.0,
            confidence=0.9,
            class_id=0,
            class_name="car",
        )

        # Convert to tracker format (tracker expects center, size, rotation)
        det_dict = {
            "center": det_3d.position_3d.tolist(),
            "size": [2.0, 4.5, 1.5],  # Estimated size
            "rotation": 0.0,
            "confidence": det_3d.confidence,
            "class_name": det_3d.class_name,
        }

        # Track
        tracker = AB3DMOTTracker()
        tracks = tracker.update([det_dict])

        assert len(tracks) == 1
        assert tracks[0].state.x == pytest.approx(10.0)

    def test_bev_to_end_to_end_compatibility(self, bev_perception_module, end_to_end_module):
        """Test data format compatibility between BEV and end-to-end."""
        BEVDetection3D = bev_perception_module.Detection3D
        E2EDetection3D = end_to_end_module.Detection3D

        # Both should have similar structure
        bev_det = BEVDetection3D(
            center=np.array([10.0, 2.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.0,
            confidence=0.9,
        )

        e2e_det = E2EDetection3D(
            center=np.array([10.0, 2.0, 0.5]),
            size=np.array([2.0, 4.5, 1.5]),
            rotation=0.0,
            confidence=0.9,
        )

        # Both should have to_dict method
        bev_dict = bev_det.to_dict()
        e2e_dict = e2e_det.to_dict()

        assert "center" in bev_dict
        assert "center" in e2e_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
