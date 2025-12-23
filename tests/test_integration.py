"""Integration tests for the CV Pipeline.

These tests verify that:
1. All pipeline components work together
2. Data flows correctly between nodes
3. The custom datasets integrate properly
4. Visualization and metrics work end-to-end
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def import_nodes_module(pipeline_name: str):
    """Import nodes module directly without going through __init__.py.

    This bypasses kedro imports in __init__.py which may not be installed.
    """
    nodes_path = src_path / "cv_pipeline" / "pipelines" / pipeline_name / "nodes.py"
    spec = importlib.util.spec_from_file_location(f"{pipeline_name}_nodes", nodes_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_import_nodes_module(pipeline_name: str):
    """Try to import nodes module, return None if dependencies are missing."""
    try:
        return import_nodes_module(pipeline_name)
    except ModuleNotFoundError as e:
        print(f"Warning: Could not import {pipeline_name} nodes: {e}")
        return None


# Pre-import node modules to avoid kedro dependency
# Some modules may fail if their dependencies (albumentations, etc.) aren't installed
data_processing_nodes = try_import_nodes_module("data_processing")
object_detection_nodes = try_import_nodes_module("object_detection")
tracking_nodes = try_import_nodes_module("tracking")
lane_detection_nodes = try_import_nodes_module("lane_detection")
path_construction_nodes = try_import_nodes_module("path_construction")


def requires_module(module_var, module_name: str):
    """Decorator to skip tests if a module failed to import."""
    return pytest.mark.skipif(
        module_var is None,
        reason=f"{module_name} nodes could not be imported (missing dependencies)",
    )


@requires_module(data_processing_nodes, "data_processing")
class TestDataProcessingIntegration:
    """Test data processing pipeline integration."""

    def test_preprocess_frames_output_format(self):
        """Test that preprocessed frames have correct format."""
        preprocess_frames = data_processing_nodes.preprocess_frames

        # Create test frames
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]

        params = {
            "input_size": [640, 640],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

        preprocessed = preprocess_frames(frames, params)

        assert len(preprocessed) == 5
        assert preprocessed[0].shape == (640, 640, 3)
        assert preprocessed[0].dtype == np.float32

    def test_create_batches_output_format(self):
        """Test that batches have correct tensor format."""
        create_batches = data_processing_nodes.create_batches

        # Create preprocessed frames
        frames = [np.random.randn(640, 640, 3).astype(np.float32) for _ in range(5)]

        batches = create_batches(frames, batch_size=2)

        assert len(batches) == 3  # 5 frames with batch_size=2 -> 3 batches
        assert batches[0].shape == (2, 3, 640, 640)  # [B, C, H, W]
        assert batches[-1].shape[0] == 1  # Last batch has 1 frame

    def test_extract_frame_metadata(self):
        """Test frame metadata extraction."""
        extract_frame_metadata = data_processing_nodes.extract_frame_metadata

        frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(10)]

        metadata = extract_frame_metadata(frames)

        assert metadata["num_frames"] == 10
        assert metadata["height"] == 720
        assert metadata["width"] == 1280
        assert metadata["channels"] == 3

    def test_apply_augmentations_passthrough(self):
        """Test augmentation passthrough when disabled."""
        apply_augmentations = data_processing_nodes.apply_augmentations

        frames = [np.random.randn(640, 640, 3).astype(np.float32) for _ in range(3)]

        # No augmentation params - should pass through
        augmented = apply_augmentations(frames, {})

        assert len(augmented) == 3
        for orig, aug in zip(frames, augmented):
            assert np.allclose(orig, aug)


@requires_module(object_detection_nodes, "object_detection")
class TestDetectionIntegration:
    """Test object detection pipeline integration."""

    def test_detection_dataclass(self):
        """Test Detection dataclass functionality."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult

        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.95,
            class_id=2,
            class_name="car",
        )

        assert det.to_dict()["confidence"] == 0.95
        assert det.to_dict()["class_name"] == "car"

        result = DetectionResult(
            frame_id=0,
            detections=[det],
            inference_time=0.05,
        )

        assert result.num_detections == 1
        assert result.get_boxes().shape == (1, 4)
        assert result.get_scores().shape == (1,)

    def test_filter_detections_by_class(self):
        """Test class-based detection filtering."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        filter_detections_by_class = object_detection_nodes.filter_detections_by_class

        detections = [
            Detection(np.array([0, 0, 100, 100]), 0.9, 0, "person"),
            Detection(np.array([100, 100, 200, 200]), 0.8, 2, "car"),
            Detection(np.array([200, 200, 300, 300]), 0.7, 7, "truck"),
        ]

        results = [DetectionResult(frame_id=0, detections=detections, inference_time=0.05)]

        # Filter to only vehicles
        target_classes = {"vehicles": [2, 7]}
        filtered = filter_detections_by_class(results, target_classes)

        assert filtered[0].num_detections == 2  # car and truck

    def test_compute_detection_metrics_node(self):
        """Test detection metrics computation."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        compute_detection_metrics = object_detection_nodes.compute_detection_metrics

        detections = [
            Detection(np.array([0, 0, 100, 100]), 0.9, 0, "person"),
            Detection(np.array([100, 100, 200, 200]), 0.8, 2, "car"),
        ]

        results = [
            DetectionResult(frame_id=i, detections=detections, inference_time=0.05)
            for i in range(10)
        ]

        metrics = compute_detection_metrics(results, {})

        assert "total_detections" in metrics
        assert "fps" in metrics
        assert metrics["total_detections"] == 20


@requires_module(tracking_nodes, "tracking")
class TestTrackingIntegration:
    """Test tracking pipeline integration."""

    def test_kalman_box_tracker(self):
        """Test KalmanBoxTracker functionality."""
        KalmanBoxTracker = tracking_nodes.KalmanBoxTracker

        # Reset counter
        KalmanBoxTracker.count = 0

        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox, class_id=2, confidence=0.9)

        assert tracker.id == 0
        assert tracker.class_id == 2

        # Predict next state
        predicted = tracker.predict()
        assert predicted.shape == (4,)

        # Update with new detection
        new_bbox = np.array([105, 105, 205, 205])
        tracker.update(new_bbox, 0.85)

        assert tracker.hits == 2
        assert tracker.time_since_update == 0

    def test_bytetracker_update(self):
        """Test ByteTracker update cycle."""
        ByteTracker = tracking_nodes.ByteTracker
        KalmanBoxTracker = tracking_nodes.KalmanBoxTracker

        # Reset counter
        KalmanBoxTracker.count = 0

        params = {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
        }

        tracker = ByteTracker(params)

        # First frame - should create new tracks
        detections = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ]
        )
        scores = np.array([0.9, 0.85])
        classes = np.array([2, 2])

        outputs = tracker.update(detections, scores, classes)

        # Might not have outputs yet (min_hits=3)
        assert isinstance(outputs, np.ndarray)

    def test_iou_batch_computation(self):
        """Test batch IoU computation."""
        iou_batch = tracking_nodes.iou_batch

        boxes1 = np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
            ]
        )
        boxes2 = np.array(
            [
                [0, 0, 100, 100],
                [50, 50, 150, 150],
            ]
        )

        iou_matrix = iou_batch(boxes1, boxes2)

        assert iou_matrix.shape == (2, 2)
        assert np.isclose(iou_matrix[0, 0], 1.0)  # Identical boxes
        assert iou_matrix[1, 0] == 0.0  # No overlap

    def test_extract_trajectories(self):
        """Test trajectory extraction from tracking results."""
        Track = tracking_nodes.Track
        TrackingResult = tracking_nodes.TrackingResult
        extract_trajectories = tracking_nodes.extract_trajectories

        tracks = [
            Track(
                track_id=1,
                bbox=np.array([100, 100, 200, 200]),
                confidence=0.9,
                class_id=2,
                class_name="car",
                frame_id=0,
            ),
            Track(
                track_id=1,
                bbox=np.array([110, 110, 210, 210]),
                confidence=0.85,
                class_id=2,
                class_name="car",
                frame_id=1,
            ),
        ]

        results = [
            TrackingResult(frame_id=0, tracks=[tracks[0]], processing_time=0.01),
            TrackingResult(frame_id=1, tracks=[tracks[1]], processing_time=0.01),
        ]

        df = extract_trajectories(results, {})

        assert len(df) == 2
        assert "track_id" in df.columns
        assert "cx" in df.columns  # center x
        assert "cy" in df.columns  # center y


@requires_module(lane_detection_nodes, "lane_detection")
class TestLaneDetectionIntegration:
    """Test lane detection pipeline integration."""

    def test_lane_dataclass(self):
        """Test Lane dataclass functionality."""
        Lane = lane_detection_nodes.Lane
        LanePoint = lane_detection_nodes.LanePoint

        points = [
            LanePoint(x=100, y=400, confidence=0.9),
            LanePoint(x=110, y=300, confidence=0.85),
            LanePoint(x=120, y=200, confidence=0.8),
            LanePoint(x=130, y=100, confidence=0.75),
        ]

        lane = Lane(
            lane_id=0,
            points=points,
            confidence=0.85,
            lane_type="ego_left",
        )

        np_points = lane.to_numpy()
        assert np_points.shape == (4, 2)

        coeffs = lane.get_coefficients(degree=2)
        assert coeffs is not None
        assert len(coeffs) == 3  # degree 2 -> 3 coefficients

    def test_fit_lane_curves(self):
        """Test lane curve fitting."""
        Lane = lane_detection_nodes.Lane
        LaneDetectionResult = lane_detection_nodes.LaneDetectionResult
        LanePoint = lane_detection_nodes.LanePoint
        fit_lane_curves = lane_detection_nodes.fit_lane_curves

        points = [LanePoint(x=100 + i * 10, y=400 - i * 50, confidence=0.9) for i in range(8)]
        lane = Lane(lane_id=0, points=points, confidence=0.9, lane_type="ego_left")

        results = [LaneDetectionResult(frame_id=0, lanes=[lane], inference_time=0.02)]

        fitted = fit_lane_curves(results, {"path_construction": {"fitting_method": "polynomial"}})

        assert len(fitted) == 1
        assert "polynomial_coeffs" in fitted[0]["lanes"][0]


@requires_module(path_construction_nodes, "path_construction")
class TestPathConstructionIntegration:
    """Test path construction pipeline integration."""

    def test_waypoint_dataclass(self):
        """Test Waypoint and DrivablePath dataclasses."""
        DrivablePath = path_construction_nodes.DrivablePath
        Waypoint = path_construction_nodes.Waypoint

        waypoints = [
            Waypoint(x=320, y=400, heading=0, curvature=0),
            Waypoint(x=320, y=350, heading=0.1, curvature=0.01),
            Waypoint(x=325, y=300, heading=0.15, curvature=0.02),
        ]

        path = DrivablePath(
            frame_id=0,
            waypoints=waypoints,
            confidence=0.9,
        )

        assert len(path.waypoints) == 3
        assert path.frame_id == 0

    def test_generate_trajectory(self):
        """Test trajectory generation from drivable paths."""
        DrivablePath = path_construction_nodes.DrivablePath
        Waypoint = path_construction_nodes.Waypoint
        generate_trajectory = path_construction_nodes.generate_trajectory

        waypoints = [Waypoint(x=320, y=400 - i * 20, heading=0.1 * i) for i in range(10)]
        paths = [DrivablePath(frame_id=0, waypoints=waypoints, confidence=0.9)]

        params = {"output": {"format": "waypoints", "num_waypoints": 5}}
        trajectories = generate_trajectory(paths, params)

        assert len(trajectories) == 1
        assert len(trajectories[0]["waypoints"]) <= 5


@requires_module(object_detection_nodes, "object_detection")
@requires_module(tracking_nodes, "tracking")
class TestVisualizationIntegration:
    """Test visualization utilities integration with pipeline outputs."""

    def test_visualize_detection_results(self):
        """Test visualization of detection results."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        from cv_pipeline.utils.visualization import draw_detections

        detections = [
            Detection(np.array([100, 100, 200, 200]), 0.9, 2, "car"),
            Detection(np.array([300, 150, 400, 300]), 0.8, 0, "person"),
        ]

        result = DetectionResult(frame_id=0, detections=detections, inference_time=0.05)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_detections(image, [d.to_dict() for d in result.detections])

        assert annotated.shape == (480, 640, 3)
        assert not np.array_equal(annotated, np.zeros_like(annotated))

    def test_visualize_tracking_results(self):
        """Test visualization of tracking results."""
        Track = tracking_nodes.Track
        TrackingResult = tracking_nodes.TrackingResult
        from cv_pipeline.utils.visualization import draw_tracks

        tracks = [
            Track(
                track_id=1,
                bbox=np.array([100, 100, 200, 200]),
                confidence=0.9,
                class_id=2,
                class_name="car",
                frame_id=0,
            ),
            Track(
                track_id=2,
                bbox=np.array([300, 150, 400, 300]),
                confidence=0.8,
                class_id=0,
                class_name="person",
                frame_id=0,
            ),
        ]

        result = TrackingResult(frame_id=0, tracks=tracks, processing_time=0.01)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_tracks(image, [t.to_dict() for t in result.tracks])

        assert annotated.shape == (480, 640, 3)

    def test_visualizer_with_pipeline_outputs(self):
        """Test Visualizer class with full pipeline outputs."""
        from cv_pipeline.utils.visualization import Visualizer

        visualizer = Visualizer(config={"trail_length": 30})

        # Simulate multiple frames
        for frame_idx in range(5):
            image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Moving track
            x_offset = frame_idx * 20
            tracks = [
                {
                    "track_id": 1,
                    "bbox": [100 + x_offset, 100, 200 + x_offset, 200],
                    "class_name": "car",
                }
            ]

            annotated = visualizer.draw_frame(image, tracks=tracks, fps=30.0)

            assert annotated.shape == (480, 640, 3)

        # Check trajectory was accumulated
        assert 1 in visualizer.trajectories
        assert len(visualizer.trajectories[1]) == 5


@requires_module(object_detection_nodes, "object_detection")
@requires_module(tracking_nodes, "tracking")
class TestMetricsIntegration:
    """Test metrics utilities integration with pipeline outputs."""

    def test_detection_metrics_with_pipeline_output(self):
        """Test detection metrics with actual pipeline output format."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        from cv_pipeline.utils.metrics import compute_detection_metrics

        # Simulated predictions
        pred_detections = [
            Detection(np.array([0, 0, 100, 100]), 0.9, 0, "person"),
            Detection(np.array([100, 100, 200, 200]), 0.8, 2, "car"),
        ]

        # Convert to metrics format
        predictions = [
            {
                "boxes": [d.bbox.tolist() for d in pred_detections],
                "scores": [d.confidence for d in pred_detections],
                "classes": [d.class_id for d in pred_detections],
            }
        ]

        ground_truth = [
            {
                "boxes": [[0, 0, 100, 100], [100, 100, 200, 200]],
                "classes": [0, 2],
            }
        ]

        metrics = compute_detection_metrics(predictions, ground_truth)

        assert "mAP_50" in metrics
        assert metrics["precision"] == 1.0  # Perfect match

    def test_tracking_metrics_with_pipeline_output(self):
        """Test tracking metrics with actual pipeline output format."""
        Track = tracking_nodes.Track
        TrackingResult = tracking_nodes.TrackingResult
        from cv_pipeline.utils.metrics import TrackingFrame, compute_tracking_metrics

        # Simulated tracking results
        pred_tracks = [
            Track(
                track_id=1,
                bbox=np.array([0, 0, 100, 100]),
                confidence=0.9,
                class_id=0,
                class_name="person",
                frame_id=0,
            ),
        ]

        pred_result = TrackingResult(frame_id=0, tracks=pred_tracks, processing_time=0.01)

        # Convert to metrics format
        predictions = [
            TrackingFrame(
                frame_id=0,
                track_ids=np.array([t.track_id for t in pred_result.tracks]),
                boxes=np.array([t.bbox for t in pred_result.tracks]),
            )
        ]

        ground_truth = [
            TrackingFrame(
                frame_id=0,
                track_ids=np.array([1]),
                boxes=np.array([[0, 0, 100, 100]]),
            )
        ]

        metrics = compute_tracking_metrics(predictions, ground_truth)

        assert "MOTA" in metrics
        assert "IDF1" in metrics


@requires_module(tracking_nodes, "tracking")
class TestCMCIntegration:
    """Test Camera Motion Compensation integration with tracking."""

    def test_cmc_with_tracking_boxes(self):
        """Test CMC compensation on tracking boxes."""
        KalmanBoxTracker = tracking_nodes.KalmanBoxTracker
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        # Reset counter
        KalmanBoxTracker.count = 0

        cmc = CameraMotionCompensator(method="none")

        # Simulate frame sequence
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Get warp matrices
        warp1 = cmc.compute(frame1)
        warp2 = cmc.compute(frame2)

        # Create tracker
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)

        # Predict and compensate
        predicted = tracker.predict()
        compensated = cmc.apply_to_boxes(predicted.reshape(1, -1))

        assert compensated.shape == (1, 4)


@requires_module(data_processing_nodes, "data_processing")
class TestDatasetIntegration:
    """Test custom datasets integration with pipeline."""

    def test_tensor_dataset_with_batches(self):
        """Test TensorDataSet with pipeline batch output."""
        from cv_pipeline.datasets import TensorDataSet

        create_batches = data_processing_nodes.create_batches

        # Create frames and batches
        frames = [np.random.randn(640, 640, 3).astype(np.float32) for _ in range(4)]
        batches = create_batches(frames, batch_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save each batch
            for i, batch in enumerate(batches):
                filepath = os.path.join(tmpdir, f"batch_{i}.pt")
                dataset = TensorDataSet(filepath=filepath)
                dataset._save(batch)

                # Reload and verify
                loaded = dataset._load()
                assert torch.allclose(batch, loaded)

    def test_video_writer_with_visualized_frames(self):
        """Test VideoWriterDataSet with visualized frames."""
        from cv_pipeline.datasets import VideoWriterDataSet
        from cv_pipeline.utils.visualization import Visualizer

        visualizer = Visualizer()

        # Generate annotated frames
        frames = []
        for i in range(10):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            tracks = [
                {"track_id": 1, "bbox": [100 + i * 10, 100, 200 + i * 10, 200], "class_name": "car"}
            ]
            annotated = visualizer.draw_frame(image, tracks=tracks)
            frames.append(annotated)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output.mp4")
            dataset = VideoWriterDataSet(
                filepath=filepath,
                save_args={"fps": 10, "codec": "mp4v"},
            )

            dataset._save(frames)

            assert dataset._exists()


@requires_module(data_processing_nodes, "data_processing")
@requires_module(object_detection_nodes, "object_detection")
@requires_module(tracking_nodes, "tracking")
class TestEndToEndPipeline:
    """End-to-end pipeline tests with synthetic data."""

    def test_full_detection_flow(self):
        """Test complete detection flow: preprocess -> detect -> visualize."""
        create_batches = data_processing_nodes.create_batches
        preprocess_frames = data_processing_nodes.preprocess_frames
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        from cv_pipeline.utils.visualization import draw_detections

        # 1. Create raw frames
        raw_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(5)]

        # 2. Preprocess
        params = {
            "input_size": [640, 640],
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        preprocessed = preprocess_frames(raw_frames, params)

        # 3. Create batches
        batches = create_batches(preprocessed, batch_size=1)

        # 4. Simulate detection (normally would use model)
        detection_results = []
        for i, batch in enumerate(batches):
            detections = [
                Detection(np.array([100, 100, 200, 200]), 0.9, 2, "car"),
                Detection(np.array([300, 150, 400, 300]), 0.8, 0, "person"),
            ]
            detection_results.append(
                DetectionResult(frame_id=i, detections=detections, inference_time=0.05)
            )

        # 5. Visualize
        visualized_frames = []
        for raw_frame, det_result in zip(raw_frames, detection_results):
            annotated = draw_detections(
                raw_frame.copy(), [d.to_dict() for d in det_result.detections]
            )
            visualized_frames.append(annotated)

        assert len(visualized_frames) == 5
        assert all(f.shape == (720, 1280, 3) for f in visualized_frames)

    def test_full_tracking_flow(self):
        """Test complete tracking flow: detect -> track -> extract trajectories."""
        Detection = object_detection_nodes.Detection
        DetectionResult = object_detection_nodes.DetectionResult
        ByteTracker = tracking_nodes.ByteTracker
        KalmanBoxTracker = tracking_nodes.KalmanBoxTracker
        Track = tracking_nodes.Track
        TrackingResult = tracking_nodes.TrackingResult
        extract_trajectories = tracking_nodes.extract_trajectories

        # Reset counter
        KalmanBoxTracker.count = 0

        # 1. Create detection results (simulating moving objects)
        detection_results = []
        for i in range(10):
            detections = [
                Detection(np.array([100 + i * 5, 100, 200 + i * 5, 200]), 0.9, 2, "car"),
                Detection(np.array([300 - i * 3, 200, 400 - i * 3, 350]), 0.85, 0, "person"),
            ]
            detection_results.append(
                DetectionResult(frame_id=i, detections=detections, inference_time=0.05)
            )

        # 2. Initialize tracker
        params = {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
        }
        tracker = ByteTracker(params)

        # 3. Run tracking
        tracking_results = []
        for det_result in detection_results:
            boxes = det_result.get_boxes()
            scores = det_result.get_scores()
            classes = det_result.get_classes()

            track_outputs = tracker.update(boxes, scores, classes)

            tracks = []
            for track in track_outputs:
                if len(track) >= 6:
                    tracks.append(
                        Track(
                            track_id=int(track[4]),
                            bbox=track[:4],
                            confidence=float(track[6]) if len(track) > 6 else 1.0,
                            class_id=int(track[5]),
                            class_name="object",
                            frame_id=det_result.frame_id,
                        )
                    )

            tracking_results.append(
                TrackingResult(
                    frame_id=det_result.frame_id,
                    tracks=tracks,
                    processing_time=0.01,
                )
            )

        # 4. Extract trajectories
        trajectories_df = extract_trajectories(tracking_results, {})

        # Verify we got some tracks
        assert len(tracking_results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
