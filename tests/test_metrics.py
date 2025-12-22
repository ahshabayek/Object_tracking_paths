"""Unit tests for metrics utilities."""

import numpy as np
import pytest


class TestIoUComputation:
    """Tests for IoU computation functions."""

    def test_compute_iou_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        from cv_pipeline.utils.metrics import compute_iou

        box = np.array([100, 100, 200, 200])
        iou = compute_iou(box, box)

        assert np.isclose(iou, 1.0)

    def test_compute_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        from cv_pipeline.utils.metrics import compute_iou

        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])

        iou = compute_iou(box1, box2)

        assert np.isclose(iou, 0.0)

    def test_compute_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        from cv_pipeline.utils.metrics import compute_iou

        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])

        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 = 0.1428...
        iou = compute_iou(box1, box2)

        assert 0.1 < iou < 0.2

    def test_compute_iou_matrix(self):
        """Test IoU matrix computation."""
        from cv_pipeline.utils.metrics import compute_iou_matrix

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
                [200, 200, 300, 300],
            ]
        )

        iou_matrix = compute_iou_matrix(boxes1, boxes2)

        assert iou_matrix.shape == (2, 3)
        assert np.isclose(iou_matrix[0, 0], 1.0)  # Identical boxes
        assert np.isclose(iou_matrix[1, 2], 1.0)  # Identical boxes
        assert iou_matrix[0, 2] == 0.0  # No overlap

    def test_compute_iou_matrix_empty(self):
        """Test IoU matrix with empty inputs."""
        from cv_pipeline.utils.metrics import compute_iou_matrix

        boxes1 = np.array([])
        boxes2 = np.array([[0, 0, 100, 100]])

        iou_matrix = compute_iou_matrix(boxes1.reshape(0, 4), boxes2)

        assert iou_matrix.shape == (0, 1)


class TestAveragePrecision:
    """Tests for Average Precision computation."""

    def test_compute_ap_perfect(self):
        """Test AP with perfect predictions."""
        from cv_pipeline.utils.metrics import compute_ap

        precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        ap = compute_ap(precisions, recalls)

        assert np.isclose(ap, 1.0)

    def test_compute_ap_voc07(self):
        """Test AP with VOC 2007 11-point interpolation."""
        from cv_pipeline.utils.metrics import compute_ap

        precisions = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ap = compute_ap(precisions, recalls, use_07_metric=True)

        assert 0 <= ap <= 1

    def test_compute_precision_recall_no_predictions(self):
        """Test PR curve with no predictions."""
        from cv_pipeline.utils.metrics import compute_precision_recall

        pred_boxes = np.array([]).reshape(0, 4)
        pred_scores = np.array([])
        gt_boxes = np.array([[0, 0, 100, 100]])

        precisions, recalls, ap = compute_precision_recall(pred_boxes, pred_scores, gt_boxes)

        assert ap == 0.0

    def test_compute_precision_recall_no_ground_truth(self):
        """Test PR curve with no ground truth."""
        from cv_pipeline.utils.metrics import compute_precision_recall

        pred_boxes = np.array([[0, 0, 100, 100]])
        pred_scores = np.array([0.9])
        gt_boxes = np.array([]).reshape(0, 4)

        precisions, recalls, ap = compute_precision_recall(pred_boxes, pred_scores, gt_boxes)

        assert ap == 0.0


class TestDetectionMetrics:
    """Tests for detection metrics computation."""

    def test_compute_detection_metrics_perfect(self):
        """Test detection metrics with perfect predictions."""
        from cv_pipeline.utils.metrics import compute_detection_metrics

        predictions = [
            {"boxes": [[0, 0, 100, 100]], "scores": [0.9], "classes": [0]},
            {"boxes": [[50, 50, 150, 150]], "scores": [0.85], "classes": [0]},
        ]
        ground_truth = [
            {"boxes": [[0, 0, 100, 100]], "classes": [0]},
            {"boxes": [[50, 50, 150, 150]], "classes": [0]},
        ]

        metrics = compute_detection_metrics(predictions, ground_truth)

        assert "mAP_50" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_compute_detection_metrics_empty(self):
        """Test detection metrics with empty predictions."""
        from cv_pipeline.utils.metrics import compute_detection_metrics

        predictions = [{"boxes": [], "scores": [], "classes": []}]
        ground_truth = [{"boxes": [[0, 0, 100, 100]], "classes": [0]}]

        metrics = compute_detection_metrics(predictions, ground_truth)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_compute_detection_metrics_multiclass(self):
        """Test detection metrics with multiple classes."""
        from cv_pipeline.utils.metrics import compute_detection_metrics

        predictions = [
            {
                "boxes": [[0, 0, 100, 100], [200, 200, 300, 300]],
                "scores": [0.9, 0.8],
                "classes": [0, 1],
            },
        ]
        ground_truth = [
            {"boxes": [[0, 0, 100, 100], [200, 200, 300, 300]], "classes": [0, 1]},
        ]

        metrics = compute_detection_metrics(predictions, ground_truth, classes=[0, 1])

        assert "mAP_50" in metrics
        assert "mAP_50_95" in metrics


class TestTrackingMetrics:
    """Tests for tracking metrics computation."""

    def test_match_tracks_perfect(self):
        """Test track matching with perfect predictions."""
        from cv_pipeline.utils.metrics import match_tracks

        pred_boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        pred_ids = np.array([1, 2])
        gt_boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        gt_ids = np.array([1, 2])

        matches, unmatched_preds, unmatched_gts = match_tracks(
            pred_boxes, pred_ids, gt_boxes, gt_ids
        )

        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_gts) == 0

    def test_match_tracks_no_predictions(self):
        """Test track matching with no predictions."""
        from cv_pipeline.utils.metrics import match_tracks

        pred_boxes = np.array([]).reshape(0, 4)
        pred_ids = np.array([])
        gt_boxes = np.array([[0, 0, 100, 100]])
        gt_ids = np.array([1])

        matches, unmatched_preds, unmatched_gts = match_tracks(
            pred_boxes, pred_ids, gt_boxes, gt_ids
        )

        assert len(matches) == 0
        assert len(unmatched_gts) == 1

    def test_compute_tracking_metrics(self):
        """Test full tracking metrics computation."""
        from cv_pipeline.utils.metrics import TrackingFrame, compute_tracking_metrics

        predictions = [
            TrackingFrame(
                frame_id=0,
                track_ids=np.array([1, 2]),
                boxes=np.array([[0, 0, 100, 100], [200, 200, 300, 300]]),
            ),
            TrackingFrame(
                frame_id=1,
                track_ids=np.array([1, 2]),
                boxes=np.array([[10, 10, 110, 110], [210, 210, 310, 310]]),
            ),
        ]
        ground_truth = [
            TrackingFrame(
                frame_id=0,
                track_ids=np.array([1, 2]),
                boxes=np.array([[0, 0, 100, 100], [200, 200, 300, 300]]),
            ),
            TrackingFrame(
                frame_id=1,
                track_ids=np.array([1, 2]),
                boxes=np.array([[10, 10, 110, 110], [210, 210, 310, 310]]),
            ),
        ]

        metrics = compute_tracking_metrics(predictions, ground_truth)

        assert "MOTA" in metrics
        assert "MOTP" in metrics
        assert "IDF1" in metrics
        assert metrics["MOTA"] == 1.0
        assert metrics["id_switches"] == 0

    def test_compute_tracking_metrics_with_id_switch(self):
        """Test tracking metrics with ID switch."""
        from cv_pipeline.utils.metrics import TrackingFrame, compute_tracking_metrics

        predictions = [
            TrackingFrame(frame_id=0, track_ids=np.array([1]), boxes=np.array([[0, 0, 100, 100]])),
            TrackingFrame(
                frame_id=1,
                track_ids=np.array([2]),  # ID switch
                boxes=np.array([[10, 10, 110, 110]]),
            ),
        ]
        ground_truth = [
            TrackingFrame(frame_id=0, track_ids=np.array([1]), boxes=np.array([[0, 0, 100, 100]])),
            TrackingFrame(
                frame_id=1, track_ids=np.array([1]), boxes=np.array([[10, 10, 110, 110]])
            ),
        ]

        metrics = compute_tracking_metrics(predictions, ground_truth)

        assert metrics["id_switches"] >= 1


class TestLaneMetrics:
    """Tests for lane detection metrics."""

    def test_compute_lane_accuracy_perfect(self):
        """Test lane accuracy with perfect predictions."""
        from cv_pipeline.utils.metrics import compute_lane_accuracy

        pred_lanes = [np.array([[100, 400], [100, 300], [100, 200], [100, 100]])]
        gt_lanes = [np.array([[100, 400], [100, 300], [100, 200], [100, 100]])]

        acc, prec, rec, f1 = compute_lane_accuracy(pred_lanes, gt_lanes, img_height=480)

        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0

    def test_compute_lane_accuracy_empty(self):
        """Test lane accuracy with empty predictions."""
        from cv_pipeline.utils.metrics import compute_lane_accuracy

        pred_lanes = []
        gt_lanes = [np.array([[100, 400], [100, 200]])]

        acc, prec, rec, f1 = compute_lane_accuracy(pred_lanes, gt_lanes, img_height=480)

        assert acc == 0.0
        assert rec == 0.0

    def test_compute_lane_metrics(self):
        """Test full lane metrics computation."""
        from cv_pipeline.utils.metrics import compute_lane_metrics

        predictions = [
            {"lanes": [[[100, 400], [100, 300], [100, 200]]]},
            {"lanes": [[[200, 400], [200, 300], [200, 200]]]},
        ]
        ground_truth = [
            {"lanes": [[[100, 400], [100, 300], [100, 200]]]},
            {"lanes": [[[200, 400], [200, 300], [200, 200]]]},
        ]

        metrics = compute_lane_metrics(predictions, ground_truth)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator class."""

    def test_init(self):
        """Test accumulator initialization."""
        from cv_pipeline.utils.metrics import MetricsAccumulator

        acc = MetricsAccumulator()

        assert len(acc.predictions) == 0
        assert len(acc.ground_truth) == 0

    def test_reset(self):
        """Test reset functionality."""
        from cv_pipeline.utils.metrics import MetricsAccumulator

        acc = MetricsAccumulator()
        acc.update_detection(
            {"boxes": [[0, 0, 100, 100]], "scores": [0.9], "classes": [0]},
            {"boxes": [[0, 0, 100, 100]], "classes": [0]},
        )

        acc.reset()

        assert len(acc.predictions) == 0

    def test_update_detection(self):
        """Test detection update."""
        from cv_pipeline.utils.metrics import MetricsAccumulator

        acc = MetricsAccumulator()

        for _ in range(5):
            acc.update_detection(
                {"boxes": [[0, 0, 100, 100]], "scores": [0.9], "classes": [0]},
                {"boxes": [[0, 0, 100, 100]], "classes": [0]},
            )

        assert len(acc.predictions) == 5
        assert len(acc.ground_truth) == 5

    def test_compute(self):
        """Test metric computation."""
        from cv_pipeline.utils.metrics import MetricsAccumulator

        acc = MetricsAccumulator()

        acc.update_detection(
            {"boxes": [[0, 0, 100, 100]], "scores": [0.9], "classes": [0]},
            {"boxes": [[0, 0, 100, 100]], "classes": [0]},
        )

        results = acc.compute()

        assert "detection" in results
        assert "mAP_50" in results["detection"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
