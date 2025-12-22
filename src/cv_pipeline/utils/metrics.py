"""Metrics for Object Detection, Tracking, and Lane Detection.

This module provides evaluation metrics for:
- Object Detection: mAP (mean Average Precision), Precision, Recall, F1
- Multi-Object Tracking: MOTA, MOTP, IDF1, ID switches, fragmentation
- Lane Detection: Accuracy, F1, precision, recall

Based on:
- COCO evaluation for detection
- MOT Challenge metrics for tracking
- CULane/TuSimple metrics for lane detection

Usage:
    from cv_pipeline.utils.metrics import (
        compute_detection_metrics,
        compute_tracking_metrics,
        compute_lane_metrics,
    )

    # Detection metrics
    det_metrics = compute_detection_metrics(predictions, ground_truth)
    print(f"mAP@0.5: {det_metrics['mAP_50']:.3f}")

    # Tracking metrics
    track_metrics = compute_tracking_metrics(predictions, ground_truth)
    print(f"MOTA: {track_metrics['MOTA']:.3f}, IDF1: {track_metrics['IDF1']:.3f}")
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# =============================================================================
# Detection Metrics
# =============================================================================


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        Intersection over Union value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes.

    Args:
        boxes1: First set of boxes [N, 4].
        boxes2: Second set of boxes [M, 4].

    Returns:
        IoU matrix [N, M].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Expand dimensions for broadcasting
    boxes1 = np.expand_dims(boxes1, 1)  # [N, 1, 4]
    boxes2 = np.expand_dims(boxes2, 0)  # [1, M, 4]

    # Intersection coordinates
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
    use_07_metric: bool = False,
) -> float:
    """Compute Average Precision from precision-recall curve.

    Args:
        precisions: Precision values at each threshold.
        recalls: Recall values at each threshold.
        use_07_metric: Use VOC 2007 11-point interpolation.

    Returns:
        Average Precision value.
    """
    if use_07_metric:
        # 11-point interpolation (VOC 2007)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask])
        return ap / 11.0
    else:
        # All-point interpolation (VOC 2010+)
        # Make precision monotonically decreasing
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Find points where recall changes
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

        # Sum areas under curve
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

        return ap


def compute_precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute precision-recall curve for a single class.

    Args:
        pred_boxes: Predicted boxes [N, 4].
        pred_scores: Prediction confidence scores [N].
        gt_boxes: Ground truth boxes [M, 4].
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (precisions, recalls, AP).
    """
    if len(pred_boxes) == 0:
        return np.array([0]), np.array([0]), 0.0

    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes)), 0.0

    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # Track matched ground truths
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        # Find best matching ground truth
        ious = iou_matrix[i]
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = True
        else:
            fp[i] = 1

    # Compute cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Compute AP
    ap = compute_ap(precisions, recalls)

    return precisions, recalls, ap


def compute_detection_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresholds: List[float] = [0.5, 0.75],
    classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute detection metrics (mAP, precision, recall, F1).

    Args:
        predictions: List of predictions per image, each with:
            - 'boxes': [N, 4] array of boxes
            - 'scores': [N] array of confidence scores
            - 'classes': [N] array of class IDs
        ground_truth: List of ground truth per image, each with:
            - 'boxes': [M, 4] array of boxes
            - 'classes': [M] array of class IDs
        iou_thresholds: IoU thresholds for AP calculation.
        classes: List of class IDs to evaluate. None for all classes.

    Returns:
        Dictionary of metrics.
    """
    # Collect all unique classes
    if classes is None:
        all_classes = set()
        for gt in ground_truth:
            if "classes" in gt and len(gt["classes"]) > 0:
                all_classes.update(gt["classes"])
        classes = sorted(all_classes) if all_classes else [0]

    metrics = {}

    # Per-class APs for each IoU threshold
    for iou_thresh in iou_thresholds:
        class_aps = []

        for cls_id in classes:
            # Collect predictions and ground truth for this class
            all_pred_boxes = []
            all_pred_scores = []
            all_gt_boxes = []

            for pred, gt in zip(predictions, ground_truth):
                pred_boxes = np.array(pred.get("boxes", []))
                pred_scores = np.array(pred.get("scores", []))
                pred_classes = np.array(pred.get("classes", []))
                gt_boxes = np.array(gt.get("boxes", []))
                gt_classes = np.array(gt.get("classes", []))

                if len(pred_boxes) > 0:
                    mask = pred_classes == cls_id
                    all_pred_boxes.append(pred_boxes[mask])
                    all_pred_scores.append(pred_scores[mask])

                if len(gt_boxes) > 0:
                    mask = gt_classes == cls_id
                    all_gt_boxes.append(gt_boxes[mask])

            # Concatenate all
            if all_pred_boxes:
                all_pred_boxes = np.concatenate(all_pred_boxes)
                all_pred_scores = np.concatenate(all_pred_scores)
            else:
                all_pred_boxes = np.array([])
                all_pred_scores = np.array([])

            if all_gt_boxes:
                all_gt_boxes = np.concatenate(all_gt_boxes)
            else:
                all_gt_boxes = np.array([])

            # Compute AP for this class
            _, _, ap = compute_precision_recall(
                all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh
            )
            class_aps.append(ap)

        # Mean AP across classes
        mAP = np.mean(class_aps) if class_aps else 0.0
        thresh_name = str(int(iou_thresh * 100))
        metrics[f"mAP_{thresh_name}"] = mAP

    # Compute mAP@0.5:0.95 (COCO-style)
    coco_thresholds = np.arange(0.5, 1.0, 0.05)
    coco_aps = []

    for iou_thresh in coco_thresholds:
        class_aps = []
        for cls_id in classes:
            all_pred_boxes = []
            all_pred_scores = []
            all_gt_boxes = []

            for pred, gt in zip(predictions, ground_truth):
                pred_boxes = np.array(pred.get("boxes", []))
                pred_scores = np.array(pred.get("scores", []))
                pred_classes = np.array(pred.get("classes", []))
                gt_boxes = np.array(gt.get("boxes", []))
                gt_classes = np.array(gt.get("classes", []))

                if len(pred_boxes) > 0:
                    mask = pred_classes == cls_id
                    all_pred_boxes.append(pred_boxes[mask])
                    all_pred_scores.append(pred_scores[mask])

                if len(gt_boxes) > 0:
                    mask = gt_classes == cls_id
                    all_gt_boxes.append(gt_boxes[mask])

            if all_pred_boxes:
                all_pred_boxes = np.concatenate(all_pred_boxes)
                all_pred_scores = np.concatenate(all_pred_scores)
            else:
                all_pred_boxes = np.array([])
                all_pred_scores = np.array([])

            if all_gt_boxes:
                all_gt_boxes = np.concatenate(all_gt_boxes)
            else:
                all_gt_boxes = np.array([])

            _, _, ap = compute_precision_recall(
                all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh
            )
            class_aps.append(ap)

        coco_aps.append(np.mean(class_aps) if class_aps else 0.0)

    metrics["mAP_50_95"] = np.mean(coco_aps)

    # Overall precision, recall, F1 at IoU=0.5
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truth):
        pred_boxes = np.array(pred.get("boxes", []))
        gt_boxes = np.array(gt.get("boxes", []))

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Greedy matching
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(pred_boxes)):
            ious = iou_matrix[i]
            max_idx = np.argmax(ious)
            if ious[max_idx] >= 0.5 and not gt_matched[max_idx]:
                total_tp += 1
                gt_matched[max_idx] = True
            else:
                total_fp += 1

        total_fn += np.sum(~gt_matched)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    return metrics


# =============================================================================
# Multi-Object Tracking Metrics
# =============================================================================


@dataclass
class TrackingFrame:
    """Container for tracking data in a single frame."""

    frame_id: int
    track_ids: np.ndarray
    boxes: np.ndarray
    classes: Optional[np.ndarray] = None


def match_tracks(
    pred_boxes: np.ndarray,
    pred_ids: np.ndarray,
    gt_boxes: np.ndarray,
    gt_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match predicted tracks to ground truth.

    Args:
        pred_boxes: Predicted boxes [N, 4].
        pred_ids: Predicted track IDs [N].
        gt_boxes: Ground truth boxes [M, 4].
        gt_ids: Ground truth track IDs [M].
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (matches, unmatched_preds, unmatched_gts).
        matches is list of (pred_idx, gt_idx) pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # Hungarian algorithm for optimal matching
    cost_matrix = 1 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_preds = list(range(len(pred_boxes)))
    unmatched_gts = list(range(len(gt_boxes)))

    for pred_idx, gt_idx in zip(row_indices, col_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
            matches.append((pred_idx, gt_idx))
            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(gt_idx)

    return matches, unmatched_preds, unmatched_gts


def compute_tracking_metrics(
    predictions: List[TrackingFrame],
    ground_truth: List[TrackingFrame],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute multi-object tracking metrics (MOTA, MOTP, IDF1, etc.).

    Based on MOT Challenge metrics:
    - MOTA: Multi-Object Tracking Accuracy
    - MOTP: Multi-Object Tracking Precision
    - IDF1: ID F1 Score
    - IDP: ID Precision
    - IDR: ID Recall
    - ID Switches: Number of ID switches
    - Fragmentation: Number of fragmentations

    Args:
        predictions: List of predicted TrackingFrame per frame.
        ground_truth: List of ground truth TrackingFrame per frame.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary of metrics.
    """
    # Create frame lookup
    pred_by_frame = {p.frame_id: p for p in predictions}
    gt_by_frame = {g.frame_id: g for g in ground_truth}

    all_frames = sorted(set(pred_by_frame.keys()) | set(gt_by_frame.keys()))

    # Counters
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    id_switches = 0
    fragmentations = 0

    # ID matching
    gt_id_to_pred_id: Dict[int, int] = {}  # Last matched pred ID for each GT ID
    pred_id_to_gt_id: Dict[int, int] = {}  # Last matched GT ID for each pred ID

    # For IDF1 calculation
    gt_id_matches: Dict[int, int] = defaultdict(int)  # GT ID -> match count
    pred_id_matches: Dict[int, int] = defaultdict(int)  # Pred ID -> match count
    gt_id_total: Dict[int, int] = defaultdict(int)  # GT ID -> total appearances
    pred_id_total: Dict[int, int] = defaultdict(int)  # Pred ID -> total appearances

    # Track continuity for fragmentation
    gt_active: Dict[int, bool] = {}  # GT ID -> was active in previous frame

    for frame_id in all_frames:
        pred_frame = pred_by_frame.get(frame_id)
        gt_frame = gt_by_frame.get(frame_id)

        if pred_frame is None:
            pred_boxes = np.array([])
            pred_ids = np.array([])
        else:
            pred_boxes = pred_frame.boxes
            pred_ids = pred_frame.track_ids

        if gt_frame is None:
            gt_boxes = np.array([])
            gt_ids = np.array([])
        else:
            gt_boxes = gt_frame.boxes
            gt_ids = gt_frame.track_ids

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # Update ID totals
        for gt_id in gt_ids:
            gt_id_total[gt_id] += 1
        for pred_id in pred_ids:
            pred_id_total[pred_id] += 1

        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_gts = match_tracks(
            pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold
        )

        # Count TP, FP, FN
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)

        # Track current frame GT IDs for fragmentation
        current_gt_ids = set(gt_ids)

        for pred_idx, gt_idx in matches:
            pred_id = pred_ids[pred_idx]
            gt_id = gt_ids[gt_idx]

            # Accumulate IoU for MOTP
            iou = compute_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
            total_iou += iou

            # Update ID matches for IDF1
            gt_id_matches[gt_id] += 1
            pred_id_matches[pred_id] += 1

            # Check for ID switch
            if gt_id in gt_id_to_pred_id:
                if gt_id_to_pred_id[gt_id] != pred_id:
                    id_switches += 1

            gt_id_to_pred_id[gt_id] = pred_id
            pred_id_to_gt_id[pred_id] = gt_id

        # Check for fragmentations
        for gt_id in gt_active:
            if gt_active[gt_id] and gt_id not in current_gt_ids:
                # GT was active but disappeared
                pass
            elif not gt_active[gt_id] and gt_id in current_gt_ids:
                # GT reappeared after being lost
                fragmentations += 1

        # Update active status
        for gt_id in current_gt_ids:
            gt_active[gt_id] = True
        for gt_id in list(gt_active.keys()):
            if gt_id not in current_gt_ids:
                gt_active[gt_id] = False

    # Compute MOTA
    # MOTA = 1 - (FN + FP + ID_switches) / total_GT
    mota = 1 - (total_fn + total_fp + id_switches) / (total_gt + 1e-6)

    # Compute MOTP
    # MOTP = sum(IoU) / total_TP
    motp = total_iou / (total_tp + 1e-6)

    # Compute IDF1
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    # IDTP = sum of correctly matched identities
    # Here we use a simplified version based on ID consistency

    idtp = sum(gt_id_matches.values())
    idfp = sum(pred_id_total.values()) - sum(pred_id_matches.values())
    idfn = sum(gt_id_total.values()) - sum(gt_id_matches.values())

    idf1 = 2 * idtp / (2 * idtp + idfp + idfn + 1e-6)
    idp = idtp / (idtp + idfp + 1e-6)
    idr = idtp / (idtp + idfn + 1e-6)

    # Detection metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    metrics = {
        "MOTA": mota,
        "MOTP": motp,
        "IDF1": idf1,
        "IDP": idp,
        "IDR": idr,
        "precision": precision,
        "recall": recall,
        "num_gt": total_gt,
        "num_pred": total_pred,
        "num_tp": total_tp,
        "num_fp": total_fp,
        "num_fn": total_fn,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "num_unique_gt_ids": len(gt_id_total),
        "num_unique_pred_ids": len(pred_id_total),
    }

    return metrics


# =============================================================================
# Lane Detection Metrics
# =============================================================================


def compute_lane_iou(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    img_height: int,
    lane_width: int = 30,
) -> float:
    """Compute IoU between predicted and ground truth lane.

    Uses mask-based IoU where lanes are rendered as thick lines.

    Args:
        pred_points: Predicted lane points [N, 2].
        gt_points: Ground truth lane points [M, 2].
        img_height: Image height for normalization.
        lane_width: Width of lane line for mask.

    Returns:
        IoU value.
    """
    if len(pred_points) < 2 or len(gt_points) < 2:
        return 0.0

    # Create masks
    import cv2

    mask_height = img_height
    mask_width = 1920  # Assume standard width

    pred_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    gt_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Draw lanes on masks
    pred_pts = pred_points.astype(np.int32).reshape((-1, 1, 2))
    gt_pts = gt_points.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(pred_mask, [pred_pts], False, 1, lane_width)
    cv2.polylines(gt_mask, [gt_pts], False, 1, lane_width)

    # Compute IoU
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)

    return intersection / (union + 1e-6)


def compute_lane_accuracy(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    img_height: int,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Compute lane detection accuracy metrics.

    Args:
        pred_lanes: List of predicted lane point arrays.
        gt_lanes: List of ground truth lane point arrays.
        img_height: Image height.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (accuracy, precision, recall, f1).
    """
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return 1.0, 1.0, 1.0, 1.0

    if len(pred_lanes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(gt_lanes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))

    for i, pred in enumerate(pred_lanes):
        for j, gt in enumerate(gt_lanes):
            iou_matrix[i, j] = compute_lane_iou(pred, gt, img_height)

    # Match using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(1 - iou_matrix)

    tp = 0
    for pred_idx, gt_idx in zip(row_indices, col_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
            tp += 1

    fp = len(pred_lanes) - tp
    fn = len(gt_lanes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = tp / (tp + fp + fn + 1e-6)

    return accuracy, precision, recall, f1


def compute_lane_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    img_height: int = 720,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute lane detection metrics across all frames.

    Args:
        predictions: List of predictions per frame, each with:
            - 'lanes': List of lane point arrays [N, 2]
        ground_truth: List of ground truth per frame, each with:
            - 'lanes': List of lane point arrays [M, 2]
        img_height: Image height for IoU calculation.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary of metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    frame_accuracies = []

    for pred, gt in zip(predictions, ground_truth):
        pred_lanes = pred.get("lanes", [])
        gt_lanes = gt.get("lanes", [])

        # Convert to numpy arrays if needed
        pred_lanes = [np.array(l) for l in pred_lanes if len(l) >= 2]
        gt_lanes = [np.array(l) for l in gt_lanes if len(l) >= 2]

        acc, prec, rec, f1 = compute_lane_accuracy(pred_lanes, gt_lanes, img_height, iou_threshold)
        frame_accuracies.append(acc)

        # Count TP, FP, FN for this frame
        if len(pred_lanes) == 0 and len(gt_lanes) == 0:
            continue

        if len(pred_lanes) == 0:
            total_fn += len(gt_lanes)
            continue

        if len(gt_lanes) == 0:
            total_fp += len(pred_lanes)
            continue

        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
        for i, pred_lane in enumerate(pred_lanes):
            for j, gt_lane in enumerate(gt_lanes):
                iou_matrix[i, j] = compute_lane_iou(pred_lane, gt_lane, img_height)

        # Match
        row_indices, col_indices = linear_sum_assignment(1 - iou_matrix)

        tp = 0
        for pred_idx, gt_idx in zip(row_indices, col_indices):
            if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                tp += 1

        total_tp += tp
        total_fp += len(pred_lanes) - tp
        total_fn += len(gt_lanes) - tp

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_frame_accuracy": np.mean(frame_accuracies) if frame_accuracies else 0.0,
        "num_tp": total_tp,
        "num_fp": total_fp,
        "num_fn": total_fn,
    }

    return metrics


# =============================================================================
# Utility Classes
# =============================================================================


class MetricsAccumulator:
    """Accumulator for computing metrics over multiple batches/frames.

    Example:
        accumulator = MetricsAccumulator()

        for batch in dataloader:
            predictions = model(batch)
            accumulator.update(predictions, batch.targets)

        metrics = accumulator.compute()
    """

    def __init__(self):
        """Initialize accumulator."""
        self.predictions = []
        self.ground_truth = []
        self.tracking_predictions = []
        self.tracking_ground_truth = []
        self.lane_predictions = []
        self.lane_ground_truth = []

    def reset(self) -> None:
        """Reset all accumulated data."""
        self.predictions.clear()
        self.ground_truth.clear()
        self.tracking_predictions.clear()
        self.tracking_ground_truth.clear()
        self.lane_predictions.clear()
        self.lane_ground_truth.clear()

    def update_detection(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> None:
        """Add detection predictions and ground truth."""
        self.predictions.append(predictions)
        self.ground_truth.append(ground_truth)

    def update_tracking(
        self,
        predictions: TrackingFrame,
        ground_truth: TrackingFrame,
    ) -> None:
        """Add tracking predictions and ground truth."""
        self.tracking_predictions.append(predictions)
        self.tracking_ground_truth.append(ground_truth)

    def update_lanes(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> None:
        """Add lane detection predictions and ground truth."""
        self.lane_predictions.append(predictions)
        self.lane_ground_truth.append(ground_truth)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute all accumulated metrics.

        Returns:
            Dictionary with 'detection', 'tracking', 'lane' keys.
        """
        results = {}

        if self.predictions and self.ground_truth:
            results["detection"] = compute_detection_metrics(self.predictions, self.ground_truth)

        if self.tracking_predictions and self.tracking_ground_truth:
            results["tracking"] = compute_tracking_metrics(
                self.tracking_predictions, self.tracking_ground_truth
            )

        if self.lane_predictions and self.lane_ground_truth:
            results["lane"] = compute_lane_metrics(self.lane_predictions, self.lane_ground_truth)

        return results


# Convenience exports
__all__ = [
    # Detection metrics
    "compute_iou",
    "compute_iou_matrix",
    "compute_ap",
    "compute_precision_recall",
    "compute_detection_metrics",
    # Tracking metrics
    "TrackingFrame",
    "match_tracks",
    "compute_tracking_metrics",
    # Lane metrics
    "compute_lane_iou",
    "compute_lane_accuracy",
    "compute_lane_metrics",
    # Utilities
    "MetricsAccumulator",
]
