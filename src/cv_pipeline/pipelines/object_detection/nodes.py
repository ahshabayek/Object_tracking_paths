"""Object Detection Pipeline Nodes.

This module contains node functions for object detection using various models:
- RF-DETR (Roboflow Detection Transformer)
- RT-DETR (Real-Time Detection Transformer)
- YOLOv11/YOLOv12 (Ultralytics)
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time

import torch
import torch.nn as nn
import numpy as np
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class representing a single detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class DetectionResult:
    """Container for detection results on a single frame."""
    frame_id: int
    detections: List[Detection]
    inference_time: float
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    def get_boxes(self) -> np.ndarray:
        if not self.detections:
            return np.empty((0, 4))
        return np.array([d.bbox for d in self.detections])
    
    def get_scores(self) -> np.ndarray:
        if not self.detections:
            return np.empty(0)
        return np.array([d.confidence for d in self.detections])
    
    def get_classes(self) -> np.ndarray:
        if not self.detections:
            return np.empty(0, dtype=np.int32)
        return np.array([d.class_id for d in self.detections], dtype=np.int32)


class DetectorFactory:
    """Factory class for creating detection models."""
    
    SUPPORTED_MODELS = ["rf_detr", "rt_detr", "yolov12", "yolov11", "yolov10"]
    
    @staticmethod
    def create(model_name: str, params: Dict[str, Any]) -> nn.Module:
        """Create a detection model based on the specified name.
        
        Args:
            model_name: Name of the model to create
            params: Model configuration parameters
        
        Returns:
            Initialized detection model
        """
        if model_name == "rf_detr":
            return DetectorFactory._create_rf_detr(params)
        elif model_name == "rt_detr":
            return DetectorFactory._create_rt_detr(params)
        elif model_name in ["yolov12", "yolov11", "yolov10"]:
            return DetectorFactory._create_yolo(model_name, params)
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported: {DetectorFactory.SUPPORTED_MODELS}")
    
    @staticmethod
    def _create_rf_detr(params: Dict[str, Any]) -> nn.Module:
        """Create RF-DETR model."""
        try:
            from rfdetr import RFDETR
            
            variant = params.get("variant", "l")
            weights = params.get("weights", {}).get("rf_detr", f"rfdetr-{variant}.pt")
            
            model = RFDETR(weights=weights)
            logger.info(f"Loaded RF-DETR model: {weights}")
            return model
        except ImportError:
            logger.warning("RF-DETR not installed, falling back to RT-DETR")
            return DetectorFactory._create_rt_detr(params)
    
    @staticmethod
    def _create_rt_detr(params: Dict[str, Any]) -> nn.Module:
        """Create RT-DETR model."""
        from ultralytics import RTDETR
        
        variant = params.get("variant", "l")
        weights = params.get("weights", {}).get("rt_detr", f"rtdetr-{variant}.pt")
        
        model = RTDETR(weights)
        logger.info(f"Loaded RT-DETR model: {weights}")
        return model
    
    @staticmethod
    def _create_yolo(model_name: str, params: Dict[str, Any]) -> nn.Module:
        """Create YOLO model (v10, v11, or v12)."""
        from ultralytics import YOLO
        
        variant = params.get("variant", "l")
        
        # Map model names to weight files
        weight_map = {
            "yolov12": f"yolov12{variant}.pt",
            "yolov11": f"yolo11{variant}.pt",
            "yolov10": f"yolov10{variant}.pt",
        }
        
        weights = params.get("weights", {}).get(model_name, weight_map[model_name])
        
        model = YOLO(weights)
        logger.info(f"Loaded {model_name.upper()} model: {weights}")
        return model


def load_detection_model(params: Dict[str, Any]) -> nn.Module:
    """Load the specified detection model.
    
    Args:
        params: Detection parameters including model type and configuration
    
    Returns:
        Loaded detection model
    """
    model_name = params.get("model", "rt_detr")
    device = params.get("device", "cuda:0")
    half_precision = params.get("half_precision", True)
    
    logger.info(f"Loading detection model: {model_name}")
    
    # Create model
    model = DetectorFactory.create(model_name, params)
    
    # Move to device
    if hasattr(model, "to"):
        model = model.to(device)
    
    # Enable half precision if supported
    if half_precision and torch.cuda.is_available():
        if hasattr(model, "half"):
            model = model.half()
    
    logger.info(f"Detection model loaded on {device}")
    return model


def run_detection_inference(
    model: nn.Module,
    batches: List[torch.Tensor],
    params: Dict[str, Any],
) -> List[Any]:
    """Run detection inference on batched inputs.
    
    Args:
        model: Detection model
        batches: List of batched input tensors [B, C, H, W]
        params: Detection parameters
    
    Returns:
        List of raw model outputs
    """
    device = params.get("device", "cuda:0")
    conf_thresh = params.get("confidence_threshold", 0.25)
    iou_thresh = params.get("iou_threshold", 0.45)
    classes = params.get("classes", None)
    
    results = []
    total_time = 0
    
    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)
        
        if params.get("half_precision", True) and torch.cuda.is_available():
            batch = batch.half()
        
        start_time = time.time()
        
        # Different inference methods for different models
        if hasattr(model, "predict"):
            # Ultralytics models (YOLO, RT-DETR)
            output = model.predict(
                batch,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=classes,
                verbose=False,
            )
        else:
            # Custom models
            with torch.no_grad():
                output = model(batch)
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        results.append({
            "batch_idx": batch_idx,
            "output": output,
            "inference_time": inference_time,
        })
    
    avg_time = total_time / len(batches) if batches else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    logger.info(f"Detection inference complete: {len(batches)} batches, {fps:.1f} FPS")
    
    return results


def post_process_detections(
    raw_results: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[DetectionResult]:
    """Post-process raw detection outputs.
    
    Args:
        raw_results: Raw model outputs
        params: Detection parameters
    
    Returns:
        List of processed DetectionResult objects
    """
    conf_thresh = params.get("confidence_threshold", 0.25)
    
    # COCO class names
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    processed_results = []
    
    for result in raw_results:
        output = result["output"]
        inference_time = result["inference_time"]
        batch_idx = result["batch_idx"]
        
        # Handle Ultralytics results
        if hasattr(output, "__iter__") and hasattr(output[0], "boxes"):
            for frame_idx, frame_result in enumerate(output):
                boxes = frame_result.boxes
                
                detections = []
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().item()
                    cls_id = int(boxes.cls[i].cpu().item())
                    
                    if conf >= conf_thresh:
                        det = Detection(
                            bbox=box,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=coco_names[cls_id] if cls_id < len(coco_names) else f"class_{cls_id}",
                        )
                        detections.append(det)
                
                processed_results.append(DetectionResult(
                    frame_id=batch_idx * len(output) + frame_idx,
                    detections=detections,
                    inference_time=inference_time / len(output),
                ))
        else:
            # Handle raw tensor outputs
            processed_results.append(DetectionResult(
                frame_id=batch_idx,
                detections=[],
                inference_time=inference_time,
            ))
    
    total_dets = sum(r.num_detections for r in processed_results)
    logger.info(f"Post-processed {len(processed_results)} frames, {total_dets} total detections")
    
    return processed_results


def filter_detections_by_class(
    detection_results: List[DetectionResult],
    target_classes: Optional[Dict[str, List[int]]],
) -> List[DetectionResult]:
    """Filter detections to keep only specified classes.
    
    Args:
        detection_results: List of detection results
        target_classes: Dictionary mapping category names to class IDs
            e.g., {"vehicles": [2, 3, 5, 7], "pedestrians": [0]}
    
    Returns:
        Filtered detection results
    """
    if target_classes is None:
        return detection_results
    
    # Flatten all target class IDs
    all_target_ids = set()
    for class_ids in target_classes.values():
        all_target_ids.update(class_ids)
    
    filtered_results = []
    
    for result in detection_results:
        filtered_detections = [
            det for det in result.detections
            if det.class_id in all_target_ids
        ]
        
        filtered_results.append(DetectionResult(
            frame_id=result.frame_id,
            detections=filtered_detections,
            inference_time=result.inference_time,
        ))
    
    kept_dets = sum(r.num_detections for r in filtered_results)
    orig_dets = sum(r.num_detections for r in detection_results)
    logger.info(f"Filtered detections: {kept_dets}/{orig_dets} kept")
    
    return filtered_results


def compute_detection_metrics(
    detection_results: List[DetectionResult],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute detection metrics.
    
    Args:
        detection_results: List of detection results
        params: Detection parameters
    
    Returns:
        Dictionary of computed metrics
    """
    if not detection_results:
        return {}
    
    total_detections = sum(r.num_detections for r in detection_results)
    total_frames = len(detection_results)
    
    # Compute timing metrics
    inference_times = [r.inference_time for r in detection_results]
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Compute confidence statistics
    all_confidences = []
    for result in detection_results:
        all_confidences.extend([d.confidence for d in result.detections])
    
    metrics = {
        "total_detections": total_detections,
        "total_frames": total_frames,
        "detections_per_frame": total_detections / total_frames if total_frames > 0 else 0,
        "avg_inference_time_ms": avg_inference_time * 1000,
        "fps": fps,
        "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
        "min_confidence": np.min(all_confidences) if all_confidences else 0,
        "max_confidence": np.max(all_confidences) if all_confidences else 0,
    }
    
    # Class distribution
    class_counts = {}
    for result in detection_results:
        for det in result.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        metrics[f"count_{class_name}"] = count
    
    logger.info(f"Detection metrics: {total_detections} detections, {fps:.1f} FPS")
    
    return metrics


def log_detection_to_mlflow(
    metrics: Dict[str, float],
    params: Dict[str, Any],
) -> None:
    """Log detection metrics to MLFlow.
    
    Args:
        metrics: Computed detection metrics
        params: Detection parameters
    """
    try:
        # Log parameters
        mlflow.log_param("detection_model", params.get("model", "unknown"))
        mlflow.log_param("detection_variant", params.get("variant", "unknown"))
        mlflow.log_param("confidence_threshold", params.get("confidence_threshold", 0.25))
        mlflow.log_param("iou_threshold", params.get("iou_threshold", 0.45))
        mlflow.log_param("device", params.get("device", "unknown"))
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"detection_{key}", value)
        
        logger.info("Detection metrics logged to MLFlow")
        
    except Exception as e:
        logger.warning(f"Failed to log to MLFlow: {e}")
