"""Lane Detection Pipeline Nodes.

This module contains node functions for lane detection using:
- CLRerNet (with LaneIoU)
- CLRNet (Cross-Layer Refinement)
- LaneATT (Attention-based)
- UFLD (Ultra-Fast)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import cv2
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class LanePoint:
    """A single point on a lane."""
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class Lane:
    """Represents a detected lane."""
    lane_id: int
    points: List[LanePoint]
    confidence: float
    lane_type: str = "unknown"  # ego_left, ego_right, adjacent_left, adjacent_right
    
    def to_numpy(self) -> np.ndarray:
        """Convert lane points to numpy array."""
        return np.array([[p.x, p.y] for p in self.points])
    
    def get_coefficients(self, degree: int = 3) -> Optional[np.ndarray]:
        """Fit polynomial coefficients to lane points."""
        points = self.to_numpy()
        if len(points) < degree + 1:
            return None
        
        try:
            # Fit polynomial y = f(x)
            coeffs = np.polyfit(points[:, 1], points[:, 0], degree)
            return coeffs
        except np.linalg.LinAlgError:
            return None


@dataclass
class LaneDetectionResult:
    """Container for lane detection results on a single frame."""
    frame_id: int
    lanes: List[Lane]
    inference_time: float
    
    @property
    def num_lanes(self) -> int:
        return len(self.lanes)


class BaseLaneDetector:
    """Base class for lane detectors."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.device = params.get("device", "cuda:0")
        self.img_h = params.get("img_h", 320)
        self.img_w = params.get("img_w", 800)
        self.model = None
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize
        img = cv2.resize(image, (self.img_w, self.img_h))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        return img.to(self.device)
    
    def postprocess(self, output: Any, ori_shape: Tuple[int, int]) -> List[Lane]:
        """Convert model output to Lane objects."""
        raise NotImplementedError


class CLRNetDetector(BaseLaneDetector):
    """CLRNet lane detector implementation."""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.num_priors = params.get("clrnet", {}).get("num_priors", 192)
        self.num_points = params.get("num_points", 72)
        self.conf_threshold = params.get("conf_threshold", 0.4)
        
    def load_model(self, weights_path: Optional[str] = None):
        """Load CLRNet model."""
        try:
            # Attempt to load from mmdet/mmseg if available
            from mmdet.apis import init_detector
            
            config_path = self.params.get("config_path", "configs/clrnet/clrnet_culane.py")
            checkpoint = weights_path or self.params.get("weights_path")
            
            self.model = init_detector(config_path, checkpoint, device=self.device)
            logger.info("Loaded CLRNet model from mmdet")
            
        except ImportError:
            logger.warning("mmdet not available, using placeholder model")
            self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create a placeholder model for demonstration."""
        
        class PlaceholderLaneNet(nn.Module):
            def __init__(self, num_lanes=4, num_points=72):
                super().__init__()
                self.num_lanes = num_lanes
                self.num_points = num_points
                
                # Simple encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                
                # Lane heads
                self.lane_cls = nn.Linear(256, num_lanes)
                self.lane_reg = nn.Linear(256, num_lanes * num_points * 2)
            
            def forward(self, x):
                features = self.encoder(x)
                features = self.pool(features).flatten(1)
                
                cls_scores = torch.sigmoid(self.lane_cls(features))
                lane_points = self.lane_reg(features)
                lane_points = lane_points.view(-1, self.num_lanes, self.num_points, 2)
                
                return cls_scores, lane_points
        
        model = PlaceholderLaneNet(num_lanes=4, num_points=self.num_points)
        return model.to(self.device)
    
    def postprocess(self, output: Tuple, ori_shape: Tuple[int, int]) -> List[Lane]:
        """Convert CLRNet output to Lane objects."""
        cls_scores, lane_points = output
        
        cls_scores = cls_scores.cpu().numpy()[0]
        lane_points = lane_points.cpu().numpy()[0]
        
        ori_h, ori_w = ori_shape
        
        lanes = []
        for i, (score, points) in enumerate(zip(cls_scores, lane_points)):
            if score < self.conf_threshold:
                continue
            
            # Scale points to original image size
            points[:, 0] = points[:, 0] * ori_w
            points[:, 1] = points[:, 1] * ori_h
            
            # Filter valid points
            valid_mask = (points[:, 0] >= 0) & (points[:, 0] < ori_w) & \
                        (points[:, 1] >= 0) & (points[:, 1] < ori_h)
            valid_points = points[valid_mask]
            
            if len(valid_points) < 2:
                continue
            
            lane_points_list = [
                LanePoint(x=float(p[0]), y=float(p[1]), confidence=float(score))
                for p in valid_points
            ]
            
            # Determine lane type based on x position
            center_x = np.mean(valid_points[:, 0])
            if center_x < ori_w * 0.35:
                lane_type = "adjacent_left"
            elif center_x < ori_w * 0.5:
                lane_type = "ego_left"
            elif center_x < ori_w * 0.65:
                lane_type = "ego_right"
            else:
                lane_type = "adjacent_right"
            
            lanes.append(Lane(
                lane_id=i,
                points=lane_points_list,
                confidence=float(score),
                lane_type=lane_type,
            ))
        
        return lanes


class LaneATTDetector(BaseLaneDetector):
    """LaneATT lane detector implementation."""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.topk_anchors = params.get("laneatt", {}).get("topk_anchors", 1000)
        self.conf_threshold = params.get("conf_threshold", 0.5)
        self.nms_threshold = params.get("nms_threshold", 15.0)
    
    def load_model(self, weights_path: Optional[str] = None):
        """Load LaneATT model."""
        logger.info("Loading LaneATT model (placeholder)")
        self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create placeholder LaneATT model."""
        return CLRNetDetector._create_placeholder_model(self)


class UFLDDetector(BaseLaneDetector):
    """Ultra-Fast Lane Detection (UFLD) implementation."""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.griding_num = params.get("ufld", {}).get("griding_num", 200)
        self.cls_num_per_lane = params.get("ufld", {}).get("cls_num_per_lane", 56)
    
    def load_model(self, weights_path: Optional[str] = None):
        """Load UFLD model."""
        logger.info("Loading UFLD model (placeholder)")
        self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create placeholder UFLD model."""
        return CLRNetDetector._create_placeholder_model(self)


def load_lane_model(params: Dict[str, Any]) -> BaseLaneDetector:
    """Load the specified lane detection model.
    
    Args:
        params: Lane detection parameters
    
    Returns:
        Loaded lane detector
    """
    model_name = params.get("model", "clrnet")
    
    logger.info(f"Loading lane detection model: {model_name}")
    
    if model_name in ["clrernet", "clrnet"]:
        detector = CLRNetDetector(params)
    elif model_name == "laneatt":
        detector = LaneATTDetector(params)
    elif model_name == "ufld":
        detector = UFLDDetector(params)
    else:
        raise ValueError(f"Unknown lane model: {model_name}")
    
    detector.load_model()
    
    return detector


def run_lane_detection(
    detector: BaseLaneDetector,
    frames: List[np.ndarray],
    params: Dict[str, Any],
) -> List[LaneDetectionResult]:
    """Run lane detection on frames.
    
    Args:
        detector: Lane detection model
        frames: List of preprocessed frames
        params: Lane detection parameters
    
    Returns:
        List of lane detection results
    """
    results = []
    
    for frame_idx, frame in enumerate(frames):
        start_time = time.time()
        
        # Denormalize frame if needed (assuming normalized input)
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)
        
        ori_shape = (frame_uint8.shape[0], frame_uint8.shape[1])
        
        # Preprocess
        input_tensor = detector.preprocess(frame_uint8)
        
        # Run inference
        with torch.no_grad():
            output = detector.model(input_tensor)
        
        # Postprocess
        lanes = detector.postprocess(output, ori_shape)
        
        inference_time = time.time() - start_time
        
        results.append(LaneDetectionResult(
            frame_id=frame_idx,
            lanes=lanes,
            inference_time=inference_time,
        ))
    
    total_lanes = sum(r.num_lanes for r in results)
    logger.info(f"Lane detection complete: {len(results)} frames, {total_lanes} total lanes")
    
    return results


def fit_lane_curves(
    lane_results: List[LaneDetectionResult],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fit curves to detected lanes.
    
    Args:
        lane_results: Raw lane detection results
        params: Lane detection parameters
    
    Returns:
        List of fitted lane curves
    """
    fitting_method = params.get("path_construction", {}).get("fitting_method", "polynomial")
    
    fitted_results = []
    
    for result in lane_results:
        frame_lanes = {
            "frame_id": result.frame_id,
            "lanes": [],
            "inference_time": result.inference_time,
        }
        
        for lane in result.lanes:
            points = lane.to_numpy()
            
            if len(points) < 4:
                continue
            
            lane_data = {
                "lane_id": lane.lane_id,
                "lane_type": lane.lane_type,
                "confidence": lane.confidence,
                "points": points.tolist(),
            }
            
            # Fit polynomial
            coeffs = lane.get_coefficients(degree=3)
            if coeffs is not None:
                lane_data["polynomial_coeffs"] = coeffs.tolist()
            
            # Fit Bezier curve if requested
            if fitting_method == "bezier":
                bezier_points = _fit_bezier_curve(points, order=3)
                if bezier_points is not None:
                    lane_data["bezier_control_points"] = bezier_points.tolist()
            
            frame_lanes["lanes"].append(lane_data)
        
        fitted_results.append(frame_lanes)
    
    logger.info(f"Fitted curves for {len(fitted_results)} frames")
    
    return fitted_results


def _fit_bezier_curve(
    points: np.ndarray,
    order: int = 3,
) -> Optional[np.ndarray]:
    """Fit a Bezier curve to points.
    
    Args:
        points: Lane points [N, 2]
        order: Bezier curve order
    
    Returns:
        Control points for Bezier curve
    """
    if len(points) < order + 1:
        return None
    
    try:
        # Simple Bezier fitting using uniform parameter
        n_control = order + 1
        indices = np.linspace(0, len(points) - 1, n_control).astype(int)
        control_points = points[indices]
        
        return control_points
        
    except Exception as e:
        logger.warning(f"Bezier fitting failed: {e}")
        return None


def compute_lane_metrics(
    lane_results: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Compute lane detection metrics.
    
    Args:
        lane_results: Fitted lane results
        params: Lane detection parameters
    
    Returns:
        Dictionary of computed metrics
    """
    if not lane_results:
        return {}
    
    total_frames = len(lane_results)
    total_lanes = sum(len(r["lanes"]) for r in lane_results)
    
    # Processing times
    inference_times = [r["inference_time"] for r in lane_results]
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    # Lane statistics
    lanes_per_frame = [len(r["lanes"]) for r in lane_results]
    
    # Confidence statistics
    all_confidences = []
    for result in lane_results:
        for lane in result["lanes"]:
            all_confidences.append(lane["confidence"])
    
    # Lane type distribution
    lane_types = {}
    for result in lane_results:
        for lane in result["lanes"]:
            lane_type = lane["lane_type"]
            lane_types[lane_type] = lane_types.get(lane_type, 0) + 1
    
    metrics = {
        "total_frames": total_frames,
        "total_lanes": total_lanes,
        "avg_lanes_per_frame": np.mean(lanes_per_frame) if lanes_per_frame else 0,
        "avg_inference_time_ms": avg_time * 1000,
        "fps": fps,
        "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
    }
    
    # Add lane type counts
    for lane_type, count in lane_types.items():
        metrics[f"count_{lane_type}"] = count
    
    logger.info(f"Lane metrics: {total_lanes} lanes, {fps:.1f} FPS")
    
    return metrics


def log_lane_to_mlflow(
    metrics: Dict[str, float],
    params: Dict[str, Any],
) -> None:
    """Log lane detection metrics to MLFlow.
    
    Args:
        metrics: Computed lane metrics
        params: Lane detection parameters
    """
    try:
        mlflow.log_param("lane_model", params.get("model", "unknown"))
        mlflow.log_param("lane_backbone", params.get("backbone", "unknown"))
        mlflow.log_param("conf_threshold", params.get("conf_threshold", 0.4))
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"lane_{key}", value)
        
        logger.info("Lane metrics logged to MLFlow")
        
    except Exception as e:
        logger.warning(f"Failed to log to MLFlow: {e}")
