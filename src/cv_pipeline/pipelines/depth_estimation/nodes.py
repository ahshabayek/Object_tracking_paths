"""Depth Estimation Pipeline Nodes.

This module provides monocular depth estimation to bridge 2D detection with 3D perception.
When only a single camera is available, depth estimation enables pseudo-3D understanding.

Supported Models:
    - ZoeDepth: Zero-shot transfer, best accuracy (Abs Rel: 0.075)
    - Metric3D: Metric depth with camera intrinsics
    - DepthAnything: Fast and robust depth estimation
    - MiDaS: Classic baseline, good generalization

Architecture Overview:

    Single Image → Depth Model → Dense Depth Map → Per-pixel distance
         ↓                              ↓
    [H, W, 3]                    [H, W] in meters

    Combined with 2D Detection:

    Image → RT-DETR → 2D Boxes ──────────────────────┐
      │                                               ↓
      └──→ ZoeDepth → Depth Map → Sample depth → 3D Position
                                  at box center    [x, y, z]

Why Multiple Models?
    - ZoeDepth: Best accuracy, use when quality matters
    - DepthAnything: Fastest, use for real-time
    - Metric3D: When you have camera intrinsics
    - MiDaS: Most robust to different domains
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Configuration for depth estimation.

    Attributes:
        model: Model name (zoedepth, metric3d, depth_anything, midas)
        variant: Model variant (e.g., 'nk' for ZoeDepth-NK)
        device: Device to run inference on
        max_depth: Maximum depth value in meters
        min_depth: Minimum depth value in meters
        output_size: Optional output size (H, W) for resizing
        use_camera_intrinsics: Whether to use camera intrinsics for metric depth
        intrinsics: Camera intrinsic matrix [fx, fy, cx, cy]
    """

    model: str = "zoedepth"
    variant: str = "nk"  # ZoeDepth variants: n, k, nk
    device: str = "cuda:0"
    max_depth: float = 80.0  # meters, typical for driving
    min_depth: float = 0.1  # meters
    output_size: Optional[Tuple[int, int]] = None
    use_camera_intrinsics: bool = False
    intrinsics: Optional[List[float]] = None  # [fx, fy, cx, cy]


@dataclass
class DepthResult:
    """Container for depth estimation results.

    Attributes:
        depth_map: Dense depth map [H, W] in meters
        confidence: Optional confidence map [H, W]
        inference_time: Time taken for inference
        min_depth: Minimum depth in the map
        max_depth: Maximum depth in the map
        median_depth: Median depth value
    """

    depth_map: np.ndarray  # [H, W] in meters
    confidence: Optional[np.ndarray] = None
    inference_time: float = 0.0
    min_depth: float = 0.0
    max_depth: float = 0.0
    median_depth: float = 0.0

    def get_depth_at(self, x: int, y: int) -> float:
        """Get depth value at a specific pixel location."""
        if 0 <= y < self.depth_map.shape[0] and 0 <= x < self.depth_map.shape[1]:
            return float(self.depth_map[y, x])
        return 0.0

    def get_depth_in_box(self, bbox: np.ndarray, method: str = "center") -> float:
        """Get depth value for a bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            method: How to compute depth - 'center', 'median', 'min', 'mean'

        Returns:
            Depth value in meters
        """
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Clamp to image bounds
        h, w = self.depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        if method == "center":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return float(self.depth_map[cy, cx])
        elif method == "median":
            roi = self.depth_map[y1:y2, x1:x2]
            return float(np.median(roi))
        elif method == "min":
            roi = self.depth_map[y1:y2, x1:x2]
            return float(np.min(roi))
        elif method == "mean":
            roi = self.depth_map[y1:y2, x1:x2]
            return float(np.mean(roi))
        else:
            raise ValueError(f"Unknown method: {method}")


@dataclass
class Detection3D:
    """3D detection result after lifting from 2D.

    Attributes:
        bbox_2d: Original 2D bounding box [x1, y1, x2, y2]
        position_3d: 3D position [x, y, z] in camera frame (meters)
        depth: Estimated depth (z-coordinate) in meters
        confidence: Detection confidence
        class_id: Class ID
        class_name: Class name
        size_3d: Estimated 3D size [width, height, length] in meters (optional)
    """

    bbox_2d: np.ndarray
    position_3d: np.ndarray  # [x, y, z] in meters
    depth: float
    confidence: float
    class_id: int
    class_name: str
    size_3d: Optional[np.ndarray] = None  # [w, h, l] in meters


class DepthEstimatorFactory:
    """Factory for creating depth estimation models.

    Supported Models:
        - zoedepth: Zero-shot depth, best accuracy
        - metric3d: Metric depth with intrinsics
        - depth_anything: Fast and robust
        - midas: Classic baseline
    """

    SUPPORTED_MODELS = ["zoedepth", "metric3d", "depth_anything", "midas"]

    @staticmethod
    def create(config: DepthConfig) -> nn.Module:
        """Create a depth estimation model.

        Args:
            config: Depth configuration

        Returns:
            Initialized depth model
        """
        model_name = config.model.lower()

        if model_name == "zoedepth":
            return DepthEstimatorFactory._create_zoedepth(config)
        elif model_name == "metric3d":
            return DepthEstimatorFactory._create_metric3d(config)
        elif model_name == "depth_anything":
            return DepthEstimatorFactory._create_depth_anything(config)
        elif model_name == "midas":
            return DepthEstimatorFactory._create_midas(config)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: {DepthEstimatorFactory.SUPPORTED_MODELS}"
            )

    @staticmethod
    def _create_zoedepth(config: DepthConfig) -> nn.Module:
        """Create ZoeDepth model.

        ZoeDepth (Zero-shot Transfer):
        - Combines relative depth (MiDaS) with metric depth heads
        - Variants: N (NYU indoor), K (KITTI outdoor), NK (both)
        - Best for autonomous driving: NK or K variant

        Paper: "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth"
        """
        try:
            # Try official ZoeDepth
            import torch.hub

            variant = config.variant.lower()

            # Map variants to model names
            variant_map = {
                "n": "ZoeD_N",  # NYU trained (indoor)
                "k": "ZoeD_K",  # KITTI trained (outdoor)
                "nk": "ZoeD_NK",  # Both (best generalization)
            }

            model_name = variant_map.get(variant, "ZoeD_NK")

            # Load from torch hub
            model = torch.hub.load(
                "isl-org/ZoeDepth",
                model_name,
                pretrained=True,
                trust_repo=True,
            )

            model.eval()
            logger.info(f"Loaded ZoeDepth model: {model_name}")
            return model

        except Exception as e:
            logger.warning(f"Failed to load ZoeDepth from hub: {e}")
            logger.info("Falling back to MiDaS")
            return DepthEstimatorFactory._create_midas(config)

    @staticmethod
    def _create_metric3d(config: DepthConfig) -> nn.Module:
        """Create Metric3D model.

        Metric3D:
        - Produces metric depth when camera intrinsics are provided
        - Better scale accuracy than relative depth methods
        - Requires camera focal length for best results

        Paper: "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image"
        """
        try:
            import torch.hub

            # Metric3D v2 is the latest
            model = torch.hub.load(
                "yvanyin/metric3d",
                "metric3d_vit_large",
                pretrained=True,
                trust_repo=True,
            )

            model.eval()
            logger.info("Loaded Metric3D model (ViT-Large)")
            return model

        except Exception as e:
            logger.warning(f"Failed to load Metric3D: {e}")
            logger.info("Falling back to ZoeDepth")
            return DepthEstimatorFactory._create_zoedepth(config)

    @staticmethod
    def _create_depth_anything(config: DepthConfig) -> nn.Module:
        """Create Depth Anything model.

        Depth Anything:
        - Trained on 62M images (largest depth dataset)
        - Excellent generalization
        - Fast inference
        - Variants: vits (small), vitb (base), vitl (large)

        Paper: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"
        """
        try:
            from transformers import pipeline

            variant = config.variant.lower()

            # Map to HuggingFace model names
            variant_map = {
                "small": "LiheYoung/depth-anything-small-hf",
                "base": "LiheYoung/depth-anything-base-hf",
                "large": "LiheYoung/depth-anything-large-hf",
                "vits": "LiheYoung/depth-anything-small-hf",
                "vitb": "LiheYoung/depth-anything-base-hf",
                "vitl": "LiheYoung/depth-anything-large-hf",
            }

            model_id = variant_map.get(variant, variant_map["base"])

            # Create pipeline
            depth_pipe = pipeline(
                "depth-estimation",
                model=model_id,
                device=0 if "cuda" in config.device else -1,
            )

            logger.info(f"Loaded Depth Anything model: {model_id}")
            return depth_pipe

        except ImportError:
            logger.warning("transformers not installed for Depth Anything")
            logger.info("Falling back to MiDaS")
            return DepthEstimatorFactory._create_midas(config)

    @staticmethod
    def _create_midas(config: DepthConfig) -> nn.Module:
        """Create MiDaS model.

        MiDaS (Monocular Depth in the Wild):
        - Classic baseline, very robust
        - Produces relative (not metric) depth
        - Variants: DPT_Large, DPT_Hybrid, MiDaS_small

        Paper: "Towards Robust Monocular Depth Estimation"
        """
        try:
            import torch.hub

            variant = config.variant.lower()

            variant_map = {
                "large": "DPT_Large",
                "hybrid": "DPT_Hybrid",
                "small": "MiDaS_small",
            }

            model_type = variant_map.get(variant, "DPT_Large")

            model = torch.hub.load(
                "intel-isl/MiDaS",
                model_type,
                pretrained=True,
                trust_repo=True,
            )

            model.eval()
            logger.info(f"Loaded MiDaS model: {model_type}")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load any depth model: {e}")


def load_depth_model(params: Dict[str, Any]) -> nn.Module:
    """Load a depth estimation model.

    Args:
        params: Dictionary with depth configuration

    Returns:
        Loaded depth model

    Example:
        params = {
            "model": "zoedepth",
            "variant": "nk",
            "device": "cuda:0",
            "max_depth": 80.0,
        }
        model = load_depth_model(params)
    """
    config = DepthConfig(
        model=params.get("model", "zoedepth"),
        variant=params.get("variant", "nk"),
        device=params.get("device", "cuda:0"),
        max_depth=params.get("max_depth", 80.0),
        min_depth=params.get("min_depth", 0.1),
    )

    model = DepthEstimatorFactory.create(config)

    # Move to device if applicable
    device = config.device
    if hasattr(model, "to") and not isinstance(model, type(lambda: None)):
        # Skip pipeline objects
        try:
            model = model.to(device)
        except Exception:
            pass

    logger.info(f"Depth model loaded: {config.model} ({config.variant})")
    return model


def estimate_depth(
    model: nn.Module,
    images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
    params: Dict[str, Any],
) -> List[DepthResult]:
    """Estimate depth from images.

    Args:
        model: Depth estimation model
        images: Input images [B, H, W, C] or list of [H, W, C]
        params: Depth parameters

    Returns:
        List of DepthResult objects

    Example:
        results = estimate_depth(model, frames, {"model": "zoedepth"})
        for result in results:
            depth_at_center = result.get_depth_at(320, 240)
    """
    device = params.get("device", "cuda:0")
    max_depth = params.get("max_depth", 80.0)
    min_depth = params.get("min_depth", 0.1)
    model_name = params.get("model", "zoedepth").lower()

    # Handle input formats
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]

    results = []

    for image in images:
        start_time = time.time()

        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # Assume BGR from OpenCV, convert to RGB
            if image.shape[-1] == 3:
                image_rgb = image[..., ::-1].copy()
            else:
                image_rgb = image

            image_tensor = torch.from_numpy(image_rgb).float()

            # Normalize to [0, 1]
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0

            # HWC to CHW
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = image

        # Run inference based on model type
        with torch.no_grad():
            if model_name == "depth_anything":
                # HuggingFace pipeline
                from PIL import Image

                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = image
                output = model(pil_image)
                depth_map = np.array(output["depth"])

            elif hasattr(model, "infer"):
                # ZoeDepth style
                image_tensor = image_tensor.to(device)
                depth_map = model.infer(image_tensor)
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.squeeze().cpu().numpy()

            else:
                # MiDaS style
                image_tensor = image_tensor.to(device)

                # MiDaS expects normalized input
                depth_map = model(image_tensor)

                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.squeeze().cpu().numpy()

                # MiDaS outputs inverse depth, convert to depth
                # Normalize to [min_depth, max_depth] range
                depth_map = depth_map - depth_map.min()
                depth_map = depth_map / (depth_map.max() + 1e-8)
                depth_map = min_depth + depth_map * (max_depth - min_depth)

        # Clamp depth values
        depth_map = np.clip(depth_map, min_depth, max_depth)

        inference_time = time.time() - start_time

        result = DepthResult(
            depth_map=depth_map,
            inference_time=inference_time,
            min_depth=float(np.min(depth_map)),
            max_depth=float(np.max(depth_map)),
            median_depth=float(np.median(depth_map)),
        )
        results.append(result)

    logger.info(f"Depth estimation complete: {len(results)} frames")
    return results


def depth_to_pointcloud(
    depth_result: DepthResult,
    intrinsics: np.ndarray,
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert depth map to 3D point cloud.

    Args:
        depth_result: DepthResult with depth map
        intrinsics: Camera intrinsic matrix [3, 3] or [fx, fy, cx, cy]
        image: Optional RGB image for colored point cloud

    Returns:
        Point cloud [N, 3] or [N, 6] if image provided (XYZ + RGB)

    Math:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth[v, u]
    """
    depth_map = depth_result.depth_map
    h, w = depth_map.shape

    # Parse intrinsics
    if intrinsics.shape == (3, 3):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx, fy, cx, cy = intrinsics[:4]

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Compute 3D coordinates
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to point cloud
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter invalid points
    valid_mask = (z.flatten() > 0) & (z.flatten() < depth_result.max_depth)
    points = points[valid_mask]

    # Add color if image provided
    if image is not None:
        if image.ndim == 3 and image.shape[-1] == 3:
            colors = image.reshape(-1, 3)[valid_mask]
            points = np.concatenate([points, colors], axis=-1)

    return points


def lift_detections_to_3d(
    detections: List[Any],  # List[Detection] from object_detection
    depth_result: DepthResult,
    intrinsics: Optional[np.ndarray] = None,
    depth_method: str = "center",
) -> List[Detection3D]:
    """Lift 2D detections to 3D using depth map.

    Args:
        detections: List of 2D Detection objects
        depth_result: DepthResult with depth map
        intrinsics: Camera intrinsics [fx, fy, cx, cy] or [3, 3] matrix
        depth_method: How to sample depth - 'center', 'median', 'min'

    Returns:
        List of Detection3D objects

    Process:
        1. For each 2D detection, sample depth from depth map
        2. Use camera intrinsics to compute 3D position
        3. Optionally estimate 3D size from 2D size + depth
    """
    # Default intrinsics (approximate for 1920x1080)
    if intrinsics is None:
        # Assume FOV ~60 degrees
        h, w = depth_result.depth_map.shape
        fx = fy = w / (2 * np.tan(np.radians(30)))
        cx, cy = w / 2, h / 2
    elif intrinsics.shape == (3, 3):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx, fy, cx, cy = intrinsics[:4]

    detections_3d = []

    for det in detections:
        # Get 2D bbox
        bbox = det.bbox if hasattr(det, "bbox") else np.array(det["bbox"])
        x1, y1, x2, y2 = bbox[:4]

        # Sample depth
        depth = depth_result.get_depth_in_box(bbox, method=depth_method)

        if depth <= 0:
            continue

        # Compute 3D position (camera frame)
        # Center of bbox in image
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        # Back-project to 3D
        x_3d = (u - cx) * depth / fx
        y_3d = (v - cy) * depth / fy
        z_3d = depth

        position_3d = np.array([x_3d, y_3d, z_3d])

        # Estimate 3D size from 2D size and depth
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        width_3d = bbox_width * depth / fx
        height_3d = bbox_height * depth / fy
        length_3d = width_3d  # Approximate as cube for now

        size_3d = np.array([width_3d, height_3d, length_3d])

        det_3d = Detection3D(
            bbox_2d=bbox,
            position_3d=position_3d,
            depth=depth,
            confidence=det.confidence if hasattr(det, "confidence") else det.get("confidence", 1.0),
            class_id=det.class_id if hasattr(det, "class_id") else det.get("class_id", -1),
            class_name=det.class_name
            if hasattr(det, "class_name")
            else det.get("class_name", "unknown"),
            size_3d=size_3d,
        )
        detections_3d.append(det_3d)

    logger.info(f"Lifted {len(detections_3d)} detections to 3D")
    return detections_3d


def compute_depth_metrics(
    predictions: List[DepthResult],
    ground_truth: List[np.ndarray],
) -> Dict[str, float]:
    """Compute depth estimation metrics.

    Args:
        predictions: List of predicted DepthResult
        ground_truth: List of ground truth depth maps

    Returns:
        Dictionary of metrics (abs_rel, sq_rel, rmse, etc.)

    Standard Metrics:
        - Abs Rel: |d* - d| / d (absolute relative error)
        - Sq Rel: |d* - d|^2 / d (squared relative error)
        - RMSE: sqrt(mean((d* - d)^2)) (root mean squared error)
        - δ1: % of pixels with max(d*/d, d/d*) < 1.25
        - δ2: % of pixels with max(d*/d, d/d*) < 1.25^2
        - δ3: % of pixels with max(d*/d, d/d*) < 1.25^3
    """
    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []

    for pred, gt in zip(predictions, ground_truth):
        pred_depth = pred.depth_map

        # Resize if needed
        if pred_depth.shape != gt.shape:
            import cv2

            pred_depth = cv2.resize(
                pred_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        # Valid mask (gt > 0)
        valid_mask = gt > 0

        if valid_mask.sum() == 0:
            continue

        pred_valid = pred_depth[valid_mask]
        gt_valid = gt[valid_mask]

        # Compute metrics
        abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
        sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

        # Threshold metrics
        thresh = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        delta1 = np.mean(thresh < 1.25)
        delta2 = np.mean(thresh < 1.25**2)
        delta3 = np.mean(thresh < 1.25**3)

        abs_rel_list.append(abs_rel)
        sq_rel_list.append(sq_rel)
        rmse_list.append(rmse)
        delta1_list.append(delta1)
        delta2_list.append(delta2)
        delta3_list.append(delta3)

    metrics = {
        "abs_rel": float(np.mean(abs_rel_list)) if abs_rel_list else 0.0,
        "sq_rel": float(np.mean(sq_rel_list)) if sq_rel_list else 0.0,
        "rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "delta1": float(np.mean(delta1_list)) if delta1_list else 0.0,
        "delta2": float(np.mean(delta2_list)) if delta2_list else 0.0,
        "delta3": float(np.mean(delta3_list)) if delta3_list else 0.0,
        "num_frames": len(predictions),
    }

    logger.info(f"Depth metrics: Abs Rel={metrics['abs_rel']:.4f}, RMSE={metrics['rmse']:.2f}")
    return metrics
