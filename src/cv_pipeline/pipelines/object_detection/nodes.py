"""Object Detection Pipeline Nodes.

This module contains node functions for object detection using various models:
- RF-DETR (Roboflow Detection Transformer)
- RT-DETR / RT-DETRv2 (Real-Time Detection Transformer)
- D-FINE (Fine-grained Distribution Refinement DETR) - ICLR 2025 Spotlight
- YOLOv10/YOLOv11/YOLOv12 (Ultralytics)
- YOLO-World (Zero-shot open-vocabulary detection)
- GroundingDINO (Text-prompted detection)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class representing a single detection.

    Attributes:
        bbox: Bounding box in [x1, y1, x2, y2] format
        confidence: Detection confidence score [0, 1]
        class_id: Class ID (for closed-vocabulary models, -1 for open-vocab)
        class_name: Class name or text prompt (for open-vocabulary models)
        phrase: Optional text phrase for GroundingDINO detections
        features: Optional feature embedding for Re-ID
    """

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    phrase: Optional[str] = None  # For GroundingDINO text-prompted detections
    features: Optional[np.ndarray] = None  # For Re-ID embeddings

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }
        if self.phrase is not None:
            result["phrase"] = self.phrase
        return result


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
    """Factory class for creating detection models.

    Supported Models:
        - rf_detr: Roboflow Detection Transformer (DINOv2 backbone)
        - rt_detr, rt_detr_v2: Real-Time Detection Transformer (CVPR 2024)
        - d_fine: Fine-grained Distribution Refinement DETR (ICLR 2025 Spotlight)
        - yolov10, yolov11, yolov12: Ultralytics YOLO family
        - yolo_world: Zero-shot open-vocabulary YOLO (Tencent AI Lab)
        - grounding_dino: Text-prompted detection (IDEA Research)
    """

    SUPPORTED_MODELS = [
        "rf_detr",
        "rt_detr",
        "rt_detr_v2",
        "d_fine",
        "yolov12",
        "yolov11",
        "yolov10",
        "yolo_world",
        "grounding_dino",
    ]

    # Models that support open-vocabulary / zero-shot detection
    OPEN_VOCAB_MODELS = ["yolo_world", "grounding_dino"]

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
        elif model_name in ["rt_detr", "rt_detr_v2"]:
            return DetectorFactory._create_rt_detr(params)
        elif model_name == "d_fine":
            return DetectorFactory._create_d_fine(params)
        elif model_name in ["yolov12", "yolov11", "yolov10"]:
            return DetectorFactory._create_yolo(model_name, params)
        elif model_name == "yolo_world":
            return DetectorFactory._create_yolo_world(params)
        elif model_name == "grounding_dino":
            return DetectorFactory._create_grounding_dino(params)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: {DetectorFactory.SUPPORTED_MODELS}"
            )

    @staticmethod
    def is_open_vocab(model_name: str) -> bool:
        """Check if a model supports open-vocabulary detection."""
        return model_name in DetectorFactory.OPEN_VOCAB_MODELS

    @staticmethod
    def _create_rf_detr(params: Dict[str, Any]) -> nn.Module:
        """Create RF-DETR model.

        RF-DETR uses DINOv2 backbone and achieves 54.7% mAP on COCO.
        First real-time detector to break 60 mAP barrier.
        """
        try:
            from rfdetr import RFDETR

            variant = params.get("variant", "l")
            weights = params.get("weights", {}).get("rf_detr", f"rfdetr-{variant}.pt")

            model = RFDETR(weights=weights)
            logger.info(f"Loaded RF-DETR model: {weights}")
            return model
        except ImportError:
            logger.warning("RF-DETR not installed. Install with: pip install rfdetr")
            logger.warning("Falling back to RT-DETR")
            return DetectorFactory._create_rt_detr(params)

    @staticmethod
    def _create_rt_detr(params: Dict[str, Any]) -> nn.Module:
        """Create RT-DETR or RT-DETRv2 model.

        RT-DETR (CVPR 2024): First DETR to beat YOLO in real-time detection.
        RT-DETRv2: Improved version with better small object detection.

        Supports flexible speed tuning by adjusting decoder layers.
        """
        from ultralytics import RTDETR

        variant = params.get("variant", "l")
        version = params.get("version", "v1")  # v1 or v2

        if version == "v2":
            weights = params.get("weights", {}).get("rt_detr_v2", f"rtdetr-{variant}.pt")
        else:
            weights = params.get("weights", {}).get("rt_detr", f"rtdetr-{variant}.pt")

        model = RTDETR(weights)
        logger.info(f"Loaded RT-DETR{version.upper()} model: {weights}")
        return model

    @staticmethod
    def _create_d_fine(params: Dict[str, Any]) -> nn.Module:
        """Create D-FINE model.

        D-FINE (ICLR 2025 Spotlight): Fine-grained Distribution Refinement for DETR.
        - Achieves 57.4% mAP on COCO (D-FINE-L)
        - Apache 2.0 license (commercial-friendly)
        - Better localization precision than YOLO

        Install: pip install dfine
        Or clone: https://github.com/Peterande/D-FINE
        """
        try:
            # Try official D-FINE package
            from dfine import DFINE

            variant = params.get("variant", "l")  # n, s, m, l, x
            weights = params.get("weights", {}).get("d_fine", f"dfine-{variant}.pth")

            model = DFINE(weights=weights)
            logger.info(f"Loaded D-FINE model: {weights}")
            return model

        except ImportError:
            try:
                # Try loading from local D-FINE repo
                import sys

                dfine_path = params.get("dfine_repo_path", "")
                if dfine_path:
                    sys.path.insert(0, dfine_path)

                from src.core import YAMLConfig
                from src.solver import DetSolver

                config_path = params.get("config_path")
                weights_path = params.get("weights_path")

                if not config_path or not weights_path:
                    raise ValueError(
                        "D-FINE requires 'config_path' and 'weights_path' in params. "
                        "Download from: https://github.com/Peterande/D-FINE"
                    )

                cfg = YAMLConfig(config_path)
                solver = DetSolver(cfg)
                solver.load_checkpoint(weights_path)
                model = solver.model
                model.eval()

                logger.info(f"Loaded D-FINE model from repo: {weights_path}")
                return model

            except ImportError:
                raise ImportError(
                    "D-FINE not installed. Install with:\n"
                    "  pip install dfine\n"
                    "Or clone the repo:\n"
                    "  git clone https://github.com/Peterande/D-FINE\n"
                    "  cd D-FINE && pip install -e ."
                )

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

    @staticmethod
    def _create_yolo_world(params: Dict[str, Any]) -> nn.Module:
        """Create YOLO-World model for zero-shot open-vocabulary detection.

        YOLO-World (Tencent AI Lab, January 2024):
        - Zero-shot detection: detect any object by text prompt
        - No retraining needed for new classes
        - Real-time performance

        Usage:
            params = {
                "model": "yolo_world",
                "variant": "l",  # s, m, l, x
                "text_prompts": ["person", "car", "dog", "traffic light"]
            }
        """
        from ultralytics import YOLO

        variant = params.get("variant", "l")
        weights = params.get("weights", {}).get("yolo_world", f"yolov8{variant}-world.pt")

        model = YOLO(weights)

        # Set custom classes if provided
        text_prompts = params.get("text_prompts", None)
        if text_prompts:
            model.set_classes(text_prompts)
            logger.info(f"YOLO-World classes set to: {text_prompts}")

        logger.info(f"Loaded YOLO-World model: {weights}")
        return model

    @staticmethod
    def _create_grounding_dino(params: Dict[str, Any]) -> nn.Module:
        """Create GroundingDINO model for text-prompted detection.

        GroundingDINO (IDEA Research):
        - Open-vocabulary detection with language grounding
        - Achieves 52.5% AP on COCO zero-shot, 63.0% after fine-tuning
        - Accepts natural language text prompts

        Usage:
            params = {
                "model": "grounding_dino",
                "variant": "base",  # tiny, base
                "text_prompt": "person . car . traffic light ."
            }

        Note: Text prompts should be separated by " . " (space-dot-space)
        """
        try:
            from groundingdino.util.inference import load_model
            from groundingdino.util.inference import predict as gd_predict

            variant = params.get("variant", "base")  # tiny, base

            # Default paths - user should override
            config_path = params.get("config_path")
            weights_path = params.get("weights_path")

            if not config_path or not weights_path:
                # Try default Hugging Face paths
                try:
                    from huggingface_hub import hf_hub_download

                    if variant == "tiny":
                        config_path = hf_hub_download(
                            repo_id="ShilongLiu/GroundingDINO",
                            filename="GroundingDINO_SwinT_OGC.cfg.py",
                        )
                        weights_path = hf_hub_download(
                            repo_id="ShilongLiu/GroundingDINO",
                            filename="groundingdino_swint_ogc.pth",
                        )
                    else:  # base
                        config_path = hf_hub_download(
                            repo_id="ShilongLiu/GroundingDINO",
                            filename="GroundingDINO_SwinB.cfg.py",
                        )
                        weights_path = hf_hub_download(
                            repo_id="ShilongLiu/GroundingDINO",
                            filename="groundingdino_swinb_cogcoor.pth",
                        )
                except ImportError:
                    raise ValueError(
                        "GroundingDINO requires 'config_path' and 'weights_path' in params, "
                        "or install huggingface_hub: pip install huggingface_hub"
                    )

            model = load_model(config_path, weights_path)
            logger.info(f"Loaded GroundingDINO ({variant}): {weights_path}")
            return model

        except ImportError:
            raise ImportError(
                "GroundingDINO not installed. Install with:\n"
                "  pip install groundingdino\n"
                "Or clone:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO\n"
                "  cd GroundingDINO && pip install -e ."
            )


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
        batches: List of batched input tensors [B, C, H, W] or list of images
        params: Detection parameters

    Returns:
        List of raw model outputs
    """
    model_name = params.get("model", "rt_detr")
    device = params.get("device", "cuda:0")
    conf_thresh = params.get("confidence_threshold", 0.25)
    iou_thresh = params.get("iou_threshold", 0.45)
    classes = params.get("classes", None)

    results = []
    total_time = 0

    for batch_idx, batch in enumerate(batches):
        start_time = time.time()

        # Handle different model types
        if model_name == "grounding_dino":
            output = _run_grounding_dino_inference(model, batch, params, conf_thresh)
        elif model_name == "d_fine":
            output = _run_d_fine_inference(model, batch, params, device, conf_thresh)
        else:
            # Standard inference for Ultralytics models (YOLO, RT-DETR, YOLO-World)
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
                if params.get("half_precision", True) and torch.cuda.is_available():
                    batch = batch.half()

            if hasattr(model, "predict"):
                # Ultralytics models (YOLO, RT-DETR, YOLO-World)
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

        results.append(
            {
                "batch_idx": batch_idx,
                "output": output,
                "inference_time": inference_time,
                "model_name": model_name,
            }
        )

    avg_time = total_time / len(batches) if batches else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0

    logger.info(f"Detection inference complete: {len(batches)} batches, {fps:.1f} FPS")

    return results


def _run_grounding_dino_inference(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    params: Dict[str, Any],
    conf_thresh: float,
) -> Dict[str, Any]:
    """Run GroundingDINO inference with text prompts.

    Args:
        model: GroundingDINO model
        image: Input image (numpy array or tensor)
        params: Detection parameters including text_prompt
        conf_thresh: Confidence threshold

    Returns:
        Dictionary with boxes, logits, and phrases
    """
    try:
        import PIL.Image as Image
        from groundingdino.util.inference import load_image
        from groundingdino.util.inference import predict as gd_predict
    except ImportError:
        raise ImportError("GroundingDINO not installed")

    text_prompt = params.get("text_prompt", "person . car . truck .")
    box_thresh = params.get("box_threshold", conf_thresh)
    text_thresh = params.get("text_threshold", 0.25)

    # Convert to PIL Image if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]  # Take first image from batch
        if image.shape[0] == 3:  # CHW -> HWC
            image = image.transpose(1, 2, 0)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Run prediction
    boxes, logits, phrases = gd_predict(
        model=model,
        image=pil_image,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
    )

    return {
        "boxes": boxes,
        "logits": logits,
        "phrases": phrases,
        "image_size": pil_image.size,
    }


def _run_d_fine_inference(
    model: nn.Module,
    batch: Union[np.ndarray, torch.Tensor],
    params: Dict[str, Any],
    device: str,
    conf_thresh: float,
) -> Dict[str, Any]:
    """Run D-FINE inference.

    Args:
        model: D-FINE model
        batch: Input batch (numpy array or tensor)
        params: Detection parameters
        device: Device to run inference on
        conf_thresh: Confidence threshold

    Returns:
        Dictionary with detection outputs
    """
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)

    batch = batch.to(device)

    if params.get("half_precision", True) and torch.cuda.is_available():
        batch = batch.half()

    with torch.no_grad():
        # D-FINE outputs predictions directly
        if hasattr(model, "predict"):
            outputs = model.predict(batch, conf_threshold=conf_thresh)
        else:
            outputs = model(batch)

    return {
        "outputs": outputs,
        "conf_thresh": conf_thresh,
    }


# COCO class names (module-level constant)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def post_process_detections(
    raw_results: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[DetectionResult]:
    """Post-process raw detection outputs.

    Handles outputs from all supported models:
    - Ultralytics (YOLO, RT-DETR, YOLO-World)
    - GroundingDINO
    - D-FINE
    - RF-DETR

    Args:
        raw_results: Raw model outputs
        params: Detection parameters

    Returns:
        List of processed DetectionResult objects
    """
    conf_thresh = params.get("confidence_threshold", 0.25)
    model_name = params.get("model", "rt_detr")

    processed_results = []

    for result in raw_results:
        output = result["output"]
        inference_time = result["inference_time"]
        batch_idx = result["batch_idx"]
        result_model = result.get("model_name", model_name)

        # Route to appropriate post-processor based on model type
        if result_model == "grounding_dino":
            detections = _post_process_grounding_dino(output, conf_thresh)
            processed_results.append(
                DetectionResult(
                    frame_id=batch_idx,
                    detections=detections,
                    inference_time=inference_time,
                )
            )
        elif result_model == "d_fine":
            detections = _post_process_d_fine(output, conf_thresh)
            processed_results.append(
                DetectionResult(
                    frame_id=batch_idx,
                    detections=detections,
                    inference_time=inference_time,
                )
            )
        elif _is_ultralytics_output(output):
            # Handle Ultralytics results (YOLO, RT-DETR, YOLO-World)
            for frame_idx, frame_result in enumerate(output):
                detections = _post_process_ultralytics(frame_result, conf_thresh, result_model)
                processed_results.append(
                    DetectionResult(
                        frame_id=batch_idx * len(output) + frame_idx,
                        detections=detections,
                        inference_time=inference_time / len(output),
                    )
                )
        else:
            # Handle raw tensor outputs or unknown formats
            processed_results.append(
                DetectionResult(
                    frame_id=batch_idx,
                    detections=[],
                    inference_time=inference_time,
                )
            )

    total_dets = sum(r.num_detections for r in processed_results)
    logger.info(f"Post-processed {len(processed_results)} frames, {total_dets} total detections")

    return processed_results


def _is_ultralytics_output(output: Any) -> bool:
    """Check if output is from an Ultralytics model."""
    try:
        # Must be a list-like object (not dict) with items that have 'boxes' attribute
        if isinstance(output, dict):
            return False
        return hasattr(output, "__iter__") and len(output) > 0 and hasattr(output[0], "boxes")
    except (IndexError, TypeError, KeyError):
        return False


def _post_process_ultralytics(
    frame_result: Any,
    conf_thresh: float,
    model_name: str,
) -> List[Detection]:
    """Post-process Ultralytics model output (YOLO, RT-DETR, YOLO-World).

    Args:
        frame_result: Single frame result from Ultralytics model
        conf_thresh: Confidence threshold
        model_name: Name of the model for class name handling

    Returns:
        List of Detection objects
    """
    boxes = frame_result.boxes
    detections = []

    # Get class names - YOLO-World may have custom classes
    if hasattr(frame_result, "names"):
        class_names = frame_result.names
    else:
        class_names = {i: name for i, name in enumerate(COCO_CLASSES)}

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        conf = boxes.conf[i].cpu().item()
        cls_id = int(boxes.cls[i].cpu().item())

        if conf >= conf_thresh:
            # Get class name from model's class mapping
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id, f"class_{cls_id}")
            elif isinstance(class_names, list) and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = (
                    COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                )

            det = Detection(
                bbox=box,
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
            )
            detections.append(det)

    return detections


def _post_process_grounding_dino(
    output: Dict[str, Any],
    conf_thresh: float,
) -> List[Detection]:
    """Post-process GroundingDINO output.

    Args:
        output: Dictionary with boxes, logits, phrases from GroundingDINO
        conf_thresh: Confidence threshold

    Returns:
        List of Detection objects
    """
    detections = []

    boxes = output.get("boxes", [])
    logits = output.get("logits", [])
    phrases = output.get("phrases", [])
    image_size = output.get("image_size", (1, 1))

    # Convert normalized boxes to pixel coordinates
    img_w, img_h = image_size

    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        conf = float(logit)

        if conf >= conf_thresh:
            # GroundingDINO returns normalized cx, cy, w, h
            if len(box) == 4:
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()

                # Convert from cxcywh normalized to xyxy pixel coords
                cx, cy, w, h = box
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h

                bbox = np.array([x1, y1, x2, y2])
            else:
                bbox = np.array(box)

            det = Detection(
                bbox=bbox,
                confidence=conf,
                class_id=-1,  # Open vocabulary, no fixed class ID
                class_name=phrase.strip(),
                phrase=phrase.strip(),
            )
            detections.append(det)

    return detections


def _post_process_d_fine(
    output: Dict[str, Any],
    conf_thresh: float,
) -> List[Detection]:
    """Post-process D-FINE output.

    Args:
        output: Dictionary with outputs from D-FINE
        conf_thresh: Confidence threshold

    Returns:
        List of Detection objects
    """
    detections = []

    outputs = output.get("outputs", output)

    # D-FINE output format: dict with 'pred_boxes', 'pred_logits', etc.
    if isinstance(outputs, dict):
        pred_boxes = outputs.get("pred_boxes", [])
        pred_logits = outputs.get("pred_logits", [])

        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_logits, torch.Tensor):
            pred_logits = torch.softmax(pred_logits, dim=-1).cpu().numpy()

        # Handle batch dimension
        if pred_boxes.ndim == 3:
            pred_boxes = pred_boxes[0]
            pred_logits = pred_logits[0]

        for i, (box, logits) in enumerate(zip(pred_boxes, pred_logits)):
            # Get class with highest probability (excluding background)
            cls_id = int(np.argmax(logits[:-1])) if len(logits) > 1 else 0
            conf = float(logits[cls_id]) if cls_id < len(logits) else float(np.max(logits))

            if conf >= conf_thresh:
                # D-FINE uses normalized cxcywh format
                if len(box) == 4:
                    cx, cy, w, h = box
                    # Keep as cxcywh for now, will be converted if needed
                    # Assuming output is already in pixel coords after model post-proc
                    bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                else:
                    bbox = np.array(box[:4])

                class_name = (
                    COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                )

                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                )
                detections.append(det)

    # Handle list of detections format
    elif isinstance(outputs, (list, tuple)):
        for item in outputs:
            if hasattr(item, "boxes"):
                # Similar to Ultralytics format
                boxes = item.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy() if hasattr(boxes, "xyxy") else boxes[i][:4]
                    conf = float(boxes.conf[i]) if hasattr(boxes, "conf") else float(boxes[i][4])
                    cls_id = int(boxes.cls[i]) if hasattr(boxes, "cls") else int(boxes[i][5])

                    if conf >= conf_thresh:
                        det = Detection(
                            bbox=np.array(box),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=COCO_CLASSES[cls_id]
                            if cls_id < len(COCO_CLASSES)
                            else f"class_{cls_id}",
                        )
                        detections.append(det)

    return detections


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
        filtered_detections = [det for det in result.detections if det.class_id in all_target_ids]

        filtered_results.append(
            DetectionResult(
                frame_id=result.frame_id,
                detections=filtered_detections,
                inference_time=result.inference_time,
            )
        )

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
