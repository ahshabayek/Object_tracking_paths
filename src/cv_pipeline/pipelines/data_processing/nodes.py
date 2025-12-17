"""Data Processing Pipeline Nodes.

This module contains all node functions for data ingestion and preprocessing.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def load_video_frames(
    video_source: Any,
    params: Dict[str, Any],
) -> List[np.ndarray]:
    """Load frames from a video source.
    
    Args:
        video_source: Video file path or camera stream
        params: Configuration parameters
            - target_fps: Target FPS for extraction
            - resize: Target resolution [width, height]
            - max_frames: Maximum frames to extract
    
    Returns:
        List of numpy arrays representing video frames
    """
    target_fps = params.get("target_fps", 30)
    resize = params.get("resize", None)
    max_frames = params.get("max_frames", None)
    
    frames = []
    
    # Handle different input types
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    elif isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        raise ValueError(f"Unsupported video source type: {type(video_source)}")
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if specified
            if resize is not None:
                frame = cv2.resize(frame, tuple(resize))
            
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    
    return frames


def load_image_batch(
    image_paths: List[str],
    params: Dict[str, Any],
) -> List[np.ndarray]:
    """Load a batch of images from file paths.
    
    Args:
        image_paths: List of image file paths
        params: Configuration parameters
    
    Returns:
        List of numpy arrays representing images
    """
    resize = params.get("resize", None)
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if resize is not None:
            img = cv2.resize(img, tuple(resize))
        
        images.append(img)
    
    logger.info(f"Loaded {len(images)} images")
    return images


def extract_frame_metadata(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Extract metadata from frames.
    
    Args:
        frames: List of image frames
    
    Returns:
        Dictionary containing frame metadata
    """
    if not frames:
        return {}
    
    sample_frame = frames[0]
    
    metadata = {
        "num_frames": len(frames),
        "height": sample_frame.shape[0],
        "width": sample_frame.shape[1],
        "channels": sample_frame.shape[2] if len(sample_frame.shape) > 2 else 1,
        "dtype": str(sample_frame.dtype),
        "frame_sizes_bytes": [f.nbytes for f in frames],
    }
    
    return metadata


def preprocess_frames(
    frames: List[np.ndarray],
    params: Dict[str, Any],
) -> List[np.ndarray]:
    """Preprocess frames for model input.
    
    Args:
        frames: List of raw frames
        params: Preprocessing parameters
            - normalize: Normalization settings (mean, std)
            - input_size: Target input size [height, width]
    
    Returns:
        List of preprocessed frames
    """
    input_size = params.get("input_size", [640, 640])
    normalize = params.get("normalize", {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    })
    
    preprocessed = []
    
    for frame in frames:
        # Resize to model input size
        processed = cv2.resize(frame, tuple(input_size))
        
        # Convert to float and normalize
        processed = processed.astype(np.float32) / 255.0
        
        # Apply normalization
        mean = np.array(normalize["mean"], dtype=np.float32)
        std = np.array(normalize["std"], dtype=np.float32)
        processed = (processed - mean) / std
        
        preprocessed.append(processed)
    
    logger.info(f"Preprocessed {len(preprocessed)} frames")
    return preprocessed


def apply_augmentations(
    frames: List[np.ndarray],
    augmentation_params: Dict[str, Any],
) -> List[np.ndarray]:
    """Apply data augmentations to frames.
    
    Args:
        frames: List of preprocessed frames
        augmentation_params: Augmentation configuration
    
    Returns:
        List of augmented frames
    """
    # Build augmentation pipeline
    transforms = []
    
    if augmentation_params.get("horizontal_flip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=augmentation_params["horizontal_flip"]))
    
    if augmentation_params.get("color_jitter", False):
        transforms.append(A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ))
    
    if augmentation_params.get("random_brightness_contrast", False):
        transforms.append(A.RandomBrightnessContrast(p=0.3))
    
    if augmentation_params.get("blur", False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.1))
    
    if augmentation_params.get("noise", False):
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.1))
    
    # Create transform pipeline
    transform = A.Compose(transforms) if transforms else None
    
    augmented = []
    for frame in frames:
        if transform is not None:
            # Denormalize for augmentation (assuming normalized input)
            frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
            result = transform(image=frame_uint8)
            augmented_frame = result["image"].astype(np.float32) / 255.0
            augmented.append(augmented_frame)
        else:
            augmented.append(frame)
    
    logger.info(f"Applied augmentations to {len(augmented)} frames")
    return augmented


def create_batches(
    frames: List[np.ndarray],
    batch_size: int = 1,
) -> List[torch.Tensor]:
    """Create batched tensors from frames.
    
    Args:
        frames: List of preprocessed frames
        batch_size: Number of frames per batch
    
    Returns:
        List of batched tensors [B, C, H, W]
    """
    batches = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        # Convert to tensors and stack
        tensors = []
        for frame in batch_frames:
            # HWC to CHW
            tensor = torch.from_numpy(frame.transpose(2, 0, 1))
            tensors.append(tensor)
        
        batch = torch.stack(tensors)
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches of size {batch_size}")
    return batches


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image while maintaining aspect ratio with padding.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        pad_value: Padding fill value
    
    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    
    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # Place resized image
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, (pad_w, pad_h)
