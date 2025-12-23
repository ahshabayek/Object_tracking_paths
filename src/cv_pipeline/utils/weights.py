"""Model Weights Manager for CV Pipeline.

This module provides utilities for managing pretrained model weights,
including downloading, caching, and loading weights for various models.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default weights directory
DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "cv_pipeline" / "weights"

# Model weight registry with download URLs and expected checksums
WEIGHT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # RT-DETR weights (Ultralytics)
    "rtdetr-l": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-l.pt",
        "filename": "rtdetr-l.pt",
        "size_mb": 126,
        "framework": "ultralytics",
    },
    "rtdetr-x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-x.pt",
        "filename": "rtdetr-x.pt",
        "size_mb": 232,
        "framework": "ultralytics",
    },
    # YOLOv11 weights
    "yolo11n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "filename": "yolo11n.pt",
        "size_mb": 5.4,
        "framework": "ultralytics",
    },
    "yolo11s": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "filename": "yolo11s.pt",
        "size_mb": 18.4,
        "framework": "ultralytics",
    },
    "yolo11m": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "filename": "yolo11m.pt",
        "size_mb": 38.8,
        "framework": "ultralytics",
    },
    "yolo11l": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "filename": "yolo11l.pt",
        "size_mb": 49.0,
        "framework": "ultralytics",
    },
    "yolo11x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "filename": "yolo11x.pt",
        "size_mb": 109.3,
        "framework": "ultralytics",
    },
    # OSNet Re-ID weights
    "osnet_x1_0": {
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_imagenet.pth",
        "filename": "osnet_x1_0_imagenet.pth",
        "size_mb": 8.7,
        "framework": "torchreid",
    },
    "osnet_x0_75": {
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_75_imagenet.pth",
        "filename": "osnet_x0_75_imagenet.pth",
        "size_mb": 5.5,
        "framework": "torchreid",
    },
    "osnet_x0_5": {
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_5_imagenet.pth",
        "filename": "osnet_x0_5_imagenet.pth",
        "size_mb": 2.8,
        "framework": "torchreid",
    },
    "osnet_x0_25": {
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_25_imagenet.pth",
        "filename": "osnet_x0_25_imagenet.pth",
        "size_mb": 1.0,
        "framework": "torchreid",
    },
    "osnet_ain_x1_0": {
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_ain_x1_0_msmt17.pth",
        "filename": "osnet_ain_x1_0_msmt17.pth",
        "size_mb": 8.8,
        "framework": "torchreid",
    },
    # CLRNet Lane Detection
    "clrnet_culane_r18": {
        "url": None,  # Manual download required
        "filename": "clrnet_culane_r18.pth",
        "size_mb": 44,
        "framework": "clrnet",
        "manual": True,
    },
    "clrnet_culane_r34": {
        "url": None,
        "filename": "clrnet_culane_r34.pth",
        "size_mb": 85,
        "framework": "clrnet",
        "manual": True,
    },
    "clrnet_tusimple_r18": {
        "url": None,
        "filename": "clrnet_tusimple_r18.pth",
        "size_mb": 44,
        "framework": "clrnet",
        "manual": True,
    },
}


class WeightsManager:
    """Manager for handling model weights download, caching, and loading.

    Example:
        >>> manager = WeightsManager()
        >>> weights_path = manager.get_weights("yolo11l")
        >>> model.load_state_dict(torch.load(weights_path))
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the weights manager.

        Args:
            cache_dir: Directory to cache downloaded weights.
                      Defaults to ~/.cache/cv_pipeline/weights
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_WEIGHTS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Weights cache directory: {self.cache_dir}")

    def get_weights(
        self,
        model_name: str,
        force_download: bool = False,
    ) -> Path:
        """Get the path to model weights, downloading if necessary.

        Args:
            model_name: Name of the model (e.g., 'yolo11l', 'osnet_x1_0').
            force_download: Force re-download even if cached.

        Returns:
            Path to the weights file.

        Raises:
            ValueError: If model not in registry.
            RuntimeError: If download fails or manual download required.
        """
        if model_name not in WEIGHT_REGISTRY:
            available = list(WEIGHT_REGISTRY.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        info = WEIGHT_REGISTRY[model_name]
        weights_path = self.cache_dir / info["filename"]

        # Check if already cached
        if weights_path.exists() and not force_download:
            logger.info(f"Using cached weights: {weights_path}")
            return weights_path

        # Check if manual download required
        if info.get("manual", False):
            raise RuntimeError(
                f"Model {model_name} requires manual download. "
                f"Please download weights from the official source and place at: {weights_path}"
            )

        # Download weights
        self._download_weights(info["url"], weights_path, info.get("size_mb", 0))

        return weights_path

    def _download_weights(
        self,
        url: str,
        destination: Path,
        expected_size_mb: float = 0,
    ) -> None:
        """Download weights from URL.

        Args:
            url: URL to download from.
            destination: Path to save the weights.
            expected_size_mb: Expected file size in MB for progress reporting.
        """
        try:
            import requests
            from tqdm import tqdm
        except ImportError:
            logger.warning("requests or tqdm not installed. Using urllib.")
            self._download_with_urllib(url, destination)
            return

        logger.info(f"Downloading weights from {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(destination, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded weights to {destination}")

    def _download_with_urllib(self, url: str, destination: Path) -> None:
        """Fallback download using urllib."""
        import urllib.request

        logger.info(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Download complete: {destination}")

    def list_cached(self) -> List[str]:
        """List all cached model weights.

        Returns:
            List of cached model names.
        """
        cached = []
        for model_name, info in WEIGHT_REGISTRY.items():
            weights_path = self.cache_dir / info["filename"]
            if weights_path.exists():
                cached.append(model_name)
        return cached

    def list_available(self) -> List[str]:
        """List all available models in the registry.

        Returns:
            List of available model names.
        """
        return list(WEIGHT_REGISTRY.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with model information.
        """
        if model_name not in WEIGHT_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        info = WEIGHT_REGISTRY[model_name].copy()
        weights_path = self.cache_dir / info["filename"]
        info["cached"] = weights_path.exists()
        info["path"] = str(weights_path) if weights_path.exists() else None

        return info

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """Clear cached weights.

        Args:
            model_name: Specific model to clear, or None for all.
        """
        if model_name:
            if model_name not in WEIGHT_REGISTRY:
                raise ValueError(f"Unknown model: {model_name}")
            weights_path = self.cache_dir / WEIGHT_REGISTRY[model_name]["filename"]
            if weights_path.exists():
                weights_path.unlink()
                logger.info(f"Cleared cache for {model_name}")
        else:
            for model_name, info in WEIGHT_REGISTRY.items():
                weights_path = self.cache_dir / info["filename"]
                if weights_path.exists():
                    weights_path.unlink()
            logger.info("Cleared all cached weights")

    def cache_size(self) -> float:
        """Get total size of cached weights in MB.

        Returns:
            Total cache size in MB.
        """
        total_bytes = 0
        for model_name, info in WEIGHT_REGISTRY.items():
            weights_path = self.cache_dir / info["filename"]
            if weights_path.exists():
                total_bytes += weights_path.stat().st_size
        return total_bytes / (1024 * 1024)


def load_weights(
    model: nn.Module,
    weights_path: Union[str, Path],
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Load weights into a PyTorch model.

    Args:
        model: PyTorch model to load weights into.
        weights_path: Path to the weights file.
        strict: Whether to strictly enforce that keys match.
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0').

    Returns:
        Tuple of (missing_keys, unexpected_keys).
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Loading weights from {weights_path}")

    # Determine map_location
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=map_location)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    # Handle DataParallel/DistributedDataParallel prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load state dict
    result = model.load_state_dict(state_dict, strict=strict)

    if result.missing_keys:
        logger.warning(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys: {result.unexpected_keys}")

    logger.info(f"Loaded weights successfully (strict={strict})")

    return result.missing_keys, result.unexpected_keys


def get_checkpoint_info(weights_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a checkpoint file.

    Args:
        weights_path: Path to the weights/checkpoint file.

    Returns:
        Dictionary with checkpoint information.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    info = {
        "path": str(weights_path),
        "size_mb": weights_path.stat().st_size / (1024 * 1024),
        "filename": weights_path.name,
    }

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            info["type"] = "checkpoint_dict"
            info["keys"] = list(checkpoint.keys())

            if "epoch" in checkpoint:
                info["epoch"] = checkpoint["epoch"]
            if "optimizer" in checkpoint:
                info["has_optimizer"] = True
            if "model" in checkpoint:
                info["model_keys"] = len(checkpoint["model"])
            elif "state_dict" in checkpoint:
                info["model_keys"] = len(checkpoint["state_dict"])
        else:
            info["type"] = "state_dict"
            if hasattr(checkpoint, "keys"):
                info["num_keys"] = len(checkpoint.keys())
    except Exception as e:
        info["error"] = str(e)

    return info


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint.

    Args:
        model: PyTorch model to save.
        path: Path to save the checkpoint.
        optimizer: Optional optimizer state to save.
        epoch: Optional epoch number.
        metrics: Optional metrics dictionary.
        config: Optional configuration dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle DataParallel/DistributedDataParallel
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model": model_to_save.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """Get model size information.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with size information.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate memory
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "param_size_mb": param_size_bytes / (1024 * 1024),
        "buffer_size_mb": buffer_size_bytes / (1024 * 1024),
        "total_size_mb": (param_size_bytes + buffer_size_bytes) / (1024 * 1024),
    }


# Convenience function for getting weights
def get_pretrained_weights(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Get path to pretrained weights, downloading if necessary.

    Args:
        model_name: Name of the model.
        cache_dir: Optional cache directory.

    Returns:
        Path to the weights file.

    Example:
        >>> weights_path = get_pretrained_weights("yolo11l")
        >>> model = YOLO(weights_path)
    """
    manager = WeightsManager(cache_dir)
    return manager.get_weights(model_name)
