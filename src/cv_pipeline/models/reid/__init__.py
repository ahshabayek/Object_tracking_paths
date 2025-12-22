"""Re-ID Feature Extraction for Multi-Object Tracking.

This module provides Re-ID (Re-Identification) feature extractors for
appearance-based tracking in BoT-SORT and DeepSORT trackers.

Supported models:
- OSNet: Omni-Scale Network for person re-identification
- FastReID: Fast and accurate Re-ID framework
- Simple CNN: Lightweight feature extractor for basic use cases

Usage:
    from cv_pipeline.models.reid import ReIDExtractor

    # Create extractor
    extractor = ReIDExtractor.create("osnet", device="cuda:0")

    # Extract features from image crops
    features = extractor.extract(image, bboxes)  # [N, feature_dim]
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseReIDExtractor(ABC):
    """Base class for Re-ID feature extractors."""

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 32,
        input_size: Tuple[int, int] = (256, 128),  # (height, width)
    ):
        """Initialize extractor.

        Args:
            device: Device to run inference on.
            batch_size: Maximum batch size for inference.
            input_size: Input image size (height, width).
        """
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.model: Optional[nn.Module] = None

        # Normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    @abstractmethod
    def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load the Re-ID model weights."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        pass

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image crop.

        Args:
            image: Image crop (H, W, C) in BGR format.

        Returns:
            Preprocessed tensor (C, H, W).
        """
        # Resize
        img = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).float()

    def crop_image(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.0,
    ) -> np.ndarray:
        """Crop image region with optional padding.

        Args:
            image: Full image (H, W, C).
            bbox: Bounding box [x1, y1, x2, y2].
            padding: Padding ratio to add around bbox.

        Returns:
            Cropped image region.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Add padding
        if padding > 0:
            pw = int((x2 - x1) * padding)
            ph = int((y2 - y1) * padding)
            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(w, x2 + pw)
            y2 = min(h, y2 + ph)

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            # Return empty crop
            return np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

        return image[y1:y2, x1:x2].copy()

    def extract(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract Re-ID features for detected objects.

        Args:
            image: Full image (H, W, C) in BGR format.
            bboxes: Detection bounding boxes [N, 4+].
            normalize: Whether to L2-normalize features.

        Returns:
            Feature matrix [N, feature_dim].
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim))

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Crop and preprocess all detections
        crops = []
        for bbox in bboxes:
            crop = self.crop_image(image, bbox)
            tensor = self.preprocess(crop)
            crops.append(tensor)

        # Stack into batch
        batch = torch.stack(crops).to(self.device)

        # Extract features in batches
        features_list = []

        with torch.no_grad():
            for i in range(0, len(batch), self.batch_size):
                batch_slice = batch[i : i + self.batch_size]
                batch_features = self.model(batch_slice)

                # Handle different output formats
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[0]

                features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0).numpy()

        # L2 normalize
        if normalize:
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norm + 1e-6)

        return features

    def compute_distance(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """Compute distance matrix between query and gallery features.

        Args:
            query_features: Query features [N, D].
            gallery_features: Gallery features [M, D].
            metric: Distance metric ('cosine' or 'euclidean').

        Returns:
            Distance matrix [N, M].
        """
        if metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(query_features, gallery_features.T)
            return 1 - similarity
        elif metric == "euclidean":
            # Euclidean distance
            diff = query_features[:, np.newaxis, :] - gallery_features[np.newaxis, :, :]
            return np.linalg.norm(diff, axis=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class OSNetExtractor(BaseReIDExtractor):
    """OSNet (Omni-Scale Network) Re-ID feature extractor.

    Paper: "Omni-Scale Feature Learning for Person Re-Identification"

    Supports multiple variants:
    - osnet_x1_0: Full model (2.2M params)
    - osnet_x0_75: 0.75x width (1.2M params)
    - osnet_x0_5: 0.5x width (0.6M params)
    - osnet_x0_25: 0.25x width (0.2M params)
    """

    VARIANTS = ["osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25"]
    FEATURE_DIMS = {
        "osnet_x1_0": 512,
        "osnet_x0_75": 512,
        "osnet_x0_5": 512,
        "osnet_x0_25": 512,
    }

    def __init__(
        self,
        variant: str = "osnet_x1_0",
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        """Initialize OSNet extractor.

        Args:
            variant: Model variant name.
            device: Device to run inference on.
            batch_size: Maximum batch size.
        """
        super().__init__(device, batch_size)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {self.VARIANTS}")

        self.variant = variant
        self._feature_dim = self.FEATURE_DIMS[variant]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load OSNet model.

        Args:
            weights_path: Path to weights file. If None, uses pretrained weights.
        """
        try:
            from torchreid import models

            self.model = models.build_model(
                name=self.variant,
                num_classes=1,  # Not used for feature extraction
                pretrained=weights_path is None,
            )

            if weights_path:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded OSNet model: {self.variant}")

        except ImportError:
            logger.warning("torchreid not installed, using placeholder model")
            self.model = self._create_placeholder_model()

    def _create_placeholder_model(self) -> nn.Module:
        """Create a simple placeholder model."""

        class PlaceholderReID(nn.Module):
            def __init__(self, feature_dim: int = 512):
                super().__init__()
                self.features = nn.Sequential(
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
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, feature_dim)

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = PlaceholderReID(self._feature_dim)
        model = model.to(self.device)
        model.eval()
        return model


class FastReIDExtractor(BaseReIDExtractor):
    """FastReID feature extractor.

    Paper: "FastReID: A Pytorch Toolbox for General Instance Re-identification"

    Supports various backbones:
    - ResNet50
    - ResNet101
    - OSNet
    - ResNeSt
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        """Initialize FastReID extractor.

        Args:
            config_path: Path to FastReID config file.
            device: Device to run inference on.
            batch_size: Maximum batch size.
        """
        super().__init__(device, batch_size)
        self.config_path = config_path
        self._feature_dim = 2048  # Default for ResNet50 backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load FastReID model.

        Args:
            weights_path: Path to weights file.
        """
        try:
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model
            from fastreid.utils.checkpoint import Checkpointer

            cfg = get_cfg()
            if self.config_path:
                cfg.merge_from_file(self.config_path)

            cfg.MODEL.BACKBONE.PRETRAIN = False
            self.model = build_model(cfg)

            if weights_path:
                Checkpointer(self.model).load(weights_path)

            self.model = self.model.to(self.device)
            self.model.eval()

            # Update feature dimension from config
            self._feature_dim = cfg.MODEL.HEADS.EMBEDDING_DIM

            logger.info("Loaded FastReID model")

        except ImportError:
            logger.warning("fastreid not installed, using OSNet placeholder")
            self.model = OSNetExtractor(device=self.device)._create_placeholder_model()


class SimpleReIDExtractor(BaseReIDExtractor):
    """Simple CNN-based Re-ID extractor.

    A lightweight feature extractor for basic use cases when
    full Re-ID models are not needed.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        """Initialize simple extractor.

        Args:
            feature_dim: Output feature dimension.
            device: Device to run inference on.
            batch_size: Maximum batch size.
        """
        super().__init__(device, batch_size, input_size=(128, 64))
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load or create simple Re-ID model."""

        class SimpleReIDNet(nn.Module):
            def __init__(self, feature_dim: int = 256):
                super().__init__()

                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )

                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, feature_dim)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        self.model = SimpleReIDNet(self._feature_dim)

        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded SimpleReID model with feature_dim={self._feature_dim}")


class ReIDExtractor:
    """Factory class for creating Re-ID extractors."""

    EXTRACTORS = {
        "osnet": OSNetExtractor,
        "fastreid": FastReIDExtractor,
        "simple": SimpleReIDExtractor,
    }

    @staticmethod
    def create(
        name: str = "osnet",
        device: str = "cuda:0",
        weights_path: Optional[str] = None,
        **kwargs,
    ) -> BaseReIDExtractor:
        """Create a Re-ID extractor.

        Args:
            name: Extractor name ('osnet', 'fastreid', 'simple').
            device: Device to run inference on.
            weights_path: Optional path to weights file.
            **kwargs: Additional arguments for the extractor.

        Returns:
            Initialized Re-ID extractor.
        """
        if name not in ReIDExtractor.EXTRACTORS:
            raise ValueError(
                f"Unknown extractor: {name}. Choose from {list(ReIDExtractor.EXTRACTORS.keys())}"
            )

        extractor_class = ReIDExtractor.EXTRACTORS[name]
        extractor = extractor_class(device=device, **kwargs)
        extractor.load_model(weights_path)

        return extractor

    @staticmethod
    def get_available() -> List[str]:
        """Get list of available extractors."""
        return list(ReIDExtractor.EXTRACTORS.keys())


# Convenience exports
__all__ = [
    "BaseReIDExtractor",
    "OSNetExtractor",
    "FastReIDExtractor",
    "SimpleReIDExtractor",
    "ReIDExtractor",
]
