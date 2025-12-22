"""Custom Kedro Datasets for CV Pipeline.

This module provides custom dataset implementations for:
- Video file I/O (VideoDataSet, VideoWriterDataSet)
- PyTorch tensors and models (TensorDataSet, PyTorchModelDataSet)
- Camera streaming (CameraStreamDataSet)
"""

import logging
import queue
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str

logger = logging.getLogger(__name__)


class VideoDataSet(AbstractDataset[List[np.ndarray], List[np.ndarray]]):
    """Dataset for loading video files as a list of frames.

    Reads video files using OpenCV VideoCapture and returns frames as numpy arrays.

    Example catalog.yml entry:
        raw_video:
            type: cv_pipeline.datasets.VideoDataSet
            filepath: data/01_raw/video.mp4
            load_args:
                target_fps: 30
                resize: [1920, 1080]
                max_frames: 1000
    """

    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize VideoDataSet.

        Args:
            filepath: Path to the video file.
            load_args: Arguments for loading:
                - target_fps: Target frame rate (skip frames if needed)
                - resize: Tuple of (width, height) to resize frames
                - max_frames: Maximum number of frames to load
                - start_frame: Frame index to start from
                - end_frame: Frame index to end at
                - grayscale: Convert to grayscale if True
            save_args: Not used for this dataset.
            credentials: Not used for this dataset.
            fs_args: Not used for this dataset.
            metadata: Optional metadata dictionary.
        """
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}

    def _load(self) -> List[np.ndarray]:
        """Load video file and return list of frames.

        Returns:
            List of frames as numpy arrays (H, W, C) in BGR format.
        """
        filepath = get_filepath_str(self._filepath, "file")

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {filepath}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Loading video: {filepath}")
        logger.info(f"Video properties: {total_frames} frames, {video_fps} FPS, {width}x{height}")

        # Parse load args
        target_fps = self._load_args.get("target_fps", video_fps)
        resize = self._load_args.get("resize")
        max_frames = self._load_args.get("max_frames", total_frames)
        start_frame = self._load_args.get("start_frame", 0)
        end_frame = self._load_args.get("end_frame", total_frames)
        grayscale = self._load_args.get("grayscale", False)

        # Calculate frame skip for target FPS
        frame_skip = max(1, int(video_fps / target_fps)) if target_fps < video_fps else 1

        # Set start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame

        while len(frames) < max_frames and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply frame skip
            if (frame_idx - start_frame) % frame_skip != 0:
                frame_idx += 1
                continue

            # Resize if specified
            if resize is not None:
                frame = cv2.resize(frame, tuple(resize))

            # Convert to grayscale if requested
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame, axis=-1)

            frames.append(frame)
            frame_idx += 1

        cap.release()

        logger.info(f"Loaded {len(frames)} frames from video")

        return frames

    def _save(self, data: List[np.ndarray]) -> None:
        """Save is not implemented - use VideoWriterDataSet instead."""
        raise NotImplementedError(
            "VideoDataSet does not support saving. Use VideoWriterDataSet instead."
        )

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "filepath": str(self._filepath),
            "load_args": self._load_args,
        }

    def _exists(self) -> bool:
        """Check if the video file exists."""
        return self._filepath.exists()


class TensorDataSet(AbstractDataset[torch.Tensor, torch.Tensor]):
    """Dataset for saving and loading PyTorch tensors.

    Uses torch.save/torch.load for serialization.

    Example catalog.yml entry:
        detection_batch:
            type: cv_pipeline.datasets.TensorDataSet
            filepath: data/05_model_input/detection_batch.pt
            save_args:
                compress: true
    """

    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize TensorDataSet.

        Args:
            filepath: Path to the tensor file (.pt or .pth).
            load_args: Arguments for loading:
                - map_location: Device to map tensor to (e.g., 'cpu', 'cuda:0')
            save_args: Arguments for saving:
                - compress: Use compression (pickle protocol 4)
            credentials: Not used.
            fs_args: Not used.
            metadata: Optional metadata dictionary.
        """
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}

    def _load(self) -> torch.Tensor:
        """Load tensor from file.

        Returns:
            Loaded PyTorch tensor.
        """
        filepath = get_filepath_str(self._filepath, "file")

        map_location = self._load_args.get("map_location", None)

        tensor = torch.load(filepath, map_location=map_location, weights_only=True)

        logger.info(f"Loaded tensor from {filepath}, shape: {tensor.shape}, dtype: {tensor.dtype}")

        return tensor

    def _save(self, data: torch.Tensor) -> None:
        """Save tensor to file.

        Args:
            data: PyTorch tensor to save.
        """
        filepath = get_filepath_str(self._filepath, "file")

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        compress = self._save_args.get("compress", False)

        if compress:
            torch.save(data, filepath, pickle_protocol=4)
        else:
            torch.save(data, filepath)

        logger.info(f"Saved tensor to {filepath}, shape: {data.shape}, dtype: {data.dtype}")

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "filepath": str(self._filepath),
            "load_args": self._load_args,
            "save_args": self._save_args,
        }

    def _exists(self) -> bool:
        """Check if the tensor file exists."""
        return self._filepath.exists()


class PyTorchModelDataSet(AbstractDataset[nn.Module, nn.Module]):
    """Dataset for saving and loading PyTorch models.

    Supports saving/loading full models or state_dict only.

    Example catalog.yml entry:
        detection_model:
            type: cv_pipeline.datasets.PyTorchModelDataSet
            filepath: data/06_models/detection_model.pt
            versioned: true
            save_args:
                map_location: cpu
                save_state_dict_only: true
    """

    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[Any] = None,
    ):
        """Initialize PyTorchModelDataSet.

        Args:
            filepath: Path to the model file (.pt or .pth).
            load_args: Arguments for loading:
                - map_location: Device to map model to (e.g., 'cpu', 'cuda:0')
                - model_class: Class to instantiate for state_dict loading
                - model_args: Arguments to pass to model_class constructor
            save_args: Arguments for saving:
                - save_state_dict_only: If True, save only state_dict
                - map_location: Move model to device before saving
            credentials: Not used.
            fs_args: Not used.
            metadata: Optional metadata dictionary.
            version: Version for versioned datasets.
        """
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}
        self._version = version

    def _load(self) -> nn.Module:
        """Load PyTorch model from file.

        Returns:
            Loaded PyTorch model.
        """
        filepath = get_filepath_str(self._filepath, "file")

        map_location = self._load_args.get("map_location", None)
        model_class = self._load_args.get("model_class")
        model_args = self._load_args.get("model_args", {})

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, nn.Module):
            # Full model saved
            model = checkpoint
            logger.info(f"Loaded full model from {filepath}")
        elif isinstance(checkpoint, dict):
            # State dict or checkpoint dict
            if model_class is None:
                raise ValueError(
                    "model_class must be specified in load_args when loading state_dict"
                )

            # Instantiate model
            if isinstance(model_class, str):
                # Import class from string
                module_path, class_name = model_class.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)

            model = model_class(**model_args)

            # Load state dict
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            logger.info(f"Loaded model state_dict from {filepath}")
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

        return model

    def _save(self, data: nn.Module) -> None:
        """Save PyTorch model to file.

        Args:
            data: PyTorch model to save.
        """
        filepath = get_filepath_str(self._filepath, "file")

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_state_dict_only = self._save_args.get("save_state_dict_only", False)
        map_location = self._save_args.get("map_location")

        # Move to specified device before saving
        if map_location:
            data = data.to(map_location)

        if save_state_dict_only:
            checkpoint = {
                "state_dict": data.state_dict(),
                "model_class": data.__class__.__name__,
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Saved model state_dict to {filepath}")
        else:
            torch.save(data, filepath)
            logger.info(f"Saved full model to {filepath}")

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "filepath": str(self._filepath),
            "load_args": self._load_args,
            "save_args": self._save_args,
        }

    def _exists(self) -> bool:
        """Check if the model file exists."""
        return self._filepath.exists()


class CameraStreamDataSet(AbstractDataset[List[np.ndarray], None]):
    """Dataset for reading from camera streams (webcam, RTSP, etc.).

    Provides buffered reading from camera sources with optional preprocessing.

    Example catalog.yml entry:
        raw_camera_stream:
            type: cv_pipeline.datasets.CameraStreamDataSet
            load_args:
                source: 0  # Camera index or RTSP URL
                buffer_size: 10
                fps: 30
                duration: 10.0  # seconds
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CameraStreamDataSet.

        Args:
            filepath: Not used (source specified in load_args).
            load_args: Arguments for loading:
                - source: Camera index (int) or RTSP URL (str)
                - buffer_size: Frame buffer size for async reading
                - fps: Target frame rate
                - duration: Recording duration in seconds
                - max_frames: Maximum number of frames to capture
                - resize: Tuple of (width, height) to resize frames
                - grayscale: Convert to grayscale if True
            save_args: Not used.
            credentials: Optional credentials for RTSP streams.
            fs_args: Not used.
            metadata: Optional metadata dictionary.
        """
        self._filepath = filepath
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._credentials = credentials or {}
        self._metadata = metadata or {}

        self._capture = None
        self._buffer: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._capture_thread = None

    def _get_source(self) -> Union[int, str]:
        """Get the camera source from load_args."""
        source = self._load_args.get("source", 0)

        # Handle RTSP with credentials
        if isinstance(source, str) and source.startswith("rtsp://"):
            username = self._credentials.get("username")
            password = self._credentials.get("password")
            if username and password:
                # Insert credentials into URL
                source = source.replace("rtsp://", f"rtsp://{username}:{password}@")

        return source

    def _capture_frames(self, cap: cv2.VideoCapture, target_fps: float) -> None:
        """Background thread for capturing frames."""
        buffer_size = self._load_args.get("buffer_size", 10)
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        last_capture_time = 0

        while not self._stop_event.is_set():
            current_time = time.time()

            # Maintain target FPS
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            last_capture_time = current_time

            # Drop oldest frame if buffer is full
            if self._buffer.qsize() >= buffer_size:
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    pass

            self._buffer.put(frame)

    def _load(self) -> List[np.ndarray]:
        """Load frames from camera stream.

        Returns:
            List of frames as numpy arrays (H, W, C) in BGR format.
        """
        source = self._get_source()

        # Open capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

        # Get/set properties
        target_fps = self._load_args.get("fps", 30)
        duration = self._load_args.get("duration")
        max_frames = self._load_args.get("max_frames")
        resize = self._load_args.get("resize")
        grayscale = self._load_args.get("grayscale", False)
        use_threading = self._load_args.get("threaded", False)

        logger.info(f"Opening camera stream: {source}")

        frames = []
        start_time = time.time()

        if use_threading:
            # Use background thread for buffered capture
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_frames, args=(cap, target_fps)
            )
            self._capture_thread.start()

            while True:
                # Check termination conditions
                if duration and (time.time() - start_time) >= duration:
                    break
                if max_frames and len(frames) >= max_frames:
                    break

                try:
                    frame = self._buffer.get(timeout=1.0)

                    # Apply preprocessing
                    if resize:
                        frame = cv2.resize(frame, tuple(resize))
                    if grayscale:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.expand_dims(frame, axis=-1)

                    frames.append(frame)
                except queue.Empty:
                    continue

            self._stop_event.set()
            self._capture_thread.join()
        else:
            # Simple synchronous capture
            frame_interval = 1.0 / target_fps if target_fps > 0 else 0
            last_capture_time = 0

            while True:
                current_time = time.time()

                # Check termination conditions
                if duration and (current_time - start_time) >= duration:
                    break
                if max_frames and len(frames) >= max_frames:
                    break

                # Maintain target FPS
                if current_time - last_capture_time < frame_interval:
                    time.sleep(0.001)
                    continue

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                last_capture_time = current_time

                # Apply preprocessing
                if resize:
                    frame = cv2.resize(frame, tuple(resize))
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)

                frames.append(frame)

        cap.release()

        logger.info(f"Captured {len(frames)} frames from camera")

        return frames

    def _save(self, data: Any) -> None:
        """Save is not supported for camera streams."""
        raise NotImplementedError("CameraStreamDataSet does not support saving.")

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "source": self._load_args.get("source", 0),
            "load_args": self._load_args,
        }

    def _exists(self) -> bool:
        """Camera streams are always considered to exist."""
        return True


class VideoWriterDataSet(AbstractDataset[None, List[np.ndarray]]):
    """Dataset for writing frames to a video file.

    Uses OpenCV VideoWriter for encoding.

    Example catalog.yml entry:
        annotated_video:
            type: cv_pipeline.datasets.VideoWriterDataSet
            filepath: data/08_reporting/annotated_output.mp4
            save_args:
                fps: 30
                codec: mp4v
                resize: [1920, 1080]
    """

    # Common codec mappings
    CODEC_MAP = {
        "mp4v": cv2.VideoWriter_fourcc(*"mp4v"),
        "avc1": cv2.VideoWriter_fourcc(*"avc1"),
        "h264": cv2.VideoWriter_fourcc(*"H264"),
        "xvid": cv2.VideoWriter_fourcc(*"XVID"),
        "mjpg": cv2.VideoWriter_fourcc(*"MJPG"),
        "divx": cv2.VideoWriter_fourcc(*"DIVX"),
    }

    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize VideoWriterDataSet.

        Args:
            filepath: Path to the output video file.
            load_args: Not used.
            save_args: Arguments for saving:
                - fps: Frame rate (default: 30)
                - codec: Codec name (default: 'mp4v')
                - resize: Tuple of (width, height) to resize frames
                - quality: Quality setting (0-100, codec-dependent)
            credentials: Not used.
            fs_args: Not used.
            metadata: Optional metadata dictionary.
        """
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}

    def _load(self) -> None:
        """Load is not supported for VideoWriterDataSet."""
        raise NotImplementedError(
            "VideoWriterDataSet does not support loading. Use VideoDataSet instead."
        )

    def _save(self, data: List[np.ndarray]) -> None:
        """Save frames to video file.

        Args:
            data: List of frames as numpy arrays (H, W, C) in BGR format.
        """
        if not data:
            logger.warning("No frames to write to video")
            return

        filepath = get_filepath_str(self._filepath, "file")

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Get save args
        fps = self._save_args.get("fps", 30)
        codec = self._save_args.get("codec", "mp4v")
        resize = self._save_args.get("resize")

        # Get frame dimensions
        first_frame = data[0]
        if resize:
            width, height = resize
        else:
            height, width = first_frame.shape[:2]

        # Get fourcc code
        if isinstance(codec, str):
            fourcc = self.CODEC_MAP.get(codec.lower(), cv2.VideoWriter_fourcc(*codec))
        else:
            fourcc = codec

        # Determine if frames are color or grayscale
        is_color = len(first_frame.shape) == 3 and first_frame.shape[2] == 3

        # Create video writer
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height), is_color)

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {filepath}")

        logger.info(f"Writing {len(data)} frames to {filepath}")

        for frame in data:
            # Resize if needed
            if resize:
                frame = cv2.resize(frame, (width, height))

            # Convert grayscale to BGR if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            writer.write(frame)

        writer.release()

        logger.info(f"Video saved to {filepath}")

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "filepath": str(self._filepath),
            "save_args": self._save_args,
        }

    def _exists(self) -> bool:
        """Check if the video file exists."""
        return self._filepath.exists()


# Export all dataset classes
__all__ = [
    "VideoDataSet",
    "TensorDataSet",
    "PyTorchModelDataSet",
    "CameraStreamDataSet",
    "VideoWriterDataSet",
]
