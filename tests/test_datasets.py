"""Unit tests for custom Kedro datasets."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestVideoDataSet:
    """Tests for VideoDataSet."""

    def test_describe(self):
        """Test dataset description."""
        from cv_pipeline.datasets import VideoDataSet

        dataset = VideoDataSet(
            filepath="test_video.mp4",
            load_args={"target_fps": 15, "resize": [640, 480]},
        )

        desc = dataset._describe()
        assert desc["filepath"] == "test_video.mp4"
        assert desc["load_args"]["target_fps"] == 15

    def test_exists_false(self):
        """Test exists returns False for non-existent file."""
        from cv_pipeline.datasets import VideoDataSet

        dataset = VideoDataSet(filepath="/nonexistent/video.mp4")
        assert not dataset._exists()

    def test_save_not_implemented(self):
        """Test that save raises NotImplementedError."""
        from cv_pipeline.datasets import VideoDataSet

        dataset = VideoDataSet(filepath="test.mp4")

        with pytest.raises(NotImplementedError):
            dataset._save([np.zeros((100, 100, 3))])


class TestTensorDataSet:
    """Tests for TensorDataSet."""

    def test_save_and_load(self):
        """Test saving and loading tensors."""
        from cv_pipeline.datasets import TensorDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tensor.pt")
            dataset = TensorDataSet(filepath=filepath)

            # Save tensor
            original = torch.randn(10, 20)
            dataset._save(original)

            assert dataset._exists()

            # Load tensor
            loaded = dataset._load()

            assert torch.allclose(original, loaded)

    def test_save_with_compression(self):
        """Test saving with compression."""
        from cv_pipeline.datasets import TensorDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tensor_compressed.pt")
            dataset = TensorDataSet(
                filepath=filepath,
                save_args={"compress": True},
            )

            original = torch.randn(100, 100)
            dataset._save(original)

            loaded = dataset._load()
            assert torch.allclose(original, loaded)

    def test_load_with_map_location(self):
        """Test loading with map_location."""
        from cv_pipeline.datasets import TensorDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tensor.pt")

            # Save
            dataset_save = TensorDataSet(filepath=filepath)
            original = torch.randn(5, 5)
            dataset_save._save(original)

            # Load with map_location
            dataset_load = TensorDataSet(
                filepath=filepath,
                load_args={"map_location": "cpu"},
            )
            loaded = dataset_load._load()

            assert loaded.device.type == "cpu"


class TestPyTorchModelDataSet:
    """Tests for PyTorchModelDataSet."""

    def test_save_full_model(self):
        """Test saving full model."""
        from cv_pipeline.datasets import PyTorchModelDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            dataset = PyTorchModelDataSet(filepath=filepath)

            # Create simple model
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2),
            )

            dataset._save(model)
            assert dataset._exists()

    def test_save_state_dict_only(self):
        """Test saving state_dict only."""
        from cv_pipeline.datasets import PyTorchModelDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model_state.pt")
            dataset = PyTorchModelDataSet(
                filepath=filepath,
                save_args={"save_state_dict_only": True},
            )

            model = nn.Linear(10, 5)
            dataset._save(model)

            # Load checkpoint
            checkpoint = torch.load(filepath)
            assert "state_dict" in checkpoint
            assert "model_class" in checkpoint

    def test_describe(self):
        """Test dataset description."""
        from cv_pipeline.datasets import PyTorchModelDataSet

        dataset = PyTorchModelDataSet(
            filepath="model.pt",
            load_args={"map_location": "cpu"},
            save_args={"save_state_dict_only": True},
        )

        desc = dataset._describe()
        assert desc["filepath"] == "model.pt"
        assert desc["load_args"]["map_location"] == "cpu"
        assert desc["save_args"]["save_state_dict_only"] is True


class TestVideoWriterDataSet:
    """Tests for VideoWriterDataSet."""

    def test_describe(self):
        """Test dataset description."""
        from cv_pipeline.datasets import VideoWriterDataSet

        dataset = VideoWriterDataSet(
            filepath="output.mp4",
            save_args={"fps": 30, "codec": "mp4v"},
        )

        desc = dataset._describe()
        assert desc["filepath"] == "output.mp4"
        assert desc["save_args"]["fps"] == 30

    def test_load_not_implemented(self):
        """Test that load raises NotImplementedError."""
        from cv_pipeline.datasets import VideoWriterDataSet

        dataset = VideoWriterDataSet(filepath="output.mp4")

        with pytest.raises(NotImplementedError):
            dataset._load()

    def test_save_empty_frames(self):
        """Test saving with empty frames list."""
        from cv_pipeline.datasets import VideoWriterDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.mp4")
            dataset = VideoWriterDataSet(filepath=filepath)

            # Should not raise, just log warning
            dataset._save([])

            # File should not be created
            assert not dataset._exists()

    def test_save_frames(self):
        """Test saving frames to video."""
        from cv_pipeline.datasets import VideoWriterDataSet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_output.mp4")
            dataset = VideoWriterDataSet(
                filepath=filepath,
                save_args={"fps": 10, "codec": "mp4v"},
            )

            # Create test frames
            frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]

            dataset._save(frames)

            assert dataset._exists()


class TestCameraStreamDataSet:
    """Tests for CameraStreamDataSet."""

    def test_describe(self):
        """Test dataset description."""
        from cv_pipeline.datasets import CameraStreamDataSet

        dataset = CameraStreamDataSet(
            load_args={"source": 0, "fps": 30, "duration": 5.0},
        )

        desc = dataset._describe()
        assert desc["source"] == 0
        assert desc["load_args"]["fps"] == 30

    def test_save_not_implemented(self):
        """Test that save raises NotImplementedError."""
        from cv_pipeline.datasets import CameraStreamDataSet

        dataset = CameraStreamDataSet()

        with pytest.raises(NotImplementedError):
            dataset._save(None)

    def test_exists_always_true(self):
        """Test that exists always returns True."""
        from cv_pipeline.datasets import CameraStreamDataSet

        dataset = CameraStreamDataSet()
        assert dataset._exists()

    def test_get_source_with_credentials(self):
        """Test RTSP source with credentials."""
        from cv_pipeline.datasets import CameraStreamDataSet

        dataset = CameraStreamDataSet(
            load_args={"source": "rtsp://192.168.1.1:554/stream"},
            credentials={"username": "admin", "password": "pass123"},
        )

        source = dataset._get_source()
        assert "admin:pass123@" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
