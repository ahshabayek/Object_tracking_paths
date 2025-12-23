"""Unit tests for model weights management utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestWeightsManager:
    """Tests for WeightsManager class."""

    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        from cv_pipeline.utils.weights import WeightsManager

        manager = WeightsManager()
        assert manager.cache_dir.exists()

    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        from cv_pipeline.utils.weights import WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)
            assert manager.cache_dir == Path(tmpdir)

    def test_list_available(self):
        """Test listing available models."""
        from cv_pipeline.utils.weights import WeightsManager

        manager = WeightsManager()
        available = manager.list_available()

        assert isinstance(available, list)
        assert len(available) > 0
        assert "yolo11l" in available
        assert "osnet_x1_0" in available

    def test_list_cached_empty(self):
        """Test listing cached models when cache is empty."""
        from cv_pipeline.utils.weights import WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)
            cached = manager.list_cached()

            assert isinstance(cached, list)
            assert len(cached) == 0

    def test_get_model_info(self):
        """Test getting model information."""
        from cv_pipeline.utils.weights import WeightsManager

        manager = WeightsManager()
        info = manager.get_model_info("yolo11l")

        assert "filename" in info
        assert "framework" in info
        assert "size_mb" in info
        assert info["framework"] == "ultralytics"

    def test_get_model_info_invalid(self):
        """Test getting info for invalid model."""
        from cv_pipeline.utils.weights import WeightsManager

        manager = WeightsManager()

        with pytest.raises(ValueError, match="Unknown model"):
            manager.get_model_info("invalid_model")

    def test_get_weights_invalid_model(self):
        """Test getting weights for invalid model."""
        from cv_pipeline.utils.weights import WeightsManager

        manager = WeightsManager()

        with pytest.raises(ValueError, match="Unknown model"):
            manager.get_weights("nonexistent_model")

    def test_get_weights_manual_download_required(self):
        """Test that manual download models raise appropriate error."""
        from cv_pipeline.utils.weights import WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)

            # CLRNet requires manual download
            with pytest.raises(RuntimeError, match="manual download"):
                manager.get_weights("clrnet_culane_r18")

    def test_cache_size_empty(self):
        """Test cache size when empty."""
        from cv_pipeline.utils.weights import WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)
            size = manager.cache_size()

            assert size == 0.0

    def test_cache_size_with_files(self):
        """Test cache size with cached files."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY, WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)

            # Create a dummy cached file
            dummy_file = Path(tmpdir) / WEIGHT_REGISTRY["yolo11n"]["filename"]
            dummy_file.write_bytes(b"x" * 1024)  # 1KB file

            size = manager.cache_size()
            assert size > 0

    def test_clear_cache_specific(self):
        """Test clearing cache for specific model."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY, WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)

            # Create dummy cached files
            dummy_file1 = Path(tmpdir) / WEIGHT_REGISTRY["yolo11n"]["filename"]
            dummy_file2 = Path(tmpdir) / WEIGHT_REGISTRY["yolo11s"]["filename"]
            dummy_file1.write_bytes(b"x" * 100)
            dummy_file2.write_bytes(b"x" * 100)

            assert dummy_file1.exists()
            assert dummy_file2.exists()

            manager.clear_cache("yolo11n")

            assert not dummy_file1.exists()
            assert dummy_file2.exists()

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY, WeightsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WeightsManager(cache_dir=tmpdir)

            # Create dummy cached files
            dummy_file1 = Path(tmpdir) / WEIGHT_REGISTRY["yolo11n"]["filename"]
            dummy_file2 = Path(tmpdir) / WEIGHT_REGISTRY["yolo11s"]["filename"]
            dummy_file1.write_bytes(b"x" * 100)
            dummy_file2.write_bytes(b"x" * 100)

            manager.clear_cache()

            assert not dummy_file1.exists()
            assert not dummy_file2.exists()


class TestLoadWeights:
    """Tests for load_weights function."""

    def test_load_weights_state_dict(self):
        """Test loading weights from state dict."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.pt"

            # Save model state dict
            torch.save(model.state_dict(), weights_path)

            # Create new model and load weights
            new_model = SimpleModel()
            missing, unexpected = load_weights(new_model, weights_path)

            assert len(missing) == 0
            assert len(unexpected) == 0

    def test_load_weights_checkpoint_dict(self):
        """Test loading weights from checkpoint dict."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "checkpoint.pt"

            # Save as checkpoint dict
            checkpoint = {
                "model": model.state_dict(),
                "epoch": 10,
                "optimizer": {"state": {}},
            }
            torch.save(checkpoint, weights_path)

            # Create new model and load weights
            new_model = SimpleModel()
            missing, unexpected = load_weights(new_model, weights_path)

            assert len(missing) == 0
            assert len(unexpected) == 0

    def test_load_weights_with_state_dict_key(self):
        """Test loading weights with 'state_dict' key."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "checkpoint.pt"

            checkpoint = {"state_dict": model.state_dict()}
            torch.save(checkpoint, weights_path)

            new_model = SimpleModel()
            missing, unexpected = load_weights(new_model, weights_path)

            assert len(missing) == 0

    def test_load_weights_nonexistent_file(self):
        """Test loading from nonexistent file."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with pytest.raises(FileNotFoundError):
            load_weights(model, "/nonexistent/path/model.pt")

    def test_load_weights_with_module_prefix(self):
        """Test loading weights with 'module.' prefix (DataParallel)."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.pt"

            # Add module. prefix (simulating DataParallel)
            state_dict = {"module." + k: v for k, v in model.state_dict().items()}
            torch.save(state_dict, weights_path)

            new_model = SimpleModel()
            missing, unexpected = load_weights(new_model, weights_path)

            assert len(missing) == 0

    def test_load_weights_non_strict(self):
        """Test non-strict weight loading."""
        from cv_pipeline.utils.weights import load_weights

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.pt"

            # Save partial state dict
            partial_state = {"fc1.weight": model.fc1.weight, "fc1.bias": model.fc1.bias}
            torch.save(partial_state, weights_path)

            new_model = SimpleModel()
            missing, unexpected = load_weights(new_model, weights_path, strict=False)

            assert len(missing) == 2  # fc2.weight and fc2.bias
            assert len(unexpected) == 0


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        from cv_pipeline.utils.weights import save_checkpoint

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            save_checkpoint(model, path)

            assert path.exists()

            checkpoint = torch.load(path)
            assert "model" in checkpoint

    def test_save_checkpoint_with_optimizer(self):
        """Test saving checkpoint with optimizer."""
        from cv_pipeline.utils.weights import save_checkpoint

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            save_checkpoint(model, path, optimizer=optimizer)

            checkpoint = torch.load(path)
            assert "model" in checkpoint
            assert "optimizer" in checkpoint

    def test_save_checkpoint_with_metadata(self):
        """Test saving checkpoint with metadata."""
        from cv_pipeline.utils.weights import save_checkpoint

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            save_checkpoint(
                model,
                path,
                epoch=10,
                metrics={"accuracy": 0.95, "loss": 0.05},
                config={"learning_rate": 0.001},
            )

            checkpoint = torch.load(path)
            assert checkpoint["epoch"] == 10
            assert checkpoint["metrics"]["accuracy"] == 0.95
            assert checkpoint["config"]["learning_rate"] == 0.001

    def test_save_checkpoint_creates_directory(self):
        """Test that save_checkpoint creates parent directories."""
        from cv_pipeline.utils.weights import save_checkpoint

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "checkpoint.pt"

            save_checkpoint(model, path)

            assert path.exists()


class TestGetCheckpointInfo:
    """Tests for get_checkpoint_info function."""

    def test_get_checkpoint_info_basic(self):
        """Test getting checkpoint info."""
        from cv_pipeline.utils.weights import get_checkpoint_info

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            checkpoint = {
                "model": model.state_dict(),
                "epoch": 10,
            }
            torch.save(checkpoint, path)

            info = get_checkpoint_info(path)

            assert info["path"] == str(path)
            assert info["size_mb"] > 0
            assert info["type"] == "checkpoint_dict"
            assert info["epoch"] == 10

    def test_get_checkpoint_info_state_dict(self):
        """Test getting info for raw state dict."""
        from cv_pipeline.utils.weights import get_checkpoint_info

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), path)

            info = get_checkpoint_info(path)

            assert info["type"] == "checkpoint_dict"  # state_dict is also a dict

    def test_get_checkpoint_info_nonexistent(self):
        """Test getting info for nonexistent file."""
        from cv_pipeline.utils.weights import get_checkpoint_info

        with pytest.raises(FileNotFoundError):
            get_checkpoint_info("/nonexistent/path/model.pt")


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_get_model_size_basic(self):
        """Test getting model size."""
        from cv_pipeline.utils.weights import get_model_size

        model = SimpleModel()
        size_info = get_model_size(model)

        assert "total_params" in size_info
        assert "trainable_params" in size_info
        assert "param_size_mb" in size_info
        assert "total_size_mb" in size_info

        # SimpleModel has: fc1 (10*5 + 5) + fc2 (5*2 + 2) = 55 + 12 = 67 params
        assert size_info["total_params"] == 67
        assert size_info["trainable_params"] == 67

    def test_get_model_size_frozen_params(self):
        """Test model size with frozen parameters."""
        from cv_pipeline.utils.weights import get_model_size

        model = SimpleModel()

        # Freeze fc1
        for param in model.fc1.parameters():
            param.requires_grad = False

        size_info = get_model_size(model)

        assert size_info["total_params"] == 67
        assert size_info["trainable_params"] == 12  # Only fc2
        assert size_info["non_trainable_params"] == 55  # fc1


class TestGetPretrainedWeights:
    """Tests for get_pretrained_weights convenience function."""

    def test_get_pretrained_weights_cached(self):
        """Test getting cached pretrained weights."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY, get_pretrained_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy cached file
            dummy_file = Path(tmpdir) / WEIGHT_REGISTRY["yolo11n"]["filename"]
            dummy_file.write_bytes(b"x" * 1024)

            path = get_pretrained_weights("yolo11n", cache_dir=tmpdir)

            assert path.exists()
            assert path.name == "yolo11n.pt"

    def test_get_pretrained_weights_invalid(self):
        """Test getting weights for invalid model."""
        from cv_pipeline.utils.weights import get_pretrained_weights

        with pytest.raises(ValueError, match="Unknown model"):
            get_pretrained_weights("invalid_model_name")


class TestWeightRegistry:
    """Tests for the weight registry."""

    def test_registry_has_required_fields(self):
        """Test that all registry entries have required fields."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        required_fields = ["filename", "framework"]

        for model_name, info in WEIGHT_REGISTRY.items():
            for field in required_fields:
                assert field in info, f"{model_name} missing field: {field}"

    def test_registry_frameworks_valid(self):
        """Test that all frameworks are valid."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        valid_frameworks = ["ultralytics", "torchreid", "clrnet", "pytorch"]

        for model_name, info in WEIGHT_REGISTRY.items():
            assert info["framework"] in valid_frameworks, f"{model_name} has invalid framework"

    def test_registry_filenames_unique(self):
        """Test that all filenames are unique."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        filenames = [info["filename"] for info in WEIGHT_REGISTRY.values()]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames in registry"
