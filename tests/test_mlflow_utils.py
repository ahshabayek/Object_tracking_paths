"""Unit tests for MLFlow utilities."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check if mlflow is available
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="mlflow is not installed")


class TestMLFlowAvailability:
    """Tests for MLFlow availability check."""

    def test_is_mlflow_available(self):
        """Test MLFlow availability check."""
        from cv_pipeline.utils.mlflow_utils import is_mlflow_available

        # Should return True when mlflow is installed
        assert is_mlflow_available() == MLFLOW_AVAILABLE


class TestFlattenDict:
    """Tests for dictionary flattening utility."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple dictionary."""
        from cv_pipeline.utils.mlflow_utils import _flatten_dict

        d = {"a": 1, "b": 2}
        result = _flatten_dict(d)

        assert result == {"a": 1, "b": 2}

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        from cv_pipeline.utils.mlflow_utils import _flatten_dict

        d = {"level1": {"level2a": 1, "level2b": {"level3": 2}}, "top": 3}
        result = _flatten_dict(d)

        assert result["level1_level2a"] == 1
        assert result["level1_level2b_level3"] == 2
        assert result["top"] == 3

    def test_flatten_with_prefix(self):
        """Test flattening with a prefix."""
        from cv_pipeline.utils.mlflow_utils import _flatten_dict

        d = {"a": 1, "b": 2}
        result = _flatten_dict(d, prefix="test_")

        assert result == {"test_a": 1, "test_b": 2}


class TestLogParamsSafe:
    """Tests for safe parameter logging."""

    def test_log_params_simple(self):
        """Test logging simple parameters."""
        from cv_pipeline.utils.mlflow_utils import log_params_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()

            params = {"learning_rate": 0.001, "batch_size": 32}
            log_params_safe(params)

            assert mock_mlflow.log_param.call_count == 2

    def test_log_params_with_prefix(self):
        """Test logging parameters with prefix."""
        from cv_pipeline.utils.mlflow_utils import log_params_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()

            params = {"model": "yolo"}
            log_params_safe(params, prefix="detection_")

            mock_mlflow.log_param.assert_called_once()
            call_args = mock_mlflow.log_param.call_args
            assert "detection_model" in str(call_args)

    def test_log_params_long_value_truncated(self):
        """Test that long parameter values are truncated."""
        from cv_pipeline.utils.mlflow_utils import log_params_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()

            # Create a very long string value
            long_value = "x" * 600
            params = {"long_param": long_value}
            log_params_safe(params)

            # Should have been called with truncated value
            call_args = mock_mlflow.log_param.call_args[0]
            assert len(call_args[1]) <= 500


class TestLogMetricsSafe:
    """Tests for safe metrics logging."""

    def test_log_metrics_simple(self):
        """Test logging simple metrics."""
        from cv_pipeline.utils.mlflow_utils import log_metrics_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()

            metrics = {"accuracy": 0.95, "loss": 0.05}
            log_metrics_safe(metrics)

            assert mock_mlflow.log_metric.call_count == 2

    def test_log_metrics_with_step(self):
        """Test logging metrics with step."""
        from cv_pipeline.utils.mlflow_utils import log_metrics_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()

            metrics = {"accuracy": 0.95}
            log_metrics_safe(metrics, step=10)

            mock_mlflow.log_metric.assert_called_once_with("accuracy", 0.95, step=10)

    def test_log_metrics_skips_nan(self):
        """Test that NaN values are skipped."""
        from cv_pipeline.utils.mlflow_utils import log_metrics_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()

            metrics = {"valid": 0.5, "nan_value": float("nan")}
            log_metrics_safe(metrics)

            # Should only log the valid metric
            assert mock_mlflow.log_metric.call_count == 1

    def test_log_metrics_skips_inf(self):
        """Test that Inf values are skipped."""
        from cv_pipeline.utils.mlflow_utils import log_metrics_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()

            metrics = {"valid": 0.5, "inf_value": float("inf")}
            log_metrics_safe(metrics)

            # Should only log the valid metric
            assert mock_mlflow.log_metric.call_count == 1

    def test_log_metrics_with_prefix(self):
        """Test logging metrics with prefix."""
        from cv_pipeline.utils.mlflow_utils import log_metrics_safe

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()

            metrics = {"mAP": 0.85}
            log_metrics_safe(metrics, prefix="det_")

            mock_mlflow.log_metric.assert_called_once_with("det_mAP", 0.85, step=None)


class TestLogDictAsArtifact:
    """Tests for logging dictionaries as artifacts."""

    def test_log_dict_as_artifact(self):
        """Test logging a dictionary as JSON artifact."""
        from cv_pipeline.utils.mlflow_utils import log_dict_as_artifact

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_artifact = MagicMock()

            data = {"key1": "value1", "key2": 123}
            log_dict_as_artifact(data, "test.json")

            # Should have been called once
            assert mock_mlflow.log_artifact.call_count == 1


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""

    def test_tracker_init(self):
        """Test ExperimentTracker initialization."""
        from cv_pipeline.utils.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_name="test_exp", run_name="test_run", tags={"version": "1.0"}
        )

        assert tracker.experiment_name == "test_exp"
        assert tracker.run_name == "test_run"
        assert tracker.tags == {"version": "1.0"}
        assert tracker._step == 0

    def test_tracker_step_counter(self):
        """Test step counter incrementing."""
        from cv_pipeline.utils.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker()

        assert tracker.step() == 1
        assert tracker.step() == 2
        assert tracker.step() == 3

    def test_tracker_context_manager(self):
        """Test tracker as context manager."""
        from cv_pipeline.utils.mlflow_utils import ExperimentTracker

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
            mock_mlflow.create_experiment = MagicMock(return_value="123")
            mock_mlflow.set_experiment = MagicMock()
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run = MagicMock(return_value=mock_run)
            mock_mlflow.end_run = MagicMock()
            mock_mlflow.set_tags = MagicMock()

            with ExperimentTracker("test_exp", "test_run") as tracker:
                assert tracker.run is not None

            mock_mlflow.end_run.assert_called_once()

    def test_tracker_log_methods(self):
        """Test tracker logging methods."""
        from cv_pipeline.utils.mlflow_utils import ExperimentTracker

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()
            mock_mlflow.set_tag = MagicMock()

            tracker = ExperimentTracker()

            tracker.log_param("key", "value")
            tracker.log_metric("accuracy", 0.95)
            tracker.set_tag("tag_key", "tag_value")

            assert mock_mlflow.log_param.called
            assert mock_mlflow.log_metric.called
            assert mock_mlflow.set_tag.called


class TestMLFlowRun:
    """Tests for mlflow_run context manager."""

    def test_mlflow_run_context(self):
        """Test mlflow_run context manager."""
        from cv_pipeline.utils.mlflow_utils import mlflow_run

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
            mock_mlflow.create_experiment = MagicMock(return_value="123")
            mock_mlflow.set_experiment = MagicMock()

            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run = MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=mock_run),
                    __exit__=MagicMock(return_value=False),
                )
            )
            mock_mlflow.set_tags = MagicMock()

            with mlflow_run("test_exp", run_name="test", tags={"v": "1"}) as run:
                pass


class TestMLFlowTrackDecorator:
    """Tests for mlflow_track decorator."""

    def test_mlflow_track_decorator(self):
        """Test mlflow_track decorator."""
        from cv_pipeline.utils.mlflow_utils import mlflow_track

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
            mock_mlflow.create_experiment = MagicMock(return_value="123")
            mock_mlflow.set_experiment = MagicMock()
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()

            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run = MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=mock_run),
                    __exit__=MagicMock(return_value=False),
                )
            )
            mock_mlflow.set_tags = MagicMock()

            @mlflow_track(experiment_name="test")
            def my_func(learning_rate: float = 0.01) -> dict:
                return {"accuracy": 0.95}

            result = my_func(learning_rate=0.001)

            assert result == {"accuracy": 0.95}


class TestPipelineSpecificLogging:
    """Tests for pipeline-specific logging functions."""

    def test_log_detection_metrics(self):
        """Test detection metrics logging."""
        from cv_pipeline.utils.mlflow_utils import log_detection_metrics

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()

            metrics = {"mAP": 0.85, "precision": 0.90}
            config = {"model": "yolo", "variant": "v11"}

            log_detection_metrics(metrics, config)

            assert mock_mlflow.log_param.called
            assert mock_mlflow.log_metric.called

    def test_log_tracking_metrics(self):
        """Test tracking metrics logging."""
        from cv_pipeline.utils.mlflow_utils import log_tracking_metrics

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()

            metrics = {"MOTA": 0.75, "IDF1": 0.80}
            config = {"tracker": "bytetrack", "track_buffer": 30}

            log_tracking_metrics(metrics, config)

            assert mock_mlflow.log_param.called
            assert mock_mlflow.log_metric.called

    def test_log_lane_detection_metrics(self):
        """Test lane detection metrics logging."""
        from cv_pipeline.utils.mlflow_utils import log_lane_detection_metrics

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()

            metrics = {"accuracy": 0.95, "f1": 0.92}
            config = {"model": "clrnet", "backbone": "resnet34"}

            log_lane_detection_metrics(metrics, config)

            assert mock_mlflow.log_param.called
            assert mock_mlflow.log_metric.called

    def test_log_pipeline_run(self):
        """Test pipeline run logging."""
        from cv_pipeline.utils.mlflow_utils import log_pipeline_run

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.set_tag = MagicMock()
            mock_mlflow.log_metric = MagicMock()
            mock_mlflow.log_param = MagicMock()

            node_outputs = {"detection": {"mAP": 0.85, "count": 100}, "tracking": {"MOTA": 0.75}}

            log_pipeline_run(
                pipeline_name="inference",
                node_outputs=node_outputs,
                execution_time=10.5,
                params={"batch_size": 16},
            )

            mock_mlflow.set_tag.assert_called_with("pipeline", "inference")
            # Should log execution time and metrics from node outputs
            assert mock_mlflow.log_metric.call_count >= 1


class TestMLFlowIntegration:
    """Integration tests for MLFlow utilities."""

    def test_full_tracking_workflow(self):
        """Test a complete tracking workflow."""
        from cv_pipeline.utils.mlflow_utils import ExperimentTracker

        with patch("cv_pipeline.utils.mlflow_utils.mlflow") as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
            mock_mlflow.create_experiment = MagicMock(return_value="123")
            mock_mlflow.set_experiment = MagicMock()
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()
            mock_mlflow.set_tag = MagicMock()
            mock_mlflow.set_tags = MagicMock()

            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run = MagicMock(return_value=mock_run)
            mock_mlflow.end_run = MagicMock()

            # Run a complete workflow
            tracker = ExperimentTracker("cv_pipeline", "detection_test")
            tracker.start()

            # Log model config
            tracker.log_params({"model": "rf_detr", "variant": "base", "input_size": 640})

            # Simulate training loop
            for epoch in range(3):
                step = tracker.step()
                tracker.log_metrics(
                    {"train_loss": 0.5 - epoch * 0.1, "val_mAP": 0.7 + epoch * 0.05}, step=step
                )

            # Log final results
            tracker.set_tag("status", "completed")
            tracker.log_metric("final_mAP", 0.85)

            tracker.end()

            # Verify calls were made
            assert mock_mlflow.start_run.called
            assert mock_mlflow.log_param.called
            assert mock_mlflow.log_metric.called
            assert mock_mlflow.end_run.called


class TestNoMLFlowFallback:
    """Tests for behavior when MLFlow is not available."""

    def test_functions_handle_no_mlflow(self):
        """Test that functions gracefully handle missing MLFlow."""
        from cv_pipeline.utils.mlflow_utils import (
            log_dict_as_artifact,
            log_metrics_safe,
            log_params_safe,
        )

        # Temporarily set MLFLOW_AVAILABLE to False
        with patch("cv_pipeline.utils.mlflow_utils.MLFLOW_AVAILABLE", False):
            # These should not raise exceptions
            log_params_safe({"key": "value"})
            log_metrics_safe({"metric": 0.5})
            log_dict_as_artifact({"data": "test"}, "test.json")
