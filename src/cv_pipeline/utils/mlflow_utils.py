"""MLFlow Utilities for CV Pipeline.

This module provides utilities for experiment tracking with MLFlow,
including context managers, decorators, and helper functions for
logging metrics, parameters, artifacts, and models.
"""

import functools
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import mlflow, provide stubs if not available
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None
    logger.warning("MLFlow not installed. Logging will be disabled.")


def is_mlflow_available() -> bool:
    """Check if MLFlow is available."""
    return MLFLOW_AVAILABLE


def get_or_create_experiment(experiment_name: str) -> Optional[str]:
    """Get or create an MLFlow experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Experiment ID or None if MLFlow is not available.
    """
    if not MLFLOW_AVAILABLE:
        return None

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


@contextmanager
def mlflow_run(
    experiment_name: str = "cv_pipeline",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
):
    """Context manager for MLFlow runs.

    Args:
        experiment_name: Name of the experiment.
        run_name: Optional name for this run.
        tags: Optional tags to add to the run.
        nested: Whether this is a nested run.

    Yields:
        Active MLFlow run or None if MLFlow is not available.

    Example:
        >>> with mlflow_run("my_experiment", run_name="test_run") as run:
        ...     mlflow.log_param("learning_rate", 0.001)
        ...     mlflow.log_metric("accuracy", 0.95)
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLFlow not available, skipping run context")
        yield None
        return

    # Set experiment
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Start run
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"Started MLFlow run: {run.info.run_id}")
        yield run
        logger.info(f"Finished MLFlow run: {run.info.run_id}")


def log_params_safe(params: Dict[str, Any], prefix: str = "") -> None:
    """Safely log parameters to MLFlow.

    Handles nested dictionaries, non-string values, and MLFlow limitations.

    Args:
        params: Dictionary of parameters to log.
        prefix: Optional prefix for parameter names.
    """
    if not MLFLOW_AVAILABLE:
        return

    flat_params = _flatten_dict(params, prefix)

    for key, value in flat_params.items():
        try:
            # MLFlow has a 500 character limit for param values
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            mlflow.log_param(key, str_value)
        except Exception as e:
            logger.warning(f"Failed to log param {key}: {e}")


def log_metrics_safe(
    metrics: Dict[str, Union[int, float]],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Safely log metrics to MLFlow.

    Args:
        metrics: Dictionary of metrics to log.
        step: Optional step number for the metrics.
        prefix: Optional prefix for metric names.
    """
    if not MLFLOW_AVAILABLE:
        return

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
            metric_name = f"{prefix}{key}" if prefix else key
            try:
                mlflow.log_metric(metric_name, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {metric_name}: {e}")


def log_dict_as_artifact(
    data: Dict[str, Any],
    filename: str,
    artifact_path: Optional[str] = None,
) -> None:
    """Log a dictionary as a JSON artifact.

    Args:
        data: Dictionary to save.
        filename: Name of the artifact file.
        artifact_path: Optional subdirectory in artifacts.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            mlflow.log_artifact(filepath, artifact_path)
            logger.info(f"Logged artifact: {filename}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {filename}: {e}")


def log_figure(
    figure,
    filename: str,
    artifact_path: Optional[str] = None,
) -> None:
    """Log a matplotlib figure as an artifact.

    Args:
        figure: Matplotlib figure object.
        filename: Name of the artifact file.
        artifact_path: Optional subdirectory in artifacts.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            figure.savefig(filepath, bbox_inches="tight", dpi=150)
            mlflow.log_artifact(filepath, artifact_path)
            logger.info(f"Logged figure: {filename}")
    except Exception as e:
        logger.warning(f"Failed to log figure {filename}: {e}")


def log_image_artifact(
    image: np.ndarray,
    filename: str,
    artifact_path: Optional[str] = None,
) -> None:
    """Log a numpy image as an artifact.

    Args:
        image: Image as numpy array (BGR or RGB).
        filename: Name of the artifact file.
        artifact_path: Optional subdirectory in artifacts.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            cv2.imwrite(filepath, image)
            mlflow.log_artifact(filepath, artifact_path)
            logger.info(f"Logged image: {filename}")
    except Exception as e:
        logger.warning(f"Failed to log image {filename}: {e}")


def log_model_info(
    model_name: str,
    model_config: Dict[str, Any],
    model_summary: Optional[str] = None,
) -> None:
    """Log model information to MLFlow.

    Args:
        model_name: Name of the model.
        model_config: Model configuration dictionary.
        model_summary: Optional model architecture summary.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_tag("model_name", model_name)
        log_params_safe(model_config, prefix="model_")

        if model_summary:
            log_dict_as_artifact(
                {"model_name": model_name, "summary": model_summary},
                "model_summary.json",
                "model_info",
            )
    except Exception as e:
        logger.warning(f"Failed to log model info: {e}")


def log_pytorch_model(
    model,
    artifact_path: str = "model",
    conda_env: Optional[Dict] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    """Log a PyTorch model to MLFlow.

    Args:
        model: PyTorch model to log.
        artifact_path: Path within the run's artifact URI.
        conda_env: Conda environment specification.
        registered_model_name: Name to register the model under.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        import mlflow.pytorch

        mlflow.pytorch.log_model(
            model,
            artifact_path,
            conda_env=conda_env,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Logged PyTorch model to {artifact_path}")
    except Exception as e:
        logger.warning(f"Failed to log PyTorch model: {e}")


def mlflow_track(
    experiment_name: str = "cv_pipeline",
    run_name: Optional[str] = None,
    log_params: bool = True,
    log_results: bool = True,
):
    """Decorator for tracking function execution with MLFlow.

    Args:
        experiment_name: Name of the experiment.
        run_name: Optional name for the run.
        log_params: Whether to log function arguments as params.
        log_results: Whether to log return values as metrics.

    Returns:
        Decorated function.

    Example:
        >>> @mlflow_track(experiment_name="detection")
        ... def run_detection(model_name: str, threshold: float) -> Dict[str, float]:
        ...     # ... detection logic ...
        ...     return {"mAP": 0.85, "precision": 0.90}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not MLFLOW_AVAILABLE:
                return func(*args, **kwargs)

            with mlflow_run(experiment_name, run_name or func.__name__) as run:
                # Log parameters from kwargs
                if log_params and kwargs:
                    log_params_safe(kwargs, prefix="arg_")

                # Execute function
                result = func(*args, **kwargs)

                # Log results if they're metrics
                if log_results and isinstance(result, dict):
                    log_metrics_safe(result)

                return result

        return wrapper

    return decorator


class ExperimentTracker:
    """Class-based experiment tracker for more complex workflows.

    Example:
        >>> tracker = ExperimentTracker("cv_pipeline", "detection_run")
        >>> tracker.start()
        >>> tracker.log_param("model", "yolo")
        >>> tracker.log_metric("mAP", 0.85)
        >>> tracker.end()
    """

    def __init__(
        self,
        experiment_name: str = "cv_pipeline",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment.
            run_name: Optional name for the run.
            tags: Optional tags for the run.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None
        self._step = 0

    def start(self) -> Optional[str]:
        """Start the MLFlow run.

        Returns:
            Run ID or None if MLFlow is not available.
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLFlow not available")
            return None

        get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

        if self.tags:
            mlflow.set_tags(self.tags)

        logger.info(f"Started tracking run: {self.run.info.run_id}")
        return self.run.info.run_id

    def end(self) -> None:
        """End the MLFlow run."""
        if self.run and MLFLOW_AVAILABLE:
            mlflow.end_run()
            logger.info(f"Ended tracking run: {self.run.info.run_id}")
            self.run = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        log_params_safe({key: value})

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        log_params_safe(params)

    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None) -> None:
        """Log a single metric."""
        log_metrics_safe({key: value}, step=step)

    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        log_metrics_safe(metrics, step=step)

    def step(self) -> int:
        """Increment and return the step counter."""
        self._step += 1
        return self._step

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact from a local path."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact: {e}")

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log a dictionary as a JSON artifact."""
        log_dict_as_artifact(data, filename)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to set tag: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end()
        return False


# Pipeline-specific logging functions


def log_detection_metrics(
    metrics: Dict[str, float],
    model_config: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log object detection metrics and configuration.

    Args:
        metrics: Detection metrics (mAP, precision, recall, etc.).
        model_config: Model configuration parameters.
        step: Optional step number.
    """
    log_params_safe(model_config, prefix="detection_")
    log_metrics_safe(metrics, step=step, prefix="det_")


def log_tracking_metrics(
    metrics: Dict[str, float],
    tracker_config: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log tracking metrics and configuration.

    Args:
        metrics: Tracking metrics (MOTA, IDF1, etc.).
        tracker_config: Tracker configuration parameters.
        step: Optional step number.
    """
    log_params_safe(tracker_config, prefix="tracking_")
    log_metrics_safe(metrics, step=step, prefix="track_")


def log_lane_detection_metrics(
    metrics: Dict[str, float],
    model_config: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log lane detection metrics and configuration.

    Args:
        metrics: Lane detection metrics (accuracy, F1, etc.).
        model_config: Model configuration parameters.
        step: Optional step number.
    """
    log_params_safe(model_config, prefix="lane_")
    log_metrics_safe(metrics, step=step, prefix="lane_")


def log_pipeline_run(
    pipeline_name: str,
    node_outputs: Dict[str, Any],
    execution_time: float,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a complete pipeline run.

    Args:
        pipeline_name: Name of the pipeline.
        node_outputs: Dictionary of outputs from each node.
        execution_time: Total execution time in seconds.
        params: Optional pipeline parameters.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_tag("pipeline", pipeline_name)
        mlflow.log_metric("execution_time_seconds", execution_time)

        if params:
            log_params_safe(params, prefix="pipeline_")

        # Log output summaries
        for node_name, output in node_outputs.items():
            if isinstance(output, dict):
                # Log metrics from dict outputs
                for key, value in output.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{node_name}_{key}", value)

        logger.info(f"Logged pipeline run: {pipeline_name}")
    except Exception as e:
        logger.warning(f"Failed to log pipeline run: {e}")


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        prefix: Prefix for keys.

    Returns:
        Flattened dictionary.
    """
    items = {}
    for key, value in d.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, f"{new_key}_"))
        else:
            items[new_key] = value
    return items
