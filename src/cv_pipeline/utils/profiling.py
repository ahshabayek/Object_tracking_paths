"""Performance Profiling and Optimization Utilities for CV Pipeline.

This module provides utilities for profiling, benchmarking, and optimizing
the performance of the CV pipeline, including:
- Timing decorators and context managers
- Memory profiling
- GPU utilization monitoring
- Throughput benchmarking
- Model optimization helpers
"""

import functools
import gc
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Container for timing results."""

    name: str
    total_time: float
    call_count: int
    times: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0

    @property
    def min_time(self) -> float:
        """Minimum time."""
        return min(self.times) if self.times else 0

    @property
    def max_time(self) -> float:
        """Maximum time."""
        return max(self.times) if self.times else 0

    @property
    def std_time(self) -> float:
        """Standard deviation of times."""
        return float(np.std(self.times)) if len(self.times) > 1 else 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_time": self.total_time,
            "call_count": self.call_count,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "std_time": self.std_time,
        }


class Timer:
    """High-precision timer for profiling.

    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do something ...
        >>> elapsed = timer.stop()
        >>> print(f"Elapsed: {elapsed:.4f}s")
    """

    def __init__(self, use_cuda_sync: bool = True):
        """Initialize timer.

        Args:
            use_cuda_sync: Synchronize CUDA before timing.
        """
        self.use_cuda_sync = use_cuda_sync and torch.cuda.is_available()
        self._start_time: Optional[float] = None
        self._elapsed: float = 0

    def start(self) -> "Timer":
        """Start the timer."""
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        if self._start_time is None:
            return 0
        self._elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        return self._elapsed

    def __enter__(self) -> "Timer":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


class Profiler:
    """Profiler for tracking timing across multiple operations.

    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile("detection"):
        ...     detect(image)
        >>> with profiler.profile("tracking"):
        ...     track(detections)
        >>> profiler.print_summary()
    """

    def __init__(self, use_cuda_sync: bool = True):
        """Initialize profiler.

        Args:
            use_cuda_sync: Synchronize CUDA before timing.
        """
        self.use_cuda_sync = use_cuda_sync
        self._timings: Dict[str, TimingResult] = {}
        self._active_timers: Dict[str, float] = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block.

        Args:
            name: Name of the operation being profiled.
        """
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            if self.use_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            if name not in self._timings:
                self._timings[name] = TimingResult(
                    name=name,
                    total_time=0,
                    call_count=0,
                    times=[],
                )

            self._timings[name].total_time += elapsed
            self._timings[name].call_count += 1
            self._timings[name].times.append(elapsed)

    def get_timing(self, name: str) -> Optional[TimingResult]:
        """Get timing result for an operation."""
        return self._timings.get(name)

    def get_all_timings(self) -> Dict[str, TimingResult]:
        """Get all timing results."""
        return self._timings.copy()

    def reset(self) -> None:
        """Reset all timings."""
        self._timings.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all timings."""
        return {name: result.to_dict() for name, result in self._timings.items()}

    def print_summary(self) -> None:
        """Print a formatted summary of all timings."""
        if not self._timings:
            print("No timings recorded.")
            return

        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        print(f"{'Operation':<30} {'Calls':>8} {'Total':>10} {'Avg':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)

        total_time = sum(t.total_time for t in self._timings.values())

        for name, result in sorted(self._timings.items(), key=lambda x: -x[1].total_time):
            pct = (result.total_time / total_time * 100) if total_time > 0 else 0
            print(
                f"{name:<30} {result.call_count:>8} "
                f"{result.total_time:>9.3f}s {result.avg_time * 1000:>9.2f}ms "
                f"{result.min_time * 1000:>9.2f}ms {result.max_time * 1000:>9.2f}ms "
                f"({pct:>5.1f}%)"
            )

        print("-" * 70)
        print(f"{'Total':<30} {'':<8} {total_time:>9.3f}s")
        print("=" * 70 + "\n")


def timed(name: Optional[str] = None, log_level: int = logging.DEBUG):
    """Decorator for timing function execution.

    Args:
        name: Optional name for the operation (defaults to function name).
        log_level: Logging level for timing output.

    Example:
        >>> @timed()
        ... def my_function():
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            logger.log(log_level, f"{op_name} took {elapsed * 1000:.2f}ms")

            return result

        return wrapper

    return decorator


@dataclass
class MemoryStats:
    """Container for memory statistics."""

    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "max_allocated_mb": self.max_allocated_mb,
            "max_reserved_mb": self.max_reserved_mb,
        }


def get_gpu_memory_stats(device: Optional[int] = None) -> Optional[MemoryStats]:
    """Get GPU memory statistics.

    Args:
        device: CUDA device index (None for current device).

    Returns:
        MemoryStats or None if CUDA not available.
    """
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()

    return MemoryStats(
        allocated_mb=torch.cuda.memory_allocated(device) / (1024**2),
        reserved_mb=torch.cuda.memory_reserved(device) / (1024**2),
        max_allocated_mb=torch.cuda.max_memory_allocated(device) / (1024**2),
        max_reserved_mb=torch.cuda.max_memory_reserved(device) / (1024**2),
    )


def reset_gpu_memory_stats(device: Optional[int] = None) -> None:
    """Reset GPU memory statistics.

    Args:
        device: CUDA device index (None for current device).
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def track_gpu_memory(device: Optional[int] = None):
    """Context manager for tracking GPU memory usage.

    Args:
        device: CUDA device index.

    Yields:
        Function to get current memory delta.

    Example:
        >>> with track_gpu_memory() as get_memory:
        ...     model = load_model()
        ...     memory_used = get_memory()
    """
    if not torch.cuda.is_available():
        yield lambda: 0.0
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_memory = torch.cuda.memory_allocated(device)

    def get_memory_delta() -> float:
        torch.cuda.synchronize(device)
        current = torch.cuda.memory_allocated(device)
        return (current - start_memory) / (1024**2)

    try:
        yield get_memory_delta
    finally:
        pass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    iterations: int
    total_time: float
    throughput: float  # items per second
    latency_mean: float  # seconds
    latency_std: float
    latency_min: float
    latency_max: float
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "throughput": self.throughput,
            "latency_mean_ms": self.latency_mean * 1000,
            "latency_std_ms": self.latency_std * 1000,
            "latency_min_ms": self.latency_min * 1000,
            "latency_max_ms": self.latency_max * 1000,
            "memory_mb": self.memory_mb,
        }


def benchmark(
    func: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict] = None,
    iterations: int = 100,
    warmup: int = 10,
    name: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark a function's performance.

    Args:
        func: Function to benchmark.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        iterations: Number of iterations for benchmarking.
        warmup: Number of warmup iterations.
        name: Optional name for the benchmark.

    Returns:
        BenchmarkResult with timing statistics.

    Example:
        >>> result = benchmark(detect, args=(image,), iterations=100)
        >>> print(f"Throughput: {result.throughput:.1f} FPS")
    """
    kwargs = kwargs or {}
    name = name or func.__name__

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Clear memory stats
    reset_gpu_memory_stats()

    # Benchmark
    latencies = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies.append(time.perf_counter() - start)

    total_time = time.perf_counter() - start_total

    # Get memory stats
    memory_stats = get_gpu_memory_stats()

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        throughput=iterations / total_time,
        latency_mean=float(np.mean(latencies)),
        latency_std=float(np.std(latencies)),
        latency_min=min(latencies),
        latency_max=max(latencies),
        memory_mb=memory_stats.max_allocated_mb if memory_stats else None,
    )


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
    half_precision: bool = False,
) -> BenchmarkResult:
    """Benchmark a PyTorch model.

    Args:
        model: PyTorch model to benchmark.
        input_shape: Shape of input tensor (including batch dimension).
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        device: Device to run on ('cuda' or 'cpu').
        half_precision: Use FP16 inference.

    Returns:
        BenchmarkResult with timing statistics.
    """
    model = model.to(device)
    model.eval()

    if half_precision and device == "cuda":
        model = model.half()

    # Create input tensor
    dtype = torch.float16 if half_precision else torch.float32
    x = torch.randn(*input_shape, dtype=dtype, device=device)

    @torch.no_grad()
    def inference():
        return model(x)

    return benchmark(
        inference,
        iterations=iterations,
        warmup=warmup,
        name=model.__class__.__name__,
    )


# Model optimization utilities


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize a model for inference.

    Applies various optimizations:
    - Sets model to eval mode
    - Freezes batch normalization
    - Fuses operations where possible

    Args:
        model: PyTorch model to optimize.

    Returns:
        Optimized model.
    """
    model.eval()

    # Freeze batch normalization
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = False

    return model


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> nn.Module:
    """Compile model using torch.compile (PyTorch 2.0+).

    Args:
        model: PyTorch model to compile.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        fullgraph: Whether to compile the entire graph.

    Returns:
        Compiled model.
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model

    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logger.info(f"Model compiled with mode={mode}")
        return compiled
    except Exception as e:
        logger.warning(f"Failed to compile model: {e}")
        return model


def enable_cudnn_benchmark() -> None:
    """Enable cuDNN benchmark mode for faster convolutions."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled")


def set_deterministic(seed: int = 42) -> None:
    """Set deterministic mode for reproducibility.

    Note: This may slow down training.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Deterministic mode enabled with seed={seed}")


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    start_batch: int = 1,
    max_batch: int = 256,
    memory_fraction: float = 0.9,
) -> int:
    """Find optimal batch size that fits in GPU memory.

    Args:
        model: PyTorch model.
        input_shape: Shape of single input (without batch dimension).
        device: Device to test on.
        start_batch: Starting batch size.
        max_batch: Maximum batch size to try.
        memory_fraction: Target fraction of GPU memory to use.

    Returns:
        Optimal batch size.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return max_batch

    model = model.to(device)
    model.eval()

    optimal_batch = start_batch
    batch_size = start_batch

    while batch_size <= max_batch:
        try:
            clear_gpu_memory()

            x = torch.randn(batch_size, *input_shape, device=device)

            with torch.no_grad():
                _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            memory_stats = get_gpu_memory_stats()
            if memory_stats:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                usage_fraction = memory_stats.allocated_mb / total_memory

                if usage_fraction < memory_fraction:
                    optimal_batch = batch_size
                else:
                    break
            else:
                optimal_batch = batch_size

            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            raise
        finally:
            clear_gpu_memory()

    logger.info(f"Optimal batch size: {optimal_batch}")
    return optimal_batch


class ThroughputTracker:
    """Track processing throughput over time.

    Example:
        >>> tracker = ThroughputTracker(window_size=100)
        >>> for batch in batches:
        ...     tracker.update(batch_size=len(batch))
        ...     print(f"Current FPS: {tracker.fps:.1f}")
    """

    def __init__(self, window_size: int = 100):
        """Initialize tracker.

        Args:
            window_size: Number of samples to use for rolling average.
        """
        self.window_size = window_size
        self._timestamps: List[float] = []
        self._counts: List[int] = []
        self._total_count = 0
        self._start_time: Optional[float] = None

    def update(self, count: int = 1) -> None:
        """Record a processing event.

        Args:
            count: Number of items processed.
        """
        current_time = time.perf_counter()

        if self._start_time is None:
            self._start_time = current_time

        self._timestamps.append(current_time)
        self._counts.append(count)
        self._total_count += count

        # Trim to window size
        if len(self._timestamps) > self.window_size:
            self._timestamps = self._timestamps[-self.window_size :]
            self._counts = self._counts[-self.window_size :]

    @property
    def fps(self) -> float:
        """Get current frames/items per second (rolling average)."""
        if len(self._timestamps) < 2:
            return 0.0

        time_delta = self._timestamps[-1] - self._timestamps[0]
        if time_delta <= 0:
            return 0.0

        return sum(self._counts) / time_delta

    @property
    def average_fps(self) -> float:
        """Get overall average FPS."""
        if self._start_time is None or self._total_count == 0:
            return 0.0

        elapsed = time.perf_counter() - self._start_time
        return self._total_count / elapsed if elapsed > 0 else 0.0

    @property
    def total_count(self) -> int:
        """Get total items processed."""
        return self._total_count

    def reset(self) -> None:
        """Reset the tracker."""
        self._timestamps.clear()
        self._counts.clear()
        self._total_count = 0
        self._start_time = None
