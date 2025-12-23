"""Unit tests for profiling and performance utilities."""

import time
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


class TestTimer:
    """Tests for Timer class."""

    def test_timer_basic(self):
        """Test basic timer functionality."""
        from cv_pipeline.utils.profiling import Timer

        timer = Timer(use_cuda_sync=False)
        timer.start()
        time.sleep(0.01)  # Sleep 10ms
        elapsed = timer.stop()

        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should not take more than 100ms

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        from cv_pipeline.utils.profiling import Timer

        with Timer(use_cuda_sync=False) as timer:
            time.sleep(0.01)

        assert timer.elapsed >= 0.01

    def test_timer_elapsed_property(self):
        """Test elapsed property."""
        from cv_pipeline.utils.profiling import Timer

        timer = Timer(use_cuda_sync=False)
        timer.start()
        time.sleep(0.01)
        timer.stop()

        assert timer.elapsed >= 0.01

    def test_timer_not_started(self):
        """Test stopping timer that wasn't started."""
        from cv_pipeline.utils.profiling import Timer

        timer = Timer(use_cuda_sync=False)
        elapsed = timer.stop()

        assert elapsed == 0


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_timing_result_properties(self):
        """Test TimingResult computed properties."""
        from cv_pipeline.utils.profiling import TimingResult

        result = TimingResult(
            name="test",
            total_time=1.0,
            call_count=10,
            times=[0.08, 0.09, 0.10, 0.11, 0.12, 0.08, 0.09, 0.10, 0.11, 0.12],
        )

        assert result.avg_time == 0.1
        assert result.min_time == 0.08
        assert result.max_time == 0.12
        assert result.std_time > 0

    def test_timing_result_empty(self):
        """Test TimingResult with no times."""
        from cv_pipeline.utils.profiling import TimingResult

        result = TimingResult(
            name="test",
            total_time=0,
            call_count=0,
            times=[],
        )

        assert result.avg_time == 0
        assert result.min_time == 0
        assert result.max_time == 0
        assert result.std_time == 0

    def test_timing_result_to_dict(self):
        """Test TimingResult to_dict method."""
        from cv_pipeline.utils.profiling import TimingResult

        result = TimingResult(
            name="test",
            total_time=1.0,
            call_count=10,
            times=[0.1] * 10,
        )

        d = result.to_dict()

        assert d["name"] == "test"
        assert d["total_time"] == 1.0
        assert d["call_count"] == 10
        assert "avg_time" in d
        assert "min_time" in d
        assert "max_time" in d


class TestProfiler:
    """Tests for Profiler class."""

    def test_profiler_basic(self):
        """Test basic profiler functionality."""
        from cv_pipeline.utils.profiling import Profiler

        profiler = Profiler(use_cuda_sync=False)

        with profiler.profile("operation1"):
            time.sleep(0.01)

        with profiler.profile("operation2"):
            time.sleep(0.02)

        timing1 = profiler.get_timing("operation1")
        timing2 = profiler.get_timing("operation2")

        assert timing1 is not None
        assert timing2 is not None
        assert timing1.call_count == 1
        assert timing2.call_count == 1
        assert timing1.total_time >= 0.01
        assert timing2.total_time >= 0.02

    def test_profiler_multiple_calls(self):
        """Test profiler with multiple calls to same operation."""
        from cv_pipeline.utils.profiling import Profiler

        profiler = Profiler(use_cuda_sync=False)

        for _ in range(5):
            with profiler.profile("operation"):
                time.sleep(0.001)

        timing = profiler.get_timing("operation")

        assert timing is not None
        assert timing.call_count == 5
        assert len(timing.times) == 5

    def test_profiler_get_all_timings(self):
        """Test getting all timings."""
        from cv_pipeline.utils.profiling import Profiler

        profiler = Profiler(use_cuda_sync=False)

        with profiler.profile("op1"):
            pass
        with profiler.profile("op2"):
            pass

        all_timings = profiler.get_all_timings()

        assert "op1" in all_timings
        assert "op2" in all_timings

    def test_profiler_reset(self):
        """Test profiler reset."""
        from cv_pipeline.utils.profiling import Profiler

        profiler = Profiler(use_cuda_sync=False)

        with profiler.profile("operation"):
            pass

        profiler.reset()

        assert profiler.get_timing("operation") is None

    def test_profiler_summary(self):
        """Test profiler summary."""
        from cv_pipeline.utils.profiling import Profiler

        profiler = Profiler(use_cuda_sync=False)

        with profiler.profile("operation"):
            time.sleep(0.01)

        summary = profiler.summary()

        assert "operation" in summary
        assert "total_time" in summary["operation"]


class TestTimedDecorator:
    """Tests for timed decorator."""

    def test_timed_decorator(self):
        """Test timed decorator."""
        from cv_pipeline.utils.profiling import timed

        @timed(name="test_func")
        def my_function():
            time.sleep(0.01)
            return 42

        result = my_function()

        assert result == 42

    def test_timed_decorator_default_name(self):
        """Test timed decorator with default name."""
        from cv_pipeline.utils.profiling import timed

        @timed()
        def another_function():
            return "hello"

        result = another_function()

        assert result == "hello"


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_memory_stats_to_dict(self):
        """Test MemoryStats to_dict method."""
        from cv_pipeline.utils.profiling import MemoryStats

        stats = MemoryStats(
            allocated_mb=100.0,
            reserved_mb=200.0,
            max_allocated_mb=150.0,
            max_reserved_mb=250.0,
        )

        d = stats.to_dict()

        assert d["allocated_mb"] == 100.0
        assert d["reserved_mb"] == 200.0
        assert d["max_allocated_mb"] == 150.0
        assert d["max_reserved_mb"] == 250.0


class TestGPUMemoryFunctions:
    """Tests for GPU memory functions."""

    def test_get_gpu_memory_stats_no_cuda(self):
        """Test getting GPU stats when CUDA not available."""
        from cv_pipeline.utils.profiling import get_gpu_memory_stats

        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            stats = get_gpu_memory_stats()
            assert stats is None

    def test_reset_gpu_memory_stats_no_cuda(self):
        """Test resetting GPU stats when CUDA not available."""
        from cv_pipeline.utils.profiling import reset_gpu_memory_stats

        # Should not raise exception
        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            reset_gpu_memory_stats()

    def test_clear_gpu_memory_no_cuda(self):
        """Test clearing GPU memory when CUDA not available."""
        from cv_pipeline.utils.profiling import clear_gpu_memory

        # Should not raise exception
        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            clear_gpu_memory()

    def test_track_gpu_memory_no_cuda(self):
        """Test tracking GPU memory when CUDA not available."""
        from cv_pipeline.utils.profiling import track_gpu_memory

        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            with track_gpu_memory() as get_memory:
                memory = get_memory()
                assert memory == 0.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult to_dict method."""
        from cv_pipeline.utils.profiling import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_time=1.0,
            throughput=100.0,
            latency_mean=0.01,
            latency_std=0.001,
            latency_min=0.008,
            latency_max=0.012,
            memory_mb=50.0,
        )

        d = result.to_dict()

        assert d["name"] == "test"
        assert d["iterations"] == 100
        assert d["throughput"] == 100.0
        assert d["latency_mean_ms"] == 10.0  # Converted to ms
        assert d["memory_mb"] == 50.0


class TestBenchmark:
    """Tests for benchmark function."""

    def test_benchmark_basic(self):
        """Test basic benchmark functionality."""
        from cv_pipeline.utils.profiling import benchmark

        def simple_func():
            time.sleep(0.001)
            return 42

        result = benchmark(simple_func, iterations=10, warmup=2)

        assert result.name == "simple_func"
        assert result.iterations == 10
        assert result.throughput > 0
        assert result.latency_mean > 0

    def test_benchmark_with_args(self):
        """Test benchmark with function arguments."""
        from cv_pipeline.utils.profiling import benchmark

        def add(a, b):
            return a + b

        result = benchmark(add, args=(1, 2), iterations=10, warmup=2)

        assert result.iterations == 10

    def test_benchmark_with_kwargs(self):
        """Test benchmark with keyword arguments."""
        from cv_pipeline.utils.profiling import benchmark

        def greet(name="World"):
            return f"Hello, {name}!"

        result = benchmark(greet, kwargs={"name": "Test"}, iterations=10, warmup=2)

        assert result.iterations == 10

    def test_benchmark_custom_name(self):
        """Test benchmark with custom name."""
        from cv_pipeline.utils.profiling import benchmark

        def my_func():
            pass

        result = benchmark(my_func, iterations=5, warmup=1, name="custom_name")

        assert result.name == "custom_name"


class TestBenchmarkModel:
    """Tests for benchmark_model function."""

    def test_benchmark_model_cpu(self):
        """Test model benchmarking on CPU."""
        from cv_pipeline.utils.profiling import benchmark_model

        model = SimpleModel()

        result = benchmark_model(
            model,
            input_shape=(1, 10),
            iterations=10,
            warmup=2,
            device="cpu",
        )

        assert result.name == "SimpleModel"
        assert result.iterations == 10
        assert result.throughput > 0


class TestOptimizeForInference:
    """Tests for optimize_for_inference function."""

    def test_optimize_for_inference(self):
        """Test model optimization for inference."""
        from cv_pipeline.utils.profiling import optimize_for_inference

        model = SimpleModel()
        model.train()

        optimized = optimize_for_inference(model)

        assert not optimized.training

    def test_optimize_freezes_batchnorm(self):
        """Test that batch normalization is frozen."""
        from cv_pipeline.utils.profiling import optimize_for_inference

        class ModelWithBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x):
                return self.bn(self.fc(x))

        model = ModelWithBN()
        model.train()

        optimized = optimize_for_inference(model)

        assert not optimized.bn.training


class TestCompileModel:
    """Tests for compile_model function."""

    def test_compile_model_no_torch_compile(self):
        """Test compile_model when torch.compile not available."""
        from cv_pipeline.utils.profiling import compile_model

        model = SimpleModel()

        # Mock torch to not have compile
        with patch.object(torch, "compile", None, create=True):
            with patch("cv_pipeline.utils.profiling.hasattr", return_value=False):
                result = compile_model(model)
                assert result is model  # Should return original model


class TestDeterministicMode:
    """Tests for set_deterministic function."""

    def test_set_deterministic(self):
        """Test setting deterministic mode."""
        from cv_pipeline.utils.profiling import set_deterministic

        set_deterministic(seed=123)

        # Check that settings were applied
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestEnableCudnnBenchmark:
    """Tests for enable_cudnn_benchmark function."""

    def test_enable_cudnn_benchmark_no_cuda(self):
        """Test enabling cuDNN benchmark when CUDA not available."""
        from cv_pipeline.utils.profiling import enable_cudnn_benchmark

        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            enable_cudnn_benchmark()  # Should not raise


class TestThroughputTracker:
    """Tests for ThroughputTracker class."""

    def test_throughput_tracker_basic(self):
        """Test basic throughput tracking."""
        from cv_pipeline.utils.profiling import ThroughputTracker

        tracker = ThroughputTracker(window_size=10)

        for i in range(5):
            tracker.update(count=1)
            time.sleep(0.01)

        assert tracker.total_count == 5
        assert tracker.fps > 0

    def test_throughput_tracker_batch_count(self):
        """Test throughput tracking with batch counts."""
        from cv_pipeline.utils.profiling import ThroughputTracker

        tracker = ThroughputTracker()

        tracker.update(count=10)
        time.sleep(0.01)
        tracker.update(count=10)

        assert tracker.total_count == 20

    def test_throughput_tracker_reset(self):
        """Test throughput tracker reset."""
        from cv_pipeline.utils.profiling import ThroughputTracker

        tracker = ThroughputTracker()

        tracker.update(count=5)
        tracker.reset()

        assert tracker.total_count == 0
        assert tracker.fps == 0

    def test_throughput_tracker_average_fps(self):
        """Test average FPS calculation."""
        from cv_pipeline.utils.profiling import ThroughputTracker

        tracker = ThroughputTracker()

        for _ in range(10):
            tracker.update(count=1)
            time.sleep(0.01)

        assert tracker.average_fps > 0

    def test_throughput_tracker_window_size(self):
        """Test window size limiting."""
        from cv_pipeline.utils.profiling import ThroughputTracker

        tracker = ThroughputTracker(window_size=5)

        for _ in range(10):
            tracker.update(count=1)

        # Internal lists should be trimmed to window size
        assert len(tracker._timestamps) <= 5
        assert len(tracker._counts) <= 5
        # But total count should still reflect all updates
        assert tracker.total_count == 10


class TestGetOptimalBatchSize:
    """Tests for get_optimal_batch_size function."""

    def test_get_optimal_batch_size_cpu(self):
        """Test optimal batch size on CPU."""
        from cv_pipeline.utils.profiling import get_optimal_batch_size

        model = SimpleModel()

        batch_size = get_optimal_batch_size(
            model,
            input_shape=(10,),
            device="cpu",
            max_batch=32,
        )

        assert batch_size == 32  # Should return max on CPU

    def test_get_optimal_batch_size_no_cuda(self):
        """Test optimal batch size when CUDA not available."""
        from cv_pipeline.utils.profiling import get_optimal_batch_size

        model = SimpleModel()

        with patch("cv_pipeline.utils.profiling.torch.cuda.is_available", return_value=False):
            batch_size = get_optimal_batch_size(
                model,
                input_shape=(10,),
                device="cuda",
                max_batch=64,
            )

            assert batch_size == 64
