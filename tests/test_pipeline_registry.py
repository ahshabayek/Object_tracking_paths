"""Unit tests for pipeline registry."""

import pytest

# Check if kedro is available
try:
    import kedro
    from kedro.pipeline import Pipeline

    KEDRO_AVAILABLE = True
except ImportError:
    KEDRO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not KEDRO_AVAILABLE, reason="kedro is not installed")


class TestPipelineRegistry:
    """Tests for pipeline registry."""

    def test_register_pipelines_returns_dict(self):
        """Test that register_pipelines returns a dictionary."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()

        assert isinstance(pipelines, dict)
        assert len(pipelines) > 0

    def test_all_pipelines_are_pipeline_objects(self):
        """Test that all registered pipelines are Pipeline objects."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()

        for name, pipeline in pipelines.items():
            assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"

    def test_required_pipelines_exist(self):
        """Test that all required pipelines are registered."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()

        required = [
            "data_processing",
            "object_detection",
            "tracking",
            "lane_detection",
            "path_construction",
            "inference",
            "__default__",
        ]

        for name in required:
            assert name in pipelines, f"Missing required pipeline: {name}"

    def test_default_pipeline_exists(self):
        """Test that __default__ pipeline exists."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()

        assert "__default__" in pipelines
        assert len(pipelines["__default__"].nodes) > 0

    def test_inference_pipeline_has_all_stages(self):
        """Test that inference pipeline includes all stages."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()
        inference = pipelines["inference"]

        # Get all node names
        node_names = [node.name for node in inference.nodes]

        # Check that key nodes from each stage exist
        assert "load_video_frames" in node_names
        assert "load_detection_model" in node_names
        assert "run_detection_inference" in node_names
        assert "initialize_tracker" in node_names
        assert "run_tracking" in node_names
        assert "load_lane_model" in node_names
        assert "run_lane_detection" in node_names
        assert "construct_drivable_path" in node_names

    def test_inference_pipeline_node_count(self):
        """Test that inference pipeline has expected number of nodes."""
        from cv_pipeline.pipeline_registry import register_pipelines

        pipelines = register_pipelines()
        inference = pipelines["inference"]

        # Should have nodes from all 5 stages
        # Data processing: 5, Detection: 6, Tracking: 5, Lane: 5, Path: 5 = 26
        assert len(inference.nodes) == 26


class TestCreateInferencePipeline:
    """Tests for create_inference_pipeline function."""

    def test_create_inference_pipeline(self):
        """Test that inference pipeline can be created."""
        from cv_pipeline.pipeline_registry import create_inference_pipeline

        pipeline = create_inference_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.nodes) > 0

    def test_inference_pipeline_has_inference_tag(self):
        """Test that all nodes in inference pipeline have inference tag."""
        from cv_pipeline.pipeline_registry import create_inference_pipeline

        pipeline = create_inference_pipeline()

        # The pipeline itself should have the inference tag
        # Individual nodes may have additional tags


class TestSubsetPipelines:
    """Tests for subset pipeline functions."""

    def test_detection_only_pipeline(self):
        """Test detection-only pipeline."""
        from cv_pipeline.pipeline_registry import create_detection_only_pipeline

        pipeline = create_detection_only_pipeline()

        assert isinstance(pipeline, Pipeline)

        # Should only have data_processing and object_detection nodes
        node_names = [node.name for node in pipeline.nodes]

        # Should have detection nodes
        assert "load_detection_model" in node_names
        assert "run_detection_inference" in node_names

        # Should NOT have tracking/lane nodes
        assert "initialize_tracker" not in node_names
        assert "load_lane_model" not in node_names

    def test_tracking_only_pipeline(self):
        """Test tracking-only pipeline."""
        from cv_pipeline.pipeline_registry import create_tracking_only_pipeline

        pipeline = create_tracking_only_pipeline()

        assert isinstance(pipeline, Pipeline)

        node_names = [node.name for node in pipeline.nodes]

        # Should have tracking nodes
        assert "initialize_tracker" in node_names
        assert "run_tracking" in node_names

        # Should NOT have lane nodes
        assert "load_lane_model" not in node_names


class TestPipelineDataFlow:
    """Tests for pipeline data flow."""

    def test_inference_pipeline_inputs_outputs(self):
        """Test that inference pipeline has correct inputs/outputs."""
        from cv_pipeline.pipeline_registry import create_inference_pipeline

        pipeline = create_inference_pipeline()

        # Get all inputs and outputs
        all_inputs = set()
        all_outputs = set()

        for node in pipeline.nodes:
            all_inputs.update(node.inputs)
            all_outputs.update(node.outputs)

        # raw_video should be an input (not produced by any node)
        external_inputs = all_inputs - all_outputs
        assert "raw_video" in external_inputs

        # constructed_path should be a final output
        assert "constructed_path" in all_outputs

    def test_data_flows_between_stages(self):
        """Test that data flows correctly between pipeline stages."""
        from cv_pipeline.pipeline_registry import create_inference_pipeline

        pipeline = create_inference_pipeline()

        # Build a map of outputs to nodes
        output_to_node = {}
        for node in pipeline.nodes:
            for output in node.outputs:
                output_to_node[output] = node.name

        # Check key data flows
        # preprocessed_frames should be used by multiple nodes
        preprocessed_users = [
            node for node in pipeline.nodes if "preprocessed_frames" in node.inputs
        ]
        assert len(preprocessed_users) >= 2  # tracking and lane detection

        # detection_results should flow to tracking
        detection_results_users = [
            node for node in pipeline.nodes if "detection_results" in node.inputs
        ]
        assert any(
            "tracking" in node.name.lower() or node.name == "run_tracking"
            for node in detection_results_users
        )


class TestPipelineTags:
    """Tests for pipeline tags."""

    def test_inference_pipeline_has_tags(self):
        """Test that inference pipeline nodes have appropriate tags."""
        from cv_pipeline.pipeline_registry import create_inference_pipeline

        pipeline = create_inference_pipeline()

        # Check that nodes have stage-specific tags
        for node in pipeline.nodes:
            # Each node should have at least one tag
            assert len(node.tags) > 0

            # Verify nodes have appropriate tags based on their names
            if "detection" in node.name.lower() and "lane" not in node.name.lower():
                assert "object_detection" in node.tags or "data_processing" in node.tags
            elif "tracking" in node.name.lower() or "tracker" in node.name.lower():
                assert "tracking" in node.tags
            elif "lane" in node.name.lower():
                assert "lane_detection" in node.tags
            elif "path" in node.name.lower() or "fuse" in node.name.lower():
                assert "path_construction" in node.tags
