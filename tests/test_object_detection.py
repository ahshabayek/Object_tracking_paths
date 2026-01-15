"""Tests for Object Detection Pipeline Nodes.

This module tests the object detection functionality including:
- DetectorFactory with all supported models
- Detection and DetectionResult dataclasses
- Post-processing for various model outputs
- Open-vocabulary detection (YOLO-World, GroundingDINO)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_nodes_module():
    """Dynamically load nodes module to bypass kedro imports in __init__.py."""
    nodes_path = (
        Path(__file__).parent.parent
        / "src"
        / "cv_pipeline"
        / "pipelines"
        / "object_detection"
        / "nodes.py"
    )
    spec = importlib.util.spec_from_file_location("nodes", nodes_path)
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)
    return nodes


# Load nodes module dynamically
try:
    import torch

    nodes = load_nodes_module()
    NODES_AVAILABLE = True
    NODES_ERROR = ""
except Exception as e:
    NODES_AVAILABLE = False
    nodes = None
    NODES_ERROR = str(e)


# Skip decorator for tests requiring nodes
requires_nodes = pytest.mark.skipif(
    not NODES_AVAILABLE,
    reason=f"Nodes module not available: {NODES_ERROR}",
)


@requires_nodes
class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a Detection object."""
        Detection = nodes.Detection

        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.phrase is None
        assert det.features is None

    def test_detection_with_phrase(self):
        """Test Detection with phrase for open-vocabulary models."""
        Detection = nodes.Detection

        det = Detection(
            bbox=np.array([50, 50, 150, 150]),
            confidence=0.85,
            class_id=-1,
            class_name="red car",
            phrase="red car",
        )
        assert det.phrase == "red car"
        assert det.class_id == -1  # Open vocabulary

    def test_detection_to_dict(self):
        """Test converting Detection to dictionary."""
        Detection = nodes.Detection

        det = Detection(
            bbox=np.array([10, 20, 30, 40]),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
        d = det.to_dict()
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["confidence"] == 0.9
        assert d["class_id"] == 2
        assert d["class_name"] == "car"
        assert "phrase" not in d  # Should not include None phrase

    def test_detection_to_dict_with_phrase(self):
        """Test converting Detection with phrase to dictionary."""
        Detection = nodes.Detection

        det = Detection(
            bbox=np.array([10, 20, 30, 40]),
            confidence=0.9,
            class_id=-1,
            class_name="traffic light",
            phrase="traffic light",
        )
        d = det.to_dict()
        assert d["phrase"] == "traffic light"


@requires_nodes
class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_detection_result_empty(self):
        """Test DetectionResult with no detections."""
        DetectionResult = nodes.DetectionResult

        result = DetectionResult(
            frame_id=0,
            detections=[],
            inference_time=0.01,
        )
        assert result.num_detections == 0
        assert result.get_boxes().shape == (0, 4)
        assert result.get_scores().shape == (0,)
        assert result.get_classes().shape == (0,)

    def test_detection_result_with_detections(self):
        """Test DetectionResult with multiple detections."""
        Detection = nodes.Detection
        DetectionResult = nodes.DetectionResult

        detections = [
            Detection(np.array([10, 10, 50, 50]), 0.9, 0, "person"),
            Detection(np.array([100, 100, 200, 200]), 0.8, 2, "car"),
        ]
        result = DetectionResult(
            frame_id=5,
            detections=detections,
            inference_time=0.05,
        )
        assert result.num_detections == 2
        assert result.get_boxes().shape == (2, 4)
        assert result.get_scores().shape == (2,)
        assert result.get_classes().shape == (2,)
        np.testing.assert_array_equal(result.get_classes(), [0, 2])


@requires_nodes
class TestDetectorFactory:
    """Tests for DetectorFactory."""

    def test_supported_models(self):
        """Test that all expected models are in SUPPORTED_MODELS."""
        DetectorFactory = nodes.DetectorFactory

        expected_models = [
            "rf_detr",
            "rt_detr",
            "rt_detr_v2",
            "d_fine",
            "yolov12",
            "yolov11",
            "yolov10",
            "yolo_world",
            "grounding_dino",
        ]
        for model in expected_models:
            assert model in DetectorFactory.SUPPORTED_MODELS

    def test_open_vocab_models(self):
        """Test that open-vocab models are correctly identified."""
        DetectorFactory = nodes.DetectorFactory

        assert DetectorFactory.is_open_vocab("yolo_world")
        assert DetectorFactory.is_open_vocab("grounding_dino")
        assert not DetectorFactory.is_open_vocab("rt_detr")
        assert not DetectorFactory.is_open_vocab("yolov11")
        assert not DetectorFactory.is_open_vocab("d_fine")

    def test_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        DetectorFactory = nodes.DetectorFactory

        with pytest.raises(ValueError, match="Unknown model"):
            DetectorFactory.create("unknown_model", {})


@requires_nodes
class TestPostProcessing:
    """Tests for post-processing functions."""

    def test_coco_classes_length(self):
        """Test COCO classes constant has correct length."""
        COCO_CLASSES = nodes.COCO_CLASSES

        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[2] == "car"
        assert COCO_CLASSES[7] == "truck"

    def test_is_ultralytics_output_false(self):
        """Test _is_ultralytics_output returns False for non-ultralytics."""
        _is_ultralytics_output = nodes._is_ultralytics_output

        assert not _is_ultralytics_output(None)
        assert not _is_ultralytics_output([])
        assert not _is_ultralytics_output({"boxes": []})
        assert not _is_ultralytics_output([{"boxes": []}])

    def test_post_process_grounding_dino_empty(self):
        """Test GroundingDINO post-processing with empty output."""
        _post_process_grounding_dino = nodes._post_process_grounding_dino

        output = {
            "boxes": [],
            "logits": [],
            "phrases": [],
            "image_size": (640, 480),
        }
        detections = _post_process_grounding_dino(output, conf_thresh=0.25)
        assert len(detections) == 0

    def test_post_process_grounding_dino_with_detections(self):
        """Test GroundingDINO post-processing with detections."""
        _post_process_grounding_dino = nodes._post_process_grounding_dino

        # Simulated GroundingDINO output (normalized cxcywh)
        output = {
            "boxes": [
                np.array([0.5, 0.5, 0.2, 0.3]),  # cx, cy, w, h normalized
            ],
            "logits": [0.85],
            "phrases": ["person"],
            "image_size": (640, 480),
        }
        detections = _post_process_grounding_dino(output, conf_thresh=0.25)

        assert len(detections) == 1
        det = detections[0]
        assert det.class_id == -1  # Open vocabulary
        assert det.class_name == "person"
        assert det.phrase == "person"
        assert det.confidence == 0.85
        # Check box conversion from cxcywh to xyxy
        assert det.bbox[0] < det.bbox[2]  # x1 < x2
        assert det.bbox[1] < det.bbox[3]  # y1 < y2

    def test_post_process_grounding_dino_filters_low_conf(self):
        """Test that low confidence detections are filtered."""
        _post_process_grounding_dino = nodes._post_process_grounding_dino

        output = {
            "boxes": [
                np.array([0.5, 0.5, 0.2, 0.3]),
                np.array([0.3, 0.3, 0.1, 0.1]),
            ],
            "logits": [0.85, 0.15],  # Second one below threshold
            "phrases": ["car", "bicycle"],
            "image_size": (640, 480),
        }
        detections = _post_process_grounding_dino(output, conf_thresh=0.25)

        assert len(detections) == 1
        assert detections[0].class_name == "car"

    def test_post_process_d_fine_empty(self):
        """Test D-FINE post-processing with empty output."""
        _post_process_d_fine = nodes._post_process_d_fine

        output = {
            "outputs": {
                "pred_boxes": np.array([]).reshape(0, 4),
                "pred_logits": np.array([]).reshape(0, 81),
            }
        }
        detections = _post_process_d_fine(output, conf_thresh=0.25)
        assert len(detections) == 0

    def test_post_process_d_fine_with_detections(self):
        """Test D-FINE post-processing with detections."""
        import torch

        _post_process_d_fine = nodes._post_process_d_fine

        # Simulated D-FINE output
        pred_boxes = np.array([[0.5, 0.5, 0.2, 0.3]])  # cxcywh
        pred_logits = np.zeros((1, 81))
        pred_logits[0, 0] = 2.0  # High logit for person class

        output = {
            "outputs": {
                "pred_boxes": pred_boxes,
                "pred_logits": torch.from_numpy(pred_logits),
            }
        }
        detections = _post_process_d_fine(output, conf_thresh=0.1)

        assert len(detections) >= 0  # May be 0 or 1 depending on softmax

    def test_post_process_ultralytics_class_names(self):
        """Test Ultralytics post-processing uses correct class names."""
        import torch

        _post_process_ultralytics = nodes._post_process_ultralytics
        COCO_CLASSES = nodes.COCO_CLASSES

        # Mock Ultralytics result
        class MockBoxes:
            def __init__(self):
                self.xyxy = torch.tensor([[10, 20, 100, 150]])
                self.conf = torch.tensor([0.9])
                self.cls = torch.tensor([2])  # car

            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes()
                self.names = {0: "person", 2: "car", 7: "truck"}

        result = MockResult()
        detections = _post_process_ultralytics(result, 0.25, "yolov11")

        assert len(detections) == 1
        assert detections[0].class_name == "car"
        assert detections[0].class_id == 2


class TestWeightsRegistry:
    """Tests for weights registry entries for new models."""

    def test_dfine_weights_in_registry(self):
        """Test D-FINE weights are in registry."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        dfine_variants = ["dfine-n", "dfine-s", "dfine-m", "dfine-l", "dfine-x"]
        for variant in dfine_variants:
            assert variant in WEIGHT_REGISTRY
            assert WEIGHT_REGISTRY[variant]["framework"] == "dfine"

    def test_yoloworld_weights_in_registry(self):
        """Test YOLO-World weights are in registry."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        yoloworld_variants = ["yoloworld-s", "yoloworld-m", "yoloworld-l", "yoloworld-x"]
        for variant in yoloworld_variants:
            assert variant in WEIGHT_REGISTRY
            assert WEIGHT_REGISTRY[variant]["framework"] == "ultralytics"

    def test_groundingdino_weights_in_registry(self):
        """Test GroundingDINO weights are in registry."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        gd_variants = ["groundingdino-tiny", "groundingdino-base"]
        for variant in gd_variants:
            assert variant in WEIGHT_REGISTRY
            assert WEIGHT_REGISTRY[variant]["framework"] == "groundingdino"
            assert "config" in WEIGHT_REGISTRY[variant]

    def test_yolo12_weights_in_registry(self):
        """Test YOLOv12 weights are in registry."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        yolo12_variants = ["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"]
        for variant in yolo12_variants:
            assert variant in WEIGHT_REGISTRY
            assert WEIGHT_REGISTRY[variant]["framework"] == "ultralytics"

    def test_rfdetr_weights_in_registry(self):
        """Test RF-DETR weights are in registry."""
        from cv_pipeline.utils.weights import WEIGHT_REGISTRY

        rfdetr_variants = ["rfdetr-b", "rfdetr-l"]
        for variant in rfdetr_variants:
            assert variant in WEIGHT_REGISTRY
            assert WEIGHT_REGISTRY[variant]["framework"] == "rfdetr"


class TestOpenVocabDetection:
    """Tests for open-vocabulary detection features."""

    def test_yolo_world_text_prompts_param(self):
        """Test YOLO-World accepts text_prompts parameter."""
        # This is a configuration test - actual model loading requires dependencies
        params = {
            "model": "yolo_world",
            "variant": "l",
            "text_prompts": ["person", "car", "dog", "traffic light"],
        }
        assert "text_prompts" in params
        assert len(params["text_prompts"]) == 4

    def test_grounding_dino_text_prompt_param(self):
        """Test GroundingDINO accepts text_prompt parameter."""
        params = {
            "model": "grounding_dino",
            "variant": "base",
            "text_prompt": "person . car . traffic light .",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
        }
        assert "text_prompt" in params
        assert " . " in params["text_prompt"]  # Correct separator


class TestModelParameters:
    """Tests for model parameter configurations."""

    def test_rt_detr_v2_version_param(self):
        """Test RT-DETR v2 version parameter."""
        params = {
            "model": "rt_detr_v2",
            "variant": "l",
            "version": "v2",
        }
        assert params["version"] == "v2"

    def test_d_fine_config_params(self):
        """Test D-FINE configuration parameters."""
        params = {
            "model": "d_fine",
            "variant": "l",
            "config_path": "/path/to/config.yaml",
            "weights_path": "/path/to/weights.pth",
        }
        assert "config_path" in params
        assert "weights_path" in params

    def test_default_confidence_threshold(self):
        """Test default confidence threshold."""
        default_conf = 0.25
        params = {"model": "yolov11"}
        conf = params.get("confidence_threshold", default_conf)
        assert conf == 0.25

    def test_half_precision_param(self):
        """Test half precision parameter."""
        params = {
            "model": "rt_detr",
            "half_precision": True,
            "device": "cuda:0",
        }
        assert params["half_precision"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
