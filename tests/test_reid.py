"""Unit tests for Re-ID feature extractors."""

import numpy as np
import pytest
import torch


class TestBaseReIDExtractor:
    """Tests for base Re-ID extractor functionality."""

    def test_preprocess_output_shape(self):
        """Test preprocessing output shape."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu")
        extractor.load_model()

        # Create test image
        image = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

        tensor = extractor.preprocess(image)

        # Should be (C, H, W) with correct input size
        assert tensor.shape == (3, extractor.input_size[0], extractor.input_size[1])

    def test_crop_image(self):
        """Test image cropping."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(device="cpu")

        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 300])

        crop = extractor.crop_image(image, bbox)

        assert crop.shape == (200, 100, 3)

    def test_crop_image_with_padding(self):
        """Test image cropping with padding."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(device="cpu")

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])

        crop = extractor.crop_image(image, bbox, padding=0.1)

        # Should be larger than 100x100 due to padding
        assert crop.shape[0] > 100
        assert crop.shape[1] > 100

    def test_crop_image_boundary_handling(self):
        """Test cropping at image boundaries."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(device="cpu")

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Bbox extends beyond image
        bbox = np.array([80, 80, 150, 150])

        crop = extractor.crop_image(image, bbox)

        # Should handle boundary gracefully
        assert crop.shape[2] == 3


class TestSimpleReIDExtractor:
    """Tests for SimpleReIDExtractor."""

    def test_init(self):
        """Test initialization."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=512, device="cpu", batch_size=16)

        assert extractor.feature_dim == 512
        assert extractor.batch_size == 16
        assert extractor.device == "cpu"

    def test_load_model(self):
        """Test model loading."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu")
        extractor.load_model()

        assert extractor.model is not None

    def test_extract_features(self):
        """Test feature extraction."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu")
        extractor.load_model()

        # Create test image and bboxes
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = np.array(
            [
                [100, 100, 200, 300],
                [300, 200, 400, 400],
            ]
        )

        features = extractor.extract(image, bboxes)

        assert features.shape == (2, 256)

    def test_extract_empty_bboxes(self):
        """Test extraction with empty bboxes."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu")
        extractor.load_model()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = np.array([]).reshape(0, 4)

        features = extractor.extract(image, bboxes)

        assert features.shape == (0, 256)

    def test_normalized_features(self):
        """Test that features are L2 normalized."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu")
        extractor.load_model()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = np.array([[100, 100, 200, 300]])

        features = extractor.extract(image, bboxes, normalize=True)

        # Check L2 norm is ~1
        norm = np.linalg.norm(features[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_compute_distance_cosine(self):
        """Test cosine distance computation."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(device="cpu")

        # Create normalized features
        query = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        gallery = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)

        distances = extractor.compute_distance(query, gallery, metric="cosine")

        assert distances.shape == (2, 2)
        assert np.isclose(distances[0, 0], 0.0)  # Same vector
        assert np.isclose(distances[0, 1], 1.0)  # Orthogonal vectors

    def test_compute_distance_euclidean(self):
        """Test euclidean distance computation."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(device="cpu")

        query = np.array([[0, 0], [1, 1]], dtype=np.float32)
        gallery = np.array([[0, 0], [1, 0]], dtype=np.float32)

        distances = extractor.compute_distance(query, gallery, metric="euclidean")

        assert distances.shape == (2, 2)
        assert np.isclose(distances[0, 0], 0.0)  # Same point


class TestOSNetExtractor:
    """Tests for OSNetExtractor."""

    def test_init_valid_variant(self):
        """Test initialization with valid variant."""
        from cv_pipeline.models.reid import OSNetExtractor

        extractor = OSNetExtractor(variant="osnet_x1_0", device="cpu")

        assert extractor.variant == "osnet_x1_0"
        assert extractor.feature_dim == 512

    def test_init_invalid_variant(self):
        """Test initialization with invalid variant."""
        from cv_pipeline.models.reid import OSNetExtractor

        with pytest.raises(ValueError):
            OSNetExtractor(variant="invalid_variant", device="cpu")

    def test_load_model_placeholder(self):
        """Test loading placeholder model when torchreid not available."""
        from cv_pipeline.models.reid import OSNetExtractor

        extractor = OSNetExtractor(variant="osnet_x1_0", device="cpu")
        extractor.load_model()

        # Should have loaded placeholder model
        assert extractor.model is not None

    def test_feature_extraction(self):
        """Test feature extraction."""
        from cv_pipeline.models.reid import OSNetExtractor

        extractor = OSNetExtractor(variant="osnet_x1_0", device="cpu")
        extractor.load_model()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = np.array([[100, 100, 200, 300]])

        features = extractor.extract(image, bboxes)

        assert features.shape[0] == 1
        assert features.shape[1] == extractor.feature_dim


class TestReIDExtractorFactory:
    """Tests for ReIDExtractor factory class."""

    def test_create_osnet(self):
        """Test creating OSNet extractor."""
        from cv_pipeline.models.reid import ReIDExtractor

        extractor = ReIDExtractor.create("osnet", device="cpu")

        assert extractor is not None
        assert extractor.model is not None

    def test_create_simple(self):
        """Test creating simple extractor."""
        from cv_pipeline.models.reid import ReIDExtractor

        extractor = ReIDExtractor.create("simple", device="cpu", feature_dim=128)

        assert extractor.feature_dim == 128

    def test_create_invalid(self):
        """Test creating with invalid name."""
        from cv_pipeline.models.reid import ReIDExtractor

        with pytest.raises(ValueError):
            ReIDExtractor.create("invalid_extractor", device="cpu")

    def test_get_available(self):
        """Test getting available extractors."""
        from cv_pipeline.models.reid import ReIDExtractor

        available = ReIDExtractor.get_available()

        assert "osnet" in available
        assert "simple" in available
        assert "fastreid" in available


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_large_batch(self):
        """Test processing many detections."""
        from cv_pipeline.models.reid import SimpleReIDExtractor

        extractor = SimpleReIDExtractor(feature_dim=256, device="cpu", batch_size=8)
        extractor.load_model()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create many bboxes
        bboxes = np.array([[i * 30, i * 30, i * 30 + 50, i * 30 + 100] for i in range(20)])

        features = extractor.extract(image, bboxes)

        assert features.shape == (20, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
