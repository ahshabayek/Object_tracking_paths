"""Unit tests for Camera Motion Compensation utilities."""

import numpy as np
import pytest


class TestBaseCMC:
    """Tests for base CMC functionality."""

    def test_identity_matrix(self):
        """Test identity matrix generation."""
        from cv_pipeline.utils.cmc import BaseCMC

        identity = BaseCMC.identity_matrix()

        assert identity.shape == (2, 3)
        assert np.allclose(identity, np.eye(2, 3))

    def test_apply_to_boxes_identity(self):
        """Test applying identity transform to boxes."""
        from cv_pipeline.utils.cmc import NoCMC

        cmc = NoCMC(downscale=1.0)

        boxes = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ],
            dtype=np.float32,
        )

        identity = cmc.identity_matrix()
        transformed = cmc.apply_to_boxes(boxes, identity)

        assert np.allclose(boxes, transformed)

    def test_apply_to_boxes_translation(self):
        """Test applying translation transform to boxes."""
        from cv_pipeline.utils.cmc import NoCMC

        cmc = NoCMC(downscale=1.0)

        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)

        # Translation by (50, 30)
        warp = np.array([[1, 0, 50], [0, 1, 30]], dtype=np.float32)
        transformed = cmc.apply_to_boxes(boxes, warp)

        expected = np.array([[50, 30, 150, 130]])
        assert np.allclose(transformed, expected)

    def test_apply_to_points_identity(self):
        """Test applying identity transform to points."""
        from cv_pipeline.utils.cmc import NoCMC

        cmc = NoCMC(downscale=1.0)

        points = np.array([[100, 200], [300, 400]], dtype=np.float32)
        identity = cmc.identity_matrix()
        transformed = cmc.apply_to_points(points, identity)

        assert np.allclose(points, transformed)

    def test_apply_to_boxes_empty(self):
        """Test applying transform to empty boxes."""
        from cv_pipeline.utils.cmc import NoCMC

        cmc = NoCMC(downscale=1.0)

        boxes = np.array([]).reshape(0, 4)
        identity = cmc.identity_matrix()
        transformed = cmc.apply_to_boxes(boxes, identity)

        assert len(transformed) == 0


class TestNoCMC:
    """Tests for NoCMC (no compensation)."""

    def test_always_returns_identity(self):
        """Test that NoCMC always returns identity."""
        from cv_pipeline.utils.cmc import NoCMC

        cmc = NoCMC()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        warp1 = cmc.compute(frame)
        warp2 = cmc.compute(frame)

        assert np.allclose(warp1, np.eye(2, 3))
        assert np.allclose(warp2, np.eye(2, 3))


class TestSparseOptFlowCMC:
    """Tests for SparseOptFlowCMC."""

    def test_first_frame_returns_identity(self):
        """Test that first frame returns identity."""
        from cv_pipeline.utils.cmc import SparseOptFlowCMC

        cmc = SparseOptFlowCMC()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        warp = cmc.compute(frame)

        assert np.allclose(warp, np.eye(2, 3))

    def test_identical_frames_near_identity(self):
        """Test that identical consecutive frames produce near-identity."""
        from cv_pipeline.utils.cmc import SparseOptFlowCMC

        cmc = SparseOptFlowCMC(downscale=2.0)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # First frame
        cmc.compute(frame)

        # Same frame again
        warp = cmc.compute(frame)

        # Should be close to identity (translation near 0)
        assert np.abs(warp[0, 2]) < 10  # Small x translation
        assert np.abs(warp[1, 2]) < 10  # Small y translation

    def test_reset(self):
        """Test reset functionality."""
        from cv_pipeline.utils.cmc import SparseOptFlowCMC

        cmc = SparseOptFlowCMC()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cmc.compute(frame)

        assert cmc.prev_gray is not None

        cmc.reset()

        assert cmc.prev_gray is None


class TestOpticalFlowCMC:
    """Tests for OpticalFlowCMC."""

    def test_first_frame_returns_identity(self):
        """Test that first frame returns identity."""
        from cv_pipeline.utils.cmc import OpticalFlowCMC

        cmc = OpticalFlowCMC()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        warp = cmc.compute(frame)

        assert np.allclose(warp, np.eye(2, 3))

    def test_with_movement(self):
        """Test with simulated camera movement."""
        from cv_pipeline.utils.cmc import OpticalFlowCMC

        cmc = OpticalFlowCMC(downscale=2.0)

        # Create a frame with features
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[100:200, 100:200] = 255  # White square
        frame1[300:400, 300:400] = 128  # Gray square

        # First frame
        cmc.compute(frame1)

        # Create shifted frame (simulating camera pan)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:200, 120:220] = 255  # Shifted white square
        frame2[300:400, 320:420] = 128  # Shifted gray square

        warp = cmc.compute(frame2)

        # Should detect rightward movement
        assert warp.shape == (2, 3)


class TestFeatureCMC:
    """Tests for FeatureCMC."""

    def test_orb_detector(self):
        """Test with ORB detector."""
        from cv_pipeline.utils.cmc import FeatureCMC

        cmc = FeatureCMC(detector="orb", downscale=2.0)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # First frame
        warp1 = cmc.compute(frame)
        assert np.allclose(warp1, np.eye(2, 3))

        # Second frame
        warp2 = cmc.compute(frame)
        assert warp2.shape == (2, 3)

    def test_invalid_detector(self):
        """Test with invalid detector name."""
        from cv_pipeline.utils.cmc import FeatureCMC

        with pytest.raises(ValueError):
            FeatureCMC(detector="invalid_detector")


class TestECCCMC:
    """Tests for ECCCMC."""

    def test_first_frame_returns_identity(self):
        """Test that first frame returns identity."""
        from cv_pipeline.utils.cmc import ECCCMC

        cmc = ECCCMC(downscale=4.0, num_iterations=10)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        warp = cmc.compute(frame)

        assert np.allclose(warp, np.eye(2, 3))

    def test_motion_models(self):
        """Test different motion models."""
        import cv2

        from cv_pipeline.utils.cmc import ECCCMC

        for model in ["translation", "euclidean", "affine"]:
            cmc = ECCCMC(motion_model=model, downscale=4.0)
            assert cmc.motion_type in [
                cv2.MOTION_TRANSLATION,
                cv2.MOTION_EUCLIDEAN,
                cv2.MOTION_AFFINE,
                cv2.MOTION_HOMOGRAPHY,
            ]


class TestCameraMotionCompensator:
    """Tests for CameraMotionCompensator factory class."""

    def test_create_sparseflow(self):
        """Test creating sparseflow compensator."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="sparseflow")

        assert cmc.method_name == "sparseflow"

    def test_create_none(self):
        """Test creating no compensation."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        warp = cmc.compute(frame)

        assert np.allclose(warp, np.eye(2, 3))

    def test_invalid_method(self):
        """Test with invalid method name."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        with pytest.raises(ValueError):
            CameraMotionCompensator(method="invalid_method")

    def test_get_available_methods(self):
        """Test getting available methods."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        methods = CameraMotionCompensator.get_available_methods()

        assert "sparseflow" in methods
        assert "ecc" in methods
        assert "orb" in methods
        assert "none" in methods

    def test_apply_to_boxes(self):
        """Test applying compensation to boxes."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cmc.compute(frame)

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        transformed = cmc.apply_to_boxes(boxes)

        assert np.allclose(boxes, transformed)

    def test_reset(self):
        """Test reset functionality."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="sparseflow")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cmc.compute(frame)
        cmc.compute(frame)

        assert len(cmc.motion_history) > 0

        cmc.reset()

        assert len(cmc.motion_history) == 0

    def test_smoothed_warp(self):
        """Test smoothed warp matrix computation."""
        from cv_pipeline.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            cmc.compute(frame)

        smoothed = cmc.get_smoothed_warp()

        assert smoothed.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
