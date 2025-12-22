"""Camera Motion Compensation (CMC) for Multi-Object Tracking.

This module provides camera motion compensation algorithms to improve
tracking accuracy when the camera is moving. CMC estimates the camera
motion between frames and compensates for it when predicting object positions.

Supported methods:
- ECC (Enhanced Correlation Coefficient): Most accurate but slowest
- ORB: Feature-based using ORB descriptors (faster)
- SIFT: Feature-based using SIFT descriptors (more robust)
- OpticalFlow: Sparse optical flow based method
- None: No compensation (for stationary cameras)

Usage:
    from cv_pipeline.utils.cmc import CameraMotionCompensator

    cmc = CameraMotionCompensator(method="ecc")

    for frame in frames:
        # Get affine transformation matrix
        warp_matrix = cmc.compute(frame)

        # Apply to tracked bounding boxes
        compensated_boxes = cmc.apply_to_boxes(boxes, warp_matrix)
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseCMC(ABC):
    """Base class for Camera Motion Compensation methods."""

    def __init__(self, downscale: float = 2.0):
        """Initialize CMC.

        Args:
            downscale: Factor to downscale images for faster processing.
        """
        self.downscale = downscale
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None

    @abstractmethod
    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute the warp matrix for camera motion compensation.

        Args:
            frame: Current frame (H, W, C) in BGR format.

        Returns:
            Affine transformation matrix (2, 3) or identity if no motion.
        """
        pass

    def reset(self) -> None:
        """Reset the CMC state."""
        self.prev_frame = None
        self.prev_gray = None

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frame for motion estimation.

        Args:
            frame: Input frame (H, W, C).

        Returns:
            Tuple of (downscaled frame, grayscale frame).
        """
        # Downscale
        if self.downscale > 1:
            h, w = frame.shape[:2]
            new_h, new_w = int(h / self.downscale), int(w / self.downscale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        return frame, gray

    @staticmethod
    def identity_matrix() -> np.ndarray:
        """Return identity transformation matrix."""
        return np.eye(2, 3, dtype=np.float32)

    def apply_to_boxes(
        self,
        boxes: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply warp transformation to bounding boxes.

        Args:
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2].
            warp_matrix: Affine transformation matrix (2, 3).

        Returns:
            Transformed bounding boxes [N, 4].
        """
        if len(boxes) == 0:
            return boxes

        # Scale warp matrix if downscaling was used
        warp_matrix = warp_matrix.copy()
        warp_matrix[0, 2] *= self.downscale
        warp_matrix[1, 2] *= self.downscale

        # Convert boxes to corner points
        # Each box has 4 corners
        corners = np.zeros((len(boxes), 4, 2), dtype=np.float32)
        corners[:, 0] = boxes[:, [0, 1]]  # top-left
        corners[:, 1] = boxes[:, [2, 1]]  # top-right
        corners[:, 2] = boxes[:, [2, 3]]  # bottom-right
        corners[:, 3] = boxes[:, [0, 3]]  # bottom-left

        # Reshape for transformation
        corners = corners.reshape(-1, 2)

        # Apply affine transformation
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])  # Homogeneous coordinates
        transformed = corners_h @ warp_matrix.T

        # Reshape back to boxes
        transformed = transformed.reshape(-1, 4, 2)

        # Get new bounding boxes from transformed corners
        new_boxes = np.zeros_like(boxes)
        new_boxes[:, 0] = np.min(transformed[:, :, 0], axis=1)  # x1
        new_boxes[:, 1] = np.min(transformed[:, :, 1], axis=1)  # y1
        new_boxes[:, 2] = np.max(transformed[:, :, 0], axis=1)  # x2
        new_boxes[:, 3] = np.max(transformed[:, :, 1], axis=1)  # y2

        return new_boxes

    def apply_to_points(
        self,
        points: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply warp transformation to points.

        Args:
            points: Points [N, 2] in format [x, y].
            warp_matrix: Affine transformation matrix (2, 3).

        Returns:
            Transformed points [N, 2].
        """
        if len(points) == 0:
            return points

        # Scale warp matrix if downscaling was used
        warp_matrix = warp_matrix.copy()
        warp_matrix[0, 2] *= self.downscale
        warp_matrix[1, 2] *= self.downscale

        # Apply affine transformation
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])
        transformed = points_h @ warp_matrix.T

        return transformed


class ECCCMC(BaseCMC):
    """ECC (Enhanced Correlation Coefficient) based CMC.

    Most accurate method but computationally expensive.
    Best for high-accuracy tracking with stationary or slow-moving cameras.
    """

    def __init__(
        self,
        downscale: float = 2.0,
        num_iterations: int = 100,
        eps: float = 1e-5,
        motion_model: str = "affine",
    ):
        """Initialize ECC CMC.

        Args:
            downscale: Downscale factor.
            num_iterations: Maximum iterations for ECC optimization.
            eps: Convergence threshold.
            motion_model: Motion model ('translation', 'euclidean', 'affine', 'homography').
        """
        super().__init__(downscale)
        self.num_iterations = num_iterations
        self.eps = eps

        # Motion model mapping
        motion_models = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY,
        }
        self.motion_type = motion_models.get(motion_model, cv2.MOTION_AFFINE)

        # Termination criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            num_iterations,
            eps,
        )

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute warp matrix using ECC algorithm."""
        frame_proc, gray = self._preprocess(frame)

        if self.prev_gray is None:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Initialize warp matrix
        if self.motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # Compute ECC transform
            _, warp_matrix = cv2.findTransformECC(
                self.prev_gray,
                gray,
                warp_matrix,
                self.motion_type,
                self.criteria,
                None,
                5,  # Gaussian blur size
            )

            # Convert homography to affine if needed
            if self.motion_type == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = warp_matrix[:2, :]

        except cv2.error as e:
            logger.warning(f"ECC failed: {e}, returning identity")
            warp_matrix = self.identity_matrix()

        self.prev_frame = frame_proc
        self.prev_gray = gray

        return warp_matrix


class FeatureCMC(BaseCMC):
    """Feature-based CMC using keypoint matching.

    Faster than ECC and works well with textured scenes.
    Supports ORB, SIFT, and other OpenCV feature detectors.
    """

    def __init__(
        self,
        downscale: float = 2.0,
        detector: str = "orb",
        max_features: int = 1000,
        match_ratio: float = 0.75,
        ransac_thresh: float = 3.0,
    ):
        """Initialize Feature-based CMC.

        Args:
            downscale: Downscale factor.
            detector: Feature detector ('orb', 'sift', 'brisk').
            max_features: Maximum number of features to detect.
            match_ratio: Lowe's ratio test threshold.
            ransac_thresh: RANSAC inlier threshold.
        """
        super().__init__(downscale)
        self.match_ratio = match_ratio
        self.ransac_thresh = ransac_thresh

        # Initialize detector and matcher
        if detector.lower() == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif detector.lower() == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif detector.lower() == "brisk":
            self.detector = cv2.BRISK_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown detector: {detector}")

        self.prev_keypoints = None
        self.prev_descriptors = None

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute warp matrix using feature matching."""
        frame_proc, gray = self._preprocess(frame)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.prev_descriptors is None or descriptors is None:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.identity_matrix()

        if len(keypoints) < 4 or len(self.prev_keypoints) < 4:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.identity_matrix()

        # Match features
        try:
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        except cv2.error:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.identity_matrix()

        # Apply Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.identity_matrix()

        # Extract matched points
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate affine transformation with RANSAC
        warp_matrix, inliers = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )

        if warp_matrix is None:
            warp_matrix = self.identity_matrix()

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return warp_matrix


class OpticalFlowCMC(BaseCMC):
    """Sparse Optical Flow based CMC.

    Uses Lucas-Kanade optical flow on sparse feature points.
    Good balance of speed and accuracy.
    """

    def __init__(
        self,
        downscale: float = 2.0,
        max_corners: int = 1000,
        quality_level: float = 0.01,
        min_distance: float = 10,
        block_size: int = 7,
    ):
        """Initialize Optical Flow CMC.

        Args:
            downscale: Downscale factor.
            max_corners: Maximum number of corners to track.
            quality_level: Minimal accepted quality of corners.
            min_distance: Minimum distance between corners.
            block_size: Size of averaging block for corner detection.
        """
        super().__init__(downscale)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute warp matrix using optical flow."""
        frame_proc, gray = self._preprocess(frame)

        if self.prev_gray is None:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Detect corners in previous frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 4:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Filter good points
        if curr_pts is None:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        status = status.flatten()
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 4:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Estimate affine transformation
        warp_matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if warp_matrix is None:
            warp_matrix = self.identity_matrix()

        self.prev_frame = frame_proc
        self.prev_gray = gray

        return warp_matrix


class SparseOptFlowCMC(BaseCMC):
    """Sparse Optical Flow with grid-based sampling.

    More robust version that samples points from a regular grid
    instead of relying on corner detection.
    """

    def __init__(
        self,
        downscale: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ):
        """Initialize Sparse Optical Flow CMC.

        Args:
            downscale: Downscale factor.
            grid_size: Grid size (rows, cols) for sampling points.
        """
        super().__init__(downscale)
        self.grid_size = grid_size

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _create_grid_points(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create grid of sample points."""
        h, w = shape
        rows, cols = self.grid_size

        y_coords = np.linspace(h * 0.1, h * 0.9, rows)
        x_coords = np.linspace(w * 0.1, w * 0.9, cols)

        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])

        return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute warp matrix using grid-based optical flow."""
        frame_proc, gray = self._preprocess(frame)

        if self.prev_gray is None:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Create grid points
        prev_pts = self._create_grid_points(gray.shape[:2])

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Filter good points
        status = status.flatten()
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 4:
            self.prev_frame = frame_proc
            self.prev_gray = gray
            return self.identity_matrix()

        # Estimate affine transformation
        warp_matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if warp_matrix is None:
            warp_matrix = self.identity_matrix()

        self.prev_frame = frame_proc
        self.prev_gray = gray

        return warp_matrix


class NoCMC(BaseCMC):
    """No Camera Motion Compensation (placeholder).

    Returns identity transformation for stationary cameras.
    """

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Return identity matrix (no compensation)."""
        return self.identity_matrix()


class CameraMotionCompensator:
    """Factory and wrapper class for Camera Motion Compensation.

    Example:
        cmc = CameraMotionCompensator(method="ecc", downscale=2.0)

        for frame in frames:
            warp_matrix = cmc.compute(frame)
            compensated_boxes = cmc.apply_to_boxes(boxes)
    """

    METHODS = {
        "ecc": ECCCMC,
        "orb": lambda **kw: FeatureCMC(detector="orb", **kw),
        "sift": lambda **kw: FeatureCMC(detector="sift", **kw),
        "brisk": lambda **kw: FeatureCMC(detector="brisk", **kw),
        "optflow": OpticalFlowCMC,
        "sparseflow": SparseOptFlowCMC,
        "none": NoCMC,
    }

    def __init__(
        self,
        method: str = "sparseflow",
        downscale: float = 2.0,
        **kwargs,
    ):
        """Initialize Camera Motion Compensator.

        Args:
            method: CMC method ('ecc', 'orb', 'sift', 'optflow', 'sparseflow', 'none').
            downscale: Downscale factor for processing.
            **kwargs: Additional arguments for the CMC method.
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS.keys())}")

        self.method_name = method

        # Create CMC instance
        cmc_class = self.METHODS[method]
        if callable(cmc_class) and not isinstance(cmc_class, type):
            # Lambda wrapper
            self.cmc = cmc_class(downscale=downscale, **kwargs)
        else:
            self.cmc = cmc_class(downscale=downscale, **kwargs)

        # Store last warp matrix
        self.last_warp_matrix = BaseCMC.identity_matrix()

        # Motion history for smoothing
        self.motion_history: deque = deque(maxlen=5)

        logger.info(f"Initialized CMC with method: {method}")

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute camera motion compensation.

        Args:
            frame: Current frame (H, W, C) in BGR format.

        Returns:
            Affine transformation matrix (2, 3).
        """
        warp_matrix = self.cmc.compute(frame)
        self.last_warp_matrix = warp_matrix
        self.motion_history.append(warp_matrix)

        return warp_matrix

    def get_smoothed_warp(self) -> np.ndarray:
        """Get smoothed warp matrix using motion history.

        Returns:
            Smoothed affine transformation matrix (2, 3).
        """
        if not self.motion_history:
            return BaseCMC.identity_matrix()

        # Average the warp matrices
        matrices = np.array(list(self.motion_history))
        return np.mean(matrices, axis=0).astype(np.float32)

    def apply_to_boxes(
        self,
        boxes: np.ndarray,
        warp_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply compensation to bounding boxes.

        Args:
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2].
            warp_matrix: Optional warp matrix. Uses last computed if None.

        Returns:
            Compensated bounding boxes [N, 4].
        """
        if warp_matrix is None:
            warp_matrix = self.last_warp_matrix

        return self.cmc.apply_to_boxes(boxes, warp_matrix)

    def apply_to_points(
        self,
        points: np.ndarray,
        warp_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply compensation to points.

        Args:
            points: Points [N, 2] in format [x, y].
            warp_matrix: Optional warp matrix. Uses last computed if None.

        Returns:
            Compensated points [N, 2].
        """
        if warp_matrix is None:
            warp_matrix = self.last_warp_matrix

        return self.cmc.apply_to_points(points, warp_matrix)

    def reset(self) -> None:
        """Reset CMC state."""
        self.cmc.reset()
        self.motion_history.clear()
        self.last_warp_matrix = BaseCMC.identity_matrix()

    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available CMC methods."""
        return list(CameraMotionCompensator.METHODS.keys())


# Convenience exports
__all__ = [
    "BaseCMC",
    "ECCCMC",
    "FeatureCMC",
    "OpticalFlowCMC",
    "SparseOptFlowCMC",
    "NoCMC",
    "CameraMotionCompensator",
]
