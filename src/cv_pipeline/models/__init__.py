"""CV Pipeline Models.

This package contains model implementations for:
- Re-ID feature extraction for BoT-SORT tracking
"""

from cv_pipeline.models.reid import FastReIDExtractor, OSNetExtractor, ReIDExtractor

__all__ = [
    "ReIDExtractor",
    "OSNetExtractor",
    "FastReIDExtractor",
]
