"""Text detectors: CRAFT, DBNet, EAST, Differentiable Binarization."""
from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.detectors.clustering import cluster_and_merge_boxes

# Side-effect imports to self-register
from docuvision.detectors import (
    craft_detector,          # noqa: F401
    dbnet_detector,          # noqa: F401
    east_detector,           # noqa: F401
    paddle_detector,         # noqa: F401
    contour_detector,        # noqa: F401 — the always-available fallback
)

__all__ = [
    "BaseTextDetector",
    "DetectorRegistry",
    "cluster_and_merge_boxes",
]
