"""
Contour-based text detector — pure OpenCV, no deep-learning dependencies.

This is a deliberately-crude but *always available* fallback so that
`DetectorRegistry.auto_build()` can always succeed. It's useful for:
    - dev environments without deep-learning libs
    - very constrained edge deployments
    - engraved/embossed plates where heavier detectors struggle with
      low-contrast text anyway

Algorithm:
    grayscale → CLAHE → adaptive threshold → dilate horizontally
    → contours → filter by aspect ratio and area
"""
from __future__ import annotations

from typing import Any, List

import cv2
import numpy as np

from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.types import BoundingBox, TextRegion


@DetectorRegistry.register
class ContourDetector(BaseTextDetector):
    name = "contour"
    requires: List[str] = []  # cv2 is a core dep — always available

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __init__(self, use_gpu: bool = False,
                 min_area: int = 100,
                 max_aspect: float = 50.0,
                 min_aspect: float = 0.1,
                 dilate_kernel: tuple = (15, 3),
                 **kwargs: Any) -> None:
        super().__init__(use_gpu=False, **kwargs)
        self.min_area = min_area
        self.max_aspect = max_aspect
        self.min_aspect = min_aspect
        self.dilate_kernel = dilate_kernel

    def _load(self) -> None:
        self._model = True  # no model

    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_kernel)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        regions: List[TextRegion] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < self.min_area:
                continue
            aspect = w / float(max(1, h))
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            regions.append(TextRegion(
                bbox=BoundingBox(x, y, x + w, y + h),
                confidence=0.5,  # low-confidence prior
            ))
        return regions
