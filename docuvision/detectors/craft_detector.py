"""CRAFT text detector.

CRAFT is the scene-text detector originally published by Clova AI. The easiest
way to use it from Python is via EasyOCR's bundled CRAFT implementation —
it ships pretrained weights and an easy-to-invoke `.detect()` method.

If you have a standalone `craft-text-detector` package installed, we prefer it.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.types import BoundingBox, TextRegion
from docuvision.utils.image_io import ensure_rgb
from docuvision.utils.lazy_import import is_available, optional_import, require


@DetectorRegistry.register
class CRAFTDetector(BaseTextDetector):
    name = "craft"
    # We require *either* craft_text_detector or easyocr; the class-level
    # `requires` list is an AND-gate, so we override `is_available` below.
    requires: List[str] = []

    @classmethod
    def is_available(cls) -> bool:
        return is_available("craft_text_detector") or is_available("easyocr")

    def __init__(self, use_gpu: bool = False, **kwargs: Any) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self._backend: str = ""

    def _load(self) -> None:
        # Prefer the dedicated craft library
        ctd = optional_import("craft_text_detector")
        if ctd is not None:
            self._model = ctd.Craft(
                output_dir=None,
                crop_type="poly",
                cuda=self.use_gpu,
                text_threshold=self.kwargs.get("text_threshold", 0.7),
                link_threshold=self.kwargs.get("link_threshold", 0.4),
                low_text=self.kwargs.get("low_text", 0.4),
            )
            self._backend = "craft_text_detector"
        else:
            easyocr = require("easyocr",
                              feature="CRAFT detector via EasyOCR bundle")
            # Use EasyOCR's detector-only mode
            self._model = easyocr.Reader(
                ["en"], gpu=self.use_gpu,
                detector=True, recognizer=False, verbose=False,
            )
            self._backend = "easyocr"
        self.log.debug("CRAFT loaded via %s; gpu=%s", self._backend, self.use_gpu)

    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        rgb = ensure_rgb(image)
        regions: List[TextRegion] = []

        if self._backend == "craft_text_detector":
            pred = self._model.detect_text(rgb)
            for poly in pred.get("boxes", []):
                pts = [(int(p[0]), int(p[1])) for p in poly]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                regions.append(TextRegion(
                    bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                    polygon=pts,
                    confidence=1.0,
                ))
            return regions

        # easyocr backend
        boxes = self._model.detect(rgb)
        # EasyOCR .detect returns (horizontal_list, free_list) where
        # horizontal_list[0] is list of [x_min,x_max,y_min,y_max]
        if boxes and len(boxes) >= 1 and boxes[0]:
            for b in boxes[0][0]:
                x_min, x_max, y_min, y_max = map(int, b)
                regions.append(TextRegion(
                    bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    confidence=1.0,
                ))
        if boxes and len(boxes) >= 2 and boxes[1]:
            for poly in boxes[1][0]:
                pts = [(int(p[0]), int(p[1])) for p in poly]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                regions.append(TextRegion(
                    bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                    polygon=pts,
                    confidence=1.0,
                ))
        return regions
