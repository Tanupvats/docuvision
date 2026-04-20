"""PaddleOCR detector-only wrapper.

PaddleOCR's text detector (DB++ by default) is a reasonable fallback when
`doctr` isn't around but `paddleocr` is.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.types import BoundingBox, TextRegion
from docuvision.utils.image_io import ensure_bgr
from docuvision.utils.lazy_import import require


@DetectorRegistry.register
class PaddleDetector(BaseTextDetector):
    name = "paddle_det"
    requires = ["paddleocr"]

    def __init__(self, use_gpu: bool = False, lang: str = "en",
                 **kwargs: Any) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.lang = lang

    def _load(self) -> None:
        paddleocr = require("paddleocr", feature="PaddleDetector")
        self._model = paddleocr.PaddleOCR(
            use_angle_cls=False,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )
        self.log.debug("Paddle detector loaded; gpu=%s lang=%s",
                       self.use_gpu, self.lang)

    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        bgr = ensure_bgr(image)
        raw = self._model.ocr(bgr, det=True, rec=False, cls=False)
        regions: List[TextRegion] = []
        if not raw:
            return regions
        results = raw[0] if (raw and isinstance(raw[0], list)) else raw
        if results is None:
            return regions
        for item in results:
            # When rec=False, Paddle returns just the quads
            quad = item if isinstance(item, list) else item[0]
            try:
                xs = [int(p[0]) for p in quad]
                ys = [int(p[1]) for p in quad]
            except Exception:
                continue
            regions.append(TextRegion(
                bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                polygon=[(int(p[0]), int(p[1])) for p in quad],
                confidence=1.0,
            ))
        return regions
