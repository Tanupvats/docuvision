"""PaddleOCR engine (Tier 2; supports CPU and GPU)."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import ensure_bgr
from docuvision.utils.lazy_import import require


# PaddleOCR uses its own language codes
_PADDLE_LANG_MAP = {
    "en": "en",
    "zh": "ch",
    "fr": "fr",
    "de": "german",
    "ja": "japan",
    "ko": "korean",
    "ru": "ru",
    "ar": "ar",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
}


@OCREngineRegistry.register
class PaddleOCREngine(BaseOCREngine):
    name = "paddleocr"
    requires = ["paddleocr"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = False, use_angle_cls: bool = True,
                 det: bool = True, rec: bool = True,
                 show_log: bool = False, **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=use_gpu, **kwargs)
        self.use_angle_cls = use_angle_cls
        self.det = det
        self.rec = rec
        self.show_log = show_log

    def _load(self) -> None:
        paddleocr = require("paddleocr", feature="PaddleOCR")
        # PaddleOCR only accepts a single language at init; pick first mapped one
        primary = self.languages[0] if self.languages else "en"
        lang = _PADDLE_LANG_MAP.get(primary, primary)
        self._model = paddleocr.PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=lang,
            use_gpu=self.use_gpu,
            show_log=self.show_log,
        )
        self.log.debug("PaddleOCR loaded; gpu=%s lang=%s", self.use_gpu, lang)

    def _predict(self, image: np.ndarray) -> OCRResult:
        bgr = ensure_bgr(image)
        # PaddleOCR accepts ndarray directly
        raw = self._model.ocr(bgr, det=self.det, rec=self.rec,
                              cls=self.use_angle_cls)

        regions: List[TextRegion] = []
        texts: List[str] = []

        # Paddle's output shape changed across versions; normalize defensively.
        # v2.6+: [ [ [quad, (text, conf)], ... ] ]
        # older: [ [quad, (text, conf)], ... ]
        if not raw:
            return OCRResult(text="", regions=[], engine=self.name)

        results = raw[0] if isinstance(raw[0], list) else raw
        if results is None:
            return OCRResult(text="", regions=[], engine=self.name)

        for entry in results:
            try:
                quad, (text, conf) = entry
            except Exception:
                # Defensive: skip malformed
                continue
            xs = [int(p[0]) for p in quad]
            ys = [int(p[1]) for p in quad]
            bbox = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            regions.append(TextRegion(
                bbox=bbox,
                text=text,
                confidence=float(conf),
                polygon=[(int(p[0]), int(p[1])) for p in quad],
            ))
            texts.append(text)

        return OCRResult(
            text="\n".join(texts),
            regions=regions,
            engine=self.name,
        )
