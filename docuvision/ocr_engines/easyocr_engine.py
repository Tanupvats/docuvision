"""EasyOCR engine (Tier 1)."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import ensure_rgb
from docuvision.utils.lazy_import import require


@OCREngineRegistry.register
class EasyOCREngine(BaseOCREngine):
    name = "easyocr"
    requires = ["easyocr"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = False, detector: bool = True,
                 recognizer: bool = True, model_storage_dir: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=use_gpu, **kwargs)
        self.detector = detector
        self.recognizer = recognizer
        self.model_storage_dir = model_storage_dir

    def _load(self) -> None:
        easyocr = require("easyocr", feature="EasyOCR")
        self._model = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu,
            detector=self.detector,
            recognizer=self.recognizer,
            model_storage_directory=self.model_storage_dir,
            verbose=False,
        )
        self.log.debug("EasyOCR loaded; gpu=%s langs=%s", self.use_gpu, self.languages)

    def _predict(self, image: np.ndarray) -> OCRResult:
        rgb = ensure_rgb(image)
        raw = self._model.readtext(rgb, detail=1, paragraph=False)

        regions: List[TextRegion] = []
        texts: List[str] = []
        for item in raw:
            # EasyOCR returns [[quad], text, conf]
            quad, text, conf = item
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
