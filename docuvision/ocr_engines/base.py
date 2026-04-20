"""Abstract base class for OCR engines."""
from __future__ import annotations

import abc
import time
from typing import Any, List, Optional

import numpy as np

from docuvision.types import OCRResult, TextRegion
from docuvision.utils.image_io import ImageLike, load_image
from docuvision.utils.logging import get_logger


class BaseOCREngine(abc.ABC):
    """Every OCR backend subclasses this and implements `_predict`.

    Subclasses MUST set `name` (the registry key) and SHOULD set
    `requires` to a list of importable module names it needs.
    """

    name: str = "base"
    requires: List[str] = []
    default_languages: List[str] = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = False, **kwargs: Any) -> None:
        self.languages = languages or list(self.default_languages)
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self._model: Any = None
        self.log = get_logger(f"ocr.{self.name}")

    # ------------------------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        """Return True if all required modules are importable."""
        from docuvision.utils.lazy_import import is_available
        return all(is_available(m) for m in cls.requires)

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _load(self) -> None:
        """Lazily load the underlying model. Called once on first predict."""

    @abc.abstractmethod
    def _predict(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a BGR uint8 ndarray and return an OCRResult."""

    # ------------------------------------------------------------------
    def predict(self, image: ImageLike,
                regions: Optional[List[TextRegion]] = None) -> OCRResult:
        """Public entry point: handles image loading, timing, and region crops.

        If `regions` is provided, the engine runs on each cropped region
        instead of the whole image. This is how we combine detectors with
        recognition-only engines (e.g. TrOCR).
        """
        img = load_image(image)
        if self._model is None:
            self._load()

        t0 = time.perf_counter()
        if regions is None:
            result = self._predict(img)
        else:
            result = self._predict_regions(img, regions)
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        result.engine = self.name
        if result.language is None and self.languages:
            result.language = self.languages[0]
        return result

    # ------------------------------------------------------------------
    def _predict_regions(self, img: np.ndarray,
                         regions: List[TextRegion]) -> OCRResult:
        """Default: crop each region, call _predict, stitch results.

        Subclasses that natively accept regions should override this.
        """
        h, w = img.shape[:2]
        out_regions: List[TextRegion] = []
        texts: List[str] = []
        for r in regions:
            bb = r.bbox
            x1 = max(0, min(w - 1, bb.x1))
            y1 = max(0, min(h - 1, bb.y1))
            x2 = max(x1 + 1, min(w, bb.x2))
            y2 = max(y1 + 1, min(h, bb.y2))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                sub = self._predict(crop)
            except Exception as e:
                self.log.warning("OCR on crop failed: %s", e)
                continue
            if sub.text:
                texts.append(sub.text.strip())
            new_region = TextRegion(
                bbox=r.bbox,
                text=sub.text.strip() if sub.text else "",
                confidence=(
                    sub.regions[0].confidence
                    if sub.regions
                    else 0.0
                ),
                polygon=r.polygon,
            )
            out_regions.append(new_region)
        return OCRResult(
            text="\n".join(texts),
            regions=out_regions,
            engine=self.name,
        )
