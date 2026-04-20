"""TrOCR engine (Tier 4).

TrOCR is a recognition-only transformer — it does not localize text. For
full-page OCR we expect the pipeline to supply bounding boxes (from a text
detector). If called on a whole page without regions, we fall back to
treating the page as a single text line, which is rarely what you want.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import ensure_rgb
from docuvision.utils.lazy_import import require


@OCREngineRegistry.register
class TrOCREngine(BaseOCREngine):
    name = "trocr"
    requires = ["transformers", "torch"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = True,
                 model_name: str = "microsoft/trocr-base-printed",
                 max_new_tokens: int = 128,
                 **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=use_gpu, **kwargs)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._device: Any = None

    def _load(self) -> None:
        transformers = require("transformers", feature="TrOCR")
        torch = require("torch", feature="TrOCR")
        self._processor = transformers.TrOCRProcessor.from_pretrained(self.model_name)
        self._model = transformers.VisionEncoderDecoderModel.from_pretrained(self.model_name)
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device).eval()
        self.log.debug("TrOCR loaded on %s; model=%s", self._device, self.model_name)

    # ------------------------------------------------------------------
    def _recognize_crop(self, crop_rgb: np.ndarray) -> str:
        import torch
        inputs = self._processor(images=crop_rgb, return_tensors="pt").pixel_values
        inputs = inputs.to(self._device)
        with torch.no_grad():
            ids = self._model.generate(inputs, max_new_tokens=self.max_new_tokens)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()

    def _predict(self, image: np.ndarray) -> OCRResult:
        # Single-crop / full-image recognition. For doc-level OCR, prefer
        # calling with `regions=...` from a detector.
        rgb = ensure_rgb(image)
        text = self._recognize_crop(rgb)
        h, w = rgb.shape[:2]
        region = TextRegion(
            bbox=BoundingBox(0, 0, w, h),
            text=text,
            confidence=1.0,  # TrOCR doesn't expose a per-seq confidence
        )
        return OCRResult(
            text=text,
            regions=[region],
            engine=self.name,
            metadata={"model": self.model_name, "warning": "full-image mode"},
        )

    # ------------------------------------------------------------------
    def _predict_regions(self, img: np.ndarray,
                         regions: List[TextRegion]) -> OCRResult:
        """Override to batch region crops efficiently through TrOCR."""
        rgb = ensure_rgb(img)
        h, w = rgb.shape[:2]
        out_regions: List[TextRegion] = []
        texts: List[str] = []

        crops: List[np.ndarray] = []
        kept: List[TextRegion] = []
        for r in regions:
            bb = r.bbox
            x1 = max(0, min(w - 1, bb.x1))
            y1 = max(0, min(h - 1, bb.y1))
            x2 = max(x1 + 1, min(w, bb.x2))
            y2 = max(y1 + 1, min(h, bb.y2))
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            kept.append(r)

        if not crops:
            return OCRResult(text="", regions=[], engine=self.name)

        # Batched inference: 8 crops at a time to control memory
        batch_size = self.kwargs.get("batch_size", 8)
        import torch
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").pixel_values
            inputs = inputs.to(self._device)
            with torch.no_grad():
                ids = self._model.generate(inputs, max_new_tokens=self.max_new_tokens)
            decoded = self._processor.batch_decode(ids, skip_special_tokens=True)
            for r, text in zip(kept[i:i + batch_size], decoded):
                text = text.strip()
                out_regions.append(TextRegion(
                    bbox=r.bbox,
                    text=text,
                    confidence=r.confidence or 1.0,
                    polygon=r.polygon,
                ))
                if text:
                    texts.append(text)

        return OCRResult(
            text="\n".join(texts),
            regions=out_regions,
            engine=self.name,
            metadata={"model": self.model_name},
        )
