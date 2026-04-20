"""docTR OCR engine (Tier 3)."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import ensure_rgb
from docuvision.utils.lazy_import import require


@OCREngineRegistry.register
class DoctrEngine(BaseOCREngine):
    name = "doctr"
    requires = ["doctr"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = False,
                 det_arch: str = "db_resnet50",
                 reco_arch: str = "crnn_vgg16_bn",
                 pretrained: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=use_gpu, **kwargs)
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained

    def _load(self) -> None:
        doctr_models = require("doctr.models", feature="docTR")
        # docTR autodetects torch/TF; we trust it.
        self._model = doctr_models.ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained,
        )
        # GPU placement — best-effort
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            except Exception as e:
                self.log.warning("Could not move docTR to GPU: %s", e)
        self.log.debug("docTR loaded; det=%s reco=%s", self.det_arch, self.reco_arch)

    def _predict(self, image: np.ndarray) -> OCRResult:
        # docTR expects a list of RGB numpy arrays
        rgb = ensure_rgb(image)
        h, w = rgb.shape[:2]
        result = self._model([rgb])

        regions: List[TextRegion] = []
        text_lines: List[str] = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_words: List[str] = []
                    for word in line.words:
                        # geometry is ((x1,y1),(x2,y2)) in normalized coords
                        (rx1, ry1), (rx2, ry2) = word.geometry
                        x1, y1 = int(rx1 * w), int(ry1 * h)
                        x2, y2 = int(rx2 * w), int(ry2 * h)
                        regions.append(TextRegion(
                            bbox=BoundingBox(x1, y1, x2, y2),
                            text=word.value,
                            confidence=float(word.confidence),
                        ))
                        line_words.append(word.value)
                    if line_words:
                        text_lines.append(" ".join(line_words))

        return OCRResult(
            text="\n".join(text_lines),
            regions=regions,
            engine=self.name,
            metadata={"det_arch": self.det_arch, "reco_arch": self.reco_arch},
        )
