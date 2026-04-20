"""DBNet (Differentiable Binarization) text detector.

Uses docTR's `db_resnet50` predictor as the canonical backend. Since docTR
is already a first-class dep for us, this detector is the closest to a
"free" modern detector we have.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.types import BoundingBox, TextRegion
from docuvision.utils.image_io import ensure_rgb
from docuvision.utils.lazy_import import require


@DetectorRegistry.register
class DBNetDetector(BaseTextDetector):
    """DBNet / Differentiable Binarization via docTR."""
    name = "dbnet"
    requires = ["doctr"]

    def __init__(self, use_gpu: bool = False,
                 arch: str = "db_resnet50",
                 pretrained: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.arch = arch
        self.pretrained = pretrained

    def _load(self) -> None:
        doctr_models = require("doctr.models", feature="DBNet via docTR")
        self._model = doctr_models.detection_predictor(
            arch=self.arch, pretrained=self.pretrained,
        )
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            except Exception as e:
                self.log.warning("Could not move DBNet to GPU: %s", e)
        self.log.debug("DBNet loaded; arch=%s gpu=%s", self.arch, self.use_gpu)

    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        rgb = ensure_rgb(image)
        h, w = rgb.shape[:2]
        out = self._model([rgb])

        regions: List[TextRegion] = []
        # docTR 0.7+: list of dicts with "words" or numpy arrays
        for page in out:
            boxes_arr = page["words"] if isinstance(page, dict) and "words" in page else page
            if isinstance(boxes_arr, np.ndarray):
                for b in boxes_arr:
                    # Format: [x_min, y_min, x_max, y_max, score]
                    x1, y1, x2, y2 = b[:4]
                    conf = float(b[4]) if len(b) >= 5 else 1.0
                    regions.append(TextRegion(
                        bbox=BoundingBox(
                            int(x1 * w), int(y1 * h),
                            int(x2 * w), int(y2 * h),
                        ),
                        confidence=conf,
                    ))
        return regions


# Alias: "Differentiable Binarization" is the same thing
@DetectorRegistry.register
class DBDetector(DBNetDetector):
    name = "db"
