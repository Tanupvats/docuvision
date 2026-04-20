"""Tesseract OCR engine (Tier 0 fallback)."""
from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.lazy_import import require


# ISO 639-1 → Tesseract 3-letter language codes (best-effort)
_TESS_LANG_MAP = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "ru": "rus", "ja": "jpn", "ko": "kor",
    "zh": "chi_sim", "ar": "ara", "hi": "hin", "bn": "ben", "ta": "tam",
    "te": "tel", "mr": "mar", "kn": "kan", "gu": "guj",
}


@OCREngineRegistry.register
class TesseractEngine(BaseOCREngine):
    name = "tesseract"
    requires = ["pytesseract"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = False, psm: int = 6, oem: int = 3,
                 config_extra: str = "", binary_path: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=False, **kwargs)
        self.psm = psm
        self.oem = oem
        self.config_extra = config_extra
        self.binary_path = binary_path

    def _load(self) -> None:
        pt = require("pytesseract", feature="Tesseract OCR")
        if self.binary_path:
            pt.pytesseract.tesseract_cmd = self.binary_path
        # Sentinel so base class doesn't reload
        self._model = pt
        self.log.debug("Tesseract loaded; binary=%s",
                       pt.pytesseract.tesseract_cmd)

    # ------------------------------------------------------------------
    def _lang_string(self) -> str:
        return "+".join(_TESS_LANG_MAP.get(l, l) for l in self.languages) or "eng"

    def _tess_config(self) -> str:
        return f"--oem {self.oem} --psm {self.psm} {self.config_extra}".strip()

    def _predict(self, image: np.ndarray) -> OCRResult:
        pt = self._model
        # Grayscale helps Tesseract; keep color for language detection scripts.
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        lang = self._lang_string()
        cfg = self._tess_config()

        data = pt.image_to_data(gray, lang=lang, config=cfg,
                                output_type=pt.Output.DICT)

        regions: List[TextRegion] = []
        text_lines: List[str] = []
        current_line: List[str] = []
        last_block = (-1, -1, -1)  # (block, par, line)

        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i]
            conf = data["conf"][i]
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = -1.0
            if not txt or txt.strip() == "" or conf_f < 0:
                continue

            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            regions.append(TextRegion(
                bbox=BoundingBox(x, y, x + w, y + h),
                text=txt,
                confidence=conf_f / 100.0,
            ))

            block_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if block_id != last_block and current_line:
                text_lines.append(" ".join(current_line))
                current_line = []
            current_line.append(txt)
            last_block = block_id

        if current_line:
            text_lines.append(" ".join(current_line))

        return OCRResult(
            text="\n".join(text_lines),
            regions=regions,
            engine=self.name,
            language=self.languages[0] if self.languages else None,
            metadata={"psm": self.psm, "oem": self.oem},
        )
