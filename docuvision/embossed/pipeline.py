"""
Embossed / engraved OCR pipeline.

This module composes the embossed preprocessor with a detector and an OCR
recognizer, in the right order, with a couple of domain-specific tricks:

- Because embossed text is usually short (chassis numbers, serials), the
  pipeline biases toward UPPERCASE alphanumeric recognition and applies a
  post-processing step that removes obvious non-alphanumeric artifacts.
- The preprocessor is run *before* detection so the detector sees a
  normalized image; then the detected regions are cropped from the
  *preprocessed* image (not the original) before OCR.
- Two recognition runs are performed: one on the preprocessed crop and one
  on the raw crop. The higher-confidence result wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np

from docuvision.detectors.clustering import cluster_and_merge_boxes
from docuvision.detectors.registry import DetectorRegistry
from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.preprocessing.embossed import (
    EmbossedPreprocessConfig,
    embossed_preprocess,
)
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import ImageLike, load_image
from docuvision.utils.logging import get_logger

log = get_logger("embossed")


@dataclass
class EmbossedPipelineConfig:
    preprocess: EmbossedPreprocessConfig = None  # type: ignore[assignment]
    detector_name: Optional[str] = None     # None → auto
    ocr_name: Optional[str] = None          # None → auto
    cluster_boxes: bool = True
    margin: int = 6
    alphanumeric_only: bool = True
    min_confidence: float = 0.0


class EmbossedOCRPipeline:
    """Specialized pipeline for embossed / engraved text.

    Usage
    -----
    >>> pipe = EmbossedOCRPipeline(use_gpu=False)
    >>> result = pipe.run("chassis_plate.jpg")
    >>> print(result.text)
    """
    # Chassis-like token: uppercase alphanumerics, 8–17 chars.
    CHASSIS_PATTERN = re.compile(r"[A-Z0-9]{6,17}")

    def __init__(self,
                 use_gpu: bool = False,
                 config: Optional[EmbossedPipelineConfig] = None,
                 preprocess_config: Optional[EmbossedPreprocessConfig] = None,
                 detector: Optional[Any] = None,
                 ocr_engine: Optional[BaseOCREngine] = None,
                 **kwargs: Any) -> None:
        self.use_gpu = use_gpu
        self.config = config or EmbossedPipelineConfig()
        if preprocess_config is not None:
            self.config.preprocess = preprocess_config
        if self.config.preprocess is None:
            self.config.preprocess = EmbossedPreprocessConfig()

        self._detector = detector
        self._ocr = ocr_engine
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    def _ensure_models(self) -> None:
        if self._detector is None:
            # Contour detector works surprisingly well on embossed plates
            # because text is usually on a relatively uniform background.
            self._detector = DetectorRegistry.auto_build(
                preferred=self.config.detector_name or "contour",
                use_gpu=self.use_gpu,
            )
        if self._ocr is None:
            # Tesseract is preferred for short alphanumeric strings with the
            # right config; fall back to whatever is available.
            try:
                self._ocr = OCREngineRegistry.build(
                    self.config.ocr_name or "tesseract",
                    languages=["en"], use_gpu=self.use_gpu,
                    psm=7,  # single line
                    config_extra="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
            except Exception:
                self._ocr = OCREngineRegistry.auto_build(
                    preferred=self.config.ocr_name,
                    use_gpu=self.use_gpu,
                )

    # ------------------------------------------------------------------
    def _recognize(self, engine: BaseOCREngine,
                   preproc: np.ndarray, raw_crop: np.ndarray) -> OCRResult:
        """Run OCR twice (preprocessed + raw), return whichever looks better."""
        try:
            r_pre = engine.predict(preproc)
        except Exception as e:
            log.debug("OCR on preprocessed crop failed: %s", e)
            r_pre = OCRResult(text="", regions=[])
        try:
            r_raw = engine.predict(raw_crop)
        except Exception as e:
            log.debug("OCR on raw crop failed: %s", e)
            r_raw = OCRResult(text="", regions=[])

        candidates = [r_pre, r_raw]
        # Score candidates by mean confidence * number of alphanumerics
        def score(r: OCRResult) -> float:
            if not r or not r.text:
                return 0.0
            alnum = sum(ch.isalnum() for ch in r.text)
            conf = (
                float(np.mean([x.confidence for x in r.regions]))
                if r.regions else 0.0
            )
            return conf * (1 + alnum / 10.0)

        return max(candidates, key=score)

    # ------------------------------------------------------------------
    def _postprocess_text(self, text: str) -> str:
        if not self.config.alphanumeric_only:
            return text.strip()
        cleaned = text.upper()
        # Keep only alphanumerics and whitespace separators
        cleaned = re.sub(r"[^A-Z0-9\s]", "", cleaned)
        return cleaned.strip()

    # ------------------------------------------------------------------
    def run(self, image: ImageLike) -> OCRResult:
        """Run the full embossed pipeline. Returns a single OCRResult."""
        self._ensure_models()

        img = load_image(image)
        h, w = img.shape[:2]

        # 1. Preprocess
        preproc_bin = embossed_preprocess(img, self.config.preprocess)
        # Convert to 3-channel BGR so it's compatible with engines that expect it
        preproc_bgr = cv2.cvtColor(preproc_bin, cv2.COLOR_GRAY2BGR)

        # 2. Detect text regions on the preprocessed image
        try:
            raw_regions = self._detector.detect(preproc_bgr)
        except Exception as e:
            log.warning("Detector failed on preprocessed image: %s", e)
            raw_regions = []

        if not raw_regions:
            # Fall back to single-region full-image OCR
            final = self._recognize(self._ocr, preproc_bgr, img)
            final.text = self._postprocess_text(final.text)
            final.metadata["embossed"] = True
            final.metadata["regions_detected"] = 0
            return final

        # 3. Cluster and merge
        if self.config.cluster_boxes:
            regions = cluster_and_merge_boxes(
                raw_regions,
                image_shape=img.shape,
                margin=self.config.margin,
            )
        else:
            regions = raw_regions

        # 4. Recognize each region — run OCR on both preprocessed and raw crops
        final_regions: List[TextRegion] = []
        texts: List[str] = []
        for r in regions:
            bb = r.bbox
            x1, y1, x2, y2 = (max(0, bb.x1), max(0, bb.y1),
                              min(w, bb.x2), min(h, bb.y2))
            raw_crop = img[y1:y2, x1:x2]
            pre_crop = preproc_bgr[y1:y2, x1:x2]
            if raw_crop.size == 0:
                continue
            res = self._recognize(self._ocr, pre_crop, raw_crop)
            cleaned = self._postprocess_text(res.text)
            if not cleaned:
                continue
            avg_conf = (
                float(np.mean([x.confidence for x in res.regions]))
                if res.regions else 0.0
            )
            if avg_conf < self.config.min_confidence:
                continue
            final_regions.append(TextRegion(
                bbox=BoundingBox(x1, y1, x2, y2),
                text=cleaned,
                confidence=avg_conf,
            ))
            texts.append(cleaned)

        # 5. Extract chassis-like tokens into metadata
        combined = " ".join(texts)
        chassis_candidates = self.CHASSIS_PATTERN.findall(combined)

        return OCRResult(
            text="\n".join(texts),
            regions=final_regions,
            engine=f"embossed+{self._ocr.name}",
            metadata={
                "embossed": True,
                "chassis_candidates": chassis_candidates,
                "regions_detected": len(regions),
            },
        )
