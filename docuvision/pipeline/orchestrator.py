"""
docuvision.pipeline.orchestrator
================================

Top-level DocumentPipeline — the user-facing façade that wires together:

    system profile → tier/engine selection
                  → (optional) document classification
                  → (optional) embossed-mode OCR  [short-circuits the rest]
                  → (optional) text detection + clustering
                  → OCR recognition
                  → (optional) masking

Design goals:
    - Zero-config: `DocumentPipeline().run(image)` just works.
    - Explicit: every knob is exposed on `PipelineConfig`.
    - Lazy: engines/detectors/models are only built when they'll be used.
    - Safe: any single stage failing degrades the result but never crashes
      the pipeline — failures are captured in `PipelineResult.metadata`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from docuvision.classifier import DocumentClassifier
from docuvision.detectors import DetectorRegistry, cluster_and_merge_boxes
from docuvision.embossed import EmbossedOCRPipeline
from docuvision.masking import DocumentMasker
from docuvision.ocr_engines import OCREngineRegistry
from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.system_profiler import SystemCapabilityReport, profile_system
from docuvision.types import OCRResult, PipelineResult, TextRegion
from docuvision.utils.image_io import ImageLike, load_image
from docuvision.utils.logging import get_logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # --- Stage flags -----------------------------------------------------
    detect_text: bool = True
    classify_doc: bool = False
    mask: bool = False
    embossed_mode: bool = False

    # --- Language(s) -----------------------------------------------------
    language: List[str] = field(default_factory=lambda: ["en"])

    # --- Engine overrides ------------------------------------------------
    ocr_engine: Optional[str] = None          # e.g. "paddleocr", "trocr"
    detector: Optional[str] = None            # e.g. "craft", "dbnet"
    use_gpu: Optional[bool] = None            # None → auto from profile

    # --- Classifier / masker config -------------------------------------
    classifier_labels: Optional[List[str]] = None
    classifier_model_path: Optional[str] = None
    mask_method: str = "gaussian"             # "gaussian" | "blackbox" | "pixelate"
    mask_labels: Optional[List[str]] = None
    mask_model_path: Optional[str] = None

    # --- Detection / clustering -----------------------------------------
    cluster_boxes: bool = True
    cluster_margin: int = 4
    cluster_algorithm: str = "dbscan"         # "dbscan" | "hdbscan"

    # --- Misc -----------------------------------------------------------
    tier_override: Optional[int] = None       # force a tier 0-4
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    detector_kwargs: Dict[str, Any] = field(default_factory=dict)
    return_preprocessed_image: bool = False


# ---------------------------------------------------------------------------
# Engine selection helper
# ---------------------------------------------------------------------------
_TIER_TO_ENGINE: Dict[int, List[str]] = {
    0: ["tesseract"],
    1: ["easyocr", "tesseract"],
    2: ["paddleocr", "easyocr", "tesseract"],
    3: ["doctr", "paddleocr", "easyocr", "tesseract"],
    4: ["trocr", "doctr", "paddleocr", "easyocr", "tesseract"],
}


def _engine_order_for_tier(tier: int) -> List[str]:
    tier = max(0, min(4, tier))
    return _TIER_TO_ENGINE[tier]


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
class DocumentPipeline:
    """High-level orchestrator.

    Minimal example
    ---------------
    >>> from docuvision import DocumentPipeline
    >>> pipe = DocumentPipeline()          # zero-config
    >>> out = pipe.run("invoice.png")
    >>> print(out.text)

    Full example
    ------------
    >>> pipe = DocumentPipeline(
    ...     detect_text=True,
    ...     classify_doc=True,
    ...     mask=True,
    ...     embossed_mode=False,
    ...     language=["en", "hi"],
    ...     ocr_engine="paddleocr",
    ... )
    >>> out = pipe.run("aadhaar.jpg")
    >>> print(out.doc_class, out.text)
    """

    def __init__(self,
                 detect_text: bool = True,
                 classify_doc: bool = False,
                 mask: bool = False,
                 embossed_mode: bool = False,
                 language: Optional[List[str]] = None,
                 config: Optional[PipelineConfig] = None,
                 report: Optional[SystemCapabilityReport] = None,
                 **kwargs: Any) -> None:
        # Let users pass either a full PipelineConfig or keyword shortcuts
        if config is None:
            config = PipelineConfig(
                detect_text=detect_text,
                classify_doc=classify_doc,
                mask=mask,
                embossed_mode=embossed_mode,
                language=language or ["en"],
                **kwargs,
            )
        self.config = config
        self.log = get_logger("pipeline")
        self.report = report or profile_system()

        # Resolve use_gpu
        if self.config.use_gpu is None:
            self.config.use_gpu = self.report.gpu_available and self.report.vram_total_gb >= 3.0

        # Lazy-built members
        self._ocr: Optional[BaseOCREngine] = None
        self._detector: Any = None
        self._classifier: Optional[DocumentClassifier] = None
        self._masker: Optional[DocumentMasker] = None
        self._embossed: Optional[EmbossedOCRPipeline] = None

        self.log.debug("Pipeline init: tier=%s, recommended=%s, use_gpu=%s",
                       self.report.tier, self.report.recommended_pipeline,
                       self.config.use_gpu)

    # ------------------------------------------------------------------
    # Lazy constructors
    # ------------------------------------------------------------------
    def _build_ocr(self) -> BaseOCREngine:
        if self._ocr is not None:
            return self._ocr

        tier = self.config.tier_override
        if tier is None:
            tier = self.report.tier

        fallback_order = _engine_order_for_tier(tier)

        preferred = self.config.ocr_engine or self.report.recommended_pipeline
        if preferred == "none":
            preferred = None

        self._ocr = OCREngineRegistry.auto_build(
            preferred=preferred,
            fallback_order=fallback_order,
            languages=self.config.language,
            use_gpu=self.config.use_gpu,
            **self.config.engine_kwargs,
        )
        self.log.info("OCR engine: %s (tier %s)", self._ocr.name, tier)
        return self._ocr

    def _build_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        self._detector = DetectorRegistry.auto_build(
            preferred=self.config.detector,
            use_gpu=self.config.use_gpu,
            **self.config.detector_kwargs,
        )
        self.log.info("Text detector: %s", self._detector.name)
        return self._detector

    def _build_classifier(self) -> DocumentClassifier:
        if self._classifier is not None:
            return self._classifier
        self._classifier = DocumentClassifier(
            labels=self.config.classifier_labels,
            model_path=self.config.classifier_model_path,
            use_gpu=self.config.use_gpu,
            keyword_ocr_engine=self._ocr,   # reuse if already built
        )
        self.log.info("Classifier backend: %s", self._classifier.backend)
        return self._classifier

    def _build_masker(self) -> DocumentMasker:
        if self._masker is not None:
            return self._masker
        self._masker = DocumentMasker(
            labels=self.config.mask_labels,
            model_path=self.config.mask_model_path,
            method=self.config.mask_method,
            use_gpu=self.config.use_gpu,
            ocr_engine=self._ocr,
        )
        self.log.info("Masker backend: %s", self._masker.backend)
        return self._masker

    def _build_embossed(self) -> EmbossedOCRPipeline:
        if self._embossed is not None:
            return self._embossed
        self._embossed = EmbossedOCRPipeline(
            use_gpu=self.config.use_gpu,
        )
        return self._embossed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image: ImageLike) -> PipelineResult:
        """Run the configured pipeline on a single image.

        Returns a `PipelineResult` with the fields populated according to
        which stages were enabled in the config.
        """
        t0 = time.perf_counter()
        img = load_image(image)
        result = PipelineResult(
            tier_used=self.config.tier_override or self.report.tier,
        )
        failures: Dict[str, str] = {}

        # ------------------ 1. Embossed mode short-circuit --------------
        if self.config.embossed_mode:
            try:
                emb = self._build_embossed()
                ocr = emb.run(img)
                result.ocr = ocr
                result.text = ocr.text
                result.regions = ocr.regions
                result.engine_used = ocr.engine
            except Exception as e:
                self.log.exception("Embossed pipeline failed: %s", e)
                failures["embossed"] = repr(e)

            # Classification (optional) runs even in embossed mode
            if self.config.classify_doc and "embossed" not in failures:
                try:
                    result.doc_class = self._build_classifier().predict(img)
                except Exception as e:
                    self.log.exception("Classifier failed: %s", e)
                    failures["classify"] = repr(e)

            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            result.metadata["failures"] = failures
            return result

        # ------------------ 2. Classification ---------------------------
        if self.config.classify_doc:
            try:
                result.doc_class = self._build_classifier().predict(img)
            except Exception as e:
                self.log.exception("Classifier failed: %s", e)
                failures["classify"] = repr(e)

        # ------------------ 3. Text detection ---------------------------
        detected_regions: Optional[List[TextRegion]] = None
        if self.config.detect_text:
            try:
                detector = self._build_detector()
                raw = detector.detect(img)
                if raw and self.config.cluster_boxes:
                    detected_regions = cluster_and_merge_boxes(
                        raw,
                        image_shape=img.shape,
                        algorithm=self.config.cluster_algorithm,
                        margin=self.config.cluster_margin,
                    )
                else:
                    detected_regions = raw
            except Exception as e:
                self.log.exception("Text detector failed: %s", e)
                failures["detect"] = repr(e)
                detected_regions = None

        # ------------------ 4. OCR recognition --------------------------
        try:
            ocr_engine = self._build_ocr()
            if detected_regions is not None and detected_regions:
                ocr_result = ocr_engine.predict(img, regions=detected_regions)
            else:
                ocr_result = ocr_engine.predict(img)
            result.ocr = ocr_result
            result.text = ocr_result.text
            result.regions = ocr_result.regions
            result.engine_used = ocr_result.engine
        except Exception as e:
            self.log.exception("OCR failed: %s", e)
            failures["ocr"] = repr(e)
            result.ocr = OCRResult(text="", regions=[])

        # ------------------ 5. Masking ----------------------------------
        if self.config.mask:
            try:
                masker = self._build_masker()
                mask_regions = masker.detect(img)
                masked_img = masker.apply_mask(img, regions=mask_regions)
                result.masks = mask_regions
                result.masked_image = masked_img
            except Exception as e:
                self.log.exception("Masking failed: %s", e)
                failures["mask"] = repr(e)

        # ------------------ Finalize ------------------------------------
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        if failures:
            result.metadata["failures"] = failures
        result.metadata["config"] = {
            "detect_text": self.config.detect_text,
            "classify_doc": self.config.classify_doc,
            "mask": self.config.mask,
            "embossed_mode": self.config.embossed_mode,
            "language": self.config.language,
        }

        if self.config.return_preprocessed_image:
            result.preprocessed_image = img

        return result

    # ------------------------------------------------------------------
    def run_batch(self, images: List[ImageLike]) -> List[PipelineResult]:
        """Run the pipeline on a list of images. No parallelism — engines
        are typically already internally batched or GIL-bound.
        """
        return [self.run(img) for img in images]
