"""
Document masker — detect sensitive regions and redact them.

Backends for region detection:

1. **YOLO model** — pass `model_path=...` pointing to a PII detector.
2. **Regex over OCR** — always available. Uses OCR output to find phones,
   ID numbers, VINs, etc. and maps text back to bounding boxes.
3. **User-supplied boxes** — call `apply_mask(image, regions=[...])` directly.

Mask methods: gaussian blur, solid black box, pixelation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

import cv2
import numpy as np

from docuvision.types import BoundingBox, MaskRegion, TextRegion
from docuvision.utils.image_io import ImageLike, ensure_rgb, load_image
from docuvision.utils.lazy_import import optional_import
from docuvision.utils.logging import get_logger


class MaskMethod(str, Enum):
    GAUSSIAN = "gaussian"
    BLACKBOX = "blackbox"
    PIXELATE = "pixelate"


DEFAULT_PII_LABELS: List[str] = [
    "name",
    "id_number",
    "address",
    "phone",
    "email",
    "vin",
    "chassis_number",
    "signature",
    "dob",
]


# ---------------------------------------------------------------------------
# Regex library (compiled once)
# ---------------------------------------------------------------------------
_REGEX_RULES: Dict[str, Pattern] = {
    # Aadhaar: 12 digits optionally spaced/hyphenated in groups of 4
    "id_number":  re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    # PAN (India): 5 letters, 4 digits, 1 letter
    # (alias under the same 'id_number' label)
    # Handled via separate rule below for cleanliness
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{0,4}\b"
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # VIN: 17 chars, no I/O/Q
    "vin": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    # Vehicle chassis — very similar to VIN but often shorter in India
    "chassis_number": re.compile(r"\b[A-HJ-NPR-Z0-9]{10,17}\b"),
    "dob": re.compile(r"\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"),
}

# PAN — handled separately to avoid clobbering id_number's 12-digit match
_PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


@dataclass
class _TokenMatch:
    label: str
    start: int
    end: int
    text: str


class DocumentMasker:
    def __init__(self,
                 labels: Optional[List[str]] = None,
                 model_path: Optional[str] = None,
                 method: str = MaskMethod.GAUSSIAN.value,
                 blur_ksize: int = 31,
                 pixelate_blocks: int = 10,
                 use_gpu: bool = False,
                 ocr_engine: Optional[Any] = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        labels : list of str
            Which PII labels to mask. Defaults to `DEFAULT_PII_LABELS`.
        model_path : str, optional
            Path to a YOLO PII detector (.pt). If given, uses YOLO backend.
        method : MaskMethod
            Redaction method. Can be overridden per-call in `apply_mask`.
        blur_ksize : int
            Gaussian kernel size (odd integer).
        pixelate_blocks : int
            Number of blocks per side for pixelation.
        ocr_engine : BaseOCREngine, optional
            Used by the regex backend. If None, auto-selected.
        """
        self.labels = set(labels or DEFAULT_PII_LABELS)
        self.model_path = model_path
        self.method = MaskMethod(method)
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.pixelate_blocks = max(2, pixelate_blocks)
        self.use_gpu = use_gpu
        self._model: Any = None
        self._ocr_engine = ocr_engine
        self.backend = "yolo" if model_path else "regex"
        self.log = get_logger("masker")
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "yolo":
            ul = optional_import("ultralytics")
            if ul is None:
                raise ImportError("Install `ultralytics` to use the YOLO PII backend.")
            self._model = ul.YOLO(self.model_path)
        else:
            if self._ocr_engine is None:
                from docuvision.ocr_engines.registry import OCREngineRegistry
                try:
                    self._ocr_engine = OCREngineRegistry.auto_build()
                except RuntimeError as e:
                    self.log.warning(
                        "No OCR engine for regex masker: %s — detect() will "
                        "return no regions.", e,
                    )
                    self._ocr_engine = None
            self._model = True

    # ------------------------------------------------------------------
    def detect(self, image: ImageLike) -> List[MaskRegion]:
        """Detect PII regions in the image."""
        self._load()
        img = load_image(image)
        if self.backend == "yolo":
            return self._detect_yolo(img)
        return self._detect_regex(img)

    # ------------------------------------------------------------------
    def _detect_yolo(self, img: np.ndarray) -> List[MaskRegion]:
        rgb = ensure_rgb(img)
        results = self._model(rgb, verbose=False)
        r = results[0]
        out: List[MaskRegion] = []
        if r.boxes is None:
            return out
        names = r.names if hasattr(r, "names") else {}
        for box, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            label = names.get(int(cls_id), str(cls_id))
            if self.labels and label not in self.labels:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            out.append(MaskRegion(
                bbox=BoundingBox(x1, y1, x2, y2),
                label=label,
                confidence=float(conf),
            ))
        return out

    # ------------------------------------------------------------------
    def _detect_regex(self, img: np.ndarray) -> List[MaskRegion]:
        if self._ocr_engine is None:
            return []
        try:
            ocr = self._ocr_engine.predict(img)
        except Exception as e:
            self.log.warning("OCR failed during regex masking: %s", e)
            return []

        # Build per-word index for bbox lookup
        word_regions: List[TextRegion] = [r for r in ocr.regions if r.text]

        # Match regex against the full concatenated text to find PII spans,
        # then use the per-word bboxes to build union boxes for each span.
        matches: List[_TokenMatch] = []
        full_text = " ".join(r.text for r in word_regions if r.text)

        for label, pattern in _REGEX_RULES.items():
            if label not in self.labels:
                continue
            for m in pattern.finditer(full_text):
                matches.append(_TokenMatch(
                    label=label, start=m.start(), end=m.end(), text=m.group(0),
                ))

        # PAN treated as id_number
        if "id_number" in self.labels:
            for m in _PAN_PATTERN.finditer(full_text):
                matches.append(_TokenMatch(
                    label="id_number", start=m.start(), end=m.end(),
                    text=m.group(0),
                ))

        if not matches:
            return []

        # Compute character spans for each word
        spans: List[Tuple[int, int, TextRegion]] = []
        cursor = 0
        for r in word_regions:
            if not r.text:
                continue
            start = cursor
            cursor += len(r.text)
            spans.append((start, cursor, r))
            cursor += 1  # for the joining space

        out: List[MaskRegion] = []
        for match in matches:
            hit_regions = [
                tr for (s, e, tr) in spans
                if not (e <= match.start or s >= match.end)
            ]
            if not hit_regions:
                continue
            xs = [r.bbox.x1 for r in hit_regions] + [r.bbox.x2 for r in hit_regions]
            ys = [r.bbox.y1 for r in hit_regions] + [r.bbox.y2 for r in hit_regions]
            out.append(MaskRegion(
                bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                label=match.label,
                confidence=0.9,  # deterministic regex
            ))
        return out

    # ------------------------------------------------------------------
    def apply_mask(self, image: ImageLike,
                   regions: Optional[List[MaskRegion]] = None,
                   method: Optional[str] = None) -> np.ndarray:
        """Apply the configured (or provided) mask method to the given regions."""
        img = load_image(image).copy()
        if regions is None:
            regions = self.detect(image)

        m = MaskMethod(method) if method else self.method
        h, w = img.shape[:2]

        for r in regions:
            bb = r.bbox
            x1 = max(0, min(w - 1, bb.x1))
            y1 = max(0, min(h - 1, bb.y1))
            x2 = max(x1 + 1, min(w, bb.x2))
            y2 = max(y1 + 1, min(h, bb.y2))
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if m is MaskMethod.GAUSSIAN:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(
                    roi, (self.blur_ksize, self.blur_ksize), 0,
                )
            elif m is MaskMethod.BLACKBOX:
                img[y1:y2, x1:x2] = 0
            elif m is MaskMethod.PIXELATE:
                blocks = self.pixelate_blocks
                rh, rw = roi.shape[:2]
                temp = cv2.resize(roi, (max(1, rw // blocks), max(1, rh // blocks)),
                                  interpolation=cv2.INTER_LINEAR)
                img[y1:y2, x1:x2] = cv2.resize(
                    temp, (rw, rh), interpolation=cv2.INTER_NEAREST,
                )

        return img

    # ------------------------------------------------------------------
    def mask(self, image: ImageLike,
             method: Optional[str] = None) -> Tuple[np.ndarray, List[MaskRegion]]:
        """Convenience: detect + apply_mask in one call. Returns (masked, regions)."""
        regions = self.detect(image)
        masked = self.apply_mask(image, regions=regions, method=method)
        return masked, regions
