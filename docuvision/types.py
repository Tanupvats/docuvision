"""
docuvision.types
================

Shared dataclass-based types used throughout the package.

These are intentionally built on stdlib `dataclasses` (rather than pydantic) so
that this module has no heavy import cost and can be reused by every submodule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1) = top-left."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        # Normalize so x1 <= x2 and y1 <= y2
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def expand(self, margin: int, max_w: Optional[int] = None,
               max_h: Optional[int] = None) -> "BoundingBox":
        x1 = max(0, self.x1 - margin)
        y1 = max(0, self.y1 - margin)
        x2 = self.x2 + margin
        y2 = self.y2 + margin
        if max_w is not None:
            x2 = min(max_w, x2)
        if max_h is not None:
            y2 = min(max_h, y2)
        return BoundingBox(x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Text / OCR
# ---------------------------------------------------------------------------
@dataclass
class TextRegion:
    """A detected text region with an optional recognized text and confidence."""

    bbox: BoundingBox
    text: Optional[str] = None
    confidence: float = 0.0
    polygon: Optional[List[Tuple[int, int]]] = None  # optional quad / polygon
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.as_xyxy(),
            "text": self.text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "language": self.language,
        }


@dataclass
class OCRResult:
    """Output of an OCR engine for a single image."""

    text: str
    regions: List[TextRegion] = field(default_factory=list)
    engine: str = ""
    language: Optional[str] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "regions": [r.to_dict() for r in self.regions],
            "engine": self.engine,
            "language": self.language,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Document classification / masking
# ---------------------------------------------------------------------------
@dataclass
class DocumentClass:
    """Classifier output for a document."""

    label: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaskRegion:
    """A region to be masked / redacted."""

    bbox: BoundingBox
    label: str                       # e.g. "name", "id_number", "chassis"
    confidence: float = 0.0
    polygon: Optional[List[Tuple[int, int]]] = None


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Top-level structured output of DocumentPipeline.run()."""

    text: str = ""
    ocr: Optional[OCRResult] = None
    doc_class: Optional[DocumentClass] = None
    regions: List[TextRegion] = field(default_factory=list)
    masks: List[MaskRegion] = field(default_factory=list)
    masked_image: Optional[Any] = None         # numpy.ndarray when mask=True
    preprocessed_image: Optional[Any] = None   # numpy.ndarray, for debugging
    elapsed_ms: float = 0.0
    tier_used: Optional[int] = None
    engine_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "text": self.text,
            "ocr": self.ocr.to_dict() if self.ocr else None,
            "doc_class": (
                {"label": self.doc_class.label, "confidence": self.doc_class.confidence}
                if self.doc_class else None
            ),
            "regions": [r.to_dict() for r in self.regions],
            "masks": [
                {"bbox": m.bbox.as_xyxy(), "label": m.label, "confidence": m.confidence}
                for m in self.masks
            ],
            "elapsed_ms": self.elapsed_ms,
            "tier_used": self.tier_used,
            "engine_used": self.engine_used,
            "metadata": self.metadata,
        }
        return out
