"""
DocuVision
==========

Adaptive, modular OCR & document-AI pipeline orchestrator.

Public API
----------
- profile_system()             : detect hardware / libs and get a SystemCapabilityReport
- DocumentPipeline(...)        : top-level orchestrator
- OCREngineRegistry            : lookup/build OCR engines by name
- DetectorRegistry             : text detectors
- DocumentClassifier           : document type classifier
- DocumentMasker               : PII masker
- EmbossedOCRPipeline          : specialized engraved-text pipeline
- types                        : OCRResult, TextRegion, PipelineResult, etc.
"""

from docuvision.types import (
    BoundingBox,
    TextRegion,
    OCRResult,
    PipelineResult,
    DocumentClass,
    MaskRegion,
)
from docuvision.system_profiler import profile_system, SystemCapabilityReport
from docuvision.ocr_engines import OCREngineRegistry, BaseOCREngine
from docuvision.detectors import DetectorRegistry, BaseTextDetector
from docuvision.classifier import DocumentClassifier
from docuvision.masking import DocumentMasker
from docuvision.embossed import EmbossedOCRPipeline
from docuvision.pipeline import DocumentPipeline, PipelineConfig

__version__ = "0.1.0"

__all__ = [
    # Core
    "__version__",
    "DocumentPipeline",
    "PipelineConfig",
    "profile_system",
    "SystemCapabilityReport",
    # Engines / stages
    "OCREngineRegistry",
    "BaseOCREngine",
    "DetectorRegistry",
    "BaseTextDetector",
    "DocumentClassifier",
    "DocumentMasker",
    "EmbossedOCRPipeline",
    # Types
    "BoundingBox",
    "TextRegion",
    "OCRResult",
    "PipelineResult",
    "DocumentClass",
    "MaskRegion",
]
