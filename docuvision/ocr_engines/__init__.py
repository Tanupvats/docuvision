"""OCR engine registry and backends."""
from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry

# Eagerly import engine modules so they self-register with the registry.
# Imports are cheap because heavy deps are loaded lazily inside each engine.
from docuvision.ocr_engines import (
    tesseract_engine,   # noqa: F401
    easyocr_engine,     # noqa: F401
    paddleocr_engine,   # noqa: F401
    doctr_engine,       # noqa: F401
    trocr_engine,       # noqa: F401
    donut_engine,       # noqa: F401
)

__all__ = ["BaseOCREngine", "OCREngineRegistry"]
