"""Registry for OCR engines.

Engines self-register via :meth:`OCREngineRegistry.register`; the registry
is used both by the pipeline (to look up engines by name) and by the
system profiler (to find available engines for a given tier).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.utils.logging import get_logger

log = get_logger("ocr.registry")


class OCREngineRegistry:
    """Class-level registry mapping engine name → engine class.

    Use via the class methods; there is no reason to instantiate it.
    """

    _engines: Dict[str, Type[BaseOCREngine]] = {}

    # ------------------------------------------------------------------
    @classmethod
    def register(cls, engine_cls: Type[BaseOCREngine]) -> Type[BaseOCREngine]:
        """Register an engine class. Idempotent; last write wins."""
        name = engine_cls.name
        if not name or name == "base":
            raise ValueError(f"Invalid engine name: {name!r}")
        cls._engines[name] = engine_cls
        log.debug("Registered OCR engine: %s", name)
        return engine_cls

    # ------------------------------------------------------------------
    @classmethod
    def list(cls, available_only: bool = False) -> List[str]:
        """List registered engines. If `available_only`, filter by `is_available()`."""
        if not available_only:
            return sorted(cls._engines.keys())
        return sorted(n for n, c in cls._engines.items() if c.is_available())

    # ------------------------------------------------------------------
    @classmethod
    def get_class(cls, name: str) -> Type[BaseOCREngine]:
        if name not in cls._engines:
            raise KeyError(
                f"Unknown OCR engine: {name!r}. "
                f"Registered: {sorted(cls._engines)}"
            )
        return cls._engines[name]

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, name: str, **kwargs: Any) -> BaseOCREngine:
        """Instantiate an engine by name, constructing it with kwargs."""
        engine_cls = cls.get_class(name)
        if not engine_cls.is_available():
            raise RuntimeError(
                f"OCR engine '{name}' is not available — "
                f"install: {engine_cls.requires}"
            )
        return engine_cls(**kwargs)

    # ------------------------------------------------------------------
    @classmethod
    def auto_build(cls, preferred: Optional[str] = None,
                   fallback_order: Optional[List[str]] = None,
                   **kwargs: Any) -> BaseOCREngine:
        """Build the best-available engine.

        Order of attempts:
            1. `preferred` (if given)
            2. `fallback_order` (if given)
            3. Default fallback: trocr → doctr → paddleocr → easyocr → tesseract
        """
        default_order = ["trocr", "doctr", "paddleocr", "easyocr", "tesseract"]
        order: List[str] = []
        if preferred:
            order.append(preferred)
        if fallback_order:
            order.extend(fallback_order)
        order.extend(default_order)

        # Deduplicate while preserving order
        seen = set()
        ordered = [n for n in order if not (n in seen or seen.add(n))]  # type: ignore[func-returns-value]

        last_error: Optional[Exception] = None
        for name in ordered:
            if name not in cls._engines:
                continue
            engine_cls = cls._engines[name]
            if not engine_cls.is_available():
                continue
            try:
                return engine_cls(**kwargs)
            except Exception as e:
                log.warning("Failed to build %s: %s", name, e)
                last_error = e

        hint = "Install at least one of: pytesseract (+ tesseract binary), easyocr, paddleocr."
        if last_error:
            raise RuntimeError(f"No OCR engine could be built. Last error: {last_error}. {hint}")
        raise RuntimeError(f"No OCR engine is available. {hint}")
