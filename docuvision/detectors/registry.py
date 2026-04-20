"""Registry for text detectors."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from docuvision.detectors.base import BaseTextDetector
from docuvision.utils.logging import get_logger

log = get_logger("detector.registry")


class DetectorRegistry:
    _detectors: Dict[str, Type[BaseTextDetector]] = {}

    @classmethod
    def register(cls, detector_cls: Type[BaseTextDetector]) -> Type[BaseTextDetector]:
        name = detector_cls.name
        if not name or name == "base":
            raise ValueError(f"Invalid detector name: {name!r}")
        cls._detectors[name] = detector_cls
        log.debug("Registered text detector: %s", name)
        return detector_cls

    @classmethod
    def list(cls, available_only: bool = False) -> List[str]:
        if not available_only:
            return sorted(cls._detectors.keys())
        return sorted(n for n, c in cls._detectors.items() if c.is_available())

    @classmethod
    def get_class(cls, name: str) -> Type[BaseTextDetector]:
        if name not in cls._detectors:
            raise KeyError(
                f"Unknown detector: {name!r}. "
                f"Registered: {sorted(cls._detectors)}"
            )
        return cls._detectors[name]

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> BaseTextDetector:
        det_cls = cls.get_class(name)
        if not det_cls.is_available():
            raise RuntimeError(
                f"Detector '{name}' is not available — install: {det_cls.requires}"
            )
        return det_cls(**kwargs)

    @classmethod
    def auto_build(cls, preferred: Optional[str] = None,
                   fallback_order: Optional[List[str]] = None,
                   **kwargs: Any) -> BaseTextDetector:
        # "contour" is our always-available pure-OpenCV fallback
        default_order = ["craft", "dbnet", "east", "paddle_det", "contour"]
        order: List[str] = []
        if preferred:
            order.append(preferred)
        if fallback_order:
            order.extend(fallback_order)
        order.extend(default_order)

        seen: set = set()
        ordered = [n for n in order if not (n in seen or seen.add(n))]  # type: ignore

        last_error: Optional[Exception] = None
        for name in ordered:
            if name not in cls._detectors:
                continue
            det_cls = cls._detectors[name]
            if not det_cls.is_available():
                continue
            try:
                return det_cls(**kwargs)
            except Exception as e:
                log.warning("Failed to build detector %s: %s", name, e)
                last_error = e

        if last_error:
            raise RuntimeError(f"No detector could be built; last error: {last_error}")
        raise RuntimeError("No text detector is available.")
