"""Abstract base class for text detectors."""
from __future__ import annotations

import abc
import time
from typing import Any, List

import numpy as np

from docuvision.types import TextRegion
from docuvision.utils.image_io import ImageLike, load_image
from docuvision.utils.logging import get_logger


class BaseTextDetector(abc.ABC):
    """Every text detector subclasses this and implements `_detect`."""

    name: str = "base"
    requires: List[str] = []

    def __init__(self, use_gpu: bool = False, **kwargs: Any) -> None:
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self._model: Any = None
        self.log = get_logger(f"detector.{self.name}")

    @classmethod
    def is_available(cls) -> bool:
        from docuvision.utils.lazy_import import is_available
        return all(is_available(m) for m in cls.requires)

    @abc.abstractmethod
    def _load(self) -> None:
        """Lazily load the underlying model."""

    @abc.abstractmethod
    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        """Return raw detected regions (no recognition)."""

    def detect(self, image: ImageLike) -> List[TextRegion]:
        img = load_image(image)
        if self._model is None:
            self._load()
        t0 = time.perf_counter()
        regions = self._detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        self.log.debug("%s detector found %d regions in %.1f ms",
                       self.name, len(regions), elapsed)
        return regions
