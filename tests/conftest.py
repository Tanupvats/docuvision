"""Shared pytest fixtures.

We synthesize tiny test images so tests run without any model downloads.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import pytest
except ImportError:  # offline fallback — our run_tests.py handles fixtures manually
    class _FakePytest:
        def fixture(self, *args, **kwargs):
            def _wrap(fn):
                return fn
            return _wrap
    pytest = _FakePytest()  # type: ignore[assignment]


def _make_text_image(text: str = "INVOICE 12345\nTotal: 99.99",
                     size=(200, 600)) -> np.ndarray:
    h, w = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 60
    for line in text.split("\n"):
        cv2.putText(img, line, (20, y), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        y += 60
    return img


def _make_embossed_image(text: str = "MAT12345XY", size=(120, 420)) -> np.ndarray:
    """Synthetic low-contrast embossed-looking text for the embossed pipeline."""
    h, w = size
    # Gray metallic base
    base = np.full((h, w, 3), 130, dtype=np.uint8)
    # Add some noise to simulate brushed metal
    noise = np.random.default_rng(0).integers(-15, 15, (h, w, 3), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Draw slightly-darker text (engraved)
    cv2.putText(base, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (90, 90, 90), 3, cv2.LINE_AA)
    return base


@pytest.fixture
def text_image() -> np.ndarray:
    return _make_text_image()


@pytest.fixture
def embossed_image() -> np.ndarray:
    return _make_embossed_image()


@pytest.fixture
def pii_text_image() -> np.ndarray:
    return _make_text_image(
        "Name: John Doe\n"
        "Aadhaar: 1234 5678 9012\n"
        "Phone: +91 98765 43210"
    )
