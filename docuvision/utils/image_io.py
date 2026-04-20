"""Image loading / color-space helpers.

Engines disagree on whether they expect RGB or BGR, PIL or ndarray.
These helpers give us one canonical representation and cheap conversions.
"""
from __future__ import annotations

import os
from typing import Any, Union

import numpy as np

try:  # cv2 is a core dep (opencv-python-headless); guard just in case
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

ImageLike = Union[str, os.PathLike, np.ndarray, "Image.Image", bytes]


def load_image(src: ImageLike) -> np.ndarray:
    """Load an image from a variety of sources into a BGR uint8 ndarray.

    Supported inputs:
        - path (str / PathLike)
        - raw bytes
        - numpy.ndarray  (passed through; assumed BGR if 3-channel)
        - PIL.Image.Image (converted to BGR)
    """
    if isinstance(src, np.ndarray):
        return _ensure_uint8(src)

    if Image is not None and isinstance(src, Image.Image):
        arr = np.array(src.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if isinstance(src, (bytes, bytearray)):
        arr = np.frombuffer(src, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Fall back to PIL for weird formats
            if Image is None:
                raise ValueError(f"Could not read image: {path}")
            with Image.open(path) as im:
                arr = np.array(im.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return img

    raise TypeError(f"Unsupported image type: {type(src)!r}")


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    # Rescale floats in [0,1] or [0,255]
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(arr.max()) if arr.size else 1.0
        if maxv <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
    return arr.astype(np.uint8)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 view of a BGR-or-gray ndarray."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Return a BGR uint8 view of an ndarray."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img  # already BGR


def to_pil(img: np.ndarray) -> "Image.Image":
    """Convert a BGR ndarray to PIL.Image (RGB)."""
    if Image is None:
        raise ImportError("Pillow is required")
    return Image.fromarray(ensure_rgb(img))
