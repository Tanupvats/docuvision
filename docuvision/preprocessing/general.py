"""General image preprocessing helpers for OCR."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_max_side(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Downscale so the longest side == max_side; never upscale."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_contrast(img: np.ndarray, clip_limit: float = 2.0,
                       tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE on luminance, preserving colour channels."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid_size, tile_grid_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, strength: int = 7) -> np.ndarray:
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def binarize(img: np.ndarray, method: str = "sauvola",
             block_size: int = 31, C: int = 10) -> np.ndarray:
    """Return a binarized grayscale image.

    method ∈ {"otsu", "adaptive_mean", "adaptive_gaussian", "sauvola"}.
    """
    gray = to_grayscale(img)

    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out

    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C,
        )

    if method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C,
        )

    if method == "sauvola":
        return _sauvola(gray, window=block_size, k=0.2)

    raise ValueError(f"Unknown binarization method: {method}")


def _sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2,
             R: float = 128.0) -> np.ndarray:
    """Sauvola's adaptive thresholding — works well on uneven lighting."""
    if window % 2 == 0:
        window += 1
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window, window))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    thresh = mean * (1 + k * ((std / R) - 1))
    out = np.where(gray > thresh, 255, 0).astype(np.uint8)
    return out


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Estimate and correct small text rotations.

    Uses the minimum-area rectangle around foreground pixels after binarization.
    Only corrects angles within ±max_angle degrees (to avoid 90° flips).
    """
    gray = to_grayscale(img)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0) with newer OpenCVs
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > max_angle or abs(angle) < 0.1:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
