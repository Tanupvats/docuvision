"""
Preprocessing pipeline for embossed / engraved text (metal plates, chassis numbers).

The strategy, informed by the CV literature on low-contrast industrial text:

    1. Grayscale + strong CLAHE          → normalize uneven illumination
    2. Bilateral filter                  → smooth metal texture, keep edges
    3. Shape-from-shading approximation  → emphasize depth ridges
       (approximated via Sobel gradient magnitude + Laplacian-of-Gaussian)
    4. Top-hat morphology                → isolate bright / dark stroke ridges
    5. Adaptive / Sauvola thresholding   → binarize
    6. Morphological close/open          → connect stroke fragments

Tuning is exposed via `EmbossedPreprocessConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from docuvision.preprocessing.general import binarize, to_grayscale


@dataclass
class EmbossedPreprocessConfig:
    clahe_clip: float = 3.0
    clahe_grid: int = 8
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    sobel_ksize: int = 3
    log_ksize: int = 5
    tophat_ksize: int = 15
    use_dark_text: bool = True               # True=engraved (dark), False=raised (light)
    binarize_method: str = "sauvola"
    binarize_block: int = 41
    morph_close_ksize: int = 2
    morph_open_ksize: int = 1
    invert_output: bool = True               # most OCR engines expect dark text on white


def _gradient_magnitude(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def _log_filter(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=ksize)
    lap = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX)
    return lap.astype(np.uint8)


def _shape_from_shading_approx(gray: np.ndarray, cfg: EmbossedPreprocessConfig) -> np.ndarray:
    """Approximate shape-from-shading emphasis using gradient magnitude + LoG.

    True SfS is expensive and ill-posed under unknown lighting. This cheap
    approximation captures most of the benefit for downstream OCR: it
    boosts raised/recessed ridges regardless of lighting direction.
    """
    mag = _gradient_magnitude(gray, ksize=cfg.sobel_ksize)
    log = _log_filter(gray, ksize=cfg.log_ksize)
    combined = cv2.addWeighted(mag, 0.6, log, 0.4, 0)
    return combined


def embossed_preprocess(
    img: np.ndarray,
    cfg: EmbossedPreprocessConfig | None = None,
) -> np.ndarray:
    """Apply the embossed-text preprocessing pipeline. Returns a uint8 image.

    Output is always single-channel (grayscale or binary) in uint8. If
    `invert_output=True`, the output has dark text on a white background,
    which is what most OCR engines prefer.
    """
    cfg = cfg or EmbossedPreprocessConfig()

    gray = to_grayscale(img)

    # 1. Normalize uneven illumination with CLAHE
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    eq = clahe.apply(gray)

    # 2. Edge-preserving smoothing to kill metal-grain texture
    smoothed = cv2.bilateralFilter(
        eq, cfg.bilateral_d,
        cfg.bilateral_sigma_color, cfg.bilateral_sigma_space,
    )

    # 3. Shape-from-shading approximation
    ridges = _shape_from_shading_approx(smoothed, cfg)

    # 4. Top-hat to pull out bright (raised) or black-hat for dark (engraved) strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.tophat_ksize, cfg.tophat_ksize))
    if cfg.use_dark_text:
        hat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel)
    else:
        hat = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, kernel)
    hat = cv2.normalize(hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. Fuse ridges + hat, equalize again
    fused = cv2.addWeighted(ridges, 0.5, hat, 0.5, 0)
    fused = clahe.apply(fused)

    # 6. Binarize
    binary = binarize(fused, method=cfg.binarize_method, block_size=cfg.binarize_block)

    # 7. Morph cleanup — connect broken strokes, remove specks
    if cfg.morph_close_ksize > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (cfg.morph_close_ksize, cfg.morph_close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)
    if cfg.morph_open_ksize > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (cfg.morph_open_ksize, cfg.morph_open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)

    if cfg.invert_output:
        # Ensure dark-text-on-white
        white_ratio = np.mean(binary > 127)
        if white_ratio < 0.5:
            binary = cv2.bitwise_not(binary)

    return binary
