"""Image preprocessing for OCR (general and embossed)."""
from docuvision.preprocessing.general import (
    deskew,
    denoise,
    binarize,
    to_grayscale,
    resize_max_side,
    normalize_contrast,
)
from docuvision.preprocessing.embossed import embossed_preprocess, EmbossedPreprocessConfig

__all__ = [
    "deskew",
    "denoise",
    "binarize",
    "to_grayscale",
    "resize_max_side",
    "normalize_contrast",
    "embossed_preprocess",
    "EmbossedPreprocessConfig",
]
