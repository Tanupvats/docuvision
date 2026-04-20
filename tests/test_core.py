"""Core smoke tests — none of these require any heavy ML dependencies.

They verify that:
  * the package imports cleanly
  * the system profiler returns a sensible report
  * the contour detector always works
  * clustering & merging work
  * preprocessing works
  * the embossed preprocessor runs
  * the masker's regex backend can find PII in synthesized OCR output
  * the pipeline orchestrator degrades gracefully when no engines are present
"""
from __future__ import annotations

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pass

from docuvision import (
    BoundingBox,
    DetectorRegistry,
    DocumentPipeline,
    OCREngineRegistry,
    PipelineConfig,
    SystemCapabilityReport,
    TextRegion,
    profile_system,
)
from docuvision.detectors.clustering import cluster_and_merge_boxes
from docuvision.preprocessing import (
    EmbossedPreprocessConfig,
    binarize,
    deskew,
    embossed_preprocess,
    normalize_contrast,
    resize_max_side,
    to_grayscale,
)


# ---------------------------------------------------------------------------
# 1. System profiler
# ---------------------------------------------------------------------------
def test_profile_runs_cleanly():
    report = profile_system(refresh=True)
    assert isinstance(report, SystemCapabilityReport)
    assert report.cpu_count >= 1
    assert report.ram_total_gb >= 0
    assert 0 <= report.tier <= 4
    assert isinstance(report.recommended_pipeline, str)
    # Summary should render
    assert "DocuVision" in report.summary()


def test_profile_installed_libs_is_dict():
    report = profile_system()
    assert isinstance(report.installed_libs, dict)
    for k, v in report.installed_libs.items():
        assert isinstance(k, str)
        assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# 2. Types
# ---------------------------------------------------------------------------
def test_boundingbox_geometry():
    b = BoundingBox(10, 10, 30, 40)
    assert b.width == 20
    assert b.height == 30
    assert b.area == 600
    assert b.center == (20.0, 25.0)


def test_boundingbox_iou_and_expand():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    iou = a.iou(b)
    assert 0 < iou < 1
    expanded = a.expand(5, max_w=12, max_h=12)
    assert expanded.x1 == 0 and expanded.x2 == 12


def test_boundingbox_normalizes_inverted_coords():
    b = BoundingBox(30, 40, 10, 10)
    assert b.x1 <= b.x2 and b.y1 <= b.y2


# ---------------------------------------------------------------------------
# 3. Registries
# ---------------------------------------------------------------------------
def test_registries_register_expected_names():
    ocr = set(OCREngineRegistry.list())
    assert {"tesseract", "easyocr", "paddleocr",
            "doctr", "trocr", "donut"}.issubset(ocr)

    det = set(DetectorRegistry.list())
    assert {"craft", "dbnet", "db", "east", "paddle_det",
            "contour"}.issubset(det)


def test_contour_detector_always_available():
    # Contour is the only detector guaranteed to work without extras
    assert "contour" in DetectorRegistry.list(available_only=True)


# ---------------------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------------------
def test_preprocessing_pipeline(text_image):
    gray = to_grayscale(text_image)
    assert gray.ndim == 2

    resized = resize_max_side(text_image, max_side=400)
    assert max(resized.shape[:2]) == 400

    eq = normalize_contrast(text_image)
    assert eq.shape == text_image.shape

    bin_img = binarize(text_image, method="otsu")
    assert bin_img.dtype == np.uint8 and bin_img.ndim == 2

    desk = deskew(text_image)
    assert desk.shape == text_image.shape


def test_embossed_preprocess(embossed_image):
    cfg = EmbossedPreprocessConfig()
    out = embossed_preprocess(embossed_image, cfg)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == embossed_image.shape[:2]


# ---------------------------------------------------------------------------
# 5. Contour detector end-to-end
# ---------------------------------------------------------------------------
def test_contour_detector_finds_regions(text_image):
    det = DetectorRegistry.build("contour")
    regions = det.detect(text_image)
    assert len(regions) >= 1
    for r in regions:
        assert isinstance(r, TextRegion)
        assert r.bbox.width > 0 and r.bbox.height > 0


# ---------------------------------------------------------------------------
# 6. Clustering + merging
# ---------------------------------------------------------------------------
def test_clustering_merges_adjacent_boxes():
    # Three horizontally-adjacent word boxes; expected to be fused into one line
    regions = [
        TextRegion(bbox=BoundingBox(10, 10, 50, 30), text="foo", confidence=0.9),
        TextRegion(bbox=BoundingBox(55, 12, 95, 30), text="bar", confidence=0.9),
        TextRegion(bbox=BoundingBox(100, 11, 140, 30), text="baz", confidence=0.9),
        # Unrelated region far below
        TextRegion(bbox=BoundingBox(10, 200, 80, 220), text="x", confidence=0.9),
    ]
    merged = cluster_and_merge_boxes(regions, image_shape=(300, 300, 3), margin=2)
    assert 1 <= len(merged) <= 2
    # The merged row must span all three boxes
    widest = max(merged, key=lambda r: r.bbox.width)
    assert widest.bbox.x1 <= 12
    assert widest.bbox.x2 >= 138


def test_clustering_empty_input_returns_empty():
    assert cluster_and_merge_boxes([], image_shape=(100, 100, 3)) == []


# ---------------------------------------------------------------------------
# 7. Masker regex fallback — no real OCR needed
# ---------------------------------------------------------------------------
def test_masker_regex_detects_pii_from_supplied_text():
    """Verify regex rules find Aadhaar / phone patterns in a synthesized string."""
    from docuvision.masking.document_masker import (
        _PAN_PATTERN, _REGEX_RULES,
    )
    text = ("Name: John Doe\n"
            "Aadhaar: 1234 5678 9012\n"
            "PAN: ABCDE1234F\n"
            "Phone: +91 98765 43210\n"
            "Email: john.doe@example.com")
    assert _REGEX_RULES["id_number"].search(text) is not None
    assert _PAN_PATTERN.search(text) is not None
    assert _REGEX_RULES["phone"].search(text) is not None
    assert _REGEX_RULES["email"].search(text) is not None


def test_masker_apply_mask_with_supplied_regions(text_image):
    from docuvision.masking import DocumentMasker
    from docuvision.types import MaskRegion

    masker = DocumentMasker()
    # Bypass OCR/yolo by supplying regions ourselves
    regions = [
        MaskRegion(bbox=BoundingBox(10, 10, 100, 60), label="name", confidence=1.0),
    ]
    out = masker.apply_mask(text_image, regions=regions, method="blackbox")
    assert out.shape == text_image.shape
    # The masked region should be all-black
    roi = out[10:60, 10:100]
    assert int(roi.sum()) == 0


# ---------------------------------------------------------------------------
# 8. Classifier keyword backend (stubbed OCR)
# ---------------------------------------------------------------------------
def test_classifier_keyword_backend(text_image):
    """Stub an OCR engine so we don't need Tesseract/EasyOCR in CI."""
    from docuvision.classifier import DocumentClassifier
    from docuvision.types import OCRResult

    class FakeOCR:
        name = "fake"
        def predict(self, image):  # noqa: D401
            return OCRResult(text="Tax Invoice #123 GSTIN ABC",
                             regions=[], engine="fake")

    clf = DocumentClassifier(backend="keyword", keyword_ocr_engine=FakeOCR())
    result = clf.predict(text_image)
    assert result.label == "invoice"
    assert result.confidence > 0


# ---------------------------------------------------------------------------
# 9. Pipeline orchestrator — graceful when no real OCR engine is present
# ---------------------------------------------------------------------------
def test_pipeline_builds_even_without_trained_models(text_image):
    """With no heavy deps, the pipeline should still report failures gracefully
    rather than raising."""
    cfg = PipelineConfig(
        detect_text=True,
        classify_doc=False,
        mask=False,
        embossed_mode=False,
        language=["en"],
    )
    pipe = DocumentPipeline(config=cfg)
    try:
        result = pipe.run(text_image)
    except RuntimeError as e:
        # Acceptable: no OCR engine available at all
        assert "OCR engine" in str(e) or "No" in str(e)
        return

    # If OCR was available, we expect an OCRResult (possibly empty text)
    assert result.ocr is not None
    assert isinstance(result.elapsed_ms, float)


def test_pipeline_result_serializes():
    from docuvision.types import PipelineResult, OCRResult
    r = PipelineResult(
        text="hello",
        ocr=OCRResult(text="hello", engine="tesseract"),
        elapsed_ms=12.3,
        tier_used=0,
    )
    d = r.to_dict()
    assert d["text"] == "hello"
    assert d["ocr"]["engine"] == "tesseract"
    assert d["elapsed_ms"] == 12.3


# ---------------------------------------------------------------------------
# 10. CLI argument parsing sanity
# ---------------------------------------------------------------------------
def test_cli_parser_builds():
    from docuvision.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["profile"])
    assert args.cmd == "profile"
    args = parser.parse_args(["ocr", "img.png", "--classify", "--mask",
                              "--mask-method", "blackbox",
                              "--language", "en,hi"])
    assert args.cmd == "ocr"
    assert args.classify is True
    assert args.mask is True
    assert args.mask_method == "blackbox"
    assert args.language == "en,hi"
