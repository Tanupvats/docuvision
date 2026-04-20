# DocuVision

**Adaptive, modular document OCR & AI pipeline orchestrator.** DocuVision detects
your system's capabilities (CPU, GPU, VRAM, installed ML libraries) and
dynamically picks the best feasible OCR and document-AI stack — from Tesseract
on a 4 GB CPU box all the way up to TrOCR / Donut on a multi-GPU workstation.

## Features

- **System profiler** — OS, CPU, GPU, CUDA, VRAM, RAM, TensorRT, installed libs
- **Tiered engine selection** — Tier 0 (CPU / Tesseract) → Tier 4 (GPU / TrOCR, Donut)
- **Unified OCR registry** — Tesseract, EasyOCR, PaddleOCR (CPU/GPU), docTR, TrOCR, Donut
- **Text detection** — CRAFT, DBNet, EAST, PaddleOCR-det + pure-OpenCV contour fallback
- **Clustering** — DBSCAN / HDBSCAN bounding-box merging with anisotropic distance
- **Document classification** — pluggable YOLO / ONNX / keyword backends, custom labels
- **PII masking** — YOLO or regex backends, gaussian blur / black box / pixelation
- **Embossed / engraved OCR** — CLAHE + Sobel + shape-from-shading + morphology, ideal
  for chassis numbers, industrial plates, metal serials
- **Graceful fallbacks** — no hard deps on heavy libs, missing engines are skipped

## Installation

```bash
# Core only — works with the pure-OpenCV detector but needs an OCR engine
# installed separately
pip install docuvision

# Common bundles
pip install "docuvision[tesseract]"   # + pytesseract (requires tesseract binary)
pip install "docuvision[easyocr]"     # + easyocr
pip install "docuvision[paddle-cpu]"  # + paddleocr + paddlepaddle (CPU)
pip install "docuvision[paddle-gpu]"  # + paddlepaddle-gpu
pip install "docuvision[doctr]"       # + python-doctr
pip install "docuvision[trocr]"       # + torch + transformers + TrOCR deps
pip install "docuvision[donut]"       # + torch + transformers + Donut deps

# Aggregate bundles
pip install "docuvision[cpu]"         # every CPU-capable engine
pip install "docuvision[gpu]"         # every engine, GPU-preferred
pip install "docuvision[all]"         # everything, library versions only

# Development
pip install "docuvision[dev]"
```

Heavy engines (torch, paddle, transformers…) are **optional**. DocuVision lazily
imports them, so a missing library simply disables that one engine; the rest of
the pipeline keeps working.

## Quick start

```python
from docuvision import DocumentPipeline, profile_system

# See what DocuVision picks for your system
print(profile_system().summary())

# Zero-config pipeline — uses the best available engine
pipe = DocumentPipeline()
result = pipe.run("invoice.png")
print(result.text)

# Full-featured pipeline
pipe = DocumentPipeline(
    detect_text=True,
    classify_doc=True,
    mask=True,
    embossed_mode=False,
    language=["en"],
)
result = pipe.run("aadhaar.jpg")
print(result.doc_class.label)  # → 'aadhaar'
print(result.text)
print(result.masks)            # detected PII regions
# result.masked_image is a numpy.ndarray ready for cv2.imwrite(...)
```

## Command-line

```bash
docuvision profile                            # show the capability report
docuvision ocr invoice.png                    # default pipeline
docuvision ocr plate.jpg --embossed           # embossed / chassis OCR
docuvision ocr id.jpg --classify --mask --mask-method blackbox
docuvision engines                            # list registered OCR engines
docuvision detectors                          # list registered detectors
```

## Design principles

1. **No hard deps on heavy libs.** Every engine is imported lazily. `pip install
   docuvision` drops in fine even without torch, paddle, or transformers.
2. **Explicit tiering.** The profiler scores your hardware and picks a tier; you
   can override it per call.
3. **Composable, not monolithic.** Each stage (detect, classify, OCR, mask) is
   an independent, swappable component with a common ABC and a registry.
4. **Sane defaults, full control.** `DocumentPipeline().run(img)` just works;
   every knob is exposed via `PipelineConfig`.
5. **Fails gracefully.** A per-stage failure is captured in
   `PipelineResult.metadata["failures"]` rather than aborting the pipeline.

## Publishing this package to PyPI

### 1. Verify the name is available

Before anything else, **check that the package name is available on PyPI**:

```bash
# If this returns a page, the name is taken — pick a different one
# and update `name = "…"` in pyproject.toml
curl -s https://pypi.org/pypi/docuvision/json -o /dev/null -w "%{http_code}\n"
```

A `404` means the name is free. A `200` means someone has already published
under that name — edit `[project].name` in `pyproject.toml` before building.

### 2. Update placeholders

- Edit `[project].authors` and `maintainers` with your name and email.
- Edit `[project.urls]` to point at your real repository.
- Bump `[project].version` to the release number (we follow [SemVer](https://semver.org/)).
- Keep `CHANGELOG.md` in sync.

### 3. Build the distributions

```bash
pip install --upgrade build twine
python -m build            # produces dist/docuvision-<ver>.tar.gz and -py3-none-any.whl
twine check dist/*         # validates README rendering and metadata
```

### 4. Upload

```bash
# Test-run against TestPyPI first
twine upload --repository testpypi dist/*

# If that looks good, push to PyPI proper
twine upload dist/*
```

Configure `~/.pypirc` or use an API token (`twine upload -u __token__ -p <token>`).

### 5. Tag the release

```bash
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0
```

## Testing

The full test suite runs against the core install only — heavy engines are not
required:

```bash
pytest tests/
# or, without pytest installed:
python run_tests.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
