# Changelog

All notable changes to DocuVision are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v2.0.0] - 2026-04-20

### Added

- Initial public release.
- System capability profiler: detects OS, CPU (cores + AVX/SSE features),
  RAM, GPU (CUDA via `torch` or `nvidia-smi`), VRAM, TensorRT, and installed
  OCR/ML libraries. Produces a heuristic-scored `SystemCapabilityReport`
  with a recommended tier (0–4) and pipeline.
- Unified OCR engine registry with lazy-loaded backends: Tesseract, EasyOCR,
  PaddleOCR (CPU + GPU), docTR, TrOCR, Donut.
- Text detector registry: CRAFT (via `craft_text_detector` or EasyOCR's
  bundled CRAFT), DBNet / DB (via docTR), EAST (via OpenCV DNN), PaddleOCR
  detector, and a pure-OpenCV contour-based fallback that always works.
- Density-based clustering + bounding-box merging (DBSCAN / HDBSCAN) with
  anisotropic distance scaling for line detection.
- Document classifier with three backends: Ultralytics YOLO, ONNX Runtime,
  and a zero-training keyword-matching fallback covering common Indian
  identity / financial / industrial documents.
- Document masker for PII redaction with YOLO or regex backends; mask
  methods: gaussian blur, solid black-box, pixelation.
- Embossed / engraved OCR pipeline: CLAHE → bilateral → Sobel + LoG
  shape-from-shading approximation → top-hat / black-hat morphology →
  Sauvola threshold → morph cleanup, tunable via
  `EmbossedPreprocessConfig`. Includes dual-crop scoring and a chassis-
  number post-processor.
- Top-level `DocumentPipeline` orchestrator composing classify / detect /
  OCR / mask / embossed stages with graceful per-stage failure handling.
- Command-line entry point `docuvision` with `profile`, `ocr`, `engines`,
  and `detectors` subcommands.
- 18 smoke tests covering every subsystem; all pass with core deps only.

[v2.0.0]: https://github.com/Tanupvats/docuvision/releases/tag/v2.0.0
