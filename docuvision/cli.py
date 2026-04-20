"""
docuvision CLI
==============

Usage:
    docuvision profile                            # show system capability report
    docuvision ocr path/to/image.png              # run the default pipeline
    docuvision ocr img.png --engine paddleocr --gpu
    docuvision ocr img.png --classify --mask --mask-method blackbox
    docuvision ocr img.png --embossed             # embossed / chassis OCR
    docuvision engines                            # list registered OCR engines
    docuvision detectors                          # list registered detectors

Any image readable by OpenCV or PIL is accepted, including PDFs via pdf2image
if installed (DocuVision itself just reads a single raster image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from docuvision import (
    DetectorRegistry,
    DocumentPipeline,
    OCREngineRegistry,
    PipelineConfig,
    __version__,
    profile_system,
)


def _cmd_profile(args: argparse.Namespace) -> int:
    report = profile_system(refresh=args.refresh)
    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        print(report.summary())
    return 0


def _report_to_dict(report: Any) -> Dict[str, Any]:
    # Drop the `notes` list into a top-level key; everything else from the
    # dataclass fields maps cleanly.
    import dataclasses
    d = dataclasses.asdict(report)
    return d


def _cmd_engines(args: argparse.Namespace) -> int:
    all_engines = OCREngineRegistry.list(available_only=False)
    available = set(OCREngineRegistry.list(available_only=True))
    print("OCR engines:")
    for name in all_engines:
        mark = "✔" if name in available else "✗"
        cls = OCREngineRegistry.get_class(name)
        reqs = ", ".join(cls.requires) or "(no deps)"
        print(f"  [{mark}] {name:<12s} requires: {reqs}")
    return 0


def _cmd_detectors(args: argparse.Namespace) -> int:
    all_det = DetectorRegistry.list(available_only=False)
    available = set(DetectorRegistry.list(available_only=True))
    print("Text detectors:")
    for name in all_det:
        mark = "✔" if name in available else "✗"
        cls = DetectorRegistry.get_class(name)
        reqs = ", ".join(cls.requires) or "(core deps only)"
        print(f"  [{mark}] {name:<12s} requires: {reqs}")
    return 0


def _cmd_ocr(args: argparse.Namespace) -> int:
    if not os.path.exists(args.image):
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    config = PipelineConfig(
        detect_text=not args.no_detect,
        classify_doc=args.classify,
        mask=args.mask,
        embossed_mode=args.embossed,
        language=[l.strip() for l in args.language.split(",") if l.strip()],
        ocr_engine=args.engine,
        detector=args.detector,
        use_gpu=(True if args.gpu else None),
        classifier_model_path=args.classifier_model,
        classifier_labels=(
            [l.strip() for l in args.classifier_labels.split(",") if l.strip()]
            if args.classifier_labels else None
        ),
        mask_method=args.mask_method,
        mask_model_path=args.mask_model,
        tier_override=args.tier,
    )

    pipe = DocumentPipeline(config=config)
    result = pipe.run(args.image)

    # Save masked image if one was produced and user asked for it
    if args.save_masked and result.masked_image is not None:
        import cv2
        cv2.imwrite(args.save_masked, result.masked_image)

    # Render result
    if args.json:
        out = result.to_dict()
        print(json.dumps(out, indent=2, default=str))
    else:
        print(_format_human_result(result))
    return 0


def _format_human_result(result: Any) -> str:
    lines: List[str] = []
    lines.append(f"Engine    : {result.engine_used or 'n/a'}")
    lines.append(f"Tier used : {result.tier_used}")
    lines.append(f"Elapsed   : {result.elapsed_ms:.1f} ms")
    if result.doc_class is not None:
        lines.append(
            f"Document  : {result.doc_class.label} "
            f"(conf {result.doc_class.confidence:.2f})"
        )
    if result.masks:
        lines.append(f"Masks     : {len(result.masks)} region(s)")
        for m in result.masks:
            lines.append(f"            - {m.label}: {m.bbox.as_xyxy()}")
    lines.append(f"Regions   : {len(result.regions)}")
    lines.append("--- Text ---")
    lines.append(result.text or "(empty)")
    failures = result.metadata.get("failures") if result.metadata else None
    if failures:
        lines.append("--- Failures ---")
        for k, v in failures.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="docuvision",
        description="Adaptive OCR and document-AI pipeline orchestrator.",
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"docuvision {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # profile
    sp = sub.add_parser("profile", help="Show system capability report.")
    sp.add_argument("--json", action="store_true", help="Output JSON.")
    sp.add_argument("--refresh", action="store_true",
                    help="Re-run detection, ignoring cache.")
    sp.set_defaults(func=_cmd_profile)

    # engines
    se = sub.add_parser("engines", help="List registered OCR engines.")
    se.set_defaults(func=_cmd_engines)

    # detectors
    sd = sub.add_parser("detectors", help="List registered text detectors.")
    sd.set_defaults(func=_cmd_detectors)

    # ocr
    so = sub.add_parser("ocr", help="Run the OCR pipeline on an image.")
    so.add_argument("image", help="Path to the input image.")
    so.add_argument("--engine", help="Force a specific OCR engine.")
    so.add_argument("--detector", help="Force a specific text detector.")
    so.add_argument("--language", default="en",
                    help="Comma-separated language codes (default: en).")
    so.add_argument("--gpu", action="store_true",
                    help="Prefer GPU when available.")
    so.add_argument("--tier", type=int,
                    help="Override the recommended tier (0-4).")
    so.add_argument("--classify", action="store_true",
                    help="Run document classification.")
    so.add_argument("--classifier-model",
                    help="Path to YOLO (.pt) or ONNX classifier weights.")
    so.add_argument("--classifier-labels",
                    help="Comma-separated custom labels.")
    so.add_argument("--mask", action="store_true",
                    help="Detect + mask PII regions.")
    so.add_argument("--mask-method", default="gaussian",
                    choices=["gaussian", "blackbox", "pixelate"])
    so.add_argument("--mask-model", help="Path to YOLO PII detector weights.")
    so.add_argument("--save-masked", metavar="PATH",
                    help="Write the masked image to PATH.")
    so.add_argument("--embossed", action="store_true",
                    help="Use the embossed/engraved OCR pipeline.")
    so.add_argument("--no-detect", action="store_true",
                    help="Skip text detection; OCR the full image.")
    so.add_argument("--json", action="store_true",
                    help="Output JSON instead of human-readable text.")
    so.set_defaults(func=_cmd_ocr)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # pragma: no cover
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
