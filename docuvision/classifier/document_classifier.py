"""
Document type classifier.

Supports three backends, chosen automatically based on what's available:

1. **YOLO classifier** (Ultralytics) — best accuracy on finetuned weights.
2. **ONNX Runtime classifier** — lightweight; loads .onnx exported from any
   PyTorch / TF training pipeline. Expects a standard image classifier
   (softmax head) with a provided list of labels.
3. **Keyword fallback** — runs OCR with a lightweight engine and matches
   against class-specific keyword dictionaries. Always works, but naturally
   less accurate than a trained model. Ideal for zero-training bootstrapping
   and for the most common Indian document types.

Users can:
    - call `DocumentClassifier()` with no args → uses keyword fallback
    - pass `model_path=...` to load YOLO or ONNX weights
    - pass `labels=[...]` to define custom classes
    - call `predict(image)` to get a DocumentClass
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from docuvision.types import DocumentClass
from docuvision.utils.image_io import ImageLike, ensure_rgb, load_image
from docuvision.utils.lazy_import import optional_import
from docuvision.utils.logging import get_logger


DEFAULT_DOCUMENT_CLASSES: List[str] = [
    "invoice",
    "aadhaar",
    "pan",
    "driving_license",
    "vehicle_rc",
    "bank_statement",
    "utility_bill",
    "industrial_label",
    "vehicle_chassis",
    "other",
]

# Keyword fallback lexicon. Casing is ignored at match time.
_KEYWORD_LEXICON: Dict[str, List[str]] = {
    "invoice": ["invoice", "bill to", "sub total", "subtotal", "gstin", "tax invoice",
                "po number", "purchase order", "amount due"],
    "aadhaar": ["aadhaar", "uidai", "government of india", "unique identification",
                "आधार"],
    "pan": ["permanent account number", "income tax department", "pan card",
            "incometaxindia"],
    "driving_license": ["driving licence", "driving license", "dl no", "transport",
                        "motor vehicle"],
    "vehicle_rc": ["registration certificate", "vehicle class", "chassis",
                   "engine number", "rc book"],
    "bank_statement": ["account statement", "opening balance", "closing balance",
                       "ifsc", "micr", "statement of account", "transaction"],
    "utility_bill": ["electricity bill", "water bill", "bill number", "meter",
                     "units consumed", "kwh"],
    "industrial_label": ["serial no", "part no", "model no", "manufactured by",
                         "certified", "iso"],
    "vehicle_chassis": ["chassis no", "vin", "frame number", "maa", "mak"],
}


class DocumentClassifier:
    def __init__(self,
                 labels: Optional[List[str]] = None,
                 model_path: Optional[str] = None,
                 backend: Optional[str] = None,
                 input_size: int = 224,
                 use_gpu: bool = False,
                 keyword_ocr_engine: Optional[Any] = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        labels : list of str
            Class labels. Defaults to `DEFAULT_DOCUMENT_CLASSES`. When using
            a trained model, this MUST match the training class order.
        model_path : str, optional
            Path to YOLO (.pt) or ONNX (.onnx) weights.
        backend : {"yolo", "onnx", "keyword"}, optional
            Force a backend. Auto-detected from model_path if None.
        input_size : int
            Square input size used for ONNX inference.
        use_gpu : bool
        keyword_ocr_engine : BaseOCREngine, optional
            When using the keyword backend, use this OCR engine. If None,
            one is auto-selected.
        """
        self.labels = list(labels) if labels else list(DEFAULT_DOCUMENT_CLASSES)
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self.log = get_logger("classifier")

        self.backend = backend or self._auto_backend()
        self._model: Any = None
        self._onnx_input_name: Optional[str] = None
        self._ocr_engine = keyword_ocr_engine

    # ------------------------------------------------------------------
    def _auto_backend(self) -> str:
        if self.model_path:
            ext = os.path.splitext(self.model_path)[1].lower()
            if ext in {".pt", ".pth"}:
                return "yolo"
            if ext == ".onnx":
                return "onnx"
        return "keyword"

    # ------------------------------------------------------------------
    def _load_yolo(self) -> None:
        ul = optional_import("ultralytics")
        if ul is None:
            raise ImportError("Install `ultralytics` to use the YOLO backend.")
        self._model = ul.YOLO(self.model_path)
        # Pull labels from model if the user didn't override
        if self.model_path and hasattr(self._model, "names") and self._model.names:
            model_labels = [self._model.names[i] for i in sorted(self._model.names)]
            if self.labels == DEFAULT_DOCUMENT_CLASSES:
                self.labels = model_labels

    def _load_onnx(self) -> None:
        ort = optional_import("onnxruntime")
        if ort is None:
            raise ImportError("Install `onnxruntime` to use the ONNX backend.")
        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            providers = ["CUDAExecutionProvider"] + providers
        self._model = ort.InferenceSession(self.model_path, providers=providers)
        self._onnx_input_name = self._model.get_inputs()[0].name

    def _load_keyword(self) -> None:
        if self._ocr_engine is None:
            # Import lazily to avoid a circular import at module load
            from docuvision.ocr_engines.registry import OCREngineRegistry
            try:
                self._ocr_engine = OCREngineRegistry.auto_build(
                    preferred=self.kwargs.get("keyword_engine"),
                )
            except RuntimeError as e:
                self.log.warning(
                    "No OCR engine for keyword classifier: %s — predictions "
                    "will default to 'other'.", e,
                )
                self._ocr_engine = None
        self._model = True  # sentinel

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "yolo":
            self._load_yolo()
        elif self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "keyword":
            self._load_keyword()
        else:
            raise ValueError(f"Unknown classifier backend: {self.backend}")

    # ------------------------------------------------------------------
    def _preprocess_onnx(self, img: np.ndarray) -> np.ndarray:
        rgb = ensure_rgb(img)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        # ImageNet normalization — can be overridden via kwargs
        mean = np.array(self.kwargs.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(self.kwargs.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    def predict(self, image: ImageLike) -> DocumentClass:
        self._load()
        img = load_image(image)

        if self.backend == "yolo":
            results = self._model(ensure_rgb(img), verbose=False)
            r = results[0]
            # Classification mode
            if hasattr(r, "probs") and r.probs is not None:
                probs = r.probs.data.cpu().numpy()
                scores = {self.labels[i]: float(probs[i])
                          for i in range(min(len(self.labels), len(probs)))}
                top = max(scores, key=scores.get)
                return DocumentClass(label=top, confidence=scores[top], all_scores=scores)
            # Detection mode — pick the most confident class across boxes
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                agg: Dict[int, float] = {}
                for c, p in zip(cls_ids, confs):
                    agg[int(c)] = max(agg.get(int(c), 0.0), float(p))
                scores = {self.labels[i]: v for i, v in agg.items()
                          if i < len(self.labels)}
                if scores:
                    top = max(scores, key=scores.get)
                    return DocumentClass(label=top, confidence=scores[top],
                                         all_scores=scores)
            return DocumentClass(label="other", confidence=0.0, all_scores={})

        if self.backend == "onnx":
            x = self._preprocess_onnx(img)
            out = self._model.run(None, {self._onnx_input_name: x})[0]
            # Softmax (stable)
            logits = out[0]
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            scores = {self.labels[i]: float(probs[i])
                      for i in range(min(len(self.labels), len(probs)))}
            top = max(scores, key=scores.get)
            return DocumentClass(label=top, confidence=scores[top], all_scores=scores)

        # Keyword backend
        if self.backend == "keyword":
            if self._ocr_engine is None:
                return DocumentClass(label="other", confidence=0.0, all_scores={})
            try:
                ocr_result = self._ocr_engine.predict(img)
            except Exception as e:
                self.log.warning("OCR failed in keyword classifier: %s", e)
                return DocumentClass(label="other", confidence=0.0, all_scores={})
            return self._classify_by_keywords(ocr_result.text)

        raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    def _classify_by_keywords(self, text: str) -> DocumentClass:
        lower = text.lower()
        scores: Dict[str, float] = {}
        for label, keywords in _KEYWORD_LEXICON.items():
            if label not in self.labels:
                continue
            hits = sum(1 for kw in keywords if kw in lower)
            if hits:
                # Score: fraction of keywords matched, with diminishing returns
                scores[label] = min(1.0, hits / 3.0)

        if not scores:
            return DocumentClass(label="other", confidence=0.0, all_scores={})

        top = max(scores, key=scores.get)
        return DocumentClass(label=top, confidence=scores[top], all_scores=scores)
