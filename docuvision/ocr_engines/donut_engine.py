"""Donut engine (Tier 4) — document understanding transformer.

Donut is an OCR-free document understanding model — it reads the whole page
and emits structured output in one shot. Unlike the other engines, its
natural output is task-specific JSON or free text, not per-word bboxes.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from docuvision.ocr_engines.base import BaseOCREngine
from docuvision.ocr_engines.registry import OCREngineRegistry
from docuvision.types import BoundingBox, OCRResult, TextRegion
from docuvision.utils.image_io import to_pil
from docuvision.utils.lazy_import import require


@OCREngineRegistry.register
class DonutEngine(BaseOCREngine):
    name = "donut"
    requires = ["transformers", "torch"]
    default_languages = ["en"]

    def __init__(self, languages: Optional[List[str]] = None,
                 use_gpu: bool = True,
                 model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
                 task_prompt: str = "<s_cord-v2>",
                 max_length: int = 768,
                 **kwargs: Any) -> None:
        super().__init__(languages=languages, use_gpu=use_gpu, **kwargs)
        self.model_name = model_name
        self.task_prompt = task_prompt
        self.max_length = max_length
        self._processor: Any = None
        self._device: Any = None

    def _load(self) -> None:
        transformers = require("transformers", feature="Donut")
        torch = require("torch", feature="Donut")
        self._processor = transformers.DonutProcessor.from_pretrained(self.model_name)
        self._model = transformers.VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        self._model.to(self._device).eval()
        self.log.debug("Donut loaded on %s; model=%s", self._device, self.model_name)

    def _predict(self, image: np.ndarray) -> OCRResult:
        import torch
        pil = to_pil(image)
        pixel_values = self._processor(pil, return_tensors="pt").pixel_values.to(self._device)

        decoder_input_ids = self._processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        seq = self._processor.batch_decode(outputs.sequences)[0]
        # Strip task prompt and special tokens
        seq = seq.replace(self._processor.tokenizer.eos_token, "")
        seq = seq.replace(self._processor.tokenizer.pad_token, "")
        seq = seq.replace(self.task_prompt, "").strip()

        # Try to convert the tag-based output to a plain-text summary
        plain_text = seq
        structured: Any = None
        try:
            structured = self._processor.token2json(seq)
            plain_text = _flatten_donut_tree(structured)
        except Exception:
            pass

        h, w = image.shape[:2]
        region = TextRegion(
            bbox=BoundingBox(0, 0, w, h),
            text=plain_text,
            confidence=1.0,
        )
        return OCRResult(
            text=plain_text,
            regions=[region],
            engine=self.name,
            metadata={
                "model": self.model_name,
                "task_prompt": self.task_prompt,
                "structured": structured,
                "raw": seq,
            },
        )


def _flatten_donut_tree(node: Any) -> str:
    """Convert the nested dict produced by `token2json` back to readable text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_flatten_donut_tree(x) for x in node)
    if isinstance(node, dict):
        parts: List[str] = []
        for k, v in node.items():
            text = _flatten_donut_tree(v).strip()
            if text:
                parts.append(f"{k}: {text}" if not isinstance(v, (dict, list)) else text)
        return "\n".join(parts)
    return str(node)
