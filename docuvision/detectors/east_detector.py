"""EAST text detector using OpenCV's `cv2.dnn` and an EAST .pb model file.

Download the pretrained model from:
    https://github.com/argman/EAST
and pass its path via `model_path=`.
"""
from __future__ import annotations

import os
from typing import Any, List

import cv2
import numpy as np

from docuvision.detectors.base import BaseTextDetector
from docuvision.detectors.registry import DetectorRegistry
from docuvision.types import BoundingBox, TextRegion
from docuvision.utils.lazy_import import is_available


@DetectorRegistry.register
class EASTDetector(BaseTextDetector):
    name = "east"
    # cv2.dnn ships in opencv-python — core dep — so we only need the weights
    requires: List[str] = []

    @classmethod
    def is_available(cls) -> bool:
        return is_available("cv2") and _find_east_model() is not None

    def __init__(self, use_gpu: bool = False,
                 model_path: str = "",
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 input_size: int = 320,
                 **kwargs: Any) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.model_path = model_path or _find_east_model() or ""
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # must be multiple of 32

    def _load(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "EAST model file not found. Set env var DOCUVISION_EAST_MODEL "
                "or pass model_path=… pointing to frozen_east_text_detection.pb"
            )
        self._model = cv2.dnn.readNet(self.model_path)
        if self.use_gpu:
            try:
                self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                self.log.warning("CUDA backend not available for EAST: %s", e)
        self.log.debug("EAST loaded from %s", self.model_path)

    def _detect(self, image: np.ndarray) -> List[TextRegion]:
        h, w = image.shape[:2]
        new_size = ((self.input_size + 31) // 32) * 32
        rW = w / float(new_size)
        rH = h / float(new_size)

        blob = cv2.dnn.blobFromImage(
            image, 1.0, (new_size, new_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False,
        )
        self._model.setInput(blob)
        scores, geometry = self._model.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])

        rects, confidences = self._decode(scores, geometry, self.conf_threshold)
        if not rects:
            return []
        # NMS
        indices = cv2.dnn.NMSBoxesRotated(
            rects, confidences, self.conf_threshold, self.nms_threshold,
        )

        regions: List[TextRegion] = []
        if indices is None or len(indices) == 0:
            return regions

        for idx in np.array(indices).flatten():
            (cx, cy), (rw, rh), angle = rects[int(idx)]
            # Axis-aligned bbox around the rotated rect
            box = cv2.boxPoints(((cx, cy), (rw, rh), angle))
            box[:, 0] *= rW
            box[:, 1] *= rH
            xs = box[:, 0].astype(int)
            ys = box[:, 1].astype(int)
            regions.append(TextRegion(
                bbox=BoundingBox(int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())),
                polygon=[(int(p[0]), int(p[1])) for p in box],
                confidence=float(confidences[int(idx)]),
            ))
        return regions

    @staticmethod
    def _decode(scores: np.ndarray, geometry: np.ndarray,
                score_thresh: float):
        """EAST decoder — returns (rotated_rects, confidences)."""
        num_rows, num_cols = scores.shape[2:4]
        rects = []
        confs = []
        for y in range(num_rows):
            scoresData = scores[0, 0, y]
            x0 = geometry[0, 0, y]
            x1 = geometry[0, 1, y]
            x2 = geometry[0, 2, y]
            x3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(num_cols):
                score = float(scoresData[x])
                if score < score_thresh:
                    continue
                offsetX, offsetY = x * 4.0, y * 4.0
                angle = anglesData[x]
                cos, sin = np.cos(angle), np.sin(angle)
                h = x0[x] + x2[x]
                w = x1[x] + x3[x]
                endX = offsetX + cos * x1[x] + sin * x2[x]
                endY = offsetY - sin * x1[x] + cos * x2[x]
                startX = endX - w
                startY = endY - h
                cx, cy = (startX + endX) / 2.0, (startY + endY) / 2.0
                rects.append(((cx, cy), (w, h), -np.degrees(angle)))
                confs.append(score)
        return rects, confs


def _find_east_model() -> "str | None":
    env = os.environ.get("DOCUVISION_EAST_MODEL")
    if env and os.path.exists(env):
        return env
    for candidate in [
        "./frozen_east_text_detection.pb",
        "/models/frozen_east_text_detection.pb",
        os.path.expanduser("~/.docuvision/frozen_east_text_detection.pb"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None
