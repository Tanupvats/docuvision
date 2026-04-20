"""
Density-based clustering + bounding-box merging.

Used by the low-resource pipeline:

    detector → many small word boxes
             → cluster spatially close boxes into lines / blocks
             → merge overlapping boxes into a single bbox
             → expand margin + crop
             → feed to a lightweight OCR recognizer
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from docuvision.types import BoundingBox, TextRegion
from docuvision.utils.lazy_import import optional_import
from docuvision.utils.logging import get_logger

log = get_logger("detector.clustering")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _cluster_labels(points: np.ndarray, eps: float,
                    min_samples: int, algorithm: str) -> np.ndarray:
    if algorithm == "hdbscan":
        hdb = optional_import("hdbscan")
        if hdb is not None:
            clusterer = hdb.HDBSCAN(min_cluster_size=max(2, min_samples),
                                    min_samples=min_samples)
            return clusterer.fit_predict(points)
        log.debug("hdbscan not installed, falling back to DBSCAN")

    # DBSCAN via scikit-learn
    sk = optional_import("sklearn.cluster")
    if sk is None:
        # Absolute fallback: put everything in one cluster
        return np.zeros(len(points), dtype=int)
    db = sk.DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(points)


def _union_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    return BoundingBox(x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Overlap-merge (IoU + containment)
# ---------------------------------------------------------------------------
def _merge_overlapping(boxes: List[BoundingBox],
                       iou_threshold: float = 0.05) -> List[BoundingBox]:
    """Greedy union-merge of overlapping boxes.

    We intentionally use a LOW IoU threshold: adjacent words in the same line
    usually overlap only slightly, but we still want to merge them together
    for region-based OCR. Recognition accuracy is unhurt by oversized crops.
    """
    if not boxes:
        return []

    remaining = list(boxes)
    out: List[BoundingBox] = []

    while remaining:
        current = remaining.pop(0)
        merged_any = True
        group = [current]
        while merged_any:
            merged_any = False
            still: List[BoundingBox] = []
            for b in remaining:
                merged = _union_boxes(group)
                if merged.iou(b) >= iou_threshold or _contains(merged, b) or _contains(b, merged):
                    group.append(b)
                    merged_any = True
                else:
                    still.append(b)
            remaining = still
        out.append(_union_boxes(group))

    return out


def _contains(a: BoundingBox, b: BoundingBox) -> bool:
    return (a.x1 <= b.x1 and a.y1 <= b.y1
            and a.x2 >= b.x2 and a.y2 >= b.y2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def cluster_and_merge_boxes(
    regions: List[TextRegion],
    image_shape: Optional[tuple] = None,
    eps: Optional[float] = None,
    min_samples: int = 1,
    algorithm: str = "dbscan",
    margin: int = 4,
    merge_overlaps: bool = True,
    iou_threshold: float = 0.05,
) -> List[TextRegion]:
    """Cluster detected text regions into coherent groups and merge them.

    Parameters
    ----------
    regions : list of TextRegion
        Raw detector output.
    image_shape : tuple, optional
        (H, W) or (H, W, C) of the source image; used to clip expanded boxes
        and to auto-tune `eps` if not provided.
    eps : float, optional
        DBSCAN neighborhood radius in pixels. If None, set to ~1.5 × median
        height of the boxes (a reasonable proxy for line spacing).
    min_samples : int
        DBSCAN/HDBSCAN min cluster size.
    algorithm : {"dbscan", "hdbscan"}
    margin : int
        Pixels to expand each merged box before returning.
    merge_overlaps : bool
        After clustering, also merge any remaining overlapping boxes.
    iou_threshold : float
        IoU threshold for the post-clustering overlap merge.
    """
    if not regions:
        return []

    boxes = [r.bbox for r in regions]

    # Feature vector: box centers. We apply anisotropic scaling — multiplying
    # y by a factor > 1 keeps lines from bleeding into each other while
    # letting word-level boxes on the same line cluster horizontally.
    y_scale = 2.5
    centers = np.array(
        [[b.center[0], b.center[1] * y_scale] for b in boxes],
        dtype=np.float32,
    )

    # Auto-tune eps. Horizontal word gaps are usually on the order of the
    # *width* of a single word — so we use the median bbox width plus some
    # slack. Using median height alone consistently under-estimates eps and
    # leaves every word in its own cluster.
    if eps is None:
        widths = np.array([max(1, b.width) for b in boxes])
        heights = np.array([max(1, b.height) for b in boxes])
        eps = float(max(np.median(widths), np.median(heights))) * 1.2

    labels = _cluster_labels(centers, eps=eps, min_samples=min_samples,
                             algorithm=algorithm)

    groups: dict = {}
    for idx, lbl in enumerate(labels):
        key = int(lbl)
        # Noise (label == -1) becomes its own singleton cluster
        if key == -1:
            key = -(idx + 2)  # unique negative id
        groups.setdefault(key, []).append(idx)

    merged_boxes: List[BoundingBox] = []
    merged_texts: List[List[str]] = []
    merged_confs: List[List[float]] = []

    for key, idxs in groups.items():
        group_boxes = [boxes[i] for i in idxs]
        union = _union_boxes(group_boxes)
        merged_boxes.append(union)
        merged_texts.append([regions[i].text or "" for i in idxs])
        merged_confs.append([regions[i].confidence for i in idxs])

    if merge_overlaps:
        final_boxes = _merge_overlapping(merged_boxes, iou_threshold=iou_threshold)
    else:
        final_boxes = merged_boxes

    # Re-attach (rough) text/confidence info from input regions whose centers
    # fall inside each final box
    h = w = None
    if image_shape is not None:
        h, w = image_shape[:2]

    out: List[TextRegion] = []
    for fb in final_boxes:
        texts: List[str] = []
        confs: List[float] = []
        for r in regions:
            cx, cy = r.bbox.center
            if fb.x1 <= cx <= fb.x2 and fb.y1 <= cy <= fb.y2:
                if r.text:
                    texts.append(r.text)
                confs.append(r.confidence)
        expanded = fb.expand(margin, max_w=w, max_h=h)
        out.append(TextRegion(
            bbox=expanded,
            text=" ".join(texts) if texts else None,
            confidence=float(np.mean(confs)) if confs else 0.0,
        ))

    # Sort top-to-bottom, left-to-right for readable downstream output
    out.sort(key=lambda r: (r.bbox.y1 // max(1, int((eps or 20) / 2)), r.bbox.x1))
    return out
