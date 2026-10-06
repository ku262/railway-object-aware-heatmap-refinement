"""Absolute-probability mask sweep; scores stay attached during merging."""
import itertools

import cv2
import numpy as np

THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
AREAS = (0.001, 0.003, 0.005, 0.01)
COMPONENTS = (1, 2, 3)
FIXED = dict(blur_ksize=3, morph_kernel_size=5, expand_ratio=0.05,
             merge_gap=8, merge_iou_th=0.05, merge_score="max_member")


def grid():
    for t, a, c in itertools.product(THRESHOLDS, AREAS, COMPONENTS):
        yield dict(threshold=t, min_area_ratio=a, max_components=c)


def should_merge(a, b, gap, iou):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    overlap = ix * iy / union if union else 0
    dx = max(0, a[0] - b[2], b[0] - a[2])
    dy = max(0, a[1] - b[3], b[1] - a[3])
    return overlap >= iou or (dx <= gap and dy <= gap)


def merge_scored(boxes, scores, gap=8, iou=0.05):
    if len(boxes) != len(scores):
        raise ValueError("Box/score length mismatch")
    groups = [(list(b), float(s)) for b, s in zip(boxes, scores)]
    # Restart after each union so a bridging component cannot leave two groups.
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, sa = groups[i]
                b, sb = groups[j]
                if should_merge(a, b, gap, iou):
                    groups[i] = ([min(a[0], b[0]), min(a[1], b[1]),
                                  max(a[2], b[2]), max(a[3], b[3])], max(sa, sb))
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [b for b, _ in groups], [s for _, s in groups]


def mask_to_boxes(prob, width, height, threshold, min_area_ratio, max_components):
    prob = np.asarray(prob, dtype=np.float32)
    if (prob.ndim != 2 or not np.isfinite(prob).all() or
            prob.min() < 0 or prob.max() > 1 or width <= 0 or height <= 0):
        raise ValueError("Expected finite sigmoid probability mask and positive dimensions")
    smooth = cv2.GaussianBlur(prob, (3, 3), 0)
    binary = (smooth >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = max(1, int(binary.size * min_area_ratio))
    components = []
    for cid in range(1, n):
        x, y, w, h, area = stats[cid]
        if area < minimum:
            continue
        values = smooth[labels == cid]
        x1, x2 = x / binary.shape[1] * width, (x + w) / binary.shape[1] * width
        y1, y2 = y / binary.shape[0] * height, (y + h) / binary.shape[0] * height
        ex, ey = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        box = [max(0., x1 - ex), max(0., y1 - ey), min(float(width), x2 + ex), min(float(height), y2 + ey)]
        components.append((float(values.mean() * np.sqrt(area)), box, float(values.max())))
    components.sort(key=lambda x: x[0], reverse=True)
    selected = components[:max_components]
    boxes, scores = merge_scored([x[1] for x in selected], [x[2] for x in selected])
    return boxes, scores, binary


def selection_key(metrics):
    values = [float(metrics[k]) for k in ("AP50", "F1", "FP")]
    if not np.isfinite(values).all() or not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and values[2] >= 0):
        raise ValueError("Scorer must return finite AP50/F1 fractions and nonnegative FP")
    return values[0], values[1], -values[2]
