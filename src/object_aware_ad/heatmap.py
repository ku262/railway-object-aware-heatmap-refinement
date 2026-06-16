import cv2
import numpy as np


def normalize_map(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def expand_box(box, w, h, ratio):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return [
        float(max(0, x1 - bw * ratio)),
        float(max(0, y1 - bh * ratio)),
        float(min(w, x2 + bw * ratio)),
        float(min(h, y2 + bh * ratio)),
    ]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _should_merge(a, b, gap, iou_th):
    if iou_xyxy(a, b) >= iou_th:
        return True
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    hgap = max(0, max(bx1 - ax2, ax1 - bx2))
    vgap = max(0, max(by1 - ay2, ay1 - by2))
    return hgap <= gap and vgap <= gap


def merge_boxes(boxes, gap=12, iou_th=0.05):
    merged = []
    for box in boxes:
        for idx, old in enumerate(merged):
            if _should_merge(old, box, gap, iou_th):
                merged[idx] = [
                    float(min(old[0], box[0])),
                    float(min(old[1], box[1])),
                    float(max(old[2], box[2])),
                    float(max(old[3], box[3])),
                ]
                break
        else:
            merged.append(box)
    return merged


def heatmap_to_boxes(
    heatmap,
    orig_w,
    orig_h,
    percentile=92.0,
    blur_ksize=5,
    min_area_ratio=0.0002,
    morph_kernel_size=7,
    max_components=3,
    expand_ratio=0.15,
    merge_gap=12,
    merge_iou_th=0.05,
):
    heat = normalize_map(heatmap)
    if blur_ksize and blur_ksize > 1:
        k = int(blur_ksize) | 1
        heat = cv2.GaussianBlur(heat, (k, k), 0)
    threshold = np.percentile(heat, float(percentile))
    mask = (heat >= threshold).astype(np.uint8)
    if morph_kernel_size and morph_kernel_size > 1:
        k = int(morph_kernel_size) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(1, int(mask.shape[0] * mask.shape[1] * float(min_area_ratio)))
    comps = []
    for cid in range(1, n):
        x, y, bw, bh, area = stats[cid]
        if area < min_area:
            continue
        comp = labels == cid
        score = float(heat[comp].max())
        rank = float(heat[comp].mean() * np.sqrt(float(area)))
        x1 = x / mask.shape[1] * orig_w
        y1 = y / mask.shape[0] * orig_h
        x2 = (x + bw) / mask.shape[1] * orig_w
        y2 = (y + bh) / mask.shape[0] * orig_h
        comps.append({"box": expand_box([x1, y1, x2, y2], orig_w, orig_h, expand_ratio), "score": score, "rank": rank})

    comps.sort(key=lambda r: r["rank"], reverse=True)
    boxes = merge_boxes([r["box"] for r in comps[:max_components]], merge_gap, merge_iou_th)
    scores = [float(comps[i]["score"]) for i in range(min(len(comps), len(boxes)))]
    scores += [1.0] * max(0, len(boxes) - len(scores))
    return boxes[:max_components], scores[:max_components]
