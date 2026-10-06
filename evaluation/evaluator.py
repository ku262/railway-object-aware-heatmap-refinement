"""Category-agnostic, non-crowd XYXY detection evaluation (NumPy only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

IOUS = np.linspace(0.5, 0.95, 10)
RECALLS = np.linspace(0, 1, 101)
FIELDS = ("image_id", "base_part", "source_id", "fold", "width", "height",
          "gt_boxes", "pred_boxes", "pred_scores")
METRICS = ("ap50", "ap50_95", "precision", "recall", "f1", "fp_perimage", "normalfpr")
PROTOCOL = "zhongche-coco101-agnostic-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def key(record):
    return (record["fold"], record["image_id"])


def validate(records):
    """Reject malformed inputs instead of silently dropping/clipping predictions."""
    seen = set()
    for r in records:
        if any(f not in r for f in FIELDS):
            raise ValueError("Missing required record fields")
        for f in ("image_id", "base_part", "source_id"):
            if not isinstance(r[f], str) or not r[f]:
                raise ValueError(f"{f} must be a nonempty string")
        if type(r["fold"]) is not int:
            raise ValueError("fold must be an integer")
        if key(r) in seen:
            raise ValueError(f"Duplicate fold/image_id: {key(r)}")
        seen.add(key(r))
        for f in ("width", "height"):
            if type(r[f]) is not int or r[f] <= 0:
                raise ValueError(f"{f} must be a positive integer")
        for f in ("gt_boxes", "pred_boxes"):
            if not isinstance(r[f], list):
                raise ValueError(f"{f} must be an array")
            for b in r[f]:
                if not isinstance(b, list) or len(b) != 4 or any(
                    not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) or not np.isfinite(v) for v in b
                ):
                    raise ValueError("Boxes must contain four finite numbers")
                if not (0 <= b[0] < b[2] <= r["width"] and
                        0 <= b[1] < b[3] <= r["height"]):
                    raise ValueError("Boxes must be positive-area in-bounds XYXY")
        scores = r["pred_scores"]
        if not isinstance(scores, list) or len(scores) != len(r["pred_boxes"]):
            raise ValueError("pred_boxes/pred_scores length mismatch")
        if any(not isinstance(s, Real) or isinstance(s, (bool, np.bool_)) or not np.isfinite(s) or not 0 <= s <= 1
               for s in scores):
            raise ValueError("Scores must be finite probabilities in [0,1]")
    if not records:
        raise ValueError("Empty dataset (empty predictions per image are supported)")
    return sorted(records, key=key)


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        return validate([json.loads(line) for line in stream if line.strip()])


def iou_matrix(pred, gt):
    p = np.asarray(pred, dtype=float).reshape(-1, 4)
    g = np.asarray(gt, dtype=float).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(p[:, None, 2:], g[None, :, 2:]) -
                    np.maximum(p[:, None, :2], g[None, :, :2]))
    inter = wh.prod(axis=2)
    return inter / ((p[:, 2:] - p[:, :2]).prod(axis=1)[:, None] +
                    (g[:, 2:] - g[:, :2]).prod(axis=1)[None, :] - inter)


def relative_areas(boxes, record):
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (b[:, 2:] - b[:, :2]).prod(axis=1) / (record["width"] * record["height"])


def prepare(records, area_by_fold=None):
    """Cache COCO greedy matches. Size bins use COCO ignore semantics, no crowd."""
    prepared = []
    for r in records:
        order = np.argsort(-np.asarray(r["pred_scores"], dtype=float), kind="stable")[:100]
        boxes = [r["pred_boxes"][i] for i in order]
        scores = np.asarray(r["pred_scores"], dtype=float)[order]
        lo, hi = (0, float("inf")) if area_by_fold is None else area_by_fold[r["fold"]]
        ga = relative_areas(r["gt_boxes"], r)
        ignored = (ga < lo) | (ga >= hi)
        go = np.argsort(ignored, kind="stable")
        ignored = ignored[go]
        overlaps = iou_matrix(boxes, [r["gt_boxes"][i] for i in go])
        pa = relative_areas(boxes, r)
        outside = (pa < lo) | (pa >= hi)
        tp = np.zeros((10, len(order)), dtype=bool)
        fp = np.zeros_like(tp)
        for ti, threshold in enumerate(IOUS):
            used = set()
            for di in range(len(order)):
                best, match = min(threshold, 1 - 1e-10), -1
                for gi in range(len(go)):
                    if gi in used:
                        continue
                    if match >= 0 and not ignored[match] and ignored[gi]:
                        break
                    if overlaps[di, gi] < best:
                        continue
                    best, match = overlaps[di, gi], gi
                if match >= 0:
                    used.add(match)
                    tp[ti, di] = not ignored[match]
                else:
                    fp[ti, di] = not outside[di]
        prepared.append({"scores": scores, "tp": tp, "fp": fp,
                         "n_gt": int((~ignored).sum()), "normal": not r["gt_boxes"],
                         "fold": r["fold"], "source_id": r["source_id"]})
    return prepared


def aggregate(prepared, thresholds):
    n_gt = sum(p["n_gt"] for p in prepared)
    n_images = len(prepared)
    scores = np.concatenate([p["scores"] for p in prepared]) if prepared else np.array([])
    order = np.argsort(-scores, kind="stable")
    aps = []
    for ti in range(10):
        tp = np.concatenate([p["tp"][ti] for p in prepared])[order] if prepared else np.array([])
        fp = np.concatenate([p["fp"][ti] for p in prepared])[order] if prepared else np.array([])
        ct, cf = np.cumsum(tp), np.cumsum(fp)
        precision = ct / (ct + cf + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        sampled = np.zeros(101)
        if n_gt:
            idx = np.searchsorted(ct / n_gt, RECALLS, side="left")
            valid = idx < len(precision)
            sampled[valid] = precision[idx[valid]]
        aps.append(float(sampled.mean()) if n_gt else None)
    tp = fp = normal_fp = n_normal = 0
    for p in prepared:
        threshold = thresholds[p["fold"]]
        keep = p["scores"] >= threshold if threshold is not None else np.zeros(len(p["scores"]), bool)
        tp += int(p["tp"][0, keep].sum())
        fp += int(p["fp"][0, keep].sum())
        n_normal += int(p["normal"])
        normal_fp += int(p["normal"] and bool(p["fp"][0, keep].any()))
    fn = n_gt - tp
    return {"ap50": aps[0], "ap50_95": float(np.mean(aps)) if n_gt else None,
            "ap_by_iou": {f"{t:.2f}": a for t, a in zip(IOUS, aps)},
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / n_gt if n_gt else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if n_gt else None,
            "fp_perimage": fp / n_images if n_images else None,
            "normalfpr": normal_fp / n_normal if n_normal else None,
            "tp": tp, "fp": fp, "fn": fn, "n_images": n_images,
            "n_gt": n_gt, "n_normal": n_normal, "normal_fp_images": normal_fp}


def evaluate(records, fixedthreshold):
    """Evaluate TEST with a fixed scalar/None or fold->threshold mapping; never tune.

    Does not mutate records or the threshold mapping. None means reject all.
    AP always uses all top-100 predictions, independent of this threshold.
    """
    records = validate(records)
    thresholds = fixedthreshold if isinstance(fixedthreshold, dict) else {
        r["fold"]: fixedthreshold for r in records}
    for fold in {r["fold"] for r in records}:
        if fold not in thresholds:
            raise ValueError(f"Missing fixed threshold for fold {fold}")
        t = thresholds[fold]
        if t is not None and (type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t <= 1):
            raise ValueError("Fixed threshold must be None or a finite number in [0,1]")
    return aggregate(prepare(records), thresholds)


def select_threshold(records):
    """Best F1 at IoU .50, tied F1 -> highest threshold; None rejects all."""
    cache = prepare(validate(records))
    total = sum(p["n_gt"] for p in cache)
    if total == 0:
        raise ValueError("Validation fold has no GT: BestF1 is undefined")
    events = {}
    for p in cache:
        for s, tp, fp in zip(p["scores"], p["tp"][0], p["fp"][0]):
            counts = events.setdefault(float(s), [0, 0])
            counts[0] += int(tp)
            counts[1] += int(fp)
    best, threshold, tp, fp = 0.0, None, 0, 0
    for score in sorted(events, reverse=True):
        dt, df = events[score]
        tp, fp = tp + dt, fp + df
        f1 = 2 * tp / (total + tp + fp)
        if f1 > best:
            best, threshold = f1, score
    return threshold, best


def freeze_thresholds(val_records):
    records = validate(val_records)
    folds = sorted({r["fold"] for r in records})
    selected = {str(f): dict(zip(("threshold", "validation_f1"),
                                select_threshold([r for r in records if r["fold"] == f])))
                for f in folds}
    payload = {"protocol": PROTOCOL, "selected_on": "validation_only",
               "validation_records_sha256": digest(records),
               "selection": "IoU=.50; score>=threshold; highest threshold on F1 tie; null=reject_all",
               "max_detections_per_image": 100, "folds": selected}
    return {**payload, "sha256": digest(payload)}


def thaw_thresholds(artifact):
    payload = {k: v for k, v in artifact.items() if k != "sha256"}
    if digest(payload) != artifact.get("sha256") or artifact.get("protocol") != PROTOCOL:
        raise ValueError("Threshold artifact hash/protocol mismatch")
    return {int(f): v["threshold"] for f, v in artifact["folds"].items()}


def check_splits(val, test, train=None):
    if {r["fold"] for r in val} != {r["fold"] for r in test}:
        raise ValueError("Validation/test fold sets must match")
    splits = [("val", val), ("test", test)] + ([("train", train)] if train is not None else [])
    all_records = [r for _, rows in splits for r in rows]
    has_groups = any("group_id" in r for r in all_records)
    if has_groups and any(not isinstance(r.get("group_id"), str) or not r["group_id"] for r in all_records):
        raise ValueError("group_id must be provided as a nonempty string on every record when used")
    overlaps = {}
    for i, (aname, a) in enumerate(splits):
        for bname, b in splits[i + 1:]:
            for field in (("image_id", "group_id") if has_groups else ("image_id",)):
                if {(r["fold"], r[field]) for r in a} & {(r["fold"], r[field]) for r in b}:
                    raise ValueError(f"Split leakage within fold: {field}")
            overlaps[aname + "_" + bname] = {
                str(f): len({r["source_id"] for r in a if r["fold"] == f} &
                            {r["source_id"] for r in b if r["fold"] == f})
                for f in sorted({r["fold"] for r in a + b})}
    return {"image_disjoint": True, "annotation_group_disjoint": True if has_groups else None,
            "source_overlap_counts": overlaps, "source_heldout_claim": False,
            "note": "Original outer folds retained; source overlap is reported, not rejected."}


def fit_size_thresholds(train_records):
    records = validate(train_records)
    folds = {}
    for fold in sorted({r["fold"] for r in records}):
        areas = [a for r in records if r["fold"] == fold for a in relative_areas(r["gt_boxes"], r)]
        if not areas:
            raise ValueError("Training fold lacks GT for size cutoffs")
        folds[str(fold)] = np.quantile(areas, [1 / 3, 2 / 3], method="linear").tolist()
    payload = {"protocol": PROTOCOL, "fitted_on": "training_only",
               "area": "box_area / image_area", "quantiles": [1 / 3, 2 / 3],
               "training_records_sha256": digest(records), "folds": folds}
    return {**payload, "sha256": digest(payload)}


def fold_statistics(rows):
    result = {}
    for metric in METRICS:
        values = [row[metric] for row in rows.values() if row[metric] is not None]
        result[metric] = {"mean": float(np.mean(values)) if values else None,
                          "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                          "n_defined_folds": len(values), "n_folds": len(rows)}
    return result


def summarize(records, thresholds, size_artifact=None):
    records = validate(records)
    cache = prepare(records)
    folds = {str(f): aggregate([p for p in cache if p["fold"] == f], thresholds)
             for f in sorted({r["fold"] for r in records})}
    parts = {part: aggregate(prepare([r for r in records if r["base_part"] == part]), thresholds)
             for part in sorted({r["base_part"] for r in records})}
    sizes = {}
    if size_artifact is not None:
        if digest({k: v for k, v in size_artifact.items() if k != "sha256"}) != size_artifact["sha256"]:
            raise ValueError("Size artifact hash mismatch")
        for i, name in enumerate(("small", "medium", "large")):
            ranges = {int(f): ([0, *cuts, float("inf")][i], [0, *cuts, float("inf")][i + 1])
                      for f, cuts in size_artifact["folds"].items()}
            row = aggregate(prepare(records, ranges), thresholds)
            # Normal-image FPR has no object size and is reported only overall/by part.
            row["normalfpr"] = None
            row["normal_fp_images"] = None
            sizes[name] = row
    return {"pooled": aggregate(cache, thresholds), "folds": folds,
            "fold_statistics": fold_statistics(folds), "parts": parts, "sizes": sizes,
            "size_status": "train_fitted" if sizes else "not_requested_no_training_records"}


def cohort_signature(records):
    return [{k: r[k] for k in FIELDS if k not in ("pred_boxes", "pred_scores")}
            for r in records]


def paired_bootstrap(method_records, method_thresholds, n_boot=2000, seed=20261006):
    """Resample whole source clusters globally; same draws for every method.

    A source appearing in several folds is sampled once with all its occurrences.
    Estimand is pooled out-of-fold detection performance, not fold-mean performance.
    """
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    methods = list(method_records)
    if not methods or set(methods) != set(method_thresholds):
        raise ValueError("Methods/thresholds must be nonempty and aligned")
    records = {m: validate(method_records[m]) for m in methods}
    reference = records[methods[0]]
    if any(cohort_signature(records[m]) != cohort_signature(reference) for m in methods):
        raise ValueError("Paired methods must have identical images, folds, sources, parts, dimensions and GT")
    sources = sorted({r["source_id"] for r in reference})
    groups = [[i for i, r in enumerate(reference) if r["source_id"] == s] for s in sources]
    caches = {m: prepare(records[m]) for m in methods}
    rng = np.random.default_rng(seed)
    samples = {m: {k: [] for k in METRICS} for m in methods}
    draws_hash = hashlib.sha256()
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        draws_hash.update(canonical(draw.tolist()) + b"\n")
        indices = [i for group in draw for i in groups[group]]
        for m in methods:
            row = aggregate([caches[m][i] for i in indices], method_thresholds[m])
            for k in METRICS:
                samples[m][k].append(row[k])

    def interval(values):
        good = [v for v in values if v is not None]
        return {"ci95": np.quantile(good, [.025, .975]).tolist() if good else None,
                "n_defined": len(good), "n_undefined": n_boot - len(good)}

    differences = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            differences[f"{b} minus {a}"] = {k: interval([
                y - x if x is not None and y is not None else None
                for x, y in zip(samples[a][k], samples[b][k])]) for k in METRICS}
    return {"unit": "source_id_global_cluster", "estimand": "pooled_test_metrics",
            "n_boot": n_boot, "seed": seed, "n_sources": len(sources),
            "sources_sha256": digest(sources), "draws_sha256": draws_hash.hexdigest(),
            "cohort_sha256": digest(cohort_signature(reference)),
            "methods": {m: {k: interval(v) for k, v in samples[m].items()} for m in methods},
            "paired_differences": differences}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-pred", required=True, action="append", type=Path)
    parser.add_argument("--test-pred", required=True, action="append", type=Path)
    parser.add_argument("--method", action="append", help="Names aligned with repeated val/test arguments")
    parser.add_argument("--train-records", type=Path, help="Same schema; training-only size quantiles")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    methods = args.method or [f"method_{i + 1}" for i in range(len(args.val_pred))]
    if len(args.val_pred) != len(args.test_pred) or len(methods) != len(args.val_pred):
        parser.error("Provide equally many --val-pred, --test-pred and --method values")
    if len(set(methods)) != len(methods) or any(not m or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in m) for m in methods):
        parser.error("Unique method names must use letters, numbers, underscore or hyphen")
    train = read_jsonl(args.train_records) if args.train_records else None
    size = fit_size_thresholds(train) if train is not None else None
    tests, thresholds, artifacts, summaries = {}, {}, {}, {}
    for method, vp, tp in zip(methods, args.val_pred, args.test_pred):
        val, test = read_jsonl(vp), read_jsonl(tp)
        split_report = check_splits(val, test, train)
        if train is not None and {r["fold"] for r in train} != {r["fold"] for r in test}:
            raise ValueError("Training/test fold sets must match for size summaries")
        artifact = freeze_thresholds(val)
        thresholds[method] = thaw_thresholds(artifact)
        tests[method], artifacts[method] = test, artifact
        summaries[method] = summarize(test, thresholds[method], size)
        summaries[method]["split_audit"] = split_report
        summaries[method]["test_records_sha256"] = digest(test)
        summaries[method]["threshold_artifact_sha256"] = artifact["sha256"]
    bootstrap = paired_bootstrap(tests, thresholds, args.bootstrap, args.seed)
    # Refuse overwrites so an evaluation cannot silently replace a frozen result.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"summary.json": {"protocol": PROTOCOL, "methods": summaries},
               "bootstrap.json": bootstrap}
    outputs.update({f"{m}.thresholds.json": a for m, a in artifacts.items()})
    if size is not None:
        outputs["size_thresholds.json"] = size
    if any((args.out_dir / name).exists() for name in outputs):
        raise FileExistsError("Evaluation output exists; choose a fresh --out-dir")
    for name, value in outputs.items():
        write_json(args.out_dir / name, value)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "files": sorted(outputs)}))


if __name__ == "__main__":
    main()
