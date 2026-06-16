import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def get_crop_path(record, dataset_dir):
    rel = record.get("relative_crop_path")
    if rel:
        rel = rel.replace("\\", "/").lstrip("/")
        return Path(dataset_dir).joinpath(*[p for p in rel.split("/") if p])
    return Path(record["crop_path"])


def load_records(metadata_jsonl, dataset_dir, valid_parts=None):
    valid_parts = set(valid_parts or [])
    records = []

    with open(metadata_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            if not r.get("saved", True):
                continue
            if r.get("exclude", False):
                continue
            if r.get("label") not in {"normal", "part_with_foreign"}:
                continue
            if valid_parts and r.get("base_part") not in valid_parts:
                continue
            if r["label"] == "part_with_foreign" and len(r.get("foreign_bboxes_crop_xyxy", [])) == 0:
                continue

            crop_path = get_crop_path(r, dataset_dir)
            if not crop_path.exists():
                continue

            r["resolved_crop_path"] = str(crop_path)
            r["y"] = 1 if r["label"] == "part_with_foreign" else 0
            r["group_id"] = (
                f"{r.get('dataset_idx', '')}_"
                f"{r.get('source_file_name', '')}_"
                f"{r.get('base_part', '')}_"
                f"{r.get('part_ann_id', '')}"
            )
            records.append(r)

    return records


def summarize_records(records):
    by_part = defaultdict(Counter)
    for r in records:
        by_part[r["base_part"]][r["label"]] += 1
    return {part: dict(counts) for part, counts in sorted(by_part.items())}


def records_by_part(records):
    out = defaultdict(list)
    for r in records:
        out[r["base_part"]].append(r)
    return dict(out)


def valid_binary_parts(records):
    parts = []
    for part, rs in records_by_part(records).items():
        if {r["y"] for r in rs} == {0, 1}:
            parts.append(part)
    return sorted(parts)


def make_part_folds(part_records, n_splits, seed):
    y = np.array([r["y"] for r in part_records])
    groups = np.array([r["group_id"] for r in part_records])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        train_records = [part_records[i] for i in train_idx]
        test_records = [part_records[i] for i in test_idx]
        yield fold, train_records, test_records

