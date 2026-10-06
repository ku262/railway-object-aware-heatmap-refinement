"""Explicit manifests and leakage-resistant Refiner preprocessing."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 256
MODES = ("rgb_heat", "rgb_only", "heat_only", "rgb_zero", "rgb_shuffle", "feature_fusion")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def indexed(rows):
    result = {}
    for row in rows:
        key = row.get("image_id")
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError(f"Missing/invalid/duplicate image_id: {key!r}")
        result[key] = row
    return result


def resolve(manifest, path):
    if not isinstance(path, str) or not path:
        raise ValueError("An explicit nonempty asset path is required")
    p = Path(path)
    return str((p if p.is_absolute() else Path(manifest).parent / p).resolve())


class Bundle:
    def __init__(self, records, splits, heatmaps, fold=None):
        self.paths = {k: str(Path(v).resolve()) for k, v in
                      dict(records=records, splits=splits, heatmaps=heatmaps).items()}
        self.manifest_hashes = {k: digest(v) for k, v in self.paths.items()}
        rows = indexed(read_jsonl(records))
        heats = indexed(read_jsonl(heatmaps))
        with open(splits, encoding="utf-8-sig") as f:
            split_document = json.load(f)
        required = {"train_ids", "val_ids", "test_ids"}
        if not required.issubset(split_document):
            raise ValueError("Splits must contain train_ids, val_ids, test_ids")
        declared = [v for v in (fold, split_document.get("fold"), split_document.get("outer_fold")) if v is not None]
        if not declared or any(type(v) is not int or v < 0 for v in declared) or len(set(declared)) != 1:
            raise ValueError("Declare consistent nonnegative run fold via --fold or split fold/outer_fold")
        self.fold = declared[0]
        self.splits = {k.removesuffix("_ids"): split_document[k] for k in ("train_ids", "val_ids", "test_ids")}
        ids = []
        for name, members in self.splits.items():
            if not isinstance(members, list) or not members:
                raise ValueError(f"Empty or invalid split: {name}")
            if not all(isinstance(x, str) for x in members):
                raise ValueError("Split IDs must be strings")
            ids.extend(members)
        if len(ids) != len(set(ids)) or set(ids) != set(rows) or set(heats) != set(rows):
            raise ValueError("Splits must partition records; heatmaps must match records exactly")
        self.rows = {}
        path_owners, group_owners = {}, {}
        for split, members in self.splits.items():
            for key in members:
                source = rows[key]
                # Never carry test labels into the runner's records.
                r = {"image_id": key, "image_path": resolve(records, source.get("image_path")),
                     "heatmap_path": resolve(heatmaps, heats[key].get("heatmap_path"))}
                for field in ("base_part", "source_id", "group_id"):
                    if field in source:
                        r[field] = source[field]
                if "fold" in source:
                    if type(source["fold"]) is not int:
                        raise ValueError("Original record fold must be an integer")
                    r["original_fold"] = source["fold"]
                r["fold"] = self.fold
                for field in ("image_path", "heatmap_path"):
                    p = r[field]
                    previous = path_owners.setdefault((field, p), split)
                    if previous != split:
                        raise ValueError(f"Cross-split asset alias: {p}")
                for group_field in ("group_id",):
                    group = source.get(group_field)
                    if group is None:
                        continue
                    if not isinstance(group, str) or not group:
                        raise ValueError(f"{group_field} must be a nonempty string")
                    if group_owners.setdefault((group_field, group), split) != split:
                        raise ValueError(f"Cross-split group: {group}")
                if split != "test":
                    if ("target_path" in source) == ("gt_boxes" in source):
                        raise ValueError(f"{key}: require exactly one target_path or gt_boxes")
                    if "target_path" in source:
                        r["target_path"] = resolve(records, source["target_path"])
                    else:
                        r["gt_boxes"] = source["gt_boxes"]
                self.rows[key] = r
        if not all("group_id" in r for r in self.rows.values()):
            raise ValueError("group_id is required on every record for annotation-group disjointness")

    def source_overlap(self):
        result = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            left = {r["source_id"] for r in self.records(a) if "source_id" in r}
            right = {r["source_id"] for r in self.records(b) if "source_id" in r}
            result[f"{a}_{b}"] = sorted(left & right)
        return result

    def records(self, split):
        return [self.rows[key] for key in self.splits[split]]

    def asset_hashes(self, splits, targets=True):
        result, image_owners = {}, {}
        for split in splits:
            for r in self.records(split):
                fields = ["image_path", "heatmap_path"]
                if targets and split != "test" and "target_path" in r:
                    fields.append("target_path")
                for field in fields:
                    sha = digest(r[field])
                    result[f"{split}:{r['image_id']}:{field}"] = sha
                    if field == "image_path":
                        if image_owners.setdefault(sha, split) != split:
                            raise ValueError("Identical image bytes occur across splits")
        return result


def read_heat(path):
    a = np.load(path, allow_pickle=False)
    if a.ndim != 2 or min(a.shape) < 1 or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"Heatmap must be numeric 2-D raw distances: {path}")
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite heatmap: {path}")
    return a


def image_array(record):
    with Image.open(record["image_path"]) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def checked_boxes(boxes, width, height):
    if not isinstance(boxes, list):
        raise ValueError("gt_boxes must be an explicit list")
    out = []
    for box in boxes:
        a = np.asarray(box, dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all():
            raise ValueError("Invalid xyxy box")
        x1, y1, x2, y2 = a
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Box outside image or nonpositive area")
        out.append(a.tolist())
    return out


def target_array(record, width, height):
    if "target_path" in record:
        path = Path(record["target_path"])
        if path.suffix.lower() == ".npy":
            a = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as im:
                a = np.asarray(im)
        if a.shape != (height, width) or not np.isin(a, [0, 1, 255]).all():
            raise ValueError("Target must be binary and have original image dimensions")
        a = (a > 0).astype(np.float32)
    else:
        a = np.zeros((height, width), np.float32)
        for x1, y1, x2, y2 in checked_boxes(record["gt_boxes"], width, height):
            x1, y1 = min(width - 1, round(x1)), min(height - 1, round(y1))
            a[y1:round(y2), x1:round(x2)] = 1
    return cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def fit_normalization(train_records, mode):
    if mode == "per_image":
        return {"mode": mode}
    if mode != "train_quantile":
        raise ValueError(mode)
    normal = []
    for r in train_records:
        with Image.open(r["image_path"]) as im:
            # Use original-resolution labels: a tiny positive must not become normal
            # merely because nearest-neighbor downsampling erased its pixels.
            if "gt_boxes" in r:
                is_normal = len(checked_boxes(r["gt_boxes"], *im.size)) == 0
            else:
                target_array(r, *im.size)  # validate before inspecting original pixels
                path = Path(r["target_path"])
                if path.suffix.lower() == ".npy":
                    mask = np.load(path, allow_pickle=False)
                else:
                    with Image.open(path) as mask_image:
                        mask = np.asarray(mask_image)
                is_normal = not np.any(mask)
        if is_normal:
            normal.append(r)
    if not normal:
        raise ValueError("train_quantile requires normal training maps")
    values = np.concatenate([read_heat(r["heatmap_path"]).ravel() for r in normal])
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= low:
        raise ValueError("Degenerate train-only distance quantiles")
    return {"mode": mode, "low": float(low), "high": float(high), "quantiles": [0.01, 0.99],
            "fit_ids": [r["image_id"] for r in normal], "fit_population": "normal_train_only"}


def normalize(heat, stats):
    if stats["mode"] == "per_image":
        return (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return np.clip((heat - stats["low"]) / (stats["high"] - stats["low"]), 0, 1)


def id_seed(seed, image_id, purpose):
    return int(canonical_hash([seed, image_id, purpose])[:16], 16)


def shuffle_grid(heat, seed, image_id, grid_size=16):
    h, w = heat.shape
    if grid_size < 2 or h % grid_size or w % grid_size:
        raise ValueError("Shuffle grid must divide both heat dimensions and be >= 2")
    ph, pw = h // grid_size, w // grid_size
    patches = heat.reshape(grid_size, ph, grid_size, pw).transpose(0, 2, 1, 3)
    patches = patches.reshape(grid_size * grid_size, ph, pw)
    rng = np.random.default_rng(id_seed(seed, image_id, "patch-grid"))
    shuffled = patches[rng.permutation(len(patches))]
    return shuffled.reshape(grid_size, grid_size, ph, pw).transpose(0, 2, 1, 3).reshape(h, w)


class RefinerDataset:
    def __init__(self, records, mode, stats, seed=42, epoch=None, targets=True, shuffle_grid_size=16):
        self.records, self.mode, self.stats = records, mode, stats
        self.seed, self.epoch, self.targets = seed, epoch, targets
        self.shuffle_grid_size = shuffle_grid_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        rgb = image_array(r)
        h, w = rgb.shape[:2]
        heat = normalize(read_heat(r["heatmap_path"]), self.stats)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        heat = cv2.resize(heat, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        if self.mode == "rgb_shuffle":
            heat = shuffle_grid(heat, self.seed, r["image_id"], self.shuffle_grid_size)
        rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        target = target_array(r, w, h) if self.targets else np.zeros((SIZE, SIZE), np.float32)
        if self.epoch is not None:
            rng = np.random.default_rng(id_seed(self.seed, r["image_id"], ["flip", self.epoch]))
            if rng.random() < 0.5:
                rgb, heat, target = rgb[:, ::-1], heat[:, ::-1], target[:, ::-1]
        rgb = rgb.transpose(2, 0, 1)
        if self.mode == "rgb_only":
            x = rgb
        elif self.mode == "heat_only":
            x = heat[None]
        elif self.mode in MODES:
            channel = np.zeros_like(heat) if self.mode == "rgb_zero" else heat
            x = np.concatenate([rgb, channel[None]], axis=0)
        else:
            raise ValueError(self.mode)
        return np.ascontiguousarray(x), np.ascontiguousarray(target[None]), index


def positive_weight(train_records):
    positives, total = 0, 0
    for r in train_records:
        with Image.open(r["image_path"]) as im:
            target = target_array(r, *im.size)
        positives += int(target.sum())
        total += target.size
    if not 0 < positives < total:
        raise ValueError("Weighted BCE requires both positive and negative training pixels")
    return (total - positives) / positives
