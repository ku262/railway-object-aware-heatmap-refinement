import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_aware_ad.bank import build_normal_bank
from object_aware_ad.data import load_records, records_by_part
from object_aware_ad.dinov2 import build_transform, get_or_extract_patch_tokens, load_dinov2
from object_aware_ad.heatmap import heatmap_to_boxes
from object_aware_ad.scoring import compute_image_scores, compute_patch_anomaly
from object_aware_ad.utils import read_json, resolve_device, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Build normal PatchBanks and score query crops.")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sampling", choices=["none", "random", "coreset", "trimmed_random", "trimmed_coreset"], default="random")
    p.add_argument("--bank-ratio", type=float, default=0.05)
    p.add_argument("--trim-ratio", type=float, default=0.0)
    p.add_argument("--box-percentile", type=float, default=92.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(args.output)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(cfg["metadata_jsonl"], cfg["dataset_dir"], cfg.get("valid_parts"))
    by_part = records_by_part(records)
    model = load_dinov2(cfg["dinov2_ckpt"], int(cfg["img_size"]), device)
    transform = build_transform(int(cfg["img_size"]))

    def get_patch(record):
        return get_or_extract_patch_tokens(record, model, transform, cfg, device)

    rows = []
    for part, part_records in sorted(by_part.items()):
        normal_records = [r for r in part_records if r["y"] == 0]
        query_records = [r for r in part_records if r["y"] == 1]
        if not normal_records or not query_records:
            continue
        payload = build_normal_bank(normal_records, get_patch, args.sampling, args.bank_ratio, args.trim_ratio)
        bank = payload["bank"].to(device)
        for record in query_records:
            patch, grid_hw = get_patch(record)
            anomaly = compute_patch_anomaly(patch, bank, int(cfg.get("score_chunk_size", 512)), device)
            heatmap = anomaly.reshape(grid_hw).astype(np.float32)
            image = Image.open(record["resolved_crop_path"])
            boxes, scores = heatmap_to_boxes(heatmap, image.width, image.height, percentile=args.box_percentile)
            heat_path = heat_dir / f"{Path(record['resolved_crop_path']).stem}.npy"
            np.save(heat_path, heatmap)
            rows.append({
                "image": record.get("relative_crop_path", record["resolved_crop_path"]),
                "base_part": part,
                "heatmap": str(heat_path),
                "scores": compute_image_scores(anomaly, cfg.get("score_top_patch_nums", []), cfg.get("score_top_ratios", [0.03])),
                "boxes": boxes,
                "box_scores": scores,
            })
        del bank
        torch.cuda.empty_cache()
    write_json(out_dir / "predictions.json", rows)
    print(out_dir / "predictions.json")


if __name__ == "__main__":
    main()
