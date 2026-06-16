import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_aware_ad.heatmap import heatmap_to_boxes
from object_aware_ad.refiner import SmallUNet, predict_mask


def parse_args():
    p = argparse.ArgumentParser(description="Apply the object-aware RGB+heatmap refiner to one crop.")
    p.add_argument("--image", required=True)
    p.add_argument("--heatmap", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output-mask", required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--input-mode", choices=["rgb_heat", "heat_only", "rgb_only"], default="rgb_heat")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    in_ch = {"rgb_heat": 4, "heat_only": 1, "rgb_only": 3}[args.input_mode]
    model = SmallUNet(in_ch=in_ch, base=args.base_channels)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    heatmap = np.load(args.heatmap)
    mask = predict_mask(model, args.image, heatmap, args.img_size, args.input_mode, args.device)
    np.save(args.output_mask, mask.astype(np.float32))
    boxes, scores = heatmap_to_boxes(mask, mask.shape[1], mask.shape[0], percentile=50.0)
    print({"mask": args.output_mask, "boxes": boxes, "scores": scores})


if __name__ == "__main__":
    main()
