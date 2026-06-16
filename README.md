# Railway Object-Aware Heatmap Refinement

Compact PatchBank anomaly scoring and RGB-guided heatmap refinement for railway
foreign-object inspection. The code keeps only the reusable method components:
DINOv2 patch-token extraction, component-specific normal PatchBank scoring,
heatmap generation, and a compact U-Net-style refiner for converting anomaly
responses into object-aware masks and boxes.

## Install

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Expected Metadata

Each JSONL row should at least provide:

- `relative_crop_path` or `crop_path`
- `label`: `normal` or `part_with_foreign`
- `base_part`
- optional `foreign_bboxes_crop_xyxy` for localization/refiner training

Use `configs/example_config.json` as a path-free template and replace paths with your own dataset and DINOv2 checkpoint.

The paper setting uses DINOv2 ViT-B/14 with `img_size=518`, a random 5% normal
reference bank per component category, and top-3% patch-distance aggregation for
image-level anomaly scoring.

## Core Idea

1. Extract normalized DINOv2 patch tokens from normal component crops.
2. Build a component-specific normal PatchBank.
3. Score query patches by nearest-normal cosine distance.
4. Reshape patch scores into a heatmap.
5. Refine RGB+heatmap with a compact U-Net to obtain an object-aware mask/box.
