# Training and evaluation

The training scripts extend the method modules in `src/` and `tools/`.
They provide a portable primary-method training/inference workflow, grouped inner
validation splits, a common detection evaluator, and heatmap control utilities.
It does not contain railway images, annotations, checkpoints or saved predictions.

The primary configuration retains 5% of normal training patch features and averages
the highest-scoring 3% of query patches. HeatmapRefiner uses RGB+heat, base width 16,
256 x 256 inputs, BCE+Dice, AdamW and 20 epochs. Model and post-processing choices
are frozen before held-out inference. The supplied outer-fold assignments are
preserved; 20% of development annotation groups are reserved for internal validation.
All views of an annotation remain together. This is not train/session/source-image
or physical-component holdout.

## Setup

Run commands from the repository root with Python 3.10+.

```sh
python -m pip install -r runtime/requirements.txt
python -m pip install pytest
```

Use an appropriate PyTorch/CUDA build for your hardware. These dependency ranges
are compatibility requirements, not a record of every historical environment.
The code does not automatically download pretrained weights.

## Input layout

```text
railway-object-aware-heatmap-refinement/
  metadata/records.jsonl
  metadata/splits.json
  images/...
  weights/dinov2_vitb14.pth
  runtime/configs/primary_fixed5.example.json
```

Each metadata record requires `image_id`, `group_id`, `source_id`, `base_part`,
`label` (0/1), original `fold` (0..4), `image_sha256`, relative `image_path`,
integer `width` and `height`, and `gt_boxes` in pixel XYXY coordinates. Normal
images have an empty box list. IDs may be anonymized; group IDs must still connect
all original and augmented views of the same annotation. Supply all five folds.
Replace the example configuration's zero weight digest with the verified SHA256
of your DINOv2 ViT-B/14 checkpoint. Paths are relative to the project directory.

## Primary workflow

```sh
python -B split.py --records metadata/records.jsonl --out metadata/splits.json
python -B train.py --config runtime/configs/primary_fixed5.example.json --check-dependencies
python -B train.py --config runtime/configs/primary_fixed5.example.json --fold 0 --output runs/fold0 --execute --device cuda --permit-gpu
python -B infer.py --run runs/fold0 --output predictions/fold0.jsonl --execute --device cuda --permit-gpu --include-ground-truth
```

Repeat training and inference for folds 1 through 4 with distinct output paths.
Without `--execute`, the training/inference commands perform preflight checks.
Training excludes test images, reference banks use internal-training normal images,
and query matching excludes reference features belonging to the query's annotation
group. Completed run artifacts are hash-checked before inference. Do not edit a
completed run in place; the commands refuse existing output paths.

Concatenate the five `validation_predictions.jsonl` files into `outputs/val.jsonl`
and the five GT-attached test files into `outputs/test.jsonl`, preserving each
record's outer-run `fold`. Then run:

```sh
python -B evaluate.py --splits metadata/splits.json --val outputs/val.jsonl --test outputs/test.jsonl --thresholds outputs/frozen.json --out outputs/metrics.json
```

The evaluator checks exact fold membership and freezes operating thresholds from
validation predictions before opening test predictions. It uses confidence-ranked
one-to-one matching, COCO-style 101-point AP, IoU 0.50:0.05:0.95 and at most 100
boxes per image. AP is calculated before operating-threshold truncation. The
report distinguishes pooled results from five-fold means and sample standard
deviations. Normal images are included in false-positive evaluation.

## Heatmap controls

`heatmap_controls.py` provides validation-only image gates for saved HeatmapBox
predictions and permutation of raw patch-grid scores for inference-only controls.
It does not retrain a model or change the RGB image or target annotations.

```sh
python -B heatmap_controls.py gate --val outputs/direct_val.jsonl --test outputs/direct_test.jsonl --gates outputs/image_gates.json --out outputs/gated_test.jsonl
python -B heatmap_controls.py shuffle --input heatmaps/example.npy --output shuffled/example.npy --seed 42
```

Gate records require `fold`, `base_part`, `image_id`, `image_score`, `gt_boxes`,
`pred_boxes` and `pred_scores`. Use the top-3% score for the proposed PatchBank and
the maximum patch distance for PatchCore. Supply both labels for each validation
fold/component pair. The gate removes boxes from low-scoring images; it does not
reassign box confidences or replace the spatial post-processing parameters.
Ground-truth fields in test rows are retained but never used to select the gate.
This utility is for HeatmapBox; the primary HeatmapRefiner does not use that gate.

The shuffle utility accepts a finite two-dimensional raw score grid and writes a
permuted grid of the same shape. Apply the model's original normalization and
interpolation afterwards. Evaluate the existing aligned-input model with these
alternative heatmaps in a separate inference run, retaining its frozen settings.
The utility itself does not execute that alternative model-inference run.

## Tests and scope

```sh
python -B -m pytest -q -p no:cacheprovider evaluation/test_evaluator.py test_release.py test_heatmap_controls.py runtime/test_evaluator_numeric.py
python -B -m unittest runtime.test_contracts -v
```

The first command tests metadata splits, evaluator behavior and heatmap controls.
Optional pycocotools parity tests are skipped when pycocotools is absent. The
second command requires PyTorch and OpenCV and uses synthetic inputs and mocked
training; it is not a full training reproduction.

This supplement packages the primary workflow, not the entire baseline and
supplementary-experiment campaign. It does not include YOLO/PatchCore training,
synthetic OOD assets, or a guarantee of identical results on another dataset.
Actual-data GPU training was not repeated when assembling this upload package.
