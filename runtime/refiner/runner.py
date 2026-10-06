"""Strict, explicit-input revision Refiner CLI. No legacy data discovery."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import random
import sys
import uuid

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import (Bundle, MODES, RefinerDataset, canonical_hash, checked_boxes, digest,
                  fit_normalization, indexed, positive_weight, read_jsonl)
from .model import SmallUNet, total_loss
from .postprocess import FIXED, grid, mask_to_boxes, selection_key
from .evaluation_bridge import EvaluationBridge, load_module

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = None


def configure(project_root):
    global PROJECT_ROOT
    PROJECT_ROOT = Path(project_root).resolve()


def portable(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


PROTOCOL = dict(base=16, input_size=256, batch_size=16, lr=1e-4,
                weight_decay=1e-4, epochs=20, optimizer="AdamW", selection="final")


def output_path(path):
    if PROJECT_ROOT is None:
        raise ValueError("Configure the portable project root first")
    path = Path(path)
    path = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(PROJECT_ROOT) or path == PROJECT_ROOT:
        raise ValueError("Output must remain within the configured project root")
    return path


def atomic(path, writer):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, obj):
    atomic(path, lambda f: f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")))


def save_jsonl(path, rows):
    atomic(path, lambda f: f.write("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode("utf-8")))


def save_npy(path, array):
    atomic(path, lambda f: np.save(f, array, allow_pickle=False))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextlib.contextmanager
def run_lock(run):
    run = output_path(run)
    run.mkdir(parents=True, exist_ok=True)
    lock = output_path(run / ".lock")
    with open(lock, "x", encoding="ascii") as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def source_hashes():
    return {name: digest(ROOT / name) for name in ("runner.py", "data.py", "model.py", "postprocess.py", "evaluation_bridge.py")}


def versions():
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                opencv=cv2.__version__, pillow=PIL.__version__)


def select_device(name, permit_gpu):
    if name != "cpu":
        if not permit_gpu:
            raise ValueError("CUDA is forbidden without explicit parent permission and --permit-gpu")
        if not name.startswith("cuda"):
            raise ValueError("Only cpu or explicit cuda device is supported")
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA is unavailable")
    return torch.device(name)


def prepare(args):
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    bundle = Bundle(args.records, args.splits, args.heatmaps, fold=args.fold)
    assets = bundle.asset_hashes(("train", "val"))
    stats = fit_normalization(bundle.records("train"), args.normalization)
    weight = positive_weight(bundle.records("train")) if args.loss == "weighted_bce_dice" else None
    # Validate every training/validation target before any optimization; no test assets.
    for split in ("train", "val"):
        ds = RefinerDataset(bundle.records(split), args.mode, stats, args.seed, shuffle_grid_size=args.shuffle_grid)
        for i in range(len(ds)):
            ds[i]
    config = dict(protocol=PROTOCOL, mode=args.mode, loss=args.loss, seed=args.seed,
                  normalization=stats, pos_weight=weight, manifests={k: portable(v) for k, v in bundle.paths.items()},
                  manifest_hashes=bundle.manifest_hashes, asset_hashes=assets,
                  sources=source_hashes(), versions=versions(), device=args.device,
                  shuffle_grid=args.shuffle_grid, source_overlap=bundle.source_overlap(), fold=bundle.fold,
                  focal=dict(gamma=2, alpha=0.25), num_workers=args.num_workers,
                  split_counts={k: len(v) for k, v in bundle.splits.items()})
    config["config_hash"] = canonical_hash(config)
    return bundle, config


def verify_run(run):
    config = load_json(run / "config.json")
    unsigned = {k: v for k, v in config.items() if k != "config_hash"}
    if canonical_hash(unsigned) != config["config_hash"]:
        raise ValueError("Corrupt config hash")
    if config["sources"] != source_hashes() or config["versions"] != versions():
        raise ValueError("Source or dependency drift; use a new run")
    bundle = Bundle(**{k: str(PROJECT_ROOT / v) for k, v in config["manifests"].items()}, fold=config["fold"])
    if bundle.manifest_hashes != config["manifest_hashes"]:
        raise ValueError("Manifest changed since training")
    if bundle.asset_hashes(("train", "val")) != config["asset_hashes"]:
        raise ValueError("Training/validation assets changed since training")
    return bundle, config


def load_checkpoint(path, config):
    # Portable final weights are tensor-only; resume state remains local/trusted.
    state = torch.load(path, map_location="cpu", weights_only=Path(path).name == "final.pt")
    if state["config_hash"] != config["config_hash"]:
        raise ValueError("Checkpoint/config mismatch")
    return state


def seed_runtime(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.random.default_generator.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def one_epoch(model, loader, config, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            loss = total_loss(model(x), y, config["loss"], config["pos_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; checkpoint not advanced")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def train(args):
    device = select_device(args.device, args.permit_gpu)
    run = output_path(args.output)
    bundle, config = prepare(args)
    with run_lock(run):
        existing = [p for p in run.iterdir() if p.name != ".lock"]
        if args.resume:
            if not (run / "config.json").exists() or load_json(run / "config.json") != config:
                raise ValueError("Resume requires an identical existing config and input hashes")
        elif existing:
            raise ValueError("Refusing nonempty output directory; use --resume for an identical run")
        else:
            save_json(run / "config.json", config)
        seed_runtime(args.seed, device)
        model = SmallUNet(args.mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        start, history = 0, []
        if args.resume and (run / "last.pt").exists():
            state = load_checkpoint(run / "last.pt", config)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"], state["history"]
            if not 0 <= start <= 20 or len(history) != start:
                raise ValueError("Invalid checkpoint epoch/history")
            restore_rng(state["rng"], device)
        for epoch in range(start, 20):
            train_ds = RefinerDataset(bundle.records("train"), args.mode, config["normalization"], args.seed, epoch,
                                      shuffle_grid_size=config["shuffle_grid"])
            val_ds = RefinerDataset(bundle.records("val"), args.mode, config["normalization"], args.seed,
                                    shuffle_grid_size=config["shuffle_grid"])
            generator = torch.Generator().manual_seed(args.seed + epoch)
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=config["num_workers"],
                                      generator=generator, pin_memory=device.type == "cuda")
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=config["num_workers"],
                                    pin_memory=device.type == "cuda")
            train_loss = one_epoch(model, train_loader, config, device, optimizer)
            val_loss = one_epoch(model, val_loader, config, device)
            history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss))
            state = dict(config_hash=config["config_hash"], epoch=epoch + 1,
                         state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=rng_state(device), history=history)
            atomic(run / "last.pt", lambda f: torch.save(state, f))
            save_jsonl(run / "history.jsonl", history)
            print(json.dumps(history[-1]), flush=True)
        # Detect changed inputs before sealing the final artifact.
        verify_run(run)
        save_jsonl(run / "history.jsonl", history)
        final = run / "final.pt"
        if not final.exists():
            payload = dict(config_hash=config["config_hash"], epoch=20, state_dict=model.state_dict())
            atomic(final, lambda f: torch.save(payload, f))
        else:
            state = load_checkpoint(final, config)
            if state["epoch"] != 20 or any(not torch.equal(v.cpu(), state["state_dict"][k])
                                          for k, v in model.state_dict().items()):
                raise ValueError("Existing final weight does not match completed training")
        save_json(run / "complete.json", dict(epoch=20, final_sha256=digest(final), config_hash=config["config_hash"]))


def final_hash(run, config):
    complete = load_json(run / "complete.json")
    sha = digest(run / "final.pt")
    if complete != dict(epoch=20, final_sha256=sha, config_hash=config["config_hash"]):
        raise ValueError("Final checkpoint not sealed or hash mismatch")
    return sha


def predict(args):
    run = output_path(args.run)
    device = select_device(args.device, args.permit_gpu)
    with run_lock(run):
        bundle, config = verify_run(run)
        sha = final_hash(run, config)
        target = output_path(run / f"predictions_{args.split}")
        if target.exists():
            raise ValueError("Prediction directory already exists; preserve it and choose a new run")
        # Only explicit test prediction hashes/opens test assets; labels remain absent.
        assets = bundle.asset_hashes(("train", "val", "test") if args.split == "test" else ("val",), targets=False)
        seed_runtime(config["seed"], device)
        model = SmallUNet(config["mode"]).to(device)
        checkpoint = load_checkpoint(run / "final.pt", config)
        if checkpoint["epoch"] != 20:
            raise ValueError("Prediction requires final epoch 20")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        records = bundle.records(args.split)
        ds = RefinerDataset(records, config["mode"], config["normalization"], config["seed"], targets=False,
                            shuffle_grid_size=config["shuffle_grid"])
        rows = []
        with torch.inference_mode():
            for x, _, indices in DataLoader(ds, batch_size=16, shuffle=False, num_workers=config["num_workers"],
                                            pin_memory=device.type == "cuda"):
                probs = model(x.to(device)).sigmoid().cpu().numpy()[:, 0]
                for index, prob in zip(indices.tolist(), probs):
                    r = records[index]
                    if not np.isfinite(prob).all():
                        raise ValueError("Nonfinite predicted probabilities")
                    with Image.open(r["image_path"]) as im:
                        width, height = im.size
                    path = target / "masks" / (canonical_hash(r["image_id"]) + ".npy")
                    save_npy(path, prob.astype(np.float32))
                    rows.append(dict(image_id=r["image_id"], split=args.split, width=width, height=height,
                                     refined_mask_path=portable(path), mask_sha256=digest(path),
                                     final_sha256=sha, config_hash=config["config_hash"],
                                     **{k: r[k] for k in ("base_part", "source_id", "fold", "original_fold", "group_id") if k in r}))
        after = bundle.asset_hashes(("train", "val", "test") if args.split == "test" else ("val",), targets=False)
        if assets != after:
            raise ValueError("Inputs changed during prediction")
        save_jsonl(target / "predictions.jsonl", rows)
        save_json(target / "provenance.json", dict(asset_hashes=assets, device=str(device),
                  predictions_sha256=digest(target / "predictions.jsonl"), final_sha256=sha,
                  config_hash=config["config_hash"]))


def prediction_rows(run, bundle, config, split):
    folder = run / f"predictions_{split}"
    path = folder / "predictions.jsonl"
    provenance = load_json(folder / "provenance.json")
    sha = final_hash(run, config)
    if (provenance["predictions_sha256"] != digest(path) or provenance["final_sha256"] != sha
            or provenance["config_hash"] != config["config_hash"]):
        raise ValueError("Prediction provenance mismatch")
    rows = read_jsonl(path)
    if set(indexed(rows)) != set(bundle.splits[split]):
        raise ValueError("Prediction IDs do not exactly match requested split")
    for row in rows:
        if row["split"] != split or row["final_sha256"] != sha or row["config_hash"] != config["config_hash"]:
            raise ValueError("Cross-split or checkpoint prediction mismatch")
        output_path(row["refined_mask_path"])
        if digest(output_path(row["refined_mask_path"])) != row["mask_sha256"]:
            raise ValueError("Prediction mask changed")
    return rows, digest(path)


def detections(rows, parameters):
    out = []
    for row in rows:
        prob = np.load(output_path(row["refined_mask_path"]), allow_pickle=False)
        boxes, scores, _ = mask_to_boxes(prob, row["width"], row["height"], **parameters)
        out.append(dict(image_id=row["image_id"], pred_boxes=boxes, pred_scores=scores))
    return out


def tune(args):
    run = output_path(args.run)
    with run_lock(run):
        bundle, config = verify_run(run)
        if (run / "tuned.json").exists():
            raise ValueError("Frozen tuning already exists; refusing replacement")
        rows, pred_sha = prediction_rows(run, bundle, config, "val")
        source = read_jsonl(args.val_ground_truth) if args.val_ground_truth else bundle.records("val")
        truth_index = indexed(source)
        if set(truth_index) != set(bundle.splits["val"]):
            raise ValueError("Ground truth must contain exactly validation IDs")
        truth = []
        for row in rows:
            gt = truth_index[row["image_id"]]
            if "gt_boxes" not in gt:
                raise ValueError("Box ground truth required from evaluation agent for mask-only records")
            truth.append(dict(image_id=row["image_id"], gt_boxes=checked_boxes(gt["gt_boxes"], row["width"], row["height"])))
        scorer_path = Path(args.evaluator or args.scorer).resolve()
        scorer_sha = digest(scorer_path)
        module = EvaluationBridge(scorer_path, bundle, rows) if args.evaluator else load_module(scorer_path)
        candidates, best = [], None
        for params in grid():
            predictions = detections(rows, params)
            metrics = module.score(copy.deepcopy(predictions), copy.deepcopy(truth))
            key = selection_key(metrics)
            row = dict(parameters=params, metrics={k: float(metrics[k]) for k in ("AP50", "F1", "FP")})
            if args.evaluator:
                row["confidence_threshold"] = metrics["confidence_threshold"]
            candidates.append(row)
            if best is None or key > best[0]:
                best = (key, row, predictions)
        if digest(scorer_path) != scorer_sha or prediction_rows(run, bundle, config, "val")[1] != pred_sha:
            raise ValueError("Evaluation inputs changed during tuning")
        save_jsonl(run / "tuning_grid.jsonl", candidates)
        save_jsonl(run / "val_detections.jsonl", best[2])
        frozen = dict(**best[1], fixed=FIXED, selection=["AP50", "F1", "-FP"],
                      split="val", validation_ids=bundle.splits["val"],
                      ground_truth_hash=canonical_hash(truth), scorer_path=scorer_path.name, scorer_sha256=scorer_sha,
                      scorer_kind="native_evaluator" if args.evaluator else "generic_adapter",
                      split_audit=module.split_audit if args.evaluator else {"source_overlap": bundle.source_overlap()},
                      validation_predictions_sha256=pred_sha, config_hash=config["config_hash"],
                      final_sha256=final_hash(run, config))
        frozen["tuning_hash"] = canonical_hash(frozen)
        save_json(run / "tuned.json", frozen)


def apply(args):
    run = output_path(args.run)
    with run_lock(run):
        bundle, config = verify_run(run)
        frozen = load_json(run / "tuned.json")
        if canonical_hash({k: v for k, v in frozen.items() if k != "tuning_hash"}) != frozen["tuning_hash"]:
            raise ValueError("Tuning hash mismatch")
        if (frozen["split"] != "val" or frozen["validation_ids"] != bundle.splits["val"]
                or frozen["config_hash"] != config["config_hash"] or frozen["fixed"] != FIXED
                or frozen["final_sha256"] != final_hash(run, config)
                or frozen["parameters"] not in list(grid())):
            raise ValueError("Frozen validation settings mismatch")
        rows, sha = prediction_rows(run, bundle, config, args.split)
        destination = output_path(run / f"postprocessed_{args.split}")
        if destination.exists():
            raise ValueError("Postprocessed output already exists")
        out = []
        for row in rows:
            prob = np.load(output_path(row["refined_mask_path"]), allow_pickle=False)
            boxes, scores, binary = mask_to_boxes(prob, row["width"], row["height"], **frozen["parameters"])
            path = destination / "binary_masks" / (canonical_hash(row["image_id"]) + ".npy")
            save_npy(path, binary)
            out.append(dict(**row, pred_boxes=boxes, pred_scores=scores, binary_mask_path=portable(path),
                            binary_mask_sha256=digest(path), tuning_hash=frozen["tuning_hash"],
                            **({"confidence_threshold": frozen["confidence_threshold"]} if "confidence_threshold" in frozen else {})))
        save_jsonl(destination / "detections.jsonl", out)
        save_json(destination / "provenance.json", dict(input_predictions_sha256=sha,
                  tuning_hash=frozen["tuning_hash"], detections_sha256=digest(destination / "detections.jsonl")))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate"):
        p = subs.add_parser(name)
        for field in ("records", "splits", "heatmaps", "output"):
            p.add_argument("--" + field, required=True)
        p.add_argument("--mode", choices=MODES, default="rgb_heat")
        p.add_argument("--loss", choices=("bce_dice", "focal_dice", "weighted_bce_dice"), default="bce_dice")
        p.add_argument("--normalization", choices=("per_image", "train_quantile"), default="per_image")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--fold", type=int, help="Held-out run fold; must agree with split fold/outer_fold if present")
        p.add_argument("--num-workers", type=int, default=4,
                       help="Saved worker count used by train, validation and prediction (default: 4)")
        p.add_argument("--shuffle-grid", type=int, choices=(2, 4, 8, 16, 32, 64, 128, 256), default=16,
                       help="Grid rows/columns for block shuffle on the 256x256 heat input")
        p.add_argument("--device", default="cpu")
        p.add_argument("--permit-gpu", action="store_true")
        p.add_argument("--resume", action="store_true")
    for name in ("predict", "tune", "apply"):
        p = subs.add_parser(name)
        p.add_argument("--run", required=True)
        if name != "tune":
            p.add_argument("--split", choices=("val", "test"), required=True)
        if name == "predict":
            p.add_argument("--device", default="cpu")
            p.add_argument("--permit-gpu", action="store_true")
        if name == "tune":
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--scorer", help="Trusted generic score(predictions, ground_truth) adapter")
            group.add_argument("--evaluator", help="Parent evaluation/evaluator.py (validation-only BestF1)")
            p.add_argument("--val-ground-truth")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command == "validate":
        output_path(args.output)
        bundle, config = prepare(args)
        print(json.dumps(dict(config_hash=config["config_hash"], splits=config["split_counts"],
                              normalization=config["normalization"], pos_weight=config["pos_weight"]), indent=2))
    else:
        {"train": train, "predict": predict, "tune": tune, "apply": apply}[args.command](args)


if __name__ == "__main__":
    main()
