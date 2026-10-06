"""Adapter to the parent's strict evaluator, without exposing test labels."""
import importlib.util
from pathlib import Path
import sys


def load_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("refiner_external_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError("Expected a trusted Python evaluator/adapter file")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


class EvaluationBridge:
    def __init__(self, path, bundle, validation_rows):
        self.api = load_module(path)
        for split in ("train", "val", "test"):
            for r in bundle.records(split):
                for key in ("base_part", "source_id"):
                    if not isinstance(r.get(key), str) or not r[key]:
                        raise ValueError(f"Native evaluator requires {key} metadata for every split")
                if type(r.get("fold")) is not int:
                    raise ValueError("Native evaluator requires integer fold metadata")
        folds = {r["fold"] for r in bundle.rows.values()}
        if len(folds) != 1:
            raise ValueError("Refiner runs and native tuning must contain exactly one fold")
        # This API checks only IDs/folds/source IDs; test targets remain stripped.
        self.split_audit = self.api.check_splits(bundle.records("val"), bundle.records("test"), bundle.records("train"))
        self.metadata = {r["image_id"]: {key: bundle.rows[r["image_id"]][key]
                                      for key in ("base_part", "source_id", "fold")}
                         | {"width": r["width"], "height": r["height"]}
                         for r in validation_rows}

    def score(self, predictions, truth):
        truth = {r["image_id"]: r["gt_boxes"] for r in truth}
        records = [dict(**p, **self.metadata[p["image_id"]], gt_boxes=truth[p["image_id"]]) for p in predictions]
        threshold, _ = self.api.select_threshold(records)
        result = self.api.evaluate(records, fixedthreshold=threshold)
        return dict(AP50=result["ap50"], F1=result["f1"], FP=result["fp_perimage"],
                    confidence_threshold=threshold)
