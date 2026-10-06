"""Freeze validation thresholds before opening test predictions; audit membership."""
import argparse
import json
from pathlib import Path
import numpy as np

from evaluation.evaluator import (check_splits, digest, evaluate, freeze_thresholds,
                                  read_jsonl, thaw_thresholds)
from split import verify_artifact, write_new


def check_membership(rows, artifact, role):
    index = {r["image_id"]: r for r in artifact["records"]}
    expected = {(int(f), i) for f, s in artifact["splits"].items() for i in s[role]}
    if {(r["fold"], r["image_id"]) for r in rows} != expected:
        raise ValueError("Predictions do not exactly cover split role: " + role)
    for r in rows:
        original = index[r["image_id"]]
        for field in ("group_id", "source_id", "base_part"):
            if r.get(field) != original[field]:
                raise ValueError("Prediction metadata mismatch: " + field)
        if bool(r["gt_boxes"]) != bool(original["label"]):
            raise ValueError("Ground-truth presence differs from metadata label")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("splits", "val", "test", "thresholds", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.thresholds.resolve() == args.out.resolve():
        raise ValueError("Threshold and metric outputs must differ")
    if args.thresholds.exists() or args.out.exists():
        raise FileExistsError("Use new output paths; frozen artifacts are not overwritten")
    artifact = verify_artifact(json.loads(args.splits.read_text(encoding="utf-8")))
    val = read_jsonl(args.val)
    check_membership(val, artifact, "val")
    thresholds = freeze_thresholds(val)
    write_new(args.thresholds, thresholds)
    # Test data becomes accessible only after the validation choice is persisted.
    test = read_jsonl(args.test)
    check_membership(test, artifact, "test")
    audit = check_splits(val, test)
    report = {"metrics": evaluate(test, thaw_thresholds(thresholds)),
              "split_audit": audit, "split_sha256": artifact["sha256"],
              "test_records_sha256": digest(test),
              "threshold_artifact_sha256": thresholds["sha256"],
              "upstream_training_and_bank_provenance_verified": False}
    frozen = thaw_thresholds(thresholds)
    per_fold = {str(f): evaluate([r for r in test if r['fold'] == f], frozen) for f in range(5)}
    metric_names = ('ap50', 'ap50_95', 'precision', 'recall', 'f1', 'fp_perimage', 'normalfpr')
    report['fold_metrics'] = per_fold
    report['five_fold_mean_sd'] = {
        key: {'mean': float(np.mean([m[key] for m in per_fold.values()])),
              'sd': float(np.std([m[key] for m in per_fold.values()], ddof=1))}
        for key in metric_names}
    report['metrics_scope'] = 'pooled predictions; use five_fold_mean_sd for fold summaries'
    write_new(args.out, report)


if __name__ == "__main__":
    main()
