import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
if importlib.util.find_spec("pycocotools") is None:
    sys.path.insert(0, str(Path(__file__).parent / "_verification_vendor"))

from evaluator import (evaluate, select_threshold, freeze_thresholds, thaw_thresholds,
                       paired_bootstrap, fit_size_thresholds, summarize, validate,
                       check_splits, fold_statistics, main, METRICS)

B = [0, 0, 10, 10]


def rec(image="a", gt=None, pred=None, scores=None, fold=0, source=None):
    return dict(image_id=image, base_part="part", source_id=source or image, fold=fold,
                width=100, height=100, gt_boxes=[B] if gt is None else gt,
                pred_boxes=[] if pred is None else pred,
                pred_scores=[] if scores is None else scores)


def test_empty_predictions():
    records = [rec()]
    row = evaluate(records, .5)
    assert row["ap50"] == row["ap50_95"] == row["f1"] == 0
    assert row["fn"] == 1 and row["normalfpr"] is None
    assert select_threshold(records) == (None, 0)


def test_normal_images_and_undefined_metrics():
    row = evaluate([rec(gt=[], pred=[B], scores=[.7]), rec("b", gt=[])], .5)
    assert row["normalfpr"] == row["fp_perimage"] == .5
    assert row["ap50"] is None and row["recall"] is None and row["f1"] is None
    with pytest.raises(ValueError, match="no GT"):
        select_threshold([rec(gt=[])])


def test_score_sorting_multigt_and_duplicate_detections():
    other = [20, 20, 30, 30]
    row = evaluate([rec(gt=[B, other], pred=[B, other, B], scores=[.1, .8, .9])], .5)
    assert (row["tp"], row["fp"], row["fn"]) == (2, 0, 0)
    row = evaluate([rec(gt=[B, other], pred=[B, other, B], scores=[.9, .9, .9])], .5)
    assert (row["tp"], row["fp"], row["fn"]) == (2, 1, 0)


def test_validation_ties_and_test_independence():
    # Adding one TP and two FP preserves F1=2/3; highest threshold wins.
    val = [rec(gt=[B, B], pred=[B, B, B, B], scores=[.9, .5, .5, .5])]
    assert select_threshold(val) == (.9, 2 / 3)
    artifact = freeze_thresholds(val)
    frozen = copy.deepcopy(artifact)
    test = [rec("test", pred=[B], scores=[.8])]
    before = copy.deepcopy(test)
    assert evaluate(test, fixedthreshold=thaw_thresholds(artifact))["tp"] == 0
    assert artifact == frozen and test == before
    assert evaluate(test, fixedthreshold=.9)["ap50"] == pytest.approx(1)
    artifact["folds"]["0"]["threshold"] = .1
    with pytest.raises(ValueError, match="hash"):
        thaw_thresholds(artifact)


def test_all_score_ties_are_selected_together():
    val = [rec(pred=[B, B], scores=[.8, .8])]
    assert select_threshold(val) == (.8, 2 / 3)


def test_max100():
    row = evaluate([rec(pred=[B] * 101, scores=[.9] * 101)], .5)
    assert row["tp"] == 1 and row["fp"] == 99


def test_101_point_interpolation_not_continuous_area():
    row = evaluate([rec(gt=[B, B, B], pred=[B], scores=[.8])], .5)
    assert row["ap50"] == pytest.approx(34 / 101)


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan"), {}, {1: .5}])
def test_invalid_fixed_threshold(threshold):
    with pytest.raises(ValueError):
        evaluate([rec()], threshold)


def test_two_fold_threshold_freeze():
    val = [rec("v0", pred=[B], scores=[.8]), rec("v1", pred=[B], scores=[.3], fold=1)]
    thresholds = thaw_thresholds(freeze_thresholds(val))
    assert thresholds == {0: .8, 1: .3}
    test = [rec("t0", pred=[B], scores=[.5]), rec("t1", pred=[B], scores=[.5], fold=1)]
    assert evaluate(test, thresholds)["tp"] == 1


@pytest.mark.parametrize("change", [
    {"pred_boxes": [B], "pred_scores": []}, {"pred_scores": [float("nan")]},
    {"gt_boxes": [[0, 0, 101, 1]]}, {"gt_boxes": [[1, 0, 1, 2]]},
    {"width": 0}, {"fold": "0"}, {"source_id": ""},
])
def test_schema_rejection(change):
    record = rec()
    record.update(change)
    with pytest.raises(ValueError):
        validate([record])


def test_duplicate_and_split_leakage():
    with pytest.raises(ValueError, match="Duplicate"):
        validate([rec(), rec()])
    val, test = rec("val", source="s"), rec("test", source="s")
    val["group_id"], test["group_id"] = "g1", "g2"
    report = check_splits([val], [test])
    assert report["source_overlap_counts"]["val_test"]["0"] == 1
    assert report["source_heldout_claim"] is False
    test["group_id"] = "g1"
    with pytest.raises(ValueError, match="group_id"):
        check_splits([val], [test])
    with pytest.raises(ValueError, match="image_id"):
        check_splits([rec()], [rec()])


def test_fold_sample_sd():
    rows = {"0": {k: 0.0 for k in METRICS}, "1": {k: 1.0 for k in METRICS}}
    assert fold_statistics(rows)["f1"]["sample_sd"] == pytest.approx(np.sqrt(.5))
    assert fold_statistics({"0": rows["0"]})["f1"]["sample_sd"] is None


def test_paired_clusters_deterministic_and_mismatch_rejection():
    records = [rec("a", pred=[B], scores=[.9], source="shared"),
               rec("b", source="shared"), rec("c", gt=[])]
    methods = {"one": records, "two": copy.deepcopy(records)}
    thresholds = {"one": {0: .5}, "two": {0: .5}}
    result = paired_bootstrap(methods, thresholds, n_boot=25, seed=12)
    assert result == paired_bootstrap(methods, thresholds, n_boot=25, seed=12)
    assert result["n_sources"] == 2
    assert result["paired_differences"]["two minus one"]["f1"]["ci95"] == [0, 0]
    methods["two"][0]["gt_boxes"] = []
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(methods, thresholds, n_boot=2)


def test_size_training_only_and_ignore_matching():
    train = [rec("train", gt=[[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]])]
    artifact = fit_size_thresholds(train)
    test = [rec("test", gt=[B, [0, 0, 30, 30]], pred=[B, [0, 0, 30, 30]], scores=[.9, .8])]
    result = summarize(test, {0: .5}, artifact)
    assert result["sizes"]["small"]["tp"] == 1
    assert result["sizes"]["small"]["fp"] == 0
    assert result["sizes"]["large"]["tp"] == 1
    assert result["sizes"]["medium"]["ap50"] is None
    assert result["sizes"]["small"]["normalfpr"] is None


def test_default_2000_bootstrap_and_crossfold_source_cluster():
    records = [rec("a", pred=[B], scores=[.9], source="shared"),
               rec("b", pred=[B], scores=[.9], fold=1, source="shared")]
    result = paired_bootstrap({"one": records, "two": records},
                              {"one": {0: .5, 1: .5}, "two": {0: .5, 1: .5}})
    assert result["n_boot"] == 2000 and result["n_sources"] == 1
    assert result["paired_differences"]["two minus one"]["ap50"]["ci95"] == [0, 0]
    assert result["methods"]["one"]["ap50"]["n_defined"] == 2000


def canonical_coco(records):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    images, annotations, detections = [], [], []
    for image_id, r in enumerate(validate(records), 1):
        images.append(dict(id=image_id, width=r["width"], height=r["height"]))
        for b in r["gt_boxes"]:
            x, y, x2, y2 = b
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                                    category_id=1, bbox=[x, y, x2-x, y2-y],
                                    area=(x2-x)*(y2-y), iscrowd=0))
        for b, score in zip(r["pred_boxes"], r["pred_scores"]):
            x, y, x2, y2 = b
            detections.append(dict(image_id=image_id, category_id=1,
                                   bbox=[x, y, x2-x, y2-y], score=score))
    gt.dataset = dict(images=images, annotations=annotations, categories=[dict(id=1, name="object")], info={})
    gt.createIndex()
    if detections:
        dt = gt.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = dict(images=images, annotations=[], categories=gt.dataset["categories"])
        dt.createIndex()
    ev = COCOeval(gt, dt, "bbox")
    ev.params.useCats = 0
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"][:, :, 0, 0, 2]
    return [None if not (p >= 0).any() else float(p[p >= 0].mean()) for p in precision]


@pytest.mark.parametrize("seed", range(8))
def test_randomized_canonical_pycocotools(seed):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(8):
        gt, pred = [], []
        for _ in range(int(rng.integers(0, 5))):
            x, y = rng.integers(0, 70, 2).tolist()
            gt.append([x, y, x + 20, y + 20])
        for _ in range(int(rng.integers(0, 130))):
            if gt and rng.random() < .6:
                b = gt[int(rng.integers(len(gt)))].copy()
                b[2] += int(rng.integers(-5, 6))
            else:
                x, y = rng.integers(0, 70, 2).tolist()
                b = [x, y, x + 20, y + 20]
            pred.append(b)
        scores = rng.choice([0, .3, .5, .9, 1], len(pred)).tolist()
        records.append(rec(str(i), gt=gt, pred=pred, scores=scores))
    actual = list(evaluate(records, .5)["ap_by_iou"].values())
    assert actual == pytest.approx(canonical_coco(records), abs=1e-12)


@pytest.mark.parametrize("records", [
    [rec()], [rec(gt=[])], [rec(gt=[], pred=[B], scores=[.7])],
    [rec(gt=[B, B], pred=[B, B, B], scores=[.7, .7, .7])],
    [rec(pred=[B] * 101, scores=[.9] * 101)],
])
def test_canonical_edge_cases(records):
    assert list(evaluate(records, .5)["ap_by_iou"].values()) == pytest.approx(canonical_coco(records))


def test_cli(tmp_path):
    val, test = tmp_path / "val.jsonl", tmp_path / "test.jsonl"
    val.write_text(json.dumps(rec("val", pred=[B], scores=[.8])) + "\n")
    test.write_text(json.dumps(rec("test", pred=[B], scores=[.7])) + "\n")
    out = tmp_path / "output"
    args = ["--val-pred", str(val), "--test-pred", str(test), "--out-dir", str(out), "--bootstrap", "8"]
    main(args)
    artifact = json.loads((out / "method_1.thresholds.json").read_text())
    assert thaw_thresholds(artifact) == {0: .8}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["methods"]["method_1"]["pooled"]["tp"] == 0
    with pytest.raises(FileExistsError):
        main(args)
