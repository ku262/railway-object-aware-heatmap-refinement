import copy
import hashlib
import json

import pytest

from evaluate import check_membership, main
from split import make_splits, verify_artifact


def records():
    return [dict(image_id=f"i{f}_{label}_{i}", group_id=f"g{f}_{label}_{i}",
                 source_id="shared_source", base_part="part", fold=f, label=label,
                 image_sha256=hashlib.sha256(f"{f}_{label}_{i}".encode()).hexdigest())
            for f in range(5) for label in (0, 1) for i in range(3)]


def predictions(artifact, role):
    lookup = {r["image_id"]: r for r in artifact["records"]}
    rows = []
    for f, s in artifact["splits"].items():
        for i in s[role]:
            r = lookup[i]
            rows.append(dict(image_id=i, group_id=r["group_id"], source_id=r["source_id"],
                             base_part=r["base_part"], fold=int(f), width=20, height=20,
                             gt_boxes=[[0, 0, 10, 10]] if r["label"] else [],
                             pred_boxes=[[0, 0, 10, 10]], pred_scores=[.8 if r["label"] else .1]))
    return rows


def test_deterministic_partition():
    a = make_splits(records())
    assert make_splits(list(reversed(records()))) == a
    assert verify_artifact(a) == a
    for f, s in a["splits"].items():
        assert set(s["train"]) | set(s["val"]) | set(s["test"]) == {r["image_id"] for r in records()}
        assert not set(s["train"]) & set(s["val"])
        assert s["source_overlap"]["train_test"] == 1
        assert set(s["test"]) == {r["image_id"] for r in records() if r["fold"] == int(f)}


def test_bad_metadata_and_tampering():
    rows = records()
    rows[1]["image_sha256"] = rows[0]["image_sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        make_splits(rows)
    rows = records()
    rows[1]["group_id"] = rows[-1]["group_id"]
    with pytest.raises(ValueError, match="Group spans"):
        make_splits(rows)
    a = make_splits(records())
    a["splits"]["0"]["val"].pop()
    with pytest.raises(ValueError, match="artifact"):
        verify_artifact(a)


def test_membership_rejects_missing_or_wrong_groups():
    a = make_splits(records())
    rows = predictions(a, "val")
    check_membership(rows, a, "val")
    with pytest.raises(ValueError, match="cover"):
        check_membership(rows[:-1], a, "val")
    bad = copy.deepcopy(rows)
    bad[0]["group_id"] = "wrong"
    with pytest.raises(ValueError, match="metadata"):
        check_membership(bad, a, "val")


def test_cli_freezes_then_evaluates_and_refuses_overwrite(tmp_path):
    a = make_splits(records())
    (tmp_path / "splits.json").write_text(json.dumps(a), encoding="utf-8")
    for role in ("val", "test"):
        (tmp_path / (role + ".jsonl")).write_text(
            "\n".join(json.dumps(r) for r in predictions(a, role)), encoding="utf-8")
    args = []
    for k, file in (("splits", "splits.json"), ("val", "val.jsonl"), ("test", "test.jsonl"),
                    ("thresholds", "frozen.json"), ("out", "metrics.json")):
        args.extend(["--" + k, str(tmp_path / file)])
    main(args)
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["metrics"]["f1"] == 1
    assert not report["upstream_training_and_bank_provenance_verified"]
    with pytest.raises(FileExistsError):
        main(args)
