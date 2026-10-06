"""Portable metadata-only strict inner splits; never regenerate outer assignments."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_splits(records):
    fields = ("image_id", "group_id", "source_id", "base_part", "label", "fold", "image_sha256")
    rows = []
    for row in records:
        if any(k not in row for k in fields):
            raise ValueError("Missing metadata field")
        for key in fields[:4]:
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("Identifiers must be nonempty strings")
        if type(row["fold"]) is not int or row["fold"] not in range(5):
            raise ValueError("Original fold must be an integer 0..4")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Label must be binary integer")
        if not isinstance(row["image_sha256"], str) or not re.fullmatch("[0-9a-f]{64}", row["image_sha256"]):
            raise ValueError("image_sha256 must be lowercase hexadecimal SHA256")
        rows.append({k: row[k] for k in fields})
    rows.sort(key=lambda r: r["image_id"])
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Empty records or duplicate image IDs")
    if {r["fold"] for r in rows} != set(range(5)):
        raise ValueError("All five original folds required")
    groups, hashes = defaultdict(list), defaultdict(set)
    for row in rows:
        groups[row["group_id"]].append(row)
        hashes[row["image_sha256"]].add(row["group_id"])
    if any(len(g) > 1 for g in hashes.values()):
        raise ValueError("Recorded byte duplicate crosses annotation groups")
    for items in groups.values():
        if len({(r["fold"], r["base_part"], r["label"]) for r in items}) != 1:
            raise ValueError("Group spans folds or strata")
    all_strata = {(r["base_part"], r["label"]) for r in rows}
    splits = {}
    for fold in range(5):
        strata = defaultdict(list)
        for group, items in sorted(groups.items()):
            r = items[0]
            if r["fold"] != fold:
                strata[(r["base_part"], r["label"])].append(group)
        if set(strata) != all_strata:
            raise ValueError("Missing development stratum")
        rng, validation = random.Random(42 + fold), set()
        for _, candidates in sorted(strata.items()):
            if len(candidates) < 5:
                raise ValueError("At least five development groups per stratum required")
            count = math.ceil(.2 * len(candidates))
            rng.shuffle(candidates)
            validation.update(candidates[:count])
        roles = {role: [] for role in ("train", "val", "test")}
        for r in rows:
            role = "test" if r["fold"] == fold else "val" if r["group_id"] in validation else "train"
            roles[role].append(r["image_id"])
        membership = {role: set(ids) for role, ids in roles.items()}
        sources = {role: {r["source_id"] for r in rows if r["image_id"] in ids}
                   for role, ids in membership.items()}
        splits[str(fold)] = {**roles, "source_overlap": {
            a + "_" + b: len(sources[a] & sources[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}}
    payload = {"protocol": "strict_annotation_group_nested_v1", "seed": 42,
               "inner_fraction": .2, "records": rows, "splits": splits,
               "raw_images_rehashed": False, "source_heldout_claim": False}
    return {**payload, "sha256": digest(payload)}


def verify_artifact(artifact):
    expected = make_splits(artifact["records"])
    if expected != artifact:
        raise ValueError("Split artifact does not match deterministic protocol")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with args.records.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    write_new(args.out, make_splits(rows))


if __name__ == "__main__":
    main()
