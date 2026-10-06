"""Validation-frozen image gates and inference-only raw-grid permutation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def image_gates(validation):
    from sklearn.metrics import precision_recall_curve
    groups, seen = {}, set()
    for row in validation:
        key = (int(row['fold']), row['base_part'])
        identity = (key[0], row['image_id'])
        if identity in seen:
            raise ValueError('Duplicate validation prediction')
        seen.add(identity)
        score = float(row['image_score'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite image score')
        groups.setdefault(key, []).append((int(bool(row['gt_boxes'])), score))
    if not groups:
        raise ValueError('Empty validation cohort')
    result = []
    for (fold, part), samples in sorted(groups.items()):
        labels, scores = zip(*samples)
        if set(labels) != {0, 1}:
            raise ValueError('Each validation fold/component needs both labels')
        p, r, thresholds = precision_recall_curve(labels, scores)
        f1 = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
        result.append(dict(fold=fold, base_part=part,
                           threshold=float(thresholds[np.argmax(f1)])))
    return {'selection': 'internal_validation_f1', 'gates': result}


def apply_gates(records, frozen):
    lookup = {(g['fold'],g['base_part']):g['threshold'] for g in frozen['gates']}
    result = copy.deepcopy(records)
    for row in result:
        key = (int(row['fold']), row['base_part'])
        if key not in lookup:
            raise ValueError('Missing validation gate for fold/component')
        score = float(row['image_score'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite image score')
        if len(row['pred_boxes']) != len(row['pred_scores']):
            raise ValueError('Box/score count mismatch')
        if score < lookup[key]:
            row['pred_boxes'], row['pred_scores'] = [], []
    return result


def shuffle_grid(grid, seed):
    values = np.asarray(grid)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected a finite, nonempty two-dimensional raw grid')
    return np.random.default_rng(seed).permutation(values.ravel()).reshape(values.shape)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    gate = modes.add_parser('gate')
    for key in ('val','test','gates','out'):
        gate.add_argument('--'+key, type=Path, required=True)
    shuffle = modes.add_parser('shuffle')
    for key in ('input','output'):
        shuffle.add_argument('--'+key, type=Path, required=True)
    shuffle.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'shuffle':
        if args.output.exists():
            raise FileExistsError(args.output)
        values = shuffle_grid(np.load(args.input, allow_pickle=False),args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
        return
    if args.gates.resolve() == args.out.resolve() or args.gates.exists() or args.out.exists():
        raise ValueError('Use two distinct new output paths')
    frozen = image_gates(read_rows(args.val))
    args.gates.parent.mkdir(parents=True, exist_ok=True)
    with args.gates.open('x', encoding='utf-8') as stream:
        json.dump(frozen,stream,indent=2,allow_nan=False)
    gated = apply_gates(read_rows(args.test),frozen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        for row in gated:
            stream.write(json.dumps(row,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
