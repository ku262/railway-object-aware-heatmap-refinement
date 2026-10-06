"""Primary two-stage orchestration; test assets are opened only by infer."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .common import (asset, check_images, cohort, dependencies, read_config, require,
                     rows, seal, sha, verify, write_json, write_rows)

PACKAGE = Path(__file__).resolve().parents[1]


def code_hashes():
    names = ['train.py', 'infer.py', 'split.py', 'evaluation/evaluator.py']
    names += ['runtime/' + n for n in ('common.py', 'features.py', 'pipeline.py')]
    names += ['runtime/refiner/' + n for n in ('data.py', 'model.py', 'postprocess.py', 'evaluation_bridge.py', 'runner.py')]
    return {name: sha(PACKAGE / name) for name in names}


def device(name, allow):
    require(name == 'cpu' or (allow and (name == 'cuda' or name.startswith('cuda:'))), 'CUDA requires --permit-gpu')
    import torch
    if name != 'cpu':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
    return name


def prepare_snapshot(root, run, records, splits, fold):
    from .features import token_path
    folder = run / 'inputs'
    folder.mkdir()
    prepared, heats = [], []
    development = set(splits['train'] + splits['val'])
    for r in records:
        item = {k: r[k] for k in ('image_id', 'base_part', 'group_id', 'source_id', 'fold', 'width', 'height', 'image_sha256')}
        item['image_path'] = os.path.relpath(asset(root, r['image_path']), folder).replace('\\', '/')
        if r['image_id'] in development:
            item.update(label=r['label'], gt_boxes=r['gt_boxes'])
        prepared.append(item)
        heats.append({'image_id': r['image_id'], 'heatmap_path': os.path.relpath(token_path(run / 'stage1/maps', r['image_id']), folder).replace('\\', '/')})
    write_rows(folder / 'records.jsonl', prepared)
    write_rows(folder / 'heatmaps.jsonl', heats)
    write_json(folder / 'fold.json', {'fold': fold, **{role + '_ids': ids for role, ids in splits.items()}})


def preflight(root, cfg, fold):
    records, splits, split_sha = cohort(root, cfg, fold)
    available = dependencies(cfg['encoder']['kind'])
    missing = [name for name, ok in available.items() if not ok]
    require(not missing, 'Install runtime dependencies first: ' + ', '.join(missing))
    encoder = cfg['encoder']
    if encoder['kind'] == 'dinov2':
        weights = asset(root, encoder['weights'])
        require(weights.is_file(), 'Pretrained DINO checkpoint is not bundled; obtain the licensed weight file and set its relative path/SHA256')
        require(sha(weights) == encoder['weights_sha256'], 'Pretrained checkpoint SHA256 mismatch')
    else:
        folder = asset(root, encoder['directory'])
        require((folder / 'signature.json').is_file(), 'Cached feature signature/download is missing')
        signature = json.loads((folder / 'signature.json').read_text())
        require(signature['records_sha256'] == sha(asset(root, cfg['records']))
                and signature['weights_sha256'] == encoder['weights_sha256']
                and signature['size'] == 518 and signature['dtype'] == 'float32'
                and signature['l2_normalized'] is True, 'Cached-token provenance mismatch')
        from .features import checked_tokens, token_path
        for r in records:
            if r['image_id'] in set(splits['train'] + splits['val']):
                checked_tokens(token_path(folder, r['image_id']))
    development = [r for r in records if r['image_id'] in set(splits['train'] + splits['val'])]
    check_images(root, development, targets=True)
    return records, splits, split_sha


def train(args):
    root = Path(args.root).resolve()
    cfg = read_config(root, args.config)
    if args.check_dependencies:
        print(json.dumps({'modules': dependencies(cfg['encoder']['kind']), 'automatic_downloads': False,
                          'weights_bundled': False, 'data_bundled': False}, indent=2))
        return
    require(args.fold in range(5), 'Choose one original fold, 0..4')
    run = asset(root, args.output)
    require(not run.exists(), 'Use a new run directory; immutable artifacts are never overwritten')
    records, splits, split_sha = preflight(root, cfg, args.fold)
    summary = {'experiment': cfg['experiment'], 'fold': args.fold, 'roles': {k: len(v) for k, v in splits.items()},
               'test_assets_opened': False, 'training_requested': args.execute}
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    from . import features
    from .refiner import runner
    import torch
    active_device = device(args.device, args.permit_gpu)
    runner.configure(root)
    runner.seed_runtime(cfg['seed'], torch.device(active_device))
    run.mkdir(parents=True)
    prepare_snapshot(root, run, records, splits, args.fold)
    by_id = {r['image_id']: r for r in records}
    train_rows = [by_id[i] for i in splits['train']]
    val_rows = [by_id[i] for i in splits['val']]
    cache, banks, heatmaps = (run / 'stage1' / name for name in ('tokens', 'banks', 'maps'))
    features.extract(root, cfg, train_rows + val_rows, cache, active_device)
    features.choose_and_build(cfg, train_rows, val_rows, cache, banks, active_device, cfg['seed'])
    features.maps(train_rows + val_rows, cache, banks, heatmaps, active_device)
    manifest = {'config': cfg, 'fold': args.fold, 'records_sha256': sha(asset(root, cfg['records'])),
                'split_artifact_sha256': split_sha, 'code_sha256': code_hashes(),
                'bank_sources': 'inner_train_normal_only', 'test_assets_opened': False,
                'bank_ratio_selection': cfg['bank']['selection'], 'refiner_checkpoint': 'final_epoch20',
                'threshold_source': 'inner_validation_only'}
    write_json(run / 'runtime.json', manifest)
    core_args = SimpleNamespace(records=str(run / 'inputs/records.jsonl'), splits=str(run / 'inputs/fold.json'),
                               heatmaps=str(run / 'inputs/heatmaps.jsonl'), output=str(run / 'refiner'), fold=args.fold,
                               mode='rgb_heat', loss='bce_dice', normalization='per_image', seed=cfg['seed'],
                               shuffle_grid=16, device=active_device, permit_gpu=args.permit_gpu,
                               num_workers=args.num_workers, resume=False)
    runner.train(core_args)
    predict = SimpleNamespace(run=str(run / 'refiner'), split='val', device=active_device, permit_gpu=args.permit_gpu)
    runner.predict(predict)
    runner.tune(SimpleNamespace(run=str(run / 'refiner'), evaluator=str(PACKAGE / 'evaluation/evaluator.py'),
                                scorer=None, val_ground_truth=None))
    runner.apply(predict)
    validation = rows(run / 'refiner/postprocessed_val/detections.jsonl')
    for r in validation:
        r['gt_boxes'] = by_id[r['image_id']]['gt_boxes']
    write_rows(run / 'validation_predictions.jsonl', validation)
    require(sha(asset(root, cfg['records'])) == manifest['records_sha256'], 'Input metadata changed during training')
    seal(run, 'TRAIN_COMPLETE.json')
    print(json.dumps({**summary, 'phase': 'complete', 'output': args.output}, indent=2))


def infer(args):
    root = Path(args.root).resolve()
    run = asset(root, args.run)
    verify(run, 'TRAIN_COMPLETE.json')
    manifest = json.loads((run / 'runtime.json').read_text(encoding='utf-8'))
    require(manifest['code_sha256'] == code_hashes(), 'Runtime code changed since training')
    cfg = manifest['config']
    require(sha(asset(root, cfg['records'])) == manifest['records_sha256'], 'Metadata changed since training')
    records, splits, split_sha = cohort(root, cfg, manifest['fold'])
    require(split_sha == manifest['split_artifact_sha256'], 'Split protocol changed since training')
    destination = asset(root, args.output)
    require(not destination.exists(), 'Refusing existing prediction output')
    require(not (run / 'INFER_COMPLETE.json').exists(), 'Held-out inference already sealed; preserve existing results')
    tuned = json.loads((run / 'refiner/tuned.json').read_text())
    require(tuned['split'] == 'val' and set(tuned['validation_ids']) == set(splits['val']), 'Missing validation-frozen settings')
    if not args.execute:
        print(json.dumps({'phase': 'ready', 'fold': manifest['fold'], 'test_images': len(splits['test']),
                          'gpu_started': False, 'thresholds_frozen': True}, indent=2))
        return
    from . import features
    from .refiner import runner
    active_device = device(args.device, args.permit_gpu)
    runner.configure(root)
    # Test image pixels become accessible only after all frozen-artifact checks.
    test_rows = [r for r in records if r['image_id'] in set(splits['test'])]
    check_images(root, test_rows, targets=False)
    features.extract(root, cfg, test_rows, run / 'stage1/tokens', active_device)
    features.maps(test_rows, run / 'stage1/tokens', run / 'stage1/banks', run / 'stage1/maps', active_device)
    core_args = SimpleNamespace(run=str(run / 'refiner'), split='test', device=active_device, permit_gpu=args.permit_gpu)
    runner.predict(core_args)
    runner.apply(core_args)
    predictions = rows(run / 'refiner/postprocessed_test/detections.jsonl')
    if args.include_ground_truth:
        from .refiner.data import checked_boxes
        by_id = {r['image_id']: r for r in test_rows}
        for r in predictions:
            r['gt_boxes'] = checked_boxes(by_id[r['image_id']]['gt_boxes'], r['width'], r['height'])
    write_rows(destination, predictions)
    write_json(run / 'INFER_COMPLETE.json', {'status': 'complete', 'output': args.output,
                                             'predictions_sha256': sha(destination), 'images': len(predictions),
                                             'frozen_tuning_sha256': sha(run / 'refiner/tuned.json'),
                                             'ground_truth_attached_after_inference': args.include_ground_truth})
    print(json.dumps({'phase': 'complete', 'images': len(predictions), 'output': args.output}, indent=2))


def train_main(argv=None):
    p = argparse.ArgumentParser(description='Portable primary two-stage training; validation-only selection; no downloads.')
    p.add_argument('--root', default='.', help='Project root; config and image paths are relative to this directory')
    p.add_argument('--config', required=True)
    p.add_argument('--fold', type=int)
    p.add_argument('--output', default='runs/fold0')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--permit-gpu', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--check-dependencies', action='store_true')
    args = p.parse_args(argv)
    require(args.num_workers >= 0, 'Negative worker count')
    train(args)


def infer_main(argv=None):
    p = argparse.ArgumentParser(description='Frozen primary two-stage held-out inference; no threshold fitting.')
    p.add_argument('--root', default='.')
    p.add_argument('--run', required=True)
    p.add_argument('--output', default='predictions/test.jsonl')
    p.add_argument('--device', default='cpu')
    p.add_argument('--permit-gpu', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--include-ground-truth', action='store_true', help='Attach evaluator ground truth only after prediction')
    infer(p.parse_args(argv))
