"""Portable paths, immutable artifacts, and public split-contract validation."""
import hashlib
import importlib.util
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def asset(root, value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value,
            'Use nonempty portable relative POSIX paths')
    p = Path(value)
    require(not p.is_absolute(), 'Config/manifest paths must be relative')
    root = Path(root).resolve()
    result = (root / p).resolve()
    require(result != root and root in result.parents, 'Asset escapes project root')
    return result


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def seal(folder, name):
    folder = Path(folder)
    hashes = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p.name not in (name, '.lock')}
    write_json(folder / name, {'status': 'complete', 'sha256': hashes})


def verify(folder, name):
    folder = Path(folder)
    marker = json.loads((folder / name).read_text(encoding='utf-8'))
    require(marker['status'] == 'complete', 'Incomplete run')
    for rel, expected in marker['sha256'].items():
        require(sha(asset(folder, rel)) == expected, 'Frozen artifact changed: ' + rel)
    return marker


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_config(root, path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    require(cfg['version'] == 1 and cfg['experiment'] in ('primary_fixed5', 'strict_nested_validation'), 'Unknown runtime experiment')
    require(cfg['encoder']['kind'] in ('dinov2', 'cached_tokens'), 'Unsupported encoder')
    require(cfg['encoder']['size'] == 518, 'Primary feature resolution is fixed at 518')
    require(cfg['refiner'] == {'mode': 'rgb_heat', 'base': 16, 'epochs': 20, 'loss': 'bce_dice',
                                'normalization': 'per_image', 'selection': 'final'}, 'Primary refiner settings are fixed; use separate experiment code for ablations')
    expected = {'ratios': [0.05], 'top_percent': [3], 'selection': 'fixed'} if cfg['experiment'] == 'primary_fixed5' else {
        'ratios': [0.01, 0.03, 0.05, 0.1, 0.2, 1.0], 'top_percent': [1, 2, 3, 4, 5, 10], 'selection': 'inner_val_macro_auc'}
    require(cfg['bank'] == expected, 'Bank settings differ from the named experiment')
    require(cfg['seed'] == 42, 'Protocol seed must be 42')
    for key in ('records', 'splits'):
        asset(root, cfg[key])
    key = 'weights' if cfg['encoder']['kind'] == 'dinov2' else 'directory'
    asset(root, cfg['encoder'][key])
    require(isinstance(cfg['encoder']['weights_sha256'], str) and len(cfg['encoder']['weights_sha256']) == 64
            and all(c in '0123456789abcdef' for c in cfg['encoder']['weights_sha256']),
            'Supply an explicit expected pretrained-weight SHA256')
    return cfg


def cohort(root, cfg, fold):
    from split import verify_artifact
    artifact = verify_artifact(json.loads(asset(root, cfg['splits']).read_text(encoding='utf-8')))
    records = rows(asset(root, cfg['records']))
    by_id = {r['image_id']: r for r in records}
    require(len(by_id) == len(records) and set(by_id) == {r['image_id'] for r in artifact['records']}, 'Record membership differs from frozen public splits')
    for item in artifact['records']:
        r = by_id[item['image_id']]
        require(all(r[k] == v for k, v in item.items()), 'Metadata differs from frozen public split record')
        asset(root, r['image_path'])
        require(type(r['width']) is int and type(r['height']) is int and r['width'] > 0 and r['height'] > 0, 'Invalid crop dimensions')
    split = artifact['splits'][str(fold)]
    return records, {role: split[role] for role in ('train', 'val', 'test')}, artifact['sha256']


def check_images(root, records, targets=True):
    from PIL import Image
    from .refiner.data import checked_boxes
    for r in records:
        path = asset(root, r['image_path'])
        require(sha(path) == r['image_sha256'], 'Image checksum mismatch')
        with Image.open(path) as im:
            require(im.size == (r['width'], r['height']), 'Image dimension mismatch')
            im.verify()
        if targets:
            boxes = checked_boxes(r['gt_boxes'], r['width'], r['height'])
            require(bool(boxes) == bool(r['label']), 'Label/box inconsistency')


def dependencies(encoder_kind):
    names = ['numpy', 'torch', 'PIL', 'cv2', 'sklearn']
    if encoder_kind == 'dinov2':
        names += ['timm', 'torchvision']
    return {name: importlib.util.find_spec(name) is not None for name in names}
