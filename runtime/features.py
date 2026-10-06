"""Portable DINO tokens, actual-ratio random banks, and owner-excluded matching."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import asset, require, sha, write_json


def token_path(folder, image_id):
    return Path(folder) / (hashlib.sha256(image_id.encode()).hexdigest() + '.npy')


def checked_tokens(path):
    a = np.load(path, allow_pickle=False)
    require(a.shape == (1369, 768) and a.dtype == np.float32, 'Expected normalized float32 37x37 DINOv2-B/14 tokens')
    require(np.isfinite(a).all() and np.all(np.abs(np.linalg.norm(a, axis=1) - 1) < 1e-4), 'Invalid normalized features')
    return a


def extract(root, cfg, records, output, device):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoder = cfg['encoder']
    if encoder['kind'] == 'cached_tokens':
        folder = asset(root, encoder['directory'])
        signature = json.loads((folder / 'signature.json').read_text())
        require(signature['records_sha256'] == sha(asset(root, cfg['records']))
                and signature['weights_sha256'] == encoder['weights_sha256']
                and signature['size'] == 518 and signature['dtype'] == 'float32'
                and signature['l2_normalized'] is True, 'Cached-token provenance mismatch')
        for r in records:
            a = checked_tokens(token_path(folder, r['image_id']))
            dest = token_path(output, r['image_id'])
            require(not dest.exists(), 'Do not overwrite frozen tokens')
            np.save(dest, a, allow_pickle=False)
        return
    import timm
    from torchvision import transforms
    from PIL import Image
    weights = asset(root, encoder['weights'])
    require(sha(weights) == encoder['weights_sha256'], 'DINO checkpoint checksum mismatch')
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, img_size=518)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    for key in ('model', 'teacher', 'state_dict'):
        if key in state:
            state = state[key]
            break
    state = {k.replace('module.', '').replace('backbone.', '').replace('teacher.', ''): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys and all(k == 'mask_token' for k in incompatible.unexpected_keys), 'Incompatible DINO weights')
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose([transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    with torch.inference_mode():
        for r in records:
            with Image.open(asset(root, r['image_path'])) as image:
                x = transform(image.convert('RGB')).unsqueeze(0).to(device)
            value = model.forward_features(x)
            tokens = value['x_norm_patchtokens'] if isinstance(value, dict) else value[:, 1:]
            tokens = torch.nn.functional.normalize(tokens.float(), dim=-1)[0].cpu().numpy()
            dest = token_path(output, r['image_id'])
            require(not dest.exists(), 'Do not overwrite frozen tokens')
            np.save(dest, tokens, allow_pickle=False)
            checked_tokens(dest)


def make_bank(records, folder, ratio, seed):
    records = sorted(records, key=lambda r: r['image_id'])
    require(records and all(r['label'] == 0 for r in records), 'Bank needs inner-train normals only')
    sizes = [len(checked_tokens(token_path(folder, r['image_id']))) for r in records]
    bounds = np.cumsum([0] + sizes)
    count = math.ceil(int(bounds[-1]) * ratio)
    selected = np.sort(np.random.default_rng(seed).choice(int(bounds[-1]), count, replace=False))
    features, owners, tokens = [], [], []
    for i, r in enumerate(records):
        indexes = selected[(selected >= bounds[i]) & (selected < bounds[i + 1])] - bounds[i]
        if len(indexes):
            features.append(checked_tokens(token_path(folder, r['image_id']))[indexes])
            owners.extend([r['group_id']] * len(indexes))
            tokens.extend({'image_id': r['image_id'], 'group_id': r['group_id'], 'token_index': int(k)} for k in indexes)
    return np.concatenate(features), owners, {'global_token_indices': selected.tolist(), 'tokens': tokens,
                                              'all_train_normal_tokens': int(bounds[-1]), 'ratio': ratio, 'seed': seed}


def match(query, bank, owners, query_group, device='cpu', bank_chunk=32768, query_chunk=512):
    import torch
    valid = np.array([owner != query_group for owner in owners])
    require(valid.any(), 'No bank reference survives annotation-group exclusion')
    result = []
    with torch.inference_mode():
        for start in range(0, len(query), query_chunk):
            q = torch.as_tensor(query[start:start + query_chunk], dtype=torch.float32, device=device)
            best = torch.full((len(q),), -torch.inf, device=device)
            for j in range(0, len(bank), bank_chunk):
                b = torch.as_tensor(bank[j:j + bank_chunk], dtype=torch.float32, device=device)
                scores = q @ b.T
                scores[:, torch.as_tensor(~valid[j:j + bank_chunk], device=device)] = -torch.inf
                best = torch.maximum(best, scores.max(1).values)
            require(torch.isfinite(best).all().item(), 'Nonfinite reference distances')
            result.append((1 - best).clamp_min(0).cpu().numpy())
    return np.concatenate(result)


def part_path(folder, part):
    return Path(folder) / hashlib.sha256(part.encode()).hexdigest()[:12]


def choose_and_build(cfg, train, val, cache, out, device, seed):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    parts = sorted({r['base_part'] for r in train})
    options = []
    for ratio in cfg['bank']['ratios']:
        predictions = []
        for part in parts:
            normal = [r for r in train if r['base_part'] == part and r['label'] == 0]
            bank, owners, _ = make_bank(normal, cache, ratio, seed)
            for r in val:
                if r['base_part'] == part:
                    d = np.sort(match(checked_tokens(token_path(cache, r['image_id'])), bank, owners, r['group_id'], device))
                    predictions.append((r, {t: float(d[-max(1, int(len(d) * t / 100)):].mean()) for t in cfg['bank']['top_percent']}))
        for top in cfg['bank']['top_percent']:
            metrics = []
            if cfg['bank']['selection'] != 'fixed':
                for part in parts:
                    selected = [(r['label'], s[top]) for r, s in predictions if r['base_part'] == part]
                    labels, scores = zip(*selected)
                    require(set(labels) == {0, 1}, 'Strict bank selection requires binary inner validation for every part')
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
                    metrics.append([roc_auc_score(labels, scores), average_precision_score(labels, scores), float(f1[:-1].max())])
            metric = np.mean(metrics, axis=0).tolist() if metrics else [0, 0, 0]
            options.append({'ratio': ratio, 'top_percent': top, 'validation_macro_auc_ap_f1': metric})
    chosen = max(options, key=lambda r: (*r['validation_macro_auc_ap_f1'], -r['ratio'], -r['top_percent']))
    for part in parts:
        dest = part_path(out, part)
        dest.mkdir(parents=True)
        bank, owners, provenance = make_bank([r for r in train if r['base_part'] == part and r['label'] == 0], cache, chosen['ratio'], seed)
        np.save(dest / 'bank.npy', bank, allow_pickle=False)
        write_json(dest / 'owners.json', owners)
        write_json(dest / 'provenance.json', {'base_part': part, **provenance})
    write_json(Path(out) / 'selection.json', {'chosen': chosen, 'candidates': options, 'selected_on': 'inner_val_only' if cfg['bank']['selection'] != 'fixed' else 'predeclared',
                                           'validation_ids': [r['image_id'] for r in val], 'candidate_pool_cap': None, 'bank_budget_cap': None})


def maps(records, cache, banks, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for part in sorted({r['base_part'] for r in records}):
        folder = part_path(banks, part)
        bank = np.load(folder / 'bank.npy', allow_pickle=False)
        owners = json.loads((folder / 'owners.json').read_text())
        for r in records:
            if r['base_part'] != part:
                continue
            d = match(checked_tokens(token_path(cache, r['image_id'])), bank, owners, r['group_id'], device)
            path = token_path(output, r['image_id'])
            require(not path.exists(), 'Do not overwrite a heatmap')
            np.save(path, d.reshape(37, 37), allow_pickle=False)
