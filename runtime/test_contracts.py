"""Synthetic contracts only: real preprocessing/inference, mocked training optimizer.

Run from release root: python -B -m unittest runtime.test_contracts -v
No pretrained weights, real data, optimization, downloads, or GPU allocation.
Temporary fixtures are confined to runtime/_test_work and removed afterwards.
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from split import make_splits
from runtime import features, pipeline
from runtime.common import asset, check_images, read_config, rows, sha, verify, write_json, write_rows
from runtime.refiner import runner
from runtime.refiner.model import SmallUNet, total_loss

torch.set_num_threads(2)
BASE = Path(__file__).resolve().parent


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = BASE / '_test_work'
        work.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=work)
        cls.root = Path(cls.temp.name)
        (cls.root / 'images').mkdir()
        (cls.root / 'cache').mkdir()
        cfg = json.loads((BASE / 'configs/primary_fixed5.example.json').read_text())
        cfg['encoder'] = {'kind': 'cached_tokens', 'directory': 'cache', 'weights_sha256': '1' * 64, 'size': 518}
        cls.cfg = cfg
        write_json(cls.root / 'config.json', cfg)
        values = []
        for fold in range(5):
            for label in (0, 1):
                for j in range(2):
                    ident = f'f{fold}_l{label}_g{j}'
                    rng = np.random.default_rng(100 * fold + 10 * label + j)
                    path = cls.root / 'images' / (ident + '.png')
                    Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(path)
                    r = {'image_id': ident, 'image_path': path.relative_to(cls.root).as_posix(),
                         'group_id': 'g_' + ident, 'source_id': 'source_' + str(j), 'base_part': 'synthetic_part',
                         'fold': fold, 'label': label, 'width': 40, 'height': 32,
                         'gt_boxes': [[5, 6, 25, 24]] if label else [], 'image_sha256': sha(path)}
                    values.append(r)
                    a = rng.standard_normal((1369, 768)).astype(np.float32)
                    a /= np.linalg.norm(a, axis=1, keepdims=True)
                    np.save(features.token_path(cls.root / 'cache', ident), a, allow_pickle=False)
        cls.records = values
        write_rows(cls.root / cfg['records'], values)
        artifact = make_splits(values)
        cls.splits = artifact['splits']['0']
        write_json(cls.root / cfg['splits'], artifact)
        write_json(cls.root / 'cache/signature.json', {'records_sha256': sha(cls.root / cfg['records']),
                   'weights_sha256': '1' * 64, 'size': 518, 'dtype': 'float32', 'l2_normalized': True})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        work = BASE / '_test_work'
        if work.exists() and not list(work.iterdir()):
            work.rmdir()

    def test_01_complete_train_contract_then_portable_infer(self):
        args = SimpleNamespace(root=str(self.root), config=str(self.root / 'config.json'), fold=0,
                               output='runs/primary', device='cpu', num_workers=0, permit_gpu=False,
                               execute=True, check_dependencies=False)
        test_ids = set(self.splits['test'])
        test_paths = {str((self.root / r['image_path']).resolve()) for r in self.records if r['image_id'] in test_ids}
        original_open = Image.open

        def development_only_open(path, *a, **kw):
            if isinstance(path, (str, Path)) and str(Path(path).resolve()) in test_paths:
                raise AssertionError('Training orchestration opened held-out pixels')
            return original_open(path, *a, **kw)

        def checkpoint_double(core_args):
            # A fixture models the trainer output contract; no optimizer is created.
            bundle, config = runner.prepare(core_args)
            path = Path(core_args.output)
            path.mkdir()
            runner.save_json(path / 'config.json', config)
            torch.random.default_generator.manual_seed(123)
            model = SmallUNet('rgb_heat').eval()
            runner.atomic(path / 'final.pt', lambda f: torch.save({'epoch': 20, 'config_hash': config['config_hash'],
                                                                 'state_dict': model.state_dict(), 'synthetic_contract_double': True}, f))
            runner.save_json(path / 'complete.json', {'epoch': 20, 'config_hash': config['config_hash'],
                                                      'final_sha256': sha(path / 'final.pt')})

        with patch.object(Image, 'open', side_effect=development_only_open), patch.object(runner, 'train', side_effect=checkpoint_double) as mocked:
            pipeline.train(args)
            mocked.assert_called_once()
        run = self.root / 'runs/primary'
        verify(run, 'TRAIN_COMPLETE.json')
        self.assertTrue((run / 'validation_predictions.jsonl').exists())
        self.assertFalse(any(features.token_path(run / 'stage1/tokens', i).exists() for i in test_ids))
        prepared = {r['image_id']: r for r in rows(run / 'inputs/records.jsonl')}
        self.assertTrue(all('gt_boxes' not in prepared[i] and 'label' not in prepared[i] for i in test_ids))
        bank_folder = features.part_path(run / 'stage1/banks', 'synthetic_part')
        owner_groups = set(json.loads((bank_folder / 'owners.json').read_text()))
        expected_groups = {r['group_id'] for r in self.records if r['image_id'] in self.splits['train'] and r['label'] == 0}
        self.assertTrue(owner_groups <= expected_groups)
        checkpoint_hash = sha(run / 'refiner/final.pt')
        frozen_hash = sha(run / 'refiner/tuned.json')
        # Copy a complete user project to a new root: configs/artifacts must relocate.
        relocated = self.root.parent / (self.root.name + '_relocated')
        try:
            shutil.copytree(self.root, relocated)
            infer_args = SimpleNamespace(root=str(relocated), run='runs/primary', output='predictions/test.jsonl',
                                         device='cpu', permit_gpu=False, execute=True, include_ground_truth=False)
            pipeline.infer(infer_args)
            predictions = rows(relocated / 'predictions/test.jsonl')
            self.assertEqual({r['image_id'] for r in predictions}, test_ids)
            self.assertTrue(all(r['fold'] == 0 and 'gt_boxes' not in r for r in predictions))
            self.assertEqual(sha(relocated / 'runs/primary/refiner/final.pt'), checkpoint_hash)
            self.assertEqual(sha(relocated / 'runs/primary/refiner/tuned.json'), frozen_hash)
            self.assertTrue(all(not Path(r['refined_mask_path']).is_absolute() for r in predictions))
            with self.assertRaises(ValueError):
                pipeline.infer(infer_args)
        finally:
            shutil.rmtree(relocated, ignore_errors=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_02_group_exclusion_uses_real_vectors(self):
        query = np.array([[1., 0.]], np.float32)
        bank = np.array([[1., 0.], [0., 1.]], np.float32)
        d = features.match(query, bank, ['same', 'other'], 'same', 'cpu', 1, 1)
        np.testing.assert_array_equal(d, [1.])
        with self.assertRaises(ValueError):
            features.match(query, bank[:1], ['same'], 'same', 'cpu')

    def test_03_real_refiner_forward_and_loss_no_optimization(self):
        model = SmallUNet('rgb_heat').eval()
        with torch.no_grad():
            logits = model(torch.zeros((1, 4, 256, 256)))
            value = total_loss(logits, torch.zeros_like(logits))
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(value).item())

    def test_04_private_absolute_paths_rejected(self):
        for value in ('/absolute/data', 'C:/private/data', '../escape'):
            with self.assertRaises(ValueError):
                asset(self.root, value)

    def test_05_strict_config_is_separate(self):
        value = json.loads((BASE / 'configs/strict_nested.example.json').read_text())
        self.assertEqual(value['experiment'], 'strict_nested_validation')
        bad = copy.deepcopy(self.cfg)
        bad['bank']['ratios'] = [0.01]
        path = self.root / 'bad_config.json'
        write_json(path, bad)
        with self.assertRaises(ValueError):
            read_config(self.root, path)

    def test_06_image_checksum_mismatch_aborts(self):
        bad = dict(self.records[0], image_sha256='0' * 64)
        with self.assertRaises(ValueError):
            check_images(self.root, [bad])

    def test_07_gpu_requires_opt_in(self):
        with self.assertRaises(ValueError):
            pipeline.device('cuda', False)

    def test_08_frozen_artifact_tampering_aborts(self):
        run = self.root / 'runs/primary'
        if not run.exists():
            self.skipTest('Depends on full contract fixture')
        p = run / 'stage1/banks/selection.json'
        original = p.read_bytes()
        try:
            p.write_bytes(original + b' ')
            with self.assertRaises(ValueError):
                verify(run, 'TRAIN_COMPLETE.json')
        finally:
            p.write_bytes(original)

    def test_09_dino_adapter_without_downloaded_weights(self):
        from types import ModuleType
        fake_timm = ModuleType('timm')

        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.ones(1))

            def forward_features(self, x):
                return {'x_norm_patchtokens': torch.ones((len(x), 1369, 768), device=x.device)}

        calls = []
        def construct(name, **kwargs):
            calls.append((name, kwargs))
            return FakeEncoder()
        fake_timm.create_model = construct
        weights = self.root / 'weights/synthetic.pth'
        weights.parent.mkdir()
        torch.save(FakeEncoder().state_dict(), weights)
        cfg = copy.deepcopy(self.cfg)
        cfg['encoder'] = {'kind': 'dinov2', 'weights': 'weights/synthetic.pth', 'weights_sha256': sha(weights), 'size': 518}
        with patch.dict('sys.modules', {'timm': fake_timm}):
            features.extract(self.root, cfg, [self.records[0]], self.root / 'adapter_tokens', 'cpu')
        tokens = features.checked_tokens(features.token_path(self.root / 'adapter_tokens', self.records[0]['image_id']))
        self.assertEqual(tokens.shape, (1369, 768))
        self.assertFalse(calls[0][1]['pretrained'])
        self.assertEqual(calls[0][0], 'vit_base_patch14_dinov2')

    def test_10_strict_selection_only_inner_validation_and_tiebreak(self):
        cfg = json.loads((BASE / 'configs/strict_nested.example.json').read_text())
        tr = [{'image_id': 'tn', 'group_id': 'tn', 'base_part': 'p', 'label': 0},
              {'image_id': 'ta', 'group_id': 'ta', 'base_part': 'p', 'label': 1}]
        val = [{'image_id': 'vn', 'group_id': 'vn', 'base_part': 'p', 'label': 0},
               {'image_id': 'va', 'group_id': 'va', 'base_part': 'p', 'label': 1}]
        def bank_builder(records, *args):
            self.assertEqual([r['image_id'] for r in records], ['tn'])
            return np.ones((1, 768), np.float32), ['tn'], {'synthetic_contract': True}
        def token_reader(path):
            v = .9 if Path(path).name == features.token_path('.', 'va').name else .1
            return np.full((100, 768), v, np.float32)
        target = self.root / 'strict_selection'
        with patch.object(features, 'make_bank', side_effect=bank_builder), \
                patch.object(features, 'checked_tokens', side_effect=token_reader), \
                patch.object(features, 'match', side_effect=lambda q, *args: q[:, 0]):
            features.choose_and_build(cfg, tr, val, self.root, target, 'cpu', 42)
        frozen = json.loads((target / 'selection.json').read_text())
        self.assertEqual(frozen['chosen']['ratio'], .01)
        self.assertEqual(frozen['chosen']['top_percent'], 1)
        self.assertEqual(frozen['validation_ids'], ['vn', 'va'])
        self.assertEqual(len(frozen['candidates']), 36)

    def test_11_cli_defaults_are_nonexecuting(self):
        with patch.object(pipeline, 'train') as train:
            pipeline.train_main(['--config', 'runtime/configs/primary_fixed5.example.json', '--fold', '0'])
            self.assertFalse(train.call_args.args[0].execute)
            self.assertFalse(train.call_args.args[0].permit_gpu)
        with patch.object(pipeline, 'infer') as infer:
            pipeline.infer_main(['--run', 'runs/fold0'])
            self.assertFalse(infer.call_args.args[0].execute)

    def test_12_config_placeholder_does_not_download(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['encoder'] = {'kind': 'dinov2', 'weights': 'weights/not_downloaded.pth', 'weights_sha256': '0' * 64, 'size': 518}
        with patch.object(pipeline, 'dependencies', return_value={'torch': True}):
            with self.assertRaisesRegex(ValueError, 'not bundled'):
                pipeline.preflight(self.root, cfg, 0)


if __name__ == '__main__':
    unittest.main()
