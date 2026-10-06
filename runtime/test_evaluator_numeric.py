"""Public evaluator accepts real NumPy scalars without weakening validation."""
import unittest

import numpy as np

from evaluation.evaluator import evaluate, validate


def record():
    return dict(image_id="fixture", base_part="part", source_id="source", fold=0,
                width=32, height=32, gt_boxes=[[0, 0, 10, 10]],
                pred_boxes=[[0, 0, 10, 10]], pred_scores=[0.8])


class NumericContracts(unittest.TestCase):
    def test_numpy_real_coordinates_and_scores(self):
        for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                row = record()
                row['gt_boxes'] = [[dtype(v) for v in row['gt_boxes'][0]]]
                row['pred_boxes'] = [[dtype(v) for v in row['pred_boxes'][0]]]
                row['pred_scores'] = [dtype(1)]
                validate([row])
                self.assertEqual(evaluate([row], .5)['tp'], 1)

    def test_invalid_coordinate_and_score_scalars(self):
        for value in (True, np.bool_(True), np.float32(np.nan),
                      np.float64(np.inf), complex(1, 0), '1'):
            for field in ('gt_boxes', 'pred_boxes', 'pred_scores'):
                with self.subTest(value=value, field=field):
                    row = record()
                    if field == 'pred_scores':
                        row[field][0] = value
                    else:
                        row[field][0][0] = value
                    with self.assertRaises(ValueError):
                        validate([row])


if __name__ == '__main__':
    unittest.main()
