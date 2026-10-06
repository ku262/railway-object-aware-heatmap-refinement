import copy
import numpy as np
import pytest
from heatmap_controls import image_gates, apply_gates, shuffle_grid


def test_gate_uses_validation_and_preserves_scores():
    rows = [dict(fold=0,base_part='part',image_id=str(i),image_score=s,
                 gt_boxes=[[0,0,1,1]] if i else [],pred_boxes=[[0,0,1,1]],pred_scores=[.7])
            for i,s in enumerate([.1,.8])]
    gates = image_gates(rows)
    original = copy.deepcopy(rows)
    result = apply_gates(rows,gates)
    assert not result[0]['pred_boxes']
    assert result[1]['pred_scores']==[.7]
    assert rows==original
    for row in rows:
        row['gt_boxes']=[]
    assert apply_gates(rows,gates)[1]['pred_scores']==[.7]


def test_shuffle_is_deterministic_and_preserves_values():
    grid = np.arange(37*37).reshape(37,37)
    actual = shuffle_grid(grid,42)
    assert actual.shape==grid.shape
    assert np.array_equal(actual,shuffle_grid(grid,42))
    assert np.array_equal(np.sort(actual.ravel()),grid.ravel())
    assert not np.array_equal(actual,grid)


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        shuffle_grid(np.array([[np.nan]]),42)
    with pytest.raises(ValueError):
        image_gates([])
    with pytest.raises(ValueError):
        image_gates([dict(fold=0,base_part='part',image_id='x',image_score=.5,gt_boxes=[])])
