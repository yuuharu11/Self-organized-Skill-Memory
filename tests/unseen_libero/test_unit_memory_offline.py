import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path('/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero/unit_memory_offline.py')
SPEC = importlib.util.spec_from_file_location('unit_memory_offline_testmod', SOURCE)
UM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UM)


def test_error_conventions_and_dct3_keep_gripper():
    y = np.zeros((2, 10, 7), np.float32)
    p = y.copy(); p[:, :, 0] = 1.; p[:, :, 6] = 2.
    assert np.allclose(UM.row_loss(p, y, 7, slice(0, 10)), 5.0 / 7.0)
    assert np.allclose(UM.row_loss(p, y, 6, slice(0, 10)), 1.0 / 7.0)
    a = np.random.default_rng(4).normal(size=(3, 10, 7)).astype(np.float32)
    assert np.array_equal(UM.dct3_pi05(a)[..., 6], a[..., 6])


def test_eval_split_guards_learned_rows_and_cache_alignment():
    from compression_seen_cuts import prepare, select_seen
    d, evals = prepare(); s = select_seen(d['rows'])
    rows = d['rows']; ids = s['ids']
    assert len(ids) == 621
    assert not np.asarray(rows['memory'])[ids].any()
    assert np.asarray(rows['validation'])[ids].all()
    learned = UM._learned_data(d, __import__('compression_seen_cuts'), evals['learned']['hard'])
    assert learned['audit']['selector_train_overlap'] == 0
    assert learned['audit']['memory_overlap'] == 0
    assert 'ground-truth recorded demo actions' in learned['audit']['history_source']


def test_serving_selector_is_deterministic_and_pool_checkpoint_matches():
    from compression_seen_cuts import prepare
    d, evals = prepare()
    data = UM._learned_data(d, __import__('compression_seen_cuts'), evals['learned']['hard'])
    data = {k: (v[:4] if isinstance(v, np.ndarray) and len(v) == len(data['row']) else v)
            for k, v in data.items()}
    sel = UM._selectors()['unit_08']
    a = UM._select(sel, data)
    b = UM._select(sel, data)
    assert np.array_equal(a, b)
    # Serving path and helper share one API; selected indices are local pool indices.
    assert np.all((a >= 0) & (a < len(sel.shapes)))
    pred = UM._pred_with_pi05_pose(sel, a, data['pi05'])
    assert pred.shape == data['pi05'].shape
    assert np.array_equal(pred[:, :, 6], data['pi05'][:, :, 6])


def test_mag1_a6_t3_supplement_reconstruction():
    # This compares against the explicitly cited supplement metric, not magnitude-01's split-window table.
    p = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/magnitude-01/eval_rows.npz')
    if not p.exists(): pytest.skip('MAG1 cache is unavailable')
    got = UM.reproduce_mag1_supplement()
    assert got['rows'] == 619 and got['hard_rows'] == 134
    assert abs(got['all_full10'] - .1348) < 5e-4
    assert abs(got['hard_full10'] - .2921) < 5e-4


def test_long_task_mapping_uses_task_id_space():
    assert UM._long_exclusion_task(0) == 5
    assert UM._long_exclusion_task(10) == 7
    assert UM._long_exclusion_task(88) == 2
    # LIBERO-90 ID 10 maps to LIBERO-40 task 7, not the numerically equal candidate 10.
    mask = UM.task_exclusion_mask(np.array([7, 10, 3]), np.array([10]), unseen=True)
    assert mask.tolist() == [[False, True, True]]
