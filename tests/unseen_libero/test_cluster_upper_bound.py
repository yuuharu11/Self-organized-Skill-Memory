import sys
from pathlib import Path

import numpy as np
import pytest

MOD = Path('/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero')
sys.path.insert(0, str(MOD))
import cluster_upper_bound as cub


def test_nearest_uses_all_requested_axes_and_returns_pool_position():
    target = np.zeros((1, 10, 7), np.float32)
    pool = np.ones((3, 10, 7), np.float32)
    pool[2] = 0.2
    idx7, e7 = cub.nearest(target, pool, 7)
    idx6, e6 = cub.nearest(target, pool, 6)
    assert idx7.tolist() == [2] and idx6.tolist() == [2]
    assert np.isclose(e7[0], 0.2**2)
    assert np.isclose(e6[0], 6 * 0.2**2 / 7)


def test_no_pose_or_scale_fit_in_candidate_metrics():
    target = np.zeros((1, 10, 7), np.float32)
    pool = np.ones((1, 10, 7), np.float32)
    idx, err = cub.nearest(target, pool)
    assert idx[0] == 0
    assert np.isclose(err[0], 1.0)
    assert np.all(pool[0] == 1.0)  # candidate is scored verbatim


def test_arm_and_gripper_contributions_sum_to_active7():
    y = np.zeros((2, 10, 7), np.float32)
    p = np.ones_like(y) * 0.1
    h = np.array([True, False])
    arm, _ = cub.metric_row(p, y, h, axes=6)
    grip = cub.grip_contribution(p, y, h)
    active, _ = cub.metric_row(p, y, h, axes=7)
    assert np.isclose(arm['all_mse'] + grip['all'], active['all_mse'])
    assert np.isclose(arm['hard_mse'] + grip['hard'], active['hard_mse'])


def test_eval_selection_is_not_memory_and_is_validation_only():
    eval_ids = np.array([1, 2])
    cub.assert_eval_rows_outside_memory({'memory': np.array([True, False, False, True]),
                                         'validation': np.array([False, True, True, False])}, eval_ids)
    with pytest.raises(AssertionError):
        cub.assert_eval_rows_outside_memory({'memory': np.array([True, False, False, True]),
                                             'validation': np.array([False, True, True, False])}, [0])


def test_cmp3_g1_reference_contract():
    assert cub.EXPECTED_G1['units'] == 4516
    assert np.isclose(cub.EXPECTED_G1['all'], 0.034447897620114326)
    assert np.isclose(cub.EXPECTED_G1['hard'], 0.06845424480130637)


def test_run_and_report_are_a_only_and_never_build_closed_loop_pools(monkeypatch, tmp_path):
    rows = {'memory': np.array([True, False]),
            'validation': np.array([False, True])}
    data = {'ids': np.array([1]), 'targets': np.zeros((1, 10, 7), np.float32),
            'hard': np.array([False]), 'traces': {'eval': np.zeros((10, 7), np.float32)},
            'rows_of': {'eval': np.array([0])}}
    d = {'rows': rows, 'mem_ids': np.array([0]), 'memory': np.zeros((1, 10, 7), np.float32)}
    evals = {'learned': data, 'unseen': data}
    counts = {k: {'tasks': 1, 'demos': 1, 'windows': 1, 'hard': 0}
              for k in ('learned', 'unseen')}
    stats = {'all_mse': 0.1, 'hard_mse': None}
    refs = [{'name': 'synthetic_reference', 'eval': {
        split: {'active7': stats, 'arm6': stats} for split in ('learned', 'unseen')}}]

    monkeypatch.setattr(cub.c3, 'prepare', lambda: (d, evals))
    monkeypatch.setattr(cub.c3, 'unit_meta', lambda d, ev: counts)
    monkeypatch.setattr(cub, 'source_variable_units',
                        lambda d: (np.zeros((1, 5, 7)), np.zeros((1, 5, 7)),
                                   np.zeros((1, 32, 7)), np.array([5])))
    monkeypatch.setattr(cub.c3, 'eval_variable', lambda items, lengths, data:
                        ({'all_mse': cub.EXPECTED_G1['all'],
                          'hard_mse': cub.EXPECTED_G1['hard']}, None, None))
    monkeypatch.setattr(cub.c3, 'subset_eval', lambda x, n_demos: x)
    monkeypatch.setattr(cub, 'variable_results', lambda d, ev, smoke: [])
    monkeypatch.setattr(cub, 'reference_rows', lambda d, ev, full: refs)

    result = cub.run(smoke=True)
    assert 'A' in result and 'references' in result
    assert 'B' not in result
    assert not hasattr(cub, 'fixed_results')
    report = tmp_path / 'report.md'
    cub.render(result, report)
    text = report.read_text(encoding='utf-8')
    assert '## A.' in text
    assert '## B.' not in text
    assert '閉ループ用 pool' not in text


@pytest.mark.skipif(not (cub.ROOT / 'unseen-libero' / 'compress-seen-cuts-01' / 'production-control' / 'evaluation-selection.json').exists(),
                    reason='CMP3 real caches absent; real-data reproduction is exercised by --smoke')
def test_real_cmp3_g1_full_reproduction():
    d, evals = cub.c3.prepare()
    raw, coded, _, lengths = cub.source_variable_units(d)
    got, _, _ = cub.c3.eval_variable(coded, lengths, evals['unseen'])
    assert len(raw) == 4516
    assert abs(got['all_mse'] - cub.EXPECTED_G1['all']) <= 2e-6
    assert abs(got['hard_mse'] - cub.EXPECTED_G1['hard']) <= 2e-6

