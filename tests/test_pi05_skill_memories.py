"""CPU-only checks for open-ended M6-style raw action memories."""
from __future__ import annotations

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.pi05_memory.skill_memories.build import (
    RawFunctional,
    assert_no_long,
    m6_fit,
    unit_medoid,
)


def test_m6_registers_supported_novel_regions_without_fixed_k():
    rng = np.random.default_rng(31)
    # Two separated functional regions, sufficiently supported over episodes.
    x = np.concatenate([
        rng.normal(0.0, 0.005, (24, 70)),
        rng.normal(0.8, 0.005, (24, 70)),
    ]).astype(np.float32)
    row_ids = np.arange(len(x), dtype=np.int64)
    tasks = np.full(len(x), 12, np.int32)
    episodes = np.arange(len(x), dtype=np.int32)
    result = m6_fit(x, row_ids, tasks, episodes, RawFunctional(), kappa=.1,
                    epsilon_override=.01, seed=4, pending_batch=8)
    assert result['registered'] == 2
    assert np.all(result['row_units'] >= 0)
    assert sorted(result['support'].tolist()) == [24, 24]
    assert all(len(es) == 24 for es in result['episode_sets'])
    assert len(result['centers']) == len(result['members_local'])


def test_medoid_is_an_actual_member_minimizing_mean_functional_distance():
    x = np.asarray([[0.0] * 70, [0.01] * 70, [0.04] * 70, [1.0] * 70], np.float32)
    members = np.asarray([0, 1, 2], np.int64)
    medoid = unit_medoid(members, x, RawFunctional())
    metric = RawFunctional()
    costs = [np.mean([metric.distance(x[i], x[j]) for j in members]) for i in members]
    assert medoid in members
    assert costs[list(members).index(medoid)] == min(costs)


def test_no_long_guard_rejects_long_rows_and_eval_overlap():
    rows = {'task': np.asarray([0, 11, 12, 2]), 'memory': np.asarray([1, 1, 1, 0])}
    eval_ids = np.asarray([3], np.int64)
    assert_no_long([1, 2], eval_ids, rows, context='test')
    try:
        assert_no_long([0], eval_ids, rows, context='test')
    except AssertionError as exc:
        assert 'Long task rows leaked' in str(exc)
    else:
        raise AssertionError('Long memory row was not rejected')
    try:
        assert_no_long([3], eval_ids, rows, context='test')
    except AssertionError as exc:
        assert 'Long Eval rows leaked' in str(exc)
    else:
        raise AssertionError('Eval row was not rejected')
