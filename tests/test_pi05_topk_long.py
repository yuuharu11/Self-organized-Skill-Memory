import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path('/work/skill_moe/skillnet/examples/memory_pilot/pi05_memory/topk_long/run.py')
SPEC = importlib.util.spec_from_file_location('pi05_topk_long_testmod', SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_medoid_returns_consensus_candidate():
    x = np.asarray([[[[0.]], [[2.]], [[10.]]], [[[5.]], [[5.1]], [[20.]]]], np.float32)
    got = mod.select_medoid(x)
    assert got.tolist() == [1, 1]


def test_min_edit_and_near_normal():
    outputs = np.asarray([[[[0.]], [[2.]], [[4.]]], [[[9.]], [[8.]], [[7.]]]], np.float32)
    drafts = np.asarray([[[[1.]], [[2.2]], [[3.]]], [[[9.1]], [[8.9]], [[6.]]]], np.float32)
    assert mod.select_min_edit(outputs, drafts).tolist() == [1, 0]
    ref = np.asarray([[[3.8]], [[7.1]]], np.float32)
    assert mod.select_near(outputs, ref).tolist() == [2, 2]


def test_episode_bootstrap_is_reproducible_and_paired():
    a = np.asarray([1., 2., 4., 8.])
    b = np.asarray([2., 3., 3., 9.])
    ep = np.asarray(['a', 'a', 'b', 'c'])
    mask = np.asarray([True, True, True, False])
    ci1 = mod.episode_bootstrap_ci(a, b, ep, mask, reps=200, seed=19)
    ci2 = mod.episode_bootstrap_ci(a, b, ep, mask, reps=200, seed=19)
    assert ci1 == ci2
    assert ci1[0] <= -0.5 <= ci1[1]


def test_candidate_validation_fails_closed_for_nonmemory_and_duplicates():
    rows = {
        'memory': np.asarray([True, True, False]),
        'task': np.asarray([0, 1, 0]),
        'episode': np.asarray([4, 4, 8]),
    }
    mod.validate_candidate_rows(rows, np.asarray([[1]]), np.asarray([0]), np.asarray([4]), 'task')
    try:
        mod.validate_candidate_rows(rows, np.asarray([[2]]), np.asarray([0]), np.asarray([4]), 'task')
    except AssertionError as exc:
        assert 'fixed memory' in str(exc)
    else:
        raise AssertionError('nonmemory candidate was accepted')
    try:
        mod.validate_candidate_rows(rows, np.asarray([[1, 1]]), np.asarray([0]), np.asarray([4]), 'task')
    except AssertionError as exc:
        assert 'duplicate' in str(exc)
    else:
        raise AssertionError('duplicate candidate was accepted')


def test_episode_exclusion_is_task_namespaced():
    rows = {
        'memory': np.asarray([True, True]),
        'task': np.asarray([0, 1]),
        'episode': np.asarray([7, 7]),
    }
    # Episode numbers can repeat in another task; only (task, episode) is identity.
    mod.validate_candidate_rows(rows, np.asarray([[1]]), np.asarray([0]), np.asarray([7]), 'episode')
    try:
        mod.validate_candidate_rows(rows, np.asarray([[0]]), np.asarray([0]), np.asarray([7]), 'episode')
    except AssertionError as exc:
        assert 'same-episode' in str(exc)
    else:
        raise AssertionError('same episode candidate was accepted')


def test_long_mapping_is_verified_from_task_text_and_data_filenames():
    rows, _, _, long_rows = mod.load_inputs()
    mapping = mod.verify_long_task_mapping(rows)
    assert set(mapping) == {str(i) for i in range(10)}
    assert len(long_rows) > 0
    assert 'turn_on_the_stove' in mapping['3']['demo_file']

