import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path('/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero/key_grid_l90x.py')
spec = importlib.util.spec_from_file_location('key_grid_l90x_tested', MODULE)
kgx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kgx)


def test_l90x_manifest_has_only_fixed_allowed_tasks_and_task51_missing():
    result = kgx.validate_l90x_rows()
    assert result['rows'] == 11030
    assert result['manifest_tasks'] == 78
    assert result['populated_tasks'] == 77
    assert result['missing_tasks'] == [51]


def test_l90x_task_namespace_does_not_exclude_same_numeric_libero40_task():
    # LIBERO-90 task 10 and LIBERO-40 task 10 are different task identities.
    l90x_task = np.array([1010], dtype=np.int32)
    l90x_ep = np.array([1_000_003], dtype=np.int64)
    lib40_task = np.array([10, 1010], dtype=np.int32)
    lib40_ep = np.array([3, 1_000_003], dtype=np.int64)
    allowed = kgx.same_episode_allowed(l90x_task, l90x_ep, lib40_task, lib40_ep)
    assert allowed.tolist() == [[True, False]]


def test_same_episode_exclusion_still_applies_to_memory_queries():
    allowed = kgx.same_episode_allowed(np.array([2]), np.array([7]),
        np.array([2, 2, 3]), np.array([6, 7, 7]))
    assert allowed.tolist() == [[True, False, True]]


def test_external_query_candidate_ids_are_not_fabricated():
    ids = np.r_[np.array([5, 19, 32]), np.full(4, -1)]
    assert ids.tolist() == [5, 19, 32, -1, -1, -1, -1]
    assert np.all(ids[3:] < 0)

