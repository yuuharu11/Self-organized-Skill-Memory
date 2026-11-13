import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/magnitude_bounds.py'
spec = importlib.util.spec_from_file_location('magnitude_bounds', MODULE)
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)


def test_scalar_fit_recovers_nonnegative_amplitude_and_keeps_gripper():
    template = np.arange(60, dtype=np.float64).reshape(2, 5, 6)
    template = np.concatenate([template, np.ones((2, 5, 1)) * -1], axis=-1)
    target = mb.scale_template(template, np.array([2., 0.]))
    got = mb.scalar_fit(template, target)
    np.testing.assert_allclose(got, [2., 0.], atol=1e-10)
    scaled = mb.scale_template(template, got)
    np.testing.assert_array_equal(scaled[..., 6], template[..., 6])


def test_duration_rules_and_oracle_type_exclusion():
    d = mb.duration_values([6, 8], [12, 20], [2., 4.], [10., 40.])
    np.testing.assert_array_equal(d['T1_oracle'], [6, 8])
    np.testing.assert_array_equal(d['T2_stored'], [12, 20])
    np.testing.assert_array_equal(d['T3_ten'], [10, 10])
    np.testing.assert_array_equal(d['T4_history'], [5, 10])

    candidates = np.zeros((2, 10, 7), np.float64)
    candidates[0, :, 0] = 1.
    candidates[1, :, 1] = 1.
    target = candidates[:1, :, :6].copy()
    picked, _, _, _ = mb.best_oracle_types(
        target, candidates[:, :, :6], pool_task=[3, 4], query_task=[3])
    assert picked.tolist() == [1]


def test_episode_bootstrap_is_repeatable_and_reports_rows():
    a = mb.paired_bootstrap([1., 3., 2.], [10, 10, 20], reps=100, seed=17)
    b = mb.paired_bootstrap([1., 3., 2.], [10, 10, 20], reps=100, seed=17)
    assert a == b
    assert a['rows'] == 3 and a['episodes'] == 2
    assert a['mean'] == 2.


def test_episode_ids_are_namespaced_by_task():
    ids = mb.episode_cluster([1, 1, 2], [0, 1, 0])
    assert len(np.unique(ids)) == 3
