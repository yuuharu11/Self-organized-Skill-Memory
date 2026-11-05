import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/fair_expert/run.py'
spec = importlib.util.spec_from_file_location('pi05_fair_expert_tested', MODULE_PATH)
fair = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fair)


REVIEW_PATH = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/fair_expert/review_top1_shuffle.py'
review_spec = importlib.util.spec_from_file_location('pi05_fair_expert_review_tested', REVIEW_PATH)
review = importlib.util.module_from_spec(review_spec)
assert review_spec.loader is not None
review_spec.loader.exec_module(review)


def test_candidate_row_ids_index_dataset_not_pool_positions():
    actions = np.zeros((20, 10, 32), np.float32)
    actions[14] = 14
    actions[3] = 3
    actions[19] = 19
    actions[2] = 2
    actions[0] = 0
    # Pool order deliberately differs from the source row IDs.
    pool = np.asarray([14, 3, 19])
    cands = np.asarray([[2, 0], [14, 3]])
    actual = fair.candidate_drafts(actions, cands)
    np.testing.assert_array_equal(actual[:, :, 0, 0], [[2, 0], [14, 3]])
    # A mistaken pool-position lookup would return pool[2]=19 and pool[0]=14.
    assert not np.array_equal(actual[0], actions[pool[cands[0]]])


def test_training_rows_exclude_long_validation_and_withheld_tasks():
    task = np.asarray([0, 11, 17, 19, 28, 29, 35, 38, 12, 13])
    validation = np.asarray([False, False, False, False, False, False, False, False, True, False])
    rows = {'task': task, 'validation': validation}
    ids = fair.training_rows(rows)
    np.testing.assert_array_equal(ids, [1, 9])
    assert np.all(task[ids] >= 10)
    assert not np.any(validation[ids])
    assert not np.any(np.isin(task[ids], fair.HOLDOUT))


def test_validation_rows_use_fixed_calibration_dev_episodes_only():
    tasks = np.repeat([17, 19, 28, 29, 35, 38], 5)
    episodes = np.tile(np.arange(5), 6)
    validation = np.ones(len(tasks), bool)
    rows = {'task': tasks, 'episode': episodes, 'validation': validation}
    ids = fair.validation_rows(rows)
    assert len(ids) == 6 * 3
    assert set(tasks[ids]) == set(fair.HOLDOUT.tolist())


def test_candidate_validation_rejects_same_task_and_nonmemory_row():
    rows = {
        'actions': np.zeros((8, 10, 32)),
        'task': np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
        'episode': np.asarray([0, 0, 0, 0, 0, 0, 0, 0]),
    }
    with pytest.raises(AssertionError, match='same-task'):
        fair.validate_candidates(np.asarray([[1]]), np.asarray([0]), rows, np.asarray([1, 2, 3]))
    with pytest.raises(AssertionError, match='outside raw'):
        fair.validate_candidates(np.asarray([[7]]), np.asarray([0]), rows, np.asarray([1, 2, 3]))


def test_shuffle_uses_a_different_task():
    rows = {
        'task': np.asarray([10, 11, 12, 10, 11, 12]),
        'episode': np.arange(6),
    }
    queries = np.asarray([0, 1, 2, 3, 4, 5])
    cands = np.asarray([[1, 2, 4, 5, 3], [2, 0, 3, 4, 5], [0, 1, 3, 4, 5],
                        [4, 2, 1, 5, 0], [5, 0, 2, 3, 1], [3, 1, 0, 2, 4]])
    shuffled, _ = fair.shuffle_other_task(cands, queries, rows, seed=5)
    assert np.all(rows['task'][shuffled] != rows['task'][queries])
    # Shuffling changes query assignment, not the source candidate distribution.
    source_top1 = cands[:, 0]
    shuffled_top1, _ = fair.shuffle_other_task(source_top1[:, None], queries, rows, seed=5)
    np.testing.assert_array_equal(np.sort(shuffled_top1), np.sort(source_top1))
    impossible_rows = {'task': np.asarray([7, 7]), 'episode': np.asarray([0, 1])}
    with pytest.raises(RuntimeError, match='preserving the candidate multiset'):
        fair.shuffle_other_task(np.asarray([[0], [1]]), np.asarray([0, 1]), impossible_rows, seed=5)


def test_same_batch_ids_for_both_experts():
    ids = np.arange(100, 200)
    a = fair.make_batch_indices(ids, steps=20, batch=8, seed=20261104)
    b = fair.make_batch_indices(ids, steps=20, batch=8, seed=20261104)
    assert a.shape == (20, 8)
    np.testing.assert_array_equal(a, b)


def test_m2_draft_mixture_is_sampled_per_query():
    c = np.tile(np.arange(5, dtype=np.int64), (10000, 1))
    picked, ranks = fair.sample_m2_training_draft(c, np.random.default_rng(20261104))
    assert picked.shape == ranks.shape == (10000,)
    assert .68 < np.mean(ranks == 0) < .72
    assert np.all((ranks >= 0) & (ranks <= 4))
    np.testing.assert_array_equal(picked, ranks)


def test_m2_fixed_dev_sample_is_reproducible_and_has_mixture_support():
    c = np.tile(np.arange(5, dtype=np.int64), (500, 1))
    a, ar = fair.sample_m2_training_draft(c, np.random.default_rng(fair.SEED + 1902))
    b, br = fair.sample_m2_training_draft(c, np.random.default_rng(fair.SEED + 1902))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ar, br)
    assert np.unique(ar).size > 1


def test_review_reads_evaluate_output_directory_by_default():
    # run.evaluate writes offline_predictions.npz and decision.json directly
    # under the output directory; the review must default to that directory,
    # not an evaluation-final subdirectory that no command creates.
    params = inspect.signature(review.evaluate).parameters
    assert params['source'].default == fair.OUT_DEFAULT
    assert 'evaluation-final' not in str(params['source'].default)
