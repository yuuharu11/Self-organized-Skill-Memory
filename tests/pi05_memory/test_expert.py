"""Experts that repair drafts: fair expert, skill expert, output by draft."""
import inspect

import numpy as np
import pytest

from tests.support import PI05, load_script

fair = load_script(PI05 / "fair_expert/run.py")
review = load_script(PI05 / "fair_expert/review_top1_shuffle.py")
skill_expert = load_script(PI05 / "skill_expert/run.py")
output_by_draft = load_script(PI05 / "output_by_draft/run.py")


# ── Fair expert ───────────────────────────────────────────────────────────────────────────────────

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


# ── Skill expert ──────────────────────────────────────────────────────────────────────────────────

def test_training_mixture_has_fixed_categorical_contract():
    labels,counts=skill_expert.mix_categories(np.random.default_rng(20261104),100000)
    assert labels.shape==(100000,)
    rates=counts/counts.sum()
    np.testing.assert_allclose(rates,[.60,.15,.25],atol=.006)

def test_confidence_schedule_edges_and_fallback_are_fixed():
    edges={'q20':.2,'q60':.6,'q80':.8}
    assert skill_expert.confidence_mode(.19,edges,'C')=={'fallback':True,'t0':None,'draft':'normal'}
    assert skill_expert.confidence_mode(.2,edges,'C')['t0']==.6
    assert skill_expert.confidence_mode(.6,edges,'C')['t0']==.4
    assert skill_expert.confidence_mode(.8,edges,'C')['t0']==.2
    assert skill_expert.confidence_mode(.1,edges,'B')['t0']==.6
    assert skill_expert.confidence_mode(.1,edges,'A')['t0']==.6
    assert skill_expert.confidence_mode(.1,edges,'D')['draft']=='zero'

def test_fixed_score_edges_use_dev_only_quantiles():
    scores=np.arange(100,dtype=float)
    edges=skill_expert.confidence_edges(scores)
    assert edges['n_dev']==100
    np.testing.assert_allclose([edges['q20'],edges['q60'],edges['q80']],np.quantile(scores,[.2,.6,.8]))
    assert 'success' not in edges

def test_environment_and_query_seed_protocol():
    for init in range(10):
        env,q=skill_expert.seed_values(init,17)
        assert env==7+init
        assert q==(7+init)*100000+17


# ── Output by draft ───────────────────────────────────────────────────────────────────────────────

def test_direction_cos_uses_summed_xyz():
    a = np.zeros((1, 10, 7)); a[0, :, 0] = 1.
    assert output_by_draft.direction_cos(a, a) == 1.
    assert output_by_draft.direction_cos(a, -a) == -1.
    b = np.zeros((1, 10, 7)); b[0, :, 1] = 1.
    assert output_by_draft.direction_cos(a, b) == 0.


def test_row_mse_ignores_inactive_dimensions():
    a = np.zeros((1, 10, 32)); b = a.copy(); b[:, :, 7:] = 5.
    assert output_by_draft.row_mse(a, b)[0] == 0.


def test_bins_split_points_and_skip_sparse_ranges():
    draft = np.r_[np.full(25, .001), np.full(30, .03), np.full(3, 1.)]
    gen = {t: draft * 2 for t in output_by_draft.T0S}
    bins = output_by_draft.bin_by_draft(draft, np.ones_like(draft), gen)
    assert [b["points"] for b in bins] == [25, 30]
    assert bins[0]["t0=0.2"] == .002
    assert bins[0]["below_normal_t0=0.2"] == 1.


def test_bootstrap_is_paired_over_episodes():
    r = output_by_draft.bootstrap(np.r_[np.full(10, 1.), np.full(10, 1.)], np.r_[np.zeros(10), np.ones(10)], reps=50)
    assert r["mean"] == 1. and r["ci95"] == [1., 1.]
