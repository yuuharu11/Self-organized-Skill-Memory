import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import xf1_run as X


def test_task_stratified_episode_folds_and_no_leakage():
    task = np.repeat(np.arange(3), 20)
    episode = np.tile(np.arange(20), 3)
    assignment = X.assign_episode_folds(task, episode, seed=7, n_folds=5)
    assert len(assignment) == 60
    for t in range(3):
        counts = [sum(f == i for (tt, _), f in assignment.items() if tt == t) for i in range(5)]
        assert counts == [4] * 5
    train, held, bank, folds = X.fold_partition(task, episode, task, episode, assignment, 2)
    held_pairs = set(zip(task[held].tolist(), episode[held].tolist()))
    assert not held_pairs.intersection(zip(task[train].tolist(), episode[train].tolist()))
    assert not held_pairs.intersection(zip(task[bank].tolist(), episode[bank].tolist()))
    assert np.all(folds[held] == 2)


def test_confidence_features_are_label_free_and_d10_d5_are_topk_disagreement():
    assert 'target' not in inspect.signature(X.confidence_from_scores).parameters
    assert 'raw_target_coeff' not in inspect.signature(X.confidence_from_scores).parameters
    scores = np.arange(12, 0, -1, dtype=np.float32)[None]
    keys = np.arange(12 * 18, dtype=np.float32).reshape(12, 18)
    actions = np.zeros((12, 10, 7), np.float32)
    for i in range(12): actions[i, :, :6] = i
    allowed = np.ones((1, 12), bool)
    out = X.confidence_from_scores(scores, keys, actions, allowed, top_k=12)
    assert out['top_ids'].tolist() == [list(range(12))]
    def expected(k):
        p = np.exp(scores[0, :k] - scores[0, :k].max()); p /= p.sum()
        return float((p * np.square(np.arange(k, dtype=np.float64))).sum())
    assert np.isclose(out['D5'][0], expected(5))
    assert np.isclose(out['D10'][0], expected(10))


def test_truth_scoring_is_separate_and_flattens_saved_coefficients():
    f = {'top_ids': np.array([[1], [0]], np.int32)}
    target_c = np.zeros((2, 18), np.float32)
    raw_keys = np.stack([np.zeros((3, 6)), np.ones((3, 6))]).astype(np.float32)
    target_a = np.zeros((2, 10, 7), np.float32)
    memory_a = np.zeros((2, 10, 7), np.float32)
    memory_a[1, :, :6] = 1.
    out = X.score_truth(f, target_c, raw_keys, target_a, memory_a)
    assert np.allclose(out['e_arm'], [60 / 70, 0.])
    assert np.allclose(out['e_coef'], [18., 0.])

