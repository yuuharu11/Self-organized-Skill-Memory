import numpy as np

from skill_moe.skillnet.examples.memory_pilot.pi05_memory.score_audit.retrieval_4b import (
    _rank_matrix,
    stable_topk,
    standardized,
)


def test_stable_topk_uses_row_id_for_ties():
    ids, scores = stable_topk(np.asarray([.2, .5, .5, .1]), np.asarray([9, 8, 3, 1]), k=3)
    np.testing.assert_array_equal(ids, [3, 8, 9])
    np.testing.assert_allclose(scores, [.5, .5, .2], rtol=0, atol=1e-7)


def test_vectorized_fixed_keys_obey_task_and_episode_exclusions():
    x = np.asarray([[1, 0], [.9, .1], [0, 1], [-1, 0], [0, -1]], np.float32)
    rows = {'task': np.asarray([0, 0, 1, 2, 3]), 'episode': np.asarray([0, 1, 2, 3, 4])}
    qids = np.asarray([0]); mids = np.arange(5)
    task_cand, _ = _rank_matrix(x, qids, mids, rows, 'task', 'cos', batch_rows=1, k=3)
    ep_cand, _ = _rank_matrix(x, qids, mids, rows, 'episode', 'cos', batch_rows=1, k=3)
    assert 0 not in task_cand[0]
    assert 1 not in task_cand[0]
    assert 0 not in ep_cand[0]
    assert 1 in ep_cand[0]


def test_standardization_is_fitted_only_on_supplied_train_rows():
    x = np.asarray([[0., 1.], [2., 3.], [100., 100.]], np.float32)
    z, mean, std = standardized(x, np.asarray([0, 1]))
    np.testing.assert_allclose(mean, [1., 2.])
    np.testing.assert_allclose(std, [1., 1.])
    np.testing.assert_allclose(z[2], [99., 98.])
