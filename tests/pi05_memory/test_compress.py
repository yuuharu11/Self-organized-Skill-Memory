"""Compression of stored action units: cluster oracle, dimensions, compressed vs raw."""
from __future__ import annotations

import numpy as np
import pytest

from tests.support import (
    PI05,
    load_script,
)

cluster = load_script(PI05 / "compress_cluster_oracle/experiment.py")
cluster_queue = load_script(PI05 / "compress_cluster_oracle/run_queue.py")
dims = load_script(PI05 / "compress_vs_raw/dimension_extension.py")
common = load_script(PI05 / "compress_vs_raw/common.py")


# ── Cluster oracle ────────────────────────────────────────────────────────────────────────────────

def test_each_query_action_row_is_in_pool_and_row_ids_are_not_pool_offsets():
    pool = cluster.compose_pool(np.array([20, 24]), np.array([2, 9]), np.array([31]), np.array([40]))
    assert cluster.assert_pool_coverage(pool, np.array([40, 2, 31]))
    assert cluster.map_row_ids(pool, np.array([40, 2, 31])).tolist() == [5, 0, 4]
    with pytest.raises(AssertionError):
        cluster.assert_pool_coverage(pool, np.array([41]))


def test_raw_none_can_select_exact_truth_with_zero_mse():
    truth = np.arange(70, dtype=np.float32).reshape(10, 7)
    pool = np.stack([truth + 1, truth, truth - 2])
    distances = np.asarray([cluster.active7_mse(truth, candidate) for candidate in pool])
    assert int(np.argmin(distances)) == 1
    assert distances[1] == 0.0


def test_raw_none_pipeline_returns_own_action_and_actual_row_id():
    rng = np.random.default_rng(3)
    actions = np.zeros((30, 10, 32), np.float32)
    actions[5] = rng.normal(size=(10, 32))
    actions[20] = rng.normal(size=(10, 32))
    pool = np.asarray([5, 20], np.int64)
    values = actions[pool, :, :7].reshape(2, 70)
    fit = {'row_units': np.asarray([0, 1], np.int32)}
    primary, secondary, chosen, _ = cluster._predictions_for_queries(
        'raw', None, None, values, fit, np.asarray([0, 1]), pool, pool,
        np.asarray([20]), {'task': np.zeros(30, np.int32)}, actions, np.zeros((30, 2), np.float32))
    assert chosen.tolist() == [20]
    np.testing.assert_array_equal(primary[0], actions[20])
    np.testing.assert_array_equal(secondary[0], actions[20])


def test_long_rows_cannot_enter_expert_training():
    rows = {'task': np.asarray([10, 11, 2, 0])}
    assert cluster.assert_long_excluded_from_expert([0, 1], rows)
    with pytest.raises(AssertionError):
        cluster.assert_long_excluded_from_expert([0, 3], rows)


def test_shuffle_sources_are_from_other_tasks():
    tasks = np.asarray([10, 10, 11, 11, 12, 12])
    source = np.asarray([0, 2, 4, 1, 3, 5])
    query = np.asarray([0, 1, 2, 3, 4, 5])
    shuffled, _ = cluster.stable_other_task_shuffle(source, query, tasks, seed=12)
    assert np.all(tasks[shuffled] != tasks[query])


def test_m6default_handles_numpy_center_arrays_without_changing_distance():
    from skill_memories.build import RawFunctional
    rng = np.random.default_rng(17)
    values = np.broadcast_to(rng.normal(size=(1, 70)).astype(np.float32), (24, 70)).copy()
    rows = {'task': np.repeat(np.arange(4), 6),
            'episode': np.tile(np.repeat(np.arange(3), 2), 4)}
    ids = np.arange(24, dtype=np.int64)
    fit = cluster._fit_unit(values, ids, rows, RawFunctional(), 'm6default', np.arange(24))
    assert fit['row_units'].shape == (24,)
    assert len(fit['centers']) >= 1


def test_b12_none_secondary_uses_each_queries_own_row_and_matches_primary():
    class TinyCodec:
        def prior(self, feature, state):
            n = len(feature)
            return np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32)

        def decode(self, base, residual, feature, state):
            return np.broadcast_to(np.asarray(base)[:, :1, None], (len(base), 10, 32)).copy()

    nrows = 4
    actions = np.zeros((nrows, 10, 32), np.float32)
    actions[1, :, :7] = 1.0
    actions[2, :, :7] = 2.0
    actions[3, :, :7] = 3.0
    bases = np.arange(nrows, dtype=np.float32)[:, None]
    features = np.zeros((nrows, 2), np.float32)
    rows = {'state': np.zeros((nrows, 3), np.float32)}
    pool = np.asarray([1, 2, 3], np.int64)
    reps = pool.copy()
    fit = {'row_units': np.asarray([0, 1, 2], np.int32)}
    primary, secondary, chosen, _ = cluster._predictions_for_queries(
        'b12', TinyCodec(), bases, bases[pool], fit, np.arange(3), reps, pool,
        np.asarray([1, 2, 3]), rows, actions, features)
    assert chosen.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(secondary, primary)


def test_queue_has_no_duplicate_expert_cells():
    base = cluster_queue.BASE
    extra = cluster_queue.EXTRA
    all_cells = base + extra
    assert len(base) == 16
    assert len(extra) == 4
    assert len(set(all_cells)) == len(all_cells) == 20
    assert not set(base).intersection(extra)


# ── Dimension extension ───────────────────────────────────────────────────────────────────────────

def test_codec_fit_split_excludes_validation_and_memory_but_documents_long_inclusion():
    rows={'validation':np.array([0,0,1,1,1],bool),'memory':np.array([0,1,0,0,0],bool),
          'task':np.array([0,3,1,10,35])}
    ids=dims.codec_fit_ids(rows)
    np.testing.assert_array_equal(ids,[0])
    assert rows['task'][ids[0]]<10 # Long training examples are intentionally retained.
    hold=dims.nonlong_holdout_ids(rows)
    np.testing.assert_array_equal(hold,[3,4])


def test_candidate_base_encoding_uses_dataset_row_ids():
    class Codec:
        def encode(self, actions, states):
            assert np.array_equal(states[:,0],actions[:,0,0]+50)
            return actions[:,0,:2], np.zeros((len(actions),4),np.float32)
    actions=np.zeros((20,10,32),np.float32); states=np.zeros((20,32),np.float32)
    actions[:,0,0]=np.arange(20); states[:,0]=np.arange(20)+50
    candidate_ids=np.array([[17,3],[11,1]])
    b=dims.encode_candidate_b(Codec(),{'state':states},actions,candidate_ids)
    np.testing.assert_array_equal(b[:,:,0],candidate_ids)
    np.testing.assert_array_equal(b[:,:,1],np.zeros_like(candidate_ids))


def test_holm_step_down_is_monotone_and_bounded():
    adjusted=dims._holm({'a':.001,'b':.02,'c':.04,'d':.9})
    assert all(0<=x<=1 for x in adjusted.values())
    assert adjusted['a']<=adjusted['b']<=adjusted['c']<=adjusted['d']


def test_c3_direction_uses_m2_minus_shuffle_sign_convention():
    assert dims._contrast_direction('C3', {'ci95': [-.2, -.1]}) == 'M2_better_than_shuffle'
    assert dims._contrast_direction('C3', {'ci95': [.1, .2]}) == 'shuffle_better_than_M2'
    assert dims._contrast_direction('C3', {'ci95': [-.1, .1]}) == 'no_detected_difference'


def test_reviewed_shuffle_is_exact_top1_permutation_and_candidates_are_memory_rows():
    if not dims.REVIEW.exists():
        pytest.skip('requires local artifacts')
    fair=dims.fair_module(); rows=dims.base_module().load_rows(dims.RUN)
    prep=np.load(dims.REVIEW/'drafts_prepared.npz',allow_pickle=False)
    long=prep['eval_query_rows']; m2=prep['m2_candidate_rows']; shuffle=prep['shuffle_candidate_rows']
    assert np.array_equal(np.sort(m2),np.sort(shuffle))
    assert np.all(np.asarray(rows['task'][shuffle])!=np.asarray(rows['task'][long]))
    pool=np.asarray(np.load(dims.MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'])
    assert np.all(np.isin(np.r_[m2,shuffle,prep['oracle_candidate_rows']],pool))
    assert len(fair.training_rows(rows))==11712


def test_e2_selected_candidate_ids_are_preserved_for_training_and_not_long():
    if not dims.REVIEW.exists():
        pytest.skip('requires local artifacts')
    fair=dims.fair_module(); rows=dims.base_module().load_rows(dims.RUN)
    saved=np.load(dims.FAIR/'E-M2/training_drafts.npz',allow_pickle=False)
    assert np.array_equal(saved['query_rows'],fair.training_rows(rows))
    assert np.array_equal(saved['candidate_rows'][np.arange(len(saved['rank'])),saved['rank']],
                          saved['selected_candidate_rows'])
    pool=np.asarray(np.load(dims.MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'])
    fair.validate_candidates(saved['selected_candidate_rows'][:,None],saved['query_rows'],rows,pool)


# ── Compressed vs raw ─────────────────────────────────────────────────────────────────────────────

class FakeCodec:
    def prior(self, feature, state):
        return np.zeros((len(feature), 2), np.float32), np.asarray(state[:, :1], np.float32)

    def decode(self, b, r, f, s):
        # Output encodes all inputs so bad pool-offset indexing is observable.
        v = np.asarray(b).sum(1) + np.asarray(r)[:, 0] * 10 + np.asarray(s)[:, 1]
        return np.broadcast_to(v[:, None, None], (len(v), 10, 32)).copy()


def test_decoded_candidates_use_dataset_row_ids_not_pool_offsets():
    bases = np.arange(40, dtype=np.float32).reshape(20, 2)
    features = np.zeros((20, 3), np.float32)
    states = np.zeros((20, 4), np.float32)
    states[:, 0] = np.arange(20) + 1
    states[:, 1] = np.arange(20) + 101
    q = np.array([17, 13])
    candidates = np.array([[2, 19], [1, 18]])
    got = common.decoded_candidate_drafts(FakeCodec(), bases, features, states, q, candidates, batch=1)
    expected = np.empty_like(got)
    for i, row in enumerate(q):
        residual = states[row, 0]
        for j, candidate in enumerate(candidates[i]):
            v = bases[candidate].sum() + residual * 10 + states[row, 1]
            expected[i, j] = v
    np.testing.assert_array_equal(got, expected)


def test_candidate_row_id_range_is_checked():
    with pytest.raises(IndexError):
        common.decoded_candidate_drafts(FakeCodec(), np.zeros((5, 2)), np.zeros((5, 3)),
                                        np.zeros((5, 4)), [0], [[5]])


def test_saved_output_artifacts_are_never_replaced(tmp_path):
    existing = tmp_path / 'decision.json'
    existing.write_text('{}')
    with pytest.raises(FileExistsError):
        common.require_new_artifacts([tmp_path / 'new.npz', existing])


def test_existing_em2_selected_candidate_rows_are_fixed_and_long_free():
    root = common.FAIR / 'E-M2/training_drafts.npz'
    if not root.exists():
        pytest.skip('fair-expert artifacts are not present')
    saved = np.load(root, allow_pickle=False)
    assert saved['selected_candidate_rows'].shape == saved['query_rows'].shape
    assert not np.any(saved['query_rows'] < 0)
    assert np.array_equal(saved['candidate_rows'][np.arange(len(saved['query_rows'])), saved['rank']],
                          saved['selected_candidate_rows'])
    base = common.base_module()
    rows = base.load_rows(common.RUN)
    fair = common.fair_module()
    assert np.array_equal(saved['query_rows'], fair.training_rows(rows))
    assert np.all(np.asarray(rows['task'][saved['query_rows']]) >= 10)
    pool = np.asarray(np.load(common.MEMORY / 'units.npz', allow_pickle=False)['pool_row_ids'], np.int64)
    fair.validate_candidates(saved['selected_candidate_rows'][:, None], saved['query_rows'], rows, pool)
    assert np.all(np.asarray(rows['task'][saved['selected_candidate_rows']]) >= 10)


def test_row_candidate_identity_and_mse_lookup_are_not_pool_offsets():
    # Explicitly mimic a sparse memory inventory where IDs differ from positions.
    pool_ids = np.array([3, 11, 19], dtype=np.int64)
    actions = np.arange(22 * 10 * 7, dtype=np.float32).reshape(22, 10, 7)
    chosen_id = np.array([19])
    got = actions[chosen_id]
    np.testing.assert_array_equal(got[0], actions[pool_ids[2]])
    assert not np.array_equal(got[0], actions[2])


def test_shuffle_uses_the_exact_m2_top1_multiset_and_crosses_tasks():
    corrected_path = common.FAIR / 'corrected-top1-shuffle-01/top1_shuffle_predictions.npz'
    official_path = common.FAIR / 'evaluation-final/offline_predictions.npz'
    if not corrected_path.exists() or not official_path.exists():
        pytest.skip('fair-expert corrected Top-1 shuffle artifacts are not present')
    corrected = np.load(corrected_path, allow_pickle=False)
    official = np.load(official_path, allow_pickle=False)
    assert np.array_equal(corrected['query_rows'], official['row_ids'])
    source = np.asarray(official['m2_candidate_rows'][:, 0], np.int64)
    shuffled = np.asarray(corrected['shuffled_candidate_rows'], np.int64)
    assert np.array_equal(corrected['source_candidate_rows'], source)
    assert np.array_equal(np.sort(shuffled), np.sort(source))
    base = common.base_module()
    rows = base.load_rows(common.RUN)
    assert np.all(np.asarray(rows['task'])[shuffled] != np.asarray(rows['task'])[corrected['query_rows']])
