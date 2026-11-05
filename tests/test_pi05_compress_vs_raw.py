import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/compress_vs_raw/common.py'
spec = importlib.util.spec_from_file_location('compress_vs_raw_test_common', SCRIPT)
common = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = common
spec.loader.exec_module(common)


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

