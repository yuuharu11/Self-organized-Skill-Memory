"""Offline retrieval must select the same eligible entries as the S2 server."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/anchor_flow"
sys.path.insert(0, str(HERE))
from memory import prepare_centered_online_query
from offline_retrieval import center_normalize, softmax_weights, topk_centered


def test_offline_and_online_match_with_ties_and_both_exclusions():
    keys = np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    queries = keys[[0, 2]]
    episodes = np.asarray(["e0", "e1", "e2", "e3", "e4", "e5"])
    initials = np.asarray(["i0", "i1", "i2", "i3", "i4", "i5"])
    query_episodes, query_initials = np.asarray(["e2", "e0"]), np.asarray(["i3", "i1"])
    centered, mean = center_normalize(keys)
    query_centered, _ = center_normalize(queries, mean)
    chosen, similarities = topk_centered(
        query_centered, centered, episodes, initials, query_episodes, query_initials,
        k=3, block_size=1)
    weights = softmax_weights(similarities, .05)
    bank = SimpleNamespace(memory=[
        dict(key=key, unit_id=i, meta=dict(episode_id=episodes[i], init_state_id=initials[i]))
        for i, key in enumerate(keys)])
    for i, query in enumerate(queries):
        online = prepare_centered_online_query(
            bank, query, dict(episode_id=query_episodes[i], init_state_id=query_initials[i]),
            top_k=3, temperature=.05)
        np.testing.assert_array_equal(chosen[i], online["_prepared_candidates"])
        np.testing.assert_allclose(similarities[i], online["_anchor_v2_similarities"], atol=1e-7)
        np.testing.assert_allclose(weights[i], online["_anchor_v2_weights"], atol=1e-7)


def test_excluded_entries_cannot_fill_a_short_candidate_set():
    with pytest.raises(ValueError, match="eligible"):
        topk_centered(np.ones((1, 2)), np.ones((2, 2)), ["same", "other"],
                      ["a", "b"], ["same"], ["b"], k=1)


def test_tiny_temperature_is_stable_and_invalid_temperature_is_rejected():
    weights = softmax_weights([[.999, .998, -np.inf]], 1e-8)
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert weights[0, 0] == 1.
    with pytest.raises(ValueError, match="positive"):
        softmax_weights([[1., 0.]], 0.)
