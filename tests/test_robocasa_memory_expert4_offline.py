"""Held-out query table: indices address held-out rows, not memory rows.

oof.subset_z() only slices memory_*/candidate_* arrays, so passing held-out
row ids as its memory selector silently built drafts from unrelated memory
rows while scoring stayed on held-out targets. _heldout_query_table rebuilds
the query side from the corresponding heldout_* arrays instead.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MP = str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa")
if MP not in sys.path:
    sys.path.insert(0, MP)

import expert4_offline as e4


def _fake_z(n_mem=6, n_hold=4, n_cand=5):
    rng = np.random.default_rng(11)
    z = {}
    z["memory_task"] = np.zeros(n_mem, np.int64)
    z["memory_frame"] = np.arange(n_mem, dtype=np.int64) * 3
    z["memory_episode"] = np.arange(n_mem, dtype=np.int64)
    z["memory_hist"] = rng.normal(size=(n_mem, 10, 12)).astype(np.float32)
    z["memory_context"] = np.arange(n_mem, dtype=np.int64)
    z["memory_state"] = rng.normal(size=(n_mem, 16)).astype(np.float32)
    z["heldout_task"] = np.ones(n_hold, np.int64)
    z["heldout_frame"] = np.arange(n_hold, dtype=np.int64) * 7 + 1
    z["heldout_episode"] = 1000 + np.arange(n_hold, dtype=np.int64)
    z["heldout_local_episode"] = np.arange(n_hold, dtype=np.int64)
    z["heldout_hist"] = rng.normal(size=(n_hold, 10, 12)).astype(np.float32) + 5
    z["heldout_context"] = np.arange(n_hold, dtype=np.int64) + 50
    z["heldout_state"] = rng.normal(size=(n_hold, 16)).astype(np.float32) - 5
    z["candidate_task"] = np.arange(n_cand, dtype=np.int64)
    return z


def test_heldout_query_table_uses_heldout_rows():
    z = _fake_z()
    qids = np.array([2, 0], np.int64)
    cids = np.array([4, 1], np.int64)
    got = e4._heldout_query_table(z, qids, cids)
    np.testing.assert_array_equal(got["memory_task"], z["heldout_task"][qids])
    np.testing.assert_array_equal(got["memory_frame"], z["heldout_frame"][qids])
    np.testing.assert_array_equal(got["memory_episode"], z["heldout_episode"][qids])
    np.testing.assert_array_equal(got["memory_hist"], z["heldout_hist"][qids])
    np.testing.assert_array_equal(got["memory_context"], z["heldout_context"][qids])
    np.testing.assert_array_equal(got["memory_state"], z["heldout_state"][qids])
    np.testing.assert_array_equal(got["candidate_task"], z["candidate_task"][cids])
    # Same-episode exclusion in _generate compares global episode ids.
    assert set(np.asarray(got["memory_episode"]).tolist()) <= {1000, 1001, 1002, 1003}


def test_heldout_query_table_differs_from_memory_rows():
    z = _fake_z()
    qids = np.array([1], np.int64)
    got = e4._heldout_query_table(z, qids, np.array([0], np.int64))
    assert not np.array_equal(got["memory_hist"], np.asarray(z["memory_hist"])[qids])
