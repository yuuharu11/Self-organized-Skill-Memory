import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/draft_reconstruction_bundles.py"
SPEC = importlib.util.spec_from_file_location("draft_reconstruction_bundles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_roundtrip_preserves_full_segment_shape_and_window_shape():
    source = np.arange(18 * 7, dtype=np.float32).reshape(18, 7)
    decoded = MODULE.roundtrip_segment(
        source,
        lambda x: (x.copy(), {"length": len(x)}),
        lambda code, pose, length: code[:length],
    )
    assert decoded.shape == source.shape
    np.testing.assert_array_equal(decoded, source)
    window = MODULE.interp_window(decoded, 5.0, 10)
    assert window.shape == (10, 7)
    np.testing.assert_array_equal(window, source[5:15])


def test_row_order_is_restored_by_dataset_row_id_not_pool_position():
    wanted = np.asarray([91, 7, 44], np.int64)
    returned = np.asarray([44, 91, 7], np.int64)
    values = np.asarray([[440], [910], [70]])
    got = MODULE.align_by_row_ids(wanted, returned, values)
    np.testing.assert_array_equal(got[:, 0], [910, 70, 440])


def test_unseen_windows_stitch_and_roundtrip_to_source_order():
    # Two consecutive stride-10 windows from a synthetic episode.
    chunks = np.zeros((2, 10, 32), np.float32)
    chunks[0, :, :7] = np.arange(70, dtype=np.float32).reshape(10, 7)
    chunks[1, :, :7] = np.arange(70, 140, dtype=np.float32).reshape(10, 7)
    un = {"task": np.asarray([88, 88]), "episode": np.asarray([3, 3]),
          "frame": np.asarray([0, 10]), "actions": chunks}
    q01, q99 = np.full(7, -1, np.float32), np.full(7, 1, np.float32)
    episodes = MODULE._unseen_episodes(un, q01, q99)
    got = episodes[(88, 3)]["actions_norm"]
    assert got.shape == (20, 7)
    np.testing.assert_allclose(got[:10], chunks[0, :, :7], atol=1e-6, rtol=0)
    np.testing.assert_allclose(got[10:], chunks[1, :, :7], atol=1e-6, rtol=0)


def test_uncovered_long_window_gets_query_only_fallback():
    rows = {"task": np.asarray([2]), "episode": np.asarray([9]),
            "frame": np.asarray([10])}
    trace = np.arange(20 * 7, dtype=np.float32).reshape(20, 7)
    episodes = {(2, 9): {"frame": np.arange(20), "actions_phys": trace,
                         "meta": {"memory": False}}}
    segments = {(2, 9): [{"task": 2, "episode": 9, "start": 0,
                          "length": 10, "row_id": 0, "a": trace[:10]}]}
    q, fallback = MODULE._long_queries(segments, np.asarray([0]), rows, episodes)
    seg, offset = q[(2, 9)][0]
    assert fallback == {0}
    assert (seg["start"], seg["length"], offset) == (10, 10, 0.0)
    np.testing.assert_array_equal(seg["a"], trace[10:])
