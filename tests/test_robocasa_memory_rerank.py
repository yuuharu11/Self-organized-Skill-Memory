import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
from experiment import first_rows
from rerank import ActionReranker, _episode_split


def test_first_rows_matches_executed_prefix_not_whole_unit_interpolation():
    rng = np.random.default_rng(0)
    unit = rng.normal(size=(50, 12)).astype(np.float32)
    got = first_rows(unit, 5)
    np.testing.assert_array_equal(got, unit[:5])
    # A whole-unit interpolation would already deviate by step 4.
    assert not np.allclose(got[4], unit[0] + (unit[-1] - unit[0]) * (4 / 49))


def test_first_rows_edge_pads_short_units_and_rejects_empty():
    unit = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    got = first_rows(unit, 5)
    assert got.shape == (5, 4)
    np.testing.assert_array_equal(got[:3], unit)
    np.testing.assert_array_equal(got[3:], np.broadcast_to(unit[-1], (2, 4)))
    try:
        first_rows(np.zeros((0, 4), np.float32), 5)
    except ValueError:
        pass
    else:
        raise AssertionError("empty unit must raise")


def test_episode_split_has_no_episode_pair_leakage():
    task = np.repeat(np.arange(3), 20)
    episode = np.tile(np.repeat(np.arange(4), 5), 3)
    val, _ = _episode_split(task, episode, seed=17)
    train_pairs = set(zip(task[~val].tolist(), episode[~val].tolist()))
    val_pairs = set(zip(task[val].tolist(), episode[val].tolist()))
    assert train_pairs.isdisjoint(val_pairs)
    assert val.any() and (~val).any()


def test_reranker_scores_candidate_list_without_task_feature():
    torch.set_num_threads(1)
    model = ActionReranker(qdim=12, cdim=38)
    q = torch.zeros(2, 12)
    candidates = torch.zeros(2, 20, 38)
    out = model(q, candidates)
    assert out.shape == (2, 20)
    assert torch.isfinite(out).all()
