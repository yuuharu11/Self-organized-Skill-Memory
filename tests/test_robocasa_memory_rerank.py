import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
from rerank import ActionReranker, _episode_split


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
