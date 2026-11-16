import importlib.util
import os
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

MODULE = Path(__file__).parents[2] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/key_split_ablation.py"
spec = importlib.util.spec_from_file_location("key_split_ablation_test", MODULE)
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)


def test_ks_a_matches_query_dot_mean_of_normalized_key_segments():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    keys = torch.tensor([
        [[2.0, 0.0], [0.0, 3.0]],
        [[1.0, 1.0], [-1.0, 1.0]],
    ])
    got = ks.ks_a_scores(q, keys)
    expected = F.normalize(q, dim=-1) @ F.normalize(keys, dim=-1).mean(dim=1).T
    torch.testing.assert_close(got, expected)


def test_ks_b_normalizes_the_mean_of_queries_and_raw_keys():
    q = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    raw_keys = torch.tensor([
        [[8.0, 0.0], [0.0, 1.0]],
        [[0.0, 4.0], [0.0, 2.0]],
    ])
    got = ks.ks_b_scores(q, raw_keys)
    expected = F.normalize(q.mean(dim=1), dim=-1) @ F.normalize(raw_keys.mean(dim=1), dim=-1).T
    torch.testing.assert_close(got, expected)

    # This is deliberately different from averaging per-segment normalized keys.
    wrong = F.normalize(q.mean(dim=1), dim=-1) @ F.normalize(F.normalize(raw_keys, dim=-1).mean(1), dim=-1).T
    assert not torch.allclose(got, wrong)
