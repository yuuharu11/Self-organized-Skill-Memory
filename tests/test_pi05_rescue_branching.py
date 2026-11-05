from pathlib import Path
import importlib.util

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/fast_search.py'
SPEC = importlib.util.spec_from_file_location('rescue_branch_fast_search', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_blockwise_fp16_normalization_matches_original_formula():
    import torch
    torch.manual_seed(12)
    values = torch.randn(11, 13, 17, dtype=torch.float16)
    expected = values.clone()
    expected /= torch.linalg.vector_norm(expected.float(), dim=-1,
                                         keepdim=True).clamp_min(1e-8).half()
    actual = MODULE.normalize_half_keys_blockwise(values.clone(), torch, block_rows=3)
    assert torch.equal(actual, expected)


def test_search_scores_identical_for_small_vs_large_blocks():
    import torch
    torch.manual_seed(31)
    torch.set_num_threads(2)
    n, ntok, width = 29, 12, 32
    keys = torch.randn(n, ntok, width).half()
    key_groups = torch.tensor(np.tile(np.repeat(np.arange(3), 4), (n, 1)))
    qgroups = np.repeat(np.arange(3), 2)
    query = torch.randn(6, width).half()
    query /= torch.linalg.vector_norm(query.float(), dim=-1, keepdim=True).half()
    selected = [query[qgroups == g] for g in range(3)]
    ranges = [(0,4),(4,8),(8,12)]
    small = MODULE.maxsim_scores_blockwise(keys, key_groups, selected, ranges, torch, block=3)
    large = MODULE.maxsim_scores_blockwise(keys, key_groups, selected, ranges, torch, block=16)
    torch.testing.assert_close(small, large, rtol=1e-6, atol=1e-6)
    assert int(torch.argmax(small)) == int(torch.argmax(large))
