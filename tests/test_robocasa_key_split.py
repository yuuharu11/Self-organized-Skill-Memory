"""CPU contract tests for the RoboCasa split motion key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"
sys.path.insert(0, str(MODULE))
import key_split


def test_ks_a_score_is_query_dot_mean_of_normalized_key_tokens():
    q = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 0.0], [0.0, -4.0]]])
    actual = key_split.ks_a_scores(q, keys, 2.5)
    expected = F.normalize(q, dim=-1) @ F.normalize(keys, dim=-1).mean(dim=1).T * 2.5
    torch.testing.assert_close(actual, expected)


def test_decoded_motion_type_keys_are_row_aligned_and_seven_dimensional():
    pool = {
        "action10": np.zeros((2, 10, 12), np.float32),
        "shape": np.zeros((2, 10, 6), np.float32),
    }
    pool["action10"][0, :, 11] = 1.0
    pool["action10"][1, :, 11] = -1.0
    pool["shape"][0, :, 0] = 2.0
    pool["shape"][1, :, 1] = 3.0
    actual = key_split._split_actions(pool)
    assert actual.shape == (2, 10, 7)
    np.testing.assert_array_equal(actual[0, :, :6], pool["shape"][0])
    np.testing.assert_array_equal(actual[1, :, :6], pool["shape"][1])
    # Policy order puts stored grip column 11 at active7 index 6.
    np.testing.assert_array_equal(actual[:, :, 6], pool["action10"][:, :, 11])


def test_same_episode_candidates_are_not_allowed():
    memory_episode = np.array([2, 7, 7])
    candidate_episode = np.array([2, 7, 9])
    allowed = memory_episode[:, None] != candidate_episode[None, :]
    assert not allowed[0, 0]
    assert not allowed[1, 1]
    assert not allowed[2, 1]
    assert allowed[0, 2]


def test_continuation_window_requires_real_demo_and_unit_steps():
    remaining = np.array([15, 9, 14, 20])
    unit_length = np.array([15, 15, 9, 10])
    np.testing.assert_array_equal(
        key_split.valid_window_mask(remaining, unit_length, 10),
        np.array([True, False, False, True]),
    )
    np.testing.assert_array_equal(
        key_split.valid_window_mask(remaining, unit_length, 15),
        np.array([True, False, False, False]),
    )
