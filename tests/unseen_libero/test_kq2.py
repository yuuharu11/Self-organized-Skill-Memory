"""Small KQ2/KQ2-CL helper tests; no cached data or policy weights required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero"
sys.path.insert(0, str(EXAMPLES))

import kq2cl_common as common  # noqa: E402


def test_history_window_shape_padding_and_validity():
    actions = np.ones((12, 7), np.float32)
    states = np.ones((12, 8), np.float32)
    window, valid = common.window_tokens(actions, states, 4)
    assert window.shape == (10, 16)  # previous action 7 + state 8 + action-valid flag
    assert valid.shape == (10,)
    assert valid.sum() == 5
    assert np.all(window[:6, :7] == 0)  # left-padded action history
    assert np.all(window[6:, :7] == 1)
    assert window[5, 15] == 0 and window[-1, 15] == 1


def test_decode_gripper_single_switch_from_previous_sign():
    pclose = np.array([[.1, .2, .9, .8, .7, .6, .5, .4, .3, .2]], np.float32)
    out_open = common.decode_gripper(pclose, np.array([-1]), .6)
    assert out_open.shape == (1, 10)
    np.testing.assert_array_equal(out_open[0, :2], [-1, -1])
    np.testing.assert_array_equal(out_open[0, 2:], [1] * 8)
    out_closed = common.decode_gripper(pclose, np.array([1]), .6)
    np.testing.assert_array_equal(out_closed[0], [-1] * 10)


def test_t0_map_is_monotone_and_clipped():
    mapper = common.T0Map(c=2, lo=.05, hi=.6)
    got = mapper(np.array([0., .01, .04, 1.], np.float32))
    np.testing.assert_allclose(got, [.05, .2, .4, .6], atol=1e-6)
    assert np.all(np.diff(got) >= 0)


def test_mcnemar_exact_counts_discordant_pairs():
    result = common.mcnemar_exact([1, 0, 1, 0], [0, 1, 1, 0])
    assert result == {"b_only": 1, "a_only": 1, "p": 1.0}
    assert common.mcnemar_exact([1, 1], [1, 1])["p"] == 1.0
