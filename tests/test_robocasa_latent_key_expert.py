from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
import latent_expert as le  # noqa: E402
import latent_key as lk  # noqa: E402


def _batch(n, seq=12, with_tokens=True):
    b = {"history": torch.randn(n, 10, 7), "history_mask": torch.ones(n, 10, dtype=torch.bool),
         "state": torch.randn(n, 16)}
    if with_tokens:
        b["tokens"] = torch.randn(n, seq, 2048, dtype=torch.float16)
        g = torch.zeros(n, seq, dtype=torch.long); g[:, -3:] = -1; g[:, 4:8] = 3
        b["groups"] = g
        b["pooled"] = torch.randn(n, 2048)
    return b


@pytest.mark.parametrize("mode", lk.MODES)
def test_models_return_normalized_token_sets(mode):
    m = lk.new_model(mode)
    b = _batch(3, with_tokens=(mode != "e_h"))
    q = lk.encode(m, b, key=False); k = lk.encode(m, b, key=True)
    assert q.shape == k.shape == (3, lk.TOKEN_COUNT, lk.WIDTH)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(3, lk.TOKEN_COUNT), atol=1e-4, rtol=0)
    assert lk.token_scores(q, k).shape == (3, 3)


def test_e_h_ignores_vlm_tokens():
    torch.manual_seed(0); m = lk.new_model("e_h").eval()
    b = _batch(2)
    b2 = dict(b); b2["tokens"] = torch.randn_like(b["tokens"])
    torch.testing.assert_close(m(b), m(b2))


def test_soft_listwise_prefers_low_error_and_respects_mask():
    err = torch.tensor([[0., 1., 1.]]); allowed = torch.tensor([[True, True, True]])
    good = lk.soft_listwise_loss(torch.tensor([[5., 0., 0.]]), err, allowed, .01)
    bad = lk.soft_listwise_loss(torch.tensor([[0., 5., 0.]]), err, allowed, .01)
    assert good < bad
    masked = torch.tensor([[False, True, True]])
    # The masked best candidate cannot be the teacher target.
    assert lk.soft_listwise_loss(torch.tensor([[0., 5., 0.]]), err, masked, .01) < bad


def test_history_corruption_schedule_shapes_and_zero_rows():
    rng = np.random.default_rng(0)
    h = np.ones((2000, 10, 7), np.float32); mk = np.ones((2000, 10), bool)
    h2, m2 = lk.history_corrupt(h, mk, rng)
    zero = ~m2.any(1)
    # 25% zeroed, plus other-row swaps that copy a zeroed row (~.25*.25).
    assert .27 < zero.mean() < .36 and not h2[zero].any()
    assert h.min() == 1  # input untouched


def test_spread_ranks_are_log_uniform_and_bounded():
    r = le.spread_ranks(50000)
    assert r.min() == 0 and r.max() < le.MAX_RANK
    assert .1 < np.mean(r == 0) < .2 and np.median(r) < 40


def test_t0_map_is_monotone_and_in_range():
    m = le.T0Map(np.linspace(0, 1, 1000) ** 2)
    t = m(np.array([-1., 0., .25, 1., 5.]))
    assert np.all(np.diff(t) >= 0)
    np.testing.assert_allclose([t[0], t[-1]], [le.T0_MIN, le.T0_MAX], rtol=1e-6)


def test_chain_key_fills_horizon_and_excludes_own_episode():
    L = np.array([7, 12, 30])
    nat = np.zeros((3, 30, 7), np.float32)
    for j in range(3):
        nat[j, :L[j]] = j + 1
    bank = {"task": np.array([0, 0, 1]), "episode": np.array([5, 6, 7]), "length": L, "natural_actions": nat}
    rows = {"task": np.array([0]), "episode": np.array([5]), "history": np.zeros((1, 10, 7), np.float32),
            "history_mask": np.zeros((1, 10), bool)}
    calls = []

    def scorer(ix, hist, hmask):
        calls.append(hist.copy())
        return np.tile(np.array([[9., 1., 0.]]), (len(ix), 1))   # unit 0 is best but same-episode
    d, first, margin = le.chain_key(scorer, bank, rows)
    assert d.shape == (1, le.HORIZON, 7) and first[0] == 1 and margin[0] == 1.
    assert (d[0] == 2).all()                                    # unit 1 repeated, unit 0 never used
    assert (calls[1][0] == 2).all()                             # history updated with own draft


def test_to_model_draft_pads_to_32():
    d = le.to_model_draft(np.ones((50, 7), np.float32))
    assert d.shape == (50, 32) and d[:, :7].all() and not d[:, 7:].any()


def test_future_past_episode_end_holds_last_action(tmp_path):
    e = le.Episodes.__new__(le.Episodes)
    e._c = {(0, 0): np.arange(5 * 12, dtype=np.float32).reshape(5, 12)}
    out, mask = e.future(0, 0, 3, 4)
    np.testing.assert_array_equal(out[:2], e._c[(0, 0)][3:5]); np.testing.assert_array_equal(out[2:], e._c[(0, 0)][[4, 4]])
    assert mask.tolist() == [True, True, False, False]
    out, mask = e.future(0, 0, 9, 3)
    assert out.shape == (3, 12) and not mask.any()
    np.testing.assert_array_equal(out, e._c[(0, 0)][[4, 4, 4]])
