import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.intermediate import anchor_flow as af
from skill_moe.skillnet.examples.emergent.light.intermediate import roundtrip as rt

SCALES = {"pos_m_per_action_unit_step": [0.01, 0.01, 0.01],
          "rot_rad_per_action_unit_step": [0.1, 0.1, 0.1]}


def _target(n=3, h=5):
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (n, h, 7)).astype(np.float32)
    a[..., 6] = np.where(rng.random((n, h)) > 0.5, 1.0, -1.0)
    return a


def test_identical_prediction_has_zero_error_and_full_gripper_agreement():
    target = _target()
    m = rt.action_metrics(np.stack([target, target]), target, SCALES)
    for key in ("mse", "best_of_8_mse", "pos_chunk_m", "pos_terminal_m", "rot_chunk_rad", "rot_terminal_rad"):
        assert np.allclose(m[key], 0.0)
    assert np.allclose(m["gripper_agreement"], 1.0)


def test_position_error_accumulates_deltas_in_metres():
    target = _target(n=1, h=5)
    pred = target.copy()
    pred[..., 0] += 0.2          # constant x offset of 0.2 action units per step
    m = rt.action_metrics(pred[None], target, SCALES)
    assert np.isclose(m["pos_terminal_m"][0], 5 * 0.2 * 0.01)
    assert np.isclose(m["pos_chunk_m"][0], np.mean(np.arange(1, 6) * 0.2 * 0.01))
    assert np.allclose(m["rot_terminal_rad"], 0.0)


def test_gripper_sign_flip_and_nan_rows():
    target = _target(n=2, h=4)
    pred = target.copy()
    pred[0, :2, 6] *= -1
    pred[1] = np.nan
    m = rt.action_metrics(pred[None], target, SCALES)
    assert np.isclose(m["gripper_agreement"][0], 0.5)
    assert all(np.isnan(m[k][1]) for k in m)


def test_raw_nearest_returns_self_in_condition_a_and_other_episode_in_b():
    keys = np.asarray([[1, 0], [0.99, 0.1], [0, 1]], np.float32)
    codes = np.asarray([0, 1, 2])
    ix_a, _ = rt.raw_nearest(keys, keys, codes, codes, exclude_same_episode=False, device="cpu")
    ix_b, _ = rt.raw_nearest(keys, keys, codes, codes, exclude_same_episode=True, device="cpu")
    assert ix_a.tolist() == [0, 1, 2]
    assert ix_b[0] == 1 and ix_b[1] == 0 and ix_b[2] != 2


class _FakeM6:
    """Two units in base space; nearest by L2 with epsilon 1.0."""
    def __init__(self):
        self.prototypes = np.asarray([[0.0, 0.0], [10.0, 0.0]], np.float32)
        self.epsilon = 1.0

    def _nearest(self, base):
        d = np.sum((self.prototypes - base[None]) ** 2, axis=1)
        return int(d.argmin()), float(d.min())


def _bank_fixture():
    m6 = _FakeM6()
    bases = np.asarray([[0.1, 0], [-0.1, 0], [0.2, 0], [10.1, 0], [9.9, 0], [10.3, 0], [5.0, 0]], np.float32)
    keys = np.asarray([[1, 0.1], [1, 0.2], [1, 0.0], [0.1, 1], [0.2, 1], [0.0, 1], [0.7, 0.7]], np.float32)
    episodes = np.asarray(["e0", "e1", "e2", "e0", "e1", "e2", "e0"])
    bank = af.UnitBank(m6, 1.0)
    own, dist, assigned, unit_rows = rt.seed_bank_with_rows(bank, m6, bases, keys, episodes)
    return m6, bank, bases, keys, episodes, own, assigned, unit_rows


def test_seed_bank_matches_unitbank_seed_training():
    m6, bank, bases, keys, episodes, own, assigned, unit_rows = _bank_fixture()
    reference = af.UnitBank(m6, 1.0)
    reference.seed_training(bases, keys, episodes)
    assert {k: len(v) for k, v in reference.members.items()} == {k: len(v) for k, v in bank.members.items()}
    assert assigned.tolist() == [True] * 6 + [False]
    assert own[6] in (0, 1)       # unassigned rows still get the nearest unit
    assert unit_rows[0].tolist() == [0, 1, 2] and unit_rows[1].tolist() == [3, 4, 5]


def test_retrieve_pure_round_trip_and_leave_episode_out():
    m6, bank, bases, keys, episodes, own, assigned, unit_rows = _bank_fixture()
    codes = rt.encode_episode_codes(episodes)
    q = slice(0, 6)
    a = rt.retrieve(bank, keys[q], episodes[q], own[q], unit_rows, codes, 0.05,
                    exclude_same_episode=False, rng=np.random.default_rng(0), samples=4)
    assert (a["top1"] == own[q]).all()
    assert a["instance_row"].tolist() == list(range(6))       # returns itself
    assert np.allclose(a["own_center"], m6.prototypes[own[q]])
    b = rt.retrieve(bank, keys[q], episodes[q], own[q], unit_rows, codes, 0.05,
                    exclude_same_episode=True, rng=np.random.default_rng(0), samples=4)
    assert (b["top1"] == own[q]).all()
    assert all(episodes[b["instance_row"][i]] != episodes[i] for i in range(6))
    # leave-episode-out center is the mean of the other episodes' members
    assert np.allclose(b["own_center"][0], bases[[1, 2]].mean(axis=0))
    assert b["comp_center"].shape == (4, 6, 2)


def test_normal_flow_never_reads_anchor_sources():
    rng = np.random.default_rng(0)
    n, h = 40, 2
    rows = {"f": rng.standard_normal((n, 3)).astype(np.float32),
            "s": rng.standard_normal((n, 2)).astype(np.float32),
            "episode_id": np.asarray([f"e{i // 4}" for i in range(n)]),
            "a": rng.standard_normal((n, h, 7)).astype(np.float32)}
    nan_sources = tuple(np.full((n, h * 7), np.nan, np.float32) for _ in range(2))
    sources = (nan_sources[0], None, nan_sources[1], None)
    model, norm, fit = af.fit_anchor_flow(rows, rows["a"], sources, source_mix=rt.NORMAL_MIX,
                                          seed=0, epochs=2, patience=2, batch_size=16, device="cpu")
    assert np.isfinite(fit["best_validation_loss"])
    assert all(np.isfinite(p.detach()).all() for p in model.parameters())


def test_t0_one_denoise_ignores_anchor():
    class Flow(torch.nn.Module):
        def forward(self, x, t, f, s):
            return 0.1 * x + t[:, None]
    n, h = 3, 2
    rows = {"a": np.zeros((n, h, 7), np.float32), "f": np.zeros((n, 1), np.float32),
            "s": np.zeros((n, 1), np.float32)}
    norm = {"a_mean": np.zeros(h * 7, np.float32), "a_scale": np.ones(h * 7, np.float32),
            "f_mean": np.zeros(1, np.float32), "f_scale": np.ones(1, np.float32),
            "s_mean": np.zeros(1, np.float32), "s_scale": np.ones(1, np.float32),
            "action_shape": (h, 7)}
    noise = np.random.default_rng(0).standard_normal((2, n, h * 7)).astype(np.float32)
    anchors = np.full((2, n, h, 7), 5.0, np.float32)
    with_anchor = rt.denoise(Flow(), norm, rows, anchors, noise, 1.0, device="cpu")
    from_noise = rt.denoise(Flow(), norm, rows, None, noise, 1.0, device="cpu")
    assert np.allclose(with_anchor, from_noise)
