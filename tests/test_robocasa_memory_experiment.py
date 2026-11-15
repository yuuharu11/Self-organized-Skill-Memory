from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"
sys.path.insert(0, str(MOD))
import experiment as ex
import units


def test_task_episode_split_is_deterministic_and_disjoint():
    ids = np.arange(100)
    a, b = ex.split_episode_ids(ids, seed=77)
    aa, bb = ex.split_episode_ids(ids, seed=77)
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)
    assert len(a) == 80 and len(b) == 20
    assert not np.intersect1d(a, b).size
    np.testing.assert_array_equal(np.sort(np.r_[a, b]), ids)


def test_prepare_resume_reenters_pool_build_when_manifest_exists(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"tasks": ["task-a"], "demo_limit": 4}))
    called = []
    monkeypatch.setattr(ex, "build_pools", lambda path: called.append(Path(path)))
    result = ex.prepare(out, tasks=("task-a",), demo_limit=4, smoke=True)
    assert result["tasks"] == ["task-a"]
    assert called == [out]


def test_type_error_cache_is_published_complete_and_shape_checked(tmp_path):
    out = tmp_path / "run"
    (out / "queries").mkdir(parents=True)
    (out / "pools").mkdir()
    qshape = np.zeros((2, 10, 6), np.float32); qshape[1, :, 0] = np.linspace(0, 1, 10)
    pshape = np.zeros((3, 10, 6), np.float32); pshape[1, :, 0] = np.linspace(0, 1, 10)
    np.savez(out / "queries/U_heldout.npz", shape=qshape)
    np.savez(out / "pools/U_memory.npz", shape=pshape)
    err = ex._type_error_file(out, "U", "heldout")
    assert err.shape == (2, 3)
    assert np.isfinite(err).all()
    assert (out / "cache/errors_U_heldout.npy").exists()
    assert not (out / "cache/errors_U_heldout.tmp.npy").exists()
    np.save(out / "cache/errors_U_heldout.npy", np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="stale/corrupt type-error cache"):
        ex._type_error_file(out, "U", "heldout")


def test_report_picks_up_separately_resumed_pi05_baseline(tmp_path):
    payload = {"available": True, "rows": 120, "mse6_5": 0.04, "mse6_50": 0.2}
    (tmp_path / "pi05_evaluation.json").write_text(json.dumps(payload))
    got = ex._report_pi05_metrics(tmp_path, {"metrics": {"pi05": {"available": False}}})
    assert got == payload


def test_action_and_state_channel_reordering_matches_fork_order_and_roundtrips():
    # Give every stored channel a distinct value so a misplaced dimension is visible.
    stored_action = np.arange(12, dtype=np.float32)
    policy_action = ex.action_storage_to_policy(stored_action)
    np.testing.assert_array_equal(policy_action, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(ex.action_policy_to_storage(policy_action), stored_action)
    stored_state = np.arange(16, dtype=np.float32)
    policy_state = ex.state_storage_to_policy(stored_state)
    np.testing.assert_array_equal(policy_state, [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15])


def test_scale_only_dct_type_keeps_direction_and_separates_gripper():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(17, 12)).astype(np.float32)
    a[:, 11] = np.where(np.arange(17) < 9, -1, 1)
    shape, key = ex.encode_shape(a)
    scaled, key_scaled = ex.encode_shape(a * 3.7)
    np.testing.assert_allclose(shape, scaled, atol=2e-6)
    np.testing.assert_allclose(key, key_scaled, atol=2e-6)
    flipped = a.copy(); flipped[:, 5:11] *= -1
    flip_shape, _ = ex.encode_shape(flipped)
    assert ex.shape_error_matrix(shape[None], flip_shape[None])[0, 0] > 1e-3
    assert key.shape == (19,) and shape.shape == (10, 6)


def test_scale_only_type_error_is_invariant_to_positive_scale():
    x = np.zeros((1, 10, 6), np.float32)
    x[0, :, 0] = np.linspace(0, 1, 10)
    y = x * 4.0
    e = ex.shape_error_matrix(x, np.stack([x[0], y[0]]))
    assert e.shape == (1, 2)
    assert e[0, 0] < 1e-8 and e[0, 1] < 1e-8


def test_segmentation_keeps_all_gripper_boundaries_and_minimum():
    a = np.zeros((51, 12), np.float32)
    a[:, 5] = np.r_[np.linspace(0, 1, 25), np.linspace(1, 0, 26)]
    a[30:, 11] = 1
    seg = units.cut(a, min_length=10)
    cuts = {int(s["end"]) for s in seg[:-1]}
    assert 30 in cuts
    assert all(int(s["length"]) > 0 for s in seg)
    assert sum(int(s["length"]) for s in seg) == len(a)


def test_pool_actions_and_offsets_preserve_candidate_alignment():
    acts = [np.full((3, 12), 1, np.float32), np.full((5, 12), 2, np.float32)]
    records = []
    for i, a in enumerate(acts):
        shape, key = ex.encode_shape(a)
        records.append({"action": a, "action10": ex.resample_rows(a), "shape": shape, "key": key,
            "length": len(a), "task": 0, "episode": i, "global_episode": i, "frame": 0})
    pool = ex._records_to_pool(records)
    np.testing.assert_array_equal(pool["actions_flat"][pool["action_offsets"][0]:pool["action_offsets"][1]], acts[0])
    np.testing.assert_array_equal(pool["actions_flat"][pool["action_offsets"][1]:pool["action_offsets"][2]], acts[1])
    assert len(pool["key"]) == len(pool["action10"]) == 2


def test_candidate_mask_removes_same_episode_and_keeps_other_episodes():
    allowed = ex._allowed(np.array([10, 11]), np.array([10, 12, 11]))
    np.testing.assert_array_equal(allowed, [[False, True, True], [True, True, False]])


def test_autoregressive_unit_chaining_returns_exactly_fifty_steps():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(8)

    class FirstCandidate(torch.nn.Module):
        def forward(self, x, keys):
            out = torch.zeros((len(x), len(keys)), device=x.device)
            out[:, 0] = 1
            return out

    raw = np.zeros((7, 12), np.float32)
    raw[:, 5] = .02
    shape, key = ex.encode_shape(raw)
    pool = {"key": key[None], "actions_flat": raw.copy(), "action_offsets": np.array([0, 7]),
            "action10": ex.resample_rows(raw)[None]}
    ck = {"norm": {"action_mean": np.zeros(12, np.float32), "action_std": np.ones(12, np.float32),
                    "state_mean": np.zeros(16, np.float32), "state_std": np.ones(16, np.float32)}}
    pred, selected = ex._generate_50(FirstCandidate(), ck, pool, np.zeros((10, 12), np.float32),
                                      np.zeros(16, np.float32), device="cpu")
    assert pred.shape == (50, 12)
    assert len(selected) == 8  # ceil(50/7)
    np.testing.assert_allclose(pred[:, 5], .02)
