from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


features = load_module("latent_probe_features_test", BASE / "latent_probe_features.py")
readout = load_module("latent_probe_readout_test", BASE / "latent_probe_readout.py")


def test_vjepa_token_pool_and_frozen_encoder():
    x = np.arange(2 * 256 * 3, dtype=np.float32).reshape(2, 256, 3)
    pooled = features.pool_2x2_tokens(x)
    assert pooled.shape == (2, 64, 3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout())
    model.train()
    features.freeze_encoder(model)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_worker_resize_is_pixel_exact_to_explicit_pil_bicubic():
    from PIL import Image
    rng = np.random.default_rng(41)
    image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    explicit = features.resize_pil_bicubic(image, 128)
    implicit = np.asarray(Image.fromarray(image).resize((128, 128)))
    assert explicit.shape == (128, 128, 3)
    assert np.array_equal(explicit, implicit)
    stat = features.preprocessor_tensor_stats(np.zeros((2, 3, 16, 16), np.float32))
    assert stat["mean"] == 0.0 and stat["variance"] == 0.0
    assert stat["high_frequency_energy_mean"] == 0.0


def test_feature_loader_requires_exact_key_order(tmp_path):
    split = tmp_path / "dev"
    split.mkdir()
    keys = np.array([[1, 2, 0], [1, 2, 5]], dtype=np.int32)
    np.save(split / "row_keys.npy", keys)
    np.save(split / "base.npy", np.zeros((2, 64, 8), np.float16))
    np.save(split / "wrist.npy", np.ones((2, 64, 8), np.float16))
    (split / "manifest.json").write_text('{"complete":true,"row_keys":"row_keys.npy",'
        '"camera_outputs":{"base":"base.npy","wrist":"wrist.npy"}}')
    loaded = features.load_keyed_features(tmp_path, "dev", keys)
    assert loaded["base"].shape == (2, 64, 8)
    with pytest.raises(ValueError, match="exactly match"):
        features.load_keyed_features(tmp_path, "dev", keys[::-1])


def test_model_shapes_for_proprio_on_and_off():
    for use_prop in (False, True):
        cfg = readout.ReadoutConfig(latent_dim=12, key_dim=5, use_proprio=use_prop, dropout=0)
        model = readout.LatentProbeReadout(cfg)
        latent = torch.randn(3, 17, 12)
        prop = torch.randn(3, 78) if use_prop else None
        keys = torch.randn(7, 5).expand(3, -1, -1).clone()
        out = model(latent, prop, keys)
        assert out["gripper_logits"].shape == (3, 10)
        assert out["switch_logit"].shape == (3,)
        assert out["switch_time_logits"].shape == (3, 10)
        assert out["log_rms6"].shape == (3, 6)
        assert out["candidate_logits"].shape == (3, 7)
        model.eval()
        out = model(latent, prop, keys)
        shared = model(latent, prop, keys[0])
        assert torch.allclose(out["candidate_logits"], shared["candidate_logits"])


def test_repaired_condition_set_and_fixed_gripper_dev_tasks():
    assert readout.CONDITIONS == (("L-none", "P-on"), ("L-vjepa", "P-off"),
                                  ("L-vjepa", "P-on"), ("L-vlm", "P-off"),
                                  ("L-vlm", "P-on"))
    tasks = np.setdiff1d(np.arange(40)[::-1], [2, 7, 12, 17, 22, 27, 32, 37])
    split = readout.gripper_dev_task_split(tasks)
    assert split["gripper_dev_tasks"] == [0, 10, 20, 30]
    assert set(split["gripper_dev_tasks"]).isdisjoint(split["train_tasks"])
    with pytest.raises(ValueError, match="at least"):
        readout.gripper_dev_task_split(np.arange(20))


def test_candidate_mask_is_excluded_from_listwise_loss():
    logits = torch.tensor([[0.0, 10.0, 0.0]], requires_grad=True)
    batch = {"gripper_targets": torch.zeros(1, 10), "switch_target": torch.zeros(1),
             "type_target": torch.tensor([[1.0, 0.0, 0.0]]),
             "candidate_allowed": torch.tensor([[True, False, True]])}
    out = {"gripper_logits": torch.zeros(1, 10, requires_grad=True),
           "switch_logit": torch.zeros(1, requires_grad=True),
           "switch_time_logits": torch.zeros(1, 10, requires_grad=True),
           "log_rms6": torch.zeros(1, 6, requires_grad=True),
           "query_embedding": torch.zeros(1, 4), "candidate_logits": logits}
    loss, _ = readout.multitask_loss(out, batch)
    loss.backward()
    assert logits.grad[0, 1] == 0
    with pytest.raises(ValueError, match="at least one allowed"):
        readout.multitask_loss(out, {**batch, "candidate_allowed": torch.zeros(1, 3, dtype=torch.bool)})


def test_collapse_detector_rejects_constant_and_accepts_variable():
    with pytest.raises(AssertionError, match="collapsed"):
        readout.check_prediction_variance(np.ones((20, 10), np.float32))
    audit = readout.check_prediction_variance(np.arange(80, dtype=np.float32).reshape(20, 4))
    assert not audit["collapsed"] and audit["max_std"] > 0


def test_key_join_is_exact_and_rejects_missing_rows():
    source = np.array([[1, 2, 0], [1, 2, 5], [3, 4, 0]])
    feats = np.array([[3, 4, 0], [1, 2, 0]])
    assert readout.join_keyed_rows(source, feats).tolist() == [2, 0]
    with pytest.raises(KeyError, match="feature keys absent"):
        readout.join_keyed_rows(source, np.array([[8, 8, 8]]))
    with pytest.raises(ValueError, match="duplicate"):
        readout.join_keyed_rows(np.vstack([source, source[:1]]), feats)


def test_supervision_history_is_causal_zero_padded(tmp_path):
    root = tmp_path / "split"
    root.mkdir()
    (root / "manifest.json").write_text('{"complete": true}')
    def save(name, value):
        np.save(root / f"{name}.npy", value, allow_pickle=False)
    save("task", np.array([2, 2], np.int32))
    save("episode", np.array([7, 7], np.int32))
    save("frame", np.array([0, 5], np.int32))
    save("trajectory_task", np.array([2], np.int32))
    save("trajectory_episode", np.array([7], np.int32))
    save("trajectory_offsets", np.array([[0, 12]], np.int64))
    trace = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    save("trajectory_actions_normalized", trace)
    save("state_normalized", np.zeros((2, 8), np.float32))
    save("actions_normalized", np.zeros((2, 10, 7), np.float32))
    save("actions_physical", np.zeros((2, 10, 7), np.float32))
    save("gripper_signs", np.zeros((2, 10), np.int8))
    save("first_gripper_switch", np.array([-1, 2], np.int8))
    save("rms6_normalized", np.ones((2, 6), np.float32))
    save("rms6_physical", np.ones((2, 6), np.float32))
    result = readout.load_supervision_rows(tmp_path, "split")
    assert result["history"].shape == (2, 10, 7)
    assert np.count_nonzero(result["history"][0]) == 0
    assert np.array_equal(result["history"][1, -5:], trace[:5])
    assert not result["history_valid"][1, :5].any()
    assert result["history_valid"][1, 5:].all()


def test_history_corruptions_preserve_state_and_change_only_history():
    x = torch.arange(2 * 78, dtype=torch.float32).reshape(2, 78)
    zeroed = readout.corrupt_history(x, "zero")
    assert torch.equal(zeroed[:, :8], x[:, :8])
    assert torch.count_nonzero(zeroed[:, 8:]) == 0
    donor = torch.full((2, 10, 7), -2.0)
    swapped = readout.corrupt_history(x, "swap", donor_history=donor)
    assert torch.equal(swapped[:, :8], x[:, :8])
    assert torch.all(swapped[:, 8:] == -2)
    with pytest.raises(ValueError, match="donor histories"):
        readout.corrupt_history(x, "swap")


def test_vlm_key_join_excludes_missing_without_substitution(tmp_path):
    tokens = np.arange(3 * 2 * 4, dtype=np.float16).reshape(3, 2, 4)
    groups = np.zeros((3, 2), dtype=np.int8)
    np.save(tmp_path / "tokens.npy", tokens)
    np.save(tmp_path / "groups.npy", groups)
    source = np.array([[0, 1, 0], [0, 1, 10], [1, 2, 0]])
    targets = np.array([[0, 1, 10], [4, 5, 0], [0, 1, 0]])
    result = readout.load_keyed_vlm_cache(tmp_path / "tokens.npy", tmp_path / "groups.npy", source, targets)
    assert result["excluded_missing_keys"] == 1
    assert result["valid_feature_indices"].tolist() == [0, 2]
    assert np.array_equal(result["tokens"], tokens[[1, 0]])


def test_soft_type_target_mask_and_retrieval_orientation():
    errors = torch.tensor([[0.0, 0.2, 1.0], [0.2, 0.1, 0.0]])
    allowed = torch.tensor([[True, False, True], [True, True, True]])
    target = readout.soft_type_distribution(errors, allowed, tau=0.1)
    assert target[0, 1] == 0
    assert torch.allclose(target.sum(-1), torch.ones(2))
    scores = np.array([[3., 999., 0.], [0., 1., 2.]])
    metrics = readout.retrieval_metrics(scores, errors.numpy(), allowed.numpy(), tau=.15, ks=(1, 2))
    assert metrics["top1_type_error_mean"] == pytest.approx(0.0)
    assert metrics["spearman_score_vs_negative_type_error_mean"] > 0
    assert metrics["recall"]["1"] == pytest.approx(1.0)


def test_episode_bootstrap_resamples_paired_episodes():
    a = np.array([2., 4., 8., 10.])
    b = np.array([1., 1., 2., 2.])
    result = readout.episode_bootstrap_delta(a, b, np.array([1, 1, 2, 2]), samples=100, seed=3)
    assert result["mean_delta"] == pytest.approx(4.5)
    assert result["ci95_low"] <= 4.5 <= result["ci95_high"]


def test_training_smoke_with_dev_selection(tmp_path):
    cfg = readout.ReadoutConfig(latent_dim=8, key_dim=4, use_proprio=True, dropout=0)
    model = readout.LatentProbeReadout(cfg)
    def batch():
        return {"latent": torch.randn(4, 5, 8), "proprio": torch.randn(4, 78),
                "candidate_keys": torch.randn(4, 3, 4), "candidate_allowed": torch.ones(4, 3, dtype=torch.bool),
                "type_target": torch.full((4, 3), 1/3), "gripper_targets": torch.randint(0, 2, (4, 10)).float(),
                "switch_target": torch.tensor([0., 1., 0., 1.]), "switch_step": torch.tensor([-1, 3, -1, 7]),
                "rms6_target": torch.randn(4, 6)}
    result = readout.run_training(model, [batch()], [batch()], updates=2, eval_every=1,
                                  patience=2, device="cpu", checkpoint_path=tmp_path / "best.pt")
    assert result["best_step"] in (1, 2)
    assert (tmp_path / "best.pt").exists()
    assert np.isfinite(result["best_dev"])


def test_switch_metrics_report_no_switch_false_alarm_and_auc():
    y = np.zeros((4, 10), dtype=bool)
    y[0, 4:] = True
    y[1, 6:] = True
    sy = np.array([1, 1, 0, 0], dtype=bool)
    slogit = np.array([3, 2, -2, -3], dtype=np.float32)
    glogit = np.where(y, 3.0, -3.0)
    result = readout.gripper_metrics(glogit, slogit, y, sy)
    assert result["switch_auc"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_switch_rate_no_switch"] == 0.0
    assert result["switch_time_abs_error_mean"] == 0.0

