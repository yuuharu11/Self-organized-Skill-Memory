from __future__ import annotations

import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.methods.m5 import M5


def _train(seed: int = 19, n: int = 40):
    rng = np.random.default_rng(seed)
    return {
        "actions": rng.normal(size=(n, 5, 2)).astype(np.float32),
        "state": rng.normal(size=(n, 3)).astype(np.float32),
        "features": rng.normal(size=(n, 6)).astype(np.float32),
    }


def test_m5_fit_encode_decode_prior_and_memory_query():
    train = _train()
    method = M5(n_atoms=8, top_k=2, epochs=2, batch_size=16, width=24)
    method.fit(train)

    b, r = method.encode(train["actions"][:4], train["state"][:4])
    alpha = b["alpha"]
    assert alpha.shape == (4, 8)
    assert np.all((alpha > 0).sum(-1) <= 2)
    assert r.shape == (4, 4)
    decoded = method.decode(b, r, train["features"][:4], train["state"][:4])
    assert decoded.shape == train["actions"][:4].shape
    assert np.isfinite(decoded).all()

    b_dist, r_prior = method.prior(train["features"][:4], train["state"][:4])
    assert b_dist["alpha"].shape == (4, 8)
    assert np.all((b_dist["alpha"] > 0).sum(-1) <= 2)
    assert r_prior.shape == (4, 4)
    prior_decoded = method.decode(b_dist, r_prior, train["features"][:4], train["state"][:4])
    assert prior_decoded.shape == train["actions"][:4].shape

    key = np.array([0.5, 1.0, -2.0], dtype=np.float32)
    result = method.memory_add(key, {name: value[:1] for name, value in b.items()}, {"task": "goal"})
    assert result["added"]
    hit = method.memory_query(key, top_k=1)
    assert hit["alphas"].shape == (1, 8)
    np.testing.assert_allclose(hit["weights"], [1.0], atol=1e-6)
    description = method.describe()
    assert "share atoms" in description["composition"]
    assert description["initial_atom_sweep"] == [8, 16, 32]
    assert description["sparsity_sweep_top_k"] == [2, 4, 6]


def test_m5_dictionary_coherence_penalty_and_residual_atom_growth_are_non_gradient():
    method = M5(n_atoms=6, top_k=2, epochs=1, batch_size=24, width=16,
                novelty_threshold=0.2, novelty_batch=3, max_atoms=10)
    method.fit(_train(n=24))
    coherence = method._coherence(method.net.dictionary)
    assert torch.isfinite(coherence)
    n_before = method.net.dictionary.shape[0]
    alpha = np.zeros((1, n_before), np.float32)
    alpha[0, :2] = 0.5
    innovation = np.zeros((1, method.base_dim), np.float32)
    innovation[0, 0] = 0.8
    dictionary_before = method.net.dictionary.detach().cpu().numpy().copy()

    for i in range(3):
        out = method.memory_add(np.array([float(i), 1.0]),
                                {"alpha": alpha, "innovation": innovation},
                                {"reconstruction_error": 0.8})
    assert out["added_atom"] is True
    assert method.net.dictionary.shape[0] == n_before + 1
    # Adding an atom appends a PCA-derived vector; existing dictionary rows do
    # not receive optimizer updates during memory_add.
    np.testing.assert_array_equal(method.net.dictionary.detach().cpu().numpy()[:n_before], dictionary_before)
    assert "first principal component" in method.describe()["atom_growth"]["initializer"]

    hit = method.memory_query(np.array([2.0, 1.0]), top_k=2)
    assert hit["alphas"].shape == (2, n_before + 1)
    assert np.isfinite(hit["weights"]).all()

    stronger = M5(n_atoms=6, top_k=2, epochs=1, batch_size=24, width=16,
                  coherence_weight=0.2, novelty_threshold=0.2, novelty_batch=3, max_atoms=10)
    stronger.fit(_train(n=24))
    report = stronger.describe()
    assert 0.2 in report["mutual_coherence_weight_sweep"]
    assert report["mutual_coherence_penalty"] == 0.2
    assert np.isfinite(report["dictionary_coherence"]["mean_squared_mutual_coherence"])
    assert np.isfinite(report["dictionary_coherence"]["max_abs_mutual_coherence"])


def test_m5_fixed_dictionary_variant_stores_memory_without_growing_atoms():
    method = M5(n_atoms=6, top_k=2, epochs=1, batch_size=24, width=16,
                variant="fixed", novelty_threshold=0.1, novelty_batch=1, max_atoms=10)
    method.fit(_train(n=24))
    before = method.net.dictionary.detach().cpu().numpy().copy()
    alpha = np.zeros((1, 6), np.float32)
    alpha[0, :2] = 0.5
    innovation = np.zeros((1, method.base_dim), np.float32)
    innovation[0, 0] = 1.0
    result = method.memory_add(np.array([0.0, 1.0]), {"alpha": alpha, "innovation": innovation})

    assert result["added"] is True
    assert result["added_atom"] is False
    assert result["reason"] == "fixed_dictionary_novelty_not_added"
    assert method.net.dictionary.shape[0] == 6
    np.testing.assert_array_equal(method.net.dictionary.detach().cpu().numpy(), before)
    assert method.describe()["atom_growth"]["enabled"] is False
    assert method.describe()["variant"] == "fixed"


def test_m5_decoder_uses_sparse_base_and_ignores_observation_by_default():
    rng = np.random.default_rng(51)
    code = np.arange(80) % 4
    actions = np.eye(4, dtype=np.float32)[code, None, :].repeat(3, axis=1)
    train = {"actions": actions, "state": np.zeros((80, 0), np.float32),
             "features": rng.normal(size=(80, 5)).astype(np.float32)}
    method = M5(n_atoms=8, top_k=2, base_dim=8, residual_dim=2, epochs=24, batch_size=32,
                width=32, residual_beta=0.1)
    method.fit(train)
    b, r = method.encode(actions[:20], train["state"][:20])
    pred = method.decode(b, r, train["features"][:20], train["state"][:20])
    zero = dict(b, alpha=np.zeros_like(b["alpha"]))
    zero_pred = method.decode(zero, r, train["features"][:20], train["state"][:20])
    assert np.mean((zero_pred - actions[:20]) ** 2) > np.mean((pred - actions[:20]) ** 2) + 0.02
    changed_f = method.decode(b, r, np.full((20, 5), -100.0, np.float32), train["state"][:20])
    np.testing.assert_array_equal(changed_f, pred)
    assert method.describe()["residual_kl"]["beta"] == 0.1


def test_m5_growth_threshold_is_train_calibrated_and_fixed_differs():
    train = _train(n=64)
    common = dict(n_atoms=8, top_k=2, epochs=2, batch_size=32, width=24,
                  novelty_batch=1, max_atoms=12, novelty_quantile=0.90)
    growing, fixed = M5(**common, variant="growing"), M5(**common, variant="fixed")
    growing.fit(train)
    fixed.fit(train)
    assert growing.novelty_threshold is not None and growing.novelty_threshold > 0
    bases, _ = growing.encode(train["actions"], train["state"])
    fixed_bases, _ = fixed.encode(train["actions"], train["state"])
    outcomes = [growing.memory_add(np.array([float(i), 1.0]),
                                   {name: value[i:i+1] for name, value in bases.items()})
                for i in range(len(train["actions"]))]
    rate = np.mean([item["added_atom"] for item in outcomes])
    assert 0.0 < rate < 1.0
    fixed_before = fixed.net.dictionary.detach().cpu().numpy().copy()
    fixed_outcomes = [fixed.memory_add(np.array([float(i), 1.0]),
                                       {name: value[i:i+1] for name, value in fixed_bases.items()})
                      for i in range(len(train["actions"]))]
    assert not any(item["added_atom"] for item in fixed_outcomes)
    np.testing.assert_array_equal(fixed.net.dictionary.detach().cpu().numpy(), fixed_before)
    assert growing.net.dictionary.shape[0] > fixed.net.dictionary.shape[0]
