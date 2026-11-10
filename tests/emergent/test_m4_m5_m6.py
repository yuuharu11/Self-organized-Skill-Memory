"""Emergent-skill methods M4, M5 and M6."""
from __future__ import annotations

import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.methods.m4 import M4
from skill_moe.skillnet.examples.emergent.light.methods.m5 import M5
from skill_moe.skillnet.examples.emergent.light.methods.m6 import M6, build_variants

# ── M4 ────────────────────────────────────────────────────────────────────────────────────────────

def _train_m4(seed: int = 7, n: int = 40):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 3)).astype(np.float32)
    feature = rng.normal(size=(n, 6)).astype(np.float32)
    actions = rng.normal(size=(n, 5, 2)).astype(np.float32)
    return {"actions": actions, "state": state, "features": feature}


def test_m4_fit_encode_decode_prior_and_memory_query():
    train = _train_m4()
    method = M4(epochs=2, batch_size=16, width=24, dpmeans_lambda=0.75, vigilance=0.85)
    method.fit(train)

    b, r = method.encode(train["actions"][:4], train["state"][:4])
    assert b["codes"].shape == (4,)
    assert b["embedding"].shape == (4, 12)
    assert r.shape == (4, 4)
    decoded = method.decode(b, r, train["features"][:4], train["state"][:4])
    assert decoded.shape == train["actions"][:4].shape
    assert np.isfinite(decoded).all()

    b_dist, r_prior = method.prior(train["features"][:4], train["state"][:4])
    assert b_dist["probs"].shape == (4, len(method.prototypes))
    np.testing.assert_allclose(b_dist["probs"].sum(-1), 1.0, atol=1e-6)
    assert r_prior.shape == (4, 4)
    prior_decoded = method.decode(b_dist, r_prior, train["features"][:4], train["state"][:4])
    assert prior_decoded.shape == train["actions"][:4].shape

    key = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    add = method.memory_add(key, {"embedding": b["embedding"][:1]}, {"task": "object"})
    assert add["added"] is True
    hit = method.memory_query(key, top_k=1)
    assert hit["bases"].shape == (1, 12)
    np.testing.assert_allclose(hit["weights"], [1.0], atol=1e-6)
    assert "DP-means" in method.describe()["method"]


def test_m4_art_resonance_updates_or_freezes_without_gradient():
    train = _train_m4(n=24)
    results = {}
    for update in (False, True):
        method = M4(epochs=1, batch_size=24, width=16, update_prototypes=update, update_rate=0.5)
        method.fit(train)
        center = method.prototypes[0].copy()
        nearby = center.copy()
        nearby[0] += 0.02
        nearby /= np.linalg.norm(nearby)
        result = method.memory_add(np.array([0.0, 1.0]), {"embedding": nearby[None, :]})
        results[update] = (result, center, method.prototypes[0].copy())
    assert results[False][0]["reason"] == "resonance_frozen"
    np.testing.assert_array_equal(results[False][1], results[False][2])
    assert results[True][0]["reason"] == "resonance_updated"
    assert not np.array_equal(results[True][1], results[True][2])
    assert results[True][0]["prototype_movement_l2"] > 0
    assert results[True][0]["resonance_update_count"] == 1


def test_m4_vigilance_miss_creates_prototype_and_reports_three_points():
    method = M4(epochs=1, width=16, max_prototypes=16, vigilance=0.95)
    method.fit(_train_m4(n=16))
    n_before = len(method.prototypes)
    novel = -method.prototypes[0]
    result = method.memory_add(np.array([1.0, 0.0]), {"embedding": novel[None, :]})
    assert result["added_prototype"]
    assert len(method.prototypes) == n_before + 1
    assert method.describe()["vigilance_sweep"] == [0.7, 0.85, 0.95]


def test_m4_capacity_path_returns_nearest_code_without_error():
    method = M4(epochs=1, width=16, max_prototypes=32)
    method.fit(_train_m4(n=20))
    method.max_prototypes = len(method.prototypes)
    novel = -method.prototypes[0]
    result = method.memory_add(np.array([1.0, 0.0]), {"embedding": novel[None, :]})
    assert result["reason"] == "vigilance_miss_capacity_reached"
    assert 0 <= result["code"] < len(method.prototypes)


def test_m4_merge_variant_remaps_ids_and_uses_count_weighted_representatives():
    method = M4(epochs=1, width=16, variant="merge", merge_threshold=0.2,
                merge_action_mse_tolerance=1.0)
    method.fit(_train_m4(n=16))
    p = np.zeros((3, method.base_dim), np.float32)
    p[0, :2] = [1.0, 0.0]
    p[1, :2] = [0.99, 0.10]
    p[1] /= np.linalg.norm(p[1])
    p[2, :2] = [-1.0, 0.0]
    method.prototypes = p
    method.prototype_counts = [2, 3, 1]
    method._prior_prototype_count = 3
    method._prior_code_map = np.arange(3, dtype=np.int64)
    method.memory = [{"key": np.array([1.0]), "code": i, "meta": None} for i in range(3)]

    report = method.memory_merge()
    assert report["pre_merge_count"] == 3
    assert report["post_merge_count"] == 2
    assert report["old_to_new"] == [0, 0, 1]
    assert report["merged_groups"] == [[0, 1], [2]]
    assert report["pre_merge_eval_mse"] is not None
    assert report["post_merge_eval_mse"] is not None
    assert report["merge_eval_mse_delta"] is not None
    assert max(report["merged_pair_action_mse_deltas"]) <= 1.0
    assert method.prototype_counts == [5, 1]
    assert [item["code"] for item in method.memory] == [0, 0, 1]
    expected = (2 * p[0] + 3 * p[1]) / 5
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(method.prototypes[0], expected, atol=1e-6)
    description = method.describe()["merge"]
    assert description["history"][0]["old_to_new"] == [0, 0, 1]
    assert "representative values change" in description["representative"]
    assert "old IDs change" in description["representative"]


def test_m4_prior_probabilities_follow_merged_id_map():
    train = _train_m4(n=20)
    method = M4(epochs=1, width=16, variant="merge", merge_threshold=3.0)
    method.fit(train)
    method.memory_merge()
    b_dist, _ = method.prior(train["features"][:3], train["state"][:3])
    assert b_dist["probs"].shape == (3, len(method.prototypes)) == (3, 1)
    np.testing.assert_allclose(b_dist["probs"].sum(-1), 1.0, atol=1e-6)


def test_m4_variant_alias_keeps_add_only_and_adaptive_modes():
    assert M4(variant="add_only").update_prototypes is False
    assert M4(variant="adaptive").update_prototypes is True
    assert M4(update_prototypes=False).variant == "add_only"
    assert M4().variant == "adaptive"


def test_m4_decoder_uses_base_and_ignores_observation_by_default():
    rng = np.random.default_rng(37)
    prototypes = np.eye(4, dtype=np.float32)
    labels = np.arange(64) % 4
    actions = prototypes[labels, None, :].repeat(3, axis=1)
    train = {"actions": actions, "state": np.zeros((64, 0), np.float32),
             "features": rng.normal(size=(64, 5)).astype(np.float32)}
    method = M4(epochs=24, batch_size=32, width=32, base_dim=8, residual_dim=2,
                dpmeans_lambda=0.2, residual_beta=0.1)
    method.fit(train)
    b, r = method.encode(actions[:16], train["state"][:16])
    pred = method.decode(b, r, train["features"][:16], train["state"][:16])
    zero_base = method.decode(np.zeros_like(b["embedding"]), r,
                              train["features"][:16], train["state"][:16])
    mse = np.mean((pred - actions[:16]) ** 2)
    zero_mse = np.mean((zero_base - actions[:16]) ** 2)
    assert zero_mse > mse + 0.02
    # With decoder_obs=False, changing f cannot alter decoder output.
    changed_f = method.decode(b, r, np.full((16, 5), 100.0, np.float32), train["state"][:16])
    np.testing.assert_array_equal(changed_f, pred)
    assert method.describe()["residual_kl"]["beta"] == 0.1


def test_m4_base_novelty_and_merge_thresholds_are_train_calibrated_and_non_degenerate():
    train = _train_m4(n=60)
    method = M4(epochs=2, batch_size=32, width=24, variant="merge", dpmeans_lambda=0.5,
                novelty_distance_quantile=0.90, merge_distance_quantile=0.20)
    method.fit(train)
    assert method.novelty_distance_threshold is not None and method.novelty_distance_threshold > 0
    assert method.merge_threshold is not None and method.merge_threshold > 0
    bases, _ = method.encode(train["actions"], train["state"])
    outcomes = [method.memory_add(np.array([float(i), 1.0]), {"embedding": bases["embedding"][i:i+1]})
                for i in range(len(train["actions"]))]
    rate = np.mean([x["added_prototype"] for x in outcomes])
    assert 0.0 < rate < 1.0
    before = len(method.prototypes)
    merged = method.memory_merge()
    assert merged["threshold"] == method.merge_threshold
    assert merged["post_merge_count"] < before


# ── M5 ────────────────────────────────────────────────────────────────────────────────────────────

def _train_m5(seed: int = 19, n: int = 40):
    rng = np.random.default_rng(seed)
    return {
        "actions": rng.normal(size=(n, 5, 2)).astype(np.float32),
        "state": rng.normal(size=(n, 3)).astype(np.float32),
        "features": rng.normal(size=(n, 6)).astype(np.float32),
    }


def test_m5_fit_encode_decode_prior_and_memory_query():
    train = _train_m5()
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
    method.fit(_train_m5(n=24))
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
    stronger.fit(_train_m5(n=24))
    report = stronger.describe()
    assert 0.2 in report["mutual_coherence_weight_sweep"]
    assert report["mutual_coherence_penalty"] == 0.2
    assert np.isfinite(report["dictionary_coherence"]["mean_squared_mutual_coherence"])
    assert np.isfinite(report["dictionary_coherence"]["max_abs_mutual_coherence"])


def test_m5_fixed_dictionary_variant_stores_memory_without_growing_atoms():
    method = M5(n_atoms=6, top_k=2, epochs=1, batch_size=24, width=16,
                variant="fixed", novelty_threshold=0.1, novelty_batch=1, max_atoms=10)
    method.fit(_train_m5(n=24))
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
    train = _train_m5(n=64)
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


# ── M6 ────────────────────────────────────────────────────────────────────────────────────────────

def _clustered_train(n=180, seed=23):
    rng = np.random.default_rng(seed)
    label = np.arange(n) % 3
    centers = np.asarray([
        np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        np.tile([0.0, 1.0, 0.0, 0.0], (4, 1)),
        np.tile([0.0, 0.0, 1.0, 0.0], (4, 1)),
    ], dtype=np.float32)
    actions = centers[label] + rng.normal(0, 0.015, size=(n, 4, 4)).astype(np.float32)
    episode = np.asarray([f"ep-{i // 6}" for i in range(n)])
    return {
        "actions": actions,
        "state": rng.normal(0, 0.02, size=(n, 3)).astype(np.float32),
        "feature": rng.normal(size=(n, 5)).astype(np.float32),
        "key": rng.normal(size=(n, 6)).astype(np.float32),
        "episode_id": episode,
        "init_state_id": np.asarray([f"init-{i // 6}" for i in range(n)]),
        "seed": seed,
    }


def test_m6_open_ended_units_match_synthetic_motion_clusters_and_outlier_stays_pending():
    train = _clustered_train()
    method = M6(epochs=4, batch_size=48, width=32, pending_batch_size=60,
                min_support=10, min_episodes=3, kappa=0.1, seed=23)
    method.fit(train)
    assert 2 <= len(method.prototypes) <= 5
    assert len(method.prototypes) != 3 or method.describe()["unit_count_is_fixed"] is False

    params_before = [p.detach().cpu().numpy().copy() for p in method.backbone.parameters()]
    centers_before = method.prototypes.copy()
    outlier_action = np.full((1, 4, 4), 50.0, np.float32)
    outlier_state = np.zeros((1, 3), np.float32)
    outlier_base, _ = method.encode(outlier_action, outlier_state)
    add = method.memory_add(np.asarray([99.0, 99.0, 99.0, 99.0, 99.0, 99.0], np.float32),
                            outlier_base[0], {"episode_id": "outlier-ep", "init_state_id": "outlier-init"})
    assert add["assignment"] == "pending_functionally_novel"
    same_center = method.memory_add(np.full((6,), -99.0, np.float32), method.prototypes[0],
                                    {"episode_id": "near-ep", "init_state_id": "near-init"})
    assert same_center["assignment"] == "assigned_existing_unit"
    # One isolated observation cannot create a unit and the default store path
    # does not adapt existing centers or any encoder/decoder/prior parameters.
    assert len(method.prototypes) == len(centers_before)
    np.testing.assert_array_equal(method.prototypes, centers_before)
    for old, new in zip(params_before, method.backbone.parameters()):
        np.testing.assert_array_equal(old, new.detach().cpu().numpy())
    assert method.describe()["pending_count"] >= 1


def test_m6_pending_component_merges_into_functionally_close_existing_unit():
    method = M6(min_support=5, min_episodes=3, pending_batch_size=5)
    method.base_dim = 1
    method._fitted = True
    method._fit_in_progress = False
    method.epsilon = 1.0
    method._centers = [np.asarray([0.0], np.float32)]
    method._sync_prototypes()
    method.support_counts = [10]
    method.episode_sets = [{"known-0", "known-1", "known-2"}]
    method._key_tau = None
    def fake_distance(left, right):
        return (
        0.5 if (0.95 <= left[0] <= 1.05 and np.isclose(right[0], 0.0))
        or (0.95 <= right[0] <= 1.05 and np.isclose(left[0], 0.0))
        else (2.0 if np.isclose(left[0], 0.0) or np.isclose(right[0], 0.0) else 0.1)
        )
    method._functional_distances = lambda base, candidates: np.asarray(
        [fake_distance(base, candidate) for candidate in candidates], dtype=np.float64)
    outcomes = []
    for i, value in enumerate([0.9, 1.1, 0.9, 1.1, 0.9]):
        outcomes.append(method.memory_add(np.asarray([float(i), 1.0], np.float32),
                                          np.asarray([value], np.float32),
                                          {"episode_id": f"ep-{i}", "init_state_id": f"init-{i}"}))
    assert len(method.prototypes) == 1
    assert method.merged_count == 1
    assert method.registered_count == 0
    assert all(item["unit_id"] == 0 for item in method.memory)
    assert outcomes[-1]["merged_total"] == 1


def test_m6_soft_query_returns_item_and_skill_blends_and_reports_open_ended_config():
    method = M6(epochs=2, batch_size=48, width=32, pending_batch_size=60,
                min_support=5, min_episodes=3, topk=8)
    train = _clustered_train(n=150)
    method.fit(train)
    base, _ = method.encode(train["actions"][:2], train["state"][:2])
    for i in range(2):
        method.memory_add(train["key"][i], base[i],
                          {"episode_id": f"memory-{i}", "init_state_id": f"memory-init-{i}", "seed": True})
    query = method.memory_query(train["key"][0], {"episode_id": "query", "init_state_id": "query-init"})
    assert query["bases"].ndim == 2
    np.testing.assert_allclose(query["weights"].sum(), 1.0, atol=1e-6)
    assert query["item_blend"].shape == (method.base_dim,)
    assert query["skill_blend"].shape == (method.base_dim,)
    assert np.isfinite(method.decode(query["item_blend"], method.prior(train["feature"][:1], train["state"][:1])[1][0],
                                     train["feature"][:1], train["state"][:1])).all()
    description = method.describe()
    assert description["kappa_sweep"] == [0.05, 0.1, 0.2]
    assert description["min_support_sweep"] == [5, 10, 20]
    assert description["min_episodes"] == 3
    assert description["unit_count_is_fixed"] is False


def test_m6_m4_encoder_ablation_uses_same_unit_interface():
    train = _clustered_train(n=48)
    method = M6(encoder="m4", epochs=1, batch_size=24, width=16,
                pending_batch_size=20, min_support=3, min_episodes=2)
    method.fit(train)
    base, residual = method.encode(train["actions"][:2], train["state"][:2])
    prior_base, prior_residual = method.prior(train["feature"][:2], train["state"][:2])
    pred = method.decode(base, residual, train["feature"][:2], train["state"][:2])
    assert base.shape == prior_base.shape == (2, 12)
    assert residual.shape == prior_residual.shape == (2, 4)
    assert pred.shape == train["actions"][:2].shape


def test_m6_variant_grid_covers_hsic_scale_support_and_m4_encoder():
    variants = build_variants()
    assert len(variants) == 27
    assert "M6_m1_hsic0.01_k0.05_m5" in variants
    assert "M6_m1_hsic0.1_k0.2_m20" in variants
    assert "M6_m4_k0.1_m10" in variants
