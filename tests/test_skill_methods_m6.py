from __future__ import annotations

import numpy as np

from skill_moe.skillnet.examples.emergent.light.methods.m6 import M6, build_variants


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
