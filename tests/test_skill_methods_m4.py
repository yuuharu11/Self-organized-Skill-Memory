from __future__ import annotations

import numpy as np

from skill_moe.skillnet.examples.emergent.light.methods.m4 import M4


def _train(seed: int = 7, n: int = 40):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 3)).astype(np.float32)
    feature = rng.normal(size=(n, 6)).astype(np.float32)
    actions = rng.normal(size=(n, 5, 2)).astype(np.float32)
    return {"actions": actions, "state": state, "features": feature}


def test_m4_fit_encode_decode_prior_and_memory_query():
    train = _train()
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
    train = _train(n=24)
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
    method.fit(_train(n=16))
    n_before = len(method.prototypes)
    novel = -method.prototypes[0]
    result = method.memory_add(np.array([1.0, 0.0]), {"embedding": novel[None, :]})
    assert result["added_prototype"]
    assert len(method.prototypes) == n_before + 1
    assert method.describe()["vigilance_sweep"] == [0.7, 0.85, 0.95]


def test_m4_capacity_path_returns_nearest_code_without_error():
    method = M4(epochs=1, width=16, max_prototypes=32)
    method.fit(_train(n=20))
    method.max_prototypes = len(method.prototypes)
    novel = -method.prototypes[0]
    result = method.memory_add(np.array([1.0, 0.0]), {"embedding": novel[None, :]})
    assert result["reason"] == "vigilance_miss_capacity_reached"
    assert 0 <= result["code"] < len(method.prototypes)


def test_m4_merge_variant_remaps_ids_and_uses_count_weighted_representatives():
    method = M4(epochs=1, width=16, variant="merge", merge_threshold=0.2,
                merge_action_mse_tolerance=1.0)
    method.fit(_train(n=16))
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
    train = _train(n=20)
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
    train = _train(n=60)
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
