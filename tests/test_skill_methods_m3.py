import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.methods.m3_rvq import ResidualVQ


def _data(n=24, horizon=4, action_dim=7, state_dim=8, feature_dim=12):
    gen = torch.Generator().manual_seed(4)
    feature = torch.randn(n, feature_dim, generator=gen)
    return {
        "actions": torch.randn(n, horizon, action_dim, generator=gen),
        "state": torch.randn(n, state_dim, generator=gen),
        "feature": feature,
        "key": feature.clone(),
        "episode_id": np.asarray([f"ep{i}" for i in range(n)]),
        "init_state_id": np.asarray([f"init{i}" for i in range(n)]),
    }


def test_m3_rvq_fit_encode_decode_prior_and_two_memory_levels():
    data = _data()
    method = ResidualVQ(action_horizon=4, action_dim=7, state_dim=8, feature_dim=12,
                        coarse_codes=8, fine_codes=8, code_dim=4, residual_dim=4,
                        hidden_dim=24, epochs=2, batch_size=8, seed=2)
    method.fit(data)
    base, residual = method.encode(data["actions"], data["state"])
    assert base.shape == (24, 2) and np.issubdtype(base.dtype, np.integer)
    assert residual.shape == (24, 4)
    decoded = method.decode(base, residual, data["feature"], data["state"])
    assert decoded.shape == data["actions"].shape and np.isfinite(decoded).all()

    distribution, prior_residual = method.prior(data["feature"], data["state"])
    assert distribution["coarse_probs"].shape == (24, 8)
    assert distribution["fine_probs"].shape == (24, 8)
    np.testing.assert_allclose(distribution["coarse_probs"].sum(-1), np.ones(24), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(distribution["fine_probs"].sum(-1), np.ones(24), atol=1e-6, rtol=1e-6)
    prior_base = method.prior_base(distribution)
    assert prior_base.shape == (24, 8)
    assert method.decode(prior_base, prior_residual, data["feature"], data["state"]).shape == data["actions"].shape

    weights_before = {k: v.detach().clone() for k, v in method.state_dict().items()}
    assert method.memory_add(data["feature"][0], base[0], {"episode_id": "e0", "init_state_id": "i0"})[0]
    coarse = method.memory_query(data["feature"][0], top_k=1, level="coarse")
    assert coarse[0].shape == (1, 2) and coarse[0][0, 1] == -1
    full = method.memory_query(data["feature"][0], top_k=1, level="both")
    assert full[0][0, 1] == base[0, 1]
    for key, value in weights_before.items():
        torch.testing.assert_close(method.state_dict()[key], value)
    assert method.describe()["compositional_base_capacity"] == 64


def test_m3_decoder_uses_base_not_observation_and_residual_kl_variants_exist():
    gen = torch.Generator().manual_seed(61)
    n, horizon = 64, 4
    mode = torch.arange(n) % 4
    templates = torch.tensor([
        [1.0, 0.0, 0.0, 0.2, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -0.2, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.2, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0, -0.2, 0.0, 1.0],
    ])
    actions = templates[mode, None, :].expand(n, horizon, 7).clone()
    actions += 0.015 * torch.randn(actions.shape, generator=gen)
    state = 0.01 * torch.randn(n, 8, generator=gen)
    feature = torch.randn(n, 12, generator=gen)
    data = {"actions": actions, "state": state, "feature": feature,
            "key": feature.clone(), "episode_id": np.asarray([f"e{i}" for i in range(n)]),
            "init_state_id": np.asarray([f"i{i}" for i in range(n)]),
            "early_stopping_patience": 100}
    method = ResidualVQ(action_horizon=horizon, action_dim=7, state_dim=8, feature_dim=12,
                        coarse_codes=8, fine_codes=8, code_dim=4, residual_dim=2,
                        hidden_dim=32, epochs=100, batch_size=16, residual_kl_weight=0.1, seed=7)
    method.fit(data)
    base, residual = method.encode(actions, state)
    pred = method.decode(base, residual, feature, state)
    changed_obs = method.decode(base, residual, feature + 50.0, state)
    np.testing.assert_allclose(pred, changed_obs, atol=1e-6, rtol=1e-6)
    zero_base = np.zeros((n, 2 * method.code_dim), dtype=np.float32)
    zero_pred = method.decode(zero_base, residual, feature, state)
    assert np.mean((zero_pred - actions.numpy()) ** 2) > np.mean((pred - actions.numpy()) ** 2) + 0.03
    assert method.decoder_obs is False
    variants = __import__(
        "skill_moe.skillnet.examples.emergent.light.methods.m3_rvq", fromlist=["build_variants"]
    ).build_variants()
    assert variants["M3_rvq16x16"].residual_kl_weight == 0.1
    assert variants["M3_rvq16x16_beta1.0"].residual_kl_weight == 1.0


def test_m3_key_additions_use_calibrated_threshold_instead_of_always_adding():
    data = _data(n=32)
    method = ResidualVQ(action_horizon=4, action_dim=7, state_dim=8, feature_dim=12,
                        coarse_codes=8, fine_codes=8, code_dim=4, residual_dim=3,
                        hidden_dim=24, epochs=3, batch_size=8, seed=9)
    method.fit(data)
    bases, _ = method.encode(data["actions"], data["state"])
    accepted = []
    for i in range(8):
        if i % 2:
            key = data["key"][i].numpy()
        else:
            key = np.zeros(12, dtype=np.float32)
            key[i // 2] = 100.0
        ok, _ = method.memory_add(key, bases[i], {"episode_id": f"new-e{i}", "init_state_id": f"new-i{i}"})
        accepted.append(ok)
    rate = np.mean(accepted)
    assert method.tau is not None and method.base_novelty_threshold is not None
    assert 0.0 < rate < 1.0
