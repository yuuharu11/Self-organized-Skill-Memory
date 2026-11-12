"""Emergent-skill bottlenecks M0-M3 and FSQ."""
import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.emergent.light.methods.fsq import FSQ
from skill_moe.skillnet.examples.emergent.light.methods.m0 import M0
from skill_moe.skillnet.examples.emergent.light.methods.m1 import M1
from skill_moe.skillnet.examples.emergent.light.methods.m2 import M2
from skill_moe.skillnet.examples.emergent.light.methods.m3_rvq import ResidualVQ

# ── M0 ────────────────────────────────────────────────────────────────────────────────────────────

def test_m0_fit_encode_decode_prior_and_memory():
    rng=np.random.default_rng(8);n=32;h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),"state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),"key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),"init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":1,"epochs":1,"batch_size":16}
    m=M0(epochs=1,batch_size=16,width=32);m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4]);pb,pr=m.prior(train["feature"][:4],train["state"][:4])
    out=m.decode(b,r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,32) and r is None and pr is None and out.shape==(4,h,7)
    assert np.isfinite(out).all() and pb.shape==(4,32)
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    base=m.blend_base(candidates,weights)
    assert candidates.shape[1:]==(32,) and np.isclose(weights.sum(),1)
    assert m.memory_add(train["key"][1],b[1],{"episode_id":"e1","init_state_id":"i1"})[1] in {"added_novel","near_existing"}
    assert m.describe()["name"]=="M0"


# ── M1 ────────────────────────────────────────────────────────────────────────────────────────────

def test_m1_fit_encode_decode_prior_and_memory():
    rng=np.random.default_rng(9);n=32;h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),"state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),"key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),"init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":2,"epochs":1,"batch_size":16}
    m=M1(epochs=1,batch_size=16,width=32,independence_weight=.01);m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4]);pb,pr=m.prior(train["feature"][:4],train["state"][:4])
    out=m.decode(b,r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,12) and r.shape==(4,4) and pb.shape==(4,12) and pr.shape==(4,4)
    assert out.shape==(4,h,7) and np.isfinite(out).all()
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    base=m.blend_base(candidates,weights);assert base.shape==(12,) and np.isclose(weights.sum(),1)
    assert m.memory_add(train["key"][1],b[1],{"episode_id":"e1","init_state_id":"i1"})[1] in {"added_novel","near_existing"}
    assert m.describe()["independence_weight"]==.01


# ── M2 ────────────────────────────────────────────────────────────────────────────────────────────

def test_m2_fit_encode_decode_prior_and_memory():
    rng=np.random.default_rng(10);n=48;h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),"state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),"key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),"init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":3,"epochs":1,"batch_size":16}
    m=M2(n_codes=16,epochs=1,batch_size=16,width=32);m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4]);bd,pr=m.prior(train["feature"][:4],train["state"][:4])
    out=m.decode(m.base_features(b),r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,) and r.shape==(4,4) and bd.shape==(4,16) and pr.shape==(4,4)
    assert out.shape==(4,h,7) and np.isfinite(out).all()
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    base=m.blend_base(candidates,weights);assert base.shape==(16,) and np.isclose(weights.sum(),1)
    assert m.memory_add(train["key"][1],b[1],{"episode_id":"e1","init_state_id":"i1"})[1] in {"added_novel","near_existing"}
    assert m.describe()["base"].startswith("EMA VQ code with 16")


# ── FSQ ───────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("codes", [16, 32, 64])
def test_fsq_fit_encode_decode_prior_and_memory(codes):
    rng=np.random.default_rng(17); n=40; h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),
           "state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),
           "key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),
           "init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":4,"epochs":1,"batch_size":16}
    m=FSQ(n_codes=codes,epochs=1,batch_size=16,width=32); m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4])
    logits,rp=m.prior(train["feature"][:4],train["state"][:4])
    y=m.decode(m.base_features(b),r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,) and b.min()>=0 and b.max()<codes
    assert r.shape==(4,4) and logits.shape==(4,codes) and rp.shape==(4,4)
    assert y.shape==(4,h,7) and np.isfinite(y).all()
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    assert m.blend_base(candidates,weights).shape==(m.base_dim,)
    assert m.describe()["neural_weights_fixed_on_memory_add"]


# ── M3 residual VQ ────────────────────────────────────────────────────────────────────────────────

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


# ── Shared bottleneck contract ────────────────────────────────────────────────────────────────────

def _data_shared_bottleneck_contract(seed=3, n=256, horizon=5):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    feature = rng.normal(size=(n, 24)).astype(np.float32)
    mode = rng.integers(0, 4, n)
    prototype = np.eye(4, dtype=np.float32)[mode]
    actions = np.repeat(prototype[:, None, :], horizon, axis=1)
    actions = np.concatenate([actions, rng.normal(scale=.03, size=(n, horizon, 3)).astype(np.float32)], axis=2)
    return {
        "actions": actions.astype(np.float32), "state": state, "feature": feature,
        "key": rng.normal(size=(n, 10)).astype(np.float32),
        "episode_id": np.asarray([f"episode-{i}" for i in range(n)]),
        "init_state_id": np.asarray([f"init-{i}" for i in range(n)]),
        "seed": seed, "epochs": 10, "batch_size": 64,
    }


@pytest.mark.parametrize("factory", [
    lambda: M0(width=64, epochs=10, batch_size=64),
    lambda: M1(beta_r=1.0, width=64, epochs=10, batch_size=64),
    lambda: M1(beta_r=10.0, width=64, epochs=10, batch_size=64),
    lambda: M2(n_codes=16, beta_r=1.0, width=64, epochs=10, batch_size=64),
    lambda: M2(n_codes=16, beta_r=10.0, width=64, epochs=10, batch_size=64),
    lambda: FSQ(n_codes=16, beta_r=1.0, width=64, epochs=10, batch_size=64),
    lambda: FSQ(n_codes=16, beta_r=10.0, width=64, epochs=10, batch_size=64),
])
def test_default_decoder_uses_base_and_residual_rate_is_finite(factory):
    train = _data_shared_bottleneck_contract()
    method = factory()
    method.fit(train)
    base, residual = method.encode(train["actions"], train["state"])
    full = method.decode(base, residual, train["feature"], train["state"])
    zero = method.decode(np.zeros_like(method.base_features(base)), residual, train["feature"], train["state"])
    mse_full = np.mean((full - train["actions"]) ** 2)
    mse_zero = np.mean((zero - train["actions"]) ** 2)
    assert mse_zero > mse_full + .005
    assert method.describe()["decoder_obs"] is False
    if residual is not None:
        bits = method.residual_kl_bits(train["actions"], train["state"])
        assert bits.shape == (len(train["actions"]),)
        assert np.isfinite(bits).all() and np.all(bits >= 0)


def test_default_decoder_does_not_read_observation_feature():
    train = _data_shared_bottleneck_contract(n=64)
    method = M0(width=32, epochs=2, batch_size=32)
    method.fit(train)
    base, residual = method.encode(train["actions"][:8], train["state"][:8])
    a = method.decode(base, residual, train["feature"][:8], train["state"][:8])
    b = method.decode(base, residual, train["feature"][:8] + 100, train["state"][:8])
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("factory", [
    lambda: M1(beta_r=1.0, width=32, epochs=2, batch_size=32),
    lambda: M2(n_codes=16, beta_r=1.0, width=32, epochs=2, batch_size=32),
    lambda: FSQ(n_codes=16, beta_r=1.0, width=32, epochs=2, batch_size=32),
])
def test_default_split_decoder_is_observation_independent(factory):
    train = _data_shared_bottleneck_contract(n=64)
    method = factory(); method.fit(train)
    base, residual = method.encode(train["actions"][:8], train["state"][:8])
    first = method.decode(base, residual, train["feature"][:8], train["state"][:8])
    second = method.decode(base, residual, train["feature"][:8] + 100, train["state"][:8])
    np.testing.assert_allclose(first, second, atol=1e-6)
