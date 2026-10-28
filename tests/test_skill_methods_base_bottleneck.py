import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.light.methods.m0 import M0
from skill_moe.skillnet.examples.emergent.light.methods.m1 import M1
from skill_moe.skillnet.examples.emergent.light.methods.m2 import M2
from skill_moe.skillnet.examples.emergent.light.methods.fsq import FSQ


def _data(seed=3, n=256, horizon=5):
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
    train = _data()
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
    train = _data(n=64)
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
    train = _data(n=64)
    method = factory(); method.fit(train)
    base, residual = method.encode(train["actions"][:8], train["state"][:8])
    first = method.decode(base, residual, train["feature"][:8], train["state"][:8])
    second = method.decode(base, residual, train["feature"][:8] + 100, train["state"][:8])
    np.testing.assert_allclose(first, second, atol=1e-6)
