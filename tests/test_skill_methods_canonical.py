import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.methods.canonical import (
    CanonicalizedMethod,
    canonicalize,
    decanonicalize,
)
from skill_moe.skillnet.examples.emergent.light.methods.m2 import M2
from skill_moe.skillnet.examples.emergent.light.methods.pressures import swap_reconstruction_loss


def _actions_and_state(n=24, horizon=5):
    gen = torch.Generator().manual_seed(10)
    actions = torch.randn(n, horizon, 7, generator=gen) * 0.03
    actions[:, :, 6] = torch.randint(0, 2, (n, horizon), generator=gen).float() * 2 - 1
    state = torch.zeros(n, 8)
    state[:, :3] = torch.randn(n, 3, generator=gen)
    state[:, 5] = torch.linspace(-1.2, 1.1, n)  # LIBERO axis-angle yaw examples
    return actions, state


def test_canonical_transform_is_yaw_scale_invariant_and_round_trips():
    actions, state = _actions_and_state(n=2)
    base1, geo1 = canonicalize(actions, state)
    # Same local trajectory, changed world start yaw and net-displacement scale.
    local = base1[0] * 2.7
    yaw = torch.tensor(0.8)
    rotated = torch.stack([torch.cos(yaw) * local[:, 0] - torch.sin(yaw) * local[:, 1],
                           torch.sin(yaw) * local[:, 0] + torch.cos(yaw) * local[:, 1], local[:, 2]], dim=-1)
    actions2 = actions[1:2].clone()
    actions2[0, :, :3] = rotated
    state2 = state[1:2].clone(); state2[:, 3:6] = torch.tensor([[0.0, 0.0, 0.8]])
    base2, _ = canonicalize(actions2, state2)
    torch.testing.assert_close(base1[0], base2[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decanonicalize(base1, geo1), actions, atol=1e-6, rtol=1e-6)


def test_c_pressure_wraps_m2_fit_encode_decode_prior_and_memory():
    actions, state = _actions_and_state()
    feature = torch.randn(len(actions), 9, generator=torch.Generator().manual_seed(11))
    train = {"actions": actions.numpy(), "state": state.numpy(), "feature": feature.numpy(), "key": feature.numpy(),
             "episode_id": np.arange(len(actions)).astype(str), "init_state_id": np.arange(len(actions)).astype(str)}
    inner = M2(n_codes=16, embedding_dim=8, residual_dim=4, width=24, epochs=2, batch_size=4, seed_value=5)
    method = CanonicalizedMethod(inner, epochs=2, batch_size=4, hidden_dim=16)
    method.fit(train)
    assert method.decoder_obs is False
    assert inner.feature_dim == 0
    base, residual = method.encode(actions, state)
    reconstructed = method.decode(base, residual, feature, state)
    assert reconstructed.shape == actions.shape and torch.isfinite(reconstructed).all()
    swap_loss, swap_count = swap_reconstruction_loss(method, actions, state, feature, return_pair_count=True)
    assert swap_count > 0 and torch.isfinite(swap_loss)
    swap_loss.backward()
    assert inner.decoder[0].weight.grad is not None
    # D1: the action decoder is independent of the observation key after the
    # base and residual have been selected.
    np.testing.assert_allclose(reconstructed.detach().numpy(),
                               method.decode(base, residual, feature + 20.0, state).detach().numpy(),
                               atol=1e-6, rtol=1e-6)

    base_dist, prior_residual = method.prior(feature, state)
    assert base_dist.shape == (len(actions), 16)
    assert method.decode(base_dist, prior_residual, feature, state).shape == actions.shape
    assert method.memory_add(feature[0].numpy(), base[:1], {"episode_id": "store", "init_state_id": "store"})[0]
    retrieved = method.memory_query(feature[1].numpy(), meta={"episode_id": "query", "init_state_id": "query"})
    assert len(retrieved[2]) == 1
    assert "yaw-rotated" in method.describe()["base_input"]


def test_m2_plus_c_uses_base_when_decoder_cannot_see_observation():
    gen = torch.Generator().manual_seed(72)
    n, horizon = 64, 5
    patterns = torch.tensor([
        [[0.14, 0.0, 0.0]] * 5,
        [[0.50, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        [[0.0, 0.14, 0.0]] * 5,
        [[0.0, 0.50, 0.0], [0.0, 0.05, 0.0], [0.0, 0.05, 0.0], [0.0, 0.05, 0.0], [0.0, 0.05, 0.0]],
    ])
    mode = torch.arange(n) % 4
    local = patterns[mode]
    state = torch.zeros(n, 8)
    state[:, 5] = torch.linspace(-1.0, 1.0, n)
    scale = torch.linspace(0.6, 1.8, n)
    c, s = torch.cos(state[:, 5]), torch.sin(state[:, 5])
    xyz = local.clone() * scale[:, None, None]
    world = torch.stack([c[:, None] * xyz[..., 0] - s[:, None] * xyz[..., 1],
                         s[:, None] * xyz[..., 0] + c[:, None] * xyz[..., 1], xyz[..., 2]], dim=-1)
    actions = torch.zeros(n, horizon, 7)
    actions[..., :3] = world + 0.001 * torch.randn(world.shape, generator=gen)
    actions[..., 3:6] = 0.005 * torch.randn(n, horizon, 3, generator=gen)
    actions[..., 6] = torch.where(mode[:, None] % 2 == 0, -1.0, 1.0)
    feature = torch.randn(n, 9, generator=gen)
    train = {"actions": actions.numpy(), "state": state.numpy(), "feature": feature.numpy(),
             "key": feature.numpy(), "episode_id": np.asarray([f"e{i}" for i in range(n)]),
             "init_state_id": np.asarray([f"i{i}" for i in range(n)])}
    inner = M2(n_codes=16, embedding_dim=8, residual_dim=2, width=32, epochs=24, batch_size=16, seed_value=12)
    method = CanonicalizedMethod(inner, epochs=2, batch_size=16, hidden_dim=16)
    method.fit(train)
    base, residual = method.encode(actions, state)
    pred = method.decode(base, residual, feature, state)
    base_zero = torch.zeros(n, inner.embedding_dim)
    zero_inner = method._decode_inner(base_zero, residual["inner"], feature, state)
    geometry_pred = method.decode(base, residual, feature, state)
    zero_actions = method.decode(base_zero, residual, feature, state)
    assert torch.mean((zero_inner - canonicalize(actions, state)[0]) ** 2) > 0.001
    assert torch.mean((zero_actions - actions) ** 2) > torch.mean((geometry_pred - actions) ** 2) + 1e-5
    np.testing.assert_allclose(pred.detach().numpy(),
                               method.decode(base, residual, feature + 100.0, state).detach().numpy(),
                               atol=1e-6, rtol=1e-6)
