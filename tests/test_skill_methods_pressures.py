import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.methods.m2 import M2
from skill_moe.skillnet.examples.emergent.light.methods.m1 import M1
from skill_moe.skillnet.examples.emergent.light.methods.m4 import M4
from skill_moe.skillnet.examples.emergent.light.methods.m5 import M5
from skill_moe.skillnet.examples.emergent.light.methods.pressures import (
    add_swap_reconstruction_pressure,
    swap_reconstruction_loss,
)


class ContinuousToy:
    def __init__(self):
        self.decoded_base = None

    def encode(self, actions, state):
        base = torch.tensor([[0.0], [0.1], [0.2]], dtype=actions.dtype)
        return base, actions

    def decode(self, base, residual, feature, state):
        self.decoded_base = base.detach().clone()
        return residual + base[:, None, :] * 0.01


def test_swap_reconstruction_pressure_composes_with_m2_and_has_gradients():
    gen = torch.Generator().manual_seed(23)
    actions = torch.randn(24, 4, 7, generator=gen) * 0.1
    state = torch.zeros(24, 8)
    feature = torch.randn(24, 12, generator=gen)
    method = M2(n_codes=16, embedding_dim=8, residual_dim=4, width=24, epochs=2, batch_size=8, seed_value=3)
    train = {"actions": actions.numpy(), "state": state.numpy(), "feature": feature.numpy(), "key": feature.numpy(),
             "episode_id": np.arange(24).astype(str), "init_state_id": np.arange(24).astype(str)}
    method.fit(train)
    base, residual = method.encode(actions.numpy(), state.numpy())
    assert method.decode(base, residual, feature.numpy(), state.numpy()).shape == actions.shape
    b_dist, _ = method.prior(feature.numpy(), state.numpy())
    assert b_dist.shape == (24, 16)
    loss, directed_swaps = swap_reconstruction_loss(method, actions, state, feature, return_pair_count=True)
    assert directed_swaps > 0
    assert loss.item() > 0
    total = add_swap_reconstruction_pressure(torch.tensor(2.0), method, actions, state, feature, weight=0.25)
    total.backward()
    assert method.decoder[0].weight.grad is not None and method.decoder[0].weight.grad.abs().sum().item() > 0
    assert total.item() > 2.0
    assert method.memory_add(feature[0].numpy(), base[:1], {"episode_id": "store", "init_state_id": "store"})[0]
    retrieved = method.memory_query(feature[1].numpy(), {"episode_id": "query", "init_state_id": "query"})
    assert len(retrieved[2]) == 1


def test_swap_pressure_uses_near_peer_base_and_residual():
    method = ContinuousToy()
    actions = torch.zeros(3, 2, 1)
    state = torch.zeros(3, 2)
    feature = torch.zeros(3, 4)
    _, pair_count = swap_reconstruction_loss(method, actions, state, feature, return_pair_count=True)
    assert pair_count == 4
    # First directed pair targets row 0 but must decode with row 1's base.
    torch.testing.assert_close(method.decoded_base[0], torch.tensor([0.1]))


def test_swap_pressure_has_gradient_paths_for_other_method_families():
    gen = torch.Generator().manual_seed(31)
    actions = torch.randn(12, 4, 7, generator=gen) * 0.1
    state = torch.randn(12, 8, generator=gen)
    feature = torch.randn(12, 6, generator=gen)
    train = {"actions": actions.numpy(), "state": state.numpy(), "feature": feature.numpy(),
             "key": feature.numpy(), "episode_id": np.arange(12).astype(str),
             "init_state_id": np.arange(12).astype(str)}
    methods = [
        M1(epochs=1, batch_size=6, width=16),
        M4(epochs=1, batch_size=6, width=16, base_dim=6, residual_dim=3, max_prototypes=8),
        M5(epochs=1, batch_size=6, width=16, base_dim=6, residual_dim=3, n_atoms=8, top_k=2),
    ]
    for method in methods:
        method.fit(train)
        loss, pairs = swap_reconstruction_loss(method, actions, state, feature, return_pair_count=True)
        assert pairs > 0 and torch.isfinite(loss)
        loss.backward()
        if isinstance(method, M1):
            decoder = method.decoder
        else:
            decoder = method.net.decoder
        assert decoder[0].weight.grad is not None
        assert decoder[0].weight.grad.abs().sum().item() > 0
