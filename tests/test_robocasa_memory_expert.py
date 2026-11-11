import pathlib
import sys
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
import experiment as ex
import expert


def test_fork_storage_action_roundtrip_for_expert_channels():
    a = np.arange(24, dtype=np.float32).reshape(2, 12)
    p = ex.action_storage_to_policy(a)
    np.testing.assert_array_equal(p[:, :6], a[:, 5:11])
    np.testing.assert_array_equal(p[:, 6], a[:, 11])
    np.testing.assert_array_equal(p[:, 7:12], a[:, :5])
    np.testing.assert_array_equal(ex.action_policy_to_storage(p), a)


def test_runtime_draft_excludes_query_episode_and_returns_storage_order():
    import torch

    class Fixed(torch.nn.Module):
        def forward(self, x, key):
            return torch.tensor([[9.0, 2.0, 1.0]], dtype=torch.float32, device=x.device).repeat(len(x), 1)

    # The highest-scored candidate is from the query episode and must be
    # removed. The selected other-episode unit retains the 12-D source order.
    pool = {
        "key": np.ones((3, 4), np.float32),
        "global_episode": np.asarray([1001, 1002, 1003], np.int32),
        "actions_flat": np.ones((30, 12), np.float32),
        "action_offsets": np.asarray([0, 10, 20, 30], np.int64),
        "length": np.asarray([10, 10, 10], np.int32),
    }
    ck = {"norm": {"action_mean": np.zeros(12, np.float32), "action_std": np.ones(12, np.float32),
                    "state_mean": np.zeros(16, np.float32), "state_std": np.ones(16, np.float32)}}
    draft, chosen = expert._chain(Fixed(), ck, pool, np.zeros((10, 12), np.float32),
                                  np.zeros(16, np.float32), episode=1001, device="cpu", length=50)
    assert draft.shape == (50, 12)
    assert chosen and all(int(i) != 0 for i in chosen)


def test_runtime_adapter_policy_state_inverse_is_correct():
    # State is supplied in fork order [EEF pose, base pose, gripper], while
    # P5 context is stored in LeRobot order [base pose, EEF pose, gripper].
    state = np.arange(16, dtype=np.float32)
    expected = np.concatenate([state[7:14], state[:7], state[14:16]])
    actual = expert.state_policy_to_storage(state)
    np.testing.assert_array_equal(actual, expected)


def test_bridge_training_step_updates_filtered_parameters():
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax

    class Toy(nnx.Module):
        def __init__(self):
            self.scale = nnx.Param(jnp.asarray(0.0))

        def velocity(self, obs, x, t):
            return jnp.ones_like(x) * self.scale.value

    model = Toy()
    filt = nnx.All(nnx.Param)
    tx = optax.adamw(1e-2)
    step = expert._bridge_step(filt, tx)
    opt = tx.init(nnx.state(model, filt))
    target = jnp.ones((2, 3, 32), jnp.float32)
    draft = jnp.zeros_like(target)
    loss, updated = step(model, opt, None, target, draft, jax.random.key(5))
    assert np.isfinite(float(loss))
    assert float(model.scale.value) != 0.0


def test_bridge_training_path_uses_inference_time_scale():
    import jax.numpy as jnp

    target = jnp.ones((2, 3, 32), jnp.float32)
    draft = jnp.zeros_like(target)
    noise = jnp.full_like(target, 2.0)
    t0 = jnp.asarray([.2, .6])
    progress = jnp.asarray([0.0, 1.0])
    x, time, velocity = expert._bridge_training_sample(target, draft, noise, t0, progress)
    np.testing.assert_allclose(np.asarray(time), [0.0, .6])
    np.testing.assert_allclose(np.asarray(x[0]), np.asarray(target[0]))
    source = (1-.6)*draft[1]+.6*noise[1]
    np.testing.assert_allclose(np.asarray(x[1]), np.asarray(source))
    source0 = (1-.2)*draft[0]+.2*noise[0]
    expected_velocity = (source0-target[0])/.2
    np.testing.assert_allclose(np.asarray(velocity[0]), np.asarray(expected_velocity))


def test_scaled_draft_chains_each_configuration_from_its_emitted_history(monkeypatch):
    # A selected memory unit is scaled from the aligned π0.5 chunk, and the
    # next P5 query must see that generated command (including its grip mode).
    calls = []
    def select(_net, _ck, _pool, hist, _state, **_kwargs):
        calls.append(np.asarray(hist).copy())
        return 0, np.asarray([1.0, 0.0], np.float32)
    monkeypatch.setattr(expert, "_p5_select", select)

    policy_action = np.ones((10, 12), np.float32)
    policy_action[:, 6] = .2
    stored_action = ex.action_policy_to_storage(policy_action)
    pool = {
        "actions_flat": np.concatenate([stored_action, stored_action]),
        "action_offsets": np.asarray([0, 10, 20], np.int64),
        "global_episode": np.asarray([1002, 1003], np.int32),
    }
    stats = SimpleNamespace(mean=np.zeros(32, np.float32), std=np.ones(32, np.float32))
    normal = np.zeros((50, 32), np.float32)
    normal[:, :6] = 2.0
    normal[:, 6] = .7
    drafts, scales = expert._p5_scale_draft(
        np.zeros((10, 12), np.float32), np.zeros(16, np.float32), normal,
        pool, [(None, None)], episode=1001, stats=stats, device="cpu")

    assert set(drafts) == {"shared_type", "shared_pi05", "separate_type", "separate_pi05"}
    assert set(scales) == set(drafts)
    # Each configuration starts from an empty history and its second lookup
    # sees exactly its first ten generated actions, converted back to storage.
    for config_index, config in enumerate(drafts):
        first_lookup = config_index * 5
        second_lookup = first_lookup + 1
        expected_policy = np.ones((10, 12), np.float32)
        expected_policy[:, :6] = 2.0
        expected_policy[:, 6] = .2 if config.endswith("_type") else .7
        expected_storage = ex.action_policy_to_storage(expected_policy)
        np.testing.assert_allclose(calls[second_lookup], expected_storage, atol=3e-6)
        np.testing.assert_allclose(drafts[config][:10, :6], 2.0, atol=3e-6)
        assert len(scales[config]) == 5
