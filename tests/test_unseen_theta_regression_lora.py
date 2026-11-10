import numpy as np
import jax
import jax.numpy as jnp
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/theta_regression_lora.py"
_SPEC = importlib.util.spec_from_file_location("theta_regression_lora_test", _SCRIPT)
lora = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lora)


def test_dense_overlay_keeps_new_lora_leaves_and_checks_base_shapes():
    target = {"frozen": {"kernel": np.zeros((2, 3), np.float32)},
              "lora": {"w_a": np.ones((2, 4), np.float32)}}
    source = {"frozen": {"kernel": np.full((2, 3), 7, np.float32)}}
    got = lora._overlay_pretrained(target, source)
    np.testing.assert_array_equal(got["frozen"]["kernel"], 7)
    np.testing.assert_array_equal(got["lora"]["w_a"], 1)
    try:
        lora._overlay_pretrained({"frozen": np.zeros((2, 3))}, {"frozen": np.zeros((3, 2))})
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("base checkpoint shape mismatch must fail closed")


def test_vlm_lora_gradient_mask_keeps_only_last_six_scanned_layers():
    x = np.ones((18, 2, 3), np.float32)
    tree = {"PaliGemma": {"llm": {"layers": {"attn_lora_a": x}}}}
    got = lora.mask_early_lora_gradients(tree)
    assert not np.any(got["PaliGemma"]["llm"]["layers"]["attn_lora_a"][:12])
    np.testing.assert_array_equal(got["PaliGemma"]["llm"]["layers"]["attn_lora_a"][12:], 1)


def test_jax_gradient_and_weight_decay_masks_freeze_early_lora_layers():
    from flax import nnx
    state = nnx.State({"PaliGemma": {"llm": {"layers": {
        "attention": {"lora_a": jnp.ones((18, 2, 3)), "dense": jnp.ones((18, 3, 3))}}}}})
    got = lora._jax_mask_late_gradients(state)
    pure = got.to_pure_dict()["PaliGemma"]["llm"]["layers"]["attention"]
    assert not np.any(np.asarray(pure["lora_a"][:12]))
    np.testing.assert_array_equal(np.asarray(pure["lora_a"][12:]), 1)
    # Dense base weights are frozen and never masked into an optimizer update.
    np.testing.assert_array_equal(np.asarray(pure["dense"]), 1)


def test_training_rows_must_be_exact_libero40_nonvalidation_split():
    task = np.array([0, 0, 1, 40, 1, 41])
    validation = np.array([False, True, False, False, True, False])
    got = lora.validate_libero40_train_rows([0, 2], task, validation, expected_count=2)
    np.testing.assert_array_equal(got, [0, 2])
    for invalid in ([0, 0], [0, 3], [0]):
        try:
            lora.validate_libero40_train_rows(invalid, task, validation, expected_count=2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid training IDs accepted: {invalid}")


def test_zero_initialized_lora_b_has_gradient():
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    w = jnp.eye(4, dtype=jnp.float32)
    a = jnp.full((4, 2), .1, jnp.float32)
    b = jnp.zeros((2, 4), jnp.float32)

    def loss(a_, b_):
        y = x @ (w + a_ @ b_)
        return jnp.mean((y - 1.3) ** 2)

    ga, gb = jax.grad(loss, argnums=(0, 1))(a, b)
    assert np.linalg.norm(np.asarray(gb)) > 0
    # With B=0, the adapter starts learning through B; A's first gradient is
    # zero, which is the standard low-rank initialization behavior.
    assert np.linalg.norm(np.asarray(ga)) == 0


def test_unit_shape_feature_contract_is_23_and_group_consistent():
    shape = jnp.arange(2 * 10 * 7, dtype=jnp.float32).reshape(2, 10, 7) / 100
    units = jnp.asarray([[0] * 4 + [1] * 6, [2] * 10], dtype=jnp.int32)
    feat = np.asarray(lora._unit_shape_features(shape, units))
    assert feat.shape == (2, 10, 23)
    np.testing.assert_allclose(feat[0, :4], np.broadcast_to(feat[0, :1], (4, 23)))
    np.testing.assert_allclose(feat[0, 4:], np.broadcast_to(feat[0, 4:5], (6, 23)))
    np.testing.assert_allclose(feat[1], np.broadcast_to(feat[1, :1], (10, 23)))
    # Mean phase is normalized to [0,1], including a unit occupying the final
    # six positions of the window.
    phase = np.asarray(feat[0, 4:, -1])
    np.testing.assert_allclose(phase, np.full(6, (4 + 9) / 2 / 9), rtol=1e-6)
