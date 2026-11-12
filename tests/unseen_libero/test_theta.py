"""Pose (theta) regression on top of the selected shape."""
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support import (
    UNSEEN,
    load_script,
)

THETA_SCRIPT = UNSEEN / "theta_regression.py"
tr = load_script(THETA_SCRIPT)
lora = load_script(UNSEEN / "theta_regression_lora.py")


# ── Theta regression ──────────────────────────────────────────────────────────────────────────────

def test_grouped_prefix_features_keep_camera_and_language_separate():
    h=np.arange(7*4,dtype=np.float32).reshape(7,4);g=np.array([0,0,1,1,1,2,2])
    x=tr.grouped_means(h,g)
    np.testing.assert_allclose(x[0],h[:2].mean(0));np.testing.assert_allclose(x[1],h[2:5].mean(0));np.testing.assert_allclose(x[2],h[5:].mean(0))
    assert x.shape==(3,4)


def test_rot6_representation_roundtrips_axis_angle():
    from scipy.spatial.transform import Rotation
    t=np.array([.2,-.1,.3]);v=tr.rotation6(t)
    np.testing.assert_allclose(Rotation.from_matrix(tr.rot6_matrix(v)).as_rotvec(),t,atol=1e-6)


def test_axis_angle_log_scale_decodes_in_canonical_physical_space():
    from scipy.spatial.transform import Rotation
    vs_path=THETA_SCRIPT.parent/'variable_shape_memory.py'
    spec=importlib.util.spec_from_file_location('theta_test_variable_shape_memory',vs_path)
    vs=importlib.util.module_from_spec(spec);spec.loader.exec_module(vs)
    shape=np.arange(70,dtype=np.float32).reshape(10,7)/50
    theta=np.r_[np.array([.2,-.1,.3],np.float32),np.log(1.4)]
    actual=tr.decode_theta(shape,theta,vs)
    r=Rotation.from_rotvec(theta[:3]).as_matrix();scale=np.exp(theta[3])
    physical=np.concatenate([shape[:,:3]@r,shape[:,3:6]@r],axis=1)*scale
    expected=np.concatenate([vs.to_normalized(physical),shape[:,6:7]],axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-6)


def test_resampling_and_candidate_scoring_ignore_gripper_for_shape():
    a=np.zeros((4,7),np.float32);a[:,0]=np.arange(4);a[:,6]=1
    b=a.copy();b[:,6]=-1
    x=tr._resample_window(a);y=tr._resample_window(b)
    assert x.shape==(10,7)
    assert tr._shape_error(x,y,None)[0]==pytest.approx(0.)


def test_random_candidate_baseline_averages_errors_not_actions():
    target=np.zeros((1,10,7),np.float32)
    replicas=np.stack([np.ones((10,7),np.float32),-np.ones((10,7),np.float32)],axis=0)[None]
    mean_error=tr._replica_mean_errors(replicas,target)
    blended=np.mean(replicas,axis=1)
    blended_error=np.mean((blended-target)**2,axis=(1,2))
    np.testing.assert_allclose(mean_error,[1.])
    np.testing.assert_allclose(blended_error,[0.])


def test_training_cache_inventory_is_fixed_and_separate_from_eval_targets():
    assert tr.SOURCE.name=='production'
    assert tr.SPLITS==('train','dev','eval')
    assert tr.CACHES==('C_vm_none','C_vm_08')
    assert len(tr.VARIANTS)==5


def test_pose_diagnostics_compare_matching_representations():
    p=np.array([.2,-.1,.3,.4],np.float32)
    t=np.array([-.1,.2,.1,-.3],np.float32)
    for kind in ('aa','6d'):
        expected=np.mean((tr._target_theta_repr(p,kind)-tr._target_theta_repr(t,kind))**2)
        assert tr._pose_parameter_error(p,t,kind)==pytest.approx(expected)
        assert tr._pose_parameter_error(p,p,kind)==pytest.approx(0.)


# ── Theta regression with LoRA ────────────────────────────────────────────────────────────────────

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
