"""Tests for the emergent skill latent z (Pi0SkillLatent) on tiny dummy variants."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

from openpi.models import model as _model
from openpi.models import pi0_config_moe_skill
from openpi.models import pi0_config_skill_latent
from openpi.models import skill_latent as _sl
import openpi.shared.nnx_utils as nnx_utils

B = 2
H = 6
NUM_STEPS = 3

_COMMON = dict(
    pi05=True,
    action_horizon=H,
    discrete_state_input=False,
    paligemma_variant="dummy",
    action_expert_variant="dummy_moe_4",
    vision_variant="mu/32",
    dtype="float32",
    max_token_len=8,
)


def _latent_config(**kw):
    values = {**_COMMON, "z_dim": 8, "enc_hidden": 32, "prior_hidden": 32}
    values.update(kw)
    return pi0_config_skill_latent.Pi0SkillLatentConfig(**values)


_MODELS = {}


def _seed_dummy_action_gates(model):
    """Use nonzero gates like the trained pi0.5 checkpoint; tiny dummy gates default to zero."""
    state = nnx.state(model.PaliGemma.llm, nnx.Param)
    flat = traverse_util.flatten_dict(state.to_pure_dict())
    for path, value in flat.items():
        name = "/".join(map(str, path))
        if "_1/Dense_0/bias" in name and ("pre_attention_norm_1" in name or "pre_ffw_norm_1" in name):
            width = value.shape[-1] // 3
            flat[path] = value.at[..., 2 * width :].set(0.1)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
    nnx.update(model.PaliGemma.llm, state)


def _model_for(key):
    if key not in _MODELS:
        cfgs = {
            "base": pi0_config_moe_skill.Pi0Config(**_COMMON),
            "disabled": _latent_config(z_dim=0, use_skill_ids=True),
            "main": _latent_config(),
            "router": _latent_config(z_inject="router"),
            "token": _latent_config(z_inject="token"),
            "ids_z": _latent_config(use_skill_ids=True),
        }
        model = cfgs[key].create(jax.random.key(0))
        if getattr(cfgs[key], "z_enabled", False):
            # The real pi0.5 start checkpoint has trained AdaRMS gates. Dummy variants initialize them to zero,
            # which blocks every action-expert residual and makes any conditioning test vacuous.
            _seed_dummy_action_gates(model)
        _MODELS[key] = (cfgs[key], model)
    return _MODELS[key]


def _obs(cfg, *, skills=False, seed=0, prompt_seed=0):
    rng = np.random.default_rng(seed)
    obs = cfg.fake_obs(B)
    images = {k: jnp.asarray(rng.uniform(-1, 1, v.shape), jnp.float32) for k, v in obs.images.items()}
    prompt = jnp.asarray(np.random.default_rng(100 + prompt_seed).integers(1, 1000, (B, cfg.max_token_len)), jnp.int32)
    kw = dict(
        images=images,
        image_masks=obs.image_masks,
        state=jnp.asarray(rng.normal(size=obs.state.shape), jnp.float32),
        tokenized_prompt=prompt,
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
    if skills:
        kw.update(skills=jnp.asarray([[1, 2, 3], [4, 5, 0]], jnp.int32), skill_mask=jnp.asarray([[1, 1, 1], [1, 1, 0]], bool))
    return _model.Observation(**kw)


def _actions(cfg, seed=1):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(B, cfg.action_horizon, cfg.action_dim)), jnp.float32)


def _flat(tree):
    return traverse_util.flatten_dict(tree, sep="/")


# ---------------------------------------------------------------- (a) disabled == existing model
def test_disabled_latent_is_exactly_the_existing_model():
    base_cfg, base = _model_for("base")
    cfg, latent = _model_for("disabled")
    base_params = nnx.state(base, nnx.Param).to_pure_dict()
    latent_state = nnx.state(latent, nnx.Param)
    assert set(_flat(base_params)) == set(_flat(latent_state.to_pure_dict()))
    latent_state.replace_by_pure_dict(base_params)
    nnx.update(latent, latent_state)

    obs = _obs(cfg, skills=True)
    actions = _actions(cfg)
    rng = jax.random.key(3)
    l0, b0 = base.compute_loss(rng, obs, actions, train=False)
    l1, b1 = latent.compute_loss(rng, obs, actions, train=False)
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    np.testing.assert_array_equal(np.asarray(b0), np.asarray(b1))
    a0 = base.sample_actions(rng, obs, num_steps=NUM_STEPS)
    a1 = latent.sample_actions(rng, obs, num_steps=NUM_STEPS)
    np.testing.assert_array_equal(np.asarray(a0), np.asarray(a1))


def test_default_configs_and_observation_payload_are_backward_compatible():
    assert pi0_config_moe_skill.Pi0Config().vision_variant == "So400m/14"
    obs = _model.Observation.from_dict(
        {
            "image": {"base_0_rgb": np.zeros((1, 4, 4, 3), np.float32)},
            "image_mask": {"base_0_rgb": np.ones((1,), bool)},
            "state": np.zeros((1, 5), np.float32),
            "skill_z": np.ones((1, 3), np.float32),
        }
    )
    assert obs.skill_z.shape == (1, 3)
    no_z = _model.Observation.from_dict(
        {"image": {"base_0_rgb": np.zeros((1, 4, 4, 3), np.float32)}, "image_mask": {"base_0_rgb": np.ones((1,), bool)},
         "state": np.zeros((1, 5), np.float32)}
    )
    assert no_z.skill_z is None
    with pytest.raises(ValueError):
        _latent_config(z_dim=0, use_skill_ids=False)


# ---------------------------------------------------------------- (b) losses and gradients
def _grads(model, loss_fn):
    return nnx.grad(loss_fn)(model)


def _nonzero(x):
    return bool(np.any(np.asarray(x) != 0))


@pytest.mark.parametrize("key", ["main", "ids_z"])
def test_training_step_has_kl_and_reconstruction_and_grads_reach_enc_prior_injection(key):
    cfg, model = _model_for(key)
    obs = _obs(cfg, skills=cfg.use_skill_ids)
    actions = _actions(cfg)
    rng = jax.random.key(4)

    recon, terms = model.compute_loss_terms(rng, obs, actions, train=True)
    assert recon.shape == (B, H)
    assert np.isfinite(np.asarray(recon)).all()
    assert float(terms["kl"]) > 0
    expected_aux = cfg.balance_loss_weight * terms["balance_loss"] + cfg.kl_beta * terms["kl_loss"]
    np.testing.assert_allclose(float(terms["aux_loss"]), float(expected_aux), rtol=1e-6)
    _, aux = model.compute_loss(rng, obs, actions, train=True)
    np.testing.assert_allclose(float(aux), float(terms["aux_loss"]), rtol=1e-6)

    def loss_fn(m):
        r, a = m.compute_loss(rng, obs, actions, train=True)
        return jnp.mean(r) + a

    g = _grads(model, loss_fn).skill_latent
    assert _nonzero(g.encoder.fc1.kernel.value)
    assert _nonzero(g.prior.fc1.kernel.value)
    assert _nonzero(g.router_tokens.proj.kernel.value)
    assert _nonzero(g.token.fc1.kernel.value)


def test_kl_gradient_does_not_reach_the_vlm_and_reconstruction_trains_the_encoder():
    cfg, model = _model_for("main")
    obs, actions, rng = _obs(cfg), _actions(cfg), jax.random.key(5)

    def kl_only(m):
        return m.compute_loss_terms(rng, obs, actions, train=True)[1]["kl_loss"]

    g = _grads(model, kl_only)
    assert _nonzero(g.skill_latent.prior.fc1.kernel.value)
    assert _nonzero(g.skill_latent.encoder.fc1.kernel.value)
    vlm = _flat(g.PaliGemma.to_pure_dict())
    assert not any(_nonzero(v) for k, v in vlm.items() if "_1" not in k and "embedder" not in k)

    def recon_only(m):
        return jnp.mean(m.compute_loss_terms(rng, obs, actions, train=True)[0])

    g = _grads(model, recon_only)
    assert _nonzero(g.skill_latent.encoder.fc1.kernel.value)
    assert not _nonzero(g.skill_latent.prior.fc1.kernel.value)


def test_freeze_paligemma_filter_keeps_action_side_trainable():
    cfg = _latent_config(freeze_paligemma=True)
    _, model = _model_for("main")
    frozen_filter = cfg.get_freeze_filter()
    params = nnx.state(model, nnx.Param)
    frozen = {"/".join(map(str, p)) for p, _ in params.filter(frozen_filter).flat_state().items()}
    trainable = {"/".join(map(str, p)) for p, _ in params.filter(nnx.Not(frozen_filter)).flat_state().items()}
    assert any(p.startswith("PaliGemma/img") for p in frozen)
    assert any("PaliGemma/llm/layers/attn/q_einsum/" in p for p in frozen)
    assert not any(p.startswith("PaliGemma/img") for p in trainable)
    for must in ("skill_latent/encoder", "skill_latent/prior", "skill_latent/token", "skill_latent/router_tokens",
                 "moe_ffn_1", "q_einsum_1", "action_in_proj", "action_out_proj", "time_mlp_in"):
        assert any(must in p for p in trainable), must
        assert not any(must in p for p in frozen), must
    assert any("skill_embedder" in p for p in frozen)
    assert not any("skill_embedder" in p for p in trainable)
    assert _latent_config(freeze_paligemma=False).get_freeze_filter() == nnx.Nothing


# ---------------------------------------------------------------- (c) inference with prior z / external z
def test_sample_actions_with_prior_and_external_z():
    cfg, model = _model_for("main")
    obs = _obs(cfg)
    rng = jax.random.key(6)
    noise = jax.random.normal(jax.random.key(7), (B, H, cfg.action_dim))

    a_prior = model.sample_actions(rng, obs, num_steps=NUM_STEPS, noise=noise)
    assert a_prior.shape == (B, H, cfg.action_dim) and np.isfinite(np.asarray(a_prior)).all()

    p_mean, p_logvar, pooled = model.infer_prior(obs)
    assert p_mean.shape == (B, cfg.z_dim) and pooled.shape[0] == B
    a_kw = model.sample_actions(rng, obs, num_steps=NUM_STEPS, noise=noise, z=p_mean)
    np.testing.assert_allclose(np.asarray(a_kw), np.asarray(a_prior), atol=1e-5)
    a_obs = model.sample_actions(rng, dataclasses.replace(obs, skill_z=p_mean), num_steps=NUM_STEPS, noise=noise)
    np.testing.assert_allclose(np.asarray(a_obs), np.asarray(a_prior), atol=1e-5)

    ext = jnp.asarray(np.random.default_rng(8).normal(size=(B, cfg.z_dim)), jnp.float32)
    a_ext = model.sample_actions(rng, obs, num_steps=NUM_STEPS, noise=noise, z=ext)
    assert np.isfinite(np.asarray(a_ext)).all()
    # kwarg wins over the observation field
    a_both = model.sample_actions(rng, dataclasses.replace(obs, skill_z=p_mean), num_steps=NUM_STEPS, noise=noise, z=ext)
    np.testing.assert_allclose(np.asarray(a_both), np.asarray(a_ext), atol=1e-6)

    # jit path used by Policy
    jitted = nnx_utils.module_jit(model.sample_actions, static_argnames=("capture_auxiliary",))
    a_jit = jitted(rng, obs, num_steps=NUM_STEPS, noise=noise)
    np.testing.assert_allclose(np.asarray(a_jit), np.asarray(a_prior), atol=1e-4)


def test_capture_auxiliary_still_works_with_z():
    cfg, model = _model_for("main")
    out = model.sample_actions(jax.random.key(9), _obs(cfg), num_steps=2, capture_auxiliary=True)
    assert len(out) == 5
    assert out[1].shape[2] == H + 1  # z token + actions


def test_memory_key_features_language_free_pass():
    cfg, model = _model_for("main")
    obs_a, obs_b = _obs(cfg, prompt_seed=0), _obs(cfg, prompt_seed=1)
    fa, fb = model.memory_key_features(obs_a), model.memory_key_features(obs_b)
    np.testing.assert_array_equal(np.asarray(fa["image_pool_no_language"]), np.asarray(fb["image_pool_no_language"]))
    assert not np.allclose(np.asarray(fa["image_pool"]), np.asarray(fb["image_pool"]))
    # The prior reads the normal-input image pool.
    _, _, pooled = model.infer_prior(obs_a)
    np.testing.assert_allclose(np.asarray(pooled), np.asarray(fa["image_pool"]), atol=1e-6)
    # Fixed generic sentence variant.
    neutral = (jnp.ones((B, 4), jnp.int32), jnp.ones((B, 4), bool))
    fn = model.memory_key_features(obs_a, neutral_prompt=neutral)
    assert fn["image_pool_no_language"].shape == fa["image_pool"].shape


# ---------------------------------------------------------------- (d) z changes the output, (e) injection switches
@pytest.mark.parametrize("key", ["main", "router", "token", "ids_z"])
def test_changing_z_changes_actions(key):
    cfg, model = _model_for(key)
    obs = _obs(cfg, skills=cfg.use_skill_ids)
    rng = jax.random.key(10)
    noise = jax.random.normal(jax.random.key(11), (B, H, cfg.action_dim))
    z1 = jnp.asarray(np.random.default_rng(12).normal(size=(B, cfg.z_dim)) * 2, jnp.float32)
    z2 = jnp.asarray(np.random.default_rng(13).normal(size=(B, cfg.z_dim)) * 2, jnp.float32)
    a1 = np.asarray(model.sample_actions(rng, obs, num_steps=NUM_STEPS, noise=noise, z=z1))
    a2 = np.asarray(model.sample_actions(rng, obs, num_steps=NUM_STEPS, noise=noise, z=z2))
    assert np.max(np.abs(a1 - a2)) > 1e-4


def test_injection_switches():
    expected = {"main": (True, True), "router": (True, False), "token": (False, True)}
    for key, (has_router, has_token) in expected.items():
        cfg, model = _model_for(key)
        assert (model.skill_latent.router_tokens is not None) == has_router
        assert (model.skill_latent.token is not None) == has_token
        obs = _obs(cfg)
        z = jnp.zeros((B, cfg.z_dim))
        tokens, input_mask, ar_mask, _ = model._embed_suffix_z(obs, _actions(cfg), jnp.ones(B), z)
        assert tokens.shape[1] == H + int(has_token)
        if has_token:
            # pi05: z forms its own block before the actions; actions keep their block rule.
            np.testing.assert_array_equal(np.asarray(ar_mask), np.asarray([True, True] + [False] * (H - 1)))
        else:
            np.testing.assert_array_equal(np.asarray(ar_mask), np.asarray([True] + [False] * (H - 1)))
        keys, mask = model._router_inputs(obs, z)
        if has_router:
            assert keys.shape[1] == cfg.z_router_tokens and bool(mask.all())
        else:
            assert not bool(mask.any())  # uniform router, no skill IDs

    # skill IDs + z: z tokens appended after the skill-ID keys, ID mask preserved.
    cfg, model = _model_for("ids_z")
    obs = _obs(cfg, skills=True)
    keys, mask = model._router_inputs(obs, jnp.zeros((B, cfg.z_dim)))
    assert keys.shape[1] == 3 + cfg.z_router_tokens
    np.testing.assert_array_equal(np.asarray(mask[:, :3]), np.asarray(obs.skill_mask))


def test_pi0_non_pi05_z_token_joins_the_state_block():
    cfg = _latent_config(pi05=False)
    model = cfg.create(jax.random.key(0))
    tokens, _, ar_mask, _ = model._embed_suffix_z(_obs(cfg), _actions(cfg), jnp.ones(B), jnp.zeros((B, cfg.z_dim)))
    assert tokens.shape[1] == H + 2
    np.testing.assert_array_equal(np.asarray(ar_mask), np.asarray([True, False, True] + [False] * (H - 1)))


# ---------------------------------------------------------------- KL utilities
def test_kl_utilities():
    m = jnp.asarray([[0.3, -0.2]])
    lv = jnp.asarray([[0.1, -0.5]])
    np.testing.assert_allclose(np.asarray(_sl.gaussian_kl(m, lv, m, lv)), 0.0, atol=1e-7)
    kl_loss, kl_raw = _sl.kl_objective(m, lv, jnp.zeros_like(m), jnp.zeros_like(lv))
    np.testing.assert_allclose(float(kl_loss), float(kl_raw), rtol=1e-6)
    fb_loss, _ = _sl.kl_objective(m, lv, m, lv, free_bits=0.5)
    np.testing.assert_allclose(float(fb_loss), 1.0, rtol=1e-6)
    bal_loss, _ = _sl.kl_objective(m, lv, jnp.zeros_like(m), jnp.zeros_like(lv), balance=0.8)
    np.testing.assert_allclose(float(bal_loss), float(kl_raw), rtol=1e-6)


def test_research_training_configs():
    import openpi.training.config_moe_skill as config

    main = config.get_config("pi05_libero90_emergent_z")
    assert isinstance(main.model, pi0_config_skill_latent.Pi0SkillLatentConfig)
    assert main.model.action_horizon == 25
    assert main.model.use_skill_ids is False and main.model.z_inject == "router_token"
    assert main.model.freeze_paligemma and main.model.balance_loss_weight == 1.0
    assert isinstance(main.data, config.LeRobotLiberoDataConfig)  # no skill-annotation column
    assert main.data.repo_id == config.LIBERO90_REPO_ID
    assert main.data.base_config.root == config.EMERGENT_DATASET_ROOT
    assert main.freeze_filter == main.model.get_freeze_filter()
    router = config.get_config("pi05_libero90_emergent_z_router_only")
    assert router.model.z_inject == "router" and router.model.action_horizon == 25
    assert router.model.balance_loss_weight == 1.0
    short = config.get_config("pi05_libero90_emergent_z_h10")
    assert short.model.z_inject == "router_token" and short.model.action_horizon == 10
    assert short.model.balance_loss_weight == 1.0
    ids = config.get_config("pi05_libero90_skillid_z")
    assert ids.model.action_horizon == 25
    assert ids.model.use_skill_ids and isinstance(ids.data, config.LeRobotLiberoSkillDataConfig)
    assert ids.model.freeze_paligemma and ids.model.balance_loss_weight == 1.0
    assert ids.weight_loader.missing_regex == ".*skill_latent.*"
    assert config._PUBLIC_CONFIG_NAMES == tuple(c.name for c in config._CONFIGS)
