#!/usr/bin/env python3
"""LoRA follow-up for observation-derived pose regression.

The frozen-MLP condition is selected on the fixed unseen Dev split first.  The
selected representation/loss is then held constant for VLM and action-expert
LoRA runs, each with both cached C representations.  Eval is only scored.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".40")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import theta_regression as tr  # noqa: E402

OUT = Path(os.environ.get("THETA_LORA_OUT", tr.OUT / "lora" / "production"))
REPRESENTATIONS = ("C_vm_none", "C_vm_08")
SPLITS = ("train", "dev", "eval")
STEPS = 4000
BATCH = 8
SEED = 20261104
LR = 1e-4
WIDTH = 256
RANK = 16
VLM_LORA_LAYERS = 6


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    tmp.replace(path)


def _npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def _load_unseen():
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
            for p in (tr.UNSEEN / "unseen_rows").glob("*.npy")}


def validate_libero40_train_rows(train_ids, task, validation, *, expected_count=23304):
    """Fail closed unless the cache is exactly the LIBERO-40 ~validation split."""
    ids = np.asarray(train_ids, np.int64)
    task = np.asarray(task, np.int64)
    validation = np.asarray(validation, bool)
    if task.ndim != 1 or validation.shape != task.shape:
        raise ValueError("task/validation must be aligned vectors")
    if np.any((ids < 0) | (ids >= len(task))):
        raise ValueError("training row ID outside LIBERO-40 rows")
    expected = np.flatnonzero((task < 40) & ~validation)
    if len(ids) != expected_count or not np.array_equal(ids, expected):
        raise ValueError("training IDs must exactly equal LIBERO-40 ~validation rows in source order")
    return ids


def _decode_rows(shapes, theta6, theta_mean, theta_std, vs):
    """NumPy path used for reporting and baselines; decoder matches exp1."""
    a = np.asarray(shapes, np.float32)
    p = np.asarray(theta6, np.float32) * theta_std + theta_mean
    out = np.zeros_like(a, dtype=np.float32)
    for i in range(len(a)):
        for t in range(a.shape[1]):
            if not np.isfinite(p[i, t]).all():
                continue
            from scipy.spatial.transform import Rotation
            mat = tr.rot6_matrix(p[i, t, :6])
            scale = float(np.exp(np.clip(p[i, t, 6], -5., 5.)))
            physical = np.r_[a[i, t, :3] @ mat, a[i, t, 3:6] @ mat] * scale
            out[i, t, :6] = vs.to_normalized(physical)
            out[i, t, 6] = a[i, t, 6]
    return out


def select_exp1_spec():
    """Pick one exp1 representation/loss using only the designated 113-row Dev."""
    decision_path = OUT / "selection" / "exp1_dev_selection.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    import torch
    torch.set_num_threads(8)
    import variable_shape_memory as vs

    base, rows, unseen, splits = tr._splits()
    source, ids = splits["dev"]
    actions = np.asarray(source["actions"][ids], np.float32)[:, :, :7]
    features = np.load(tr.OUT / "production" / "features" / "dev.npy", mmap_mode="r")
    outcomes = []
    for rep in REPRESENTATIONS:
        cache = np.load(tr.SOURCE / "drafts" / f"{rep}_dev.npz", allow_pickle=False)
        for variant in tr.VARIANTS:
            kind = "aa" if variant == "R-aa-theta" else "6d"
            cam = "nowrist" if variant.endswith("nowrist") else (
                "wristonly" if variant.endswith("wristonly") else "all")
            x, _, meta, _ = tr._make_data(
                cache, features, np.asarray(source["state"][ids], np.float32), kind, cam)
            mp = tr.OUT / "production" / "models" / rep / variant / "model.pt"
            if not mp.exists():
                raise FileNotFoundError(f"missing exp1 model needed for Dev-only selection: {mp}")
            pred_theta = tr._predict_model(mp, x)
            prediction = np.zeros_like(actions)
            for j, (row, _uid, shape, pos, _target, _src) in enumerate(meta):
                prediction[row, pos] = tr.decode_theta(shape[:len(pos)], pred_theta[j], vs)
            mse = float(np.mean((prediction - actions) ** 2))
            outcomes.append({"representation": rep, "variant": variant, "cam_mode": cam,
                             "theta_kind": kind, "dev_active7_mse": mse,
                             "rows": len(ids), "source": "fixed unseen Dev only"})
    selected = min(outcomes, key=lambda x: x["dev_active7_mse"])
    result = {"selected": selected, "candidates": outcomes,
              "policy": "minimum 113-row Dev active7 MSE; no Eval actions or labels used"}
    _json(decision_path, result)
    return result


def _overlay_pretrained(target, source, path=()):
    """Overlay dense checkpoint leaves, preserving newly-created LoRA leaves."""
    if isinstance(target, dict):
        if not isinstance(source, dict):
            raise ValueError(f"tree mismatch at {'.'.join(path)}")
        result = {}
        for key, value in target.items():
            if key in source:
                result[key] = _overlay_pretrained(value, source[key], path + (str(key),))
            elif "lora" in ".".join(path + (str(key),)).lower() or key in ("w_a", "w_b"):
                result[key] = value
            else:
                raise ValueError(f"base checkpoint missing {'/'.join(path + (str(key),))}")
        return result
    ta, sa = np.asarray(target), np.asarray(source)
    if ta.shape != sa.shape:
        raise ValueError(f"dense checkpoint shape mismatch at {'/'.join(path)}: {ta.shape} != {sa.shape}")
    return source


def _zero_lora_b(tree, path=()):
    """Zero LoRA B factors so the initialized model equals its dense checkpoint."""
    if isinstance(tree, dict):
        return {k: _zero_lora_b(v, path + (str(k),)) for k, v in tree.items()}
    leaf = path[-1].lower() if path else ""
    parent = ".".join(path).lower()
    if leaf in ("w_b", "lora_b", "gating_einsum_lora_b", "linear_lora_b") or leaf.endswith("lora_b"):
        return np.zeros_like(np.asarray(tree))
    return tree


def mask_early_lora_gradients(pure, *, train_last=VLM_LORA_LAYERS, depth=18, path=()):
    """For scanned Gemma LoRA leaves, freeze all but the final layers."""
    if isinstance(pure, dict):
        return {k: mask_early_lora_gradients(v, train_last=train_last, depth=depth,
                                             path=path + (str(k),)) for k, v in pure.items()}
    a = np.asarray(pure)
    joined = ".".join(path).lower()
    leaf = path[-1].lower() if path else ""
    if "layers" in joined and ("lora" in joined or leaf in ("w_a", "w_b")) and a.ndim >= 2 and a.shape[0] == depth:
        # This is a host-side utility used only in the test/report path. The
        # training implementation below uses the equivalent JAX tree mask.
        out = a.copy()
        out[:depth - train_last] = 0
        return out
    return pure


def _jax_mask_late_gradients(grads, *, train_last=VLM_LORA_LAYERS, depth=18):
    pure = grads.to_pure_dict()
    def rec(node, path=()):
        import jax.numpy as jnp
        if isinstance(node, dict):
            return {k: rec(v, path + (str(k),)) for k, v in node.items()}
        a = jnp.asarray(node)
        joined = ".".join(path).lower(); leaf = path[-1].lower() if path else ""
        if ("layers" in joined and ("lora" in joined or leaf in ("w_a", "w_b"))
                and a.ndim >= 2 and a.shape[0] == depth):
            return a.at[:depth - train_last].set(0)
        return node
    grads.replace_by_pure_dict(rec(pure))
    return grads


def _load_rows_for_lora():
    base_mod = tr._module(tr.PI05 / "run.py", "theta_lora_data")
    return base_mod, base_mod.load_rows(tr.RUN)


def _make_model(architecture, *, seed=SEED):
    """Create Pi0.5 with rank-16 LoRA on PaliGemma or action expert."""
    import flax.nnx as nnx
    import flax.nnx.bridge as nnx_bridge
    import jax
    import openpi.models.gemma as gemma
    import openpi.models.pi0 as pi0
    import openpi.models.pi0_config as config_mod
    import openpi.models.siglip as siglip
    from openpi.models import model as model_api
    from openpi.models.lora import LoRAConfig

    lora_cfg = {"attn": LoRAConfig(rank=RANK, alpha=RANK),
                "ffn": LoRAConfig(rank=RANK, alpha=RANK)}
    paligemma_name = "gemma_2b_lora" if architecture == "L-vlm" else "gemma_2b"
    action_name = "gemma_300m" if architecture == "L-vlm" else "gemma_300m_lora"
    dense_paligemma = gemma.get_config(paligemma_name)
    dense_action = gemma.get_config(action_name)
    if architecture == "L-expert":
        dense_action = dataclasses.replace(dense_action, lora_configs=lora_cfg)

    class LoRAPi0(pi0.Pi0):
        def __init__(self, cfg, rngs):
            model_api.BaseModel.__init__(self, cfg.action_dim, cfg.action_horizon, cfg.max_token_len)
            self.pi05 = True
            llm = nnx_bridge.ToNNX(gemma.Module(
                configs=[dense_paligemma, dense_action], embed_dtype=cfg.dtype, adarms=True))
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
            img = nnx_bridge.ToNNX(siglip.Module(
                num_classes=dense_paligemma.width, variant="So400m/14", pool_type="none",
                scan=True, dtype_mm=cfg.dtype))
            img.lazy_init(next(iter(cfg.fake_obs().images.values())), train=False, rngs=rngs)
            self.PaliGemma = nnx.Dict(llm=llm, img=img)
            self.action_in_proj = nnx.Linear(cfg.action_dim, dense_action.width, rngs=rngs)
            self.time_mlp_in = nnx.Linear(dense_action.width, dense_action.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(dense_action.width, dense_action.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(dense_action.width, cfg.action_dim, rngs=rngs)
            self.deterministic = True

    cfg = config_mod.Pi0Config(pi05=True, action_dim=32, action_horizon=10,
                               max_token_len=200, discrete_state_input=False,
                               paligemma_variant=paligemma_name, action_expert_variant=action_name)
    model = LoRAPi0(cfg, rngs=nnx.Rngs(seed))
    return model


def _restore_dense_checkpoint(model):
    import flax.nnx as nnx
    from openpi.models.model import restore_params
    import jax
    target = nnx.state(model).to_pure_dict()
    source = restore_params(tr.CHECKPOINT / "params", restore_type=np.ndarray)
    merged = _overlay_pretrained(target, source)
    merged = _zero_lora_b(merged)
    state = nnx.state(model)
    state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray, merged))
    nnx.update(model, state)
    return {"dense_leaves": sum(1 for _ in jax.tree.leaves(source)),
            "lora_zero_initialized": True}


def _wrapper_type():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    class ThetaLoRA(nnx.Module):
        def __init__(self, base, architecture, input_width, cam_mode):
            self.base = base
            self.architecture = architecture
            self.cam_mode = cam_mode
            self.theta_head1 = nnx.Linear(input_width, WIDTH, rngs=nnx.Rngs(391))
            # Start at the train-set mean pose (zero in standardized target
            # coordinates), avoiding an unstable random pose/scale at init.
            self.theta_head2 = nnx.Linear(WIDTH, 7,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(), rngs=nnx.Rngs(392))
            self.theta_mean = nnx.Variable(jnp.zeros((7,), jnp.float32))
            self.theta_std = nnx.Variable(jnp.ones((7,), jnp.float32))

        def _features(self, obs, shape, unit_ids, theta_src):
            obs = _mask_observation(obs, self.cam_mode)
            base = self.base
            if self.architecture == "L-vlm":
                prefix, valid, ar = base.embed_prefix(obs)
                (hidden, _), _ = base.PaliGemma.llm(
                    [prefix, None], mask=make_attn_mask(valid, ar),
                    positions=jnp.cumsum(valid, axis=1) - 1)
                prompt_len = obs.tokenized_prompt.shape[1]
                image_len = (prefix.shape[1] - prompt_len) // 3
                means = []
                for k, name in enumerate(("base_0_rgb", "left_wrist_0_rgb")):
                    sl = hidden[:, k * image_len:(k + 1) * image_len]
                    mask = obs.image_masks[name][:, None, None].astype(sl.dtype)
                    means.append((sl * mask).sum(1) / jnp.maximum(mask.sum(1), 1.))
                text = hidden[:, 3 * image_len:]
                tm = obs.tokenized_prompt_mask[..., None].astype(text.dtype)
                means.append((text * tm).sum(1) / jnp.maximum(tm.sum(1), 1.))
                pooled = jnp.concatenate(means, axis=-1)
                if self.cam_mode == "wristonly":
                    pooled = pooled.at[:, :2048].set(0).at[:, 4096:].set(0)
                elif self.cam_mode == "nowrist":
                    pooled = pooled.at[:, 2048:4096].set(0)
                meta = _unit_shape_features(shape, unit_ids)
                x = jnp.concatenate([jnp.broadcast_to(pooled[:, None], (len(shape), 10, pooled.shape[-1])),
                                     jnp.broadcast_to(obs.state[:, None], (len(shape), 10, obs.state.shape[-1])),
                                     meta], axis=-1)
            else:
                prefix, pmask, par = base.embed_prefix(obs)
                _, cache = base.PaliGemma.llm([prefix, None], mask=make_attn_mask(pmask, par),
                    positions=jnp.cumsum(pmask, axis=1) - 1)
                shape_action = _decode_source_jax(shape, theta_src)
                suffix, smask, sar, cond = base.embed_suffix(obs, shape_action, jnp.full((len(shape),), .5))
                full_mask = jnp.concatenate([
                    jnp.broadcast_to(pmask[:, None, :], (len(shape), suffix.shape[1], pmask.shape[1])),
                    make_attn_mask(smask, sar)], axis=-1)
                pos = jnp.sum(pmask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1
                (_, hidden), _ = base.PaliGemma.llm([None, suffix], mask=full_mask, positions=pos,
                    kv_cache=cache, adarms_cond=[None, cond])
                hidden = hidden[:, -10:]
                hidden = _unit_group_mean(hidden, unit_ids)
                meta = _unit_shape_features(shape, unit_ids)
                state = jnp.broadcast_to(obs.state[:, None], (len(shape), 10, obs.state.shape[-1]))
                x = jnp.concatenate([hidden, state, meta], axis=-1)
            return _unit_group_mean(x, unit_ids)

        def __call__(self, obs, shape, unit_ids, theta_src):
            import jax
            x = self._features(obs, shape, unit_ids, theta_src)
            return _unit_group_mean(self.theta_head2(jax.nn.gelu(self.theta_head1(x))), unit_ids)
    return ThetaLoRA


def _mask_observation(obs, cam_mode):
    if cam_mode == "all":
        return obs
    images = dict(obs.image_masks)
    if cam_mode == "wristonly":
        images["base_0_rgb"] = images["base_0_rgb"] & False
        images["right_wrist_0_rgb"] = images["right_wrist_0_rgb"] & False
        prompt_mask = obs.tokenized_prompt_mask & False
    elif cam_mode == "nowrist":
        images["left_wrist_0_rgb"] = images["left_wrist_0_rgb"] & False
        images["right_wrist_0_rgb"] = images["right_wrist_0_rgb"] & False
        prompt_mask = obs.tokenized_prompt_mask
    else:
        raise ValueError(cam_mode)
    return obs.replace(image_masks=images, tokenized_prompt_mask=prompt_mask)


def _unit_group_mean(x, unit_ids):
    import jax.numpy as jnp
    eq = (unit_ids[:, :, None] == unit_ids[:, None, :]) & (unit_ids[:, :, None] >= 0)
    denom = jnp.maximum(eq.sum(-1, keepdims=True), 1)
    return jnp.einsum("bij,bjd->bid", eq.astype(x.dtype), x) / denom


def _unit_shape_features(shape, unit_ids):
    import jax.numpy as jnp
    # Three DCT-II coefficients per channel over the window positions occupied
    # by each unit, plus unit length, mean phase, and source pose.
    n = shape.shape[1]
    t = jnp.arange(n, dtype=jnp.float32)
    k = jnp.arange(3, dtype=jnp.float32)[:, None]
    basis = jnp.cos(jnp.pi / n * (t[None, :] + .5) * k)
    basis = basis.at[0].multiply(1 / jnp.sqrt(n))
    basis = basis.at[1:].multiply(jnp.sqrt(2 / n))
    eq = (unit_ids[:, :, None] == unit_ids[:, None, :]) & (unit_ids[:, :, None] >= 0)
    coeff = jnp.einsum("bqj,bjd,kj->bqkd", eq.astype(shape.dtype), shape, basis)
    count = jnp.maximum(eq.sum(-1, keepdims=True), 1)
    pos = jnp.broadcast_to(t[None, :], unit_ids.shape)
    phase = ((eq * pos[:, None, :]).sum(-1, keepdims=True)
             / jnp.maximum(count, 1) / max(n - 1, 1))
    return jnp.concatenate([coeff.reshape((len(shape), n, -1)),
                            jnp.log1p(count.astype(shape.dtype)), phase.astype(shape.dtype)], axis=-1)


def _decode_source_jax(shape, theta_src):
    import jax.numpy as jnp
    # The cached source pose is constant input; reproduce exp1's axis-angle path.
    th = theta_src
    v = th[..., :3]
    angle = jnp.linalg.norm(v, axis=-1, keepdims=True)
    x, y, z = jnp.moveaxis(v, -1, 0)
    zero = jnp.zeros_like(x)
    K = jnp.stack([jnp.stack([zero, -z, y], -1), jnp.stack([z, zero, -x], -1),
                   jnp.stack([-y, x, zero], -1)], -2)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=v.dtype), K.shape)
    A = jnp.where(angle > 1e-5, jnp.sin(angle) / jnp.maximum(angle, 1e-5), 1.)[..., None]
    B = jnp.where(angle > 1e-5, (1 - jnp.cos(angle)) / jnp.maximum(angle**2, 1e-10), .5)[..., None]
    r = eye + A * K + B * (K @ K)
    scale = jnp.exp(jnp.clip(th[..., 3:4], -5., 5.))
    p = jnp.einsum("bti,btij->btj", shape[:, :, :3], r) * scale
    rot = jnp.einsum("bti,btij->btj", shape[:, :, 3:6], r) * scale
    motion = jnp.concatenate([p, rot], -1)
    import variable_shape_memory as vs
    lo = jnp.asarray(vs.LOW, jnp.float32); span = jnp.asarray(vs.SPAN, jnp.float32)
    norm = (motion - lo) / (span + 1e-6) * 2 - 1
    return jnp.zeros((len(shape), 10, 32), jnp.float32).at[:, :, :6].set(norm).at[:, :, 6].set(shape[:, :, 6])


def _decode_pred_jax(shape, pred_norm, ymean, ystd):
    import jax.numpy as jnp
    pred = pred_norm * ystd + ymean
    a, b = pred[:, :, :3], pred[:, :, 3:6]
    x = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    y0 = b - jnp.sum(x * b, axis=-1, keepdims=True) * x
    y = y0 / jnp.maximum(jnp.linalg.norm(y0, axis=-1, keepdims=True), 1e-8)
    z = jnp.cross(x, y)
    r = jnp.stack([x, y, z], axis=-1)
    scale = jnp.exp(jnp.clip(pred[:, :, 6:7], -5, 5))
    xyz = jnp.einsum("bti,btij->btj", shape[:, :, :3], r) * scale
    rot = jnp.einsum("bti,btij->btj", shape[:, :, 3:6], r) * scale
    motion = jnp.concatenate([xyz, rot], -1)
    import variable_shape_memory as vs
    lo = jnp.asarray(vs.LOW, jnp.float32); span = jnp.asarray(vs.SPAN, jnp.float32)
    action = (motion - lo) / (span + 1e-6) * 2 - 1
    return jnp.concatenate([action, shape[:, :, 6:7]], axis=-1)


def _aa_targets(theta):
    return np.stack([tr._target_theta_repr(x, "6d") for x in np.asarray(theta).reshape(-1, 4)]).reshape(*theta.shape[:-1], 7)


def _trainable_filter():
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    return nnx.All(nnx.Param, PathRegex(r".*(lora|theta_head|w_[ab]).*"))


def _condition_dir(architecture, rep):
    return OUT / "models" / architecture / rep


def _train_one(architecture, rep, *, steps=STEPS, threads=8, device="cuda", smoke=False):
    import torch
    torch.set_num_threads(threads)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex

    spec = select_exp1_spec()["selected"]
    base_mod, rows = _load_rows_for_lora()
    train_cache = np.load(tr.SOURCE / "drafts" / f"{rep}_train.npz", allow_pickle=False)
    train_ids = np.asarray(train_cache["row_ids"], np.int64)
    task = np.asarray(rows["task"]); validation = np.asarray(rows["validation"], bool)
    train_ids = validate_libero40_train_rows(train_ids, task, validation)
    if smoke:
        train_ids = train_ids[:16]
        train_cache = {k: train_cache[k][:16] for k in train_cache.files}
    dev_cache = np.load(tr.SOURCE / "drafts" / f"{rep}_dev.npz", allow_pickle=False)
    if not np.array_equal(train_cache["row_ids"], train_ids):
        raise AssertionError("training cache row alignment changed")
    unseen = _load_unseen()
    dev_source = unseen
    dev_ids = np.asarray(dev_cache["row_ids"], np.int64)
    expected_dev_ids = np.flatnonzero(np.asarray(unseen["is_dev"], bool))
    if not np.array_equal(dev_ids, expected_dev_ids):
        raise AssertionError("checkpoint selection rows must exactly match the fixed Dev split")
    if smoke:
        dev_ids = dev_ids[:8]
        dev_cache = {k: dev_cache[k][:8] for k in dev_cache.files}

    model = _make_model(architecture)
    restore_info = _restore_dense_checkpoint(model)
    filt = _trainable_filter()
    selected_state = nnx.state(model, filt)
    if not selected_state.flat_state():
        raise AssertionError("LoRA/head filter selected no trainable parameters")
    # The head is attached outside the Pi0 module after base checkpoint load.
    # Hidden representation widths: pooled three-group VLM, or action expert tokens.
    head_input = (3 * 2048 + 32 + 23) if architecture == "L-vlm" else (1024 + 32 + 23)
    Wrapper = _wrapper_type()
    wrapper = Wrapper(model, architecture, head_input, spec["cam_mode"])
    filt = _trainable_filter(); selected_state = nnx.state(wrapper, filt)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(LR, weight_decay=1e-5))
    opt = tx.init(selected_state)
    fit_dir = OUT / ("smoke" if smoke else "models") / architecture / rep
    fit_dir.mkdir(parents=True, exist_ok=True)
    done = fit_dir / "trained.json"
    if done.exists():
        return json.loads(done.read_text())

    theta_targets = _aa_targets(np.asarray(train_cache["theta_target"], np.float32))
    valid = np.asarray(train_cache["unit_ids"]) >= 0
    mu = theta_targets[valid].mean(0).astype(np.float32)
    sd = theta_targets[valid].std(0).astype(np.float32); sd[sd < 1e-5] = 1.
    theta_src = np.asarray(train_cache["theta_src"], np.float32)
    shapes = np.asarray(train_cache["shape"], np.float32)
    units = np.asarray(train_cache["unit_ids"], np.int32)
    targets = np.asarray(rows["actions"][train_ids], np.float32)[:, :, :7]
    yn = ((theta_targets - mu) / sd).astype(np.float32)
    if smoke:
        run_steps = 2
        batches = [np.arange(8, dtype=np.int64), np.arange(8, 16, dtype=np.int64)]
    else:
        fair = tr._module(tr.PI05 / "fair_expert" / "run.py", "theta_lora_batch_index")
        batches = fair.make_batch_indices(train_ids, steps=steps, batch=BATCH, seed=SEED)
        rowloc = {int(r): i for i, r in enumerate(train_ids)}
        batches = [np.asarray([rowloc[int(r)] for r in ids], np.int64) for ids in batches]
        run_steps = steps
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.40)
    else:
        device = "cpu"

    def loss_fn(current, obs, sh, ui, src, ytarget, action_target):
        pred = current(obs, sh, ui, src)
        generated = _decode_pred_jax(sh, pred, jnp.asarray(mu), jnp.asarray(sd))
        mask = (ui >= 0)[..., None].astype(jnp.float32)
        return jnp.sum(jnp.square(generated - action_target) * mask) / jnp.maximum(mask.sum() * 7., 1.)

    @nnx.jit
    def train_step(current, opt_state, obs, sh, ui, src, ytarget, action_target):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(
            current, obs, sh, ui, src, ytarget, action_target)
        if architecture == "L-vlm":
            _jax_mask_late_gradients(grads, train_last=VLM_LORA_LAYERS)
        state = nnx.state(current, filt)
        updates, opt_state = tx.update(grads, opt_state, state)
        if architecture == "L-vlm":
            # AdamW applies weight decay even to a zero gradient. Mask updates
            # as well, so early PaliGemma LoRA slices remain bitwise frozen.
            _jax_mask_late_gradients(updates, train_last=VLM_LORA_LAYERS)
        nnx.update(current, optax.apply_updates(state, updates))
        return loss, opt_state

    # Avoid saving model internals over prior artifacts; each training run owns
    # a condition-scoped directory and complete artifacts are reused.
    started = time.monotonic(); losses = []; dev_curve = []; best = float("inf"); best_meta = None
    ntrain = len(train_ids)
    for step in range(run_steps):
        bi = batches[step]
        ids = train_ids[bi]
        obs = base_mod.obs_batch(rows, ids)
        sh = jnp.asarray(shapes[bi]); ui = jnp.asarray(units[bi]); src = jnp.asarray(theta_src[bi])
        ynb = jnp.asarray(yn[bi]); act = jnp.asarray(targets[bi])
        loss, opt = train_step(wrapper, opt, obs, sh, ui, src, ynb, act)
        value = float(loss); losses.append(value)
        if not np.isfinite(value):
            raise FloatingPointError(f"nonfinite {architecture}/{rep} loss at {step+1}")
        if (step+1)%50==0 or smoke:
            print(json.dumps({"stage":"train","architecture":architecture,"representation":rep,
                "step":step+1,"loss":value,"elapsed_s":time.monotonic()-started}),flush=True)
            _json(fit_dir / "losses.json", {"losses":losses,"steps":step+1})
        if not smoke and ((step + 1) % 500 == 0 or step + 1 == run_steps):
            pred = _predict_rows(wrapper, base_mod, dev_source, dev_cache, dev_ids, mu, sd,
                                 threads=threads, batch=BATCH)
            dev_target = np.asarray(dev_source["actions"][dev_ids], np.float32)[:, :, :7]
            dev_mse = float(np.mean((pred - dev_target) ** 2))
            dev_curve.append({"step": step + 1, "active7_mse": dev_mse})
            ckpt = fit_dir / "checkpoints" / f"{step+1:06d}"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            if ckpt.exists():
                raise FileExistsError(ckpt)
            ocp.PyTreeCheckpointer().save(ckpt, nnx.state(wrapper, filt).to_pure_dict())
            if dev_mse < best:
                best = dev_mse
                best_meta = {"step": step + 1, "dev_active7_mse": best, "path": str(ckpt)}
                _json(fit_dir / "best.json", best_meta)
            _json(fit_dir / "progress.json", {"status": "running", "latest_step": step + 1,
                "train_loss": value, "dev_curve": dev_curve, "best": best_meta})
        elif smoke:
            dev_curve.append({"step": step + 1, "train_loss": value})
    result = {"status": "smoke_complete" if smoke else "complete", "architecture": architecture,
              "representation": rep, "selected_exp1_spec": spec, "steps": run_steps, "batch_size": BATCH,
              "seed": SEED, "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": 1e-5},
              "lora_rank": RANK, "trainable_parameter_count": int(sum(np.size(np.asarray(x.value)) for x in selected_state.flat_state().values())),
              "dense_checkpoint": str(tr.CHECKPOINT), "restore": restore_info,
              "train_rows": len(train_ids), "unseen_dev_rows": len(dev_ids), "dev_curve": dev_curve,
              "theta_mean": mu.tolist(), "theta_std": sd.tolist(),
              "best_dev": best_meta, "train_loss_start_end": [losses[0], losses[-1]],
              "elapsed_s": time.monotonic() - started,
              "training_data_policy": "LIBERO-40 ~validation only; unseen labels not used for gradients"}
    _json(done, result)
    return result


def _predict_rows(wrapper, base_mod, source, cache, ids, mu, sd, *, threads=8, batch=8):
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn = module_jit(wrapper.__call__)
    out = []
    for start in range(0, len(ids), batch):
        sl = slice(start, min(start + batch, len(ids)))
        ix = np.arange(sl.start, sl.stop)
        obs = base_mod.obs_batch(source, ids[sl])
        sh = jnp.asarray(cache["shape"][sl]); ui = jnp.asarray(cache["unit_ids"][sl])
        src = jnp.asarray(cache["theta_src"][sl])
        pn = np.asarray(fn(obs, sh, ui, src))
        pred = _decode_rows(sh, pn, mu, sd, _import_vs())
        out.append(pred)
    return np.concatenate(out) if out else np.zeros((0, 10, 7), np.float32)


def _import_vs():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    return vs


def _evaluate_one(architecture, rep, *, threads=8, split="eval"):
    import torch
    torch.set_num_threads(threads)
    import flax.nnx as nnx
    import jax
    import orbax.checkpoint as ocp
    spec = select_exp1_spec()["selected"]
    fit_dir = _condition_dir(architecture, rep)
    best = json.loads((fit_dir / "best.json").read_text())
    model = _make_model(architecture)
    _restore_dense_checkpoint(model)
    Wrapper = _wrapper_type(); width = (3 * 2048 + 32 + 23) if architecture == "L-vlm" else 1024 + 32 + 23
    wrapper = Wrapper(model, architecture, width, spec["cam_mode"])
    filt = _trainable_filter()
    state = nnx.state(wrapper, filt)
    saved = ocp.PyTreeCheckpointer().restore(best["path"])
    state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray, saved)); nnx.update(wrapper, state)
    base_mod, rows = _load_rows_for_lora(); unseen = _load_unseen()
    cache = np.load(tr.SOURCE / "drafts" / f"{rep}_{split}.npz", allow_pickle=False)
    ids = np.asarray(cache["row_ids"], np.int64)
    source = rows if split == "train" else unseen
    trained = json.loads((fit_dir / "trained.json").read_text())
    mu = np.asarray(trained["theta_mean"], np.float32)
    sd = np.asarray(trained["theta_std"], np.float32)
    pred = _predict_rows(wrapper, base_mod, source, cache, ids, mu, sd, threads=threads)
    target = np.asarray(source["actions"][ids], np.float32)[:, :, :7]
    shape = np.asarray(cache["shape"], np.float32)
    unit = np.asarray(cache["unit_ids"], np.int32)
    src_6d = _aa_targets(np.asarray(cache["theta_src"], np.float32))
    star_6d = _aa_targets(np.asarray(cache["theta_target"], np.float32))
    src = _decode_rows(shape, (src_6d - mu) / sd, mu, sd, _import_vs())
    star = _decode_rows(shape, (star_6d - mu) / sd, mu, sd, _import_vs())
    tr_cache = np.load(tr.SOURCE / "drafts" / f"{rep}_train.npz", allow_pickle=False)
    train_theta = np.asarray(tr_cache["theta_target"], np.float32)
    train_valid = np.asarray(tr_cache["unit_ids"]) >= 0
    const_aa = train_theta[train_valid].mean(0).astype(np.float32)
    const = np.zeros_like(target)
    for i in range(len(ids)):
        for uid in np.unique(unit[i]):
            if uid < 0:
                continue
            pos = np.flatnonzero(unit[i] == uid)
            const[i, pos] = tr.decode_theta(shape[i, pos], const_aa, _import_vs())
    official = np.load(tr.UNSEEN / "official_normal_eval.npz", allow_pickle=False)
    official_loc = {int(r): i for i, r in enumerate(official["row_ids"])}
    # unseen rows use local IDs after LIBERO-40; official normal cache stores
    # those global IDs in the same alignment convention as exp1.
    normal_ids = ids if split == "train" else len(_load_rows_for_lora()[1]["actions"]) + ids
    normal = np.asarray(official["prediction"][[official_loc[int(r)] for r in normal_ids]], np.float32)[:, :, :7]
    normal_pose = np.zeros_like(target)
    import sys as _sys
    if str(HERE) not in _sys.path:
        _sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    from compressed_denoise import _procrustes
    for i in range(len(ids)):
        for uid in np.unique(unit[i]):
            if uid < 0:
                continue
            pos = np.flatnonzero(unit[i] == uid)
            pose, _, _ = _procrustes(shape[i, pos, :6], vs.to_physical(normal[i, pos, :6]))
            normal_pose[i, pos] = tr.decode_theta(shape[i, pos], pose, vs)
    hard = np.asarray(cache["hard"], bool)
    ep = np.asarray(cache["task"], np.int64) * 100000 + np.asarray(cache["episode"], np.int64)
    methods = {"lora_theta": pred, "theta_src": src, "train_constant_theta": const,
               "pi05_pose": normal_pose, "theta_star_upper": star, "official_pi05": normal}
    metric_rows = {}
    row_errors = {}
    import sys as _sys
    if str(HERE) not in _sys.path:
        _sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    move_target = vs.to_physical(target[:, :, :6])[:, :, :3].sum(1)
    move_norm = np.linalg.norm(move_target, axis=1)
    switch = np.zeros((len(target), 10), bool)
    signs = np.sign(target[:, :, 6])
    switch[:, 1:] = signs[:, 1:] != signs[:, :-1]
    near_switch = np.asarray([bool(np.any(np.abs(np.flatnonzero(s)[:, None] - np.arange(10)[None]) <= 5))
                              for s in switch], bool)
    for name, value in methods.items():
        e = np.mean((value - target) ** 2, axis=(1, 2))
        row_errors[name] = e
        move_pred = vs.to_physical(value[:, :, :6])[:, :, :3].sum(1)
        move_pred_norm = np.linalg.norm(move_pred, axis=1)
        good = move_norm > 1e-6
        cos = np.sum(move_pred * move_target, axis=1) / np.maximum(move_pred_norm * move_norm, 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos[good], -1, 1))) if good.any() else np.asarray([np.nan])
        metric_rows[name] = {"active7_mse": float(e.mean()),
            "hard_active7_mse": float(e[hard].mean()) if hard.any() else None,
            "episode_bootstrap_ci": tr.bootstrap_episode(e, ep),
            "xyz_mse": float(np.mean((value[:, :, :3] - target[:, :, :3]) ** 2)),
            "rotation_mse": float(np.mean((value[:, :, 3:6] - target[:, :, 3:6]) ** 2)),
            "gripper_mse": float(np.mean((value[:, :, 6] - target[:, :, 6]) ** 2)),
            "translation_direction_error_degrees": float(np.nanmean(angle)),
            "translation_magnitude_ratio_median": float(np.median(move_pred_norm[good] / move_norm[good])) if good.any() else None,
            "gripper_switch_neighborhood_mse": float(np.mean((value[near_switch, :, 6] - target[near_switch, :, 6]) ** 2)) if near_switch.any() else None}
    paired = {}
    for baseline in ("theta_src", "train_constant_theta", "pi05_pose", "official_pi05"):
        delta = row_errors["lora_theta"] - row_errors[baseline]
        paired[f"lora_theta_minus_{baseline}"] = {
            "mean": float(delta.mean()), "episode_bootstrap_ci": tr.bootstrap_episode(delta, ep)}
    err = np.mean((pred - target) ** 2, axis=(1, 2))
    rowpath = fit_dir / f"{split}_rows.npz"
    if rowpath.exists():
        raise FileExistsError(rowpath)
    tr._atomic_npz(rowpath, row_ids=ids, episode=cache["episode"], task=cache["task"], hard=hard,
                   pred=pred, theta_src=src, theta_star=star, train_constant=const,
                   pi05_pose=normal_pose, official_pi05=normal, target=target,
                   **{f"err_{k}": np.mean((v-target)**2, axis=(1,2)) for k,v in methods.items()})
    result = {"architecture": architecture, "representation": rep, "split": split, "rows": len(ids),
              "hard_rows": int(hard.sum()), "methods": metric_rows, "paired_differences": paired,
              "checkpoint_selected_on_dev": best,
              "theta_ground_truth_used_for_fit": False}
    _json(fit_dir / f"{split}_result.json", result)
    return result


def run(*, steps=STEPS, threads=8, device="cuda", smoke=False, evaluate=False):
    spec = select_exp1_spec()
    results = []
    for rep in REPRESENTATIONS:
        for architecture in ("L-vlm", "L-expert"):
            trained = _train_one(architecture, rep, steps=steps, threads=threads,
                                 device=device, smoke=smoke)
            results.append(trained)
            if evaluate and not smoke:
                results.append(_evaluate_one(architecture, rep, threads=threads, split="eval"))
    result = {"status": "smoke_complete" if smoke else "complete", "selected_exp1_spec": spec,
              "runs": results, "unseen_eval_used_for_selection": False}
    _json(OUT / ("smoke_summary.json" if smoke else "summary.json"), result)
    return result


def smoke():
    """Tiny CPU-only contract smoke; production training remains opt-in."""
    import jax
    import jax.numpy as jnp
    x = jnp.ones((3, 4), jnp.float32)
    w = jnp.eye(4, dtype=jnp.float32)
    a = jnp.full((4, 2), .1, jnp.float32)
    b = jnp.zeros((2, 4), jnp.float32)
    def fn(a_, b_):
        return jnp.mean((x @ w + (x @ a_) @ b_ - 1.2) ** 2)
    ga, gb = jax.grad(fn, argnums=(0, 1))(a, b)
    if not np.isfinite(float(fn(a, b))) or not np.linalg.norm(np.asarray(gb)) > 0:
        raise AssertionError("zero-init LoRA B must receive gradient")
    synthetic = {"llm": {"layers": {"q_lora_a": np.ones((18, 2, 3), np.float32),
                                     "q_lora_b": np.ones((18, 3, 2), np.float32)}}}
    masked = mask_early_lora_gradients(synthetic)
    if np.any(masked["llm"]["layers"]["q_lora_a"][:12]) or not np.all(masked["llm"]["layers"]["q_lora_a"][12:]):
        raise AssertionError("VLM LoRA must train only final six layers")
    overlaid = _overlay_pretrained({"base": np.zeros((2,), np.float32), "lora": {"w_a": np.ones((2, 2))}},
                                   {"base": np.ones((2,), np.float32)})
    if not np.array_equal(overlaid["base"], np.ones(2)) or not np.all(overlaid["lora"]["w_a"] == 1):
        raise AssertionError("checkpoint overlay must preserve added LoRA leaves")
    decision = select_exp1_spec()
    result = {"status": "complete", "lora_gradient_norm": float(np.linalg.norm(np.asarray(gb))),
              "selected_exp1_spec_dev": decision["selected"],
              "selection_candidates": len(decision["candidates"]),
              "full_model_weights_loaded": False, "gpu_used": False,
              "note": "Contract smoke only; two-step end-to-end LoRA update is separately available with --stage train --smoke."}
    _json(OUT / "smoke_contract.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("select", "smoke", "train", "run"))
    p.add_argument("--architecture", choices=("L-vlm", "L-expert"))
    p.add_argument("--representation", choices=REPRESENTATIONS)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    a = p.parse_args()
    if a.stage == "select":
        print(json.dumps(select_exp1_spec(), ensure_ascii=False))
    elif a.stage == "smoke":
        print(json.dumps(smoke(), ensure_ascii=False))
    elif a.stage == "train":
        if not a.architecture or not a.representation:
            p.error("train requires --architecture and --representation")
        print(json.dumps(_train_one(a.architecture, a.representation, steps=a.steps,
            threads=a.threads, device=a.device, smoke=a.smoke), ensure_ascii=False))
    else:
        print(json.dumps(run(steps=a.steps, threads=a.threads, device=a.device,
            smoke=a.smoke, evaluate=a.evaluate), ensure_ascii=False))


if __name__ == "__main__":
    main()
