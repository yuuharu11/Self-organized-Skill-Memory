#!/usr/bin/env python3
"""Train matched G1-dmp and G0 runs on the fixed R012/S1-v3 rows.

The parent launcher runs this script once per condition, sequentially, with
the same persisted batch_order.npy. No S1 labels outside the Train partition
are used as targets.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader, Dataset

from openpi.models import model as _model
from openpi.models.pi0_config_dmp_route import Pi0DmpRouteConfig
from openpi.training import config_moe_skill as _configs
from openpi.training import data_loader_skill as _data_loader
from openpi.training import optimizer as _optimizer
from openpi.training import checkpoints as _checkpoints
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.shared import normalize as _normalize

from command_dmp import command_draft
from train_utils import (
    initialize_from_skillnet90,
    load_s1_library,
    make_s3_train_step,
    state_history_windows,
    validate_s1_frames,
    low_memory_adamw,
)

STATE = Path(os.environ.get("SKILLNET_STATE_DIR",
             Path(__file__).resolve().parents[5] / ".skillnet-state")).expanduser()
ROUTE = STATE / "outputs/memory-pilot/dmp-route-20261102"
OUT = ROUTE / "s3-train-v3c"
S1 = ROUTE / "s1-v3c"
S1_FRAMES = ROUTE / "s1-v3" / "frames.npz"
SPLIT = STATE / "outputs/memory-pilot/r012-20261101T004734Z/split.json"
TRAIN_ROWS = STATE / "outputs/memory-pilot/r012-20261101T004734Z/rerank69-v1/train_features.npz"
DEV_ROWS = STATE / "outputs/memory-pilot/r012-20261101T004734Z/rerank69-v1/development_features.npz"
PRIV = STATE / "outputs/memory-pilot/privstate-20261101T003957Z"
STAGE_ROOT = STATE / "data/lerobot/.stage-libero90-1ed1d7032e2f4d128d39fa8364da4102/emergent/libero_90_unsegmented"
PUBLIC_PARAMS = STATE / "checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params"
PUBLIC_ASSETS = PUBLIC_PARAMS.parent / "assets"
SEED = 690311
STEPS = 1500
BATCH = 32
MICROBATCH = 16


def log(path, event, **payload):
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, **payload}
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _task_name(s):
    return str(s).removesuffix("_demo.hdf5").removesuffix(".hdf5")


def _rlds_number(episode_id):
    m = re.search(r"rlds-(\d+)$", str(episode_id))
    return int(m.group(1)) if m else None


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def make_model(base, k, *, g0=False):
    cfg = Pi0DmpRouteConfig(
        dtype=base.dtype,
        paligemma_variant=base.paligemma_variant,
        action_expert_variant=base.action_expert_variant,
        vision_variant=base.vision_variant,
        action_dim=32,
        action_horizon=10,
        max_token_len=base.max_token_len,
        pi05=True,
        discrete_state_input=base.discrete_state_input,
        skill_num=base.skill_num,
        skill_embed_dim=base.skill_embed_dim,
        positional_encoding=base.positional_encoding,
        t_coding=base.t_coding,
        enable_primitive_router=not g0,
        shape_basis_n=k,
        state_history_len=5,
        injection_mode="none" if g0 else "both",
        bridge_t0=0.4,
    )
    model = cfg.create(jax.random.key(SEED))
    state = nnx.state(model)
    loaded = CheckpointWeightLoader(str(PUBLIC_PARAMS)).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    return cfg, model


def make_data(base_cfg, model_cfg):
    data_cfg = base_cfg.data.create(base_cfg.assets_dirs, base_cfg.model)
    stats = _checkpoints.load_norm_stats(PUBLIC_ASSETS, data_cfg.asset_id)
    if stats is None:
        raise RuntimeError(f"public normalization stats not found for {data_cfg.asset_id}")
    repack = _data_loader._transforms.Group(inputs=[
        _data_loader._transforms.RepackTransform({
            "observation/image": "image", "observation/wrist_image": "wrist_image",
            "observation/state": "state", "actions": "actions", "prompt": "prompt",
        })
    ])
    data_cfg = dataclasses.replace(data_cfg, root=STAGE_ROOT, norm_stats=stats,
                                   repack_transforms=repack, prompt_from_task=True)
    raw = _data_loader.create_torch_dataset(data_cfg, 10, model_cfg)
    transformed = _data_loader.transform_dataset(raw, data_cfg)
    # create_torch_dataset may wrap the raw LeRobot dataset only for its task
    # prompt transform; locate the underlying Arrow table without copying it.
    source = raw
    while not hasattr(source, "hf_dataset"):
        if not hasattr(source, "_dataset"):
            raise TypeError("cannot locate LeRobot Arrow dataset")
        source = source._dataset
    src_ids = np.fromiter((int(x) for x in source.hf_dataset["source_episode_index"]), np.int64)
    frames = np.fromiter((int(x) for x in source.hf_dataset["frame_index"]), np.int64)
    raw_lookup = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(src_ids, frames, strict=True))}
    return data_cfg, stats, transformed, raw_lookup


def make_rows(features, frames, allowed_tasks, raw_lookup):
    row_episode = np.asarray(features["episode_id"]).astype(str)
    row_t = np.asarray(features["frame"], np.int64)
    row_task = np.asarray([_task_name(x) for x in features["source_file"]])
    frame_index = {}
    for i in range(len(frames["t"])):
        if _task_name(frames["task_id"][i]) not in allowed_tasks:
            continue
        key = (str(frames["episode_id"][i]), int(frames["t"][i]))
        if key in frame_index:
            raise ValueError(f"ambiguous S1 v3 frame label {key}")
        frame_index[key] = i
    selected_features, selected_raw, selected_frame = [], [], []
    for i, (ep, t, task) in enumerate(zip(row_episode, row_t, row_task, strict=True)):
        if task not in allowed_tasks:
            continue
        fi = frame_index.get((ep, int(t)))
        source_id = _rlds_number(ep)
        raw = raw_lookup.get((source_id, int(t))) if source_id is not None else None
        if fi is not None and raw is not None:
            selected_features.append(i)
            selected_raw.append(raw)
            selected_frame.append(fi)
    return (np.asarray(selected_features, np.int64), np.asarray(selected_raw, np.int64),
            np.asarray(selected_frame, np.int64))


class IndexedDataset(Dataset):
    def __init__(self, transformed, raw_indices, feature_indices):
        self.data = transformed
        self.raw_indices = np.asarray(raw_indices, np.int64)
        self.feature_indices = np.asarray(feature_indices, np.int64)

    def __len__(self):
        return len(self.raw_indices)

    def __getitem__(self, i):
        row = self.data[int(self.raw_indices[i])]
        row["feature_index"] = np.int64(self.feature_indices[i])
        return row


def make_batches(order):
    return [order[i:i + BATCH].tolist() for i in range(0, len(order), BATCH)]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return np.asarray(x)


def _observation(batch):
    raw = {k: _to_numpy(batch[k]) for k in ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")}
    return jax.tree.map(jnp.asarray, _model.Observation.from_dict(raw))


def previous_raw_commands(features, feature_indices, q01, q99):
    """Return the command at t-1 from cached chunk actions; t=0 starts at zero."""
    eps = np.asarray(features["episode_id"]).astype(str)
    times = np.asarray(features["frame"], np.int64)
    actions = np.asarray(features["actions"], np.float32)
    lookup = {}
    for i, (ep, t) in enumerate(zip(eps, times, strict=True)):
        lookup.setdefault(ep, []).append((int(t), i))
    for rows in lookup.values():
        rows.sort()
    q01 = np.asarray(q01, np.float32)[:7]
    q99 = np.asarray(q99, np.float32)[:7]
    result = np.zeros((len(feature_indices), 6), np.float32)
    for oi, fi in enumerate(np.asarray(feature_indices, np.int64)):
        ep, t = eps[fi], int(times[fi])
        if t <= 0:
            continue
        candidates = lookup[ep]
        prior = [pair for pair in candidates if pair[0] < t]
        if not prior:
            continue
        pt, pi = prior[-1]
        offset = t - pt - 1
        if not 0 <= offset < actions.shape[1]:
            continue
        normalized = actions[pi, offset, :6]
        result[oi] = ((normalized + 1.) * .5 * (q99[:6] - q01[:6] + 1e-6) + q01[:6])
    return result


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).splitlines()[0]
        return int(out.strip())
    except Exception:
        return None


def save_checkpoint(model, output, asset_id):
    output.mkdir(parents=True, exist_ok=True)
    params = nnx.state(model).to_pure_dict()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(output / "params", {"params": params})
    _normalize.save(output / "assets" / asset_id, _checkpoints.load_norm_stats(PUBLIC_ASSETS, asset_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("G1-dmp", "G0"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=MICROBATCH)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = OUT / "attempts" / f"{args.condition}-attempt-{args.attempt}"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    started = time.perf_counter()
    log(metrics_path, "start", condition=args.condition, steps=args.steps, batch_size=BATCH,
        checkpoint=str(PUBLIC_PARAMS), s1=str(S1), cpu_threads=args.threads,
        xla_preallocate=os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE"),
        xla_fraction=os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION"))
    log(metrics_path, "memory_strategy", logical_batch_size=BATCH,
        microbatch_size=args.microbatch_size, optimizer_updates_per_batch=1)

    library = load_s1_library(S1 / "library.npz")
    frames = validate_s1_frames(S1_FRAMES, len(library["shapes"]))
    with open(SPLIT) as f:
        split = json.load(f)["full"]
    train_tasks = set(split["train"])
    dev_tasks = set(split["development"])
    train_features = _load_npz(TRAIN_ROWS)
    dev_features = _load_npz(DEV_ROWS)
    frame_train_mask = np.isin(frames["split"].astype(str), ["train"])
    frame_dev_mask = np.isin(frames["split"].astype(str), ["development"])
    train_frame_subset = {k: v[frame_train_mask] for k, v in frames.items()}
    dev_frame_subset = {k: v[frame_dev_mask] for k, v in frames.items()}

    base_train = _configs.get_config("pi05_libero_moe_skill_4_90")
    data_cfg, norm_stats, transformed, raw_lookup = make_data(base_train, base_train.model)
    tr_f, tr_raw, tr_fr = make_rows(train_features, train_frame_subset, train_tasks, raw_lookup)
    dv_f, dv_raw, dv_fr = make_rows(dev_features, dev_frame_subset, dev_tasks, raw_lookup)
    if not len(tr_f):
        raise RuntimeError("no exact Train feature rows joined to S1-v3 labels and staged LeRobot episodes")
    if not len(dv_f):
        log(metrics_path, "warning_no_dev_rows", available_dev_tasks=0)
    log(metrics_path, "data_join", train_rows=int(len(tr_f)), dev_rows=int(len(dv_f)),
        train_tasks=sorted(set(np.asarray(train_features["source_file"])[tr_f].astype(str))),
        train_task_count=len(set(_task_name(x) for x in np.asarray(train_features["source_file"])[tr_f])),
        dev_task_count=len(set(_task_name(x) for x in np.asarray(dev_features["source_file"])[dv_f])),
        stage_rows=len(raw_lookup),)

    # Require the exact same row permutation and batch order in both runs.
    order_file = OUT / "batch_order.npy"
    if order_file.exists():
        order = np.load(order_file, allow_pickle=False)
        if order.shape != (STEPS, BATCH):
            raise ValueError("persisted batch_order has wrong shape")
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(tr_f))
        order = np.stack([perm[(np.arange(BATCH) + step * BATCH) % len(perm)] for step in range(STEPS)])
        np.save(order_file, order)
    order_hash = hashlib.sha256(np.ascontiguousarray(order).view(np.uint8)).hexdigest()
    log(metrics_path, "batch_order", shape=list(order.shape), sha256=order_hash,
        row_indices_sha256=hashlib.sha256(np.ascontiguousarray(tr_f).view(np.uint8)).hexdigest())
    batches = make_batches(order.reshape(-1))
    train_ds = IndexedDataset(transformed, tr_raw, tr_f)
    loader = DataLoader(
        train_ds, batch_sampler=batches, num_workers=4, persistent_workers=True,
        collate_fn=_data_loader._collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        pin_memory=False,
    )
    loader_iter = iter(loader)

    dmp_spec = __import__("importlib.util").util.spec_from_file_location("s1_v3_dmp_for_s3", S1 / "dmp.py")
    dmp = __import__("importlib.util").util.module_from_spec(dmp_spec)
    dmp_spec.loader.exec_module(dmp)
    # Attach the frame targets to the aligned feature rows. v3 g is already a
    # raw 6-D command delta (no world-pose conversion or ridge map). S1-v3c
    # explicitly stores coefficient weights; sampled shape curves are unused.
    # Build causal histories on the exact retained rows, in episode/time order.
    train_hist = state_history_windows(
        train_features["state"][tr_f], train_features["episode_id"][tr_f],
        train_features["frame"][tr_f], history_len=5,
    )
    dev_hist = state_history_windows(
        dev_features["state"][dv_f], dev_features["episode_id"][dv_f],
        dev_features["frame"][dv_f], history_len=5,
    ) if len(dv_f) else np.empty((0, 5, 32), np.float32)
    tr_y0 = previous_raw_commands(train_features, tr_f, norm_stats["actions"].q01,
                                  norm_stats["actions"].q99)
    dv_y0 = previous_raw_commands(dev_features, dv_f, norm_stats["actions"].q01,
                                  norm_stats["actions"].q99)
    # Pre-expand only the selected train samples; y0 is the prior executed
    # command from the previous cached action chunk, never physical state.
    tr_drafts = np.empty((len(tr_f), 10, 7), np.float32)
    for j, fi in enumerate(tr_fr):
        route = type("Route", (), {"prim_id": np.asarray([train_frame_subset["prim_id"][fi]]),
                                     "goal": train_frame_subset["g"][fi:fi+1],
                                     "tau": np.asarray([train_frame_subset["tau"][fi]]),
                                     "phase_s": np.asarray([train_frame_subset["phase_s"][fi]])})()
        tr_drafts[j] = command_draft(dmp, library, route, tr_y0[j:j+1], q01=norm_stats["actions"].q01,
                                     q99=norm_stats["actions"].q99, horizon=10,
                                     dt=float(library["dt"]))[0]
    if len(dv_f):
        dv_drafts = np.empty((len(dv_f), 10, 7), np.float32)
        for j, fi in enumerate(dv_fr):
            route = type("Route", (), {"prim_id": np.asarray([dev_frame_subset["prim_id"][fi]]),
                                         "goal": dev_frame_subset["g"][fi:fi+1],
                                         "tau": np.asarray([dev_frame_subset["tau"][fi]]),
                                         "phase_s": np.asarray([dev_frame_subset["phase_s"][fi]])})()
            dv_drafts[j] = command_draft(dmp, library, route, dv_y0[j:j+1], q01=norm_stats["actions"].q01,
                                         q99=norm_stats["actions"].q99, horizon=10,
                                         dt=float(library["dt"]))[0]
    else:
        dv_drafts = np.empty((0, 10, 7), np.float32)

    base = base_train.model
    cfg, model = make_model(base, library["shapes"].shape[-1], g0=(args.condition == "G0"))
    filter_primitives = args.condition == "G1-dmp"
    lr_schedule = _optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-5,
                                                 decay_steps=STEPS, decay_lr=1e-6).create()
    # Keep AdamW's public coefficients and schedule but store both moments in
    # bf16 so action-expert updates fit the shared 0.30 GPU cap.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        low_memory_adamw(lr_schedule, b1=0.9, b2=0.95, eps=1e-8,
                         weight_decay=1e-10, moment_dtype=jnp.bfloat16),
    )
    from openpi.models.pi0_dmp_route import dmp_trainable_filter, dmp_g0_trainable_filter
    params = nnx.state(model, dmp_trainable_filter() if filter_primitives else dmp_g0_trainable_filter())
    opt_state = optimizer.init(params)
    shared_hash = parameter_hash(nnx.state(model, dmp_g0_trainable_filter()).to_pure_dict())
    log(metrics_path, "initialization", shared_action_expert_sha256=shared_hash,
        train_rows_sha256=hashlib.sha256(np.ascontiguousarray(tr_f).view(np.uint8)).hexdigest(),
        batch_order_sha256=order_hash,
        optimizer="AdamW b1=.9 b2=.95 lr=1e-5 cosine warmup100; mu/nu_dtype=bfloat16")
    step_fn = make_s3_train_step(optimizer, use_primitives=filter_primitives,
                                 microbatch_size=args.microbatch_size)
    shape_array = jnp.asarray(library["shapes"], jnp.float32)
    local_for_feature = {int(v): i for i, v in enumerate(tr_f)}
    step_times = []
    requested_steps = 1 if args.preflight_only else args.steps
    for step in range(requested_steps):
        batch = next(loader_iter)
        fids = np.asarray(_to_numpy(batch["feature_index"]), np.int64)
        # The wrapper exposes original feature row indices. Translate them to
        # positions in the persisted joined row table, used by the batch order.
        local = np.asarray([local_for_feature[int(v)] for v in fids], np.int64)
        obs = _observation(batch)
        actions = jnp.asarray(_to_numpy(batch["actions"]), jnp.float32)
        if step == 0:
            max_action_alignment = float(np.max(np.abs(
                np.asarray(_to_numpy(batch["actions"]))[..., :7] - train_features["actions"][fids]
            )))
            log(metrics_path, "action_alignment", max_abs_normalized_error=max_action_alignment,
                tolerance=1e-3, passed=max_action_alignment <= 1e-3)
            if max_action_alignment > 1e-3:
                raise RuntimeError("transformed actions do not match the cached row sequence")
        targets = {
            "state_history": jnp.asarray(train_hist[local], jnp.float32),
            "prim_id": jnp.asarray(train_frame_subset["prim_id"][tr_fr[local]], jnp.int32),
            "g": jnp.asarray(train_frame_subset["g"][tr_fr[local]], jnp.float32),
            "tau": jnp.asarray(train_frame_subset["tau"][tr_fr[local]], jnp.float32),
            "phase_s": jnp.asarray(train_frame_subset["phase_s"][tr_fr[local]], jnp.float32),
        }
        draft = jnp.asarray(tr_drafts[local], jnp.float32)
        tick = time.perf_counter()
        loss, metrics, opt_state = step_fn(
            jax.random.key(SEED + step), model, opt_state, obs, actions,
            draft, shape_array, targets,
        )
        jax.block_until_ready(loss)
        step_times.append(time.perf_counter() - tick)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == requested_steps:
            m = {k: float(np.asarray(v)) for k, v in metrics.items()}
            log(metrics_path, "train", step=step + 1, loss=float(loss), metrics=m,
                step_seconds=float(step_times[-1]), mean_step_seconds=float(np.mean(step_times[-50:])),
                batch_order_sha256=order_hash, gpu_used_mib=gpu_used_mib())
    del loader_iter, loader
    if args.preflight_only:
        log(metrics_path, "preflight_complete", updates=1, elapsed_seconds=time.perf_counter() - started,
            batch_order_sha256=order_hash)
        return

    # Save the trained weights before any optional post-training evaluation so
    # a diagnostic failure never discards a completed 1,500-update run.
    checkpoint_dir = out / f"step_{args.steps}"
    save_checkpoint(model, checkpoint_dir, data_cfg.asset_id)

    # Pointer agreement and action MSE are measured on a fixed sample. These
    # calls use prediction-only routing; the S1 targets stay outside infer().
    train_eval = eval_subset(model, transformed, tr_raw, tr_f, tr_fr, train_features,
                             train_frame_subset, train_hist, tr_y0, library, norm_stats,
                             dmp, use_primitives=filter_primitives, sample_n=128)
    dev_eval = eval_subset(model, transformed, dv_raw, dv_f, dv_fr, dev_features,
                           dev_frame_subset, dev_hist, dv_y0, library, norm_stats,
                           dmp, use_primitives=filter_primitives,
                           sample_n=min(256, len(dv_f))) if len(dv_f) else {}
    log(metrics_path, "complete", updates=args.steps, elapsed_seconds=time.perf_counter() - started,
        mean_step_seconds=float(np.mean(step_times[10:])), train_eval=train_eval,
        dev_eval=dev_eval, checkpoint=str(checkpoint_dir / "params"),
        batch_order_sha256=order_hash)
    (out / "evaluation_metrics.json").write_text(json.dumps({"train": train_eval, "development": dev_eval}, indent=2) + "\n")


def parameter_hash(tree):
    import flax.traverse_util
    flat = flax.traverse_util.flatten_dict(tree, sep="/")
    digest = hashlib.sha256()
    for key in sorted(flat):
        value = np.asarray(flat[key])
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def eval_subset(model, transformed, raw_idx, feature_idx, frame_idx, features, frames, histories,
                y0_all, library, norm_stats, dmp, *, use_primitives, sample_n):
    sample_n = min(sample_n, len(feature_idx))
    selected = np.linspace(0, len(feature_idx) - 1, sample_n).round().astype(np.int64)
    ds = IndexedDataset(transformed, raw_idx[selected], feature_idx[selected])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=_data_loader._collate_fn, worker_init_fn=_data_loader._worker_init_fn)
    from openpi.shared import nnx_utils
    sample_fn = nnx_utils.module_jit(model.sample_actions_s3,
                                     static_argnames=("num_steps", "injection_mode", "t0"))
    pred_ids, expected_ids, draft_mse, generated_mse = [], [], [], []
    cursor = 0
    for batch in dl:
        b = len(batch["feature_index"])
        obs = _observation(batch)
        local = selected[cursor:cursor+b]
        history = jnp.asarray(histories[local], jnp.float32)
        feature_rows = feature_idx[local]
        truth = np.asarray(features["actions"])[feature_rows, :, :7]
        if use_primitives:
            route = model.predict_route(obs, jnp.asarray(library["shapes"], jnp.float32), history)
            route_np = jax.tree.map(np.asarray, route)
            pred_ids.extend(route_np.prim_id.astype(int).tolist())
            expected_ids.extend(np.asarray(frames["prim_id"])[frame_idx[local]].astype(int).tolist())
            draft = command_draft(dmp, library, route_np, y0_all[local],
                                  q01=norm_stats["actions"].q01, q99=norm_stats["actions"].q99,
                                  horizon=10, dt=float(library["dt"]))
            route_arg, draft_arg, mode = route, jnp.asarray(draft, jnp.float32), "both"
        else:
            draft = np.zeros((b, 10, 7), np.float32)
            route_arg, draft_arg, mode = None, None, "none"
        draft_mse.extend(np.mean((draft - truth) ** 2, axis=(1, 2)).tolist())
        noise = jax.random.normal(jax.random.key(SEED + 900000 + cursor), (b, 10, 32), jnp.float32)
        output = sample_fn(jax.random.key(SEED + 910000 + cursor), obs, draft=draft_arg,
                           noise=noise, route=route_arg, t0=0.4, num_steps=2,
                           injection_mode=mode)
        generated_mse.extend(np.mean((np.asarray(output)[..., :7] - truth) ** 2,
                                     axis=(1, 2)).tolist())
        cursor += b
    result = {"rows": len(generated_mse), "draft_mse_normalized_7d": float(np.mean(draft_mse)),
              "generated_mse_normalized_7d": float(np.mean(generated_mse)),
              "sample": "evenly spaced rows; task split preserved; fixed per-row noise"}
    if use_primitives:
        result["pointer_top1_agreement"] = float(np.mean(np.asarray(pred_ids) == np.asarray(expected_ids)))
    return result


if __name__ == "__main__":
    main()
