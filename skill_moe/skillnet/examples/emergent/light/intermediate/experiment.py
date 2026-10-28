"""Lightweight held-out-suite study of action-flow intermediate skills."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_CACHE = Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-20261026T050453Z-0cb536/feature-cache")
DEFAULT_DATASET = Path("/work/.skillnet-state/data/lerobot/emergent/libero_40_unsegmented")
DEFAULT_OUTPUT_ROOT = Path("/work/.skillnet-state/outputs/memory-pilot")
TRAIN_TASK_INDICES = tuple(range(20, 40))
GOAL_TASK_INDICES = tuple(range(10, 20))
HORIZONS = (25, 10)
T0_VALUES = (0.8, 0.6, 0.4)


def _standardize_fit(values: np.ndarray):
    mean = np.asarray(values, dtype=np.float32).mean(axis=0)
    scale = np.asarray(values, dtype=np.float32).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _standardize(values, mean, scale):
    return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(np.float32)


def dct_matrix(length: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (frequency, time)."""
    n = np.arange(length, dtype=np.float64)
    k = np.arange(length, dtype=np.float64)[:, None]
    matrix = np.cos(np.pi / length * (n + 0.5) * k)
    matrix[0] *= np.sqrt(1.0 / length)
    matrix[1:] *= np.sqrt(2.0 / length)
    return matrix.astype(np.float32)


def low_frequency_intermediate(actions: np.ndarray, train_actions: np.ndarray, t0: float):
    """Keep DCT coefficients whose train signal power beats noise power."""
    if not 0.0 < t0 < 1.0:
        raise ValueError("t0 must be in (0, 1)")
    actions = np.asarray(actions, dtype=np.float32)
    train_actions = np.asarray(train_actions, dtype=np.float32)
    length = actions.shape[1]
    matrix = dct_matrix(length)
    train_coeff = np.einsum("kt,ntd->nkd", matrix, train_actions, optimize=True)
    variance = train_coeff.var(axis=0)
    keep = variance * (1.0 - t0) ** 2 > t0 ** 2
    coeff = np.einsum("kt,ntd->nkd", matrix, actions, optimize=True)
    filtered = coeff * keep[None, :, :]
    value = np.einsum("kt,nkd->ntd", matrix, filtered, optimize=True)
    return value.astype(np.float32), keep, variance.astype(np.float32)


def align_segment_labels(cache_steps, frame_index, skill_class, skill_segment):
    """Join annotation values at each cached observation's window-start frame."""
    frame_index = np.asarray(frame_index, dtype=np.int64)
    skill_class = np.asarray(skill_class, dtype=np.int64)
    skill_segment = np.asarray(skill_segment, dtype=np.int64)
    if not (len(frame_index) == len(skill_class) == len(skill_segment)):
        raise ValueError("parquet annotation columns have different row counts")
    lookup = {int(frame): i for i, frame in enumerate(frame_index)}
    steps = np.asarray(cache_steps, dtype=np.int64)
    missing = [int(step) for step in steps if int(step) not in lookup]
    if missing:
        raise ValueError(f"cached window start frames missing from annotation parquet: {missing[:5]}")
    rows = np.asarray([lookup[int(step)] for step in steps], dtype=np.int64)
    return skill_class[rows], skill_segment[rows]


def split_split_a(rows):
    train_mask = np.isin(rows["task_index"], TRAIN_TASK_INDICES)
    goal_mask = np.isin(rows["task_index"], GOAL_TASK_INDICES)
    return ({key: value[train_mask] for key, value in rows.items()},
            {key: value[goal_mask] for key, value in rows.items()})


def make_memory_payload(train_rows, values, unit_labels):
    """Build the store only from training suites; reject accidental goal leakage."""
    if np.isin(train_rows["task_index"], GOAL_TASK_INDICES).any():
        raise ValueError("held-out goal rows must never be inserted into memory")
    if not np.isin(train_rows["task_index"], TRAIN_TASK_INDICES).all():
        raise ValueError("memory initialization accepts only object/spatial task_index 20-39")
    return {"rows": train_rows, "values": np.asarray(values), "unit_ids": np.asarray(unit_labels)}


def load_rows(cache_dir: Path, dataset_dir: Path, max_horizon: int = 25) -> dict[str, np.ndarray]:
    """Reuse cached features/actions and read only skill labels/frame from parquet."""
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parquet_by_episode = {int(path.stem.split("_")[-1]): path
                          for path in (dataset_dir / "data").rglob("episode_*.parquet")}
    cols = {name: [] for name in ("episode_id", "task_index", "frame", "skill_class", "skill_segment",
                                  "f", "k", "s", "a", "valid_h10", "valid_h25")}
    for episode in manifest["episodes"]:
        episode_id = str(episode["episode_id"])
        episode_index = int(episode_id.rsplit("-", 1)[-1])
        parquet = parquet_by_episode.get(episode_index)
        if parquet is None:
            continue
        with np.load(episode["cache"], allow_pickle=False) as cache:
            steps = np.asarray(cache["step"], dtype=np.int64)
            task_index = int(np.asarray(cache["task_index"]).item())
            all_steps = np.asarray(cache.get("env_step", np.arange(len(cache["action"]))), dtype=np.int64)
            all_actions = np.asarray(cache["action"], dtype=np.float32)
            all_states = np.asarray(cache["env_state"], dtype=np.float32)
            features = np.asarray(cache["f"], dtype=np.float32)
            keys = np.asarray(cache["k"], dtype=np.float32)
            proprio = np.asarray(cache["proprio"], dtype=np.float32)
        if task_index not in (*TRAIN_TASK_INDICES, *GOAL_TASK_INDICES):
            continue
        table = pq.read_table(parquet, columns=["frame_index", "eval_skill_class", "eval_skill_segment"])
        frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        classes, segments = align_segment_labels(
            steps, frame_index, table["eval_skill_class"].to_numpy(), table["eval_skill_segment"].to_numpy())
        step_to_idx = {int(value): i for i, value in enumerate(all_steps)}
        for row, step in enumerate(steps):
            start = step_to_idx.get(int(step), -1)
            if start < 0 or start + 10 >= len(all_states) or start + 10 > len(all_actions):
                continue
            available = min(max_horizon, len(all_actions) - start, len(all_states) - start - 1)
            action = np.full((max_horizon, all_actions.shape[-1]), np.nan, dtype=np.float32)
            action[:available] = all_actions[start:start + available]
            cols["episode_id"].append(episode_id)
            cols["task_index"].append(task_index)
            cols["frame"].append(int(step))
            cols["skill_class"].append(int(classes[row]))
            cols["skill_segment"].append(int(segments[row]))
            cols["f"].append(features[row])
            cols["k"].append(keys[row])
            cols["s"].append(proprio[row])
            cols["a"].append(action)
            cols["valid_h10"].append(available >= 10)
            cols["valid_h25"].append(available >= 25)
    if not cols["episode_id"]:
        raise ValueError("no feature rows joined to annotated parquet episodes")
    result = {}
    for name, values in cols.items():
        if name in ("f", "k", "s", "a"):
            result[name] = np.asarray(values, dtype=np.float32)
        elif name in ("valid_h10", "valid_h25"):
            result[name] = np.asarray(values, dtype=bool)
        elif name in ("episode_id",):
            result[name] = np.asarray(values, dtype=str)
        else:
            result[name] = np.asarray(values, dtype=np.int64)
    return result


class FlowActionHead(nn.Module):
    """Conditional flow field F(x_t, t, f, s) trained with velocity eps-A."""
    def __init__(self, feature_dim: int, state_dim: int, action_dim: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim + state_dim + action_dim + 1, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, action_dim))

    def forward(self, x_t, t, feature, state):
        return self.net(torch.cat([x_t, t.reshape(-1, 1), feature, state], dim=-1))


class VRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim))

    def forward(self, x):
        return self.net(x)


def _episode_train_validation_indices(rows, seed: int, validation_fraction: float = 0.15):
    """Episode-level holdout so adjacent windows cannot leak across validation."""
    episode_ids = np.asarray(rows["episode_id"]).astype(str)
    unique = np.unique(episode_ids)
    if len(unique) < 2:
        raise ValueError("early stopping needs at least two training episodes")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_valid = min(len(unique) - 1, max(1, int(round(len(unique) * validation_fraction))))
    valid_episodes = set(shuffled[:n_valid].tolist())
    valid = np.asarray([episode in valid_episodes for episode in episode_ids], dtype=bool)
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def train_flow(train, *, horizon: int, epochs: int, batch_size: int, seed: int,
               device: torch.device, patience: int = 5):
    torch.manual_seed(seed)
    actions = train["a"][:, :horizon]
    train_ix, valid_ix = _episode_train_validation_indices(train, seed + 1)
    f_mean, f_scale = _standardize_fit(train["f"][train_ix])
    s_mean, s_scale = _standardize_fit(train["s"][train_ix])
    a_mean, a_scale = _standardize_fit(actions[train_ix].reshape(len(train_ix), -1))
    ft = torch.as_tensor(_standardize(train["f"][train_ix], f_mean, f_scale), device=device)
    st = torch.as_tensor(_standardize(train["s"][train_ix], s_mean, s_scale), device=device)
    at = torch.as_tensor(_standardize(actions[train_ix].reshape(len(train_ix), -1), a_mean, a_scale), device=device)
    vf = torch.as_tensor(_standardize(train["f"][valid_ix], f_mean, f_scale), device=device)
    vs = torch.as_tensor(_standardize(train["s"][valid_ix], s_mean, s_scale), device=device)
    va = torch.as_tensor(_standardize(actions[valid_ix].reshape(len(valid_ix), -1), a_mean, a_scale), device=device)
    model = FlowActionHead(ft.shape[-1], st.shape[-1], at.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = torch.Generator(device=device).manual_seed(seed + 100)
    valid_rng = torch.Generator(device=device).manual_seed(seed + 200)
    valid_eps = torch.randn(va.shape, generator=valid_rng, device=device)
    valid_t = torch.rand((len(va),), generator=valid_rng, device=device)
    valid_xt = valid_t[:, None] * valid_eps + (1.0 - valid_t[:, None]) * va
    valid_target = valid_eps - va
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        order = torch.randperm(len(at), generator=rng, device=device)
        losses = []
        model.train()
        for start in range(0, len(at), batch_size):
            ix = order[start:start + batch_size]
            a = at[ix]
            eps = torch.randn_like(a)
            t = torch.rand((len(ix),), generator=rng, device=device)
            x_t = t[:, None] * eps + (1.0 - t[:, None]) * a
            target = eps - a
            prediction = model(x_t, t, ft[ix], st[ix])
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_prediction = model(valid_xt, valid_t, vf, vs)
            valid_loss = float(F.mse_loss(valid_prediction, valid_target).cpu())
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": valid_loss})
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"f_mean": f_mean, "f_scale": f_scale, "s_mean": s_mean, "s_scale": s_scale,
                   "a_mean": a_mean, "a_scale": a_scale, "action_shape": actions.shape[1:]}, {
                       "history": history, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                       "validation_rows": int(len(valid_ix)), "training_rows": int(len(train_ix)),
                       "early_stopped": len(history) < epochs}


def train_v_regressor(train, targets, subtask_onehot, *, use_subtask: bool, epochs: int,
                      batch_size: int, seed: int, device: torch.device,
                      target_mean=None, target_scale=None, patience: int = 5):
    torch.manual_seed(seed)
    train_ix, valid_ix = _episode_train_validation_indices(train, seed + 1)
    f_mean, f_scale = _standardize_fit(train["f"][train_ix])
    s_mean, s_scale = _standardize_fit(train["s"][train_ix])
    if target_mean is None or target_scale is None:
        action_mean, action_scale = _standardize_fit(np.asarray(targets).reshape(len(targets), -1))
    else:
        action_mean, action_scale = target_mean, target_scale
    f = _standardize(train["f"], f_mean, f_scale)
    s = _standardize(train["s"], s_mean, s_scale)
    st = subtask_onehot if use_subtask else np.zeros_like(subtask_onehot)
    x = np.concatenate([st, f, s], axis=-1).astype(np.float32)
    y = _standardize(np.asarray(targets).reshape(len(targets), -1), action_mean, action_scale)
    xt = torch.as_tensor(x[train_ix], device=device)
    yt = torch.as_tensor(y[train_ix], device=device)
    xv = torch.as_tensor(x[valid_ix], device=device)
    yv = torch.as_tensor(y[valid_ix], device=device)
    model = VRegressionHead(xt.shape[-1], yt.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = torch.Generator(device=device).manual_seed(seed + 300)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        order = torch.randperm(len(xt), generator=rng, device=device)
        losses = []
        model.train()
        for start in range(0, len(xt), batch_size):
            ix = order[start:start + batch_size]
            prediction = model(xt[ix])
            loss = F.mse_loss(prediction, yt[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(F.mse_loss(model(xv), yv).cpu())
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": valid_loss})
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"f_mean": f_mean, "f_scale": f_scale, "s_mean": s_mean, "s_scale": s_scale,
                   "a_mean": action_mean, "a_scale": action_scale,
                   "target_shape": tuple(np.asarray(targets).shape[1:])}, {
                       "history": history, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                       "training_rows": int(len(train_ix)), "validation_rows": int(len(valid_ix))}


def predict_v(model, normalizers, rows, subtask_onehot, *, use_subtask: bool, device: torch.device):
    f = _standardize(rows["f"], normalizers["f_mean"], normalizers["f_scale"])
    s = _standardize(rows["s"], normalizers["s_mean"], normalizers["s_scale"])
    st = subtask_onehot if use_subtask else np.zeros_like(subtask_onehot)
    x = torch.as_tensor(np.concatenate([st, f, s], axis=-1), device=device)
    with torch.no_grad():
        y = model(x).cpu().numpy()
    y = y * normalizers["a_scale"] + normalizers["a_mean"]
    return y.reshape(len(rows["a"]), *normalizers["target_shape"])


def integrate_flow(model, normalizers, rows, noise, start_t: float, init_v=None,
                   *, device: torch.device):
    """Euler integrate t=1→0, or initialize at t0 from a supplied estimate."""
    f = torch.as_tensor(_standardize(rows["f"], normalizers["f_mean"], normalizers["f_scale"]), device=device)
    s = torch.as_tensor(_standardize(rows["s"], normalizers["s_mean"], normalizers["s_scale"]), device=device)
    noise = np.asarray(noise, dtype=np.float32).reshape(len(rows["a"]), -1)
    a_mean, a_scale = normalizers["a_mean"], normalizers["a_scale"]
    eps = torch.as_tensor(noise, device=device)
    if init_v is None:
        x = eps.clone()
    else:
        v = _standardize(np.asarray(init_v).reshape(len(rows["a"]), -1), a_mean, a_scale)
        v = torch.as_tensor(v, device=device)
        x = start_t * eps + (1.0 - start_t) * v
    n_steps = int(round(start_t * 10))
    dt = -0.1
    with torch.no_grad():
        for step in range(n_steps):
            t = torch.full((len(x),), start_t + step * dt, device=device)
            x = x + dt * model(x, t, f, s)
    decoded = x.cpu().numpy() * a_scale + a_mean
    return decoded.reshape(len(rows["a"]), *normalizers["action_shape"])


def integrate_flow_sources(model, normalizers, rows, noises, sources, start_t: float,
                           *, device: torch.device, batch_rows: int = 384,
                           return_per_row: bool = False):
    """Integrate several source values and fixed noises in batched chunks."""
    names = list(sources)
    n_noises, n_rows = len(noises), len(rows["a"])
    action_shape = tuple(normalizers["action_shape"])
    flat_dim = int(np.prod(action_shape))
    a_mean = np.asarray(normalizers["a_mean"], np.float32)
    a_scale = np.asarray(normalizers["a_scale"], np.float32)
    mse_sums = {name: 0.0 for name in names}
    mse_counts = {name: 0 for name in names}
    row_error_sums = {name: np.zeros(n_rows, dtype=np.float64) for name in names}
    row_error_counts = {name: np.zeros(n_rows, dtype=np.int64) for name in names}
    target = np.asarray(rows["a"], dtype=np.float32).reshape(n_rows, flat_dim)
    for lo in range(0, n_rows, batch_rows):
        hi = min(n_rows, lo + batch_rows)
        count = hi - lo
        eps_np = np.stack([np.asarray(noise[lo:hi], np.float32) for noise in noises])
        eps_np = eps_np.reshape(n_noises, count, flat_dim)
        eps_np = np.broadcast_to(eps_np[:, None], (n_noises, len(names), count, flat_dim))
        x_np = np.empty_like(eps_np)
        for si, name in enumerate(names):
            value = sources[name]
            if value is None:
                x_np[:, si] = eps_np[:, si]
            else:
                v = _standardize(np.asarray(value[lo:hi]).reshape(count, flat_dim), a_mean, a_scale)
                x_np[:, si] = start_t * eps_np[:, si] + (1.0 - start_t) * v[None]
        x = torch.as_tensor(x_np.reshape(-1, flat_dim), device=device)
        feature = torch.as_tensor(_standardize(rows["f"][lo:hi], normalizers["f_mean"],
                                                normalizers["f_scale"]), device=device)
        state = torch.as_tensor(_standardize(rows["s"][lo:hi], normalizers["s_mean"],
                                              normalizers["s_scale"]), device=device)
        repeat = n_noises * len(names)
        feature = feature.repeat(repeat, 1)
        state = state.repeat(repeat, 1)
        n_steps = int(round(start_t * 10))
        dt = -0.1
        with torch.no_grad():
            for step in range(n_steps):
                t_value = start_t + step * dt
                t = torch.full((len(x),), t_value, device=device)
                x = x + dt * model(x, t, feature, state)
        decoded = x.cpu().numpy().reshape(n_noises, len(names), count, flat_dim)
        decoded = decoded * a_scale + a_mean
        target_chunk = target[lo:hi]
        mask = np.isfinite(target_chunk)
        for si, name in enumerate(names):
            err = np.square(decoded[:, si] - target_chunk[None])
            valid_count = int(mask.sum())
            if valid_count:
                mse_sums[name] += float((err * mask[None]).sum())
                mse_counts[name] += n_noises * valid_count
                row_error_sums[name][lo:hi] = (err * mask[None]).sum(axis=(0, 2))
                row_error_counts[name][lo:hi] = n_noises * mask.sum(axis=1)
    means = {name: (mse_sums[name] / mse_counts[name] if mse_counts[name] else float("nan"))
             for name in names}
    if return_per_row:
        per_row = {name: np.divide(row_error_sums[name], row_error_counts[name],
                                   out=np.full(n_rows, np.nan), where=row_error_counts[name] > 0)
                   for name in names}
        return means, per_row
    return means


def make_units(values: np.ndarray, episode_ids: np.ndarray, *, kappa: float = 0.1,
               min_support: int = 10, min_episodes: int = 3,
               pending_batch: int = 200, pending_max_age: int = 1000, seed: int = 0):
    """M6-style open-ended Euclidean DP-means with support/episode gates."""
    values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    episode_ids = np.asarray(episode_ids).astype(str)
    if len(values) != len(episode_ids):
        raise ValueError("value and episode arrays have different row counts")
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(values), size=(min(2048, max(64, len(values))), 2))
    pair_distances = np.linalg.norm(values[pairs[:, 0]] - values[pairs[:, 1]], axis=1)
    median_distance = float(np.median(pair_distances))
    epsilon = max(kappa * median_distance, 1e-12)
    centers: list[np.ndarray] = []
    supports: list[int] = []
    episodes: list[set[str]] = []
    labels = np.full(len(values), -1, dtype=np.int64)
    pending_clusters: list[dict[str, Any]] = []
    fresh_pending: list[int] = []
    history = []
    registered = merged = 0
    data_seen = 0
    expired_total = 0
    max_pending_rows = 0
    order = rng.permutation(len(values))

    def assign(index):
        if not centers:
            return False
        d = np.linalg.norm(np.asarray(centers) - values[index][None, :], axis=1)
        nearest = int(d.argmin())
        if d[nearest] > epsilon:
            return False
        count = supports[nearest]
        centers[nearest] = (centers[nearest] * count + values[index]) / (count + 1)
        supports[nearest] += 1
        episodes[nearest].add(episode_ids[index])
        labels[index] = nearest
        return True

    def flush(force=False):
        nonlocal fresh_pending, pending_clusters, registered, merged, expired_total, max_pending_rows
        if not fresh_pending or (not force and len(fresh_pending) < pending_batch):
            return
        clusters: list[dict[str, Any]] = []
        new_rows = fresh_pending[:pending_batch] if not force else fresh_pending[:]
        fresh_pending = fresh_pending[len(new_rows):]
        if not force:
            for component in pending_clusters:
                component["age"] += pending_batch
        for index in new_rows:
            if not clusters:
                clusters.append({"center": values[index].copy(), "rows": [index],
                                 "episodes": {episode_ids[index]}})
                continue
            d = np.linalg.norm(np.asarray([c["center"] for c in clusters]) - values[index][None, :], axis=1)
            nearest = int(d.argmin())
            if d[nearest] <= epsilon:
                cluster = clusters[nearest]
                cluster["rows"].append(index)
                cluster["center"] = values[cluster["rows"]].mean(axis=0)
                cluster["episodes"].add(episode_ids[index])
            else:
                clusters.append({"center": values[index].copy(), "rows": [index],
                                 "episodes": {episode_ids[index]}})
        for cluster in clusters:
            center = np.asarray(cluster["center"], dtype=np.float32)
            if pending_clusters:
                pd = np.linalg.norm(np.asarray([p["center"] for p in pending_clusters]) - center[None, :], axis=1)
                pending_nearest = int(pd.argmin())
            else:
                pd, pending_nearest = np.asarray([np.inf]), 0
            if pending_clusters and pd[pending_nearest] <= epsilon:
                previous = pending_clusters[pending_nearest]
                old_n, new_n = len(previous["rows"]), len(cluster["rows"])
                previous["center"] = ((previous["center"] * old_n + center * new_n) / (old_n + new_n)).astype(np.float32)
                previous["rows"].extend(cluster["rows"])
                previous["episodes"].update(cluster["episodes"])
                previous["age"] = 0
                candidate = previous
            else:
                candidate = {"center": center.copy(), "rows": list(cluster["rows"]),
                             "episodes": set(cluster["episodes"]), "age": 0}
                pending_clusters.append(candidate)

        # Candidate components that satisfy both support gates become units.
        # Existing unresolved components age only when a new 200-row batch is
        # considered; expired rows stay out of the unit vocabulary and remain
        # available to the observation/action memory at the caller level.
        retained = []
        for component in pending_clusters:
            count = len(component["rows"])
            if count >= min_support and len(component["episodes"]) >= min_episodes:
                center = component["center"]
                if centers:
                    d = np.linalg.norm(np.asarray(centers) - center[None, :], axis=1)
                    nearest = int(d.argmin())
                else:
                    d, nearest = np.asarray([np.inf]), 0
                if centers and d[nearest] <= epsilon:
                    old_n = supports[nearest]
                    centers[nearest] = (centers[nearest] * old_n + center * count) / (old_n + count)
                    supports[nearest] += count
                    episodes[nearest].update(component["episodes"])
                    merged += 1
                    assigned = nearest
                else:
                    centers.append(center.copy())
                    supports.append(count)
                    episodes.append(set(component["episodes"]))
                    registered += 1
                    assigned = len(centers) - 1
                labels[component["rows"]] = assigned
                continue
            if not force and component["age"] >= pending_max_age:
                expired_total += count
                continue
            retained.append(component)
        pending_clusters = retained
        pending_count = sum(len(component["rows"]) for component in pending_clusters) + len(fresh_pending)
        max_pending_rows = max(max_pending_rows, pending_count)
        history.append({"data_seen": int(data_seen), "unit_count": len(centers),
                        "pending_count": pending_count, "registered_total": registered,
                        "merged_total": merged, "expired_total": expired_total})

    for index in order:
        data_seen += 1
        if not assign(int(index)):
            fresh_pending.append(int(index))
            if len(fresh_pending) >= pending_batch:
                flush()
    flush(force=True)
    if not history or history[-1]["data_seen"] != data_seen:
        history.append({"data_seen": int(data_seen), "unit_count": len(centers),
                        "pending_count": sum(len(c["rows"]) for c in pending_clusters) + len(fresh_pending),
                        "registered_total": registered, "merged_total": merged,
                        "expired_total": expired_total})
    pending_indices = [index for component in pending_clusters for index in component["rows"]] + fresh_pending
    return {"centers": np.asarray(centers, np.float32).reshape(len(centers), values.shape[1]),
            "supports": np.asarray(supports, np.int64), "episodes": episodes, "labels": labels,
            "epsilon": epsilon, "median_random_pair_distance": median_distance,
            "pending_indices": np.asarray(pending_indices, np.int64), "pending_count": len(pending_indices),
            "pending_expired": expired_total, "max_pending_rows": max_pending_rows, "history": history,
            "registered": registered, "merged": merged,
            "min_support": min_support, "min_episodes": min_episodes, "kappa": kappa}


def nearest_units(values, centers):
    values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size == 0:
        return np.full(len(values), -1, np.int64), np.full(len(values), np.inf)
    centers = centers.reshape(len(centers), -1)
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=-1)
    ids = distances.argmin(axis=1)
    return ids.astype(np.int64), distances[np.arange(len(values)), ids]


def retrieve_top1(memory_q, memory_v, memory_unit, query_q):
    """Top-1 Euclidean query retrieval; no blending of selected values."""
    memory_q = np.asarray(memory_q, dtype=np.float32)
    query_q = np.asarray(query_q, dtype=np.float32)
    result_v = np.empty((len(query_q), memory_v.shape[1]), np.float32)
    result_unit = np.empty((len(query_q),), np.int64)
    result_index = np.empty((len(query_q),), np.int64)
    memory_norm = np.sum(memory_q * memory_q, axis=1)
    for start in range(0, len(query_q), 128):
        stop = min(len(query_q), start + 128)
        q = query_q[start:stop]
        d2 = np.maximum(np.sum(q * q, axis=1)[:, None] + memory_norm[None, :] - 2.0 * (q @ memory_q.T), 0.0)
        ix = d2.argmin(axis=1)
        result_index[start:stop] = ix
        result_v[start:stop] = memory_v[ix]
        result_unit[start:stop] = memory_unit[ix]
    return result_v, result_unit, result_index


def retrieve_top1_same_class(memory_q, memory_v, memory_unit, query_q,
                             memory_classes, query_classes):
    """Nearest key among same-class memory rows; missing classes remain unmatched."""
    memory_classes = np.asarray(memory_classes)
    query_classes = np.asarray(query_classes)
    values = np.full((len(query_q), memory_v.shape[1]), np.nan, dtype=np.float32)
    units = np.full(len(query_q), -1, dtype=np.int64)
    indices = np.full(len(query_q), -1, dtype=np.int64)
    unmatched = np.zeros(len(query_q), dtype=bool)
    for cls in np.unique(query_classes):
        query_ix = np.flatnonzero(query_classes == cls)
        candidates = np.flatnonzero(memory_classes == cls)
        if not len(candidates):
            unmatched[query_ix] = True
        else:
            v, u, ix = retrieve_top1(memory_q[candidates], memory_v[candidates],
                                     memory_unit[candidates], query_q[query_ix])
            values[query_ix] = v
            units[query_ix] = u
            indices[query_ix] = candidates[ix]
    return values, units, indices, unmatched


def unit_match_metrics(goal_values, centers, retrieved_unit_ids, epsilon):
    """Compare retrieved item units to nearest units of the correct values."""
    gold_ids, distances = nearest_units(goal_values, centers)
    retrieved_unit_ids = np.asarray(retrieved_unit_ids, dtype=np.int64)
    if len(gold_ids) != len(retrieved_unit_ids):
        raise ValueError("gold and retrieved unit rows have different lengths")
    covered = distances <= epsilon
    # An unassigned/retrieved pending item (-1) is a mismatch, even when the
    # correct value has a nearest center. The denominator is every query.
    match_rate = float(np.mean((gold_ids == retrieved_unit_ids) & (retrieved_unit_ids >= 0))) if len(gold_ids) else float("nan")
    assigned = retrieved_unit_ids >= 0
    conditional_match = (float(np.mean(gold_ids[assigned] == retrieved_unit_ids[assigned]))
                         if assigned.any() else None)
    return gold_ids, distances, covered, match_rate, float(np.mean(assigned)), conditional_match


def _masked_mse(prediction, target):
    mask = np.isfinite(target)
    if not mask.any():
        return float("nan")
    return float(np.square(np.asarray(prediction)[mask] - np.asarray(target)[mask]).mean())


def _sample_eval(rows, max_per_task: int, seed: int):
    if max_per_task <= 0:
        return rows
    rng = np.random.default_rng(seed)
    keep = []
    for task in GOAL_TASK_INDICES:
        candidates = np.flatnonzero(rows["task_index"] == task)
        if len(candidates) > max_per_task:
            candidates = np.sort(rng.choice(candidates, size=max_per_task, replace=False))
        keep.extend(candidates.tolist())
    return {name: values[np.asarray(keep, dtype=np.int64)] for name, values in rows.items()}


def _onehot(classes, known_classes):
    class_to_index = {int(value): i for i, value in enumerate(known_classes)}
    unknown = len(class_to_index)
    result = np.zeros((len(classes), unknown + 1), np.float32)
    for i, value in enumerate(np.asarray(classes, dtype=np.int64)):
        result[i, class_to_index.get(int(value), unknown)] = 1.0
    return result


def _query_key(rows):
    return np.concatenate([rows["k"], rows["s"]], axis=-1).astype(np.float32)


def _query_features(rows, class_onehot, variant, key_mean, key_scale):
    key = _standardize(_query_key(rows), key_mean, key_scale)
    if variant in ("none", "same_class"):
        return key
    if variant != "concat":
        raise ValueError(f"unknown retrieval variant: {variant}")
    task = np.asarray(class_onehot, dtype=np.float32).copy()
    # Give the subtask embedding the same per-row vector energy as the full
    # normalized image/state key. This prevents a one-hot block from vanishing
    # beside the higher-dimensional key.
    key_norm = np.linalg.norm(key, axis=1, keepdims=True)
    task_norm = np.linalg.norm(task, axis=1, keepdims=True)
    task *= key_norm / np.maximum(task_norm, 1e-8)
    return np.concatenate([key, task], axis=-1).astype(np.float32)


def _eval_configuration(train, train_v, goal, goal_v, unit, retrieval, q_prediction,
                        flow, flow_norm, *, retrieval_variant, t0, horizon,
                        noise_samples, random_values, normal_mse, normal_per_row, device):
    retrieved_v, retrieved_unit, retrieved_rows, fallback = retrieval
    retrieved_v = np.asarray(retrieved_v, dtype=np.float32).reshape(goal_v.shape)
    known_mask = np.isin(goal["skill_class"], np.unique(train["skill_class"]))
    gold_unit, nearest_distance, covered, match_rate, assigned_fraction, conditional_match = unit_match_metrics(
        goal_v, unit["centers"], retrieved_unit, unit["epsilon"])
    known_dist = nearest_distance[known_mask]
    new_dist = nearest_distance[~known_mask]
    sources = {"memory_top1": retrieved_v, "Q": q_prediction,
               "oracle": goal_v, "random_train": random_values}
    eligible_mask = np.isin(goal["skill_class"], np.unique(train["skill_class"]))
    eligible_rows = int(eligible_mask.sum())
    eligible_assigned = retrieved_unit[eligible_mask] >= 0
    eligible_unit_match = (
        float(np.mean((gold_unit[eligible_mask] == retrieved_unit[eligible_mask]) & eligible_assigned))
        if eligible_rows and len(unit["centers"]) else None)
    eligible_assigned_fraction = (float(np.mean(eligible_assigned)) if eligible_rows else None)
    if retrieval_variant == "same_class":
        mse_mask = ~fallback
        metric_goal = {key: value[mse_mask] for key, value in goal.items()}
        metric_sources = {name: value[mse_mask] for name, value in sources.items()}
        metric_noises = [noise[mse_mask] for noise in noise_samples]
        mse = integrate_flow_sources(flow, flow_norm, metric_goal, metric_noises,
                                     {name: value[:, :horizon] for name, value in metric_sources.items()},
                                     t0, device=device)
        mse["normal"] = float(np.mean(normal_per_row[mse_mask]))
        mse_eligible = mse.copy()
        mse_rows = int(mse_mask.sum())
    else:
        mse, per_row = integrate_flow_sources(flow, flow_norm, goal, noise_samples,
                                              {name: value[:, :horizon] for name, value in sources.items()},
                                              t0, device=device, return_per_row=True)
        mse["normal"] = normal_mse
        mse_eligible = {name: float(np.nanmean(values[eligible_mask]))
                        for name, values in per_row.items()}
        mse_eligible["normal"] = float(np.mean(normal_per_row[eligible_mask]))
        mse_rows = len(goal_v)
    result = {"horizon": horizon, "t0": t0, "retrieval": retrieval_variant,
              "unit_count": len(unit["centers"]), "epsilon": unit["epsilon"],
              "memory_rows": len(train["a"]), "goal_rows": len(goal_v),
              "mse_rows": mse_rows, "eligible_query_rows": eligible_rows,
              "unit_match_rate": match_rate if len(unit["centers"]) else None,
              "retrieved_assigned_fraction": assigned_fraction,
              "unit_match_given_assigned": conditional_match,
              "unit_match_eligible": eligible_unit_match,
              "retrieved_assigned_fraction_eligible": eligible_assigned_fraction,
              "retrieval_unmatched_rows": int(fallback.sum()),
              "coverage_known_motion": float(np.mean(covered[known_mask])) if known_mask.any() else None,
              "coverage_new_motion": float(np.mean(covered[~known_mask])) if (~known_mask).any() else None,
              "nearest_distance_known_median": float(np.median(known_dist)) if known_dist.size else None,
              "nearest_distance_new_median": float(np.median(new_dist)) if new_dist.size else None,
              "known_motion_rows": int(known_mask.sum()), "new_motion_rows": int((~known_mask).sum()),
              "retrieved_indices": retrieved_rows.tolist(), "retrieved_unit_ids": retrieved_unit.tolist(),
              "gold_nearest_unit_ids": gold_unit.tolist()}
    result["mse"] = {name: float(value) for name, value in mse.items()}
    result["mse_eligible"] = {name: float(value) for name, value in mse_eligible.items()}
    result["memory_minus_random"] = result["mse"]["memory_top1"] - result["mse"]["random_train"]
    result["q_minus_memory"] = result["mse"]["Q"] - result["mse"]["memory_top1"]
    result["memory_minus_random_eligible"] = (result["mse_eligible"]["memory_top1"]
                                                - result["mse_eligible"]["random_train"])
    result["q_minus_memory_eligible"] = (result["mse_eligible"]["Q"]
                                         - result["mse_eligible"]["memory_top1"])
    assigned = unit["labels"] >= 0
    train_v_flat = np.asarray(train_v, dtype=np.float32).reshape(len(train_v), -1)
    within = (np.linalg.norm(train_v_flat[assigned] - unit["centers"][unit["labels"][assigned]], axis=1)
              if assigned.any() else np.empty(0))
    if len(unit["centers"]) > 1:
        pair = np.linalg.norm(unit["centers"][:, None, :] - unit["centers"][None, :, :], axis=-1)
        between = pair[np.triu_indices(len(unit["centers"]), 1)]
        ratio = float(np.mean(between) / max(float(np.mean(within)) if len(within) else 0.0, 1e-12))
        result["unit_distance_ratio"] = ratio
    else:
        result["unit_distance_ratio"] = None
    return result


def run_experiment(*, cache_dir: Path, dataset_dir: Path, output_root: Path,
                   epochs: int = 150, batch_size: int = 512, max_eval_per_task: int = 0,
                   seed: int = 0, device_name: str = "cpu", patience: int = 8) -> Path:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    run_id = datetime.now(timezone.utc).strftime("intermediate-action-%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    all_rows = load_rows(cache_dir, dataset_dir, max_horizon=25)
    train_all, goal_all = split_split_a(all_rows)
    goal_all = _sample_eval(goal_all, max_eval_per_task, seed)
    if not len(train_all["task_index"]) or not len(goal_all["task_index"]):
        raise ValueError("Split A requires training tasks 20-39 and goal tasks 10-19")
    train_classes_all = np.unique(train_all["skill_class"])
    retrieval_classes = np.unique(np.concatenate([train_all["skill_class"], goal_all["skill_class"]]))
    retrieval_train_onehot = _onehot(train_all["skill_class"], retrieval_classes)
    retrieval_goal_onehot = _onehot(goal_all["skill_class"], retrieval_classes)
    q_train_onehot = _onehot(train_all["skill_class"], train_classes_all)
    q_goal_onehot = _onehot(goal_all["skill_class"], train_classes_all)
    rng = np.random.default_rng(seed)
    report_rows = []
    training_summaries = []
    unit_summaries = []

    for horizon in HORIZONS:
        valid_train = train_all[f"valid_h{horizon}"]
        valid_goal = goal_all[f"valid_h{horizon}"]
        train = {key: value[valid_train] for key, value in train_all.items()}
        goal = {key: value[valid_goal] for key, value in goal_all.items()}
        goal_h = dict(goal)
        goal_h["a"] = goal["a"][:, :horizon].copy()
        st_train = q_train_onehot[valid_train]
        st_goal = q_goal_onehot[valid_goal]
        task_train = retrieval_train_onehot[valid_train]
        task_goal = retrieval_goal_onehot[valid_goal]
        flow, flow_norm, flow_fit = train_flow(train, horizon=horizon, epochs=epochs,
                                                batch_size=batch_size, seed=seed + horizon,
                                                device=device, patience=patience)
        training_summaries.append({"horizon": horizon, "train_rows": len(train["a"]),
                                   "goal_rows": len(goal["a"]), "flow_best_epoch": flow_fit["best_epoch"],
                                   "flow_best_validation_loss": flow_fit["best_validation_loss"],
                                   "flow_final_validation_loss": flow_fit["history"][-1]["validation_loss"],
                                   "flow_validation_rows": flow_fit["validation_rows"],
                                   "flow_training_rows": flow_fit["training_rows"],
                                   "flow_early_stopped": flow_fit["early_stopped"],
                                   "flow_history": flow_fit["history"]})
        horizon_noise = [rng.standard_normal((len(goal["a"]), horizon * goal["a"].shape[-1])).astype(np.float32)
                         for _ in range(4)]
        normal_metrics, normal_per_row = integrate_flow_sources(
            flow, flow_norm, goal_h, horizon_noise, {"normal": None}, 1.0,
            device=device, return_per_row=True)
        normal_mse = normal_metrics["normal"]
        random_indices = rng.integers(0, len(train["a"]), size=len(goal["a"]))
        key_mean, key_scale = _standardize_fit(_query_key(train))
        dummy_values = np.zeros((len(train["a"]), 1), np.float32)
        dummy_units = np.zeros(len(train["a"]), np.int64)
        query_matches = {}
        for variant in ("none", "concat", "same_class"):
            memory_q = _query_features(train, task_train, variant, key_mean, key_scale)
            goal_q = _query_features(goal, task_goal, variant, key_mean, key_scale)
            if variant == "same_class":
                _, _, indices, fallback = retrieve_top1_same_class(
                    memory_q, dummy_values, dummy_units, goal_q,
                    train["skill_class"], goal["skill_class"])
            else:
                _, _, indices = retrieve_top1(memory_q, dummy_values, dummy_units, goal_q)
                fallback = np.zeros(len(goal["a"]), dtype=bool)
            query_matches[variant] = (indices, fallback)
        target_specs = [("raw", None), *(('dct_lowpass', t) for t in T0_VALUES)]
        for value_kind, target_t0 in target_specs:
            if value_kind == "raw":
                train_v = train["a"][:, :horizon].copy()
                goal_v = goal["a"][:, :horizon].copy()
                selected_coefficients = None
            else:
                train_standard = _standardize(train["a"][:, :horizon].reshape(len(train["a"]), -1),
                                              flow_norm["a_mean"], flow_norm["a_scale"]).reshape(len(train["a"]), horizon, -1)
                goal_standard = _standardize(goal["a"][:, :horizon].reshape(len(goal["a"]), -1),
                                             flow_norm["a_mean"], flow_norm["a_scale"]).reshape(len(goal["a"]), horizon, -1)
                low_train, selected_coefficients, _ = low_frequency_intermediate(train_standard, train_standard, target_t0)
                # H=10 / target t0=.8 can contain no signal-dominant DCT
                # coefficients. Omit that undefined skill-space condition.
                if not selected_coefficients.any():
                    continue
                low_goal, _, _ = low_frequency_intermediate(goal_standard, train_standard, target_t0)
                train_v = (low_train.reshape(len(train["a"]), -1) * flow_norm["a_scale"]
                           + flow_norm["a_mean"]).reshape(len(train["a"]), horizon, -1)
                goal_v = (low_goal.reshape(len(goal["a"]), -1) * flow_norm["a_scale"]
                          + flow_norm["a_mean"]).reshape(len(goal["a"]), horizon, -1)
            unit = make_units(train_v, train["episode_id"], kappa=0.1, min_support=10,
                              min_episodes=3, pending_batch=200, seed=seed + horizon)
            unit_summaries.append({"horizon": horizon, "value_kind": value_kind, "target_t0": target_t0,
                                   "unit_count": len(unit["centers"]), "pending_count": len(unit["pending_indices"]),
                                   "memory_assigned_fraction": float(np.mean(unit["labels"] >= 0)),
                                   "pending_expired": unit["pending_expired"], "max_pending_rows": unit["max_pending_rows"],
                                   "registered": unit["registered"], "merged": unit["merged"],
                                   "epsilon": unit["epsilon"],
                                   "selected_dct_coefficients": (int(selected_coefficients.sum())
                                                                  if selected_coefficients is not None else None),
                                   "unit_supports": unit["supports"].tolist(),
                                   "unit_episode_counts": [len(e) for e in unit["episodes"]],
                                   "unit_count_history": unit["history"]})
            random_v = train_v[random_indices]
            q_models = {}
            for q_subtask in (False, True):
                q_model, q_norm, q_fit = train_v_regressor(
                    train, train_v, st_train, use_subtask=q_subtask, epochs=min(epochs, 50),
                    batch_size=batch_size, seed=seed + horizon * 100 + int(q_subtask) * 11
                    + (int(target_t0 * 100) if target_t0 else 0), device=device,
                    target_mean=flow_norm["a_mean"], target_scale=flow_norm["a_scale"], patience=patience)
                q_prediction = predict_v(q_model, q_norm, goal, st_goal,
                                         use_subtask=q_subtask, device=device)
                q_models[q_subtask] = {"prediction": q_prediction, "fit": q_fit}
            q_loss = {str(key).lower(): _masked_mse(value["prediction"], goal_v)
                      for key, value in q_models.items()}

            memory_v_flat = train_v.reshape(len(train_v), -1)
            for variant in ("none", "concat", "same_class"):
                ri, fallback = query_matches[variant]
                retrieval = (memory_v_flat[ri], unit["labels"][ri], ri, fallback)
                retrieval_v = retrieval[0].reshape(goal_v.shape)
                q_key = variant != "none"
                q_prediction = q_models[q_key]["prediction"]
                for t0 in T0_VALUES:
                    result = _eval_configuration(train, train_v, goal_h, goal_v, unit, retrieval,
                                                 q_prediction, flow, flow_norm,
                                                 retrieval_variant=variant, t0=t0, horizon=horizon,
                                                 noise_samples=horizon_noise, random_values=random_v,
                                                 normal_mse=normal_mse, normal_per_row=normal_per_row["normal"],
                                                 device=device)
                    result.update({"value_kind": value_kind, "target_t0": target_t0,
                                   "subtask": variant != "none", "q_v_mse": q_loss[str(q_key).lower()],
                                   "q_train_loss_first": q_models[q_key]["fit"]["history"][0]["train_loss"],
                                   "q_train_loss_last": q_models[q_key]["fit"]["history"][-1]["train_loss"],
                                   "q_best_epoch": q_models[q_key]["fit"]["best_epoch"],
                                   "q_best_validation_loss": q_models[q_key]["fit"]["best_validation_loss"],
                                   "dct_coefficients_kept": (int(selected_coefficients.sum())
                                                              if selected_coefficients is not None else None),
                                   "memory_contains_goal": False,
                                   "goal_motion_labels": {"known": result["known_motion_rows"],
                                                          "new": result["new_motion_rows"]}})
                    report_rows.append(result)
                    print(f"H={horizon} {value_kind} target={target_t0} t0={t0} retrieval={variant}: "
                          f"F={normal_mse:.4f} mem={result['mse']['memory_top1']:.4f} "
                          f"random={result['mse']['random_train']:.4f} match={result['unit_match_rate']}", flush=True)

    summary = {"seed": seed, "split": "A: train task_index 20-39 object+spatial; goal task_index 10-19 held out",
               "data": {"cache": str(cache_dir), "dataset": str(dataset_dir),
                        "train_rows_by_task": {str(i): int(np.sum(train_all["task_index"] == i))
                                               for i in TRAIN_TASK_INDICES},
                        "goal_rows_by_task": {str(i): int(np.sum(goal_all["task_index"] == i))
                                              for i in GOAL_TASK_INDICES},
                        "train_episode_count": int(len(np.unique(train_all["episode_id"]))),
                        "goal_episode_count": int(len(np.unique(goal_all["episode_id"]))),
                        "goal_query_cap_per_task": max_eval_per_task},
               "method": {"flow": "x_t=t*epsilon+(1-t)*A; train v=epsilon-A; Euler t=1→0, 10 steps",
                          "flow_validation": "episode-level 15% holdout, early stopping on fixed-noise flow-matching validation loss",
                          "intermediate_t_values": list(T0_VALUES),
                          "dct_rule": "train coefficient variance*(1-t0)^2 > t0^2; zero-coefficient configuration omitted",
                          "units": "open-ended Euclidean DP-means; epsilon=0.1*median random training pair distance; pending batch 200; m=10 and >=3 episodes; unresolved pending components expire after 1000 arrivals",
                          "retrieval": "top-1: (a) q=(k,s); (b) concatenate class one-hot with its norm scaled to the normalized key norm; (c) same-class candidate filter then nearest (k,s); unmatched classes have no retrieval and are excluded from c's MSE denominator",
                          "direct_regressor": "Q(subtask,f,s)->v; episode-level validation early stopping",
                          "goal_added_to_memory": False},
               "flow_training": training_summaries, "skill_units": unit_summaries, "results": report_rows}
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (run_dir / "report.md").write_text(render_report(summary))
    return run_dir


def render_report(data):
    cap = data["data"]["goal_query_cap_per_task"]
    lines = ["# Intermediate action skill experiment", "", data["split"], "",
             f"Seed: {data['seed']}; goal query cap/task: {'none (all valid rows)' if cap <= 0 else cap}", "",
             "Metric is finite-element masked action MSE; lower is better. Goal rows are query-only and never enter memory.",
             "Subtask modes: `none`=(k,s); `concat`=scaled class embedding+(k,s); `same_class`=only same-class candidates, then (k,s) top-1. If a class has no training items, it has no retrieval; its rows are excluded from that mode's MSE and counted below.", "",
             "## Flow head fit", "",
             "Validation episodes are held out by episode ID (15%); early stopping uses fixed-noise flow-matching validation loss.", "",
             "| H | fit rows | validation rows | best epoch | best validation loss | final validation loss | stopped early |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for fit in data["flow_training"]:
        lines.append(f"| {fit['horizon']} | {fit['flow_training_rows']} | {fit['flow_validation_rows']} | "
                     f"{fit['flow_best_epoch']} | {fit['flow_best_validation_loss']:.6f} | "
                     f"{fit['flow_final_validation_loss']:.6f} | "
                     f"{'yes' if fit['flow_early_stopped'] else 'no'} |")
    lines += ["", "Train and validation losses by epoch (full data are also in `results.json`):", "",
              "| epoch | H=25 train | H=25 validation | H=10 train | H=10 validation |",
              "|---:|---:|---:|---:|---:|"]
    by_horizon = {fit["horizon"]: {entry["epoch"]: entry for entry in fit["flow_history"]}
                  for fit in data["flow_training"]}
    for epoch in range(1, max((len(x) for x in by_horizon.values()), default=0) + 1):
        values = []
        for horizon in HORIZONS:
            point = by_horizon.get(horizon, {}).get(epoch)
            values.extend([f"{point['train_loss']:.6f}" if point else "—",
                           f"{point['validation_loss']:.6f}" if point else "—"])
        lines.append(f"| {epoch} | " + " | ".join(values) + " |")
    lines += ["", "## Goal query results", "",
              "`memory-random` isolates the retrieval contribution; `Q-memory` compares direct regression against retrieval. Negative values favor memory top-1.", "",
              "| H | v(A) | target t0 | eval t0 | retrieval | MSE rows | normal F | memory top-1 | random v | memory-random | Q | Q-memory | oracle v | unit match/assigned | known/new coverage | no-class rows |",
              "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["results"]:
        cov = f"{row['coverage_known_motion'] if row['coverage_known_motion'] is not None else float('nan'):.3f}/{row['coverage_new_motion'] if row['coverage_new_motion'] is not None else float('nan'):.3f}"
        match = (f"{row['unit_match_rate']:.3f}/{row['retrieved_assigned_fraction']:.3f}"
                 if row["unit_match_rate"] is not None else "n/a")
        m = row["mse"]
        target_t0 = "—" if row["target_t0"] is None else f"{row['target_t0']:.1f}"
        lines.append(f"| {row['horizon']} | {row['value_kind']} | {target_t0} | {row['t0']:.1f} | {row['retrieval']} | {row['mse_rows']} | "
                     f"{m['normal']:.4f} | {m['memory_top1']:.4f} | {m['random_train']:.4f} | "
                     f"{row['memory_minus_random']:+.4f} | {m['Q']:.4f} | {row['q_minus_memory']:+.4f} | "
                     f"{m['oracle']:.4f} | {match} | {cov} | {row['retrieval_unmatched_rows']} |")
    lines += ["", "### Matched query population", "",
              "Fair comparison for (c): all modes are recomputed on goal rows whose eval_skill_class occurs in training. This excludes classes where (c) has no legal candidate; row count and no-class fraction remain visible in the main table.", "",
              "| H | retrieval | eligible rows | normal F | memory top-1 | random v | memory-random | Q | Q-memory | unit match (eligible) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        for variant in ("none", "concat", "same_class"):
            group = [r for r in data["results"] if r["horizon"] == horizon and r["retrieval"] == variant]
            if not group:
                continue
            eligible_mse = {name: float(np.mean([r["mse_eligible"][name] for r in group]))
                            for name in ("normal", "memory_top1", "random_train", "Q")}
            eligible_matches = [r["unit_match_eligible"] for r in group if r["unit_match_eligible"] is not None]
            match = float(np.mean(eligible_matches)) if eligible_matches else float("nan")
            lines.append(f"| {horizon} | {variant} | {group[0]['eligible_query_rows']} | "
                         f"{eligible_mse['normal']:.4f} | {eligible_mse['memory_top1']:.4f} | "
                         f"{eligible_mse['random_train']:.4f} | "
                         f"{eligible_mse['memory_top1'] - eligible_mse['random_train']:+.4f} | "
                         f"{eligible_mse['Q']:.4f} | {eligible_mse['Q'] - eligible_mse['memory_top1']:+.4f} | "
                         f"{match:.3f} |")
    lines += ["", "## Interpreting the comparison", "",
              "Coverage is the fraction whose correct v is within epsilon of its nearest learned unit. The corresponding median nearest-unit distances are in `results.json`; zero coverage means all nearest distances exceed epsilon, not a missing denominator.", "",
              "Unit match compares the retrieved memory item's stored unit ID with the nearest unit to the correct v. The table shows all-query exact match / retrieved-assigned fraction; pending/unassigned items (-1) count as incorrect. `results.json` includes both ID arrays for auditing.", "",
              "DCT coefficient counts, unit supports/episode counts, pending counts, unit-count histories, and Q validation fits are in `results.json`.", ""]
    lines[-1:-1] = ["## Skill-unit coverage", "",
                    "Coverage and median distance are independent of the retrieval mode; this table uses the same gold v/learned units for all three query modes.", "",
                    "| H | v(A) | target t0 | units | epsilon | assigned memory fraction | pending | coverage known/new | median distance known/new | distance ratio |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if lines and lines[-1] == "":
        lines.pop()
    for unit in data["skill_units"]:
        row = next(r for r in data["results"] if r["horizon"] == unit["horizon"]
                   and r["value_kind"] == unit["value_kind"] and r["target_t0"] == unit["target_t0"]
                   and r["retrieval"] == "none")
        cov = f"{row['coverage_known_motion']:.3f}/{row['coverage_new_motion']:.3f}"
        med = f"{row['nearest_distance_known_median']:.3f}/{row['nearest_distance_new_median']:.3f}"
        ratio = row["unit_distance_ratio"]
        target_t0 = "—" if unit["target_t0"] is None else f"{unit['target_t0']:.1f}"
        lines.append(f"| {unit['horizon']} | {unit['value_kind']} | {target_t0} | {unit['unit_count']} | "
                     f"{unit['epsilon']:.3f} | {unit['memory_assigned_fraction']:.3f} | {unit['pending_count']} | "
                     f"{cov} | {med} | {ratio if ratio is not None else float('nan'):.2f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-eval-per-task", type=int, default=0,
                        help="Deterministic per-task query cap; <=0 evaluates all goal rows")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    path = run_experiment(cache_dir=args.cache_dir, dataset_dir=args.dataset,
                          output_root=args.output_root, epochs=args.epochs,
                          batch_size=args.batch_size, max_eval_per_task=args.max_eval_per_task,
                          seed=args.seed, device_name=args.device, patience=args.patience)
    print(f"output: {path}")


if __name__ == "__main__":
    main()
