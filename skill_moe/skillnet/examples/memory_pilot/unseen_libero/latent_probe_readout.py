#!/usr/bin/env python3
"""Train/evaluate the shared small readout used by LP1 latent probes.

This module defines the reusable model, objectives, metrics, keyed feature
joins, and a resumable-by-checkpoint training primitive. It deliberately does
not launch the expensive multi-condition/three-seed production sweep on import.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONDITIONS = (
    ("L-none", "P-on"),
    ("L-vjepa", "P-off"), ("L-vjepa", "P-on"),
    ("L-vlm", "P-off"), ("L-vlm", "P-on"),
)


@dataclass(frozen=True)
class ReadoutConfig:
    latent_dim: int
    proprio_dim: int = 78  # state[8] + previous executed actions[10,7]
    key_dim: int = 32
    num_queries: int = 4
    width: int = 256
    heads: int = 4
    layers: int = 2
    use_proprio: bool = True
    dropout: float = 0.1


class CrossReadLayer(nn.Module):
    def __init__(self, width=256, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * width, width), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(width)

    def forward(self, q, kv, key_padding_mask=None):
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm1(q + a)
        return self.norm2(q + self.ff(q))


class LatentProbeReadout(nn.Module):
    """Four learned queries attend to frozen latent tokens, then share heads."""
    def __init__(self, config: ReadoutConfig):
        super().__init__()
        self.config = config
        self.latent_proj = nn.Sequential(nn.LayerNorm(config.latent_dim),
                                         nn.Linear(config.latent_dim, config.width))
        self.query_tokens = nn.Parameter(torch.randn(1, config.num_queries, config.width) * 0.02)
        self.layers = nn.ModuleList([CrossReadLayer(config.width, config.heads, config.dropout)
                                     for _ in range(config.layers)])
        self.proprio = (nn.Sequential(nn.LayerNorm(config.proprio_dim),
                                     nn.Linear(config.proprio_dim, config.width), nn.GELU(),
                                     nn.Linear(config.width, config.width))
                        if config.use_proprio else None)
        fusion_in = config.width * (2 if config.use_proprio else 1)
        self.fusion = nn.Sequential(nn.LayerNorm(fusion_in), nn.Linear(fusion_in, config.width),
                                    nn.GELU(), nn.Dropout(config.dropout),
                                    nn.Linear(config.width, config.width), nn.GELU())
        self.gripper_head = nn.Linear(config.width, 10)
        self.switch_head = nn.Linear(config.width, 1)
        self.switch_time_head = nn.Linear(config.width, 10)
        self.rms_head = nn.Linear(config.width, 6)
        self.key_proj = nn.Sequential(nn.LayerNorm(config.key_dim), nn.Linear(config.key_dim, config.width),
                                      nn.GELU(), nn.Linear(config.width, config.width))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, latent, proprio=None, candidate_keys=None, latent_padding_mask=None):
        if latent.ndim != 3:
            raise ValueError(f"latent must be [B,T,D], got {tuple(latent.shape)}")
        if latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latent width {latent.shape[-1]} != {self.config.latent_dim}")
        kv = self.latent_proj(latent)
        q = self.query_tokens.expand(latent.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, kv, latent_padding_mask)
        pooled = q.mean(dim=1)
        if self.config.use_proprio:
            if proprio is None or proprio.ndim != 2 or proprio.shape[-1] != self.config.proprio_dim:
                raise ValueError(f"P-on expects [B,{self.config.proprio_dim}] proprio")
            pooled = torch.cat([pooled, self.proprio(proprio)], dim=-1)
        fused = self.fusion(pooled)
        result = {
            "gripper_logits": self.gripper_head(fused),
            "switch_logit": self.switch_head(fused).squeeze(-1),
            "switch_time_logits": self.switch_time_head(fused),
            "log_rms6": self.rms_head(fused),
            "query_embedding": F.normalize(fused, dim=-1),
        }
        if candidate_keys is not None:
            if candidate_keys.ndim == 2:
                candidate_keys = candidate_keys.unsqueeze(0).expand(latent.shape[0], -1, -1)
            if candidate_keys.ndim != 3 or candidate_keys.shape[0] != latent.shape[0]:
                raise ValueError("candidate_keys must be [C,key_dim] or [B,C,key_dim]")
            ck = F.normalize(self.key_proj(candidate_keys), dim=-1)
            scale = self.logit_scale.clamp(math.log(1), math.log(100)).exp()
            result["candidate_logits"] = scale * torch.einsum("bd,bcd->bc", result["query_embedding"], ck)
        return result


def multitask_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], *,
                   grip_weight=1.0, switch_weight=1.0, type_weight=1.0, rms_weight=0.1):
    """Equal-interface loss; padded/forbidden candidates are excluded exactly."""
    losses = {}
    losses["gripper"] = F.binary_cross_entropy_with_logits(
        output["gripper_logits"], batch["gripper_targets"].float())
    losses["switch"] = F.binary_cross_entropy_with_logits(
        output["switch_logit"], batch["switch_target"].float())
    if "switch_step" in batch:
        switched = batch["switch_target"].bool()
        if switched.any():
            step_target = batch["switch_step"][switched].long().clamp(0, 9)
            losses["switch_time"] = F.cross_entropy(output["switch_time_logits"][switched], step_target)
        else:
            losses["switch_time"] = output["switch_time_logits"].sum() * 0.0
    else:
        losses["switch_time"] = output["switch_time_logits"].sum() * 0.0
    if "candidate_logits" in output:
        logits = output["candidate_logits"]
        allowed = batch.get("candidate_allowed")
        target = batch["type_target"].float()
        if allowed is not None:
            allowed = allowed.bool()
            if torch.any(~allowed.any(dim=-1)):
                raise ValueError("each query must have at least one allowed candidate")
            logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
            target = target.masked_fill(~allowed, 0)
        target_sum = target.sum(dim=-1, keepdim=True)
        if torch.any(target_sum <= 0):
            raise ValueError("type target must have positive mass over allowed candidates")
        target = target / target_sum
        losses["type"] = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    else:
        losses["type"] = output["query_embedding"].sum() * 0.0
    if "rms6_target" in batch:
        losses["rms"] = F.smooth_l1_loss(output["log_rms6"], batch["rms6_target"].float())
    else:
        losses["rms"] = output["log_rms6"].sum() * 0.0
    total = (grip_weight * losses["gripper"] + switch_weight * losses["switch"] +
             0.25 * switch_weight * losses["switch_time"] +
             type_weight * losses["type"] + rms_weight * losses["rms"])
    return total, {k: float(v.detach().cpu()) for k, v in losses.items()}


def check_prediction_variance(prediction, *, min_std=1e-4, label="prediction"):
    """Fail fast if a readout collapsed to constant outputs across examples."""
    x = np.asarray(prediction, dtype=np.float64)
    if x.ndim < 2 or x.shape[0] < 2:
        raise ValueError(f"{label} needs at least two examples")
    std = np.std(x.reshape(x.shape[0], -1), axis=0)
    q10 = float(np.quantile(std, 0.10))
    if not np.isfinite(std).all() or float(np.max(std)) < min_std:
        raise AssertionError(f"{label} collapsed: maximum per-output std={float(np.max(std)):.3g}")
    return {"min_std": float(std.min()), "median_std": float(np.median(std)),
            "q10_std": q10, "max_std": float(std.max()), "collapsed": False}


def gripper_dev_task_split(task_ids, *, positions=(0, 8, 16, 24)):
    """Fixed task-level grip Dev rule: sorted fit IDs at indices 0,8,16,24."""
    tasks = np.asarray(task_ids, dtype=np.int64)
    unique = np.unique(tasks)
    if len(unique) <= max(positions):
        raise ValueError(f"need at least {max(positions)+1} fit tasks, got {len(unique)}")
    held = unique[np.asarray(positions, dtype=np.int64)]
    if len(np.unique(held)) != len(positions):
        raise AssertionError("gripper Dev task selection contains duplicates")
    return {"fit_tasks_sorted": unique.tolist(), "heldout_positions": list(map(int, positions)),
            "gripper_dev_tasks": held.tolist(), "train_tasks": np.setdiff1d(unique, held).tolist()}


def _binary_auc(target, score):
    """Tie-correct Mann-Whitney AUC; returns NaN if either class is absent."""
    y = np.asarray(target, bool).reshape(-1)
    s = np.asarray(score, np.float64).reshape(-1)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def gripper_metrics(gripper_logits, switch_logits, gripper_targets, switch_target, *,
                    switch_step=None, switch_time_logits=None, threshold=0.5):
    """Metrics requested by LP1, including event timing and no-event FPR."""
    gl = np.asarray(gripper_logits)
    sl = np.asarray(switch_logits).reshape(-1)
    gt = np.asarray(gripper_targets).astype(bool)
    st = np.asarray(switch_target).astype(bool).reshape(-1)
    if gl.shape != gt.shape or gl.ndim != 2 or gl.shape[1] != 10 or len(sl) != len(st) or len(st) != len(gt):
        raise ValueError("gripper metric arrays must be aligned [N,10] and [N]")
    switch_prob = 1 / (1 + np.exp(-np.clip(sl, -40, 40)))
    pred_event = switch_prob >= threshold
    tp, fp, fn = int((pred_event & st).sum()), int((pred_event & ~st).sum()), int((~pred_event & st).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_sign = 1 / (1 + np.exp(-np.clip(gl, -40, 40))) >= 0.5
    pred_change = pred_sign[:, 1:] != pred_sign[:, :-1]
    pred_step = np.where(pred_change.any(axis=1), pred_change.argmax(axis=1) + 1, 0)
    if switch_step is not None:
        true_step = np.asarray(switch_step, dtype=np.int64).reshape(-1).clip(0, 9)
    else:
        true_change = gt[:, 1:] != gt[:, :-1]
        true_step = np.where(true_change.any(axis=1), true_change.argmax(axis=1) + 1, 0)
    if switch_time_logits is not None:
        pred_step = np.asarray(switch_time_logits).argmax(axis=1)
    switch_rows = np.flatnonzero(st)
    timing = np.abs(pred_step[switch_rows] - true_step[switch_rows]) if len(switch_rows) else np.asarray([])
    return {"switch_auc": _binary_auc(st, switch_prob), "precision": precision, "recall": recall,
            "f1": f1, "false_switch_rate_no_switch": float(pred_event[~st].mean()) if np.any(~st) else float("nan"),
            "switch_time_abs_error_mean": float(timing.mean()) if len(timing) else float("nan"),
            "switch_time_abs_error_median": float(np.median(timing)) if len(timing) else float("nan"),
            "n_switch": int(st.sum()), "n_no_switch": int((~st).sum())}


def join_keyed_rows(source_keys, feature_keys, *, require_all=True):
    """Return source row indices in feature order; keys are exact triplets."""
    src = np.asarray(source_keys, dtype=np.int64)
    feat = np.asarray(feature_keys, dtype=np.int64)
    if src.ndim != 2 or feat.ndim != 2 or src.shape[1] != 3 or feat.shape[1] != 3:
        raise ValueError("row keys must be [N,3] task/episode/frame arrays")
    lookup = {tuple(map(int, key)): i for i, key in enumerate(src)}
    if len(lookup) != len(src):
        raise ValueError("duplicate source row keys")
    missing = [tuple(map(int, k)) for k in feat if tuple(map(int, k)) not in lookup]
    if missing and require_all:
        raise KeyError(f"{len(missing)} feature keys absent from source; first={missing[:3]}")
    return np.asarray([lookup.get(tuple(map(int, k)), -1) for k in feat], dtype=np.int64)


def load_supervision_rows(data_root, split):
    """Load aligned labels and reconstruct causal 10-step action history.

    History is zero-left-padded and contains only actions strictly before the
    query frame. The returned labels are separated from the input arrays so a
    caller can construct model inputs without accidentally passing targets.
    """
    root = Path(data_root) / split
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"incomplete LP1 data split: {root}")
    load = lambda name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    keys = np.stack([load(k) for k in ("task", "episode", "frame")], axis=1).astype(np.int32)
    n = len(keys)
    traj_task, traj_ep = load("trajectory_task"), load("trajectory_episode")
    traj_offsets = load("trajectory_offsets")
    lookup = {(int(t), int(e)): i for i, (t, e) in enumerate(zip(traj_task, traj_ep))}
    if len(lookup) != len(traj_task):
        raise ValueError("duplicate trajectory task/episode keys")
    trajectories = load("trajectory_actions_normalized")
    history = np.zeros((n, 10, 7), np.float32)
    history_valid = np.zeros((n, 10), np.bool_)
    for i, (task, episode, frame) in enumerate(keys):
        ti = lookup.get((int(task), int(episode)))
        if ti is None:
            raise KeyError(f"missing source trajectory for query key {tuple(map(int, keys[i]))}")
        start, end = map(int, traj_offsets[ti])
        if frame > end - start:
            raise ValueError(f"query frame exceeds source trajectory: {tuple(map(int, keys[i]))}")
        lo = max(0, int(frame) - 10)
        count = int(frame) - lo
        if count:
            history[i, 10-count:] = trajectories[start + lo:start + int(frame), :7]
            history_valid[i, 10-count:] = True
    return {
        "keys": keys,
        "state": np.asarray(load("state_normalized"), dtype=np.float32),
        "proprio": np.concatenate([np.asarray(load("state_normalized"), dtype=np.float32),
                                   history.reshape(n, 70)], axis=1),
        "history": history,
        "history_valid": history_valid,
        "actions_physical": load("actions_physical"),
        "actions_normalized": load("actions_normalized"),
        "gripper_targets": (np.asarray(load("gripper_signs")) > 0).astype(np.float32),
        "switch_step": np.asarray(load("first_gripper_switch"), dtype=np.int64),
        "switch_target": (np.asarray(load("first_gripper_switch")) >= 0).astype(np.float32),
        "log_rms6_target": np.log(np.maximum(np.asarray(load("rms6_normalized"), dtype=np.float32), 1e-6)),
        "rms6_physical_target": np.asarray(load("rms6_physical"), dtype=np.float32),
        "task": keys[:, 0], "episode": keys[:, 1], "frame": keys[:, 2],
    }


def corrupt_history(proprio, mode, *, donor_history=None):
    """Return P-on input with causal action history zeroed or episode-swapped."""
    x = torch.as_tensor(proprio).clone()
    if x.ndim != 2 or x.shape[1] != 78:
        raise ValueError("proprio must be [B, state8 + history10x7]")
    if mode == "intact":
        return x
    if mode == "zero":
        x[:, 8:] = 0
        return x
    if mode == "swap":
        if donor_history is None:
            raise ValueError("swap mode needs donor histories from other episodes")
        donor = torch.as_tensor(donor_history, dtype=x.dtype, device=x.device)
        if donor.shape == (x.shape[0], 10, 7):
            donor = donor.reshape(x.shape[0], 70)
        if donor.shape != (x.shape[0], 70):
            raise ValueError("donor history must be [B,10,7] or [B,70]")
        x[:, 8:] = donor
        return x
    raise ValueError(f"unknown history corruption mode: {mode}")


def load_keyed_vlm_cache(tokens_path, groups_path, source_keys, feature_keys, *, require_all=False):
    """Join cached VLM tokens by exact row identity; report and omit key gaps.

    `source_keys` describes the VLM cache row order, `feature_keys` the target
    LP1 rows. No interpolation, nearest-row substitution, or label-based fill
    is allowed. The returned indices map feature rows into cache rows.
    """
    cache_tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    cache_groups = np.load(groups_path, mmap_mode="r", allow_pickle=False)
    source_keys = np.asarray(source_keys, dtype=np.int64)
    feature_keys = np.asarray(feature_keys, dtype=np.int64)
    if len(cache_tokens) != len(source_keys) or len(cache_groups) != len(source_keys):
        raise ValueError("VLM token/group cache rows do not align with source keys")
    lookup = {tuple(map(int, key)): i for i, key in enumerate(source_keys)}
    if len(lookup) != len(source_keys):
        raise ValueError("duplicate VLM cache keys")
    mapped = np.asarray([lookup.get(tuple(map(int, k)), -1) for k in feature_keys], dtype=np.int64)
    valid = mapped >= 0
    if require_all and not valid.all():
        raise KeyError(f"{int((~valid).sum())} LP1 rows are absent from VLM cache")
    ix = mapped[valid]
    return {"tokens": cache_tokens[ix], "groups": cache_groups[ix],
            "keys": feature_keys[valid], "source_indices": ix,
            "valid_feature_indices": np.flatnonzero(valid),
            "excluded_missing_keys": int((~valid).sum()),
            "feature_rows": int(len(feature_keys)), "matched_rows": int(valid.sum())}


def soft_type_distribution(type_errors, allowed, *, tau=0.01):
    """Create the LPG-style soft target from S-shared type errors."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    err = torch.as_tensor(type_errors, dtype=torch.float32)
    mask = torch.as_tensor(allowed, dtype=torch.bool, device=err.device)
    if err.shape != mask.shape or err.ndim != 2:
        raise ValueError("type_errors and allowed must share [B,C] shape")
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("each row must allow at least one candidate")
    logits = (-err / float(tau)).masked_fill(~mask, torch.finfo(err.dtype).min)
    target = torch.softmax(logits, dim=-1)
    if not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite soft type targets")
    return target


def _rankdata(values):
    x = np.asarray(values, np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def retrieval_metrics(scores, type_errors, allowed, *, tau=0.01, ks=(1, 5, 20)):
    """Per-query retrieval metrics; positive correlation means better ranking."""
    s = np.asarray(scores, np.float64)
    e = np.asarray(type_errors, np.float64)
    m = np.asarray(allowed, bool)
    if s.shape != e.shape or s.shape != m.shape or s.ndim != 2:
        raise ValueError("scores/errors/allowed must share [queries,candidates]")
    top1, rho, recall = [], [], {int(k): [] for k in ks}
    for qi in range(len(s)):
        ix = np.flatnonzero(m[qi])
        if not len(ix):
            continue
        order = ix[np.argsort(-s[qi, ix], kind="mergesort")]
        err_order = e[qi, ix]
        top1.append(float(e[qi, order[0]]))
        rho.append(_pearson(_rankdata(s[qi, ix]), _rankdata(-e[qi, ix])))
        positives = e[qi, ix] <= float(tau)
        pos_ids = set(map(int, ix[positives]))
        for k in ks:
            kk = min(int(k), len(order))
            recall[int(k)].append(float(any(int(j) in pos_ids for j in order[:kk])) if pos_ids else float("nan"))
    return {"n_queries": len(top1), "top1_type_error_mean": float(np.mean(top1)) if top1 else float("nan"),
            "top1_type_error_median": float(np.median(top1)) if top1 else float("nan"),
            "spearman_score_vs_negative_type_error_mean": float(np.mean(rho)) if rho else float("nan"),
            "spearman_score_vs_negative_type_error_median": float(np.median(rho)) if rho else float("nan"),
            "spearman_constant_window_count": int(sum(_query_constant(s[i], m[i]) for i in range(len(s)))),
            "recall": {str(k): float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in recall.items()}}


def _query_constant(scores, allowed):
    x = np.asarray(scores)[np.asarray(allowed, bool)]
    return len(x) < 2 or np.all(x == x[0])


def episode_bootstrap_delta(values_a, values_b, episodes, *, samples=5000, seed=4101):
    """Paired episode bootstrap CI for mean(A-B), resampling whole episodes."""
    a, b = np.asarray(values_a, np.float64), np.asarray(values_b, np.float64)
    ep = np.asarray(episodes)
    if a.shape != b.shape or a.shape[0] != ep.shape[0]:
        raise ValueError("paired arrays and episode ids must align")
    labels = [tuple(row.tolist()) for row in ep] if ep.ndim > 1 else ep.tolist()
    unique = list(dict.fromkeys(labels))
    if not len(unique):
        raise ValueError("cannot bootstrap an empty evaluation")
    by_episode = {e: np.asarray([i for i, label in enumerate(labels) if label == e], dtype=np.int64)
                  for e in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(samples), np.float64)
    for i in range(int(samples)):
        sampled = rng.integers(0, len(unique), size=len(unique))
        ix = np.concatenate([by_episode[unique[j]] for j in sampled])
        estimates[i] = float(np.mean(a[ix] - b[ix]))
    return {"mean_delta": float(np.mean(a - b)), "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)), "episodes": int(len(unique)),
            "bootstrap_samples": int(samples), "seed": int(seed)}


def set_seed(seed: int):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(model, train_batches, dev_batches, *, updates=20_000, learning_rate=3e-4,
                 weight_decay=1e-4, eval_every=500, patience=8, seed=4101,
                 checkpoint_path=None, device="cuda", dev_metric="loss"):
    """Generic Dev-only early-stop trainer. Batches yield model kwargs + targets.

    A production driver should create fixed episode-disjoint iterators and
    provide `dev_metric`; the held-out Eval iterator must never be passed here.
    Each checkpoint stores optimizer-independent live model weights and curve.
    """
    if updates < 1 or eval_every < 1:
        raise ValueError("updates and eval_every must be positive")
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_it = iter(train_batches)
    best_value, best_step, bad = float("inf"), 0, 0
    curve = []
    for step in range(1, int(updates) + 1):
        model.train()
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_batches)
            batch = next(train_it)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["latent"], batch.get("proprio"), batch.get("candidate_keys"),
                    batch.get("latent_padding_mask"))
        loss, loss_parts = multitask_loss(out, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite train loss at step={step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every and step != updates:
            continue
        dev = evaluate_loss(model, dev_batches, device=device)
        metric_key = "total" if dev_metric == "loss" else dev_metric
        if metric_key not in dev:
            raise KeyError(f"unknown Dev selection metric {dev_metric!r}; choices={sorted(dev)}")
        val = float(dev[metric_key])
        item = {"step": step, "train_loss": float(loss.detach().cpu()), **{f"train_{k}": v for k, v in loss_parts.items()},
                **{f"dev_{k}": v for k, v in dev.items()}}
        curve.append(item)
        if val < best_value:
            best_value, best_step, bad = val, step, 0
            if checkpoint_path:
                _save_checkpoint(Path(checkpoint_path), model, optimizer, step, seed, curve)
        else:
            bad += 1
            if bad >= patience:
                break
    if checkpoint_path and Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    return {"best_step": best_step, "best_dev": best_value, "updates_completed": curve[-1]["step"] if curve else 0,
            "curve": curve}


@torch.no_grad()
def evaluate_loss(model, batches, *, device="cuda"):
    model.eval()
    sums = {"total": 0.0, "gripper": 0.0, "switch": 0.0, "switch_time": 0.0,
            "type": 0.0, "rms": 0.0}
    n = 0
    for batch in batches:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["latent"], batch.get("proprio"), batch.get("candidate_keys"),
                    batch.get("latent_padding_mask"))
        loss, parts = multitask_loss(out, batch)
        b = int(batch["latent"].shape[0])
        sums["total"] += float(loss.cpu()) * b
        for k, v in parts.items():
            sums[k] += v * b
        n += b
    if not n:
        raise ValueError("empty Dev batches")
    return {k: v / n for k, v in sums.items()}


def _save_checkpoint(path, model, optimizer, step, seed, curve):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": int(step), "seed": int(seed), "curve": curve,
                "config": vars(model.config)}, tmp)
    tmp.replace(path)


def save_prediction_collapse_audit(path, predictions):
    audit = check_prediction_variance(predictions)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite collapse audit: {path}")
    path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def train_seed_sweep(model_factory, train_batches_factory, dev_batches_factory, output_dir, *,
                     seeds=(4101, 4102, 4103), updates=20_000, learning_rate=3e-4,
                     eval_every=500, patience=8, device="cuda", dev_metric="loss"):
    """Train a fixed condition for the requested seeds, selecting on Dev only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        seed_dir = output_dir / f"seed-{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        curve_path = seed_dir / "curve.json"
        if curve_path.exists():
            raise FileExistsError(f"refusing to overwrite completed seed result: {curve_path}")
        model = model_factory()
        result = run_training(model, train_batches_factory(int(seed)),
            dev_batches_factory(int(seed)), updates=updates, learning_rate=learning_rate,
            eval_every=eval_every, patience=patience, seed=int(seed),
            checkpoint_path=seed_dir / "best.pt", device=device, dev_metric=dev_metric)
        curve_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append({"seed": int(seed), "best_step": result["best_step"], "best_dev": result["best_dev"]})
    summary = {"seeds": list(map(int, seeds)), "updates_per_seed": int(updates),
               "selection_split": "libero90_dev", "dev_metric": dev_metric, "results": results}
    summary_path = output_dir / "seed_sweep.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite seed summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---- LP1 runnable experiment driver -------------------------------------------------
LP1_ROOT = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01")
LP1_MEMORY = Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01")
LP1_UNSEEN = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/unseen-rows-02")
LP1_SCALE_PREP = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/scale-only-02/production/prep")
LP1_GATE_SUMMARY = LP1_ROOT / "gate-full/summary.json"


def _row_lookup(keys):
    return {tuple(map(int, row)): i for i, row in enumerate(np.asarray(keys))}


class _Rows:
    """LP1 row labels plus one exact-key latent source; arrays stay mmap-backed."""
    def __init__(self, split, data_root, modality, vjepa_root=None):
        self.split = split
        self.data = load_supervision_rows(data_root, split)
        self.keys = np.asarray(self.data["keys"], np.int32)
        self.modality = modality
        self.indices = np.arange(len(self.keys), dtype=np.int64)
        self.latent_source = self.mask_source = None
        if modality == "vlm":
            if split == "libero40_fit":
                source_root = LP1_MEMORY / "rows"
                source_keys = np.stack([np.load(source_root / f"{x}.npy", mmap_mode="r")
                                        for x in ("task", "episode", "frame")], axis=1)
                self.latent_source = np.load(LP1_MEMORY / "tokens.npy", mmap_mode="r")
                self.mask_source = np.load(LP1_MEMORY / "groups.npy", mmap_mode="r")
            else:
                source_root = LP1_UNSEEN
                source_keys = np.stack([np.load(source_root / f"{x}.npy", mmap_mode="r")
                                        for x in ("task", "episode", "frame")], axis=1)
                self.latent_source = np.load(source_root / "tokens.npy", mmap_mode="r")
                self.mask_source = np.load(source_root / "groups.npy", mmap_mode="r")
            lookup = _row_lookup(source_keys)
            mapped = np.asarray([lookup.get(tuple(map(int, k)), -1) for k in self.keys], np.int64)
            self.indices = np.flatnonzero(mapped >= 0)
            self.latent_ids = mapped[self.indices]
            self.latent_dim = int(self.latent_source.shape[-1])
            if self.latent_source.ndim != 3 or self.mask_source.shape[:2] != self.latent_source.shape[:2]:
                raise ValueError(f"bad VLM cache contract for {split}")
            self.feature_count = len(self.indices)
        elif modality == "vjepa":
            if vjepa_root is None:
                raise ValueError("V-JEPA condition requires --vjepa-root")
            from latent_probe_features import load_keyed_features
            z = load_keyed_features(vjepa_root, split, mmap_mode="r")
            lookup = _row_lookup(self.keys)
            mapped = np.asarray([lookup.get(tuple(map(int, k)), -1) for k in z["keys"]], np.int64)
            if np.any(mapped < 0):
                raise KeyError(f"V-JEPA {split} has keys outside LP1 row set")
            self.indices = mapped
            self.latent_ids = np.arange(len(mapped), dtype=np.int64)
            self.latent_source = (z["base"], z["wrist"])
            self.latent_dim = int(z["base"].shape[-1])
            if z["base"].shape[1:] != z["wrist"].shape[1:]:
                raise ValueError("base/wrist V-JEPA token shapes differ")
            self.feature_count = len(self.indices)
        elif modality == "none":
            self.latent_dim = 1
            self.feature_count = len(self.keys)
        else:
            raise ValueError(modality)
        self.lookup_local = {int(source): i for i, source in enumerate(self.indices)}
        self.feature_count = len(self.indices)

    def restrict(self, source_indices):
        wanted = np.asarray(source_indices, dtype=np.int64)
        keep = np.isin(self.indices, wanted)
        self.indices = self.indices[keep]
        if self.modality != "none":
            self.latent_ids = self.latent_ids[keep]
        self.lookup_local = {int(source): i for i, source in enumerate(self.indices)}

    def batch(self, local_ids, *, device="cpu", history_mode="intact", donor_local=None):
        local_ids = np.asarray(local_ids, dtype=np.int64)
        source_ids = self.indices[local_ids]
        d = self.data
        if self.modality == "none":
            latent = np.zeros((len(local_ids), 1, 1), np.float32)
            padding = None
        elif self.modality == "vlm":
            cache_ids = self.latent_ids[local_ids]
            latent = np.asarray(self.latent_source[cache_ids], np.float32)
            padding = np.asarray(self.mask_source[cache_ids] < 0, bool)
        else:
            cache_ids = self.latent_ids[local_ids]
            base, wrist = self.latent_source
            latent = np.concatenate([np.asarray(base[cache_ids], np.float32),
                                     np.asarray(wrist[cache_ids], np.float32)], axis=1)
            padding = None
        p = np.asarray(d["proprio"][source_ids], np.float32)
        if history_mode == "zero":
            p[:, 8:] = 0
        elif history_mode == "swap":
            if donor_local is None:
                raise ValueError("swap requires donor_local")
            p[:, 8:] = np.asarray(d["proprio"][self.indices[np.asarray(donor_local, np.int64)], 8:], np.float32)
        elif history_mode != "intact":
            raise ValueError(history_mode)
        batch = {
            "latent": torch.as_tensor(latent, dtype=torch.float32, device=device),
            "proprio": torch.as_tensor(p, dtype=torch.float32, device=device),
            "latent_padding_mask": torch.as_tensor(padding, dtype=torch.bool, device=device) if padding is not None else None,
            "source_ids": source_ids,
            "keys": self.keys[source_ids],
            "gripper_targets": torch.as_tensor(d["gripper_targets"][source_ids], dtype=torch.float32, device=device),
            "switch_target": torch.as_tensor(d["switch_target"][source_ids], dtype=torch.float32, device=device),
            "switch_step": torch.as_tensor(d["switch_step"][source_ids], dtype=torch.long, device=device),
            "log_rms6_target": torch.as_tensor(d["log_rms6_target"][source_ids], dtype=torch.float32, device=device),
            "rms6_target": torch.as_tensor(d["log_rms6_target"][source_ids], dtype=torch.float32, device=device),
            "rms6_physical_target": np.asarray(d["rms6_physical_target"][source_ids], np.float32),
            "actions_physical": np.asarray(d["actions_normalized"][source_ids], np.float32),
            "task": np.asarray(d["task"][source_ids], np.int64),
            "episode": np.asarray(d["episode"][source_ids], np.int64),
        }
        return batch


def _pool_for_readout():
    here = Path(__file__).resolve().parent
    if str(here) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(here))
    import latent_probe_gate as gate
    pool = gate._load_combined_pool()
    return pool, gate


def _candidate_arrays(rows: _Rows, pool, pool_keys, *, split, chunk=128):
    """Precompute per-query S-shared errors and same-task masks on CPU."""
    here = Path(__file__).resolve().parent
    if str(here) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(here))
    import scale_only_query as sq
    ids = rows.indices
    d = rows.data
    actions = np.asarray(d["actions_physical"][ids, :, :6], np.float32)
    errors = sq.scale_only_error(actions, np.asarray(pool["phys"], np.float32), "S-shared", chunk=chunk)
    pool_task = np.asarray(pool.get("source_task", pool.get("task")), np.int64)
    if split == "libero40_fit":
        allowed = np.asarray(d["task"][ids], np.int64)[:, None] != pool_task[None, :]
    else:
        allowed = np.ones(errors.shape, bool)
    if not allowed.any(axis=1).all():
        raise AssertionError("LP1 query has no allowed candidate")
    targets = soft_type_distribution(errors, allowed, tau=0.01).numpy().astype(np.float32)
    return errors, allowed, targets


def _make_batch(rows, local_ids, errors, allowed, type_targets, candidate_keys, *, device="cpu",
               history_mode="intact", donor_local=None):
    b = rows.batch(local_ids, device=device, history_mode=history_mode, donor_local=donor_local)
    src = b["source_ids"]
    b["candidate_keys"] = torch.as_tensor(candidate_keys, dtype=torch.float32, device=device)
    b["candidate_allowed"] = torch.as_tensor(allowed[local_ids], dtype=torch.bool, device=device)
    b["type_target"] = torch.as_tensor(type_targets[local_ids], dtype=torch.float32, device=device)
    return b


def _model_config(modality, use_prop, key_dim=60):
    dim = {"none": 1, "vjepa": 1024, "vlm": 2048}[modality]
    return ReadoutConfig(latent_dim=dim, key_dim=int(key_dim), use_proprio=use_prop, dropout=0.1)


def _dev_batchset(rows, errors, allowed, targets, candidate_keys, local_ids, *, batch_size, device):
    return [_make_batch(rows, local_ids[i:i + batch_size], errors, allowed, targets, candidate_keys,
                        device=device) for i in range(0, len(local_ids), batch_size)]


@torch.no_grad()
def _validate_heads(model, grip_batches, type_scale_batches, *, device):
    model.eval()
    grip_sum, grip_n = 0., 0
    grip_outputs = []
    for b in grip_batches:
        out = model(b["latent"], b.get("proprio"), b.get("candidate_keys"), b.get("latent_padding_mask"))
        n = len(b["latent"])
        gl = F.binary_cross_entropy_with_logits(out["gripper_logits"], b["gripper_targets"])
        sl = F.binary_cross_entropy_with_logits(out["switch_logit"], b["switch_target"])
        mask = b["switch_target"].bool()
        tl = F.cross_entropy(out["switch_time_logits"][mask], b["switch_step"][mask].clamp(0, 9)) if mask.any() else 0.
        grip_sum += (float(gl) + float(sl) + .25 * (float(tl) if torch.is_tensor(tl) else tl)) * n
        grip_n += n
        grip_outputs.append((out, b))
    type_pred, type_true, scale_pred, scale_true, variances = [], [], [], [], {}
    type_sum = scale_sum = 0.; type_n = scale_n = 0
    all_out = []
    for b in type_scale_batches:
        out = model(b["latent"], b.get("proprio"), b.get("candidate_keys"), b.get("latent_padding_mask"))
        logits = out["candidate_logits"].masked_fill(~b["candidate_allowed"], -1e9)
        pred = logits.argmax(-1).cpu().numpy()
        err = b["_errors"]
        type_sum += float(err[np.arange(len(pred)), pred].sum()); type_n += len(pred)
        scale_sum += float(torch.abs(out["log_rms6"] - b["log_rms6_target"]).sum().cpu()); scale_n += int(out["log_rms6"].numel())
        type_pred.extend(logits.cpu().numpy()); type_true.extend(err)
        scale_pred.extend(out["log_rms6"].cpu().numpy()); scale_true.extend(b["log_rms6_target"].cpu().numpy())
        all_out.append((out, b))
    # Use output distributions over the complete Dev sets for collapse auditing.
    def concat_out(pairs, name):
        return np.concatenate([p[0][name].detach().float().cpu().numpy() for p in pairs], axis=0)
    grip_raw = np.concatenate([p[0]["gripper_logits"].detach().float().cpu().numpy() for p in grip_outputs], axis=0)
    switch_raw = np.concatenate([p[0]["switch_logit"].detach().float().cpu().numpy() for p in grip_outputs], axis=0)[:, None]
    rms_raw = concat_out(all_out, "log_rms6") if all_out else np.zeros((0, 6))
    typ_raw = concat_out(all_out, "candidate_logits") if all_out else np.zeros((0, 1))
    variances = {
        "gripper": check_prediction_variance(grip_raw, label="gripper head"),
        "switch": check_prediction_variance(switch_raw, label="switch head"),
        "type": check_prediction_variance(typ_raw, label="type head"),
        "scale": check_prediction_variance(rms_raw, label="scale head"),
    }
    return {"grip": grip_sum / max(grip_n, 1), "type": type_sum / max(type_n, 1),
            "scale": scale_sum / max(scale_n, 1), "variance": variances,
            "type_outputs": type_pred, "type_errors": type_true,
            "scale_outputs": scale_pred, "scale_targets": scale_true}


def train_lp1_condition(condition, rows_train, train_errors, train_allowed, train_targets,
                        rows_grip_dev, grip_errors, grip_allowed, grip_targets,
                        rows_dev, dev_errors, dev_allowed, dev_targets, pool_keys, *,
                        output_dir, seed=4101, updates=20000, eval_every=500,
                        batch_size=16, device="cuda"):
    """Train shared fixed-loss model; checkpoint each head on its own predeclared Dev metric."""
    modality, prop_label = condition
    use_prop = prop_label == "P-on"
    set_seed(seed)
    model = LatentProbeReadout(_model_config(modality, use_prop, pool_keys.shape[-1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    # Dev source arrays are cached into small batches once; grip selection uses
    # the fixed four LIBERO-40 tasks, type/scale use only LIBERO-90 dense Dev.
    grip_local = np.arange(len(rows_grip_dev.indices))
    dev_local = np.arange(len(rows_dev.indices))
    grip_batches = _dev_batchset(rows_grip_dev, grip_errors, grip_allowed, grip_targets, pool_keys,
                                 grip_local, batch_size=64, device=device)
    dev_batches = _dev_batchset(rows_dev, dev_errors, dev_allowed, dev_targets, pool_keys,
                                dev_local, batch_size=64, device=device)
    for b, loc in zip(dev_batches, range(0, len(dev_local), 64)):
        b["_errors"] = torch.as_tensor(dev_errors[dev_local[loc:loc + 64]], device=device)
    best = {"grip": float("inf"), "type": float("inf"), "scale": float("inf")}
    states, best_step = {}, {}
    stale = {"grip": 0, "type": 0, "scale": 0}
    curve = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, int(updates) + 1):
        model.train()
        local = rng.integers(0, len(rows_train.indices), size=min(batch_size, len(rows_train.indices)))
        batch = _make_batch(rows_train, local, train_errors, train_allowed, train_targets, pool_keys, device=device)
        out = model(batch["latent"], batch.get("proprio"), batch["candidate_keys"], batch.get("latent_padding_mask"))
        loss, parts = multitask_loss(out, batch, grip_weight=1., switch_weight=1., type_weight=1., rms_weight=.1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss {condition} step {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % int(eval_every) and step != updates:
            continue
        model.eval()
        grip_metrics_now = _validate_grip(model, grip_batches)
        type_scale_now = _validate_type_scale(model, dev_batches)
        # Collapse checks are component-specific and performed on their actual
        # selection split, not on the optimization loss.
        var = {
            "gripper": check_prediction_variance(grip_metrics_now["grip_logits"], label="gripper head"),
            "switch": check_prediction_variance(grip_metrics_now["switch_logits"][:, None], label="switch head"),
            "type": check_prediction_variance(type_scale_now["type_logits"], label="type head"),
            "scale": check_prediction_variance(type_scale_now["scale"], label="scale head"),
        }
        current = {"grip": grip_metrics_now["loss"], "type": type_scale_now["type_error"],
                   "scale": type_scale_now["scale_mae"]}
        for head, value in current.items():
            if value < best[head]:
                best[head] = float(value); best_step[head] = step
                stale[head] = 0
                states[head] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model": states[head], "config": vars(model.config), "condition": condition,
                            "seed": seed, "head_selection": head, "step": step, "metric": value},
                           out_dir / f"best_{head}.pt")
            else:
                stale[head] += 1
        curve.append({"step": step, "train_total": float(loss.detach().cpu()), **{f"train_{k}": v for k, v in parts.items()},
                      "dev_selection": current, "dev_variance": var})
        (out_dir / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        if all(v >= 8 for v in stale.values()):
            break
    return {"condition": list(condition), "seed": int(seed), "updates": int(updates),
            "best_step": best_step, "best_metric": best, "curve": curve}


@torch.no_grad()
def _validate_grip(model, batches):
    model.eval(); loss_sum = 0.; n = 0; grips = []; switches = []
    for b in batches:
        o = model(b["latent"], b["proprio"], b["candidate_keys"], b["latent_padding_mask"])
        n += len(b["latent"])
        loss_sum += (F.binary_cross_entropy_with_logits(o["gripper_logits"], b["gripper_targets"], reduction="sum") +
                     F.binary_cross_entropy_with_logits(o["switch_logit"], b["switch_target"], reduction="sum"))
        mask = b["switch_target"].bool()
        if mask.any(): loss_sum += .25 * F.cross_entropy(o["switch_time_logits"][mask], b["switch_step"][mask].clamp(0, 9), reduction="sum")
        grips.append(o["gripper_logits"].cpu().numpy()); switches.append(o["switch_logit"].cpu().numpy()[:, None])
    return {"loss": float(loss_sum.cpu()) / max(n,1), "grip_logits": np.concatenate(grips),
            "switch_logits": np.concatenate(switches)}


@torch.no_grad()
def _validate_type_scale(model, batches):
    model.eval(); type_sum = scale_sum = 0.; nt = ns = 0; logits_all=[]; rms_all=[]
    for b in batches:
        o = model(b["latent"], b["proprio"], b["candidate_keys"], b["latent_padding_mask"])
        logits = o["candidate_logits"].masked_fill(~b["candidate_allowed"], -1e9)
        e = b["_errors"].detach().cpu().numpy(); pred = logits.argmax(-1).cpu().numpy()
        type_sum += float(e[np.arange(len(pred)), pred].sum()); nt += len(pred)
        scale_sum += float(torch.abs(o["log_rms6"] - b["log_rms6_target"]).sum().cpu()); ns += int(o["log_rms6"].numel())
        logits_all.append(logits.cpu().numpy()); rms_all.append(o["log_rms6"].cpu().numpy())
    return {"type_error": type_sum / max(nt,1), "scale_mae": scale_sum / max(ns,1),
            "type_logits": np.concatenate(logits_all), "scale": np.concatenate(rms_all)}


def _select_rows(rows, *, tasks=None, limit=None, seed=4101):
    base = rows.indices.copy()
    if tasks is not None:
        base = base[np.isin(rows.data["task"][base], np.asarray(tasks, np.int64))]
    if limit is not None and len(base) > limit:
        rng = np.random.default_rng(seed)
        # Equal-ish sampling across tasks keeps the smoke from silently
        # dropping the fixed held-out grip tasks.
        task_ids = np.unique(rows.data["task"][base]); chosen=[]
        per = max(1, int(limit) // len(task_ids))
        for t in task_ids:
            tix = base[np.asarray(rows.data["task"][base]) == t]
            chosen.extend(rng.choice(tix, min(per, len(tix)), replace=False).tolist())
        if len(chosen) < limit:
            remain = np.setdiff1d(base, np.asarray(chosen, np.int64))
            chosen.extend(rng.choice(remain, min(limit-len(chosen), len(remain)), replace=False).tolist())
        base = np.asarray(chosen[:limit], np.int64)
    return base


def _make_component_dataset(data_root, split, modality, vjepa_root, pool, *, tasks=None, limit=None, seed=4101):
    rows = _Rows(split, data_root, modality, vjepa_root)
    source = _select_rows(rows, tasks=tasks, limit=limit, seed=seed)
    rows.restrict(source)
    errors, allowed, targets = _candidate_arrays(rows, pool, None, split=split)
    return rows, errors, allowed, targets


def _donor_local(rows):
    keys = rows.keys[rows.indices]
    groups = {}
    for i, (t,e,_) in enumerate(keys): groups.setdefault((int(t),int(e)), []).append(i)
    labels = list(groups)
    if len(labels) < 2: raise ValueError("need at least two episodes for swap corruption")
    rng = np.random.default_rng(4101)
    donor = np.empty(len(keys), np.int64)
    for i, key in enumerate(keys):
        eligible = [j for j, other in enumerate(labels) if other != (int(key[0]), int(key[1]))]
        group = groups[labels[int(rng.choice(eligible))]]
        donor[i] = int(rng.choice(group))
    return donor


def _evaluate_component(model_path, rows, errors, allowed, targets, pool_keys, *, device="cuda",
                        history_mode="intact", donor=None, batch_size=64):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ReadoutConfig(**payload["config"])
    model = LatentProbeReadout(cfg); model.load_state_dict(payload["model"]); model.to(device).eval()
    all_outputs = {k: [] for k in ("gripper_logits", "switch_logit", "switch_time_logits", "log_rms6", "candidate_logits")}
    targets_grip=[]; switch=[]; step=[]; log_rms=[]; rms_phys=[]; episodes=[]; keys=[]; e_parts=[]; allowed_parts=[]
    donor_local = donor
    for start in range(0, len(rows.indices), batch_size):
        local = np.arange(start, min(start+batch_size,len(rows.indices)))
        dloc = donor_local[local] if donor_local is not None else None
        b = rows.batch(local, device=device, history_mode=history_mode, donor_local=dloc)
        ck = torch.as_tensor(pool_keys, dtype=torch.float32, device=device)
        out = model(b["latent"], b["proprio"], ck, b["latent_padding_mask"])
        for k in all_outputs: all_outputs[k].append(out[k].detach().float().cpu().numpy())
        sid=b["source_ids"]; y=rows.data
        targets_grip.append(y["gripper_targets"][sid]); switch.append(y["switch_target"][sid]); step.append(y["switch_step"][sid])
        log_rms.append(y["log_rms6_target"][sid]); rms_phys.append(y["rms6_physical_target"][sid])
        episodes.append(y["episode"][sid]); keys.append(rows.keys[sid])
        e_parts.append(errors[local]); allowed_parts.append(allowed[local])
    result={k:np.concatenate(v) for k,v in all_outputs.items()}
    result.update({"gripper_targets":np.concatenate(targets_grip),"switch_target":np.concatenate(switch),
                   "switch_step":np.concatenate(step),"log_rms6_target":np.concatenate(log_rms),
                   "rms6_physical_target":np.concatenate(rms_phys),
                   "episode":np.concatenate(episodes),"keys":np.concatenate(keys),
                   "type_errors":np.concatenate(e_parts),"candidate_allowed":np.concatenate(allowed_parts)})
    result["candidate_logits"] = np.where(result["candidate_allowed"], result["candidate_logits"], -1e9)
    return result


def _metric_summary(pred, *, pool, baseline=None):
    grip = gripper_metrics(pred["gripper_logits"], pred["switch_logit"], pred["gripper_targets"],
                           pred["switch_target"], switch_step=pred["switch_step"],
                           switch_time_logits=pred["switch_time_logits"])
    type_m = retrieval_metrics(pred["candidate_logits"], pred["type_errors"], pred["candidate_allowed"], tau=.01)
    top = pred["candidate_logits"].argmax(-1)
    top_err = pred["type_errors"][np.arange(len(top)), top]
    oracle = np.where(pred["candidate_allowed"], pred["type_errors"], np.inf).min(1)
    random = np.where(pred["candidate_allowed"], pred["type_errors"], 0).sum(1) / pred["candidate_allowed"].sum(1)
    scale = pred["log_rms6"]
    scale_mae = np.abs(scale - pred["log_rms6_target"])
    return {"n":len(top),"gripper":grip,"type":{"top1_mean":float(top_err.mean()),"top1_median":float(np.median(top_err)),
            "oracle_mean":float(oracle.mean()),"random_mean":float(random.mean()),
            "recovery":float(np.mean((random-top_err)/np.maximum(random-oracle,1e-9))),**type_m},
            "scale":{"log_mae_mean":float(scale_mae.mean()),"log_mae_median":float(np.median(scale_mae))}}


def _evaluate_heads(model_paths, rows, errors, allowed, targets, pool_keys, *, device="cuda",
                    history_mode="intact", donor_local=None, batch_size=64):
    """Evaluate each independently selected head checkpoint on the same rows."""
    per_head = {}
    for head in ("grip", "type", "scale"):
        raw = _evaluate_component(model_paths[head], rows, errors, allowed, targets, pool_keys,
                                  device=device, history_mode=history_mode, donor=donor_local,
                                  batch_size=batch_size)
        per_head[head] = raw
    return {"gripper_logits": per_head["grip"]["gripper_logits"],
            "switch_logit": per_head["grip"]["switch_logit"],
            "switch_time_logits": per_head["grip"]["switch_time_logits"],
            "log_rms6": per_head["scale"]["log_rms6"],
            "candidate_logits": per_head["type"]["candidate_logits"],
            **{k: per_head["grip"][k] for k in ("gripper_targets", "switch_target", "switch_step", "episode", "keys")},
            "log_rms6_target": per_head["scale"]["log_rms6_target"],
            "rms6_physical_target": per_head["scale"]["rms6_physical_target"],
            "type_errors": per_head["type"]["type_errors"],
            "candidate_allowed": per_head["type"]["candidate_allowed"]}


def _eval_reference_arrays(rows):
    """Attach cached π0.5, hold-last and P5/R5 ranks by exact window key."""
    import latent_probe_gate as gate
    inp = dict(np.load(LP1_SCALE_PREP / "W_inputs_eval.npz"))
    source_keys = np.stack([inp["task"], inp["episode"], inp["frame"]], axis=1)
    lookup = _row_lookup(source_keys)
    rows.keys = np.asarray(rows.keys, np.int32)
    ix = np.asarray([lookup.get(tuple(map(int, k)), -1) for k in rows.keys[rows.indices]], np.int64)
    if np.any(ix < 0): raise KeyError("Eval LP1 windows do not align with cached π0.5 reference rows")
    refs = {"pi05": np.asarray(inp["pi05"][ix], np.float32), "hist": np.asarray(inp["hist"][ix], np.float32),
            "state": np.asarray(inp["state"][ix], np.float32), "hard": np.asarray(inp["hard"][ix], bool),
            "source_rows": ix}
    gate_summary = json.loads(LP1_GATE_SUMMARY.read_text())
    refs["picks_p5"] = np.asarray(gate_summary["picks"]["eval"]["T-P5"], np.int64)[ix]
    refs["picks_r5"] = np.asarray(gate_summary["picks"]["eval"]["T-R5"], np.int64)[ix]
    refs["picks_oracle"] = np.asarray(gate_summary["picks"]["eval"]["T-oracle"], np.int64)[ix]
    return refs


def _reference_metrics(pred, refs, *, pool):
    """Model-independent P5/R5/Pi05/state/scale baselines on aligned rows."""
    from latent_probe_gate import hold_last_actions
    type_errors = pred["type_errors"]
    allowed = pred["candidate_allowed"]
    out = {}
    for name, picks in (("P5", refs["picks_p5"]), ("R5", refs["picks_r5"]), ("oracle_type", refs["picks_oracle"])):
        valid = picks < type_errors.shape[1]
        chosen = np.full(len(picks), np.nan, np.float32)
        chosen[valid] = type_errors[np.flatnonzero(valid), picks[valid]]
        out[name] = {"top1_type_error_mean": float(np.nanmean(chosen)), "n": int(np.isfinite(chosen).sum())}
    out["random_type"] = {"mean": float((np.where(allowed, type_errors, 0).sum(1) / allowed.sum(1)).mean())}
    h = np.asarray(refs["hist"], np.float32)
    pi05 = np.asarray(refs["pi05"], np.float32)
    scale_target = pred["log_rms6_target"]
    def log_rms(x): return np.log(np.maximum(np.sqrt(np.mean(np.asarray(x)[..., :6]**2, axis=1)), 1e-6))
    out["scale"] = {"pi05_log_mae": float(np.abs(log_rms(pi05)-scale_target).mean()),
                     "history_log_mae": float(np.abs(log_rms(h)-scale_target).mean())}
    p5 = np.clip(np.asarray(refs["picks_p5"], np.int64), 0, len(pool["phys"])-1)
    candidate_rms = np.sqrt(np.mean(np.asarray(pool["phys"])[p5, :, :6] ** 2, axis=1))
    physical_target = np.maximum(pred["rms6_physical_target"], 1e-6)
    out["scale"]["original_P5_candidate_physical_log_mae"] = float(
        np.abs(np.log(np.maximum(candidate_rms,1e-6))-np.log(physical_target)).mean())
    hold = hold_last_actions({"hist": h, "state": refs["state"], "frame": pred["keys"][:,2],
                              "has_hist": pred["keys"][:,2] > 0})
    pi_gl = np.where(pi05[...,6] >= 0, 8., -8.)
    hold_gl = np.where(hold[...,6] >= 0, 8., -8.)
    pi_switch = (np.any(np.diff(pi_gl,axis=1) != 0,axis=1)).astype(np.float32)
    out["gripper_pi05"] = gripper_metrics(pi_gl, np.where(pi_switch>0,8.,-8.),
                 pred["gripper_targets"],pred["switch_target"],switch_step=pred["switch_step"])
    hold_switch = (np.any(hold[:,1:,6] != hold[:,:-1,6],axis=1)).astype(np.float32)
    out["gripper_hold"] = gripper_metrics(hold_gl, np.where(hold_switch>0,8.,-8.),
                 pred["gripper_targets"],pred["switch_target"],switch_step=pred["switch_step"])
    return out


def _constant_baseline(train_data, train_tasks, pred):
    """Training-only constant head baseline (the L-none×P-off limit)."""
    ix=np.flatnonzero(np.isin(np.asarray(train_data["task"],np.int64),np.asarray(train_tasks,np.int64)))
    if len(ix)<2: raise ValueError("constant baseline requires training rows")
    gt=np.asarray(train_data["gripper_targets"][ix],np.float32)
    sw=np.asarray(train_data["switch_target"][ix],np.float32)
    mean_g=np.clip(gt.mean(0),1e-5,1-1e-5)
    mean_s=float(np.clip(sw.mean(),1e-5,1-1e-5))
    logit=lambda p: np.log(p/(1-p))
    g_logits=np.broadcast_to(logit(mean_g)[None,:],(len(pred["gripper_targets"]),10))
    s_logits=np.full(len(g_logits),logit(mean_s),np.float32)
    ts=np.asarray(train_data["switch_step"][ix]); pos=ts[ts>=0]
    time_prob=np.bincount(pos.clip(0,9),minlength=10).astype(np.float32)+1e-5
    time_logits=np.broadcast_to(np.log(time_prob/time_prob.sum())[None,:],(len(g_logits),10))
    grip=gripper_metrics(g_logits,s_logits,pred["gripper_targets"],pred["switch_target"],
                         switch_step=pred["switch_step"],switch_time_logits=time_logits)
    scale_mean=np.asarray(train_data["log_rms6_target"][ix],np.float32).mean(0)
    scale_mae=np.abs(pred["log_rms6_target"]-scale_mean).mean()
    random=np.where(pred["candidate_allowed"],pred["type_errors"],0).sum(1)/pred["candidate_allowed"].sum(1)
    uniform_score=np.zeros_like(pred["candidate_logits"])
    tie_metrics=retrieval_metrics(uniform_score,pred["type_errors"],pred["candidate_allowed"],tau=.01)
    return {"training_rows":int(len(ix)),"gripper":grip,
            "type":{"uniform_expected_error":float(random.mean()),"uniform_score_spearman":0.0,
                    "fixed_tie_break_top1_error":float(pred["type_errors"][np.arange(len(random)),
                         pred["candidate_allowed"].argmax(1)].mean()),
                    "n_queries":tie_metrics["n_queries"]},
            "scale":{"log_mae_mean":float(scale_mae)}}


def evaluate_condition(output_dir, modality, prop_label, eval_rows, eval_errors, eval_allowed,
                       eval_targets, pool_keys, pool, *, train_data=None, train_tasks=None,
                       device="cuda", batch_size=64):
    model_paths = {head: Path(output_dir) / f"best_{head}.pt" for head in ("grip", "type", "scale")}
    modes = ("intact", "zero", "swap") if prop_label == "P-on" else ("intact",)
    results = {}
    donor = _donor_local(eval_rows) if "swap" in modes else None
    refs = _eval_reference_arrays(eval_rows)
    for mode in modes:
        pred = _evaluate_heads(model_paths, eval_rows, eval_errors, eval_allowed, eval_targets, pool_keys,
                               device=device, history_mode=mode, donor_local=donor, batch_size=batch_size)
        results[mode] = _metric_summary(pred, pool=pool)
        results[mode]["hard"] = _metric_summary(_slice_prediction(pred, refs["hard"]), pool=pool)
        if mode == "intact":
            results[mode]["baselines"] = _reference_metrics(pred, refs, pool=pool)
            if train_data is not None and train_tasks is not None:
                results[mode]["constant_Lnone_Poff"] = _constant_baseline(train_data,train_tasks,pred)
            results[mode]["n_common_l_vlm_rows"] = int(np.sum(refs["source_rows"] >= 0))
    return results


def _slice_prediction(pred, mask):
    n = len(mask)
    return {k: (v[np.asarray(mask)] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
            for k,v in pred.items()}


def _common_eval_sources(data_root, vjepa_root, *, limit=None):
    rows = {m: _Rows("libero90_eval", data_root, m, vjepa_root) for m in ("none","vjepa","vlm")}
    common = set(map(int, rows["none"].indices))
    for r in rows.values(): common.intersection_update(map(int, r.indices))
    common = np.asarray(sorted(common), np.int64)
    if limit is not None and len(common) > limit:
        rng = np.random.default_rng(4101)
        # Keep the smoke set deterministic and broadly spread across the fixed Eval rows.
        common = np.sort(rng.choice(common, int(limit), replace=False))
    return rows, common


def _build_eval_dataset(modality, rows, common, pool):
    rows.restrict(common)
    errors, allowed, targets = _candidate_arrays(rows, pool, None, split="eval")
    return rows, errors, allowed, targets


def _write_vlm_resolution_origin(path):
    payload={"training_cache":{"path":str(LP1_MEMORY/"tokens.npy"),"shape":[26581,712,2048],
              "source_images":"run-01 resized by source_snapshot/run.py ResizeImages(224,224); LIBERO-40 stored source is 256x256"},
             "dev_eval_cache":{"path":str(LP1_UNSEEN/"tokens.npy"),"shape":[732,712,2048],
              "source_images":"unseen-rows-02 prepared from worker.py 128x128 frames then ResizeImages(224,224)"},
             "recreated":False,"conclusion":"cached VLM tensors are both 224x224 model inputs, but their source paths differ (256->224 vs 128->224); this source-resolution shift is recorded and the VLM cache is not regenerated."}
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def run_smoke(*, data_root=LP1_ROOT/"data", vjepa_root=None, output_dir=None,
              updates=200, batch_size=8, memory_fraction=.12):
    """One-seed 200-update GPU smoke across the five requested conditions."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("LP1 smoke requires CUDA")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    pool, _ = _pool_for_readout()
    here=Path(__file__).resolve().parent
    import scale_only_query as sq
    pool_keys=sq.key_inputs("W",pool,"S-shared")
    fit_probe=load_supervision_rows(data_root,"libero40_fit")
    split=gripper_dev_task_split(fit_probe["task"])
    if vjepa_root is None: raise ValueError("smoke needs 128-resolution V-JEPA features")
    cap=768
    reports=[]
    for modality, prop in (("none","P-on"),("vjepa","P-off"),("vjepa","P-on"),("vlm","P-off"),("vlm","P-on")):
        c=(f"L-{modality}",prop)
        tr,err,allow,targ=_make_component_dataset(data_root,"libero40_fit",modality,vjepa_root,pool,
             tasks=split["train_tasks"],limit=cap,seed=4101)
        gd,gerr,gallow,gtarg=_make_component_dataset(data_root,"libero40_fit",modality,vjepa_root,pool,
             tasks=split["gripper_dev_tasks"],limit=256,seed=4102)
        dv,derr,dallow,dtarg=_make_component_dataset(data_root,"libero90_dev",modality,vjepa_root,pool,limit=128,seed=4103)
        # Eval rows are not used for model or checkpoint selection in the smoke.
        od=Path(output_dir)/f"{modality}-{prop}" if output_dir else None
        if od.exists(): raise FileExistsError(od)
        r=train_lp1_condition((modality,prop),tr,err,allow,targ,gd,gerr,gallow,gtarg,dv,derr,dallow,dtarg,
              pool_keys,output_dir=od,seed=4101,updates=updates,eval_every=50,batch_size=batch_size,device="cuda")
        reports.append({"condition":list(c),"train_rows":len(tr.indices),"gripper_dev_rows":len(gd.indices),
                        "type_scale_dev_rows":len(dv.indices),"feature_rows_fit":tr.feature_count,
                        "best_step":r["best_step"],"best_metric":r["best_metric"],
                        "first_train_loss":r["curve"][0]["train_total"],"last_train_loss":r["curve"][-1]["train_total"]})
        if not r["curve"][-1]["train_total"] < r["curve"][0]["train_total"]:
            raise AssertionError(f"smoke loss did not decline for {c}")
    # One report-only Eval pass after every checkpoint was selected on Dev.
    eval_sources, common = _common_eval_sources(data_root, vjepa_root, limit=64)
    eval_records=[]
    for modality, prop in (("none","P-on"),("vjepa","P-off"),("vjepa","P-on"),("vlm","P-off"),("vlm","P-on")):
        ds, ee, ea, et = _build_eval_dataset(modality, eval_sources[modality], common, pool)
        out_dir=Path(output_dir)/f"{modality}-{prop}"
        result=evaluate_condition(out_dir,modality,prop,ds,ee,ea,et,pool_keys,pool,
                                  train_data=fit_probe,train_tasks=split["train_tasks"],
                                  device="cuda",batch_size=32)
        eval_records.append({"condition":[f"L-{modality}",prop],"n_common_rows":len(common),"metrics":result})
    summary={"mode":"smoke","conditions":reports,"eval_report_only":eval_records,
             "eval_common_row_ids":common.tolist(),
             "gripper_dev_split":split,
             "updates_per_seed":int(updates),"seeds":[4101],"eval_used_for_selection":False,
             "memory_fraction":float(memory_fraction),"device":torch.cuda.get_device_name()}
    return summary


def run_production(*, data_root=LP1_ROOT/"data", vjepa_root=None, output_dir=None,
                   seeds=(4101,4102,4103), updates=20000, batch_size=16,
                   eval_every=500, memory_fraction=.2):
    """Full LP1 run entrypoint, deliberately separate from the smoke default."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("LP1 production requires CUDA")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    pool,_=_pool_for_readout()
    here=Path(__file__).resolve().parent
    import scale_only_query as sq
    pool_keys=sq.key_inputs("W",pool,"S-shared")
    fit=load_supervision_rows(data_root,"libero40_fit")
    split=gripper_dev_task_split(fit["task"])
    root=Path(output_dir)
    if root.exists(): raise FileExistsError(f"refusing to reuse production output: {root}")
    root.mkdir(parents=True)
    all_runs=[]
    for modality,prop in (("none","P-on"),("vjepa","P-off"),("vjepa","P-on"),
                          ("vlm","P-off"),("vlm","P-on")):
        condition=f"L-{modality}_{prop}"
        tr,te,ta,tt=_make_component_dataset(data_root,"libero40_fit",modality,vjepa_root,pool,
                                            tasks=split["train_tasks"],seed=4101)
        gd,ge,ga,gt=_make_component_dataset(data_root,"libero40_fit",modality,vjepa_root,pool,
                                             tasks=split["gripper_dev_tasks"],seed=4101)
        dv,de,da,dt=_make_component_dataset(data_root,"libero90_dev",modality,vjepa_root,pool,seed=4101)
        for seed in map(int,seeds):
            od=root/condition/f"seed-{seed}"
            result=train_lp1_condition((modality,prop),tr,te,ta,tt,gd,ge,ga,gt,dv,de,da,dt,pool_keys,
                    output_dir=od,seed=seed,updates=updates,eval_every=eval_every,batch_size=batch_size,device="cuda")
            all_runs.append({"condition":condition,"modality":modality,"proprio":prop,"seed":seed,
                "train_rows":len(tr.indices),"gripper_dev_rows":len(gd.indices),"type_scale_dev_rows":len(dv.indices),
                "best_step":result["best_step"],"best_metric":result["best_metric"],
                "last_dev_variance":result["curve"][-1]["dev_variance"]})
    # Exact common row set is the intersection of all required feature sources.
    eval_sources,common=_common_eval_sources(data_root,vjepa_root,limit=None)
    eval_reports=[]
    for modality,prop in (("none","P-on"),("vjepa","P-off"),("vjepa","P-on"),
                          ("vlm","P-off"),("vlm","P-on")):
        ds,ee,ea,et=_build_eval_dataset(modality,eval_sources[modality],common,pool)
        for seed in map(int,seeds):
            od=root/f"L-{modality}_{prop}"/f"seed-{seed}"
            mpaths={h:od/f"best_{h}.pt" for h in ("grip","type","scale")}
            modes=("intact","zero","swap") if prop=="P-on" else ("intact",)
            donor=_donor_local(ds) if "swap" in modes else None
            refs=_eval_reference_arrays(ds)
            for mode in modes:
                pred=_evaluate_heads(mpaths,ds,ee,ea,et,pool_keys,device="cuda",history_mode=mode,
                                     donor_local=donor,batch_size=64)
                result={"condition":f"L-{modality}_{prop}","seed":seed,"history":mode,
                        "all":_metric_summary(pred,pool=pool),
                        "hard":_metric_summary(_slice_prediction(pred,refs["hard"]),pool=pool)}
                if mode=="intact":
                    result["baselines"]=_reference_metrics(pred,refs,pool=pool)
                    result["constant_Lnone_Poff"]=_constant_baseline(fit,split["train_tasks"],pred)
                    result["n_common_l_vlm_rows"]=len(common)
                eval_reports.append(result)
    summary={"mode":"production","conditions":all_runs,"eval":eval_reports,
             "gripper_dev_split":split,"seeds":list(map(int,seeds)),"updates_max":int(updates),
             "eval_rows_common_all_conditions":int(len(common)),"eval_used_for_selection":False,
             "vlm_expected_fit_matches":15678,"vlm_expected_dev_matches":113,"vlm_expected_eval_matches":619,
             "device":torch.cuda.get_device_name(),"memory_fraction":float(memory_fraction)}
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# LP1 latent readout results","","Selection is head-wise: grip on held-out LIBERO-40 tasks, type/scale on dense LIBERO-90 Dev. Eval labels were only used for final reporting.","",
           "## Training / selection","","| Condition | Seed | Rows train/grip-Dev/type-Dev | Best steps (grip/type/scale) | Best Dev metrics |","|---|---:|---:|---|---|"]
    for r in all_runs: lines.append(f"| {r['condition']} | {r['seed']} | {r['train_rows']}/{r['gripper_dev_rows']}/{r['type_scale_dev_rows']} | {r['best_step']} | {r['best_metric']} |")
    lines += ["","## Eval","","| Condition | Seed | History | Grip AUC / F1 / timing | Type top-1 error / recovery / Spearman | Scale log MAE |","|---|---:|---|---|---|---|"]
    for r in eval_reports:
        m=r["all"]; g=m["gripper"]; t=m["type"]; s=m["scale"]
        lines.append(f"| {r['condition']} | {r['seed']} | {r['history']} | {g['switch_auc']:.4f}/{g['f1']:.4f}/{g['switch_time_abs_error_mean']:.3f} | {t['top1_mean']:.6f}/{t['recovery']:.4f}/{t['spearman_score_vs_negative_type_error_mean']:.4f} | {s['log_mae_mean']:.5f} |")
    lines += ["","## Hard subset Eval","","| Condition | Seed | History | n | Grip AUC/F1 | Type top-1 / recovery | Scale log-MAE |","|---|---:|---|---:|---|---|---:|"]
    for r in eval_reports:
        m=r["hard"]; g=m["gripper"]; t=m["type"]; s=m["scale"]
        lines.append(f"| {r['condition']} | {r['seed']} | {r['history']} | {m['n']} | {g['switch_auc']:.4f}/{g['f1']:.4f} | {t['top1_mean']:.6f}/{t['recovery']:.4f} | {s['log_mae_mean']:.5f} |")
    first_intact=next(r for r in eval_reports if r["history"]=="intact")
    bas=first_intact["baselines"]
    lines += ["","## Fixed reference rows (same Eval windows)","",
              "| Reference | Value |","|---|---:|"]
    lines.append(f"| P5 top-1 type error | {bas['P5']['top1_type_error_mean']:.6f} |")
    lines.append(f"| R5 top-1 type error | {bas['R5']['top1_type_error_mean']:.6f} |")
    lines.append(f"| Oracle type top-1 error (diagnostic) | {bas['oracle_type']['top1_type_error_mean']:.6f} |")
    lines.append(f"| Random type mean error | {bas['random_type']['mean']:.6f} |")
    lines.append(f"| π0.5 / previous-history / original P5 candidate scale log-MAE | {bas['scale']['pi05_log_mae']:.5f} / {bas['scale']['history_log_mae']:.5f} / {bas['scale']['original_P5_candidate_physical_log_mae']:.5f} |")
    lines.append(f"| π0.5 gripper AUC / F1 | {bas['gripper_pi05']['switch_auc']:.4f} / {bas['gripper_pi05']['f1']:.4f} |")
    lines.append(f"| State-hold gripper AUC / F1 | {bas['gripper_hold']['switch_auc']:.4f} / {bas['gripper_hold']['f1']:.4f} |")
    lines.append(f"| L-none×P-off constant head grip AUC/F1, type expected-random error, scale log-MAE | {bas['constant_Lnone_Poff']['gripper']['switch_auc']:.4f}/{bas['constant_Lnone_Poff']['gripper']['f1']:.4f}, {bas['constant_Lnone_Poff']['type']['uniform_expected_error']:.6f}, {bas['constant_Lnone_Poff']['scale']['log_mae_mean']:.5f} |")
    lines += ["",f"Common Eval rows across L-vlm and all other required conditions: {len(common)}.",
              f"Exact VLM matches: LIBERO-40 fit {summary['vlm_expected_fit_matches']}, LIBERO-90 Dev {summary['vlm_expected_dev_matches']}, Eval {summary['vlm_expected_eval_matches']}.",
              "P-on includes intact, zero-history, and other-episode-swapped histories. Detailed per-seed values and output-variance checks are in summary.json and each condition's curve.json.",""]
    (root/"report.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true",help="run 1 seed x 200 updates for each condition")
    parser.add_argument("--data-root",type=Path,default=LP1_ROOT/"data")
    parser.add_argument("--vjepa-root",type=Path,default=None)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--updates",type=int,default=20000)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--memory-fraction",type=float,default=.2)
    parser.add_argument("--seeds",type=int,nargs="+",default=[4101,4102,4103])
    parser.add_argument("--eval-every",type=int,default=500)
    args=parser.parse_args(argv)
    if args.smoke:
        out=args.output_dir or LP1_ROOT/"repair-01/smoke"
        out.mkdir(parents=True,exist_ok=False)
        result=run_smoke(data_root=args.data_root,vjepa_root=args.vjepa_root,output_dir=out,
                         updates=min(args.updates,200),batch_size=min(args.batch_size,8),memory_fraction=args.memory_fraction)
        train_map={tuple(x["condition"]):x for x in result["conditions"]}
        lines=["# LP1 repair smoke","",
               "1 seed × 200 updates per requested condition. All three heads use fixed loss weights; grip checkpoints are selected on LIBERO-40 task holdout and type/scale on LIBERO-90 dense Dev. Eval is report-only.","",
               "## Train / Dev","","| Condition | Rows (train / grip Dev / type-scale Dev) | Steps grip/type/scale | Train loss first→last | Dev selection grip/type/scale |","|---|---:|---|---:|---|"]
        for r in result["conditions"]:
            lines.append(f"| {r['condition'][0]} {r['condition'][1]} | {r['train_rows']}/{r['gripper_dev_rows']}/{r['type_scale_dev_rows']} | {r['best_step']['grip']}/{r['best_step']['type']}/{r['best_step']['scale']} | {r['first_train_loss']:.4f}→{r['last_train_loss']:.4f} | {r['best_metric']} |")
        lines += ["","## Report-only Eval (same rows for all conditions)","",
                  "| Condition | n | Grip AUC/F1/time error | Type top-1 / recovery / Spearman | Scale log-MAE | zero-history type error | swapped-history type error |","|---|---:|---|---|---:|---:|---:|"]
        for r in result["eval_report_only"]:
            m=r["metrics"]; intact=m["intact"];g=intact["gripper"];t=intact["type"];s=intact["scale"]
            zero=m.get("zero",{}).get("type",{}).get("top1_mean",float("nan"));swap=m.get("swap",{}).get("type",{}).get("top1_mean",float("nan"))
            lines.append(f"| {r['condition'][0]} {r['condition'][1]} | {r['n_common_rows']} | {g['switch_auc']:.4f}/{g['f1']:.4f}/{g['switch_time_abs_error_mean']:.3f} | {t['top1_mean']:.5f}/{t['recovery']:.4f}/{t['spearman_score_vs_negative_type_error_mean']:.4f} | {s['log_mae_mean']:.4f} | {zero:.5f} | {swap:.5f} |")
        intact0=result["eval_report_only"][0]["metrics"]["intact"]
        base=intact0["baselines"]
        constant=intact0["constant_Lnone_Poff"]
        lines += ["","Fixed references on the same smoke Eval windows:",
                  f"- L-none×P-off constant-head baseline (fit-only label means): grip AUC/F1 {constant['gripper']['switch_auc']:.4f}/{constant['gripper']['f1']:.4f}; type expected-random error {constant['type']['uniform_expected_error']:.6f}; scale log-MAE {constant['scale']['log_mae_mean']:.5f}.",
                  f"- Grip: π0.5 AUC/F1 {base['gripper_pi05']['switch_auc']:.4f}/{base['gripper_pi05']['f1']:.4f}; hold-last {base['gripper_hold']['switch_auc']:.4f}/{base['gripper_hold']['f1']:.4f}.",
                  f"- Type top-1 S-shared error: P5 {base['P5']['top1_type_error_mean']:.6f}; R5 {base['R5']['top1_type_error_mean']:.6f}; oracle {base['oracle_type']['top1_type_error_mean']:.6f}; random {base['random_type']['mean']:.6f}.",
                  f"- Scale log-MAE: π0.5 {base['scale']['pi05_log_mae']:.5f}; previous actions {base['scale']['history_log_mae']:.5f}; original P5 candidate (physical units) {base['scale']['original_P5_candidate_physical_log_mae']:.5f}.",""]
        lines += ["","## Input audit","",
                  "Each head has an independent Dev-selected best checkpoint; `curve.json` records the per-head validation metric and output variance at every check. A collapsed output raises immediately.",
                  "V-JEPA uses a fresh 128×128 preprocessing path for all splits; 2×2 pooled tokens (64/camera) were saved to keep the cache modest. Resolution statistics are in `resolution_audit_128.json`.",
                  "The cached VLM path was not rebuilt. Its 40-side images go through the run-01 ResizeImages(224,224) path from source 256×256; unseen rows first use worker.py 128×128 and then ResizeImages(224,224). See `vlm_resolution_origin.json`.",
                  f"Fresh V-JEPA feature root: `{args.vjepa_root}`. Smoke row counts per split: fit {len(np.load(args.vjepa_root/'libero40_fit/row_keys.npy'))}, Dev {len(np.load(args.vjepa_root/'libero90_dev/row_keys.npy'))}, Eval {len(np.load(args.vjepa_root/'libero90_eval/row_keys.npy'))}.",
                  "Exact VLM feature joins (no nearest-row fill): 15,678/30,324 LIBERO-40 fit rows, 113/1,054 LIBERO-90 Dev rows, 619/619 Eval rows; the common smoke Eval set is recorded in summary.json.",
                  "This is only a smoke run; no conclusion about final representation quality or 3-seed performance is drawn.",""]
        audit_path=LP1_ROOT/"resolution_audit_128.json"
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            lines += ["","### V-JEPA processor tensor statistics (64 sampled windows/camera)","",
                      "| Camera | Input variant | Mean | Variance | HF energy mean |","|---|---|---:|---:|---:|"]
            for camera,variants in audit["statistics"].items():
                for variant,stat in variants.items():
                    lines.append(f"| {camera} | {variant} | {stat['mean']:.6f} | {stat['variance']:.6f} | {stat['high_frequency_energy_mean']:.6f} |")
        (out/"report.md").write_text("\n".join(lines))
    else:
        out=args.output_dir or LP1_ROOT/"repair-01/production"
        result=run_production(data_root=args.data_root,vjepa_root=args.vjepa_root,output_dir=out,
            seeds=args.seeds,updates=args.updates,batch_size=args.batch_size,eval_every=args.eval_every,
            memory_fraction=args.memory_fraction)
        print(json.dumps(result,indent=2))
        return
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
