"""Skill-centred v2 action generation for the lightweight LIBERO study.

Skills are unlabelled clusters of multi-second, gripper-bounded relative
action trajectories. The v2 flow head is trained only from a skill anchor (or
pure noise as the explicit fifth source); the ordinary policy is a separate
F_normal baseline. Existing experiment modules are imported read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import anchor_flow as af
from . import fair_compare as fc
from .compare_splits import SPLITS, _subset
from .experiment import (DEFAULT_CACHE, DEFAULT_DATASET, FlowActionHead,
                         _episode_train_validation_indices,
                         _standardize, _standardize_fit, load_rows)
from .hier_skills import _adjacent_grip_switches, _resample
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR

HORIZON = fc.HORIZON
CURVE_POINTS = 32
KAPPA = 0.1
MIN_SUPPORT = 10
MIN_EPISODES = 3
PENDING_BATCH = 200
TAU_GRID = tuple(np.round(np.arange(0.1, 1.01, 0.1), 1).tolist())
SOURCE_NAMES = ("correct", "phase_shift", "near_skill", "random_skill", "none")
DEFAULT_SOURCE_MIX = (0.40, 0.20, 0.20, 0.10, 0.10)
V21_SOURCE_MIX = (0.30, 0.15, 0.15, 0.10, 0.30)
V22_TAU = 0.3
V22_PHASE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
V22_LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _normalise_mix(mix):
    a = np.asarray(mix, np.float64)
    if a.shape != (5,) or np.any(a < 0) or a.sum() <= 0:
        raise ValueError("source_mix needs five non-negative weights")
    return (a / a.sum()).astype(np.float32)


def _relative_commands(actions):
    """Translate xyz/rotation command coordinates to the interval origin."""
    x = np.asarray(actions, np.float32).copy()
    if x.ndim != 2 or x.shape[1] < 7 or not len(x):
        raise ValueError("actions must have shape (time, >=7)")
    # Position and rotation coordinates become relative; gripper remains an
    # absolute open/close command because differencing it erases its meaning.
    x[:, :6] -= x[:1, :6]
    return x


def trajectory_representation(actions, stride_seconds, points=CURVE_POINTS):
    """Return origin-relative L-point motion and a separate log-duration term."""
    curve = _resample(_relative_commands(actions), points).astype(np.float32)
    duration = max(float(len(actions)) * float(stride_seconds), 1e-4)
    return curve, float(np.log(duration))


def _raw_gripper_spans(rows):
    """Initial boundaries, before data-derived minimum/maximum lengths."""
    episode = np.asarray(rows["episode_id"]).astype(str)
    output = []
    for ep in np.unique(episode):
        ix = np.flatnonzero(episode == ep)
        ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
        acts = np.asarray(rows["a"][ix, 0], np.float32)
        switches = _adjacent_grip_switches(acts[:, None, :])
        starts = [0, *(int(i + 1) for i in np.flatnonzero(switches)), len(ix)]
        output.extend((ix[a:b] for a, b in zip(starts[:-1], starts[1:]) if b > a))
    return output


def derive_length_limits(rows, *, min_quantile=0.10, max_quantile=0.90):
    spans = _raw_gripper_spans(rows)
    lengths = np.asarray([len(s) for s in spans], np.int64)
    if not len(lengths):
        return {"minimum": 2, "maximum": 2, "train_lengths": []}
    minimum = max(2, int(np.floor(np.quantile(lengths, min_quantile))))
    maximum = max(minimum, int(np.ceil(np.quantile(lengths, max_quantile))))
    return {"minimum": minimum, "maximum": maximum, "train_lengths": lengths.tolist()}


def constrain_spans(spans, minimum, maximum):
    """Split long spans and merge short spans, preserving order and coverage."""
    maximum, minimum = int(maximum), int(minimum)
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid span length limits")
    split = []
    for start, end in spans:
        cursor = int(start)
        end = int(end)
        while end - cursor > maximum:
            split.append((cursor, cursor + maximum))
            cursor += maximum
        if end > cursor:
            split.append((cursor, end))
    i = 0
    while i < len(split) and len(split) > 1:
        start, end = split[i]
        if end - start >= minimum:
            i += 1
            continue
        if i == 0:
            split[1] = (start, split[1][1])
            split.pop(0)
        else:
            split[i - 1] = (split[i - 1][0], end)
            split.pop(i)
            i -= 1
    return split


def split_episode_rows(rows, limits):
    """Gripper segmentation using learned duration limits, episode by episode."""
    episode = np.asarray(rows["episode_id"]).astype(str)
    result = []
    for ep in np.unique(episode):
        ix = np.flatnonzero(episode == ep)
        ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
        acts = np.asarray(rows["a"][ix, 0], np.float32)
        switches = _adjacent_grip_switches(acts[:, None, :])
        starts = [0, *(int(i + 1) for i in np.flatnonzero(switches)), len(ix)]
        spans = constrain_spans(list(zip(starts[:-1], starts[1:])),
                                limits["minimum"], limits["maximum"])
        result.extend(ix[a:b] for a, b in spans)
    return result


@dataclass
class SkillUnit:
    skill_id: int
    mean: np.ndarray
    medoid: np.ndarray
    variance: np.ndarray
    median_length: float
    key: np.ndarray
    members: list[dict] = field(default_factory=list)
    support: int = 0
    episodes: set[str] = field(default_factory=set)
    tasks: set[int] = field(default_factory=set)
    classes: list[int] = field(default_factory=list)


class TrajectorySkillMemory:
    """Open-ended DP-means-style memory; updates are non-gradient prototype updates."""
    def __init__(self, epsilon, *, kappa=KAPPA, min_support=MIN_SUPPORT,
                 min_episodes=MIN_EPISODES, pending_batch=PENDING_BATCH):
        self.epsilon = max(float(epsilon), 1e-8)
        self.kappa = float(kappa)
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.pending_batch = int(pending_batch)
        self.units: list[SkillUnit] = []
        self.pending: list[tuple[dict, np.ndarray]] = []
        self.assignments: dict[int, int] = {}
        self.registered = 0
        self.merged = 0
        self.expired = 0
        self.history = []

    @staticmethod
    def distance(left, right):
        return float(np.mean(np.square(np.asarray(left, np.float32) -
                                       np.asarray(right, np.float32)), dtype=np.float64))

    def _nearest(self, rep):
        if not self.units:
            return None, float("inf")
        ds = [self.distance(rep, u.mean_rep) for u in self.units]
        ix = int(np.argmin(ds))
        return self.units[ix], ds[ix]

    def _refresh(self, unit):
        curves = np.stack([m["curve"] for m in unit.members])
        center = curves.mean(axis=0)
        ix = int(np.argmin(np.mean(np.square(curves - center[None]), axis=(1, 2))))
        unit.mean = center.astype(np.float32)
        unit.medoid = curves[ix].copy()
        unit.variance = curves.var(axis=0).astype(np.float32)
        unit.median_length = float(np.median([m["length"] for m in unit.members]))
        unit.key = np.mean(np.stack([m["key"] for m in unit.members]), axis=0).astype(np.float32)
        unit.support = len(unit.members)
        unit.episodes = {m["episode_id"] for m in unit.members}
        unit.tasks = {int(m["task_index"]) for m in unit.members}
        unit.classes = [int(m["skill_class"]) for m in unit.members if int(m["skill_class"]) >= 0]
        unit.mean_rep = _rep(unit.mean, unit.median_length)

    def _register(self, members, *, merge=True):
        eps = {m[0]["episode_id"] for m in members}
        if len(members) < self.min_support or len(eps) < self.min_episodes:
            return False
        if merge:
            cand = np.mean(np.stack([m[1] for m in members]), axis=0)
            old, d = self._nearest(cand)
            if old is not None and d <= self.epsilon:
                old.members.extend(m[0] for m in members)
                self._refresh(old)
                for seg, _ in members:
                    self.assignments[int(seg["segment_id"])] = old.skill_id
                self.merged += 1
                return True
        next_id = len(self.units)
        unit = SkillUnit(next_id, np.zeros((CURVE_POINTS, 7), np.float32),
                         np.zeros((CURVE_POINTS, 7), np.float32),
                         np.zeros((CURVE_POINTS, 7), np.float32), 1.0,
                         np.zeros_like(members[0][0]["key"]))
        unit.members = [m[0] for m in members]
        self._refresh(unit)
        self.units.append(unit)
        for seg, _ in members:
            self.assignments[int(seg["segment_id"])] = unit.skill_id
        self.registered += 1
        return True

    def fit(self, segments, seed=0):
        rng = np.random.default_rng(seed)
        if not segments:
            raise ValueError("no training trajectory segments")
        reps = [_rep(s["curve"], s["length"]) for s in segments]
        n = len(reps)
        pairs = rng.integers(0, n, size=(min(4096, max(64, n * 2)), 2))
        pair_d = np.asarray([self.distance(reps[i], reps[j]) for i, j in pairs])
        median = float(np.median(pair_d[pair_d > 0])) if np.any(pair_d > 0) else 1.0
        self.epsilon = max(self.kappa * median, 1e-8)
        order = rng.permutation(n)
        groups: list[list[tuple[dict, np.ndarray]]] = []
        centers: list[np.ndarray] = []
        for ix in order:
            seg, rep = segments[int(ix)], reps[int(ix)]
            if not centers:
                centers.append(rep.copy()); groups.append([(seg, rep)])
                continue
            ds = [self.distance(rep, c) for c in centers]
            j = int(np.argmin(ds))
            if ds[j] <= self.epsilon:
                groups[j].append((seg, rep))
                centers[j] = np.mean(np.stack([x[1] for x in groups[j]]), axis=0)
            else:
                centers.append(rep.copy()); groups.append([(seg, rep)])
        leftovers = 0
        for members in groups:
            if not self._register(members, merge=False):
                leftovers += len(members)
        self.expired = leftovers
        self.history.append({"data_seen": n, "skills": len(self.units), "pending": 0,
                             "registered": self.registered, "expired": leftovers})
        return self

    def add_episode(self, segments):
        """Prequential, non-gradient insertion for D2; flushes at M6-sized batches."""
        for seg in segments:
            rep = _rep(seg["curve"], seg["length"])
            old, d = self._nearest(rep)
            if old is not None and d <= self.epsilon:
                old.members.append(seg); self._refresh(old)
                self.assignments[int(seg["segment_id"])] = old.skill_id
            else:
                self.pending.append((seg, rep))
        if len(self.pending) >= self.pending_batch:
            self._flush_pending(force=False)

    def _flush_pending(self, force):
        if not self.pending:
            return
        groups, centers = [], []
        for seg, rep in self.pending:
            if not centers:
                centers.append(rep.copy()); groups.append([(seg, rep)]); continue
            ds = [self.distance(rep, x) for x in centers]
            j = int(np.argmin(ds))
            if ds[j] <= self.epsilon:
                groups[j].append((seg, rep))
                centers[j] = np.mean(np.stack([m[1] for m in groups[j]]), axis=0)
            else:
                groups.append([(seg, rep)]); centers.append(rep.copy())
        remain = []
        for group in groups:
            if not self._register(group, merge=True):
                remain.extend(group)
        self.pending = [] if force else remain
        if force:
            self.expired += len(remain)
        self.history.append({"data_seen": sum(u.support for u in self.units) + len(self.pending),
                             "skills": len(self.units), "pending": len(self.pending),
                             "registered": self.registered, "expired": self.expired})

    def finish_stream(self):
        self._flush_pending(force=True)

    def clone(self):
        return copy.deepcopy(self)

    def describe(self):
        supports = [u.support for u in self.units]
        return {"n_skills": len(self.units), "epsilon": self.epsilon,
                "epsilon_rule": "0.1 x median random-pair trajectory MSE from training suite",
                "min_support": self.min_support, "min_episodes": self.min_episodes,
                "pending_batch": self.pending_batch, "registered_groups": self.registered,
                "merged_groups": self.merged, "expired": self.expired,
                "pending_final": len(self.pending), "support": supports,
                "history": self.history}


def _rep(curve, length):
    return np.concatenate([np.asarray(curve, np.float32).reshape(-1),
                           np.asarray([0.10 * np.log(max(float(length), 1e-4))], np.float32)])


def make_segments(rows, query_keys, limits, stride_seconds, bases=None):
    out = []
    row_segment = np.full(len(rows["a"]), -1, np.int64)
    for ix in split_episode_rows(rows, limits):
        ix = np.asarray(ix, np.int64)
        actions = np.asarray(rows["a"][ix, 0], np.float32)
        curve, log_duration = trajectory_representation(actions, stride_seconds)
        seg_id = len(out)
        segment = {"segment_id": seg_id, "rows": ix,
                   "episode_id": str(rows["episode_id"][ix[0]]),
                   "task_index": int(rows["task_index"][ix[0]]),
                   "skill_class": int(rows["skill_class"][ix[0]]) if "skill_class" in rows else -1,
                   "curve": curve, "length": len(ix),
                   "length_seconds": len(ix) * float(stride_seconds),
                   "log_duration": log_duration,
                   "key": np.asarray(query_keys[ix[0]], np.float32).copy()}
        if bases is not None:
            segment["base_seq"] = np.asarray(bases[ix], np.float32).copy()
        out.append(segment)
        row_segment[ix] = seg_id
    return out, row_segment


def _trajectory_chunk(curve, phase, length_primitives, stride_seconds, horizon=HORIZON):
    curve = np.asarray(curve, np.float32)
    phase = float(np.clip(phase, 0.0, 1.0))
    sample_times = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    span = min(1.0 - phase, horizon * float(stride_seconds) /
               max(float(length_primitives) * float(stride_seconds), 1e-4))
    positions = np.clip(phase + sample_times * span, 0.0, 1.0)
    src = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    return np.stack([np.interp(positions, src, curve[:, d]) for d in range(curve.shape[1])], axis=-1).astype(np.float32)


def _cosine_scores(q, units):
    q = np.asarray(q, np.float32)
    keys = np.stack([u.key for u in units]).astype(np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    return keys @ q


class MultiPositiveSelector(nn.Module):
    """Learned query/trajectory towers trained with multi-positive InfoNCE."""
    def __init__(self, query_dim, skill_dim, width=64):
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(query_dim, width), nn.GELU(),
                                         nn.Linear(width, width))
        self.skill_tower = nn.Sequential(nn.Linear(skill_dim, width), nn.GELU(),
                                         nn.Linear(width, width))
        self.log_temperature = nn.Parameter(torch.tensor(-2.3))

    def forward(self, queries, skill_features):
        q = nn.functional.normalize(self.query_tower(queries), dim=-1)
        k = nn.functional.normalize(self.skill_tower(skill_features), dim=-1)
        return (q @ k.T) / self.log_temperature.exp().clamp(0.02, 1.0)


def fit_multi_positive_selector(segments, memory, *, validation_segments=(), seed=0, device="cpu",
                                epochs=120, learning_rate=1e-3):
    """Train using only training-suite segment starts and functional positives."""
    if not segments or not memory.units:
        raise ValueError("selector needs training segments and learned skills")
    ordered = sorted(memory.units, key=lambda u: u.skill_id)
    features = np.stack([u.mean_rep for u in ordered]).astype(np.float32)
    unit_to_col = {u.skill_id: i for i, u in enumerate(ordered)}
    def collect(source, require_assignment):
        qrows, positives = [], []
        for seg in source:
            uid = memory.assignments.get(int(seg["segment_id"])) if require_assignment else None
            if require_assignment and uid is None:
                continue
            qrows.append(np.asarray(seg["key"], np.float32))
            if require_assignment:
                pos_ids = functional_positive_ids(seg, memory, ordered)
            else:
                rep = _rep(seg["curve"], seg["length"])
                pos_ids = [u.skill_id for u in ordered
                           if memory.distance(rep, u.mean_rep) <= memory.epsilon]
            if not pos_ids:
                rep = _rep(seg["curve"], seg["length"])
                pos_ids = [min(ordered, key=lambda u: memory.distance(rep, u.mean_rep)).skill_id]
            positives.append(sorted({unit_to_col[int(x)] for x in pos_ids}))
        if not qrows:
            return np.empty((0, features.shape[-1]), np.float32), np.zeros((0, len(ordered)), bool)
        pos_mask = np.zeros((len(qrows), len(ordered)), bool)
        for row, cols in enumerate(positives):
            pos_mask[row, cols] = True
        return np.stack(qrows), pos_mask

    q_np, positive = collect(segments, True)
    if not len(q_np):
        raise ValueError("no assigned training segments for selector")
    val_np, val_positive = collect(validation_segments, False)
    torch.manual_seed(seed)
    model = MultiPositiveSelector(q_np.shape[-1], features.shape[-1]).to(device)
    q = torch.as_tensor(q_np, device=device)
    k = torch.as_tensor(features, device=device)
    pos = torch.as_tensor(positive, device=device)
    vq = torch.as_tensor(val_np, device=device)
    vp = torch.as_tensor(val_positive, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, val_history = [], []
    best_state, best_val, stale = None, float("inf"), 0
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(q), generator=generator, device=device)
        losses = []
        for ix in order.split(256):
            logits = model(q[ix], k)
            pos_logits = logits.masked_fill(~pos[ix], -torch.inf)
            loss = (torch.logsumexp(logits, dim=1) -
                    torch.logsumexp(pos_logits, dim=1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            if len(vq):
                val_logits = model(vq, k)
                val_loss = (torch.logsumexp(val_logits, dim=1) -
                            torch.logsumexp(val_logits.masked_fill(~vp, -torch.inf), dim=1)).mean()
                score = float(val_loss.cpu())
            else:
                score = history[-1]
        val_history.append(score)
        if score < best_val - 1e-5:
            best_val, stale = score, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 15:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(q, k).cpu().numpy()
    correct = positive[np.arange(len(q_np)), logits.argmax(axis=1)]
    topk = np.argpartition(-logits, min(4, logits.shape[1]-1), axis=1)[:, :min(5, logits.shape[1])]
    recall5 = positive[np.arange(len(q_np))[:, None], topk].any(axis=1)
    return model, {"segments": len(q_np), "validation_segments": len(val_np), "units": len(ordered),
                   "positive_count_mean": float(positive.sum(axis=1).mean()),
                   "training_top1_positive_rate": float(correct.mean()),
                   "training_top5_positive_recall": float(recall5.mean()),
                   "loss_start": history[0], "loss_final": history[-1],
                   "validation_loss_best": best_val, "validation_loss_history": val_history,
                   "epochs": len(history), "temperature": float(model.log_temperature.detach().exp().cpu())}


def functional_positive_ids(segment, memory, units=None):
    """All learned units within the fit-calibrated functional-distance ε."""
    units = sorted(memory.units, key=lambda u: u.skill_id) if units is None else units
    rep = _rep(segment["curve"], segment["length"])
    ids = [u.skill_id for u in units
           if memory.distance(rep, u.mean_rep) <= memory.epsilon]
    assigned = memory.assignments.get(int(segment["segment_id"]))
    if assigned is not None and int(assigned) not in ids:
        ids.append(int(assigned))
    return ids


def selector_scores(model, query, units, device="cpu"):
    ordered = sorted(units, key=lambda u: u.skill_id)
    q = torch.as_tensor(np.asarray(query, np.float32), device=device)
    k = torch.as_tensor(np.stack([u.mean_rep for u in ordered]).astype(np.float32), device=device)
    with torch.no_grad():
        scores = model(q, k).cpu().numpy()
    return scores, [u.skill_id for u in ordered]


def _interpolate_base_sequence(sequence, phase):
    seq = np.asarray(sequence, np.float32)
    if seq.ndim != 2 or not len(seq):
        raise ValueError("base sequence must have shape (time, base_dim)")
    pos = float(np.clip(phase, 0, 1)) * (len(seq) - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    return ((1 - (pos-lo)) * seq[lo] + (pos-lo) * seq[hi]).astype(np.float32)


def decode_member_anchor(m1, member, phase, feature, state):
    """Decode the chosen member's b at the current observation/state."""
    if "base_seq" not in member:
        raise ValueError("member has no saved M1 base sequence")
    base = _interpolate_base_sequence(member["base_seq"], phase)[None]
    residual_dim = int(getattr(m1, "residual_dim", 4))
    residual = np.zeros((1, residual_dim), np.float32)
    return np.asarray(m1.decode(base, residual, np.asarray(feature)[None],
                                np.asarray(state)[None])[0], np.float32)


def _confidence_for(scores, selected):
    """Same q/key evidence for training and inference, including wrong choices."""
    selected = int(selected)
    sim = float(scores[selected])
    other = np.delete(np.asarray(scores, np.float32), selected)
    competitor = float(np.max(other)) if len(other) else -1.0
    gap = sim - competitor
    sim01 = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    gap01 = np.clip((gap + 2.0) / 4.0, 0.0, 1.0)
    return float(0.5 * sim01 + 0.5 * gap01), sim, gap


def tau_from_confidence(confidence):
    return np.clip(1.0 - np.asarray(confidence, np.float32), 0.1, 1.0).astype(np.float32)


def _monotone_bins(confidence, errors, tau_grid=TAU_GRID, bins=8):
    """Validation-MSE-optimal tau by confidence quantile, monotone decreasing."""
    c = np.asarray(confidence, np.float64)
    err = np.asarray(errors, np.float64)  # (tau, row)
    edges = np.unique(np.quantile(c, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, c, side="right")
    vals, scores, counts = [], [], []
    for b in range(len(edges) + 1):
        ix = np.flatnonzero(which == b)
        counts.append(len(ix))
        if len(ix):
            j = int(np.argmin(err[:, ix].mean(axis=1)))
            vals.append(float(tau_grid[j])); scores.append(float(err[j, ix].mean()))
        else:
            vals.append(float(tau_grid[-1])); scores.append(float("nan"))
    unconstrained = list(vals)
    # Confidence bins are low→high. Enforce tau non-increasing.
    for i in range(1, len(vals)):
        vals[i] = min(vals[i], vals[i - 1])
    return {"edges": edges.tolist(), "tau": vals,
            "unconstrained_tau": unconstrained,
            "monotone_adjustment": [float(a-b) for a, b in zip(vals, unconstrained)],
            "steps": [max(2, int(round(10 * x))) for x in vals],
            "validation_mse": scores, "counts": counts}


def apply_confidence_map(calibration, confidence):
    c = np.asarray(confidence, np.float32)
    ix = np.searchsorted(np.asarray(calibration["edges"], np.float32), c, side="right")
    tau = np.asarray(calibration["tau"], np.float32)[ix]
    steps = np.asarray(calibration["steps"], np.int64)[ix]
    return tau, steps


def _map_quantile(values, reference, low=0.1, high=1.0):
    reference = np.asarray(reference, np.float32)
    if not len(reference):
        return np.full_like(np.asarray(values, np.float32), 0.6)
    ranks = np.searchsorted(np.sort(reference), np.asarray(values, np.float32), side="right")
    pct = ranks / max(len(reference), 1)
    return (low + (high - low) * pct).astype(np.float32)


def _fit_skill_flow(rows, actions, anchors, confidences, source_mix, *, seed,
                    epochs, patience, batch_size, device, tau_calibration=None,
                    initial_model=None, initial_norm=None, teacher_model=None,
                    teacher_weight=0.0):
    n, h, da = actions.shape
    dim = h * da
    train_ix, val_ix = _episode_train_validation_indices(rows, seed + 11)
    f_mean, f_scale = _standardize_fit(rows["f"][train_ix])
    s_mean, s_scale = _standardize_fit(rows["s"][train_ix])
    a_mean, a_scale = _standardize_fit(actions[train_ix].reshape(len(train_ix), dim))
    at_np = _standardize(actions.reshape(n, dim), a_mean, a_scale)
    anc_np = _standardize(anchors.reshape(5, n, dim), a_mean, a_scale)
    ft_np = _standardize(rows["f"], f_mean, f_scale)
    st_np = _standardize(rows["s"], s_mean, s_scale)
    if initial_norm is not None:
        # F_normal expects these exact input/output scales.  Reusing its
        # normalizer makes the tau=1 initialization an exact functional copy.
        f_mean, f_scale = initial_norm["f_mean"], initial_norm["f_scale"]
        s_mean, s_scale = initial_norm["s_mean"], initial_norm["s_scale"]
        a_mean, a_scale = initial_norm["a_mean"], initial_norm["a_scale"]
        at_np = _standardize(actions.reshape(n, dim), a_mean, a_scale)
        anc_np = _standardize(anchors.reshape(5, n, dim), a_mean, a_scale)
        ft_np = _standardize(rows["f"], f_mean, f_scale)
        st_np = _standardize(rows["s"], s_mean, s_scale)
    model = FlowActionHead(ft_np.shape[-1], st_np.shape[-1], dim).to(device)
    init_state_equal = None
    if initial_model is not None:
        model.load_state_dict(initial_model.state_dict())
        init_state_equal = all(torch.equal(model.state_dict()[k], value)
                               for k, value in initial_model.state_dict().items())
    if teacher_model is not None:
        teacher_model.eval()
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator(device=device).manual_seed(seed + 29)
    at = torch.as_tensor(at_np, device=device)
    anc = torch.as_tensor(anc_np, device=device)
    ft = torch.as_tensor(ft_np, device=device)
    st = torch.as_tensor(st_np, device=device)
    probs = torch.as_tensor(source_mix, device=device)
    history = []
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    val_rng = np.random.default_rng(seed + 71)
    val_kind = val_rng.choice(5, len(val_ix), p=source_mix)
    val_eps = torch.as_tensor(val_rng.standard_normal((len(val_ix), dim)).astype(np.float32), device=device)
    val_t = torch.as_tensor(val_rng.random(len(val_ix)).astype(np.float32), device=device)
    val_anchor = anc[torch.as_tensor(val_kind, device=device), torch.as_tensor(val_ix, device=device)].clone()
    if tau_calibration is None:
        val_tau = tau_from_confidence(confidences[val_kind, val_ix])
    else:
        val_tau, _ = apply_confidence_map(tau_calibration, confidences[val_kind, val_ix])
    val_tau[val_kind == 4] = 1.0
    vtau = torch.as_tensor(val_tau, device=device)
    val_src = (1 - vtau[:, None]) * val_anchor + vtau[:, None] * val_eps
    val_src[val_kind == 4] = val_eps[val_kind == 4]
    val_xt = (1 - val_t[:, None]) * at[torch.as_tensor(val_ix, device=device)] + val_t[:, None] * val_src
    val_noise_only_xt = ((1 - val_t[:, None]) * at[torch.as_tensor(val_ix, device=device)] +
                         val_t[:, None] * val_eps)
    for epoch in range(int(epochs)):
        order = torch.as_tensor(train_ix, device=device)
        order = order[torch.randperm(len(order), generator=gen, device=device)]
        losses = []
        model.train()
        for start in range(0, len(order), int(batch_size)):
            ix = order[start:start + int(batch_size)]
            kind = torch.multinomial(probs.expand(len(ix), -1), 1, generator=gen).squeeze(1)
            eps = torch.randn((len(ix), dim), generator=gen, device=device)
            anchor = anc[kind, ix]
            batch_conf = torch.as_tensor(confidences, device=device)[kind, ix]
            if tau_calibration is None:
                tau = torch.clamp(1.0 - batch_conf, 0.1, 1.0)
            else:
                tau_np, _ = apply_confidence_map(tau_calibration, batch_conf.detach().cpu().numpy())
                tau = torch.as_tensor(tau_np, device=device)
            tau = torch.where(kind == 4, torch.ones_like(tau), tau)
            src = (1 - tau[:, None]) * anchor + tau[:, None] * eps
            src = torch.where((kind == 4)[:, None], eps, src)
            t = torch.rand((len(ix),), generator=gen, device=device)
            target = at[ix]
            xt = (1 - t[:, None]) * target + t[:, None] * src
            pred = model(xt, t, ft[ix], st[ix])
            loss = (pred - (src - target)).square().mean()
            if teacher_model is not None and teacher_weight > 0:
                no_anchor = kind == 4
                if bool(no_anchor.any()):
                    with torch.no_grad():
                        teacher = teacher_model(xt[no_anchor], t[no_anchor],
                                                ft[ix][no_anchor], st[ix][no_anchor])
                    loss = loss + float(teacher_weight) * (pred[no_anchor] - teacher).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pv = model(val_xt, val_t, ft[torch.as_tensor(val_ix, device=device)],
                       st[torch.as_tensor(val_ix, device=device)])
            val_loss = float((pv - (val_src - at[torch.as_tensor(val_ix, device=device)])).square().mean().cpu())
            if teacher_model is not None and teacher_weight > 0:
                val_f = ft[torch.as_tensor(val_ix, device=device)]
                val_s = st[torch.as_tensor(val_ix, device=device)]
                noise_pred = model(val_noise_only_xt, val_t, val_f, val_s)
                noise_teacher = teacher_model(val_noise_only_xt, val_t, val_f, val_s)
                val_loss += float(teacher_weight) * float((noise_pred - noise_teacher).square().mean().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    norm = {"f_mean": f_mean, "f_scale": f_scale, "s_mean": s_mean, "s_scale": s_scale,
            "a_mean": a_mean, "a_scale": a_scale, "action_shape": tuple(actions.shape[1:])}
    return model, norm, {"history": history, "best_epoch": best_epoch,
                         "best_validation_loss": best_loss, "source_mix": source_mix.tolist(),
                         "source_names": list(SOURCE_NAMES),
                         "initialized_from": "F_normal" if initial_model is not None else "random",
                         "initial_state_equal": init_state_equal,
                         "teacher_model": "F_normal" if teacher_model is not None else None,
                         "teacher_weight_on_none_source": float(teacher_weight)}


def _make_training_sources(rows, segments, memory, query_keys, action_shape, stride_seconds, seed,
                           selector=None, device="cpu"):
    n = len(rows["a"])
    h, da = action_shape
    sources = np.zeros((5, n, h, da), np.float32)
    confidence = np.ones((5, n), np.float32) * 0.5
    rng = np.random.default_rng(seed)
    seg_by_row = {}
    for seg in segments:
        uid = memory.assignments.get(int(seg["segment_id"]))
        if uid is None:
            # A rare low-support segment still gets the closest learned skill;
            # its membership is not used to change the learned units.
            rep = _rep(seg["curve"], seg["length"])
            uid = int(np.argmin([memory.distance(rep, u.mean_rep) for u in memory.units]))
        seg_by_row[int(seg["segment_id"])] = (seg, uid)
    for seg in segments:
        uid = seg_by_row[int(seg["segment_id"])][1]
        span = seg["rows"]
        units = memory.units
        if selector is None:
            scores = _cosine_scores(query_keys[span[0]], units)
            id_order = [u.skill_id for u in units]
        else:
            score_matrix, id_order = selector_scores(selector, query_keys[span[0]][None], units, device)
            scores = score_matrix[0]
        rank_cols = np.argsort(-scores)
        rank = np.asarray([id_order[i] for i in rank_cols], np.int64)
        near_choices = [int(x) for x in rank[1:5]] or [uid]
        for local, row in enumerate(span):
            phi = local / max(len(span) - 1, 1)
            correct = units[uid]
            sources[0, row] = _trajectory_chunk(correct.mean, phi, correct.median_length, stride_seconds, h)
            shifted_phi = float(np.clip(phi + rng.choice([-1, 1]) * rng.uniform(0.15, 0.5), 0, 1))
            sources[1, row] = _trajectory_chunk(correct.mean, shifted_phi, correct.median_length, stride_seconds, h)
            near_id = int(rng.choice(near_choices))
            near = units[near_id]
            sources[2, row] = _trajectory_chunk(near.mean, phi, near.median_length, stride_seconds, h)
            random_id = int(rng.integers(len(units)))
            rnd = units[random_id]
            sources[3, row] = _trajectory_chunk(rnd.mean, phi, rnd.median_length, stride_seconds, h)
            for m, selected in enumerate((uid, uid, near_id, random_id)):
                c, _, _ = _confidence_for(scores, id_order.index(selected))
                confidence[m, row] = c
            confidence[4, row] = 0.0
    return sources, confidence


def _segment_truth_unit(segment, memory):
    rep = _rep(segment["curve"], segment["length"])
    return int(np.argmin([memory.distance(rep, u.mean_rep) for u in memory.units]))


def _schedule(rows, query_keys, segments, memory, stride_seconds, *, variant="mean",
              force="top1", seed=0, selector=None, m1=None, device="cpu"):
    n, h, da = len(rows["a"]), HORIZON, rows["a"].shape[-1]
    anchor = np.zeros((n, h, da), np.float32)
    ids = np.full(n, -1, np.int64); phase = np.zeros(n, np.float32)
    conf = np.zeros(n, np.float32); sim = np.zeros(n, np.float32); margin = np.zeros(n, np.float32)
    sigma = np.zeros(n, np.float32); oracle = np.full(n, -1, np.int64)
    rng = np.random.default_rng(seed)
    for seg in segments:
        rows_ix = seg["rows"]
        oracle_id = _segment_truth_unit(seg, memory)
        cursor = 0
        while cursor < len(rows_ix):
            row0 = int(rows_ix[cursor])
            if selector is None:
                scores = _cosine_scores(query_keys[row0], memory.units)
                id_order = [u.skill_id for u in memory.units]
            else:
                score_matrix, id_order = selector_scores(
                    selector, query_keys[row0][None], memory.units, device)
                scores = score_matrix[0]
            ranked_cols = np.argsort(-scores)
            ranked = np.asarray([id_order[i] for i in ranked_cols], np.int64)
            top_id = int(ranked[0])
            if force == "top1":
                uid = top_id
            elif force == "near":
                uid = int(rng.choice(ranked[1:min(5, len(ranked))])) if len(ranked) > 1 else top_id
            elif force == "random":
                uid = int(rng.integers(len(memory.units)))
            elif force == "oracle":
                uid = oracle_id
            else:
                raise ValueError(force)
            unit = memory.units[uid]
            run_len = max(1, int(round(unit.median_length)))
            end = min(len(rows_ix), cursor + run_len)
            selected_col = id_order.index(uid)
            c, csim, mg = _confidence_for(scores, selected_col)
            curve = unit.mean if variant == "mean" else unit.medoid
            for local in range(cursor, end):
                row = int(rows_ix[local])
                phi = min((local - cursor) / max(run_len, 1), 1.0)
                if variant == "state_decode":
                    if m1 is None:
                        raise ValueError("state_decode anchor requires the fitted M1 decoder")
                    member = max(unit.members, key=lambda m: float(np.dot(
                        m["key"] / max(float(np.linalg.norm(m["key"])), 1e-8),
                        query_keys[row] / max(float(np.linalg.norm(query_keys[row])), 1e-8))))
                    anchor[row] = decode_member_anchor(m1, member, phi,
                                                       rows["f"][row], rows["s"][row])[:h]
                else:
                    anchor[row] = _trajectory_chunk(curve, phi, unit.median_length, stride_seconds, h)
                ids[row] = uid; phase[row] = phi; conf[row] = c
                sim[row] = csim; margin[row] = mg
                sigma[row] = float(np.mean(unit.variance[min(int(phi * (CURVE_POINTS - 1)), CURVE_POINTS - 1)]))
                oracle[row] = oracle_id
            cursor = end
    return {"anchor": anchor, "skill_id": ids, "phi": phase, "confidence": conf,
            "similarity": sim, "margin": margin, "sigma": sigma, "oracle_id": oracle,
            "correct": ids == oracle}


def _metrics(pred, target):
    per = np.mean(np.square(np.asarray(pred) - np.asarray(target)[None]), axis=(2, 3))
    return {"mse": float(per.mean()), "best_of_8_mse": float(per.min(axis=0).mean()),
            "per_row_mse": per.mean(axis=0).astype(np.float32)}


def _predict(model, norm, rows, anchor, eps, tau, steps, device):
    return fc.rollout(model, norm, rows["f"], rows["s"], anchor, eps,
                      np.asarray(tau, np.float32), n=np.asarray(steps, np.int64), device=device)


def _calibrate_tau(model, norm, val, schedule, eps, *, device, bins=8):
    # A fixed validation subset bounds calibration cost while preserving paired noise.
    if len(val["a"]) > 384:
        ix = np.sort(np.random.default_rng(19).choice(len(val["a"]), 384, replace=False))
        old_n = len(val["a"])
        val = _subset(val, np.isin(np.arange(len(val["a"])), ix))
        schedule = {k: (v[ix] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == old_n else v)
                    for k, v in schedule.items()}
        eps = eps[:, ix]
    errors = []
    target = val["a"][:, :HORIZON]
    for tau in TAU_GRID:
        tau_arr = np.full(len(target), tau, np.float32)
        steps = np.full(len(target), max(2, int(round(10 * tau))), np.int64)
        pred = _predict(model, norm, val, schedule["anchor"], eps, tau_arr, steps, device)
        errors.append(np.mean(np.square(pred - target[None]), axis=(0, 2, 3)))
    return _monotone_bins(schedule["confidence"], np.stack(errors), bins=bins)


def _confidence_bins(schedule, pred, target, bins=8):
    c = schedule["confidence"]
    edges = np.unique(np.quantile(c, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, c, side="right")
    result = []
    per = np.mean(np.square(pred - target[None]), axis=(0, 2, 3))
    for b in range(len(edges) + 1):
        ix = np.flatnonzero(which == b)
        result.append({"bin": b, "n": len(ix),
                       "correct_skill_rate": float(np.mean(schedule["correct"][ix])) if len(ix) else None,
                       "mse": float(per[ix].mean()) if len(ix) else None,
                       "confidence_mean": float(c[ix].mean()) if len(ix) else None})
    return {"edges": edges.tolist(), "bins": result}


def selection_metrics(query, segments, memory, selector=None, device="cpu"):
    """Segment-start top-1 accuracy and top-5 recall against functional truth."""
    old_ok, new_ok, topk_ok = [], [], []
    for seg in segments:
        truth = _segment_truth_unit(seg, memory)
        old_scores = _cosine_scores(query[int(seg["rows"][0])], memory.units)
        old_ids = np.asarray([u.skill_id for u in memory.units])
        old_order = old_ids[np.argsort(-old_scores)]
        old_ok.append(bool(old_order[0] == truth))
        if selector is not None:
            scores, ids = selector_scores(selector, query[int(seg["rows"][0])][None],
                                          memory.units, device)
            order = np.asarray(ids)[np.argsort(-scores[0])]
            new_ok.append(bool(order[0] == truth))
            topk_ok.append(bool(truth in order[:min(5, len(order))]))
    return {"previous_cosine_top1": float(np.mean(old_ok)) if old_ok else None,
            "multi_positive_top1": float(np.mean(new_ok)) if new_ok else None,
            "multi_positive_top5_recall": float(np.mean(topk_ok)) if topk_ok else None,
            "segments": len(old_ok)}


def _candidate_choice(similarity, correction, lam):
    """Choose (skill, phase) by normalized similarity minus denoising repair."""
    sim = np.asarray(similarity, np.float32)
    corr = np.asarray(correction, np.float32)
    if sim.ndim != 2 or corr.ndim != 3 or sim.shape != corr.shape[:2]:
        raise ValueError("similarity must be (rows,k), correction (rows,k,phases)")
    span = np.ptp(sim, axis=1, keepdims=True)
    sim01 = (sim - sim.min(axis=1, keepdims=True)) / np.maximum(span, 1e-6)
    score = sim01[:, :, None] - float(lam) * corr
    flat = score.reshape(len(score), -1).argmax(axis=1)
    return flat // corr.shape[2], flat % corr.shape[2], score


def _fit_repair_lambda(similarity, correction, candidate_mse, *, lambdas=V22_LAMBDA_GRID,
                       row_indices=None):
    """Pick lambda on held-out training-suite rows using paired candidate MSE."""
    mse = np.asarray(candidate_mse, np.float64)
    row_indices = np.arange(len(mse)) if row_indices is None else np.asarray(row_indices, np.int64)
    similarity, correction, mse = similarity[row_indices], correction[row_indices], mse[row_indices]
    records = []
    for lam in lambdas:
        skill_ix, phase_ix, _ = _candidate_choice(similarity, correction, lam)
        picked = mse[np.arange(len(mse)), skill_ix, phase_ix]
        records.append({"lambda": float(lam), "validation_mse": float(picked.mean()),
                        "selected_rows": int(len(picked))})
    best = min(records, key=lambda item: item["validation_mse"])
    return {"selected_lambda": best["lambda"], "validation_mse": best["validation_mse"],
            "grid": records, "calibration_rows": len(mse)}


def _candidate_bundle(rows, query_keys, memory, model, norm, eps, *, selector,
                      device, stride_seconds=0.5, max_k=10,
                      phases=V22_PHASE_GRID, chunk_rows=32):
    """Denoise top-k x phase-grid mean-trajectory anchors at common tau=0.3.

    Correction is RMS in per-action standardized coordinates. Candidate
    scoring uses one paired noise sample; final evaluation uses all eight.
    """
    count, h, da = len(rows["a"]), HORIZON, rows["a"].shape[-1]
    ordered_units = sorted(memory.units, key=lambda u:u.skill_id)
    unit_ids = np.asarray([u.skill_id for u in ordered_units], np.int64)
    phase_grid = np.asarray(phases, np.float32)
    if len(unit_ids) < max_k:
        max_k = len(unit_ids)
    sim_out = np.zeros((count, max_k), np.float32)
    correction_out = np.zeros((count, max_k, len(phase_grid)), np.float32)
    mse_out = np.zeros_like(correction_out)
    ids_out = np.zeros((count, max_k), np.int64)
    action_scale = np.asarray(norm["a_scale"], np.float32).reshape(h, da)
    for lo in range(0, count, int(chunk_rows)):
        hi = min(count, lo + int(chunk_rows)); ix = np.arange(lo, hi); b = len(ix)
        if selector is None:
            score_rows = np.stack([_cosine_scores(query_keys[i], ordered_units) for i in ix])
            id_order = unit_ids.tolist()
        else:
            score_rows, id_order = selector_scores(selector, query_keys[ix], ordered_units, device)
        col_rank = np.argsort(-score_rows, axis=1)[:, :max_k]
        ids = np.asarray(id_order, np.int64)[col_rank]
        sims = np.take_along_axis(score_rows, col_rank, axis=1)
        sim_out[ix] = sims; ids_out[ix] = ids
        anchors = np.empty((b, max_k, len(phase_grid), h, da), np.float32)
        for bi in range(b):
            for ki in range(max_k):
                unit = memory.units[int(ids[bi, ki])]
                for pi, phi in enumerate(phase_grid):
                    anchors[bi, ki, pi] = _trajectory_chunk(unit.mean, float(phi),
                                                            unit.median_length, stride_seconds, h)
        n_candidates = max_k * len(phase_grid)
        flat_anchor = anchors.reshape(b * n_candidates, h, da)
        rep_rows = {"f": np.repeat(rows["f"][ix], n_candidates, axis=0),
                    "s": np.repeat(rows["s"][ix], n_candidates, axis=0)}
        score_eps = np.repeat(np.asarray(eps[:1, ix]), n_candidates, axis=1)
        pred = fc.rollout(model, norm, rep_rows["f"], rep_rows["s"], flat_anchor,
                          score_eps, np.full(b*n_candidates, V22_TAU, np.float32),
                          n=np.full(b*n_candidates, max(2, round(10*V22_TAU)), np.int64),
                          device=device, chunk=512)[0].reshape(b, max_k, len(phase_grid), h, da)
        delta_std = (pred - anchors) / np.maximum(action_scale[None, None, None], 1e-6)
        correction_out[ix] = np.sqrt(np.mean(np.square(delta_std), axis=(3, 4)))
        targets = rows["a"][ix, :h]
        mse_out[ix] = np.mean(np.square(pred - targets[:, None, None]), axis=(3, 4))
    return {"similarity": sim_out, "correction": correction_out,
            "candidate_mse": mse_out, "skill_ids": ids_out, "phases": phase_grid}


def _v22_schedule(rows, segments, memory, bundle, *, lam, k, fixed, stride_seconds=0.5):
    """Build the denoising-selected skill/phase schedule."""
    n, h, da = len(rows["a"]), HORIZON, rows["a"].shape[-1]
    anchor = np.zeros((n, h, da), np.float32)
    ids = np.full(n, -1, np.int64); phase = np.zeros(n, np.float32)
    correction = np.zeros(n, np.float32); correct = np.zeros(n, bool)
    oracle = np.full(n, -1, np.int64)
    for seg in segments:
        row_ix = np.asarray(seg["rows"], np.int64)
        if not len(row_ix):
            continue
        truth = _segment_truth_unit(seg, memory)
        if fixed:
            cursor = 0
            while cursor < len(row_ix):
                candidate_row = int(row_ix[cursor])
                si, pi, _ = _candidate_choice(bundle["similarity"][candidate_row:candidate_row+1, :k],
                                              bundle["correction"][candidate_row:candidate_row+1, :k], lam)
                unit_id = int(bundle["skill_ids"][candidate_row, si[0]])
                phi0 = float(bundle["phases"][pi[0]])
                unit = memory.units[unit_id]
                run_len = max(1, int(round(unit.median_length)))
                end = min(len(row_ix), cursor + run_len)
                selected_corr = float(bundle["correction"][candidate_row, si[0], pi[0]])
                for local in range(cursor, end):
                    row = int(row_ix[local])
                    phi = float(np.clip(phi0 + (local - cursor) / max(float(unit.median_length), 1.0), 0, 1))
                    anchor[row] = _trajectory_chunk(unit.mean, phi, unit.median_length, stride_seconds, h)
                    ids[row], phase[row], oracle[row] = unit_id, phi, truth
                    correction[row] = selected_corr
                    correct[row] = unit_id == truth
                cursor = end
        else:
            si, pi, _ = _candidate_choice(bundle["similarity"][row_ix, :k],
                                          bundle["correction"][row_ix, :k], lam)
            chosen_units = bundle["skill_ids"][row_ix, :k][np.arange(len(row_ix)), si]
            chosen_phases = bundle["phases"][pi]
            for pos, row in enumerate(row_ix):
                unit_id, phi = int(chosen_units[pos]), float(chosen_phases[pos])
                unit = memory.units[unit_id]
                anchor[row] = _trajectory_chunk(unit.mean, phi, unit.median_length, stride_seconds, h)
                ids[row], phase[row], oracle[row] = unit_id, phi, truth
                correction[row] = bundle["correction"][row, si[pos], pi[pos]]
                correct[row] = unit_id == truth
    return {"anchor": anchor, "skill_id": ids, "phi": phase,
            "correction_norm": correction, "oracle_id": oracle, "correct": correct}


def _segment_truth_rows(segments, memory, count):
    truth = np.full(count, -1, np.int64)
    for seg in segments:
        truth[np.asarray(seg["rows"], np.int64)] = _segment_truth_unit(seg, memory)
    return truth


def _calibrate_v22_lambdas(val, val_q, val_segments, memory, model, norm, eps,
                           selector, device, stride_seconds):
    bundle = _candidate_bundle(val, val_q, memory, model, norm, eps,
                               selector=selector, device=device,
                               stride_seconds=stride_seconds)
    truth = _segment_truth_rows(val_segments, memory, len(val["a"]))
    starts = np.asarray([int(seg["rows"][0]) for seg in val_segments if len(seg["rows"])], np.int64)
    result = {}
    for requested_k in (5, 10):
        k = min(requested_k, bundle["skill_ids"].shape[1])
        for fixed in (False, True):
            key = f"k{requested_k}_{'fixed' if fixed else 'reselect'}"
            rows = starts if fixed else np.arange(len(val["a"]))
            fit = _fit_repair_lambda(bundle["similarity"][:, :k],
                                     bundle["correction"][:, :k],
                                     bundle["candidate_mse"][:, :k],
                                     row_indices=rows)
            for record in fit["grid"]:
                si, pi, _ = _candidate_choice(bundle["similarity"][rows, :k],
                                              bundle["correction"][rows, :k], record["lambda"])
                chosen = bundle["skill_ids"][rows, :k][np.arange(len(rows)), si]
                record["skill_accuracy"] = float(np.mean(chosen == truth[rows]))
            fit["candidate_count"] = k
            fit["fixed_until_segment_end"] = fixed
            fit["tau"] = V22_TAU
            fit["phase_grid"] = bundle["phases"].tolist()
            result[key] = fit
    return result


def _v22_candidate_diagnostics(bundle, segments, memory, *, k):
    truth = _segment_truth_rows(segments, memory, len(bundle["skill_ids"]))
    seen = truth >= 0
    ids = bundle["skill_ids"][:, :k]
    corr = bundle["correction"][:, :k]
    contains, true_best = [], []
    true_corr, wrong_corr = [], []
    for row in np.flatnonzero(seen):
        hit = np.flatnonzero(ids[row] == truth[row])
        contains.append(bool(len(hit)))
        all_min = float(corr[row].min())
        if len(hit):
            this = float(corr[row, hit[0]].min())
            true_best.append(this <= all_min + 1e-7)
            true_corr.append(this)
            wrong_values = np.delete(corr[row], hit[0], axis=0)
            wrong_corr.append(float(wrong_values.min()))
    return {"oracle_in_topk_rate": float(np.mean(contains)) if contains else None,
            "oracle_has_smallest_repair_rate_given_in_topk": float(np.mean(true_best)) if true_best else None,
            "oracle_candidate_repair_rms_mean": float(np.mean(true_corr)) if true_corr else None,
            "best_wrong_candidate_repair_rms_mean": float(np.mean(wrong_corr)) if wrong_corr else None,
            "rows": int(seen.sum())}


def _v22_evaluate_condition(rows, query, segments, memory, model, norm, eps,
                            lambda_calibration, *, selector, device, stride_seconds,
                            do_stream=False, seed=0, normal_prediction=None):
    configs = [(requested_k, fixed, f"k{requested_k}_{'fixed' if fixed else 'reselect'}")
               for requested_k in (5, 10) for fixed in (False, True)]
    n, h, da = len(rows["a"]), HORIZON, rows["a"].shape[-1]
    samples = eps.shape[0]
    pred_store = {key: np.zeros((samples, n, h, da), np.float32) for _, _, key in configs}
    schedule_store = {key: {name: np.zeros(n, dtype=dtype) for name, dtype in
                            (("skill_id", np.int64), ("phi", np.float32),
                             ("correction_norm", np.float32), ("oracle_id", np.int64),
                             ("correct", bool))} for _, _, key in configs}
    candidate_diag = {f"k{k}": [] for k in (5, 10)}

    def process(part, qpart, segpart, mem, epspart, original_ix):
        bundle = _candidate_bundle(part, qpart, mem, model, norm, epspart,
                                   selector=selector, device=device,
                                   stride_seconds=stride_seconds)
        for requested_k in (5, 10):
            k = min(requested_k, bundle["skill_ids"].shape[1])
            candidate_diag[f"k{requested_k}"].append(
                _v22_candidate_diagnostics(bundle, segpart, mem, k=k))
        for requested_k, fixed, key in configs:
            k = min(requested_k, bundle["skill_ids"].shape[1])
            lam = lambda_calibration[key]["selected_lambda"]
            schedule = _v22_schedule(part, segpart, mem, bundle, lam=lam, k=k,
                                     fixed=fixed, stride_seconds=stride_seconds)
            pred = _predict(model, norm, part, schedule["anchor"], epspart,
                            np.full(len(part["a"]), V22_TAU, np.float32),
                            np.full(len(part["a"]), max(2, round(10*V22_TAU)), np.int64), device)
            pred_store[key][:, original_ix] = pred
            for field in schedule_store[key]:
                schedule_store[key][field][original_ix] = schedule[field]
        return bundle

    if not do_stream:
        bundle = process(rows, query, segments, memory, eps, np.arange(n))
    else:
        working = memory.clone()
        episode_ids = np.unique(rows["episode_id"].astype(str))
        ordering = np.random.default_rng(seed + 841).permutation(episode_ids)
        bundle = None
        for episode in ordering:
            ix = np.flatnonzero(rows["episode_id"].astype(str) == episode)
            ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
            segpart = [dict(seg, rows=np.searchsorted(ix, seg["rows"]))
                       for seg in segments if seg["episode_id"] == episode]
            if not segpart:
                continue
            part = _subset(rows, np.isin(np.arange(n), ix))
            bundle = process(part, query[ix], segpart, working, eps[:, ix], ix)
            working.add_episode(segpart)
        working.finish_stream()

    target = rows["a"][:, :h]
    result = {}
    for requested_k, fixed, key in configs:
        schedule = schedule_store[key]
        metrics = _metrics(pred_store[key], target)
        per_row = metrics.pop("per_row_mse")
        metrics.update({"skill_accuracy": float(np.mean(schedule["correct"])),
                        "mean_repair_norm": float(schedule["correction_norm"].mean()),
                        "mean_phase": float(schedule["phi"].mean()),
                        "lambda": lambda_calibration[key]["selected_lambda"],
                        "tau": V22_TAU, "step_count": max(2, round(10*V22_TAU)),
                        "fixed_until_segment_end": fixed})
        if normal_prediction is not None:
            normal_per = np.mean(np.square(np.asarray(normal_prediction) - target[None]), axis=(0, 2, 3))
            metrics["D1_worse_row_fraction"] = float(np.mean(per_row > normal_per))
        result[key] = metrics
    diagnostics = {}
    for k in (5, 10):
        chunks = candidate_diag[f"k{k}"]
        # D2 uses per-episode memories; average diagnostic summaries over episodes.
        numeric = [x for x in chunks if x["rows"] > 0]
        diagnostics[f"k{k}"] = {
            name: float(np.mean([x[name] for x in numeric if x[name] is not None]))
            if any(x[name] is not None for x in numeric) else None
            for name in ("oracle_in_topk_rate", "oracle_has_smallest_repair_rate_given_in_topk",
                         "oracle_candidate_repair_rms_mean", "best_wrong_candidate_repair_rms_mean")}
    return result, diagnostics


def _run_condition(model_n, norm_n, model_pa, norm_pa, model_v2, norm_v2, rows, segments,
                   query, memory, stride, eps, tau_map, sigma_reference,
                   pa_calibration, raw_calibration, primitive, raw,
                   *, split, seed, device, do_stream=False, representation="mean",
                   selector=None, m1=None):
    target = rows["a"][:, :HORIZON].copy()
    normal = fc.rollout(model_n, norm_n, rows["f"], rows["s"], None, eps,
                        np.ones(len(rows["a"]), np.float32),
                        n=np.full(len(rows["a"]), 10, np.int64), device=device)
    outputs = {"normal": _metrics(normal, target)}
    # fair_compare P-a: M6 decoded unit with F_anchor.
    primitive_anchor, primitive_confidence = primitive(rows, query, do_stream)
    primitive_tau = fc.apply_g(pa_calibration, primitive_confidence)
    primitive_steps = fc.steps_for(primitive_tau)
    pa = fc.rollout(model_pa, norm_pa, rows["f"], rows["s"], primitive_anchor, eps,
                    primitive_tau, n=primitive_steps, device=device)
    outputs["primitive_Pa"] = _metrics(pa, target)
    raw_anchor, raw_confidence = raw(rows, query, do_stream)
    raw_tau = fc.apply_g(raw_calibration, raw_confidence)
    raw_pred = fc.rollout(model_n, norm_n, rows["f"], rows["s"], raw_anchor, eps,
                          raw_tau, n=fc.steps_for(raw_tau), device=device)
    outputs["raw_interval"] = _metrics(raw_pred, target)

    recovery_memory = memory.clone()
    original = memory
    if do_stream:
        original = memory.clone()
        predictions = []
        ordering = np.random.default_rng(seed + 841).permutation(np.unique(rows["episode_id"].astype(str)))
        row_chunks, seg_chunks = [], []
        for ep in ordering:
            ix = np.flatnonzero(rows["episode_id"].astype(str) == ep)
            ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
            sub = _subset(rows, np.isin(np.arange(len(rows["a"])), ix))
            qsub = query[ix]
            seg_sub = [dict(s, rows=np.searchsorted(ix, s["rows"])) for s in segments if s["episode_id"] == ep]
            if not seg_sub:
                continue
            sched = _schedule(sub, qsub, seg_sub, original, stride, variant=representation,
                              force="top1", seed=seed, selector=selector, m1=m1, device=device)
            tau, steps = tau_map(sched)
            p = _predict(model_v2, norm_v2, sub, sched["anchor"], eps[:, ix], tau, steps, device)
            predictions.append((ix, p, sched, tau, steps, normal[:, ix], target[ix]))
            original.add_episode(seg_sub)
        original.finish_stream()
        for ix, pred, sched, tau, steps, normal_part, tgt in predictions:
            row_chunks.append((ix, pred, sched, tau, steps, normal_part, tgt))
        # Restore full row order for all aggregate metrics.
        pred_all = np.zeros_like(normal); normal_all = normal.copy()
        tau_all = np.zeros(len(target), np.float32); steps_all = np.zeros(len(target), np.int64)
        sched_all = {"anchor": np.zeros_like(target), "skill_id": np.full(len(target), -1, np.int64),
                     "phi": np.zeros(len(target), np.float32), "confidence": np.zeros(len(target), np.float32),
                     "similarity": np.zeros(len(target), np.float32), "margin": np.zeros(len(target), np.float32),
                     "sigma": np.zeros(len(target), np.float32), "oracle_id": np.full(len(target), -1, np.int64),
                     "correct": np.zeros(len(target), bool)}
        for ix, pred, sched, tau, steps, _, _ in row_chunks:
            pred_all[:, ix] = pred; tau_all[ix] = tau; steps_all[ix] = steps
            for k in sched_all: sched_all[k][ix] = sched[k]
        sched, tau, steps, skill_pred = sched_all, tau_all, steps_all, pred_all
    else:
        sched = _schedule(rows, query, segments, original, stride, variant=representation,
                          force="top1", seed=seed, selector=selector, m1=m1, device=device)
        tau, steps = tau_map(sched)
        skill_pred = _predict(model_v2, norm_v2, rows, sched["anchor"], eps, tau, steps, device)
    outputs["skill_v2"] = _metrics(skill_pred, target)
    outputs["skill_v2"]["tau_mean"] = float(np.mean(tau))
    outputs["skill_v2"]["steps_mean"] = float(np.mean(steps))
    outputs["skill_v2"]["D1_worse_row_fraction"] = float(np.mean(
        outputs["skill_v2"]["per_row_mse"] > outputs["normal"]["per_row_mse"]))
    outputs["skill_v2"]["confidence_bins"] = _confidence_bins(sched, skill_pred, target)
    # Fixed-tau variants are useful both as a baseline and for wrong-skill recovery.
    outputs["fixed_tau"] = {}
    for t in (0.2, 0.6, 1.0):
        p = _predict(model_v2, norm_v2, rows, sched["anchor"], eps,
                     np.full(len(target), t, np.float32), np.full(len(target), max(2, round(10*t))), device)
        outputs["fixed_tau"][str(t)] = _metrics(p, target)
    # Spread-only noise policy, calibrated on training unit phase variance.
    tau_spread = _map_quantile(sched["sigma"], sigma_reference, .1, 1.0)
    spread_pred = _predict(model_v2, norm_v2, rows, sched["anchor"], eps,
                            tau_spread, np.maximum(2, np.rint(10*tau_spread)).astype(np.int64), device)
    outputs["skill_v2_sigma"] = {**_metrics(spread_pred, target), "tau_mean": float(tau_spread.mean())}

    recover = {}
    for force in ("near", "random", "oracle"):
        forced = _schedule(rows, query, segments, recovery_memory, stride, variant=representation,
                           force=force, seed=seed + (0 if force == "near" else 1),
                           selector=selector, m1=m1, device=device)
        forced_tau, forced_steps = tau_map(forced)
        pred = _predict(model_v2, norm_v2, rows, forced["anchor"], eps, forced_tau, forced_steps, device)
        recover[force] = {"confidence": float(np.mean(forced["confidence"])),
                          "correct_skill_rate": float(np.mean(forced["correct"])),
                          "confidence_bins": _confidence_bins(forced, pred, target)}
        recover[force]["confidence_tau"] = _metrics(pred, target)
        recover[force]["fixed_tau"] = {}
        for t in (0.2, 0.6, 1.0):
            p = _predict(model_v2, norm_v2, rows, forced["anchor"], eps,
                         np.full(len(target), t, np.float32),
                         np.full(len(target), max(2, round(10*t))), device)
            recover[force]["fixed_tau"][str(t)] = _metrics(p, target)
    outputs["forced_skill_recovery"] = recover
    row_detail = {"skill_id": sched["skill_id"], "phi": sched["phi"], "tau": tau,
                  "confidence": sched["confidence"], "anchor": sched["anchor"],
                  "anchor_to_target": (sched["anchor"] - target).mean(axis=1),
                  "denoise_correction": (skill_pred.mean(axis=0) - sched["anchor"]),
                  "final_minus_target": (skill_pred.mean(axis=0) - target),
                  "target": target, "normal_prediction": normal,
                  "final_prediction": skill_pred.mean(axis=0),
                  "oracle_skill_id": sched["oracle_id"], "correct_skill": sched["correct"],
                  "frame": rows["frame"], "episode_id": rows["episode_id"],
                  "task_index": rows["task_index"]}
    # Do not put row arrays into results.json; the complete arrays are in per_row.npz.
    for method in ("normal", "primitive_Pa", "raw_interval", "skill_v2", "skill_v2_sigma"):
        outputs[method].pop("per_row_mse", None)
    for v in outputs["fixed_tau"].values(): v.pop("per_row_mse", None)
    for force in recover.values():
        force["confidence_tau"].pop("per_row_mse", None)
        for v in force["fixed_tau"].values(): v.pop("per_row_mse", None)
    return outputs, row_detail


def _summary(payload):
    lines = ["# スキル中心 v2：軽量版", "",
             f"実行: Split={payload['splits']}、seed={payload['seeds']}、stride={payload['stride_seconds']:.3f}s、H={HORIZON}、instruction条件。", "",
             "## D3（未見suite、メモリ追加なし）の主比較", "",
             "| Split／表現 | スキル数 | 被覆 | D3 F_normal | 生区間 | primitive P-a | v2 信頼度τ | v2 固定τ=0.6 | v2 Σのみ | 正解スキル上限 | 切替率 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        for repr_name in ("mean", "medoid"):
            values = []
            for r in runs:
                d = r["conditions"]["D3"][repr_name]
                values.append([r["skill_formation"]["n_skills"], r["skill_formation"]["coverage"],
                               d["normal"]["mse"], d["raw_interval"]["mse"], d["primitive_Pa"]["mse"],
                               d["skill_v2"]["mse"], d["fixed_tau"]["0.6"]["mse"],
                               d["skill_v2_sigma"]["mse"], d["forced_skill_recovery"]["oracle"]["confidence_tau"]["mse"],
                               r["skill_formation"]["switch_rate"]])
            m = np.mean(values, axis=0)
            lines.append(f"| {split}/{repr_name} | {m[0]:.1f} | {m[1]:.2f} | {m[2]:.4f} | {m[3]:.4f} | {m[4]:.4f} | {m[5]:.4f} | {m[6]:.4f} | {m[7]:.4f} | {m[8]:.4f} | {m[9]:.3f} |")
    lines += ["", "## 誤ったスキル選択からの回復（D3、平均表現）", "",
              "| Split | 強制選択 | 正解スキル率 | 信頼度τ | 固定τ=.2 | 固定τ=.6 | 固定τ=1.0 | F_normal |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        for force in ("oracle", "near", "random"):
            a = [r["conditions"]["D3"]["mean"]["forced_skill_recovery"][force] for r in runs]
            mse = lambda k: float(np.mean([x[k]["mse"] for x in a]))
            lines.append(f"| {split} | {force} | {np.mean([x['correct_skill_rate'] for x in a]):.2f} | {mse('confidence_tau'):.4f} | {np.mean([x['fixed_tau']['0.2']['mse'] for x in a]):.4f} | {np.mean([x['fixed_tau']['0.6']['mse'] for x in a]):.4f} | {np.mean([x['fixed_tau']['1.0']['mse'] for x in a]):.4f} | {np.mean([r['conditions']['D3']['mean']['normal']['mse'] for r in runs]):.4f} |")
    lines += ["", "## 解釈", "",
              "τと反復数は、保留エピソードで信頼度ビンごとに検証MSEが小さくなる組を選び、信頼度が高いほどτ・nが減るよう単調化した。v2アクションヘッドはスキル軌道アンカーの混合だけで学習し、F_normalは別モデルの比較対象。", "",
              "スキル形成・長さ・タスク横断性・動詞純度・D1/D2/D3全条件・信頼度ビン表はresults.json、行ごとのφ・τ・アンカー・補正はper_row.npz、軌道と補正図はfigures/を参照。", "",
              f"総実行時間: {payload['duration_minutes']:.1f}分。"]
    return "\n".join(lines) + "\n"


def _summary_v21(payload):
    lines = ["# スキル中心 v2.1：軽量版", "",
             f"実行: Split={payload['splits']}、seed={payload['seeds']}、stride={payload['stride_seconds']:.3f}s、H={HORIZON}。τ=1 を含む学習比率={payload['source_mix']}。", "",
             "## D3 の比較（v2 の結果と v2.1 のアンカー表現）", "",
             "| Split | F_normal | raw interval | P-a | v2 前回(τ) | v2.1 平均形 | v2.1 medoid | v2.1 状態依存 b | oracle skill | 選択 top1 前→後 | top5 | D1悪化率 mean/state |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        cols = []
        for r in runs:
            d = r["conditions"]["D3"]
            cols.append([d["mean"]["normal"]["mse"], d["mean"]["raw_interval"]["mse"],
                         d["mean"]["primitive_Pa"]["mse"], r.get("previous_v2_d3_mse", np.nan),
                         d["mean"]["skill_v2"]["mse"], d["medoid"]["skill_v2"]["mse"],
                         d["state_decode"]["skill_v2"]["mse"],
                         d["mean"]["forced_skill_recovery"]["oracle"]["confidence_tau"]["mse"],
                         d["mean"]["selection"]["previous_cosine_top1"],
                         d["mean"]["selection"]["multi_positive_top1"],
                         d["mean"]["selection"]["multi_positive_top5_recall"],
                         r["conditions"]["D1"]["mean"]["skill_v2"]["D1_worse_row_fraction"],
                         r["conditions"]["D1"]["state_decode"]["skill_v2"]["D1_worse_row_fraction"]])
        m = np.nanmean(np.asarray(cols, dtype=np.float64), axis=0)
        lines.append(f"| {split} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {m[3]:.4f} | {m[4]:.4f} | {m[5]:.4f} | {m[6]:.4f} | {m[7]:.4f} | {m[8]:.2f}→{m[9]:.2f} | {m[10]:.2f} | {m[11]:.2f}/{m[12]:.2f} |")
    lines += ["", "## τ=1 の復帰と誤選択からの回復（D3）", "",
              "| Split | 正解選択 | near誤選択 | random誤選択 | F_normal |", "|---|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        def avg(force):
            return float(np.mean([r["conditions"]["D3"]["mean"]["forced_skill_recovery"][force]["confidence_tau"]["mse"] for r in runs]))
        normal = float(np.mean([r["conditions"]["D3"]["mean"]["normal"]["mse"] for r in runs]))
        lines.append(f"| {split} | {avg('oracle'):.4f} | {avg('near'):.4f} | {avg('random'):.4f} | {normal:.4f} |")
    checks = [r["tau1_initialization_check"] for r in payload["runs"]]
    lines += ["", f"τ=1 の初期化確認: 全runの重みコピー一致={all(r['F_skill']['initial_state_equal'] for r in payload['runs'])}。ノイズ・条件も同じ設定で評価した。"]
    lines += [f"- {k}: 最大絶対差={max(x[k]['pretraining_clone_max_abs_difference'] for x in checks):.3g}、bitwise equal={all(x[k]['pretraining_clone_bitwise_equal'] for x in checks)}; 混合学習後MSE={np.mean([x[k]['after_mix_training_mse'] for x in checks]):.4f}、F_normal={np.mean([x[k]['normal_mse'] for x in checks]):.4f}" for k in ("D1", "D3")]
    lines += ["", "τ=1 混合学習後の相対差（D3、seed別）:"]
    for split in payload["splits"]:
        rs = [r for r in payload["runs"] if r["split"] == split]
        ratios = [100 * (r["tau1_initialization_check"]["D3"]["after_mix_training_mse"] /
                         r["tau1_initialization_check"]["D3"]["normal_mse"] - 1) for r in rs]
        lines.append(f"- {split}: 平均 {np.mean(ratios):+.1f}%、範囲 {min(ratios):+.1f}%〜{max(ratios):+.1f}%")
    lines += ["", "τ較正ビンごとの unconstrained/monotone τ・正解スキル率・MSE は results.json の `tau_calibration` に記録。学習の30%をアンカーなしに割り当て、初期重みは F_normal からコピーした。τ=1 の更新では F_normal の出力蒸留を加え、ノイズ経路を保つ。スキル選択器は学習suiteの開始区間だけで multi-positive InfoNCE を学習し、ε以内の機能的近傍を正例とした。top-5 は正解単位が候補集合に含まれる割合。", "",
              "スキル選択とτは区間開始で決め、状態依存アンカーはメンバーの M1 base を現在の状態で復号した。誤選択回復・全条件 D1/D2/D3・信頼度ビンは results.json、行ごとの予測は per_row.npz を参照。", "",
              f"総実行時間: {payload['duration_minutes']:.1f}分。"]
    return "\n".join(lines) + "\n"


def _summary_v22(payload):
    configs = ("k5_reselect", "k5_fixed", "k10_reselect", "k10_fixed")
    labels = ("k5 再選択", "k5 固定", "k10 再選択", "k10 固定")
    lines = ["# スキル中心 v2.2：デノイズ修正量による候補選択", "",
             f"実行: Split={payload['splits']}、seed={payload['seeds']}、候補τ={V22_TAU}、phase grid={V22_PHASE_GRID}。λは学習suiteの保留行で校正。", "",
             "## D3 MSE と選択正答率", "",
             "| Split | 通常 | 生区間 | P-a | v2.1 | 正解スキル | " + " | ".join(labels) + " | v2.1 row top1 | " + " | ".join(f"{x} row正解率" for x in labels) + " |",
             "|---|" + "---:|" * (6 + len(configs)*2)]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        def avg(path):
            vals = []
            for r in runs:
                v = r
                for key in path: v = v[key]
                vals.append(v)
            return float(np.mean(vals))
        values = [
            avg(("conditions","D3","mean","normal","mse")),
            avg(("conditions","D3","mean","raw_interval","mse")),
            avg(("conditions","D3","mean","primitive_Pa","mse")),
            avg(("conditions","D3","mean","skill_v2","mse")),
            avg(("conditions","D3","mean","forced_skill_recovery","oracle","confidence_tau","mse")),
        ]
        values += [avg(("conditions","D3","v22",key,"mse")) for key in configs]
        top1 = float(np.mean([r.get("previous_v2_d3_row_selection_accuracy",
                                    r["conditions"]["D3"]["mean"]["selection"]["multi_positive_top1"])
                              for r in runs]))
        acc = [avg(("conditions","D3","v22",key,"skill_accuracy")) for key in configs]
        cells = [f"{x:.4f}" for x in values] + [f"{top1:.2f}"] + [f"{x:.2f}" for x in acc]
        lines.append("| " + split + " | " + " | ".join(cells) + " |")
    lines += ["", "## D1 / D2 / D3 の MSE（通常・生区間・P-a・v2.1・v2.2 k5再選択・正解スキル）", "",
              "| Split | 条件 | 通常 | 生区間 | P-a | v2.1 | v2.2 k5再選択 | 正解スキル |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        for cond in ("D1", "D2", "D3"):
            def cond_avg(method_path):
                vals = []
                for r in runs:
                    value = r["conditions"][cond]
                    for key in method_path: value = value[key]
                    vals.append(value["mse"])
                return float(np.mean(vals))
            # Oracle selection metric is reported on D3; use the forced oracle on each condition for completeness.
            vals = [cond_avg(("mean","normal")), cond_avg(("mean","raw_interval")),
                    cond_avg(("mean","primitive_Pa")), cond_avg(("mean","skill_v2")),
                    float(np.mean([r["conditions"][cond]["v22"]["k5_reselect"]["mse"] for r in runs])),
                    float(np.mean([r["conditions"][cond]["mean"]["forced_skill_recovery"]["oracle"]["confidence_tau"]["mse"] for r in runs]))]
            lines.append("| " + split + " | " + cond + " | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    lines += ["", "## 何が修正量選択を制限したか（D3）", "",
              "| Split | k | 正解候補がtop-k内 | 正解候補が最小修正 | 正解候補修正RMS | 最良誤候補修正RMS | v2.2 k固定/再選択の正解率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        for k in (5, 10):
            diag = [r["conditions"]["D3"]["v22_candidate_diagnostics"][f"k{k}"] for r in runs]
            mean = lambda key: float(np.mean([x[key] for x in diag if x[key] is not None]))
            acc_fixed = float(np.mean([r["conditions"]["D3"]["v22"][f"k{k}_fixed"]["skill_accuracy"] for r in runs]))
            acc_reselect = float(np.mean([r["conditions"]["D3"]["v22"][f"k{k}_reselect"]["skill_accuracy"] for r in runs]))
            lines.append(f"| {split} | {k} | {mean('oracle_in_topk_rate'):.2f} | {mean('oracle_has_smallest_repair_rate_given_in_topk'):.2f} | {mean('oracle_candidate_repair_rms_mean'):.3f} | {mean('best_wrong_candidate_repair_rms_mean'):.3f} | {acc_fixed:.2f}/{acc_reselect:.2f} |")
    lines += ["", "D1 で通常より悪化した行の割合（v2.1平均形 / k5再選択 / k5固定 / k10再選択 / k10固定）:"]
    for split in payload["splits"]:
        runs = [r for r in payload["runs"] if r["split"] == split]
        v21 = np.mean([r["conditions"]["D1"]["mean"]["skill_v2"]["D1_worse_row_fraction"] for r in runs])
        v22 = [np.mean([r["conditions"]["D1"]["v22"][key]["D1_worse_row_fraction"] for r in runs])
               for key in configs]
        lines.append(f"- {split}: " + " / ".join([f"{v21:.2f}", *(f"{x:.2f}" for x in v22)]))
    lines += ["", "λごとの保留MSE・正解率、候補τと反復数、全runの詳細は results.json に保存。スキル候補の修正量は標準化行動次元上のRMS。最終行動MSEは8サンプル・同一ノイズで計測。", "",
              f"総実行時間: {payload['duration_minutes']:.1f}分。"]
    return "\n".join(lines) + "\n"


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _skill_formation_metrics(memory, segments, row_segment, fit, stride):
    member_lengths = [m["length_seconds"] for u in memory.units for m in u.members]
    durations = [u.median_length * stride for u in memory.units]
    task_counts = [len(u.tasks) for u in memory.units]
    purity = []
    for u in memory.units:
        if u.classes:
            purity.append(max(np.bincount(u.classes)) / len(u.classes))
    assignment = np.full(len(row_segment), -1, np.int64)
    for i, sid in enumerate(row_segment):
        if sid >= 0:
            assignment[i] = memory.assignments.get(int(sid), -1)
    switches = pairs = 0
    for ep in np.unique(fit["episode_id"].astype(str)):
        ix = np.flatnonzero(fit["episode_id"].astype(str) == ep)
        ix = ix[np.argsort(fit["frame"][ix], kind="stable")]
        vals = assignment[ix]
        for a, b in zip(vals[:-1], vals[1:]):
            if a >= 0 and b >= 0:
                pairs += 1; switches += int(a != b)
    return {"n_skills": len(memory.units), "n_segments": len(segments),
            "covered_segments": len(memory.assignments),
            "coverage": float(len(memory.assignments) / max(len(segments), 1)),
            "epsilon": memory.epsilon, "pending_expired": memory.expired,
            "member_segment_duration_seconds_quantiles": np.quantile(member_lengths, [0,.25,.5,.75,1]).tolist()
            if member_lengths else [],
            "skill_median_length_seconds_quantiles": np.quantile(durations, [0,.25,.5,.75,1]).tolist()
            if durations else [],
            "tasks_per_skill_quantiles": np.quantile(task_counts, [0,.25,.5,.75,1]).tolist()
            if task_counts else [],
            "verb_purity_mean_posthoc": float(np.mean(purity)) if purity else None,
            "switch_rate": float(switches / pairs) if pairs else None,
            "primitive_reference_switch_rate": 0.79,
            "cluster": memory.describe()}


def run_one(rows, *, split, seed, texts_by_source, text_embeddings, stride_seconds,
            device, smoke=False, m1_epochs=60, flow_epochs=80, batch_size=512,
            source_mix=DEFAULT_SOURCE_MIX, v21=False, v22=False):
    tic = time.perf_counter()
    train_tasks, eval_tasks, _ = SPLITS[split]
    train_all = _subset(rows, np.isin(rows["task_index"], train_tasks) & rows["valid_h25"])
    test = _subset(rows, np.isin(rows["task_index"], eval_tasks) & rows["valid_h25"])
    if smoke:
        train_all = af._select_episodes(train_all, train_tasks, 12, seed + 101)
        test = af._select_episodes(test, eval_tasks, 4, seed + 211)
    fit_rows, d1 = fc.holdout_split(train_all, split, seed, per_task=2 if smoke else fc.HOLDOUT_PER_TASK)
    fit_ix, val_ix = _episode_train_validation_indices(fit_rows, seed + 401)
    fit = _subset(fit_rows, np.isin(np.arange(len(fit_rows["a"])), fit_ix))
    val = _subset(fit_rows, np.isin(np.arange(len(fit_rows["a"])), val_ix))
    for x in (fit, val, d1, test):
        if "subtask_text" not in x:
            pass
    fc.assert_calibration_rows(val, split, set(map(str, d1["episode_id"])) | set(map(str, test["episode_id"])))
    m6, fit_bases, actions = af._fit_m6_backbone_and_units(
        fit, seed=seed + 19, device=device, epochs=m1_epochs)
    m1 = m6.backbone
    fit_text = fc.attach_text(fit, texts_by_source, text_embeddings, "instruction")
    e_model, e_fit, fit_q = af._fit_query_encoder(
        fit_text, fit_bases, seed=seed + 701, device=device,
        allowed_tasks=tuple(train_tasks), subtask_source="instruction")

    def embed(rows_part):
        text_rows = fc.attach_text(rows_part, texts_by_source, text_embeddings, "instruction")
        return af._query_embedding(e_model, e_fit, af._normal_query_rows(text_rows), device=device)

    query = {"fit": fit_q, "val": embed(val), "D1": embed(d1), "D2": embed(test), "D3": embed(test)}
    limits = derive_length_limits(fit)
    train_segments, fit_row_segment = make_segments(fit, fit_q, limits, stride_seconds,
                                                    bases=fit_bases if v21 else None)
    memory = TrajectorySkillMemory(KAPPA).fit(train_segments, seed=seed + 811)
    if not memory.units:
        raise RuntimeError("no supported skill clusters were formed from the training suite")
    formation = _skill_formation_metrics(memory, train_segments, fit_row_segment, fit, stride_seconds)
    val_bases = m6.encode(val["a"][:, :HORIZON], val["s"])[0] if v21 else None
    val_segments, _ = make_segments(val, query["val"], limits, stride_seconds,
                                    bases=val_bases)
    selector = None
    selector_metrics = None
    if v21:
        selector, selector_metrics = fit_multi_positive_selector(
            train_segments, memory, validation_segments=val_segments,
            seed=seed + 733, device=device)

    # F_normal exactly follows fair_compare's ordinary-only recipe.
    dim = int(np.prod(actions.shape[1:]))
    zero_source = np.zeros((len(actions), dim), np.float32)
    normal_sources = (zero_source, np.zeros(len(actions), np.float32),
                      zero_source.copy(), np.zeros(len(actions), np.float32))
    model_n, norm_n, fit_n = af.fit_anchor_flow(
        fit, actions, normal_sources, source_mix=fc.NORMAL_MIX, seed=seed + 827,
        epochs=flow_epochs, patience=8, batch_size=batch_size, device=device)

    # Fair_compare P-a comparator, using the same M6 primitive bank and F_anchor.
    global_scale = float(np.sqrt(np.mean(np.square(fit_bases - fit_bases.mean(0)))) )
    primitive_bank = af.UnitBank(m6, global_scale)
    primitive_bank.seed_training(fit_bases, fit_q, fit["episode_id"])
    flow_ix, _ = _episode_train_validation_indices(fit, seed + 827 + 11)
    am, ass = _standardize_fit(actions[flow_ix].reshape(len(flow_ix), -1))
    fm, fs = _standardize_fit(fit["f"][flow_ix]); sm, ss = _standardize_fit(fit["s"][flow_ix])
    flow_norm = {"a_mean": am, "a_scale": ass, "f_mean": fm, "f_scale": fs,
                 "s_mean": sm, "s_scale": ss, "action_shape": tuple(actions.shape[1:])}
    anchor_sources = af._make_flow_sources(m1, primitive_bank, fit, fit_bases, fit_q,
                                           e_fit["temperature"], flow_norm, seed=seed + 823)
    model_pa, norm_pa, fit_pa = af.fit_anchor_flow(
        fit, actions, anchor_sources, source_mix=fc.ANCHOR_MIX, seed=seed + 827,
        epochs=flow_epochs, patience=8, batch_size=batch_size, device=device)

    train_anchors, train_conf = _make_training_sources(
        fit, train_segments, memory, fit_q, actions.shape[1:], stride_seconds, seed + 853,
        selector=selector, device=device)
    effective_mix = _normalise_mix(source_mix)
    model_v2_init, norm_v2, fit_v2_init = _fit_skill_flow(
        fit, actions, train_anchors, train_conf, effective_mix,
        seed=seed + 907, epochs=flow_epochs, patience=8, batch_size=batch_size, device=device,
        initial_model=model_n if v21 else None, initial_norm=norm_n if v21 else None,
        teacher_model=model_n if v21 else None, teacher_weight=2.0 if v21 else 0.0)

    # Holdout suite episodes calibrate confidence→(tau,n). The score is the
    # same confidence feature used for skill selection and deliberately excludes
    # target-to-anchor distance.
    val_schedule = _schedule(val, query["val"], val_segments, memory, stride_seconds,
                             variant="mean", force="top1", seed=seed + 941,
                             selector=selector, m1=m1 if v21 else None, device=device)
    rng = np.random.default_rng(seed * 1_000_003 + 31)
    noises = {"val": rng.standard_normal((fc.SAMPLES, len(val["a"]), dim)).astype(np.float32),
              "D1": rng.standard_normal((fc.SAMPLES, len(d1["a"]), dim)).astype(np.float32),
              "D3": rng.standard_normal((fc.SAMPLES, len(test["a"]), dim)).astype(np.float32)}
    calibration = _calibrate_tau(model_v2_init, norm_v2, val, val_schedule,
                                 noises["val"], device=device)
    if v21:
        edges = np.asarray(calibration["edges"], np.float32)
        which = np.searchsorted(edges, val_schedule["confidence"], side="right")
        calibration["correct_skill_rate_by_bin"] = [
            float(np.mean(val_schedule["correct"][which == b])) if np.any(which == b) else None
            for b in range(len(calibration["tau"]))]

    # Fair_compare's P-a and raw controls calibrate noise time on validation
    # episodes, then hold that map fixed for D1/D2/D3. Keep the same paired
    # validation noise and per-row cosine confidence definition here.
    primitive_ids, primitive_centers, primitive_keys, _, _ = primitive_bank.centers_keys_sigmas()
    primitive_keys_n = primitive_keys / np.maximum(np.linalg.norm(primitive_keys, axis=1, keepdims=True), 1e-8)
    val_qn = query["val"] / np.maximum(np.linalg.norm(query["val"], axis=1, keepdims=True), 1e-8)
    val_pa_scores = val_qn @ primitive_keys_n.T
    val_pa_top = val_pa_scores.argmax(axis=1)
    val_pa_conf = val_pa_scores[np.arange(len(val_pa_top)), val_pa_top]
    val_pa_anchor = m1.decode(primitive_centers[val_pa_top],
                              np.zeros((len(val_pa_top), m1.residual_dim), np.float32),
                              val["f"], val["s"])
    train_qn = fit_q / np.maximum(np.linalg.norm(fit_q, axis=1, keepdims=True), 1e-8)
    val_raw_scores = val_qn @ train_qn.T
    val_raw_top = val_raw_scores.argmax(axis=1)
    val_raw_conf = val_raw_scores[np.arange(len(val_raw_top)), val_raw_top]
    val_raw_anchor = actions[val_raw_top]

    def fit_anchor_confidence_calibration(model, norm, anchor, confidence):
        row_errors = []
        val_target = val["a"][:, :HORIZON]
        for t0 in fc.T0_GRID:
            pred = fc.rollout(model, norm, val["f"], val["s"], anchor,
                              noises["val"], np.full(len(val_target), t0, np.float32),
                              n=fc.steps_for(np.full(len(val_target), t0)), device=device)
            row_errors.append(np.square(pred - val_target[None]).mean(axis=(0, 2, 3)))
        return fc.fit_monotone_g(confidence, np.stack(row_errors), grid=fc.T0_GRID)

    pa_calibration = fit_anchor_confidence_calibration(model_pa, norm_pa, val_pa_anchor, val_pa_conf)
    raw_calibration = fit_anchor_confidence_calibration(model_n, norm_n, val_raw_anchor, val_raw_conf)

    def calibrated(schedule):
        return apply_confidence_map(calibration, schedule["confidence"])

    # One permitted training/calibration round: refit F_skill with the calibrated
    # confidence-derived tau instead of the initial monotone tau=clip(1-c).
    model_v2, norm_v2_final, fit_v2 = _fit_skill_flow(
        fit, actions, train_anchors, train_conf, _normalise_mix(source_mix),
        seed=seed + 907, epochs=flow_epochs, patience=8, batch_size=batch_size,
        device=device, tau_calibration=calibration,
        initial_model=model_n if v21 else None, initial_norm=norm_n if v21 else None,
        teacher_model=model_n if v21 else None, teacher_weight=2.0 if v21 else 0.0)
    norm_v2 = norm_v2_final
    init_tau1_check = None
    if v21:
        check_rows = {"D1": d1, "D3": test}
        check_noise = {"D1": noises["D1"], "D3": noises["D3"]}
        init_tau1_check = {}
        for name, check in check_rows.items():
            p_normal = fc.rollout(model_n, norm_n, check["f"], check["s"], None,
                                  check_noise[name], np.ones(len(check["a"]), np.float32),
                                  n=np.full(len(check["a"]), 10, np.int64), device=device)
            p_skill = fc.rollout(model_v2_init, norm_v2, check["f"], check["s"], None,
                                 check_noise[name], np.ones(len(check["a"]), np.float32),
                                 n=np.full(len(check["a"]), 10, np.int64), device=device)
            clone = copy.deepcopy(model_n).eval()
            p_clone = fc.rollout(clone, norm_n, check["f"], check["s"], None,
                                 check_noise[name], np.ones(len(check["a"]), np.float32),
                                 n=np.full(len(check["a"]), 10, np.int64), device=device)
            init_tau1_check[name] = {"normal_mse": _metrics(p_normal, check["a"][:, :HORIZON])["mse"],
                                     "after_mix_training_mse": _metrics(p_skill, check["a"][:, :HORIZON])["mse"],
                                     "pretraining_clone_max_abs_difference": float(np.max(np.abs(p_normal-p_clone))),
                                     "pretraining_clone_bitwise_equal": bool(np.array_equal(p_normal, p_clone))}
    # Variance-only policy uses the per-phase ensemble spread seen in training.
    sigma_reference = np.concatenate([u.variance.mean(axis=1) for u in memory.units])
    v22_lambda_calibration = None
    if v22:
        v22_lambda_calibration = _calibrate_v22_lambdas(
            val, query["val"], val_segments, memory, model_v2, norm_v2,
            noises["val"], selector, device, stride_seconds)

    # Inference helpers. D1 has no episode overlap by construction. D2 and D3
    # share the exact held-out rows/noise; only D2 inserts after each episode.
    ids, centers, keys, _, _ = primitive_bank.centers_keys_sigmas()
    keys_n = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    raw_kn = fit_q / np.maximum(np.linalg.norm(fit_q, axis=1, keepdims=True), 1e-8)
    test_b, _ = m6.encode(test["a"][:, :HORIZON], test["s"])
    d2_primitive, _ = fc.stream_d2(
        m6, {"instruction": primitive_bank}, test, test_b,
        {"instruction": query["D2"]}, {"instruction": fit_q}, actions,
        {"instruction": float(e_fit["temperature"])},
        initial_ids=np.arange(len(m6.prototypes)), seed=seed, device=device)
    d2_primitive = d2_primitive["instruction"]

    def primitive_anchor(rows_part, q, streaming):
        if streaming:
            conf = d2_primitive["conf_sim"]
            out = m1.decode(d2_primitive["unit_b"],
                            np.zeros((len(rows_part["a"]), m1.residual_dim), np.float32),
                            rows_part["f"], rows_part["s"])
            return out, conf
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        scores = qn @ keys_n.T
        ix = scores.argmax(axis=1)
        return (m1.decode(centers[ix], np.zeros((len(ix), m1.residual_dim), np.float32),
                           rows_part["f"], rows_part["s"]), scores[np.arange(len(ix)), ix])

    train_actions = fit["a"][:, :HORIZON]
    def raw_anchor(rows_part, q, streaming):
        if streaming:
            return d2_primitive["raw_action"], d2_primitive["raw_conf_sim"]
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        scores = qn @ raw_kn.T
        ix = scores.argmax(axis=1)
        return train_actions[ix], scores[np.arange(len(ix)), ix]

    val_outputs = {}
    # Calibration diagnostics compare initial and calibrated flow on the same holdout.
    val_outputs["tau_calibration"] = calibration
    row_details = {}
    conditions = {"D1": (d1, query["D1"], noises["D1"], False),
                  "D2": (test, query["D2"], noises["D3"], True),
                  "D3": (test, query["D3"], noises["D3"], False)}
    all_outputs = {}
    for condition, (eval_rows, eval_q, eval_eps, stream) in conditions.items():
        eval_bases = m6.encode(eval_rows["a"][:, :HORIZON], eval_rows["s"])[0] if v21 else None
        eval_segments, _ = make_segments(eval_rows, eval_q, limits, stride_seconds,
                                         bases=eval_bases)
        # Mean and medoid are explicit alternatives. Baselines are identical and
        # use the same noise tensor for paired comparison.
        all_outputs[condition] = {}
        representations = ("mean", "medoid", "state_decode") if v21 else ("mean", "medoid")
        for representation in representations:
            output, detail = _run_condition(
                model_n, norm_n, model_pa, norm_pa, model_v2, norm_v2,
                eval_rows, eval_segments, eval_q, memory, stride_seconds, eval_eps,
                calibrated, sigma_reference, pa_calibration, raw_calibration,
                lambda r, q, stream: primitive_anchor(r, q, stream),
                lambda r, q, stream: raw_anchor(r, q, stream),
                split=split, seed=seed, device=device, do_stream=stream,
                representation=representation, selector=selector,
                m1=m1 if v21 else None)
            all_outputs[condition][representation] = output
            output["selection"] = selection_metrics(eval_q, eval_segments, memory,
                                                    selector if v21 else None, device)
            row_details[f"{condition}_{representation}"] = detail
            if v22 and representation == "mean":
                v22_output, v22_diagnostics = _v22_evaluate_condition(
                    eval_rows, eval_q, eval_segments, memory, model_v2, norm_v2,
                    eval_eps, v22_lambda_calibration, selector=selector, device=device,
                    stride_seconds=stride_seconds, do_stream=stream, seed=seed,
                    normal_prediction=detail["normal_prediction"])
                all_outputs[condition]["v22"] = v22_output
                all_outputs[condition]["v22_candidate_diagnostics"] = v22_diagnostics

    # Post-hoc task/verb interpretation; neither field is used in clustering.
    segment_lengths = [s["length_seconds"] for s in train_segments]
    result = {"split": split, "seed": seed, "smoke": smoke,
              "rows": {"fit": len(fit["a"]), "validation": len(val["a"]),
                       "D1": len(d1["a"]), "D2": len(test["a"]), "D3": len(test["a"])},
              "limits": {"minimum_primitives": limits["minimum"], "maximum_primitives": limits["maximum"],
                         "quantiles": [0.10, 0.90]},
              "skill_formation": formation,
              "segment_length_seconds_quantiles": np.quantile(segment_lengths, [0,.25,.5,.75,1]).tolist(),
              "F_normal": {"best_epoch": fit_n["best_epoch"],
                           "validation_loss": fit_n["best_validation_loss"]},
              "F_primitive_anchor": {"best_epoch": fit_pa["best_epoch"],
                                     "validation_loss": fit_pa["best_validation_loss"]},
              "F_skill": {"initial_best_epoch": fit_v2_init["best_epoch"],
                          "best_epoch": fit_v2["best_epoch"],
                          "validation_loss": fit_v2["best_validation_loss"],
                          "source_mix": _normalise_mix(source_mix).tolist(),
                          "initialized_from": fit_v2["initialized_from"],
                          "initial_state_equal": fit_v2_init["initial_state_equal"],
                          "training_history": fit_v2["history"]},
              "selector": selector_metrics,
              "tau1_initialization_check": init_tau1_check,
              "tau_calibration": calibration,
              "v22_lambda_calibration": v22_lambda_calibration,
              "P_a_calibration": pa_calibration,
              "raw_interval_calibration": raw_calibration,
              "conditions": all_outputs,
              "duration_seconds": time.perf_counter() - tic}
    return result, row_details, memory


def _figures(out, runs, memories, per_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    # Representative learned trajectories with per-phase standard-deviation bands.
    mem = memories[0]
    chosen = sorted(mem.units, key=lambda u: u.support, reverse=True)[:4]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(8, max(3, 2.4 * len(chosen))), squeeze=False)
    phase = np.linspace(0, 1, CURVE_POINTS)
    for ax, unit in zip(axes[:, 0], chosen):
        ax.plot(phase, unit.mean[:, :3], linewidth=1.3)
        std = np.sqrt(np.maximum(unit.variance[:, :3], 0))
        for d in range(3):
            ax.fill_between(phase, unit.mean[:, d] - std[:, d], unit.mean[:, d] + std[:, d], alpha=.12)
        ax.set_title(f"skill {unit.skill_id}, support={unit.support}, seconds={unit.median_length:.1f}")
        ax.set_ylabel("relative command")
    axes[-1, 0].set_xlabel("phase φ")
    fig.tight_layout(); fig.savefig(figdir / "skill_trajectories.png", dpi=140); plt.close(fig)
    # One representative query: anchor, denoised output, and target by xyz/gripper.
    detail = next(iter(per_rows.values()))
    row = 0
    pred = detail["final_prediction"][row]
    target = detail["target"][row]
    anchor = detail["anchor"][row]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(anchor[:, :3], linestyle="--", alpha=.8, label=["anchor x", "anchor y", "anchor z"][0])
    for d, name in enumerate(("x", "y", "z")):
        axes[0].plot(pred[:, d], label=f"final {name}")
        axes[0].plot(target[:, d], linestyle=":", label=f"target {name}")
    axes[0].set_ylabel("relative position command"); axes[0].legend(ncol=3, fontsize=7)
    axes[1].plot(anchor[:, 6], "--", label="anchor grip")
    axes[1].plot(pred[:, 6], label="final grip"); axes[1].plot(target[:, 6], ":", label="target grip")
    axes[1].set_xlabel("action step"); axes[1].set_ylabel("gripper"); axes[1].legend()
    fig.tight_layout(); fig.savefig(figdir / "denoising_correction.png", dpi=140); plt.close(fig)
    return ["figures/skill_trajectories.png", "figures/denoising_correction.png"]


def run(*, output_root=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'), cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seeds=(0,1,2), splits=("A","B"), device_name="cuda",
        smoke=False, m1_epochs=60, flow_epochs=80, batch_size=512, threads=8,
        source_mix=None, v21=False, v22=False):
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device = torch.device(device_name)
    v21 = bool(v21 or v22)
    source_mix = V21_SOURCE_MIX if source_mix is None and v21 else (DEFAULT_SOURCE_MIX if source_mix is None else source_mix)
    if source_mix is None:
        source_mix = DEFAULT_SOURCE_MIX
    stamp = datetime.now(timezone.utc).strftime(
        "skill-v22-%Y%m%dT%H%M%SZ" if v22 else ("skill-v21-%Y%m%dT%H%M%SZ" if v21 else "skill-v2-%Y%m%dT%H%M%SZ"))
    if smoke: stamp += "-smoke"
    out = Path(output_root) / f"{stamp}-{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir()
    start = time.perf_counter()
    rows = load_rows(Path(cache_dir), Path(dataset_dir), max_horizon=HORIZON)
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    info = json.loads((Path(dataset_dir) / "meta" / "info.json").read_text())
    stride_seconds = float(manifest["stride"]) / float(info["fps"])
    task_ids = sorted({t for split in splits for t in (*SPLITS[split][0], *SPLITS[split][1])})
    rows = _subset(rows, np.isin(rows["task_index"], task_ids) & rows["valid_h25"])
    texts, text_emb = fc.precompute_texts(rows, ("instruction",), Path(cache_dir))
    runs, detail_rows, memories = [], {}, []
    for split in splits:
        for seed in seeds:
            result, details, memory = run_one(
                rows, split=split, seed=int(seed), texts_by_source=texts,
                text_embeddings=text_emb, stride_seconds=stride_seconds,
                device=device, smoke=smoke, m1_epochs=m1_epochs,
                flow_epochs=flow_epochs, batch_size=batch_size, source_mix=source_mix,
                v21=v21, v22=v22)
            runs.append(result); memories.append(memory)
            for name, detail in details.items():
                detail_rows[f"{split}_seed{seed}_{name}"] = detail
            partial = {"runs": runs, "seeds": list(seeds), "splits": list(splits), "smoke": smoke}
            (out / "results.partial.json").write_text(json.dumps(partial, ensure_ascii=False,
                                                                  default=_json_default))
    # Save row-level skill/phase/tau/anchor/correction audit data separately.
    per_row_path = out / "per_row.npz"
    flattened = {}
    for prefix, detail in detail_rows.items():
        for key, value in detail.items():
            flattened[f"{prefix}__{key}"] = np.asarray(value)
    np.savez_compressed(per_row_path, **flattened)
    figures = _figures(out, runs, memories, detail_rows)
    previous = {}
    if v21:
        old_path = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/skill-v2-20261029T050130Z-daab6b/results.json'))
        old_rows_path = old_path.parent / "per_row.npz"
        if old_path.exists():
            old = json.loads(old_path.read_text())
            previous = {(r["split"], int(r["seed"])):
                        r["conditions"]["D3"]["mean"]["skill_v2"]["mse"]
                        for r in old["runs"]}
            previous_row_accuracy = {}
            if old_rows_path.exists():
                with np.load(old_rows_path, allow_pickle=True) as old_rows:
                    for r in old["runs"]:
                        key = f"{r['split']}_seed{int(r['seed'])}_D3_mean__correct_skill"
                        if key in old_rows:
                            previous_row_accuracy[(r["split"], int(r["seed"]))] = float(old_rows[key].mean())
            for r in runs:
                r["previous_v2_d3_mse"] = previous.get((r["split"], int(r["seed"])), float("nan"))
                r["previous_v2_d3_row_selection_accuracy"] = previous_row_accuracy.get(
                    (r["split"], int(r["seed"])), float("nan"))
    payload = {"spec": "/work/docs/research/architecture_v2.md §0, §2, §4",
               "seeds": list(seeds), "splits": list(splits), "smoke": bool(smoke),
               "subtask_source": "instruction", "stride_seconds": stride_seconds,
               "cache_stride": int(manifest["stride"]), "dataset_fps": int(info["fps"]),
               "horizon": HORIZON, "curve_points": CURVE_POINTS,
               "representations": ["mean", "medoid", "state_decode"] if v21 else ["mean", "medoid"],
               "source_names": list(SOURCE_NAMES),
               "source_mix": _normalise_mix(source_mix).tolist(),
               "v21": bool(v21),
               "v22": bool(v22),
               "noise_samples": fc.SAMPLES, "runs": runs,
               "duration_minutes": (time.perf_counter() - start) / 60,
               "output_dir": str(out), "per_row_path": str(per_row_path),
               "figures": figures, "results_path": str(out / "results.json")}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                 default=_json_default))
    summary = _summary_v22(payload) if v22 else (_summary_v21(payload) if v21 else _summary(payload))
    (out / "summary.md").write_text(summary)
    (out / "results.partial.json").unlink(missing_ok=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot')))
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--splits", default="A,B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--m1-epochs", type=int, default=60)
    p.add_argument("--flow-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--v21", action="store_true")
    p.add_argument("--v22", action="store_true")
    p.add_argument("--source-mix", default=None)
    a = p.parse_args(argv)
    out = run(output_root=a.output_root, cache_dir=a.cache_dir, dataset_dir=a.dataset,
              seeds=tuple(int(x) for x in a.seeds.split(",")),
              splits=tuple(a.splits.split(",")), device_name=a.device,
              smoke=a.smoke, m1_epochs=a.m1_epochs, flow_epochs=a.flow_epochs,
              batch_size=a.batch_size, threads=a.threads,
              source_mix=tuple(float(x) for x in a.source_mix.split(",")) if a.source_mix else None,
              v21=a.v21, v22=a.v22)
    print(f"output: {out}")


if __name__ == "__main__":
    main()
