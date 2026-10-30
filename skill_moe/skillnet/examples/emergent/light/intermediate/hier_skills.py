"""S3: group short M6 motion primitives into open-ended higher-level skills.

The algorithm is deliberately label-free. Task and verb labels are attached
only after training for interpretation tables. See anchor_flow_skill_memory.md
§S3. This module imports the frozen M1/M6, query encoder and flow machinery
from the existing lightweight experiments; it does not edit those modules.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from . import anchor_flow as af
from . import fair_compare as fc
from .splits import SPLITS, subset_rows as _subset
from .experiment import (DEFAULT_CACHE, DEFAULT_DATASET, _episode_train_validation_indices,
                         _standardize_fit, load_rows)

HORIZON = fc.HORIZON
K_POINTS = 4
SKILL_KAPPA = 0.1
MIN_SEGMENTS = 10
MIN_EPISODES = 3
PENDING_BATCH = 200
MIN_PRIMITIVES_PER_SEGMENT = 2
METHODS = ("normal", "primitive_anchor", "primitive_normal", "hier_skill",
           "raw_interval", "shuffle")


def _resample(values, count=K_POINTS):
    """Linear time-normalized resampling on axis 0."""
    x = np.asarray(values, np.float32)
    if x.ndim < 2 or len(x) == 0:
        raise ValueError("a non-empty sequence is required")
    if len(x) == 1:
        return np.repeat(x, count, axis=0)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, count)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(count, *x.shape[1:]).astype(np.float32)


def _merge_short_segments(bounds, min_length=MIN_PRIMITIVES_PER_SEGMENT):
    """Merge short adjacent runs until every returned half-open span is long enough."""
    bounds = [(int(a), int(b)) for a, b in bounds if b > a]
    if not bounds:
        return []
    changed = True
    while changed and len(bounds) > 1:
        changed = False
        for i, (a, b) in enumerate(bounds):
            if b - a >= min_length:
                continue
            if i == 0:
                bounds[1] = (a, bounds[1][1])
                bounds.pop(0)
            else:
                bounds[i - 1] = (bounds[i - 1][0], b)
                bounds.pop(i)
            changed = True
            break
    return bounds


def make_boundaries(n, *, mode, grip_switch=None, functional_distance=None,
                    functional_threshold=None, min_length=MIN_PRIMITIVES_PER_SEGMENT):
    """Build episode spans for gripper, functional change, or their union."""
    if mode not in ("gripper", "functional", "both"):
        raise ValueError("mode must be gripper, functional, or both")
    cut = np.zeros(max(0, n - 1), dtype=bool)
    if mode in ("gripper", "both"):
        if grip_switch is None or len(grip_switch) != len(cut):
            raise ValueError("grip_switch must have n-1 entries")
        cut |= np.asarray(grip_switch, bool)
    if mode in ("functional", "both"):
        if functional_distance is None or len(functional_distance) != len(cut):
            raise ValueError("functional_distance must have n-1 entries")
        if functional_threshold is None:
            raise ValueError("functional_threshold is required")
        cut |= np.asarray(functional_distance) > float(functional_threshold)
    starts = [0, *(int(i + 1) for i in np.flatnonzero(cut)), n]
    return _merge_short_segments(list(zip(starts[:-1], starts[1:])), min_length)


def _base_interp(bases, count=K_POINTS):
    return _resample(bases, count)


def segment_representation(bases, stride_seconds):
    """Fixed-size b trajectory plus log duration, as prescribed by S3."""
    b = _base_interp(bases)
    length_seconds = max(float(len(bases)) * float(stride_seconds), 1e-6)
    length_feature = np.asarray([np.log(length_seconds)], np.float32)
    return b, length_feature


def _adjacent_grip_switches(actions):
    a = np.asarray(actions, np.float32)
    if len(a) < 2:
        return np.zeros(0, bool)
    # Each row is a fair_compare stride window. Use the first five action steps
    # to identify its commanded gripper state, then mark state transitions.
    state = np.sign(np.mean(a[:, :min(5, a.shape[1]), 6], axis=1))
    return (state[1:] != state[:-1]) & (state[1:] != 0) & (state[:-1] != 0)


def adjacent_functional_distances(m1, rows, bases, *, chunk=256):
    """MSE of adjacent interval decodes under the *same* row context.

    The left interval's proprioception, language-free image feature and zero
    residual are held fixed while both b codes are decoded. This makes the
    distance a learned action-function difference rather than a scene change.
    """
    out = np.zeros(max(0, len(bases) - 1), np.float32)
    if len(out) == 0:
        return out
    left, right = np.asarray(bases[:-1], np.float32), np.asarray(bases[1:], np.float32)
    residual = np.zeros((len(left), m1.residual_dim), np.float32)
    values = []
    for lo in range(0, len(left), chunk):
        hi = min(len(left), lo + chunk)
        state, feature = rows["s"][lo:hi], rows["f"][lo:hi]
        dl = m1.decode(left[lo:hi], residual[lo:hi], feature, state)
        dr = m1.decode(right[lo:hi], residual[lo:hi], feature, state)
        values.append(np.mean(np.square(dl - dr), axis=(1, 2)))
    out[:] = np.concatenate(values)
    return out


def _make_episode_segments(rows, bases, keys, *, mode, stride_seconds,
                           functional_threshold, functional_m1=None,
                           primitive_ids=None):
    """Create row-aligned primitive runs; no task or verb label is consulted."""
    episodes = np.asarray(rows["episode_id"]).astype(str)
    output, row_segment = [], np.full(len(episodes), -1, np.int64)
    # Input rows must be in frame order within each episode.
    for episode in np.unique(episodes):
        ix = np.flatnonzero(episodes == episode)
        ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
        local = _subset(rows, np.isin(np.arange(len(episodes)), ix))
        local_ix = np.asarray(ix, np.int64)
        local_b = np.asarray(bases[local_ix], np.float32)
        local_k = np.asarray(keys[local_ix], np.float32)
        local_primitive = (np.full(len(local_ix), -1, np.int64) if primitive_ids is None
                           else np.asarray(primitive_ids, np.int64)[local_ix])
        if len(local_ix) < MIN_PRIMITIVES_PER_SEGMENT:
            continue
        grip = _adjacent_grip_switches(local["a"])
        if functional_m1 is None:
            fd = np.zeros(len(local_ix) - 1, np.float32)
        else:
            fd = adjacent_functional_distances(functional_m1, local, local_b)
        spans = make_boundaries(len(local_ix), mode=mode, grip_switch=grip,
                                functional_distance=fd,
                                functional_threshold=functional_threshold)
        for a, b in spans:
            selected = local_ix[a:b]
            rep, length_feature = segment_representation(local_b[a:b], stride_seconds)
            seg = {"episode_id": str(episode), "rows": selected,
                   "bases": local_b[a:b].copy(), "keys": local_k[a:b].copy(),
                   "primitive_ids": local_primitive[a:b].copy(),
                   "base_curve": rep, "length_feature": length_feature,
                   "key": local_k[a:b].mean(0).astype(np.float32),
                   "length_primitives": int(b - a),
                   "length_seconds": float((b - a) * stride_seconds),
                   "task_index": int(local["task_index"][a]),
                   "skill_class": int(local["skill_class"][a]) if "skill_class" in local else -1}
            seg["local_segment"] = len(output)
            output.append(seg)
            row_segment[selected] = seg["local_segment"]
    return output, row_segment


def _decoded_curve_embedding(m1, curves, probe_rows, *, chunk=32):
    """Decode each curve point on common states and return action-function features."""
    curves = np.asarray(curves, np.float32)
    if curves.ndim == 2:
        curves = curves[None]
    probe_f = np.asarray(probe_rows["f"], np.float32)
    probe_s = np.asarray(probe_rows["s"], np.float32)
    prior_r = np.asarray(m1.prior(probe_f, probe_s)[1], np.float32)
    n, k, d = curves.shape
    out = np.empty((n, k, len(probe_f), int(np.prod(m1.action_shape))), np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        c = curves[lo:hi]
        repeated_b = np.repeat(c[:, :, None, :], len(probe_f), axis=2).reshape(-1, d)
        repeated_r = np.tile(prior_r, ((hi - lo) * k, 1))
        repeated_f = np.tile(probe_f, ((hi - lo) * k, 1))
        repeated_s = np.tile(probe_s, ((hi - lo) * k, 1))
        dec = m1.decode(repeated_b, repeated_r, repeated_f, repeated_s)
        out[lo:hi] = dec.reshape(hi - lo, k, len(probe_f), -1)
    return out.reshape(n, -1) / np.sqrt(float(k * len(probe_f) * np.prod(m1.action_shape)))


@dataclass
class SkillUnit:
    skill_id: int
    base_curve: np.ndarray
    key: np.ndarray
    functional_center: np.ndarray
    members: list[dict] = field(default_factory=list)
    support: int = 0
    episodes: set[str] = field(default_factory=set)


class HierSkillMemory:
    """M6-style unbounded, gradient-free clustering over segment functions."""
    def __init__(self, epsilon, *, min_segments=MIN_SEGMENTS, min_episodes=MIN_EPISODES,
                 pending_batch=PENDING_BATCH, max_pending_age=1000):
        self.epsilon = max(float(epsilon), 1e-12)
        self.min_segments = int(min_segments)
        self.min_episodes = int(min_episodes)
        self.pending_batch = int(pending_batch)
        self.max_pending_age = int(max_pending_age)
        self.units: list[SkillUnit] = []
        self.pending: list[tuple[dict, np.ndarray, int]] = []
        self.next_id = 0
        self.seen = 0
        self.registered = 0
        self.merged = 0
        self.expired = 0
        self.history = []
        self.assignments = {}

    def _nearest(self, functional):
        if not self.units:
            return None, float("inf")
        distances = np.asarray([np.mean(np.square(functional - u.functional_center), dtype=np.float64)
                                for u in self.units])
        i = int(np.argmin(distances))
        return self.units[i], float(distances[i])

    def _add_member(self, unit, segment, functional):
        n = unit.support
        unit.base_curve = ((unit.base_curve * n + segment["base_curve"]) / (n + 1)).astype(np.float32)
        unit.key = ((unit.key * n + segment["key"]) / (n + 1)).astype(np.float32)
        unit.functional_center = ((unit.functional_center * n + functional) / (n + 1)).astype(np.float32)
        unit.support += 1
        unit.episodes.add(str(segment["episode_id"]))
        unit.members.append({"segment": segment, "functional": functional})
        self.assignments[segment["local_segment"]] = unit.skill_id

    def add(self, segment, functional, *, allow_pending=True):
        functional = np.asarray(functional, np.float32)
        unit, distance = self._nearest(functional)
        if unit is not None and distance <= self.epsilon:
            self._add_member(unit, segment, functional)
            self.seen += 1
            if allow_pending and len(self.pending) >= self.pending_batch:
                self.flush(force=False)
            self._record()
            return unit.skill_id, "assigned"
        self.pending.append((segment, functional.copy(), self.seen))
        self.seen += 1
        if allow_pending and len(self.pending) >= self.pending_batch:
            self.flush(force=False)
        self._expire()
        self._record()
        return None, "pending"

    def _expire(self):
        kept = []
        for seg, emb, age in self.pending:
            if self.seen - age >= self.max_pending_age:
                self.expired += 1
            else:
                kept.append((seg, emb, age))
        self.pending = kept

    def flush(self, *, force):
        pending, self.pending = self.pending, []
        if not pending:
            return
        # DP-means over the fixed-size functional signatures, no fixed K.
        groups = []
        for segment, functional, age in pending:
            if not groups:
                groups.append([segment, functional.copy(), 1, {str(segment["episode_id"])}, [(segment, functional)]])
                continue
            dist = [np.mean(np.square(functional - g[1]), dtype=np.float64) for g in groups]
            j = int(np.argmin(dist))
            if dist[j] <= self.epsilon:
                g = groups[j]
                g[1] = ((g[1] * g[2] + functional) / (g[2] + 1)).astype(np.float32)
                g[2] += 1
                g[3].add(str(segment["episode_id"]))
                g[4].append((segment, functional))
            else:
                groups.append([segment, functional.copy(), 1, {str(segment["episode_id"])}, [(segment, functional)]])
        leftovers = []
        for first, center_func, count, episodes, members in groups:
            if count < self.min_segments or len(episodes) < self.min_episodes:
                leftovers.extend(members)
                continue
            curve = np.mean(np.stack([m[0]["base_curve"] for m in members]), axis=0).astype(np.float32)
            key = np.mean(np.stack([m[0]["key"] for m in members]), axis=0).astype(np.float32)
            old, distance = self._nearest(center_func)
            if old is not None and distance <= self.epsilon:
                for segment, functional in members:
                    self._add_member(old, segment, functional)
                self.merged += 1
            else:
                unit = SkillUnit(self.next_id, curve, key, np.asarray(center_func, np.float32), [], 0, set())
                self.next_id += 1
                self.units.append(unit)
                for segment, functional in members:
                    self._add_member(unit, segment, functional)
                self.registered += 1
        if not force:
            self.pending = [(seg, emb, self.seen) for seg, emb in leftovers]
        else:
            self.expired += len(leftovers)
        self._record()

    def fit(self, segments, functional, *, seed):
        order = np.random.default_rng(seed).permutation(len(segments))
        for i in order:
            self.add(segments[int(i)], functional[int(i)])
        self.flush(force=True)
        return self

    def _record(self):
        self.history.append({"segments_seen": int(self.seen), "skills": len(self.units),
                             "pending": len(self.pending), "registered": self.registered,
                             "merged": self.merged, "expired": self.expired})

    def top1(self, query):
        if not self.units:
            raise ValueError("no skill units have been registered")
        q = np.asarray(query, np.float32).reshape(-1)
        keys = np.stack([u.key for u in self.units])
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
        sims = keys @ q
        j = int(np.argmax(sims))
        return self.units[j], float(sims[j])

    def position_base(self, unit, query):
        """Choose the closest stored primitive key within this skill."""
        q = np.asarray(query, np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-8)
        candidates = []
        for member in unit.members:
            seg = member["segment"]
            keys = seg["keys"]
            kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
            sim = kn @ q
            ix = int(np.argmax(sim))
            candidates.append((float(sim[ix]), seg["bases"][ix]))
        if not candidates:
            return unit.base_curve[0], -1.0
        score, base = max(candidates, key=lambda x: x[0])
        return np.asarray(base, np.float32), score

    def describe(self):
        return {"epsilon": self.epsilon, "threshold_source": "q=0.1 times median pairwise training-segment functional distance",
                "min_segments": self.min_segments, "min_episodes": self.min_episodes,
                "pending_batch": self.pending_batch, "n_skills": len(self.units),
                "registered_groups": self.registered, "merged_groups": self.merged,
                "pending_expired": self.expired, "pending_final": len(self.pending),
                "unit_count_history": self.history,
                "covered_segments": int(sum(u.support for u in self.units)),
                "coverage": float(sum(u.support for u in self.units) /
                                   max(sum(u.support for u in self.units) + len(self.pending) + self.expired, 1)),
                "support": [u.support for u in self.units],
                "episode_count": [len(u.episodes) for u in self.units]}


def _functional_threshold(m1, train_rows, bases):
    distances = []
    for ep in np.unique(np.asarray(train_rows["episode_id"]).astype(str)):
        ix = np.flatnonzero(np.asarray(train_rows["episode_id"]).astype(str) == ep)
        ix = ix[np.argsort(train_rows["frame"][ix], kind="stable")]
        if len(ix) > 1:
            local = _subset(train_rows, np.isin(np.arange(len(bases)), ix))
            distances.extend(adjacent_functional_distances(m1, local, bases[ix]).tolist())
    if not distances:
        return 0.0, np.asarray([], np.float32)
    values = np.asarray(distances, np.float32)
    return float(np.quantile(values, 0.9)), values


def _primitive_unit_ids(m6, bases):
    """Assign each row to its nearest learned M6 primitive (including pending-only rows)."""
    out = np.full(len(bases), -1, np.int64)
    for i, base in enumerate(np.asarray(bases, np.float32)):
        uid, _ = m6._nearest(base)
        if uid is not None:
            out[i] = int(uid)
    return out


def _segment_memory(segments, m1, probe_rows, seed):
    if not segments:
        raise ValueError("training split produced no segments")
    curves = np.stack([s["base_curve"] for s in segments])
    functional = _decoded_curve_embedding(m1, curves, probe_rows)
    rng = np.random.default_rng(seed + 5107)
    n = len(functional)
    pairs = rng.integers(0, n, size=(min(64, max(1, n * 2)), 2))
    random_dist = np.mean(np.square(functional[pairs[:, 0]] - functional[pairs[:, 1]]), axis=1)
    epsilon = max(SKILL_KAPPA * float(np.median(random_dist)), 1e-12)
    memory = HierSkillMemory(epsilon).fit(segments, functional, seed=seed + 6021)
    return memory, functional, random_dist


def _retrieve(method, rows, q, m6, primitive_bank, skill_mem, train_keys, train_actions,
              train_bases, *, seed, split, shuffle=False):
    n = len(q)
    anchors = np.empty((n, HORIZON, 7), np.float32)
    confidence = np.empty(n, np.float32)
    ids, centers, keys, _, _ = primitive_bank.centers_keys_sigmas()
    unit_keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(seed + 8801)
    shuffle_perm = rng.permutation(len(skill_mem.units)) if skill_mem.units else []
    if method in ("primitive_anchor", "primitive_normal"):
        sims = qn @ unit_keys.T
        top = sims.argmax(axis=1)
        confidence[:] = sims[np.arange(n), top]
        b = centers[top]
        rng_dec = np.random.default_rng(seed + 9001)
        residual = np.zeros((n, m6.residual_dim), np.float32)
        anchors[:] = m6.decode(b, residual, rows["f"], rows["s"])
        return {"anchor": anchors, "confidence": confidence, "unit_id": ids[top]}
    if method == "raw_interval":
        tkeys = train_keys / np.maximum(np.linalg.norm(train_keys, axis=1, keepdims=True), 1e-8)
        sim = qn @ tkeys.T
        top = sim.argmax(axis=1)
        anchors[:] = train_actions[top, :HORIZON]
        confidence[:] = sim[np.arange(n), top]
        return {"anchor": anchors, "confidence": confidence, "raw_index": top}
    if method in ("hier_skill", "shuffle"):
        for i, query in enumerate(qn):
            unit, score = skill_mem.top1(query)
            picked = unit
            if method == "shuffle" and len(shuffle_perm) > 1:
                picked = skill_mem.units[int(shuffle_perm[unit.skill_id % len(shuffle_perm)])]
            b, _ = skill_mem.position_base(picked, query)
            anchors[i] = m6.decode(b[None], np.zeros((1, m6.residual_dim), np.float32),
                                   rows["f"][i:i + 1], rows["s"][i:i + 1])[0]
            confidence[i] = score
        return {"anchor": anchors, "confidence": confidence}
    raise ValueError(method)


def _calibrate_confidence(model, norm, rows, q, anchors, confidence, eps, target,
                          *, device, samples=fc.SAMPLES):
    errors = []
    for t0 in fc.T0_GRID:
        if t0 >= 1:
            pred = fc.rollout(model, norm, rows["f"], rows["s"], None, eps,
                              np.ones(len(target), np.float32), device=device)
        else:
            pred = fc.rollout(model, norm, rows["f"], rows["s"], anchors, eps,
                              np.full(len(target), t0, np.float32), device=device)
        errors.append(np.square(pred - target[None]).mean(axis=(0, 2, 3)))
    err = np.stack(errors)
    return fc.fit_monotone_g(confidence, err, grid=fc.T0_GRID)


def _evaluate_condition(model_n, model_a, norm_n, norm_a, rows, target, eps,
                        methods, retrievals, calibrations, *, device):
    out = {}
    normal = fc.rollout(model_n, norm_n, rows["f"], rows["s"], None, eps,
                        np.ones(len(target), np.float32), device=device)
    out["normal"] = _simple_metrics(normal, target)
    for method in methods:
        ret = retrievals[method]
        g = calibrations[method]
        t0 = fc.apply_g(g, ret["confidence"])
        nsteps = fc.steps_for(t0)
        model = model_n if method in ("primitive_normal", "raw_interval") else model_a
        norm = norm_n if model is model_n else norm_a
        pred = fc.rollout(model, norm, rows["f"], rows["s"], ret["anchor"], eps,
                          t0, n=nsteps, device=device)
        out[method] = {**_simple_metrics(pred, target),
                       "mean_t0": float(np.mean(t0)), "mean_steps": float(np.mean(nsteps))}
    return out


def _simple_metrics(pred, target):
    per = np.square(np.asarray(pred) - np.asarray(target)[None]).mean(axis=(2, 3))
    return {"mse": float(per.mean()), "best_of_8_mse": float(per.min(axis=0).mean()),
            "n_rows": int(per.shape[1])}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _metrics_for_units(memory, segments, task_names=None):
    durations, n_tasks, purities, supports = [], [], [], []
    for unit in memory.units:
        segs = [m["segment"] for m in unit.members]
        durations.extend(s["length_seconds"] for s in segs)
        n_tasks.append(len({s["task_index"] for s in segs}))
        classes = [s["skill_class"] for s in segs if s["skill_class"] >= 0]
        purities.append(max(np.bincount(classes)) / len(classes) if classes else None)
        supports.append(len(segs))
    all_lengths = [s["length_seconds"] for s in segments]
    return {"n_skills": len(memory.units), "n_segments": len(segments),
            "covered_segments": int(sum(u.support for u in memory.units)),
            "coverage": float(sum(u.support for u in memory.units) / max(len(segments), 1)),
            "segment_duration_seconds_q": np.quantile(all_lengths, [0, .25, .5, .75, 1]).tolist() if all_lengths else [],
            "unit_member_duration_seconds_q": np.quantile(durations, [0, .25, .5, .75, 1]).tolist() if durations else [],
            "tasks_per_skill_q": np.quantile(n_tasks, [0, .25, .5, .75, 1]).tolist() if n_tasks else [],
            "verb_purity_mean": float(np.mean([p for p in purities if p is not None])) if any(p is not None for p in purities) else None,
            "support_q": np.quantile(supports, [0, .25, .5, .75, 1]).tolist() if supports else [],
            "cluster": memory.describe()}


def _switch_stats(rows, row_segment, memory):
    ids = np.asarray([memory.assignments.get(int(s), -1) if s >= 0 else -1 for s in row_segment])
    switches, pairs = 0, 0
    run_lengths, current = [], 0
    for ep in np.unique(np.asarray(rows["episode_id"]).astype(str)):
        ix = np.flatnonzero(np.asarray(rows["episode_id"]).astype(str) == ep)
        ix = ix[np.argsort(rows["frame"][ix], kind="stable")]
        vals = ids[ix]
        for a, b in zip(vals[:-1], vals[1:]):
            if a >= 0 and b >= 0:
                pairs += 1
                switches += (a != b)
        for v in vals:
            if v < 0:
                if current:
                    run_lengths.append(current); current = 0
            elif not current:
                current = 1; last = v
            elif v == last:
                current += 1
            else:
                run_lengths.append(current); current = 1; last = v
        if current:
            run_lengths.append(current); current = 0
    return {"assigned_neighbor_switch_rate": float(switches / pairs) if pairs else None,
            "same_skill_run_primitives_q": np.quantile(run_lengths, [0, .25, .5, .75, 1]).tolist() if run_lengths else [],
            "primitive_reference_switch_rate": 0.79}


def run_one(rows, *, split, seed, texts_by_source, text_embeddings, device, smoke=False, m1_epochs=60,
            flow_epochs=80, batch_size=512, stride_seconds=0.25):
    tic = time.perf_counter()
    train_tasks, eval_tasks, _ = SPLITS[split]
    train_all = _subset(rows, np.isin(rows["task_index"], train_tasks) & rows["valid_h25"])
    test = _subset(rows, np.isin(rows["task_index"], eval_tasks) & rows["valid_h25"])
    if smoke:
        train_all = af._select_episodes(train_all, train_tasks, 12, seed + 101)
        test = af._select_episodes(test, eval_tasks, 4, seed + 211)
    train, d1 = fc.holdout_split(train_all, split, seed, per_task=2 if smoke else fc.HOLDOUT_PER_TASK)
    fit_ix, val_ix = _episode_train_validation_indices(train, seed + 401)
    fit = _subset(train, np.isin(np.arange(len(train["a"])), fit_ix))
    val = _subset(train, np.isin(np.arange(len(train["a"])), val_ix))
    eval_episodes = set(map(str, d1["episode_id"])) | set(map(str, test["episode_id"]))
    fc.assert_calibration_rows(val, split, eval_episodes)
    actions = fit["a"][:, :HORIZON].copy()
    m6, train_bases, _ = af._fit_m6_backbone_and_units(fit, seed=seed + 19,
                                                       device=device, epochs=m1_epochs)
    m1 = m6.backbone
    fit_text = fc.attach_text(fit, texts_by_source, text_embeddings, "instruction")
    e_model, efit, train_keys = af._fit_query_encoder(
        fit_text, train_bases,
        seed=seed + 701, device=device, allowed_tasks=tuple(train_tasks),
        subtask_source="instruction")
    val_b, _ = m1.encode(val["a"][:, :HORIZON], val["s"])
    d1_b, _ = m1.encode(d1["a"][:, :HORIZON], d1["s"])
    test_b, _ = m1.encode(test["a"][:, :HORIZON], test["s"])
    query = {}
    text_rows = {name: fc.attach_text(rws, texts_by_source, text_embeddings, "instruction")
                 for name, rws in (("val", val), ("D1", d1), ("D3", test))}
    for name, rws in text_rows.items():
        query[name] = af._query_embedding(e_model, efit, af._normal_query_rows(rws), device=device)

    # M6 primitive bank, identical to fair_compare's D1/D3 memory construction.
    base_scale = float(np.sqrt(np.mean(np.square(train_bases - train_bases.mean(0)))))
    primitive_bank = af.UnitBank(m6, base_scale)
    primitive_bank.seed_training(train_bases, train_keys, fit["episode_id"])

    # F_normal and F_anchor use the same fitting recipes and random seeds as
    # fair_compare. F_anchor mixes retrieved, gold and ordinary starts.
    dim = int(np.prod(m1.action_shape))
    zeros = np.zeros((len(actions), dim), np.float32)
    model_n, norm_n, fit_n = af.fit_anchor_flow(
        fit, actions, (zeros, np.zeros(len(zeros), np.float32), zeros, np.zeros(len(zeros), np.float32)),
        source_mix=fc.NORMAL_MIX, seed=seed + 827, epochs=flow_epochs,
        patience=8, batch_size=batch_size, device=device)
    norm_ix, _ = _episode_train_validation_indices(fit, seed + 827 + 11)
    am, ass = _standardize_fit(actions[norm_ix].reshape(len(norm_ix), -1))
    fm, fs = _standardize_fit(fit["f"][norm_ix]); sm, ss = _standardize_fit(fit["s"][norm_ix])
    flow_norm = {"a_mean": am, "a_scale": ass, "f_mean": fm, "f_scale": fs,
                 "s_mean": sm, "s_scale": ss, "action_shape": m1.action_shape}
    source = af._make_flow_sources(m1, primitive_bank, fit, train_bases, train_keys,
                                   efit["temperature"], flow_norm, seed=seed + 823)
    model_a, norm_a, fit_a = af.fit_anchor_flow(
        fit, actions, source, source_mix=fc.ANCHOR_MIX, seed=seed + 827,
        epochs=flow_epochs, patience=8, batch_size=batch_size, device=device)

    # Infer the cache stride (frames / 20 Hz LIBERO stream); fair_compare's
    # cache sampling takes the same rows. Caller may set this explicitly.
    threshold, train_adjacent = _functional_threshold(m1, fit, train_bases)
    primitive_sequence = _primitive_unit_ids(m6, train_bases)
    segments_by_mode, assignments = {}, {}
    for mode in ("gripper", "functional", "both"):
        segs, row_ids = _make_episode_segments(fit, train_bases, train_keys, mode=mode,
                                               stride_seconds=stride_seconds,
                                               functional_threshold=threshold,
                                               functional_m1=m1,
                                               primitive_ids=primitive_sequence)
        # Shared functional probe, sampled from fitting rows only.
        rng_probe = np.random.default_rng(seed + 334)
        probe_ix = np.sort(rng_probe.choice(len(fit["a"]), min(32, len(fit["a"])), replace=False))
        probe_rows = _subset(fit, np.isin(np.arange(len(fit["a"])), probe_ix))
        mem, seg_func, random_dist = _segment_memory(segs, m1, probe_rows, seed)
        segments_by_mode[mode] = (segs, row_ids, mem, seg_func, random_dist)

    target_by_name = {"val": val["a"][:, :HORIZON].copy(), "D1": d1["a"][:, :HORIZON].copy(),
                      "D3": test["a"][:, :HORIZON].copy()}
    dim = int(np.prod(m1.action_shape))
    rng = np.random.default_rng(seed * 1_000_003 + 31)
    eps = {name: rng.standard_normal((fc.SAMPLES, len(tgt), dim)).astype(np.float32)
           for name, tgt in target_by_name.items()}
    methods_by_mode, cal_by_mode, results_by_mode = {}, {}, {}
    primitive_ids, primitive_centers, primitive_keys, _, _ = primitive_bank.centers_keys_sigmas()
    primitive_kn = primitive_keys / np.maximum(np.linalg.norm(primitive_keys, axis=1, keepdims=True), 1e-8)

    for mode, (segments, train_row_segment, skill_mem, seg_func, random_dist) in segments_by_mode.items():
        # Validation-derived confidence calibrations; all use the exact same
        # common noise and monotone t0->steps rule as fair_compare.
        val_retr = {}
        for method in ("primitive_anchor", "primitive_normal", "hier_skill", "raw_interval", "shuffle"):
            if method.startswith("primitive"):
                val_retr[method] = _retrieve(method, val, query["val"], m6, primitive_bank, skill_mem,
                                             train_keys, actions, train_bases, seed=seed, split=split)
            elif method == "raw_interval":
                val_retr[method] = _retrieve(method, val, query["val"], m6, primitive_bank, skill_mem,
                                             train_keys, actions, train_bases, seed=seed, split=split)
            elif method == "hier_skill":
                val_retr[method] = _retrieve(method, val, query["val"], m6, primitive_bank, skill_mem,
                                             train_keys, actions, train_bases, seed=seed, split=split)
            else:
                val_retr[method] = _retrieve(method, val, query["val"], m6, primitive_bank, skill_mem,
                                             train_keys, actions, train_bases, seed=seed, split=split,
                                             shuffle=True)
        val_cal = {}
        for method, ret in val_retr.items():
            model = model_n if method in ("primitive_normal", "raw_interval") else model_a
            norm = norm_n if model is model_n else norm_a
            val_cal[method] = _calibrate_confidence(model, norm, val, query["val"], ret["anchor"],
                                                    ret["confidence"], eps["val"], target_by_name["val"],
                                                    device=device)
        # D1 and D3 share identical memory construction, and paired noise is
        # generated once per suite/seed. Skill modes change only the anchor.
        dsets = {"D1": d1, "D3": test}
        mode_results = {}
        for cond, rws in dsets.items():
            retr = {method: _retrieve(method, rws, query[cond], m6, primitive_bank, skill_mem,
                                      train_keys, actions, train_bases, seed=seed, split=split,
                                      shuffle=(method == "shuffle"))
                    for method in val_retr}
            mode_results[cond] = _evaluate_condition(model_n, model_a, norm_n, norm_a, rws,
                                                     target_by_name[cond], eps[cond], METHODS[1:],
                                                     retr, val_cal, device=device)
            mode_results[cond]["retrieval"] = {
                method: {"mean_confidence": float(np.mean(ret["confidence"])),
                         "mean_t0": float(np.mean(fc.apply_g(val_cal[method], ret["confidence"]))) }
                for method, ret in retr.items()}
        methods_by_mode[mode] = mode_results
        mem_metrics = _metrics_for_units(skill_mem, segments)
        mem_metrics["functional_boundary_threshold_q90"] = threshold
        mem_metrics["functional_boundary_distance_training_q"] = (
            np.quantile(train_adjacent, [0, .5, .9, .95, 1]).tolist() if len(train_adjacent) else [])
        mem_metrics["random_segment_function_distance_median"] = float(np.median(random_dist))
        mem_metrics["epsilon"] = float(skill_mem.epsilon)
        mem_metrics["switching"] = _switch_stats(fit, train_row_segment, skill_mem)
        mem_metrics["boundary_mode"] = mode
        cal_by_mode[mode] = {k: v for k, v in val_cal.items()}
        results_by_mode[mode] = mem_metrics

    # D2 primitive/raw controls use fair_compare's exact episode-block stream.
    # Copy the pair together so the bank keeps the copied M6 instance.
    import copy
    m6_stream, primitive_stream = copy.deepcopy((m6, primitive_bank))
    d2_primitive, _ = fc.stream_d2(
        m6_stream, {"instruction": primitive_stream}, test, test_b,
        {"instruction": query["D3"]}, {"instruction": train_keys}, actions,
        {"instruction": float(efit["temperature"])},
        initial_ids=np.arange(len(m6_stream.prototypes)), seed=seed, device=device)
    d2_primitive = d2_primitive["instruction"]
    # D2 prequential evaluation for hierarchical memory. Retrieve a whole
    # episode before inserting any segment from that episode.
    d2_results = {}
    for mode, (segments, _, skill_mem, _, _) in segments_by_mode.items():
        # Copy memory to keep D3 objects immutable.
        online = copy.deepcopy(skill_mem)
        test_keys = query["D3"]
        rows_ep = np.asarray(test["episode_id"]).astype(str)
        order = np.random.default_rng(seed).permutation(np.unique(rows_ep))
        preds, targets, row_t0, coverage = [], [], [], []
        online_rows, online_bases = [], []
        for ep in order:
            ix = np.flatnonzero(rows_ep == ep)
            ix = ix[np.argsort(test["frame"][ix], kind="stable")]
            q_ep = test_keys[ix]
            sub = _subset(test, np.isin(np.arange(len(rows_ep)), ix))
            ret = _retrieve("hier_skill", sub, q_ep, m6, primitive_bank, online,
                            train_keys, actions, train_bases, seed=seed, split=split)
            g = cal_by_mode[mode]["hier_skill"]
            t0 = fc.apply_g(g, ret["confidence"])
            pred = fc.rollout(model_a, norm_a, sub["f"], sub["s"], ret["anchor"], eps["D3"][:, ix],
                              t0, n=fc.steps_for(t0), device=device)
            preds.append(pred); targets.append(target_by_name["D3"][ix]); row_t0.extend(t0.tolist())
            coverage.extend([len(online.units)] * len(ix))
            # Add only after all episode queries have been answered.
            ep_b = test_b[ix]
            ep_k = test_keys[ix]
            ep_primitive = _primitive_unit_ids(m6, ep_b)
            ep_segments, _ = _make_episode_segments(sub, ep_b, ep_k, mode=mode, stride_seconds=stride_seconds,
                                                    functional_threshold=threshold, functional_m1=m1,
                                                    primitive_ids=ep_primitive)
            if ep_segments:
                # Test-segment function embeddings use the same training probe.
                femb = _decoded_curve_embedding(m1, np.stack([x["base_curve"] for x in ep_segments]), probe_rows)
                for seg, fvec in zip(ep_segments, femb):
                    seg["local_segment"] = int(online.seen)
                    online.add(seg, fvec)
                online.flush(force=False)
        pred_all = np.concatenate(preds, axis=1)
        target_all = np.concatenate(targets, axis=0)
        d2_results[mode] = {"hier_skill": _simple_metrics(pred_all, target_all),
                            "mean_t0": float(np.mean(row_t0)),
                            "mean_online_skill_count": float(np.mean(coverage)),
                            "post_stream_skill_count": len(online.units),
                            "registered_after_stream": online.registered,
                            "pending_after_stream": len(online.pending)}

        primitive_anchor = m6_stream.decode(
            d2_primitive["unit_b"], np.zeros((len(test_b), m6_stream.residual_dim), np.float32),
            test["f"], test["s"])
        shuffle_anchor = m6_stream.decode(
            d2_primitive["shuf_b"], np.zeros((len(test_b), m6_stream.residual_dim), np.float32),
            test["f"], test["s"])
        controls = {
            "primitive_anchor": (primitive_anchor, d2_primitive["conf_sim"], model_a, norm_a,
                                 cal_by_mode[mode]["primitive_anchor"]),
            "primitive_normal": (primitive_anchor, d2_primitive["conf_sim"], model_n, norm_n,
                                 cal_by_mode[mode]["primitive_normal"]),
            "raw_interval": (d2_primitive["raw_action"], d2_primitive["raw_conf_sim"], model_n, norm_n,
                             cal_by_mode[mode]["raw_interval"]),
            "shuffle": (shuffle_anchor, d2_primitive["conf_sim"], model_a, norm_a,
                        cal_by_mode[mode]["shuffle"]),
        }
        d2_results[mode]["normal"] = _simple_metrics(
            fc.rollout(model_n, norm_n, test["f"], test["s"], None, eps["D3"],
                       np.ones(len(test_b), np.float32), device=device), target_by_name["D3"])
        for name, (anchor, confidence, model, norm, calibration) in controls.items():
            t0 = fc.apply_g(calibration, confidence)
            pred = fc.rollout(model, norm, test["f"], test["s"], anchor, eps["D3"],
                              t0, n=fc.steps_for(t0), device=device)
            d2_results[mode][name] = {**_simple_metrics(pred, target_by_name["D3"]),
                                      "mean_t0": float(np.mean(t0)),
                                      "mean_steps": float(np.mean(fc.steps_for(t0)))}

    return {"split": split, "seed": seed, "smoke": smoke,
            "rows": {"train_fit": len(fit["a"]), "validation": len(val["a"]),
                     "D1": len(d1["a"]), "D3": len(test["a"])},
            "M6_units": len(m6.prototypes), "M6_epsilon": float(m6.epsilon),
            "F_normal_best_epoch": int(fit_n["best_epoch"]),
            "F_anchor_best_epoch": int(fit_a["best_epoch"]),
            "F_normal_validation_loss": float(fit_n["best_validation_loss"]),
            "F_anchor_validation_loss": float(fit_a["best_validation_loss"]),
            "segmentation": results_by_mode, "D1_D3": methods_by_mode,
            "D2_hierarchical_prequential": d2_results,
            "calibration": {mode: {method: cal[method] for method in cal}
                            for mode, cal in cal_by_mode.items()},
            "duration_seconds": time.perf_counter() - tic}


def _summary(payload):
    lines = ["# S3: プリミティブ列から作る上位スキル（軽量オフライン実験）", "",
             f"実行条件: seeds={payload['seeds']}、splits={payload['splits']}、smoke={payload['smoke']}。"
             "ラベルはスキル形成に使わず、解釈の集計にだけ使った。行動誤差は同じ F・同じ行の同一ノイズで評価。", "",
             "## 区切り方式と形成診断", "",
             "| Split/seed | 区切り | 上位スキル数 | 形成区間/被覆 | 区間長中央値 秒 | 1スキル当たりタスク中央値 | 動詞純度 | 同一スキル継続中央値 | 切替率 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        for mode, m in run["segmentation"].items():
            duration_q = m["unit_member_duration_seconds_q"]
            tasks_q = m["tasks_per_skill_q"]
            run_q = m["switching"]["same_skill_run_primitives_q"]
            lines.append(f"| {run['split']}/{run['seed']} | {mode} | {m['n_skills']} | {m['covered_segments']}/{m['n_segments']} ({m['coverage']:.2f}) | {duration_q[2]:.2f} | {tasks_q[2]:.1f} | {m['verb_purity_mean'] if m['verb_purity_mean'] is not None else 'NA'} | {run_q[2] if run_q else 'NA'} | {m['switching']['assigned_neighbor_switch_rate'] if m['switching']['assigned_neighbor_switch_rate'] is not None else 'NA'} |")
    lines += ["", "## 行動 MSE", "", "D3（未見 suite・メモリ追加なし）が主指標。D1 は保留エピソード、D2 は新 suite をエピソード単位で追加する逐次条件。", "",
              "| Split/seed/条件 | 通常 F | Primitive + F_anchor | Primitive + F_normal | 上位 skill | Raw interval | Shuffle |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        for mode, conds in run["D1_D3"].items():
            for cond, result in conds.items():
                lines.append(f"| {run['split']}/{run['seed']}/{cond}/{mode} | {result['normal']['mse']:.4f} | {result['primitive_anchor']['mse']:.4f} | {result['primitive_normal']['mse']:.4f} | {result['hier_skill']['mse']:.4f} | {result['raw_interval']['mse']:.4f} | {result['shuffle']['mse']:.4f} |")
        for mode, result in run["D2_hierarchical_prequential"].items():
            lines.append(f"| {run['split']}/{run['seed']}/D2/{mode} | {result['normal']['mse']:.4f} | {result['primitive_anchor']['mse']:.4f} | {result['primitive_normal']['mse']:.4f} | {result['hier_skill']['mse']:.4f} | {result['raw_interval']['mse']:.4f} | {result['shuffle']['mse']:.4f} |")
    lines += ["", "## 読み方", "", "スキル数は自動決定であり、上限 K は置いていない。機能閾値は学習データの隣接区間距離 q90、形成閾値は学習区間表現のランダム組距離中央値×0.1。保留バッファは200区間、登録条件は10区間・3エピソード。", "",
              "切り替わり率は学習エピソードの割当済み隣接プリミティブだけで計算し、解釈資料の79%と比較できる参考値。未割当区間は率の分母から除く。動詞純度は事後ラベルのクラスタ内最大クラス比。", "",
              f"詳細な各seed/splitの結果と学習条件は [{payload['results_path']}](./results.json)。"]
    return "\n".join(lines) + "\n"


def _figures(out, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    names = ("gripper", "functional", "both")
    vals = {m: [] for m in names}
    for run in runs:
        for m in names:
            vals[m].append(run["segmentation"][m]["switching"]["assigned_neighbor_switch_rate"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.boxplot([[v for v in vals[m] if v is not None] for m in names], labels=names)
    ax.axhline(.79, color="tab:red", linestyle="--", label="primitive ref 0.79")
    ax.set_ylabel("neighboring skill switch rate")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figdir / "switching.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in names:
        ax.plot(range(len(vals[m])), vals[m], marker="o", label=m)
    ax.set_ylabel("skill switch rate"); ax.set_xlabel("run index"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "switching_by_run.png", dpi=140); plt.close(fig)
    return ["figures/switching.png", "figures/switching_by_run.png"]


def _load_partial_resume(out, seeds, splits, smoke):
    partial_path = Path(out) / "results.partial.json"
    if not partial_path.is_file():
        raise FileNotFoundError(f"resume file missing: {partial_path}")
    partial = json.loads(partial_path.read_text())
    if (partial["seeds"] != list(seeds) or partial["splits"] != list(splits)
            or bool(partial["smoke"]) != bool(smoke)):
        raise ValueError("resume configuration differs from results.partial.json")
    return partial["runs"]


def run(*, output_root=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'), cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seeds=(0, 1, 2), splits=("A", "B"),
        device_name="cuda", m1_epochs=60, flow_epochs=80, batch_size=512,
        smoke=False, threads=8, stride_seconds=None, resume_dir=None):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(device_name)
    if resume_dir is not None:
        out = Path(resume_dir)
        runs = _load_partial_resume(out, seeds, splits, smoke)
    else:
        stamp = datetime.now(timezone.utc).strftime("hier-skills-%Y%m%dT%H%M%SZ")
        if smoke:
            stamp += "-smoke"
        out = Path(output_root) / f"{stamp}-{uuid.uuid4().hex[:6]}"
        out.mkdir(parents=True, exist_ok=False)
        (out / "figures").mkdir()
        runs = []
    start = time.perf_counter()
    rows = load_rows(cache_dir, dataset_dir, max_horizon=HORIZON)
    cache_manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    dataset_info = json.loads((Path(dataset_dir) / "meta" / "info.json").read_text())
    if stride_seconds is None:
        stride_seconds = float(cache_manifest["stride"]) / float(dataset_info["fps"])
    task_ids = sorted({t for split in splits for t in (*SPLITS[split][0], *SPLITS[split][1])})
    rows = _subset(rows, np.isin(rows["task_index"], task_ids) & rows["valid_h25"])
    texts, emb = fc.precompute_texts(rows, ("instruction",), cache_dir)
    finished = {(r["split"], int(r["seed"])) for r in runs}
    for split in splits:
        for seed in seeds:
            if (split, int(seed)) in finished:
                continue
            result = run_one(rows, split=split, seed=seed, texts_by_source=texts,
                             text_embeddings=emb, device=device, smoke=smoke,
                             m1_epochs=m1_epochs, flow_epochs=flow_epochs,
                             batch_size=batch_size, stride_seconds=stride_seconds)
            runs.append(result)
            partial = {"seeds": list(seeds), "splits": list(splits), "smoke": smoke, "runs": runs}
            (out / "results.partial.json").write_text(json.dumps(partial, ensure_ascii=False,
                                                                    default=_json_default))
    payload = {"spec": "/work/docs/research/anchor_flow_skill_memory.md §S3",
               "seeds": list(seeds), "splits": list(splits), "smoke": bool(smoke),
               "subtask_source": "instruction", "data_rows": len(rows["a"]),
               "stride_seconds": float(stride_seconds), "cache_stride_frames": int(cache_manifest["stride"]),
               "dataset_fps": int(dataset_info["fps"]), "K_points": K_POINTS,
               "skill_kappa": SKILL_KAPPA, "min_segments": MIN_SEGMENTS,
               "min_episodes": MIN_EPISODES, "pending_batch": PENDING_BATCH,
               "flow_architecture": "fair_compare F_normal and F_anchor recipes",
               "noise_samples": fc.SAMPLES, "t0_grid": list(fc.T0_GRID), "runs": runs,
               "duration_minutes": (time.perf_counter() - start) / 60,
               "output_dir": str(out)}
    payload["figures"] = _figures(out, runs)
    payload["results_path"] = str(out / "results.json")
    (out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                   default=_json_default))
    (out / "summary.md").write_text(_summary(payload))
    (out / "results.partial.json").unlink(missing_ok=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot')))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--flow-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-dir", type=Path, default=None,
                        help="continue a matching results.partial.json without rerunning completed split/seed pairs")
    a = parser.parse_args(argv)
    out = run(output_root=a.output_root, cache_dir=a.cache_dir, dataset_dir=a.dataset,
              seeds=tuple(int(x) for x in a.seeds.split(",")), splits=tuple(a.splits.split(",")),
              device_name=a.device, m1_epochs=a.m1_epochs, flow_epochs=a.flow_epochs,
              batch_size=a.batch_size, smoke=a.smoke, threads=a.threads,
              stride_seconds=a.stride_seconds, resume_dir=a.resume_dir)
    print(f"output: {out}")


if __name__ == "__main__":
    main()
