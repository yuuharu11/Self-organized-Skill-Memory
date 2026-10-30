"""Round-trip test: demo -> M6 skill memory -> same-input retrieval -> final action.

For every training-suite interval (o, A) of Split A/B, the memory built from the
same demonstrations is queried with the same o, and the distance to A is
measured stage by stage:

  0   Dec(Enc(A, s), s)               (r ~ N(0, I); 0a uses the encoder's own r)
  1   Dec(c_{u(A)}, s)                u(A): unit the interval belongs to
  2   retrieval agreement             top-1 == u(A), w_{u(A)}
  3   Dec(c_top1, s) / component draw b'~N(c_k, sigma_k^2), k~Cat(w)
  3'  two-stage: top-1 unit, then the member interval with the closest key
  4   anchors {0, 1, 3, 3', 3-comp} denoised from t0 in T0S by the mixed F
  5   F_normal from noise, mixed F from noise, raw-interval nearest neighbour
      (RAGDP-type) denoised from t0 in T0S

Memory condition (a): the query's own episode stays in memory (pure round
trip); (b): the query's episode is removed from values, keys and sigma.

M1/M6/E/mixed F are built by the S1 procedure (anchor_flow.py); anchor_flow.py
is imported, never edited. F_normal has the same architecture, rows, seed,
epoch cap and patience but is fitted on the ordinary path only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import anchor_flow as af
from .clip_retrieval import (
    DEFAULT_ANNOTATIONS, DEFAULT_PALIGEMMA_TOKENIZER, DEFAULT_PI05_BASE_PARAMS,
    _query_embedding, frozen_pi05_plan_embeddings,
)
from .splits import SPLITS, subset_rows as _subset
from .physical_scales import fit_physical_scales as _fit_physical_scales
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT,
    _episode_train_validation_indices, _standardize, _standardize_fit, load_rows,
)

HORIZON = af.HORIZON
T0S = (0.2, 0.4, 0.6)
SAMPLES = af.EVAL_SAMPLES
QUERY_CHUNK = 128
SOURCE_MIX = (0.65, 0.05, 0.30)
NORMAL_MIX = (0.0, 0.0, 1.0)
CONDITIONS = ("a_same_episode_in_memory", "b_same_episode_removed")
METRIC_KEYS = ("mse", "best_of_8_mse", "pos_chunk_m", "pos_terminal_m",
               "rot_chunk_rad", "rot_terminal_rad", "gripper_agreement")


# --------------------------------------------------------------------------
# Physical units
# --------------------------------------------------------------------------

def fit_physical_scales(cache_dir, episode_ids, *, horizon=HORIZON, max_episodes=400,
                        seed=0, stride=5):
    """Fit motion scales with this experiment's historical episode cap."""
    return _fit_physical_scales(
        cache_dir, episode_ids, horizon=horizon, max_episodes=max_episodes,
        seed=seed, stride=stride)


def action_metrics(pred, target, scales):
    """Per-row metrics, averaged over samples. pred (S,N,H,7), target (N,H,7)."""
    pred = np.asarray(pred, np.float32)
    if pred.ndim == 3:
        pred = pred[None]
    target = np.asarray(target, np.float32)
    k_pos = np.asarray(scales["pos_m_per_action_unit_step"], np.float32)
    k_rot = np.asarray(scales["rot_rad_per_action_unit_step"], np.float32)
    sq = (pred - target[None]) ** 2
    per_sample_mse = sq.mean(axis=(2, 3))                       # (S,N)
    diff = pred - target[None]
    pos_err = np.linalg.norm(np.cumsum(diff[..., :3], axis=2) * k_pos, axis=-1)   # (S,N,H)
    rot_err = np.linalg.norm(np.cumsum(diff[..., 3:6], axis=2) * k_rot, axis=-1)
    grip = (np.sign(pred[..., 6]) == np.sign(target[None, ..., 6])).mean(axis=2)
    grip = np.where(np.isnan(pred).any(axis=(2, 3)), np.nan, grip)
    return {"mse": per_sample_mse.mean(0),
            "best_of_8_mse": per_sample_mse.min(0),
            "pos_chunk_m": pos_err.mean(axis=2).mean(0),
            "pos_terminal_m": pos_err[..., -1].mean(0),
            "rot_chunk_rad": rot_err.mean(axis=2).mean(0),
            "rot_terminal_rad": rot_err[..., -1].mean(0),
            "gripper_agreement": grip.mean(0)}


# --------------------------------------------------------------------------
# Memory views and retrieval (vectorised over queries)
# --------------------------------------------------------------------------

def _normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def seed_bank_with_rows(bank, m6, bases, keys, episodes):
    """Exactly UnitBank.seed_training, but also return each row's unit.

    Returns nearest unit id (always defined), functional distance, the
    within-epsilon flag (== bank membership) and member row indices per unit.
    """
    n = len(bases)
    nearest = np.full(n, -1, np.int64)
    distance = np.full(n, np.inf, np.float64)
    unit_rows: dict[int, list[int]] = {}
    for i in range(n):
        uid, dist = m6._nearest(bases[i])
        if uid is None:
            continue
        nearest[i], distance[i] = int(uid), float(dist)
        if dist <= m6.epsilon:
            bank.add(uid, bases[i], keys[i], episodes[i])
            unit_rows.setdefault(int(uid), []).append(i)
    assigned = distance <= m6.epsilon
    return nearest, distance, assigned, {k: np.asarray(v, np.int64) for k, v in unit_rows.items()}


def retrieve(bank, query_keys, episodes, own_unit, unit_rows, train_episode_codes,
             temperature, *, exclude_same_episode, rng, samples=SAMPLES):
    """Unit retrieval and two-stage instance retrieval for every query row.

    The memory view is bank.centers_keys_sigmas() (condition a) or the
    leave-this-episode-out view (condition b), as in anchor_flow.py.
    """
    n = len(query_keys)
    qn = _normalize(query_keys)
    member_keys_n = _normalize(np.asarray([m[1] for uid in sorted(bank.members) for m in bank.members[uid]]))
    # Map unit -> positions into member_keys_n, aligned with unit_rows order.
    offsets, pos = {}, 0
    for uid in sorted(bank.members):
        count = len(bank.members[uid])
        offsets[uid] = np.arange(pos, pos + count)
        pos += count
        if len(unit_rows.get(uid, ())) != count:
            raise AssertionError("bank members and recorded member rows disagree")
    base_dim = bank.m6.prototypes.shape[1]
    out = {"top1": np.full(n, -1, np.int64), "top1_center": np.zeros((n, base_dim), np.float32),
           "own_center": np.full((n, base_dim), np.nan, np.float32),
           "own_present": np.zeros(n, bool), "w_own": np.zeros(n, np.float32),
           "w_max": np.zeros(n, np.float32), "entropy": np.zeros(n, np.float32),
           "unit_count": np.zeros(n, np.int64),
           "comp_center": np.zeros((samples, n, base_dim), np.float32),
           "comp_sigma": np.zeros((samples, n), np.float32),
           "instance_row": np.full(n, -1, np.int64),
           "instance_similarity": np.full(n, np.nan, np.float32)}
    episodes = np.asarray(episodes).astype(str)
    groups = [(None, np.arange(n))] if not exclude_same_episode else [
        (ep, np.flatnonzero(episodes == ep)) for ep in np.unique(episodes)]
    full_view = bank.centers_keys_sigmas() if not exclude_same_episode else None
    for ep, ix in groups:
        ids, centers, keys, sigmas, _ = (full_view if ep is None else bank.centers_keys_sigmas([ep]))
        if not len(ids):
            raise ValueError("empty memory view")
        logits = (qn[ix] @ _normalize(keys).T) / max(float(temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(axis=1, keepdims=True)
        top = w.argmax(axis=1)
        out["top1"][ix] = ids[top]
        out["top1_center"][ix] = centers[top]
        out["w_max"][ix] = w.max(axis=1)
        out["entropy"][ix] = -(w * np.log(np.maximum(w, 1e-12))).sum(axis=1)
        out["unit_count"][ix] = len(ids)
        where = {int(u): j for j, u in enumerate(ids)}
        for local, row in enumerate(ix):
            j = where.get(int(own_unit[row]))
            if j is not None:
                out["own_present"][row] = True
                out["own_center"][row] = centers[j]
                out["w_own"][row] = w[local, j]
            draw = rng.choice(len(ids), size=samples, p=w[local])
            out["comp_center"][:, row] = centers[draw]
            out["comp_sigma"][:, row] = sigmas[draw]
            uid = int(ids[top[local]])
            cand_rows = unit_rows[uid]
            cand_pos = offsets[uid]
            if ep is not None:
                keep = train_episode_codes[cand_rows] != train_episode_codes_lookup(ep)
                cand_rows, cand_pos = cand_rows[keep], cand_pos[keep]
            if not len(cand_rows):
                raise AssertionError("top-1 unit has no eligible member")
            sims = member_keys_n[cand_pos] @ qn[row]
            best = int(np.argmax(sims))
            out["instance_row"][row] = cand_rows[best]
            out["instance_similarity"][row] = sims[best]
    return out


_EPISODE_CODE_TABLE: dict[str, int] = {}


def train_episode_codes_lookup(episode):
    return _EPISODE_CODE_TABLE[str(episode)]


def encode_episode_codes(episodes):
    episodes = np.asarray(episodes).astype(str)
    for ep in np.unique(episodes):
        _EPISODE_CODE_TABLE.setdefault(str(ep), len(_EPISODE_CODE_TABLE))
    return np.asarray([_EPISODE_CODE_TABLE[e] for e in episodes], np.int64)


def raw_nearest(query_keys, memory_keys, query_codes, memory_codes, *, exclude_same_episode,
                device, chunk=1024):
    """RAGDP-type raw-interval top-1 by cosine, batched on the device."""
    kn = torch.as_tensor(_normalize(memory_keys), device=device)
    mc = torch.as_tensor(memory_codes, device=device)
    result = np.zeros(len(query_keys), np.int64)
    sims_out = np.zeros(len(query_keys), np.float32)
    qn_all = _normalize(query_keys)
    for lo in range(0, len(query_keys), chunk):
        hi = min(lo + chunk, len(query_keys))
        q = torch.as_tensor(qn_all[lo:hi], device=device)
        sims = q @ kn.T
        if exclude_same_episode:
            qc = torch.as_tensor(query_codes[lo:hi], device=device)
            sims = sims.masked_fill(qc[:, None] == mc[None, :], -float("inf"))
        values, index = sims.max(dim=1)
        result[lo:hi] = index.cpu().numpy()
        sims_out[lo:hi] = values.cpu().numpy()
    return result, sims_out


# --------------------------------------------------------------------------
# Decoding and denoising
# --------------------------------------------------------------------------

def decode_bases(m1, bases, residuals, rows, *, batch=4096):
    """Dec(b, r, s) for bases (S,N,B) and residuals (S,N,R) -> (S,N,H,7)."""
    bases = np.asarray(bases, np.float32)
    residuals = np.asarray(residuals, np.float32)
    s_count, n = bases.shape[:2]
    out = np.empty((s_count, n, *m1.action_shape), np.float32)
    for s in range(s_count):
        for lo in range(0, n, batch):
            hi = min(n, lo + batch)
            out[s, lo:hi] = np.asarray(m1.decode(bases[s, lo:hi], residuals[s, lo:hi],
                                                 rows["f"][lo:hi], rows["s"][lo:hi]),
                                       np.float32).reshape(hi - lo, *m1.action_shape)
    return out


def denoise(flow, norm, rows, anchors, noise, t0, *, device):
    """Mixed/normal F from x_t0=(1-t0)*anchor+t0*eps (anchors None: pure noise)."""
    n = len(rows["a"])
    dim = int(np.prod(norm["action_shape"]))
    sources = None
    if anchors is not None:
        sources = _standardize(np.asarray(anchors, np.float32).reshape(-1, n, dim),
                               norm["a_mean"], norm["a_scale"])
    return af._rollout_flow(flow, norm, rows, sources, noise, np.full(n, t0, np.float32),
                            steps=10, device=device, chunk=QUERY_CHUNK)


# --------------------------------------------------------------------------
# One split x seed (M6/F_normal shared across subtask sources)
# --------------------------------------------------------------------------

def _split_rows(rows, split, *, seed, smoke, smoke_train_episodes):
    train_tasks = SPLITS[split][0]
    train = _subset(rows, np.isin(rows["task_index"], train_tasks))
    train = _subset(train, train["valid_h25"])
    if smoke:
        train = af._select_episodes(train, train_tasks, smoke_train_episodes, seed + 101)
    fit_ix, val_ix = _episode_train_validation_indices(train, seed + 401)
    train_fit = _subset(train, np.isin(np.arange(len(train["a"])), fit_ix))
    validation = _subset(train, np.isin(np.arange(len(train["a"])), val_ix))
    return train_fit, validation


def _attach_text(rows, texts_by_source, embedding_by_text, source):
    out = dict(rows)
    key = list(zip(map(str, rows["episode_id"]), map(int, rows["frame"])))
    out["subtask_text"] = np.asarray([texts_by_source[source][k] for k in key], dtype=str)
    out["subtask_embedding"] = np.stack([embedding_by_text[t] for t in out["subtask_text"]]).astype(np.float32)
    return out


def _flow_norm_seed(train_fit, train_actions, m1, seed):
    norm_ix, _ = _episode_train_validation_indices(train_fit, seed + 827 + 11)
    a_mean, a_scale = _standardize_fit(train_actions[norm_ix].reshape(len(norm_ix), -1))
    f_mean, f_scale = _standardize_fit(train_fit["f"][norm_ix])
    s_mean, s_scale = _standardize_fit(train_fit["s"][norm_ix])
    return {"a_mean": a_mean, "a_scale": a_scale, "f_mean": f_mean, "f_scale": f_scale,
            "s_mean": s_mean, "s_scale": s_scale, "action_shape": m1.action_shape}


def _aggregate(metrics, masks):
    return {pop: {k: (float(np.nanmean(v[mask])) if mask.any() else None) for k, v in metrics.items()}
            for pop, mask in masks.items()}


def run_split_seed(rows, *, split, seed, sources, texts_by_source, embedding_by_text,
                   device, scales, m1_epochs, f_epochs, batch_size, smoke,
                   smoke_train_episodes, query_limit, log):
    tic = time.perf_counter()
    train_fit, validation = _split_rows(rows, split, seed=seed, smoke=smoke,
                                        smoke_train_episodes=smoke_train_episodes)
    train_actions = train_fit["a"][:, :HORIZON].copy()
    m6, bases, _ = af._fit_m6_backbone_and_units(train_fit, seed=seed + 19, device=device,
                                                 epochs=m1_epochs)
    m1 = m6.backbone
    log(f"  split={split} seed={seed}: M6 fitted ({len(m6.prototypes)} units, eps={m6.epsilon:.4g}) "
        f"in {time.perf_counter() - tic:.0f}s")
    m1_hash = af.state_hash(m1)
    _, true_residual = m6.encode(train_actions, train_fit["s"])
    norm_seed = _flow_norm_seed(train_fit, train_actions, m1, seed)
    train_codes = encode_episode_codes(train_fit["episode_id"])

    # Query rows: all training intervals (optionally a fixed random subset).
    n_all = len(train_actions)
    q_ix = np.arange(n_all)
    if query_limit and n_all > query_limit:
        q_ix = np.sort(np.random.default_rng(seed + 5).choice(n_all, query_limit, replace=False))
    qrows = _subset(train_fit, np.isin(np.arange(n_all), q_ix))
    target = train_actions[q_ix]
    n = len(q_ix)
    rng_common = np.random.default_rng(seed * 1_000_003 + 17)
    residual_draws = rng_common.standard_normal((SAMPLES, n, m1.residual_dim)).astype(np.float32)
    noise = rng_common.standard_normal((SAMPLES, n, int(np.prod(m1.action_shape)))).astype(np.float32)

    results, f_normal, normal_norm, normal_fit = [], None, None, None
    for source in sources:
        t1 = time.perf_counter()
        tf = _attach_text(train_fit, texts_by_source, embedding_by_text, source)
        query_model, query_fit, train_keys = af._fit_query_encoder(
            tf, bases, seed=seed + 701, device=device,
            allowed_tasks=tuple(SPLITS[split][0]), subtask_source=source)
        temperature = query_fit["temperature"]
        bank = af.UnitBank(m6, float(np.sqrt(np.mean((bases - bases.mean(axis=0)) ** 2))))
        own_unit, own_distance, assigned, unit_rows = seed_bank_with_rows(
            bank, m6, bases, train_keys, train_fit["episode_id"])
        if not len(bank.ids()):
            raise RuntimeError("M6 produced no assignable training units")
        flow_sources = af._make_flow_sources(m1, bank, tf, bases, train_keys, temperature,
                                             norm_seed, seed=seed + 823)
        f_mixed, mixed_norm, mixed_fit = af.fit_anchor_flow(
            tf, train_actions, flow_sources, source_mix=SOURCE_MIX, seed=seed + 827,
            epochs=f_epochs, patience=8, batch_size=batch_size, device=device)
        if f_normal is None:
            # Identical architecture/rows/seed/epoch cap; ordinary path only.
            f_normal, normal_norm, normal_fit = af.fit_anchor_flow(
                tf, train_actions, flow_sources, source_mix=NORMAL_MIX, seed=seed + 827,
                epochs=f_epochs, patience=8, batch_size=batch_size, device=device)
        log(f"  split={split} seed={seed} source={source}: E+bank+F fitted in {time.perf_counter() - t1:.0f}s "
            f"(F mixed best epoch {mixed_fit['best_epoch']}, F_normal {normal_fit['best_epoch']})")
        hashes = (af.state_hash(m1), af.state_hash(query_model), af.state_hash(f_mixed), af.state_hash(f_normal))

        qkeys = train_keys[q_ix]
        q_own = own_unit[q_ix]
        q_assigned = assigned[q_ix]
        q_bases = bases[q_ix]
        masks = {"all": np.ones(n, bool), "assigned_within_epsilon": q_assigned}
        rec = {"split": split, "seed": seed, "subtask_source": source, "query_rows": int(n),
               "train_rows": int(n_all), "validation_rows_unused": int(len(validation["a"])),
               "unit_count": int(len(bank.ids())), "epsilon": float(m6.epsilon),
               "assigned_fraction": float(q_assigned.mean()),
               "query_encoder_epochs": int(query_fit["epochs"]), "temperature": float(temperature),
               "F_mixed_best_epoch": int(mixed_fit["best_epoch"]),
               "F_mixed_validation_loss": float(mixed_fit["best_validation_loss"]),
               "F_normal_best_epoch": int(normal_fit["best_epoch"]),
               "F_normal_validation_loss": float(normal_fit["best_validation_loss"]),
               "stages": [], "retrieval": {}}

        def add(stage, condition, pred, t0=None, extra=None):
            m = action_metrics(pred, target, scales)
            rec["stages"].append({"stage": stage, "condition": condition, "t0": t0,
                                  "metrics": _aggregate(m, masks), **(extra or {})})

        # Condition-independent stages.
        gold_anchor = decode_bases(m1, np.broadcast_to(q_bases[None], (SAMPLES, *q_bases.shape)),
                                   residual_draws, qrows)
        add("0a_encode_true_r", "both", decode_bases(m1, q_bases[None], true_residual[q_ix][None], qrows))
        add("0_encode", "both", gold_anchor)
        for t0 in T0S:
            add("4_from_0_encode", "both", denoise(f_mixed, mixed_norm, qrows, gold_anchor, noise, t0, device=device), t0)
        add("5_F_normal_from_noise", "both", denoise(f_normal, normal_norm, qrows, None, noise, 1.0, device=device), 1.0)
        add("5_F_mixed_from_noise", "both", denoise(f_mixed, mixed_norm, qrows, None, noise, 1.0, device=device), 1.0)

        for condition in CONDITIONS:
            exclude = condition.startswith("b_")
            ret = retrieve(bank, qkeys, qrows["episode_id"], q_own, unit_rows, train_codes,
                           temperature, exclude_same_episode=exclude,
                           rng=np.random.default_rng(seed * 1_000_003 + (29 if exclude else 23)))
            top1_match = ret["top1"] == q_own
            rec["retrieval"][condition] = {
                pop: {"top1_equals_own_unit": float(top1_match[mask].mean()) if mask.any() else None,
                      "w_own_unit": float(ret["w_own"][mask].mean()) if mask.any() else None,
                      "w_max": float(ret["w_max"][mask].mean()) if mask.any() else None,
                      "normalized_entropy": float((ret["entropy"][mask] / np.log(np.maximum(ret["unit_count"][mask], 2))).mean()) if mask.any() else None,
                      "own_unit_present": float(ret["own_present"][mask].mean()) if mask.any() else None,
                      "instance_is_self": float((ret["instance_row"][mask] == q_ix[mask]).mean()) if mask.any() else None,
                      "instance_same_episode": float((train_codes[ret["instance_row"][mask]] == train_codes[q_ix][mask]).mean()) if mask.any() else None,
                      "instance_same_task": float((train_fit["task_index"][ret["instance_row"][mask]] == qrows["task_index"][mask]).mean()) if mask.any() else None}
                for pop, mask in {**masks, "top1_correct": top1_match, "top1_wrong": ~top1_match}.items()}
            own_center = np.where(ret["own_present"][:, None], ret["own_center"], q_bases)  # absent -> NaN rows below
            own_anchor = decode_bases(m1, np.broadcast_to(own_center[None], (SAMPLES, *own_center.shape)),
                                      residual_draws, qrows)
            own_anchor[:, ~ret["own_present"]] = np.nan
            top1_anchor = decode_bases(m1, np.broadcast_to(ret["top1_center"][None], (SAMPLES, n, own_center.shape[1])),
                                       residual_draws, qrows)
            comp_rng = np.random.default_rng(seed * 1_000_003 + 31)
            comp_b = ret["comp_center"] + comp_rng.standard_normal(ret["comp_center"].shape).astype(np.float32) * ret["comp_sigma"][..., None]
            comp_anchor = decode_bases(m1, comp_b, residual_draws, qrows)
            inst_action = train_actions[ret["instance_row"]]
            inst_anchor = np.broadcast_to(inst_action[None], (SAMPLES, *inst_action.shape))
            inst_dec = decode_bases(m1, np.broadcast_to(bases[ret["instance_row"]][None], (SAMPLES, n, bases.shape[1])),
                                    residual_draws, qrows)
            raw_ix, raw_sim = raw_nearest(qkeys, train_keys, train_codes[q_ix], train_codes,
                                          exclude_same_episode=exclude, device=device)
            raw_action = train_actions[raw_ix]
            raw_anchor = np.broadcast_to(raw_action[None], (SAMPLES, *raw_action.shape))
            rec["retrieval"][condition]["raw_nn"] = {
                "is_self": float((raw_ix == q_ix).mean()),
                "same_task": float((train_fit["task_index"][raw_ix] == qrows["task_index"]).mean()),
                "mean_similarity": float(raw_sim.mean())}
            if (~ret["own_present"]).any():
                own_anchor_safe = np.nan_to_num(own_anchor)
            else:
                own_anchor_safe = own_anchor
            add("1_own_unit_center", condition, own_anchor)
            add("3_top1_center", condition, top1_anchor)
            add("3_component_sample", condition, comp_anchor)
            add("3p_two_stage_instance", condition, inst_anchor)
            add("3p_two_stage_instance_decoded", condition, inst_dec)
            add("5_raw_nn_action", condition, raw_anchor)
            for t0 in T0S:
                own_pred = denoise(f_mixed, mixed_norm, qrows, own_anchor_safe, noise, t0, device=device)
                own_pred[:, ~ret["own_present"]] = np.nan
                add("4_from_1_own_center", condition, own_pred, t0)
                add("4_from_3_top1_center", condition, denoise(f_mixed, mixed_norm, qrows, top1_anchor, noise, t0, device=device), t0)
                add("4_from_3_component", condition, denoise(f_mixed, mixed_norm, qrows, comp_anchor, noise, t0, device=device), t0)
                add("4_from_3p_instance", condition, denoise(f_mixed, mixed_norm, qrows, inst_anchor, noise, t0, device=device), t0)
                add("5_RAGDP_raw_nn_F_mixed", condition, denoise(f_mixed, mixed_norm, qrows, raw_anchor, noise, t0, device=device), t0)
                add("5_RAGDP_raw_nn_F_normal", condition, denoise(f_normal, normal_norm, qrows, raw_anchor, noise, t0, device=device), t0)
        if hashes != (af.state_hash(m1), af.state_hash(query_model), af.state_hash(f_mixed), af.state_hash(f_normal)) \
                or af.state_hash(m1) != m1_hash:
            raise AssertionError("a frozen model changed during evaluation")
        rec["duration_seconds"] = time.perf_counter() - t1
        log(f"  split={split} seed={seed} source={source}: evaluated {n} rows in {rec['duration_seconds']:.0f}s")
        results.append(rec)
    return results, time.perf_counter() - tic


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

STAGE_ORDER = [
    ("0a_encode_true_r", "both", None, "0a 符号化（r も正解）"),
    ("0_encode", "both", None, "0 符号化 Dec(Enc(A,s))"),
    ("1_own_unit_center", None, None, "1 自分の単位の中心"),
    ("3_top1_center", None, None, "3 取り出し top-1 の中心"),
    ("3_component_sample", None, None, "3 成分サンプル"),
    ("3p_two_stage_instance", None, None, "3' 2段（単位→最も近い区間の行動）"),
    ("3p_two_stage_instance_decoded", None, None, "3' 2段（区間の b を復号）"),
    ("5_raw_nn_action", None, None, "5 生の区間の最近傍（行動そのまま）"),
]


def _mean_sd(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(records):
    """Mean +- seed SD for every (split, source, condition, stage, t0, population)."""
    table = {}
    for rec in records:
        for st in rec["stages"]:
            conds = CONDITIONS if st["condition"] == "both" else (st["condition"],)
            for cond in conds:
                for pop, metrics in st["metrics"].items():
                    key = (rec["split"], rec["subtask_source"], cond, st["stage"], st["t0"], pop)
                    for mk, v in metrics.items():
                        table.setdefault(key, {}).setdefault(mk, []).append(v)
    out = []
    for key, metrics in sorted(table.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        split, source, cond, stage, t0, pop = key
        out.append({"split": split, "subtask_source": source, "condition": cond, "stage": stage,
                    "t0": t0, "population": pop, "n_seeds": len(next(iter(metrics.values()))),
                    **{mk: dict(zip(("mean", "sd"), _mean_sd(v))) for mk, v in metrics.items()}})
    return out


def _fmt(cell, scale=1.0, digits=3):
    if cell is None or cell.get("mean") is None:
        return "—"
    mean, sd = cell["mean"] * scale, (cell["sd"] or 0.0) * scale
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


def _decomposition_rows(summary, split, source, cond, pop="all", best_t0_metric="mse"):
    by = {(r["stage"], r["t0"]): r for r in summary
          if r["split"] == split and r["subtask_source"] == source and r["condition"] == cond and r["population"] == pop}
    rows = []
    for stage, _, _, label in STAGE_ORDER:
        if (stage, None) in by:
            rows.append((label, None, by[(stage, None)]))
    final = [("4_from_0_encode", "4 ← 0 符号化"), ("4_from_1_own_center", "4 ← 1 自分の単位の中心"),
             ("4_from_3_top1_center", "4 ← 3 top-1 中心"), ("4_from_3_component", "4 ← 3 成分サンプル"),
             ("4_from_3p_instance", "4 ← 3' 2段"), ("5_RAGDP_raw_nn_F_mixed", "5 RAGDP 型（生の最近傍, 混合 F）"),
             ("5_RAGDP_raw_nn_F_normal", "5 RAGDP 型（生の最近傍, F_normal）")]
    for stage, label in final:
        for t0 in T0S:
            if (stage, t0) in by:
                rows.append((label, t0, by[(stage, t0)]))
    for stage, label in (("5_F_normal_from_noise", "5 F_normal ノイズから"), ("5_F_mixed_from_noise", "5 混合 F ノイズから")):
        if (stage, 1.0) in by:
            rows.append((label, 1.0, by[(stage, 1.0)]))
    return rows


def write_report(path, payload):
    summary, records = payload["summary"], payload["records"]
    scales = payload["physical_scales"]
    lines = ["# 往復テスト：デモ → スキルメモリ → 同じ入力で取り出し → 最終行動", "",
             f"出力：`{path.parent}`。seed={payload['seeds']}、split={payload['splits']}、サブタスク={payload['sources']}、"
             f"smoke={payload['smoke']}。所要 {payload['duration_seconds'] / 60:.1f} 分。", "",
             "## 手順", "",
             "- 学習 suite のデモだけを使う（Split A：object+spatial、Split B：object+goal）。S1 と同じく学習 suite を episode 単位で fit/validation に分け、fit 側で M1（b=12, r=4）・M6（κ=0.1, m=10, e=3）・E・混合 F（0.65/0.05/0.30）を S1 のコード（anchor_flow.py を import）で作る。validation 側は使わない。",
             "- F_normal：同じ FlowActionHead・同じ行・同じ seed・エポック上限 80・patience 8 で、通常の経路（x_t = tε + (1−t)A、損失 ‖F − (ε − A)‖²）だけで学習。",
             "- M6 の単位と F_normal は gold と instruction で共通（M6 は行動・状態・特徴だけで決まり、同じ seed）。E と混合 F はサブタスク条件ごとに学習。",
             "- クエリ：fit 側の全区間（学習区間）を、同じ o（画像・サブタスク・状態）でクエリ。u(A) = M6 の機能的距離で最も近い単位（_nearest）。ε 以内の区間だけが単位のメンバー（鍵・σ に入る）。",
             "- 条件 (a)：同じエピソードもメモリにある（中心は M6 の prototype）。(b)：同じエピソードの区間を値・鍵・σ から除く（中心はメンバー平均。anchor_flow.UnitBank の leave-out と同じ）。生の区間の最近傍も (b) では同じエピソードを除く。",
             "- 復号のアンカーは r ~ N(0, I) を 8 本（全段で同じ乱数）。デノイズは 10 Euler ステップ、8 本のノイズは全手法で共通。各値は 8 本の平均（best-of-8 は results.json）。",
             "", "## 指標", "",
             "- MSE：S1 と同じ（LIBERO の [-1,1] 行動空間、25×7 要素の平均二乗誤差、8 本の平均）。",
             f"- 位置：行動の差の累積 Σ_τ≤t (â−a)[:3] を、学習 suite の env_state から最小二乗で当てた換算係数（m / 行動1単位・1ステップ、軸ごと {np.round(scales['pos_m_per_action_unit_step'], 4).tolist()}、相関 {np.round(scales['pos_fit_correlation'], 2).tolist()}）で m に直し、ユークリッド距離を全25ステップで平均（チャンク）と、25ステップ目（終端）。表は mm。指令値の累積からの推定で、実際の手先の軌跡ではない。",
             f"- 回転：同様に Σ(â−a)[3:6] × 換算係数（rad / 単位・ステップ、{np.round(scales['rot_rad_per_action_unit_step'], 4).tolist()}、相関 {np.round(scales['rot_fit_correlation'], 2).tolist()}）の回転ベクトルのノルム（小角近似）。表は度。",
             "- グリッパー：各ステップの符号の一致率（正解は ±1）。", ""]

    def table(title, metric, scale, digits, cond, pop="all"):
        lines.extend([f"### {title}", "",
                      "| 段階 | t0 | " + " | ".join(f"{s} {src}" for s in payload['splits'] for src in payload['sources']) + " |",
                      "|---|---:|" + "---:|" * (len(payload['splits']) * len(payload['sources']))])
        cols = [(s, src) for s in payload['splits'] for src in payload['sources']]
        per_col = [dict(((lab, t0), r) for lab, t0, r in _decomposition_rows(summary, s, src, cond, pop)) for s, src in cols]
        order = [(lab, t0) for lab, t0, _ in _decomposition_rows(summary, cols[0][0], cols[0][1], cond, pop)]
        for lab, t0 in order:
            cells = [_fmt(pc.get((lab, t0), {}).get(metric), scale, digits) for pc in per_col]
            lines.append(f"| {lab} | {'' if t0 is None else t0} | " + " | ".join(cells) + " |")
        lines.append("")

    for cond, cname in ((CONDITIONS[1], "(b) 同じエピソードを除いたメモリ"), (CONDITIONS[0], "(a) 同じエピソードもメモリに（純粋な往復）")):
        lines.extend([f"## 分解表 {cname}（全区間、平均±seed間SD）", ""])
        table("行動 MSE", "mse", 1.0, 3, cond)
        table("位置の誤差・終端（mm）", "pos_terminal_m", 1000.0, 1, cond)
        table("位置の誤差・チャンク平均（mm）", "pos_chunk_m", 1000.0, 1, cond)
        table("回転の誤差・終端（度）", "rot_terminal_rad", 180.0 / np.pi, 2, cond)
        table("グリッパー符号の一致率", "gripper_agreement", 1.0, 3, cond)

    lines.extend(["## 取り出しの一致（段階 2、平均±seed間SD）", "",
                  "| split | サブタスク | 条件 | top-1 = u(A) | w[u(A)] | max w | 正規化エントロピー | 3' が自分自身 | 3' が同じタスク | 生の最近傍が自分 | 生の最近傍が同じタスク | ε 以内の区間の割合 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for split in payload["splits"]:
        for src in payload["sources"]:
            recs = [r for r in records if r["split"] == split and r["subtask_source"] == src]
            for cond in CONDITIONS:
                def ms(getter):
                    m, s = _mean_sd([getter(r) for r in recs])
                    return "—" if m is None else f"{m:.3f}±{s:.3f}"
                g = lambda k: (lambda r: r["retrieval"][cond]["all"][k])
                lines.append(f"| {split} | {src} | {cond[:1]} | {ms(g('top1_equals_own_unit'))} | {ms(g('w_own_unit'))} | {ms(g('w_max'))} | "
                             f"{ms(g('normalized_entropy'))} | {ms(g('instance_is_self'))} | {ms(g('instance_same_task'))} | "
                             f"{ms(lambda r: r['retrieval'][cond]['raw_nn']['is_self'])} | {ms(lambda r: r['retrieval'][cond]['raw_nn']['same_task'])} | "
                             f"{ms(lambda r: r['assigned_fraction'])} |")
    lines.extend(["", "## 実行情報", "",
                  "| split | seed | サブタスク | 区間 | 単位 | ε | E epochs | F 混合 best epoch | F_normal best epoch | 秒 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in records:
        lines.append(f"| {r['split']} | {r['seed']} | {r['subtask_source']} | {r['query_rows']} | {r['unit_count']} | {r['epsilon']:.4g} | "
                     f"{r['query_encoder_epochs']} | {r['F_mixed_best_epoch']} | {r['F_normal_best_epoch']} | {r['duration_seconds']:.0f} |")
    lines.extend(["", "ε 以内の区間だけの集計、top-1 の正誤別の取り出し統計、best-of-8 MSE は results.json にある。", "",
                  "## 再現", "", "```", payload["command"], "```", ""])
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _precompute_texts(rows, sources, cache_dir):
    texts_by_source, unique = {}, set()
    for source in sources:
        attached = af.attach_subtask_source(rows, source, split="A", cache_dir=cache_dir,
                                            annotations_path=DEFAULT_ANNOTATIONS, predicted_root=None)
        texts_by_source[source] = {(str(e), int(f)): str(t) for e, f, t in
                                   zip(attached["episode_id"], attached["frame"], attached["subtask_text"])}
        unique.update(map(str, attached["subtask_text"]))
    unique = sorted(unique)
    embeddings = frozen_pi05_plan_embeddings(unique, params_path=DEFAULT_PI05_BASE_PARAMS,
                                             tokenizer_path=DEFAULT_PALIGEMMA_TOKENIZER)
    return texts_by_source, dict(zip(unique, embeddings))


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE, dataset_dir=DEFAULT_DATASET,
        seeds=(0, 1, 2), splits=("A", "B"), sources=("gold", "instruction"), device_name="cuda",
        m1_epochs=60, f_epochs=80, batch_size=512, smoke=False, smoke_train_episodes=3,
        query_limit=0, threads=6, command=""):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(device_name)
    start = time.perf_counter()
    output_root = Path(output_root)
    output = output_root / datetime.now(timezone.utc).strftime("roundtrip-%Y%m%dT%H%M%SZ")
    if smoke:
        output = output.with_name(output.name + "-smoke")
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "progress.log"

    def log(message):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    rows = load_rows(cache_dir, dataset_dir, max_horizon=HORIZON)
    train_tasks = sorted({t for s in splits for t in SPLITS[s][0]})
    rows = _subset(rows, np.isin(rows["task_index"], train_tasks) & rows["valid_h25"])
    log(f"loaded {len(rows['a'])} training-suite rows in {time.perf_counter() - start:.0f}s")
    texts_by_source, embedding_by_text = _precompute_texts(rows, sources, cache_dir)
    log(f"text embeddings: {len(embedding_by_text)} unique texts")
    scales = fit_physical_scales(cache_dir, np.unique(rows["episode_id"]))
    log(f"physical scales: {scales}")
    records, timings = [], []
    for split in splits:
        for seed in seeds:
            recs, seconds = run_split_seed(
                rows, split=split, seed=seed, sources=sources, texts_by_source=texts_by_source,
                embedding_by_text=embedding_by_text, device=device, scales=scales,
                m1_epochs=m1_epochs, f_epochs=f_epochs, batch_size=batch_size, smoke=smoke,
                smoke_train_episodes=smoke_train_episodes, query_limit=query_limit, log=log)
            records.extend(recs)
            timings.append({"split": split, "seed": seed, "seconds": seconds})
            log(f"done split={split} seed={seed} in {seconds / 60:.1f} min")
            payload = {"spec": "roundtrip test; /work/docs/research/anchor_flow_skill_memory.md §2, §4 S1",
                       "seeds": list(seeds), "splits": list(splits), "sources": list(sources),
                       "smoke": bool(smoke), "t0_grid": list(T0S), "samples": SAMPLES,
                       "flow_steps": 10, "source_mix_mixed_F": list(SOURCE_MIX),
                       "conditions": list(CONDITIONS), "physical_scales": scales,
                       "m1_epochs": m1_epochs, "f_epochs": f_epochs, "query_limit": query_limit,
                       "timings": timings, "duration_seconds": time.perf_counter() - start,
                       "command": command, "records": records, "summary": summarize(records)}
            (output / "results.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
            write_report(output / "summary.md", payload)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--subtask-sources", default="gold,instruction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--flow-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-episodes", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=0, help="0 = every training interval")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    sources = tuple(x.strip() for x in args.subtask_sources.split(",") if x.strip())
    if any(s not in SPLITS for s in splits) or any(s not in ("gold", "instruction", "none") for s in sources):
        parser.error("bad --splits or --subtask-sources")
    import sys
    output = run(output_root=args.output_root, cache_dir=args.cache_dir, dataset_dir=args.dataset,
                 seeds=seeds, splits=splits, sources=sources, device_name=args.device,
                 m1_epochs=args.m1_epochs, f_epochs=args.flow_epochs, batch_size=args.batch_size,
                 smoke=args.smoke, smoke_train_episodes=args.smoke_train_episodes,
                 query_limit=args.query_limit, threads=args.threads,
                 command="python -m skill_moe.skillnet.examples.emergent.light.intermediate.roundtrip " + " ".join(sys.argv[1:] if argv is None else argv))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
