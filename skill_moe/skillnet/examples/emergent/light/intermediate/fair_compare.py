"""Fair comparison: confidence-adaptive anchor start vs. the ordinary flow policy.

Lightweight setting (frozen pi0.5 VLM features + small MLPs, offline). The
proposal, exactly as specified by the researcher:

    q   = E(o)                           image key k (language-free), subtask text, state
    w_k = softmax(cos(q, kappa_k)/tau),  confidence c (max |cos| or 1 - H(w)/log K)
    b   = c_{top-1}                      top-1 unit centre in M1 base space
    a   = Dec(b, r=0, s)                 M1 decoder into action space
    t0  = g(c),  n = max(1, round(10 t0))   (step size equal to the ordinary 0.1)
    x   = (1 - t0) a + t0 eps,  Euler from t0 to 0 in n steps

g is calibrated ONLY on training-suite validation episodes (confidence bins,
per-bin validation-MSE-optimal t0, then the monotone (non-increasing in
confidence) assignment with the least total validation error, solved exactly
by DP). Fixed t0 in {0.2, 0.4, 0.6} is reported too.

Fairness: F_normal (ordinary path only) and F_anchor (S1 mixture) share the
architecture, rows, seed, init, data order, epoch cap and early-stopping rule
(anchor_flow.fit_anchor_flow). Every method on a row sees the same eps (paired),
the same E, the same subtask source. The ordinary policy is also run with
n = 1..10 steps so it can be matched to each proposal's mean step count.

Conditions: D1 held-out training-suite episodes (memory = training suite);
D2 new suite, prequential insertion (S1 stream); D3 new suite, no insertion.

anchor_flow.py / roundtrip.py are imported (or not at all), never edited.
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
from .clip_retrieval import (DEFAULT_ANNOTATIONS, DEFAULT_PALIGEMMA_TOKENIZER,
                             DEFAULT_PI05_BASE_PARAMS, frozen_pi05_plan_embeddings)
from .splits import SPLITS, subset_rows as _subset
from .physical_scales import fit_physical_scales as _fit_physical_scales
from .experiment import (DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT,
                         _episode_train_validation_indices, _standardize,
                         _standardize_fit, load_rows)

HORIZON = 25
SAMPLES = 8
N_REF = 10
T0_GRID = tuple(round(0.1 * i, 1) for i in range(1, 11))      # 0.1 .. 1.0
FIXED_T0 = (0.2, 0.4, 0.6)
ANCHOR_MIX = (0.65, 0.05, 0.30)   # S1: retrieved, gold-b, ordinary path
NORMAL_MIX = (0.0, 0.0, 1.0)
CAL_BINS = 8
HOLDOUT_PER_TASK = 5
QUERY_CHUNK = 128
CHANGE_BAND = 0.05                # |relative per-row MSE change| below this = unchanged
ANCHORS = ("unit", "raw", "shuf", "gold_unit")
RULES = ("fix0.2", "fix0.4", "fix0.6", "cal_sim", "cal_ent")
SOURCES_DEFAULT = ("instruction", "gold")


def steps_for(t0):
    """n = max(1, round(10 t0)); with t0 on the 0.1 grid the step size is 0.1."""
    return np.maximum(1, np.rint(N_REF * np.asarray(t0, np.float64))).astype(np.int64)


# --------------------------------------------------------------------------
# Rollout (batched, per-row t0 and n)
# --------------------------------------------------------------------------

def rollout(model, norm, f, s, anchors, eps, t0, n=None, *, device, chunk=QUERY_CHUNK):
    """Euler flow from x_t0=(1-t0)a+t0*eps (anchors None: pure noise).

    f, s, anchors: (N, ...) raw (anchors in action units, (N,H,7)); eps (S,N,D)
    standardized noise; t0 (N,); n (N,) steps (default steps_for(t0)). Uses
    the same arithmetic as anchor_flow._rollout_flow so t0=1,n=10 reproduces it.
    Returns (S,N,H,7) in action units.
    """
    eps = np.asarray(eps, np.float32)
    samples, count, dim = eps.shape
    t0 = np.broadcast_to(np.asarray(t0, np.float32), (count,)).copy()
    n = steps_for(t0) if n is None else np.broadcast_to(np.asarray(n, np.int64), (count,)).copy()
    out = np.empty((samples, count, dim), np.float32)
    fs = _standardize(f, norm["f_mean"], norm["f_scale"])
    ss = _standardize(s, norm["s_mean"], norm["s_scale"])
    src_all = None
    if anchors is not None:
        src_all = _standardize(np.asarray(anchors, np.float32).reshape(count, dim),
                               norm["a_mean"], norm["a_scale"])
    for lo in range(0, count, chunk):
        hi = min(count, lo + chunk)
        c = hi - lo
        e = torch.as_tensor(eps[:, lo:hi].reshape(samples * c, dim), device=device)
        t0_t = torch.as_tensor(t0[lo:hi], device=device).repeat(samples)
        n_np = np.tile(n[lo:hi], samples)
        n_t = torch.as_tensor(n_np, device=device)
        nf = n_t.to(torch.float32)
        if src_all is None:
            x = e.clone()
        else:
            src = torch.as_tensor(src_all[lo:hi], device=device).repeat(samples, 1)
            x = t0_t[:, None] * e + (1 - t0_t[:, None]) * src
        ft = torch.as_tensor(fs[lo:hi], device=device).repeat(samples, 1)
        st = torch.as_tensor(ss[lo:hi], device=device).repeat(samples, 1)
        dt = -t0_t / nf
        with torch.no_grad():
            for j in range(int(n_np.max())):
                frac = torch.as_tensor((1 - j / n_np).astype(np.float32), device=device)
                t_now = t0_t * frac
                active = (j < n_t)[:, None]
                x = torch.where(active, x + dt[:, None] * model(x, t_now, ft, st), x)
        out[:, lo:hi] = x.reshape(samples, c, dim).cpu().numpy()
    out = out * norm["a_scale"] + norm["a_mean"]
    return out.reshape(samples, count, *norm["action_shape"])


# --------------------------------------------------------------------------
# Calibration of g (training-suite validation episodes only)
# --------------------------------------------------------------------------

def assert_calibration_rows(rows, split, forbidden_episodes=()):
    tasks = set(np.unique(rows["task_index"]).tolist())
    allowed = set(SPLITS[split][0])
    if not tasks <= allowed:
        raise ValueError(f"calibration rows include non-training-suite tasks {sorted(tasks - allowed)}")
    overlap = set(map(str, rows["episode_id"])) & set(map(str, forbidden_episodes))
    if overlap:
        raise ValueError(f"calibration rows overlap evaluation episodes: {sorted(overlap)[:3]}")


def fit_monotone_g(confidence, err_by_t0, grid=T0_GRID, bins=CAL_BINS):
    """Bins by confidence quantile; t0 non-increasing in confidence; exact DP.

    err_by_t0: (len(grid), N) per-row validation error. Returns dict with
    'edges' (bins-1 inner edges) and 't0' per bin (low->high confidence).
    """
    confidence = np.asarray(confidence, np.float64)
    err = np.asarray(err_by_t0, np.float64)
    grid = np.asarray(grid, np.float64)
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, confidence, side="right")
    nb = len(edges) + 1
    cost = np.zeros((nb, len(grid)))
    for b in range(nb):
        m = which == b
        if m.any():
            cost[b] = err[:, m].sum(axis=1)
    # Order grid descending: bin b must use index >= index used by bin b-1.
    order = np.argsort(-grid)
    c = cost[:, order]
    dp = np.full_like(c, np.inf)
    arg = np.zeros(c.shape, np.int64)
    dp[0] = c[0]
    for b in range(1, nb):
        best_prev = np.minimum.accumulate(dp[b - 1])
        best_idx = np.zeros(len(grid), np.int64)
        cur = 0
        for j in range(len(grid)):
            if dp[b - 1, j] < dp[b - 1, cur]:
                cur = j
            best_idx[j] = cur
        dp[b] = c[b] + best_prev
        arg[b] = best_idx
    j = int(np.argmin(dp[-1]))
    chosen = [0] * nb
    for b in range(nb - 1, -1, -1):
        chosen[b] = j
        j = int(arg[b, j]) if b else j
    t0 = [float(grid[order[k]]) for k in chosen]
    per_bin_best = [float(grid[int(np.argmin(cost[b]))]) for b in range(nb)]
    return {"edges": edges.tolist(), "t0": t0, "per_bin_unconstrained_best": per_bin_best,
            "counts": np.bincount(which, minlength=nb).tolist(),
            "validation_mse": float(dp[-1].min() / max(len(confidence), 1))}


def apply_g(g, confidence):
    which = np.searchsorted(np.asarray(g["edges"]), np.asarray(confidence), side="right")
    return np.asarray(g["t0"], np.float32)[which]


def conf_bin(g, confidence):
    return np.searchsorted(np.asarray(g["edges"]), np.asarray(confidence), side="right")


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _nrm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cosine_stats(q, keys, tau, *, device, chunk=1024):
    """Top-1 index, max cos, normalized-entropy confidence 1-H/log K."""
    kt = torch.as_tensor(_nrm(keys), device=device)
    top, mx, conf = [], [], []
    log_k = max(float(np.log(len(keys))), 1e-8)
    for lo in range(0, len(q), chunk):
        qt = torch.as_tensor(_nrm(q[lo:lo + chunk]), device=device)
        cos = qt @ kt.T
        m, i = cos.max(dim=1)
        logp = torch.log_softmax(cos / tau, dim=1)
        h = -(logp.exp() * logp).sum(dim=1)
        top.append(i.cpu().numpy()); mx.append(m.cpu().numpy())
        conf.append((1 - h / log_k).cpu().numpy())
    return np.concatenate(top), np.concatenate(mx), np.concatenate(conf)


def functional_embeddings(m6, bases, chunk=256):
    """Batched M6 decoder signature (same as M6._functional_embedding)."""
    probe = m6._functional_probe
    p = len(probe["state"])
    out = []
    for lo in range(0, len(bases), chunk):
        b = np.asarray(bases[lo:lo + chunk], np.float32)
        rep = np.repeat(b, p, axis=0)
        dec = np.asarray(m6._decode(rep, np.tile(probe["residual"], (len(b), 1)),
                                    np.tile(probe["feature"], (len(b), 1)),
                                    np.tile(probe["state"], (len(b), 1))), np.float32)
        dec = dec.reshape(len(b), -1)
        out.append(dec / np.sqrt(dec.shape[1]))
    return np.concatenate(out)


def gold_unit_index(m6, gold_b, centers, *, device):
    """Functional nearest unit centre to the true interval's code."""
    ce = torch.as_tensor(functional_embeddings(m6, centers), device=device)
    out = []
    for lo in range(0, len(gold_b), 512):
        ge = torch.as_tensor(functional_embeddings(m6, gold_b[lo:lo + 512]), device=device)
        out.append(torch.cdist(ge, ce).argmin(dim=1).cpu().numpy())
    return np.concatenate(out)


def derangement(k, rng):
    order = rng.permutation(k)
    perm = np.empty(k, np.int64)
    perm[order] = np.roll(order, -1)
    return perm if k > 1 else np.zeros(1, np.int64)


def retrieve_static(view, q, gold_b, raw_keys, raw_actions, tau, m6, rng, *, device):
    ids, centers, keys = view
    top, mx, ent = cosine_stats(q, keys, tau, device=device)
    perm = derangement(len(ids), rng)
    gidx = gold_unit_index(m6, gold_b, centers, device=device)
    rtop, rmx, rent = cosine_stats(q, raw_keys, tau, device=device)
    return {"unit_b": centers[top], "shuf_b": centers[perm[top]], "gold_b_unit": centers[gidx],
            "unit_id": ids[top], "gold_unit_id": ids[gidx],
            "conf_sim": mx, "conf_ent": ent, "raw_action": raw_actions[rtop],
            "raw_conf_sim": rmx, "raw_conf_ent": rent,
            "post_add": np.zeros(len(q), bool), "unit_count": np.full(len(q), len(ids))}


def stream_d2(m6, banks, test, test_b, keys_by_source, train_keys_by_source, train_actions,
              tau_by_source, *, initial_ids, seed, device):
    """S1 prequential stream (episode blocks: retrieve all, then insert).

    Unit formation in M6 does not depend on query keys, so one stream feeds
    one UnitBank per subtask source. Raw memory grows with earlier episodes.
    """
    rng = np.random.default_rng(seed + 853)
    episodes = np.asarray(test["episode_id"]).astype(str)
    blocks = []
    for ep in rng.permutation(np.unique(episodes)):
        ix = np.flatnonzero(episodes == ep)
        blocks.append(ix[np.argsort(test["frame"][ix], kind="stable")])
    order = np.concatenate(blocks)
    n = len(episodes)
    sources = list(banks)
    res = {src: {"unit_b": np.zeros((n, test_b.shape[1]), np.float32),
                 "shuf_b": np.zeros((n, test_b.shape[1]), np.float32),
                 "gold_b_unit": np.zeros((n, test_b.shape[1]), np.float32),
                 "unit_id": np.zeros(n, np.int64), "gold_unit_id": np.zeros(n, np.int64),
                 "conf_sim": np.zeros(n, np.float32), "conf_ent": np.zeros(n, np.float32),
                 "raw_action": np.zeros((n, HORIZON, 7), np.float32),
                 "raw_conf_sim": np.zeros(n, np.float32), "raw_conf_ent": np.zeros(n, np.float32),
                 "post_add": np.zeros(n, bool), "unit_count": np.zeros(n, np.int64)}
           for src in sources}
    initial = set(int(x) for x in initial_ids)
    perm_rng = np.random.default_rng(seed + 859)
    raw_k = {src: np.concatenate([train_keys_by_source[src], keys_by_source[src][order]]) for src in sources}
    raw_a = np.concatenate([train_actions, test["a"][order, :HORIZON]])
    used = len(train_actions)
    for ix in blocks:
        gold_cache = None
        for src in sources:
            ids, centers, keys, _, _ = banks[src].centers_keys_sigmas()
            q = keys_by_source[src][ix]
            top, mx, ent = cosine_stats(q, keys, tau_by_source[src], device=device)
            perm = derangement(len(ids), perm_rng)
            if gold_cache is None or gold_cache[0] != tuple(ids):
                gold_cache = (tuple(ids), gold_unit_index(m6, test_b[ix], centers, device=device))
            gidx = gold_cache[1]
            rtop, rmx, rent = cosine_stats(q, raw_k[src][:used], tau_by_source[src], device=device)
            r = res[src]
            r["unit_b"][ix] = centers[top]; r["shuf_b"][ix] = centers[perm[top]]
            r["gold_b_unit"][ix] = centers[gidx]; r["unit_id"][ix] = ids[top]
            r["gold_unit_id"][ix] = ids[gidx]
            r["conf_sim"][ix] = mx; r["conf_ent"][ix] = ent
            r["raw_action"][ix] = raw_a[rtop]; r["raw_conf_sim"][ix] = rmx; r["raw_conf_ent"][ix] = rent
            r["post_add"][ix] = ~np.isin(ids[top], list(initial))
            r["unit_count"][ix] = len(ids)
        before = len(m6.unit_registration_events)
        main = sources[0]
        for old in ix:
            meta = {"episode_id": str(test["episode_id"][old]), "init_state_id": str(test["episode_id"][old]),
                    "stream_index": int(old), "query_key": keys_by_source[main][old].copy()}
            m6.memory_add(keys_by_source[main][old], test_b[old], meta)
        pending = int(m6._pending_since_cluster)
        if pending:
            orig = m6.pending_batch_size
            try:
                m6.pending_batch_size = pending
                m6._cluster_pending(force=True)
            finally:
                m6.pending_batch_size = orig
        for event in m6.unit_registration_events[before:]:
            sid = event.get("stream_index")
            if sid is None:
                continue
            for src in sources:
                banks[src].add(event["unit_id"], event["base"], keys_by_source[src][int(sid)],
                               event.get("episode_id", ""), stream_id=int(sid))
        for old in ix:
            uid, dist = None, None
            for src in sources:
                if int(old) in banks[src].seen_stream_ids:
                    continue
                if uid is None:
                    uid, dist = m6._nearest(test_b[old])
                if uid is not None and dist <= m6.epsilon:
                    banks[src].add(uid, test_b[old], keys_by_source[src][old],
                                   test["episode_id"][old], stream_id=int(old))
        used += len(ix)
    return res, order


# --------------------------------------------------------------------------
# Metrics (torch, per row; averaged over the 8 samples)
# --------------------------------------------------------------------------

METRICS = ("mse", "best8", "mse_pos", "mse_rot", "mse_grip", "pos_chunk_mm", "pos_term_mm",
           "rot_term_deg", "grip_agree", "dn", "dn_pos", "dn_rot", "dn_grip",
           "spread", "spread_pos_term_mm", "grip_split")


def row_metrics(pred, target, normal, scales, *, device):
    """pred/normal (S,N,H,7), target (N,H,7) -> dict of (N,) arrays."""
    p = torch.as_tensor(pred, device=device)
    t = torch.as_tensor(target, device=device)[None]
    nn_ = torch.as_tensor(normal, device=device)
    kp = torch.as_tensor(np.asarray(scales["pos_m_per_action_unit_step"], np.float32), device=device)
    kr = torch.as_tensor(np.asarray(scales["rot_rad_per_action_unit_step"], np.float32), device=device)
    d = p - t
    sq = d.square()
    per_sample = sq.mean(dim=(2, 3))
    pos = torch.linalg.norm(torch.cumsum(d[..., :3], dim=2) * kp, dim=-1) * 1000
    rot = torch.linalg.norm(torch.cumsum(d[..., 3:6], dim=2) * kr, dim=-1) * 180 / np.pi
    dn = (p - nn_).square()
    term = torch.cumsum(p[..., :3], dim=2)[:, :, -1] * kp * 1000          # (S,N,3)
    pair = torch.cdist(term.permute(1, 0, 2), term.permute(1, 0, 2))     # (N,S,S)
    s_count = p.shape[0]
    gs = torch.sign(p[..., 6])
    split = (gs.max(dim=0).values != gs.min(dim=0).values).float().mean(dim=1)
    out = {"mse": per_sample.mean(0), "best8": per_sample.min(0).values,
           "mse_pos": sq[..., :3].mean(dim=(2, 3)).mean(0), "mse_rot": sq[..., 3:6].mean(dim=(2, 3)).mean(0),
           "mse_grip": sq[..., 6].mean(dim=2).mean(0),
           "pos_chunk_mm": pos.mean(dim=2).mean(0), "pos_term_mm": pos[..., -1].mean(0),
           "rot_term_deg": rot[..., -1].mean(0),
           "grip_agree": (torch.sign(p[..., 6]) == torch.sign(t[..., 6])).float().mean(dim=2).mean(0),
           "dn": dn.mean(dim=(2, 3)).mean(0), "dn_pos": dn[..., :3].mean(dim=(2, 3)).mean(0),
           "dn_rot": dn[..., 3:6].mean(dim=(2, 3)).mean(0), "dn_grip": dn[..., 6].mean(dim=2).mean(0),
           "spread": p.std(dim=0, unbiased=False).mean(dim=(1, 2)),
           "spread_pos_term_mm": pair.sum(dim=(1, 2)) / max(s_count * (s_count - 1), 1),
           "grip_split": split}
    return {k: v.float().cpu().numpy() for k, v in out.items()}


# --------------------------------------------------------------------------
# Evaluation of one condition
# --------------------------------------------------------------------------

def method_specs(retr, cal):
    """(name, F, anchor key or None, t0 (N,), n (N,)) for every compared method."""
    n_rows = len(retr["conf_sim"])
    specs = [("N:noise:n10", "N", None, np.ones(n_rows, np.float32), np.full(n_rows, 10))]
    for k in range(1, 10):
        specs.append((f"N:noise:n{k}", "N", None, np.ones(n_rows, np.float32), np.full(n_rows, k)))
    specs.append(("A:noise:n10", "A", None, np.ones(n_rows, np.float32), np.full(n_rows, 10)))
    for fkey in ("A", "N"):
        unit_t0 = {"cal_sim": apply_g(cal[fkey]["unit"]["sim"], retr["conf_sim"]),
                   "cal_ent": apply_g(cal[fkey]["unit"]["ent"], retr["conf_ent"])}
        raw_t0 = {"cal_sim": apply_g(cal[fkey]["raw"]["sim"], retr["raw_conf_sim"]),
                  "cal_ent": apply_g(cal[fkey]["raw"]["ent"], retr["raw_conf_ent"])}
        for anchor in ANCHORS:
            for rule in RULES:
                if rule.startswith("fix"):
                    t0 = np.full(n_rows, float(rule[3:]), np.float32)
                else:
                    t0 = (raw_t0 if anchor == "raw" else unit_t0)[rule]
                specs.append((f"{fkey}:{anchor}:{rule}", fkey, anchor, t0, steps_for(t0)))
    return specs


def decode_anchor(m1, bases, rows):
    out = np.empty((len(bases), *m1.action_shape), np.float32)
    zero = np.zeros((1, m1.residual_dim), np.float32)
    for lo in range(0, len(bases), 4096):
        hi = min(len(bases), lo + 4096)
        out[lo:hi] = m1.decode(bases[lo:hi], np.repeat(zero, hi - lo, 0), rows["f"][lo:hi], rows["s"][lo:hi])
    return out


def evaluate(rows, retr, eps, flows, cal, m1, scales, *, device, keep_rows=()):
    """Per-row metrics for every method. eps (S,N,D) shared by all methods."""
    target = rows["a"][:, :HORIZON]
    anchors = {"unit": decode_anchor(m1, retr["unit_b"], rows),
               "shuf": decode_anchor(m1, retr["shuf_b"], rows),
               "gold_unit": decode_anchor(m1, retr["gold_b_unit"], rows),
               "raw": retr["raw_action"]}
    specs = method_specs(retr, cal)
    n = len(target)
    per = {name: {m: np.zeros(n, np.float32) for m in METRICS} for name, *_ in specs}
    keep_rows = np.asarray(sorted(set(int(x) for x in keep_rows)), np.int64)
    kept = {}
    for lo in range(0, n, QUERY_CHUNK):
        hi = min(n, lo + QUERY_CHUNK)
        sl = slice(lo, hi)
        f, s = rows["f"][sl], rows["s"][sl]
        e = eps[:, sl]
        normal = None
        for name, fkey, anchor, t0, steps in specs:
            model, norm = flows[fkey]
            a = None if anchor is None else anchors[anchor][sl]
            pred = rollout(model, norm, f, s, a, e, t0[sl], steps[sl], device=device)
            if normal is None:
                normal = pred
            m = row_metrics(pred, target[sl], normal, scales, device=device)
            for k, v in m.items():
                per[name][k][sl] = v
            sel = keep_rows[(keep_rows >= lo) & (keep_rows < hi)]
            if len(sel):
                kept.setdefault(name, {}).update({int(r): pred[:, r - lo] for r in sel})
    for name, _, anchor, t0, steps in specs:
        per[name]["t0"] = t0.astype(np.float32)
        per[name]["n"] = steps.astype(np.float32)
    examples = None
    if len(keep_rows):
        examples = {"pred": kept, "target": {int(r): target[r] for r in keep_rows},
                    "anchor": {k: {int(r): v[r] for r in keep_rows} for k, v in anchors.items()}}
    return per, examples


# --------------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------------

KEY_METHODS = ("A:unit:cal_sim", "A:unit:cal_ent", "N:unit:cal_sim", "N:unit:cal_ent",
               "N:raw:cal_sim", "A:raw:cal_sim", "A:shuf:cal_sim", "N:shuf:cal_sim",
               "A:gold_unit:cal_sim", "N:gold_unit:cal_sim")


def analyze(per, masks, retr, cal, task_index):
    base = per["N:noise:n10"]
    out = {"populations": {}, "bins": {}, "risk_coverage": {}, "tasks": {}}
    for pop, mask in masks.items():
        if not mask.any():
            continue
        cell = {}
        for name, mets in per.items():
            c = {k: float(np.mean(v[mask])) for k, v in mets.items()}
            rel = (mets["mse"][mask] - base["mse"][mask]) / np.maximum(base["mse"][mask], 1e-8)
            c["frac_better"] = float(np.mean(rel < -CHANGE_BAND))
            c["frac_worse"] = float(np.mean(rel > CHANGE_BAND))
            c["frac_lower"] = float(np.mean(mets["mse"][mask] < base["mse"][mask]))
            cell[name] = c
        cell["_n"] = int(mask.sum())
        cell["_top1_is_gold_unit"] = float(np.mean(retr["unit_id"][mask] == retr["gold_unit_id"][mask]))
        cell["_post_add_top1"] = float(np.mean(retr["post_add"][mask]))
        out["populations"][pop] = cell
        # Confidence bins (calibration bin edges) and risk-coverage.
        bins_pop, rc_pop = {}, {}
        for name in KEY_METHODS:
            fkey, anchor, rule = name.split(":")
            kind = rule.split("_")[1]
            ck = ("raw_conf_" if anchor == "raw" else "conf_") + kind
            gcal = cal[fkey]["raw" if anchor == "raw" else "unit"][kind]
            conf = retr[ck][mask]
            b = conf_bin(gcal, conf)
            mm, bm = per[name]["mse"][mask], base["mse"][mask]
            rel = (mm - bm) / np.maximum(bm, 1e-8)
            rows_b = []
            for k in range(len(gcal["t0"])):
                sel = b == k
                rows_b.append({"bin": k, "n": int(sel.sum()), "t0": gcal["t0"][k],
                               "mse": float(mm[sel].mean()) if sel.any() else None,
                               "normal_mse": float(bm[sel].mean()) if sel.any() else None,
                               "frac_better": float(np.mean(rel[sel] < -CHANGE_BAND)) if sel.any() else None,
                               "frac_worse": float(np.mean(rel[sel] > CHANGE_BAND)) if sel.any() else None,
                               "dn": float(per[name]["dn"][mask][sel].mean()) if sel.any() else None})
            bins_pop[name] = rows_b
            orderc = np.argsort(-conf, kind="stable")
            curve = []
            for cov in np.linspace(0.1, 1.0, 10):
                k = max(1, int(round(cov * len(orderc))))
                cv = orderc[:k]
                rest = orderc[k:]
                curve.append({"coverage": float(cov), "method_mse": float(mm[cv].mean()),
                              "normal_mse": float(bm[cv].mean()),
                              "selective_all_mse": float((mm[cv].sum() + bm[rest].sum()) / len(orderc))})
            rc_pop[name] = curve
        out["bins"][pop] = bins_pop
        out["risk_coverage"][pop] = rc_pop
    allm = masks["all"]
    for task in np.unique(task_index[allm]):
        sel = allm & (task_index == task)
        out["tasks"][int(task)] = {"n": int(sel.sum()), **{
            name: {"mse": float(per[name]["mse"][sel].mean()),
                   "best8": float(per[name]["best8"][sel].mean()),
                   "pos_term_mm": float(per[name]["pos_term_mm"][sel].mean())}
            for name in ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "N:unit:cal_sim",
                         "A:unit:fix0.4", "N:raw:cal_sim", "A:shuf:cal_sim", "A:gold_unit:cal_sim")}}
        if "novel" in masks:
            nsel = sel & masks["novel"]
            if nsel.any():
                out["tasks"][int(task)]["novel"] = {
                    name: float(per[name]["mse"][nsel].mean())
                    for name in ("N:noise:n10", "A:unit:cal_sim", "N:unit:cal_sim", "N:raw:cal_sim")}
                out["tasks"][int(task)]["novel_n"] = int(nsel.sum())
    return out


# --------------------------------------------------------------------------
# Text, physical scales
# --------------------------------------------------------------------------

def precompute_texts(rows, sources, cache_dir):
    texts_by_source, unique = {}, set()
    for source in sources:
        attached = af.attach_subtask_source(rows, source, split="A", cache_dir=cache_dir,
                                            annotations_path=DEFAULT_ANNOTATIONS, predicted_root=None)
        texts_by_source[source] = {(str(e), int(f)): str(t) for e, f, t in
                                   zip(attached["episode_id"], attached["frame"], attached["subtask_text"])}
        unique.update(map(str, attached["subtask_text"]))
    unique = sorted(unique)
    emb = frozen_pi05_plan_embeddings(unique, params_path=DEFAULT_PI05_BASE_PARAMS,
                                      tokenizer_path=DEFAULT_PALIGEMMA_TOKENIZER)
    return texts_by_source, dict(zip(unique, emb))


def attach_text(rows, texts_by_source, emb, source):
    out = dict(rows)
    keys = list(zip(map(str, rows["episode_id"]), map(int, rows["frame"])))
    out["subtask_text"] = np.asarray([texts_by_source[source][k] for k in keys], dtype=str)
    out["subtask_embedding"] = np.stack([emb[t] for t in out["subtask_text"]]).astype(np.float32)
    return out


def fit_physical_scales(cache_dir, episode_ids, *, horizon=HORIZON, max_episodes=300,
                        seed=0, stride=5):
    """Fit motion scales with this experiment's historical episode cap."""
    return _fit_physical_scales(
        cache_dir, episode_ids, horizon=horizon, max_episodes=max_episodes,
        seed=seed, stride=stride)


# --------------------------------------------------------------------------
# One split x seed
# --------------------------------------------------------------------------

def holdout_split(train_all, split, seed, per_task=HOLDOUT_PER_TASK):
    rng = np.random.default_rng(seed + 3)
    held = set()
    for task in SPLITS[split][0]:
        eps = np.unique(train_all["episode_id"][train_all["task_index"] == task])
        if len(eps) > per_task + 3:
            held.update(map(str, rng.choice(eps, per_task, replace=False)))
    mask = np.isin(train_all["episode_id"].astype(str), list(held))
    return _subset(train_all, ~mask), _subset(train_all, mask)


def calibrate_all(flows, m1, view, raw_keys, raw_actions, val, val_keys, val_b, tau, m6, eps, *, device, split):
    retr = retrieve_static(view, val_keys, val_b, raw_keys, raw_actions, tau, m6,
                           np.random.default_rng(0), device=device)
    target = val["a"][:, :HORIZON]
    anchors = {"unit": decode_anchor(m1, retr["unit_b"], val), "raw": retr["raw_action"]}
    cal, detail = {}, {}
    for fkey, (model, norm) in flows.items():
        cal[fkey], detail[fkey] = {}, {}
        normal = rollout(model, norm, val["f"], val["s"], None, eps, np.ones(len(target)), device=device)
        err_normal = ((normal - target[None]) ** 2).mean(axis=(0, 2, 3))
        for anchor in ("unit", "raw"):
            errs = []
            for t0 in T0_GRID:
                if t0 >= 1.0:
                    errs.append(err_normal)
                    continue
                pred = rollout(model, norm, val["f"], val["s"], anchors[anchor], eps,
                               np.full(len(target), t0, np.float32), device=device)
                errs.append(((pred - target[None]) ** 2).mean(axis=(0, 2, 3)))
            errs = np.stack(errs)
            prefix = "raw_" if anchor == "raw" else ""
            cal[fkey][anchor] = {"sim": fit_monotone_g(retr[prefix + "conf_sim"], errs),
                                 "ent": fit_monotone_g(retr[prefix + "conf_ent"], errs)}
            detail[fkey][anchor] = {"fixed_t0_validation_mse": {str(t): float(e.mean()) for t, e in zip(T0_GRID, errs)}}
    return cal, detail, int(len(target))


def run_split_seed(rows, *, split, seed, sources, texts, emb, device, scales, m1_epochs, f_epochs,
                   batch_size, smoke, log, example_rows=6):
    tic = time.perf_counter()
    train_tasks, eval_tasks, _ = SPLITS[split]
    train_all = _subset(rows, np.isin(rows["task_index"], train_tasks) & rows["valid_h25"])
    test = _subset(rows, np.isin(rows["task_index"], eval_tasks) & rows["valid_h25"])
    if smoke:
        train_all = af._select_episodes(train_all, train_tasks, 12, seed + 101)
        test = af._select_episodes(test, eval_tasks, 4, seed + 211)
    train, heldout = holdout_split(train_all, split, seed, per_task=2 if smoke else HOLDOUT_PER_TASK)
    fit_ix, val_ix = _episode_train_validation_indices(train, seed + 401)
    train_fit = _subset(train, np.isin(np.arange(len(train["a"])), fit_ix))
    validation = _subset(train, np.isin(np.arange(len(train["a"])), val_ix))
    eval_eps = set(map(str, heldout["episode_id"])) | set(map(str, test["episode_id"]))
    assert_calibration_rows(validation, split, eval_eps)
    if set(map(str, train_fit["episode_id"])) & (eval_eps | set(map(str, validation["episode_id"]))):
        raise ValueError("training rows overlap validation/evaluation episodes")
    train_actions = train_fit["a"][:, :HORIZON].copy()
    m6, bases, _ = af._fit_m6_backbone_and_units(train_fit, seed=seed + 19, device=device, epochs=m1_epochs)
    m1 = m6.backbone
    m1_hash = af.state_hash(m1)
    log(f"  {split}/s{seed}: rows fit/val/D1/new = {len(train_fit['a'])}/{len(validation['a'])}/"
        f"{len(heldout['a'])}/{len(test['a'])}; M6 {len(m6.prototypes)} units ({time.perf_counter()-tic:.0f}s)")
    dim = int(np.prod(m1.action_shape))
    zeros = np.zeros((len(train_actions), dim), np.float32)
    f_normal, n_norm, n_fit = af.fit_anchor_flow(
        train_fit, train_actions, (zeros, np.zeros(len(zeros), np.float32), zeros, np.zeros(len(zeros), np.float32)),
        source_mix=NORMAL_MIX, seed=seed + 827, epochs=f_epochs, patience=8, batch_size=batch_size, device=device)
    log(f"  {split}/s{seed}: F_normal best epoch {n_fit['best_epoch']} ({time.perf_counter()-tic:.0f}s)")
    norm_ix, _ = _episode_train_validation_indices(train_fit, seed + 827 + 11)
    a_m, a_s = _standardize_fit(train_actions[norm_ix].reshape(len(norm_ix), -1))
    f_m, f_s = _standardize_fit(train_fit["f"][norm_ix])
    s_m, s_s = _standardize_fit(train_fit["s"][norm_ix])
    norm_seed = {"a_mean": a_m, "a_scale": a_s, "f_mean": f_m, "f_scale": f_s,
                 "s_mean": s_m, "s_scale": s_s, "action_shape": m1.action_shape}
    val_b, _ = m1.encode(validation["a"][:, :HORIZON], validation["s"])
    held_b, _ = m1.encode(heldout["a"][:, :HORIZON], heldout["s"])
    test_b, _ = m1.encode(test["a"][:, :HORIZON], test["s"])
    rng_noise = np.random.default_rng(seed * 1_000_003 + 31)
    eps = {"val": rng_noise.standard_normal((SAMPLES, len(validation["a"]), dim)).astype(np.float32),
           "D1": rng_noise.standard_normal((SAMPLES, len(heldout["a"]), dim)).astype(np.float32),
           "new": rng_noise.standard_normal((SAMPLES, len(test["a"]), dim)).astype(np.float32)}
    novel = af._novel_motion_mask(test, split, np.unique(train_fit["skill_class"]))
    masks_new = {"all": np.ones(len(test["a"]), bool)}
    if split == "A":
        masks_new.update({"novel": novel, "known": ~novel})
    masks_d1 = {"all": np.ones(len(heldout["a"]), bool)}
    scale = float(np.sqrt(np.mean((bases - bases.mean(axis=0)) ** 2)))
    ctx = {}
    out = {"split": split, "seed": seed, "rows": {"train_fit": len(train_fit["a"]), "validation": len(validation["a"]),
                                                  "D1": len(heldout["a"]), "new": len(test["a"]),
                                                  "novel": int(novel.sum())},
           "M6_units_train": int(len(m6.prototypes)), "M6_epsilon": float(m6.epsilon),
           "F_normal_best_epoch": int(n_fit["best_epoch"]),
           "F_normal_validation_loss": float(n_fit["best_validation_loss"]), "sources": {}}
    for source in sources:
        t1 = time.perf_counter()
        tf, vf, hf, te = (attach_text(r, texts, emb, source) for r in (train_fit, validation, heldout, test))
        qm, qfit, train_keys = af._fit_query_encoder(tf, bases, seed=seed + 701, device=device,
                                                     allowed_tasks=tuple(train_tasks), subtask_source=source)
        tau = float(qfit["temperature"])
        bank = af.UnitBank(m6, scale)
        bank.seed_training(bases, train_keys, tf["episode_id"])
        sources_train = af._make_flow_sources(m1, bank, tf, bases, train_keys, tau, norm_seed, seed=seed + 823)
        f_anchor, a_norm, a_fit = af.fit_anchor_flow(
            tf, train_actions, sources_train, source_mix=ANCHOR_MIX, seed=seed + 827,
            epochs=f_epochs, patience=8, batch_size=batch_size, device=device)
        flows = {"N": (f_normal, n_norm), "A": (f_anchor, a_norm)}
        hashes = [af.state_hash(x) for x in (m1, qm, f_normal, f_anchor)]
        val_keys = af._query_embedding(qm, qfit, af._normal_query_rows(vf), device=device)
        held_keys = af._query_embedding(qm, qfit, af._normal_query_rows(hf), device=device)
        test_keys = af._query_embedding(qm, qfit, af._normal_query_rows(te), device=device)
        ids, centers, keys, _, _ = bank.centers_keys_sigmas()
        view = (ids, centers, keys)
        cal, cal_detail, n_cal = calibrate_all(flows, m1, view, train_keys, train_actions, vf, val_keys, val_b,
                                               tau, m6, eps["val"], device=device, split=split)
        log(f"  {split}/s{seed}/{source}: E+F_anchor(best {a_fit['best_epoch']})+calibration "
            f"({time.perf_counter()-t1:.0f}s); g_sim(A,unit)={cal['A']['unit']['sim']['t0']}")
        conds = {}
        for cname, rws, q, gb, e, masks, rs in (("D1", hf, held_keys, held_b, eps["D1"], masks_d1, 1),
                                                 ("D3", te, test_keys, test_b, eps["new"], masks_new, 2)):
            retr = retrieve_static(view, q, gb, train_keys, train_actions, tau, m6,
                                   np.random.default_rng(seed + 900 + rs), device=device)
            per, _ = evaluate(rws, retr, e, flows, cal, m1, scales, device=device)
            conds[cname] = {"analysis": analyze(per, masks, retr, cal, rws["task_index"]),
                            "per": per, "retr": retr}
            log(f"  {split}/s{seed}/{source}/{cname}: normal {per['N:noise:n10']['mse'].mean():.4f} "
                f"P-a {per['A:unit:cal_sim']['mse'].mean():.4f} P-b {per['N:unit:cal_sim']['mse'].mean():.4f} "
                f"raw {per['N:raw:cal_sim']['mse'].mean():.4f} ({time.perf_counter()-t1:.0f}s)")
        ctx[source] = {"flows": flows, "cal": cal, "bank": bank, "train_keys": train_keys,
                       "test_keys": test_keys, "tau": tau, "hashes": hashes, "te": te, "conds": conds}
        out["sources"][source] = {"temperature": tau, "E_epochs": int(qfit["epochs"]),
                                  "F_anchor_best_epoch": int(a_fit["best_epoch"]),
                                  "F_anchor_validation_loss": float(a_fit["best_validation_loss"]),
                                  "calibration": cal, "calibration_detail": cal_detail,
                                  "calibration_rows": n_cal, "units_in_bank": int(len(ids))}
    # D2: one prequential stream feeds a bank per source.
    t2 = time.perf_counter()
    initial_ids = np.arange(len(m6.prototypes))
    d2, order = stream_d2(m6, {s: ctx[s]["bank"] for s in sources}, test, test_b,
                          {s: ctx[s]["test_keys"] for s in sources},
                          {s: ctx[s]["train_keys"] for s in sources}, train_actions,
                          {s: ctx[s]["tau"] for s in sources}, initial_ids=initial_ids, seed=seed, device=device)
    out["M6_units_after_stream"] = int(len(m6.prototypes))
    log(f"  {split}/s{seed}: D2 stream {len(m6.prototypes)} units ({time.perf_counter()-t2:.0f}s)")
    examples = {}
    for source in sources:
        c = ctx[source]
        per, _ = evaluate(c["te"], d2[source], eps["new"], c["flows"], c["cal"], m1, scales, device=device)
        c["conds"]["D2"] = {"analysis": analyze(per, masks_new, d2[source], c["cal"], test["task_index"]),
                            "per": per, "retr": d2[source]}
        if [af.state_hash(x) for x in (m1,)] != [m1_hash] or \
                [af.state_hash(x) for x in (c["flows"]["N"][0], c["flows"]["A"][0])] != c["hashes"][2:]:
            raise AssertionError("weights changed during evaluation")
        log(f"  {split}/s{seed}/{source}/D2: normal {per['N:noise:n10']['mse'].mean():.4f} "
            f"P-a {per['A:unit:cal_sim']['mse'].mean():.4f} raw {per['N:raw:cal_sim']['mse'].mean():.4f}")
        out["sources"][source]["conditions"] = {k: v["analysis"] for k, v in c["conds"].items()}
        # Examples (main source only): best/worst rows of P-a vs normal.
        if source == sources[0] and example_rows:
            for cname in ("D3", "D2"):
                cd = c["conds"][cname]
                pa, nm = cd["per"]["A:unit:cal_sim"]["mse"], cd["per"]["N:noise:n10"]["mse"]
                pool = np.flatnonzero(masks_new.get("novel", masks_new["all"])) if split == "A" else np.arange(len(pa))
                if not len(pool):
                    pool = np.arange(len(pa))
                diff = pa[pool] - nm[pool]
                k = example_rows // 2
                pick = np.concatenate([pool[np.argsort(diff)[:k]], pool[np.argsort(-diff)[:k]]])
                retr = cd["retr"]
                sub = _subset(c["te"], np.isin(np.arange(len(pa)), pick))
                sub_ix = np.sort(pick)
                sub_retr = {kk: v[sub_ix] for kk, v in retr.items()}
                _, ex = evaluate(sub, sub_retr, eps["new"][:, sub_ix], c["flows"], c["cal"], m1, scales,
                                 device=device, keep_rows=range(len(sub_ix)))
                names = ("N:noise:n10", "A:unit:cal_sim", "N:unit:cal_sim", "N:raw:cal_sim")
                examples[f"{split}_s{seed}_{cname}"] = [{
                    "row": int(r), "task": int(test["task_index"][r]), "episode": str(test["episode_id"][r]),
                    "frame": int(test["frame"][r]), "text": str(c["te"]["subtask_text"][r]),
                    "kind": "improved" if r in pick[:k] else "worsened",
                    "mse": {nm_: float(cd["per"][nm_]["mse"][r]) for nm_ in names},
                    "t0": float(cd["per"]["A:unit:cal_sim"]["t0"][r]),
                    "conf_sim": float(retr["conf_sim"][r]),
                    "target": ex["target"][j].tolist(),
                    "anchor_unit": ex["anchor"]["unit"][j].tolist(),
                    "anchor_raw": ex["anchor"]["raw"][j].tolist(),
                    "pred": {nm_: ex["pred"][nm_][j].tolist() for nm_ in names}}
                    for j, r in enumerate(sub_ix)]
    out["duration_seconds"] = time.perf_counter() - tic
    per_rows = {}
    for source in sources:
        for cname, cd in ctx[source]["conds"].items():
            tag = f"{source}_{cname}"
            for name in ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "N:unit:cal_sim", "A:unit:cal_ent",
                         "N:unit:cal_ent", "N:raw:cal_sim", "A:shuf:cal_sim", "A:gold_unit:cal_sim"):
                for met in ("mse", "best8", "t0", "dn"):
                    per_rows[f"{tag}|{name}|{met}"] = cd["per"][name][met]
            per_rows[f"{tag}|conf_sim"] = cd["retr"]["conf_sim"]
            per_rows[f"{tag}|conf_ent"] = cd["retr"]["conf_ent"]
    per_rows["new_task"] = test["task_index"]; per_rows["new_novel"] = novel
    per_rows["D1_task"] = heldout["task_index"]
    return out, examples, per_rows


# --------------------------------------------------------------------------
# Aggregation and report
# --------------------------------------------------------------------------

def _ms(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def aggregate(runs, sources):
    agg = {}
    for split in sorted({r["split"] for r in runs}):
        rs = [r for r in runs if r["split"] == split]
        for source in sources:
            for cond in ("D1", "D2", "D3"):
                pops = rs[0]["sources"][source]["conditions"][cond]["populations"]
                for pop in pops:
                    cells = [r["sources"][source]["conditions"][cond]["populations"].get(pop) for r in rs]
                    cells = [c for c in cells if c]
                    entry = {"_n": [c["_n"] for c in cells],
                             "_top1_is_gold_unit": _ms([c["_top1_is_gold_unit"] for c in cells]),
                             "_post_add_top1": _ms([c["_post_add_top1"] for c in cells])}
                    for name in cells[0]:
                        if name.startswith("_"):
                            continue
                        entry[name] = {m: _ms([c[name][m] for c in cells]) for m in cells[0][name]}
                    # Ordinary policy at the proposal's mean step count (per seed).
                    for name in cells[0]:
                        if name.startswith("_") or ":noise:" in name:
                            continue
                        vals = []
                        for c in cells:
                            k = int(np.clip(np.rint(c[name]["n"]), 1, 10))
                            vals.append(c[f"N:noise:n{k}"]["mse"])
                        entry[name]["matched_normal_mse"] = _ms(vals)
                    agg[f"{split}|{source}|{cond}|{pop}"] = entry
    return agg


LABELS = {
    "N:noise:n10": "通常（F_normal、10ステップ）",
    "A:noise:n10": "F_anchor をノイズから（10）",
    "A:unit:cal_sim": "P-a 単位 + g(類似度)",
    "A:unit:cal_ent": "P-a 単位 + g(エントロピー)",
    "A:unit:fix0.2": "P-a 単位 t0=0.2", "A:unit:fix0.4": "P-a 単位 t0=0.4", "A:unit:fix0.6": "P-a 単位 t0=0.6",
    "N:unit:cal_sim": "P-b 単位 + g(類似度)",
    "N:unit:cal_ent": "P-b 単位 + g(エントロピー)",
    "N:unit:fix0.2": "P-b 単位 t0=0.2", "N:unit:fix0.4": "P-b 単位 t0=0.4", "N:unit:fix0.6": "P-b 単位 t0=0.6",
    "N:raw:cal_sim": "生区間NN（F_normal）+ g(類似度)", "N:raw:cal_ent": "生区間NN（F_normal）+ g(エントロピー)",
    "N:raw:fix0.4": "生区間NN（F_normal）t0=0.4",
    "A:raw:cal_sim": "生区間NN（F_anchor）+ g(類似度)",
    "A:shuf:cal_sim": "シャッフル（F_anchor、P-aと同じt0）", "N:shuf:cal_sim": "シャッフル（F_normal、P-bと同じt0）",
    "A:gold_unit:cal_sim": "正解の単位（F_anchor、上限）", "N:gold_unit:cal_sim": "正解の単位（F_normal、上限）",
}
MAIN_ROWS = ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "A:unit:cal_ent", "A:unit:fix0.2", "A:unit:fix0.4",
             "A:unit:fix0.6", "N:unit:cal_sim", "N:unit:cal_ent", "N:unit:fix0.4", "N:raw:cal_sim",
             "N:raw:cal_ent", "N:raw:fix0.4", "A:raw:cal_sim", "A:shuf:cal_sim", "N:shuf:cal_sim",
             "A:gold_unit:cal_sim", "N:gold_unit:cal_sim")
COND_DESC = {"D1": "D1 既知（学習 suite の保留エピソード、メモリ＝学習 suite）",
             "D2": "D2 新 suite・追加あり（prequential）", "D3": "D3 新 suite・追加なし（メモリ＝学習 suite）"}
POP_DESC = {"all": "全区間", "novel": "新しい運動の区間（4タスク）", "known": "既知の運動の区間"}


def _f(cell, digits=4, scale=1.0):
    if not cell:
        return "–"
    sd = cell["sd"] * scale
    return f"{cell['mean'] * scale:.{digits}f} ± {sd:.{digits}f}"


def write_summary(path, payload):
    agg, sources = payload["aggregate"], payload["sources"]
    L = ["# 公平な比較：確信度で開始点を変えるアンカー方策 vs 通常の方策（軽量版・オフライン）", ""]
    L += [f"seed {payload['seeds']}、Split {payload['splits']}。値は seed 平均 ± seed 間 SD（ddof=1）。各 seed 内は区間（行）で平均。"
          "同じ行・同じノイズ（8本）を全方式で共有する対応のある比較。統計的有意性・閉ループ成功率は主張しない。", ""]
    L += ["## 方式", "",
          "- 通常：F_normal（通常の経路だけで学習）をノイズから10ステップ。F_anchor は S1 と同じ混合（取り出し0.65/正解b0.05/通常0.30）。両 F は同じ構造・行・seed・初期値・データ順・エポック上限・早期終了。",
          "- 提案：q=E(o)（言語なし画像k＋サブタスク文＋状態）→ top-1 単位 b=c_top1 → a=Dec(b, r=0, s) → t0=g(c)、n=max(1,round(10 t0))、x=(1−t0)a+t0ε から刻み0.1で n ステップ。P-a は F_anchor、P-b は F_normal（学習し直さない）。",
          "- g：学習 suite の検証エピソードだけで、確信度の分位8ビンごとに t0∈{0.1..1.0} の検証MSEを測り、確信度に対して t0 が単調非増加になる割り当てのうち総検証誤差最小のもの（DP）。新 suite は見ない。確信度は (i) 最大コサイン類似度、(ii) 1−H(w)/log K。",
          "- 生区間NN（RAGDP型）：同じ E のキー、値＝学習 suite（D2 は先行エピソードも）の生の行動チャンク。同じ手順で自分の確信度から g を較正。",
          "- シャッフル：鍵と値の対応を巡回置換（不動点なし）で入れ替えた単位の中心。t0/n は提案と同一（行ごと）。",
          "- 正解の単位：正解区間の b に機能的距離で最も近い単位の中心（上限の参考）。t0/n は提案と同一。",
          "- 反復回数を揃えた通常：各方式の平均 n（四捨五入）と同じ回数で F_normal をノイズから（刻み 1/n）。", ""]
    L += ["## 指標", "",
          "- MSE：8本×25ステップ×7次元の平均（元の行動単位）。best-of-8：8本中最小のMSE。pos/rot/grip：次元群別MSE。",
          f"- 手先の位置誤差：(予測−正解) の xyz 累積 × 換算係数（学習 suite の env_state から最小二乗、m/単位・ステップ {np.round(payload['physical_scales'][payload['splits'][0]]['pos_m_per_action_unit_step'], 4).tolist()}）。終端（25ステップ目）mm。指令値からの推定。",
          "- グリッパー一致：符号の一致率（8本×25ステップ）。",
          "- 通常からの距離 dn：同じノイズでの通常の出力とのMSE。良化/悪化：行ごとMSEが通常より5%超低い/高い行の割合。",
          "- 8本のばらつき：要素ごとの標準偏差の平均、終端位置の8本間ペア距離（mm）、グリッパー符号が8本で割れるステップの割合。", ""]
    for source in sources:
        L += [f"## サブタスクの出どころ：{source}" + ("（主、閉ループで使える）" if source == "instruction" else "（上限）"), ""]
        for cond in ("D1", "D2", "D3"):
            for split in payload["splits"]:
                for pop in ("all", "novel", "known"):
                    key = f"{split}|{source}|{cond}|{pop}"
                    if key not in agg:
                        continue
                    e = agg[key]
                    L += [f"### {COND_DESC[cond]} / Split {split} / {POP_DESC[pop]}（行数 {e['_n']}；top-1＝正解の単位 {_f(e['_top1_is_gold_unit'], 3)}"
                          + (f"；追加単位がtop-1 {_f(e['_post_add_top1'], 3)}" if cond == "D2" else "") + "）", "",
                          "| 方式 | MSE | best-of-8 | 終端位置 mm | grip一致 | 良化/悪化 % | 平均n | 同じnの通常 MSE |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"]
                    for name in MAIN_ROWS:
                        c = e.get(name)
                        if not c:
                            continue
                        fb, fw = c["frac_better"], c["frac_worse"]
                        L.append(f"| {LABELS.get(name, name)} | {_f(c['mse'])} | {_f(c['best8'])} | {_f(c['pos_term_mm'], 1)} | "
                                 f"{_f(c['grip_agree'], 3)} | {fb['mean']*100:.0f} / {fw['mean']*100:.0f} | {c['n']['mean']:.1f} | "
                                 f"{_f(c.get('matched_normal_mse')) if ':noise:' not in name else '–'} |")
                    L.append("")
        # Output-change table (D3 & D2 main pops)
        L += [f"### 通常の出力からどう変わったか（{source}）：次元群別の通常からの距離と誤差の変化", "",
              "| 条件 | 方式 | dn 位置 | dn 回転 | dn グリップ | ΔMSE 位置 | ΔMSE 回転 | ΔMSE グリップ | ばらつき（通常→） | 終端ばらつき mm（通常→） | grip割れ（通常→） |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cond in ("D1", "D2", "D3"):
            for split in payload["splits"]:
                for pop in ("all", "novel"):
                    key = f"{split}|{source}|{cond}|{pop}"
                    if key not in agg:
                        continue
                    e = agg[key]
                    b = e["N:noise:n10"]
                    for name in ("A:unit:cal_sim", "N:unit:cal_sim", "N:raw:cal_sim"):
                        c = e[name]
                        d = lambda m: f"{c[m]['mean'] - b[m]['mean']:+.4f}"
                        L.append(f"| {cond}/{split}/{pop} | {LABELS[name]} | {c['dn_pos']['mean']:.4f} | {c['dn_rot']['mean']:.4f} | "
                                 f"{c['dn_grip']['mean']:.4f} | {d('mse_pos')} | {d('mse_rot')} | {d('mse_grip')} | "
                                 f"{b['spread']['mean']:.3f}→{c['spread']['mean']:.3f} | {b['spread_pos_term_mm']['mean']:.1f}→{c['spread_pos_term_mm']['mean']:.1f} | "
                                 f"{b['grip_split']['mean']:.3f}→{c['grip_split']['mean']:.3f} |")
        L.append("")
    L += ["## 確信度ビン別（seed 0、instruction、P-a 類似度 g）", ""]
    r0 = [r for r in payload["runs"] if r["seed"] == payload["seeds"][0]]
    for r in r0:
        src = sources[0]
        for cond in ("D1", "D2", "D3"):
            an = r["sources"][src]["conditions"][cond]
            for pop in ("all", "novel"):
                if pop not in an["bins"]:
                    continue
                L += [f"**{cond} / Split {r['split']} / {POP_DESC[pop]}**", "",
                      "| ビン（低→高確信） | 行数 | t0 | P-a MSE | 通常 MSE | 良化 % | 悪化 % | dn |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
                for b in an["bins"][pop]["A:unit:cal_sim"]:
                    if not b["n"]:
                        continue
                    L.append(f"| {b['bin']} | {b['n']} | {b['t0']:.1f} | {b['mse']:.4f} | {b['normal_mse']:.4f} | "
                             f"{b['frac_better']*100:.0f} | {b['frac_worse']*100:.0f} | {b['dn']:.4f} |")
                L.append("")
    L += ["## タスク別 MSE（instruction、seed 平均）", ""]
    for split in payload["splits"]:
        for cond in ("D3", "D2"):
            rs = [r for r in payload["runs"] if r["split"] == split]
            tasks = sorted(rs[0]["sources"][sources[0]]["conditions"][cond]["tasks"], key=int)
            L += [f"**{cond} / Split {split}**", "", "| task | 名前 | 行数 | 通常 | P-a g類似度 | P-b g類似度 | 生区間NN | シャッフル | 正解の単位 | 新運動区間: 通常 / P-a / P-b / 生 |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for t in tasks:
                cells = [r["sources"][sources[0]]["conditions"][cond]["tasks"][t] for r in rs]
                m = lambda nm: np.mean([c[nm]["mse"] for c in cells])
                nov = ""
                if "novel" in cells[0]:
                    nov = " / ".join(f"{np.mean([c['novel'][nm] for c in cells if 'novel' in c]):.3f}"
                                     for nm in ("N:noise:n10", "A:unit:cal_sim", "N:unit:cal_sim", "N:raw:cal_sim"))
                L.append(f"| {t} | {payload['task_names'].get(str(t), '')[:48]} | {cells[0]['n']} | {m('N:noise:n10'):.3f} | {m('A:unit:cal_sim'):.3f} | "
                         f"{m('N:unit:cal_sim'):.3f} | {m('N:raw:cal_sim'):.3f} | {m('A:shuf:cal_sim'):.3f} | {m('A:gold_unit:cal_sim'):.3f} | {nov} |")
            L.append("")
    L += ["## 較正された g（seed 0、instruction）", ""]
    for r in r0:
        cal = r["sources"][sources[0]]["calibration"]
        for f in ("A", "N"):
            for anc in ("unit", "raw"):
                for kind in ("sim", "ent"):
                    g = cal[f][anc][kind]
                    L.append(f"- Split {r['split']} F_{'anchor' if f == 'A' else 'normal'} {anc} {kind}: t0（低→高確信）={g['t0']}、"
                             f"制約なし最良={g['per_bin_unconstrained_best']}、検証MSE {g['validation_mse']:.4f}")
    L += ["", "## 実行条件", "", f"- コマンド: `{payload['command']}`", f"- 所要時間: {payload['duration_minutes']:.1f} 分",
          f"- 図: {payload['figures']}", ""]
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

COLORS = {"gt": "#0b0b0b", "normal": "#8a8984", "pa": "#2a78d6", "pb": "#eb6834", "raw": "#1baf7a",
          "shuf": "#eda100", "gold": "#e87ba4", "anchor": "#6250d6"}


def make_figures(out, payload, examples, scales):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    files = []
    agg, src = payload["aggregate"], payload["sources"][0]
    # 1. MSE by method, per condition/split/pop.
    panels = [k for k in agg if k.split("|")[1] == src and k.split("|")[3] in ("all", "novel")]
    methods = [("N:noise:n10", "normal"), ("A:unit:cal_sim", "pa"), ("N:unit:cal_sim", "pb"),
               ("N:raw:cal_sim", "raw"), ("A:shuf:cal_sim", "shuf"), ("A:gold_unit:cal_sim", "gold")]
    names = ["normal", "P-a (F_anchor)", "P-b (F_normal)", "raw NN", "shuffle", "gold unit"]
    cols = 3
    rows_n = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.6 * rows_n), squeeze=False)
    for ax, key in zip(axes.flat, panels):
        e = agg[key]
        vals = [e[m]["mse"]["mean"] for m, _ in methods]
        sds = [e[m]["mse"]["sd"] for m, _ in methods]
        ax.bar(range(len(methods)), vals, yerr=sds, color=[COLORS[c] for _, c in methods], width=0.7, capsize=2)
        ax.set_xticks(range(len(methods)), names, rotation=30, ha="right")
        sp, _, cond, pop = key.split("|")
        ax.set_title(f"{cond} Split {sp} {pop}  (MSE)")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    p = fig_dir / "mse_by_method.png"; fig.savefig(p, dpi=140); plt.close(fig); files.append(str(p))
    # 2. Risk-coverage and confidence bins (seed-mean over runs).
    for split in payload["splits"]:
        rs = [r for r in payload["runs"] if r["split"] == split]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3), squeeze=False)
        for ax, cond in zip(axes[0], ("D1", "D2", "D3")):
            pop = "novel" if split == "A" and cond != "D1" else "all"
            for name, col, lab in (("A:unit:cal_sim", "pa", "P-a"), ("N:unit:cal_sim", "pb", "P-b"),
                                   ("N:raw:cal_sim", "raw", "raw NN")):
                curves = [r["sources"][src]["conditions"][cond]["risk_coverage"][pop][name] for r in rs]
                cov = [c["coverage"] for c in curves[0]]
                ax.plot(cov, np.mean([[c["method_mse"] for c in cv] for cv in curves], 0), color=COLORS[col], lw=2, label=lab)
                if name == "A:unit:cal_sim":
                    ax.plot(cov, np.mean([[c["normal_mse"] for c in cv] for cv in curves], 0), color=COLORS["normal"],
                            lw=2, ls="--", label="normal (same rows, by P-a conf.)")
            ax.set_title(f"{cond} Split {split} {pop}: risk–coverage")
            ax.set_xlabel("coverage (highest-confidence fraction)"); ax.set_ylabel("MSE on covered rows")
        axes[0][0].legend(frameon=False)
        fig.tight_layout()
        p = fig_dir / f"risk_coverage_{split}.png"; fig.savefig(p, dpi=140); plt.close(fig); files.append(str(p))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3), squeeze=False)
        for ax, cond in zip(axes[0], ("D1", "D2", "D3")):
            pop = "novel" if split == "A" and cond != "D1" else "all"
            for name, col, lab in (("A:unit:cal_sim", "pa", "P-a"), ("N:unit:cal_sim", "pb", "P-b")):
                bins = [r["sources"][src]["conditions"][cond]["bins"][pop][name] for r in rs]
                xs = [b["bin"] for b in bins[0]]
                better = np.nanmean([[np.nan if b["frac_better"] is None else b["frac_better"] for b in bb] for bb in bins], 0)
                worse = np.nanmean([[np.nan if b["frac_worse"] is None else b["frac_worse"] for b in bb] for bb in bins], 0)
                ax.plot(xs, better * 100, color=COLORS[col], lw=2, marker="o", ms=4, label=f"{lab} better")
                ax.plot(xs, worse * 100, color=COLORS[col], lw=2, ls=":", marker="s", ms=4, label=f"{lab} worse")
            ax.set_title(f"{cond} Split {split} {pop}: rows better/worse vs normal")
            ax.set_xlabel("confidence bin (validation quantiles, low→high)"); ax.set_ylabel("% rows")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = fig_dir / f"confidence_bins_{split}.png"; fig.savefig(p, dpi=140); plt.close(fig); files.append(str(p))
    # 3. Examples.
    for tag, items in examples.items():
        if not items:
            continue
        kp = np.asarray(scales[tag.split("_")[0]]["pos_m_per_action_unit_step"]) * 1000
        fig, axes = plt.subplots(len(items), 4, figsize=(12, 2.2 * len(items)), squeeze=False)
        for row_axes, it in zip(axes, items):
            series = [("target", np.asarray(it["target"]), COLORS["gt"], "-"),
                      ("anchor (unit)", np.asarray(it["anchor_unit"]), COLORS["anchor"], "--"),
                      ("anchor (raw)", np.asarray(it["anchor_raw"]), COLORS["raw"], "--")]
            preds = [("normal", "N:noise:n10", COLORS["normal"]), ("P-a", "A:unit:cal_sim", COLORS["pa"]),
                     ("P-b", "N:unit:cal_sim", COLORS["pb"])]
            for d, ax in enumerate(row_axes[:3]):
                for lab, key, col in preds:
                    arr = np.asarray(it["pred"][key])
                    cum = np.cumsum(arr[..., d], axis=-1) * kp[d]
                    for s_i, c in enumerate(cum):
                        ax.plot(c, color=col, lw=0.8, alpha=0.45, label=lab if s_i == 0 else None)
                for lab, arr, col, ls in series:
                    ax.plot(np.cumsum(arr[:, d]) * kp[d], color=col, lw=2, ls=ls, label=lab)
                ax.set_title(f"cum. {'xyz'[d]} (mm)")
            ax = row_axes[3]
            for lab, key, col in preds:
                arr = np.asarray(it["pred"][key])
                ax.plot(arr[..., 6].T, color=col, lw=0.8, alpha=0.45)
            for lab, arr, col, ls in series:
                ax.plot(arr[:, 6], color=col, lw=2, ls=ls)
            ax.set_title("gripper")
            m = it["mse"]
            row_axes[0].set_ylabel(f"{it['kind']}\ntask {it['task']} t0={it['t0']:.1f}\n"
                                   f"MSE n {m['N:noise:n10']:.3f} / Pa {m['A:unit:cal_sim']:.3f}\n/ Pb {m['N:unit:cal_sim']:.3f} / raw {m['N:raw:cal_sim']:.3f}",
                                   fontsize=7)
        axes[0][0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        p = fig_dir / f"examples_{tag}.png"; fig.savefig(p, dpi=120); plt.close(fig); files.append(str(p))
    return files


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------

def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE, dataset_dir=DEFAULT_DATASET,
        seeds=(0, 1, 2), splits=("A", "B"), sources=SOURCES_DEFAULT, device_name="cuda",
        m1_epochs=60, f_epochs=80, batch_size=512, smoke=False, threads=6, command=""):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(device_name)
    start = time.perf_counter()
    out = Path(output_root) / (datetime.now(timezone.utc).strftime("fair-compare-%Y%m%dT%H%M%SZ")
                               + ("-smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=False)
    log_path = out / "progress.log"

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    rows = load_rows(cache_dir, dataset_dir, max_horizon=HORIZON)
    tasks = sorted({t for s in splits for t in (*SPLITS[s][0], *SPLITS[s][1])})
    rows = _subset(rows, np.isin(rows["task_index"], tasks) & rows["valid_h25"])
    log(f"loaded {len(rows['a'])} rows ({time.perf_counter() - start:.0f}s)")
    texts, emb = precompute_texts(rows, sources, cache_dir)
    log(f"text embeddings: {len(emb)} unique ({time.perf_counter() - start:.0f}s)")
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    task_by_episode = {str(e["episode_id"]): str(e["task"]) for e in manifest["episodes"]}
    task_names = {}
    for e, t in zip(rows["episode_id"], rows["task_index"]):
        task_names.setdefault(str(int(t)), task_by_episode.get(str(e), ""))
    scales = {}
    for split in splits:
        train_eps = np.unique(rows["episode_id"][np.isin(rows["task_index"], SPLITS[split][0])])
        scales[split] = fit_physical_scales(cache_dir, train_eps, max_episodes=60 if smoke else 300)
    log(f"physical scales {scales}")
    runs, examples, per_rows_all = [], {}, {}
    for split in splits:
        for seed in seeds:
            r, ex, pr = run_split_seed(rows, split=split, seed=seed, sources=sources, texts=texts, emb=emb,
                                       device=device, scales=scales[split], m1_epochs=m1_epochs,
                                       f_epochs=f_epochs, batch_size=batch_size, smoke=smoke, log=log,
                                       example_rows=6 if seed == seeds[0] else 0)
            runs.append(r); examples.update(ex)
            per_rows_all.update({f"{split}_s{seed}|{k}": v for k, v in pr.items()})
            log(f"done {split}/s{seed} in {r['duration_seconds'] / 60:.1f} min")
            partial = {"runs": runs}
            (out / "results.partial.json").write_text(json.dumps(partial, ensure_ascii=False))
    np.savez_compressed(out / "per_row.npz", **{k.replace("/", "_"): np.asarray(v) for k, v in per_rows_all.items()})
    (out / "examples.json").write_text(json.dumps(examples))
    payload = {"spec": "fair comparison per researcher request 2026-10-28", "seeds": list(seeds), "splits": list(splits),
               "sources": list(sources), "smoke": smoke, "command": command, "physical_scales": scales,
               "task_names": task_names, "t0_grid": list(T0_GRID), "fixed_t0": list(FIXED_T0),
               "anchor_mix": list(ANCHOR_MIX), "calibration_bins": CAL_BINS, "samples": SAMPLES,
               "holdout_episodes_per_task": HOLDOUT_PER_TASK, "change_band": CHANGE_BAND,
               "m1_epochs": m1_epochs, "f_epochs": f_epochs, "runs": runs}
    payload["aggregate"] = aggregate(runs, sources)
    payload["figures"] = make_figures(out, payload, examples, scales)
    payload["duration_minutes"] = (time.perf_counter() - start) / 60
    (out / "results.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (out / "results.partial.json").unlink(missing_ok=True)
    write_summary(out / "summary.md", payload)
    log(f"finished in {payload['duration_minutes']:.1f} min: {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--splits", default="A,B")
    p.add_argument("--sources", default=",".join(SOURCES_DEFAULT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--m1-epochs", type=int, default=60)
    p.add_argument("--flow-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    out = run(output_root=a.output_root, cache_dir=a.cache_dir, dataset_dir=a.dataset,
              seeds=tuple(int(x) for x in a.seeds.split(",")), splits=tuple(a.splits.split(",")),
              sources=tuple(a.sources.split(",")), device_name=a.device, m1_epochs=a.m1_epochs,
              f_epochs=a.flow_epochs, batch_size=a.batch_size, smoke=a.smoke, threads=a.threads,
              command="python -m skill_moe.skillnet.examples.emergent.light.intermediate.fair_compare "
                      + " ".join(argv if argv is not None else os.sys.argv[1:]))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
