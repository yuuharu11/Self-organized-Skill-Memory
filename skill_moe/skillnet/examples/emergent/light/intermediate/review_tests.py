"""Offline intervention experiments for reviewer checks R1--R8.

This module deliberately does not modify the canonical S1 implementation.  It
provides reusable, model-agnostic diagnostics and an audit/report reader for an
existing fair_compare run.  The frozen fair_compare artifacts contain row
metrics, not model weights or query/action tensors; consequently the CLI marks
model-dependent variants unavailable instead of silently re-fitting them on a
different split or reporting reused metrics as new experiments.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import torch
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR


def motion_labels(action: np.ndarray, *, grip_threshold: float = 0.1) -> dict[str, str]:
    """Coarse post-hoc movement label for an action chunk [H,7]."""
    a = np.asarray(action, np.float64)
    translation = a[:, :3].sum(axis=0)
    rotation = a[:, 3:6].sum(axis=0)
    def axis(v: np.ndarray, threshold: float) -> str:
        if np.linalg.norm(v) < threshold:
            return "none"
        i = int(np.argmax(np.abs(v)))
        return ("+" if v[i] >= 0 else "-") + "xyz"[i]
    grip = float(np.mean(a[:, 6]))
    return {"direction": axis(translation, .05), "rotation": axis(rotation, .05),
            "gripper": "close" if grip > grip_threshold else "open" if grip < -grip_threshold else "neutral"}


def r1_transfer_diagnostics(bases: np.ndarray, states_a: np.ndarray, states_b: np.ndarray,
                            actions: np.ndarray, decode: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
                            ) -> dict[str, Any]:
    """Hold b fixed, decode at cross-scene states, compare post-hoc motion labels."""
    b = np.asarray(bases, np.float32)
    sa, sb = np.asarray(states_a, np.float32), np.asarray(states_b, np.float32)
    target = [motion_labels(x) for x in np.asarray(actions)]
    zero = np.zeros((len(b), 0), np.float32)
    outputs_a = np.asarray(decode(b, sa, zero))
    outputs_b = np.asarray(decode(b, sb, zero))
    predicted_a = [motion_labels(x) for x in outputs_a]
    predicted_b = [motion_labels(x) for x in outputs_b]
    channels = ("direction", "gripper", "rotation")
    return {"n": len(b), "agreement": {name: float(np.mean([
                p[name] == t[name] for p, t in zip(predicted_b, target, strict=True)]))
                for name in channels},
            "cross_state_agreement": {name: float(np.mean([
                a[name] == b_[name] for a, b_ in zip(predicted_a, predicted_b, strict=True)]))
                for name in channels}, "labels_are_posthoc_only": True}


def residual_ablation_metrics(decode: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                              bases: np.ndarray, residuals: np.ndarray, states: np.ndarray,
                              target: np.ndarray, *, seed: int = 0) -> dict[str, float]:
    """Compare encoded residual, zero residual, and paired standard-normal residual."""
    b, r, s, a = map(lambda x: np.asarray(x, np.float32), (bases, residuals, states, target))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(r.shape).astype(np.float32)
    def mse(value):
        y = np.asarray(decode(b, value, s), np.float32)
        return float(np.mean((y - a) ** 2))
    return {"r_encoded_mse": mse(r), "r_zero_mse": mse(np.zeros_like(r)),
            "r_standard_normal_mse": mse(noise),
            "delta_zero_minus_encoded": mse(np.zeros_like(r)) - mse(r),
            "delta_normal_minus_zero": mse(noise) - mse(np.zeros_like(r))}


def functional_distance_grid(bases: np.ndarray, member_states: list[np.ndarray],
                             decode: Callable[[np.ndarray, np.ndarray], np.ndarray], *,
                             common_states: np.ndarray, residual_dim: int = 0) -> dict[str, Any]:
    """Compare center and state-set choices; distance is RMS action distance."""
    b = np.asarray(bases, np.float32)
    if len(b) != len(member_states):
        raise ValueError("one state array is required per base")
    centers = {"mean": b.mean(0), "medoid": b[int(np.argmin(
        np.sum((b[:, None] - b[None]) ** 2, axis=-1).mean(1)))]}
    state_sets = {"common": np.asarray(common_states, np.float32),
                  "members": np.concatenate([np.asarray(x, np.float32) for x in member_states], axis=0)}
    result = {}
    for center_name, center in centers.items():
        for state_name, states in state_sets.items():
            repeated = np.broadcast_to(center, (len(states), len(center)))
            residual = np.zeros((len(states), residual_dim), np.float32)
            outputs = np.asarray(decode(repeated, states, residual), np.float32).reshape(len(states), -1)
            result[f"{center_name}/{state_name}"] = {"center": center.tolist(),
                "decoded_rms": float(np.sqrt(np.mean(outputs ** 2))),
                "between_state_rms": float(np.sqrt(np.mean((outputs - outputs.mean(0)) ** 2)))}
    return result


def multipositive_infonce(logits: np.ndarray, positive: np.ndarray) -> float:
    """Stable multi-positive InfoNCE: logsumexp(all)-logsumexp(positives)."""
    x = np.asarray(logits, np.float64)
    pos = np.asarray(positive, bool)
    if x.ndim != 2 or pos.shape != x.shape or np.any(~pos.any(axis=1)):
        raise ValueError("each query needs a nonempty positive set matching logits")
    def lse(y):
        m = np.max(y, axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(y - m).sum(axis=1)))
    return float(np.mean(lse(x) - lse(np.where(pos, x, -np.inf))))


def topk_candidates(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stable top-k candidate indices and normalized softmax weights."""
    x = np.asarray(scores, np.float64)
    if x.ndim != 2 or k < 1:
        raise ValueError("scores must be [queries,candidates] and k positive")
    k = min(int(k), x.shape[1])
    ix = np.argpartition(-x, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(x, ix, axis=1).argsort(axis=1)[:, ::-1]
    ix = np.take_along_axis(ix, order, axis=1)
    v = np.take_along_axis(x, ix, axis=1)
    e = np.exp(v - v.max(axis=1, keepdims=True)); return ix, e / e.sum(axis=1, keepdims=True)


def fit_confidence_model(features: np.ndarray, anchor_error: np.ndarray, *, ridge: float = 1e-3
                         ) -> dict[str, Any]:
    """Fit a small standardized ridge predictor for anchor error on train-suite holdout."""
    x, y = np.asarray(features, np.float64), np.asarray(anchor_error, np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 4:
        raise ValueError("confidence fit requires aligned [N,D] features and at least 4 rows")
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = np.c_[(x - mean) / scale, np.ones(len(x))]
    penalty = np.eye(z.shape[1]) * ridge; penalty[-1, -1] = 0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    pred = z @ coef
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
            "train_mae": float(np.mean(np.abs(pred - y))), "train_rows": len(y),
            "fit_scope": "training-suite held-out episodes only"}


def predict_confidence_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, np.float64)
    z = np.c_[(x - np.asarray(model["mean"])) / np.asarray(model["scale"]), np.ones(len(x))]
    return np.maximum(z @ np.asarray(model["coef"]), 0)


def risk_coverage(error: np.ndarray, predicted_risk: np.ndarray,
                  fractions=(.1, .2, .4, .6, .8, 1.0)) -> list[dict[str, float]]:
    err, risk = np.asarray(error, np.float64), np.asarray(predicted_risk, np.float64)
    if err.shape != risk.shape:
        raise ValueError("risk and error arrays must align")
    order = np.argsort(risk, kind="stable")
    return [{"coverage": float(c), "risk_threshold": float(risk[order[max(1, int(c*len(order)))-1]]),
             "mean_error": float(err[order[:max(1, int(c*len(order))) ]].mean())} for c in fractions]


def t0_confidence_bins(predicted_risk: np.ndarray, grid=(.1, .2, .4, .6, .8, 1.0), bins=6
                       ) -> dict[str, Any]:
    """Training-only quantile bins; larger predicted risk falls back toward t0=1."""
    risk = np.asarray(predicted_risk, np.float64)
    edges = np.unique(np.quantile(risk, np.linspace(0, 1, bins + 1)[1:-1]))
    chosen = np.linspace(min(grid), max(grid), len(edges) + 1)[::-1]
    return {"risk_edges": edges.tolist(), "t0_low_to_high_risk": chosen.tolist(),
            "counts": np.bincount(np.searchsorted(edges, risk), minlength=len(edges)+1).tolist()}


def independent_t0_n_grid() -> list[dict[str, float | int]]:
    return [{"t0": t, "n": n} for t in (.1, .2, .4, .6, .8, 1.0) for n in (1, 2, 3, 5, 10)]


def bridge_training_targets(actions: np.ndarray, anchors: np.ndarray, noise: np.ndarray,
                            t: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """R7 bridge path x=(1-t)A+t(a+sigma*eps), with dx/dt target."""
    a, anchor, eps = map(lambda x: np.asarray(x, np.float32), (actions, anchors, noise))
    tt = np.asarray(t, np.float32).reshape(-1, *([1] * (a.ndim - 1)))
    target_point = anchor + float(sigma) * eps
    return (1 - tt) * a + tt * target_point, target_point - a


def select_equal_count_raw_memory(keys: np.ndarray, count: int, *, seed: int
                                  ) -> dict[str, np.ndarray]:
    """Random and K-means-medoid subsets, with an identical memory item budget."""
    x = np.asarray(keys, np.float32)
    if x.ndim != 2 or not 0 < count <= len(x):
        raise ValueError("count must be within raw memory size")
    rng = np.random.default_rng(seed)
    random_ids = np.sort(rng.choice(len(x), count, replace=False))
    # Lloyd iterations on normalized keys; store actual training examples nearest
    # to each centroid, not synthetic centroid values.
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    centroids = z[np.sort(rng.choice(len(z), count, replace=False))].copy()
    for _ in range(20):
        labels = np.argmax(z @ centroids.T, axis=1)
        for c in range(count):
            assigned = z[labels == c]
            if len(assigned):
                v = assigned.mean(0); centroids[c] = v / max(np.linalg.norm(v), 1e-8)
    medoid_ids = np.empty(count, np.int64)
    for c in range(count):
        assigned_ids = np.flatnonzero(labels == c)
        if len(assigned_ids):
            medoid_ids[c] = assigned_ids[np.argmax(z[assigned_ids] @ centroids[c])]
        else:
            medoid_ids[c] = np.argmax(z @ centroids[c])
    medoid_ids = np.unique(medoid_ids)
    if len(medoid_ids) < count:
        remaining = np.setdiff1d(np.arange(len(x)), medoid_ids)
        medoid_ids = np.r_[medoid_ids, rng.choice(remaining, count-len(medoid_ids), replace=False)]
    medoid_ids = np.sort(medoid_ids[:count])
    return {"random_ids": random_ids, "kmedoids_ids": medoid_ids}


def audit_fair_compare(path: Path) -> dict[str, Any]:
    """Read existing fair-comparison summaries as context; do not relabel them R results."""
    p = Path(path)
    payload = json.loads((p / "results.json").read_text())
    rows = np.load(p / "per_row.npz", allow_pickle=False)
    return {"path": str(p), "payload": payload, "per_row_fields": list(rows.files),
            "stored_models": False, "stored_per_step_predictions": False}


def _bridge_flow(train_rows: dict[str, np.ndarray], actions: np.ndarray, sources: tuple,
                 seed: int, epochs: int, batch_size: int, device: torch.device,
                 *, sigma: float) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Train the R7 bridge path with the canonical train/validation episode split."""
    from . import anchor_flow as af
    from .experiment import _episode_train_validation_indices, _standardize, _standardize_fit
    memory, _, gold, _ = sources
    train_ix, valid_ix = _episode_train_validation_indices(train_rows, seed + 11)
    f_mean, f_scale = _standardize_fit(train_rows["f"][train_ix])
    s_mean, s_scale = _standardize_fit(train_rows["s"][train_ix])
    a_mean, a_scale = _standardize_fit(actions[train_ix].reshape(len(train_ix), -1))
    dim = int(np.prod(actions.shape[1:]))
    at = torch.as_tensor(_standardize(actions.reshape(len(actions), dim), a_mean, a_scale), device=device)
    ft = torch.as_tensor(_standardize(train_rows["f"], f_mean, f_scale), device=device)
    st = torch.as_tensor(_standardize(train_rows["s"], s_mean, s_scale), device=device)
    mem = torch.as_tensor(memory, device=device); gold_t = torch.as_tensor(gold, device=device)
    torch.manual_seed(seed)
    model = af.FlowActionHead(ft.shape[-1], st.shape[-1], dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed + 29)
    mix = torch.as_tensor((.65, .05, .30), device=device)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(epochs):
        order = torch.as_tensor(train_ix, device=device)
        order = order[torch.randperm(len(order), generator=generator, device=device)]
        model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            kind = torch.multinomial(mix.expand(len(ix), -1), 1, generator=generator).squeeze(-1)
            eps = torch.randn((len(ix), dim), generator=generator, device=device)
            anchor = at[ix].clone()
            anchor[kind == 0] = mem[ix[kind == 0]]
            anchor[kind == 1] = gold_t[ix[kind == 1]]
            t = torch.rand((len(ix),), generator=generator, device=device)
            endpoint = anchor + float(sigma) * eps
            xt = (1 - t[:, None]) * at[ix] + t[:, None] * endpoint
            target_v = endpoint - at[ix]
            pred = model(xt, t, ft[ix], st[ix])
            loss = (pred - target_v).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            vt = torch.full((len(valid_ix),), .5, device=device)
            vx = .5 * at[valid_ix] + .5 * mem[valid_ix]
            pred = model(vx, vt, ft[valid_ix], st[valid_ix])
            val_loss = float((pred - (mem[valid_ix] - at[valid_ix])).square().mean().cpu())
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8: break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    norm = {"f_mean": f_mean, "f_scale": f_scale, "s_mean": s_mean, "s_scale": s_scale,
            "a_mean": a_mean, "a_scale": a_scale, "action_shape": tuple(actions.shape[1:])}
    return model, norm, {"sigma": sigma, "best_epoch": best_epoch, "best_validation_loss": best}


def _bridge_rollout(model, norm, rows, anchors, eps, *, sigma: float, n: int, device):
    from .experiment import _standardize
    samples, count, dim = eps.shape
    fs = _standardize(rows["f"], norm["f_mean"], norm["f_scale"])
    ss = _standardize(rows["s"], norm["s_mean"], norm["s_scale"])
    aa = _standardize(np.asarray(anchors).reshape(count, dim), norm["a_mean"], norm["a_scale"])
    result = np.empty_like(eps)
    for lo in range(0, count, 128):
        hi = min(count, lo + 128); c = hi - lo
        noise = torch.as_tensor(eps[:, lo:hi].reshape(samples*c, dim), device=device)
        x = torch.as_tensor(aa[None, lo:hi].repeat(samples, axis=0).reshape(samples*c, dim),
                            device=device) + float(sigma) * noise
        ft = torch.as_tensor(fs[lo:hi], device=device).repeat(samples, 1)
        st = torch.as_tensor(ss[lo:hi], device=device).repeat(samples, 1)
        with torch.no_grad():
            for j in range(n):
                t = torch.full((samples*c,), 1 - j/n, device=device)
                x = x - (1/n) * model(x, t, ft, st)
        result[:, lo:hi] = x.reshape(samples, c, dim).cpu().numpy()
    return (result * norm["a_scale"] + norm["a_mean"]).reshape(
        samples, count, *norm["action_shape"])


def _capture_fair_run(fc, af, kwargs):
    """Run canonical fair_compare while retaining train/eval artifacts in memory."""
    captured: dict[str, Any] = {"eval": [], "retrievals": [], "flows": {}, "banks": []}
    original = {"m6": af._fit_m6_backbone_and_units, "e": af._fit_query_encoder,
                "flow": af.fit_anchor_flow, "bank": af.UnitBank,
                "retrieve": fc.retrieve_static, "evaluate": fc.evaluate, "cal": fc.calibrate_all,
                "stream": fc.stream_d2}
    def fit_m6(*a, **kw):
        out = original["m6"](*a, **kw)
        captured["m6"], captured["bases"], captured["actions"] = out
        captured["train_fit"] = a[0]
        return out
    def fit_e(*a, **kw):
        out = original["e"](*a, **kw)
        captured["query_encoder"], captured["query_fit"], captured["train_keys"] = out
        return out
    def fit_flow(*a, **kw):
        out = original["flow"](*a, **kw)
        kind = "normal" if tuple(kw["source_mix"]) == (0., 0., 1.) else "anchor"
        captured["flows"][kind] = out
        if kind == "anchor": captured["flow_sources"] = a[2]
        return out
    def make_bank(*a, **kw):
        obj = original["bank"](*a, **kw); captured["banks"].append(obj); return obj
    def retrieve(*a, **kw):
        out = original["retrieve"](*a, **kw)
        captured["retrievals"].append({"query": a[1], "view": a[0], "retr": out})
        return out
    def evaluate(*a, **kw):
        result = original["evaluate"](*a, **kw)
        captured["eval"].append({"rows": a[0], "retr": a[1], "eps": a[2], "flows": a[3],
                                 "cal": a[4], "m1": a[5], "scales": a[6],
                                 "device": kw["device"], "per": result[0]})
        return result
    def calibrate(*a, **kw):
        captured["calibration_input"] = {"flows": a[0], "m1": a[1], "view": a[2],
            "raw_keys": a[3], "raw_actions": a[4], "val": a[5], "val_keys": a[6],
            "val_b": a[7], "tau": a[8], "m6": a[9], "eps": a[10], "device": kw["device"]}
        return original["cal"](*a, **kw)
    def stream(*a, **kw):
        captured["m6_initial_centers"] = np.asarray(a[0].prototypes, np.float32).copy()
        result = original["stream"](*a, **kw)
        captured["stream_input"] = {"test": a[2], "test_b": a[3], "keys": a[4],
                                     "train_keys": a[5], "train_actions": a[6], "order": result[1]}
        return result
    with contextlib.ExitStack() as stack:
        for obj, name, replacement in ((af, "_fit_m6_backbone_and_units", fit_m6),
                                       (af, "_fit_query_encoder", fit_e), (af, "fit_anchor_flow", fit_flow),
                                       (af, "UnitBank", make_bank), (fc, "retrieve_static", retrieve),
                                       (fc, "evaluate", evaluate), (fc, "calibrate_all", calibrate),
                                       (fc, "stream_d2", stream)):
            stack.enter_context(patch.object(obj, name, replacement))
        captured["baseline_result"] = fc.run_split_seed(**kwargs)
    captured["d1"], captured["d3"], captured["d2"] = captured["eval"]
    captured["val_retr"] = captured["retrievals"][0]["retr"]
    captured["val_query"] = captured["retrievals"][0]["query"]
    return captured


def _mean_metrics(per: dict[str, np.ndarray], baseline: np.ndarray) -> dict[str, Any]:
    mse = np.asarray(per["mse"]); base = np.asarray(baseline)
    rel = (mse - base) / np.maximum(base, 1e-8)
    return {k: float(np.mean(v)) for k, v in per.items() if np.asarray(v).ndim == 1
            and np.issubdtype(np.asarray(v).dtype, np.number)} | {
            "frac_better_vs_normal": float(np.mean(rel < -.05)),
            "frac_worse_vs_normal": float(np.mean(rel > .05))}


def _flow_eval(fc, model, norm, rows, anchor, eps, t0, n, normal, scales, device):
    pred = fc.rollout(model, norm, rows["f"], rows["s"], anchor, eps,
                      np.broadcast_to(np.asarray(t0, np.float32), (len(rows["a"]),)),
                      np.broadcast_to(np.asarray(n, np.int64), (len(rows["a"]),)), device=device)
    metrics = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON], normal, scales, device=device)
    return metrics, pred


def _key_features(fc, query, keys, anchor_action, rows, m1):
    q = fc._nrm(query); k = fc._nrm(keys)
    sims = q @ k.T
    top = np.sort(sims, axis=1)[:, -min(5, sims.shape[1]):]
    density = top.mean(1)
    prior_b, _ = m1.prior(rows["f"], rows["s"])
    prior_a = fc.decode_anchor(m1, prior_b, rows)
    disagreement = np.mean((np.asarray(anchor_action) - prior_a) ** 2, axis=(1, 2))
    return np.c_[sims.max(1), density, disagreement].astype(np.float32)


def _fit_risk_g(fc, model, norm, m1, val, val_q, val_retr, view, eps, device):
    """Fit h and t0 only on disjoint episode-heldout subsets of the train suite."""
    feature = _key_features(fc, val_q, view[2],
                            fc.decode_anchor(m1, val_retr["unit_b"], val), val, m1)
    target = val["a"][:, :fc.HORIZON]
    anchor = fc.decode_anchor(m1, val_retr["unit_b"], val)
    actual_anchor_error = np.mean((anchor - target) ** 2, axis=(1, 2))
    episode_ids = np.unique(val["episode_id"].astype(str))
    rng = np.random.default_rng(173)
    rng.shuffle(episode_ids)
    model_eps = set(episode_ids[:max(1, len(episode_ids)//2)])
    model_rows = np.isin(val["episode_id"].astype(str), list(model_eps))
    cal_rows = ~model_rows
    if not cal_rows.any():
        cal_rows = model_rows.copy()
    h = fit_confidence_model(feature[model_rows], actual_anchor_error[model_rows])
    risk = predict_confidence_model(h, feature)
    errors = []
    for t0 in fc.T0_GRID:
        steps = fc.steps_for(np.full(len(val["a"]), t0))
        pred = fc.rollout(model, norm, val["f"], val["s"], anchor, eps,
                          np.full(len(val["a"]), t0), steps, device=device)
        errors.append(np.mean((pred - target[None]) ** 2, axis=(0, 2, 3)))
    g = fc.fit_monotone_g(-risk[cal_rows], np.asarray(errors)[:, cal_rows], grid=fc.T0_GRID)
    h["training_episode_count"] = len(model_eps)
    h["calibration_episode_count"] = len(set(val["episode_id"].astype(str)[cal_rows]))
    return h, g, feature, risk, np.asarray(errors)


def _select_raw(keys, actions, ids, query, fc, device):
    selected_keys, selected_actions = keys[ids], actions[ids]
    top, conf, _ = fc.cosine_stats(query, selected_keys, .1, device=device)
    return selected_actions[top], conf


def _calibrate_raw(fc, model, norm, val, val_q, val_eps, candidate_keys, candidate_actions, device):
    anchors, conf = _select_raw(candidate_keys, candidate_actions,
                                np.arange(len(candidate_keys)), val_q, fc, device)
    err = []
    for t0 in fc.T0_GRID:
        pred = fc.rollout(model, norm, val["f"], val["s"], anchors, val_eps,
                          np.full(len(anchors), t0), fc.steps_for(np.full(len(anchors), t0)), device=device)
        err.append(np.mean((pred - val["a"][None, :, :fc.HORIZON]) ** 2, axis=(0, 2, 3)))
    return fc.fit_monotone_g(conf, np.asarray(err), grid=fc.T0_GRID)


def _eval_raw_d2_prequential(fc, model, norm, rows, query, eps, scales, normal_mse,
                             train_keys, train_actions, stream_order, *, budget, seed, device):
    """Evaluate an equal-count raw-interval store with episode-block prequentiality.

    For a query episode the store contains all training intervals and only D2
    episodes completed before it. The current episode is appended after every
    row in that episode has been answered, matching fair_compare.stream_d2.
    The random store is a seeded reservoir sample; the medoid store recomputes
    actual-example medoids from the same visible pool.
    """
    episodes = np.asarray(rows["episode_id"]).astype(str)
    frames = np.asarray(rows["frame"])
    actions = np.asarray(rows["a"][:, :fc.HORIZON], np.float32)
    train_keys = np.asarray(train_keys, np.float32)
    train_actions = np.asarray(train_actions, np.float32)
    query = np.asarray(query, np.float32)
    order = np.asarray(stream_order, np.int64)
    if len(order) != len(rows["a"]):
        raise ValueError("D2 order must cover every query row exactly once")
    rng = np.random.default_rng(seed)
    pool_keys = [train_keys]
    pool_actions = [train_actions]
    outputs = {"random_ids": np.zeros_like(actions), "kmedoids_ids": np.zeros_like(actions)}
    counts = []
    # fair_compare order is episode-blocked; process the same blocks in order.
    i = 0
    while i < len(order):
        ep = episodes[order[i]]
        j = i + 1
        while j < len(order) and episodes[order[j]] == ep:
            j += 1
        block = order[i:j]
        visible_keys = np.concatenate(pool_keys)
        visible_actions = np.concatenate(pool_actions)
        count = min(int(budget), len(visible_keys))
        counts.append(count)
        random_ids = np.sort(rng.choice(len(visible_keys), count, replace=False))
        # Bound the repeated Lloyd fit in the stream while retaining examples
        # from the full visible pool; selection remains episode-block causal.
        medoid_pool = np.arange(len(visible_keys))
        medoid_pool_cap = max(4 * count, count)
        if len(medoid_pool) > medoid_pool_cap:
            medoid_pool = np.sort(rng.choice(medoid_pool, medoid_pool_cap, replace=False))
        medoid_local = select_equal_count_raw_memory(
            visible_keys[medoid_pool], count, seed=seed + 1_000_003 + i)["kmedoids_ids"]
        selections = {"random_ids": random_ids, "kmedoids_ids": medoid_pool[medoid_local]}
        for name, ids in selections.items():
            selected_keys, selected_actions = visible_keys[ids], visible_actions[ids]
            anchor, _ = _select_raw(selected_keys, selected_actions,
                                    np.arange(len(ids)), query[block], fc, device)
            pred = fc.rollout(model, norm, rows["f"][block], rows["s"][block], anchor,
                              eps[:, block], np.ones(len(block)),
                              fc.steps_for(np.ones(len(block))), device=device)
            outputs[name][block] = pred.mean(0)
        # Append only after all queries in this episode are answered.
        pool_keys.append(query[block]); pool_actions.append(actions[block])
        i = j
    metrics = {}
    for name, pred in outputs.items():
        raw = fc.row_metrics(pred[None], actions, np.zeros_like(pred[None]), scales, device=device)
        metrics[name] = {**_mean_metrics(raw, normal_mse), "visible_budget_min": int(min(counts)),
                         "visible_budget_max": int(max(counts)), "episode_block_prequential": True}
    return metrics, outputs


def _run_priority_experiments(captured, *, split, seed, device, output_dir, logger,
                             bridge_epochs=80):
    from . import fair_compare as fc
    from . import anchor_flow as af
    train = captured["train_fit"]; m6 = captured["m6"]; m1 = m6.backbone
    bases = captured["bases"]; train_actions = captured["actions"]
    bank = captured["banks"][0]
    model_a, norm_a, fit_a = captured["flows"]["anchor"]
    model_n, norm_n, fit_n = captured["flows"]["normal"]
    flow_a, flow_n = captured["eval"][0]["flows"]["A"], captured["eval"][0]["flows"]["N"]
    valctx = captured["calibration_input"]
    val, val_q, val_retr, view = valctx["val"], valctx["val_keys"], captured["val_retr"], valctx["view"]
    val_eps = valctx["eps"]
    result: dict[str, Any] = {"split": split, "seed": seed, "R5": {}, "R6": {}, "R7": {}, "R8": {},
                              "fair_compare_consistency": {}}
    result["R7"]["canonical_D3"] = {
        "F_normal": float(captured["d3"]["per"]["N:noise:n10"]["mse"].mean()),
        "F_anchor": float(captured["d3"]["per"]["A:noise:n10"]["mse"].mean()),
        "unit_F_anchor": float(captured["d3"]["per"]["A:unit:cal_sim"]["mse"].mean())}
    result["R7"]["canonical_by_condition"] = {
        condition: {name: float(captured[condition_key]["per"][name]["mse"].mean())
                    for name in ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim")}
        for condition, condition_key in (("D1", "d1"), ("D2", "d2"), ("D3", "d3"))}
    normal_per_d3 = captured["d3"]["per"]["N:noise:n10"]["mse"]
    archive = np.load(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/fair-compare-20261028T101842Z/per_row.npz'))
    key = f"{split}_s{seed}|instruction_D3|N:noise:n10|mse"
    ref = archive[key]
    result["fair_compare_consistency"] = {"current_mse": float(normal_per_d3.mean()),
        "archive_mse": float(ref.mean()), "absolute_difference": float(abs(normal_per_d3.mean()-ref.mean())),
        "within_1e-6": bool(ref.shape == normal_per_d3.shape and
                             abs(normal_per_d3.mean()-ref.mean()) < 1e-6),
        "same_row_count": bool(ref.shape == normal_per_d3.shape)}
    logger(f"{split}/s{seed} canonical baseline MSE={normal_per_d3.mean():.6f}; archived={ref.mean():.6f}")

    # Persist exact model weights and normalization used by all intervention tests.
    model_path = output_dir / f"{split}_s{seed}_models.pt"
    torch.save({"m1": {k: v.detach().cpu() for k, v in m1.encoder.state_dict().items()},
                "m1_prior": {k: v.detach().cpu() for k, v in m1.prior_net.state_dict().items()},
                "m1_base_decoder": {k: v.detach().cpu() for k, v in m1.base_decoder.state_dict().items()},
                "m1_residual_decoder": {k: v.detach().cpu() for k, v in m1.residual_decoder.state_dict().items()},
                "E": {k: v.detach().cpu() for k, v in captured["query_encoder"].state_dict().items()},
                "F_normal": {k: v.detach().cpu() for k, v in model_n.state_dict().items()},
                "F_anchor": {k: v.detach().cpu() for k, v in model_a.state_dict().items()},
                "F_normal_norm": norm_n, "F_anchor_norm": norm_a,
                "M1_stats": {name: getattr(m1, name) for name in
                             ("am", "ass", "sm", "ss", "fm", "fs", "action_shape", "a_dim",
                              "state_dim", "feature_dim", "base_dim", "residual_dim")},
                "train_keys": np.asarray(captured["train_keys"], np.float16),
                "M6_centers": np.asarray(m6.prototypes), "M6_epsilon": m6.epsilon,
                "M6_units": len(m6.prototypes), "E_fit": captured["query_fit"],
                "canonical_F_anchor_fit": fit_a, "canonical_F_normal_fit": fit_n}, model_path)

    # Save query embeddings and retrieval/query predictions for future reuse.
    eval_by_name = {"D1": captured["d1"], "D3": captured["d3"], "D2": captured["d2"]}
    embeds = {"validation_query": np.asarray(val_q, np.float16),
              "validation_target": np.asarray(val["a"][:, :fc.HORIZON], np.float16)}
    for condition, ctx in eval_by_name.items():
        embeds[f"{condition}_unit_base"] = np.asarray(ctx["retr"]["unit_b"], np.float16)
        query = (captured["retrievals"][1]["query"] if condition == "D1" else
                 captured["retrievals"][2]["query"] if condition == "D3" else
                 captured["stream_input"]["keys"]["instruction"])
        embeds[f"{condition}_query"] = np.asarray(query, np.float16)
    np.savez_compressed(output_dir / f"{split}_s{seed}_embeddings.npz", **embeds)

    h, g, val_features, val_risk, val_errors = _fit_risk_g(
        fc, model_a, norm_a, m1, val, val_q, val_retr, view, val_eps, device)
    result["R5"]["risk_model"] = h; result["R5"]["t0_calibration"] = g
    val_t0 = fc.apply_g(g, -val_risk)
    val_grid = np.asarray(fc.T0_GRID, np.float32)
    val_error = np.asarray([val_errors[int(np.argmin(np.abs(val_grid - t))), i]
                             for i, t in enumerate(val_t0)])
    result["R5"]["validation_risk_coverage"] = risk_coverage(val_error, val_risk)
    result["R5"]["validation_average_t0"] = float(val_t0.mean())
    result["R5"]["heldout"] = {}
    logger(f"{split}/s{seed} R5 risk model calibrated on {h['calibration_episode_count']} held-out episodes")

    # R6: independently cross all specified start positions and step counts.
    result["R6"]["grid"] = {}
    for condition, ctx in eval_by_name.items():
        rows, eps, retr = ctx["rows"], ctx["eps"], ctx["retr"]
        normal = ctx["per"]["N:noise:n10"]["mse"]
        anchor = fc.decode_anchor(m1, retr["unit_b"], rows)
        result["R6"]["grid"][condition] = {}
        pred_for_save = {}
        for item in independent_t0_n_grid():
            t0, n = float(item["t0"]), int(item["n"])
            pred = fc.rollout(model_a, norm_a, rows["f"], rows["s"], anchor, eps,
                              np.full(len(rows["a"]), t0), np.full(len(rows["a"]), n), device=device)
            mets = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON],
                                  np.zeros_like(pred), ctx["scales"], device=device)
            result["R6"]["grid"][condition][f"t0={t0:g},n={n}"] = {
                k: float(np.mean(v)) for k, v in mets.items() if k in
                ("mse", "best8", "mse_pos", "mse_rot", "mse_grip", "pos_term_mm", "rot_term_deg", "grip_agree")}
            if condition == "D3" and n == 10 and t0 in (.2, .6):
                pred_for_save[f"t0{t0:g}_n{n}"] = pred.mean(0).astype(np.float16)
        result["R6"]["grid"][condition]["normal_noise_by_n"] = {}
        for n in (1, 2, 3, 5, 10):
            pred = fc.rollout(model_n, norm_n, rows["f"], rows["s"], None, eps,
                              np.ones(len(rows["a"])), np.full(len(rows["a"]), n), device=device)
            result["R6"]["grid"][condition]["normal_noise_by_n"][str(n)] = float(
                np.mean((pred - rows["a"][None, :, :fc.HORIZON]) ** 2))
        if condition == "D3":
            np.savez_compressed(output_dir / f"{split}_s{seed}_R6_predictions.npz", **pred_for_save)
        logger(f"{split}/s{seed} R6 {condition} completed 30 anchor grid cells")

    # R7: compare existing F_normal/F_anchor with two bridge noise scales.
    result["R7"]["conditions"] = {}
    for sigma in (.03, .1):
        bridge, bridge_norm, bridge_fit = _bridge_flow(train, train_actions,
            captured["flow_sources"], seed + 827, bridge_epochs, 512, device, sigma=sigma)
        condition_results = {}
        for condition, ctx in eval_by_name.items():
            anchor = fc.decode_anchor(m1, ctx["retr"]["unit_b"], ctx["rows"])
            pred = _bridge_rollout(bridge, bridge_norm, ctx["rows"], anchor, ctx["eps"],
                                   sigma=sigma, n=10, device=device)
            metrics = fc.row_metrics(pred, ctx["rows"]["a"][:, :fc.HORIZON],
                                     ctx["per"]["N:noise:n10"].get("pred", np.zeros_like(pred)),
                                     ctx["scales"], device=device)
            condition_results[condition] = {k: float(np.mean(v)) for k, v in metrics.items()
                                             if k in ("mse", "best8", "mse_pos", "mse_rot", "mse_grip",
                                                      "pos_term_mm", "rot_term_deg", "grip_agree")}
            if condition == "D3":
                np.savez_compressed(output_dir / f"{split}_s{seed}_R7_sigma{sigma:g}.npz",
                                    prediction_mean=pred.mean(0).astype(np.float16),
                                    target=ctx["rows"]["a"][:, :fc.HORIZON].astype(np.float16))
        result["R7"]["conditions"][f"sigma={sigma:g}"] = {"fit": bridge_fit, "by_condition": condition_results}
        logger(f"{split}/s{seed} R7 bridge sigma={sigma:g}, best epoch={bridge_fit['best_epoch']}")

    # R8: item budget equals the number of initial M6 units; fit each raw
    # memory's own validation t0 on the same held-out training episodes.
    initial_count = len(bank.ids())
    raw_keys = np.asarray(captured["train_keys"], np.float32)
    raw_actions = np.asarray(train_actions, np.float32)
    selected = select_equal_count_raw_memory(raw_keys, initial_count, seed=seed + 912)
    result["R8"]["item_budget"] = initial_count; result["R8"]["conditions"] = {}
    result["R8"]["full_raw_reference_D3"] = float(captured["d3"]["per"]["N:raw:cal_sim"]["mse"].mean())
    for kind, ids in selected.items():
        ckeys, cactions = raw_keys[ids], raw_actions[ids]
        g_raw = _calibrate_raw(fc, model_n, norm_n, val, val_q, val_eps, ckeys, cactions, device)
        by_condition = {}
        for condition, ctx in eval_by_name.items():
            if condition == "D2":
                continue
            q = ctx["retr"].get("query")
            if q is None:
                # Static D1/D3 query vectors are captured in the retrieval trace.
                order = {"D1": 1, "D3": 2}.get(condition)
                q = captured["retrievals"][order]["query"] if order is not None else None
            if condition == "D2":
                q = captured["stream_input"]["keys"]["instruction"]
            raw_anchor, conf = _select_raw(ckeys, cactions, np.arange(len(ckeys)), q, fc, device)
            t0 = fc.apply_g(g_raw, conf); n = fc.steps_for(t0)
            pred = fc.rollout(model_n, norm_n, ctx["rows"]["f"], ctx["rows"]["s"], raw_anchor,
                              ctx["eps"], t0, n, device=device)
            base = ctx["per"]["N:noise:n10"]["mse"]
            mets = fc.row_metrics(pred, ctx["rows"]["a"][:, :fc.HORIZON],
                                  np.zeros_like(pred), ctx["scales"], device=device)
            by_condition[condition] = {**_mean_metrics(mets, base),
                "average_t0": float(t0.mean()), "average_n": float(n.mean()),
                "g_calibration": g_raw}
            if condition == "D3":
                np.savez_compressed(output_dir / f"{split}_s{seed}_R8_{kind}.npz",
                                    selected_training_ids=ids,
                                    selected_keys=ckeys.astype(np.float16),
                                    prediction_mean=pred.mean(0).astype(np.float16))
        result["R8"]["conditions"][kind] = by_condition
        logger(f"{split}/s{seed} R8 {kind} equal-count memory evaluated")

    d2 = captured["d2"]
    d2_query = captured["stream_input"]["keys"]["instruction"]
    d2_order = captured["stream_input"]["order"]
    d2_metrics, d2_predictions = _eval_raw_d2_prequential(
        fc, model_n, norm_n, d2["rows"], d2_query, d2["eps"], d2["scales"],
        d2["per"]["N:noise:n10"]["mse"], raw_keys, raw_actions, d2_order,
        budget=initial_count, seed=seed + 912, device=device)
    for kind in selected:
        result["R8"]["conditions"][kind]["D2"] = d2_metrics[kind]
        np.savez_compressed(output_dir / f"{split}_s{seed}_R8_{kind}_D2.npz",
                            prediction=d2_predictions[kind].astype(np.float16),
                            target=d2["rows"]["a"][:, :fc.HORIZON].astype(np.float16),
                            order=np.asarray(d2_order, np.int32))
    logger(f"{split}/s{seed} R8 D2 prequential raw-memory controls evaluated")

    # R5 risk predictor on D1/D2/D3; t0 is selected without using evaluation targets.
    result["R5"]["heldout"] = {}
    for condition, ctx in eval_by_name.items():
        query = (captured["retrievals"][1]["query"] if condition == "D1" else
                 captured["retrievals"][2]["query"] if condition == "D3" else
                 captured["stream_input"]["keys"]["instruction"])
        feats = _key_features(fc, query, view[2],
                              fc.decode_anchor(m1, ctx["retr"]["unit_b"], ctx["rows"]), ctx["rows"], m1)
        risk = predict_confidence_model(h, feats)
        t0 = fc.apply_g(g, -risk); n = fc.steps_for(t0)
        anchor = fc.decode_anchor(m1, ctx["retr"]["unit_b"], ctx["rows"])
        pred = fc.rollout(model_a, norm_a, ctx["rows"]["f"], ctx["rows"]["s"], anchor,
                          ctx["eps"], t0, n, device=device)
        mets = fc.row_metrics(pred, ctx["rows"]["a"][:, :fc.HORIZON],
                              np.zeros_like(pred), ctx["scales"], device=device)
        result["R5"]["heldout"][condition] = {**_mean_metrics(mets, ctx["per"]["N:noise:n10"]["mse"]),
            "risk_coverage": risk_coverage(mets["mse"], risk),
            "t0_bins": t0_confidence_bins(risk), "average_t0": float(t0.mean()),
            "average_n": float(n.mean())}
        if condition == "D3":
            np.savez_compressed(output_dir / f"{split}_s{seed}_R5_prediction.npz",
                                risk=risk.astype(np.float32), t0=t0,
                                prediction_mean=pred.mean(0).astype(np.float16))
        logger(f"{split}/s{seed} R5 {condition} evaluated")
    return result


def _json_value(x):
    if isinstance(x, dict): return {str(k): _json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_json_value(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def run_interventions(output: Path, *, splits=("A", "B"), seeds=(0, 1, 2),
                      device_name="cuda", threads=8, cache_dir=None, dataset_dir=None,
                      m1_epochs=60, f_epochs=80, batch_size=512, smoke=False,
                      resume=False) -> Path:
    """Retrain the canonical fair_compare setup and run R5--R8 across D1/D2/D3."""
    from datetime import datetime, timezone
    from . import anchor_flow as af
    from . import fair_compare as fc
    from .splits import SPLITS, subset_rows as _subset
    from .experiment import DEFAULT_CACHE, DEFAULT_DATASET, load_rows
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if output.exists() and not resume: raise FileExistsError(f"refusing to overwrite {output}")
    if resume:
        partial = output / "results.partial.json"
        if not partial.is_file(): raise FileNotFoundError(f"resume requires {partial}")
        prior = json.loads(partial.read_text())
        if prior.get("config", {}).get("splits") != list(splits) or prior.get("config", {}).get("seeds") != list(seeds):
            raise ValueError("resume split/seed configuration differs from partial run")
        runs = list(prior.get("runs", []))
    else:
        output.mkdir(parents=True)
        runs = []
    (output / "figures").mkdir(exist_ok=True)
    log_path = output / "progress.log"
    def log(message):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as stream: stream.write(line + "\n")
    cache_dir = Path(cache_dir or DEFAULT_CACHE); dataset_dir = Path(dataset_dir or DEFAULT_DATASET)
    start = time.perf_counter(); device = torch.device(device_name)
    rows = load_rows(cache_dir, dataset_dir, max_horizon=fc.HORIZON)
    task_ids = sorted({t for split in splits for t in (*SPLITS[split][0], *SPLITS[split][1])})
    rows = _subset(rows, np.isin(rows["task_index"], task_ids) & rows["valid_h25"])
    log(f"loaded {len(rows['a'])} H=25 rows")
    texts, emb = fc.precompute_texts(rows, ("instruction",), cache_dir)
    scales = {}
    for split in splits:
        train_eps = np.unique(rows["episode_id"][np.isin(rows["task_index"], SPLITS[split][0])])
        scales[split] = fc.fit_physical_scales(cache_dir, train_eps, max_episodes=300)
    cfg = {"seeds": list(seeds), "splits": list(splits), "source": "instruction",
           "horizon": fc.HORIZON, "m1_epochs": m1_epochs, "f_epochs": f_epochs,
           "batch_size": batch_size, "paired_evaluation_noise_samples": fc.SAMPLES,
           "device": device_name, "threads": threads,
           "source_conditions": ["D1", "D2", "D3"], "progressive": True}
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    completed = {(str(run["split"]), int(run["seed"])) for run in runs}
    for split in splits:
        for seed in seeds:
            if (str(split), int(seed)) in completed:
                log(f"resume skip completed {split}/seed={seed}")
                continue
            log(f"start canonical fit {split}/seed={seed}")
            kw = dict(rows=rows, split=split, seed=seed, sources=("instruction",), texts=texts,
                      emb=emb, device=device, scales=scales[split], m1_epochs=m1_epochs,
                      f_epochs=f_epochs, batch_size=batch_size, smoke=smoke, log=log, example_rows=0)
            captured = _capture_fair_run(fc, af, kw)
            run = _run_priority_experiments(captured, split=split, seed=seed, device=device,
                                             output_dir=output, logger=log, bridge_epochs=f_epochs)
            runs.append(run)
            payload = {"config": cfg, "physical_scales": scales, "runs": runs}
            (output / "results.partial.json").write_text(json.dumps(_json_value(payload), indent=1))
            log(f"done {split}/seed={seed}, total elapsed={(time.perf_counter()-start)/60:.1f} min")
    payload = {"schema_version": "anchor-flow-review-tests-v2", "config": cfg,
               "physical_scales": scales, "runs": runs,
               "measured_interventions": {"R5": True, "R6": True, "R7": True, "R8": True,
                                          "R1_to_R4": False},
               "limitations": [
                   "D2 raw-memory control uses a fixed item budget equal to the initial M6 unit count; M6 itself grows prequentially.",
                   "Bridge sigma is in standardized action coordinates; two predeclared values are .03 and .10."]}
    (output / "results.json").write_text(json.dumps(_json_value(payload), indent=1, ensure_ascii=False) + "\n")
    (output / "results.partial.json").unlink(missing_ok=True)
    _write_intervention_summary(output / "summary.md", payload)
    _plot_intervention_results(output / "figures", runs)
    log(f"finished {len(runs)} split-seed runs in {(time.perf_counter()-start)/60:.1f} min")
    return output


def _secondary_r2(captured, *, seed, device):
    """Paired R2 action-flow comparison: explicit zero r versus N(0,I) r."""
    from . import fair_compare as fc
    m1 = captured["m6"].backbone
    rng = np.random.default_rng(seed + 1701)
    result, predictions = {}, {}
    for name, ctx in (("D1", captured["d1"]), ("D2", captured["d2"]),
                      ("D3", captured["d3"])):
        rows, retr = ctx["rows"], ctx["retr"]
        base = retr["unit_b"]
        zero = np.zeros((len(base), m1.residual_dim), np.float32)
        normal = rng.standard_normal(zero.shape).astype(np.float32)
        anchor0 = m1.decode(base, zero, rows["f"], rows["s"])
        anchorN = m1.decode(base, normal, rows["f"], rows["s"])
        t0 = ctx["per"]["A:unit:cal_sim"]["t0"]
        n = ctx["per"]["A:unit:cal_sim"]["n"]
        out = {}
        for tag, anchor in (("r_zero", anchor0), ("r_normal", anchorN)):
            target = rows["a"][:, :fc.HORIZON]
            out[tag] = {"decoder_mse": float(np.mean((anchor-target) ** 2)),
                        "decoder_mse_pos": float(np.mean((anchor[..., :3]-target[..., :3]) ** 2)),
                        "decoder_mse_rot": float(np.mean((anchor[..., 3:6]-target[..., 3:6]) ** 2)),
                        "decoder_mse_grip": float(np.mean((anchor[..., 6]-target[..., 6]) ** 2))}
            pred = fc.rollout(captured["eval"][0]["flows"]["A"][0],
                              captured["eval"][0]["flows"]["A"][1], rows["f"], rows["s"],
                              anchor, ctx["eps"], t0, n, device=device)
            metrics = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON], np.zeros_like(pred),
                                     ctx["scales"], device=device)
            out[tag].update(_mean_metrics(metrics, ctx["per"]["N:noise:n10"]["mse"]))
            if name == "D3": predictions[tag] = pred.mean(0).astype(np.float16)
        out["delta_normal_minus_zero"] = out["r_normal"]["mse"] - out["r_zero"]["mse"]
        result[name] = out
    return result, predictions


def _secondary_r3(captured):
    """R3 center/state-set functional-distance comparison over M6 prototypes."""
    m6 = captured["m6"]; m1 = m6.backbone
    centers = np.asarray(captured.get("m6_initial_centers", m6.prototypes), np.float32)
    bases = np.asarray(captured["bases"], np.float32)
    # Membership is reconstructed from the nearest base-space prototype; this
    # matches M6's first candidate stage, with functional ties being rare.
    labels = np.argmin(((bases[:, None] - centers[None]) ** 2).sum(-1), axis=1)
    members = [np.flatnonzero(labels == i) for i in range(len(centers))]
    medoids = centers.copy()
    for i, ids in enumerate(members):
        if len(ids) < 2: continue
        if len(ids) > 32:
            ids = ids[np.linspace(0, len(ids)-1, 32, dtype=int)]
        sig = np.stack([m6._functional_embedding(bases[j]) for j in ids])
        medoids[i] = bases[ids[np.argmin(((sig[:, None] - sig[None]) ** 2).mean(-1))]]

    common = captured["calibration_input"]["val"]
    common_ids = np.linspace(0, len(common["s"])-1, min(64, len(common["s"])), dtype=int)
    common_f, common_s = common["f"][common_ids], common["s"][common_ids]
    rng = np.random.default_rng(211)
    pairs = [(i, j) for i in range(len(centers)) for j in range(i+1, len(centers))]
    if len(pairs) > 500:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 500, replace=False)]
    values = {f"{center}/{states}": [] for center in ("mean", "medoid")
              for states in ("common", "pair_members")}
    def distance(left, right, f, s):
        _, r = m1.prior(f, s)
        dl = m1.decode(np.repeat(left[None], len(s), 0), r, f, s)
        dr = m1.decode(np.repeat(right[None], len(s), 0), r, f, s)
        return float(np.mean((dl - dr) ** 2))
    for i, j in pairs:
        for center_name, arr in (("mean", centers), ("medoid", medoids)):
            values[f"{center_name}/common"].append(distance(arr[i], arr[j], common_f, common_s))
            ids = np.r_[members[i][:8], members[j][:8]]
            if len(ids):
                f = captured["train_fit"]["f"][ids]
                s = captured["train_fit"]["s"][ids]
                values[f"{center_name}/pair_members"].append(distance(arr[i], arr[j], f, s))
    return {"unit_count": len(centers), "pairs": len(pairs),
            "membership_note": "M6 row membership reconstructed by nearest base prototype",
            "mean_functional_mse": {k: float(np.mean(v)) if v else None for k, v in values.items()},
            "unit_support_counts_reconstructed": [int(len(x)) for x in members]}


def _secondary_r1(captured, *, seed, device):
    """Cross-state motion invariance and two consistency weights for M1."""
    from ..methods.m1 import M1
    from ..methods._common import norm as _norm, tensor as _tensor
    class ConsistencyM1(M1):
        """M1 variant whose post-fit update penalizes state-driven base drift."""
        pass
    mbase = captured["m6"].backbone
    train = captured["train_fit"]
    train_data = {"actions": train["a"][:, :25], "state": train["s"],
                  "feature": train["f"], "key": train["k"],
                  "episode_id": train["episode_id"], "init_state_id": train["episode_id"],
                  "seed": seed + 19}
    fitted = ConsistencyM1(independence_weight=.01, beta_r=mbase.beta_r,
                width=mbase.width, epochs=60, batch_size=512, device=str(device),
                seed_value=seed+19, base_dim=mbase.base_dim, residual_dim=mbase.residual_dim)
    fitted.fit(train_data)
    def finetune(model, weight):
        torch.manual_seed(seed + int(weight * 10000) + 501)
        opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.base_decoder.parameters()),
                                lr=1e-4, weight_decay=1e-5)
        a = _tensor(_norm(train["a"][:, :25].reshape(len(train["a"]), -1), model.am, model.ass), device)
        s = _tensor(_norm(train["s"], model.sm, model.ss), device)
        rng = np.random.default_rng(seed + 601)
        std = torch.as_tensor(np.maximum(train["s"].std(0), 1e-3) / model.ss, device=device)
        for _ in range(4):
            order = rng.permutation(len(a))
            for lo in range(0, len(order), 512):
                ix = torch.as_tensor(order[lo:lo+512], device=device)
                latent = model.encoder(torch.cat([a[ix], s[ix]], 1))
                perturbed_state = s[ix] + torch.randn_like(s[ix]) * .05 * std
                latent_pert = model.encoder(torch.cat([a[ix], perturbed_state], 1))
                b = latent[:, :model.base_dim]; bp = latent_pert[:, :model.base_dim]
                z = torch.zeros((len(ix), model.residual_dim), device=device)
                pred = model._decode_base(b, torch.zeros((len(ix), model.feature_dim), device=device), s[ix])
                predp = model._decode_base(bp, torch.zeros((len(ix), model.feature_dim), device=device), s[ix])
                loss = (pred-a[ix]).square().mean() + weight * (
                    (b-bp).square().mean() + .1*(pred-predp).square().mean())
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.encoder.eval(); model.base_decoder.eval()
    result, saved = {}, {}
    eval_rows = captured["d3"]["rows"]
    rng = np.random.default_rng(seed + 702)
    other = np.arange(len(eval_rows["s"]))
    for weight in (.1, 1.0):
        finetune(fitted, weight) if weight == .1 else None
        model = fitted
        if weight == 1.0:
            # Start the stronger penalty from the same supervised checkpoint.
            model = ConsistencyM1(independence_weight=.01, beta_r=mbase.beta_r, width=mbase.width,
                       epochs=60, batch_size=512, device=str(device), seed_value=seed+19,
                       base_dim=mbase.base_dim, residual_dim=mbase.residual_dim)
            model.fit(train_data); finetune(model, weight)
        base, _ = model.encode(eval_rows["a"][:, :25], eval_rows["s"])
        permutation = rng.permutation(len(other))
        alt_s = eval_rows["s"][permutation]
        alt_f = eval_rows["f"][permutation]
        zero = np.zeros((len(base), model.residual_dim), np.float32)
        dec = lambda b, s, r: model.decode(b, r, eval_rows["f"], s)
        target = eval_rows["a"][:, :25]
        result[f"lambda={weight:g}"] = r1_transfer_diagnostics(
            base, eval_rows["s"], alt_s, target,
            lambda b, s, r: model.decode(b, np.zeros((len(b), model.residual_dim), np.float32),
                                          alt_f if s is alt_s else eval_rows["f"], s))
        result[f"lambda={weight:g}"]["state_perturbation_base_mse"] = float(np.mean(
            (model.encode(eval_rows["a"][:, :25], eval_rows["s"] + .05*eval_rows["s"].std(0))[0] - base)**2))
        saved[f"lambda{weight:g}"] = {
            name: {k: v.detach().cpu() for k, v in getattr(model, name).state_dict().items()}
            for name in ("encoder", "prior_net", "base_decoder", "residual_decoder")
        }
    base, _ = mbase.encode(eval_rows["a"][:, :25], eval_rows["s"])
    result["canonical_M1"] = r1_transfer_diagnostics(
        base, eval_rows["s"], alt_s, eval_rows["a"][:, :25],
        lambda b, s, r: mbase.decode(b, np.zeros((len(b), mbase.residual_dim), np.float32),
                                     eval_rows["f"], s))
    return result, saved


def _secondary_r4(captured, *, seed, device, cache_dir):
    """Train multi-positive E with M6 functional-distance positives, then top-k blend."""
    from . import fair_compare as fc
    from . import anchor_flow as af
    from . import clip_retrieval as cr
    m6, m1 = captured["m6"], captured["m6"].backbone
    train = captured["train_fit"]
    actions = train["a"][:, :fc.HORIZON]
    bases = np.asarray(captured["bases"], np.float32)
    # M6's fixed probe defines functional distance and epsilon. Build a compact
    # lookup once so each InfoNCE minibatch can form all functional positives.
    probe = m6._functional_probe
    p = len(probe["state"]); h = int(np.prod(m1.action_shape))
    signatures = np.empty((len(bases), p*h), np.float16)
    for lo in range(0, len(bases), 32):
        hi = min(len(bases), lo+32); count = hi-lo
        bb = np.repeat(bases[lo:hi], p, axis=0)
        rr = np.tile(probe["residual"], (count, 1))
        ff = np.tile(probe["feature"], (count, 1)); ss = np.tile(probe["state"], (count, 1))
        decoded = m1.decode(bb, rr, ff, ss).reshape(count, -1)
        signatures[lo:hi] = (decoded / np.sqrt(p*h)).astype(np.float16)
    base_lookup = {np.ascontiguousarray(b, dtype=np.float32).tobytes(): i
                   for i, b in enumerate(bases)}
    sig_tensor = torch.as_tensor(signatures.astype(np.float32), device=device)
    original_positive = cr._positive_mask
    def functional_positive(batch_b, batch_episode, _distance_threshold, *, enabled):
        n = len(batch_b); identity = torch.eye(n, dtype=torch.bool, device=batch_b.device)
        if not enabled: return identity
        array = batch_b.detach().cpu().numpy().astype(np.float32, copy=False)
        ids = [base_lookup.get(np.ascontiguousarray(row).tobytes(), -1) for row in array]
        if min(ids, default=-1) < 0:
            return original_positive(batch_b, batch_episode, _distance_threshold, enabled=True)
        vectors = sig_tensor[torch.as_tensor(ids, device=device)]
        distance = torch.cdist(vectors, vectors).square()
        cross = batch_episode[:, None] != batch_episode[None, :]
        return identity | ((distance <= float(m6.epsilon)) & cross)

    texts, emb = fc.precompute_texts(train, ("instruction",), Path(cache_dir))
    tf = fc.attach_text(train, texts, emb, "instruction")
    qrows = af._normal_query_rows(tf)
    task_ids = tuple(np.unique(train["task_index"]).tolist())
    with patch.object(cr, "_positive_mask", functional_positive):
        model, fit = cr.fit_dual_encoder(
            qrows, bases, use_subtask=True, subtask_mode="sentence", nearby_positive=True,
            seed=seed+701, epochs=cr.CLIP_EPOCHS, patience=cr.CLIP_PATIENCE, device=device,
            allowed_task_indices=task_ids)
    fit["positive_rule"] = "all cross-episode pairs with M6 functional squared distance <= epsilon"
    center = np.asarray(captured.get("m6_initial_centers", m6.prototypes), np.float32)
    with torch.no_grad():
        hunit = model.encode_skill(torch.as_tensor(center, device=device)).cpu().numpy()
    result, saved = {"fit": {k: v for k, v in fit.items() if k not in ("normalizers", "history")},
                     "functional_epsilon": float(m6.epsilon), "functional_probe_rows": p,
                     "conditions": {}}, {}
    for name, ctx in (("D1", captured["d1"]), ("D2", captured["d2"]),
                      ("D3", captured["d3"])):
        rows = ctx["rows"]
        q = af._query_embedding(model, fit, af._normal_query_rows(rows), device=device)
        score = fc._nrm(q) @ fc._nrm(hunit).T * float(model.logit_scale.exp().detach().cpu())
        outputs = {}
        for k in (1, 5):
            ix, weights = topk_candidates(score, k)
            blend = np.sum(center[ix] * weights[..., None], axis=1)
            anchor = fc.decode_anchor(m1, blend, rows)
            t0 = ctx["per"]["A:unit:cal_sim"]["t0"]
            n = ctx["per"]["A:unit:cal_sim"]["n"]
            pred = fc.rollout(captured["eval"][0]["flows"]["A"][0],
                              captured["eval"][0]["flows"]["A"][1], rows["f"], rows["s"],
                              anchor, ctx["eps"], t0, n, device=device)
            mets = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON], np.zeros_like(pred),
                                  ctx["scales"], device=device)
            outputs[f"top{k}"] = _mean_metrics(mets, ctx["per"]["N:noise:n10"]["mse"])
            outputs[f"top{k}"]["mean_candidate_count"] = float(ix.shape[1])
            if name == "D3": saved[f"{name}_top{k}"] = pred.mean(0).astype(np.float16)
            if k == 5: saved[f"{name}_query"] = q.astype(np.float16)
        result["conditions"][name] = outputs
    saved["E_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    saved["unit_skill_embeddings"] = hunit.astype(np.float16)
    return result, saved


def _secondary_r6_r7(captured, bridges, *, device):
    """Paired per-row metrics for every independent R6 grid cell and R7 path."""
    from . import fair_compare as fc
    normal_model, normal_norm = captured["eval"][0]["flows"]["N"]
    anchor_model, anchor_norm = captured["eval"][0]["flows"]["A"]
    by_condition = {"D1": captured["d1"], "D2": captured["d2"], "D3": captured["d3"]}
    r6, r7 = {}, {}
    for condition, ctx in by_condition.items():
        rows, eps = ctx["rows"], ctx["eps"]
        anchor = fc.decode_anchor(captured["m6"].backbone, ctx["retr"]["unit_b"], rows)
        baseline = ctx["per"]["N:noise:n10"]["mse"]
        grid = {}
        for item in independent_t0_n_grid():
            t0, n = float(item["t0"]), int(item["n"])
            pred = fc.rollout(anchor_model, anchor_norm, rows["f"], rows["s"], anchor,
                              eps, np.full(len(rows["a"]), t0), np.full(len(rows["a"]), n), device=device)
            mets = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON], np.zeros_like(pred),
                                  ctx["scales"], device=device)
            grid[f"t0={t0:g},n={n}"] = _mean_metrics(mets, baseline)
        r6[condition] = grid
        t0 = ctx["per"]["A:unit:cal_sim"]["t0"]
        n = ctx["per"]["A:unit:cal_sim"]["n"]
        canonical_pred = fc.rollout(anchor_model, anchor_norm, rows["f"], rows["s"], anchor,
                                    eps, t0, n, device=device)
        canonical_metrics = fc.row_metrics(canonical_pred, rows["a"][:, :fc.HORIZON],
                                           np.zeros_like(canonical_pred), ctx["scales"], device=device)
        paths = {"F_anchor": _mean_metrics(canonical_metrics, baseline)}
        for sigma, (model, norm) in bridges.items():
            pred = _bridge_rollout(model, norm, rows, anchor, eps, sigma=sigma, n=10, device=device)
            mets = fc.row_metrics(pred, rows["a"][:, :fc.HORIZON], np.zeros_like(pred),
                                  ctx["scales"], device=device)
            paths[f"bridge_sigma={sigma:g}"] = _mean_metrics(mets, baseline)
        r7[condition] = paths
    return r6, r7


def run_secondary(output: Path, *, splits=("A", "B"), device_name="cuda", threads=8,
                  m1_epochs=60, f_epochs=80, batch_size=512) -> Path:
    """Run R1--R4 for seed 0 and append the measured values to a completed run."""
    from datetime import datetime, timezone
    from . import anchor_flow as af
    from . import fair_compare as fc
    from .splits import SPLITS, subset_rows as _subset
    from .experiment import DEFAULT_CACHE, DEFAULT_DATASET, load_rows
    output = Path(output)
    if not output.is_dir() or not (output / "results.json").is_file():
        raise FileNotFoundError("complete --execute-priority results.json is required")
    if (output / "R1_R4.json").exists():
        raise FileExistsError("refusing to overwrite R1_R4.json")
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device = torch.device(device_name)
    cache_dir = Path(DEFAULT_CACHE); dataset_dir = Path(DEFAULT_DATASET)
    rows = load_rows(cache_dir, dataset_dir, max_horizon=fc.HORIZON)
    tasks = sorted({t for split in splits for t in (*SPLITS[split][0], *SPLITS[split][1])})
    rows = _subset(rows, np.isin(rows["task_index"], tasks) & rows["valid_h25"])
    texts, emb = fc.precompute_texts(rows, ("instruction",), cache_dir)
    scales = {}
    for split in splits:
        train_eps = np.unique(rows["episode_id"][np.isin(rows["task_index"], SPLITS[split][0])])
        scales[split] = fc.fit_physical_scales(cache_dir, train_eps, max_episodes=300)
    results = []
    log_path = output / "secondary_progress.log"
    def log(message):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as stream: stream.write(line + "\n")
    bridge_records = []
    for split in splits:
        for seed in (0, 1, 2):
            log(f"secondary canonical capture {split}/seed{seed}")
            captured = _capture_fair_run(fc, af, dict(
                rows=rows, split=split, seed=seed, sources=("instruction",), texts=texts, emb=emb,
                device=device, scales=scales[split], m1_epochs=m1_epochs, f_epochs=f_epochs,
                batch_size=batch_size, smoke=False, log=log, example_rows=0))
            # Persist the R7 bridge checkpoints omitted by the first-pass runner.
            bridge_states = {}
            bridge_live = {}
            bridge_reproduction = {}
            for sigma in (.03, .1):
                bridge, bridge_norm, bridge_fit = _bridge_flow(
                    captured["train_fit"], captured["actions"], captured["flow_sources"],
                    seed+827, f_epochs, batch_size, device, sigma=sigma)
                bridge_states[f"sigma{sigma:g}"] = {"state": {k: v.detach().cpu() for k, v in bridge.state_dict().items()},
                    "norm": bridge_norm, "fit": bridge_fit}
                bridge_live[sigma] = (bridge, bridge_norm)
                ctx = captured["d3"]
                anchor = fc.decode_anchor(captured["m6"].backbone, ctx["retr"]["unit_b"], ctx["rows"])
                regenerated = _bridge_rollout(bridge, bridge_norm, ctx["rows"], anchor, ctx["eps"],
                                               sigma=sigma, n=10, device=device).mean(0)
                old_path = output / f"{split}_s{seed}_R7_sigma{sigma:g}.npz"
                old_pred = np.load(old_path)["prediction_mean"]
                bridge_reproduction[f"sigma{sigma:g}"] = float(np.max(np.abs(regenerated-old_pred)))
            torch.save(bridge_states, output / f"{split}_s{seed}_R7_bridge_models.pt")
            paired_r6, paired_r7 = _secondary_r6_r7(captured, bridge_live, device=device)
            bridge_records.append({"split": split, "seed": seed,
                                   "sigma03_best_epoch": bridge_states["sigma0.03"]["fit"]["best_epoch"],
                                   "sigma10_best_epoch": bridge_states["sigma0.1"]["fit"]["best_epoch"],
                                   "prediction_max_abs_difference_vs_primary": bridge_reproduction,
                                   "R6_paired": paired_r6, "R7_paired": paired_r7})
            if seed != 0:
                continue
            r1, r1_models = _secondary_r1(captured, seed=seed, device=device)
            r2, r2_pred = _secondary_r2(captured, seed=seed, device=device)
            r3 = _secondary_r3(captured)
            r4, r4_saved = _secondary_r4(captured, seed=seed, device=device, cache_dir=cache_dir)
            # Persist canonical and consistency-tuned M1 models for later inspection.
            torch.save({"R4_E_state": r4_saved.pop("E_state"), "R1_M1_variants": r1_models,
                        **r4_saved,
                        "M1_state": {name: {k: v.detach().cpu() for k, v in
                                             getattr(captured["m6"].backbone, name).state_dict().items()}
                                     for name in ("encoder", "prior_net", "base_decoder", "residual_decoder")},
                        "M6_centers": np.asarray(captured.get("m6_initial_centers", captured["m6"].prototypes))},
                       output / f"{split}_s0_R1_R4_models.pt")
            np.savez_compressed(output / f"{split}_s0_R1_R4_predictions.npz", **r2_pred,
                                **{k: v for k, v in r4_saved.items() if isinstance(v, np.ndarray)})
            results.append({"split": split, "seed": 0, "R1": r1, "R2": r2, "R3": r3, "R4": r4})
            log(f"R1-R4 complete {split}/seed0")
    payload = {"conditions": {"horizon": 25, "seeds": [0], "splits": list(splits),
                               "subtask": "instruction", "same_fair_compare_rows_and_noise": True},
               "method_details": {
                   "R1": "ConsistencyM1 inherits M1; after canonical fit, four passes optimize base reconstruction + lambda*(state-perturbed base MSE + 0.1 decoded action MSE), lambda 0.1 and 1.0; labels are post-hoc.",
                   "R2": "M1 retrieved base, paired F_anchor with residual fixed to zero versus one unscaled N(0,I) draw per query.",
                   "R3": "M6 functional decoder distance; 500 unit pairs at most; state sets are 64 common validation rows or up to 8 sampled members per unit; M6 row membership approximated by nearest base prototype.",
                   "R4": "InfoNCE positives are cross-episode pairs with squared M6 functional distance <= M6 epsilon; retrieve and softmax-blend top-1/top-5 unit bases.",
                   "R8_D2": "In the priority run's k-medoids selector, each episode uses a seeded visible-pool sample of max(4*budget,4096) candidates; random memory samples from the full visible pool. Both store the exact same item budget."},
               "runs": _json_value(results), "R7_bridge_model_checkpoints": bridge_records}
    (output / "R1_R4.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    main_path = output / "results.json"
    main = json.loads(main_path.read_text()); main["R1_R4"] = _json_value(results)
    main["R7_bridge_model_checkpoints"] = bridge_records
    main["R6_R7_paired_row_metrics"] = bridge_records
    main["measured_interventions"]["R1_to_R4"] = True
    main_path.write_text(json.dumps(main, indent=1, ensure_ascii=False) + "\n")
    with (output / "summary.md").open("a") as stream:
        stream.write("\n## R1–R4 — seed 0 measurements\n\n")
        stream.write("R1 labels are post-hoc only; R4 positives use the M6 functional-distance epsilon. Full D1/D2/D3 and dimension-wise metrics are in `R1_R4.json`.\n\n")
        stream.write("| Split | R1 canonical cross-state dir/grip/rot | λ=.1 | λ=1.0 | R2 D3 r=0 MSE | r~N MSE | R3 mean/common vs medoid/common | R4 D3 top1 | top5 |\n|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in results:
            r1 = row["R1"]
            fmt = lambda m: "/".join(f"{m['cross_state_agreement'][k]:.3f}" for k in ("direction", "gripper", "rotation"))
            r2 = row["R2"]["D3"]
            r3 = row["R3"]["mean_functional_mse"]
            r4 = row["R4"]["conditions"]["D3"]
            stream.write(f"| {row['split']} | {fmt(r1['canonical_M1'])} | {fmt(r1['lambda=0.1'])} | {fmt(r1['lambda=1'])} | {r2['r_zero']['mse']:.4f} | {r2['r_normal']['mse']:.4f} | {r3['mean/common']:.5f} / {r3['medoid/common']:.5f} | {r4['top1']['mse']:.4f} | {r4['top5']['mse']:.4f} |\n")
        stream.write("\nR7 bridge weights for all six split/seed cells are saved in `*_R7_bridge_models.pt`; query embeddings and predictions are stored in the corresponding NPZ files.\n")
    return output


def _write_intervention_summary(path: Path, payload: dict[str, Any]) -> None:
    runs = payload["runs"]
    lines = ["# Anchor flow R1–R8 intervention results (offline light)", "",
             f"- Conditions: H=25, instruction, Split {payload['config']['splits']}, seeds {payload['config']['seeds']}; same fair_compare fit/validation split, flow models, evaluation rows and paired noise.",
             "- Canonical reruns are checked rowwise against the saved fair_compare baseline; the absolute MSE difference is recorded per split/seed.",
             "- R7 bridge interpolation follows x_t=(1−t)A+t(a+σε), v=(a+σε)−A, integrated from t=1 to 0.", ""]
    for r in runs:
        lines.append(f"- {r['split']}/seed {r['seed']}: canonical D3 MSE difference {r['fair_compare_consistency']['absolute_difference']:.3g}; R5 t0={r['R5']['heldout']['D3']['average_t0']:.3f}, R8 item budget={r['R8']['item_budget']}.")
    lines += ["", "## R7 — correction path", "",
              "F_normal and F_anchor are the canonical fair_compare controls; bridge values below use the same training rows and paired D1/D2/D3 evaluation noise.",
              "| Split/seed | F_normal D3 | F_anchor/noise D3 | unit + F_anchor D3 | bridge σ=.03 | bridge σ=.10 |", "|---|---:|---:|---:|---:|---:|"]
    for r in runs:
        d3 = r["R7"]["conditions"]
        lines.append(f"| {r['split']}/{r['seed']} | {r['R7']['canonical_D3']['F_normal']:.4f} | {r['R7']['canonical_D3']['F_anchor']:.4f} | {r['R7']['canonical_D3']['unit_F_anchor']:.4f} | {d3['sigma=0.03']['by_condition']['D3']['mse']:.4f} | {d3['sigma=0.1']['by_condition']['D3']['mse']:.4f} |")
    lines += ["", "## R5 — learned confidence", "",
              "h is ridge-fit on one half of held-out training-suite episodes using max cosine, top-5 key density, and anchor/Prior disagreement; the other half chooses monotone t0. New-suite labels/errors are not used for fitting.",
              "| Split/seed | MSE D1 | MSE D2 | MSE D3 | D3 mean t0 | D3 risk coverage figure |", "|---|---:|---:|---:|---:|---|"]
    for r in runs:
        h = r["R5"]["heldout"]
        lines.append(f"| {r['split']}/{r['seed']} | {h['D1']['mse']:.4f} | {h['D2']['mse']:.4f} | {h['D3']['mse']:.4f} | {h['D3']['average_t0']:.3f} | risk-coverage in results.json |")
    lines += ["", "## R8 — equal-count raw interval memory", "",
              "Random and K-means-medoid raw stores use the same initial item count as M6 units and own validation-calibrated t0; full raw-NN fair_compare remains the capacity-unmatched reference.",
              "| Split/seed | budget | random D3 MSE | k-medoids D3 MSE | full raw reference |", "|---|---:|---:|---:|---:|"]
    for r in runs:
        x = r["R8"]["conditions"]
        lines.append(f"| {r['split']}/{r['seed']} | {r['R8']['item_budget']} | {x['random_ids']['D3']['mse']:.4f} | {x['kmedoids_ids']['D3']['mse']:.4f} | {r['R8']['full_raw_reference_D3']:.4f} |")
    lines += ["", "## R6 — independent t0 × n", "",
              "The full 6×5 grid is stored for D1/D2/D3 in results.json. Heatmaps show D3 MSE; n is varied independently of t0.",
              "", "## R1–R4", "",
              "Seed-0 follow-up measurements are reported in the section below and in R1_R4.json.",
              "", "## Reuse artifacts", "",
              "Per split/seed, `*_models.pt` stores M1/M6/E/F states and normalization, `*_embeddings.npz` stores validation/D1/D2/D3 query embeddings and retrieved bases, and R5/R6/R7/R8 NPZ files store predictions/metrics. `progress.log` records the run order and timing.", ""]
    path.write_text("\n".join(lines))


def _plot_intervention_results(directory: Path, runs: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Split/seed comparison for the requested R7 paths.
    labels = [f"{r['split']}/s{r['seed']}" for r in runs]
    x = np.arange(len(runs)); width = .2
    vals = [[r["R7"]["canonical_D3"]["F_normal"] for r in runs],
            [r["R7"]["canonical_D3"]["F_anchor"] for r in runs],
            [r["R7"]["canonical_D3"]["unit_F_anchor"] for r in runs],
            [r["R7"]["conditions"]["sigma=0.03"]["by_condition"]["D3"]["mse"] for r in runs],
            [r["R7"]["conditions"]["sigma=0.1"]["by_condition"]["D3"]["mse"] for r in runs]]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, v in enumerate(vals): ax.bar(x + (i-2)*width, v, width, label=("F_normal", "F_anchor/noise", "unit F_anchor", "bridge .03", "bridge .10")[i])
    ax.set_xticks(x, labels); ax.set_ylabel("D3 MSE"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(directory / "R7_D3_mse.png", dpi=140); plt.close(fig)
    t0s, ns = (.1, .2, .4, .6, .8, 1.), (1, 2, 3, 5, 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, split in zip(axes, ("A", "B"), strict=True):
        selected = [r for r in runs if r["split"] == split]
        matrix = np.zeros((len(t0s), len(ns)), np.float64)
        for i, t0 in enumerate(t0s):
            for j, n in enumerate(ns):
                vals = [r["R6"]["grid"]["D3"][f"t0={t0:g},n={n}"]["mse"] for r in selected]
                matrix[i, j] = np.mean(vals)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ns)), ns); ax.set_yticks(range(len(t0s)), t0s)
        ax.set_xlabel("Euler steps n"); ax.set_ylabel("start t0")
        ax.set_title(f"Split {split}: D3 MSE, seed mean")
        fig.colorbar(im, ax=ax, label="MSE")
    fig.savefig(directory / "R6_t0_n_heatmap.png", dpi=140); plt.close(fig)


def _build_report(payload: dict[str, Any], out: Path) -> str:
    baseline = payload["baseline"]
    lines = ["# Anchor-flow external review checks (offline light)", "",
             f"- Fair-comparison reference: `{baseline['path']}`.",
             "- This run audits reusable baselines and validates the R1–R8 analysis primitives. The referenced fair_compare run stores per-row metrics but not model weights or per-step predictions; no R variant is claimed as newly measured from that archive.",
             "- H=25, instruction-only, splits A/B, seeds 0–2 are the reference conditions. The reference run used paired rows/noise and F_normal.", "",
             "## R results", ""]
    conclusions = {
        "R1": "未計算。fair_compare アーカイブに M1/M6 の重みと復号行動がなく、状態摂動・scene transfer を再評価できない。診断 API と事後 motion label 関数は実装・合成テスト済み。",
        "R2": "未計算。r=0 / N(0,I) の新しい復号比較には M1 decoder が必要で、参照アーカイブには重みがない。",
        "R3": "未計算。状態集合×mean/medoid の機能距離は decoder が必要。比較 API は実装・合成テスト済み。",
        "R4": "未計算。多正例 E の追加学習状態が保存されていない。多正例 InfoNCE と top-k 候補 API は実装・合成テスト済み。",
        "R5": "既存比較の参考値のみ。fair_compare は sim/entropy から検証データのみで単調 t0 を較正しているが、今回指定の密度・予測アンカー誤差を入力にした h は未学習。risk–coverage API は実装・合成テスト済み。",
        "R6": "既存比較では t0 と n が連動し独立格子ではない。t0×n 30条件を定義する API は実装・合成テスト済みだが、本番ロールアウトは未計算。",
        "R7": "fair_compare の F_anchor / F_normal 結果を基準として参照可能。bridge 型 F は参照アーカイブにモデル重みがなく未学習・未評価。bridge 軌道と速度目標 API は実装・合成テスト済み。",
        "R8": "既存 fair_compare の全件 raw NN を参考比較として保持。単位数と件数を揃えた random / k-medoids raw memory は選択 API を実装・合成テスト済みだが、本番 MSE は未計算。",
    }
    for key, text in conclusions.items():
        lines.append(f"- **{key}:** {text}")
    lines += ["", "## Artifact limitations", "",
              "The supplied fair_compare output contains aggregate JSON and per-row MSE/best8/t0/dn, but no encoder/decoder/F state_dict, query embeddings, anchors, or per-step predictions. Reusing it cannot answer R1–R4 or execute the new R5–R8 interventions. To keep the comparison fair, those new values are left explicitly uncomputed rather than reconstructed from aggregate metrics or trained under unverified conditions.",
              "", "## Implementation checks", "",
              "`results.json` records the implemented diagnostic primitives and this run's scope. Synthetic tests cover label agreement, residual ablation, state/center distance cases, multi-positive InfoNCE, top-k weighting, confidence risk–coverage, independent t0/n grid, bridge targets, and matched-budget raw memory selection.", ""]
    lines[lines.index("## Artifact limitations") - 1:lines.index("## Artifact limitations")] = [
        "## Reused fair_compare baseline (D3 / instruction)", "",
        "| Split | F_normal | F_anchor/noise | unit + F_anchor | raw + F_normal | shuffled | oracle unit |",
        "|---|---:|---:|---:|---:|---:|---:|"] + [
        "| " + split + " | " + " | ".join(
            f"{payload['baseline']['D3_instruction_baseline'][split][name]['mse']['mean']:.4f} ± "
            f"{payload['baseline']['D3_instruction_baseline'][split][name]['mse']['sd']:.4f}"
            for name in ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "N:raw:cal_sim",
                         "A:shuf:cal_sim", "A:gold_unit:cal_sim")) + " |"
        for split in ("A", "B")] + [
        "", "![Existing fair_compare D3 baselines](figures/fair_compare_d3_baseline.png)", ""]
    return "\n".join(lines)


def run(output: Path, baseline: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    figures = output / "figures"
    figures.mkdir()
    audit = audit_fair_compare(baseline)
    reference = audit["payload"]
    methods = ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "N:raw:cal_sim",
               "A:shuf:cal_sim", "A:gold_unit:cal_sim")
    baseline_d3 = {}
    for split in ("A", "B"):
        key = f"{split}|instruction|D3|all"
        cell = reference["aggregate"][key]
        baseline_d3[split] = {name: {metric: cell[name][metric]
                                    for metric in ("mse", "best8", "pos_term_mm", "grip_agree")}
                              for name in methods}
    _plot_baseline_d3(baseline_d3, figures / "fair_compare_d3_baseline.png")
    payload = {"schema_version": "anchor-flow-review-tests-v1",
               "conditions": {"horizon": 25, "splits": ["A", "B"], "seeds": [0, 1, 2],
                              "subtask_source": "instruction", "comparison_reference": "fair_compare"},
               "baseline": {"path": audit["path"],
                            "stored_models": audit["stored_models"],
                            "stored_per_step_predictions": audit["stored_per_step_predictions"],
                            "run_count": len(audit["payload"]["runs"]),
                            "duration_minutes": audit["payload"].get("duration_minutes"),
                            "metrics": ["mse", "best8", "dimension-wise mse", "paired better/worse", "dn"],
                            "D3_instruction_baseline": baseline_d3,
                            "figure": "figures/fair_compare_d3_baseline.png"},
               "implemented_analyses": {"R1": "cross-state decode motion-label agreement",
                   "R2": "encoded/zero/normal residual ablation", "R3": "common/member state sets × mean/medoid",
                   "R4": "multi-positive InfoNCE and top-k candidates", "R5": "heldout ridge risk predictor + risk coverage",
                   "R6": independent_t0_n_grid(), "R7": "bridge interpolation and velocity target",
                   "R8": "equal-count random and k-means-medoid raw memories"},
               "measured_interventions": {k: "not_run" for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")},
               "limitation": "Reference artifacts omit state_dicts and per-step arrays needed to reproduce the interventions."}
    (output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output / "summary.md").write_text(_build_report(payload, output))
    return output


def _plot_baseline_d3(data: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"N:noise:n10": "F_normal", "A:noise:n10": "F_anchor/noise",
              "A:unit:cal_sim": "unit/F_anchor", "N:raw:cal_sim": "raw/F_normal",
              "A:shuf:cal_sim": "shuffled", "A:gold_unit:cal_sim": "oracle unit"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, split in zip(axes, ("A", "B"), strict=True):
        names = list(labels)
        means = [data[split][name]["mse"]["mean"] for name in names]
        errors = [data[split][name]["mse"]["sd"] for name in names]
        ax.bar(np.arange(len(names)), means, yerr=errors, capsize=3)
        ax.set_xticks(np.arange(len(names)), [labels[x] for x in names], rotation=35, ha="right")
        ax.set_title(f"Split {split}: D3 instruction / H=25")
        ax.set_ylabel("MSE (seed mean ± SD)"); ax.grid(axis="y", alpha=.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-priority", action="store_true",
                        help="retrain canonical models and run R5--R8 across seeds/splits")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted --execute-priority run from results.partial.json")
    parser.add_argument("--execute-secondary", action="store_true",
                        help="run seed-0 R1--R4 using canonical fair_compare captures")
    parser.add_argument("--smoke", action="store_true", help="small data/epoch smoke; not reportable results")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--flow-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--baseline", type=Path, default=Path(
        str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/fair-compare-20261028T101842Z')))
    args = parser.parse_args(argv)
    if args.execute_secondary:
        result = run_secondary(args.output, splits=tuple(args.splits.split(",")),
            device_name=args.device, threads=args.threads, m1_epochs=args.m1_epochs,
            f_epochs=args.flow_epochs, batch_size=args.batch_size)
    elif args.execute_priority:
        splits = tuple(args.splits.split(",")); seeds = tuple(int(x) for x in args.seeds.split(","))
        if args.smoke:
            splits, seeds = ("A",), (0,)
            args.m1_epochs = min(args.m1_epochs, 3); args.flow_epochs = min(args.flow_epochs, 3)
        result = run_interventions(args.output, splits=splits,
            seeds=seeds, device_name=args.device,
            threads=args.threads, m1_epochs=args.m1_epochs, f_epochs=args.flow_epochs,
            batch_size=args.batch_size, smoke=args.smoke, resume=args.resume)
    else:
        result = run(args.output, args.baseline)
    print(result)


if __name__ == "__main__":
    main()
