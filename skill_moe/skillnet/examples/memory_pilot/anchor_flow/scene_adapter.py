#!/usr/bin/env python3
"""Train small scene adapters for retrieved action drafts and evaluate offline.

The LIBERO-90 SkillNet/M1 weights and memory are read-only.  Adapters see only
the same centered-key top-5 retrieval used by S2, except for explicit oracle
diagnostics.  No environment rollout is started.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
for p in (str(HERE), str(EXAMPLES)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from key_check import (DEFAULT_CHECKPOINT, DEFAULT_HELDOUT, DEFAULT_ROWS,
                       ROW_FIELDS, _read_npz)
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from memory import load_memory
from train_retrieval_key import _sample_task_queries

DEFAULT_MEMORY = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/memory/initial_libero90.pkl')
DEFAULT_OUT = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot')
DEFAULT_NORM = str(RESEARCH_STATE_DIR / 'checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/assets/jsw19/libero_90_v1/norm_stats.json')
TAU = 0.05
TOP_K = 5
T0S = (0.2, 0.4, 0.6, 1.0)
MM_SCALE = np.asarray([0.0111, 0.0122, 0.0110], np.float32)


def _offline_helpers():
    """Reuse the offline sampler metrics through normal module imports."""
    from anchor_v2_offline import cluster_bootstrap, denoising_steps, row_errors
    return denoising_steps, row_errors, cluster_bootstrap


def _norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _retrieval(query_keys, query_eps, query_init, candidate_ids, keys, episodes, initials):
    """Exact S2 centered-key top-5 and cosine-softmax weights."""
    mem = np.asarray(keys[candidate_ids], np.float32)
    mean = mem.mean(axis=0, dtype=np.float64).astype(np.float32)
    memc = _norm(mem - mean)
    qc = _norm(np.asarray(query_keys, np.float32) - mean)
    local_ids = np.empty((len(qc), TOP_K), np.int64)
    sims = np.empty((len(qc), TOP_K), np.float32)
    for start in range(0, len(qc), 128):
        stop = min(start + 128, len(qc))
        scores = qc[start:stop] @ memc.T
        for j, row in enumerate(scores):
            i = start + j
            row[(episodes[candidate_ids] == query_eps[i])
                | (initials[candidate_ids] == query_init[i])] = -np.inf
            idx = np.argpartition(row, -TOP_K)[-TOP_K:]
            idx = idx[np.lexsort((idx, -row[idx]))]
            local_ids[i] = idx
            sims[i] = row[idx]
    logits = sims.astype(np.float64) / TAU
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return local_ids, sims, weights.astype(np.float32), mean


def _nearest_action(query_actions, query_eps, query_init, candidate_ids, actions, episodes, initials):
    """Nearest normalized action chunk, excluding same episode and init."""
    q = np.asarray(query_actions, np.float32)[..., :7].reshape(len(query_actions), -1)
    m = np.asarray(actions[candidate_ids], np.float32)[..., :7].reshape(len(candidate_ids), -1)
    qn = np.sum(q * q, axis=1)
    mn = np.sum(m * m, axis=1)
    out = np.empty(len(q), np.int64)
    cand_eps, cand_init = episodes[candidate_ids], initials[candidate_ids]
    for start in range(0, len(q), 64):
        stop = min(start + 64, len(q))
        d2 = qn[start:stop, None] + mn[None, :] - 2.0 * q[start:stop] @ m.T
        for j in range(stop - start):
            i = start + j
            d2[j, (cand_eps == query_eps[i]) | (cand_init == query_init[i])] = np.inf
            out[i] = int(np.argmin(d2[j]))
    return candidate_ids[out]


def _make_anchors(memory, qkey, state, ids_local, weights, candidate_ids, actions, bases,
                  *, block=128):
    """Return Ap, weighted raw R, and top-1 candidates' per-item decoded draft."""
    qn = _norm(qkey)
    n = len(qn)
    ap = np.empty((n, 10, 32), np.float32)
    raw = np.empty_like(ap)
    raw_items = np.asarray(actions[candidate_ids[ids_local]], np.float32)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        local = ids_local[lo:hi]
        selected = candidate_ids[local]
        b = np.asarray(bases[selected], np.float32).reshape(-1, bases.shape[-1])
        q = np.repeat(qn[lo:hi], TOP_K, axis=0)
        s = np.repeat(np.asarray(state[lo:hi], np.float32), TOP_K, axis=0)
        _, residual = memory.prior(q, s)
        decoded = np.asarray(memory.decode(b, residual, q, s), np.float32).reshape(hi-lo, TOP_K, 10, 32)
        ap[lo:hi] = np.einsum("nk,nkhd->nhd", weights[lo:hi], decoded)
        raw[lo:hi] = np.einsum("nk,nkhd->nhd", weights[lo:hi], raw_items[lo:hi])
    return ap, raw


def _assemble(anchor, q, state):
    n = len(q)
    return np.concatenate([np.asarray(anchor, np.float32).reshape(n, -1),
                           _norm(q), np.asarray(state, np.float32)], axis=-1).astype(np.float32)


def _build_model(kind, input_dim):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(),
                                      nn.Linear(256, 256), nn.GELU())
            out = 327 if kind == "geometry" else 320
            self.head = nn.Linear(256, out)
            if kind in ("residual", "geometry"):
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

        def forward(self, x, anchor, q01, q99):
            h = self.body(x)
            raw = self.head(h)
            a = anchor.reshape(len(anchor), 10, 32)
            if self.kind == "residual":
                return a + raw.reshape(-1, 10, 32)
            if self.kind == "direct":
                return raw.reshape(-1, 10, 32)
            # Identity-initialized physical xyz translation/scale/rotation plus
            # a bounded small normalized residual. LIBERO controls are delta pose.
            trans = 0.01 * torch.tanh(raw[:, 0:3])
            scale = torch.exp(0.30 * torch.tanh(raw[:, 3:4]))
            omega = 0.50 * torch.tanh(raw[:, 4:7])
            delta = 0.05 * torch.tanh(raw[:, 7:327]).reshape(-1, 10, 32)
            theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
            axis = omega / torch.clamp(theta, min=1e-7)
            v = a[..., :3] * (q99[:3] - q01[:3] + 1e-6)[None, None, :] * 0.5
            v = v + (q99[:3] + q01[:3])[None, None, :] * 0.5
            c = torch.cos(theta)[:, None, :]
            sn = torch.sin(theta)[:, None, :]
            ax = axis[:, None, :]
            rotated = (v * c + torch.linalg.cross(ax.expand_as(v), v, dim=-1) * sn
                       + ax * torch.sum(ax * v, dim=-1, keepdim=True) * (1.0 - c))
            moved = scale[:, None, :] * rotated + trans[:, None, :]
            xyz = (moved - (q99[:3] + q01[:3])[None, None, :] * 0.5) / (
                (q99[:3] - q01[:3] + 1e-6)[None, None, :] * 0.5)
            out = a.clone()
            out[..., :3] = xyz
            out = out + delta
            return out

    return Adapter()


def _train_one(name, mode, x_np, anchors_np, target_np, val_x, val_anchor, val_target,
               *, q01, q99, steps, batch, seed, out_path):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    model = _build_model(mode, x_np.shape[-1]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy(x_np)
    A = torch.from_numpy(anchors_np.reshape(len(anchors_np), -1))
    Y = torch.from_numpy(target_np)
    VX, VA, VY = map(torch.from_numpy, (val_x, val_anchor.reshape(len(val_anchor), -1), val_target))
    qlow, qhigh = torch.from_numpy(q01), torch.from_numpy(q99)
    rng = np.random.default_rng(seed)
    train_probe = np.random.default_rng(seed + 400).choice(len(X), min(512, len(X)), replace=False)
    log_at = {1, 50, 100, 200, 300, 400, 500, 600, 800, 1000}
    curve, best, best_step, best_state = [], float("inf"), 0, None
    for step in range(1, steps + 1):
        ids = torch.from_numpy(rng.integers(0, len(X), size=batch))
        pred = model(X[ids], A[ids], qlow, qhigh)
        mse = torch.mean((pred[..., :7] - Y[ids][..., :7]) ** 2)
        if mode == "residual":
            penalty = torch.mean((pred[..., :7] - A[ids].reshape(-1, 10, 32)[..., :7]) ** 2)
            loss = mse + 0.005 * penalty
        elif mode == "geometry":
            raw = model.head(model.body(X[ids]))
            delta = 0.05 * torch.tanh(raw[:, 7:327]).reshape(-1, 10, 32)[..., :7]
            loss = mse + 0.005 * torch.mean(delta ** 2)
        else:
            loss = mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in log_at:
            model.eval()
            with torch.no_grad():
                train_pred = model(X[train_probe], A[train_probe], qlow, qhigh)
                val_pred = model(VX, VA, qlow, qhigh)
                train_mse = float(torch.mean((train_pred[..., :7] - Y[train_probe][..., :7]) ** 2))
                val_mse = float(torch.mean((val_pred[..., :7] - VY[..., :7]) ** 2))
            curve.append({"step": step, "train_mse": train_mse, "validation_mse": val_mse,
                          "minibatch_loss": float(loss.detach())})
            if val_mse < best:
                best, best_step = val_mse, step
                best_state = copy.deepcopy(model.state_dict())
            print(name, curve[-1], flush=True)
            model.train()
    model.load_state_dict(best_state)
    torch.save({"name": name, "kind": mode, "state_dict": model.state_dict(),
                "input_dim": int(x_np.shape[-1]), "best_step": best_step,
                "best_validation_mse": best}, out_path)
    return model, {"kind": mode, "steps": steps, "batch_size": batch, "best_step": best_step,
                   "best_validation_mse": best, "curve": curve}


def _predict(model, anchor, q, state, q01, q99, *, batch=256):
    import torch
    X = _assemble(anchor, q, state)
    out = np.empty_like(anchor, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(X), batch):
            hi = min(lo + batch, len(X))
            xb = torch.from_numpy(X[lo:hi])
            ab = torch.from_numpy(np.asarray(anchor[lo:hi], np.float32).reshape(hi-lo, -1))
            pred = model(xb, ab, torch.from_numpy(q01), torch.from_numpy(q99))
            out[lo:hi] = pred.numpy()
    return out


def _summarize(pred, target, episodes, row_errors, cluster_bootstrap, base_mse, base_mm,
               skillless_mse, skillless_mm, *, reps=2000, seed=0):
    mse, mm = row_errors(pred, target)
    mse, mm = np.asarray(mse).reshape(-1), np.asarray(mm).reshape(-1)
    dm, dlo, dhi = cluster_bootstrap(mse - base_mse, episodes, reps=reps, seed=seed)
    dmm, mmlo, mmhi = cluster_bootstrap(mm - base_mm, episodes, reps=reps, seed=seed+1)
    ds, dslo, dshi = cluster_bootstrap(mse - skillless_mse, episodes, reps=reps, seed=seed+2)
    dss, dsslo, dsshi = cluster_bootstrap(mm - skillless_mm, episodes, reps=reps, seed=seed+3)
    return {"mse": float(mse.mean()), "endpoint_mm": float(mm.mean()),
            "delta_mse_vs_t0_1": dm, "delta_mse_ci95": [dlo, dhi],
            "delta_mm_vs_t0_1": dmm, "delta_mm_ci95": [mmlo, mmhi],
            "delta_mse_vs_no_skill": ds, "delta_mse_vs_no_skill_ci95": [dslo, dshi],
            "delta_mm_vs_no_skill": dss, "delta_mm_vs_no_skill_ci95": [dsslo, dsshi]}


def _denoise_condition(name, query, anchors, models, config, checkpoint, row_errors,
                       cluster_bootstrap, denoising_steps, *, seeds, batch_size, gpu_fraction):
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from smoke import load_model
    from train import observation_batch
    model, config = load_model(checkpoint)
    obs = observation_batch(query["obs"], config)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "t0"))
    n = len(query["actions"])
    rngs = [np.random.default_rng(20261029 + i).standard_normal((n, 10, 32)).astype(np.float32)
            for i in range(seeds)]
    out = {}
    base_mse = np.zeros((seeds, n), np.float32)
    base_mm = np.zeros((seeds, n), np.float32)
    for si in range(seeds):
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            ob = jax.tree.map(lambda x: x[lo:hi], obs)
            eps = jnp.asarray(rngs[si][lo:hi])
            pred = sample(jax.random.key(0), ob, num_steps=10, noise=eps, x_src=eps, t0=1.0)
            base_mse[si, lo:hi], base_mm[si, lo:hi] = row_errors(np.asarray(pred), query["actions"][lo:hi])
    pred_anchors = {}
    for key, model_adapter in models.items():
        draft = anchors[key]
        pred_anchors[key] = _predict(model_adapter, draft, query["q"], query["state"],
                                     query["q01"], query["q99"])
    # Oracle-input intervention: the Ap-trained residual F sees the nearest raw action.
    if "residual_ap" in models and "oracle_raw" in anchors:
        pred_anchors["residual_ap_oracle_input"] = _predict(
            models["residual_ap"], anchors["oracle_raw"], query["q"], query["state"],
            query["q01"], query["q99"])
    # Optional scene-use control: keep draft and state, permute only q.
    if "residual_ap" in models and n > 1:
        perm = np.random.default_rng(7301).permutation(n)
        pred_anchors["residual_ap_q_shuffled"] = _predict(
            models["residual_ap"], anchors["residual_ap"], query["q"][perm], query["state"],
            query["q01"], query["q99"])

    result = {"rows": n, "episodes": len(set(query["episodes"])), "baseline_t0_1": {},
              "anchor_only": {}, "denoising": {}}
    baseline_row_mse = base_mse.mean(axis=0)
    baseline_row_mm = base_mm.mean(axis=0)
    masks = {"all": np.ones(n, bool)}
    if "partition" in query:
        masks.update({"fit": query["partition"] == "fit", "select": query["partition"] == "select"})
    for part, mask in masks.items():
        result["baseline_t0_1"][part] = {
            "mse": float(baseline_row_mse[mask].mean()), "endpoint_mm": float(baseline_row_mm[mask].mean()),
            "rows": int(mask.sum()), "episodes": len(set(query["episodes"][mask]))}

    noise_keys = ["residual_ap", "residual_raw", "geometry_ap", "direct_ap", "no_skill",
                  "oracle_trained", "residual_ap_oracle_input", "residual_ap_q_shuffled"]
    denoise_rows = {key: {t: [np.zeros((seeds, n), np.float32), np.zeros((seeds, n), np.float32)]
                          for t in T0S if t != 1.0} for key in noise_keys}
    anchor_only_rows = {}
    for key, pred_anchor in pred_anchors.items():
        amse, amm = row_errors(pred_anchor, query["actions"])
        anchor_only_rows[key] = (np.asarray(amse), np.asarray(amm))
    for si in range(seeds):
        for key, pred_anchor in pred_anchors.items():
            for t0 in T0S:
                if t0 == 1.0:
                    continue
                nsteps = denoising_steps(t0)
                for lo in range(0, n, batch_size):
                    hi = min(lo + batch_size, n)
                    ob = jax.tree.map(lambda x: x[lo:hi], obs)
                    eps = jnp.asarray(rngs[si][lo:hi])
                    a = jnp.asarray(pred_anchor[lo:hi])
                    xsrc = (1.0 - t0) * a + t0 * eps
                    pred = sample(jax.random.key(0), ob, num_steps=nsteps, noise=eps,
                                  x_src=xsrc, t0=float(t0))
                    denoise_rows[key][t0][0][si, lo:hi], denoise_rows[key][t0][1][si, lo:hi] = row_errors(
                        np.asarray(pred), query["actions"][lo:hi])
    # Use the no-skill row at each t0 as the paired skill-free comparator.
    for key in pred_anchors:
        result["anchor_only"][key] = {}
        for part, mask in masks.items():
            amse, amm = anchor_only_rows[key]
            result["anchor_only"][key][part] = {
                "mse": float(amse[mask].mean()), "endpoint_mm": float(amm[mask].mean()),
                "rows": int(mask.sum())}
        result["denoising"][key] = {}
        for t0 in T0S:
            if t0 == 1.0:
                for part, mask in masks.items():
                    result["denoising"][key]["1.0"] = result["denoising"][key].get("1.0", {})
                    result["denoising"][key]["1.0"][part] = {
                        "mse": float(baseline_row_mse[mask].mean()),
                        "endpoint_mm": float(baseline_row_mm[mask].mean()),
                        "delta_mse_vs_t0_1": 0.0, "delta_mse_ci95": [0.0, 0.0],
                        "delta_mm_vs_t0_1": 0.0, "delta_mm_ci95": [0.0, 0.0],
                        "delta_mse_vs_no_skill": 0.0, "delta_mse_vs_no_skill_ci95": [0.0, 0.0],
                        "delta_mm_vs_no_skill": 0.0, "delta_mm_vs_no_skill_ci95": [0.0, 0.0]}
                continue
            mrow = denoise_rows[key][t0][0].mean(axis=0)
            mmrow = denoise_rows[key][t0][1].mean(axis=0)
            srow = denoise_rows["no_skill"][t0][0].mean(axis=0)
            ssrow = denoise_rows["no_skill"][t0][1].mean(axis=0)
            for part, mask in masks.items():
                dm, lo, hi = cluster_bootstrap(mrow[mask] - baseline_row_mse[mask],
                    query["episodes"][mask], reps=2000, seed=1000 + int(t0*100) + len(part))
                dmm, mmlo, mmhi = cluster_bootstrap(mmrow[mask] - baseline_row_mm[mask],
                    query["episodes"][mask], reps=2000, seed=2000 + int(t0*100) + len(part))
                ds, dslo, dshi = cluster_bootstrap(mrow[mask] - srow[mask],
                    query["episodes"][mask], reps=2000, seed=3000 + int(t0*100) + len(part))
                dss, dsslo, dsshi = cluster_bootstrap(mmrow[mask] - ssrow[mask],
                    query["episodes"][mask], reps=2000, seed=4000 + int(t0*100) + len(part))
                result["denoising"][key].setdefault(str(t0), {})[part] = {
                    "mse": float(mrow[mask].mean()), "endpoint_mm": float(mmrow[mask].mean()),
                    "delta_mse_vs_t0_1": dm, "delta_mse_ci95": [lo, hi],
                    "delta_mm_vs_t0_1": dmm, "delta_mm_ci95": [mmlo, mmhi],
                    "delta_mse_vs_no_skill": ds, "delta_mse_vs_no_skill_ci95": [dslo, dshi],
                    "delta_mm_vs_no_skill": dss, "delta_mm_vs_no_skill_ci95": [dsslo, dsshi],
                    "rows": int(mask.sum()), "episodes": len(set(query["episodes"][mask]))}
    result["noise"] = {"seeds": seeds, "seed_start": 20261029,
                       "same_noise_per_row_across_methods_and_t0": True}
    return result


def _obs_dict(rows, ids=None):
    result = {}
    n = len(rows["actions"]) if ids is None else len(ids)
    for name in ("image_base", "image_left_wrist", "image_right_wrist", "state",
                 "tokenized_prompt", "tokenized_prompt_mask", "skills", "skill_mask"):
        if name in rows:
            value = np.asarray(rows[name])
            result[name] = value if ids is None else value[ids]
    if "image_right_wrist" not in result:
        result["image_right_wrist"] = np.zeros_like(result["image_base"])
    if "skills" not in result:
        result["skills"] = np.zeros((n, 1), np.int32)
        result["skill_mask"] = np.ones((n, 1), bool)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--query-keys", default=str(Path(DEFAULT_HELDOUT).with_name("query-keys.npy")))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", default=DEFAULT_NORM)
    parser.add_argument("--output-root", default=DEFAULT_OUT)
    parser.add_argument("--output-dir", default=None,
                        help="resume an existing incomplete run with saved adapters")
    parser.add_argument("--train-rows", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gpu-fraction", type=float, default=0.16)
    args = parser.parse_args()
    if args.steps < 100 or args.train_rows < 512:
        raise ValueError("steps must be >=100 and train-rows >=512")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.gpu_fraction))
    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_dir() or (out / "results.json").exists():
            raise FileExistsError("resume requires an existing incomplete scene-adapter run")
        if not (out / "training_curves.json").is_file():
            raise FileNotFoundError("resume run is missing training_curves.json")
        resume_models = True
    else:
        run_id = time.strftime("scene-adapter-%Y%m%dT%H%M%SZ", time.gmtime())
        out = Path(args.output_root) / run_id
        out.mkdir(parents=True, exist_ok=False)
        resume_models = False
    started = time.perf_counter()
    import torch
    torch.set_num_threads(8)

    memory = load_memory(args.memory)
    held = _read_npz(args.heldout, ROW_FIELDS + ("partition", "suite", "skills", "skill_mask"))
    qkeys_held = np.load(args.query_keys).astype(np.float32)
    with np.load(args.rows, allow_pickle=True) as z:
        rows = {k: z[k] for k in z.files}
    n = len(rows["actions"])
    if len(memory.memory) != n or not np.all(np.asarray(rows["suite"]).astype(str) == "libero90"):
        raise ValueError("LIBERO-90 row cache and M6 memory do not align")
    for i in range(n):
        item = memory.memory[i]
        if str(item["meta"]["episode_id"]) != str(rows["episode_id"][i]) or str(item["meta"]["init_state_id"]) != str(rows["init_state_id"][i]):
            raise ValueError(f"memory/cache leakage alignment mismatch at row {i}")
    keys = np.stack([np.asarray(x["key"], np.float32).reshape(-1) for x in memory.memory])
    bases = np.stack([np.asarray(x["base"], np.float32).reshape(-1) for x in memory.memory])
    actions = np.stack([np.asarray(x["meta"]["raw_action"], np.float32) for x in memory.memory])
    eps = np.asarray(rows["episode_id"]).astype(str)
    init = np.asarray(rows["init_state_id"]).astype(str)
    sources = np.asarray(rows["source_file"]).astype(str)
    state = np.asarray(rows["state"], np.float32)
    task_query_ids, heldout_tasks, task_plan = _sample_task_queries(
        sources, eps, 20261030, episodes_per_task=2, rows_per_episode=8)
    task_mem_ids = np.flatnonzero(~np.isin(sources, heldout_tasks))
    train_rng = np.random.default_rng(20261030)
    train_ids = np.sort(train_rng.choice(task_mem_ids, min(args.train_rows, len(task_mem_ids)), replace=False))
    if set(sources[task_mem_ids]) & set(heldout_tasks):
        raise AssertionError("task-heldout task leaked into scene-adapter memory")
    if set(np.asarray(held["episode_id"]).astype(str)) & set(eps[task_mem_ids]):
        raise AssertionError("known validation episode leaked into training memory")

    # Query splits: 256 known-task heldout rows (fit/select), and the same
    # 10-task/160-row whole-task holdout used in key-projection.
    known_query = {"q": qkeys_held, "state": np.asarray(held["state"], np.float32),
        "actions": np.asarray(held["actions"], np.float32),
        "episodes": np.asarray(held["episode_id"]).astype(str),
        "initials": np.asarray(held["init_state_id"]).astype(str),
        "partition": np.asarray(held["partition"]).astype(str),
        "obs": _obs_dict(held)}
    task_query = {"q": keys[task_query_ids], "state": state[task_query_ids],
        "actions": actions[task_query_ids], "episodes": eps[task_query_ids],
        "initials": init[task_query_ids], "obs": _obs_dict(rows, task_query_ids)}

    # Search candidates are the same 79-task memory for F train, known query,
    # and whole-task test. Training queries exclude their own episode/init.
    query_train = {"q": keys[train_ids], "state": state[train_ids], "actions": actions[train_ids],
        "episodes": eps[train_ids], "initials": init[train_ids]}
    q01 = np.asarray(json.load(open(args.norm_stats))["norm_stats"]["actions"]["q01"], np.float32)[:7]
    q99 = np.asarray(json.load(open(args.norm_stats))["norm_stats"]["actions"]["q99"], np.float32)[:7]
    if q01.shape != (7,) or q99.shape != (7,):
        raise ValueError("expected 7D LIBERO action quantile statistics")

    def prepare(query, label):
        ids, sims, w, keymean = _retrieval(query["q"], query["episodes"], query["initials"],
            task_mem_ids, keys, eps, init)
        ap, raw = _make_anchors(memory, query["q"], query["state"], ids, w,
            task_mem_ids, actions, bases)
        oracle_local = _nearest_action(query["actions"], query["episodes"], query["initials"],
            task_mem_ids, actions, eps, init)
        oracle_raw = actions[oracle_local]
        print(f"prepared {label}: rows={len(query['actions'])} Ap_mse="
              f"{float(np.mean((ap[..., :7]-query['actions'][..., :7])**2)):.5f} "
              f"R_mse={float(np.mean((raw[..., :7]-query['actions'][..., :7])**2)):.5f}", flush=True)
        return {**query, "Ap": ap, "R": raw, "oracle_raw": oracle_raw,
                "retrieved_ids": task_mem_ids[ids], "similarities": sims, "weights": w,
                "q01": q01, "q99": q99}

    train_data = prepare(query_train, "train")
    known_data = prepare(known_query, "known-validation")
    task_data = prepare(task_query, "task-test")
    # Serialize retrieval-derived data before fitting so all model variants use
    # exactly the same real retrieval distribution and can be audited.
    if not (out / "prepared_retrieval.npz").exists():
        np.savez_compressed(out / "prepared_retrieval.npz", train_ids=train_ids,
            train_Ap=train_data["Ap"], train_R=train_data["R"], train_oracle=train_data["oracle_raw"],
            train_targets=train_data["actions"], known_Ap=known_data["Ap"], known_R=known_data["R"],
            known_oracle=known_data["oracle_raw"], known_targets=known_data["actions"],
            task_Ap=task_data["Ap"], task_R=task_data["R"], task_oracle=task_data["oracle_raw"],
            task_targets=task_data["actions"], heldout_tasks=np.asarray(heldout_tasks))

    # Known fit half is the only checkpoint-selection/validation split.
    fit_mask = known_data["partition"] == "fit"
    val = {k: np.asarray(known_data[k])[fit_mask] for k in
           ("q", "state", "actions", "episodes", "Ap", "R", "oracle_raw")}
    models, training = {}, {}
    specs = [
        ("residual_ap", "residual", "Ap", "Ap", 1101),
        ("residual_raw", "residual", "R", "R", 1102),
        ("geometry_ap", "geometry", "Ap", "Ap", 1103),
        ("direct_ap", "direct", "Ap", "Ap", 1104),
        ("no_skill", "direct", "zero", "Ap", 1105),
        ("oracle_trained", "residual", "oracle_raw", "oracle_raw", 1106),
    ]
    if resume_models:
        training = json.loads((out / "training_curves.json").read_text())
    for name, mode, anchor_name, input_anchor, seed in specs:
        anchor = np.zeros_like(train_data["Ap"]) if anchor_name == "zero" else train_data[anchor_name]
        val_anchor = np.zeros_like(val["Ap"]) if anchor_name == "zero" else val[anchor_name]
        x = _assemble(anchor, train_data["q"], train_data["state"])
        vx = _assemble(val_anchor, val["q"], val["state"])
        if resume_models:
            import torch
            saved_path = out / f"{name}.pt"
            if not saved_path.is_file():
                raise FileNotFoundError(saved_path)
            adapter = _build_model(mode, x.shape[-1]).cpu()
            saved = torch.load(saved_path, map_location="cpu", weights_only=False)
            adapter.load_state_dict(saved["state_dict"])
            log = training[name]
        else:
            adapter, log = _train_one(name, mode, x, anchor, train_data["actions"],
                vx, val_anchor, val["actions"], q01=q01, q99=q99,
                steps=args.steps, batch=args.batch_size, seed=seed, out_path=out / f"{name}.pt")
        models[name] = adapter
        training[name] = log
    if not resume_models:
        (out / "training_curves.json").write_text(json.dumps(training, indent=2) + "\n")

    # Map model names to drafts that define their deployment condition.
    model_sets = {
        "heldout_known_tasks": {
            "residual_ap": models["residual_ap"], "residual_raw": models["residual_raw"],
            "geometry_ap": models["geometry_ap"], "direct_ap": models["direct_ap"],
            "no_skill": models["no_skill"], "oracle_trained": models["oracle_trained"]},
        "heldout_10_tasks": {
            "residual_ap": models["residual_ap"], "residual_raw": models["residual_raw"],
            "geometry_ap": models["geometry_ap"], "direct_ap": models["direct_ap"],
            "no_skill": models["no_skill"], "oracle_trained": models["oracle_trained"]},
    }
    # The residual_ap model is also tested on an oracle draft at evaluation.
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"frozen-head evaluation needs GPU; backend={jax.default_backend()}")
    results = {}
    for name, query, anch in (("heldout_known_tasks", known_data,
            {"residual_ap": known_data["Ap"], "residual_raw": known_data["R"],
             "geometry_ap": known_data["Ap"], "direct_ap": known_data["Ap"],
             "no_skill": np.zeros_like(known_data["Ap"]),
             "oracle_trained": known_data["oracle_raw"], "oracle_raw": known_data["oracle_raw"]}),
        ("heldout_10_tasks", task_data,
            {"residual_ap": task_data["Ap"], "residual_raw": task_data["R"],
             "geometry_ap": task_data["Ap"], "direct_ap": task_data["Ap"],
             "no_skill": np.zeros_like(task_data["Ap"]),
             "oracle_trained": task_data["oracle_raw"], "oracle_raw": task_data["oracle_raw"]})):
        eval_models = {k: model_sets[name][k] for k in model_sets[name]}
        results[name] = _denoise_condition(name, query, anch, eval_models, None, args.checkpoint,
            _offline_helpers()[1], _offline_helpers()[2], _offline_helpers()[0], seeds=args.seeds,
            batch_size=args.eval_batch_size, gpu_fraction=args.gpu_fraction)

    metadata = {
        "memory": args.memory, "rows": args.rows, "heldout": args.heldout,
        "checkpoint": args.checkpoint, "normalization_stats": args.norm_stats,
        "memory_entries": len(memory.memory), "source_task_files_available": len(set(sources)),
        "heldout_task_files": heldout_tasks, "candidate_memory_task_files": len(set(sources[task_mem_ids])),
        "missing_source_task_note": "The cached LIBERO-90 row archive contains 89 task files; selected 10 heldout leaves 79 candidate-memory tasks.",
        "train_query_rows": len(train_ids), "candidate_memory_rows": len(task_mem_ids),
        "train_queries_sampling": "uniform without replacement from candidate-memory rows, seed 20261030; search excludes same episode/init",
        "init_state_exclusion_note": "The exported cache uses conservative task-file init_state_id grouping; same-init exclusion therefore excludes other rows from the same source task even in the known-task condition.",
        "retrieval": "centered full-token-mean key, cosine top-5, softmax(similarity/0.05), exclude same episode and init_state",
        "anchor_ap": "weighted average of per-entry Dec(b_i, prior(q,s), q, s)",
        "anchor_raw": "softmax-weighted mean of retrieved normalized 10x32 action chunks",
        "oracle_action": "nearest eligible normalized target action chunk in active memory",
        "all_rows_normalized_action_shape": [10, 32], "metric_dims": "first 7 active action dims",
        "adapter_training_steps": args.steps, "batch_size": args.batch_size,
        "adapter_architectures": {
            "residual": "Ahat=a_ret+DeltaF(a_ret,q,s), final layer zero initialized, 0.005*mean(Delta^2) penalty",
            "geometry": "physical xyz affine motion warp (translation, isotropic scale, axis-angle rotation) plus bounded 0.05 normalized residual; identity initialization",
            "direct": "F(a_ret,q,s) direct normalized chunk regression",
            "no_skill": "same direct architecture/updates, zero draft input; uses q,s only",
            "oracle_trained": "residual F trained on nearest eligible true-action draft"},
        "evaluation": {"heldout_known_tasks": {"rows": 256, "episodes": len(set(known_data["episodes"])),
                         "fit_rows": int(fit_mask.sum()), "select_rows": int((~fit_mask).sum())},
                       "heldout_10_tasks": {"rows": len(task_data["actions"]), "episodes": len(set(task_data["episodes"])),
                         "tasks": heldout_tasks, "sampling": "2 episodes per task, up to 8 evenly spaced rows"}},
        "frozen_policy_and_memory": True, "gpu_preallocate": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result = {"metadata": metadata, "training_curves": training, "conditions": results}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps({"output": str(out), "elapsed_seconds": metadata["elapsed_seconds"],
                      "conditions": {k: {"rows": v["rows"], "episodes": v["episodes"]} for k,v in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
