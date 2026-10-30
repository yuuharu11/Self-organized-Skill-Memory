#!/usr/bin/env python3
"""Offline paired check of the centered-key skill-memory anchor (v2).

Training-free: the frozen public SkillNet head integrates from
x_t0=(1-t0)a+t0*eps to 0 in n=max(2,round(10*t0)) Euler steps, exactly the
S2 server path (``Pi0AnchorFlow.sample_actions``; t0=1 is the ordinary policy).

Anchors (same centered-key top-5 retrieval and softmax(sim/tau) weights):
  A  : Dec(sum_i w_i b_i, prior residual, original key, state)
  A' : sum_i w_i Dec(b_i, prior residual, original key, state)
  R  : sum_i w_i raw_action_i
All anchor types and t0 share the same per-row noise eps for each seed.
No weights are updated; no closed-loop evaluation is started.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
for _p in (str(HERE), str(EXAMPLES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from key_check import DEFAULT_CHECKPOINT, DEFAULT_HELDOUT, DEFAULT_MEMORY, ROW_FIELDS, _read_npz
from offline_retrieval import center_normalize, topk_centered, softmax_weights

DEFAULT_QUERY_KEYS = str(Path(DEFAULT_HELDOUT).with_name("query-keys.npy"))
ENDPOINT_SCALES_M = np.asarray([0.0111, 0.0122, 0.0110], np.float32)  # key_check_extension
TAUS = (0.001, 0.01, 0.03, 0.05, 0.1)
T0S = (0.2, 0.4, 0.6, 0.8)
ANCHORS = ("A", "Ap", "R")


def denoising_steps(t0):
    return max(2, round(10 * float(t0)))


def row_errors(pred, target):
    """Per-row normalized MSE (7 active dims) and endpoint mm (key_check_extension)."""
    err = np.asarray(pred, np.float32)[..., :7] - np.asarray(target, np.float32)[..., :7]
    mse = np.mean(np.square(err), axis=(-2, -1))
    mm = np.linalg.norm(np.sum(err[..., :3] * ENDPOINT_SCALES_M, axis=-2), axis=-1) * 1000.0
    return mse, mm


def cluster_bootstrap(values, clusters, *, reps=4000, seed=0):
    """Mean and 95% percentile CI, resampling episodes with replacement."""
    v = np.asarray(values, np.float64)
    c = np.asarray(clusters).astype(str)
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--query-keys", default=DEFAULT_QUERY_KEYS)
    ap.add_argument("--memory", default=DEFAULT_MEMORY)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--subset-frac", type=float, default=0.3)
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from memory import load_memory

    # ---------------- data and leakage checks ----------------
    held = _read_npz(args.heldout, ROW_FIELDS + ("partition", "skills", "skill_mask"))
    qkey_raw = np.load(args.query_keys).astype(np.float32)
    n = len(held["actions"])
    if qkey_raw.shape[0] != n:
        raise ValueError("query keys and held-out rows differ")
    memory = load_memory(args.memory)
    M = len(memory.memory)
    mem_keys = np.stack([np.asarray(r["key"], np.float32).reshape(-1) for r in memory.memory])
    mem_base = np.stack([np.asarray(r["base"], np.float32).reshape(-1) for r in memory.memory])
    mem_raw = np.stack([np.asarray(r["meta"]["raw_action"], np.float32) for r in memory.memory])
    mem_ep = np.asarray([str(r["meta"]["episode_id"]) for r in memory.memory])
    mem_init = np.asarray([str(r["meta"]["init_state_id"]) for r in memory.memory])
    mem_unit = np.asarray([-1 if r.get("unit_id") is None else int(r["unit_id"]) for r in memory.memory])
    q_ep = np.asarray(held["episode_id"]).astype(str)
    q_init = np.asarray(held["init_state_id"]).astype(str)
    ep_overlap = len(set(q_ep) & set(mem_ep))
    init_overlap = len(set(q_init) & set(mem_init))
    if ep_overlap:
        raise ValueError(f"held-out episode leakage: {ep_overlap}")
    actions = np.asarray(held["actions"], np.float32)
    state = np.asarray(held["state"], np.float32)
    part = np.asarray(held["partition"]).astype(str)
    print(f"memory {M} entries; heldout {n} rows / {len(set(q_ep))} eps; init overlap {init_overlap}", flush=True)

    # ---------------- retrieval: centered full-token-mean key ----------------
    # Stored memory keys equal the recomputed full-token mean (key-check cosine=1.0);
    # center with the mean of all memory keys (L2-normalized first), as key_check did.
    mem_n = mem_keys / np.maximum(np.linalg.norm(mem_keys, axis=1, keepdims=True), 1e-8)
    q_n = qkey_raw / np.maximum(np.linalg.norm(qkey_raw, axis=1, keepdims=True), 1e-8)
    mem_c, key_mean = center_normalize(mem_n)
    q_c, _ = center_normalize(q_n, key_mean)
    ids, sims = topk_centered(q_c, mem_c, mem_ep, mem_init, q_ep, q_init, k=5, block_size=32)
    for i in range(n):
        if (mem_ep[ids[i]] == q_ep[i]).any() or (mem_init[ids[i]] == q_init[i]).any():
            raise AssertionError("exclusion violated")
    enc_base, _ = memory.encode(actions, state)
    oracle_unit = np.asarray([-1 if u is None else int(u) for u, _ in (memory._nearest(b) for b in enc_base)])
    hit1 = float(np.mean(mem_unit[ids[:, 0]] == oracle_unit))
    hit5 = float(np.mean([(oracle_unit[i] in mem_unit[ids[i]]) for i in range(n)]))
    print(f"centered retrieval unit hit@1={hit1:.3f} hit@5={hit5:.3f}", flush=True)

    # ---------------- anchors ----------------
    _, prior_res = memory.prior(q_n, state)
    dec = lambda b: np.asarray(memory.decode(b, prior_res, q_n, state), np.float32).reshape(n, 10, 32)
    dec_each = np.stack([dec(mem_base[ids[:, k]]) for k in range(5)], 1)  # [n,5,10,32]
    raw_each = mem_raw[ids]                                              # [n,5,10,32]
    anchors = np.zeros((len(TAUS), len(ANCHORS), n, 10, 32), np.float32)
    weights_all, entropy = {}, {}
    for ti, tau in enumerate(TAUS):
        w = softmax_weights(sims, tau)
        weights_all[tau] = w
        h = -np.sum(w * np.log(np.maximum(w, 1e-12)), 1) / np.log(5)
        entropy[str(tau)] = {"norm_entropy_q10_q50_q90": [float(x) for x in np.quantile(h, [.1, .5, .9])],
                             "norm_entropy_mean": float(h.mean()), "top1_weight_mean": float(w[:, 0].mean()),
                             "effective_k_mean": float(np.mean(1.0 / np.sum(w ** 2, 1)))}
        anchors[ti, 0] = dec(np.einsum("nk,nkd->nd", w, mem_base[ids]))
        anchors[ti, 1] = np.einsum("nk,nkhd->nhd", w, dec_each)
        anchors[ti, 2] = np.einsum("nk,nkhd->nhd", w, raw_each)
    anchor_only = {ANCHORS[a]: {str(t): dict(zip(("mse", "mm"), [float(x.mean()) for x in row_errors(anchors[ti, a], actions)]))
                                for ti, t in enumerate(TAUS)} for a in range(len(ANCHORS))}
    zero_mse, zero_mm = row_errors(np.zeros_like(actions), actions)
    print("anchor-only:", json.dumps(anchor_only), flush=True)

    # ---------------- frozen head, paired noise ----------------
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from smoke import load_model
    from train import observation_batch
    if jax.default_backend() != "gpu":
        raise RuntimeError("expected GPU")
    model, config = load_model(args.checkpoint)
    rows = {k: held[k] for k in ("image_base", "image_left_wrist", "state", "tokenized_prompt",
                                 "tokenized_prompt_mask", "skills", "skill_mask")}
    obs_all = observation_batch(rows, config)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "t0"))
    encode = nnx_utils.module_jit(model.encode_prefix_key)
    chk = np.asarray(encode(jax.tree.map(lambda v: v[:16], obs_all)), np.float32)
    key_cos = float(np.min(np.sum(chk * q_n[:16], 1) / np.linalg.norm(chk, axis=1)))
    print(f"saved query key vs recomputed encode_prefix_key: min cos {key_cos:.5f}", flush=True)

    # configs: index 0 = t0=1 baseline; then (tau, anchor, t0)
    cfgs = [(None, None, 1.0)] + [(ti, ai, t0) for ti in range(len(TAUS)) for ai in range(len(ANCHORS)) for t0 in T0S]
    S, C, B = args.seeds, len(cfgs), args.batch_size
    mse = np.zeros((S, C, n), np.float32)
    mm = np.zeros((S, C, n), np.float32)
    noises = np.stack([np.random.default_rng(20261029 + s).standard_normal((n, 10, 32)).astype(np.float32)
                       for s in range(S)])
    t_gpu = time.perf_counter()
    for s in range(S):
        for lo in range(0, n, B):
            hi = min(lo + B, n)
            obs = jax.tree.map(lambda v: v[lo:hi], obs_all)
            eps = jnp.asarray(noises[s, lo:hi])
            for ci, (ti, ai, t0) in enumerate(cfgs):
                if t0 == 1.0:  # exact S2 'anchor_disabled_t0_1' call
                    pred = sample(jax.random.key(0), obs, num_steps=10, noise=eps, x_src=eps, t0=1.0)
                else:
                    a = jnp.asarray(anchors[ti, ai, lo:hi])
                    x0 = (1.0 - t0) * a + t0 * eps
                    pred = sample(jax.random.key(0), obs, num_steps=denoising_steps(t0), x_src=x0, t0=float(t0))
                e1, e2 = row_errors(np.asarray(pred), actions[lo:hi])
                mse[s, ci, lo:hi], mm[s, ci, lo:hi] = e1, e2
        print(f"seed {s} done; gpu {time.perf_counter()-t_gpu:.0f}s", flush=True)
    gpu_seconds = time.perf_counter() - t_gpu
    np.savez_compressed(out / "paired-errors.npz", mse=mse, mm=mm, cfg_tau=np.asarray([-1 if c[0] is None else TAUS[c[0]] for c in cfgs]),
                        cfg_anchor=np.asarray(["base" if c[1] is None else ANCHORS[c[1]] for c in cfgs]),
                        cfg_t0=np.asarray([c[2] for c in cfgs]), candidate_ids=ids, similarities=sims,
                        episode_id=q_ep, partition=part, anchors=anchors, oracle_unit=oracle_unit)

    # ---------------- analysis ----------------
    base_mse = mse[:, 0].mean(0)
    k = int(round(args.subset_frac * n))
    worst = np.zeros(n, bool); worst[np.argsort(-base_mse)[:k]] = True
    simtop = np.zeros(n, bool); simtop[np.argsort(-sims[:, 0])[:k]] = True
    subsets = {"all": np.ones(n, bool), "worst30_base": worst, "top30_sim": simtop}
    splits = {"all": np.ones(n, bool), "fit": part == "fit", "select": part == "select"}
    cfg_index = {(c[0], c[1], c[2]): i for i, c in enumerate(cfgs)}

    def cell(ci, mask, seed=0):
        d_mse = (mse[:, ci] - mse[:, 0]).mean(0)[mask]
        d_mm = (mm[:, ci] - mm[:, 0]).mean(0)[mask]
        m, lo_, hi_ = cluster_bootstrap(d_mse, q_ep[mask], seed=seed)
        mmm, mlo, mhi = cluster_bootstrap(d_mm, q_ep[mask], seed=seed + 1)
        return {"n_rows": int(mask.sum()), "n_episodes": int(len(set(q_ep[mask]))),
                "mse": float(mse[:, ci].mean(0)[mask].mean()), "mm": float(mm[:, ci].mean(0)[mask].mean()),
                "base_mse": float(base_mse[mask].mean()), "base_mm": float(mm[:, 0].mean(0)[mask].mean()),
                "d_mse": m, "d_mse_ci95": [lo_, hi_], "d_mm": mmm, "d_mm_ci95": [mlo, mhi]}

    def anchor_only_cell(ti, ai, mask):
        e1, e2 = row_errors(anchors[ti, ai], actions)
        d1 = e1 - base_mse
        d2 = e2 - mm[:, 0].mean(0)
        m, lo_, hi_ = cluster_bootstrap(d1[mask], q_ep[mask]); mmm, mlo, mhi = cluster_bootstrap(d2[mask], q_ep[mask])
        return {"n_rows": int(mask.sum()), "mse": float(e1[mask].mean()), "mm": float(e2[mask].mean()),
                "d_mse": m, "d_mse_ci95": [lo_, hi_], "d_mm": mmm, "d_mm_ci95": [mlo, mhi]}

    grid = {}
    for sp, spm in splits.items():
        for sb, sbm in subsets.items():
            mask = spm & sbm
            for ti, tau in enumerate(TAUS):
                for ai, an in enumerate(ANCHORS):
                    key = f"{sp}|{sb}|{an}|tau={tau}"
                    grid[key] = {f"t0={t0}": cell(cfg_index[(ti, ai, t0)], mask) for t0 in T0S}
                    grid[key]["t0=0(anchor_only)"] = anchor_only_cell(ti, ai, mask)
    # tau selection on the fit half (all fit rows): minimal mean MSE over t0<1
    chosen = {}
    for ai, an in enumerate(ANCHORS):
        score = {tau: min(float(mse[:, cfg_index[(ti, ai, t0)]].mean(0)[splits["fit"]].mean()) for t0 in T0S)
                 for ti, tau in enumerate(TAUS)}
        chosen[an] = {"tau": min(score, key=score.get), "fit_best_t0_mse_by_tau": {str(k_): v for k_, v in score.items()}}

    result = {
        "memory": str(Path(args.memory).resolve()), "memory_entries_used": M,
        "memory_key_note": "all stored keys used (stored key == recomputed full-token mean, cosine 1.0 in key-check)",
        "heldout": str(Path(args.heldout).resolve()), "rows": n, "episodes": int(len(set(q_ep))),
        "fit_rows": int(splits["fit"].sum()), "select_rows": int(splits["select"].sum()),
        "heldout_episode_overlap": ep_overlap, "heldout_init_state_overlap": init_overlap,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "saved_vs_recomputed_query_key_min_cos_16rows": key_cos,
        "retrieval": "L2-normalize, subtract mean of all memory keys, L2-normalize; cosine top-5; exclude same episode_id OR init_state_id",
        "retrieval_unit_hit1": hit1, "retrieval_unit_hit5": hit5,
        "weights": "softmax(cosine/tau) over top-5", "weight_entropy": entropy,
        "anchors": {"A": "Dec(sum w b, prior residual, uncentered key, state)", "Ap": "sum w Dec(b_i,...)", "R": "sum w raw_action_i"},
        "anchor_only_all_rows": anchor_only,
        "zero_action_all_rows": {"mse": float(zero_mse.mean()), "mm": float(zero_mm.mean())},
        "sampler": "Pi0AnchorFlow.sample_actions (S2 server path); t0=1 -> parent sampler with noise=eps; t0<1 -> Euler from x_t0 with n=max(2,round(10 t0)); skills/skill_mask from held-out annotation",
        "seeds": S, "noise": "per-row eps shared across all anchors/tau/t0 within a seed",
        "error": "MSE over 7 active channels x 10 steps (normalized); endpoint mm = ||sum_t err[:3]*[0.0111,0.0122,0.0110]||*1000",
        "subsets": {"worst30_base": "top 30% rows (all 256) by t0=1 MSE averaged over seeds", "top30_sim": "top 30% rows by top-1 centered cosine"},
        "ci": "episode-cluster bootstrap 4000 reps, 95% percentile; Delta = paired (same eps) mean over seeds of err(t0)-err(t0=1)",
        "tau_selection": "per anchor type, tau minimising min_{t0<1} mean MSE on fit-partition rows; report on select partition",
        "chosen_tau": chosen,
        "base_t0_1": {sp: {"mse": float(base_mse[m].mean()), "mm": float(mm[:, 0].mean(0)[m].mean())} for sp, m in splits.items()},
        "seed_to_seed_base_mse": [float(mse[s, 0].mean()) for s in range(S)],
        "grid": grid, "gpu_seconds": gpu_seconds, "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    # ---------------- report ----------------
    def fmt(c, unit="mse"):
        if unit == "mse":
            return f"{c['mse']:.4f} ({c['d_mse']:+.4f} [{c['d_mse_ci95'][0]:+.4f},{c['d_mse_ci95'][1]:+.4f}])"
        return f"{c['mm']:.1f} ({c['d_mm']:+.1f} [{c['d_mm_ci95'][0]:+.1f},{c['d_mm_ci95'][1]:+.1f}])"
    L = ["# Anchor v2 offline paired check", "",
         f"Memory {M:,} entries (all, centered keys); held-out {n} rows / {result['episodes']} eps (LIBERO-90 known tasks, episodes excluded from memory). "
         f"Seeds {S}. Retrieval unit hit@1 {hit1:.3f}, hit@5 {hit5:.3f}.", "",
         f"t0=1 (normal policy): " + ", ".join(f"{sp} MSE {v['mse']:.4f} / {v['mm']:.1f} mm" for sp, v in result["base_t0_1"].items()), "",
         "## Weight entropy (normalized by log 5)", "", "| tau | mean | q10/q50/q90 | top1 w | eff k |", "|---|---:|---|---:|---:|"]
    for t, e in entropy.items():
        L.append(f"| {t} | {e['norm_entropy_mean']:.3f} | {'/'.join(f'{x:.2f}' for x in e['norm_entropy_q10_q50_q90'])} | {e['top1_weight_mean']:.3f} | {e['effective_k_mean']:.2f} |")
    L += ["", "## Chosen tau (fit half)", ""] + [f"- {a}: tau={v['tau']}" for a, v in chosen.items()]
    for unit in ("mse", "mm"):
        L += ["", f"## Select half, chosen tau — {'normalized MSE' if unit=='mse' else 'endpoint mm'} (Δ vs t0=1 [95% CI])", "",
              "| anchor | t0 | all | worst30_base | top30_sim |", "|---|---|---|---|---|"]
        for an in ANCHORS:
            tau = chosen[an]["tau"]
            for t in [f"t0={x}" for x in T0S] + ["t0=0(anchor_only)"]:
                L.append(f"| {an} τ={tau} | {t} | " + " | ".join(fmt(grid[f'select|{sb}|{an}|tau={tau}'][t], unit) for sb in subsets) + " |")
    L += ["", f"Full grid (all/fit/select × subsets × anchor × tau × t0) in results.json. Wall clock {result['elapsed_seconds']:.0f}s (GPU {gpu_seconds:.0f}s).", ""]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
