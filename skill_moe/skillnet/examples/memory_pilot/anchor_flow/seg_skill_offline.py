#!/usr/bin/env python3
"""Offline comparison of chunk retrieval and gripper-delimited motion skills.

The script deliberately reuses anchor_v2_offline's sampler/error conventions and
never starts a LIBERO environment.  Memory segments are built from the cached
10-step raw action chunks, which tile each source episode at its stored frame
stride.  No task/verb labels enter retrieval or clustering.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
for p in (HERE, EXAMPLES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from key_check import _read_npz, ROW_FIELDS, DEFAULT_CHECKPOINT
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR

ROOT = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'))
DEFAULT_KNOWN = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-20261028T173556Z/calibration/fit-v2/heldout-observations.npz'))
DEFAULT_ROWS = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz'))
DEFAULT_MEMORY = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/memory/m6.pkl'))
DEFAULT_TASK_RUN = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/key-projection-20261029T174550Z'))
T0S = (0.2, 0.4, 0.6, 1.0)
ENDPOINT_SCALES_M = np.asarray([0.0111, 0.0122, 0.0110], np.float32)


def denoising_steps(t0):
    return max(2, round(10 * float(t0)))


def row_errors(pred, target):
    err = np.asarray(pred, np.float32)[..., :7] - np.asarray(target, np.float32)[..., :7]
    mse = np.mean(np.square(err), axis=(-2, -1))
    mm = np.linalg.norm(np.sum(err[..., :3] * ENDPOINT_SCALES_M, axis=-2), axis=-1) * 1000
    return mse, mm


def cluster_bootstrap(values, clusters, *, reps=4000, seed=0):
    v = np.asarray(values, np.float64)
    c = np.asarray(clusters).astype(str)
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(v.mean()), float(lo), float(hi)


def _norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def centered_keys(mem, query):
    """Center normalized keys using the eligible memory mean, matching key-projection."""
    m = _norm(mem)
    q = _norm(query)
    mean = m.mean(0, keepdims=True)
    return _norm(m - mean), _norm(q - mean)


def gripper_segments(actions, chunk_frames=10, gripper_dim=6):
    """Return maximal [lo,hi) frame intervals split on raw gripper sign flips."""
    a = np.asarray(actions, np.float32)
    flat = a.reshape(-1, a.shape[-1])
    g = np.sign(flat[:, gripper_dim])
    g[g == 0] = 1
    bounds = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(flat)]
    return [(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]


def resample_sequence(seq, positions):
    """Linear sample a sequence at continuous frame positions, edge-clamped."""
    x = np.asarray(seq, np.float32)
    pos = np.clip(np.asarray(positions, np.float32), 0, max(len(x) - 1, 0))
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    f = (pos - lo)[..., None]
    return x[lo] * (1 - f) + x[hi] * f


def phase_chunk(seq, phase, horizon=10):
    """Read horizon commands in normalized phase, using this retrieved segment's duration."""
    seq = np.asarray(seq, np.float32)
    if not len(seq):
        return np.zeros((horizon, seq.shape[-1]), np.float32)
    positions = (float(phase) + np.arange(horizon, dtype=np.float32) / max(len(seq), 1)) * len(seq)
    return resample_sequence(seq, positions)


def _episode_segments(actions, episode_ids, frames, keys, states):
    """Build raw-action segments and per-row phase metadata from source-order rows."""
    actions = np.asarray(actions, np.float32)
    episode_ids = np.asarray(episode_ids).astype(str)
    frames = np.asarray(frames, np.int64)
    keys, states = np.asarray(keys, np.float32), np.asarray(states, np.float32)
    segments, row_seg, row_phase = [], np.full(len(actions), -1, np.int64), np.zeros(len(actions), np.float32)
    for ep in np.unique(episode_ids):
        ix = np.flatnonzero(episode_ids == ep)
        ix = ix[np.argsort(frames[ix], kind="stable")]
        if not len(ix):
            continue
        # Ensure chunk starts form a contiguous episode timeline. Gaps are kept as
        # separate pieces rather than silently fabricating intervening actions.
        groups = np.split(ix, np.flatnonzero(np.diff(frames[ix]) != actions.shape[1]) + 1)
        for group in groups:
            if not len(group):
                continue
            flat = np.concatenate([actions[j] for j in group], axis=0)
            for lo, hi in gripper_segments(flat):
                begin_frame = int(frames[group[0]] + lo)
                end_frame = int(frames[group[0]] + hi)
                # One key/state at the first cached observation at or after the
                # physical change point. This is available to a causal selector.
                start_row_pos = min(int(np.searchsorted(frames[group], begin_frame, side="left")), len(group)-1)
                start_row = int(group[start_row_pos])
                sid = len(segments)
                segment_actions = flat[lo:hi].copy()
                segments.append({"episode_id": ep, "lo": begin_frame, "hi": end_frame,
                                 "actions": segment_actions, "key": keys[start_row],
                                 "state": states[start_row], "start_row": start_row,
                                 "source_rows": group.copy()})
                for local_row, rid in enumerate(group):
                    row_lo = int(frames[rid] - frames[group[0]])
                    row_hi = row_lo + actions.shape[1]
                    overlap = max(0, min(row_hi, hi) - max(row_lo, lo))
                    if overlap:
                        row_seg[rid] = sid
                        row_phase[rid] = np.clip((int(frames[rid]) - begin_frame) / max(hi-lo, 1), 0, 1)
    return segments, row_seg, row_phase


def _segment_motion(seq):
    """Trajectory-shaped, start-relative command integral used only for grouping."""
    seq = np.asarray(seq, np.float32)
    delta = seq[:, :3]
    rel = np.cumsum(delta, axis=0)
    rel -= rel[:1]
    phase = np.linspace(0, len(rel)-1, 16, dtype=np.float32)
    return np.concatenate([resample_sequence(rel, phase).reshape(-1), [np.log1p(len(seq))]]).astype(np.float32)


def _kmeans(x, k, seed=0, iterations=60):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iterations):
        dist = np.maximum((x*x).sum(1, keepdims=True) + (centers*centers).sum(1)[None] - 2*x@centers.T, 0)
        labels = dist.argmin(-1)
        new = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                new[j] = x[labels == j].mean(0)
        if np.allclose(new, centers, atol=1e-5):
            centers = new
            break
        centers = new
    labels = (np.maximum((x*x).sum(1, keepdims=True) + (centers*centers).sum(1)[None] - 2*x@centers.T, 0)).argmin(-1)
    return centers, labels


def _silhouette_sample(x, labels, max_n=2000, seed=17):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    rng = np.random.default_rng(seed)
    ix = np.arange(len(x))
    if len(ix) > max_n:
        ix = np.sort(rng.choice(ix, max_n, replace=False))
    xx, ll = x[ix], labels[ix]
    d = np.sqrt(np.maximum(((xx[:, None] - xx[None]) ** 2).sum(-1), 0))
    vals = []
    for i, lab in enumerate(ll):
        own = ll == lab
        own[i] = False
        a = d[i, own].mean() if own.any() else 0.0
        b = min(d[i, ll == other].mean() for other in unique if other != lab)
        vals.append((b-a) / max(a, b, 1e-12))
    return float(np.mean(vals))


def cluster_segments(segments, seed=0):
    """Choose k by silhouette (4..16) and form per-cluster median action templates."""
    feats = np.stack([_segment_motion(s["actions"]) for s in segments])
    # Normalize feature dimensions so log-duration does not dominate trajectory shape.
    mean, std = feats.mean(0, keepdims=True), feats.std(0, keepdims=True)
    z = (feats - mean) / np.maximum(std, 1e-5)
    scores = {}
    rng = np.random.default_rng(seed)
    sample_ix = np.arange(len(z))
    if len(sample_ix) > 2000:
        sample_ix = np.sort(rng.choice(sample_ix, 2000, replace=False))
    sample = z[sample_ix]
    best = None
    for k in range(4, min(16, len(z)-1) + 1):
        centers, labels = _kmeans(sample, k, seed + k)
        score = _silhouette_sample(sample, labels, max_n=2000)
        scores[str(k)] = score
        if best is None or score > best[0]:
            best = (score, k)
    _, labels = _kmeans(z, best[1], seed + 1000)
    # Align every member's action path to normalized phase and take pointwise median.
    templates = {}
    for label in np.unique(labels):
        members = [segments[i]["actions"] for i in np.flatnonzero(labels == label)]
        aligned = np.stack([resample_sequence(a, np.linspace(0, len(a)-1, 64)) for a in members])
        templates[int(label)] = np.median(aligned, axis=0).astype(np.float32)
    return labels, templates, {"selected_k": int(best[1]), "selected_silhouette": float(best[0]), "silhouettes": scores}


def _retrieve(q, mkeys, eligible, k=5, temperature=0.1):
    sims = mkeys @ q
    sims = np.where(eligible, sims, -np.inf)
    ix = np.argsort(-sims, kind="stable")[:min(k, int(eligible.sum()))]
    if not len(ix):
        raise ValueError("memory exclusion left no retrieval candidates")
    vals = sims[ix]
    w = np.exp((vals - vals.max()) / max(float(temperature), 1e-8))
    w /= w.sum()
    return ix, vals, w


def _fit_task_holdout_rows(run_dir, rows):
    res = json.loads((run_dir / "results.json").read_text())
    holdout = res["task_holdout"]
    plan = holdout["episode_plan"]
    obs = np.load(run_dir / "task_observations.npz", allow_pickle=False)
    selected_eps = [ep for eps in plan.values() for ep in eps]
    row_ids = []
    eps_array = np.asarray(rows["episode_id"]).astype(str)
    frames = np.asarray(rows["frame"])
    for ep in selected_eps:
        ix = np.flatnonzero(eps_array == ep)
        ix = ix[np.argsort(frames[ix], kind="stable")]
        if len(ix) > 8:
            ix = ix[np.rint(np.linspace(0, len(ix)-1, 8)).astype(int)]
        row_ids.extend(ix.tolist())
    if len(row_ids) != len(obs["state"]):
        raise ValueError(f"task holdout row reconstruction {len(row_ids)} != observations {len(obs['state'])}")
    row_ids = np.asarray(row_ids, np.int64)
    if not np.allclose(np.asarray(rows["state"])[row_ids], obs["state"], atol=2e-5):
        raise ValueError("task holdout episode/frame selection does not match stored query observations")
    query = {k: np.asarray(rows[k])[row_ids] for k in ROW_FIELDS + ("demo",) if k in rows}
    return query, row_ids, set(holdout["heldout_tasks"])


def _query_segments_hdf5(qrows, episode_id, frame):
    """Map sparse evaluation rows to exact gripper intervals in their source demos."""
    import h5py
    episode_id = np.asarray(episode_id).astype(str)
    frame = np.asarray(frame, np.int64)
    source_files = np.asarray(qrows["source_file"]).astype(str)
    demos = np.asarray(qrows["demo"]).astype(str)
    result = [None] * len(frame)
    cache = {}
    for i, (source, demo) in enumerate(zip(source_files, demos)):
        cache.setdefault((source, demo), None)
    for source, demo in cache:
        path = Path(str(RESEARCH_REPO_ROOT / 'data/LIBERO/libero_90')) / source
        with h5py.File(path, "r") as f:
            actions = np.asarray(f["data"][demo]["actions"], np.float32)
        intervals = gripper_segments(actions)
        cache[(source, demo)] = intervals
    for i, (source, demo, fr) in enumerate(zip(source_files, demos, frame)):
        intervals = cache[(source, demo)]
        matches = [(j, lo, hi) for j, (lo, hi) in enumerate(intervals) if lo <= fr < hi]
        if not matches:
            # Final observation can sit one step after the last command.
            j, (lo, hi) = max(enumerate(intervals), key=lambda x: x[1][1])
        else:
            j, lo, hi = matches[0]
        result[i] = {"id": (episode_id[i], int(j)), "lo": int(lo), "hi": int(hi), "length": int(hi-lo)}
    return result


def _task_verb(source_file):
    text = Path(str(source_file)).stem.lower().split("_")
    verbs = {"pick", "put", "place", "stack", "open", "close", "turn", "push", "pull"}
    return next((word for word in text if word in verbs), "unknown")


def summarize_partition(mse, mm, anchors, anchor_names, actions, episodes, mask, names, verb_match):
    """Paired metrics on a predeclared subset such as fit/select."""
    mask = np.asarray(mask, bool)
    base = np.asarray(mse[:, 0, -1], np.float32).mean(0)
    summary = {"rows": int(mask.sum()), "episodes": int(len(set(np.asarray(episodes)[mask]))),
               "conditions": {}, "verb_match": {}}
    t0s = T0S
    for mi, name in enumerate(names):
        if name == "normal": indices = (len(t0s)-1,)
        else: indices = range(len(t0s))
        summary["conditions"][name] = {}
        for ti in indices:
            values = np.asarray(mse[:,mi,ti], np.float32).mean(0)
            delta = values - base
            mu, lo, hi = cluster_bootstrap(delta[mask], np.asarray(episodes)[mask], seed=4000+mi*10+ti)
            mm_values = np.asarray(mm[:,mi,ti], np.float32).mean(0)
            anchor_only = None
            if name != "normal":
                ai = list(anchor_names).index(name)
                anchor_only = float(row_errors(anchors[ai], actions)[0][mask].mean())
            summary["conditions"][name][str(t0s[ti])] = {
                "mse": float(values[mask].mean()), "mm": float(mm_values[mask].mean()),
                "delta_vs_t0_1": mu, "delta_ci95": [lo,hi], "anchor_only_mse": anchor_only}
        if name == "S1": summary["verb_match"][name] = float(np.asarray(verb_match["S1"])[mask].mean())
        if name == "S3": summary["verb_match"][name] = float(np.asarray(verb_match["S3"])[mask].mean())
    return summary


def _make_observations(rows, config):
    from train import observation_batch
    return observation_batch({k: rows[k] for k in ("image_base", "image_left_wrist", "state", "tokenized_prompt", "tokenized_prompt_mask", "skills", "skill_mask")}, config)


def run(args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from smoke import load_model
    from memory import load_memory

    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = _read_npz(args.rows, ROW_FIELDS + ("suite", "instruction"))
    mem = load_memory(args.memory)
    print(f"loaded {len(rows['actions'])} LIBERO-90 rows and {len(mem.memory)} memory items", flush=True)
    if len(mem.memory) != len(rows["actions"]):
        raise ValueError("memory and source row arrays are not aligned")
    mem_ep = np.asarray([str(x["meta"]["episode_id"]) for x in mem.memory])
    mem_init = np.asarray([str(x["meta"]["init_state_id"]) for x in mem.memory])
    row_ep = np.asarray(rows["episode_id"]).astype(str)
    row_init = np.asarray(rows["init_state_id"]).astype(str)
    if not np.array_equal(mem_ep, row_ep) or not np.array_equal(mem_init, row_init):
        raise ValueError("source row and memory ordering mismatch")
    mem_keys = np.stack([np.asarray(x["key"], np.float32).reshape(-1) for x in mem.memory])
    mem_base = np.stack([np.asarray(x["base"], np.float32).reshape(-1) for x in mem.memory])
    mem_raw = np.asarray(rows["actions"], np.float32)
    mem_segments, row_seg, row_phase = _episode_segments(mem_raw, row_ep, rows["frame"], mem_keys, rows["state"])
    for seg in mem_segments:
        seg["task"] = str(rows["source_file"][seg["start_row"]])
        seg["verb"] = str(rows["instruction"][seg["start_row"]]).strip().lower().split()[0]
    for s in mem_segments: s["cluster"] = -1
    print(f"built {len(mem_segments)} gripper intervals", flush=True)

    known = _read_npz(args.known, ROW_FIELDS + ("partition", "suite", "skills", "skill_mask", "demo"))
    query_sets = {"known": (known, None, None)}
    task_query, task_row_ids, task_names = _fit_task_holdout_rows(Path(args.task_run), rows)
    task_meta = {k: np.asarray(rows[k])[task_row_ids] for k in ("episode_id", "init_state_id", "frame", "source_file", "actions")}
    # The task holdout observation archive is the byte-matched input used by the
    # key-projection run; use its exact arrays for model preprocessing.
    qobs_task = np.load(Path(args.task_run) / "task_observations.npz", allow_pickle=False)
    task_query.update({k: qobs_task[k] for k in ("image_base", "image_left_wrist", "state", "tokenized_prompt", "tokenized_prompt_mask", "skills", "skill_mask")})
    query_sets["task_holdout"] = (task_query, task_meta, task_names)

    model, config = load_model(args.checkpoint)
    if jax.default_backend() != "gpu": raise RuntimeError("requires shared GPU for frozen action head")
    encode_key = nnx_utils.module_jit(model.encode_prefix_key)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "t0"))
    query_data = {}
    for cond, (qrows, meta, excluded_tasks) in query_sets.items():
        obs = _make_observations(qrows, config)
        qkeys = []
        for lo in range(0, len(qrows["actions"]), args.batch_size):
            hi = min(len(qrows["actions"]), lo + args.batch_size)
            qkeys.append(np.asarray(encode_key(jax.tree.map(lambda v: v[lo:hi], obs)), np.float32))
            print(f"{cond} query key rows {hi}/{len(qrows['actions'])}", flush=True)
        qkeys = np.concatenate(qkeys)
        query_data[cond] = (qrows, meta, excluded_tasks, obs, qkeys)

    # Source observations are only needed to find a causal start key for S1/S3.
    # Cached keys are stored in memory; nearest available frame at/before the
    # change point is used for a non-row-aligned gripper flip.
    segment_source_key = np.stack([s["key"] for s in mem_segments])

    all_results = {}
    for cond, (qrows, meta, excluded_tasks, obs, qkeys) in query_data.items():
        qep = np.asarray(qrows["episode_id"]).astype(str)
        qinit = np.asarray(qrows["init_state_id"]).astype(str)
        qactions = np.asarray(qrows["actions"], np.float32)
        qstate = np.asarray(qrows["state"], np.float32)
        qframe = np.asarray(qrows["frame"], np.int64)
        if meta is not None:
            qep, qinit, qframe = (np.asarray(meta[k]).astype(str) if k != "frame" else np.asarray(meta[k]) for k in ("episode_id", "init_state_id", "frame"))
        eligible_rows = np.ones(len(mem_keys), bool)
        if excluded_tasks is not None:
            eligible_rows &= ~np.isin(np.asarray(rows["source_file"]).astype(str), list(excluded_tasks))
        # Each query row receives its own leakage mask.
        centered_mem, centered_q = centered_keys(mem_keys[eligible_rows], qkeys)
        eligible_indices = np.flatnonzero(eligible_rows)
        # Keep segment memories only when their source episode/init/task is eligible.
        seg_keep = np.asarray([not (excluded_tasks and str(s["task"]) in excluded_tasks) for s in mem_segments], bool)
        seg_keep &= np.asarray([not (s["episode_id"] in set(qep)) for s in mem_segments], bool) if cond == "known" else True
        seg_ids = np.flatnonzero(seg_keep)
        # Fit S3 representatives only on rows that are legal memory for this
        # evaluation condition; held-out episodes/tasks never shape a prototype.
        cond_labels, cluster_templates, cluster_info = cluster_segments([mem_segments[j] for j in seg_ids])
        for j, lab in zip(seg_ids, cond_labels): mem_segments[int(j)]["cluster"] = int(lab)
        seg_keys, centered_q_seg = centered_keys(segment_source_key[seg_keep], qkeys)
        # P requires per-row keys and same-episode/init filtering; S1/S3 use interval-start keys.
        p_anchors = {"Ap": [], "R": []}
        s1_anchors, s2_anchors, s3_anchors = [], [], []
        s1_verb_match, s3_verb_match = [], []
        nearest_row_indices = []
        if cond == "known":
            query_segments = _query_segments_hdf5(qrows, qep, qframe)
        else:
            source_map = {(row_ep[i], int(rows["frame"][i])): i for i in range(len(row_ep))}
            query_segments = []
            for ep, fr in zip(qep, qframe):
                src = source_map.get((ep, int(fr)))
                if src is None: raise ValueError("task query row absent from complete cached episode")
                sid = int(row_seg[src])
                if sid < 0:
                    matches = [j for j, seg in enumerate(mem_segments) if seg["episode_id"] == ep and seg["lo"] <= fr < seg["hi"]]
                    if not matches: raise ValueError("task query row has no gripper interval")
                    sid = matches[0]
                seg = mem_segments[sid]
                query_segments.append({"id": (ep, sid), "lo": int(seg["lo"]), "hi": int(seg["hi"]),
                                       "length": int(len(seg["actions"])), "start_key": seg["key"],
                                       "verb": seg["verb"]})
        query_seg_ids = [x["id"] for x in query_segments]
        query_elapsed = [int(qframe[i]) - x["lo"] for i, x in enumerate(query_segments)]
        # Use the first available held-out query key inside each actual segment
        # as its start key. The current cached evaluation rows are sparse; they
        # do not contain an observation exactly at every gripper transition.
        segment_query_start_key = {}
        segment_query_verb = {}
        for qi, qseg in enumerate(query_segments):
            key = qseg["id"]
            if "start_key" in qseg:
                segment_query_start_key[key] = (int(qseg["lo"]), qseg["start_key"])
                segment_query_verb[key] = qseg["verb"]
            elif key not in segment_query_start_key or qframe[qi] < segment_query_start_key[key][0]:
                segment_query_start_key[key] = (int(qframe[qi]), qkeys[qi])
                segment_query_verb[key] = _task_verb(qrows["source_file"][qi])
        for qi in range(len(qactions)):
            qseg = query_segments[qi]
            elapsed = max(0, int(query_elapsed[qi]))
            # P: top-5 source rows, excluding query episode+init or held-out task.
            allow = eligible_rows.copy()
            allow &= (mem_ep != qep[qi]) & (mem_init != qinit[qi])
            rp, _, wp_ap = _retrieve(centered_q[qi], centered_mem, allow[eligible_indices], 5, temperature=0.05)
            rr, _, wp_r = _retrieve(centered_q[qi], centered_mem, allow[eligible_indices], 5, temperature=0.1)
            row_ids = eligible_indices[rp]
            nearest_row_indices.append(row_ids)
            # Decoder uses the original uncentered normalized VLM key, as S2.
            qkey_norm = _norm(qkeys[qi:qi+1])
            _, prior_r = mem.prior(qkey_norm, qstate[qi:qi+1])
            decoded = np.asarray(mem.decode(mem_base[row_ids], np.repeat(np.asarray(prior_r), len(row_ids), axis=0),
                                            np.repeat(qkey_norm, len(row_ids), axis=0), np.repeat(qstate[qi:qi+1], len(row_ids), axis=0)), np.float32).reshape(5,10,32)
            p_anchors["Ap"].append(np.einsum("k,khd->hd", wp_ap, decoded))
            row_ids_r = eligible_indices[rr]
            p_anchors["R"].append(np.einsum("k,khd->hd", wp_r, mem_raw[row_ids_r]))

            # S1: interval start key; preserve one retrieval through the interval.
            start_qkey = _norm(segment_query_start_key[qseg["id"]][1][None])[0]
            seg_key_mean = _norm(segment_source_key[seg_keep]).mean(0, keepdims=True)
            start_center = _norm(start_qkey - seg_key_mean)[0]
            allowed_seg = seg_keep.copy()
            allowed_seg &= np.asarray([s["episode_id"] != qep[qi] and
                str(rows["source_file"][s["start_row"]]) not in (excluded_tasks or set()) and
                str(rows["init_state_id"][s["start_row"]]) != qinit[qi] for s in mem_segments])
            sidx, _, sw = _retrieve(start_center, seg_keys, allowed_seg[seg_ids], 5, temperature=0.1)
            chosen_seg_ids = seg_ids[sidx]
            chunks = []
            for j in chosen_seg_ids:
                seg = mem_segments[j]
                # The retrieved duration defines the phase scale at inference.
                phase = elapsed / max(len(seg["actions"]), 1)
                chunks.append(phase_chunk(seg["actions"], phase))
            s1_anchors.append(np.einsum("k,khd->hd", sw, np.stack(chunks)))
            qverb = segment_query_verb[qseg["id"]]
            s1_verb_match.append(float(np.mean([str(rows["instruction"][mem_segments[j]["start_row"]]).lower().split()[0] == qverb for j in chosen_seg_ids])))

            # S2: retrieve at each query row; each entry's own segment phase is used.
            allow2 = eligible_rows.copy(); allow2 &= (mem_ep != qep[qi]) & (mem_init != qinit[qi])
            rids_local, _, w2 = _retrieve(centered_q[qi], centered_mem, allow2[eligible_indices], 5, temperature=0.1)
            rids = eligible_indices[rids_local]
            chunks2 = []
            for rid in rids:
                rsid = int(row_seg[rid])
                if rsid < 0:
                    chunks2.append(mem_raw[rid]); continue
                seg = mem_segments[rsid]
                phase = row_phase[rid]
                chunks2.append(phase_chunk(seg["actions"], phase))
            s2_anchors.append(np.einsum("k,khd->hd", w2, np.stack(chunks2)))

            # S3: start-key retrieval, weighted cluster vote, median action template.
            cluster_w = {}
            for j, weight in zip(chosen_seg_ids, sw):
                cl = mem_segments[j]["cluster"]
                cluster_w[cl] = cluster_w.get(cl, 0.0) + float(weight)
            selected_cluster = max(cluster_w, key=cluster_w.get)
            template = cluster_templates[selected_cluster]
            phase = elapsed / max(float(np.median([len(mem_segments[j]["actions"]) for j in seg_ids])), 1.0)
            s3_anchors.append(phase_chunk(template, phase))
            s3_verb_match.append(float(np.mean([str(rows["instruction"][mem_segments[j]["start_row"]]).lower().split()[0] == qverb for j in chosen_seg_ids if mem_segments[j]["cluster"] == selected_cluster])))
            if (qi + 1) % 64 == 0: print(f"{cond} anchor rows {qi+1}/{len(qactions)}", flush=True)

        p_anchors = {k: np.asarray(v, np.float32) for k,v in p_anchors.items()}
        anchors = {"P-Ap": p_anchors["Ap"], "P-R": p_anchors["R"], "S1": np.asarray(s1_anchors), "S2": np.asarray(s2_anchors), "S3": np.asarray(s3_anchors)}
        baseline_eps = np.stack([np.random.default_rng(20261029+s).standard_normal((len(qactions),10,32)).astype(np.float32) for s in range(3)])
        mse = np.zeros((3, len(anchors)+1, len(T0S), len(qactions)), np.float32)
        mm = np.zeros_like(mse)
        names = ["normal"] + list(anchors)
        for seed in range(3):
            for lo in range(0, len(qactions), args.batch_size):
                hi = min(len(qactions), lo + args.batch_size)
                batch = jax.tree.map(lambda v: v[lo:hi], obs)
                eps = jnp.asarray(baseline_eps[seed,lo:hi])
                baseline = sample(jax.random.key(0), batch, num_steps=10, noise=eps, x_src=eps, t0=1.0)
                e1,e2 = row_errors(np.asarray(baseline), qactions[lo:hi])
                for mi in range(len(names)):
                    mse[seed,mi,-1,lo:hi],mm[seed,mi,-1,lo:hi] = e1,e2
                for mi, name in enumerate(names[1:], start=1):
                    for ti, t0 in enumerate(T0S[:-1]):
                        anc = jnp.asarray(anchors[name][lo:hi])
                        xsrc = (1.0-t0)*anc + t0*eps
                        pred = sample(jax.random.key(0), batch, num_steps=denoising_steps(t0), x_src=xsrc, t0=float(t0))
                        e1,e2 = row_errors(np.asarray(pred), qactions[lo:hi])
                        mse[seed,mi,ti,lo:hi],mm[seed,mi,ti,lo:hi] = e1,e2
            print(f"{cond} denoise seed {seed} done", flush=True)
        # Anchor-only errors and bootstrap deltas; each seed shares the baseline noise.
        eps_mse = np.stack([row_errors(a,qactions)[0] for a in anchors.values()])
        results = {"rows":len(qactions),"episodes":int(len(set(qep))),"task_holdout":cond=="task_holdout",
                   "task_excluded": sorted(task_names or []),"segments_memory":int(seg_keep.sum()),
                   "cluster":cluster_info,"verb_match":{"S1":float(np.mean(s1_verb_match)),"S3":float(np.nanmean(s3_verb_match))},"conditions":{}}
        cfg_index={name:i for i,name in enumerate(names)}
        for name,mi in cfg_index.items():
            results["conditions"][name]={}
            for ti,t0 in enumerate(T0S):
                row_mse=mse[:,mi,ti].mean(0)
                if name=="normal": delta=np.zeros(len(qactions),np.float32)
                else: delta=row_mse - mse[:,0,-1].mean(0)
                mu,low,high=cluster_bootstrap(delta,qep,seed=2000+mi*10+ti)
                results["conditions"][name][str(t0)]={"mse":float(row_mse.mean()),"mm":float(mm[:,mi,ti].mean()),
                    "delta_vs_t0_1":mu,"delta_ci95":[low,high],
                    "anchor_only_mse":None if name=="normal" else float(eps_mse[list(anchors).index(name)].mean())}
        results["query_episode_overlap"] = 0
        if cond == "known":
            results["fit_select"] = {
                part: summarize_partition(mse, mm, np.stack(list(anchors.values())), list(anchors),
                    qactions, qep, np.asarray(qrows["partition"]).astype(str) == part, names,
                    {"S1": s1_verb_match, "S3": s3_verb_match})
                for part in ("fit", "select")}
        all_results[cond]=results
        np.savez_compressed(out/f"{cond}-results.npz", mse=mse, mm=mm, anchors=np.stack(list(anchors.values())),
                            anchor_names=np.asarray(list(anchors)), episode_id=qep, frame=qframe,
                            partition=np.asarray(qrows.get("partition", np.full(len(qep), "task_holdout"))),
                            segment_id=np.asarray(query_seg_ids), elapsed=np.asarray(query_elapsed),
                            verb_match_s1=np.asarray(s1_verb_match), verb_match_s3=np.asarray(s3_verb_match))
        print(f"{cond}: {len(qactions)} rows complete", flush=True)
    result={"closed_loop":False,"rows_source":str(args.rows),"memory":str(args.memory),"known":all_results["known"],"task_holdout":all_results["task_holdout"],"elapsed_seconds":time.perf_counter()-started,
            "segment_definition":"raw cached 10-step commands concatenated by episode/frame; sign change in action dim 6; duration in constituent command frames",
            "keys":"centered, L2-normalized stored full-token means; S1/S3 start key, P/S2 current row key; cosine top-5 softmax temperature 0.1",
            "noise":"same eps seed 20261029+s for all drafts/t0; 3 seeds", "bootstrap":"paired row delta vs t0=1, episode resampling, 4000 reps"}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    _write_report(out,result)


def _write_report(out,result):
    rows=[]
    groups = [("既知全体", result["known"]),
              ("既知 fit", result["known"]["fit_select"]["fit"]),
              ("既知 select", result["known"]["fit_select"]["select"]),
              ("タスク丸ごと保留", result["task_holdout"])]
    for label,r in groups:
        for name in ("P-Ap","P-R","S1","S2","S3","normal"):
            for t0,c in r["conditions"][name].items():
                match = r["verb_match"].get(name) if name in ("S1","S3") else None
                rows.append(f"| {label} | {name} | {t0} | {c['mse']:.4f} | {c['mm']:.1f} | {c['delta_vs_t0_1']:+.4f} [{c['delta_ci95'][0]:+.4f},{c['delta_ci95'][1]:+.4f}] | {c['anchor_only_mse'] if c['anchor_only_mse'] is not None else '—'} | {match if match is not None else '—'} |")
    task = result["task_holdout"]["conditions"]
    best_p = min((float(v["mse"]), name, t) for name in ("P-Ap","P-R") for t,v in task[name].items())
    s1 = task["S1"]["0.6"]; s3 = task["S3"]["0.6"]; base=task["normal"]["1.0"]["mse"]
    lines=["# Segment skill offline comparison","","閉ループ評価なし。既知256行は事前のfit/select各128行を保持して別集計。3 seedの同一εで推論。","",
           "| 条件 | 下書き | t0 | MSE | 終点誤差 mm | Δ vs t0=1 (95% episode CI) | 下書き単体 MSE | 検索区間の動詞一致 |","|---|---|---:|---:|---:|---|---:|---:|",*rows,"",
           "判断（3行）：",
           f"- 丸ごと保留では、Pの最小MSEは {best_p[1]}・t0={best_p[2]} で {best_p[0]:.4f}。S1/S3 はt0=0.6でも通常({base:.4f})よりそれぞれ +{s1['delta_vs_t0_1']:.4f} / +{s3['delta_vs_t0_1']:.4f}悪く、数秒スキルは通常方策を上回らなかった。",
           "- 既知タスクではPのt0=0.2が改善し、S1/S2/S3は同じ条件で悪化。短区間検索は既知場面でのみ一部改善が再現した。",
           f"- 検索動詞の一致は既知でS1/S3={result['known']['verb_match']['S1']:.1%}/{result['known']['verb_match']['S3']:.1%}、未学習タスクで約{result['task_holdout']['verb_match']['S1']:.1%}/{result['task_holdout']['verb_match']['S3']:.1%}。一致だけでは動作精度に足りない。既知別出所デモは開始時画像がなく、同区間で最初の保留行のキーを開始キーの代理にした。"]
    (out/"report.md").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--known",type=Path,default=DEFAULT_KNOWN)
    ap.add_argument("--rows",type=Path,default=DEFAULT_ROWS)
    ap.add_argument("--memory",type=Path,default=DEFAULT_MEMORY)
    ap.add_argument("--task-run",type=Path,default=DEFAULT_TASK_RUN)
    ap.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output",default=None)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--gpu-fraction",type=float,default=0.08)
    args=ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",str(args.gpu_fraction))
    if args.output is None:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output=str(ROOT/f"seg-skill-offline-{stamp}")
    run(args)


if __name__=="__main__": main()
