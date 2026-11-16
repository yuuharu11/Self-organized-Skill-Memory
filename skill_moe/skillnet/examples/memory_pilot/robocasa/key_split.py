#!/usr/bin/env python3
"""RoboCasa context-history retrieval with a split motion-type key.

The split key encoder is adapted from `key_split_ablation.py` in the
query-invariance worktree (KS-a): 5 two-frame tokens, shared 14->256->256
projection, positional embeddings, one Transformer encoder layer, and
per-token normalization before mean pooling.  Dataset paths are controlled by
SKILLNET_STATE_DIR; source artifacts are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
ROOT = STATE / "outputs/robocasa-memory"
CTX = ROOT / "ctx-01"
GT = ROOT / "gt-def2"
SOURCE = ROOT / "memory-01/policy-order-v2"
OUT = Path(os.environ.get("ROBOCASA_KEY_SPLIT_OUT", ROOT / "key-split-01"))
EXAMPLES = Path(__file__).resolve().parents[2] / "robocasa"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
SEEDS = (4101, 4102, 4103)
FOLD_SEED = 20261112
CHUNKS = 5
WIDTH = 256

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex
import latent_query as lq


def _append(stage, **values):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **values}
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=ex._json_default) + "\n")
    print(json.dumps(row, ensure_ascii=False, default=ex._json_default), flush=True)


def ks_a_scores(query, key_tokens, scale=1.0):
    """KS-a score: normalized query dot mean of individually normalized tokens."""
    import torch.nn.functional as F
    if query.ndim != 2 or key_tokens.ndim != 3 or query.shape[-1] != key_tokens.shape[-1]:
        raise ValueError(f"expected query [B,D], keys [M,K,D], got {query.shape} / {key_tokens.shape}")
    q = F.normalize(query, dim=-1)
    keys = F.normalize(key_tokens, dim=-1)
    return (q @ keys.mean(dim=1).T) * scale


def _model_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class QueryMLP(nn.Module):
        def __init__(self, q_dim=2194):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(q_dim, 512), nn.GELU(), nn.Linear(512, 512),
                                     nn.GELU(), nn.Linear(512, WIDTH))

        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    class ContextRanker(nn.Module):
        def __init__(self, q_dim=2194, k_dim=2064):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(q_dim, 512), nn.GELU(), nn.Linear(512, 512),
                                       nn.GELU(), nn.Linear(512, WIDTH))
            self.key = nn.Sequential(nn.Linear(k_dim, 512), nn.GELU(), nn.Linear(512, WIDTH))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))

        def forward(self, query, keys):
            q = F.normalize(self.query(query), dim=-1)
            k = F.normalize(self.key(keys), dim=-1)
            return q @ k.T * self.log_scale.exp()

    class SplitRanker(nn.Module):
        def __init__(self, q_dim=2194):
            super().__init__()
            self.query = QueryMLP(q_dim)
            self.key_frame = nn.Sequential(nn.Linear(14, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
            self.key_pos = nn.Parameter(torch.zeros(1, CHUNKS, WIDTH))
            self.key_attn = nn.TransformerEncoderLayer(WIDTH, 8, 4 * WIDTH, batch_first=True,
                                                       activation="gelu", norm_first=True)
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))

        def encode_keys(self, action10x7):
            if action10x7.ndim != 3 or tuple(action10x7.shape[1:]) != (10, 7):
                raise ValueError(f"split-key input must be [N,10,7], got {action10x7.shape}")
            chunks = action10x7.reshape(len(action10x7), CHUNKS, 14).float()
            raw = self.key_attn(self.key_frame(chunks) + self.key_pos)
            return F.normalize(raw, dim=-1)

        def forward(self, query, keys):
            return ks_a_scores(self.query(query), keys, self.log_scale.exp())

    return ContextRanker, SplitRanker


def test_score_contract():
    import torch
    q = torch.tensor([[3., 4.], [1., 0.]])
    keys = torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]])
    got = ks_a_scores(q, keys, 2.)
    expected = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(keys, dim=-1).mean(1).T * 2.
    torch.testing.assert_close(got, expected)
    return True


def _history_mask(frames):
    mask = np.zeros((len(frames), 10), bool)
    for i, frame in enumerate(frames):
        n = min(max(int(frame), 0), 10)
        if n:
            mask[i, 10 - n:] = True
    return mask


def _history_replacement(hist, task, episode, seed):
    """One deterministic different-episode row per query, within its task."""
    rng = np.random.default_rng(int(seed))
    result = np.empty(len(task), np.int64)
    groups = {}
    for i, (t, e) in enumerate(zip(task, episode, strict=True)):
        groups.setdefault(int(t), []).append(i)
    for i, (t, e) in enumerate(zip(task, episode, strict=True)):
        options = [j for j in groups[int(t)] if episode[j] != e]
        result[i] = options[int(rng.integers(len(options)))] if options else i
    return result


def _corrupt_history(hist, mask, ids, rng, replacement, shifted):
    """Match key-lite context_history training: 50% zero, 25% damaged, 25% true."""
    raw = hist[ids].copy()
    valid = mask[ids].copy()
    draw = rng.random(len(ids))
    zero = draw < .50
    damage = (draw >= .50) & (draw < .75)
    raw[zero] = 0.; valid[zero] = False
    for local in np.flatnonzero(damage):
        row = int(ids[local])
        # Same-episode temporal offsets where available; otherwise a distinct
        # same-task episode. This is the key-lite corruption recipe.
        options = [int(shifted[delta][row]) for delta in (-20, -10, 10, 20)
                   if int(shifted[delta][row]) >= 0]
        if options and rng.random() < .8:
            src = options[int(rng.integers(len(options)))]
        else:
            src = int(replacement[row])
        raw[local] = hist[src]
        valid[local] = mask[src]
    return raw, valid


def _query_arrays(rows, features, pool, fit_ids, memory=True):
    """Fit fold-specific normalizers and assemble same 2194-D query inputs."""
    ctx_index = np.asarray(rows["memory_context"], np.int64)
    state = np.asarray(rows["memory_state"], np.float32)
    hist = ex.action_storage_to_policy(np.asarray(rows["memory_hist"], np.float32))
    mask = _history_mask(rows["memory_frame"])
    f = features[ctx_index].astype(np.float32)
    fm = f[fit_ids].mean(0); fs = np.maximum(f[fit_ids].std(0), .01)
    sm = state[fit_ids].mean(0); ss = np.maximum(state[fit_ids].std(0), 1e-4)
    # Match key-lite: moments are computed over valid history entries only.
    valid_hist = hist[fit_ids][mask[fit_ids]]
    hm = valid_hist.mean(0) if len(valid_hist) else np.zeros(12, np.float32)
    hs = np.maximum(valid_hist.std(0) if len(valid_hist) else np.ones(12, np.float32), 1e-4)
    base = np.concatenate(((f - fm) / fs, (state - sm) / ss), axis=1).astype(np.float32)
    hn = ((hist - hm) / hs).astype(np.float32) * mask[..., None]
    q = np.concatenate((base, hn.reshape(len(hn), -1), mask.astype(np.float32)), axis=1)
    cix = np.asarray(rows["candidate_context"], np.int64)
    cstate = np.asarray(rows["candidate_state"], np.float32)
    cfeat = features[cix].astype(np.float32)
    keys = np.concatenate(((cfeat - fm) / fs, (cstate - sm) / ss), axis=1).astype(np.float32)
    return q.astype(np.float32), keys, {"feature_mean": fm, "feature_std": fs, "state_mean": sm,
            "state_std": ss, "history_mean": hm, "history_std": hs}


def _split_actions(pool):
    """Build [candidate,10,7] type keys: EEF DCT shape plus saved grip trace.

    The six EEF channels come from the already decoded S-shared DCT shape;
    the seventh channel is the unit's recorded gripper command, which is not
    part of DCT compression and is preserved separately in the memory.
    """
    actions = ex.action_storage_to_policy(np.asarray(pool["action10"], np.float32))
    shape = np.asarray(pool["shape"], np.float32)
    if shape.shape != (len(actions), 10, 6):
        raise ValueError(f"expected stored 10x6 DCT type reconstructions; got {shape.shape}")
    result = np.concatenate((shape, actions[:, :, 6:7]), axis=2).astype(np.float32)
    return result


def _evaluate_training_seed(*, net, kind, ck, query, candidate_context, split_keys, teacher, allowed,
                            memory_episode, held_fold, folds, batch_size=128, device="cuda"):
    import torch
    import torch.nn.functional as F
    net.eval()
    with torch.no_grad():
        cand = (torch.as_tensor(candidate_context, dtype=torch.float32, device=device) if kind == "context"
                else torch.as_tensor(split_keys, dtype=torch.float32, device=device))
        ids = np.flatnonzero(folds == held_fold)
        picks = []
        for st in range(0, len(ids), batch_size):
            ix = ids[st:st + batch_size]
            q = torch.as_tensor(query[ix], dtype=torch.float32, device=device)
            logits = net(q, cand).masked_fill(~torch.as_tensor(allowed[ix], device=device), -1e9)
            picks.extend(logits.argmax(1).cpu().numpy().tolist())
    err = np.asarray(teacher, np.float32)
    local = np.arange(len(ids))
    return ids, np.asarray(picks, np.int64), err[ids, np.asarray(picks, np.int64)]


def train_and_evaluate(*, epochs=12, batch_size=128, device="cuda", threads=8):
    """Train context+history and split-key models under identical episode folds."""
    import torch
    from scipy.stats import spearmanr
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing result directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(threads))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.15, 0)
    rows = np.load(CTX / "cache/rows.npz", allow_pickle=False)
    pool = dict(np.load(CTX / "cache/pool.npz", allow_pickle=False))
    feat = np.asarray(np.load(CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
    train_teacher = np.load(CTX / "cache/train_teacher_final.npz", allow_pickle=False)
    eval_data = np.load(GT / "per_query.npz", allow_pickle=False)
    gt_features = np.load(GT / "query_features.npz", allow_pickle=False)
    errors = np.asarray(train_teacher["error"], np.float32)
    allowed = np.asarray(train_teacher["allowed"], bool)
    memtask = np.asarray(rows["memory_task"], np.int64)
    memep = np.asarray(rows["memory_episode"], np.int64)
    folds, _ = lq._episode_folds(memtask, memep, FOLD_SEED, 5)
    candidate_split = _split_actions(pool)
    query_task = np.asarray(eval_data["task"], np.int64)
    query_global_ep = np.asarray(eval_data["episode"], np.int64)
    query_frame = np.asarray(eval_data["frame"], np.int64)
    if len(query_task) != 2133 or not np.array_equal(query_task, np.asarray(eval_data["task"])):
        raise AssertionError("gt-def2 query contract changed; expected the fixed 2,133-row set")
    if eval_data["eligible"].shape != (len(query_task), len(pool["task"])):
        raise AssertionError("heldout candidate/error matrix is not aligned with pool")

    # Assemble memory query metadata and exact heldout action rows from cached episodes.
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    global_to_local = {(int(d["global_episode"]), int(d["task_id"])): int(d["episode"]) for d in manifest["demos"]}
    states, histories, targets, remaining_steps = [], [], [], []
    episode_actions = {}
    for t, ge, frame in zip(query_task, query_global_ep, query_frame, strict=True):
        local_ep = global_to_local[(int(ge), int(t))]
        key = (int(t), local_ep)
        if key not in episode_actions:
            episode_actions[key] = ex._load_cached_episode(SOURCE, *key)[:2]
        acts, sts = episode_actions[key]
        f = int(frame)
        states.append(ex.state_storage_to_policy(sts[f]))
        h = np.zeros((10, 12), np.float32); past = acts[max(0, f - 10):f]
        if len(past): h[-len(past):] = past
        histories.append(ex.action_storage_to_policy(h))
        target = ex.action_storage_to_policy(acts[f:min(f + 15, len(acts))])
        if len(target) < 15:
            target = np.concatenate((target, np.repeat(target[-1:], 15 - len(target), axis=0)), axis=0)
        targets.append(target)
        remaining_steps.append(len(acts) - f)
    eval_state = np.asarray(states, np.float32)
    eval_hist = np.asarray(histories, np.float32)
    eval_target = np.asarray(targets, np.float32)
    remaining_steps = np.asarray(remaining_steps, np.int32)
    if eval_target.shape != (2133, 15, 12):
        raise AssertionError(f"evaluation targets should cover 15 steps: {eval_target.shape}")

    # Context-history memory input used for training; heldout query prefix is from gt-def2 itself.
    eval_features = np.asarray(gt_features["prefix"], np.float32)
    feature_ref = np.asarray(np.load(GT / "query_features.npz", allow_pickle=False)["prefix"], np.float32)
    if not np.array_equal(eval_features, feature_ref):
        raise AssertionError("prefix query cache mismatch")

    result = {"n_eval": len(query_task), "n_memory_queries": len(memtask), "n_candidates": len(pool["task"]),
              "fold_seed": FOLD_SEED, "epochs_max": int(epochs), "batch_size": int(batch_size),
              "history_schedule": {"zero": .50, "corrupt": .25, "correct": .25}, "seeds": {}}
    ContextRanker, SplitRanker = _model_classes()
    import serve_memonly_policy as memserve
    _RankNet, ContextNet, MagnitudeNet = memserve._network_classes()
    context_ck = torch.load(CTX / "context_model.pt", map_location="cpu", weights_only=False)
    context_net = ContextNet(int(context_ck["input_dim"]))
    context_net.load_state_dict(context_ck["state"]); context_net.eval()
    magnitude_ck = torch.load(ROOT / "key-lite-01/magnitude_v2/magnitude_head.pt",
                              map_location="cpu", weights_only=False)
    magnitude_net = MagnitudeNet(int(magnitude_ck["input_dim"]))
    magnitude_net.load_state_dict(magnitude_ck["state"]); magnitude_net.eval()
    unit_meta, _ = memserve.unit_metadata(pool)
    action_norm_file = lq.expert.CHECKPOINT / "assets/norm_stats.json"
    norm_stats = json.loads(action_norm_file.read_text())["norm_stats"]["actions"]
    action_mean = np.asarray(norm_stats["mean"], np.float32)
    action_std = np.asarray(norm_stats["std"], np.float32)
    task_eval = query_task
    episode_eval = query_global_ep
    frame0 = query_frame == 0
    context_cache = np.load(CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r")
    mem_context = np.asarray(rows["memory_context"], np.int64)
    candidate_context = np.asarray(rows["candidate_context"], np.int64)
    replacement = _history_replacement(np.asarray(rows["memory_hist"]), memtask, memep, 9107)
    memframe = np.asarray(rows["memory_frame"], np.int64)
    mem_lookup = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(
        zip(memtask, memep, memframe, strict=True))}
    shifted = {d: np.asarray([mem_lookup.get((int(t), int(e), int(f) + d), -1)
                              for t, e, f in zip(memtask, memep, memframe, strict=True)], np.int64)
               for d in (-20, -10, 10, 20)}
    target_error = np.asarray(eval_data["type_error"], np.float32)
    eval_eligible = np.asarray(eval_data["eligible"], bool)
    if not np.array_equal(eval_eligible, np.asarray(eval_data["valid"], bool) &
                          (np.asarray(pool["length"])[None, :] >= 5)):
        # The source's eligible mask additionally accounts for query-relative
        # suffix length; it is the authoritative gt-def2 comparison mask.
        if np.any(eval_eligible & ~np.asarray(eval_data["valid"], bool)):
            raise AssertionError("gt-def2 eligible contains invalid candidates")

    for seed in SEEDS:
        held_fold = (int(seed) - SEEDS[0]) % 5
        fit_ids = np.flatnonzero(folds != held_fold)
        dev_ids = np.flatnonzero(folds == held_fold)
        if not len(fit_ids) or not len(dev_ids):
            raise AssertionError(f"seed {seed}: empty episode fold")
        qmem, cctx, norms = _query_arrays(rows, context_cache, pool, fit_ids)
        # Rebuild all memory context inputs using the fold-only normalizers.
        qbase = qmem[:, :2064]
        hnorm = ((ex.action_storage_to_policy(np.asarray(rows["memory_hist"], np.float32)) - norms["history_mean"])
                 / norms["history_std"]).astype(np.float32)
        hmask = _history_mask(rows["memory_frame"])
        hnorm *= hmask[..., None]
        # Candidate split key inputs are deterministic decoded type shapes.
        if candidate_split.shape != (len(pool["task"]), 10, 7):
            raise AssertionError(f"split key alignment error: {candidate_split.shape}")

        for kind in ("context", "split"):
            torch.manual_seed(int(seed))
            rng = np.random.default_rng(int(seed))
            if kind == "context":
                net = ContextRanker(2194, 2064).to(device)
                cand_tensor = torch.as_tensor(cctx, dtype=torch.float32, device=device)
            else:
                net = SplitRanker(2194).to(device)
                key_raw = torch.as_tensor(candidate_split, dtype=torch.float32, device=device)
                with torch.no_grad():
                    cand_tensor = net.encode_keys(key_raw)
                # Re-encode with gradients for every update below.
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            gap = np.mean(np.where(allowed[fit_ids], errors[fit_ids], 0.), axis=1) - np.min(
                np.where(allowed[fit_ids], errors[fit_ids], np.inf), axis=1)
            tau = max(.1 * float(np.median(gap)), 1e-8)
            log = []
            best = (float("inf"), None, None)
            stale = 0
            teacher = np.zeros_like(errors, np.float32)
            tl = np.where(allowed[fit_ids], -errors[fit_ids] / tau, -1e9)
            tl -= tl.max(axis=1, keepdims=True)
            tp = np.exp(tl).astype(np.float32)
            tp /= np.maximum(tp.sum(axis=1, keepdims=True), 1e-20)
            teacher[fit_ids] = tp
            for epoch in range(1, int(epochs) + 1):
                net.train(); order = rng.permutation(fit_ids); losses = []
                for start in range(0, len(order), int(batch_size)):
                    ids = order[start:start + int(batch_size)]
                    raw, mask = _corrupt_history(np.asarray(rows["memory_hist"], np.float32), hmask,
                                                 ids, rng, replacement, shifted)
                    raw = ex.action_storage_to_policy(raw)
                    histx = ((raw - norms["history_mean"]) / norms["history_std"]).astype(np.float32)
                    histx *= mask[..., None]
                    q = np.concatenate((qbase[ids], histx.reshape(len(ids), -1), mask.astype(np.float32)), axis=1)
                    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
                    if kind == "context":
                        scores = net(q_t, cand_tensor)
                    else:
                        key_t = net.encode_keys(key_raw)
                        scores = net(q_t, key_t)
                    allow_t = torch.as_tensor(allowed[ids], dtype=torch.bool, device=device)
                    tgt = torch.as_tensor(teacher[ids], dtype=torch.float32, device=device)
                    scores = scores.masked_fill(~allow_t, -1e9)
                    loss = -(tgt * torch.log_softmax(scores, dim=-1)).sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                net.eval(); pred = []
                with torch.no_grad():
                    if kind == "split": cand_tensor = net.encode_keys(key_raw)
                    for start in range(0, len(dev_ids), int(batch_size)):
                        ids = dev_ids[start:start + int(batch_size)]
                        scores = net(torch.as_tensor(qmem[ids], dtype=torch.float32, device=device), cand_tensor)
                        scores = scores.masked_fill(~torch.as_tensor(allowed[ids], device=device), -1e9)
                        pred.append(scores.argmax(1).cpu().numpy())
                picks = np.concatenate(pred)
                metric = float(errors[dev_ids, picks].mean())
                log.append({"epoch": epoch, "train_listwise": float(np.mean(losses)), "dev_type_error": metric})
                _append("train_epoch", seed=int(seed), model=kind, epoch=epoch,
                        train_listwise=log[-1]["train_listwise"], dev_type_error=metric)
                if metric < best[0]:
                    best = (metric, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
                    stale = 0
                else:
                    stale += 1
                    if stale >= 3 and epoch >= 5:
                        break
            net.load_state_dict(best[1]); net.eval()
            ckpath = OUT / f"{kind}_seed{seed}.pt"
            torch.save({"state": best[1], "seed": seed, "held_fold": held_fold, "best_epoch": best[2],
                        "dev_type_error": best[0], "teacher_tau": tau, "norms": norms,
                        "history_schedule": result["history_schedule"], "curve": log}, ckpath)

            # Heldout query normalization follows the fold's training statistics.
            held_context = np.asarray(rows["heldout_context"], np.int64)
            held_state = np.asarray(rows["heldout_state"], np.float32)
            # gt-def2 rows are a fixed stride-5 subset; use their own saved prefix
            # and state from the source episode while retaining gt-def2 ordering.
            base_eval = np.concatenate(((eval_features - norms["feature_mean"]) / norms["feature_std"],
                                        (eval_state - norms["state_mean"]) / norms["state_std"]), axis=1).astype(np.float32)
            # Draft postprocessing is exactly the existing gt-def2 path:
            # Context gripper head + v2 magnitude head, then active7 asset norm.
            c_norm = context_ck["norm"]
            eval_qctx = np.concatenate(((eval_features - np.asarray(c_norm["feature_mean"], np.float32)) /
                                        np.asarray(c_norm["feature_std"], np.float32),
                                        (eval_state - np.asarray(c_norm["state_mean"], np.float32)) /
                                        np.asarray(c_norm["state_std"], np.float32)), axis=1).astype(np.float32)
            with torch.no_grad():
                grip_sign = np.where(context_net.grip(torch.as_tensor(eval_qctx, dtype=torch.float32))[:, :5].numpy() >= 0,
                                     1., -1.).astype(np.float32)
            magnitude_base = eval_qctx if magnitude_ck.get("history_free", False) else base_eval
            hist_eval_mask = _history_mask(query_frame)
            # Different heldout episode, same task; deterministic per seed.
            rep_eval = _history_replacement(eval_hist, query_task, query_global_ep, int(seed) + 917)
            conds = {}
            for condition in ("correct", "zero", "replace"):
                h = eval_hist.copy(); hm = hist_eval_mask.copy()
                if condition == "zero":
                    h[:] = 0.; hm[:] = False
                elif condition == "replace":
                    h = h[rep_eval].copy(); hm = hm[rep_eval].copy()
                hn = ((h - norms["history_mean"]) / norms["history_std"]).astype(np.float32) * hm[..., None]
                qe = np.concatenate((base_eval, hn.reshape(len(hn), -1), hm.astype(np.float32)), axis=1)
                with torch.no_grad():
                    if kind == "split": cand_tensor = net.encode_keys(key_raw)
                    rowscores = []
                    for start in range(0, len(qe), int(batch_size)):
                        sc = net(torch.as_tensor(qe[start:start + int(batch_size)], dtype=torch.float32, device=device), cand_tensor)
                        # Candidate episode exclusion aligns to global episode IDs.
                        qep = query_global_ep[start:start + len(sc)]
                        permit = qep[:, None] != np.asarray(pool["global_episode"], np.int64)[None, :]
                        permit &= eval_eligible[start:start + len(sc)]
                        sc = sc.masked_fill(~torch.as_tensor(permit, device=device), -1e9)
                        rowscores.append(sc.cpu().numpy())
                score = np.concatenate(rowscores)
                pick = score.argmax(1)
                ix = np.arange(len(pick))
                te = target_error[ix, pick]
                ae = np.asarray(eval_data["action_error"], np.float32)[ix, pick]
                conds[condition] = {"pick": pick, "scores": score, "type_error": te, "action_first5": ae}

            # Persist predictions and summarize by selected segment windows.
            summary = {}
            for condition, pred in conds.items():
                pick = pred["pick"]
                type_err = pred["type_error"]
                q_action = eval_target[:, :, :7]
                offsets = np.asarray(pool["action_offsets"], np.int64)
                flat_actions = np.asarray(pool["actions_flat"], np.float32)
                selected = np.zeros((len(pick), 15, 7), np.float32)
                selected_scale = np.empty(len(pick), np.float32)
                for ri, unit_id in enumerate(pick):
                    raw12 = ex.action_storage_to_policy(flat_actions[offsets[unit_id]:offsets[unit_id + 1]])
                    n = min(15, len(raw12)); raw15 = np.zeros((15, 12), np.float32)
                    raw15[:n] = raw12[:n]
                    if n < 15: raw15[n:] = raw12[-1]
                    mag_input = np.concatenate((magnitude_base[ri], unit_meta[unit_id]), axis=0)
                    mag_input = (mag_input - np.asarray(magnitude_ck["feature_mean"], np.float32)) / np.asarray(
                        magnitude_ck["feature_std"], np.float32)
                    with torch.no_grad():
                        scale = float(torch.exp(magnitude_net(torch.as_tensor(mag_input[None], dtype=torch.float32)))
                                      .clamp(.25, 4.).item())
                    selected_scale[ri] = scale
                    raw12[:, :6] *= scale
                    raw12[:5, 6] = grip_sign[ri, :min(5, len(raw12))]
                    if len(raw12) < 15: raw12 = np.concatenate((raw12, np.repeat(raw12[-1:], 15-len(raw12), axis=0)))
                    selected[ri] = raw12[:15, :7]
                truth_norm = ((eval_target[:, :, :7] - action_mean[:7]) /
                              (action_std[:7] + 1e-6)).astype(np.float32)
                draft_norm = ((selected - action_mean[None, None, :7]) /
                              (action_std[None, None, :7] + 1e-6)).astype(np.float32)
                # The stored first-five errors are authoritative gt-def2 values;
                # assert the duplicated postprocessor agrees before reporting.
                computed_first5 = np.mean((draft_norm[:, :5] - truth_norm[:, :5]) ** 2, axis=(1, 2))
                exact_first5 = pred["action_first5"]
                if not np.allclose(computed_first5, exact_first5, rtol=2e-3, atol=2e-5):
                    max_delta = float(np.max(np.abs(computed_first5 - exact_first5)))
                    raise AssertionError(f"gt-def2 draft construction mismatch (max first5 MSE delta {max_delta})")
                # The current gt-def2 metric is the exact first-five active7
                # reconstruction with v2 magnitude/context grip; use it as the
                # primary measure. Later windows are direct stored-segment
                # comparisons and are labeled as raw, unscaled diagnostics.
                action_segments = {}
                for name, lo, hi, min_remaining in (("first5", 0, 5, 5), ("steps5_9", 5, 10, 10),
                                                    ("steps10_14", 10, 15, 15)):
                    use = remaining_steps >= min_remaining
                    if lo == 0:
                        vals = pred["action_first5"][use]
                    else:
                        # Do not count padded demonstration tails or segments
                        # that have ended; continuation MSE is only defined
                        # where both the heldout target and candidate unit have
                        # the requested five real steps.
                        use &= np.asarray(pool["length"], np.int32)[pick] >= hi
                        vals = np.mean((draft_norm[use, lo:hi] - truth_norm[use, lo:hi]) ** 2, axis=(1, 2))
                    action_segments[name] = {**lq._episode_bootstrap(vals, query_task[use], query_global_ep[use],
                                                                     seed=int(seed) + len(name), n_boot=5000),
                                             "n_rows": int(use.sum()),
                                             "unit_too_short_or_demo_tail_rows": int((~use).sum())}
                eligible = eval_eligible
                score = pred["scores"]
                order = np.argsort(-score, axis=1, kind="stable")[:, :5]
                recall = []
                for i in range(len(pick)):
                    nbest = max(1, int(np.ceil(.01 * eligible[i].sum())))
                    oracle = np.argsort(np.where(eligible[i], target_error[i], np.inf), kind="stable")[:nbest]
                    recall.append(bool(np.intersect1d(order[i], oracle).size))
                rho = []
                for i in range(len(pick)):
                    valid_i = eligible[i]
                    if valid_i.sum() > 2:
                        val = spearmanr(score[i, valid_i], target_error[i, valid_i]).statistic
                        if np.isfinite(val): rho.append(float(-val))
                summary[condition] = {
                    "n": len(pick),
                    "top1_type_error": lq._episode_bootstrap(type_err, query_task, query_global_ep,
                                                               seed=int(seed) + 100 + len(condition), n_boot=5000),
                    "frame0_top1_type_error": float(type_err[frame0].mean()),
                    "top1_action_first5_gtdef2": lq._episode_bootstrap(pred["action_first5"], query_task, query_global_ep,
                                                                        seed=int(seed) + 200 + len(condition), n_boot=5000),
                    "draft_active7_mse_windows": action_segments,
                    "recall_at_5_best_type_1pct": float(np.mean(recall)),
                    "spearman_score_vs_type_error": float(np.median(rho)) if rho else float("nan"),
                    "frame0_n": int(frame0.sum()),
                    "task_match_diagnostic": float(np.mean(np.asarray(pool["task"])[pick] == query_task)),
                }
                np.savez_compressed(OUT / f"pred_{kind}_{condition}_seed{seed}.npz", pick=pick,
                                    type_error=type_err, action_first5=pred["action_first5"],
                                    task=query_task, episode=query_global_ep, frame=query_frame)
            result["seeds"].setdefault(str(seed), {})[kind] = {"checkpoint": ckpath.name,
                                                               "best_epoch": best[2], "dev_type_error": best[0],
                                                               "teacher_tau": tau, "curve": log,
                                                               "conditions": summary}
            _append("evaluation_model_complete", seed=int(seed), model=kind, best_epoch=best[2])
            del net, optimizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_json = OUT / "offline_results.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    return result


def _load_expert_cpu_init(overlay=None):
    """Initialize/restore on CPU, then move one model to the shared GPU.

    This avoids holding randomly initialized parameters and restored parameters
    simultaneously in the constrained JAX GPU allocator. It respects the
    caller's XLA memory fraction; it does not raise the fraction to fit.
    """
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.models.model import restore_params
    from openpi.shared.nnx_utils import PathRegex
    core_dir = HERE.parent / "pi05_memory"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    import core
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    with jax.default_device(cpu):
        model, cfg = core.make_model(action_horizon=50, discrete_state_input=True, robocasa=True)
        params = restore_params(lq.expert.CHECKPOINT / "params", restore_type=np.ndarray)
        state = nnx.state(model)
        expected = state.to_pure_dict()
        exp_leaves, exp_tree = jax.tree.flatten(expected)
        got_leaves, got_tree = jax.tree.flatten(params)
        if exp_tree != got_tree or any(a.shape != b.shape for a, b in zip(exp_leaves, got_leaves, strict=True)):
            raise ValueError("RoboCasa pi05 checkpoint tree mismatch")
        state.replace_by_pure_dict(jax.tree.map(jnp.asarray, params))
        nnx.update(model, state)
        if overlay is not None:
            filt = nnx.All(nnx.Param, PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
            selected = nnx.state(model, filt)
            from flax import serialization
            loaded = serialization.from_bytes(selected.to_pure_dict(), Path(overlay).read_bytes())
            selected.replace_by_pure_dict(loaded); nnx.update(model, selected)
    # Transfer the restored, non-random model weights as the only GPU model.
    state = nnx.state(model)
    pure = state.to_pure_dict()
    moved = jax.tree.map(lambda x: jax.device_put(x, gpu), pure)
    state.replace_by_pure_dict(moved); nnx.update(model, state)
    return model, cfg


def valid_window_mask(remaining_steps, unit_length, window_end):
    """Rows where both the demonstrated suffix and selected unit have a full window."""
    remaining_steps = np.asarray(remaining_steps)
    unit_length = np.asarray(unit_length)
    if remaining_steps.shape != unit_length.shape:
        raise ValueError("remaining_steps and unit_length must have equal shapes")
    return (remaining_steps >= int(window_end)) & (unit_length >= int(window_end))


def summarize_saved_windows(*, output_path=None, threads=8):
    """Recompute selected-unit MSE with real suffix/unit coverage masks.

    Saved first-five errors are checked against this reconstruction. The later
    windows exclude padded demonstration tails and units that have ended.
    """
    import torch
    import serve_memonly_policy as memserve
    if output_path is None:
        output_path = OUT / "window_metrics.json"
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.set_num_threads(int(threads))
    pool = dict(np.load(CTX / "cache/pool.npz", allow_pickle=False))
    ev = np.load(GT / "per_query.npz", allow_pickle=False)
    feat = np.asarray(np.load(GT / "query_features.npz", allow_pickle=False)["prefix"], np.float32)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    global_to_local = {(int(d["global_episode"]), int(d["task_id"])): int(d["episode"])
                       for d in manifest["demos"]}
    states, targets, remaining, cache = [], [], [], {}
    for t, ge, frame in zip(ev["task"], ev["episode"], ev["frame"], strict=True):
        key = (int(t), global_to_local[(int(ge), int(t))])
        if key not in cache:
            cache[key] = ex._load_cached_episode(SOURCE, *key)[:2]
        acts, sts = cache[key]; frame = int(frame)
        states.append(ex.state_storage_to_policy(sts[frame]))
        a = ex.action_storage_to_policy(acts[frame:min(frame + 15, len(acts))])
        remaining.append(len(acts) - frame)
        if len(a) < 15:
            a = np.concatenate((a, np.repeat(a[-1:], 15 - len(a), axis=0)))
        targets.append(a)
    states, targets = np.asarray(states, np.float32), np.asarray(targets, np.float32)
    cnet_cls, mnet_cls = memserve._network_classes()[1:]
    cck = torch.load(CTX / "context_model.pt", map_location="cpu", weights_only=False)
    cnet = cnet_cls(int(cck["input_dim"])); cnet.load_state_dict(cck["state"]); cnet.eval()
    mck = torch.load(ROOT / "key-lite-01/magnitude_v2/magnitude_head.pt", map_location="cpu", weights_only=False)
    mnet = mnet_cls(int(mck["input_dim"])); mnet.load_state_dict(mck["state"]); mnet.eval()
    qctx = np.concatenate(((feat - np.asarray(cck["norm"]["feature_mean"], np.float32)) /
                           np.asarray(cck["norm"]["feature_std"], np.float32),
                           (states - np.asarray(cck["norm"]["state_mean"], np.float32)) /
                           np.asarray(cck["norm"]["state_std"], np.float32)), axis=1).astype(np.float32)
    with torch.no_grad():
        grip = np.where(cnet.grip(torch.as_tensor(qctx))[:, :5].numpy() >= 0, 1., -1.).astype(np.float32)
    unit_meta, _ = memserve.unit_metadata(pool)
    mean_ck, std_ck = np.asarray(mck["feature_mean"], np.float32), np.asarray(mck["feature_std"], np.float32)
    action_stats = json.loads((lq.expert.CHECKPOINT / "assets/norm_stats.json").read_text())["norm_stats"]["actions"]
    action_mean, action_std = np.asarray(action_stats["mean"], np.float32), np.asarray(action_stats["std"], np.float32)
    flat, offsets = np.asarray(pool["actions_flat"], np.float32), np.asarray(pool["action_offsets"], np.int64)
    lengths = np.asarray(pool["length"], np.int32)
    result = {"n_queries": len(states), "coverage_rule": "remaining demonstration steps >= window end AND candidate unit length >= window end; stored first-five gt-def2 error asserted", "conditions": {}}
    for seed in SEEDS:
        for model in ("context", "split"):
            for cond in ("correct", "zero", "replace"):
                pred = np.load(OUT / f"pred_{model}_{cond}_seed{seed}.npz", allow_pickle=False)
                pick = np.asarray(pred["pick"], np.int64); unit_len = lengths[pick]
                draft = np.zeros((len(pick), 15, 7), np.float32)
                for i, unit in enumerate(pick):
                    raw = ex.action_storage_to_policy(flat[offsets[unit]:offsets[unit + 1]]).astype(np.float32)
                    x = np.concatenate((qctx[i], unit_meta[unit])).astype(np.float32)
                    x = (x - mean_ck) / std_ck
                    with torch.no_grad():
                        scale = float(torch.exp(mnet(torch.as_tensor(x[None])).reshape(())).clamp(.25, 4.).item())
                    raw[:, :6] *= scale
                    raw[:min(5, len(raw)), 6] = grip[i, :min(5, len(raw))]
                    n = min(15, len(raw)); draft[i, :n] = raw[:n, :7]
                truth_norm = (targets[:, :, :7] - action_mean[None, None, :7]) / (action_std[None, None, :7] + 1e-6)
                draft_norm = (draft - action_mean[None, None, :7]) / (action_std[None, None, :7] + 1e-6)
                cond_result = {}
                for name, lo, hi in (("first5", 0, 5), ("steps5_9", 5, 10), ("steps10_14", 10, 15)):
                    keep = valid_window_mask(remaining, unit_len, hi)
                    vals = np.mean((draft_norm[keep, lo:hi] - truth_norm[keep, lo:hi]) ** 2, axis=(1, 2))
                    if name == "first5" and not np.allclose(vals, pred["action_first5"][keep], rtol=2e-3, atol=2e-5):
                        raise AssertionError("recomputed first-five metric does not match saved gt-def2 values")
                    cond_result[name] = {**lq._episode_bootstrap(vals, np.asarray(ev["task"])[keep],
                                                                  np.asarray(ev["episode"])[keep],
                                                                  seed=int(seed) + len(name), n_boot=5000),
                                         "n_rows": int(keep.sum()),
                                         "excluded": int((~keep).sum())}
                result["conditions"][f"{seed}/{model}/{cond}"] = cond_result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def denoise_selected(*, batch_size=4, threads=8):
    """Bridge split-key top-1 and type-oracle single-unit drafts on gt-def2 rows.

    The retrieved/oracle unit is expanded to 50 commands by holding its final
    command after its stored interval ends. This isolates the effect of the
    selected type; it does not simulate a second retrieval at unit boundaries.
    """
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import module_jit
    if not (OUT / "offline_results.json").is_file():
        raise FileNotFoundError("run the retrieval training/evaluation stage first")
    if (OUT / "denoise_results.json").exists():
        raise FileExistsError(f"refusing to overwrite {(OUT / 'denoise_results.json')}")
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.15, 0)
    pool = dict(np.load(CTX / "cache/pool.npz", allow_pickle=False))
    ev = np.load(GT / "per_query.npz", allow_pickle=False)
    gt_features = np.load(GT / "query_features.npz", allow_pickle=False)
    rows = np.load(CTX / "cache/rows.npz", allow_pickle=False)
    # The first seed is fixed in advance for the bridge extension; the 3-seed
    # retrieval results remain the primary comparison.
    pred = np.load(OUT / "pred_split_correct_seed4101.npz", allow_pickle=False)
    top1 = np.asarray(pred["pick"], np.int64)
    oracle = np.argmin(np.where(ev["eligible"], ev["type_error"], np.inf), axis=1).astype(np.int64)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    demo_info = {(int(d["task_id"]), int(d["episode"])): d for d in manifest["demos"]}
    global_to_local = {(int(d["global_episode"]), int(d["task_id"])): int(d["episode"]) for d in manifest["demos"]}
    # Load frozen magnitude and grip heads used by the gt-def2 draft constructor.
    memserve = __import__("serve_memonly_policy")
    _Rank, ContextNet, MagnitudeNet = memserve._network_classes()
    cck = torch.load(CTX / "context_model.pt", map_location="cpu", weights_only=False)
    cnet = ContextNet(int(cck["input_dim"])); cnet.load_state_dict(cck["state"]); cnet.eval()
    mck = torch.load(ROOT / "key-lite-01/magnitude_v2/magnitude_head.pt", map_location="cpu", weights_only=False)
    mag = MagnitudeNet(int(mck["input_dim"])); mag.load_state_dict(mck["state"]); mag.eval()
    # Reuse the cache's exact context norm and query states for head inputs.
    qstate = []
    local_ids = []
    episode_cache = {}
    for t, ge, f in zip(ev["task"], ev["episode"], ev["frame"], strict=True):
        le = global_to_local[(int(ge), int(t))]; local_ids.append(le)
        key = (int(t), le)
        if key not in episode_cache: episode_cache[key] = ex._load_cached_episode(SOURCE, *key)[:2]
        qstate.append(ex.state_storage_to_policy(episode_cache[key][1][int(f)]))
    qstate = np.asarray(qstate, np.float32)
    eval_prefix = np.asarray(gt_features["prefix"], np.float32)
    c_norm = cck["norm"]
    qctx = np.concatenate(((eval_prefix - np.asarray(c_norm["feature_mean"], np.float32)) /
                           np.asarray(c_norm["feature_std"], np.float32),
                           (qstate - np.asarray(c_norm["state_mean"], np.float32)) /
                           np.asarray(c_norm["state_std"], np.float32)), axis=1).astype(np.float32)
    with torch.no_grad():
        grip = np.where(cnet.grip(torch.as_tensor(qctx))[:, :5].numpy() >= 0, 1., -1.).astype(np.float32)
    unit_meta, _ = memserve.unit_metadata(pool)
    scales = np.ones((2, len(top1)), np.float32)
    selected_ids = np.stack((top1, oracle))
    for method in range(2):
        for i, unit_id in enumerate(selected_ids[method]):
            x = np.concatenate((qctx[i], unit_meta[int(unit_id)]), axis=0)
            x = (x - np.asarray(mck["feature_mean"], np.float32)) / np.asarray(mck["feature_std"], np.float32)
            with torch.no_grad(): scales[method, i] = float(torch.exp(mag(torch.as_tensor(x[None])).reshape(()))
                                                                .clamp(.25, 4.).item())

    stats = normalize.load(lq.expert.CHECKPOINT / "assets")["actions"]
    mean = np.asarray(stats.mean, np.float32); std = np.asarray(stats.std, np.float32)
    drafts = np.zeros((2, len(top1), 50, 32), np.float32)
    offsets = np.asarray(pool["action_offsets"], np.int64); flat = np.asarray(pool["actions_flat"], np.float32)
    for method in range(2):
        for i, unit_id in enumerate(selected_ids[method]):
            raw = ex.action_storage_to_policy(flat[offsets[unit_id]:offsets[unit_id + 1]]).astype(np.float32)
            raw[:, :6] *= scales[method, i]
            raw[:min(5, len(raw)), 6] = grip[i, :min(5, len(raw))]
            if len(raw) >= 50: raw = raw[:50]
            elif len(raw) < 50: raw = np.concatenate((raw, np.repeat(raw[-1:], 50 - len(raw), axis=0)), axis=0)
            padded = np.zeros((50, 32), np.float32); padded[:, :12] = raw
            drafts[method, i] = (padded - mean[:32]) / (std[:32] + 1e-6)

    records = []
    obs_cpu_device = jax.devices("cpu")[0]
    for i, (t, ge, frame) in enumerate(zip(ev["task"], ev["episode"], ev["frame"], strict=True)):
        le = int(local_ids[i]); info = demo_info[(int(t), le)]
        with jax.default_device(obs_cpu_device):
            obs, target, _raw_target, _state = lq.expert._build_obs_and_target(
                info, int(frame), SOURCE, lq.expert.CHECKPOINT)
        # `_build_obs_and_target` returns device arrays. Keep the complete
        # query inventory in host RAM instead of holding image tensors on the
        # accelerator while checkpoint parameters are initialized.
        obs_host = jax.tree.map(np.asarray, jax.device_get(obs))
        records.append((obs_host, np.asarray(jax.device_get(target), np.float32)))
        if (i + 1) % 200 == 0 or i + 1 == len(ev["task"]):
            _append("denoise_observations_cached", rows=i + 1, total=len(ev["task"]))
    result = {"n": len(records), "seed_for_split_top1": 4101,
              "draft_rule": "one selected variable-length unit; repeat its last command through 50 steps; v2 magnitude + Context grip; no boundary re-retrieval",
              "noise_rule": "fold_in(key(4101), gt-def2 row index), shared across draft methods and t0",
              "conditions": {}}
    actions = {"split_top1": drafts[0], "type_oracle": drafts[1]}
    # Cache the ordinary π0.5 reference in host RAM first. Never keep the
    # base and Expert parameter trees resident on the shared GPU together.
    base, _cfg = _load_expert_cpu_init()
    normal_fn = module_jit(base.sample_actions, static_argnames=("num_steps",))
    all_normal = []
    for start in range(0, len(records), int(batch_size)):
        end = min(start + int(batch_size), len(records)); items = records[start:end]
        obs_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *[x[0] for x in items])
        noises = np.concatenate([np.asarray(jax.random.normal(
            jax.random.fold_in(jax.random.key(4101), int(row)), (1, 50, 32)))
            for row in range(start, end)], axis=0)
        all_normal.append(np.asarray(normal_fn(jax.random.key(1), obs_batch, num_steps=10,
                                                noise=jnp.asarray(noises))))
    normal_host = np.concatenate(all_normal)
    del base, normal_fn, all_normal
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    jax.clear_caches()
    import gc
    gc.collect()
    for name, step, t0, ckpt in (("expert03_step3000_t0.2", 3000, .2,
                                  ROOT / "expert-03/train/expert/expert_003000.msgpack"),
                                 ("expert03_step500_t0.4", 500, .4,
                                  ROOT / "expert-03/train/expert/expert_000500.msgpack")):
        trained, _cfg = _load_expert_cpu_init(ckpt)
        bridge_fn = module_jit(trained.sample_bridge, static_argnames=("num_steps",))
        per_method = {}
        for method, drafts_arr in actions.items():
            all_pred = []; all_target = []
            for start in range(0, len(records), int(batch_size)):
                end = min(start + int(batch_size), len(records)); items = records[start:end]
                obs_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *[x[0] for x in items])
                target_batch = np.stack([x[1] for x in items]).astype(np.float32)
                noises = np.concatenate([np.asarray(jax.random.normal(
                    jax.random.fold_in(jax.random.key(4101), int(row)), (1, 50, 32)))
                    for row in range(start, end)], axis=0)
                d = jnp.asarray(drafts_arr[start:end])
                t0s = jnp.full((end - start,), float(t0), jnp.float32)
                pred_b = np.asarray(bridge_fn(jax.random.key(0), obs_batch, d, t0s,
                                               num_steps=10, noise=jnp.asarray(noises)))
                all_pred.append(pred_b); all_target.append(target_batch)
            p = np.concatenate(all_pred); n = normal_host; y = np.concatenate(all_target)
            # Active7 MSE on normalized model actions; the 12 action channels
            # include base/mode after the EEF+gripper prefix.
            per_method[method] = {
                "bridge_first5_active7_mse": float(np.mean((p[:, :5, :7] - y[:, :5, :7]) ** 2)),
                "bridge_horizon50_active7_mse": float(np.mean((p[:, :, :7] - y[:, :, :7]) ** 2)),
                "output_minus_draft_first5_active7_mse": float(np.mean((p[:, :5, :7] - drafts_arr[:, :5, :7]) ** 2)),
                "output_minus_draft_horizon50_active7_mse": float(np.mean((p[:, :, :7] - drafts_arr[:, :, :7]) ** 2)),
                "output_minus_pi05_first5_active7_mse": float(np.mean((p[:, :5, :7] - n[:, :5, :7]) ** 2)),
                "output_minus_pi05_horizon50_active7_mse": float(np.mean((p[:, :, :7] - n[:, :, :7]) ** 2)),
            }
        result["conditions"][name] = per_method
        del trained, bridge_fn
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        jax.clear_caches(); gc.collect()
        _append("denoise_condition_complete", condition=name, step=step, t0=t0, n=len(records))
    path = OUT / "denoise_results.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    parser.add_argument("--test-contract", action="store_true")
    parser.add_argument("--denoise", action="store_true", help="run Expert-03 bridge on saved split top-1/oracle units")
    parser.add_argument("--summarize-windows", action="store_true", help="recompute valid continuation windows from saved picks")
    parser.add_argument("--window-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.test_contract:
        import torch
        torch.set_num_threads(args.threads)
        test_score_contract()
        print("split-key score contract: ok")
        return
    if args.denoise:
        denoise_selected(batch_size=args.batch_size, threads=args.threads)
        return
    if args.summarize_windows:
        summarize_saved_windows(output_path=args.window_output, threads=args.threads)
        return
    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size,
                       device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()

