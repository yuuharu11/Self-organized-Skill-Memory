#!/usr/bin/env python3
"""Scene-invariant retrieval with latent query and action tokens (INV2).

This module intentionally leaves ``scene_invariance.py`` untouched.  It reuses
its fixed data split, candidate pool, directional type errors and reporting
metrics.  Evaluation labels are only opened by ``evaluate`` after selection is
frozen.  VLM features are copied from read-only memmaps into fp16 host memory
once per process, then mini-batched to the device.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in os.sys.path:
    os.sys.path.insert(0, str(HERE))
import scene_invariance as step1

OUT = step1.OUT / "step2" / "attempt-fix1"
SEED = 4101
K = 5
D = 256
TOKEN_DIM = 2048
CONDITIONS = ("V2-A", "V2-A-L1", "V2-B", "V2-B-L1", "V2-B-pool")
# One-change-at-a-time ablation. R0 is implemented by the untouched step1
# _make_model/_train_once path; these entries define the cumulative V2 stages.
ABLATION_STAGES = {
    "R0": {"implementation": "step1_legacy", "kind": "A", "k": 1, "key_mode": "legacy", "query_pool": False, "l0": "legacy", "l1": False, "lr": 3e-4},
    "R1": {"implementation": "latent", "kind": "A", "k": 1, "key_mode": "legacy", "query_pool": True, "no_history_cross": True, "l0": "legacy", "l1": False, "lr": 3e-4},
    "R2": {"implementation": "latent", "kind": "B", "k": 1, "key_mode": "legacy", "query_pool": True, "no_history_cross": True, "l0": "legacy", "l1": False, "lr": 3e-4},
    "R3": {"implementation": "latent", "kind": "B", "k": 1, "key_mode": "legacy", "query_pool": True, "no_history_cross": True, "l0": "legacy", "l1": True, "lr": 3e-4},
    "R4": {"implementation": "latent", "kind": "B", "k": 5, "key_mode": "split", "query_pool": False, "no_history_cross": True, "l0": "legacy", "l1": True, "lr": 3e-4},
    "R5": {"implementation": "latent", "kind": "B", "k": 5, "key_mode": "split", "query_pool": False, "no_history_cross": True, "l0": "dual", "l1": True, "lr": 3e-4},
}


def _atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    tmp.replace(p)


def _json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return str(x)


bootstrap_mean = step1.bootstrap_mean  # single episode-clustered CI implementation


def _log(out, **row):
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    line = json.dumps(rec, ensure_ascii=False, default=_json_default)
    print(line, flush=True)
    with (Path(out) / "progress.jsonl").open("a") as f: f.write(line + "\n")


def split_action_tokens(actions, k=K):
    """Split a [B,10,7] sequence into K equal temporal tokens.

    K=5 gives exactly two frames per token. K=4 uses four ordered contiguous
    intervals and linearly resamples each interval to two endpoint frames, so
    the shared 14->256 projection is unchanged.
    """
    a = np.asarray(actions)
    if a.ndim != 3 or a.shape[1:] != (10, 7):
        raise ValueError(f"expected [B,10,7], got {a.shape}")
    k=int(k)
    if k not in (4,5): raise ValueError("INV2 supports K in {4,5}")
    bounds=np.rint(np.linspace(0,10,k+1)).astype(int)
    chunks=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        part=a[:,lo:hi]
        if len(part[0])==2: out=part
        else:
            src=np.linspace(0.,1.,part.shape[1]);dst=np.array([0.,1.])
            out=np.stack([np.stack([np.interp(dst,src,row[:,j]) for j in range(7)],axis=-1) for row in part],axis=0)
        chunks.append(out.reshape(len(a),14))
    return np.stack(chunks,axis=1)


def aligned_token_score(q, m, scale=1.0):
    """s=scale/K sum_k <q_k,m_k>, accepting normalized or raw token vectors."""
    import torch
    import torch.nn.functional as F
    if q.ndim != 3 or m.ndim != 3 or q.shape[1:] != m.shape[1:]:
        raise ValueError(f"aligned token shape mismatch: {tuple(q.shape)} / {tuple(m.shape)}")
    qn = F.normalize(q, dim=-1); mn = F.normalize(m, dim=-1)
    return (qn[:, None] * mn[None]).sum(-1).mean(-1) * scale


def maxsim_token_score(q, m, scale=1.0):
    """Alternative token set score: sum over query tokens of best key-token dot."""
    import torch.nn.functional as F
    import torch
    if q.ndim != 3 or m.ndim != 3 or q.shape[-1] != m.shape[-1]:
        raise ValueError("MaxSim requires [B,K,D] token sets")
    sim = torch.einsum("bkd,mjd->bmkj", F.normalize(q, dim=-1), F.normalize(m, dim=-1))
    return sim.amax(-1).sum(-1) * scale


def scene_stratified_indices(scene, eligible_ids, batch_size=256, scenes_per_batch=8, per_scene=32, seed=SEED,
                             different_type_pairs=None, excluded_scenes=()):
    """Yield unique-scene batches and optionally force one within-scene negative."""
    scene = np.asarray(scene); ids = np.asarray(eligible_ids, np.int64)
    if excluded_scenes:
        ids=ids[~np.isin(scene[ids],np.asarray(tuple(excluded_scenes),dtype=scene.dtype))]
    if not len(ids): raise ValueError("no eligible rows after held-scene exclusion")
    rng = np.random.default_rng(seed)
    by_scene = {s: ids[scene[ids] == s] for s in sorted(set(scene[ids].tolist()))}
    by_scene = {s: q for s, q in by_scene.items() if len(q)}
    names = np.asarray(list(by_scene), dtype=object)
    nsc = min(int(scenes_per_batch), len(names), int(batch_size))
    if nsc < 2: raise ValueError("scene-stratified L1 needs at least two distinct scenes")
    count_each = min(int(per_scene), max(1, int(batch_size) // nsc))
    steps = max(1, int(np.ceil(len(ids) / (nsc * count_each))))
    for _ in range(steps):
        if different_type_pairs:
            forced=[s for s in names if s in different_type_pairs]
            if not forced: raise ValueError("no scene has an available different-type pair")
            forced_scene=rng.choice(np.asarray(forced,object))
            others=names[names!=forced_scene]
            chosen=np.concatenate([[forced_scene],rng.choice(others,nsc-1,replace=False)])
        else:
            chosen = rng.choice(names, nsc, replace=False)
        batch = []
        pair_scene = next((s for s in chosen if different_type_pairs and s in different_type_pairs), None)
        for s in chosen:
            pool = by_scene[s]
            if s == pair_scene:
                pair=np.asarray(different_type_pairs[s],np.int64)
                if not np.all(np.isin(pair,pool)): raise AssertionError("hard-negative pair is outside its scene")
                rest=pool[~np.isin(pool,pair)]
                extra=rng.choice(rest,min(max(0,count_each-2),len(rest)),replace=False).tolist()
                batch.extend(pair.tolist()+extra)
            else:
                batch.extend(rng.choice(pool, min(count_each, len(pool)), replace=False).tolist())
        yield np.asarray(batch, np.int64)


def _build_model(kind, *, k=K, token_dim=TOKEN_DIM, d=D, pooled=False,
                 key_mode="split", query_pool=False, no_history_cross=False):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class CrossLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
            self.norm1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            self.norm2 = nn.LayerNorm(d)

        def forward(self, q, kv, key_padding_mask=None):
            a, w = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                             need_weights=True, average_attn_weights=False)
            q = self.norm1(q + a)
            q = self.norm2(q + self.ffn(q))
            return q, w

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind; self.k = int(k); self.pooled = bool(pooled)
            self.key_mode = key_mode; self.query_pool = bool(query_pool)
            self.no_history_cross = bool(no_history_cross)
            self.z0 = nn.Parameter(torch.randn(1, k, d) * .02)
            self.vlm_in = nn.Sequential(nn.Linear(token_dim, d), nn.LayerNorm(d))
            self.vlm_cross = CrossLayer()
            self.hist_in = nn.Linear(7, d)
            self.hist_pos = nn.Parameter(torch.randn(1, 10, d) * .01)
            self.state_in = nn.Sequential(nn.Linear(8, d), nn.LayerNorm(d))
            self.ctx_cross = CrossLayer()
            self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.legacy_context = nn.Sequential(nn.Linear(78, 512), nn.GELU(), nn.Linear(512, d))
            self.query_fuse = nn.Sequential(nn.Linear(2*d, 512), nn.GELU(), nn.Linear(512, d))
            self.query_add = nn.Linear(d, d)
            self.key_legacy = nn.Sequential(nn.Linear(70, 512), nn.GELU(), nn.Linear(512, d))
            self.key_frame = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
            self.key_pos = nn.Parameter(torch.randn(1, k, d) * .01)
            self.key_attn = nn.TransformerEncoderLayer(d, 8, 4*d, batch_first=True,
                                                       activation="gelu", norm_first=True)
            self.log_scale = nn.Parameter(torch.tensor(np.log(10.), dtype=torch.float32))

        def history_tokens(self, history):
            """Order-aware history token sequence (one learned position per step)."""
            return self.hist_in(history.float()) + self.hist_pos

        def encode_query(self, tokens, history, state, *, token_groups=None, train_mask=False):
            b = history.shape[0]
            pad = None
            if self.kind == "A":
                if token_groups is not None:
                    keep = token_groups == 1
                    tokens = tokens[:, keep]
            elif self.kind == "B":
                if token_groups is not None and train_mask and self.training:
                    # Drop the whole third-person view for each selected sample.
                    base = token_groups == 0
                    if base.any():
                        drop = torch.rand(b, device=tokens.device) < .30
                        pad = torch.zeros((b, tokens.shape[1]), device=tokens.device, dtype=torch.bool)
                        pad[:, base] = drop[:, None]
                    else: pad = None
                else: pad = None
            else: raise ValueError(kind)
            q = self.z0.expand(b, -1, -1)
            v = self.vlm_in(tokens.float())
            q, attn_vlm = self.vlm_cross(q, v, pad)
            if self.no_history_cross:
                attn_ctx = None
            else:
                hist_tokens = self.history_tokens(history)
                ctx = torch.cat([hist_tokens, self.state_in(state.float())[:, None]], 1)
                q, attn_ctx = self.ctx_cross(q, ctx)
            # Keep the old 78->512->256 global context path. R1 concatenates
            # it with the mean of the K latent reads before producing one key.
            global_ctx = self.legacy_context(torch.cat([history.float().flatten(1), state.float()], -1))
            if self.query_pool:
                query_tokens = F.normalize(q, dim=-1)
                z = self.query_fuse(torch.cat([q.mean(1), global_ctx], -1))[:, None]
            else:
                z = self.out(q + self.query_add(global_ctx)[:, None])
                query_tokens = F.normalize(z, dim=-1)
            z = F.normalize(z, dim=-1)
            attn = {"vlm": attn_vlm, "context": attn_ctx, "query_tokens": query_tokens}
            return z, attn

        def encode_keys(self, action):
            if self.key_mode == "legacy":
                return F.normalize(self.key_legacy(action.float().flatten(1)), dim=-1)[:, None]
            if self.key_mode != "split": raise ValueError(self.key_mode)
            if self.k==5:
                chunks=action.reshape(len(action),5,14)
            else:
                bounds=np.rint(np.linspace(0,10,self.k+1)).astype(int);parts=[]
                for lo,hi in zip(bounds[:-1],bounds[1:]):
                    x=action[:,lo:hi]
                    parts.append(torch.stack([x[:,0],x[:,-1]],dim=1).reshape(len(action),14))
                chunks=torch.stack(parts,dim=1)
            z = self.key_frame(chunks) + self.key_pos
            z = self.key_attn(z)
            return F.normalize(z, dim=-1)

        def scores(self, q, m):
            if self.pooled:
                qp = F.normalize(q.mean(1), dim=-1)
                mp = F.normalize(m.mean(1), dim=-1)
                return qp @ mp.T * self.log_scale.exp()
            return (q[:, None] * m[None]).sum(-1).mean(-1) * self.log_scale.exp()

    return Encoder()


def l0_prime(scores, errors, allowed, *, mode="dual"):
    """L0′ with temperatures on teacher distributions, not scaled logits."""
    import torch
    import torch.nn.functional as F
    a = torch.as_tensor(allowed, device=scores.device, dtype=torch.bool)
    e = torch.as_tensor(errors, device=scores.device, dtype=scores.dtype)
    logits = scores.masked_fill(~a, -1e4)
    if mode == "single03":
        target = torch.softmax((-e / .03).masked_fill(~a, -1e4), -1)
        return -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    if mode != "dual": raise ValueError(mode)
    target1 = torch.softmax((-e / .01).masked_fill(~a, -1e4), -1)
    target5 = torch.softmax((-e / .05).masked_fill(~a, -1e4), -1)
    lp = F.log_softmax(logits, -1)
    ce = -(target1 * lp).sum(-1).mean()
    kl = (target5 * (torch.log(target5.clamp_min(1e-12)) - lp)).sum(-1).mean()
    return ce + kl


def legacy_l0(scores, errors, allowed, *, tau=.01):
    """Original soft-target cross entropy: temperature is target-only."""
    import torch
    import torch.nn.functional as F
    a=torch.as_tensor(allowed,device=scores.device,dtype=torch.bool)
    e=torch.as_tensor(errors,device=scores.device,dtype=scores.dtype)
    target=torch.softmax((-e/float(tau)).masked_fill(~a,-1e4),-1)
    logits=scores.masked_fill(~a,-1e4)
    return -(target*F.log_softmax(logits,-1)).sum(-1).mean()


def relational_pair_masks(action_type, scene, tau_pos):
    """Return symmetric distances and (positive, valid-denominator) pair masks."""
    import torch
    if action_type.ndim != 3 or len(scene) != len(action_type):
        raise ValueError("relational batch metadata mismatch")
    # action_type carries the fixed six physical axes used by INV1's teacher.
    x = action_type.float().flatten(1)
    den = x.square().sum(-1).clamp_min(1e-10)
    dot = (x @ x.T).clamp_min(0)
    d_ij = ((den[:, None] - dot.square() / den[None, :]) / x.shape[1]).clamp_min(0)
    d_ji = ((den[None, :] - dot.square() / den[:, None]) / x.shape[1]).clamp_min(0)
    dist = .5 * (d_ij + d_ji)
    same_scene = torch.as_tensor(np.asarray(scene)[:, None] == np.asarray(scene)[None, :], device=action_type.device)
    off = ~torch.eye(len(action_type), device=action_type.device, dtype=torch.bool)
    positive = (dist <= float(tau_pos)) & ~same_scene & off
    valid = ~((dist <= float(tau_pos)) & same_scene | ~off)
    return dist, positive, valid


def relational_soft_infonce(q, action_key, action_type, scene, *, tau_pos, score_scale=1.0):
    """CLASS-style weighted multi-positive objective with ignored false negatives.

    Positives are cross-scene pairs whose symmetric directional type error is
    <= tau_pos. Same-scene close pairs are ignored (not in the denominator);
    all other non-self pairs remain valid negatives. Positive weights are the
    exp(-distance/tau_pos) analogue of CLASS's 1-empirical-CDF(DTW).
    """
    import torch
    import torch.nn.functional as F
    b = len(q)
    if len(action_type) != b or len(scene) != b: raise ValueError("relational batch metadata mismatch")
    dist,positive,valid=relational_pair_masks(action_type,scene,tau_pos)
    if not positive.any(): return q.sum() * 0., 0
    keys = action_key
    logits = (q[:, None] * keys[None]).sum(-1).mean(-1) * score_scale
    logits = logits.masked_fill(~valid, -1e4)
    logp = F.log_softmax(logits, -1)
    weight = torch.exp(-dist / max(float(tau_pos), 1e-6)) * positive
    mass = weight.sum(-1)
    rows = mass > 0
    target = weight / mass[:, None].clamp_min(1e-12)
    loss = -(target * logp).sum(-1)[rows].mean()
    return loss, int(positive.sum().item())


def _load_step1(out=step1.OUT):
    p = Path(out) / "fit_rows.npz"
    if not p.exists(): raise FileNotFoundError(f"run scene_invariance.prepare first: {p}")
    with np.load(p, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    return data


def prepare(out=OUT, *, source_out=step1.OUT, force=False):
    """Write training-only row metadata; never reads LIBERO-90 target actions."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dest = out / "prepared.json"
    if dest.exists() and not force:
        meta=json.loads(dest.read_text())
        manifest=out/"run_manifest.json"
        if not manifest.exists(): _write_run_manifest(manifest)
        return meta
    data = _load_step1(source_out)
    # Check that step1's fit/selection are task-disjoint and include no unseen rows.
    fit = np.asarray(data["fit"], bool); sel = np.asarray(data["select"], bool)
    if np.any(fit & sel) or not fit.any() or not sel.any(): raise AssertionError("invalid fit/selection split")
    task = np.asarray(data["task"])
    if set(task[fit].tolist()) & set(task[sel].tolist()): raise AssertionError("held-out task leakage")
    if len(data["pool_shape"]) != data["error"].shape[1]: raise AssertionError("pool/error mismatch")
    meta = {"status": "prepared", "source_fit_rows": str(Path(source_out) / "fit_rows.npz"),
            "fit_rows": int(fit.sum()), "selection_rows": int(sel.sum()), "pool_size": int(len(data["pool_shape"])),
            "eval_labels_read": False, "seed": SEED, "k_grid": [4,5], "token_cache": str(step1.TOKENS),
            "token_dtype": "fp16 host cache loaded once per run; selected valid VLM groups only",
            "conditions": list(CONDITIONS)}
    _atomic_json(dest, meta)
    manifest=out/"run_manifest.json"
    if not manifest.exists(): _write_run_manifest(manifest)
    return meta


def _write_run_manifest(path):
    import subprocess
    try: revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE.parents[5],text=True).strip()
    except Exception: revision="unknown"
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _atomic_json(path,{"status":"prepared","base_revision":"2fc3e6e11fad0473d9bfe4e8d4e377b69b32686c",
        "worktree_head":revision,"module_sha256":digest,"seed":SEED,"epochs_per_fit":3,
        "max_tuning_candidates":10,"K_grid":[4,5],"conditions":list(CONDITIONS),
        "gpu_fraction_cap":.22,"eval_labels_used_for_tuning":False,
        "source_step1_api":str(step1.OUT/"STEP1_API.md")})


def load_token_cache(data, *, kind, row_ids=None):
    """Load requested train-row tokens once to host fp16, retaining group ids."""
    if row_ids is None: row_ids = np.arange(len(data["row_id"]), dtype=np.int64)
    row_ids = np.asarray(row_ids, np.int64)
    vals = np.load(step1.TOKENS, mmap_mode="r")
    groups = np.load(step1.GROUPS, mmap_mode="r")
    g = np.asarray(groups[0])
    keep = np.flatnonzero(g >= 0)
    if kind == "A": keep = keep[g[keep] == 1]
    tok = np.asarray(vals[np.asarray(data["row_id"])[row_ids]][:, keep], dtype=np.float16)
    return tok, g[keep].astype(np.int8)


def load_training_actions(data, out=OUT):
    """Map each fit row to its closest allowed, exact C_vm_none template.

    Pool shapes are variable-segment M6/DCT reconstructions selected from
    stitched episodes and then phase-cropped/resampled. They are not a direct
    per-row action transform, so the reviewed nearest-template fallback is used.
    """
    path=Path(out)/"training_pool_template_keys_v2.npy"
    if path.exists():
        actions=np.load(path,mmap_mode="r")
    else:
        actions,_=pool_template_key_actions(data["error"],data["allowed"],data["pool_shape"])
        np.save(path,actions,allow_pickle=False)
    if actions.shape!=(len(data["row_id"]),10,7):raise AssertionError("training pool-key shape mismatch")
    return actions


def pool_template_key_actions(error, allowed, pool_shape):
    """Select each row's minimum-type-error allowed pool key, returning IDs."""
    e=np.asarray(error,np.float32);a=np.asarray(allowed,bool);p=np.asarray(pool_shape,np.float32)
    if e.ndim!=2 or a.shape!=e.shape or p.shape!=(e.shape[1],10,7):
        raise ValueError(f"pool mapping shapes mismatch: error={e.shape}, allowed={a.shape}, pool={p.shape}")
    if np.any(~a.any(1)):raise ValueError("some fit rows have no allowed pool template")
    ranked=np.where(a,e,np.inf);idx=np.argmin(ranked,axis=1)
    if np.any(~np.isfinite(ranked[np.arange(len(idx)),idx])):raise AssertionError("selected a forbidden template")
    return p[idx].copy(),idx.astype(np.int64)


def _query_tokens(batch_tokens, kind):
    if kind == "A":
        # Caller may have loaded B tokens for cache sharing; select wrist here.
        raise ValueError("A token cache must already contain wrist tokens")
    return batch_tokens


def _hard_negative_pairs(data, fit_ids, *, seed, tau_pos, type_actions=None):
    scenes=np.asarray(data["scene"])
    target=np.asarray(data["target"] if type_actions is None else type_actions,np.float32)
    rng=np.random.default_rng(seed);pairs={}
    for s in sorted(set(scenes[fit_ids].tolist())):
        local=fit_ids[scenes[fit_ids]==s]
        if len(local)<2:continue
        sample=rng.choice(local,min(96,len(local)),replace=False)
        err=step1._pair_err(target[sample]);np.fill_diagonal(err,-1.)
        if np.max(err)>tau_pos:
            i,j=np.unravel_index(np.argmax(err),err.shape)
            pairs[s]=(int(sample[i]),int(sample[j]))
    return pairs


def _tau_pos_for_key_actions(data, type_actions, row_ids, *, seed=SEED, per_task=48):
    """Fold-local positive threshold measured in the same scale-only key space."""
    ids=np.asarray(row_ids,np.int64);tasks=np.asarray(data["task"]);scenes=np.asarray(data["scene"])
    rng=np.random.default_rng(seed);picked=[]
    for task in sorted(set(tasks[ids].tolist())):
        local=ids[tasks[ids]==task]
        picked.extend(rng.choice(local,min(int(per_task),len(local)),replace=False).tolist())
    picked=np.asarray(picked,np.int64)
    if len(picked)<2:return .02
    err=step1._pair_err(np.asarray(type_actions,np.float32)[picked]);t=tasks[picked];s=scenes[picked]
    eligible=(t[:,None]!=t[None,:])&(s[:,None]!=s[None,:])
    vals=err[eligible]
    return float(np.quantile(vals,.25)) if len(vals) else .02


def _train_config(data, tokens, groups, key_actions, *, kind, l1, pooled, l0_mode, l1_weight, k=K,
                  epochs=3, device="cuda", batch_size=64, seed=SEED, save_path=None,
                  key_mode="split", query_pool=False, no_history_cross=False, learning_rate=3e-4,
                  legacy_r0=False):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("V2_GPU_FRACTION", ".22")), device=0)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    if legacy_r0:
        if kind!="A" or int(k)!=1 or l1 or key_mode!="legacy" or l0_mode!="legacy":
            raise ValueError("legacy_r0 requires the exact R0 architecture/key/loss")
        net=step1._make_model("A",1,TOKEN_DIM).to(device)
    else:
        net = _build_model(kind,k=k,pooled=pooled,key_mode=key_mode,query_pool=query_pool,
                           no_history_cross=no_history_cross).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    fit_ids = np.flatnonzero(np.asarray(data["fit"], bool))
    task = np.asarray(data["task"]); scene = np.asarray(data["scene"])
    target = torch.as_tensor(data["target"], dtype=torch.float32, device=device)
    pool_action = torch.as_tensor(data["pool_shape"], dtype=torch.float32, device=device)
    history = torch.as_tensor(data["history"], dtype=torch.float32, device=device)
    state = torch.as_tensor(data["state"], dtype=torch.float32, device=device)
    err_np = np.asarray(data["error"], np.float32)
    allowed_np = np.asarray(data["allowed"], bool)
    type_actions=np.asarray(key_actions,np.float32)[:,:,:6]
    tau_pos = _tau_pos_for_key_actions(data,type_actions,fit_ids,seed=seed) if l1 else .02
    hard_pairs = _hard_negative_pairs(data,fit_ids,seed=seed,tau_pos=tau_pos,type_actions=type_actions) if l1 else None
    rng = np.random.default_rng(seed)
    if l1:
        n_scene=min(8,len(set(scene[fit_ids].tolist())))
        l1_batch=max(1,n_scene*32)
        n_epoch_batches=int(np.ceil(len(fit_ids)/l1_batch))
    else:
        n_epoch_batches = int(np.ceil(len(fit_ids) / batch_size))
    logs=[]
    for ep in range(int(epochs)):
        net.train(); losses=[]; pos_total=0
        if l1:
            batches = (next(scene_stratified_indices(scene,fit_ids,batch_size=256,scenes_per_batch=8,
                per_scene=32,seed=int(rng.integers(2**31-1)),different_type_pairs=hard_pairs))
                for _ in range(n_epoch_batches))
        else:
            order = rng.permutation(fit_ids)
            batches = (order[i:i+batch_size] for i in range(0, len(order), batch_size))
        for ids in batches:
            if len(ids)<2: continue
            bi = torch.as_tensor(ids, dtype=torch.long, device=device)
            tok = torch.as_tensor(tokens[ids], device=device)
            if legacy_r0:
                qscore=legacy_r0_logits(net,history[bi],state[bi],tok,pool_action)
                loss=legacy_l0(qscore,err_np[ids],allowed_np[ids])
            else:
                q, _ = net.encode_query(tok, history[bi], state[bi], token_groups=torch.as_tensor(groups, device=device), train_mask=(kind=="B"))
                pool_keys = net.encode_keys(pool_action)
                qscore = net.scores(q, pool_keys)
                loss = legacy_l0(qscore,err_np[ids],allowed_np[ids]) if l0_mode=="legacy" else l0_prime(qscore, err_np[ids], allowed_np[ids], mode=l0_mode)
            npos=0
            if l1:
                # Same batch's observed action keys, with the same two-frame partition.
                kb = net.encode_keys(torch.as_tensor(key_actions[ids],dtype=torch.float32,device=device))
                l1_types=torch.as_tensor(type_actions[ids],dtype=torch.float32,device=device)
                rel, npos = relational_soft_infonce(q, kb, l1_types, scene[ids], tau_pos=tau_pos,
                                                       score_scale=net.log_scale.exp())
                loss = loss + float(l1_weight) * rel
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach())); pos_total += npos
        rec={"epoch":ep+1,"loss":float(np.mean(losses)),"batches":len(losses),"positive_pairs":pos_total}
        logs.append(rec)
    if save_path:
        p=Path(save_path);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists(): raise FileExistsError(p)
        tmp=p.with_suffix(p.suffix+".tmp")
        torch.save({"state_dict":net.cpu().state_dict(),"kind":kind,"pooled":pooled,"l1":l1,
                    "l0_mode":l0_mode,"l1_weight":l1_weight,"seed":seed,"epochs":epochs,"k":k,
                    "key_mode":key_mode,"query_pool":query_pool,"learning_rate":learning_rate,
                    "no_history_cross":no_history_cross,"legacy_r0":legacy_r0,
                    "tau_pos":tau_pos,"token_dim":TOKEN_DIM},tmp);os.replace(tmp,p)
    return net, logs


def _cfg(name,kind,k,*,l0="dual",l1=False,l1w=0.):
    return {"name":name,"kind":kind,"k":int(k),"l1":bool(l1),"pooled":False,"l0":l0,"l1w":float(l1w)}


def config_grid():
    """First-stage K comparison: 2 inputs × K{4,5}, three-epoch dual L0′."""
    return [_cfg(f"V2-{kind}",kind,k) for kind in ("A","B") for k in (4,5)]


def _old_checkpoint(out,cfg):
    """Reuse completed pilot checkpoints only when architecture/objective match exactly."""
    if cfg["k"]!=5:return None
    if cfg["name"] in ("V2-A","V2-B") and not cfg["l1"]:
        old=Path(out)/"tuning"/f"V2_{cfg['kind']}_{cfg['l0']}_l10.pt"
    elif cfg["l1"]:
        old=Path(out)/"tuning"/(cfg["name"].replace("-","_")+f"_{cfg['l0']}_l1{cfg['l1w']:g}_scenehard_v2.pt")
    else:old=None
    return old if old is not None and old.exists() else None


def _run_tune_candidate(out,data,tok,groups,key_actions,selected,cfg,*,device,batch_size,index,total):
    import torch
    old=_old_checkpoint(out,cfg)
    label=f"{cfg['name'].replace('-','_')}_k{cfg['k']}_{cfg['l0']}_l1{cfg['l1w']:g}"
    if cfg["l1"]:label+="_scenehard_v2"
    cp=old or (Path(out)/"tuning_v2"/f"{label}.pt")
    if cp.exists():
        ck=torch.load(cp,map_location="cpu",weights_only=False)
        net=_build_model(ck["kind"],k=ck.get("k",5),pooled=ck["pooled"]);net.load_state_dict(ck["state_dict"]);losses=[]
    else:
        net,losses=_train_config(data,tok,groups,key_actions,kind=cfg["kind"],k=cfg["k"],l1=cfg["l1"],pooled=False,
            l0_mode=cfg["l0"],l1_weight=cfg["l1w"],epochs=3,device=device,
            batch_size=256 if cfg["l1"] else batch_size,seed=SEED,save_path=cp)
    sc,rank,_=_score_ids(net,data,tok,groups,selected,kind=cfg["kind"],device=device)
    er=np.asarray(data["error"])[selected];ix=rank[:,0];err=float(er[np.arange(len(ix)),ix].mean())
    rec={**cfg,"checkpoint":str(cp),"selection_top1_type_error":err,
         "l1_sampler":"forced_same_scene_different_type_pair" if cfg["l1"] else None,
         "epochs":3,"seed":SEED,"train_loss":None if not losses else losses[-1]["loss"]}
    _log(out,stage="tune_v2",index=index,total=total,**rec)
    del net
    return rec


def _candidate_scores(net, q, pool_actions, *, chunk=256):
    import torch
    out=[]
    for st in range(0,len(pool_actions),chunk):
        aa = torch.as_tensor(pool_actions[st:st+chunk],dtype=torch.float32,device=q.device)
        mk = net.encode_keys(aa)
        out.append(net.scores(q,mk).detach().cpu().numpy())
    return np.concatenate(out,axis=1)


def _score_ids(net, data, tokens, groups, ids, *, kind, batch=32, device="cuda", return_token_cosine=False):
    import torch
    net.eval().to(device); scores=[]; attn=[]; token_cos=[]
    pool=np.asarray(data["pool_shape"],np.float32)
    with torch.no_grad():
        pool_action=torch.as_tensor(pool,dtype=torch.float32,device=device)
        pool_keys=torch.cat([net.encode_keys(pool_action[i:i+256]) for i in range(0,len(pool_action),256)],0)
        for st in range(0,len(ids),batch):
            ix=np.asarray(ids[st:st+batch],np.int64);bi=torch.as_tensor(ix,device=device)
            q,a=net.encode_query(torch.as_tensor(tokens[ix],device=device),
                torch.as_tensor(data["history"][ix],device=device),torch.as_tensor(data["state"][ix],device=device),
                token_groups=torch.as_tensor(groups,device=device))
            scores.append(net.scores(q,pool_keys).cpu().numpy())
            if kind in ("A","B"):
                # Attention averaged over heads but kept for every latent token.
                attn.append(a["vlm"].mean(1).cpu().numpy())
                qt=a.get("query_tokens",q)
                if qt.shape[1]>1:
                    gram=torch.einsum("bkd,bjd->bkj",qt,qt)
                    mask=~torch.eye(qt.shape[1],device=qt.device,dtype=torch.bool)
                    token_cos.extend(gram[:,mask].mean(1).cpu().numpy().tolist())
    s=np.concatenate(scores);allow=np.asarray(data["allowed"])[ids];s[~allow]=-np.inf
    base=(s, np.argsort(-s,axis=1,kind="stable"), (np.concatenate(attn) if attn else None))
    if return_token_cosine:
        return (*base, float(np.mean(token_cos)) if token_cos else None)
    return base


def ablation_config(stage, *, k=None, l1_weight=None):
    """Return one immutable R0–R5 switch configuration."""
    if stage not in ABLATION_STAGES: raise ValueError(f"unknown ablation stage {stage}")
    cfg=dict(ABLATION_STAGES[stage])
    if stage in ("R1", "R2", "R3") and k not in (None,1):
        raise ValueError(f"{stage} uses one query/key token; K search is removed until R4")
    if stage in ("R4", "R5") and k not in (None,5):
        raise ValueError(f"{stage} fixes K=5")
    if k is not None: cfg["k"]=int(k)
    if stage=="R5" and l1_weight is not None:
        if float(l1_weight) not in (.1,.3): raise ValueError("R5 weight must be .1 or .3")
        cfg["l1_weight"]=float(l1_weight)
    else: cfg["l1_weight"]=float(l1_weight or .1)
    cfg["stage"]=stage
    return cfg


def stage_candidate_configs(retained, stage):
    """Apply only this stage's patch to the currently retained configuration."""
    if stage not in ("R1","R2","R3","R4","R5"):
        raise ValueError(f"not an incremental stage: {stage}")
    retained=dict(retained)
    scaffold=False
    if retained.get("implementation")=="step1_legacy":
        # Later stages require the one-slot latent encoder introduced in R1.
        # If R1 was rejected, use only that structural prerequisite and record it;
        # the stage under test remains the sole additional change.
        retained=ablation_config("R1")
        scaffold=True
    cfg=retained.copy()
    if stage=="R1":
        cfg=ablation_config("R1")
    elif stage=="R2":
        cfg["kind"]="B"  # full token groups + train-only third-person mask
    elif stage=="R3":
        cfg["l1"]=True
    elif stage=="R4":
        cfg.update(k=5,key_mode="split",query_pool=False)
    elif stage=="R5":
        cfg["l0"]="dual"
    cfg["stage"]=stage
    cfg["scaffold_from_R1"]=scaffold
    return cfg


def stage_acceptance(candidate, incumbent):
    """Mechanical preregistered rule: <=2% error and strictly higher rho."""
    ce=float(candidate["top1_type_error"]);ie=float(incumbent["top1_type_error"])
    cr=float(candidate["spearman_positive"]);ir=float(incumbent["spearman_positive"])
    err_ok=ce<=ie*1.02
    rho_ok=cr>ir
    return {"accepted":bool(err_ok and rho_ok),"error_ok":bool(err_ok),"spearman_ok":bool(rho_ok),
            "candidate_error":ce,"incumbent_error":ie,"error_limit":ie*1.02,
            "candidate_spearman":cr,"incumbent_spearman":ir}


def multiseed_acceptance(candidate, baseline):
    """Noise-aware gate: error <= 1.02x R0 and rho gain > 2x R0 sample SD."""
    err=float(candidate["top1_type_error_mean"]);base=float(baseline["top1_type_error_mean"])
    rho=float(candidate["spearman_positive_mean"]);base_rho=float(baseline["spearman_positive_mean"])
    rho_sd=float(baseline["spearman_positive_sd"])
    err_ok=err<=base*1.02
    rho_gain=rho-base_rho;rho_limit=2.*rho_sd
    return {"accepted":bool(err_ok and rho_gain>rho_limit),"error_ok":bool(err_ok),
        "spearman_ok":bool(rho_gain>rho_limit),"error_limit":base*1.02,
        "spearman_gain":rho_gain,"required_spearman_gain_gt":rho_limit}


def aggregate_seed_metrics(rows):
    """Mean and sample SD for per-seed heldout metrics."""
    if len(rows)<2:raise ValueError("need at least two seeds for noise estimate")
    out={"seeds":[int(r["seed"]) for r in rows],"n_seeds":len(rows)}
    for key in ("top1_type_error","recovery","spearman_positive","mean_query_token_cosine"):
        vals=np.asarray([r[key] for r in rows if r.get(key) is not None],np.float64)
        out[f"{key}_mean"]=float(vals.mean()) if len(vals) else None
        out[f"{key}_sd"]=float(vals.std(ddof=1)) if len(vals)>1 else (0. if len(vals)==1 else None)
    return out


def legacy_r0_logits(model, history, state, tokens, pool_actions):
    """Old A-L0 forward/key path used by the new optimizer loop control."""
    import torch
    import torch.nn.functional as F
    tokens=tokens.to(dtype=next(model.parameters()).dtype)
    mask=torch.ones(tokens.shape[:2],device=tokens.device,dtype=torch.bool)
    q,_=model(history,state,tokens,mask)
    keys=F.normalize(model.key(pool_actions.flatten(1)),dim=-1)
    return F.normalize(q,dim=-1)@keys.T*model.log_scale.exp()


def choose_stage_candidate(candidates, incumbent, stage):
    """Annotate candidates with gates and choose only from passing candidates."""
    checked=[]
    for candidate in candidates:
        item=dict(candidate);item["rule"]=stage_acceptance(item,incumbent)
        item["accepted"]=item["rule"]["accepted"];checked.append(item)
    eligible=[x for x in checked if x["accepted"]]
    if not eligible:return None,checked
    if stage=="R3":
        chosen=max(eligible,key=lambda x:(x["spearman_positive"],-x["top1_type_error"]))
    else:
        chosen=eligible[0]
    return chosen,checked


def history_variant(data, ids, variant, *, seed=SEED):
    """Return a shallow data view with selection histories preserved, zeroed, or cross-episode swapped."""
    if variant not in ("true", "zero", "other_episode"):
        raise ValueError(f"unknown history variant: {variant}")
    out=dict(data)
    h=np.asarray(data["history"],np.float32).copy()
    ids=np.asarray(ids,np.int64)
    if variant=="zero":
        h[ids]=0
    elif variant=="other_episode":
        episodes=np.asarray(data["episode"])
        candidates=np.arange(len(h),dtype=np.int64)
        rng=np.random.default_rng(seed)
        for i in ids:
            other=candidates[episodes[candidates]!=episodes[i]]
            if not len(other):raise ValueError("cannot replace history: no other episode")
            h[i]=data["history"][rng.choice(other)]
    out["history"]=h
    return out


def _metric_record(value, cfg, data, tokens, groups, ids, *, device):
    """Small JSON-safe heldout summary for one fitted model."""
    met=value["metrics"]
    return {"config":cfg,"top1_type_error":value["top1_type_error"],
        "recovery":float(met["recovery"]),"spearman_positive":float(met["spearman_mean"]),
        "mean_query_token_cosine":value.get("mean_query_token_cosine")}


def _evaluate_history_variants(model,cfg,data,tokens,groups,ids,*,device,seed,base_record=None):
    rows={}
    if base_record is not None:
        rows["true"]={k:base_record[k] for k in ("top1_type_error","recovery","spearman_positive")}
        names=("zero","other_episode")
    else:
        names=("true","zero","other_episode")
    for offset,name in enumerate(names):
        d=history_variant(data,ids,name,seed=seed+offset)
        v=_score_ablation_candidate(model,cfg,d,tokens,groups,ids,device=device)
        rows[name]=_metric_record(v,cfg,d,tokens,groups,ids,device=device)
    return rows


def _fit_for_diagnostics(out,data,tokens,groups,key_actions,cfg,*,device,seed,checkpoint):
    """Use the frozen 4101 selection fit once; fit other seeds under distinct paths."""
    return _fit_ablation_candidate(out,data,tokens,groups,key_actions,cfg,device=device,
                                   save_path=checkpoint,seed=seed)


def run_noise_history_diagnostics(out=OUT, *, source_out=step1.OUT, device="cuda"):
    """Three-seed stage noise/history diagnostics; deliberately does not read Eval or run group-CV."""
    out=Path(out)
    selection_path=out/"ablation_selection.json"
    if not selection_path.exists():raise FileNotFoundError("run tune first; ablation_selection.json missing")
    selection=json.loads(selection_path.read_text())
    if selection.get("status")!="selection_frozen":raise ValueError("initial selection is not frozen")
    dest=out/"ablation_selection_3seed.json"
    if dest.exists():raise FileExistsError(dest)
    data=step1._step1_prepare_data(source_out)
    ids=np.flatnonzero(np.asarray(data["select"],bool));fit_ids=np.flatnonzero(np.asarray(data["fit"],bool))
    if np.any(np.asarray(data["task"])[ids] == -1):raise AssertionError("invalid selection task ids")
    tokens,groups=load_token_cache(data,kind="B",row_ids=np.arange(len(data["row_id"])))
    key_actions=load_training_actions(data,out)
    seeds=(4101,4102,4103)
    # Preserve the exact one-seed configs chosen during the original staged run.
    stage_cfgs={"R0":[selection["stages"][0]["candidates"][0]["config"]]}
    stage_cps={"R0":[selection["stages"][0]["candidates"][0].get("checkpoint")]}
    for stage_rec in selection["stages"][1:]:
        stage_cfgs[stage_rec["stage"]]=[x["config"] for x in stage_rec["candidates"]]
        stage_cps[stage_rec["stage"]]=[x.get("checkpoint") for x in stage_rec["candidates"]]
    partial_path=out/"noise_history_partial.json"
    partial=json.loads(partial_path.read_text()) if partial_path.exists() else {"stages":{}}
    stage_rows={};history_rows={};seed_models={}
    for stage in ("R0","R1","R2","R3","R4","R5"):
        stage_rows[stage]=[]
        candidate_configs=stage_cfgs[stage]
        for ci,cfg0 in enumerate(candidate_configs):
            cfg=dict(cfg0)
            stage_id=stage+(f"_c{ci}" if len(candidate_configs)>1 else "")
            if stage_id in partial["stages"]:
                saved=partial["stages"][stage_id]
                stage_rows[stage_id]=saved["rows"]
                history_rows[stage_id]=saved["history"]
                stage_rows[stage_id+"_aggregate"]=aggregate_seed_metrics(stage_rows[stage_id])
                seed_models[stage_id]={str(r["seed"]):r["checkpoint"] for r in stage_rows[stage_id]}
                continue
            stage_rows[stage_id]=[]
            history_rows[stage_id]={}
            for seed in seeds:
                init_cp=stage_cps[stage][ci]
                original=Path(init_cp) if seed==4101 and init_cp else None
                cp=original if original and original.exists() else out/"noise_history_models"/f"{stage_id}_seed{seed}.pt"
                model,logs=_fit_for_diagnostics(out,data,tokens,groups,key_actions,cfg,device=device,
                                                seed=seed,checkpoint=cp)
                base=_score_ablation_candidate(model,cfg,data,tokens,groups,ids,device=device)
                record=_metric_record(base,cfg,data,tokens,groups,ids,device=device)
                record.update(seed=seed,checkpoint=str(cp),train_log=logs)
                stage_rows[stage_id].append(record)
                _log(out,diagnostic="noise_history_fit",stage_name=stage_id,seed=seed,
                     checkpoint=str(cp),top1_type_error=record["top1_type_error"],
                     recovery=record["recovery"],spearman_positive=record["spearman_positive"])
                if stage=="R0" and stage_id=="R0":
                    # Stage R0 itself is also exposed under the compact key.
                    pass
                history_rows[stage_id][str(seed)]=_evaluate_history_variants(
                    model,cfg,data,tokens,groups,ids,device=device,seed=seed*10,base_record=record)
                seed_models.setdefault(stage_id,{})[str(seed)]=str(cp)
                if hasattr(model,"cpu"):model.cpu()
                del model
            stage_rows[stage_id+"_aggregate"]=aggregate_seed_metrics(stage_rows[stage_id])
            partial["stages"][stage_id]={"rows":stage_rows[stage_id],"history":history_rows[stage_id]}
            _atomic_json(partial_path,partial)
        if len(candidate_configs)==1:
            stage_rows[stage]=stage_rows.pop(stage)
            stage_rows[stage+"_aggregate"]=aggregate_seed_metrics(stage_rows[stage])
            history_rows[stage]=history_rows.pop(stage)

    r0=stage_rows["R0_aggregate"]
    gates={};retained=selection["stages"][0]["candidates"][0]["config"]
    retained_name="R0"
    for stage in ("R1","R2","R3","R4","R5"):
        keys=[stage+(f"_c{i}" if len(stage_cfgs[stage])>1 else "") for i in range(len(stage_cfgs[stage]))]
        checked=[]
        for name in keys:
            agg=stage_rows[name+"_aggregate"]
            gate=multiseed_acceptance(agg,r0)
            item={"candidate":name,"config":stage_rows[name][0]["config"],"aggregate":agg,"rule":gate}
            checked.append(item)
        eligible=[x for x in checked if x["rule"]["accepted"]]
        chosen=max(eligible,key=lambda x:(x["aggregate"]["spearman_positive_mean"],-x["aggregate"]["top1_type_error_mean"])) if eligible else None
        gates[stage]={"candidates":checked,"accepted":chosen is not None,
                      "chosen":None if chosen is None else chosen["candidate"]}
        if chosen:
            retained=dict(chosen["config"]);retained_name=chosen["candidate"]

    # State-only arm: the selected architecture is trained with all history
    # inputs clamped to zero, making H/history constant and therefore uninformative.
    nohist_cfg=dict(retained)
    nohist_rows=[];nohist_history={}
    zero_data=dict(data);zero_data["history"]=np.zeros_like(data["history"],dtype=np.float32)
    for seed in seeds:
        cp=out/"noise_history_models"/f"state_only_seed{seed}.pt"
        model,logs=_fit_for_diagnostics(out,zero_data,tokens,groups,key_actions,nohist_cfg,
            device=device,seed=seed,checkpoint=cp)
        vals=_score_ablation_candidate(model,nohist_cfg,zero_data,tokens,groups,ids,device=device)
        row=_metric_record(vals,nohist_cfg,zero_data,tokens,groups,ids,device=device)
        row.update(seed=seed,checkpoint=str(cp),train_log=logs,history_input="zero for fit and evaluation")
        nohist_rows.append(row)
        same={k:row[k] for k in ("top1_type_error","recovery","spearman_positive")}
        nohist_history[str(seed)]={v:dict(same) for v in ("true","zero","other_episode")}
        if hasattr(model,"cpu"):model.cpu()
        del model

    # Control for the optimizer/minibatch loop: old A-L0 model and objective,
    # but fitted through this module's new loop, for the same three seeds.
    r0new_cfg=dict(ablation_config("R0"),implementation="step1_legacy_new_loop",stage="R0-new")
    tokens_a,groups_a=load_token_cache(data,kind="A",row_ids=np.arange(len(data["row_id"])))
    r0new_rows=[];r0new_history={}
    for seed in seeds:
        cp=out/"noise_history_models"/f"R0_new_seed{seed}.pt"
        model,logs=_fit_for_diagnostics(out,data,tokens_a,groups_a,key_actions,r0new_cfg,
            device=device,seed=seed,checkpoint=cp)
        val=_score_ablation_candidate(model,r0new_cfg,data,tokens_a,groups_a,ids,device=device)
        row=_metric_record(val,r0new_cfg,data,tokens_a,groups_a,ids,device=device)
        row.update(seed=seed,checkpoint=str(cp),train_log=logs)
        r0new_rows.append(row)
        r0new_history[str(seed)]=_evaluate_history_variants(model,r0new_cfg,data,tokens_a,groups_a,ids,
            device=device,seed=seed*10,base_record=row)
        if hasattr(model,"cpu"):model.cpu()
        del model

    # Historical P5 baseline, trained on the same fit tasks, measured under all history variants.
    p5_rows=[];p5_history={}
    for seed in seeds:
        cp=out/"noise_history_models"/f"P5_seed{seed}.pt"
        if cp.exists():
            model,_=step1._torch_load(cp,device)
            logs=[]
        else:
            model,logs=step1._train_once(out,"P5-L0",seed=seed,tau=.01,k=1,layer="final",epochs=3,
                save_path=cp,device=device,batch_size=64,report_each=False,data_override=data)
        p5_history[str(seed)]={}
        for offset,name in enumerate(("true","zero","other_episode")):
            d=history_variant(data,ids,name,seed=seed*10+offset)
            scores,rank,_=step1._score_model(model,"P5-L0",d,ids,layer="final",device=device,batch=32)
            met=step1._step1_metrics(scores,d,ids);err=np.asarray(d["error"])[ids];ix=rank[:,0]
            row={"top1_type_error":float(err[np.arange(len(ix)),ix].mean()),
                "recovery":float(met["recovery"]),"spearman_positive":float(met["spearman_mean"])}
            p5_history[str(seed)][name]=row
        p5_rows.append({"seed":seed,"checkpoint":str(cp),"train_log":logs})
        _log(out,diagnostic="noise_history_P5",seed=seed,checkpoint=str(cp))
        if hasattr(model,"cpu"):model.cpu()
        del model

    p5_agg={}
    for variant in ("true","zero","other_episode"):
        p5_agg[variant]={}
        for metric in ("top1_type_error","recovery","spearman_positive"):
            v=[p5_history[str(s)][variant][metric] for s in seeds]
            p5_agg[variant][metric+"_mean"]=float(np.mean(v))
            p5_agg[variant][metric+"_sd"]=float(np.std(v,ddof=1))

    result={"status":"complete","source_selection":str(selection_path),"source_selection_seed":4101,
        "seeds":list(seeds),"epochs":3,"selection_rows":int(len(ids)),"fit_rows":int(len(fit_ids)),
        "eval_rows_read":False,"group_cv_run":False,"retained_config":retained,
        "retained_candidate":retained_name,"acceptance_vs_R0":gates,"stages":stage_rows,
        "history_variants":history_rows,"state_only_arm":{"aggregate":aggregate_seed_metrics(nohist_rows),
            "per_seed":nohist_rows,"history_variants":nohist_history},
        "R0_new_loop":{"aggregate":aggregate_seed_metrics(r0new_rows),"per_seed":r0new_rows,
            "history_variants":r0new_history},
        "P5":{"per_seed":p5_rows,"history_variants":p5_history,"aggregate":p5_agg},
        "model_paths":seed_models}
    _atomic_json(dest,result)
    _atomic_json(out/"history_dependence.json",{"status":"complete","variants":["true","zero","other_episode"],
        "stages":history_rows,"state_only":result["state_only_arm"],"R0_new_loop":r0new_history,"P5":p5_agg})
    write_noise_history_report(out,result)
    return result


def write_noise_history_report(out,result):
    """Write concise tables from the frozen multi-seed diagnostics."""
    out=Path(out);s=result["stages"]
    lines=["# INV2 noise and history diagnostics","",
        "Evaluation uses only the LIBERO-40 heldout-task selection rows. No LIBERO-90 Eval labels or group-CV were read/run.",
        "The initial staged configs are frozen from ablation_selection.json. Multi-seed gates compare every candidate with R0: top-1 error mean ≤ 1.02×R0 mean and Spearman gain > 2×R0 sample SD.","",
        "## Three-seed stage metrics (mean ± sample SD)","",
        "| stage/candidate | top-1 type error | recovery | Spearman (+ is good) | K-token cosine | gate vs R0 |",
        "|---|---:|---:|---:|---:|---|"]
    names=["R0","R1","R2","R3_c0","R3_c1","R4","R5"]
    for name in names:
        if name not in s:continue
        a=s[name+"_aggregate"]
        stage=name.split("_")[0]
        gate=("baseline" if name=="R0" else "pass" if result["acceptance_vs_R0"].get(stage,{}).get("accepted") else "reject")
        def fmt(metric):
            m=a.get(metric+"_mean");sd=a.get(metric+"_sd")
            return "—" if m is None else f"{m:.4f} ± {sd:.4f}"
        lines.append(f"| {name} | {fmt('top1_type_error')} | {fmt('recovery')} | {fmt('spearman_positive')} | {fmt('mean_query_token_cosine')} | {gate} |")
    lines += ["",f"Retained multi-seed configuration: `{result['retained_candidate']}` — `{result['retained_config']}`.",
        "R3 candidates are both reported; the higher-Spearman passing candidate is preferred.","",
        "## History dependence (three-seed mean ± sample SD)","",
        "| stage/candidate | history | recovery | Spearman | top-1 type error |","|---|---|---:|---:|---:|"]
    for name,seed_map in result["history_variants"].items():
        variants=[seed_map[str(seed)] for seed in result["seeds"]]
        for v in ("true","zero","other_episode"):
            vals=[r[v] for r in variants]
            fields=[]
            for metric in ("recovery","spearman_positive","top1_type_error"):
                x=np.asarray([r[metric] for r in vals],float)
                fields.append(f"{x.mean():.4f} ± {x.std(ddof=1):.4f}")
            lines.append(f"| {name} | {v} | {fields[0]} | {fields[1]} | {fields[2]} |")
    for title,key in (("State-only history-clamped arm","state_only_arm"),("R0-new loop","R0_new_loop")):
        agg=result[key]["aggregate"]
        lines += ["",f"## {title}","", "| top-1 type error | recovery | Spearman |","|---:|---:|---:|"]
        lines.append("| "+" | ".join(f"{agg[k+'_mean']:.4f} ± {agg[k+'_sd']:.4f}" for k in ("top1_type_error","recovery","spearman_positive"))+" |")
    lines += ["","## P5 baseline under history variants","","| history | top-1 type error | recovery | Spearman |","|---|---:|---:|---:|"]
    for v,r in result["P5"]["aggregate"].items():
        lines.append(f"| {v} | {r['top1_type_error_mean']:.4f} ± {r['top1_type_error_sd']:.4f} | {r['recovery_mean']:.4f} ± {r['recovery_sd']:.4f} | {r['spearman_positive_mean']:.4f} ± {r['spearman_positive_sd']:.4f} |")
    lines += ["","The state-only arm is fitted and evaluated with history clamped to zero; its history variants are not informative and are recorded as such in JSON. All reported comparisons are offline heldout-task retrieval diagnostics, not LIBERO-90 Eval results."]
    (out/"noise_history_report.md").write_text("\n".join(lines)+"\n")


def _load_ablation_checkpoint(path, cfg, device):
    import torch
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if cfg["implementation"] in ("step1_legacy","step1_legacy_new_loop"):
        model=step1._make_model("A",1,TOKEN_DIM)
    else:
        model=_build_model(cfg["kind"],k=cfg["k"],key_mode=cfg["key_mode"],
                           query_pool=cfg["query_pool"],pooled=False,
                           no_history_cross=cfg.get("no_history_cross",False))
    model.load_state_dict(ck["state_dict"])
    return model.to(device)


def _fit_ablation_candidate(out, data, tokens, groups, key_actions, cfg, *, device, save_path, seed=SEED):
    """Fit one candidate with the historical optimizer and exact stage flags."""
    import torch
    p=Path(save_path)
    if p.exists(): return _load_ablation_checkpoint(p,cfg,device),[]
    if cfg["implementation"]=="step1_legacy":
        step1._preload_final_tokens(data,"A",device)
        model,logs=step1._train_once(out,"A-L0",seed=seed,tau=.01,k=1,layer="final",epochs=3,
            save_path=p,device=device,batch_size=64,report_each=False,data_override=data)
        return model,logs
    if cfg["implementation"]=="step1_legacy_new_loop":
        return _train_config(data,tokens,groups,key_actions,kind="A",l1=False,pooled=False,
            l0_mode="legacy",l1_weight=0.,k=1,epochs=3,device=device,batch_size=64,
            seed=seed,save_path=p,key_mode="legacy",query_pool=False,
            learning_rate=3e-4,legacy_r0=True)
    return _train_config(data,tokens,groups,key_actions,kind=cfg["kind"],l1=cfg["l1"],pooled=False,
        l0_mode=cfg["l0"],l1_weight=cfg["l1_weight"],k=cfg["k"],key_mode=cfg["key_mode"],
        query_pool=cfg["query_pool"],no_history_cross=cfg.get("no_history_cross",False),
        learning_rate=cfg["lr"],epochs=3,device=device,
        batch_size=256 if cfg["l1"] else 64,seed=seed,save_path=p)


def _score_ablation_candidate(model, cfg, data, tokens, groups, ids, *, device):
    if cfg["implementation"] in ("step1_legacy","step1_legacy_new_loop"):
        scores,rank,_=step1._score_model(model,"A-L0",data,ids,layer="final",device=device,batch=32)
        token_cos=None
    else:
        scores,rank,_,token_cos=_score_ids(model,data,tokens,groups,ids,kind=cfg["kind"],
            batch=32,device=device,return_token_cosine=True)
    met=step1._step1_metrics(scores,data,ids)
    ix=rank[:,0];err=np.asarray(data["error"])[ids]
    return {"metrics":met,"top1_type_error":float(err[np.arange(len(ix)),ix].mean()),
            "mean_query_token_cosine":token_cos,"scores":scores}


def tune_stages(out=OUT, *, source_out=step1.OUT, device="cuda"):
    """Holdout-only R0–R5 ablation with mechanical accept/reject patches."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/"ablation_selection.json"
    if dest.exists(): return json.loads(dest.read_text())
    if not (step1.OUT/"STEP1_API.md").exists(): raise FileNotFoundError(step1.OUT/"STEP1_API.md")
    # Use the STEP1 API-prepared mapping, which includes the fixed group labels
    # needed by the shared retrieval metrics (the raw fit cache omits `group`).
    data=step1._step1_prepare_data(source_out); fit_sel=np.flatnonzero(np.asarray(data["fit"])|np.asarray(data["select"]))
    if not np.array_equal(fit_sel,np.arange(len(data["row_id"]))): raise AssertionError("fit/selection cache row order mismatch")
    tokens,groups=load_token_cache(data,kind="B",row_ids=fit_sel)
    key_actions=load_training_actions(data,out)
    selected=np.flatnonzero(np.asarray(data["select"],bool))
    if not len(selected): raise AssertionError("empty heldout-task selection set")
    records=[]
    # R0 is the untouched reference implementation.
    cfg0=ablation_config("R0");cp=out/"ablation_tuning/R0.pt"
    model,logs=_fit_ablation_candidate(out,data,tokens,groups,key_actions,cfg0,device=device,save_path=cp)
    val=_score_ablation_candidate(model,cfg0,data,tokens,groups,selected,device=device)
    incumbent={"config":cfg0,"top1_type_error":val["top1_type_error"],
        "recovery":val["metrics"]["recovery"],"spearman_positive":val["metrics"]["spearman_mean"],
        "mean_query_token_cosine":val["mean_query_token_cosine"],"checkpoint":str(cp)}
    records.append({"stage":"R0","candidates":[{**incumbent,"accepted":True,"train_log":logs}],
        "accepted":True,"retained_stage_after":"R0","retained_metrics_after":incumbent})
    if hasattr(model,"cpu"):model.cpu()
    del model
    retained=cfg0
    for stage in ("R1","R2","R3","R4","R5"):
        weights=(.1,.3) if stage=="R3" else (None,)
        stage_results=[]
        for weight in weights:
            cfg=stage_candidate_configs(retained,stage)
            if weight is not None:
                cfg["l1_weight"]=weight
            tag=f"{stage}_from_{retained.get('stage','R0')}_k{cfg['k']}_w{cfg['l1_weight']:g}"
            cp=out/"ablation_tuning"/(tag+".pt")
            model,logs=_fit_ablation_candidate(out,data,tokens,groups,key_actions,cfg,device=device,save_path=cp)
            val=_score_ablation_candidate(model,cfg,data,tokens,groups,selected,device=device)
            metrics={"config":cfg,"top1_type_error":val["top1_type_error"],
                "recovery":val["metrics"]["recovery"],"spearman_positive":val["metrics"]["spearman_mean"],
                "mean_query_token_cosine":val["mean_query_token_cosine"],"checkpoint":str(cp),
                "train_log":logs}
            stage_results.append(metrics);_log(out,stage="ablation_select",stage_name=stage,**metrics)
            if hasattr(model,"cpu"):model.cpu()
            del model
        # R3 explicitly chooses the highest Spearman among candidates that pass
        # both preregistered gates; other stages have one candidate.
        chosen,stage_results=choose_stage_candidate(stage_results,incumbent,stage)
        if chosen is not None:
            retained=dict(chosen["config"])
            incumbent={k:chosen[k] for k in ("config","top1_type_error","recovery","spearman_positive",
                "mean_query_token_cosine","checkpoint")}
        records.append({"stage":stage,"candidates":stage_results,"accepted":chosen is not None,
            "chosen_config":None if chosen is None else chosen["config"],
            "retained_stage_after":incumbent["config"].get("stage","R0"),
            "retained_metrics_after":incumbent})
    result={"status":"selection_frozen","seed":SEED,"epochs_per_candidate":3,
        "selection_source":"LIBERO-40 heldout tasks only","eval_targets_read":False,
        "acceptance_rule":"top1 type error <= retained * 1.02 AND Spearman strictly higher",
        "candidate_fit_count":sum(len(x["candidates"]) for x in records),"candidate_cap":10,
        "stages":records,"retained_config":incumbent["config"],
        "stage_switch":ABLATION_STAGES}
    _atomic_json(dest,result);return result


def tune(out=OUT, *, source_out=step1.OUT, device="cuda", epochs=3, batch_size=64):
    if int(epochs)!=3: raise ValueError("R0-R5 tuning uses exactly three epochs per fit")
    return tune_stages(out,source_out=source_out,device=device)

def train_selected(out=OUT, *, source_out=step1.OUT, device="cuda", batch_size=64):
    """Optional fit-only training of R0 and the heldout-selected final stage."""
    out=Path(out);selection=json.loads((out/"ablation_selection.json").read_text());data=_load_step1(source_out)
    rows=np.flatnonzero(np.asarray(data["fit"],bool))
    if not len(rows) or np.any(np.asarray(data["select"],bool)[rows]):raise AssertionError("invalid fit-only refit rows")
    tokens,groups=load_token_cache(data,kind="B")
    key_actions=load_training_actions(data,out);models={}
    for name,cfg in (("R0",ablation_config("R0")),("V2-final",selection["retained_config"])):
        path=out/"models"/(name.replace("-","_")+".pt")
        if name=="V2-final" and cfg["implementation"]=="step1_legacy":
            models[name]=models["R0"];continue
        net,logs=_fit_ablation_candidate(out,data,tokens,groups,key_actions,cfg,device=device,save_path=path)
        _log(out,stage="fit_selected",condition=name,logs=logs,checkpoint=str(path))
        if hasattr(net,"cpu"):net.cpu()
        del net;models[name]=str(path)
    _atomic_json(out/"models.json",{"status":"fit_only_models_ready","models":models,"seed":SEED,"epochs":3,
        "heldout_selection_tasks_excluded_from_fit":True})
    return models


def smoke(out=OUT, *, source_out=step1.OUT, device="cpu"):
    """Synthetic, one-mini-batch shape/loss smoke; never reads Eval labels."""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    torch.manual_seed(SEED)
    b=16; tokens=torch.randn(b,64,TOKEN_DIM);hist=torch.randn(b,10,7);state=torch.randn(b,8)
    groups=torch.tensor(np.repeat([0,1,2,2],16))
    actions=torch.randn(b,10,7);type6=torch.randn(b,10,6)
    allowed=torch.ones(b,40,dtype=torch.bool);errors=torch.rand(b,40)
    results=[]
    for name in CONDITIONS:
        kind="A" if "-A" in name or name=="V2-A" else "B"
        pooled=name.endswith("pool");net=_build_model(kind,pooled=pooled).to(device)
        q,attn=net.encode_query(tokens.to(device),hist.to(device),state.to(device),token_groups=groups.to(device),train_mask=True)
        keys=net.encode_keys(actions.to(device));scores=net.scores(q,net.encode_keys(torch.randn(40,10,7,device=device)))
        l0=l0_prime(scores,errors.to(device),allowed.to(device),mode="dual")
        l1,npos=relational_soft_infonce(q,keys,type6.to(device),np.repeat(["scene0","scene1"],8),tau_pos=1.)
        loss=l0+(0.1*l1 if "L1" in name else 0.)
        if not torch.isfinite(loss):raise AssertionError(f"nonfinite loss: {name}")
        if q.shape != (b,K,D):raise AssertionError(f"latent tokens were pooled: {q.shape}")
        results.append({"condition":name,"q_shape":list(q.shape),"score_shape":list(scores.shape),
                        "l0":float(l0.detach()),"l1":float(l1.detach()),"positive_pairs":npos})
    _atomic_json(Path(out)/"smoke.json",{"status":"smoke_complete","eval_labels_read":False,"results":results})
    return results


def _api_callbacks(out, data, tokens, groups, key_actions, *, device, batch):
    """Adapters for the shared step1 group-CV/Eval evaluator."""
    import torch
    import torch.nn.functional as F
    kq, tr, vs, sq = step1._modules()
    row_ids=np.asarray(data["row_id"],np.int64)
    if len(tokens)!=len(row_ids): raise AssertionError("token cache is not row-aligned to fit_rows")
    lookup=np.full(int(row_ids.max())+1,-1,np.int64);lookup[row_ids]=np.arange(len(row_ids))
    action_lookup=np.full(int(row_ids.max())+1,-1,np.int64);action_lookup[row_ids]=np.arange(len(row_ids))
    unseen_token_cache={}
    group_all=np.asarray(groups,np.int8)
    def token_for(rows):
        rid=np.asarray(rows["row_id"],np.int64)
        if "target_action" not in rows:
            ix=lookup[rid]
            if np.any(ix<0): raise AssertionError("query row missing from frozen LIBERO-40 feature cache")
            return tokens[ix], group_all
        # Eval features are fetched only after selection is frozen. Target
        # actions are not used in this branch; only row IDs address VLM tokens.
        key=tuple(rid.tolist())
        if key not in unseen_token_cache:
            vals=np.load(step1.UNSEEN_ROWS/"tokens.npy",mmap_mode="r")
            gr=np.load(step1.UNSEEN_ROWS/"groups.npy",mmap_mode="r")
            g=np.asarray(gr[0]);keep=np.flatnonzero(g>=0)
            unseen_token_cache[key]=(np.asarray(vals[rid][:,keep],np.float16),g[keep].astype(np.int8))
        return unseen_token_cache[key]

    def train_fn(condition, train_rows, *, seed, epochs, save_path, fold):
        if int(seed)!=SEED or int(epochs)!=3: raise ValueError("INV2 training must remain seed 4101 / 3 epochs")
        cfg=json.loads((Path(out)/"selection.json").read_text())["selected"][condition]
        path=Path(save_path)
        if condition == "V2-B-pool":
            # This is a scoring ablation, not another trained configuration.
            # The shared evaluator calls V2-B immediately before its pool control.
            source = path.with_name(path.name.replace("V2_B_pool", "V2_B"))
            if not source.exists():
                raise FileNotFoundError(f"V2-B source checkpoint required for pooling control: {source}")
            ck = torch.load(source, map_location="cpu", weights_only=False)
            model = _build_model("B", k=ck.get("k", 5), pooled=True)
            model.load_state_dict(ck["state_dict"])
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({**ck, "pooled": True, "source_condition": "V2-B"}, path)
            return model
        if path.exists():
            ck=torch.load(path,map_location="cpu",weights_only=False)
            net=_build_model(ck["kind"],k=ck.get("k",5),pooled=ck["pooled"]);net.load_state_dict(ck["state_dict"])
            return net
        rid=np.asarray(train_rows["row_id"],np.int64);ix=lookup[rid];ai=action_lookup[rid]
        if np.any(ix<0) or np.any(ai<0): raise AssertionError("fold row IDs escaped fit/selection cache")
        local={k:v for k,v in train_rows.items()}
        local["fit"]=np.ones(len(rid),bool)
        if np.any(np.asarray(local["group"]).astype(str)==fold): raise AssertionError("held-out group leaked into fold training")
        # Calibrate L1's positive threshold from this fold's fit rows only.
        local["tau_pos"]=np.asarray(step1._tau_pos_for_rows(local,np.arange(len(rid)),seed=seed),np.float32)
        net,logs=_train_config(local,tokens[ix],groups,key_actions[ai],kind=cfg["kind"],l1=cfg["l1"],
            pooled=cfg["pooled"],l0_mode=cfg["l0"],l1_weight=cfg["l1w"],epochs=3,
            device=device,batch_size=64 if not cfg["l1"] else 256,seed=seed,save_path=path)
        _log(out,stage="shared_api_train",condition=condition,fold=fold,seed=seed,epochs=3,logs=logs)
        return net

    def score_fn(model, condition, query_rows):
        kind=json.loads((Path(out)/"selection.json").read_text())["selected"][condition]["kind"]
        qtok,qgroups=token_for(query_rows)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
        if str(device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("V2_GPU_FRACTION",".22")),device=0)
        model.eval().to(device)
        allow=np.asarray(query_rows["allowed"],bool)
        pshape=np.asarray(query_rows["pool_shape"],np.float32)
        scores=[]
        with torch.no_grad():
            p=torch.as_tensor(pshape,dtype=torch.float32,device=device)
            pm=torch.cat([model.encode_keys(p[i:i+256]) for i in range(0,len(p),256)],0)
            for st in range(0,len(qtok),batch):
                ids=np.arange(st,min(st+batch,len(qtok)))
                q,_=model.encode_query(torch.as_tensor(qtok[ids],device=device),
                    torch.as_tensor(query_rows["history"][ids],device=device),
                    torch.as_tensor(query_rows["state"][ids],device=device),
                    token_groups=torch.as_tensor(qgroups,device=device))
                scores.append(model.scores(q,pm).cpu().numpy())
        scores=np.concatenate(scores);scores[~allow]=-np.inf
        extras={}
        if "target_action" in query_rows:
            rank=np.argsort(-scores,axis=1,kind="stable");chosen=rank[:,0]
            pool=query_rows["_pool"]
            pred=kq.decode_pool_actions(pool,chosen[:,None],vs,theta_source="src")[:,0,:,:6]
            truth=np.asarray(query_rows["target_action"],np.float32)
            # Same closed-loop six-axis convention as the R0 path (sum of the
            # six active-axis squared errors divided by 7), not a scalar mean.
            mse=step1._six_axis_mse_over7(pred,truth)
            hold=np.repeat(np.asarray(query_rows["history"],np.float32)[:,-1:,:6],truth.shape[1],axis=1)
            hmse=step1._six_axis_mse_over7(hold,truth);hard=np.asarray(query_rows["hard"],bool)
            episode=np.asarray(query_rows["episode_bootstrap"])
            extras["action_mse"]={"all_mean":float(mse.mean()),"hard_mean":float(mse[hard].mean()),
                "all_ci95":bootstrap_mean(mse,episode),"hard_ci95":bootstrap_mean(mse[hard],episode[hard]),
                "vs_hold_last":step1._bootstrap_diff(mse,hmse,episode),
                "vs_hold_last_hard":step1._bootstrap_diff(mse[hard],hmse[hard],episode[hard]),
                "hold_last_all":float(hmse.mean()),"hold_last_hard":float(hmse[hard].mean()),
                "hard_rows":int(hard.sum()),"rows":len(mse)}
        return {"scores":scores,"extras":extras}
    return train_fn,score_fn


def evaluate_previous_grid(out=OUT, *, source_out=step1.OUT, device="cuda", batch=16):
    """Run the API-published four group folds and single Eval pass."""
    out=Path(out);api_path=step1.OUT/"STEP1_API.md"
    if not api_path.exists(): raise FileNotFoundError(f"waiting for step1 evaluation handoff: {api_path}")
    if not (out/"run_manifest_evaluation.json").exists(): _write_run_manifest(out/"run_manifest_evaluation.json")
    data=step1._step1_prepare_data(source_out)
    if len(data["task"])!=23304: raise AssertionError("unexpected fit/selection row count")
    # The API's split group names are exactly the four user-specified buckets.
    group_names=set(np.unique(np.asarray(data["group"]).astype(str)).tolist())
    required={"libero_goal","libero_object","libero_spatial","libero10_all"}
    if group_names!=required: raise AssertionError(f"step1 group API mismatch: {group_names}")
    if not (out/"selection.json").exists(): raise RuntimeError("tune must complete before group/Eval scoring")
    # Only fit+selection rows are loaded; unseen Eval features and labels remain unopened.
    tokens,groups=load_token_cache(data,kind="B")
    key_actions=load_training_actions(data,out)
    train_fn,score_fn=_api_callbacks(out,data,tokens,groups,key_actions,device=device,batch=batch)
    if not hasattr(step1,"run_step1_evaluations"):
        raise RuntimeError("published STEP1_API function is absent from module")
    result=step1.run_step1_evaluations(out/"shared_evaluation",train_fn,score_fn,
        data=data,eval_data=step1._step1_eval_data,conditions=CONDITIONS,
        groups=data["group"],seed=SEED,epochs=3)
    _atomic_json(out/"evaluation.json",{"status":"complete","selection_frozen":True,
        "eval_calls_per_condition":1,"api_path":str(api_path),"api_text":api_path.read_text(),
        "selection":json.loads((out/"selection.json").read_text()),"result":result})
    return result


def report_previous_grid(out=OUT):
    out=Path(out);p=out/"evaluation.json"
    if not p.exists():raise FileNotFoundError("evaluation.json does not exist; wait for STEP1_API.md and evaluate")
    d=json.loads(p.read_text());result=d["result"]
    folds=result["cv"];ev=result["eval"]
    lines=["# INV2 step 2", "",
        "設定選択はLIBERO-40保留タスクのみ。Evalの行動は、選択を凍結してから採点に一度だけ使用。seed=4101、各fitは3 epoch。K=4/5 を入力族ごとに先に選択した。",
        "L0′ dual は CE(T=.01)+KL(T=.05)、単一温度対照は CE(T=.03)。保留タスクで各入力族の K∈{4,5} を先に比較し、選んだ K で L0′ と L1′ を比較した（最大10 fit）。L1′は8場面×各32行を基本とした関係損失。",
        "",
        "| 条件 | group-CV held top1型誤差 | seen top1型誤差 | seen−held | held回収率 | seen回収率 | held Spearman | seen Spearman | Eval MSE6 all [95% CI] | hard [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in CONDITIONS:
        rr=[f for f in folds if f["condition"]==cond]
        hm=[f["metrics"]["held_group"] for f in rr]
        sm=[f["metrics"]["unseen_holdout_tasks_in_seen_groups"] for f in rr]
        em=ev[cond]["extras"]["action_mse"]
        mean=lambda rows,k:float(np.mean([x[k] for x in rows]))
        lines.append(f"| {cond} | {mean(hm,'top1_type_error'):.6f} | {mean(sm,'top1_type_error'):.6f} | {mean(sm,'top1_type_error')-mean(hm,'top1_type_error'):+.6f} | {mean(hm,'recovery'):.4f} | {mean(sm,'recovery'):.4f} | {mean(hm,'spearman_mean'):.4f} | {mean(sm,'spearman_mean'):.4f} | {em['all_mean']:.6f} [{em['all_ci95'][0]:.6f},{em['all_ci95'][1]:.6f}] | {em['hard_mean']:.6f} [{em['hard_ci95'][0]:.6f},{em['hard_ci95'][1]:.6f}] |")
    lines += ["", "## 場面4群ごとの CV", "", "| 条件 | 外した群 | held n | seen n | held回収率 | seen回収率 | heldρ | seenρ | Δ回収率 (seen−held) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in CONDITIONS:
        for f in folds:
            if f["condition"]!=cond:continue
            h=f["metrics"]["held_group"];s=f["metrics"]["unseen_holdout_tasks_in_seen_groups"]
            lines.append(f"| {cond} | {f['fold']} | {h['n']} | {s['n']} | {h['recovery']:.4f} | {s['recovery']:.4f} | {h['spearman_mean']:.4f} | {s['spearman_mean']:.4f} | {s['recovery']-h['recovery']:+.4f} |")
    lines += ["", "## Step1との比較", "",
        "同じ `run_step1_evaluations` に投入した。下表はstep1の固定5条件とINV2のEval指標で、群抜きCVの同一指標は上表と上の群別表を参照。",
        "", "| 条件 | Eval top1型誤差 | 回収率 | Spearman | MSE6 all | MSE6 hard |", "|---|---:|---:|---:|---:|---:|"]
    s1_candidates=[step1.OUT/"step1/eval/summary.json"]
    s1_candidates.extend(sorted(step1.OUT.glob("step1/attempt-*/eval/summary.json")))
    s1ev=next((p for p in reversed(s1_candidates) if p.exists()),None)
    if s1ev is not None:
        s1=json.loads(s1ev.read_text())["conditions"]
        for cond,r in s1.items():
            m=r["extras"]["action_mse"]
            lines.append(f"| Step1 {cond} | {r['metrics']['top1_type_error']:.6f} | {r['metrics']['recovery']:.4f} | {r['metrics']['spearman_mean']:.4f} | {m['all_mean']:.6f} | {m['hard_mean']:.6f} |")
    for cond,r in ev.items():
        m=r["extras"]["action_mse"]
        lines.append(f"| INV2 {cond} | {r['metrics']['top1_type_error']:.6f} | {r['metrics']['recovery']:.4f} | {r['metrics']['spearman_mean']:.4f} | {m['all_mean']:.6f} | {m['hard_mean']:.6f} |")
    step1_source=str(s1ev) if s1ev is not None else "summary.json未生成（Step1が未完了）"
    lines += ["", "## 損失定義・先行法からの変更", "",
        "CLASS §4.1–4.2 はDTWで正例を閾値選択し、類似度を `1−empirical-CDF(DTW)` として正例だけを重み付けし、重み付きlog-softmaxを使う。本実装は10×6のINV1型誤差を系列距離に使い、選択用保留タスクの q25 を閾値にし、重みは `exp(−d/τpos)` とした（DTWは未使用）。",
        "Robota-Embed Appendix A.2 (§Positive, negative, and ignored pairs / §Semantic conflict rules) は、明示的な意味衝突の負例だけ分母に残し、意味的に両立・不確実なペアは無理に負例化しない。本実装は意味ラベルの代わりに、同一場面かつ型距離≤τposのペアを損失分母から除外し、同一場面で距離が大きいペアは負例として残す。",
        "参照: [CLASS](https://arxiv.org/html/2508.01600), [Robota-Embed](https://arxiv.org/html/2610.19495).",
        "", "## 再現・制約", "",
        f"API handoff: `{d['api_path']}`。CUDA 学習・評価は凍結済み token cache を使い、新たな VLM 前向き計算はしていない。V2-B は学習中に三人称カメラ群を確率0.30でまとめてマスクした。V2-B-pool のみ、得点時にクエリ・キー双方の K トークンを平均する。",
        f"選択済み K: A={d['selection']['selected_k_by_input']['A']}、B={d['selection']['selected_k_by_input']['B']}。Step1 比較元: `{step1_source}`。",
        "本結果は固定 scale-only 型プール上のオフライン検索であり、閉ループ成功率ではない。"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    return d



ABLAT_EVAL_CONDITIONS=("R0","V2-final")


def evaluate_ablation(out=OUT, *, source_out=step1.OUT, device="cuda", batch=16):
    """Evaluate R0 and the holdout-selected cumulative stage through STEP1_API."""
    import torch
    out=Path(out);api_path=step1.OUT/"STEP1_API.md"
    if not api_path.exists():raise FileNotFoundError(api_path)
    sel_path=out/"ablation_selection.json"
    if not sel_path.exists():raise RuntimeError("run tune_stages before group/Eval evaluation")
    selection=json.loads(sel_path.read_text());final_cfg=selection["retained_config"]
    data=step1._step1_prepare_data(source_out)
    if len(data["task"])!=23304:raise AssertionError("unexpected fit/selection row count")
    required={"libero_goal","libero_object","libero_spatial","libero10_all"}
    if set(np.unique(np.asarray(data["group"]).astype(str)))!=required:raise AssertionError("step1 group set changed")
    tokens,groups=load_token_cache(data,kind="B")
    key_actions=load_training_actions(data,out)
    rowids=np.asarray(data["row_id"],np.int64)
    lookup=np.full(int(rowids.max())+1,-1,np.int64);lookup[rowids]=np.arange(len(rowids))
    eval_cache={}
    kq,tr,vs,sq=step1._modules()
    s1_scorer=step1._step1_score_factory(device=device,batch=batch)

    def train_fn(condition, train_rows, *, seed, epochs, save_path, fold):
        if seed!=SEED or epochs!=3:raise ValueError("ablation API fit fixed to seed4101, three epochs")
        path=Path(save_path)
        cfg=ablation_config("R0") if condition=="R0" else final_cfg
        if condition=="V2-final" and cfg["implementation"]=="step1_legacy":
            src=path.with_name(path.name.replace("V2_final","R0"))
            if not src.exists():raise FileNotFoundError(f"selected final stage aliases R0 but checkpoint missing: {src}")
            ck=torch.load(src,map_location="cpu",weights_only=False)
            model=step1._make_model("A",1,TOKEN_DIM);model.load_state_dict(ck["state_dict"])
            if not path.exists():torch.save(ck,path)
            return model
        if path.exists():
            return _load_ablation_checkpoint(path,cfg,device)
        if cfg["implementation"]=="step1_legacy":
            step1._preload_final_tokens(train_rows,"A",device)
            model,logs=step1._train_once(out,"A-L0",seed=SEED,tau=.01,k=1,layer="final",epochs=3,
                save_path=path,device=device,batch_size=64,report_each=False,data_override=train_rows)
            _log(out,stage="api_train",condition=condition,fold=fold,logs=logs)
            return model
        rid=np.asarray(train_rows["row_id"],np.int64);loc=lookup[rid]
        if np.any(loc<0):raise AssertionError("fold training row absent from LIBERO-40 feature cache")
        if np.any(np.asarray(train_rows["group"]).astype(str)==fold):raise AssertionError("held-out group entered fold fit")
        local={k:v for k,v in train_rows.items()};local["fit"]=np.ones(len(rid),bool)
        ai=loc
        model,logs=_train_config(local,tokens[loc],groups,key_actions[ai],kind=cfg["kind"],
            l1=cfg["l1"],pooled=False,l0_mode=cfg["l0"],l1_weight=cfg["l1_weight"],k=cfg["k"],
            key_mode=cfg["key_mode"],query_pool=cfg["query_pool"],
            no_history_cross=cfg.get("no_history_cross",False),learning_rate=cfg["lr"],
            epochs=3,device=device,batch_size=256 if cfg["l1"] else 64,seed=SEED,save_path=path)
        _log(out,stage="api_train",condition=condition,fold=fold,logs=logs)
        return model

    def get_query_tokens(query_rows):
        rid=np.asarray(query_rows["row_id"],np.int64)
        if "target_action" not in query_rows:
            ix=lookup[rid]
            if np.any(ix<0):raise AssertionError("CV query row absent from frozen train/selection tokens")
            return tokens[ix],groups
        cachekey=tuple(rid.tolist())
        if cachekey not in eval_cache:
            vals=np.load(step1.UNSEEN_ROWS/"tokens.npy",mmap_mode="r")
            gmem=np.load(step1.UNSEEN_ROWS/"groups.npy",mmap_mode="r");g=np.asarray(gmem[0]);keep=np.flatnonzero(g>=0)
            eval_cache[cachekey]=(np.asarray(vals[rid][:,keep],np.float16),g[keep].astype(np.int8))
        return eval_cache[cachekey]

    def score_fn(model,condition,query_rows):
        cfg=ablation_config("R0") if condition=="R0" else final_cfg
        if cfg["implementation"]=="step1_legacy":
            return s1_scorer(model,"A-L0",query_rows)
        qtokens,qgroups=get_query_tokens(query_rows)
        model.eval().to(device)
        if str(device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("V2_GPU_FRACTION",".22")),device=0)
        allow=np.asarray(query_rows["allowed"],bool);pool_actions=np.asarray(query_rows["pool_shape"],np.float32);pieces=[]
        with torch.no_grad():
            pa=torch.as_tensor(pool_actions,dtype=torch.float32,device=device)
            mk=torch.cat([model.encode_keys(pa[i:i+256]) for i in range(0,len(pa),256)],0)
            for st in range(0,len(qtokens),batch):
                end=min(st+batch,len(qtokens));ix=np.arange(st,end)
                q,_=model.encode_query(torch.as_tensor(qtokens[ix],device=device),
                    torch.as_tensor(query_rows["history"][ix],device=device),
                    torch.as_tensor(query_rows["state"][ix],device=device),
                    token_groups=torch.as_tensor(qgroups,device=device))
                pieces.append(model.scores(q,mk).cpu().numpy())
        scores=np.concatenate(pieces);scores[~allow]=-np.inf;extras={}
        if "target_action" in query_rows:
            rank=np.argsort(-scores,axis=1,kind="stable");chosen=rank[:,0]
            pred=kq.decode_pool_actions(query_rows["_pool"],chosen[:,None],vs,theta_source="src")[:,0,:,:6]
            truth=np.asarray(query_rows["target_action"],np.float32);mse=step1._six_axis_mse_over7(pred,truth)
            hist=np.asarray(query_rows["history"],np.float32);hold=np.repeat(hist[:,-1:,:6],truth.shape[1],axis=1)
            hmse=step1._six_axis_mse_over7(hold,truth);hard=np.asarray(query_rows["hard"],bool);ep=np.asarray(query_rows["episode_bootstrap"])
            extras["action_mse"]={"all_mean":float(mse.mean()),"hard_mean":float(mse[hard].mean()),
                "all_ci95":bootstrap_mean(mse,ep),"hard_ci95":bootstrap_mean(mse[hard],ep[hard]),
                "vs_hold_last":step1._bootstrap_diff(mse,hmse,ep),
                "vs_hold_last_hard":step1._bootstrap_diff(mse[hard],hmse[hard],ep[hard]),
                "hold_last_all":float(hmse.mean()),"hold_last_hard":float(hmse[hard].mean()),
                "hard_rows":int(hard.sum()),"rows":len(mse)}
        return {"scores":scores,"extras":extras}

    result=step1.run_step1_evaluations(out/"shared_evaluation",train_fn,score_fn,
        data=data,eval_data=step1._step1_eval_data,conditions=ABLAT_EVAL_CONDITIONS,
        groups=data["group"],seed=SEED,epochs=3)
    obj={"status":"complete","api":str(api_path),"conditions":list(ABLAT_EVAL_CONDITIONS),
        "selection":selection,"result":result,"eval_call_count_per_condition":1}
    _atomic_json(out/"ablation_evaluation.json",obj);return obj


def evaluate(out=OUT, *, source_out=step1.OUT, device="cuda", batch=16):
    return evaluate_ablation(out,source_out=source_out,device=device,batch=batch)


def report(out=OUT):
    out=Path(out);p=out/"ablation_evaluation.json"
    if not p.exists():raise FileNotFoundError("run evaluate after tune; ablation_evaluation.json missing")
    d=json.loads(p.read_text());res=d["result"];sel=d["selection"]
    lines=["# INV2 step 2: 修正後 R0–R5", "",
        "LIBERO-40 保留タスクのみで設定を選択し、Evalは選択後に一度だけ採点する。seed=4101、各fit 3 epoch。",
        "採否は top-1 型誤差 ≤ 現構成×1.02 かつ Spearman の厳密な上昇。R3は両条件を満たす重み候補から Spearman 最大を採る。R1が不採用の状態で後段を試す場合、後段に必要な1スロット潜在骨格だけを足し、その事実を記録する。", "",
        "## 保留タスクの段階別切り分け", "",
        "| 段/候補 | K | L0 | L1重み | top-1型誤差 | 回収率 | Spearman | K内平均cos | 判定 |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for st in sel["stages"]:
        for r in st["candidates"]:
            c=r["config"];cos=r["mean_query_token_cosine"]
            accepted=r.get("accepted",st.get("accepted",False))
            limit=r.get("rule",{}).get("error_limit")
            detail="採用" if accepted else "除外"
            if limit is not None:detail+=f" (誤差上限 {limit:.6f})"
            if c.get("scaffold_from_R1"):detail+="; R1骨格を前提追加"
            value=r.get("top1_type_error",r.get("selection_top1_type_error"))
            lines.append(f"| {st['stage']} | {c['k']} | {c['l0']} | {c['l1_weight']:.1f} | {value:.6f} | {r['recovery']:.4f} | {r['spearman_positive']:.4f} | {'—' if cos is None else f'{cos:.4f}'} | {detail} |")
    lines += ["", f"保持された最終設定: `{sel['retained_config']}`", "", "## 4群抜きCV・Eval", "",
        "| 条件 | 外した群 | held型誤差 | 学習側型誤差 | held回収率 | 学習側回収率 | heldρ | 学習側ρ | 差(回収率) | Eval MSE6 all [95% CI] | hard [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cv=res["cv"];ev=res["eval"]
    bycond={c:[x for x in cv if x["condition"]==c] for c in ABLAT_EVAL_CONDITIONS}
    for cond in ABLAT_EVAL_CONDITIONS:
        for f in bycond[cond]:
            h=f["metrics"]["held_group"];s=f["metrics"]["unseen_holdout_tasks_in_seen_groups"]
            em=ev[cond]["extras"]["action_mse"]
            lines.append(f"| {cond} | {f['fold']} | {h['top1_type_error']:.6f} | {s['top1_type_error']:.6f} | {h['recovery']:.4f} | {s['recovery']:.4f} | {h['spearman_mean']:.4f} | {s['spearman_mean']:.4f} | {s['recovery']-h['recovery']:+.4f} | {em['all_mean']:.6f} [{em['all_ci95'][0]:.6f},{em['all_ci95'][1]:.6f}] | {em['hard_mean']:.6f} [{em['hard_ci95'][0]:.6f},{em['hard_ci95'][1]:.6f}] |")
    lines += ["", "## Step1との比較", "", "同じSTEP1_APIの指標・分割・Eval行を使用。Step1 summary が存在する場合は比較表に追加。", "",
        "| 条件 | Eval top-1型誤差 | 回収率 | Spearman | MSE6 all | MSE6 hard |", "|---|---:|---:|---:|---:|---:|"]
    candidates=[step1.OUT/"step1/eval/summary.json",*sorted(step1.OUT.glob("step1/attempt-*/eval/summary.json"))]
    s1p=next((x for x in reversed(candidates) if x.exists()),None)
    if s1p:
        for c,r in json.loads(s1p.read_text())["conditions"].items():
            m=r["extras"]["action_mse"]
            lines.append(f"| Step1 {c} | {r['metrics']['top1_type_error']:.6f} | {r['metrics']['recovery']:.4f} | {r['metrics']['spearman_mean']:.4f} | {m['all_mean']:.6f} | {m['hard_mean']:.6f} |")
    else:lines.append("| Step1 | 実行中またはsummary未生成 | — | — | — | — |")
    for c,r in ev.items():
        m=r["extras"]["action_mse"]
        lines.append(f"| INV2 {c} | {r['metrics']['top1_type_error']:.6f} | {r['metrics']['recovery']:.4f} | {r['metrics']['spearman_mean']:.4f} | {m['all_mean']:.6f} | {m['hard_mean']:.6f} |")
    lines += ["", "## 修正・実装判断", "",
        "L0′の .01/.05/.03 は教師分布だけに適用し、logits は学習scale込みの値のまま同じ log_softmax に入力する。L1′キーはC_vm_none候補shapeの生成を元行だけから再現できないため、各行で最小型誤差となる許可済みpool template自体を使う。",
        "R1はZ0を1個だけ使い、手首トークンへのcross-attention後に旧78→512→256履歴・状態ベクトルと結合する。履歴列を読むcross-attentionは使わない。R4以降でK=5分割キー・対応トークン内積に切り替える。",
        "Bの三人称群は学習時のみサンプル単位30%でkey-padding maskし、evalはmaskなし。", "",
        "R0–R5の候補設定は `ablation_selection.json`、API結果は `ablation_evaluation.json` に保存。Evalの正解は選択後の採点だけに使用。"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    return d

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","smoke","tune","diagnostics","train","evaluate","report"])
    p.add_argument("--out",default=str(OUT));p.add_argument("--device",default="cuda")
    p.add_argument("--source-out",default=str(step1.OUT));p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--ablation-stage",choices=tuple(ABLATION_STAGES),help="print the exact R0–R5 switch config")
    a=p.parse_args(argv)
    if a.ablation_stage:
        print(json.dumps(ablation_config(a.ablation_stage),indent=2));return
    if a.stage=="prepare":print(json.dumps(prepare(a.out,source_out=a.source_out),indent=2))
    elif a.stage=="smoke":print(json.dumps(smoke(a.out,source_out=a.source_out,device=a.device),indent=2))
    elif a.stage=="tune":print(json.dumps(tune(a.out,source_out=a.source_out,device=a.device,epochs=a.epochs,batch_size=a.batch_size),indent=2))
    elif a.stage=="diagnostics":print(json.dumps(run_noise_history_diagnostics(a.out,source_out=a.source_out,device=a.device),indent=2))
    elif a.stage=="train":print(json.dumps(train_selected(a.out,source_out=a.source_out,device=a.device,batch_size=a.batch_size),indent=2))
    elif a.stage=="evaluate":print(json.dumps(evaluate(a.out,source_out=a.source_out,device=a.device),indent=2))
    else:print(json.dumps(report(a.out),indent=2))


if __name__=="__main__":main()
