#!/usr/bin/env python3
"""Shared latent-token keys for variable-length LIBERO motion units.

This module keeps the reusable model, objective, ranking metrics and cache
alignment helpers separate from CMP3's read-only segmentation/value pipeline.
The model consumes frozen π0.5 prefix tokens plus causal history/state for both
queries and memory-unit starts; it never consumes a target action as an input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stable_row_map(keys: Any) -> dict[tuple[int, int, int], int]:
    """Map (task, episode, frame) identities to cache rows, rejecting duplicates."""
    result: dict[tuple[int, int, int], int] = {}
    for i, raw in enumerate(keys):
        key = tuple(map(int, raw))
        if len(key) != 3 or key in result:
            raise ValueError(f"invalid or duplicate cache key at row {i}: {key}")
        result[key] = i
    return result


def join_feature_rows(
    keys: Any,
    primary_keys: Any,
    primary_tokens: np.ndarray,
    primary_groups: np.ndarray,
    secondary_keys: Any | None = None,
    secondary_tokens: np.ndarray | None = None,
    secondary_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Join exact-key prefix rows from caches, preserving requested row order.

    Cache keys are dataset row identities, never pool offsets. The function
    allows two disjoint caches (e.g. stride-10 and stride-5 extraction) while
    rejecting conflicting duplicate entries.
    """
    wanted = [tuple(map(int, x)) for x in keys]
    maps: list[tuple[dict[tuple[int, int, int], int], np.ndarray, np.ndarray]] = [
        (stable_row_map(primary_keys), primary_tokens, primary_groups)
    ]
    if secondary_keys is not None:
        if secondary_tokens is None or secondary_groups is None:
            raise ValueError("secondary keys require token and group arrays")
        maps.append((stable_row_map(secondary_keys), secondary_tokens, secondary_groups))
    width = max(x.shape[1] for _, x, _ in maps)
    dim = maps[0][1].shape[-1]
    if any(x.ndim != 3 or x.shape[-1] != dim or g.shape != x.shape[:2] for _, x, g in maps):
        raise ValueError("token/group cache shapes do not align")
    out = np.zeros((len(wanted), width, dim), np.float16)
    groups = np.full((len(wanted), width), -1, np.int8)
    missing = []
    for i, key in enumerate(wanted):
        found = [(m[key], tok, grp) for m, tok, grp in maps if key in m]
        if not found:
            missing.append(key)
            continue
        if len(found) > 1:
            # The two source caches can differ in padding width, but valid data
            # must agree exactly because they encode the same observation.
            ref_i, ref_t, ref_g = found[0]
            for other_i, other_t, other_g in found[1:]:
                n0 = int(np.sum(ref_g[ref_i] >= 0)); n1 = int(np.sum(other_g[other_i] >= 0))
                if n0 != n1 or not np.array_equal(ref_g[ref_i, :n0], other_g[other_i, :n1]):
                    raise ValueError(f"overlapping token caches disagree in valid mask for {key}")
                if not np.array_equal(ref_t[ref_i, :n0], other_t[other_i, :n1]):
                    raise ValueError(f"overlapping token caches disagree in tokens for {key}")
        src_i, src_t, src_g = found[0]
        n = int(np.sum(src_g[src_i] >= 0))
        out[i, :n] = src_t[src_i, :n]
        groups[i, :n] = src_g[src_i, :n]
    if missing:
        raise KeyError(f"{len(missing)} exact prefix keys missing; examples={missing[:5]}")
    return out, groups


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, q: torch.Tensor, memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.q_norm(q), self.kv_norm(memory), self.kv_norm(memory),
                         key_padding_mask=padding_mask, need_weights=False)
        q = q + a
        return q + self.ff(self.ff_norm(q))


INPUT_CONDITIONS = {
    "e_vlm": (True, False, False, False),
    "e_vlm_h": (True, False, True, True),
    "e_vj_h": (False, True, True, True),
    "e_vj_vlm_h": (True, True, True, True),
    "e_h": (False, False, True, True),
}
ALL_INPUT_CONDITIONS = tuple(INPUT_CONDITIONS)
LEGACY_MODES = ("shared", "separate", "p0", "p5")


class SharedLatentEncoder(nn.Module):
    """One E for query and memory starts; modalities are ablated by input_mode."""

    def __init__(self, token_dim: int = 2048, context_dim: int = 256, token_count: int = 5,
                 history_steps: int = 10, history_action_dim: int = 7, state_dim: int = 8,
                 group_count: int = 6, heads: int = 8, input_mode: str = "e_vlm_h",
                 vjepa_dim: int = 1024):
        super().__init__()
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)
        self.token_count = int(token_count)
        self.history_steps = int(history_steps)
        self.history_action_dim = int(history_action_dim)
        self.state_dim = int(state_dim)
        if input_mode not in INPUT_CONDITIONS:
            raise ValueError(f"unknown latent input condition: {input_mode}")
        self.input_mode = input_mode
        self.use_vlm, self.use_vjepa, self.use_history, self.use_state = INPUT_CONDITIONS[input_mode]
        self.vlm_projection = nn.Linear(token_dim, context_dim)
        self.vjepa_projection = nn.Linear(vjepa_dim, context_dim)
        self.group_embedding = nn.Embedding(group_count, context_dim)
        self.history_projection = nn.Linear(history_action_dim + 1, context_dim)
        self.history_position = nn.Parameter(torch.zeros(1, history_steps, context_dim))
        self.state_projection = nn.Sequential(nn.Linear(state_dim, context_dim), nn.GELU(), nn.Linear(context_dim, context_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.latent = nn.Parameter(torch.randn(1, token_count, context_dim) * .02)
        self.cross = _CrossAttentionBlock(context_dim, heads)
        self.self_attention = nn.MultiheadAttention(context_dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(context_dim)
        self.out_norm = nn.LayerNorm(context_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(LOGIT_SCALE_INITIAL))))

    def forward(self, vlm_tokens: torch.Tensor | None = None, groups: torch.Tensor | None = None,
                history: torch.Tensor | None = None, history_mask: torch.Tensor | None = None,
                state: torch.Tensor | None = None, vjepa_tokens: torch.Tensor | None = None,
                vjepa_groups: torch.Tensor | None = None) -> torch.Tensor:
        tensors = [x for x in (vlm_tokens, vjepa_tokens, history, state) if x is not None]
        if not tensors:
            raise ValueError("at least one input modality is required")
        b = int(tensors[0].shape[0]); memories=[]; masks=[]
        if self.use_vlm:
            if vlm_tokens is None or groups is None or vlm_tokens.ndim != 3 or vlm_tokens.shape[-1] != self.token_dim:
                raise ValueError(f"{self.input_mode} requires VLM tokens [B,T,{self.token_dim}] and groups")
            if groups.shape != vlm_tokens.shape[:2]: raise ValueError("VLM group IDs must match tokens")
            valid=groups>=0; safe=groups.clamp(0,self.group_embedding.num_embeddings-1).long()
            memories.append(self.vlm_projection(vlm_tokens.float())+self.group_embedding(safe));masks.append(~valid)
        if self.use_vjepa:
            if vjepa_tokens is None or vjepa_groups is None or vjepa_tokens.ndim != 3 or vjepa_tokens.shape[-1] != 1024:
                raise ValueError(f"{self.input_mode} requires V-JEPA tokens [B,T,1024] and groups")
            if vjepa_groups.shape != vjepa_tokens.shape[:2]: raise ValueError("V-JEPA group IDs must match tokens")
            # V-JEPA base/wrist groups are disjoint from the four VLM groups.
            vgroups=vjepa_groups.long()+4;valid=vjepa_groups>=0
            memories.append(self.vjepa_projection(vjepa_tokens.float())+self.group_embedding(vgroups.clamp(0,self.group_embedding.num_embeddings-1)))
            masks.append(~valid)
        if self.use_history:
            if history is None or history_mask is None or history.shape != (b,self.history_steps,self.history_action_dim) or history_mask.shape != (b,self.history_steps):
                raise ValueError("this input condition requires causal history and its validity mask")
            hist_in=torch.cat([history.float(),history_mask.float().unsqueeze(-1)],dim=-1)
            memories.append(self.history_projection(hist_in)+self.history_position);masks.append(~history_mask.bool())
        if self.use_state:
            if state is None or state.shape != (b,self.state_dim): raise ValueError("this input condition requires current state")
            memories.append(self.state_projection(state.float()).unsqueeze(1)+self.state_type)
            masks.append(torch.zeros((b,1),dtype=torch.bool,device=state.device))
        memory=torch.cat(memories,dim=1);padding=torch.cat(masks,dim=1)
        q = self.latent.expand(b, -1, -1)
        q = self.cross(q, memory, padding)
        normalized = self.self_norm(q)
        q = q + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        return F.normalize(self.out_norm(q), dim=-1)


def shared_token_scores(query: torch.Tensor, keys: torch.Tensor, *, chunk_size: int = 1024) -> torch.Tensor:
    """Raw mean of aligned token dot products; caller applies learned logit scale."""
    if query.ndim != 3 or keys.ndim != 3 or query.shape[1:] != keys.shape[1:]:
        raise ValueError(f"expected [B,K,D] and [N,K,D], got {query.shape}, {keys.shape}")
    pieces = []
    for start in range(0, len(keys), chunk_size):
        pieces.append(torch.einsum("bkd,nkd->bnk", query, keys[start:start + chunk_size]).mean(-1))
    return torch.cat(pieces, dim=1) if pieces else query.new_empty((len(query), 0))


def soft_listwise_loss(scores: torch.Tensor, type_errors: torch.Tensor, allowed: torch.Tensor,
                       teacher_temperature: float) -> torch.Tensor:
    """Soft listwise CE; temperature affects teacher distribution only."""
    if scores.shape != type_errors.shape or scores.shape != allowed.shape:
        raise ValueError("scores, type errors and allowed mask must have identical shapes")
    valid_rows = allowed.any(dim=-1)
    if not valid_rows.any():
        return scores.sum() * 0.
    masked_err = type_errors.float().masked_fill(~allowed.bool(), torch.inf)
    target_logits = (-masked_err / float(teacher_temperature)).masked_fill(~allowed.bool(), -1e9)
    target = torch.softmax(target_logits, dim=-1)
    logp = torch.log_softmax(scores.float().masked_fill(~allowed.bool(), -1e9), dim=-1)
    return -(target * logp).sum(dim=-1)[valid_rows].mean()


def hard_info_nce_loss(scores: torch.Tensor, type_errors: torch.Tensor,
                       allowed: torch.Tensor) -> torch.Tensor:
    """Hard InfoNCE control: the lowest-error allowed unit is the sole positive."""
    if scores.shape != type_errors.shape or scores.shape != allowed.shape:
        raise ValueError("scores, errors and candidate mask must have identical shapes")
    valid=allowed.bool().any(-1)
    errors=type_errors.float().masked_fill(~allowed.bool(),torch.inf)
    target=torch.zeros_like(scores,dtype=torch.float32)
    target.scatter_(1,errors.argmin(-1,keepdim=True),1.)
    logq=torch.log_softmax(scores.float().masked_fill(~allowed.bool(),-1e9),-1)
    return -(target*logq).sum(-1)[valid].mean() if valid.any() else scores.sum()*0.


def contrastive_candidate_set(batch_key_ids, random_memory_ids, query_task, query_episode,
                              candidate_task, candidate_episode):
    """Batch positives/negatives plus random memory rows, with self-episode masks.

    Candidate sampling depends only on batch identities and a uniform RNG draw,
    not on continuation actions or their MSE labels.
    """
    cids=np.unique(np.r_[np.asarray(batch_key_ids,np.int64),np.asarray(random_memory_ids,np.int64)])
    qt=np.asarray(query_task);qe=np.asarray(query_episode);ct=np.asarray(candidate_task)[cids];ce=np.asarray(candidate_episode)[cids]
    allowed=~((qt[:,None]==ct[None,:])&(qe[:,None]==ce[None,:]))
    return cids,allowed


def episode_bootstrap_ci(values: np.ndarray, task: np.ndarray, episode: np.ndarray,
                         *, reps: int = 5000, seed: int = 4101) -> dict[str, float]:
    """Episode-cluster bootstrap mean and percentile interval."""
    v = np.asarray(values, np.float64)
    ids = np.asarray([f"{int(t)}:{int(e)}" for t, e in zip(task, episode, strict=True)])
    unique = np.unique(ids)
    by_ep = [v[ids == x] for x in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(reps, np.float64)
    for j in range(reps):
        draw = rng.integers(0, len(unique), size=len(unique))
        means[j] = np.concatenate([by_ep[i] for i in draw]).mean()
    return {"mean": float(v.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "episodes": int(len(unique)), "rows": int(len(v))}


def ranking_metrics(scores: np.ndarray, type_errors: np.ndarray) -> dict[str, float]:
    """Top-1 type error, recovery against random/oracle, and positive-is-good Spearman."""
    from scipy.stats import spearmanr
    s, e = np.asarray(scores, np.float64), np.asarray(type_errors, np.float64)
    if s.shape != e.shape or s.ndim != 2:
        raise ValueError("expected matching [queries,candidates] scores/errors")
    order = np.argsort(-s, axis=1, kind="stable")
    top_error = e[np.arange(len(e)), order[:, 0]]
    best_error = e.min(axis=1)
    random_error = e.mean(axis=1)
    # Larger score should associate with smaller error, hence negate error.
    correlations = [spearmanr(s[i], -e[i]).statistic for i in range(len(e))]
    rho = np.asarray(correlations, np.float64)
    return {"top1_arm6_mse": float(top_error.mean()),
            "recovery": float(np.mean((random_error - top_error) / np.maximum(random_error - best_error, 1e-12))),
            "spearman_positive": float(np.nanmean(rho)), "random_arm6_mse": float(random_error.mean()),
            "querywise_oracle_arm6_mse": float(best_error.mean())}


def action_errors(pred: np.ndarray, target: np.ndarray, pi05_grip: np.ndarray | None = None) -> dict[str, float]:
    """Per-query active7 and arm6 errors under the CMP3 sum-of-six / 7 convention."""
    p, y = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[-1] < 7:
        raise ValueError("expected matching [N,T,>=7] actions")
    arm = np.square(p[..., :6] - y[..., :6]).sum((1, 2)) / (p.shape[1] * 7)
    grip = np.square(p[..., 6] - y[..., 6]).sum(1) / (p.shape[1] * 7)
    out = {"arm6": arm, "mse_gripper_truth": arm, "active7": arm + grip,
           "active7_saved_gripper": arm + grip}
    if pi05_grip is not None:
        g = np.asarray(pi05_grip, np.float64)
        if g.shape != p[..., 6].shape:
            raise ValueError("π0.5 grip shape mismatch")
        out["active7_pi05_grip"] = arm + np.square(g - y[..., 6]).sum(1) / (p.shape[1] * 7)
    return out


def scale_only_type_errors(query_actions: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
    """Pairwise six-axis shape MSE after removing one positive global scale.

    Inputs are aligned [query,time,6] and [candidate,time,6] windows. A single
    RMS scale per whole query/candidate is removed, preserving relative axis
    amplitudes and direction. This is the teacher quantity; grip is excluded.
    """
    q = np.asarray(query_actions, np.float64)
    c = np.asarray(candidate_actions, np.float64)
    if q.ndim != 3 or c.ndim != 3 or q.shape[1:] != c.shape[1:] or q.shape[-1] != 6:
        raise ValueError(f"expected [Q,T,6] and [N,T,6], got {q.shape} and {c.shape}")
    qn = q / np.maximum(np.sqrt(np.mean(q * q, axis=(1, 2), keepdims=True)), 1e-8)
    cn = c / np.maximum(np.sqrt(np.mean(c * c, axis=(1, 2), keepdims=True)), 1e-8)
    return np.square(qn[:, None] - cn[None]).mean(axis=(2, 3))


class VectorRanker(nn.Module):
    """P0/P5 ablation: pooled observation query and independently encoded key."""
    def __init__(self, token_dim=2048, context_dim=256, history_steps=10, state_dim=8,
                 use_history=True, key_input_dim=None, token_count=5):
        super().__init__()
        self.use_history = bool(use_history)
        self.token_count = int(token_count)
        qdim = token_dim + ((history_steps * 7 + history_steps + state_dim) if use_history else 0)
        self.q = nn.Sequential(nn.Linear(qdim, 512), nn.GELU(), nn.Linear(512, context_dim))
        self.k = nn.Sequential(nn.Linear(key_input_dim or token_dim, 512), nn.GELU(),
                                nn.Linear(512, context_dim * token_count))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(LOGIT_SCALE_INITIAL))))

    def query(self, pooled, history, history_mask, state):
        x = [pooled]
        if self.use_history:
            x.extend([history.flatten(1), history_mask.float(), state])
        q = F.normalize(self.q(torch.cat(x, -1)), dim=-1)
        return q[:, None, :].expand(-1, self.token_count, -1)

    def key(self, features):
        return F.normalize(self.k(features).reshape(-1, self.token_count,
                                                     self.k[-1].out_features // self.token_count), dim=-1)


class TypeKeyEncoder(nn.Module):
    """Separate-key ablation; unit motion shape, no query/context leakage."""
    def __init__(self, input_dim=60, width=256, token_count=5):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(LOGIT_SCALE_INITIAL))))
        self.token_count = token_count
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(),
                                 nn.Linear(512, width * token_count))
        self.width = width

    def forward(self, x):
        return F.normalize(self.net(x.flatten(1)).reshape(-1, self.token_count, self.width), dim=-1)


def two_temperature_target(errors: torch.Tensor, allowed: torch.Tensor,
                            temperatures=(.01, .03)) -> torch.Tensor:
    """Average two soft targets; neither temperature rescales model logits."""
    e = errors.float().masked_fill(~allowed.bool(), torch.inf)
    dists = []
    for tau in temperatures:
        logits = (-e / float(tau)).masked_fill(~allowed.bool(), -1e9)
        dists.append(torch.softmax(logits, -1))
    target = torch.stack(dists).mean(0)
    return target / target.sum(-1, keepdim=True).clamp_min(1e-12)


def contrastive_teacher_loss(scores: torch.Tensor, errors: torch.Tensor,
                             allowed: torch.Tensor, *, objective: str,
                             teacher_temperature: float = .005) -> torch.Tensor:
    """Soft-label InfoNCE or its tau-to-zero hard-positive control."""
    if objective == "hard":
        return hard_info_nce_loss(scores, errors, allowed)
    if objective != "soft":
        raise ValueError(f"unknown contrastive objective {objective}")
    return soft_listwise_loss(scores,errors,allowed,teacher_temperature)


class ExactFeatureStore:
    """Read-only token cache join by (task,episode,frame), including exact unit starts."""
    def __init__(self, run_root, missing_root, seen_stride5, unseen_stride5):
        self.caches=[]
        run_root=Path(run_root)
        rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (run_root/'rows').glob('*.npy')}
        rkeys=list(zip(rows['task'].tolist(),rows['episode'].tolist(),rows['frame'].tolist()))
        self.caches.append(('libero40',stable_row_map(rkeys),np.load(run_root/'tokens.npy',mmap_mode='r'),
                            np.load(run_root/'groups.npy',mmap_mode='r')))
        for name,path in (('libero40',Path(seen_stride5)),('libero90',Path(unseen_stride5)),('libero40',Path(missing_root))):
            manifest=json.loads((path/'manifest.json').read_text())
            keys=manifest['keys']
            self.caches.append((name,stable_row_map(keys),np.load(path/'tokens.npy',mmap_mode='r'),
                                np.load(path/'groups.npy',mmap_mode='r')))
        self.rows=rows
        # Exact duplicates are checked on demand in get(); the very large cache
        # arrays remain memory-mapped instead of being copied into RAM.

    def get(self, keys, *, domain='libero40'):
        out=np.zeros((len(keys),532,2048),np.float16);groups=np.full((len(keys),532),-1,np.int8)
        for oi,key0 in enumerate(keys):
            key=tuple(map(int,key0)); found=[]
            for name,mapping,tokens,grps in self.caches:
                if name==domain and key in mapping:
                    ix=mapping[key]; valid=np.asarray(grps[ix])>=0
                    found.append((np.asarray(tokens[ix])[valid],np.asarray(grps[ix])[valid]))
            if not found: raise KeyError(f"no exact prefix tokens for {key}")
            ref_t,ref_g=found[0]
            if any(not np.array_equal(ref_t,t) or not np.array_equal(ref_g,g) for t,g in found[1:]):
                raise ValueError(f"duplicate exact token caches disagree for {key}")
            n=min(len(ref_t),532);out[oi,:n]=ref_t[:n];groups[oi,:n]=ref_g[:n]
        return out,groups

    def pooled(self, keys, *, domain='libero40'):
        t,g=self.get(keys,domain=domain);valid=g>=0
        return (t.astype(np.float32)*valid[...,None]).sum(1)/np.maximum(valid.sum(1,keepdims=True),1)


class ExactVJEPAFeatureStore:
    """Exact (task,episode,frame) joins for the frozen two-camera V-JEPA cache."""
    def __init__(self, key2_cache: Path, eval_cache: Path):
        self.maps={"libero40":[],"libero90":[]}
        for domain,path in (("libero40",Path(key2_cache)/"libero40"),
                            ("libero90",Path(key2_cache)/"libero90"),
                            ("libero90",Path(eval_cache))):
            manifest=json.loads((path/"manifest.json").read_text())
            if manifest.get("complete") is not True:raise ValueError(f"incomplete V-JEPA rows in {path}")
            keys=np.load(path/"keys.npy",mmap_mode="r",allow_pickle=False)
            base=np.load(path/"base.npy",mmap_mode="r",allow_pickle=False)
            wrist=np.load(path/"wrist.npy",mmap_mode="r",allow_pickle=False)
            if base.shape!=(len(keys),64,1024) or wrist.shape!=base.shape:
                raise ValueError(f"bad V-JEPA arrays in {path}")
            self.maps[domain].append((stable_row_map(keys),base,wrist))

    def get(self, keys, *, domain):
        if domain not in self.maps:raise ValueError(f"unknown V-JEPA domain {domain}")
        result=np.empty((len(keys),128,1024),np.float16)
        mapping_sets=self.maps[domain]
        for i,key0 in enumerate(keys):
            key=tuple(map(int,key0));found=[]
            for mapping,base,wrist in mapping_sets:
                if key in mapping:
                    j=mapping[key];found.append((np.asarray(base[j]),np.asarray(wrist[j])))
            if not found:raise KeyError(f"missing exact V-JEPA token row {domain}:{key}")
            b,w=found[0]
            if any(not np.array_equal(b,x) or not np.array_equal(w,y) for x,y in found[1:]):
                raise ValueError(f"overlapping V-JEPA cache rows disagree: {domain}:{key}")
            result[i,:64]=b;result[i,64:]=w
        groups=np.tile(np.r_[np.zeros(64,np.int8),np.ones(64,np.int8)],(len(keys),1))
        return result,groups


def build_vjepa_eval_cache(output: Path, *, source_root=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/data/libero90_eval"),
                           feature_root=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/features128/vjepa2-vitl-fpc64-256"),
                           model_dir=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/data/vjepa2-vitl-fpc64-256"),
                           device="cuda:0", memory_fraction=.04):
    """Complete exact Eval V-JEPA rows from cached corrected-orientation images."""
    output=Path(output);source_root=Path(source_root)
    if output.exists() and (output/"manifest.json").exists():
        old=json.loads((output/"manifest.json").read_text())
        if old.get("complete") is True:return old
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite incomplete/unrecognized cache directory {output}")
    output.mkdir(parents=True,exist_ok=True)
    keys=np.load(source_root/"task.npy",mmap_mode="r",allow_pickle=False)
    task=keys;episode=np.load(source_root/"episode.npy",mmap_mode="r",allow_pickle=False)
    frame=np.load(source_root/"frame.npy",mmap_mode="r",allow_pickle=False)
    key_array=np.column_stack([task,episode,frame]).astype(np.int32)
    if len(key_array)!=619 or len(np.unique(key_array,axis=0))!=619:
        raise AssertionError("fixed LIBERO-90 Eval key set must contain 619 unique rows")
    source=Path(feature_root)/"libero90_eval"
    source_keys=np.load(source/"row_keys.npy",mmap_mode="r",allow_pickle=False)
    done=np.load(source/"completed.npy",mmap_mode="r",allow_pickle=False)
    if not np.array_equal(source_keys,key_array):raise AssertionError("latent-probe V-JEPA Eval keys do not match fixed rows")
    base_path=output/"base.npy";wrist_path=output/"wrist.npy"
    base=np.lib.format.open_memmap(base_path,mode="w+",dtype=np.float16,shape=(len(key_array),64,1024))
    wrist=np.lib.format.open_memmap(wrist_path,mode="w+",dtype=np.float16,shape=(len(key_array),64,1024))
    srcb=np.load(source/"base.npy",mmap_mode="r",allow_pickle=False);srcw=np.load(source/"wrist.npy",mmap_mode="r",allow_pickle=False)
    present=np.asarray(done,bool)
    base[present]=srcb[present];wrist[present]=srcw[present]
    missing=np.flatnonzero(~present)
    if len(missing):
        import torch
        from PIL import Image
        from transformers import AutoModel,AutoVideoProcessor
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
        torch_device=torch.device(device)
        if torch_device.type=="cuda":torch.cuda.set_per_process_memory_fraction(float(memory_fraction),torch_device.index or 0)
        processor=AutoVideoProcessor.from_pretrained(model_dir,local_files_only=True)
        model=AutoModel.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float16).eval().to(torch_device)
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from key_inputs_vjepa import _encode_one
        ib=np.load(source_root/"image_base_prev.npy",mmap_mode="r",allow_pickle=False)
        ic=np.load(source_root/"image_base.npy",mmap_mode="r",allow_pickle=False)
        iw=np.load(source_root/"image_wrist_prev.npy",mmap_mode="r",allow_pickle=False)
        iwc=np.load(source_root/"image_wrist.npy",mmap_mode="r",allow_pickle=False)
        for offset,index in enumerate(missing,1):
            images={"base":[Image.fromarray(np.asarray(ib[index]).astype(np.uint8)),Image.fromarray(np.asarray(ic[index]).astype(np.uint8))],
                    "wrist":[Image.fromarray(np.asarray(iw[index]).astype(np.uint8)),Image.fromarray(np.asarray(iwc[index]).astype(np.uint8))]}
            b,w=_encode_one(model,processor,images,device=str(torch_device));base[index]=b;wrist[index]=w
            if offset%64==0:base.flush();wrist.flush()
        base.flush();wrist.flush();del model,processor
    del base,wrist
    manifest={"contract":"key2-e-vjepa-eval-v1","source":"latent-probe corrected-orientation image memmaps",
        "rows":len(key_array),"keys_file":"keys.npy","shape_per_camera":[64,1024],"dtype":"float16",
        "source_completed_rows_reused":int(present.sum()),"rows_encoded_here":int(len(missing)),
        "exact_selection_keys":True,"complete":True}
    np.save(output/"keys.npy",key_array,allow_pickle=False)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def _load_key2_rows(run_root):
    root=Path(run_root)
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (root/'rows').glob('*.npy')}


def _training_candidate_ids(tasks, episodes, eval_pairs):
    """Candidate indices allowed to shape training teachers (exclude Eval demos)."""
    tasks=np.asarray(tasks);episodes=np.asarray(episodes)
    if tasks.shape!=episodes.shape:raise ValueError('candidate task/episode arrays do not align')
    heldout={tuple(map(int,p)) for p in eval_pairs}
    return np.asarray([i for i,(t,e) in enumerate(zip(tasks,episodes,strict=True))
                       if (int(t),int(e)) not in heldout],np.int64)


def _episode_history(rows, task, episode, frame):
    ids=np.flatnonzero((rows['task']==task)&(rows['episode']==episode))
    ids=ids[np.argsort(rows['frame'][ids])]
    if not len(ids): raise KeyError((task,episode))
    base=int(rows['frame'][ids[0]])
    trace=np.asarray(rows['actions'][ids,:,:7],np.float32).reshape(-1,7)
    f=int(frame)-base
    if not 0<=f<=len(trace): raise ValueError(f"frame outside episode trace: {task,episode,frame}")
    hist=np.zeros((10,7),np.float32);mask=np.zeros(10,bool)
    lo=max(0,f-10);n=f-lo
    if n: hist[-n:]=trace[lo:f];mask[-n:]=True
    state=np.asarray(rows['state'][ids[min(max(f//10,0),len(ids)-1)],:8],np.float32)
    return hist,mask,state,trace[f:f+10]


def _exact_start_states(keys, rows, run_root, missing_root):
    """Read exact normalized current state for every memory-unit start."""
    import pyarrow.parquet as pq
    from collections import OrderedDict
    from openpi.shared.normalize import load as load_norm
    from openpi.transforms import Normalize
    run_root=Path(run_root);manifest=json.loads((run_root/'data_manifest.json').read_text())
    records={(int(r['task']),int(r['episode'])):r for r in manifest['episodes']}
    normfiles=list((Path(manifest['checkpoint'])/'assets').rglob('norm_stats.json'))
    if len(normfiles)!=1:raise ValueError(f'expect one q01/q99 stats file, got {normfiles}')
    normalizer=Normalize(load_norm(normfiles[0].parent),use_quantiles=True)
    # Reuse state written by the exact-frame VLM extraction when available.
    extra=Path(missing_root);extra_map={}
    if (extra/'manifest.json').exists():
        m=json.loads((extra/'manifest.json').read_text())
        states=np.load(extra/'state.npy',mmap_mode='r',allow_pickle=False)
        extra_map={tuple(map(int,k)):states[i] for i,k in enumerate(m['keys'])}
    rowmap={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows['task'],rows['episode'],rows['frame']))}
    cache:OrderedDict[Path,Any]=OrderedDict();out=[]
    for key0 in keys:
        key=tuple(map(int,key0))
        if key in extra_map:
            out.append(np.asarray(extra_map[key],np.float32)[:8]);continue
        if key in rowmap:
            out.append(np.asarray(rows['state'][rowmap[key]],np.float32)[:8]);continue
        rec=records[(key[0],key[1])];path=Path(rec['path'])
        if path not in cache:
            if len(cache)>=2:cache.popitem(last=False)
            cache[path]=pq.read_table(path,columns=['state'])
        raw=np.asarray(cache[path]['state'][key[2]].as_py(),np.float32)
        norm=np.asarray(normalizer({'state':raw})['state'],np.float32)
        out.append(norm[:8])
    result=np.asarray(out,np.float32)
    if result.shape!=(len(keys),8) or not np.isfinite(result).all():raise ValueError('invalid exact unit-start states')
    return result


def _arm6_candidate_mse(query_actions, candidates):
    """Stored-value arm MSE, with no scale/direction fit and grip excluded.

    Denominator intentionally remains 10*7 to match the active7 convention:
    sum of six squared arm errors divided by the number of seven-axis elements.
    """
    q=np.asarray(query_actions,np.float64)
    c=np.asarray(candidates,np.float64)
    if q.ndim!=3 or c.ndim!=3 or q.shape[1]!=c.shape[1] or q.shape[2]<6 or c.shape[2]<6:
        raise ValueError(f'expected [Q,T,>=6] and [M,T,>=6], got {q.shape}, {c.shape}')
    denominator=q.shape[1]*7.
    q=q[...,:6].reshape(len(q),-1);c=c[...,:6].reshape(len(c),-1)
    cn=np.einsum('nd,nd->n',c,c);out=np.empty((len(q),len(c)),np.float32)
    for s in range(0,len(q),64):
        x=q[s:s+64];dot=x@c.T;qn=np.einsum('nd,nd->n',x,x)
        out[s:s+len(x)]=np.maximum(qn[:,None]+cn[None]-2.*dot,0.)/denominator
    return out


def _pack_query_observation(split, indices, pooled=None, type_features=None):
    """Pack only causal query inputs; labels and episode/task IDs are omitted."""
    ix=np.asarray(indices,np.int64)
    allowed=('tokens','groups','history','history_mask','state')
    missing=[k for k in allowed if k not in split]
    if missing:raise KeyError(f'missing query observation fields: {missing}')
    allowed=allowed+tuple(k for k in ('vjepa_tokens','vjepa_groups') if k in split)
    result={k:np.asarray(split[k])[ix] for k in allowed}
    if pooled is not None:
        pooled=np.asarray(pooled)
        result['pooled']=pooled if len(pooled)==len(ix) else pooled[ix]
    if type_features is not None:result['type_features']=np.asarray(type_features)
    return result


def _assert_eval_candidates_disjoint(split, bank):
    """Check fixed evaluation episodes are absent from the memory candidate pool."""
    # `episode` may be domain-offset for bootstrap grouping; use immutable
    # source identity keys for this audit, never for a retrieval decision.
    if 'key' in split:
        keys=np.asarray(split['key'])
        eval_pairs={(int(k[0]),int(k[1])) for k in keys}
    else:
        eval_pairs=set(zip(np.asarray(split['task']).astype(int).tolist(),
                           np.asarray(split['episode']).astype(int).tolist(),strict=True))
    memory_pairs=set(zip(np.asarray(bank['task']).astype(int).tolist(),
                         np.asarray(bank['episode']).astype(int).tolist(),strict=True))
    overlap=eval_pairs&memory_pairs
    if overlap:raise AssertionError(f'evaluation episodes appear in memory candidates: {sorted(overlap)[:5]}')


def _softmax_np(logits, axis=-1):
    x=np.asarray(logits,np.float64)
    x=x-np.max(x,axis=axis,keepdims=True)
    exp=np.exp(x)
    return exp/np.maximum(exp.sum(axis=axis,keepdims=True),np.finfo(np.float64).tiny)


def candidate_rank_metrics(scores, errors, *, tau, temperature=1., random_orders=None):
    """Full-candidate ranking/distribution scores; lower arm6 MSE is relevant."""
    from scipy.stats import spearmanr
    s=np.asarray(scores,np.float64);e=np.asarray(errors,np.float64)
    if s.shape!=e.shape or s.ndim!=2:raise ValueError('scores/errors must be matching [queries,candidates]')
    if tau<=0 or temperature<=0:raise ValueError('tau and temperature must be positive')
    q=_softmax_np(s/temperature,axis=1)
    p=_softmax_np(-e/tau,axis=1)
    ce=float(np.mean(-np.sum(p*np.log(np.maximum(q,1e-300)),axis=1)))
    kl=float(np.mean(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))),axis=1)))
    n,m=e.shape;row=np.arange(n);true_order=np.argsort(e,axis=1,kind='stable')
    pred_order=np.argsort(-s,axis=1,kind='stable')
    rho=[]
    for i in range(n):
        if np.ptp(s[i])>0 and np.ptp(e[i])>0:rho.append(float(spearmanr(s[i],-e[i]).statistic))
        else:rho.append(np.nan)
    requested_ks=(1,5,20)
    recall={}
    for requested_k in requested_ks:
        k=min(requested_k,m)
        recall[f'recall@{requested_k}']=float(np.mean([
            len(set(true_order[i,:k]).intersection(pred_order[i,:k]))/k for i in range(n)]))
    k20=min(20,m);relevance=np.exp(-e/float(tau))
    denom=np.log2(np.arange(2,k20+2))[None,:]
    dcg=np.sum(np.take_along_axis(relevance,pred_order[:,:k20],axis=1)/denom,axis=1)
    ideal=np.sum(np.take_along_axis(relevance,true_order[:,:k20],axis=1)/denom,axis=1)
    ndcg=float(np.mean(dcg/np.maximum(ideal,1e-300)))
    best=true_order[:,0]
    best_rank=np.argmax(pred_order==best[:,None],axis=1)+1
    top1=e[row,pred_order[:,0]]
    top20=np.take_along_axis(e,pred_order[:,:k20],axis=1).mean(axis=1)
    oracle_top1=e[row,best]
    if random_orders is None:
        random_top1=float(np.mean(e[row,np.random.default_rng(0).integers(m,size=n)]))
    else:
        ords=np.asarray(random_orders,np.int64)
        if ords.ndim!=3 or ords.shape[1:]!=s.shape:raise ValueError('random_orders must be [R,Q,M]')
        random_top1=float(np.mean(np.stack([e[row,o[:,0]] for o in ords])))
    recovery=float((random_top1-top1.mean())/max(random_top1-oracle_top1.mean(),1e-12))
    return {'spearman_positive':float(np.nanmean(rho)) if np.isfinite(rho).any() else 0.,
        'cross_entropy':ce,'kl_pq':kl,**recall,'ndcg@20':ndcg,
        'mrr':float(np.mean(1./best_rank)),'top1_arm6_mse':float(top1.mean()),
        'top20_arm6_mse':float(top20.mean()),'oracle_top1_arm6_mse':float(oracle_top1.mean()),
        'random_top1_arm6_mse':random_top1,'recovery':recovery,
        'ranking_reference_tau':float(tau),'score_temperature':float(temperature)}


def random_candidate_metrics(errors, *, tau, seeds=(4101,4102,4103,4104,4105)):
    """Random rankings plus a uniform candidate probability distribution."""
    e=np.asarray(errors,np.float64);n,m=e.shape;row=np.arange(n)
    orders=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed))
        orders.append(np.stack([rng.permutation(m) for _ in range(n)]))
    orders=np.asarray(orders)
    # Uniform q has an unambiguous distributional interpretation; random order
    # permutations supply the rank-sensitive measures.
    metrics=candidate_rank_metrics(np.zeros_like(e),e,tau=tau,temperature=1.,random_orders=orders)
    true_order=np.argsort(e,axis=1,kind='stable');k20=min(20,m)
    relevance=np.exp(-e/float(tau));denom=np.log2(np.arange(2,k20+2))[None,:]
    ndcg=[];mrr=[];recall={k:[] for k in (1,5,20)};top1=[];top20=[]
    for order in orders:
        for k in recall:
            kk=min(k,m);recall[k].append(np.mean([len(set(true_order[i,:kk])&set(order[i,:kk]))/kk for i in range(n)]))
        dcg=np.sum(np.take_along_axis(relevance,order[:,:k20],axis=1)/denom,axis=1)
        ideal=np.sum(np.take_along_axis(relevance,true_order[:,:k20],axis=1)/denom,axis=1)
        ndcg.append(np.mean(dcg/np.maximum(ideal,1e-300)))
        best=true_order[:,0];rank=np.argmax(order==best[:,None],axis=1)+1;mrr.append(np.mean(1./rank))
        top1.append(np.mean(e[row,order[:,0]]));top20.append(np.mean(np.take_along_axis(e,order[:,:k20],axis=1)))
    metrics.update({f'recall@{k}':float(np.mean(v)) for k,v in recall.items()})
    metrics['ndcg@20']=float(np.mean(ndcg));metrics['mrr']=float(np.mean(mrr))
    metrics['top1_arm6_mse']=float(np.mean(top1));metrics['top20_arm6_mse']=float(np.mean(top20))
    metrics['random_top1_arm6_mse']=float(np.mean(top1));metrics['oracle_top1_arm6_mse']=float(np.min(e,axis=1).mean())
    metrics['recovery']=0.;metrics['random_seeds']=list(map(int,seeds))
    return metrics


def oracle_candidate_metrics(errors, *, tau):
    """Perfect candidate ranking under the same full-candidate diagnostics."""
    e=np.asarray(errors,np.float64)
    return candidate_rank_metrics(-e,e,tau=tau,temperature=tau)


def natural_topk_blend_actions(scores, bank, *, k, temperature):
    """Blend raw natural-unit action windows; never average DCT coefficients.

    Each query is one 10-step selection interval. A unit longer than ten steps
    is cut at the next selection point; a shorter unit ends naturally and its
    terminal command is held until that point. Only then are the raw action
    windows averaged with softmax score weights. No time stretching, pose fit,
    magnitude fit, or coefficient-space averaging is applied.
    """
    s=np.asarray(scores,np.float64);n,m=s.shape;k=min(int(k),m)
    if k<1 or temperature<=0:raise ValueError('k and temperature must be positive')
    top=np.argsort(-s,axis=1,kind='stable')[:,:k]
    weight=_softmax_np(np.take_along_axis(s,top,axis=1)/float(temperature),axis=1)
    natural=np.asarray(bank['natural_actions'],np.float32);lengths=np.asarray(bank['length'],np.int64)
    out=np.empty((n,10,natural.shape[-1]),np.float32)
    for i in range(n):
        chunks=[]
        for candidate in top[i]:
            length=int(lengths[candidate])
            if length<1:raise ValueError('unit length must be positive')
            unit=natural[candidate,:length]
            window=np.empty((10,unit.shape[-1]),np.float32)
            n=min(length,10)
            window[:n]=unit[:n]
            if n<10:window[n:]=unit[n-1]
            chunks.append(window)
        out[i]=np.einsum('k,ktd->td',weight[i].astype(np.float32),np.stack(chunks),optimize=True)
    return out


# Keep the old private name for call sites in notebooks, but its semantics are
# now explicitly the saved-value arm6 MSE used by KEY2b's listwise target.
_fast_type_errors=_arm6_candidate_mse


def _atomic_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
    tmp.replace(path)


def _last10(history40, mask40):
    out=np.zeros((10,7),np.float32);mask=np.zeros(10,bool)
    valid=np.asarray(mask40,bool)
    a=np.asarray(history40,np.float32)[valid][-10:]
    if len(a):out[-len(a):]=a;mask[-len(a):]=True
    return out,mask


def _build_key2_queries(run_root, missing_root, seen_stride5, unseen_stride5, vjepa_eval_cache):
    """Create memory anchors, fixed learned holdout, fixed unseen and Dev rows."""
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(here.parent/'pi05_memory')]
    rows=_load_key2_rows(run_root)
    outroot=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
    selection=json.loads((outroot/'compress-seen-cuts-01/production-control/evaluation-selection.json').read_text())
    ref=np.load(outroot/'pi05-reference-01/selection.npz',allow_pickle=False)
    pairs={tuple(map(int,x)) for x in selection['seen_demo_pairs']}
    # Recreate the exact 4,516 CMP3 variable DCT3 units from the run's memory rows.
    import seg_context_retrieval as sc
    import gripper_split as grip
    import motion_core_eval as motion
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    pool,meta,raw_pool=grip._variable_pool(rows,memory_ids)
    if len(meta)!=4516: raise AssertionError(f"expected CMP3 pool 4,516, got {len(meta)}")
    trace_map=grip._traces(rows,memory_ids);base_frames={}
    for (t,e),ids in trace_map.items():base_frames[(int(t),int(e))]=int(rows['frame'][ids[0]])
    coeff=[];mags=[];lengths=[];source=[];values=[];natural=[];unit_keys=[];starts_local=[];rowids=[];startvel=[]
    for i,(t,e,start,length,_g) in enumerate(meta):
        # `grip._variable_pool` already returns the natural-length DCT3
        # reconstruction in physical coordinates. Re-encoding that decoded
        # signal would apply DCT3 twice and break CMP3 oracle parity.
        a=np.asarray(pool[i],np.float32)
        norm=np.column_stack([grip._to_norm(a[:,:6]),a[:,6]]).astype(np.float32)
        c,m,l=sc.segment_value_encode(norm,coefficients=3)
        coeff.append(c);mags.append(m);lengths.append(int(length))
        absstart=base_frames[(int(t),int(e))]+int(start)
        source.append((int(t),int(e),absstart,absstart+int(length)))
        unit_keys.append((int(t),int(e),absstart));starts_local.append(int(start))
        values.append(grip._resample10(norm).astype(np.float32))
        natural.append(norm.astype(np.float32))
        epids=trace_map[(int(t),int(e))]
        rowids.append(int(epids[min(len(epids)-1,int(start)//10)]))
        startvel.append(grip._to_physical(np.asarray(raw_pool[i],np.float32)[:1,:6])[0,:3])
    natural_width=max(map(len,natural));natural_actions=np.zeros((len(natural),natural_width,7),np.float32)
    for i,a in enumerate(natural):natural_actions[i,:len(a)]=a
    bank={'coefficients':np.asarray(coeff,np.float32),'magnitude':np.asarray(mags,np.float32),
        'length':np.asarray(lengths,np.int32),'task':meta[:,0].astype(np.int32),'episode':meta[:,1].astype(np.int32),
        'start':np.asarray(starts_local,np.int32),'source':np.asarray(source,np.int32),
        'row_id':np.asarray(rowids,np.int64),'start_velocity':np.asarray(startvel,np.float32),
        'value_actions10':np.asarray(values,np.float32),'natural_actions':natural_actions}
    # Natural-length decoded CMP3 DCT3 unit values, resampled at the query window.
    keys=unit_keys
    feats=ExactFeatureStore(run_root,missing_root,seen_stride5,unseen_stride5)
    vjepa=ExactVJEPAFeatureStore(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/vjepa'),vjepa_eval_cache)
    bt,bg=feats.get(keys)
    hist=[];hmask=[]
    exact_states=_exact_start_states(keys,rows,run_root,missing_root)
    for task,episode,frame in keys:
        h,m,_,_=_episode_history(rows,task,episode,frame);hist.append(h);hmask.append(m)
    bank_input={'key':np.asarray(keys,np.int32),'history':np.asarray(hist,np.float32),
                'history_mask':np.asarray(hmask,bool),'state':exact_states,
                'tokens':bt,'groups':bg}
    bank_input['vjepa_tokens'],bank_input['vjepa_groups']=vjepa.get(keys,domain='libero40')

    # Training anchors are memory-demo unit starts, excluding every fixed learned-eval episode.
    train_ids=np.asarray([i for i,(t,e,_,_) in enumerate(bank['source']) if (int(t),int(e)) not in pairs],np.int64)
    if set(zip(bank['task'][train_ids].tolist(),bank['episode'][train_ids].tolist())) & pairs:
        raise AssertionError('fixed learned-evaluation episodes leaked into memory-query training')
    train={k:v[train_ids] for k,v in bank_input.items()}
    train_actions=[];train_task=[];train_ep=[]
    for t,e,s,l in bank['source'][train_ids]:
        h,m,st,future=_episode_history(rows,int(t),int(e),int(s))
        arr=np.zeros((10,7),np.float32);arr[:min(10,len(future))]=future[:10]
        if len(future)<10 and len(future):arr[len(future):]=future[-1]
        train_actions.append(arr);train_task.append(t);train_ep.append(e)
    train.update(actions=np.asarray(train_actions),task=np.asarray(train_task),episode=np.asarray(train_ep))
    if len(train_ids)<1000: raise AssertionError('unexpectedly small memory-only fit pool')
    # Fixed learned-evaluation demos are excluded from both training queries
    # and the candidate set used to form their action-distance teacher. They
    # remain in the full bank for the locked learned/unseen retrieval runs.
    fit_candidate_ids=_training_candidate_ids(bank['task'],bank['episode'],pairs)
    if set(zip(bank['task'][fit_candidate_ids].tolist(),bank['episode'][fit_candidate_ids].tolist())) & pairs:
        raise AssertionError('fixed learned-evaluation candidates leaked into training teacher bank')
    fit_bank={k:(v[fit_candidate_ids] if isinstance(v,np.ndarray) and len(v)==len(bank['length']) else v)
              for k,v in bank.items()}
    fit_bank_input={k:(v[fit_candidate_ids] if isinstance(v,np.ndarray) and len(v)==len(bank['length']) else v)
                    for k,v in bank_input.items()}

    learned_ids=np.asarray(ref['learned_row'],np.int64)
    if len(learned_ids)!=621 or int(np.asarray(ref['learned_hard']).sum())!=125: raise AssertionError('fixed learned selection changed')
    fixed_rows=np.asarray(selection['seen_row_ids'],np.int64)
    if len(fixed_rows)!=621 or not np.array_equal(np.sort(learned_ids),np.sort(fixed_rows)):
        raise AssertionError('PI0REF learned rows differ from CMP3 fixed evaluation-selection.json')
    learned_keys=[(int(rows['task'][i]),int(rows['episode'][i]),int(rows['frame'][i])) for i in learned_ids]
    learned_hist=[];learned_masks=[];learned_state=[];learned_action=[]
    for i,k in zip(learned_ids,learned_keys,strict=True):
        h,m,s,future=_episode_history(rows,*k);learned_hist.append(h);learned_masks.append(m);learned_state.append(np.asarray(rows['state'][i,:8],np.float32));learned_action.append(np.asarray(rows['actions'][i,:10,:7],np.float32))
    learned={'key':np.asarray(learned_keys,np.int32),'history':np.asarray(learned_hist),
        'history_mask':np.asarray(learned_masks),'state':np.asarray(learned_state),'actions':np.asarray(learned_action),
        'task':np.asarray(ref['learned_task']),'episode':np.asarray(ref['learned_episode']),
        'hard':np.asarray(ref['learned_hard']), 'row':learned_ids}

    # Fixed unseen 619 rows and PI0REF ordering are joined by their recorded IDs, never resampled.
    from seg_context_run import build_unseen_queries
    unseen_store,unseen_all=build_unseen_queries(unseen_stride5,split='eval',official_stride10=True)
    # Use the selected task/episode/frame arrays in selection.npz as identity keys.
    keys_u=list(zip(ref['unseen_task'].tolist(),ref['unseen_episode'].tolist(),ref['unseen_frame'].tolist()))
    # unseen query keys contain row indexes; the source cache is the same fixed
    # stride-5 extraction and retains exact (task,episode,frame) identities.
    usroot=Path(unseen_stride5); ua={n:np.load(usroot/f'{n}.npy',mmap_mode='r',allow_pickle=False) for n in ('task','episode','frame','actions','state','is_dev')}
    ulookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ua['task'],ua['episode'],ua['frame']))}
    if any(tuple(map(int,k)) not in ulookup for k in keys_u): raise KeyError('fixed unseen keys not present in stride5 token cache')
    ix=np.asarray([ulookup[tuple(map(int,k))] for k in keys_u],np.int64)
    # `build_unseen_queries` has already rebuilt the action stream and masks
    # causally for this exact stride-5 cache. Join by row identity.
    # The builder's key is its source-cache row index; recover absolute frame
    # from that index rather than assuming row order matches PI0REF.
    q_by_identity={}
    for j,rowid in enumerate(unseen_all['key'][:,0]):
        q_by_identity[(int(unseen_all['task'][j]),int(unseen_all['episode'][j])-1_000_000,
                        int(ua['frame'][int(rowid)]))]=j
    qix=np.asarray([q_by_identity[tuple(map(int,k))] for k in keys_u],np.int64)
    uh=[];um=[]
    for j in qix:
        # For fixed Eval queries use the exact causal trace assembled by
        # build_unseen_queries. history_actions is the last completed segment
        # and can be much older than the immediately preceding ten actions.
        h,m=_last10(unseen_all['history_stream_actions'][j],unseen_all['history_stream_mask'][j]);uh.append(h);um.append(m)
    unseen={'key':np.asarray(keys_u,np.int32),'history':np.asarray(uh,np.float32),
        'history_mask':np.asarray(um,bool),'state':np.asarray(ua['state'][ix,:8],np.float32),
        'actions':np.asarray(ua['actions'][ix,:10,:7],np.float32),'task':np.asarray(ref['unseen_task']),
        'episode':np.asarray(ref['unseen_episode'],np.int64)+1_000_000,'hard':np.asarray(ref['unseen_hard']),'row':ix}
    # Dev-only setting selection uses the already-tokenized 217 fixed Dev rows, not Eval labels.
    dev_store,dev_all=build_unseen_queries(unseen_stride5,split='dev')
    devix=np.flatnonzero(np.asarray(ua['is_dev'],bool));devkeys=[(int(ua['task'][i]),int(ua['episode'][i]),int(ua['frame'][i])) for i in devix]
    dev_by_identity={}
    for j,rowid in enumerate(dev_all['key'][:,0]):
        dev_by_identity[(int(dev_all['task'][j]),int(dev_all['episode'][j])-1_000_000,
                         int(ua['frame'][int(rowid)]))]=j
    dix=np.asarray([dev_by_identity[k] for k in devkeys],np.int64)
    dh=[];dm=[]
    for j in dix:
        h,m=_last10(dev_all['history_stream_actions'][j],dev_all['history_stream_mask'][j]);dh.append(h);dm.append(m)
    dev={'key':np.asarray(devkeys,np.int32),'history':np.asarray(dh,np.float32),
        'history_mask':np.asarray(dm,bool),
        'state':np.asarray(ua['state'][devix,:8],np.float32),'actions':np.asarray(ua['actions'][devix,:10,:7],np.float32),
        'task':np.asarray(ua['task'][devix]),'episode':np.asarray(ua['episode'][devix],np.int64)+2_000_000,
        'hard':np.zeros(len(devix),bool),'row':devix}
    for name, split in [('learned',learned),('unseen',unseen),('dev',dev)]:
        split['tokens'],split['groups']=feats.get(split['key'],domain='libero90' if name!='learned' else 'libero40')
        split['vjepa_tokens'],split['vjepa_groups']=vjepa.get(split['key'],domain='libero90' if name!='learned' else 'libero40')
    _assert_eval_candidates_disjoint(learned,bank)
    _assert_eval_candidates_disjoint(unseen,bank)
    _assert_eval_candidates_disjoint(dev,bank)
    # Fixed-normal policy predictions and row alignment are parity checked.
    reflearn=np.load(outroot/'pi05-reference-01/predictions/learned.npz',allow_pickle=False)
    refun=np.load(outroot/'pi05-reference-01/predictions/unseen_cached.npz',allow_pickle=False)
    for split,pred,label in ((learned,reflearn,'learned'),(unseen,refun,'unseen')):
        if not np.array_equal(split['actions'],pred['target']):
            raise AssertionError(f'{label} fixed target rows do not match PI0REF selection ordering')
        split['pi05']=np.asarray(pred['actions'],np.float32)
        split['hard']=np.asarray(pred['hard'],bool)
    return bank,bank_input,fit_bank,fit_bank_input,train,learned,unseen,dev


def _history_corrupt(h, mask, rng):
    """Requested schedule: 25% zero, 25% other-row, 15% reversed XYZ, 35% true."""
    h=np.asarray(h,np.float32).copy();mask=np.asarray(mask,bool).copy();n=len(h)
    modes=rng.choice(4,size=n,p=[.25,.25,.15,.35])
    zero=modes==0;h[zero]=0;mask[zero]=False
    swap=np.flatnonzero(modes==1)
    if len(swap) and n>1:
        for i in swap:
            j=int(rng.integers(n-1));j+=int(j>=i)
            h[i]=h[j].copy();mask[i]=mask[j].copy()
    rev=modes==2;h[rev,:,:3]*=-1
    return h,mask


def _pool_tokens(tok, groups):
    # Cache arrays are large fp16 memmaps (roughly 10 GB for the memory bank).
    # Reduce them in modest blocks rather than materializing a full float32
    # copy.
    tok=np.asarray(tok)
    valid=np.asarray(groups)>=0
    if tok.ndim!=3 or valid.shape!=tok.shape[:2]:
        raise ValueError("token cache and group mask shapes do not align")
    out=np.empty((len(tok),tok.shape[-1]),np.float32)
    for start in range(0,len(tok),64):
        end=min(start+64,len(tok))
        count=np.maximum(valid[start:end].sum(1,keepdims=True),1)
        block=np.where(valid[start:end,...,None],tok[start:end],0)
        out[start:end]=block.sum(1,dtype=np.float32)/count
    return out


def _new_model(mode, token_count=5):
    if mode in ('shared','separate') or mode in INPUT_CONDITIONS:
        input_mode='e_vlm_h' if mode in ('shared','separate') else mode
        encoder=SharedLatentEncoder(token_dim=2048,context_dim=256,token_count=token_count,
            history_steps=10,history_action_dim=7,state_dim=8,group_count=6,heads=8,input_mode=input_mode)
        if mode=='shared' or mode in INPUT_CONDITIONS:return encoder
        return nn.ModuleDict({'query':encoder,'key':TypeKeyEncoder(60,256,token_count)})
    if mode=='p0':return VectorRanker(token_dim=2048,context_dim=256,use_history=False,key_input_dim=2048,token_count=token_count)
    if mode=='p5':return VectorRanker(token_dim=2048,context_dim=256,use_history=True,key_input_dim=2048+10*7+10+8,token_count=token_count)
    raise ValueError(f'unknown model mode {mode}')


def _shared_encode(module, part, device):
    import torch
    def as_tensor(name,dtype):
        value=part.get(name)
        return None if value is None else torch.as_tensor(value,device=device,dtype=dtype)
    tok=as_tensor('tokens',torch.float32);grp=as_tensor('groups',torch.long)
    vj=as_tensor('vjepa_tokens',torch.float32);vjg=as_tensor('vjepa_groups',torch.long)
    hist=as_tensor('history',torch.float32);hmask=as_tensor('history_mask',torch.bool);state=as_tensor('state',torch.float32)
    if isinstance(module,nn.ModuleDict):
        return module['query'](tok,grp,hist,hmask,state,vj,vjg)
    if isinstance(module,SharedLatentEncoder):return module(tok,grp,hist,hmask,state,vj,vjg)
    pooled=torch.as_tensor(part['pooled'],device=device,dtype=torch.float32)
    return module.query(pooled,hist,hmask,state)


def _type_features(actions):
    a=np.asarray(actions,np.float32)[...,:6]
    # Separate-key control receives saved command values at the same native
    # 10-step selection interval; do not normalize away their magnitude.
    return a.reshape(len(a),-1).astype(np.float32)


def _model_batch_encode(model,mode,part,device,*,key=False):
    import torch
    if mode=='shared' or mode in INPUT_CONDITIONS:
        return _shared_encode(model,part,device)
    if mode=='separate':
        if key:
            return model['key'](torch.as_tensor(part['type_features'],device=device,dtype=torch.float32))
        return _shared_encode(model,part,device)
    pooled=torch.as_tensor(part['pooled'],device=device,dtype=torch.float32)
    history=torch.as_tensor(part['history'],device=device,dtype=torch.float32)
    hmask=torch.as_tensor(part['history_mask'],device=device,dtype=torch.bool)
    state=torch.as_tensor(part['state'],device=device,dtype=torch.float32)
    if key and mode=='p5':
        x=torch.cat([pooled,history.flatten(1),hmask.float(),state],-1)
        return model.key(x)
    if key:return model.key(pooled)
    return model.query(pooled,history,hmask,state)


def _pack_part(part,indices,pooled=None,type_features=None):
    result={k:np.asarray(v)[indices] for k,v in part.items()
            if isinstance(v,np.ndarray) and len(v)==len(part['key'])}
    if pooled is not None:
        x=np.asarray(pooled);result['pooled']=x if len(x)==len(indices) else x[indices]
    if type_features is not None:
        x=np.asarray(type_features);result['type_features']=x if len(x)==len(indices) else x[indices]
    return result


def _train_ranker(mode, bank, bank_input, train, *, seed, lr, epochs, device,
                  steps_per_epoch=None, max_train_queries=None, token_count=5,
                  objective="soft", random_candidate_count=256, teacher_temperature=.005):
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    model=_new_model(mode,token_count).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    n=len(train['key']);rng=np.random.default_rng(seed+101)
    qactions=train['actions'];cactions=_candidate_window_actions(np.arange(len(bank['length'])),bank)
    # Full train queries are unit starts; the optional cap is reserved for smoke only.
    qids=np.arange(n)
    if max_train_queries is not None:qids=qids[:int(max_train_queries)]
    steps=int(steps_per_epoch or max(1,int(np.ceil(len(qids)/8))))
    # The shared and separate-key models consume full tokens or type values;
    # only pooled baselines need the extra bank-wide reduction.
    mem_pooled=_pool_tokens(bank_input['tokens'],bank_input['groups']) if mode in ('p0','p5') else None
    mem_type=_type_features(cactions)
    if len(train['key'])>len(bank_input['key']) or not np.array_equal(
            np.asarray(train['key']),np.asarray(bank_input['key'])[:len(train['key'])]):
        raise AssertionError("memory training queries and candidate keys must share exact unit-start rows")
    if objective not in ("soft","hard"):raise ValueError("objective must be soft or hard")
    histlog=[]
    for ep in range(int(epochs)):
        model.train();rng.shuffle(qids);losses=[]
        for step in range(steps):
            qi=qids[(step*8)%len(qids):min((step+1)*8,len(qids))]
            if len(qi)<1:continue
            # Contrastive set C: the other keys in this query minibatch plus
            # uniformly sampled memory units. Labels affect the teacher only,
            # never candidate generation.
            nbank=len(bank['length'])
            random_ids=rng.choice(nbank,size=min(int(random_candidate_count),nbank),replace=False)
            cids,allowed=contrastive_candidate_set(qi,random_ids,train['task'][qi],train['episode'][qi],
                bank['task'],bank['episode'])
            qpart=_pack_query_observation(train,qi,pooled=_pool_tokens(train['tokens'][qi],train['groups'][qi]))
            qpart['history'],qpart['history_mask']=_history_corrupt(qpart['history'],qpart['history_mask'],rng)
            kpart=_pack_query_observation(bank_input,cids,pooled=mem_pooled[cids] if mem_pooled is not None else None,
                                          type_features=mem_type[cids])
            # Float16 cache inputs are converted only for the active minibatch.
            qrep=_model_batch_encode(model,mode,qpart,device)
            krep=_model_batch_encode(model,mode,kpart,device,key=(mode=='separate'))
            if qrep.ndim==3:
                score=(qrep[:,None]*krep[None]).sum(-1).mean(-1)
            else:score=qrep@krep.T
            score=_scale_logits(model,score)
            emat=torch.as_tensor(_arm6_candidate_mse(qactions[qi],cactions[cids]),device=device)
            amask=torch.as_tensor(allowed,device=device)
            loss=contrastive_teacher_loss(score,emat,amask,objective=objective,
                                          teacher_temperature=teacher_temperature)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            _clamp_logit_scale_(model)
            losses.append(float(loss.detach()))
        histlog.append({'epoch':ep+1,'loss':float(np.mean(losses)) if losses else None,'steps':len(losses),
                        'objective':objective,'teacher_temperature':float(teacher_temperature),
                        'candidate_rule':'minibatch memory-start keys + uniform memory units; self-episode masked'})
    return model,histlog


def _encode_candidates(model,mode,bank_input,bank,device,batch=12):
    import torch
    pooled=_pool_tokens(bank_input['tokens'],bank_input['groups']) if mode in ('p0','p5') else None
    typef=_type_features(_candidate_window_actions(np.arange(len(bank['length'])),bank))
    reps=[];model.eval()
    with torch.no_grad():
        for s in range(0,len(bank['length']),batch):
            ix=np.arange(s,min(len(bank['length']),s+batch));part=_pack_query_observation(
                bank_input,ix,pooled=pooled[ix] if pooled is not None else None,type_features=typef[ix])
            reps.append(_model_batch_encode(model,mode,part,device,key=(mode=='separate')).cpu().numpy())
    return np.concatenate(reps),pooled,typef


def _query_representations(model,mode,split,device,batch=8):
    import torch
    pooled=_pool_tokens(split['tokens'],split['groups']) if mode in ('p0','p5') else None
    reps=[];model.eval()
    with torch.no_grad():
        for s in range(0,len(split['key']),batch):
            ix=np.arange(s,min(len(split['key']),s+batch));part=_pack_query_observation(
                split,ix,pooled=pooled[ix] if pooled is not None else None)
            reps.append(_model_batch_encode(model,mode,part,device).cpu().numpy())
    return np.concatenate(reps),pooled


def _score_model(model,mode,bank,bank_input,split,device):
    keys,_,_=_encode_candidates(model,mode,bank_input,bank,device)
    q,_=_query_representations(model,mode,split,device)
    if q.ndim==3:scores=np.einsum('qkd,nkd->qn',q,keys,optimize=True)/q.shape[1]
    else:scores=q@keys.T
    scale=float(_logit_scale_parameter(model).detach().exp().clamp(max=LOGIT_SCALE_MAX).cpu())
    scores=scores*scale
    candidate_windows=_candidate_window_actions(np.arange(len(bank['length'])),bank)
    errors=_arm6_candidate_mse(split['actions'],candidate_windows)
    # Fixed learned/unseen demos are disjoint from the memory pool. No query
    # task/episode ID is used to score, filter, or rank candidates.
    return scores.astype(np.float32),errors


def _candidate_window_actions(indices, bank):
    """Read stored action values over one selection interval, holding endpoints."""
    ix=np.asarray(indices,np.int64)
    natural=np.asarray(bank['natural_actions'],np.float32)
    lengths=np.asarray(bank['length'],np.int64)[ix]
    out=np.empty((len(ix),10,natural.shape[-1]),np.float32)
    for i,(candidate,length) in enumerate(zip(ix,lengths,strict=True)):
        if length<1:raise ValueError('unit length must be positive')
        unit=natural[candidate,:int(length)];n=min(int(length),10)
        out[i,:n]=unit[:n]
        if n<10:out[i,n:]=unit[n-1]
    return out


def _metric_block(scores,errors,split,bank,*,seed=4101,random_runs=5,
                  tau=0.01,temperature=0.1,k=1):
    n,m=scores.shape;row=np.arange(n);allowed=np.isfinite(scores)
    if np.any(~allowed.any(axis=1)):
        raise ValueError('each query must have at least one allowed memory candidate')
    top=np.argmax(scores,axis=1)
    top_err=errors[row,top]
    best=np.where(allowed,errors,np.inf).min(axis=1)
    # Candidate sets are fixed/disjoint; no task or episode mask is used here.
    rank_orders=[]
    for j in range(random_runs):
        rng=np.random.default_rng(int(seed+j))
        rank_orders.append(np.stack([rng.permutation(m) for _ in range(n)]))
    # Every candidate is played at its native cadence up to the next 10-step
    # selection point. The selected-window action values are averaged only
    # after this natural playback, never in DCT space.
    pred=_blend_prediction(scores,split,bank,k,temperature)
    action_rows=_action_metric_block(pred,split['actions'])
    candidate_rank=candidate_rank_metrics(scores,errors,tau=tau,temperature=temperature,
                                           random_orders=np.asarray(rank_orders))
    random_rank=random_candidate_metrics(errors,tau=tau,seeds=tuple(seed+j for j in range(random_runs)))
    oracle_rank=oracle_candidate_metrics(errors,tau=tau)
    result={'top_index':top,'score':scores,'arm6_errors':errors,'selected_actions':pred,
        'rank':candidate_rank,'random_rank':random_rank,'oracle_rank':oracle_rank,
        'actions':{k:float(v.mean()) for k,v in action_rows.items()}}
    rr=[]
    seeds=[int(seed+i) for i in range(random_runs)]
    for r,random_seed in enumerate(seeds):
        rng=np.random.default_rng(random_seed)
        picks=np.asarray([rng.choice(np.flatnonzero(allowed[i])) for i in range(n)])
        rr.append(picks)
    result['random_indices']=np.asarray(rr)
    random_type=np.stack([errors[row,p] for p in rr])
    random_preds=np.stack([_candidate_window_actions(picks,bank) for picks in rr])
    random_actions=[action_errors(random_preds[j],split['actions']) for j in range(len(rr))]
    result['random_arm6_mse']=float(random_type.mean())
    result['random_arm6_mse_seed_mean']=random_type.mean(0)
    result['random_actions_seed_mean']={k:np.mean(np.stack([x[k] for x in random_actions]),axis=0) for k in ('arm6','mse_gripper_truth','active7','active7_saved_gripper')}
    result['random_actions_seed_sd']={k:float(np.std([np.mean(x[k]) for x in random_actions])) for k in ('arm6','active7')}
    result['random_playback_actions']=random_preds
    result['random_seeds']=seeds
    # Oracle selects by the same raw saved-value arm6 MSE and uses the same
    # one-interval natural playback as learned ranking methods.
    action_error_matrix=errors
    result['oracle_action_index']=np.where(allowed,action_error_matrix,np.inf).argmin(axis=1)
    oi=result['oracle_action_index']
    oracle_window=_candidate_window_actions(oi,bank)
    result['querywise_oracle_actions']={k:float(np.asarray(v).mean()) for k,v in action_errors(oracle_window,split['actions']).items()}
    oracle_play=oracle_window
    result['oracle_playback_actions']=oracle_play
    result['oracle_playback_metrics']={k:float(np.asarray(v).mean()) for k,v in action_errors(oracle_play,split['actions']).items()}
    return result


def natural_unit_replay(selected,split,bank):
    """Replay selected variable-length decoded units, then return 10-step rows."""
    selected=np.asarray(selected,np.int64);actions=np.asarray(split['actions'],np.float32)
    if selected.shape!=(len(actions),):raise ValueError('one selected candidate is required per query row')
    out=np.empty_like(actions);keys=np.asarray(split['key'],np.int64)
    tasks=np.asarray(split['task']);episodes=np.asarray(split['episode'])
    identities=list(dict.fromkeys(zip(tasks.tolist(),episodes.tolist(),strict=True)))
    for task,episode in identities:
        ids=np.flatnonzero((tasks==task)&(episodes==episode))
        ids=ids[np.argsort(keys[ids,2],kind='stable')]
        starts=keys[ids,2]-keys[ids[0],2]
        if len(ids)>1 and not np.all(np.diff(starts)==10):
            raise AssertionError(f'natural replay requires contiguous stride-10 target rows: {(task,episode)}')
        target=np.asarray(actions[ids],np.float32).reshape(-1,actions.shape[-1]);play=np.empty_like(target)
        cursor=0;last_q=0
        while cursor<len(play):
            # A retrieval at frame s is unavailable before s. A left-sided
            # search selected the next query when a natural unit ended between
            # stride-10 observations, leaking a future query into playback.
            qpos=max(0,min(int(np.searchsorted(starts,cursor,side='right'))-1,len(ids)-1))
            if qpos<last_q:qpos=last_q
            candidate=int(selected[ids[qpos]]);length=int(bank['length'][candidate])
            if length<1:raise AssertionError('empty DCT3 unit')
            unit=bank['natural_actions'][candidate,:length]
            n=min(length,len(play)-cursor);play[cursor:cursor+n]=unit[:n]
            cursor+=n;last_q=qpos
        out[ids]=play.reshape(len(ids),actions.shape[1],actions.shape[2])
    return out


def _action_metric_block(pred, target):
    """Per-row action metrics for predictions aligned to query rows."""
    pred=np.asarray(pred,np.float32);target=np.asarray(target,np.float32)
    err=action_errors(pred,target)
    axis={f'axis_{j}_mse_over7':np.square(pred[...,j]-target[...,j]).mean(1)/7 for j in range(6)}
    axis['gripper_mse_over7']=np.square(pred[...,6]-target[...,6]).mean(1)/7
    return {k:np.asarray(v) for k,v in {**err,**axis}.items()}


def cmp3_natural_dct3_oracle(split,bank):
    """Re-run CMP3's natural-duration sequence-level oracle on fixed rows."""
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path:sys.path.insert(0,str(here))
    import compression_matched as cmp3
    traces={};rows_of={}
    task=np.asarray(split['task']);episode=np.asarray(split['episode']);keys=np.asarray(split['key'])
    for t,e in list(dict.fromkeys(zip(task.tolist(),episode.tolist(),strict=True))):
        ids=np.flatnonzero((task==t)&(episode==e));ids=ids[np.argsort(keys[ids,2],kind='stable')]
        traces[(int(t),int(e))]=np.asarray(split['actions'][ids],np.float32).reshape(-1,7)
        rows_of[(int(t),int(e))]=ids
    items=[np.asarray(bank['natural_actions'][i,:int(length)],np.float32)
           for i,length in enumerate(bank['length'])]
    had_target=hasattr(cmp3.ev.Shared,'target')
    old=getattr(cmp3.ev.Shared,'target',None)
    try:
        cmp3.ev.Shared.target=np.asarray(split['actions'],np.float32)
        pred,_=cmp3.score_variable_memory(items,bank['length'],np.arange(len(items)),traces,rows_of)
    finally:
        if had_target:cmp3.ev.Shared.target=old
        else:delattr(cmp3.ev.Shared,'target')
    errs=action_errors(pred,split['actions']);hard=np.asarray(split['hard'],bool)
    return {'all':{k:float(np.mean(v)) for k,v in errs.items()},
            'hard':{k:float(np.mean(v[hard])) for k,v in errs.items()} if hard.any() else None}


def dim3_arm6_dct3_oracle_parity(locked_splits, *, atol=2e-7):
    """Recompute DIM3's arm6 variable-DCT3 natural-length sequence DP.

    This is a distinct, future-aware reference upper bound, not the main
    causal same-playback oracle. It uses DIM3's exact segmentation, candidate
    pool, DP, and fixed evaluation traces and asserts against its saved row.
    """
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path:sys.path.insert(0,str(here))
    import compression_per_axis_segmentation as dim3
    d,evals=dim3.cmp3.prepare()
    pool=dim3.candidate_pools(d)['G1_variable_dct3']
    items,lengths,_=pool
    if len(items)!=4516:raise AssertionError(f'DIM3 candidate count changed: {len(items)}')
    saved=json.loads(dim3.OUTPUT.joinpath('results.json').read_text())
    got={}
    for name in ('learned','unseen'):
        data=evals[name]
        # Exact targets and masks validate fixed row identity and order;
        # KEY2's builder separately validates those rows against PI0REF keys.
        if not np.array_equal(np.asarray(data['targets']),np.asarray(locked_splits[name]['actions'])):
            raise AssertionError(f'DIM3 {name} target rows/order differ from locked KEY2 rows')
        if not np.array_equal(np.asarray(data['hard']),np.asarray(locked_splits[name]['hard'])):
            raise AssertionError(f'DIM3 {name} hard mask differs from locked KEY2 rows')
        pred=dim3._best_sequence_prediction(data,items,lengths,6)
        row=dim3.per_row_errors(pred,data['targets'])['arm6']
        hard=np.asarray(data['hard'],bool)
        stats={'all':float(row.mean()),'hard':float(row[hard].mean()),'rows':int(len(row)),
               'hard_rows':int(hard.sum())}
        ref=saved['distributions']['G1_variable_dct3'][name]['arm6']
        for subset,key in (('all','all'),('hard','raw_memory_q80')):
            expected=float(ref[key]['arm6']['mean'])
            if not np.isclose(stats[subset],expected,atol=atol,rtol=0):
                raise AssertionError(f'DIM3 arm6 DP parity {name}/{subset}: {stats[subset]} != {expected}')
        got[name]=stats
    return got


BLEND_KS=(1,5,20)
BLEND_TEMPERATURES=(.01,.03,.05,.1,.2,.5,1.0)
TEACHER_TEMPERATURES=(.002,.005,.01,.02)
REPORT_RANKING_TAU=.01
LOGIT_SCALE_INITIAL=1.0/.07
LOGIT_SCALE_MAX=100.0
RANDOM20_BLEND_SEED=92831


def _logit_scale_parameter(model):
    """Return the one learned CLIP-style scale owned by each ranker."""
    if isinstance(model, nn.ModuleDict):
        return model['query'].logit_scale
    return model.logit_scale


def _scale_logits(model, scores):
    """Scale similarities by exp(logit_scale), capped at 100."""
    return scores * _logit_scale_parameter(model).exp().clamp(max=LOGIT_SCALE_MAX)


def _clamp_logit_scale_(model):
    with torch.no_grad():
        _logit_scale_parameter(model).clamp_(max=float(np.log(LOGIT_SCALE_MAX)))


def _hold_last_actions(split):
    """Repeat the last valid strictly-prior action across the next 10 steps."""
    history=np.asarray(split['history'],np.float32)
    mask=np.asarray(split['history_mask'],bool)
    if history.ndim!=3 or history.shape[1:]!=(10,7) or mask.shape!=history.shape[:2]:
        raise ValueError('hold-last expects [N,10,7] history and [N,10] mask')
    out=np.zeros((len(history),10,7),np.float32)
    for i in range(len(history)):
        valid=np.flatnonzero(mask[i])
        if len(valid):out[i]=history[i,valid[-1]]
    return out


def _assert_hold_last_gate(learned, unseen, *, atol=5e-4):
    """Protect exact causal-history semantics and fixed-row hold-last parity."""
    got={}
    for name,split in (('learned',learned),('unseen',unseen)):
        pred=_hold_last_actions(split)
        err=action_errors(pred,split['actions'])['arm6']
        hard=np.asarray(split['hard'],bool)
        got[name]={'all':float(err.mean()),'hard':float(err[hard].mean()) if hard.any() else None,
                   'rows':int(len(err)),'hard_rows':int(hard.sum())}
    expected={'learned':(.05639717563886132,.09724447740896078),
              'unseen':(.113994986,.20097965)}
    for name,(all_ref,hard_ref) in expected.items():
        if not np.isclose(got[name]['all'],all_ref,atol=atol,rtol=0):
            raise AssertionError(f'{name} hold-last all parity failed: {got[name]}')
        if got[name]['hard'] is None or not np.isclose(got[name]['hard'],hard_ref,atol=atol,rtol=0):
            raise AssertionError(f'{name} hold-last hard parity failed: {got[name]}')
    return got


def _blend_prediction(scores,split,bank,k,temperature):
    return natural_topk_blend_actions(scores,bank,k=int(k),temperature=float(temperature))


def _select_dev_blends(scores,split,bank,temperature_grid=BLEND_TEMPERATURES):
    """Select k/temp from Dev action MSE only; return per-k and joint choices."""
    grid=[];by_k={}
    for k in BLEND_KS:
        temps=(1.0,) if k==1 else tuple(map(float,temperature_grid))
        rows=[]
        for temp in temps:
            pred=_blend_prediction(scores,split,bank,k,temp)
            mse=float(np.mean(action_errors(pred,split['actions'])['mse_gripper_truth']))
            item={'k':k,'temperature':temp,'mse_gripper_truth':mse}
            grid.append(item);rows.append(item)
        by_k[k]=min(rows,key=lambda x:(x['mse_gripper_truth'],x['temperature']))
    selected=min(by_k.values(),key=lambda x:(x['mse_gripper_truth'],x['k'],x['temperature']))
    return {'grid':grid,'per_k':{str(k):v for k,v in by_k.items()},'selected':selected}


def _summarize_split(name,scores,errors,split,bank,*,tau,selected_blend,blend_by_k,reps=5000):
    whole=_metric_block(scores,errors,split,bank,tau=tau,temperature=selected_blend['temperature'],
                        k=selected_blend['k'])
    selected_actions=_blend_prediction(scores,split,bank,selected_blend['k'],selected_blend['temperature'])
    blend_rows={}
    blend_predictions={}
    for kstr,cfg in blend_by_k.items():
        k=int(kstr);pred=_blend_prediction(scores,split,bank,k,cfg['temperature'])
        blend_predictions[k]=pred
        allm=action_errors(pred,split['actions'])
        hard=np.asarray(split['hard'],bool)
        hm={a:float(np.mean(v)) for a,v in action_errors(pred[hard],split['actions'][hard]).items()} if hard.any() else {}
        blend_rows[kstr]={'temperature':float(cfg['temperature']),
            'mse_gripper_truth':float(np.mean(allm['mse_gripper_truth'])),
            'hard_mse_gripper_truth':hm.get('mse_gripper_truth')}
    selected_action_rows=_action_metric_block(selected_actions,split['actions'])
    selected_actions_all={k:float(v.mean()) for k,v in selected_action_rows.items()}
    hard=np.asarray(split['hard'],bool)
    if hard.any():
        hsplit={k:(v[hard] if isinstance(v,np.ndarray) and len(v)==len(hard) else v) for k,v in split.items()}
        hm=_metric_block(scores[hard],errors[hard],hsplit,bank,tau=tau,
                         temperature=selected_blend['temperature'],k=selected_blend['k'])
        hm['selected_actions']=selected_actions[hard]
        hard_action_rows=_action_metric_block(hm['selected_actions'],split['actions'][hard])
        hm['actions']={k:float(v.mean()) for k,v in hard_action_rows.items()}
        hm['random_actions_seed_mean']={k:v[hard] for k,v in whole['random_actions_seed_mean'].items()}
    else:hm=None
    # Baselines use the same selected fixed rows and target arrays.
    last=_hold_last_actions(split)
    base={'hold_last':action_errors(last,split['actions'])}
    zero=np.zeros_like(split['actions'],dtype=np.float32)
    base['zero']=action_errors(zero,split['actions'])
    # A uniform blend of twenty randomly selected natural action windows is a
    # reference for error reduction caused only by averaging.
    rng20=np.random.default_rng(RANDOM20_BLEND_SEED)
    random20=np.empty_like(split['actions'],dtype=np.float32)
    candidates=_candidate_window_actions(np.arange(len(bank['length'])),bank)
    allowed=np.isfinite(scores)
    for i in range(len(random20)):
        available=np.flatnonzero(allowed[i])
        chosen=rng20.choice(available,size=min(20,len(available)),replace=False)
        random20[i]=candidates[chosen].mean(axis=0)
    base['random20_blend']=action_errors(random20,split['actions'])
    if 'pi05' in split:base['pi05']=action_errors(split['pi05'],split['actions'])
    base_hard={k:{a:float(np.mean(v[hard])) for a,v in vals.items()} for k,vals in base.items() if hard.any()}
    if 'pi05' in split:
        whole['actions']['active7_pi05_grip']=float(np.mean(action_errors(selected_actions,split['actions'],
            pi05_grip=split['pi05'][...,6])['active7_pi05_grip']))
        if hm is not None:
            hm['actions']['active7_pi05_grip']=float(np.mean(action_errors(hm['selected_actions'],split['actions'][hard],
                pi05_grip=split['pi05'][hard,...,6])['active7_pi05_grip']))
    random_all={k:float(np.mean(v)) for k,v in whole['random_actions_seed_mean'].items()}
    random_hard={}
    if hm is not None:
        for metric in ('arm6','mse_gripper_truth','active7','active7_saved_gripper'):
            # These querywise errors were measured after natural-duration
            # replay for each random draw; preserve that protocol for hard rows.
            random_hard[metric]=float(np.mean(whole['random_actions_seed_mean'][metric][hard]))
    oracle_hard=None
    if hard.any():
        oracle_hard={k:float(np.mean(v)) for k,v in action_errors(
            whole['oracle_playback_actions'][hard],split['actions'][hard]).items()}
    top1=_metric_block(scores,errors,split,bank,tau=tau,temperature=selected_blend['temperature'],k=1)
    top1_hard=None
    if hard.any():
        hsplit={k:(v[hard] if isinstance(v,np.ndarray) and len(v)==len(hard) else v) for k,v in split.items()}
        top1_hard=_metric_block(scores[hard],errors[hard],hsplit,bank,tau=tau,
                                temperature=selected_blend['temperature'],k=1)
    return {'name':name,'n':len(hard),'hard_n':int(hard.sum()),'rank_all':whole['rank'],
        'random_rank':whole['random_rank'],'oracle_rank':whole['oracle_rank'],
        'selected_blend':selected_blend,'blend_by_k':blend_rows,
        'actions_all':selected_actions_all,'top1_actions_all':top1['actions'],
        'rank_hard':hm['rank'] if hm else None,
        'random_rank_hard':hm['random_rank'] if hm else None,'oracle_rank_hard':hm['oracle_rank'] if hm else None,
        'actions_hard':{k:float(v[hard].mean()) for k,v in selected_action_rows.items()} if hard.any() else None,
        'top1_actions_hard':top1_hard['actions'] if top1_hard else None,'random5_actions_all':random_all,
        'random5_actions_hard':random_hard,'random5_arm6_mse':whole['random_arm6_mse'],
        'random5_actions_seed_sd':whole['random_actions_seed_sd'],'baselines':{k:{a:float(np.mean(v[a])) for a in v}
            for k,v in base.items()},'baselines_hard':base_hard,
        'querywise_oracle_all':whole['querywise_oracle_actions'],
        'oracle_playback_all':whole['oracle_playback_metrics'],'oracle_playback_hard':oracle_hard,
        'ranking_reference_tau':float(tau),'_selected_actions':selected_actions,'_blend_predictions':blend_predictions,
        '_whole':whole,'_hard':hm}


def _paired_bootstrap(diff,task,episode,*,reps=5000,seed=4101):
    return episode_bootstrap_ci(np.asarray(diff),np.asarray(task),np.asarray(episode),reps=reps,seed=seed)


def _without_private(value):
    if isinstance(value,dict):return {k:_without_private(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,list):return [_without_private(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.floating,np.integer,np.bool_)):return value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _input_fields_for_mode(mode):
    if mode in INPUT_CONDITIONS:
        vlm,vj,hist,state=INPUT_CONDITIONS[mode]
        return ((["VLM tokens (current image and instruction)","VLM group IDs"] if vlm else [])+
                (["V-JEPA base/wrist tokens (current image pair)","V-JEPA camera IDs"] if vj else [])+
                (["strictly prior executed-action history","history-valid mask"] if hist else [])+
                (["current state"] if state else []))
    if mode in ("shared","separate"):return ["VLM tokens","VLM group IDs","strictly prior executed-action history","history-valid mask","current state"]
    if mode=="p0":return ["mean VLM tokens"]
    if mode=="p5":return ["mean VLM tokens","strictly prior executed-action history","history-valid mask","current state"]
    return []


def run_experiment(output: Path, *, mode='all', seed=1, learning_rate=3e-4,
                   epochs=2, steps_per_epoch=None, smoke=False,
                   dev_only=False, objective="soft", teacher_temperature=.005, vjepa_eval_cache=None,
                   eval_only_checkpoint=None,
                   baseline_json=None,
                   run_root=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'),
                   missing_root=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'),
                   seen_stride5=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache'),
                   unseen_stride5=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'),
                   device=None):
    """End-to-end experiment. Eval row IDs are frozen by CMP3/PI0REF manifests."""
    # Keep PyTorch's internal pools aligned with the requested per-process CPU
    # allocation used by the launcher.
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.20')))
    if vjepa_eval_cache is None:raise ValueError("vjepa_eval_cache is required; first prepare it in a fresh attempt")
    bank,bank_input,fit_bank,fit_bank_input,train,learned,unseen,dev=_build_key2_queries(run_root,missing_root,seen_stride5,unseen_stride5,vjepa_eval_cache)
    hold_last_parity=_assert_hold_last_gate(learned,unseen)
    full_unit_count=len(bank['length'])
    if len(bank['length'])!=4516:raise AssertionError('CMP3 DCT3 pool cardinality mismatch')
    baseline_doc=json.loads(Path(baseline_json).read_text()) if baseline_json is not None else None
    # Expensive fixed-row natural-duration DP and full-candidate oracle checks
    # are done once by the CPU-reference stage (and revalidated by the smoke),
    # never redundantly by every parallel Dev worker or final method process.
    if dev_only:
        dim3_arm6_reference=None
        querywise_oracle_all=querywise_oracle_hard=None
    elif baseline_doc is not None and 'reference_diagnostics' in baseline_doc:
        diag=baseline_doc['reference_diagnostics']
        dim3_arm6_reference=diag['dim3_arm6_natural_dp']
        querywise_oracle_all=float(diag['querywise_best_arm6']['unseen_all'])
        querywise_oracle_hard=float(diag['querywise_best_arm6']['unseen_hard'])
    else:
        dim3_arm6_reference=dim3_arm6_dct3_oracle_parity({'learned':learned,'unseen':unseen})
        candidate_windows=_candidate_window_actions(np.arange(len(bank['length'])),bank)
        oracle_arm=np.square(unseen['actions'][:,None,:,:6]-candidate_windows[None,:,:,:6]).sum((2,3))/(10*7)
        arm_best=np.min(oracle_arm,axis=1)
        querywise_oracle_all=float(arm_best.mean());querywise_oracle_hard=float(arm_best[unseen['hard']].mean())
    # Exact fixed-row parity and the published CMP3/PI0REF gates.
    pi05_unseen=float(np.square(unseen['pi05']-unseen['actions']).mean())
    pi05_hard=float(np.square(unseen['pi05'][unseen['hard']]-unseen['actions'][unseen['hard']]).mean())
    if not np.isclose(pi05_unseen,.14133905,atol=3e-4) or not np.isclose(pi05_hard,.28695038,atol=5e-4):
        raise AssertionError(f'PI0REF parity mismatch: all={pi05_unseen}, hard={pi05_hard}')
    # The CMP3 number is a natural-duration, sequence-level DP replay oracle,
    # not a per-window nearest-action minimum. Read its frozen row and assert
    # identity/counts; keep the distinct oracle quantities separately labeled.
    cmp3_path=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/compress-seen-cuts-01/production/results.json')
    cmp3_results=json.loads(cmp3_path.read_text())
    cmp3_row=next(x for x in cmp3_results['E1'] if x.get('method')=='anchor_variable_dct' and x.get('units')==4516)
    cmp3_eval=cmp3_row['eval']['unseen']
    cmp3_learned=cmp3_row['eval']['learned']
    cmp3_all=float(cmp3_eval['all_mse']);cmp3_hard=float(cmp3_eval['hard_mse'])
    if cmp3_eval.get('n')!=619 or cmp3_eval.get('hard_n')!=134 or not np.isclose(cmp3_all,.03444789789394605,atol=1e-10) or not np.isclose(cmp3_hard,.06845424465082015,atol=1e-10):
        raise AssertionError(f'frozen CMP3 natural DCT3 reference changed: {cmp3_eval}')
    cmp3_natural_local={}
    if smoke:
        # Full fixed-set oracle is cheap relative to fitting and explicitly
        # replays decoded units at their stored durations, as CMP3 does.
        cmp3_natural_local={'learned':cmp3_natural_dct3_oracle(learned,bank),
                            'unseen':cmp3_natural_dct3_oracle(unseen,bank)}
        for split_name,reference in (('learned',cmp3_learned),('unseen',cmp3_eval)):
            local=cmp3_natural_local[split_name]
            if not np.isclose(local['all']['active7'],reference['all_mse'],atol=2e-6) or not np.isclose(local['hard']['active7'],reference['hard_mse'],atol=2e-6):
                raise AssertionError(f'{split_name} natural DCT3 oracle does not reproduce CMP3: {local} vs {reference}')
    # Only small smoke fixtures keep a copy; production workers share the
    # immutable source bank instead of writing one duplicate archive each.
    if smoke:np.savez_compressed(output/'bank.npz',**{k:v for k,v in bank.items()})
    if smoke:
        # Bound the smoke to a small candidate/train subset, while keeping the
        # exact fixed full-set oracle and PI0REF integrity gates below.
        train={k:(v[:64] if isinstance(v,np.ndarray) and len(v)==len(train['key']) else v) for k,v in train.items()}
        fit_bank={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==len(fit_bank['length']) else v) for k,v in fit_bank.items()}
        fit_bank_input={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==len(fit_bank_input['key']) else v) for k,v in fit_bank_input.items()}
        bank={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==4516 else v) for k,v in bank.items()}
        bank_input={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==4516 else v) for k,v in bank_input.items()}
        learned={k:(v[:16] if isinstance(v,np.ndarray) and len(v)==621 else v) for k,v in learned.items()}
        unseen={k:(v[:16] if isinstance(v,np.ndarray) and len(v)==619 else v) for k,v in unseen.items()}
        dev={k:(v[:24] if isinstance(v,np.ndarray) and len(v)==len(dev['key']) else v) for k,v in dev.items()}
    if mode=='all':modes=['shared','separate','p0','p5']
    elif mode=='all_inputs':modes=list(ALL_INPUT_CONDITIONS)
    elif mode=='all_methods':modes=list(ALL_INPUT_CONDITIONS)+['separate','p0','p5']
    else:modes=[mode]
    if objective=='soft' and not any(np.isclose(float(teacher_temperature),x) for x in TEACHER_TEMPERATURES):
        raise ValueError(f'teacher_temperature must be Dev-grid value in {TEACHER_TEMPERATURES}')
    # Training teacher sharpness is selected on Dev from the fixed grid.
    # Distribution/ranking reports instead use one shared tau=.01.
    training_tau=float(teacher_temperature)
    summaries={};models={}
    for md in modes:
        if eval_only_checkpoint is not None:
            model=_new_model(md).to(device)
            state=torch.load(eval_only_checkpoint,map_location=device,weights_only=True)
            model.load_state_dict(state);hist=[]
        else:
            model,hist=_train_ranker(md,fit_bank,fit_bank_input,train,seed=seed,lr=learning_rate,epochs=epochs,
                device=device,steps_per_epoch=steps_per_epoch or (2 if smoke else None),
                max_train_queries=64 if smoke else None,objective=objective,teacher_temperature=training_tau)
        modelpath=output/f'{md}_seed{seed}.pt';torch.save(model.state_dict(),modelpath)
        model.eval();models[md]=model
        mode_results={}
        # Full-candidate Dev alone selects blend size and score temperature.
        dev_scores,dev_errors=_score_model(model,md,bank,bank_input,dev,device)
        dev_blend=_select_dev_blends(dev_scores,dev,bank)
        dev_rank=candidate_rank_metrics(dev_scores,dev_errors,tau=REPORT_RANKING_TAU,
            temperature=float(dev_blend['selected']['temperature']))
        dev_metric={'rank':dev_rank,'blend_selection':dev_blend,'training_teacher_tau':training_tau,
                    'ranking_reference_tau':REPORT_RANKING_TAU,
                    'logit_scale':float(_logit_scale_parameter(model).detach().exp().clamp(max=LOGIT_SCALE_MAX).cpu())}
        if not dev_only:
            for name,split in [('learned',learned),('unseen',unseen)]:
                scores,errors=_score_model(model,md,bank,bank_input,split,device)
                mode_results[name]=_summarize_split(name,scores,errors,split,bank,tau=REPORT_RANKING_TAU,
                    selected_blend=dev_blend['selected'],blend_by_k=dev_blend['per_k'])
                whole=mode_results[name]['_whole']
                random_actions=whole['random_playback_actions']
                np.savez_compressed(output/f'{md}_{name}_paired_predictions.npz',
                    selected_actions=mode_results[name]['_selected_actions'],
                    top1_actions=mode_results[name]['_blend_predictions'][1],random_actions=random_actions,
                    blend_k1=mode_results[name]['_blend_predictions'][1],
                    blend_k5=mode_results[name]['_blend_predictions'][5],
                    blend_k20=mode_results[name]['_blend_predictions'][20],
                    oracle_playback_actions=whole['oracle_playback_actions'],
                    target_actions=split['actions'],task=split['task'],episode=split['episode'],
                    hard=split['hard'],key=split['key'])
        summaries[md]={'train':{'seed':seed,'learning_rate':learning_rate,'epochs':epochs,'objective':objective,'history':hist,
            'candidate_set':'in-batch memory unit keys plus 256 uniform memory units; self-episode masked',
            'train_queries':len(train['key']),'training_candidate_units':len(fit_bank['length']),
            'retrieval_units':full_unit_count},
            'dev_selection':dev_metric,'splits':mode_results,
            'checkpoint':str(modelpath)}
        del model
        if device=='cuda':torch.cuda.empty_cache()
    if dev_only:
        results={'status':'dev_complete','mode':mode,'objective':objective,'seed':seed,'fixed_selection':{
            'learned_rows':len(learned['key']),'learned_hard':int(learned['hard'].sum()),
            'unseen_rows':len(unseen['key']),'unseen_hard':int(unseen['hard'].sum()),'dev_rows':len(dev['key'])},
            'models':summaries,'training_teacher_temperature':training_tau,
            'selection_split':'Dev labels select teacher tau, learning rate, epochs, k and score temperature; fixed Eval labels are used only for parity and metrics.'}
        results=_without_private(results)
        _atomic_json(output/'dev_results.json',results)
        (output/'report.md').write_text(f"# KEY2 Dev-only run\n\nMode: {mode}; objective: {objective}; Dev rows: {len(dev['key'])}; Dev metric: {summaries[modes[0]]['dev_selection']}\n")
        return results
    # Random-5, last-action, PI0REF, and both oracles on the same fixed rows.
    if baseline_json is not None:
        baseline=baseline_doc['metrics']
    else:
        baseline={}
        for name,split in [('learned',learned),('unseen',unseen)]:
            errors=_arm6_candidate_mse(split['actions'],_candidate_window_actions(np.arange(len(bank['length'])),bank))
            scores=np.zeros_like(errors)
            result=_summarize_split(name,scores,errors,split,bank,tau=REPORT_RANKING_TAU,
                selected_blend={'k':1,'temperature':1.0},blend_by_k={'1':{'k':1,'temperature':1.0}})
            result['rank_all']=result['random_rank'];result['rank_hard']=result['random_rank_hard']
            result['actions_all']=result['random5_actions_all'];result['actions_hard']=result['random5_actions_hard']
            baseline[name]=result
    for splitname,block in (('learned',cmp3_learned),('unseen',cmp3_eval)):
        all_active=(cmp3_natural_local[splitname]['all']['active7'] if cmp3_natural_local else float(block['all_mse']))
        hard_active=(cmp3_natural_local[splitname]['hard']['active7'] if cmp3_natural_local else float(block['hard_mse']))
        baseline[splitname]['baselines']['CMP3-7axis-natural-DP-secondary']={'mse_gripper_truth':None,'arm6':None,'active7':all_active}
        baseline[splitname]['baselines_hard']['CMP3-7axis-natural-DP-secondary']={'mse_gripper_truth':None,'arm6':None,'active7':hard_active}
        baseline[splitname]['baselines']['DIM3-arm6-natural-DP-reference']={
            'mse_gripper_truth':dim3_arm6_reference[splitname]['all'],'arm6':dim3_arm6_reference[splitname]['all']}
        baseline[splitname]['baselines_hard']['DIM3-arm6-natural-DP-reference']={
            'mse_gripper_truth':dim3_arm6_reference[splitname]['hard'],'arm6':dim3_arm6_reference[splitname]['hard']}
    comparisons={}
    for splitname in ('learned','unseen'):
        sdict={md:summaries[md]['splits'][splitname] for md in summaries if splitname in summaries[md]['splits']}
        if 'shared' not in sdict:continue
        sp=sdict.get('p5');se=sdict.get('separate')
        split=learned if splitname=='learned' else unseen
        hard=np.asarray(split['hard'],bool)
        sharedall=sdict['shared']['_selected_actions']
        sharedhard=sdict['shared']['_selected_actions'][hard] if hard.any() else None
        comparisons[splitname]={}
        for baseline_name,baseline_summary in [('p5',sp),('separate',se)]:
            if baseline_summary is None:continue
            bpred=baseline_summary['_selected_actions']
            ash=action_errors(sharedall,split['actions']);abh=action_errors(bpred,split['actions'])
            diffs={m:np.asarray(ash[m])-np.asarray(abh[m]) for m in ('arm6','active7')}
            comparisons[splitname][f'shared_minus_{baseline_name}_all']={m:_paired_bootstrap(d,split['task'],split['episode'],seed=seed) for m,d in diffs.items()}
            if hard.any():
                sh=action_errors(sharedhard,split['actions'][hard]);bh=action_errors(baseline_summary['_selected_actions'][hard],split['actions'][hard])
                comparisons[splitname][f'shared_minus_{baseline_name}_hard']={m:_paired_bootstrap(np.asarray(sh[m])-np.asarray(bh[m]),split['task'][hard],split['episode'][hard],seed=seed+1) for m in ('arm6','active7')}
        randpred=sdict['shared']['_whole']['random_playback_actions']
        # Average the five random draws per query before episode-level resampling.
        rmetrics=[action_errors(randpred[i],split['actions']) for i in range(len(randpred))]
        rmean={m:np.mean(np.stack([x[m] for x in rmetrics]),axis=0) for m in ('arm6','active7')}
        ash=action_errors(sharedall,split['actions'])
        comparisons[splitname]['shared_minus_random5_all']={m:_paired_bootstrap(np.asarray(ash[m])-rmean[m],split['task'],split['episode'],seed=seed+2) for m in ('arm6','active7')}
        if hard.any():
            rh={m:v[hard] for m,v in rmean.items()};ah=action_errors(sharedhard,split['actions'][hard])
            comparisons[splitname]['shared_minus_random5_hard']={m:_paired_bootstrap(np.asarray(ah[m])-rh[m],split['task'][hard],split['episode'][hard],seed=seed+3) for m in ('arm6','active7')}
    results={'status':'complete' if not smoke else 'smoke_complete','mode':mode,'objective':objective,'seed':seed,
        'fixed_selection':{'learned_rows':len(learned['key']),'learned_hard':int(learned['hard'].sum()),
            'unseen_rows':len(unseen['key']),'unseen_hard':int(unseen['hard'].sum()),'dev_rows':len(dev['key'])},
        'parity':{'unit_count':full_unit_count,'smoke_candidate_count':len(bank['length']),'unit_start_feature_rows':4516,
            'learned_rows_match_pi05ref':True,'unseen_rows_match_pi05ref':True,
            'pi05_unseen_active7':pi05_unseen,
            'pi05_unseen_hard_active7':pi05_hard,
            'querywise_best_arm6':querywise_oracle_all,'querywise_best_arm6_hard':querywise_oracle_hard,
            'dim3_arm6_natural_dp':dim3_arm6_reference,
            'dim3_arm6_natural_dp_parity':True,
            'cmp3_natural_dct3_active7':cmp3_all,'cmp3_natural_dct3_active7_hard':cmp3_hard,
            'cmp3_natural_dct3_reference_matches':True,
            'cmp3_natural_dct3_recomputed_on_full_fixed_sets':bool(cmp3_natural_local),
            'cmp3_natural_dct3_recomputed':cmp3_natural_local},
        'baselines':baseline,'models':summaries,'paired_episode_bootstrap_5000':comparisons,
        'bootstrap_reps':5000,'training_teacher_temperature':training_tau,
        'ranking_reference_tau':REPORT_RANKING_TAU,'hold_last_parity':hold_last_parity,
        'random20_blend_seed':RANDOM20_BLEND_SEED,
        'contrastive_objective':('hard InfoNCE with one minimum-arm-MSE positive' if objective=='hard' else
            f'soft-label InfoNCE cross entropy H(p,q), p=softmax(-arm6_MSE/tau), training tau={training_tau:.8g}; q=softmax(exp(logit_scale)*similarity), logit_scale initialized 1/.07 and capped at 100'),
        'selection_note':'Fixed learned/unseen rows and hard masks; hyperparameters evaluated on tokenized Dev rows only. No Eval-based selection.',
        'teacher_tau_definition':f'Training teacher tau selected from {TEACHER_TEMPERATURES} by Dev. Ranking/distribution metrics use fixed tau={REPORT_RANKING_TAU}; no Eval labels used for selection.',
        'blend_definition':'Each stored unit is played at native cadence for one 10-step selection interval (clip if longer, hold terminal value if shorter); raw action windows are then softmax(score/temperature)-averaged. No DCT averaging or pose/scale/time alignment.',
        'input_fields':_input_fields_for_mode(mode),
        'leakage_audit':{'query_model_fields':_input_fields_for_mode(mode),
            'excluded_from_query_model':['future actions','target continuation','task ID','episode ID','subtask label','ground-truth boundary'],
            'memory_key_fields':['unit-start VLM tokens/group IDs','prior memory-demo history','unit-start state','stored decoded unit actions only for separate-key control'],
            'evaluation_candidate_overlap_assertions':True,
            'training_self_episode_exclusion':'task/episode IDs are used only to mask same-episode memory candidates when forming memory-side training batches; they are not model inputs.',
            'evaluation_labels':'used only for metrics and Dev-only configuration selection; learned/unseen query episodes are asserted absent from the candidate pool.'}}
    results=_without_private(results)
    _atomic_json(output/'results.json',results)
    (output/'report.md').write_text(_render_report(results)+'\n')
    return results


def _render_report(r):
    def fmt(x,places=5):
        try:return f'{float(x):.{places}f}' if np.isfinite(float(x)) else '—'
        except (TypeError,ValueError):return '—'
    lines=['# KEY2-F report','',
        'attempt-25 is invalidated: unseen/Dev histories used last-completed-segment actions, and its soft teacher tau / unscaled logits made contrastive supervision nearly uniform. It must not be used as a result.',
        '', '## 主結果：top-1（グリッパー正解）','',
        '| split |方式|行/難|全体 MSE|難 MSE|',
        '|---|---|---:|---:|---:|']
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            a=s['top1_actions_all'];h=s['top1_actions_hard'] or {}
            lines.append(f"|{split}|{mode}|{s['n']}/{s['hard_n']}|{fmt(a.get('mse_gripper_truth'))}|{fmt(h.get('mse_gripper_truth'))}|")
    for split,s in r.get('baselines',{}).items():
        lines.append(f"|{split}|random-5|{s['n']}/{s['hard_n']}|{fmt(s['random5_actions_all'].get('mse_gripper_truth'))}|{fmt((s.get('random5_actions_hard') or {}).get('mse_gripper_truth'))}|")
        for name,vals in s.get('baselines',{}).items():
            hv=s.get('baselines_hard',{}).get(name,{})
            lines.append(f"|{split}|{name}|{s['n']}/{s['hard_n']}|{fmt(vals.get('mse_gripper_truth'))}|{fmt(hv.get('mse_gripper_truth'))}|")
        lines.append(f"|{split}|same-playback greedy oracle|{s['n']}/{s['hard_n']}|{fmt(s['oracle_playback_all'].get('mse_gripper_truth'))}|{fmt((s.get('oracle_playback_hard') or {}).get('mse_gripper_truth'))}|")
        dp=s['baselines'].get('DIM3-arm6-natural-DP-reference',{});dph=s['baselines_hard'].get('DIM3-arm6-natural-DP-reference',{})
        lines.append(f"|{split}|whole-column DP oracle (DIM3; reference)|{s['n']}/{s['hard_n']}|{fmt(dp.get('mse_gripper_truth'))}|{fmt(dph.get('mse_gripper_truth'))}|")
    lines += ['', '## Devで選んだ top-k ブレンド（付録）','',
        '| split |方式|行/難|k / 温度|全体 MSE|難 MSE|','|---|---|---:|---:|---:|---:|']
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            a=s['actions_all'];h=s['actions_hard'] or {};cfg=s['selected_blend']
            lines.append(f"|{split}|{mode}|{s['n']}/{s['hard_n']}|{cfg['k']} / {fmt(cfg['temperature'],3)}|{fmt(a.get('mse_gripper_truth'))}|{fmt(h.get('mse_gripper_truth'))}|")
    lines += ['', '## 全候補の順位・分布診断','',
        '| split |方式|対象|Spearman|CE H(p,q)|KL(p‖q)|Recall@1/5/20|NDCG@20|MRR|Top-1 / Top-20 腕MSE|回収率|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def rankline(split,method,subset,m):
        return (f"|{split}|{method}|{subset}|{fmt(m.get('spearman_positive'),3)}|{fmt(m.get('cross_entropy'),4)}|"
            f"{fmt(m.get('kl_pq'),4)}|{fmt(m.get('recall@1'),3)}/{fmt(m.get('recall@5'),3)}/{fmt(m.get('recall@20'),3)}|"
            f"{fmt(m.get('ndcg@20'),3)}|{fmt(m.get('mrr'),3)}|{fmt(m.get('top1_arm6_mse'))}/{fmt(m.get('top20_arm6_mse'))}|{fmt(m.get('recovery'),3)}|")
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            lines.append(rankline(split,mode,'全体',s['rank_all']))
            if s.get('rank_hard'):lines.append(rankline(split,mode,'難',s['rank_hard']))
    for split,s in r.get('baselines',{}).items():
        lines.append(rankline(split,'random-5','全体',s['random_rank']))
        if s.get('random_rank_hard'):lines.append(rankline(split,'random-5','難',s['random_rank_hard']))
        # Every method shares the same correct-action oracle distribution.
        lines.append(rankline(split,'oracle','全体',s['oracle_rank']))
        if s.get('oracle_rank_hard'):lines.append(rankline(split,'oracle','難',s['oracle_rank_hard']))
    lines += ['', '## 付録：Devで選んだ温度による k 別アクション誤差','',
        '| split |方式|集合|k|温度|MSE（グリッパー正解）|', '|---|---|---|---:|---:|---:|']
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            for k,cfg in s['blend_by_k'].items():
                lines.append(f"|{split}|{mode}|全体 / 難|{k}|{fmt(cfg['temperature'],3)}|{fmt(cfg['mse_gripper_truth'])} / {fmt(cfg['hard_mse_gripper_truth'])}|")
    lines += ['', '### Dev選択の監査','']
    for mode,summary in r.get('models',{}).items():
        d=summary['dev_selection']
        lines.append(f"- {mode}: training teacher tau={fmt(d['training_teacher_tau'],7)}、ranking tau={fmt(d['ranking_reference_tau'],7)}、logit scale={fmt(d['logit_scale'],4)}、選択={d['blend_selection']['selected']}（Dev action MSE 最小）。各kの最良温度={d['blend_selection']['per_k']}。")
    lines += ['', '### 軸別寄与（1/7で規格化）','', '| split |方式|集合|x|y|z|rx|ry|rz|grip|', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            for subset,key in [('all','actions_all'),('hard','actions_hard')]:
                vals=s.get(key) or {}
                fields=[vals.get(f'axis_{j}_mse_over7') for j in range(6)]+[vals.get('gripper_mse_over7')]
                lines.append(f"|{split}|{mode}|{subset}|"+'|'.join(fmt(v) for v in fields)+'|')
    lines += ['', '### 付録：保存グリッパー／π0.5グリッパーを使ったactive7', '',
              '| split |方式|全体 active7（保存grip）|難 active7（保存grip）|全体 active7（π0.5 grip）|難 active7（π0.5 grip）|',
              '|---|---|---:|---:|---:|---:|']
    for mode,summary in r.get('models',{}).items():
        for split,s in summary['splits'].items():
            a=s['actions_all'];h=s['actions_hard'] or {}
            lines.append(f"|{split}|{mode}|{fmt(a.get('active7_saved_gripper'))}|{fmt(h.get('active7_saved_gripper'))}|{fmt(a.get('active7_pi05_grip'))}|{fmt(h.get('active7_pi05_grip'))}|")
    for split,s in r.get('baselines',{}).items():
        for method,vals in s.get('baselines',{}).items():
            hv=s.get('baselines_hard',{}).get(method,{})
            lines.append(f"|{split}|{method}|{fmt(vals.get('active7_saved_gripper'))}|{fmt(hv.get('active7_saved_gripper'))}|{fmt(vals.get('active7_pi05_grip'))}|{fmt(hv.get('active7_pi05_grip'))}|")
    lines += ['',f"状態: {r['status']}。Random は seed 4101–4105 の一様選択。",]
    lines += ['',f"固定行: learned={r['fixed_selection']['learned_rows']} (hard {r['fixed_selection']['learned_hard']}), unseen={r['fixed_selection']['unseen_rows']} (hard {r['fixed_selection']['unseen_hard']}), Dev={r['fixed_selection']['dev_rows']}.",
        f"PI0REF parity: unseen active7={r['parity']['pi05_unseen_active7']:.6f}, hard={r['parity']['pi05_unseen_hard_active7']:.6f}.",
        f"DIM3 arm6 natural-DP parity: {r['parity']['dim3_arm6_natural_dp']}.",
        f"候補順位の正解分布: p=softmax(-MSE/tau)、report tau={REPORT_RANKING_TAU}; 学習 teacher tau は Dev 候補 {TEACHER_TEMPERATURES} から選択。q=softmax(score/Dev-selected temperature)、score は exp(logit_scale) 倍（初期 1/.07、上限100）。NDCG gain=exp(-MSE/tau)。",
        f"Hold-last parity gate: {r.get('hold_last_parity')}",
        f"選択は全4,516候補に対して task/episode によるマスクなし。選択方式ごとの主表はDevで選んだk、各kの温度固定値は付録。",
        f"漏洩監査: {r['leakage_audit']}.",
        '注意: オフライン行動誤差であり、閉ループ性能ではない。']
    return '\n'.join(lines)


def run(*, output: Path, **_: Any) -> None:
    """Run an end-to-end train, Dev-select and locked-evaluation attempt."""
    run_experiment(Path(output), mode=_.get("mode", "all"), seed=int(_.get("seed", 1)),
                   learning_rate=float(_.get("learning_rate", 3e-4)),
                   epochs=int(_.get("epochs", 2)), steps_per_epoch=_.get("steps_per_epoch"),
                   smoke=bool(_.get("smoke", False)),objective=_.get("objective","soft"),
                   teacher_temperature=float(_.get("teacher_temperature",.005)),
                   vjepa_eval_cache=Path(_.get("vjepa_eval_cache")) if _.get("vjepa_eval_cache") else None)


def extract_missing_memory_prefixes(*, output: Path, run_root: Path, primary_cache: Path,
                                    secondary_cache: Path, checkpoint: Path | None = None,
                                    batch_size: int = 4, verify_count: int = 8) -> dict[str, Any]:
    """Extract exact π0.5 prefix rows for missing CMP3 unit-start timestamps.

    This follows SEG2's LIBERO transforms/prefix_features path and uses the
    source frame identified by (task,episode,frame). Existing stride-10 and
    stride-5 caches are read-only. Several existing keys are deliberately
    re-extracted and compared bit-for-bit as a parity gate.
    """
    import io
    import os
    import sys
    import importlib.util
    from collections import OrderedDict

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".20")
    import numpy as _np
    import pyarrow.parquet as pq
    from PIL import Image

    here = Path(__file__).resolve().parent
    pi05_dir = here.parent / "pi05_memory"
    sys.path[:0] = [str(here), str(pi05_dir)]
    import gripper_split as grip
    import motion_core_eval as motion
    import seg_context_extract as extract

    run_root = Path(run_root)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        raise FileExistsError(f"refusing to reuse incomplete/unmanifested token output {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = grip._load_rows()
    memory_ids = _np.flatnonzero(_np.asarray(rows["memory"], bool))
    starts = []
    for (task, episode), row_ids in grip._traces(rows, memory_ids).items():
        trace = _np.concatenate([_np.asarray(rows["actions"][i, :, :7], _np.float32) for i in row_ids])
        base_frame = int(rows["frame"][row_ids[0]])
        starts.extend((int(task), int(episode), base_frame + int(s))
                      for s, _ in motion.anchor_spans(trace, min_length=5))
    if len(starts) != 4516 or len(set(starts)) != 4516:
        raise AssertionError(f"CMP3 unit-start keys changed: {len(starts)} / {len(set(starts))}")

    def read_keys(cache: Path) -> list[tuple[int, int, int]]:
        manifest = cache / "manifest.json"
        if manifest.exists():
            return [tuple(map(int, k)) for k in json.loads(manifest.read_text())["keys"]]
        row_dir = cache / "rows"
        if (cache / "data_manifest.json").exists() and row_dir.exists():
            task = _np.load(row_dir / "task.npy", mmap_mode="r", allow_pickle=False)
            episode = _np.load(row_dir / "episode.npy", mmap_mode="r", allow_pickle=False)
            frame = _np.load(row_dir / "frame.npy", mmap_mode="r", allow_pickle=False)
            return [(int(t), int(e), int(f)) for t, e, f in zip(task, episode, frame, strict=True)]
        raise FileNotFoundError(f"cache needs manifest.json or run-01 rows/: {cache}")

    primary_keys = read_keys(primary_cache)
    secondary_keys = read_keys(secondary_cache)
    primary_set, secondary_set = set(primary_keys), set(secondary_keys)
    missing = sorted(set(starts) - primary_set - secondary_set)
    # Re-extract entries already in either cache, sampled deterministically, to
    # verify the same checkpoint, transform path, grouping and image direction.
    overlap = sorted((set(starts) & (primary_set | secondary_set)))
    rng = _np.random.default_rng(4101)
    verify = [overlap[i] for i in sorted(rng.choice(len(overlap), min(verify_count, len(overlap)), replace=False))]
    required = sorted(set(missing) | set(verify))
    previous_manifest = output / "manifest.json"
    if previous_manifest.exists():
        old_keys = [tuple(map(int, k)) for k in json.loads(previous_manifest.read_text())["keys"]]
        if old_keys != required:
            raise ValueError("existing extraction manifest differs from requested exact key set")
    if len(missing) != 1546:
        raise AssertionError(f"expected 1546 missing unit-start tokens, found {len(missing)}")

    data_manifest = json.loads((run_root / "data_manifest.json").read_text())
    source = Path(data_manifest["source"])
    ckpt = Path(checkpoint or data_manifest["checkpoint"])
    records = data_manifest["episodes"]
    record_map = {(int(r["task"]), int(r["episode"])): r for r in records}
    tasks = {int(k): v for k, v in data_manifest["tasks"].items()}

    import jax
    from openpi.transforms import Normalize, ResizeImages, TokenizePrompt, PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load as load_norm
    from openpi.shared.nnx_utils import module_jit

    run_py = pi05_dir / "run.py"
    spec = importlib.util.spec_from_file_location("key2_pi05_run", run_py)
    if spec is None or spec.loader is None:
        raise ImportError(run_py)
    run_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = run_module
    spec.loader.exec_module(run_module)
    norm_files = list((ckpt / "assets").rglob("norm_stats.json"))
    if len(norm_files) != 1:
        raise ValueError(f"expected one pi05_libero norm_stats.json, got {norm_files}")
    transforms = [LiberoInputs(ModelType.PI05), Normalize(load_norm(norm_files[0].parent), use_quantiles=True),
                  ResizeImages(224, 224), TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                  PadStatesAndActions(32)]
    model, _ = run_module.load_policy(ckpt)
    encode = module_jit(model.prefix_features)
    parquet_cache: OrderedDict[Path, Any] = OrderedDict()

    def observe(key):
        task, episode, frame = map(int, key)
        rec = record_map[(task, episode)]
        p = Path(rec["path"])
        if p not in parquet_cache:
            if len(parquet_cache) >= 2:
                _, old = parquet_cache.popitem(last=False)
                old = None
            parquet_cache[p] = pq.read_table(p, columns=["image", "wrist_image", "state", "actions"])
        table = parquet_cache[p]
        images = {}
        for name in ("image", "wrist_image"):
            obj = table[name][frame].as_py()
            with Image.open(io.BytesIO(obj["bytes"])) as im:
                images[name] = _np.asarray(im.convert("RGB"))
        item = {"observation/image": images["image"], "observation/wrist_image": images["wrist_image"],
                "observation/state": _np.asarray(table["state"][frame].as_py(), _np.float32),
                "actions": _np.asarray(table["actions"].slice(frame, 10).to_pylist(), _np.float32),
                "prompt": tasks[task]}
        for transform in transforms:
            item = transform(item)
        return {"image_base": _np.asarray(item["image"]["base_0_rgb"]),
                "image_wrist": _np.asarray(item["image"]["left_wrist_0_rgb"]),
                "state": _np.asarray(item["state"]), "prompt": _np.asarray(item["tokenized_prompt"]),
                "prompt_mask": _np.asarray(item["tokenized_prompt_mask"])}

    def encode_batch(items):
        packed = {k: _np.stack([it[k] for it in items]) for k in items[0]}
        obs = {"image_base": packed["image_base"], "image_wrist": packed["image_wrist"],
               "state": packed["state"], "prompt": packed["prompt"], "prompt_mask": packed["prompt_mask"]}
        hidden, groups = encode(run_module.obs_batch(obs, _np.arange(len(items))))
        return _np.asarray(hidden), _np.asarray(groups)

    extract.extract_prefix_memmap(required, observe, encode_batch, output, batch_size=batch_size,
                                  token_shape=(532, 2048), valid_only=True)
    extracted_keys = [tuple(map(int, k)) for k in json.loads((output / "manifest.json").read_text())["keys"]]
    extracted = _np.load(output / "tokens.npy", mmap_mode="r", allow_pickle=False)
    extracted_groups = _np.load(output / "groups.npy", mmap_mode="r", allow_pickle=False)
    slot = {k: i for i, k in enumerate(extracted_keys)}

    def compare_existing(cache_dir, cache_keys):
        cache_tokens = _np.load(cache_dir / "tokens.npy", mmap_mode="r", allow_pickle=False)
        cache_groups = _np.load(cache_dir / "groups.npy", mmap_mode="r", allow_pickle=False)
        errors = []
        for key in verify:
            if key not in set(cache_keys):
                continue
            i = slot[key]
            j = cache_keys.index(key)
            valid = cache_groups[j] >= 0
            n = int(valid.sum())
            if not _np.array_equal(extracted_groups[i, :n], cache_groups[j, valid]):
                raise AssertionError(f"group IDs differ after exact prefix re-extraction for {key}")
            delta = _np.abs(extracted[i, :n].astype(_np.float32) - cache_tokens[j, valid].astype(_np.float32))
            errors.append(float(delta.max(initial=0)))
        return errors

    parity_errors = compare_existing(primary_cache, primary_keys) + compare_existing(secondary_cache, secondary_keys)
    if not parity_errors or max(parity_errors) != 0.:
        raise AssertionError(f"existing cache parity failed: {parity_errors}")
    result = {"unit_starts": len(starts), "missing_before_extraction": len(missing),
              "verify_rows_reextracted": len(parity_errors), "cache_max_abs_error": max(parity_errors),
              "extracted_rows": len(required), "tokens_shape": list(extracted.shape),
              "groups_shape": list(extracted_groups.shape), "output": str(output),
              "checkpoint": str(ckpt), "orientation": "SEG2 transforms; corrected LIBERO source orientation",
              "source_cache_dirs": [str(primary_cache), str(secondary_cache)], "complete": True}
    (output / "extract_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-vjepa-cache",action="store_true",help="complete exact V-JEPA features for the fixed 619 Eval rows")
    parser.add_argument("--extract-missing", action="store_true")
    parser.add_argument("--run-root", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    parser.add_argument("--primary-cache", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache"))
    parser.add_argument("--secondary-cache", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode",choices=('all','all_inputs','all_methods','shared','separate','p0','p5',*ALL_INPUT_CONDITIONS),default='all')
    parser.add_argument("--objective",choices=('soft','hard'),default='soft')
    parser.add_argument("--teacher-temperature",type=float,default=.005,
                        help="memory-side soft teacher tau; choose from .002,.005,.01,.02 on Dev")
    parser.add_argument("--vjepa-eval-cache",type=Path)
    parser.add_argument("--eval-only-checkpoint",type=Path)
    parser.add_argument("--baseline-json",type=Path)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--learning-rate",type=float,default=3e-4)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--steps-per-epoch",type=int,default=None)
    parser.add_argument("--smoke",action='store_true')
    parser.add_argument("--dev-only",action='store_true')
    parser.add_argument("--missing-root",type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'))
    parser.add_argument("--unseen-stride5",type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'))
    args = parser.parse_args()
    if args.prepare_vjepa_cache:
        result=build_vjepa_eval_cache(args.output)
        print(json.dumps(result,indent=2))
    elif args.extract_missing:
        result = extract_missing_memory_prefixes(output=args.output, run_root=args.run_root,
            primary_cache=args.primary_cache, secondary_cache=args.secondary_cache,
            checkpoint=args.checkpoint, batch_size=args.batch_size)
        print(json.dumps(result, indent=2))
    else:
        run_experiment(args.output,mode=args.mode,seed=args.seed,learning_rate=args.learning_rate,
            epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,smoke=args.smoke,dev_only=args.dev_only,
            objective=args.objective,vjepa_eval_cache=args.vjepa_eval_cache,
            teacher_temperature=args.teacher_temperature,
            eval_only_checkpoint=args.eval_only_checkpoint,
            baseline_json=args.baseline_json,
            run_root=args.run_root,missing_root=args.missing_root,seen_stride5=args.primary_cache,
            unseen_stride5=args.unseen_stride5)


if __name__ == "__main__":
    main()
