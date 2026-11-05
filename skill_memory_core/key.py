"""M2 salient-token key, training loss, retrieval and exact MaxSim ranking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def build_context(state, previous_actions=None, has_previous=None, *, layout: str = "as_trained") -> np.ndarray:
    """Build the 79-D key context.

    `as_trained` preserves the historical key-training bug for checkpoint
    parity: previous 70-D actions are first written at offset 0, then state
    overwrites offsets 0:8, leaving actions 8:70 in 8:70 and zeros in 70:78.
    `fixed` is the intended `[state(8), previous_actions(70), flag]` layout.
    """
    s = np.asarray(state, np.float32)
    if s.ndim == 1:
        s = s[None]
    if s.ndim != 2 or s.shape[1] < 8:
        raise ValueError("state must be [B,>=8]")
    n = len(s)
    prev = np.zeros((n, 70), np.float32) if previous_actions is None else np.asarray(previous_actions, np.float32).reshape(n, 70)
    flag = np.zeros(n, np.float32) if has_previous is None else np.asarray(has_previous, np.float32).reshape(n)
    out = np.zeros((n, 79), np.float32)
    if layout == "as_trained":
        out[:, :70] = prev
        out[:, :8] = s[:, :8]
    elif layout == "fixed":
        out[:, :8] = s[:, :8]
        out[:, 8:78] = prev
    else:
        raise ValueError("layout must be 'as_trained' or 'fixed'")
    out[:, 78] = flag
    return out


def normalize_tokens(x, eps: float = 1e-8):
    if torch.is_tensor(x):
        return F.normalize(x.float(), dim=-1, eps=eps)
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class SalientTokenKey(nn.Module):
    """State-dict-compatible M2SalientTokens (`dim=96`, 512 image tokens)."""

    def __init__(self, dim: int = 96, *, token_count: int = 512,
                 token_dim: int = 2048, context_dim: int = 79):
        super().__init__()
        self.dim, self.token_count = int(dim), int(token_count)
        self.token_proj = nn.Linear(token_dim, dim, bias=False)
        self.position = nn.Parameter(torch.randn(token_count, dim) * .01)
        self.context = nn.Sequential(nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.salience = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.out = nn.Linear(dim, dim)

    def encode_tokens(self, tokens, context, mask):
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_proj.in_features:
            raise ValueError("tokens must be [B,N,token_dim]")
        if tokens.shape[1] > self.token_count:
            raise ValueError("token count exceeds positional table")
        mask = mask.bool()
        h = F.normalize(self.token_proj(tokens) + self.position[:tokens.shape[1]][None], dim=-1)
        ctx = self.context(context)[:, None]
        logits = self.salience(torch.tanh(h + ctx)).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.)
        key = F.normalize(self.out(torch.sum(h * weights[:, :, None], dim=1)), dim=-1)
        return key, h, weights

    def forward(self, tokens, context, mask):
        return self.encode_tokens(tokens, context, mask)[0]

    def load_production_weights(self, path: str | Path, *, map_location="cpu", strict=True):
        """Load the nested M2 tensor weights file without fixing a device."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload.get("models", {}).get("M2_salient_maxsim", payload)
        return self.load_state_dict(state, strict=strict)


def positive_pairs(train_row_ids, actions, tasks, *, tau: float, seed: int,
                   max_anchors: int = 4096, max_neighbors: int = 8) -> np.ndarray:
    """Select near-action positive pairs across tasks using active-action MSE."""
    ids = np.asarray(train_row_ids, np.int64)
    a = np.asarray(actions[ids, :, :7], np.float32).reshape(len(ids), -1)
    task = np.asarray(tasks)[ids]
    if not len(ids):
        raise ValueError("empty training rows")
    rng = np.random.default_rng(int(seed) + 19)
    anchors = rng.choice(ids, min(len(ids), max_anchors), replace=False)
    row_to_local = {int(r): i for i, r in enumerate(ids)}
    out = []
    for start in range(0, len(anchors), 128):
        qids = anchors[start:start + 128]
        q = np.stack([a[row_to_local[int(r)]] for r in qids])
        d = np.mean((q[:, None] - a[None]) ** 2, axis=-1)
        for i, rid in enumerate(qids):
            d[i, task == int(tasks[rid])] = np.inf
            near = np.flatnonzero(d[i] <= tau)
            if len(near):
                near = near[np.argsort(d[i, near], kind="stable")[:max_neighbors]]
                out.append((int(rid), int(ids[int(rng.choice(near))])))
    if len(out) < 1:
        raise ValueError("no cross-task positive pairs at this tau")
    return np.asarray(out, np.int64)


def infonce_loss(keys, actions, tasks, *, tau: float, temperature: float = .12):
    """Multi-positive in-batch InfoNCE used by the learned pooled M2 key."""
    z = F.normalize(keys.float(), dim=-1)
    sim = z @ z.T / float(temperature)
    dist = torch.mean((actions[:, None].float() - actions[None].float()) ** 2, dim=-1)
    same = tasks[:, None] == tasks[None, :]
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = (dist <= float(tau)) & ~same & ~eye
    valid = positive.any(1)
    if not bool(valid.any()):
        return z.sum() * 0.
    sim = sim.masked_fill(eye, -1e4)
    den = torch.logsumexp(sim, dim=1)
    num = torch.logsumexp(sim.masked_fill(~positive, -1e4), dim=1)
    return (den[valid] - num[valid]).mean()


def train_key(model: SalientTokenKey, token_batches, context_batches, mask_batches,
              action_batches, task_batches, *, tau: float, epochs: int = 1,
              lr: float = 4e-4, device=None, weight_decay: float = 1e-4):
    """Train on prebatched observations; device defaults to the model parameters."""
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        losses = []
        for tok, ctx, mask, act, task in zip(token_batches, context_batches, mask_batches,
                                             action_batches, task_batches, strict=True):
            tok = torch.as_tensor(tok, device=dev, dtype=torch.float32)
            ctx = torch.as_tensor(ctx, device=dev, dtype=torch.float32)
            mask = torch.as_tensor(mask, device=dev, dtype=torch.bool)
            act = torch.as_tensor(act, device=dev, dtype=torch.float32)
            task = torch.as_tensor(task, device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=dev.type == "cuda" and torch.cuda.is_bf16_supported()):
                key = model(tok, ctx, mask)
            loss = infonce_loss(key, act, task, tau=tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else float("nan")})
    return model.eval(), history


def quantize_for_storage(hidden, weights):
    """Round both arrays to fp16 at write time; downstream computes in fp32."""
    if torch.is_tensor(hidden):
        hidden = hidden.detach().to(dtype=torch.float16, device="cpu").numpy()
    if torch.is_tensor(weights):
        weights = weights.detach().to(dtype=torch.float16, device="cpu").numpy()
    return np.asarray(hidden, np.float16), np.asarray(weights, np.float16)


def coarse_search(query_keys, memory_keys, query_row_ids, memory_row_ids,
                  query_tasks, memory_tasks, *, k: int = 64):
    """Normalized cosine shortlist, excluding same tasks and tie-breaking by row ID."""
    q = normalize_tokens(query_keys)
    m = normalize_tokens(memory_keys)
    qids, mids = np.asarray(query_row_ids, np.int64), np.asarray(memory_row_ids, np.int64)
    qt, mt = np.asarray(query_tasks), np.asarray(memory_tasks)
    if m.ndim != 2 or q.ndim != 2 or len(m) != len(mids) or len(q) != len(qids):
        raise ValueError("key arrays and dataset row IDs do not align")
    scores = q @ m.T
    out = np.empty((len(q), min(k, len(m))), np.int64)
    out_s = np.empty(out.shape, np.float32)
    for i in range(len(q)):
        vals = scores[i].copy()
        vals[mt == qt[i]] = -np.inf
        order = np.lexsort((mids, -vals))[:out.shape[1]]
        if not np.isfinite(vals[order]).all():
            raise ValueError("not enough cross-task memory rows")
        out[i], out_s[i] = mids[order], vals[order]
    return out, out_s


def maxsim_rerank(query_hidden, query_weights, query_mask, memory_hidden,
                  memory_weights, memory_mask, query_row_ids, candidate_row_ids,
                  memory_row_ids, *, batch_size: int = 4, device=None):
    """Bidirectional weighted MaxSim; exact stored-fp16 then fp32 behavior."""
    qid = np.asarray(query_row_ids, np.int64)
    mids = np.asarray(memory_row_ids, np.int64)
    cand = np.asarray(candidate_row_ids, np.int64)
    # Encoder hidden vectors are normalized before they are rounded to fp16.
    # Production MaxSim casts stored fp16 values back to fp32 without a second
    # normalization, so do exactly that here.
    qh = np.asarray(query_hidden, np.float16).astype(np.float32)
    mh = np.asarray(memory_hidden, np.float16).astype(np.float32)
    qw = np.asarray(query_weights, np.float16).astype(np.float32)
    mw = np.asarray(memory_weights, np.float16).astype(np.float32)
    qm, mm = np.asarray(query_mask, bool), np.asarray(memory_mask, bool)
    if cand.shape[0] != len(qid) or len(qh) != len(qid) or len(mh) != len(mids):
        raise ValueError("MaxSim query/candidate rows do not align")
    lookup = {int(r): i for i, r in enumerate(mids)}
    if any(int(r) not in lookup for r in cand.flat):
        raise ValueError("candidate dataset row ID missing from token memory")
    local = np.asarray([[lookup[int(r)] for r in c] for c in cand], np.int64)
    scores = np.empty(cand.shape, np.float32)
    dev = torch.device(device or "cpu")
    th = torch.as_tensor(qh, device=dev, dtype=torch.float32)
    tw = torch.as_tensor(qw, device=dev, dtype=torch.float32)
    tm = torch.as_tensor(mh, device=dev, dtype=torch.float32)
    tmw = torch.as_tensor(mw, device=dev, dtype=torch.float32)
    for start in range(0, len(qid), batch_size):
        stop = min(start + batch_size, len(qid))
        ix = torch.as_tensor(local[start:stop], device=dev, dtype=torch.long)
        cand_h, cand_w = tm[ix], tmw[ix]
        sim = torch.einsum("bnd,bkmd->bknm", th[start:stop], cand_h)
        q2m = sim.max(dim=-1).values
        m2q = sim.max(dim=-2).values
        a = torch.sum(q2m * tw[start:stop, None, :], dim=-1)
        b = torch.sum(m2q * cand_w, dim=-1)
        scores[start:stop] = ((a + b) * .5).cpu().numpy()
    # Stable sort preserves coarse shortlist order on tied MaxSim scores.
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(cand, order, axis=1), np.take_along_axis(scores, order, axis=1)


@dataclass
class Retriever:
    model: SalientTokenKey
    memory_row_ids: np.ndarray
    memory_tasks: np.ndarray
    memory_keys: np.ndarray
    memory_hidden: np.ndarray
    memory_weights: np.ndarray
    memory_mask: np.ndarray

    def search(self, query_keys, query_hidden, query_weights, query_mask,
               query_row_ids, query_tasks, *, k=64, rerank=True):
        coarse, scores = coarse_search(query_keys, self.memory_keys, query_row_ids,
                                       self.memory_row_ids, query_tasks, self.memory_tasks, k=k)
        if not rerank:
            return coarse, scores
        return maxsim_rerank(query_hidden, query_weights, query_mask,
            self.memory_hidden, self.memory_weights, self.memory_mask,
            query_row_ids, coarse, self.memory_row_ids)
