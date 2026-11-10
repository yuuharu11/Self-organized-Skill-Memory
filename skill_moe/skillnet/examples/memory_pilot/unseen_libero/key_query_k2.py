#!/usr/bin/env python3
"""K2: layer-wise and executed-history query diagnostics for unseen LIBERO.

This module is deliberately separate from K1. It imports K1's data split,
candidate pool, exclusion, listwise objective, and evaluation helpers; it
never alters K1 artifacts. Production stages are resumable.

Unseen action targets are structurally absent from extraction and fitting.
They may be loaded by the evaluation scorer only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
K1_PATH = HERE / "key_query_probe.py"
OUT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/key-query-02')
SEED = 4101
LAYER_COUNT = 18
HISTORY_LENGTHS = (10, 30, 50)


_K1_MODULE=None
def _k1():
    global _K1_MODULE
    if _K1_MODULE is not None:return _K1_MODULE
    spec = importlib.util.spec_from_file_location("k2_imported_k1", K1_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(K1_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _K1_MODULE=module
    return module


def _atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _atomic_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def _atomic_torch_save(torch, value, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    torch.save(value,tmp);tmp.replace(path)


def pool_prefix_layer_values(values: np.ndarray, language_mask: np.ndarray,
                             *, image_tokens: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Pool each layer to 16 tokens/camera plus one instruction mean.

    Args:
      values: [L,B,3*256+T,D] V projections from the prefix KV cache.
      language_mask: [B,T], true for non-padding prompt tokens.

    Returns ``[B,L,33,D]`` and ``[B,33]``. The two active cameras are reduced
    from 16x16 to 4x4. The third image slot is the masked placeholder camera
    in this tokenizer and is omitted. PI05 is configured with
    ``discrete_state_input=False``; state is therefore a separate numeric
    feature, not a text-token group. The 33rd feature is instruction mean.
    """
    v = np.asarray(values, np.float32)
    lm = np.asarray(language_mask, bool)
    if v.ndim != 4 or lm.ndim != 2 or v.shape[1] != lm.shape[0]:
        raise ValueError("expected V=[layers,batch,prefix,D], mask=[batch,text]")
    if v.shape[2] != 3 * image_tokens + lm.shape[1]:
        raise ValueError("prefix token count does not match camera and language lengths")
    layers, batch, _, dim = v.shape
    out = np.zeros((batch, layers, 33, dim), np.float32)
    mask = np.ones((batch, 33), bool)
    for b in range(batch):
        for layer in range(layers):
            for camera in range(2):
                start = camera * image_tokens
                grid = v[layer, b, start:start + image_tokens].reshape(16, 16, dim)
                out[b, layer, camera * 16:(camera + 1) * 16] = grid.reshape(4, 4, 4, 4, dim).mean((1, 3)).reshape(16, dim)
            text = v[layer, b, 3 * image_tokens:]
            valid = lm[b]
            if np.any(valid):
                out[b, layer, 32] = text[valid].mean(0)
    return out, mask


def build_history_arrays(rows: dict[str, np.ndarray], row_ids: np.ndarray,
                         actions: np.ndarray, *, lengths=HISTORY_LENGTHS,
                         state_width: int = 8) -> dict[str, np.ndarray]:
    """Create strictly past-only executed action/state histories.

    Each source row covers ten executed steps. Histories are right-aligned;
    unavailable prefix values are zero and marked false. Row identity is
    ``(task, episode, frame)`` and no current/future action is read.
    """
    ids = np.asarray(row_ids, np.int64)
    act = np.asarray(actions, np.float32)
    if act.ndim != 3 or act.shape[1] != 10 or act.shape[2] < 7:
        raise ValueError("actions must be [rows,10,>=7]")
    task = np.asarray(rows["task"], np.int64)
    episode = np.asarray(rows["episode"], np.int64)
    frame = np.asarray(rows["frame"], np.int64)
    if len(task) != len(act):
        raise ValueError("rows/actions source length mismatch")
    lookup = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(task, episode, frame, strict=True))}
    current_state = np.asarray(rows["state"], np.float32)
    result: dict[str, np.ndarray] = {}
    for length in lengths:
        length = int(length)
        if length not in (10, 30, 50):
            raise ValueError("supported executed-action histories are 10/30/50 steps")
        h = np.zeros((len(ids), length, 7), np.float32)
        m = np.zeros((len(ids), length), bool)
        hs = np.zeros((len(ids), length // 10, state_width), np.float32)
        sm = np.zeros((len(ids), length // 10), bool)
        for j, rid in enumerate(ids):
            t, e, f = int(task[rid]), int(episode[rid]), int(frame[rid])
            past = []
            for prior_frame in range(f - 10, max(-10, f - length - 10), -10):
                prev = lookup.get((t, e, prior_frame))
                if prev is None:
                    break
                past.append((prev, act[prev, :, :7]))
                if len(past) * 10 >= length:
                    break
            cursor = length
            for prev, chunk in past:
                cursor -= 10
                h[j, cursor:cursor + 10] = chunk
                m[j, cursor:cursor + 10] = True
                state_slot = cursor // 10
                hs[j, state_slot] = current_state[prev, :state_width]
                sm[j, state_slot] = True
        result[f"action_{length}"] = h
        result[f"action_mask_{length}"] = m
        result[f"state_{length}"] = hs
        result[f"state_mask_{length}"] = sm
    return result


def history_unit_codes(actions: np.ndarray, valid_mask: np.ndarray | None = None, *,
                       max_units: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build a past-history unit sequence without task/plan labels.

    The input is right-aligned by ``build_history_arrays``. Its mask is used to
    remove unavailable zero padding before segmentation; without this, an
    episode-start query would acquire a fake all-zero unit. Segmentation uses
    the same smoothed physical-speed minima, minimum duration, and gripper
    switch rules as ``motion_core_eval.anchor_spans``. Each completed unit is
    represented by its first three temporal DCT coefficients over active-seven
    actions. The latest units are returned.
    """
    from scipy.fft import dct
    x = np.asarray(actions, np.float32)
    if x.ndim != 2 or x.shape[1] != 7:
        raise ValueError("history actions must be [steps,7]")
    if valid_mask is None:
        valid = np.ones(len(x), bool)
    else:
        valid = np.asarray(valid_mask, bool)
        if valid.shape != (len(x),):
            raise ValueError("history valid mask must match action steps")
        # Histories are right-aligned; reject malformed masks instead of
        # silently treating holes as executed actions.
        first_valid = int(np.flatnonzero(valid)[0]) if valid.any() else len(valid)
        if valid[first_valid:].size and not valid[first_valid:].all():
            raise ValueError("history valid mask must be a contiguous suffix")
        x = x[first_valid:]
    if len(x) == 0:
        return np.zeros((max_units, 21), np.float32), np.zeros(max_units, bool)
    # Match the reviewed LIBERO anchor segmentation used for memory units.
    # Keep imports local so importing this diagnostic does not load the rest
    # of the experimental stack.
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    speed = np.linalg.norm(vs.to_physical(x[:, :6])[:, :3], axis=1)
    smooth = np.convolve(speed, np.ones(3, np.float32) / 3, mode="same")
    n = len(x)
    minima = [t for t in range(1, n - 1)
              if smooth[t] <= smooth[t - 1] and smooth[t] < smooth[t + 1]]
    switches = (np.flatnonzero(np.sign(x[1:, 6]) != np.sign(x[:-1, 6])) + 1).tolist()
    bounds = [0]
    for t in sorted(set(minima) | set(switches)):
        if t in switches or t - bounds[-1] >= 5:
            bounds.append(t)
    if n - bounds[-1] < 5 and len(bounds) > 1 and bounds[-1] not in switches:
        bounds.pop()
    spans = list(zip(bounds, bounds[1:] + [n]))
    units = [x[a:b] for a, b in spans]
    encoded = []
    for unit in units:
        coeff = dct(unit, type=2, norm="ortho", axis=0)[:3]
        z = np.zeros((3, 7), np.float32)
        z[:len(coeff)] = coeff
        encoded.append(z.reshape(-1))
    out = np.zeros((max_units, 21), np.float32)
    mask = np.zeros(max_units, bool)
    selected = encoded[-max_units:]
    out[-len(selected):] = selected
    mask[-len(selected):] = True
    return out, mask


def history_only_inputs(history: dict[str, np.ndarray], *, length: int,
                        encoder: str) -> np.ndarray:
    """Deterministic history-only input layouts for h1/h2/h3/h4 baselines."""
    if encoder == "h1":
        return np.concatenate([history[f"action_{length}"].reshape(len(history[f"action_{length}"]), -1),
                               history[f"action_mask_{length}"].astype(np.float32)], axis=-1)
    if encoder == "h4":
        return np.concatenate([history[f"state_{length}"].reshape(len(history[f"state_{length}"]), -1),
                               history[f"state_mask_{length}"].astype(np.float32)], axis=-1)
    if encoder == "h3":
        encoded = [history_unit_codes(a, mask)
                   for a, mask in zip(history[f"action_{length}"],
                                      history[f"action_mask_{length}"], strict=True)]
        return np.concatenate([np.stack([x[0] for x in encoded]).reshape(len(encoded), -1),
                               np.stack([x[1] for x in encoded]).astype(np.float32)], axis=-1)
    if encoder == "h2":
        # The raw sequence/mask is consumed by K2HistoryGRU below.
        return np.concatenate([history[f"action_{length}"].reshape(len(history[f"action_{length}"]), -1),
                               history[f"action_mask_{length}"].astype(np.float32)], axis=-1)
    raise ValueError(f"unknown history encoder: {encoder}")


def _fit_gru_history_encoder(history, current_actions, length, fit_ids, hold_ids, *,
                             epochs=12, batch_size=128, threads=8, device="cpu"):
    """Fit h2 by next-window action prediction on LIBERO-40 fit episodes only."""
    import torch
    from torch import nn
    torch.set_num_threads(int(threads));dev=torch.device(device)
    if device=="cuda":torch.cuda.set_per_process_memory_fraction(.20)
    gru=K2HistoryGRU().to(dev);head=nn.Linear(128,70).to(dev)
    params=list(gru.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-3)
    acts=torch.as_tensor(history[f"action_{length}"],dtype=torch.float32,device=dev)
    mask=torch.as_tensor(history[f"action_mask_{length}"],dtype=torch.bool,device=dev)
    target=torch.as_tensor(np.asarray(current_actions,np.float32).reshape(len(current_actions),70),device=dev)
    fit_ids=np.asarray(fit_ids,np.int64);hold_ids=np.asarray(hold_ids,np.int64)
    best=float("inf");best_state=None;wait=0;curve=[]
    for ep in range(int(epochs)):
        order=np.random.default_rng(SEED+ep).permutation(fit_ids);gru.train();head.train();losses=[]
        for start in range(0,len(order),int(batch_size)):
            ix=order[start:start+int(batch_size)];z=gru(acts[ix],mask[ix]);pred=head(z)
            loss=(pred-target[ix]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
        gru.eval();head.eval()
        with torch.no_grad():
            val=[]
            for start in range(0,len(hold_ids),int(batch_size)):
                ix=hold_ids[start:start+int(batch_size)];val.append(float((head(gru(acts[ix],mask[ix]))-target[ix]).square().mean().cpu()))
            hold_loss=float(np.mean(val)) if val else float(np.mean(losses))
        curve.append({"epoch":ep+1,"fit_mse":float(np.mean(losses)),"holdout_mse":hold_loss})
        if hold_loss<best:
            best=hold_loss;best_state={"gru":{k:v.detach().cpu().clone() for k,v in gru.state_dict().items()},
                "head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},"best_epoch":ep+1};wait=0
        else:
            wait+=1
            if wait>=3:break
    gru.load_state_dict(best_state["gru"]);gru.eval()
    with torch.no_grad():
        vectors=[]
        for start in range(0,len(acts),int(batch_size)):
            vectors.append(gru(acts[start:start+int(batch_size)],mask[start:start+int(batch_size)]).cpu().numpy())
    return np.concatenate(vectors),best_state,curve


def _history_variant_arrays(source, row_ids, histories, *, length, encoder, gru_state=None,
                            device="cpu",threads=8):
    if encoder in ("h1","h3","h4"):
        return history_only_inputs(histories,length=length,encoder=encoder)
    if encoder!="h2" or gru_state is None:raise ValueError("h2 needs trained GRU state")
    import torch
    torch.set_num_threads(int(threads));gru=K2HistoryGRU().to(device);gru.load_state_dict(gru_state["gru"]);gru.eval()
    acts=torch.as_tensor(histories[f"action_{length}"],dtype=torch.float32,device=device)
    mask=torch.as_tensor(histories[f"action_mask_{length}"],dtype=torch.bool,device=device)
    with torch.no_grad():
        return np.concatenate([gru(acts[i:i+128],mask[i:i+128]).cpu().numpy() for i in range(0,len(acts),128)],axis=0)


class K2HistoryGRU:
    """Small masked GRU for h2; padding cannot update the recurrent state."""
    def __new__(cls, input_dim=7, hidden_dim=128):
        import torch
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            def forward(self, x, mask):
                h = torch.zeros((1, len(x), self.gru.hidden_size), device=x.device, dtype=x.dtype)
                for t in range(x.shape[1]):
                    _, candidate = self.gru(x[:, t:t+1], h)
                    h = torch.where(mask[:, t].view(1, -1, 1), candidate, h)
                return h[0]
        return Net()


def make_k2_layer_query(d_model: int = 256, token_dim: int = 256, slots: int = 4,
                        history_mode: str = "attention", history_dim: int = 128):
    """K1 P1-style cross-attention query generalized to any layer/history.

    ``history_mode='attention'`` uses history to condition attention slots;
    ``concat`` fuses a history vector with pooled visual output.
    """
    import torch
    from torch import nn
    if history_mode not in ("attention", "concat", "visual", "history"):
        raise ValueError("invalid history fusion")
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.hist = nn.Sequential(nn.Linear(history_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.kv = nn.Linear(token_dim, d_model)
            self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
            self.query = nn.Linear(d_model, slots * d_model)
            self.out = nn.Sequential(nn.Linear(slots * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        def forward(self, tokens, token_mask, state, hist):
            v = self.state(state)
            h = self.hist(hist)
            if history_mode == "history":
                return h
            if history_mode == "visual":
                h = torch.zeros_like(h)
            qctx = v + h
            q = self.query(qctx).view(len(tokens), slots, d_model)
            y, weights = self.attn(q, self.kv(tokens), self.kv(tokens),
                                   key_padding_mask=~token_mask.bool(), need_weights=True)
            visual = self.out(y.flatten(1))
            if history_mode == "concat":
                visual = self.fuse(torch.cat([visual, h], dim=-1))
            return visual, weights
    return Net()


def make_candidate_key(candidate_dim: int = 70, d_model: int = 256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(candidate_dim, 512), nn.GELU(), nn.Linear(512, d_model))


class K2ElmoMixer:
    """Serializable scalar layer mixture plus the standard K2 query/key heads."""
    def __new__(cls, layer_count=18, token_dim=256, history_dim=1,
                history_mode="visual", d_model=256):
        import torch
        from torch import nn
        from torch.nn import functional as F
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha=nn.Parameter(torch.zeros(int(layer_count)))
                self.query=make_k2_layer_query(d_model=d_model,token_dim=token_dim,
                    history_dim=history_dim,history_mode=history_mode)
                self.key=make_candidate_key(70,d_model)
            def forward(self,x,mask,state,hist,cand):
                mix=torch.einsum("l,nltd->ntd",torch.softmax(self.alpha,0),x)
                q=self.query(mix,mask,state,hist)
                if isinstance(q,tuple):q=q[0]
                return F.normalize(q,dim=-1) @ F.normalize(self.key(cand),dim=-1).T / .1
        return Net()


def combine_layer_tokens(tokens: np.ndarray, layers: tuple[int, ...], *, method: str = "concat",
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Combine a chosen layer set by token concatenation or ELMo scalar mix."""
    x = np.asarray(tokens, np.float32)
    if x.ndim != 4: raise ValueError("layer tokens must be [N,L,T,D]")
    if not layers or min(layers) < 0 or max(layers) >= x.shape[1]: raise ValueError("invalid layer selection")
    selected = x[:, np.asarray(layers, np.int64)]
    if method == "concat": return selected.transpose(0, 2, 1, 3).reshape(len(x), x.shape[2], -1)
    if method != "elmo": raise ValueError("method must be concat or elmo")
    w = np.ones(len(layers), np.float32) if weights is None else np.asarray(weights, np.float32)
    if w.shape != (len(layers),) or not np.isfinite(w).all(): raise ValueError("layer weight mismatch")
    w = np.exp(w - w.max()); w /= w.sum()
    return np.einsum("l,nltd->ntd", w, selected)


def select_top_layers(holdout_mse: dict[int, float], *, count: int = 3) -> tuple[int, ...]:
    """Choose layer IDs from LIBERO-40 holdout metrics only (lower is better)."""
    if count < 1 or len(holdout_mse) < count: raise ValueError("insufficient holdout layers")
    if not all(np.isfinite(float(v)) for v in holdout_mse.values()): raise ValueError("non-finite layer metric")
    return tuple(int(k) for k, _ in sorted(holdout_mse.items(), key=lambda kv: (float(kv[1]), int(kv[0])))[:count])


def drop_history_modality(history: np.ndarray, probability: float, *, seed: int, training: bool = True) -> np.ndarray:
    """Drop complete past-history vectors per row; never affects visual/state inputs."""
    x = np.asarray(history, np.float32).copy()
    if not 0 <= probability <= 1: raise ValueError("history dropout must be in [0,1]")
    if not training or probability == 0: return x
    rng = np.random.default_rng(int(seed)); dropped = rng.random(len(x)) < probability
    x[dropped] = 0
    return x


def fit_layer_retriever(tokens, token_mask, state, history, candidate_shapes, teacher, allowed,
                        train_indices, *, epochs=12, batch_size=64, lr=1e-3,
                        history_mode="attention", dropout=0.0, seed=SEED,
                        device="cpu", threads=8, train_metadata=None,
                        hard_negative_weight=0.0):
    """Fit one P1-style layer/history query against K1's listwise teacher.

    This routine accepts only LIBERO-40 rows. It has no argument for unseen
    target actions. Holdout model/hyperparameter selection is performed by the
    caller using K1's fixed task holdout and the same teacher/error matrices.
    """
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(int(threads)); torch.manual_seed(int(seed))
    if device == "cuda": torch.cuda.set_per_process_memory_fraction(.20)
    dev = torch.device(device)
    tok = torch.as_tensor(np.asarray(tokens, np.float32), device=dev)
    mask = torch.as_tensor(np.asarray(token_mask, bool), device=dev)
    st = torch.as_tensor(np.asarray(state, np.float32), device=dev)
    hist = torch.as_tensor(np.asarray(history, np.float32), device=dev)
    shapes = np.asarray(candidate_shapes, np.float32)
    teach = np.asarray(teacher, np.float32); allow = np.asarray(allowed, bool)
    fit = np.asarray(train_indices, np.int64)
    if tok.shape[:2] != mask.shape or len(tok) != len(st) or len(tok) != len(hist): raise ValueError("K2 fit feature mismatch")
    if teach.shape != allow.shape or teach.shape != (len(tok), len(shapes)): raise ValueError("teacher/memory mismatch")
    if np.any(teach[~allow] != 0) or not np.allclose(teach.sum(1), 1., atol=1e-4): raise ValueError("teacher must be normalized over allowed items")
    query = make_k2_layer_query(token_dim=tok.shape[-1], history_mode=history_mode, history_dim=hist.shape[-1]).to(dev)
    key = make_candidate_key(70).to(dev)
    candidate = torch.as_tensor(shapes.reshape(len(shapes), -1), device=dev)
    opt = torch.optim.AdamW([*query.parameters(), *key.parameters()], lr=float(lr))
    rng = np.random.default_rng(seed); losses=[]
    for epoch in range(int(epochs)):
        order = fit[rng.permutation(len(fit))]; epoch_losses=[]; query.train(); key.train()
        for start in range(0, len(order), int(batch_size)):
            ix = order[start:start+int(batch_size)]
            hist_batch = hist[ix]
            if dropout:
                if not 0 <= dropout <= 1: raise ValueError("history dropout outside [0,1]")
                keep = torch.as_tensor(rng.random((len(ix), 1)) >= dropout, device=dev)
                hist_batch = hist_batch * keep
            qout = query(tok[ix], mask[ix], st[ix], hist_batch)
            if isinstance(qout, tuple): qout=qout[0]
            q = F.normalize(qout, dim=-1); k = F.normalize(key(candidate), dim=-1)
            logits = q @ k.T / .1
            m = torch.as_tensor(allow[ix], device=dev); p = torch.as_tensor(teach[ix], device=dev)
            logits = logits.masked_fill(~m, -1e9)
            loss = -(p * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            if hard_negative_weight and train_metadata is not None and len(ix) > 1:
                k1 = _k1()
                meta = {name: np.asarray(value)[ix] for name, value in train_metadata.items()}
                hard_context=meta.get("hard_context",hist_batch.detach().cpu().numpy())
                pairs = k1.branching_hard_negative_pairs(
                    np.asarray(hard_context,np.float32), np.argmax(teach[ix], axis=1),
                    meta["episode"], meta["frame"] // 10)
                if len(pairs):
                    loss = loss + float(hard_negative_weight) * k1.hard_negative_margin_loss(q, pairs)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return {"query": query, "key": key, "history": losses,
            "fit_rows": fit.copy(), "heldout_target_access": False,
            "hard_negative_weight": float(hard_negative_weight)}


def score_layer_retriever(model, tokens, token_mask, state, history, candidate_shapes,
                          allowed, *, batch_size=128, device="cpu"):
    import torch
    import torch.nn.functional as F
    dev=torch.device(device); query=model["query"].to(dev).eval(); key=model["key"].to(dev).eval()
    tok=torch.as_tensor(np.asarray(tokens,np.float32),device=dev); mask=torch.as_tensor(np.asarray(token_mask,bool),device=dev)
    st=torch.as_tensor(np.asarray(state,np.float32),device=dev); hist=torch.as_tensor(np.asarray(history,np.float32),device=dev)
    cand=np.asarray(candidate_shapes,np.float32); ck=torch.as_tensor(cand.reshape(len(cand),-1),device=dev)
    with torch.no_grad(): keys=F.normalize(key(ck),dim=-1)
    scores=np.empty((len(tok),len(cand)),np.float32)
    with torch.no_grad():
        for start in range(0,len(tok),int(batch_size)):
            end=min(start+int(batch_size),len(tok)); q=query(tok[start:end],mask[start:end],st[start:end],hist[start:end])
            if isinstance(q,tuple): q=q[0]
            scores[start:end]=(F.normalize(q,dim=-1)@keys.T).detach().cpu().numpy()
    a=np.asarray(allowed,bool)
    if a.shape!=scores.shape: raise ValueError("allowed mask differs from candidate scores")
    scores[~a]=-np.inf
    rank=np.argsort(-scores,axis=1,kind="stable")
    return scores,rank


def fit_elmo_layer_mixer(layer_tokens, token_mask, state, history, candidate_shapes,
                         teacher, allowed, train_indices, *, epochs=12, batch_size=64,
                         lr=1e-3, history_mode="attention", dropout=0.0,
                         device="cpu", threads=8):
    """Learn scalar softmax weights over all layers with listwise retrieval loss."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(int(threads))
    dev=torch.device(device)
    if device=="cuda":torch.cuda.set_per_process_memory_fraction(.20)
    x=torch.as_tensor(np.asarray(layer_tokens,np.float32),device=dev)
    mask=torch.as_tensor(np.asarray(token_mask,bool),device=dev)
    st=torch.as_tensor(np.asarray(state,np.float32),device=dev)
    hist=torch.as_tensor(np.asarray(history,np.float32),device=dev)
    shapes=np.asarray(candidate_shapes,np.float32); teach=np.asarray(teacher,np.float32); allow=np.asarray(allowed,bool)
    fit=np.asarray(train_indices,np.int64)
    net=K2ElmoMixer(x.shape[1],x.shape[-1],hist.shape[-1],history_mode).to(dev)
    candidate=torch.as_tensor(shapes.reshape(len(shapes),-1),device=dev)
    opt=torch.optim.AdamW(net.parameters(),lr=float(lr));rng=np.random.default_rng(SEED);losses=[]
    for _ in range(int(epochs)):
        order=fit[rng.permutation(len(fit))];epoch=[];net.train()
        for start in range(0,len(order),int(batch_size)):
            ix=order[start:start+int(batch_size)];hh=hist[ix]
            if dropout:hh=hh*torch.as_tensor(rng.random((len(ix),1))>=dropout,device=dev)
            logits=net(x[ix],mask[ix],st[ix],hh,candidate)
            valid=torch.as_tensor(allow[ix],device=dev);p=torch.as_tensor(teach[ix],device=dev)
            logits=logits.masked_fill(~valid,-1e9)
            loss=-(p*F.log_softmax(logits,dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();epoch.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch)))
    return {"module":net,"history":losses,"layer_weights":torch.softmax(net.alpha,0).detach().cpu().numpy(),
            "fit_rows":fit.copy(),"heldout_target_access":False}


def score_elmo_layer_mixer(model, layer_tokens, token_mask, state, history,
                           candidate_shapes, allowed, *, batch_size=128,device="cpu"):
    import torch
    net=model["module"].to(device).eval();x=torch.as_tensor(np.asarray(layer_tokens,np.float32),device=device)
    mask=torch.as_tensor(np.asarray(token_mask,bool),device=device);st=torch.as_tensor(np.asarray(state,np.float32),device=device)
    hist=torch.as_tensor(np.asarray(history,np.float32),device=device);cand=np.asarray(candidate_shapes,np.float32)
    scores=[]
    with torch.no_grad():
        for start in range(0,len(x),int(batch_size)):
            end=min(start+int(batch_size),len(x));cc=torch.as_tensor(cand.reshape(len(cand),-1),device=device)
            logits=net(x[start:end],mask[start:end],st[start:end],hist[start:end],cc)
            scores.append(logits.detach().cpu().numpy())
    scores=np.concatenate(scores,axis=0);allow=np.asarray(allowed,bool)
    if scores.shape!=allow.shape:raise ValueError("ELMo score mask mismatch")
    scores[~allow]=-np.inf
    return scores,np.argsort(-scores,axis=1,kind="stable")


def fit_type_code_linear_probe(train_repr, train_shapes, holdout_repr, holdout_shapes,
                               *, clusters=32, seed=SEED, epochs=100, lr=1e-2,
                               threads=8, device="cpu", codebook_shapes=None):
    """Fit the K2 information probe: KMeans(32) codes + linear classifier.

    Codebook and classifier are learned on LIBERO-40 training rows only;
    returned accuracy is for a separately supplied held-out feature/shape set.
    """
    from sklearn.cluster import KMeans
    import torch
    torch.set_num_threads(int(threads)); torch.manual_seed(int(seed))
    x=np.asarray(train_repr,np.float32); yx=np.asarray(train_shapes,np.float32).reshape(len(train_shapes),-1)
    xv=np.asarray(holdout_repr,np.float32); yv=np.asarray(holdout_shapes,np.float32).reshape(len(holdout_shapes),-1)
    if len(x)<clusters: raise ValueError("fewer train rows than type clusters")
    codebook = yx if codebook_shapes is None else np.asarray(codebook_shapes,np.float32).reshape(-1,yx.shape[1])
    if len(codebook)<clusters: raise ValueError("fewer candidate/codebook shapes than clusters")
    km=KMeans(n_clusters=int(clusters),random_state=int(seed),n_init=10).fit(codebook)
    labels=km.predict(yx); val_labels=km.predict(yv)
    dev=torch.device(device); model=torch.nn.Linear(x.shape[1],int(clusters)).to(dev)
    tx=torch.as_tensor(x,device=dev);ty=torch.as_tensor(labels,device=dev,dtype=torch.long)
    opt=torch.optim.AdamW(model.parameters(),lr=float(lr));order=np.arange(len(x));rng=np.random.default_rng(seed)
    for _ in range(int(epochs)):
        rng.shuffle(order)
        for start in range(0,len(order),256):
            ix=order[start:start+256];loss=torch.nn.functional.cross_entropy(model(tx[ix]),ty[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad(): pred=model(torch.as_tensor(xv,device=dev)).argmax(-1).cpu().numpy()
    return {"accuracy":float(np.mean(pred==val_labels)),"labels":pred,
        "holdout_labels":val_labels,"kmeans_centers":km.cluster_centers_,"clusters":int(clusters),
        "fit_rows":len(x),"holdout_rows":len(xv),"unseen_fit_access":False,
        "classifier_state":{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}}


def masked_top1(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Stable Top-1 in the candidate-local pool, enforcing K1 exclusions."""
    s = np.asarray(scores, np.float32)
    a = np.asarray(allowed, bool)
    if s.ndim != 2 or s.shape != a.shape or np.any(a.sum(1) == 0):
        raise ValueError("score/mask mismatch or empty candidate pool")
    ix = np.argmax(np.where(a, s, -np.inf), axis=1)
    if np.any(~a[np.arange(len(ix)), ix]):
        raise AssertionError("masked candidate selected")
    return ix.astype(np.int64)


def _capture_all_layers(model, observation, row_id: int, *, include_expert: bool = True):
    """Capture layer-wise prefix V and suffix V (suffix is an expert-layer proxy).

    Gemma's public NNX scan returns final residuals and a per-layer KV cache,
    not every block residual. Prefix and suffix V projections are therefore
    captured at every block. They are layer-wise hidden-input projections,
    explicitly labelled as V-cache features rather than post-MLP residuals.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs = model_api.preprocess_observation(None, observation, train=False)
    prefix, pmask, par = model.embed_prefix(obs)
    _, pcache = model.PaliGemma.llm([prefix, None], mask=make_attn_mask(pmask, par),
                                    positions=jnp.cumsum(pmask, axis=1) - 1)
    p_v = pcache[1]
    # Attention caches retain separate KV-head and head-width axes; K1's
    # feature cache flattens them to the per-token channel dimension.
    p_v = p_v.reshape(p_v.shape[0], p_v.shape[1], p_v.shape[2], -1)
    result = {"prefix_v": p_v}
    if include_expert:
        noise_key = jax.random.fold_in(jax.random.key(SEED), jnp.asarray(row_id, jnp.uint32))
        x = jax.random.normal(noise_key, (1, model.action_horizon, model.action_dim))
        suffix, smask, sar, cond = model.embed_suffix(obs, x, jnp.ones((1,), jnp.float32))
        attn = jnp.concatenate([jnp.broadcast_to(pmask[:, None, :],
            (1, suffix.shape[1], pmask.shape[1])), make_attn_mask(smask, sar)], axis=-1)
        pos = jnp.sum(pmask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1
        (_, final), all_cache = model.PaliGemma.llm([None, suffix], mask=attn, positions=pos,
            kv_cache=pcache, adarms_cond=[None, cond])
        # The cache is prefix concatenated with suffix, so last horizon positions
        # are the action-token V projections for each transformer layer.
        suffix_v = all_cache[1][:, :, -model.action_horizon:]
        result["expert_suffix_v"] = suffix_v.reshape(suffix_v.shape[0], suffix_v.shape[1], suffix_v.shape[2], -1)
        result["expert_final"] = final[:, -model.action_horizon:]
        result["noise"] = x
    return result, obs


def _capture_batch_all_layers(model, observation, row_ids):
    """Batched all-layer prefix and t=1 suffix V-cache extraction."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs = model_api.preprocess_observation(None, observation, train=False)
    prefix, pmask, par = model.embed_prefix(obs)
    _, pcache = model.PaliGemma.llm([prefix, None], mask=make_attn_mask(pmask, par),
                                    positions=jnp.cumsum(pmask, axis=1)-1)
    b = len(row_ids)
    rid = jnp.asarray(row_ids, jnp.uint32)
    keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.key(SEED), r))(rid)
    x = jax.vmap(lambda key: jax.random.normal(key, (model.action_horizon, model.action_dim)))(keys)
    suffix, smask, sar, cond = model.embed_suffix(obs, x, jnp.ones((b,), jnp.float32))
    attn = jnp.concatenate([jnp.broadcast_to(pmask[:, None, :],
        (b, suffix.shape[1], pmask.shape[1])), make_attn_mask(smask, sar)], axis=-1)
    pos = jnp.sum(pmask, axis=1)[:, None] + jnp.cumsum(smask, axis=1)-1
    (_, final), full_cache = model.PaliGemma.llm([None, suffix], mask=attn, positions=pos,
        kv_cache=pcache, adarms_cond=[None, cond])
    prefix_v = pcache[1].reshape(pcache[1].shape[0], b, pcache[1].shape[2], -1)
    suffix_v = full_cache[1][:, :, -model.action_horizon:]
    suffix_v = suffix_v.reshape(suffix_v.shape[0], b, suffix_v.shape[2], -1)
    return prefix_v, suffix_v, final[:, -model.action_horizon:], obs.tokenized_prompt_mask


def real_model_smoke(*, output=OUT, threads=8):
    """One real LIBERO-40 observation; validates all-layer extraction shapes."""
    import torch
    torch.set_num_threads(int(threads))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".15")
    k1 = _k1()
    base, rows, unseen, splits = k1._production_data_splits(include_eval_targets=False)
    model, config = base.load_policy(k1.CHECKPOINT)
    src, ids = splits["train"]
    row_id = int(ids[0])
    obs = base.obs_batch(src, np.asarray([row_id], np.int64))
    captured, preprocessed = _capture_all_layers(model, obs, row_id, include_expert=True)
    prefix_v = np.asarray(captured["prefix_v"])
    suffix_v = np.asarray(captured["expert_suffix_v"])
    # Gemma's config exposes per-layer configs rather than a num_layers field.
    n_layers = int(prefix_v.shape[0])
    if n_layers != LAYER_COUNT:
        raise AssertionError(f"expected all {LAYER_COUNT} prefix blocks, got {n_layers}")
    pooled, mask = pool_prefix_layer_values(prefix_v, np.asarray(preprocessed.tokenized_prompt_mask))
    if prefix_v.shape[0] != n_layers or suffix_v.shape[0] != n_layers:
        raise AssertionError(f"layer count mismatch: config={n_layers}, prefix={prefix_v.shape}, suffix={suffix_v.shape}")
    if pooled.shape[:3] != (1, n_layers, 33) or suffix_v.shape[1] != 1:
        raise AssertionError(f"pooled/suffix shape mismatch: {pooled.shape}, {suffix_v.shape}")
    out = Path(output) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    # Small actual activations only; this is smoke evidence, not a production cache.
    np.savez_compressed(out / "real_model_one_row.npz", prefix=pooled.astype(np.float16),
                        prefix_mask=mask, expert_suffix_v=suffix_v.astype(np.float16),
                        expert_final=np.asarray(captured["expert_final"], np.float16),
                        row_id=np.asarray([row_id], np.int64))
    return {"status": "real_model_smoke_complete", "row_id": row_id,
        "prefix_v_shape": list(prefix_v.shape), "pooled_all_18_shape": list(pooled.shape),
        "expert_suffix_v_shape": list(suffix_v.shape),
        "expert_final_shape": list(np.asarray(captured["expert_final"]).shape),
        "n_layers": n_layers, "model_variant": config.paligemma_variant,
        "language_tokens_are_instruction_only": True,
        "state_tokens": "not present as text with discrete_state_input=False; numeric state is separate",
        "expert_feature_note": "layer-wise suffix V projections from KV cache; final residual also saved",
        "target_actions_read": False, "memory_fraction_cap": .15}


def synthetic_smoke(*, output=OUT):
    """Exercise histories, encoder variants, listwise retrieval and leakage guard."""
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(8)
    rng = np.random.default_rng(SEED)
    n = 12
    rows = {"task": np.repeat([0, 1], 6), "episode": np.repeat(np.arange(6), 2),
            "frame": np.tile([0, 10], 6), "state": rng.normal(size=(n, 8)).astype(np.float32)}
    actions = rng.normal(size=(n, 10, 7)).astype(np.float32)
    hist = build_history_arrays(rows, np.arange(n), actions)
    assert not hist["action_mask_50"][0].any()
    assert hist["action_mask_10"][1].all()
    units, umask = history_unit_codes(actions[0].reshape(-1, 7))
    assert units.shape == (3, 21) and umask.shape == (3,)
    layer_tokens = torch.tensor(rng.normal(size=(n, 33, 256)).astype(np.float32))
    token_mask = torch.ones((n, 33), dtype=torch.bool)
    state = torch.tensor(rows["state"])
    histvec = torch.tensor(rng.normal(size=(n, 64)).astype(np.float32))
    outputs = {}
    for fusion in ("attention", "concat", "visual", "history"):
        net = make_k2_layer_query(history_mode=fusion, history_dim=64)
        out = net(layer_tokens, token_mask, state, histvec)
        if isinstance(out, tuple): out = out[0]
        if out.shape != (n, 256): raise AssertionError((fusion, out.shape))
        outputs[fusion] = list(out.shape)
    allow = np.ones((n, n), bool)
    allow[0, 0] = False
    scores = rng.normal(size=(n, n)).astype(np.float32)
    pick = masked_top1(scores, allow)
    if np.any(~allow[np.arange(n), pick]): raise AssertionError("masked top1 leaked")
    # Task holdout and same-task exclusion are re-used verbatim from K1.
    k1 = _k1()
    pool = {"source_task": np.asarray([0, 1, 2, 3])}
    allowed = k1.filter_cross_task(pool, np.asarray([0, 2]), heldout=(3,))
    if allowed.tolist() != [[False, True, True, False], [True, True, False, False]]:
        raise AssertionError(f"K1 exclusion behavior changed: {allowed}")
    out = Path(output) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "synthetic_smoke_complete", "rows": n, "history_shapes":
        {k: list(v.shape) for k, v in hist.items()}, "fusion_outputs": outputs,
        "top1_pool_local": pick.tolist(), "same_task_exclusion_asserted": True,
        "eval_targets_read": False}
    _atomic_json(out / "synthetic_smoke.json", result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("smoke","extract","train","select","evaluate","report"),default="smoke")
    ap.add_argument("--output",type=Path,default=OUT);ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--batch",type=int,default=1);ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--limit",type=int,default=None);ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--epochs",type=int,default=8);ap.add_argument("--device",default="cpu")
    args=ap.parse_args();root=args.output/"smoke_pipeline" if args.smoke else args.output/"production"
    if args.stage=="smoke":
        # A true four-stage smoke is intentionally isolated from production.
        extract_stage(output=root,batch=args.batch,threads=args.threads,limit=args.limit,smoke=True)
        train_stage(output=root,threads=args.threads,batch_size=args.batch_size,epochs=1,device="cpu",smoke=True)
        select_stage(output=root);evaluate_stage(output=root,threads=args.threads,batch=32,device="cpu",smoke=True)
        result=report_stage(output=root)
    elif args.stage=="extract":result=extract_stage(output=root,batch=args.batch,threads=args.threads,limit=args.limit,smoke=args.smoke)
    elif args.stage=="train":result=train_stage(output=root,threads=args.threads,batch_size=args.batch_size,epochs=args.epochs,device=args.device,smoke=args.smoke)
    elif args.stage=="select":result=select_stage(output=root)
    elif args.stage=="evaluate":result=evaluate_stage(output=root,threads=args.threads,batch=args.batch_size,device=args.device,smoke=args.smoke)
    else:result=report_stage(output=root)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)))


def extract_stage(*, output=OUT, batch=1, threads=8, limit=None, smoke=False):
    """Resumable fp16 cache for all configured splits, storing all 18 layers.

    This stage is implemented for the later production request but is not run
    in this implementation turn. Evaluation targets are excluded from the
    loaded unseen metadata.
    """
    import torch
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    torch.set_num_threads(int(threads))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".25")
    k1 = _k1()
    base, rows, unseen, splits = k1._production_data_splits(include_eval_targets=False)
    model, config = base.load_policy(k1.CHECKPOINT)
    root = Path(output) / "features"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    selected_ids={}
    if smoke:
        train_source,train_all=splits["train"]
        tasks=np.asarray(train_source["task"],np.int64)
        hold=set(k1.deterministic_holdout_tasks(tasks))
        chosen=[]
        for task in sorted(set(tasks.tolist())):
            match=np.asarray(train_all)[tasks[np.asarray(train_all)]==task]
            if len(match):chosen.append(int(match[0]))
        if not (set(np.asarray(train_source["task"])[np.asarray(chosen)].tolist())&hold and
                set(np.asarray(train_source["task"])[np.asarray(chosen)].tolist())-hold):
            raise AssertionError("smoke rows must cover both fit and holdout tasks")
        selected_ids["train"]=np.asarray(chosen,np.int64)
        for split in ("dev","eval"):
            selected_ids[split]=np.asarray(splits[split][1],np.int64)[:4]
    else:
        for split,(_,all_ids) in splits.items():
            ids=np.asarray(all_ids,np.int64)
            if limit is not None:ids=ids[:int(limit)]
            selected_ids[split]=ids
    counts={name:len(ids) for name,ids in selected_ids.items()}
    hashes={name:hashlib.sha256(ids.tobytes()).hexdigest() for name,ids in selected_ids.items()}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("complete") and previous.get("row_counts") == counts and previous.get("row_ids_sha256")==hashes:
            return previous
        raise FileExistsError("feature cache exists with incompatible/incomplete manifest; choose new output")
    import flax.nnx as nnx
    graphdef, state = nnx.split(model)
    def run(model_state, obs, row_ids):
        live = nnx.merge(graphdef, model_state)
        return _capture_batch_all_layers(live, obs, row_ids)
    fn = jax.jit(run)
    from openpi.models import gemma
    gemma_config = gemma.get_config(config.paligemma_variant)
    layer_dim = int(gemma_config.num_kv_heads * gemma_config.head_dim)
    for split, (source, all_ids) in splits.items():
        ids = selected_ids[split]
        stem = root / split
        path = stem.with_suffix(".npy"); state_path = root / f"{split}.progress.json"
        if path.exists() and not state_path.exists(): raise FileExistsError(path)
        done = int(json.loads(state_path.read_text()).get("done", 0)) if state_path.exists() else 0
        # Observations contain two active camera grids, a masked third slot,
        # and variable instruction text. Cache layout is padded [18,33,D].
        array = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
            dtype=np.float16, shape=(len(ids), LAYER_COUNT, 33, layer_dim))
        masks = np.lib.format.open_memmap(root/f"{split}_mask.npy", mode="r+" if (root/f"{split}_mask.npy").exists() else "w+",
            dtype=np.bool_, shape=(len(ids), 33))
        suffix_path = root/f"{split}_expert_suffix_v.npy"
        final_path = root/f"{split}_expert_final.npy"
        suffix = np.lib.format.open_memmap(suffix_path, mode="r+" if suffix_path.exists() else "w+",
            dtype=np.float16, shape=(len(ids), LAYER_COUNT, 10, layer_dim))
        final_dim = int(model.action_out_proj.kernel.value.shape[0])
        final = np.lib.format.open_memmap(final_path, mode="r+" if final_path.exists() else "w+",
            dtype=np.float16, shape=(len(ids), 10, final_dim))
        state_values = np.lib.format.open_memmap(root/f"{split}_state.npy",
            mode="r+" if (root/f"{split}_state.npy").exists() else "w+",
            dtype=np.float16, shape=(len(ids), 8))
        row_ids_path=root/f"{split}_row_ids.npy"
        if row_ids_path.exists():
            old_ids=np.load(row_ids_path,allow_pickle=False)
            if not np.array_equal(old_ids,ids):raise FileExistsError(f"row-ID manifest mismatch: {row_ids_path}")
        else:np.save(row_ids_path,ids,allow_pickle=False)
        for i in range(done, len(ids), int(batch)):
            chunk = ids[i:min(i+int(batch), len(ids))]
            obs = base.obs_batch(source, chunk)
            cache_v, suffix_v, final_h, prompt_mask = fn(state, obs, jnp.asarray(chunk, jnp.uint32))
            cache_v, suffix_v, final_h = map(np.asarray, (cache_v, suffix_v, final_h))
            lm = np.asarray(prompt_mask, bool)
            pooled, mask = pool_prefix_layer_values(cache_v, lm)
            if pooled.shape[1] != LAYER_COUNT or pooled.shape[2] != 33:
                raise ValueError(f"unexpected all-layer prefix output {pooled.shape}")
            end = i + len(chunk)
            array[i:end] = pooled.astype(np.float16); masks[i:end] = mask
            suffix[i:end] = suffix_v.transpose(1, 0, 2, 3).astype(np.float16)
            final[i:end] = final_h.astype(np.float16)
            state_values[i:end] = np.asarray(source["state"][chunk, :8], np.float16)
            if end % 16 == 0 or end == len(ids):
                array.flush(); masks.flush()
                suffix.flush(); final.flush(); state_values.flush()
                _atomic_json(state_path, {"split": split, "done": end, "total": len(ids), "complete": end == len(ids)})
        del array, masks, suffix, final, state_values
    manifest = {"complete": True, "row_counts": counts, "row_ids_sha256":hashes,
        "smoke":bool(smoke),"layers": list(range(LAYER_COUNT)),
        "prefix_cache": "per-layer KV V projections", "expert_suffix_cache": "per-layer action suffix KV V at fixed-noise t=1; final suffix residual separately stored",
        "image_pool": "4x4 per two active cameras",
        "instruction_group": "valid tokenizer prompt mean", "state_group": "numeric state kept separate (discrete_state_input=False)",
        "unseen_actions_loaded": False, "reusable": True}
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_k2_context(root, *, target_actions=False):
    """Load cache arrays in the exact K1 split order; labels are opt-in."""
    k1 = _k1()
    base, rows, unseen, splits = k1._production_data_splits(include_eval_targets=False)
    froot = Path(root) / "features"
    manifest = json.loads((froot / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError("K2 extraction is incomplete")
    data, meta = {}, {}
    for split, (source, expected) in splits.items():
        ids = np.load(froot / f"{split}_row_ids.npy", allow_pickle=False)
        exp=np.asarray(expected,np.int64)
        if manifest.get("smoke"):
            if len(np.unique(ids))!=len(ids) or not np.isin(ids,exp).all():
                raise ValueError(f"K2 {split} smoke row IDs are not a unique subset of K1")
        elif not np.array_equal(ids,exp[:len(ids)]):
            raise ValueError(f"K2 {split} row IDs do not match K1 compact split")
        data[split] = {"tokens": np.load(froot / f"{split}.npy", mmap_mode="r", allow_pickle=False),
            "mask": np.load(froot / f"{split}_mask.npy", mmap_mode="r", allow_pickle=False),
            "suffix": np.load(froot / f"{split}_expert_suffix_v.npy", mmap_mode="r", allow_pickle=False),
            "final": np.load(froot / f"{split}_expert_final.npy", mmap_mode="r", allow_pickle=False),
            "state": np.load(froot / f"{split}_state.npy", mmap_mode="r", allow_pickle=False), "ids": ids}
        meta[split] = {k: np.asarray(source[k])[ids] for k in ("task", "episode", "frame")}
    if target_actions:
        # The unseen target mmap is opened only by evaluate, never by extract,
        # fit, or selection. Dev actions are used for reporting only as well.
        unseen_actions = np.load(k1.UNSEEN / "actions.npy", mmap_mode="r", allow_pickle=False)
        for split in ("dev", "eval"):
            data[split]["actions"] = np.asarray(unseen_actions[data[split]["ids"]], np.float32)[..., :7]
    return k1, rows, unseen, splits, data, meta


def _k2_pool(k1, excluded_tasks=()):
    theta = k1._load_theta()
    pool0 = theta.candidate_shape_pool(rep="C_vm_none")
    keep = ~np.isin(np.asarray(pool0["source_task"], np.int64), np.asarray(excluded_tasks, np.int64))
    return theta, {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v)
                   for k, v in pool0.items()}


def _candidate_metrics(k1, theta, pool, errors, allowed, rank, target, hard):
    rank = np.asarray(rank, np.int64); first = rank[:, 0]
    shape_mse = errors[np.arange(len(first)), first]
    # Caller supplies decoded action MSE separately; retain action-independent
    # retrieval metrics here to avoid accidental target use in training.
    positive = (errors <= .01) & allowed
    hit5 = positive[np.arange(len(first))[:, None], rank[:, :min(5, rank.shape[1])]].any(axis=1)
    result = {"rows":len(first), "shape_mse":float(shape_mse.mean()),
        "shape_mse_median":float(np.median(shape_mse)),
        "recall_at_1_eps_0.01":float(np.mean(positive[np.arange(len(first)), first])),
        "recall_at_5_eps_0.01":float(hit5.mean()),
        "hard_rows":int(np.asarray(hard, bool).sum())}
    hard=np.asarray(hard,bool)
    if hard.any():
        result["hard"]={"rows":int(hard.sum()),"shape_mse":float(shape_mse[hard].mean()),
            "recall_at_5_eps_0.01":float(hit5[hard].mean())}
    return result


def _fit_save_retriever(name, tokens, mask, state, hist, shapes, teacher, allowed,
                        fit_ids, val_ids, val_errors, outdir, *, epochs, batch_size,
                        device, threads, histories_meta, history_mode="attention", dropout=0., layer_ids=None,
                        candidate_actions=None, target_actions=None):
    import torch
    ckpt=Path(outdir)/f"{name}.pt"
    if ckpt.exists():
        saved=torch.load(ckpt,map_location="cpu",weights_only=False)
        return saved["record"]
    model=fit_layer_retriever(tokens,mask,state,hist,shapes,teacher,allowed,fit_ids,
        epochs=epochs,batch_size=batch_size,history_mode=history_mode,dropout=dropout,
        device=device,threads=threads,train_metadata=histories_meta,hard_negative_weight=.05)
    scores,rank=score_layer_retriever(model,tokens[val_ids],mask[val_ids],state[val_ids],hist[val_ids],
        shapes,allowed[val_ids],device=device,batch_size=batch_size)
    val_e=val_errors[val_ids]
    mse=np.nan_to_num(val_e[np.arange(len(val_ids)),rank[:,0]],nan=1e6,posinf=1e6,neginf=1e6)
    record={"name":name,"checkpoint":str(ckpt),"val_shape_mse":float(mse.mean()),
        "val_shape_mse_median":float(np.median(mse)),"val_rows":len(val_ids),"fit_rows":len(fit_ids),
        "history_mode":history_mode,"history_dropout":float(dropout)}
    if candidate_actions is not None and target_actions is not None:
        chosen=np.asarray(candidate_actions,np.float32)[rank[:,0]]
        true=np.asarray(target_actions,np.float32)[val_ids]
        record["val_theta_src_action_mse"]=float(np.mean((chosen-true)**2))
    if layer_ids is not None:
        record["layers"]=list(map(int,layer_ids))
    elif "_l" in name:
        try:record["layers"]=[int(name.split("_l",1)[1][:2])]
        except ValueError:pass
    hit=(val_e <= .01) & allowed[val_ids]
    record["recall_at_5_eps_0.01"]=float(np.mean(hit[np.arange(len(val_ids))[:,None],rank[:,:min(5,rank.shape[1])]].any(1)))
    _atomic_torch_save(torch,{"query":model["query"].state_dict(),"key":model["key"].state_dict(),
        "token_dim":int(tokens.shape[-1]),"history_dim":int(hist.shape[-1]),
        "history_mode":history_mode,"dropout":float(dropout),"record":record},ckpt)
    return record


def train_stage(*, output=OUT, threads=8, batch_size=128, epochs=8, device="cpu", smoke=False):
    """Train the layer, layer-combination, and executed-history comparisons."""
    import torch
    torch.set_num_threads(int(threads))
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20)
    root=Path(output);models=root/"models";models.mkdir(parents=True,exist_ok=True)
    summary_path=root/"train_summary.json"
    if summary_path.exists():return json.loads(summary_path.read_text())
    k1,rows,unseen,splits,data,meta=_load_k2_context(root)
    theta,pool=_k2_pool(k1)
    tr=data["train"]; ids=np.asarray(tr["ids"],np.int64); task=np.asarray(rows["task"][ids],np.int64)
    hold=tuple(k1.deterministic_holdout_tasks(task))
    keep=~np.isin(np.asarray(pool["source_task"],np.int64),hold)
    pool={k:(np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v) for k,v in pool.items()}
    allowed=k1.filter_cross_task(pool,task,heldout=hold)
    import sys as _sys
    if str(HERE) not in _sys.path:_sys.path.insert(0,str(HERE))
    import variable_shape_memory as vs
    actions=np.asarray(rows["actions"][ids,:,:7],np.float32)
    qshape=k1._motion_shape_from_actions(actions,vs)
    errors=k1.shape_error_matrix(qshape,pool["shape"],vs,batch=512 if device.startswith("cuda") else 8,device=device)
    errors=np.nan_to_num(errors,nan=1e6,posinf=1e6,neginf=1e6)
    errors[~allowed]=np.inf
    # Degenerate Procrustes fits occasionally return +inf for an otherwise
    # valid candidate (e.g. zero-motion fragments). Keep them rankable last
    # without contaminating means/soft targets with infinities.
    teacher=k1.target_distribution(errors,allowed,.02)
    fit_ids=np.flatnonzero(~np.isin(task,hold));val_ids=np.flatnonzero(np.isin(task,hold))
    if not len(fit_ids) or not len(val_ids):raise ValueError("empty K2 fit/holdout task split")
    histories=build_history_arrays(rows,ids,rows["actions"])
    hist_meta={"episode":np.asarray(rows["episode"][ids]),"frame":np.asarray(rows["frame"][ids]),
        "task":task,"hard_context":histories["action_10"].reshape(len(ids),-1)}
    zero=np.zeros((len(ids),1),np.float32);records={};probes={}
    candidate_action_src=k1.decode_pool_actions(pool,np.arange(len(pool["shape"]),dtype=np.int64)[None,:],vs,theta_source="src")[0]
    prefix_layers=(9,) if smoke else tuple(range(LAYER_COUNT))
    expert_layers=(9,) if smoke else tuple(range(LAYER_COUNT))
    epochs=int(1 if smoke else epochs)
    for family,source_key,layers in (("prefix","tokens",prefix_layers),("expert_v","suffix",expert_layers)):
        values=np.asarray(tr[source_key],np.float32)
        for layer in layers:
            tok=values[:,layer];mask=np.asarray(tr["mask"],bool) if family=="prefix" else np.ones(tok.shape[:2],bool)
            name=f"{family}_l{layer:02d}_visual"
            records[name]=_fit_save_retriever(name,tok,mask,np.asarray(tr["state"],np.float32),zero,
                pool["shape"],teacher,allowed,fit_ids,val_ids,errors,models,epochs=epochs,batch_size=batch_size,
                device=device,threads=threads,histories_meta=hist_meta,history_mode="visual",
                candidate_actions=candidate_action_src,target_actions=actions)
            # K=32 code probe uses only the memory codebook and LIBERO-40 fit labels.
            code_path=models/f"{family}_l{layer:02d}_type_probe.npz"
            if not code_path.exists():
                code=fit_type_code_linear_probe(tok.mean(1)[fit_ids],k1._motion_shape_from_actions(actions[fit_ids],vs),
                    tok.mean(1)[val_ids],k1._motion_shape_from_actions(actions[val_ids],vs),clusters=32,seed=SEED,
                    epochs=20 if smoke else 100,threads=threads,device=device,
                    codebook_shapes=np.asarray(pool["shape"]))
                rec={"accuracy":code["accuracy"],"fit_rows":code["fit_rows"],"holdout_rows":code["holdout_rows"],"clusters":32}
                np.savez_compressed(code_path,weight=code["classifier_state"]["weight"],bias=code["classifier_state"]["bias"],
                    centers=code["kmeans_centers"],summary=np.asarray(json.dumps(rec)))
            else:rec=json.loads(str(np.load(code_path,allow_pickle=False)["summary"].item()))
            probes[name]=rec
    # Best visual layer is chosen only from LIBERO-40 holdout rows.
    best_prefix=min((x for x in records if x.startswith("prefix_l")),key=lambda x:(records[x]["val_shape_mse"],x))
    best_layer=int(best_prefix.split("_l")[1][:2]);v=np.asarray(tr["tokens"],np.float32);best_tokens=v[:,best_layer]
    configs=[]
    if smoke:
        configs=[(10,"h1","concat",0.),(10,"h2","attention",0.),(10,"h3","attention",0.),
                 (10,"h4","attention",0.),(10,"h1","visual",0.),(10,"h1","history",0.)]
    else:
        configs=[(n,e,f,d) for n in HISTORY_LENGTHS for e in ("h1","h2","h3","h4")
                 for f in ("attention","concat") for d in (0.,.3,.5)]
        configs += [(10,"h1","visual",0.),(10,"h1","history",0.)]
    for length,enc,fusion,drop in configs:
        gru_state=None
        if enc=="h2":
            _,gru_state,_=_fit_gru_history_encoder(histories,actions,length,fit_ids,val_ids,
                epochs=3 if smoke else 12,batch_size=batch_size,threads=threads,device=device)
        histv=_history_variant_arrays(rows,ids,histories,length=length,encoder=enc,gru_state=gru_state,device="cpu",threads=threads)
        name=f"history_l{best_layer:02d}_{enc}_{fusion}_h{length}_drop{drop:g}"
        records[name]=_fit_save_retriever(name,best_tokens,np.asarray(tr["mask"],bool),np.asarray(tr["state"],np.float32),histv,
            pool["shape"],teacher,allowed,fit_ids,val_ids,errors,models,epochs=epochs,batch_size=batch_size,
            device=device,threads=threads,histories_meta=hist_meta,history_mode=fusion,dropout=drop,
            candidate_actions=candidate_action_src,target_actions=actions)
        if gru_state is not None:_atomic_torch_save(torch,gru_state,models/f"{name}_gru.pt")
    layer_scores={int(n.split("_l")[1][:2]):r["val_shape_mse"] for n,r in records.items() if n.startswith("prefix_l") and n.endswith("visual")}
    top3=select_top_layers(layer_scores,count=min(3,len(layer_scores)))
    if not smoke:
        combo=combine_layer_tokens(v,top3,method="concat");combo_mask=np.ones(combo.shape[:2],bool)
        name="prefix_top3_concat"
        records[name]=_fit_save_retriever(name,combo,combo_mask,np.asarray(tr["state"],np.float32),zero,
            pool["shape"],teacher,allowed,fit_ids,val_ids,errors,models,epochs=epochs,batch_size=batch_size,
            device=device,threads=threads,histories_meta=hist_meta,history_mode="visual",layer_ids=top3,
            candidate_actions=candidate_action_src,target_actions=actions)
        mixer=fit_elmo_layer_mixer(v,np.asarray(tr["mask"],bool),np.asarray(tr["state"],np.float32),zero,
            pool["shape"],teacher,allowed,fit_ids,epochs=epochs,batch_size=batch_size,device=device,threads=threads)
        scores,rank=score_elmo_layer_mixer(mixer,v[val_ids],np.asarray(tr["mask"],bool)[val_ids],
            np.asarray(tr["state"],np.float32)[val_ids],zero[val_ids],pool["shape"],allowed[val_ids],device=device)
        name="prefix_elmo_all";ck=models/f"{name}.pt";_atomic_torch_save(torch,{"elmo":mixer["module"].state_dict(),
            "record":{"name":name,"checkpoint":str(ck),"layers":list(range(LAYER_COUNT)),
                      "history_mode":"visual","history_dropout":0.,"val_shape_mse":float(errors[val_ids,rank[:,0]].mean())}},ck)
        elmo_actions=candidate_action_src[rank[:,0]]
        records[name]={"name":name,"checkpoint":str(ck),"val_shape_mse":float(errors[val_ids,rank[:,0]].mean()),
            "val_theta_src_action_mse":float(np.mean((elmo_actions-actions[val_ids])**2)),
            "recall_at_5_eps_0.01":float(np.mean(((errors[val_ids]<=.01)&allowed[val_ids])[
                np.arange(len(val_ids))[:,None],rank[:,:min(5,rank.shape[1])]].any(1))),
            "layer_weights":mixer["layer_weights"].tolist(),"layers":list(range(LAYER_COUNT)),"history_mode":"visual"}
    summary={"status":"complete","smoke":bool(smoke),"holdout_tasks":list(map(int,hold)),
        "fit_tasks":[int(x) for x in np.unique(task) if x not in hold],"best_visual_layer":best_layer,
        "top3_layers":list(map(int,top3)),"candidate_count":len(pool["shape"]),"models":records,
        "type_probes":probes,"train_rows":len(ids),"unseen_target_actions_read":False,
        "expert_layer_representation":"suffix V-cache projection proxy; final residual cached separately",
        "resume_policy":"per-model checkpoints are reused; completed train_summary is immutable"}
    _atomic_json(summary_path,summary);return summary


def select_stage(*, output=OUT):
    root=Path(output);path=root/"train_summary.json"
    if not path.exists():raise FileNotFoundError("run train first")
    train=json.loads(path.read_text());models=train["models"]
    primary=min(models,key=lambda n:(float(models[n]["val_shape_mse"]),n))
    groups={}
    for name,rec in models.items():
        group="history" if name.startswith("history_") else ("expert_layer" if name.startswith("expert_v") else "prefix_layer")
        if group not in groups or rec["val_shape_mse"]<groups[group]["val_shape_mse"]:groups[group]=rec
    visual=[(n,r) for n,r in models.items() if n.startswith("prefix_l") and n.endswith("_visual")]
    hist_only=[(n,r) for n,r in models.items() if n.startswith("history_") and "_history_" in n]
    both=[(n,r) for n,r in models.items() if n.startswith("history_") and ("_attention_" in n or "_concat_" in n)]
    if visual:groups["visual_only"]=min(visual,key=lambda z:(z[1]["val_shape_mse"],z[0]))[1]
    if hist_only:groups["history_only"]=min(hist_only,key=lambda z:(z[1]["val_shape_mse"],z[0]))[1]
    if both:groups["visual_plus_history"]=min(both,key=lambda z:(z[1]["val_shape_mse"],z[0]))[1]
    if "prefix_top3_concat" in models:groups["prefix_top3_concat"]=models["prefix_top3_concat"]
    if "prefix_elmo_all" in models:groups["prefix_elmo_all"]=models["prefix_elmo_all"]
    result={"status":"selected","primary":primary,"primary_config":models[primary],
        "best_by_family":groups,"holdout_tasks":train["holdout_tasks"],
        "selection_source":"LIBERO-40 fixed task holdout only","unseen_targets_used":False}
    _atomic_json(root/"selection_summary.json",result);return result


def evaluate_stage(*, output=OUT, threads=8, batch=128, device="cpu", smoke=False):
    import torch
    torch.set_num_threads(int(threads));root=Path(output);out=root/"evaluation"
    if (out/"summary.json").exists():return json.loads((out/"summary.json").read_text())
    if not (root/"selection_summary.json").exists():raise FileNotFoundError("run select first")
    k1,rows,unseen,splits,data,meta=_load_k2_context(root,target_actions=True)
    train=json.loads((root/"train_summary.json").read_text());selection=json.loads((root/"selection_summary.json").read_text())
    theta,pool=_k2_pool(k1,train["holdout_tasks"])
    import sys as _sys
    if str(HERE) not in _sys.path:_sys.path.insert(0,str(HERE))
    import variable_shape_memory as vs
    all_results={};saved={}
    for split in ("dev","eval"):
        d=data[split];n=min(len(d["ids"]),4) if smoke else len(d["ids"])
        ids=np.asarray(d["ids"][:n]);actions=np.asarray(d["actions"][:n],np.float32)
        tasks=np.asarray(meta[split]["task"][:n],np.int64);allowed=k1.filter_cross_task(pool,tasks,heldout=train["holdout_tasks"])
        qshape=k1._motion_shape_from_actions(actions,vs);errors=k1.shape_error_matrix(qshape,pool["shape"],vs,batch=8)
        errors=np.nan_to_num(errors,nan=1e6,posinf=1e6,neginf=1e6);errors[~allowed]=np.inf
        try:
            hard_cache=np.load(k1.POOL_DIR/f"C_vm_none_{split}.npz",allow_pickle=False)
            hard=np.asarray(hard_cache["hard"],bool)[:n]
            unit_ids=np.asarray(hard_cache["unit_ids"],np.int64)[:n]
        except (FileNotFoundError,KeyError):
            hard=np.zeros(n,bool);unit_ids=np.full((n,10),-1,np.int64)
        episode_ids=np.asarray(meta[split]["task"][:n],np.int64)*100000+np.asarray(meta[split]["episode"][:n],np.int64)
        boundary=np.zeros(n,bool);comparable=np.zeros(n,bool)
        for i in range(1,n):
            if meta[split]["task"][i]==meta[split]["task"][i-1] and meta[split]["episode"][i]==meta[split]["episode"][i-1] and meta[split]["frame"][i]==meta[split]["frame"][i-1]+10:
                prev=unit_ids[i-1][unit_ids[i-1]>=0];cur=unit_ids[i][unit_ids[i]>=0]
                if len(prev) and len(cur):comparable[i]=True;boundary[i]=prev[-1]!=cur[0]
        split_metrics={};split_saved={}
        for family,rec in selection["best_by_family"].items():
            ck=Path(rec["checkpoint"])
            if not ck.exists():continue
            ckdata=torch.load(ck,map_location="cpu",weights_only=False)
            name=rec.get("name",Path(ck).stem);is_expert=name.startswith("expert_v")
            layers=list(map(int,rec.get("layers",[train["best_visual_layer"]])))
            if name=="prefix_elmo_all":
                tok=np.asarray(d["tokens"][:n],np.float32);mask=np.asarray(d["mask"][:n],bool)
            elif name=="prefix_top3_concat":
                tok=combine_layer_tokens(np.asarray(d["tokens"][:n],np.float32),tuple(layers),method="concat")
                mask=np.ones(tok.shape[:2],bool)
            else:
                layer=layers[0]
                tok=np.asarray(d["suffix" if is_expert else "tokens"][:n,layer],np.float32)
                mask=np.ones(tok.shape[:2],bool) if is_expert else np.asarray(d["mask"][:n],bool)
            if name.startswith("history_"):
                parts=name.split("_");enc=parts[2];fusion=parts[3];length=int(parts[4][1:]);drop=float(parts[5].removeprefix("drop"))
                histories=build_history_arrays(unseen,ids,np.load(k1.UNSEEN/"actions.npy",mmap_mode="r",allow_pickle=False))
                gs=torch.load(ck.parent/f"{name}_gru.pt",map_location="cpu",weights_only=False) if enc=="h2" else None
                histv=_history_variant_arrays(unseen,ids,histories,length=length,encoder=enc,gru_state=gs,device="cpu",threads=threads)
            else:fusion="visual";histv=np.zeros((n,1),np.float32)
            if name=="prefix_elmo_all":
                mixer=K2ElmoMixer(tok.shape[1],tok.shape[-1],histv.shape[-1],fusion)
                mixer.load_state_dict(ckdata["elmo"])
                scores,rank=score_elmo_layer_mixer({"module":mixer},tok,mask,np.asarray(d["state"][:n],np.float32),
                    histv,pool["shape"],allowed,device=device,batch_size=batch)
            else:
                query=make_k2_layer_query(token_dim=tok.shape[-1],history_dim=histv.shape[-1],history_mode=fusion)
                key=make_candidate_key(70);query.load_state_dict(ckdata["query"]);key.load_state_dict(ckdata["key"])
                scores,rank=score_layer_retriever({"query":query,"key":key},tok,mask,np.asarray(d["state"][:n],np.float32),
                    histv,pool["shape"],allowed,device=device,batch_size=batch)
            metrics=_candidate_metrics(k1,theta,pool,errors,allowed,rank,actions,hard)
            action_pred=k1.decode_pool_actions(pool,rank[:,:1],vs,theta_source="src")[:,0]
            metrics["theta_src_action_mse"]=float(np.mean((action_pred-actions)**2))
            if hard.any():metrics["hard"]["theta_src_action_mse"]=float(np.mean((action_pred[hard]-actions[hard])**2))
            metrics["shape_mse_episode_bootstrap"]=k1.episode_bootstrap(errors[np.arange(n),rank[:,0]],episode_ids)
            metrics["theta_src_action_mse_episode_bootstrap"]=k1.episode_bootstrap(np.mean((action_pred-actions)**2,axis=(1,2)),episode_ids)
            metrics["unit_transition_strata"]={}
            per_row_mse=np.mean((action_pred-actions)**2,axis=(1,2))
            for group,mask_group in (("immediately_after_switch",comparable&boundary),("within_unit",comparable&~boundary)):
                metrics["unit_transition_strata"][group]={"rows":int(mask_group.sum()),
                    "theta_src_action_mse":float(per_row_mse[mask_group].mean()) if mask_group.any() else None}
            split_metrics[family]=metrics;split_saved[f"rank_{family}"]=rank[:,:5];split_saved[f"action_{family}"]=action_pred
        all_results[split]=split_metrics;saved[split]=split_saved
    out.mkdir(parents=True,exist_ok=False)
    _atomic_npz(out/"rows.npz",**{f"{s}_{k}":v for s,vsaved in saved.items() for k,v in vsaved.items()})
    result={"status":"complete","splits":all_results,"rows":{"dev":len(data["dev"]["ids"]),"eval":len(data["eval"]["ids"])},
        "smoke":bool(smoke),"unseen_action_labels_used_only_for_scoring":True,
        "hard_mask_source":"K1 C_vm_none split cache"}
    _atomic_json(out/"summary.json",result);return result


def report_stage(*, output=OUT):
    import matplotlib.pyplot as plt
    root=Path(output);train=json.loads((root/"train_summary.json").read_text());selection=json.loads((root/"selection_summary.json").read_text())
    ev=json.loads((root/"evaluation"/"summary.json").read_text());figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True)
    layers=sorted((int(n.split("_l")[1][:2]),v["val_shape_mse"],"prefix") for n,v in train["models"].items() if n.startswith("prefix_l") and n.endswith("visual"))
    expert_layers=sorted((int(n.split("_l")[1][:2]),v["val_shape_mse"],"expert-V") for n,v in train["models"].items() if n.startswith("expert_v_l") and n.endswith("visual"))
    if layers:
        fig,ax=plt.subplots(figsize=(9,4))
        for group,records in (("prefix",layers),("Expert suffix V",expert_layers)):
            if records:ax.plot([r[0] for r in records],[r[1] for r in records],marker="o",linewidth=2,label=group)
        ax.set_xlabel("Layer index");ax.set_ylabel("LIBERO-40 holdout shape MSE");ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(figdir/"layer_curves.png",dpi=180);plt.close(fig)
    lines=["# K2 層・履歴検索", "",f"- ステージ: train={train['status']}, select={selection['status']}, evaluate={ev['status']}",
        f"- 保留タスク: {train['holdout_tasks']}（選択にだけ使用）",f"- primary: `{selection['primary']}`",
        "- Expert の層別特徴は suffix V-cache 射影を使用。残差 hidden の層別取得ではない。",
        "- 未学習 Eval の行動ラベルは evaluate で採点する時のみ読み込んだ。", "",
        "## 保留タスク層曲線", "", "| Family | 層 | Holdout shape MSE | Recall@5 (ε=.01) | K=32 linear type accuracy | θ_src action MSE |", "|---|---:|---:|---:|---:|---:|"]
    for layer,val,_family in layers:
        name=f"prefix_l{layer:02d}_visual";rec=train["models"][name]
        probe=train["type_probes"].get(name,{})
        lines.append(f"| Prefix | {layer} | {val:.6f} | {rec.get('recall_at_5_eps_0.01',float('nan')):.3f} | {probe.get('accuracy',float('nan')):.3f} | {rec.get('val_theta_src_action_mse',float('nan')):.6f} |")
    for layer,val,_ in expert_layers:
        name=f"expert_v_l{layer:02d}_visual";rec=train["models"][name]
        lines.append(f"| Expert suffix V | {layer} | {val:.6f} | {rec.get('recall_at_5_eps_0.01',float('nan')):.3f} | — | {rec.get('val_theta_src_action_mse',float('nan')):.6f} |")
    lines += ["", "## 未学習セット", "", "| Split | 条件 | Shape MSE (95% episode CI) | θ_src action MSE | Recall@5 (ε=.01) |", "|---|---|---:|---:|---:|"]
    for split,methods in ev["splits"].items():
        for name,m in methods.items():
            ci=m.get("shape_mse_episode_bootstrap",{}).get("ci95",[float('nan'),float('nan')])
            lines.append(f"| {split} | {name} | {m['shape_mse']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {m['theta_src_action_mse']:.6f} | {m['recall_at_5_eps_0.01']:.3f} |")
            if "unit_transition_strata" in m:
                for group,values in m["unit_transition_strata"].items():lines.append(f"| {split} / {name} / {group} | {values['rows']} rows; action MSE {values['theta_src_action_mse']} | — | — | — |")
    path=root/"report.md"
    if path.exists():
        ix=1
        while (root/f"report_update_{ix}.md").exists():ix+=1
        path=root/f"report_update_{ix}.md"
    path.write_text("\n".join(lines)+"\n")
    return {"status":"reported","report":str(path),"figure":str(figdir/"layer_curves.png")}


def run_pipeline_stage(stage, *, output=OUT, threads=8, batch=1, batch_size=128,
                       limit=None, epochs=8, device="cpu", smoke=False):
    """Run one resumable K2 stage without modifying K1."""
    if stage=="extract":return extract_stage(output=output,batch=batch,threads=threads,limit=limit,smoke=smoke)
    if stage=="train":return train_stage(output=output,threads=threads,batch_size=batch_size,epochs=epochs,device=device,smoke=smoke)
    if stage=="select":return select_stage(output=output)
    if stage=="evaluate":return evaluate_stage(output=output,threads=threads,batch=batch_size,device=device,smoke=smoke)
    if stage=="report":return report_stage(output=output)
    raise ValueError(f"unknown K2 stage {stage}")


if __name__ == "__main__":
    main()
