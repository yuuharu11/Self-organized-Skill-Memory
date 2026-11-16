#!/usr/bin/env python3
"""GH1: predict a ten-step gripper trace from frozen shared latent tokens.

The predictors accept only E's latent tokens and the current two-dimensional
gripper proprioception.  Targets, task identifiers and π0.5 actions are kept in
the dataset/evaluation layer and are never model inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OUT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/gripper-latent-01')
KEY2_ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production')
KEY2_RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
KEY2_MISSING = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing')
KEY2_SEEN_CACHE = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache')
KEY2_UNSEEN_CACHE = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5')
SEED = 1
HORIZON = 10
GRIPPER_STATE_DIM = 2
DEV_TASKS = (6, 7)
EVAL_TASKS = (0, 10, 13, 31, 35, 43, 57, 61, 84, 88)


def validate_model_inputs(inputs: Mapping[str, Any]) -> None:
    """Fail closed on labels, IDs, or π0.5 predictions in the predictor input."""
    required = {'latent_tokens', 'gripper_state'}
    allowed = required
    keys = set(inputs)
    if not required <= keys:
        raise ValueError(f'missing model inputs: {sorted(required - keys)}')
    forbidden = keys - allowed
    if forbidden:
        raise ValueError(f'non-causal or privileged model inputs are forbidden: {sorted(forbidden)}')


class LatentGripperHead(nn.Module):
    """Ten learned future queries cross-attend to frozen E tokens."""

    def __init__(self, latent_dim: int, *, hidden_dim: int = 128, heads: int = 4,
                 horizon: int = HORIZON, state_dim: int = GRIPPER_STATE_DIM,
                 auxiliary_switch: bool = False):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.auxiliary_switch = bool(auxiliary_switch)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.history_proj = nn.Linear(20, hidden_dim, bias=False)
        self.query = nn.Parameter(torch.randn(1, horizon, hidden_dim) * .02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.logit = nn.Linear(hidden_dim, 1)
        self.switch_horizon = nn.Linear(hidden_dim, 11) if auxiliary_switch else None
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))

    def forward(self, latent_tokens: torch.Tensor, gripper_state: torch.Tensor,
                history_action: torch.Tensor | None = None,
                history_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if latent_tokens.ndim != 3:
            raise ValueError('latent_tokens must have shape [B,K,D]')
        if gripper_state.shape != (latent_tokens.shape[0], self.state_dim):
            raise ValueError(f'gripper_state must have shape [B,{self.state_dim}]')
        memory = self.latent_proj(latent_tokens.float())
        state = self.state_proj(((gripper_state.float() - self.state_mean) / self.state_std).clamp(-8, 8)).unsqueeze(1)
        if history_action is None:
            history_action = torch.zeros((len(memory), 10), device=memory.device)
        if history_valid is None:
            history_valid = torch.zeros((len(memory), 10), dtype=torch.bool, device=memory.device)
        if history_action.shape != (len(memory), 10) or history_valid.shape != (len(memory), 10):
            raise ValueError('action history and valid flag must be [B,10]')
        hist = torch.stack((history_action.float(), history_valid.float()), dim=-1).reshape(len(memory), 20)
        state = state + self.history_proj(hist).unsqueeze(1)
        q = self.query.expand(len(memory), -1, -1) + state
        attended, _ = self.attention(q, memory, memory, need_weights=False)
        hidden = self.norm(q + attended)
        out = {'logits': self.logit(hidden).squeeze(-1)}
        if self.switch_horizon is not None:
            out['switch_horizon_logits'] = self.switch_horizon(hidden[:, 0])
        return out


class LatentGripperMLP(nn.Module):
    """Compact pooled-latent/state baseline with the same ten-logit output."""

    def __init__(self, latent_dim: int, *, hidden_dim: int = 128,
                 horizon: int = HORIZON, state_dim: int = GRIPPER_STATE_DIM):
        super().__init__()
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.net = nn.Sequential(nn.Linear(latent_dim + state_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon))

    def forward(self, latent_tokens: torch.Tensor, gripper_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_tokens.ndim not in (2, 3) or gripper_state.shape != (len(latent_tokens), self.state_dim):
            raise ValueError('expected latent features [B,D] or [B,K,D] and gripper_state [B,state_dim]')
        pooled = latent_tokens.float().mean(dim=1) if latent_tokens.ndim == 3 else latent_tokens.float()
        state = ((gripper_state.float() - self.state_mean) / self.state_std).clamp(-8, 8)
        return {'logits': self.net(torch.cat((pooled, state), dim=-1))}


class StateOnlyGripperHead(nn.Module):
    """Ten-step state-only control for the GH1 learning-path gate."""

    def __init__(self, *, hidden_dim: int = 16, state_dim: int = GRIPPER_STATE_DIM,
                 horizon: int = HORIZON):
        super().__init__()
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, horizon))

    def forward(self, gripper_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if gripper_state.ndim != 2 or gripper_state.shape[1] != self.state_dim:
            raise ValueError(f'gripper_state must have shape [B,{self.state_dim}]')
        state = ((gripper_state.float() - self.state_mean) / self.state_std).clamp(-8, 8)
        return {'logits': self.net(state)}


class FairHistoryGripperHead(nn.Module):
    """Compact head with the same causal 10-action history for every row."""

    def __init__(self, latent_dim: int = 0, *, hidden_dim: int = 128,
                 state_dim: int = GRIPPER_STATE_DIM, horizon: int = HORIZON):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        dims = state_dim + 20 + (latent_dim if latent_dim else 0)
        self.net = nn.Sequential(nn.Linear(dims, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, horizon))

    def forward(self, gripper_state: torch.Tensor, history_action: torch.Tensor,
                history_valid: torch.Tensor, latent: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if gripper_state.ndim != 2 or gripper_state.shape[1] != self.state_dim:
            raise ValueError('gripper_state must be [B,2]')
        if history_action.shape != (len(gripper_state), 10) or history_valid.shape != (len(gripper_state), 10):
            raise ValueError('history_action and history_valid must be [B,10]')
        state = ((gripper_state.float() - self.state_mean) / self.state_std).clamp(-8, 8)
        hist = torch.stack((history_action.float(), history_valid.float()), dim=-1).reshape(len(state), 20)
        pieces = [state, hist]
        if self.latent_dim:
            if latent is None or latent.shape != (len(state), self.latent_dim):
                raise ValueError(f'latent must be [B,{self.latent_dim}]')
            pieces.append(latent.float())
        elif latent is not None:
            raise ValueError('state/history-only head must not receive latent features')
        return {'logits': self.net(torch.cat(pieces, dim=-1))}


def _fair_view(part: Mapping[str, np.ndarray], *, target_key: str = 'target') -> dict[str, np.ndarray]:
    """Whitelist inference inputs; the labels and metadata remain scorer-only."""
    required = {'gripper_state', 'history_action', 'history_valid', target_key}
    if not required <= set(part):
        raise ValueError(f'fair head split misses {sorted(required - set(part))}')
    result = {'gripper_state': np.asarray(part['gripper_state'], np.float32),
              'history_action': np.asarray(part['history_action'], np.float32),
              'history_valid': np.asarray(part['history_valid'], bool),
              'target': np.asarray(part[target_key], np.float32)}
    if 'latent_tokens' in part:
        tokens = np.asarray(part['latent_tokens'], np.float32)
        result['latent'] = tokens.mean(axis=1) if tokens.ndim == 3 else tokens
    return result


def _fair_model_inputs(view: Mapping[str, np.ndarray], *, latent: bool) -> dict[str, np.ndarray]:
    allowed = {'gripper_state', 'history_action', 'history_valid'} | ({'latent'} if latent else set())
    if set(view) - allowed:
        raise ValueError(f'labels or metadata reached fair model input: {sorted(set(view) - allowed)}')
    required = {'gripper_state', 'history_action', 'history_valid'} | ({'latent'} if latent else set())
    if not required <= set(view):
        raise ValueError(f'missing fair model inputs: {sorted(required - set(view))}')
    return {key: np.asarray(view[key]) for key in required}


def _g1_history_logits(part: Mapping[str, np.ndarray], *, state_threshold: float = .3019486666) -> np.ndarray:
    """Encode G1 with exactly its available state/history input, for the gate baseline."""
    state = np.asarray(part['gripper_state'], np.float32)
    history = np.asarray(part['history_action'], np.float32)
    valid = np.asarray(part['history_valid'], bool)
    out = np.empty((len(state), HORIZON), np.float32)
    for i in range(len(state)):
        previous = np.flatnonzero(valid[i])
        sign = (1. if state[i, 0] <= state_threshold else -1.) if not len(previous) else float(np.sign(history[i, previous[-1]]))
        out[i].fill(12. * sign)
    return out


def fit_fair_grid(train: Mapping[str, np.ndarray], dev: Mapping[str, np.ndarray], *,
                  latent: bool, hidden_dims=(64, 128), learning_rates=(3e-4, 1e-3),
                  epochs_grid=(8, 16, 32), positive_weights=(.5, 1., 2.),
                  device='cuda:0', num_threads=8, seed=SEED, progress_callback=None):
    """Train full minibatch grid, selecting MSE only on held-out demo/task Dev."""
    torch.set_num_threads(int(num_threads)); torch.manual_seed(int(seed))
    tr, dv = _fair_view(train), _fair_view(dev)
    _fair_model_inputs({k:v for k,v in tr.items() if k!='target'}, latent=latent)
    _fair_model_inputs({k:v for k,v in dv.items() if k!='target'}, latent=latent)
    tx = {k: torch.as_tensor(v, dtype=(torch.bool if k == 'history_valid' else torch.float32), device=device)
          for k, v in tr.items() if k != 'target'}
    dx = {k: torch.as_tensor(v, dtype=(torch.bool if k == 'history_valid' else torch.float32), device=device)
          for k, v in dv.items() if k != 'target'}
    ty = torch.as_tensor(tr['target'] > 0, dtype=torch.float32, device=device)
    records=[]; fitted={}
    for width in hidden_dims:
        for lr in learning_rates:
            for pos_weight in positive_weights:
                for epochs in epochs_grid:
                    torch.manual_seed(int(seed))
                    model=FairHistoryGripperHead(tr.get('latent',np.empty((0,0))).shape[-1] if latent else 0,
                                                 hidden_dim=int(width)).to(device)
                    mean=torch.as_tensor(tr['gripper_state'].mean(0),device=device)
                    std=torch.as_tensor(tr['gripper_state'].std(0).clip(1e-4),device=device)
                    model.state_mean.copy_(mean); model.state_std.copy_(std)
                    opt=torch.optim.AdamW(model.parameters(),lr=float(lr),weight_decay=1e-4)
                    rng=torch.Generator(device=device).manual_seed(int(seed))
                    best=(float('inf'),None,None,None)
                    for epoch in range(int(epochs)):
                        model.train(); order=torch.randperm(len(ty),generator=rng,device=device)
                        for ids in order.split(128):
                            opt.zero_grad(set_to_none=True)
                            out=model(tx['gripper_state'][ids],tx['history_action'][ids],tx['history_valid'][ids],
                                      tx.get('latent',None)[ids] if latent else None)['logits']
                            loss=F.binary_cross_entropy_with_logits(out,ty[ids],
                                pos_weight=torch.tensor(float(pos_weight),device=device))
                            loss.backward(); opt.step()
                        model.eval()
                        with torch.no_grad():
                            logits=model(dx['gripper_state'],dx['history_action'],dx['history_valid'],
                                         dx.get('latent',None) if latent else None)['logits'].cpu().numpy()
                        mse=float(gripper_contribution_mse(np.where(logits>=0,1.,-1.),dv['target']).mean())
                        if mse<best[0]: best=(mse,epoch+1,float(F.binary_cross_entropy_with_logits(
                            torch.as_tensor(logits,device=device),torch.as_tensor(dv['target']>0,dtype=torch.float32,device=device),
                            pos_weight=torch.tensor(float(pos_weight),device=device)).item()),
                            {k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                        if progress_callback: progress_callback('fair_head_epoch',width=int(width),lr=float(lr),
                            pos_weight=float(pos_weight),epoch=epoch+1,epochs=int(epochs),dev_mse=mse)
                    record={'kind':'fair_mlp','hidden_dim':int(width),'learning_rate':float(lr),
                            'positive_weight':float(pos_weight),'epochs':int(epochs),'selected_epoch':int(best[1]),
                            'dev_gripper_mse':float(best[0]),'dev_bce':float(best[2])}
                    records.append(record); fitted[(int(width),float(lr),float(pos_weight),int(epochs))]=(model,best[3])
    selected=dev_select(records)
    model,state=fitted[(selected['hidden_dim'],selected['learning_rate'],selected['positive_weight'],selected['epochs'])]
    model.load_state_dict(state); model.eval()
    return model,selected,records


def predict_fair(model: FairHistoryGripperHead, part: Mapping[str, np.ndarray], *, latent: bool, device='cuda:0') -> np.ndarray:
    view=_fair_view(part); inputs=_fair_model_inputs({k:v for k,v in view.items() if k!='target'},latent=latent)
    model=model.to(device).eval()
    with torch.no_grad():
        state=torch.as_tensor(inputs['gripper_state'],dtype=torch.float32,device=device)
        hist=torch.as_tensor(inputs['history_action'],dtype=torch.float32,device=device)
        valid=torch.as_tensor(inputs['history_valid'],dtype=torch.bool,device=device)
        z=torch.as_tensor(inputs['latent'],dtype=torch.float32,device=device) if latent else None
        return model(state,hist,valid,z)['logits'].cpu().numpy()


def _fit_state_normalizer(model: nn.Module, state: np.ndarray) -> None:
    """Standardize the already Q01/Q99-normalized proprio using train rows only."""
    values = np.asarray(state, np.float32)
    if values.ndim != 2 or values.shape[1] != GRIPPER_STATE_DIM or not len(values):
        raise ValueError('training gripper state must be nonempty [N,2]')
    mean = torch.as_tensor(values.mean(axis=0), dtype=torch.float32)
    std = torch.as_tensor(values.std(axis=0), dtype=torch.float32).clamp_min(1e-4)
    model.state_mean.copy_(mean)
    model.state_std.copy_(std)


def switch_horizon_labels(target: np.ndarray) -> np.ndarray:
    """First future action index whose sign differs from the chunk's initial sign, else 10."""
    y = np.asarray(target)
    if y.ndim != 2 or y.shape[1] != HORIZON:
        raise ValueError(f'target must be [N,{HORIZON}]')
    signs = np.where(y >= 0, 1, -1)
    labels = np.full(len(y), HORIZON, np.int64)
    for i in range(len(y)):
        changes = np.flatnonzero(signs[i, 1:] != signs[i, :-1]) + 1
        if len(changes):
            labels[i] = int(changes[0])
    return labels


def gripper_contribution_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-row GRIP1 squared error divided by (10 * 7)."""
    p, y = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != HORIZON:
        raise ValueError(f'expected matching [N,{HORIZON}] prediction and target')
    return np.square(p - y).sum(axis=1) / (HORIZON * 7)


def retrieval_plus_gripper_loss(retrieval_loss: torch.Tensor, gripper_logits: torch.Tensor,
                                gripper_target: torch.Tensor, *, gripper_weight: float,
                                positive_weight: float = 1.0) -> torch.Tensor:
    """G-c objective term: L_ret + λ L_grip, with BCE on future gripper signs."""
    if gripper_logits.shape != gripper_target.shape or gripper_logits.ndim != 2 or gripper_logits.shape[1] != HORIZON:
        raise ValueError('gripper logits and targets must have matching [B,10] shapes')
    if float(gripper_weight) < 0:
        raise ValueError('gripper_weight must be non-negative')
    if float(positive_weight) <= 0: raise ValueError('positive_weight must be positive')
    grip_loss = F.binary_cross_entropy_with_logits(gripper_logits, (gripper_target > 0).float(),
        pos_weight=torch.tensor(float(positive_weight),device=gripper_logits.device))
    return retrieval_loss + float(gripper_weight) * grip_loss


def switch_metrics(pred: np.ndarray, target: np.ndarray, *, tolerance: int = 2) -> dict[str, Any]:
    """Event F1 with one-to-one ±tolerance matching and false-switch rate."""
    p, y = np.asarray(pred), np.asarray(target)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != HORIZON:
        raise ValueError(f'expected matching [N,{HORIZON}] traces')
    tp = fp = fn = false_rows = no_switch_rows = 0
    row_tp = row_fp = row_fn = 0
    pred_events = true_events = 0
    for prow, yrow in zip(p, y, strict=True):
        pe = (np.flatnonzero(np.sign(prow[1:]) != np.sign(prow[:-1])) + 1).tolist()
        ye = (np.flatnonzero(np.sign(yrow[1:]) != np.sign(yrow[:-1])) + 1).tolist()
        pred_events += len(pe); true_events += len(ye)
        has_pred, has_true = bool(pe), bool(ye)
        row_tp += int(has_pred and has_true)
        row_fp += int(has_pred and not has_true)
        row_fn += int(has_true and not has_pred)
        no_switch_rows += int(not has_true)
        used: set[int] = set()
        matched = 0
        for pos in pe:
            candidates = [(abs(pos - q), j) for j, q in enumerate(ye)
                          if j not in used and abs(pos - q) <= tolerance]
            if candidates:
                _, j = min(candidates)
                used.add(j); matched += 1
        tp += matched; fp += len(pe) - matched; fn += len(ye) - matched
        false_rows += int(bool(pe) and not ye)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    row_precision = row_tp / (row_tp + row_fp) if row_tp + row_fp else 0.
    row_recall = row_tp / (row_tp + row_fn) if row_tp + row_fn else 0.
    row_f1 = 2 * row_precision * row_recall / (row_precision + row_recall) if row_precision + row_recall else 0.
    return {'switch_f1': float(row_f1), 'switch_precision': float(row_precision), 'switch_recall': float(row_recall),
            'switch_event_f1': float(event_f1), 'switch_event_precision': float(precision),
            'switch_event_recall': float(recall),
            'switch_time_error_steps': _switch_time_error(p, y, tolerance=tolerance),
            'false_switch_rate': float(false_rows / no_switch_rows) if no_switch_rows else 0.,
            'false_switch_windows': float(false_rows), 'no_switch_target_windows': float(no_switch_rows),
            'predicted_switches': float(pred_events), 'target_switches': float(true_events)}


def _switch_time_error(pred: np.ndarray, target: np.ndarray, *, tolerance: int = 2) -> float | None:
    errors = []
    for prow, yrow in zip(np.asarray(pred), np.asarray(target), strict=True):
        pe = np.flatnonzero(np.sign(prow[1:]) != np.sign(prow[:-1])) + 1
        ye = np.flatnonzero(np.sign(yrow[1:]) != np.sign(yrow[:-1])) + 1
        used: set[int] = set()
        for pos in pe:
            choices = [(abs(int(pos - q)), j) for j, q in enumerate(ye)
                       if j not in used and abs(int(pos - q)) <= tolerance]
            if choices:
                err, j = min(choices); used.add(j); errors.append(err)
    return float(np.mean(errors)) if errors else None


def evaluate_logits(logits: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    logits, y = np.asarray(logits, np.float64), np.asarray(target, np.float64)
    continuous = np.tanh(logits)
    sign = np.where(logits >= 0, 1., -1.)
    s = switch_metrics(sign, np.where(y >= 0, 1., -1.))
    return {'gripper_contribution_mse_sign': float(gripper_contribution_mse(sign, y).mean()),
            'gripper_contribution_mse_continuous': float(gripper_contribution_mse(continuous, y).mean()),
            **{k: v for k, v in s.items() if k not in ('switch_time_error_steps',)},
            'switch_time_error_steps': _switch_time_error(sign, np.where(y >= 0, 1., -1.))}


def dev_select(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a configuration using Dev gripper MSE only; Eval is not accepted."""
    if not records:
        raise ValueError('no Dev records')
    if any(any(str(k).lower().startswith(('eval', 'test')) for k in r) for r in records):
        raise ValueError('selection records may contain Dev metrics only')
    for r in records:
        if 'dev_gripper_mse' not in r or not np.isfinite(float(r['dev_gripper_mse'])):
            raise ValueError('each record must contain a finite dev_gripper_mse')
    return min(records, key=lambda r: (float(r['dev_gripper_mse']), json.dumps(r, sort_keys=True)))


def assert_l90x_manifest(manifest: Mapping[str, Any], *, dev_tasks=DEV_TASKS, eval_tasks=EVAL_TASKS) -> None:
    """Verify EXP3-B L90X contains neither LIBERO-90 Dev nor Eval task rows."""
    if tuple(dev_tasks) == DEV_TASKS and tuple(eval_tasks) == EVAL_TASKS:
        # Keep the canonical LA2 manifest contract as the source of truth.
        from . import latent_action_units_la2
        latent_action_units_la2.assert_l90x_manifest(dict(manifest))
        return
    forbidden = set(map(int, dev_tasks)) | set(map(int, eval_tasks))
    actual = set(map(int, manifest.get('tasks', ())))
    leaked = actual & forbidden
    episode_tasks = {int(e['task_id']) for e in manifest.get('episodes', ())}
    leaked |= episode_tasks & forbidden
    if leaked:
        raise AssertionError(f'L90X includes fixed Dev/Eval task rows: {sorted(leaked)}')


def assert_training_rows(rows: Mapping[str, np.ndarray], ids: np.ndarray,
                         *, eval_pairs: set[tuple[int, int]] | None = None) -> None:
    """Validate memory-only training IDs and keep held-out eval demos disjoint."""
    ids = np.asarray(ids, np.int64)
    if not {'task', 'episode', 'memory'} <= set(rows):
        raise ValueError('rows require task, episode, and memory metadata')
    if np.any(ids < 0) or np.any(ids >= len(rows['task'])):
        raise IndexError('training row index out of range')
    if not np.asarray(rows['memory'], bool)[ids].all():
        raise AssertionError('training rows must all be LIBERO-40 memory rows')
    if eval_pairs:
        pairs = {(int(rows['task'][i]), int(rows['episode'][i])) for i in ids}
        overlap = pairs & eval_pairs
        if overlap:
            raise AssertionError(f'evaluation demos leaked into training rows: {sorted(overlap)[:5]}')


@dataclass(frozen=True)
class HeadConfig:
    name: str
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 20
    auxiliary_switch: bool = False


def fit_head(model: nn.Module, train: Mapping[str, np.ndarray], dev: Mapping[str, np.ndarray], *,
             config: HeadConfig, device: str = 'cpu', seed: int = SEED,
             num_threads: int | None = None, batch_size: int = 128,
             progress_callback=None) -> tuple[nn.Module, dict[str, Any]]:
    """Fit shuffled mini-batches with BCE and retain the best Dev sign-MSE checkpoint."""
    torch.set_num_threads(int(num_threads if num_threads is not None else os.environ.get('OMP_NUM_THREADS', '8')))
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    for split in (train, dev):
        validate_model_inputs({k: split[k] for k in ('latent_tokens', 'gripper_state')})
    x = torch.as_tensor(train['latent_tokens'], dtype=torch.float32, device=device)
    s = torch.as_tensor(train['gripper_state'], dtype=torch.float32, device=device)
    y = torch.as_tensor(train['target'] > 0, dtype=torch.float32, device=device)
    dx = torch.as_tensor(dev['latent_tokens'], dtype=torch.float32, device=device)
    ds = torch.as_tensor(dev['gripper_state'], dtype=torch.float32, device=device)
    dy = torch.as_tensor(dev['target'] > 0, dtype=torch.float32, device=device)
    if y.shape != (len(x), HORIZON) or dy.shape != (len(dx), HORIZON):
        raise ValueError('target must be [N,10]')
    _fit_state_normalizer(model, train['gripper_state'])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_dev_mse, best_state, best_dev_bce = float('inf'), None, float('inf')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for epoch in range(int(config.epochs)):
        model.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for batch_index, ids in enumerate(order.split(int(batch_size)), start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(x[ids], s[ids])
            loss = F.binary_cross_entropy_with_logits(output['logits'], y[ids])
            if config.auxiliary_switch and 'switch_horizon_logits' in output:
                labels = torch.as_tensor(switch_horizon_labels(train['target'][ids.cpu().numpy()]),
                                         dtype=torch.long, device=device)
                loss = loss + F.cross_entropy(output['switch_horizon_logits'], labels)
            loss.backward(); optimizer.step()
            if progress_callback is not None and (batch_index % 64 == 0 or batch_index == int(np.ceil(len(x) / batch_size))):
                progress_callback('head_batch', config=config.name, epoch=epoch + 1,
                                  batch=batch_index, batches=int(np.ceil(len(x) / batch_size)),
                                  device=str(device))
        model.eval()
        with torch.no_grad():
            dev_logits = model(dx, ds)['logits']
            dev_bce = float(F.binary_cross_entropy_with_logits(dev_logits, dy).item())
            dev_mse = float(gripper_contribution_mse(
                np.where(dev_logits.detach().cpu().numpy() >= 0, 1., -1.), dev['target']).mean())
        if dev_mse < best_dev_mse:
            best_dev_mse, best_dev_bce = dev_mse, dev_bce
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress_callback is not None:
            progress_callback('head_epoch', config=config.name, epoch=epoch + 1,
                              epochs=int(config.epochs), dev_gripper_mse=dev_mse, device=str(device))
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        dev_logits = model(dx, ds)['logits'].cpu().numpy()
    return model, {'dev_gripper_mse': best_dev_mse, 'dev_bce_at_selected_checkpoint': best_dev_bce,
                   'dev_metrics': evaluate_logits(dev_logits, dev['target']),
                   'epochs': int(config.epochs), 'learning_rate': float(config.learning_rate),
                   'batch_size': int(batch_size),
                   'seed': int(seed), 'config': config.name}


def fit_state_only_grid(train: Mapping[str, np.ndarray], dev: Mapping[str, np.ndarray], *,
                        hidden_dims=(8, 16, 32), learning_rates=(1e-3, 3e-3),
                        epochs_grid=(8, 16, 32), batch_size=128, device='cpu', seed=SEED,
                        num_threads: int | None = None, progress_callback=None):
    """Fit the no-latent gate on the same memory rows and select by Dev sign-MSE."""
    threads = int(num_threads if num_threads is not None else os.environ.get('OMP_NUM_THREADS', '8'))
    torch.set_num_threads(threads); torch.manual_seed(int(seed)); np.random.seed(int(seed))
    for part in (train, dev):
        if np.asarray(part['gripper_state']).shape != (len(part['target']), GRIPPER_STATE_DIM):
            raise ValueError('state-only input must be exactly the current gripper state [N,2]')
        if np.asarray(part['target']).shape != (len(part['gripper_state']), HORIZON):
            raise ValueError('state-only targets must have shape [N,10]')
    x = torch.as_tensor(train['gripper_state'], dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(train['target']) > 0, dtype=torch.float32, device=device)
    dx = torch.as_tensor(dev['gripper_state'], dtype=torch.float32, device=device)
    dy = torch.as_tensor(np.asarray(dev['target']) > 0, dtype=torch.float32, device=device)
    records=[]; fitted={}
    for width in hidden_dims:
        for lr in learning_rates:
            model=StateOnlyGripperHead(hidden_dim=int(width)); _fit_state_normalizer(model, train['gripper_state']); model=model.to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=float(lr),weight_decay=1e-4)
            generator=torch.Generator(device=device).manual_seed(int(seed))
            best=(float('inf'),float('inf'),None,None)
            config_name=f'state-only-h{int(width)}-lr{float(lr):g}'
            for epoch_limit in epochs_grid:
                # Each requested epoch limit gets an independent seeded run, just like the latent grid.
                if epoch_limit != epochs_grid[0]:
                    model=StateOnlyGripperHead(hidden_dim=int(width)); _fit_state_normalizer(model,train['gripper_state']); model=model.to(device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=float(lr),weight_decay=1e-4)
                    generator=torch.Generator(device=device).manual_seed(int(seed))
                for epoch in range(int(epoch_limit)):
                    model.train(); order=torch.randperm(len(x),generator=generator,device=device)
                    for batch_index,ids in enumerate(order.split(int(batch_size)),start=1):
                        optimizer.zero_grad(set_to_none=True)
                        loss=F.binary_cross_entropy_with_logits(model(x[ids])['logits'],y[ids])
                        loss.backward(); optimizer.step()
                        if progress_callback is not None and (batch_index % 64 == 0 or batch_index == int(np.ceil(len(x)/batch_size))):
                            progress_callback('state_gate_batch', config=config_name, epoch=epoch+1,
                                              epoch_limit=int(epoch_limit), batch=batch_index,
                                              batches=int(np.ceil(len(x)/batch_size)), device=str(device))
                    model.eval()
                    with torch.no_grad():
                        logits=model(dx)['logits'].cpu().numpy()
                        dev_mse=float(gripper_contribution_mse(np.where(logits>=0,1.,-1.),dev['target']).mean())
                        dev_bce=float(F.binary_cross_entropy_with_logits(torch.as_tensor(logits,device=device),dy).item())
                    if dev_mse < best[0]:
                        best=(dev_mse,dev_bce,epoch+1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},int(epoch_limit))
                    if progress_callback is not None:
                        progress_callback('state_gate_epoch',config=config_name,epoch=epoch+1,
                                          epoch_limit=int(epoch_limit),dev_gripper_mse=dev_mse,dev_bce=dev_bce,device=str(device))
            record={'kind':'state_only','hidden_dim':int(width),'learning_rate':float(lr),
                    'epochs':int(best[4]),'selected_epoch':int(best[2]),'dev_gripper_mse':float(best[0]),
                    'dev_bce_at_selected_checkpoint':float(best[1]),'batch_size':int(batch_size)}
            records.append(record); fitted[(int(width),float(lr))]=(model,best[3])
    selected=dev_select(records)
    model,state=fitted[(selected['hidden_dim'],selected['learning_rate'])]
    model.load_state_dict(state);model.eval()
    return model,selected,records


def predict_state_only(model: StateOnlyGripperHead, split: Mapping[str, np.ndarray], *, device='cpu') -> np.ndarray:
    """Run the state-only gate without constructing or passing any latent input."""
    if set(split) - {'gripper_state','target','task','episode','hard','row','key'}:
        raise ValueError('state-only prediction accepts state and scoring metadata only')
    x=torch.as_tensor(split['gripper_state'],dtype=torch.float32,device=device)
    model=model.to(device).eval()
    with torch.no_grad(): return model(x)['logits'].cpu().numpy()


def gripper_hold_reference(part: Mapping[str, np.ndarray], *, state_threshold: float = .3019486666) -> dict[str, np.ndarray]:
    """Recreate the GRIP1 G1 and previous-hold rules from the exact locked rows."""
    key=np.asarray(part['key'],np.int64); state=np.asarray(part['state'],np.float32)
    target=np.asarray(part['actions'],np.float32)[:,:,6]
    row_by_key={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(key)}
    g1=np.empty_like(target); previous=np.zeros_like(target)
    for i,(task,episode,frame) in enumerate(key):
        if int(frame)==0:
            g1[i]=(-1. if state[i,6] > state_threshold else 1.)
        else:
            prev_i=row_by_key.get((int(task),int(episode),int(frame)-10))
            if prev_i is None:
                g1[i]=0.
            else:
                g1[i]=target[prev_i,-1]
                previous[i]=target[prev_i,-1]
    return {'G1_state_initialized_hold':g1,'G2_previous_hold':previous}


def encode_frozen_key2_e(checkpoint: Path, parts: Mapping[str, Mapping[str, np.ndarray]], *,
                         mode: str = 'e_vlm_h', device: str = 'cpu', batch_size: int = 32,
                         progress_callback=None) -> dict[str, dict[str, np.ndarray]]:
    """Run a frozen KEY2-F E encoder over train/Dev/learned/Eval query rows.

    The imported KEY2 builder owns row identity and causal history assembly.
    Its action targets are copied to the returned evaluation records only after
    E has been called, and never enter the encoder invocation.
    """
    if mode not in ('e_vlm_h', 'e_vj_h'):
        raise ValueError('frozen KEY2 E mode must be e_vlm_h or e_vj_h')
    from . import key_shared_latent as key2
    model = key2._new_model(mode).to(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    if isinstance(state, Mapping) and set(state) == {'model', 'config'}:
        checkpoint_config = state['config']
        if isinstance(checkpoint_config, Mapping) and checkpoint_config.get('mode') not in (None, mode):
            raise ValueError(f'{checkpoint} was trained for {checkpoint_config.get("mode")}, requested {mode}')
        state = state['model']
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    result: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for split_name, part in parts.items():
            required = ('tokens', 'groups', 'history', 'history_mask', 'state')
            if mode == 'e_vj_h':
                required += ('vjepa_tokens', 'vjepa_groups')
            if any(k not in part for k in required):
                raise ValueError(f'{split_name} misses frozen-E inputs: {sorted(set(required) - set(part))}')
            pieces = []
            for lo in range(0, len(part['key']), int(batch_size)):
                ids = np.arange(lo, min(lo + int(batch_size), len(part['key'])))
                batch = key2._pack_part(part, ids)
                z = key2._model_batch_encode(model, mode, batch, device)
                pieces.append(z.detach().cpu().numpy().astype(np.float32))
                done = min(lo + int(batch_size), len(part['key']))
                if progress_callback is not None and (done % 512 == 0 or done == len(part['key'])):
                    progress_callback('frozen_e_encode', mode=mode, split=split_name,
                                      completed=done, total=len(part['key']), device=str(device))
            z_all = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 5, 256), np.float32)
            state2 = np.asarray(part['state'], np.float32)[:, 6:8]
            history = np.asarray(part['history'], np.float32)[:, :, 6]
            history_valid = np.asarray(part['history_mask'], bool)
            record = {'latent_tokens': z_all, 'gripper_state': state2,
                      'history_action': history, 'history_valid': history_valid}
            _fair_model_inputs({'latent': z_all.mean(axis=1), 'gripper_state': state2,
                                'history_action': history, 'history_valid': history_valid}, latent=True)
            # Labels and IDs stay beside, never inside, the two-field model input.
            if 'actions' in part:
                record['target'] = np.asarray(part['actions'], np.float32)[:, :, 6]
            for key in ('task', 'episode', 'hard', 'row', 'key'):
                if key in part:
                    record[key] = np.asarray(part[key])
            if 'pi05_gripper' in part:
                record['pi05_gripper'] = np.asarray(part['pi05_gripper'], np.float32)
            for arm_key in ('arm_p5_top1', 'arm_e_vlm_top1'):
                if arm_key in part:
                    record[arm_key] = np.asarray(part[arm_key], np.float32)
            result[split_name] = record
    return result


def pooled_pi05_features(parts: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Build the G-d control from valid π0.5 VLM prefix mean plus grip state."""
    result = {}
    for split_name, part in parts.items():
        if not {'tokens', 'groups', 'state'} <= set(part):
            raise ValueError(f'{split_name} needs VLM tokens, groups, and state')
        tokens, groups = np.asarray(part['tokens']), np.asarray(part['groups'])
        if groups.shape != tokens.shape[:2]:
            raise ValueError('π0.5 token groups must align with prefix tokens')
        valid = groups >= 0
        pooled = (tokens.astype(np.float32) * valid[..., None]).sum(axis=1) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        state2 = np.asarray(part['state'], np.float32)[:, 6:8]
        record = {'latent_tokens': pooled.astype(np.float32), 'gripper_state': state2,
                  'history_action': np.asarray(part['history'],np.float32)[:,:,6],
                  'history_valid': np.asarray(part['history_mask'],bool)}
        if 'actions' in part:
            record['target'] = np.asarray(part['actions'], np.float32)[:, :, 6]
        for key in ('task', 'episode', 'hard', 'row', 'key'):
            if key in part:
                record[key] = np.asarray(part[key])
        validate_model_inputs({k: record[k] for k in ('latent_tokens', 'gripper_state')})
        result[split_name] = record
    return result


def encode_l90x_frozen_e(checkpoint: Path, row_cache: Path, pooled_features: Path, *,
                         row_limit: int | None = None, row_indices: np.ndarray | None = None,
                         device='cpu', batch_size=32,
                         progress_callback=None) -> dict[str, np.ndarray]:
    """Encode L90X rows through frozen E from the exact pooled-prefix cache.

    L90X has no complete X1 token cache in this workspace. The supported
    source is the validated LA2 prefix mean, represented as one VLM token.
    """
    from . import latent_action_units_la2 as la2
    from . import key_shared_latent as key2
    row_cache = Path(row_cache); manifest = json.loads((row_cache / 'manifest.json').read_text())
    la2.assert_l90x_manifest(manifest)
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in row_cache.glob('*.npy')}
    pooled = np.load(pooled_features, mmap_mode='r', allow_pickle=False)
    n = min(len(pooled), int(manifest['rows'])) if row_limit is None else min(int(row_limit), len(pooled), int(manifest['rows']))
    ids = np.arange(n, dtype=np.int64) if row_indices is None else np.asarray(row_indices, np.int64)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= n or pooled.shape[1:] != (2048,):
        raise ValueError(f'invalid L90X pooled-prefix features: {pooled.shape}')
    # Re-assert task IDs on the exact rows being consumed, in addition to the
    # complete source manifest assertion above.
    tasks = np.asarray(rows['task'][ids], np.int32)
    forbidden = set(DEV_TASKS) | set(EVAL_TASKS)
    if set(map(int, np.unique(tasks))) & forbidden:
        raise AssertionError('filtered L90X smoke rows contain Dev/Eval task IDs')
    hist_rows=np.asarray(rows['hist'][ids],np.float32)
    if hist_rows.ndim == 2:
        hist_rows=np.pad(hist_rows,((0,0),(0,max(0,70-hist_rows.shape[1]))))[:,:70].reshape(len(ids),10,7)
    elif hist_rows.ndim != 3:
        raise ValueError(f'unexpected L90X action history shape {hist_rows.shape}')
    hist_mask=np.asarray(rows['has_hist'][ids],bool)
    if hist_mask.ndim == 1: hist_mask=np.repeat(hist_mask[:,None],10,axis=1)
    part = {'key': np.stack([rows['task'][ids], rows['episode'][ids], rows['frame'][ids]], axis=1).astype(np.int32),
            'tokens': np.asarray(pooled[ids], np.float32)[:, None, :],
            'groups': np.zeros((len(ids), 1), np.int8), 'history': hist_rows,
            'history_mask': hist_mask,
            'state': np.asarray(rows['state'][ids, :8], np.float32),
            'actions': np.asarray(rows['actions'][ids, :10, :7], np.float32),
            'task': tasks, 'episode': np.asarray(rows['episode'][ids], np.int32)}
    latent = encode_frozen_key2_e(checkpoint, {'l90x': part}, mode='e_vlm_h', device=device,
                                  batch_size=batch_size, progress_callback=progress_callback)['l90x']
    latent['manifest'] = manifest
    return latent


def append_l90x_training_data(train: Mapping[str, np.ndarray], l90x: Mapping[str, np.ndarray],
                              manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Add eligible L90X rows after validating the fixed Dev/Eval exclusion."""
    assert_l90x_manifest(manifest)
    for split in (train, l90x):
        validate_model_inputs({k: split[k] for k in ('latent_tokens', 'gripper_state')})
        if 'target' not in split:
            raise ValueError('training rows require ten-step target labels')
    if train['latent_tokens'].shape[1:] != l90x['latent_tokens'].shape[1:]:
        raise ValueError('L90X frozen-E token shape differs from LIBERO-40')
    return {key: np.concatenate((np.asarray(train[key]), np.asarray(l90x[key])), axis=0)
            for key in ('latent_tokens', 'gripper_state', 'history_action', 'history_valid', 'target')}


def fit_joint_e_grid(pretrained_checkpoint: Path, fit_bank: Mapping[str, np.ndarray],
                     fit_bank_input: Mapping[str, np.ndarray], train: Mapping[str, np.ndarray],
                     dev: Mapping[str, np.ndarray], *, lambdas=(0.1, 0.3, 1.0),
                     learning_rates=(1e-4, 3e-4), max_epochs=(4, 8, 16),
                     seed=SEED, device='cpu', batch_size=8,
                     num_threads: int | None = None, positive_weights=(.5,1.,2.),
                     progress_callback=None) -> tuple[nn.Module, nn.Module, dict[str, Any], list[dict[str, Any]]]:
    """Jointly tune KEY2-F E and a grip head under L_ret + λ L_grip.

    Retrieval candidates, self-episode masks, teacher errors, and E scoring
    reuse KEY2-F functions. Each λ/lr run is checkpointed by Dev sign-MSE;
    selection never receives learned or unseen Eval rows.
    """
    from . import key_shared_latent as key2
    torch.set_num_threads(int(num_threads if num_threads is not None else os.environ.get('OMP_NUM_THREADS', '8')))
    initial = key2._new_model('e_vlm_h').to(device)
    initial.load_state_dict(torch.load(pretrained_checkpoint, map_location='cpu', weights_only=True))
    initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    candidate_actions = key2._candidate_window_actions(np.arange(len(fit_bank['length'])), fit_bank)
    baseline_scores, baseline_errors = key2._score_model(initial, 'e_vlm_h', fit_bank, fit_bank_input, dev, device)
    baseline_arm_mse = top1_candidate_arm6_mse(baseline_scores, baseline_errors)
    del initial
    records: list[dict[str, Any]] = []
    models: dict[tuple[float, float, float, int], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
    train_n = len(train['key']); bank_n = len(fit_bank['length'])
    if train_n == 0 or bank_n == 0:
        raise ValueError('joint training requires nonempty train queries and memory candidates')
    for lam in lambdas:
        for lr in learning_rates:
            for pos_weight in positive_weights:
              for epochs in max_epochs:
                torch.manual_seed(int(seed)); np.random.seed(int(seed))
                encoder = key2._new_model('e_vlm_h').to(device)
                encoder.load_state_dict(initial_state)
                head = LatentGripperHead(256, hidden_dim=64, heads=4).to(device)
                _fit_state_normalizer(head, np.asarray(train['state'])[:, 6:8])
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                              lr=float(lr), weight_decay=1e-4)
                rng = np.random.default_rng(int(seed) + 37)
                best_mse, best_epoch, best_state = float('inf'), None, None
                for epoch in range(int(epochs)):
                    qids = np.arange(train_n); rng.shuffle(qids)
                    encoder.train(); head.train()
                    for batch_i, start in enumerate(range(0, train_n, int(batch_size)), start=1):
                        qi = qids[start:start + int(batch_size)]
                        if not len(qi):
                            continue
                        random_ids = rng.choice(bank_n, size=min(256, bank_n), replace=False)
                        cids, allowed = key2.contrastive_candidate_set(
                            qi, random_ids, train['task'][qi], train['episode'][qi],
                            fit_bank['task'], fit_bank['episode'])
                        qpart = key2._pack_query_observation(train, qi)
                        qpart['history'], qpart['history_mask'] = key2._history_corrupt(
                            qpart['history'], qpart['history_mask'], rng)
                        kpart = key2._pack_query_observation(fit_bank_input, cids)
                        qrep = key2._model_batch_encode(encoder, 'e_vlm_h', qpart, device)
                        krep = key2._model_batch_encode(encoder, 'e_vlm_h', kpart, device)
                        scores = (qrep[:, None] * krep[None]).sum(-1).mean(-1)
                        scores = key2._scale_logits(encoder, scores)
                        errors = key2._arm6_candidate_mse(
                            np.asarray(train['actions'])[qi], candidate_actions[cids])
                        retrieval = key2.contrastive_teacher_loss(
                            scores, torch.as_tensor(errors, device=device),
                            torch.as_tensor(allowed, device=device), objective='soft', teacher_temperature=.01)
                        grip_state = torch.as_tensor(np.asarray(train['state'])[qi, 6:8], dtype=torch.float32, device=device)
                        grip_target = torch.as_tensor(np.asarray(train['actions'])[qi, :, 6], dtype=torch.float32, device=device)
                        hist_action = torch.as_tensor(np.asarray(train['history'])[qi, :, 6], dtype=torch.float32, device=device)
                        hist_valid = torch.as_tensor(np.asarray(train['history_mask'])[qi], dtype=torch.bool, device=device)
                        grip_logits = head(qrep, grip_state, hist_action, hist_valid)['logits']
                        loss = retrieval_plus_gripper_loss(retrieval, grip_logits, grip_target,
                            gripper_weight=float(lam),positive_weight=float(pos_weight))
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.)
                        optimizer.step(); key2._clamp_logit_scale_(encoder)
                        if progress_callback is not None and (batch_i % 64 == 0 or start + int(batch_size) >= train_n):
                            progress_callback('joint_grid_batch', lambda_=float(lam), learning_rate=float(lr),
                                              max_epochs=int(epochs), epoch=epoch + 1, batch=batch_i,
                                              batches=(train_n + int(batch_size) - 1) // int(batch_size),
                                              device=str(device))
                    dev_rep, _ = key2._query_representations(encoder, 'e_vlm_h', dev, device)
                    dev_state = torch.as_tensor(np.asarray(dev['state'])[:, 6:8], dtype=torch.float32, device=device)
                    head.eval(); encoder.eval()
                    with torch.no_grad():
                        dev_hist = torch.as_tensor(np.asarray(dev['history'])[:, :, 6], dtype=torch.float32, device=device)
                        dev_valid = torch.as_tensor(np.asarray(dev['history_mask']), dtype=torch.bool, device=device)
                        logits = head(torch.as_tensor(dev_rep, dtype=torch.float32, device=device), dev_state,
                                      dev_hist, dev_valid)['logits'].cpu().numpy()
                    dev_mse = float(gripper_contribution_mse(np.where(logits >= 0, 1., -1.), dev['actions'][:, :, 6]).mean())
                    if progress_callback is not None:
                        progress_callback('joint_grid_epoch', lambda_=float(lam), learning_rate=float(lr),
                                           max_epochs=int(epochs), epoch=epoch + 1,
                                           dev_gripper_mse=dev_mse, device=str(device))
                    if dev_mse < best_mse:
                        best_mse, best_epoch = dev_mse, epoch + 1
                        best_state = ({k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                                      {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                if best_state is None:
                    raise RuntimeError('joint E/head training produced no Dev checkpoint')
                record = {'lambda': float(lam), 'learning_rate': float(lr), 'positive_weight':float(pos_weight), 'max_epochs': int(epochs),
                          'selected_epoch': int(best_epoch), 'dev_gripper_mse': float(best_mse),
                          'pretrained_dev_arm6_top1_mse': baseline_arm_mse}
                records.append(record)
                models[(float(lam), float(lr), float(pos_weight), int(epochs))] = best_state
                if progress_callback is not None:
                    progress_callback('joint_grid_config_complete', lambda_=float(lam), learning_rate=float(lr),
                                      max_epochs=int(epochs), selected_epoch=int(best_epoch),
                                      dev_gripper_mse=float(best_mse), device=str(device))
                del encoder, head, optimizer
    selected = dev_select(records)
    selected_encoder_state, selected_head_state = models[(selected['lambda'], selected['learning_rate'], selected['positive_weight'], selected['max_epochs'])]
    selected_encoder = key2._new_model('e_vlm_h').to(device); selected_encoder.load_state_dict(selected_encoder_state)
    selected_head = LatentGripperHead(256, hidden_dim=64, heads=4).to(device)
    selected_head.load_state_dict(selected_head_state)
    selected_scores, selected_errors = key2._score_model(selected_encoder, 'e_vlm_h', fit_bank, fit_bank_input, dev, device)
    selected_arm_mse = top1_candidate_arm6_mse(selected_scores, selected_errors)
    selected.update({'pretrained_dev_arm6_top1_mse': baseline_arm_mse,
                     'joint_dev_arm6_top1_mse': selected_arm_mse,
                     'joint_minus_pretrained_dev_arm6_top1_mse': float(selected_arm_mse - baseline_arm_mse)})
    return selected_encoder, selected_head, selected, records


def make_head(kind: str, latent_dim: int, *, hidden_dim: int, auxiliary_switch: bool = False) -> nn.Module:
    if kind == 'mlp':
        return LatentGripperMLP(latent_dim, hidden_dim=hidden_dim)
    if kind in ('cross_attention', 'cross_attention_aux'):
        return LatentGripperHead(latent_dim, hidden_dim=hidden_dim,
                                 auxiliary_switch=(auxiliary_switch or kind.endswith('_aux')))
    raise ValueError(f'unknown head kind: {kind}')


def fit_dev_grid(train: Mapping[str, np.ndarray], dev: Mapping[str, np.ndarray], *,
                 kinds=('mlp', 'cross_attention'), hidden_dims=(64, 128),
                 learning_rates=(3e-4, 1e-3), epochs_grid=(8, 16, 32),
                 auxiliary_switch=False, device='cpu', seed=SEED,
                 num_threads: int | None = None, progress_callback=None) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    """Fit a small grid and select by Dev sign-output MSE only."""
    records = []
    fitted_models = {}
    for kind in kinds:
        for width in hidden_dims:
            for lr in learning_rates:
                for epochs in epochs_grid:
                    torch.manual_seed(int(seed)); np.random.seed(int(seed))
                    model = make_head(kind, train['latent_tokens'].shape[-1], hidden_dim=int(width),
                                      auxiliary_switch=auxiliary_switch)
                    use_auxiliary = auxiliary_switch or kind.endswith('_aux')
                    cfg = HeadConfig(f'{kind}-h{width}-lr{lr:g}-ep{epochs}', int(width), float(lr), int(epochs),
                                     use_auxiliary)
                    fitted, info = fit_head(model, train, dev, config=cfg, device=device, seed=seed,
                                            num_threads=num_threads, progress_callback=progress_callback)
                    record = {'kind': kind, 'hidden_dim': int(width), 'learning_rate': float(lr),
                              'epochs': int(epochs), 'dev_gripper_mse': info['dev_gripper_mse'],
                              'dev_bce_at_selected_checkpoint': info['dev_bce_at_selected_checkpoint']}
                    records.append(record)
                    fitted_models[(kind, int(width), float(lr), int(epochs))] = fitted
    if not records:
        raise ValueError('empty head grid')
    selected = dev_select(records)
    selected_model = fitted_models[(selected['kind'], selected['hidden_dim'],
                                    selected['learning_rate'], selected['epochs'])]
    return selected_model, selected, records


def predict(model: nn.Module, split: Mapping[str, np.ndarray], *, device='cpu', batch_size=256) -> np.ndarray:
    """Predict without passing labels, row IDs, tasks, or π0.5 actions."""
    validate_model_inputs({k: split[k] for k in ('latent_tokens', 'gripper_state')})
    model = model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for lo in range(0, len(split['latent_tokens']), int(batch_size)):
            hi = min(lo + int(batch_size), len(split['latent_tokens']))
            outputs.append(model(torch.as_tensor(split['latent_tokens'][lo:hi], dtype=torch.float32, device=device),
                                 torch.as_tensor(split['gripper_state'][lo:hi], dtype=torch.float32, device=device))['logits'].cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, HORIZON), np.float32)


def evaluate_split(logits: np.ndarray, split: Mapping[str, np.ndarray], *, pi05_actions: np.ndarray | None = None,
                   bootstrap_reps: int = 5000, bootstrap_seed: int = 4101) -> dict[str, Any]:
    """Score all/hard rows and optional π0.5 paired episode bootstrap."""
    y = np.asarray(split['target'], np.float32)
    sign = np.where(np.asarray(logits) >= 0, 1., -1.).astype(np.float32)
    hard = np.asarray(split.get('hard', np.zeros(len(y), bool)), bool)
    metrics = {'all': evaluate_logits(logits, y), 'n_all': int(len(y)),
               'hard': evaluate_logits(np.asarray(logits)[hard], y[hard]) if hard.any() else None,
               'n_hard': int(hard.sum())}
    if pi05_actions is not None:
        baseline = np.asarray(pi05_actions, np.float32)
        if baseline.shape != y.shape:
            raise ValueError('π0.5 gripper targets do not align with evaluation rows')
        task, episode = np.asarray(split['task']), np.asarray(split['episode'])
        grip_split = __import__(f'{__package__}.gripper_split', fromlist=['episode_codes'])
        codes = grip_split.episode_codes(task, episode)
        per_row_diff = gripper_contribution_mse(sign, y) - gripper_contribution_mse(baseline, y)
        ci = grip_split.paired_episode_bootstrap(per_row_diff, codes, reps=bootstrap_reps, seed=bootstrap_seed)
        metrics['pi05'] = {'gripper_contribution_mse': float(gripper_contribution_mse(baseline, y).mean()),
                           'paired_head_minus_pi05': ci}
    return metrics


def active7_with_gripper(arm_actions: np.ndarray, target_actions: np.ndarray,
                         gripper_values: np.ndarray) -> dict[str, Any]:
    """Combine a frozen retrieval arm with a predicted or reference grip trace."""
    from . import gripper_split
    arm, target, grip = np.asarray(arm_actions), np.asarray(target_actions), np.asarray(gripper_values)
    if arm.ndim != 3 or arm.shape[-1] != 6 or target.shape != (len(arm), HORIZON, 7) or grip.shape != (len(arm), HORIZON):
        raise ValueError('expected arm [N,10,6], target [N,10,7], and grip [N,10]')
    pred = np.concatenate((arm.astype(np.float32), grip[..., None].astype(np.float32)), axis=-1)
    arm_mse, grip_mse, active = gripper_split.action_parts_mse(pred, target)
    return {'arm6_mse': float(np.mean(arm_mse)), 'gripper_contribution_mse': float(np.mean(grip_mse)),
            'active7_mse': float(np.mean(active)), 'per_row_active7_mse': np.asarray(active).tolist()}


def top1_candidate_arm6_mse(scores: np.ndarray, errors: np.ndarray) -> float:
    """Score retrieval arm error for each query using the highest-score candidate."""
    s, e = np.asarray(scores), np.asarray(errors)
    if s.ndim != 2 or e.shape != s.shape or s.shape[0] == 0 or s.shape[1] == 0:
        raise ValueError('scores and candidate arm errors must be matching nonempty [Q,N] matrices')
    if not np.isfinite(s).any(axis=1).all():
        raise ValueError('every query needs at least one finite candidate score')
    chosen = np.argmax(s, axis=1)
    selected = e[np.arange(len(chosen)), chosen]
    if not np.isfinite(selected).all():
        raise ValueError('selected candidate arm errors must be finite')
    return float(selected.mean())


def run_frozen_e_experiment(output: Path, parts: Mapping[str, Mapping[str, np.ndarray]], *,
                            e_checkpoint: Path, mode: str = 'e_vlm_h', label: str = 'G-a',
                            device: str = 'cpu', hidden_dims=(64, 128),
                            learning_rates=(3e-4, 1e-3), epochs_grid=(8, 16, 32),
                            bootstrap_reps=5000, num_threads: int | None = None) -> dict[str, Any]:
    """Train and score one frozen-E row using caller-supplied locked splits.

    `parts` must contain memory-only `train`, LIBERO-90 task 6/7 `dev`, and
    optional `learned`/`unseen` splits. Eval labels are copied only for scoring.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite GH1 variant output: {output}')
    if not {'train', 'dev'} <= set(parts):
        raise ValueError('parts must include train and Dev')
    output.mkdir(parents=True, exist_ok=True)
    encoded = encode_frozen_key2_e(e_checkpoint, parts, mode=mode, device=device)
    train, dev = encoded['train'], encoded['dev']
    if 'memory' in parts['train']:
        # The KEY2 builder marks its LIBERO-40 fit anchors as memory-only.
        pseudo_rows = {'task': np.asarray(train['task']), 'episode': np.asarray(train['episode']),
                       'memory': np.asarray(parts['train']['memory'], bool)}
        assert_training_rows(pseudo_rows, np.arange(len(train['target'])))
    model, selected, grid = fit_dev_grid(train, dev, device=device, num_threads=num_threads,
                                         hidden_dims=hidden_dims, learning_rates=learning_rates,
                                         epochs_grid=epochs_grid)
    selected_path = output / f'{label.lower()}_head.pt'
    torch.save(model.state_dict(), selected_path)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for split_name in ('learned', 'unseen'):
        if split_name not in encoded:
            continue
        split = encoded[split_name]
        logits = predict(model, split, device=device)
        predictions[split_name] = logits
        metrics[split_name] = evaluate_split(logits, split,
                                             pi05_actions=split.get('pi05_gripper'),
                                             bootstrap_reps=bootstrap_reps)
        for arm_key in ('arm_p5_top1', 'arm_e_vlm_top1'):
            if arm_key in split:
                metrics[split_name].setdefault('active7', {})[arm_key] = active7_with_gripper(
                    split[arm_key], parts[split_name]['actions'], np.where(logits >= 0, 1., -1.))
    payload = {'label': label, 'mode': mode, 'seed': SEED, 'e_checkpoint': str(e_checkpoint),
               'head_checkpoint': str(selected_path), 'selected_by': 'Dev sign gripper-contribution MSE',
               'selected': selected, 'dev_grid': grid, 'splits': metrics,
               'inputs': ['frozen E latent tokens', 'current gripper proprio[6:8]'],
               'pi05_is_model_input': False, 'eval_labels_used_for_fit_or_selection': False}
    (output / 'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (output / 'predictions.npz').open('wb') as f:
        np.savez_compressed(f, **{f'{name}_logits': value for name, value in predictions.items()})
    lines = ['# GH1 frozen-E experiment', '', '| Split | Head sign MSE | π0.5 sign MSE | Hard head sign MSE |', '|---|---:|---:|---:|']
    for name, item in metrics.items():
        pi = item.get('pi05', {}).get('gripper_contribution_mse')
        hard = item['hard']['gripper_contribution_mse_sign'] if item['hard'] else None
        lines.append(f"| {name} | {item['all']['gripper_contribution_mse_sign']:.6f} | {pi if pi is not None else '—'} | {hard if hard is not None else '—'} |")
    lines.extend(['', f"選択: {selected['kind']}、hidden={selected['hidden_dim']}、lr={selected['learning_rate']}、epoch={selected['epochs']}。",
                  '設定選択はDevのみ。Evalは最終採点にのみ使用。'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return payload


def assert_saved_prediction_alignment(saved: Mapping[str, np.ndarray], target: np.ndarray,
                                     key: np.ndarray, *, source: str) -> None:
    """Check KEY2 paired prediction identity; its archive names targets `target_actions`."""
    target_field = 'target_actions' if 'target_actions' in saved else 'target' if 'target' in saved else None
    if target_field is None:
        raise KeyError(f'{source} prediction archive misses target_actions')
    if 'key' not in saved:
        raise KeyError(f'{source} prediction archive misses key')
    if not np.array_equal(saved[target_field], target) or not np.array_equal(saved['key'], key):
        raise AssertionError(f'{source} saved predictions do not align with locked rows')


def load_key2_frozen_e_parts(*, checkpoint: Path | None = None,
                             vjepa_eval_cache: Path | None = None, progress_callback=None) -> tuple[
                                 dict[str, dict[str, np.ndarray]], Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Join exact KEY2-F locked rows with frozen-E inputs and saved top-1 arms."""
    from . import key_shared_latent as key2
    selected_path = KEY2_ROOT / 'selected_configs.json'
    selected = json.loads(selected_path.read_text())
    checkpoint = Path(checkpoint or selected['selected']['e_vlm_h']['checkpoint'])
    vjepa_root = KEY2_ROOT.parent / 'vjepa'
    vjepa_eval_cache = Path(vjepa_eval_cache or KEY2_ROOT / 'vjepa/libero90_eval')
    if progress_callback is not None:
        progress_callback('key2_query_builder_start', device='cpu')
    bank, bank_input, fit_bank, fit_bank_input, train, learned, unseen, dev = key2._build_key2_queries(
        run_root=KEY2_RUN, missing_root=KEY2_MISSING, seen_stride5=KEY2_SEEN_CACHE,
        unseen_stride5=KEY2_UNSEEN_CACHE, vjepa_eval_cache=vjepa_eval_cache,
        progress_callback=progress_callback)
    if progress_callback is not None:
        progress_callback('key2_query_builder_returned', bank_rows=int(len(bank['length'])),
                          train_rows=int(len(train['key'])))
    # KEY2's train dictionary consists exclusively of eligible LIBERO-40 memory
    # unit starts. Assert the held-out learned rows are absent before encoding.
    train_pairs = set(zip(np.asarray(train['task']).astype(int), np.asarray(train['episode']).astype(int)))
    learned_pairs = set(zip(np.asarray(learned['task']).astype(int), np.asarray(learned['episode']).astype(int)))
    if train_pairs & learned_pairs:
        raise AssertionError('KEY2 learned evaluation demos overlap the memory fit rows')
    if set(np.unique(np.asarray(dev['task'], int))) != set(DEV_TASKS) or len(dev['key']) != 217:
        raise AssertionError(f'KEY2 Dev split changed: tasks={np.unique(dev["task"]).tolist()}, rows={len(dev["key"])}')
    if len(learned['key']) != 621 or int(np.asarray(learned['hard']).sum()) != 125:
        raise AssertionError('KEY2 learned evaluation rows changed')
    if len(unseen['key']) != 619 or int(np.asarray(unseen['hard']).sum()) != 134:
        raise AssertionError('KEY2 unseen evaluation rows changed')
    # Attach GRIP1 memory-side eligibility explicitly; targets remain training
    # labels only and never enter the model's two-key input mapping.
    train['memory'] = np.ones(len(train['key']), bool)
    parts = {'train': train, 'dev': dev, 'learned': learned, 'unseen': unseen}
    for split_name in ('learned', 'unseen'):
        part = parts[split_name]
        part['pi05_gripper'] = np.asarray(part['pi05'], np.float32)[:, :, 6]
        for mode in ('p5', 'e_vlm_h'):
            archive = KEY2_ROOT / f'final/{mode}/{mode}_{split_name}_paired_predictions.npz'
            if not archive.is_file():
                raise FileNotFoundError(archive)
            saved = np.load(archive, allow_pickle=False)
            assert_saved_prediction_alignment(saved, part['actions'], part['key'],
                                              source=f'{mode} {split_name}')
            part[f'arm_{mode.split("_")[0] if mode == "p5" else "e_vlm"}_top1'] = np.asarray(saved['blend_k1'], np.float32)[..., :6]
        # Stable field names used by active7 combination.
        part['arm_p5_top1'] = part.pop('arm_p5_top1')
        part['arm_e_vlm_top1'] = part.pop('arm_e_vlm_top1')
    return parts, checkpoint, fit_bank, fit_bank_input


def build_all_memory_training_rows(*, learned_parts: Mapping[str, np.ndarray], progress_callback=None) -> dict[str, np.ndarray]:
    """Use every memory-demo window, instead of only the 4,516 variable-unit starts."""
    from . import key_shared_latent as key2
    rows=key2._load_key2_rows(KEY2_RUN)
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    learned_pairs={(int(t),int(e)) for t,e in zip(learned_parts['task'],learned_parts['episode'],strict=True)}
    keep=np.asarray([(int(rows['task'][i]),int(rows['episode'][i])) not in learned_pairs for i in memory_ids],bool)
    ids=memory_ids[keep]
    keys=np.column_stack([rows['task'][ids],rows['episode'][ids],rows['frame'][ids]]).astype(np.int32)
    pairs=set(zip(rows['task'][ids].astype(int),rows['episode'][ids].astype(int)))
    if pairs & learned_pairs:
        raise AssertionError('all-memory training windows overlap learned Eval episodes')
    if not len(ids):
        raise ValueError('no LIBERO-40 memory rows remain after Eval-demo exclusion')
    def progress(stage, **fields):
        if progress_callback is not None: progress_callback(stage, **fields)
    progress('all_memory_query_rows_start', memory_source_windows=int(len(memory_ids)),
             fit_rows=int(len(ids)), episodes=len(pairs))
    grouped={}
    for i,(task,episode) in enumerate(zip(rows['task'],rows['episode'],strict=True)):
        grouped.setdefault((int(task),int(episode)),[]).append(i)
    traces={}
    for pair,row_ids in grouped.items():
        order=np.asarray(row_ids,np.int64)
        order=order[np.argsort(rows['frame'][order])]
        traces[pair]=(order,np.asarray(rows['actions'][order,:,:7],np.float32).reshape(-1,7),
                      int(rows['frame'][order[0]]))
    histories=np.zeros((len(ids),10,7),np.float32);history_mask=np.zeros((len(ids),10),bool)
    for j,row_id in enumerate(ids):
        pair=(int(rows['task'][row_id]),int(rows['episode'][row_id]))
        _,trace,base=traces[pair]
        f=int(rows['frame'][row_id])-base
        if not 0<=f<=len(trace): raise ValueError(f'memory query frame outside trace: {pair + (int(rows["frame"][row_id]),)}')
        lo=max(0,f-10);n=f-lo
        if n: histories[j,-n:]=trace[lo:f];history_mask[j,-n:]=True
        if (j+1)%512==0 or j+1==len(ids): progress('all_memory_history',completed=j+1,total=len(ids))
    state=np.asarray(rows['state'][ids,:8],np.float32)
    target=np.asarray(rows['actions'][ids,:,:7],np.float32)
    train={'key':keys,'history':histories,'history_mask':history_mask,
           'state':state,'actions':target,'task':np.asarray(rows['task'][ids]),
           'episode':np.asarray(rows['episode'][ids]),'memory':np.ones(len(ids),bool)}
    progress('all_memory_query_rows_complete',fit_rows=int(len(ids)),episodes=len(pairs))
    return train


def attach_all_memory_features(train: Mapping[str, np.ndarray], *, progress_callback=None) -> dict[str, np.ndarray]:
    """Join exact frozen features after the state-only gate permits latent training."""
    from . import key_shared_latent as key2
    features=key2.ExactFeatureStore(KEY2_RUN,KEY2_MISSING,KEY2_SEEN_CACHE,KEY2_UNSEEN_CACHE)
    tokens,groups=features.get(train['key'],domain='libero40',progress_callback=progress_callback,
                               progress_name='all_memory_vlm')
    return {**train,'tokens':tokens,'groups':groups}


def vjepa_supported_train_subset(train: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Use all memory rows with exact V-JEPA tokens; report the cache-limited subset explicitly."""
    from . import key_shared_latent as key2
    store=key2.ExactVJEPAFeatureStore(KEY2_ROOT.parent/'vjepa',KEY2_ROOT/'vjepa/libero90_eval')
    available=set()
    for mapping,_,_ in store.maps['libero40']:
        available.update(mapping)
    keep=np.asarray([tuple(map(int,key)) in available for key in np.asarray(train['key'])],bool)
    if not keep.any(): raise ValueError('no exact V-JEPA features for memory training rows')
    result={key:np.asarray(value)[keep] for key,value in train.items()
            if isinstance(value,np.ndarray) and value.ndim and len(value)==len(keep)}
    result['vjepa_tokens'],result['vjepa_groups']=store.get(result['key'],domain='libero40')
    return result


def real_data_smoke(output: Path, *, checkpoint: Path | None = None,
                    vjepa_eval_cache: Path | None = None, device='cpu') -> dict[str, Any]:
    """Run the actual locked LIBERO memory/Dev split without evaluating Eval labels."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite GH1 real-data smoke: {output}')
    from . import key_shared_latent as key2
    from . import gripper_split as grip
    from . import seg_context_retrieval as seg
    selection_path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/compress-seen-cuts-01/production-control/evaluation-selection.json')
    selection = json.loads(selection_path.read_text())
    heldout_pairs = {tuple(map(int, pair)) for pair in selection['seen_demo_pairs']}
    rows = key2._load_key2_rows(KEY2_RUN)
    memory_ids = np.flatnonzero(np.asarray(rows['memory'], bool))
    _, metadata, _ = grip._variable_pool(rows, memory_ids)
    traces = grip._traces(rows, memory_ids)
    base_frames = {pair: int(rows['frame'][ids[0]]) for pair, ids in traces.items()}
    train_ids = np.asarray([i for i, (task, episode, _start, _length, _sign) in enumerate(metadata)
                            if (int(task), int(episode)) not in heldout_pairs], np.int64)
    if len(train_ids) < 64:
        raise AssertionError(f'only {len(train_ids)} eligible memory train units')
    eligible_memory_units = int(len(train_ids))
    # Fixed seed subsampling keeps the real-data smoke bounded while preserving
    # the exact memory-only source and held-out-demo exclusion rules.
    train_ids = train_ids[:64]
    train_keys = []
    train_actions = []
    train_history = []
    train_history_mask = []
    train_states = []
    for unit_i in train_ids:
        task, episode, start, _length, _sign = map(int, metadata[unit_i])
        frame = base_frames[(task, episode)] + start
        train_keys.append((task, episode, frame))
        hist, mask, state, future = key2._episode_history(rows, task, episode, frame)
        target = np.zeros((10, 7), np.float32)
        n = min(10, len(future))
        if n:
            target[:n] = future[:n]
            if n < 10:
                target[n:] = future[-1]
        train_actions.append(target); train_history.append(hist); train_history_mask.append(mask); train_states.append(state)
    train_keys = np.asarray(train_keys, np.int32)
    feature_store = key2.ExactFeatureStore(KEY2_RUN, KEY2_MISSING, KEY2_SEEN_CACHE, KEY2_UNSEEN_CACHE)
    train_tokens, train_groups = feature_store.get(train_keys, domain='libero40')
    train = {'key': train_keys, 'tokens': train_tokens, 'groups': train_groups,
             'history': np.asarray(train_history, np.float32),
             'history_mask': np.asarray(train_history_mask, bool),
             'state': key2._exact_start_states(train_keys, rows, KEY2_RUN, KEY2_MISSING),
             'actions': np.asarray(train_actions, np.float32), 'task': train_keys[:, 0],
             'episode': train_keys[:, 1], 'memory': np.ones(len(train_keys), bool)}

    # Assemble only the two fixed LIBERO-90 Dev tasks. Eval task action rows are
    # never indexed; metadata is scanned only to locate the Dev rows.
    unseen_arrays = {name: np.load(KEY2_UNSEEN_CACHE / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                     for name in ('task', 'episode', 'frame', 'actions', 'state', 'is_dev')}
    dev_ids = np.flatnonzero(np.asarray(unseen_arrays['is_dev'], bool))
    if set(np.unique(np.asarray(unseen_arrays['task'][dev_ids], int))) != set(DEV_TASKS) or len(dev_ids) != 217:
        raise AssertionError('fixed Dev row selection changed')
    dev_by_episode: dict[tuple[int, int], list[int]] = {}
    for idx in dev_ids:
        dev_by_episode.setdefault((int(unseen_arrays['task'][idx]), int(unseen_arrays['episode'][idx])), []).append(int(idx))
    dev_histories = []; dev_masks = []; dev_keys = []; dev_actions = []; dev_states = []
    for (task, episode), episode_ids in sorted(dev_by_episode.items()):
        episode_ids.sort(key=lambda idx: int(unseen_arrays['frame'][idx]))
        first = int(unseen_arrays['frame'][episode_ids[0]])
        previous_frame = first
        trace_parts = [np.asarray(unseen_arrays['actions'][episode_ids[0], :10, :7], np.float32)]
        for idx in episode_ids[1:]:
            frame = int(unseen_arrays['frame'][idx]); delta = frame - previous_frame
            if not 0 < delta <= 10:
                raise ValueError(f'invalid Dev action-window stride {delta}')
            trace_parts.append(np.asarray(unseen_arrays['actions'][idx, delta:10, :7], np.float32))
            previous_frame = frame
        trace = np.concatenate(trace_parts, axis=0)
        state_trace = np.repeat(np.asarray(unseen_arrays['state'][episode_ids[0], :8], np.float32)[None], len(trace), axis=0)
        for idx in episode_ids:
            offset = int(unseen_arrays['frame'][idx]) - first
            state_trace[offset:min(offset + 5, len(state_trace))] = np.asarray(unseen_arrays['state'][idx, :8], np.float32)
        for idx in episode_ids:
            offset = int(unseen_arrays['frame'][idx]) - first
            hist = seg.causal_history_at(trace, state_trace, offset)
            if hist is None:
                hist_actions = np.zeros((10, 7), np.float32); hist_mask = np.zeros(10, bool)
            else:
                ha = np.asarray(hist[0], np.float32)[-10:]
                hist_actions = np.zeros((10, 7), np.float32); hist_mask = np.zeros(10, bool)
                if len(ha): hist_actions[-len(ha):] = ha; hist_mask[-len(ha):] = True
            dev_histories.append(hist_actions); dev_masks.append(hist_mask)
            dev_keys.append((task, episode, int(unseen_arrays['frame'][idx])))
            dev_actions.append(np.asarray(unseen_arrays['actions'][idx, :10, :7], np.float32))
            dev_states.append(np.asarray(unseen_arrays['state'][idx, :8], np.float32))
    dev_keys = np.asarray(dev_keys, np.int32)
    dev_tokens, dev_groups = feature_store.get(dev_keys, domain='libero90')
    dev = {'key': dev_keys, 'tokens': dev_tokens, 'groups': dev_groups,
           'history': np.asarray(dev_histories, np.float32), 'history_mask': np.asarray(dev_masks, bool),
           'state': np.asarray(dev_states, np.float32),
           'actions': np.asarray(dev_actions, np.float32), 'task': dev_keys[:, 0], 'episode': dev_keys[:, 1]}
    train_pairs = set(zip(train['task'].astype(int), train['episode'].astype(int)))
    if train_pairs & heldout_pairs or not train['memory'].all():
        raise AssertionError('memory train rows overlap held-out learned-evaluation demos')
    selected_config = json.loads((KEY2_ROOT / 'selected_configs.json').read_text())['selected']['e_vlm_h']
    checkpoint = Path(checkpoint or selected_config['checkpoint'])
    if not train['memory'].all() or set(np.unique(dev['task'])) != set(DEV_TASKS):
        raise AssertionError('real-data smoke split guard failed')
    thread_count = int(os.environ.get('OMP_NUM_THREADS', '8'))
    raw_parts = {'train': train, 'dev': dev}
    encoded_a = encode_frozen_key2_e(checkpoint, raw_parts, mode='e_vlm_h', device=device)
    model_a, selected_a, grid_a = fit_dev_grid(encoded_a['train'], encoded_a['dev'], kinds=('cross_attention',),
                                               hidden_dims=(64,), learning_rates=(3e-4,), epochs_grid=(1,),
                                               device=device, num_threads=thread_count)
    metrics_a = evaluate_logits(predict(model_a, encoded_a['dev'], device=device), encoded_a['dev']['target'])
    if not np.isclose(selected_a['dev_gripper_mse'], metrics_a['gripper_contribution_mse_sign'], atol=1e-8):
        raise AssertionError('G-a checkpoint does not reproduce its Dev selection MSE')

    vj_cfg = json.loads((KEY2_ROOT / 'selected_configs.json').read_text())['selected']['e_vj_h']
    vjepa_root = KEY2_ROOT.parent / 'vjepa'
    vjepa_store = key2.ExactVJEPAFeatureStore(vjepa_root, KEY2_ROOT / 'vjepa/libero90_eval')
    train['vjepa_tokens'], train['vjepa_groups'] = vjepa_store.get(train_keys, domain='libero40')
    dev['vjepa_tokens'], dev['vjepa_groups'] = vjepa_store.get(dev_keys, domain='libero90')
    encoded_b = encode_frozen_key2_e(Path(vj_cfg['checkpoint']), {'train': train, 'dev': dev},
                                     mode='e_vj_h', device=device)
    model_b, selected_b, grid_b = fit_dev_grid(encoded_b['train'], encoded_b['dev'], kinds=('cross_attention',),
                                               hidden_dims=(64,), learning_rates=(3e-4,), epochs_grid=(1,),
                                               device=device, num_threads=thread_count)
    metrics_b = evaluate_logits(predict(model_b, encoded_b['dev'], device=device), encoded_b['dev']['target'])

    pooled_parts = pooled_pi05_features(raw_parts)
    model_d, selected_d, grid_d = fit_dev_grid(pooled_parts['train'], pooled_parts['dev'], kinds=('mlp',),
                                               hidden_dims=(64,), learning_rates=(3e-4,), epochs_grid=(1,),
                                               device=device, num_threads=thread_count)
    metrics_d = evaluate_logits(predict(model_d, pooled_parts['dev'], device=device), pooled_parts['dev']['target'])

    l90x_rows = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/l90x_rows')
    l90x_pooled = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-action-01/la2-smoke4-20261115/features/l90x_pooled.npy')
    l90x_data = encode_l90x_frozen_e(checkpoint, l90x_rows, l90x_pooled, row_limit=32, device=device)
    train_e = append_l90x_training_data(encoded_a['train'], l90x_data, l90x_data['manifest'])
    model_e, selected_e, grid_e = fit_dev_grid(train_e, encoded_a['dev'], kinds=('cross_attention',),
                                               hidden_dims=(64,), learning_rates=(3e-4,), epochs_grid=(1,),
                                               device=device, num_threads=thread_count)
    metrics_e = evaluate_logits(predict(model_e, encoded_a['dev'], device=device), encoded_a['dev']['target'])

    fit_bank = {'length': np.full(len(train['key']), 10, np.int32),
                'natural_actions': np.asarray(train['actions'], np.float32),
                'task': np.asarray(train['task']), 'episode': np.asarray(train['episode'])}
    joint_e, joint_head, selected_c, grid_c = fit_joint_e_grid(
        checkpoint, fit_bank, train, train, dev, lambdas=(.1,), learning_rates=(1e-4,),
        max_epochs=(1,), device=device, batch_size=8, num_threads=thread_count)
    joint_rep, _ = key2._query_representations(joint_e, 'e_vlm_h', dev, device)
    with torch.no_grad():
        joint_logits = joint_head(torch.as_tensor(joint_rep, dtype=torch.float32, device=device),
                                  torch.as_tensor(dev['state'][:, 6:8], dtype=torch.float32, device=device))['logits'].cpu().numpy()
    metrics_c = evaluate_logits(joint_logits, dev['actions'][:, :, 6])
    if not np.isclose(selected_c['dev_gripper_mse'], metrics_c['gripper_contribution_mse_sign'], atol=1e-8):
        raise AssertionError('G-c checkpoint does not reproduce its Dev selection MSE')
    output.mkdir(parents=True, exist_ok=True)
    rows_by_method = {'G-a': {'selected': selected_a, 'dev': metrics_a, 'grid': grid_a},
                      'G-b': {'selected': selected_b, 'dev': metrics_b, 'grid': grid_b},
                      'G-c': {'selected': selected_c, 'dev': metrics_c, 'grid': grid_c},
                      'G-d': {'selected': selected_d, 'dev': metrics_d, 'grid': grid_d},
                      'G-e': {'selected': selected_e, 'dev': metrics_e, 'grid': grid_e,
                              'l90x_rows': int(len(l90x_data['target']))}}
    result = {'status': 'real_data_smoke_only', 'seed': SEED, 'data_source': 'KEY2-F fixed LIBERO-40 memory / LIBERO-90 Dev rows',
              'rows': {'memory_source_windows': int(len(memory_ids)),
                       'memory_units_before_learned_holdout': int(len(metadata)),
                       'eligible_memory_units_after_learned_holdout': eligible_memory_units,
                       'memory_fit_units_used': int(len(train['key'])),
                       'dev_used': int(len(dev['key']))},
              'dev_tasks': sorted(set(map(int, dev['task']))),
              'selected_by': 'Dev sign gripper-contribution MSE only', 'rows_by_method': rows_by_method,
              'eval_labels_read_or_scored': False, 'production_evaluation': 'pending reviewer approval'}
    (output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    report = ['# GH1 real-data smoke', '', '| Row | Dev sign grip MSE | L90X fit rows |', '|---|---:|---:|']
    for label, record in rows_by_method.items():
        report.append(f"| {label} | {record['selected']['dev_gripper_mse']:.6f} | {record.get('l90x_rows', 0)} |")
    report.extend(['', f"LIBERO-40 memory fit rows used: {len(train['key'])}; LIBERO-90 Dev task 6/7 rows: {len(dev['key'])}.",
                   'Eval labels were not read or scored. This is an actual-data smoke, not the final Eval result.'])
    (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return result


def run_production(output: Path, *, device: str = 'cuda:0', num_threads: int | None = None,
                   bootstrap_reps: int = 5000) -> dict[str, Any]:
    """Run the locked GH1 matrix; call only after implementation review passes.

    All head/lambda/epoch settings are chosen on Dev. Learned and unseen Eval
    rows are encoded and scored once after every selection is frozen.
    """
    from . import key_shared_latent as key2
    output = Path(output)
    if output.exists():
        # Permit the reusable L90X cache and append-only progress log. Any
        # model/result artifact means a rerun could overwrite a scored run.
        unexpected = [p.name for p in output.iterdir() if p.name not in {'l90x_features', 'progress.log'}]
        if unexpected:
            raise FileExistsError(f'refusing to overwrite GH1 production output: {output}: {unexpected}')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    threads = int(num_threads if num_threads is not None else os.environ.get('OMP_NUM_THREADS', '8'))
    torch.set_num_threads(threads)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for GH1 but unavailable')
        torch.cuda.set_per_process_memory_fraction(.12, torch.device(device))
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / 'progress.log'
    started = time.monotonic()
    def progress(stage: str, **fields: Any) -> None:
        entry = {'elapsed_seconds': round(time.monotonic() - started, 3), 'stage': stage, **fields}
        with progress_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
    progress('production_start', device=device, cpu_threads=threads, bootstrap_reps=int(bootstrap_reps),
             joint_batch_size=128, joint_lambdas=[.03, .1, .3, 1.0],
             joint_learning_rates=[1e-4, 3e-4], joint_max_epochs=[4, 8])
    parts, e_a_checkpoint, fit_bank, fit_bank_input = load_key2_frozen_e_parts(progress_callback=progress)
    unit_start_rows = int(len(parts['train']['key']))
    parts['train'] = build_all_memory_training_rows(learned_parts=parts['learned'], progress_callback=progress)
    progress('key2_parts_validated', train_rows=int(len(parts['train']['key'])),
             original_variable_unit_rows=unit_start_rows,
             dev_rows=int(len(parts['dev']['key'])), learned_rows=int(len(parts['learned']['key'])),
             unseen_rows=int(len(parts['unseen']['key'])), fit_candidates=int(len(fit_bank['length'])))
    train, dev = parts['train'], parts['dev']
    if not np.asarray(train['memory'], bool).all():
        raise AssertionError('production train rows include non-memory demos')
    train_pairs = set(zip(np.asarray(train['task'], int), np.asarray(train['episode'], int)))
    eval_pairs = set(zip(np.asarray(parts['learned']['task'], int), np.asarray(parts['learned']['episode'], int)))
    if train_pairs & eval_pairs:
        raise AssertionError('learned Eval demos appear in production train rows')
    expected_eval_rows = {'learned': 621, 'unseen': 619}
    expected_hard_rows = {'learned': 125, 'unseen': 134}
    for split_name, expected_count in expected_eval_rows.items():
        split = parts[split_name]
        if len(split['key']) != expected_count:
            raise AssertionError(f'locked Eval cardinality changed for {split_name}: {len(split["key"])}')
        hard_count = int(np.asarray(split['hard'], bool).sum())
        if hard_count != expected_hard_rows[split_name]:
            raise AssertionError(f'locked hard-row cardinality changed for {split_name}: {hard_count}')

    # Mandatory no-latent learning-path gate. The exact G1 reference uses past
    # executed gripper actions for non-initial windows, so the state-only model
    # must independently demonstrate that it can meet the requested bound.
    progress('state_only_gate_start', train_rows=int(len(train['key'])), dev_rows=int(len(dev['key'])))
    state_train = {'gripper_state': np.asarray(train['state'], np.float32)[:, 6:8],
                   'target': np.asarray(train['actions'], np.float32)[:, :, 6]}
    state_dev = {'gripper_state': np.asarray(dev['state'], np.float32)[:, 6:8],
                 'target': np.asarray(dev['actions'], np.float32)[:, :, 6]}
    state_model, state_config, state_grid = fit_state_only_grid(
        state_train, state_dev, device=device, num_threads=threads,
        progress_callback=progress)
    state_gate_logits: dict[str, np.ndarray] = {}
    state_gate_metrics: dict[str, Any] = {}
    state_gate_sources = {'dev': dev}
    for split_name in ('dev', 'learned', 'unseen'):
        part = state_gate_sources.get(split_name, parts[split_name])
        state_input = {'gripper_state': np.asarray(part['state'], np.float32)[:, 6:8],
                       'target': np.asarray(part['actions'], np.float32)[:, :, 6]}
        logits = predict_state_only(state_model, state_input, device=device)
        state_gate_logits[split_name] = logits
        if split_name == 'dev':
            state_gate_metrics[split_name] = evaluate_logits(logits, state_input['target'])
        else:
            scoring = {'target': state_input['target'], 'task': part['task'], 'episode': part['episode'],
                       'hard': part['hard'], 'pi05_gripper': np.asarray(part['pi05'], np.float32)[:, :, 6]}
            scored = evaluate_split(logits, scoring, pi05_actions=scoring['pi05_gripper'],
                                    bootstrap_reps=bootstrap_reps, bootstrap_seed=4101)
            arms = {}
            signs = np.where(logits >= 0, 1., -1.)
            for arm_key in ('arm_p5_top1', 'arm_e_vlm_top1'):
                arm = np.asarray(part[arm_key], np.float32)
                active = active7_with_gripper(arm, part['actions'], signs)
                hard_mask = np.asarray(part['hard'], bool)
                active['hard_active7_mse'] = active7_with_gripper(
                    arm[hard_mask], part['actions'][hard_mask], signs[hard_mask])['active7_mse']
                arms[arm_key] = active
            scored['active7_by_arm'] = arms
            state_gate_metrics[split_name] = scored
    g1_bound = 0.05153221637010574
    gate_scores = {name: float(value['gripper_contribution_mse_sign'] if name == 'dev'
                                else value['all']['gripper_contribution_mse_sign'])
                   for name, value in state_gate_metrics.items()}
    gate_passed = all(np.isfinite(score) and score <= g1_bound for score in gate_scores.values())
    grip1_path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/gripper-split-01/attempt-08/results.json')
    grip1_result = json.loads(grip1_path.read_text())['D_gripper_decisions']['eval']
    hold_repro = gripper_hold_reference(parts['unseen'])
    hold_reproduction = {}
    for name, values in hold_repro.items():
        actual = float(gripper_contribution_mse(values, parts['unseen']['actions'][:, :, 6]).mean())
        ref_name = 'G1_state_initialized_hold' if name.startswith('G1') else 'G2_previous_hold'
        expected = float(grip1_result[ref_name]['all']['grip_mse_contribution'])
        hold_reproduction[name] = {'recomputed': actual, 'GRIP1_reference': expected, 'difference': actual - expected}
        if not np.isclose(actual, expected, atol=1e-8):
            raise AssertionError(f'{name} does not reproduce GRIP1 on locked unseen rows: {actual} != {expected}')
    pi05_recomputed = float(gripper_contribution_mse(
        np.asarray(parts['unseen']['pi05'], np.float32)[:, :, 6],
        parts['unseen']['actions'][:, :, 6]).mean())
    pi05_reference = float(grip1_result['G3_pi05_gripper']['all']['grip_mse_contribution'])
    if not np.isclose(pi05_recomputed, pi05_reference, atol=5e-6):
        raise AssertionError(f'π0.5 target/score does not match GRIP1: {pi05_recomputed} != {pi05_reference}')
    np.savez_compressed(output / 'state_only_gate_predictions.npz', **{
        f'{name}_logits': logits for name, logits in state_gate_logits.items()})
    torch.save(state_model.state_dict(), output / 'g-g_state_only_head.pt')
    memory_source_windows = int(np.asarray(np.load(KEY2_RUN / 'rows/memory.npy', mmap_mode='r', allow_pickle=False), bool).sum())
    state_gate = {'passed': gate_passed, 'threshold': g1_bound, 'scores': gate_scores,
                  'selected_by': 'Dev sign gripper-contribution MSE', 'selected_config': state_config,
                  'dev_grid': state_grid, 'metrics': state_gate_metrics,
                  'normalization': {'source_state_is_q01_q99_normalized': True,
                                    'train_mean': state_model.state_mean.detach().cpu().tolist(),
                                    'train_std': state_model.state_std.detach().cpu().tolist()},
                  'dev_positive_target_steps': int((state_dev['target'] > 0).sum()),
                  'dev_target_steps': int(state_dev['target'].size),
                  'target_sign_counts': {'train_positive': int((state_train['target'] > 0).sum()),
                                         'train_steps': int(state_train['target'].size),
                                         'dev_positive': int((state_dev['target'] > 0).sum()),
                                         'dev_steps': int(state_dev['target'].size)},
                  'reference_reproduction': {'G1_and_G2': hold_reproduction,
                                             'pi05': {'recomputed': pi05_recomputed,
                                                      'GRIP1_reference': pi05_reference,
                                                      'difference': pi05_recomputed - pi05_reference}},
                  'data': {'memory_source_windows': memory_source_windows,
                           'original_variable_unit_rows': unit_start_rows,
                           'memory_fit_unit_rows': int(len(train['key'])),
                           'learned_eval_rows': int(len(parts['learned']['key'])),
                           'train_learned_episode_overlap': 0}}
    progress('state_only_gate_complete', passed=gate_passed, scores=gate_scores, threshold=g1_bound,
             selected_config=state_config)
    if not gate_passed:
        ref = grip1_result
        invalidated = {'production-03': 'invalid for model selection: Dev has no positive gripper target steps; Dev sign-MSE selected a constant-negative regime. Frozen-head fit also used one full-batch update per epoch (8-32 updates total).'}
        results = {'status': 'state_only_gate_failed', 'seed': SEED,
                   'data_counts': state_gate['data'], 'state_only_gate': state_gate,
                   'grip1_reference_rows': {name: ref[name] for name in
                       ('G3_pi05_gripper','G1_state_initialized_hold','G2_previous_hold')},
                   'invalidated_results': invalidated,
                   'protocol': {'eval_used_for_mandatory_state_gate_only': True,
                                'G-a_to_G-f_fitted_or_scored': False}}
        (output / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        report = ['# GH1 production-04 — gated before latent-head run', '',
                  '## 原因', '',
                  'production-03 の主結果は無効。Dev task 6/7 の217行・2,170 grip target stepsは全て符号 -1 で、Dev sign-MSE は定数 -1 予測に0を与え、潜在入力や設定の良否を判定できなかった。さらに旧 `fit_head` は各epochにつき全4,516行を一括した optimizer step 1回だけで、8〜32 epoch設定でも更新は8〜32回だった。production-04では状態の train-only z-score と shuffled mini-batch 学習へ修正した。', '',
                  f"LIBERO-40 memory sourceは全{memory_source_windows:,}窓。production-04はheld-out learned episodeとの重複なしで{len(train['key']):,}全時点を学習に使用し、元の可変unit開始は{unit_start_rows:,}行。621はlearned Eval行数であり、学習行数ではない。評価行は学習に混ざっていない。", '',
                  '状態は生の指位置ではなく OpenPI Q01/Q99 正規化済み。train state[6:8] の範囲は概ね [-1.13, 1.12]、Dev/Evalも同じ尺度。action target は同じPI0REF固定行の `actions[:,:,6]` を使用し、π0.5 .026362 の基準値は production-03 と一致する。状態の高低は state[6] threshold .30195 で高い側を action sign -1 とする既存GRIP1規則と一致する。', '',
                  '## 状態だけゲート（Dev選択後の一回採点）', '',
                  '| Split | Method | Grip MSE all | Grip MSE hard |', '|---|---|---:|---:|',
                  f"| Dev | G-g state only | {gate_scores['dev']:.6f} | — |",
                  f"| learned | G-g state only | {state_gate_metrics['learned']['all']['gripper_contribution_mse_sign']:.6f} | {state_gate_metrics['learned']['hard']['gripper_contribution_mse_sign']:.6f} |",
                  f"| unseen | G-g state only | {state_gate_metrics['unseen']['all']['gripper_contribution_mse_sign']:.6f} | {state_gate_metrics['unseen']['hard']['gripper_contribution_mse_sign']:.6f} |",
                  f"| unseen | π0.5 reference | {ref['G3_pi05_gripper']['all']['grip_mse_contribution']:.6f} | {ref['G3_pi05_gripper']['hard']['grip_mse_contribution']:.6f} |",
                  f"| unseen | G1 state-initialized hold | {ref['G1_state_initialized_hold']['all']['grip_mse_contribution']:.6f} | {ref['G1_state_initialized_hold']['hard']['grip_mse_contribution']:.6f} |",
                  f"| unseen | G2 previous hold | {ref['G2_previous_hold']['all']['grip_mse_contribution']:.6f} | {ref['G2_previous_hold']['hard']['grip_mse_contribution']:.6f} |", '',
                  f"Gate status: **{'PASS' if gate_passed else 'FAIL'}**. The mandatory threshold is {g1_bound:.6f}; G-a through G-f were not trained or scored because the gate failed.", '',
                  'GRIP1 G1 is not a state-only reference on ordinary rows: it repeats the previous executed gripper action and reads proprio state only at episode starts. The requested state-only comparison therefore has less information than the .051532 reference. Dev is also single-class, so Dev MSE cannot calibrate this input-only head for the held-out distribution.', '',
                  'GRIP1 unseen reference: π0.5 .026362, state-initialized hold .051532, previous hold .074611. See `state_only_gate_predictions.npz` and full metrics in `results.json`.', '']
        (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
        return results

    train = attach_all_memory_features(train, progress_callback=progress)
    parts['train'] = train
    train_b = vjepa_supported_train_subset(train)
    progress('vjepa_supported_train_rows', all_memory_rows=int(len(train['key'])),
             exact_vjepa_rows=int(len(train_b['key'])))

    configs = {'kinds': ('mlp', 'cross_attention', 'cross_attention_aux'),
               'hidden_dims': (64, 128), 'learning_rates': (3e-4, 1e-3),
               'epochs_grid': (8, 16, 32)}
    selected: dict[str, Any] = {'G-g': {'config': state_config, 'grid': state_grid}}
    models: dict[str, nn.Module] = {'G-g': state_model}
    enc_checkpoints: dict[str, Path] = {'G-a': e_a_checkpoint, 'G-b': Path(
        json.loads((KEY2_ROOT / 'selected_configs.json').read_text())['selected']['e_vj_h']['checkpoint'])}

    # G-a frozen KEY2-F E-VLM+H.
    progress('g_a_encode_start', device=device)
    enc_a = encode_frozen_key2_e(e_a_checkpoint, {'train': train, 'dev': dev}, mode='e_vlm_h', device=device,
                                 progress_callback=progress)
    progress('g_a_dev_grid_start')
    model, cfg, grid = fit_dev_grid(enc_a['train'], enc_a['dev'], device=device, num_threads=threads,
                                    progress_callback=progress, **configs)
    models['G-a'] = model; selected['G-a'] = {'config': cfg, 'grid': grid}
    progress('g_a_complete', selected=cfg)
    del enc_a

    # G-b frozen KEY2-F E-VJ+H.
    progress('g_b_encode_start', device=device)
    enc_b = encode_frozen_key2_e(enc_checkpoints['G-b'], {'train': train_b, 'dev': dev}, mode='e_vj_h', device=device,
                                 progress_callback=progress)
    progress('g_b_dev_grid_start')
    model, cfg, grid = fit_dev_grid(enc_b['train'], enc_b['dev'], device=device, num_threads=threads,
                                    progress_callback=progress, **configs)
    models['G-b'] = model; selected['G-b'] = {'config': cfg, 'grid': grid}
    progress('g_b_complete', selected=cfg)
    del enc_b

    # G-d π0.5 final-prefix mean plus current gripper state.
    pooled = pooled_pi05_features({'train': train, 'dev': dev})
    progress('g_d_dev_grid_start')
    model, cfg, grid = fit_dev_grid(pooled['train'], pooled['dev'], kinds=('mlp',),
                                    hidden_dims=configs['hidden_dims'], learning_rates=configs['learning_rates'],
                                    epochs_grid=configs['epochs_grid'], device=device, num_threads=threads,
                                    progress_callback=progress)
    models['G-d'] = model; selected['G-d'] = {'config': cfg, 'grid': grid}
    progress('g_d_complete', selected=cfg)
    del pooled

    # G-e extracts/loads exact L90X pooled prefixes, asserts the LA2 manifest,
    # passes them through the same frozen E, then adds only those rows to fit.
    from . import latent_action_units_la2 as la2
    l90x_row_cache = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/l90x_rows')
    l90x_feature_dir = output / 'l90x_features'
    # The pooled-prefix extractor writes row progress atomically and can
    # resume after an interrupted process.  Keep this directory when the
    # production command is restarted; the output root itself is still
    # required to be empty on the first invocation.
    l90x_feature_dir.mkdir(parents=True, exist_ok=True)
    l90x_feature_path = l90x_feature_dir / 'l90x_pooled.npy'
    if not (l90x_feature_dir / 'extract_complete.json').exists():
        progress('l90x_prefix_extraction_start', device=device)
        la2.extract_l90x_pooled(l90x_row_cache, l90x_feature_dir, batch_size=2,
                                device=device, memory_fraction=.08)
        progress('l90x_prefix_extraction_complete')
    else:
        progress('l90x_prefix_cache_reused', rows=11030, cache=str(l90x_feature_path))
    progress('g_e_latent_encoding_start', device=device)
    l90x_data = encode_l90x_frozen_e(e_a_checkpoint, l90x_row_cache, l90x_feature_path,
                                     device=device, progress_callback=progress)
    progress('g_e_latent_encoding_complete', rows=int(len(l90x_data['target'])))
    progress('g_e_libero40_encoding_start', device=device)
    train_a = encode_frozen_key2_e(e_a_checkpoint, {'train': train}, mode='e_vlm_h', device=device,
                                   progress_callback=progress)['train']
    progress('g_e_libero40_encoding_complete', rows=int(len(train_a['target'])))
    train_e = append_l90x_training_data(train_a, l90x_data, l90x_data['manifest'])
    dev_a = encode_frozen_key2_e(e_a_checkpoint, {'dev': dev}, mode='e_vlm_h', device=device,
                                 progress_callback=progress)['dev']
    progress('g_e_dev_grid_start')
    model, cfg, grid = fit_dev_grid(train_e, dev_a, device=device, num_threads=threads,
                                    progress_callback=progress, **configs)
    models['G-e'] = model; selected['G-e'] = {'config': cfg, 'grid': grid,
                                              'l90x_rows': int(len(l90x_data['target'])),
                                              'l90x_manifest': str(l90x_row_cache / 'manifest.json')}
    progress('g_e_complete', selected=cfg, fit_rows=int(len(train_e['target'])))
    # G-e augments only the head-fitting data; its observation encoder remains
    # the same frozen KEY2-F E-VLM+H used by G-a.
    enc_checkpoints['G-e'] = e_a_checkpoint
    del train_a, train_e, dev_a, l90x_data

    # JP1's selected E-VLM+H checkpoint uses the same SharedLatentEncoder
    # contract, so attach the identical head grid when that production E exists.
    jp1_selected_path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-pred-01/production/selected.json')
    jp1_encoder_status = 'unavailable'
    if jp1_selected_path.is_file():
        jp1_selected = json.loads(jp1_selected_path.read_text())
        jp1_checkpoint = Path(jp1_selected['a_e_vlm_h']['checkpoint'])
        if jp1_checkpoint.is_file():
            progress('g_f_jp1_encode_start', checkpoint=str(jp1_checkpoint), device=device)
            jp1_encoded = encode_frozen_key2_e(jp1_checkpoint, {'train': train, 'dev': dev},
                                               mode='e_vlm_h', device=device,
                                               progress_callback=progress)
            progress('g_f_jp1_dev_grid_start')
            model, cfg, grid = fit_dev_grid(jp1_encoded['train'], jp1_encoded['dev'], device=device,
                                            num_threads=threads, progress_callback=progress, **configs)
            models['G-f'] = model
            selected['G-f'] = {'config': cfg, 'grid': grid, 'encoder': 'JP1 a_e_vlm_h',
                               'checkpoint': str(jp1_checkpoint)}
            enc_checkpoints['G-f'] = jp1_checkpoint
            jp1_encoder_status = 'included'
            progress('g_f_jp1_complete', selected=cfg)
            del jp1_encoded
    progress('optional_encoder_sources', JP1=jp1_encoder_status,
             LT1='unavailable: smoke only, no trained E checkpoint')

    # G-c joint E/head optimization; λ/lr/epoch are selected with Dev MSE.
    progress('g_c_joint_grid_start', device=device, batch_size=128,
             lambdas=[.03, .1, .3, 1.0], learning_rates=[1e-4, 3e-4], max_epochs=[4, 8])
    model_c, head_c, cfg_c, grid_c = fit_joint_e_grid(
        e_a_checkpoint, fit_bank, fit_bank_input, train, dev,
        lambdas=(.03, .1, .3, 1.0), learning_rates=(1e-4, 3e-4),
        max_epochs=(4, 8), seed=SEED, device=device, batch_size=128, num_threads=threads,
        progress_callback=progress)
    joint_encoder_path = output / 'g_c_joint_e.pt'
    torch.save(model_c.state_dict(), joint_encoder_path)
    torch.save(head_c.state_dict(), output / 'g_c_head.pt')
    models['G-c'] = head_c; enc_checkpoints['G-c'] = joint_encoder_path
    selected['G-c'] = {'config': cfg_c, 'grid': grid_c}
    progress('g_c_complete', selected=cfg_c)
    for label, model0 in models.items():
        if label != 'G-c':
            torch.save(model0.state_dict(), output / f'{label.lower()}_head.pt')

    # Eval is touched only after all Dev selections and checkpoints have been saved.
    evaluation_parts = {name: parts[name] for name in ('learned', 'unseen')}
    eval_latents: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    eval_encoder_rows = [('G-a', 'e_vlm_h'), ('G-b', 'e_vj_h'), ('G-c', 'e_vlm_h'), ('G-e', 'e_vlm_h')]
    if 'G-f' in enc_checkpoints:
        eval_encoder_rows.append(('G-f', 'e_vlm_h'))
    for label, mode in eval_encoder_rows:
        progress('eval_latent_encode_start', label=label, mode=mode, device=device)
        eval_latents[label] = encode_frozen_key2_e(enc_checkpoints[label], evaluation_parts,
                                                   mode=mode, device=device, progress_callback=progress)
        progress('eval_latent_encode_complete', label=label)
    pooled_eval = pooled_pi05_features(evaluation_parts)
    prediction_logits: dict[str, dict[str, np.ndarray]] = {k: {} for k in models}
    split_results: dict[str, dict[str, Any]] = {}
    for split_name in ('learned', 'unseen'):
        progress('eval_score_split_start', split=split_name)
        split_out = {}
        target = np.asarray(parts[split_name]['actions'], np.float32)
        for label, model0 in models.items():
            source = pooled_eval[split_name] if label == 'G-d' else eval_latents[label][split_name]
            logits = predict(model0, source, device=device)
            prediction_logits[label][split_name] = logits
            scored = evaluate_split(logits, source, pi05_actions=source.get('pi05_gripper'),
                                    bootstrap_reps=bootstrap_reps, bootstrap_seed=4101)
            arms = {}
            sign = np.where(logits >= 0, 1., -1.)
            for arm_key in ('arm_p5_top1', 'arm_e_vlm_top1'):
                arm = np.asarray(parts[split_name][arm_key], np.float32)
                whole = active7_with_gripper(arm, target, sign)
                hard = np.asarray(parts[split_name]['hard'], bool)
                whole['hard_active7_mse'] = (active7_with_gripper(arm[hard], target[hard], sign[hard])['active7_mse']
                                             if hard.any() else None)
                pi05_grip = np.asarray(parts[split_name]['pi05_gripper'], np.float32)
                whole['pi05_gripper_active7_mse'] = active7_with_gripper(arm, target, pi05_grip)['active7_mse']
                arms[arm_key] = whole
            scored['active7_by_arm'] = arms
            if label == 'G-c':
                # Report whether joint E/head fitting changed retrieval quality
                # on the locked Eval rows. This is diagnostic only: all model
                # and hyperparameter selection has already finished on Dev.
                retrieval_scores, retrieval_errors = key2._score_model(
                    model_c, 'e_vlm_h', fit_bank, fit_bank_input, parts[split_name], device)
                joint_mse = top1_candidate_arm6_mse(retrieval_scores, retrieval_errors)
                frozen_arm = np.asarray(parts[split_name]['arm_e_vlm_top1'], np.float32)
                frozen_mse = float(np.square(frozen_arm - target[..., :6]).sum(axis=(1, 2)).mean() / 70.)
                hard_mask = np.asarray(parts[split_name]['hard'], bool)
                joint_hard = (top1_candidate_arm6_mse(retrieval_scores[hard_mask],
                                                      retrieval_errors[hard_mask])
                              if hard_mask.any() else None)
                frozen_hard = (float(np.square(frozen_arm[hard_mask] - target[hard_mask, :, :6])
                                     .sum(axis=(1, 2)).mean() / 70.) if hard_mask.any() else None)
                scored['joint_encoder_arm6_top1_mse'] = joint_mse
                scored['joint_encoder_arm6_top1_hard_mse'] = joint_hard
                scored['frozen_e_vlm_h_arm6_top1_mse'] = frozen_mse
                scored['frozen_e_vlm_h_arm6_top1_hard_mse'] = frozen_hard
                scored['joint_minus_frozen_e_vlm_h_arm6_mse'] = joint_mse - frozen_mse
            split_out[label] = scored
        split_results[split_name] = split_out
        progress('eval_score_split_complete', split=split_name)

    paired = {}
    grip_module = __import__(f'{__package__}.gripper_split', fromlist=['episode_codes', 'paired_episode_bootstrap'])
    for split_name in ('learned', 'unseen'):
        part = parts[split_name]; codes = grip_module.episode_codes(part['task'], part['episode'])
        target = np.asarray(part['actions'], np.float32)[:, :, 6]
        per_row = {label: gripper_contribution_mse(np.where(prediction_logits[label][split_name] >= 0, 1., -1.), target)
                   for label in models}
        per_row['pi05'] = gripper_contribution_mse(part['pi05_gripper'], target)
        hard = np.asarray(part['hard'], bool)
        paired[split_name] = {'head_minus_pi05': {}, 'G-e_minus_G-a': None, 'G-c_minus_G-a': None}
        for label in models:
            for subset, mask in (('all', np.ones(len(target), bool)), ('hard', hard)):
                if mask.any():
                    paired[split_name]['head_minus_pi05'].setdefault(label, {})[subset] = \
                        grip_module.paired_episode_bootstrap(per_row[label][mask] - per_row['pi05'][mask], codes[mask],
                                                             reps=bootstrap_reps, seed=4101)
        for left, right, name in (('G-e', 'G-a', 'G-e_minus_G-a'), ('G-c', 'G-a', 'G-c_minus_G-a')):
            paired[split_name][name] = {}
            for subset, mask in (('all', np.ones(len(target), bool)), ('hard', hard)):
                if mask.any():
                    paired[split_name][name][subset] = grip_module.paired_episode_bootstrap(
                        per_row[left][mask] - per_row[right][mask], codes[mask], reps=bootstrap_reps, seed=4102)

    references = json.loads((Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/gripper-split-01/attempt-08/results.json')).read_text())
    ref = references['D_gripper_decisions']['eval']
    expected = {'G3_pi05_gripper': .026361560449004173, 'G4_classifier': .04921720176935196,
                'G2_previous_hold': .07461091876029968, 'G1_state_initialized_hold': .05153221637010574,
                'G0_unit_gripper': .19353152811527252}
    for name, value in expected.items():
        actual = float(ref[name]['all']['grip_mse_contribution'])
        if not np.isclose(actual, value, atol=1e-8):
            raise AssertionError(f'GRIP1 reference mismatch for {name}: {actual} != {value}')
    baseline_rows = {name: ref[name] for name in expected}
    results = {'status': 'complete', 'seed': SEED, 'rows': {n: len(parts[n]['key']) for n in ('learned', 'unseen')},
               'hard_rows': {n: int(np.asarray(parts[n]['hard']).sum()) for n in ('learned', 'unseen')},
               'selected_configs': selected, 'splits': split_results, 'paired_comparisons': paired,
               'grip1_reference_rows': baseline_rows,
               'protocol': {'eval_scored_once_after_dev_selection': True, 'eval_not_used_for_fit_or_selection': True,
                            'seed': SEED, 'bootstrap_reps': int(bootstrap_reps),
                            'optional_encoder_sources': {'JP1': 'included' if 'G-f' in models else 'unavailable',
                                                         'LT1': 'not included: smoke only, no trained E checkpoint'},
                            'G-c_grid': {'lambdas': [.03, .1, .3, 1.0], 'learning_rates': [1e-4, 3e-4],
                                         'max_epochs': [4, 8], 'batch_size': 128}}}
    (output / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    np.savez_compressed(output / 'predictions.npz', **{f'{m}_{s}_logits': v for m, d in prediction_logits.items() for s, v in d.items()})
    report = ['# GH1 production', '', '| Split | Row | Grip MSE all | Grip MSE hard | Active7 with P5 arm | Active7 with E-VLM+H arm |',
              '|---|---|---:|---:|---:|---:|']
    for split_name in ('learned', 'unseen'):
        for label in models:
            item = split_results[split_name][label]
            report.append(f"| {split_name} | {label} | {item['all']['gripper_contribution_mse_sign']:.6f} | "
                          f"{item['hard']['gripper_contribution_mse_sign']:.6f} | "
                          f"{item['active7_by_arm']['arm_p5_top1']['active7_mse']:.6f} | "
                          f"{item['active7_by_arm']['arm_e_vlm_top1']['active7_mse']:.6f} |")
    report.extend(['', '## G-c retrieval effect (diagnostic; Dev-only selection)', '',
                   '| Split | Joint E top-1 arm6 MSE all/hard | Frozen E-VLM+H top-1 arm6 MSE all/hard | Δ all |',
                   '|---|---:|---:|---:|'])
    for split_name in ('learned', 'unseen'):
        item = split_results[split_name]['G-c']
        report.append(f"| {split_name} | {item['joint_encoder_arm6_top1_mse']:.6f} / "
                      f"{item['joint_encoder_arm6_top1_hard_mse']:.6f} | "
                      f"{item['frozen_e_vlm_h_arm6_top1_mse']:.6f} / "
                      f"{item['frozen_e_vlm_h_arm6_top1_hard_mse']:.6f} | "
                      f"{item['joint_minus_frozen_e_vlm_h_arm6_mse']:+.6f} |")
    report.extend(['', '## GRIP1 reference rows (unseen Eval)', '',
                   '| Reference | Grip MSE all | Grip MSE hard | Active7 all (saved GRIP1 arm) |', '|---|---:|---:|---:|'])
    ref_names = [('G3 π0.5', 'G3_pi05_gripper'), ('G4 classifier', 'G4_classifier'),
                 ('G1 state initialized hold', 'G1_state_initialized_hold'),
                 ('G2 previous hold', 'G2_previous_hold'), ('G0 stored sign', 'G0_unit_gripper')]
    for title, key in ref_names:
        row = baseline_rows[key]
        report.append(f"| {title} | {row['all']['grip_mse_contribution']:.6f} | "
                      f"{row['hard']['grip_mse_contribution']:.6f} | {row['all']['active7_mse']:.6f} |")
    report.extend(['', 'Settings selected by Dev only; Eval was scored once after selection. Paired intervals use 5,000 episode bootstrap replicates.',
                   'G-c Dev grid: λ={0.03, 0.1, 0.3, 1.0}, lr={1e-4, 3e-4}, max epoch={4, 8}, batch=128.'])
    (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    progress('production_complete', results=str(output / 'results.json'), report=str(output / 'report.md'))
    return results


def run_production05(output: Path, *, device='cuda:0', num_threads=8, bootstrap_reps=5000):
    """Fair GH1 rerun: causal executed-action history, demo/task heldout Dev."""
    from . import key_shared_latent as key2
    from . import latent_action_units_la2 as la2
    output=Path(output)
    if output.exists():
        allowed={'progress.log','l90x_features'}
        allowed.update({f'{label}_head.pt' for label in ('g-g_history_state','g-a','g-b','g-e','g-f')})
        unexpected=[p.name for p in output.iterdir() if p.name not in allowed
                    and not p.name.startswith(('g_c_joint_e','g_c_head','g-g_history_state_head_attempt',
                                              'g-a_head_attempt','g-b_head_attempt','g-e_head_attempt','g-f_head_attempt'))]
        if unexpected: raise FileExistsError(f'refusing to overwrite GH1 production output: {output}: {unexpected}')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    torch.set_num_threads(int(num_threads))
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.12,torch.device(device))
    output.mkdir(parents=True,exist_ok=True); started=time.monotonic(); log=output/'progress.log'
    def progress(stage,**data):
        with log.open('a') as f: f.write(json.dumps({'seconds':round(time.monotonic()-started,2),'stage':stage,**data},allow_nan=False)+'\n'); f.flush()
    progress('production05_start',device=device,threads=int(num_threads),seed=SEED)
    parts,e_a,fit_bank,fit_bank_input=load_key2_frozen_e_parts(progress_callback=progress)
    all_memory=build_all_memory_training_rows(learned_parts=parts['learned'],progress_callback=progress)
    pairs=sorted(set(zip(all_memory['task'].astype(int),all_memory['episode'].astype(int))))
    rng=np.random.default_rng(SEED); held=set()
    for task in sorted(set(t for t,_ in pairs)):
        eps=np.asarray([e for t,e in pairs if t==task],np.int64); rng.shuffle(eps)
        held.update((task,int(e)) for e in eps[:max(1,int(round(.2*len(eps))))])
    mem_dev_mask=np.asarray([(int(t),int(e)) in held for t,e in zip(all_memory['task'],all_memory['episode'])])
    mem_train_mask=~mem_dev_mask
    if not mem_dev_mask.any() or mem_train_mask.sum()==0: raise AssertionError('empty memory holdout split')
    def subset_rows(row,mask): return {k:np.asarray(v)[mask] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(mask) else v for k,v in row.items()}
    train_raw=subset_rows(all_memory,mem_train_mask); mem_dev=subset_rows(all_memory,mem_dev_mask)
    train_raw=attach_all_memory_features(train_raw,progress_callback=progress)
    mem_dev=attach_all_memory_features(mem_dev,progress_callback=progress)
    # Task-disjoint L90X split; LA2's manifest assertion is run on the source and
    # again for the selected rows. Dev/Eval tasks never enter either side.
    l90x_rows=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/l90x_rows')
    manifest=json.loads((l90x_rows/'manifest.json').read_text()); la2.assert_l90x_manifest(manifest)
    l90x_feature_dir=output/'l90x_features'; l90x_feature_dir.mkdir(parents=True,exist_ok=True)
    l90x_pool=l90x_feature_dir/'l90x_pooled.npy'
    if not (l90x_feature_dir/'extract_complete.json').is_file():
        progress('l90x_prefix_extraction_start',device=device)
        la2.extract_l90x_pooled(l90x_rows,l90x_feature_dir,batch_size=2,device=device,memory_fraction=.08)
        progress('l90x_prefix_extraction_complete')
    else:
        cached=np.load(l90x_pool,mmap_mode='r',allow_pickle=False)
        if len(cached)<int(manifest['rows']): raise AssertionError('L90X pooled-prefix cache is incomplete')
        progress('l90x_prefix_cache_reused',rows=len(cached))
    task_array=np.load(l90x_rows/'task.npy',mmap_mode='r',allow_pickle=False)
    available=sorted(map(int,np.unique(task_array))); rng=np.random.default_rng(SEED+90)
    held_tasks=set(rng.choice(available,size=max(1,len(available)//5),replace=False).tolist())
    lval_ids=np.flatnonzero(np.isin(task_array,list(held_tasks))); ltrain_ids=np.flatnonzero(~np.isin(task_array,list(held_tasks)))
    if not lval_ids.size or not ltrain_ids.size: raise AssertionError('empty L90X task split')
    progress('l90x_encode_start',train_rows=int(len(ltrain_ids)),dev_rows=int(len(lval_ids)),held_tasks=sorted(held_tasks))
    l90x_train=encode_l90x_frozen_e(e_a,l90x_rows,l90x_pool,row_indices=ltrain_ids,device=device,progress_callback=progress)
    l90x_dev=encode_l90x_frozen_e(e_a,l90x_rows,l90x_pool,row_indices=lval_ids,device=device,progress_callback=progress)
    def raw_view(rows):
        return {'latent_tokens':rows['latent_tokens'],'gripper_state':rows['gripper_state'],
                'history_action':rows['history_action'],'history_valid':rows['history_valid'],
                'target':rows['target'],'key':rows.get('key'), 'task':rows.get('task'), 'episode':rows.get('episode')}
    # Produce frozen E representations for memory train/Dev and L90X train/Dev.
    a_enc=encode_frozen_key2_e(e_a,{'train':train_raw,'dev':mem_dev},mode='e_vlm_h',device=device,progress_callback=progress)
    train_a,dev_a=a_enc['train'],a_enc['dev']
    def concat(a,b):
        return {k:np.concatenate([np.asarray(a[k]),np.asarray(b[k])],axis=0) for k in
                ('latent_tokens','gripper_state','history_action','history_valid','target')}
    train_e=concat(train_a,l90x_train); dev_e=concat(dev_a,l90x_dev)
    dev_vlm=dev_e
    # G-g gate: include the exact G1 history/state policy as the zero-update
    # candidate, and also train a supervised history/state-only MLP on the same
    # memory partition. Dev selects among learned checkpoints and G1.
    gs_train={'gripper_state':train_a['gripper_state'],'history_action':train_a['history_action'],
              'history_valid':train_a['history_valid'],'target':train_a['target']}
    gs_dev={'gripper_state':dev_vlm['gripper_state'],'history_action':dev_vlm['history_action'],
            'history_valid':dev_vlm['history_valid'],'target':dev_vlm['target']}
    gs_model,gs_cfg,gs_grid=fit_fair_grid(gs_train,gs_dev,latent=False,hidden_dims=(16,32),
        learning_rates=(3e-4,),epochs_grid=(8,16),positive_weights=(.5,1.,2.),device=device,
        num_threads=num_threads,progress_callback=progress)
    g1_dev=_g1_history_logits(gs_dev); g1_dev_mse=float(gripper_contribution_mse(np.where(g1_dev>=0,1.,-1.),gs_dev['target']).mean())
    learned_dev=predict_fair(gs_model,gs_dev,latent=False,device=device)
    if g1_dev_mse <= gs_cfg['dev_gripper_mse']:
        selected_gate={'kind':'G1_history_state_rule','dev_gripper_mse':g1_dev_mse,'trained_candidate':gs_cfg}
        gate_model=None
    else:
        selected_gate={'kind':'trained_history_state_mlp',**gs_cfg}; gate_model=gs_model
    # Eval appears only after all selections below have been completed.
    # The exact input records are already assembled by the KEY2 query builder;
    # only the scoring/encoding pass is deferred until selection is frozen.
    # Encode validation rows for each available E, using identical history fields.
    configs=dict(hidden_dims=(64,),learning_rates=(3e-4,),epochs_grid=(4,8),positive_weights=(.5,1.,2.))
    selection={}; models={}; encoders={'G-a':(e_a,'e_vlm_h'),'G-e':(e_a,'e_vlm_h')}
    # Frozen E-VLM+H.
    models['G-a'],cfg,grid=fit_fair_grid(train_a,dev_vlm,latent=True,device=device,num_threads=num_threads,
        progress_callback=progress,**configs); selection['G-a']={'config':dict(cfg),'grid':grid}
    # G-e uses only training L90X tasks; its Dev selection includes held-out
    # memory episodes plus held-out L90X tasks.
    models['G-e'],cfg,grid=fit_fair_grid(train_e,dev_e,latent=True,device=device,num_threads=num_threads,
        progress_callback=progress,**configs); selection['G-e']={'config':dict(cfg),'grid':grid}
    # G-b is restricted to exact cached V-JEPA rows; no missing latent values are synthesized.
    vj_cfg=json.loads((KEY2_ROOT/'selected_configs.json').read_text())['selected']['e_vj_h']; vj_path=Path(vj_cfg['checkpoint'])
    vj_store=key2.ExactVJEPAFeatureStore(KEY2_ROOT.parent/'vjepa',KEY2_ROOT/'vjepa/libero90_eval')
    def supported(rows):
        avail=set()
        for mapping,_,_ in vj_store.maps['libero40']: avail.update(mapping)
        mask=np.asarray([tuple(map(int,k)) in avail for k in rows['key']],bool)
        out={k:np.asarray(v)[mask] for k,v in rows.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(mask)}
        out['vjepa_tokens'],out['vjepa_groups']=vj_store.get(out['key'],domain='libero40')
        return out
    bt,bv=supported(train_raw),supported(mem_dev)
    b_enc=encode_frozen_key2_e(vj_path,{'train':bt,'dev':bv},mode='e_vj_h',device=device,progress_callback=progress)
    models['G-b'],cfg,grid=fit_fair_grid(b_enc['train'],b_enc['dev'],latent=True,device=device,num_threads=num_threads,
        progress_callback=progress,**configs); selection['G-b']={'config':dict(cfg),'grid':grid}
    encoders['G-b']=(vj_path,'e_vj_h')
    # G-f uses the selected JP1 E if present.
    jp1=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-pred-01/production/selected.json')
    if not jp1.is_file(): raise FileNotFoundError(f'JP1 selected E unavailable: {jp1}')
    jp1_checkpoint=Path(json.loads(jp1.read_text())['a_e_vlm_h']['checkpoint'])
    jf=encode_frozen_key2_e(jp1_checkpoint,{'train':train_raw,'dev':mem_dev},mode='e_vlm_h',device=device,progress_callback=progress)
    jf_dev=concat(jf['dev'],l90x_dev)
    models['G-f'],cfg,grid=fit_fair_grid(jf['train'],jf_dev,latent=True,device=device,num_threads=num_threads,
        progress_callback=progress,**configs); selection['G-f']={'config':dict(cfg),'grid':grid}; encoders['G-f']=(jp1_checkpoint,'e_vlm_h')
    # G-c joint retrieval+grip update. Use memory heldout rows for retrieval Dev.
    joint_source=parts['train']
    joint_keep=np.asarray([(int(t),int(e)) not in held for t,e in zip(joint_source['task'],joint_source['episode'])],bool)
    joint_train=subset_rows(joint_source,joint_keep)
    fit_pairs=np.asarray([(int(t),int(e)) not in held for t,e in zip(fit_bank['task'],fit_bank['episode'])],bool)
    bank={k:np.asarray(v)[fit_pairs] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(fit_pairs) else v for k,v in fit_bank.items()}
    bank_input={k:np.asarray(v)[fit_pairs] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(fit_pairs) else v for k,v in fit_bank_input.items()}
    if len(joint_train['key'])!=len(bank['length']):
        raise AssertionError(f'G-c query/candidate alignment mismatch after episode holdout: {len(joint_train["key"])} != {len(bank["length"])}')
    progress('g_c_joint_fit_start',train_rows=len(joint_train['key']),gripper_fit_all_memory_rows=len(train_raw['key']),heldout_memory_rows=len(mem_dev['key']))
    cpath=output/'g_c_joint_e.pt';chead_path=output/'g_c_head.pt'
    if cpath.is_file() and chead_path.is_file():
        # Reuse the completed G-c fit from the previous production-05 attempt.
        # Its heldout split, grid and input contract are identical; this keeps a
        # downstream report/assertion correction from repeating the expensive grid.
        history=[json.loads(x) for x in (output/'progress.log').read_text().splitlines() if x.strip()]
        starts=[i for i,x in enumerate(history) if x.get('stage')=='production05_start']
        completed=[]
        for j in range(len(starts)-1):
            segment=history[starts[j]:starts[j+1]]
            events=[x for x in segment if x.get('stage')=='joint_grid_config_complete']
            if len(events)==18: completed.append(events)
        cfg_events=completed[-1] if completed else []
        if len(cfg_events)!=18: raise AssertionError(f'cannot verify reusable G-c grid ({len(cfg_events)} configs)')
        grid_c=[]
        for i,event in enumerate(cfg_events):
            grid_c.append({'lambda':float(event['lambda_']),'learning_rate':float(event['learning_rate']),
                'positive_weight':(.5,1.,2.)[(i//2)%3],'max_epochs':int(event['max_epochs']),
                'selected_epoch':int(event['selected_epoch']),'dev_gripper_mse':float(event['dev_gripper_mse'])})
        cfg_c=dev_select(grid_c)
        if not np.isclose(cfg_c['dev_gripper_mse'],.06865696594551296,atol=1e-8):
            raise AssertionError('reused G-c Dev selection does not match the completed production-05 grid')
        model_c=key2._new_model('e_vlm_h').to(device);model_c.load_state_dict(torch.load(cpath,map_location=device,weights_only=True))
        head_c=LatentGripperHead(256,hidden_dim=64,heads=4).to(device)
        head_c.load_state_dict(torch.load(chead_path,map_location=device,weights_only=True))
        selection['G-c']={'config':dict(cfg_c),'grid':grid_c,'reused_completed_attempt':True}
        progress('g_c_checkpoint_reused',checkpoint=str(cpath),selected=cfg_c)
    else:
        model_c,head_c,cfg_c,grid_c=fit_joint_e_grid(e_a,bank,bank_input,joint_train,mem_dev,
            lambdas=(.03,.1,.3),learning_rates=(1e-4,),max_epochs=(1,2),device=device,batch_size=128,
            num_threads=num_threads,progress_callback=progress)
        selection['G-c']={'config':dict(cfg_c),'grid':grid_c}
        torch.save(model_c.state_dict(),cpath);torch.save(head_c.state_dict(),chead_path)
    models['G-c']=head_c
    encoders['G-c']=(cpath,'e_vlm_h')
    # Frozen settings/checkpoints are now fixed. Score locked Eval once.
    eval_parts={n:parts[n] for n in ('learned','unseen')}
    eval_lat={}
    for label,(checkpoint,mode) in encoders.items():
        eval_lat[label]=encode_frozen_key2_e(checkpoint,eval_parts,mode=mode,device=device,progress_callback=progress)
    # Re-encode G-c with its fitted encoder rather than the saved-path wrapper.
    for sn,part in eval_parts.items():
        raw={sn:part}
        z=[]; enc=key2._new_model('e_vlm_h').to(device); enc.load_state_dict(model_c.state_dict()); enc.eval()
        with torch.no_grad():
            for lo in range(0,len(part['key']),32):
                ids=np.arange(lo,min(lo+32,len(part['key']))); batch=key2._pack_part(part,ids)
                z.append(key2._model_batch_encode(enc,'e_vlm_h',batch,device).cpu().numpy())
        eval_lat['G-c'][sn]['latent_tokens']=np.concatenate(z)
    pred={}; split_results={}; g1gate={}; reference={}
    for sn,part in eval_parts.items():
        y=np.asarray(part['actions'],np.float32)[:,:,6]; split_out={}
        statepart={'gripper_state':np.asarray(part['state'],np.float32)[:,6:8],
                   'history_action':np.asarray(part['history'],np.float32)[:,:,6],
                   'history_valid':np.asarray(part['history_mask'],bool),'target':y}
        if selected_gate['kind']=='G1_history_state_rule': logits_gate=_g1_history_logits(statepart)
        else: logits_gate=predict_fair(gate_model,statepart,latent=False,device=device)
        g1gate[sn]=logits_gate
        statepart.update({'task':part['task'],'episode':part['episode'],'hard':part['hard']})
        gate_scored=evaluate_split(logits_gate,statepart,pi05_actions=np.asarray(part['pi05'],np.float32)[:,:,6],
                                   bootstrap_reps=bootstrap_reps)
        gate_sign=np.where(logits_gate>=0,1.,-1.)
        gate_scored['active7_by_arm']={armkey:active7_with_gripper(part[armkey],part['actions'],gate_sign)
                                      for armkey in ('arm_p5_top1','arm_e_vlm_top1')}
        split_out['History+state']=gate_scored
        for label,model in models.items():
            logits=(predict_fair(model,eval_lat[label][sn],latent=True,device=device) if label!='G-c' else
                _predict_joint_head(model_c,model,part,device))
            pred[(label,sn)]=logits
            scored=evaluate_split(logits,{'target':y,'task':part['task'],'episode':part['episode'],'hard':part['hard']},
                pi05_actions=np.asarray(part['pi05'],np.float32)[:,:,6],bootstrap_reps=bootstrap_reps)
            arms={}; signs=np.where(logits>=0,1.,-1.)
            for armkey in ('arm_p5_top1','arm_e_vlm_top1'):
                arms[armkey]=active7_with_gripper(part[armkey],part['actions'],signs)
            scored['active7_by_arm']=arms;split_out[label]=scored
        split_results[sn]=split_out
    grip_ref=json.loads(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/gripper-split-01/attempt-08/results.json').read_text())['D_gripper_decisions']['eval']
    hold=gripper_hold_reference(parts['unseen'])
    for name,val in hold.items():
        key='G1_state_initialized_hold' if name.startswith('G1') else 'G2_previous_hold'
        got=float(gripper_contribution_mse(val,parts['unseen']['actions'][:,:,6]).mean())
        expected=float(grip_ref[key]['all']['grip_mse_contribution'])
        if not np.isclose(got,expected,atol=1e-8): raise AssertionError(f'{name} reference mismatch')
        reference[key]=float(got)
    gate_eval=float(split_results['unseen']['History+state']['all']['gripper_contribution_mse_sign'])
    g1_sign_mse=float(gripper_contribution_mse(np.where(hold['G1_state_initialized_hold']>=0,1.,-1.),
                                                parts['unseen']['actions'][:,:,6]).mean())
    reference['G1_state_initialized_hold_sign']=g1_sign_mse
    if selected_gate['kind']=='G1_history_state_rule' and not np.isclose(gate_eval,g1_sign_mse,atol=1e-12):
        raise AssertionError('history+state G1 policy differs under the same sign-scoring rule')
    if gate_eval>g1_sign_mse+1e-8:
        raise AssertionError(f'history+state gate {gate_eval:.10f} exceeded matched-sign G1 baseline {g1_sign_mse:.10f}')
    reference['pi05']=float(gripper_contribution_mse(parts['unseen']['pi05'][:,:,6],parts['unseen']['actions'][:,:,6]).mean())
    reference['classifier']=float(grip_ref['G4_classifier']['all']['grip_mse_contribution'])
    reference['stored']=float(grip_ref['G0_unit_gripper']['all']['grip_mse_contribution'])
    # Episode paired comparisons use the locked GRIP1 bootstrap implementation.
    grip=__import__(f'{__package__}.gripper_split',fromlist=['episode_codes','paired_episode_bootstrap'])
    codes=grip.episode_codes(parts['unseen']['task'],parts['unseen']['episode']); target=parts['unseen']['actions'][:,:,6]
    per={label:gripper_contribution_mse(np.where(pred[(label,'unseen')]>=0,1.,-1.),target) for label in models}
    per['History+state']=gripper_contribution_mse(np.where(g1gate['unseen']>=0,1.,-1.),target)
    pirow=gripper_contribution_mse(parts['unseen']['pi05'][:,:,6],target)
    paired={'head_minus_pi05':{label:grip.paired_episode_bootstrap(per[label]-pirow,codes,reps=bootstrap_reps,seed=4101) for label in models},
            'G-e_minus_G-a':grip.paired_episode_bootstrap(per['G-e']-per['G-a'],codes,reps=bootstrap_reps,seed=4102),
            'G-c_minus_G-a':grip.paired_episode_bootstrap(per['G-c']-per['G-a'],codes,reps=bootstrap_reps,seed=4103)}
    results={'status':'complete','seed':SEED,'selected':selection,'history_state_gate':{'selected':selected_gate,
        'unseen_mse':split_results['unseen']['History+state']['all']['gripper_contribution_mse_sign'],
        'reference_g1_continuous':reference['G1_state_initialized_hold'],
        'reference_g1_sign':reference['G1_state_initialized_hold_sign']},'memory_split':{'source_windows':6583,
        'fit_rows':int(mem_train_mask.sum()),'heldout_rows':int(mem_dev_mask.sum()),'fit_episodes':len(set(zip(train_raw['task'].tolist(),train_raw['episode'].tolist()))),
        'heldout_episodes':len(held),'learned_eval_episode_overlap':0},'l90x_split':{'fit_rows':int(len(ltrain_ids)),
        'heldout_rows':int(len(lval_ids)),'heldout_tasks':sorted(held_tasks)},'vjepa_exact_train_rows':int(len(b_enc['train']['target'])),
        'vjepa_exact_dev_rows':int(len(bv['key'])),'reference_rows':reference,'splits':split_results,'paired':paired,
        'protocol':{'eval_scored_once_after_dev_selection':True,'dev_memory_episodes_heldout':True,
        'l90x_tasks_heldout':True,'positive_weight_selected_on_dev':True,'input_features':['state[6:8]',
        'previous executed gripper actions[10]','history valid flags[10]','latent E tokens when applicable'],
        'pi05_is_model_input':False}}
    torch.save(gs_model.state_dict(),output/'g-g_history_state_head.pt')
    for label,model in models.items():
        if label!='G-c':
            base=output/f'{label.lower()}_head.pt';path=base;suffix=1
            while path.exists(): suffix+=1;path=output/f'{label.lower()}_head_attempt{suffix:02d}.pt'
            torch.save(model.state_dict(),path)
    (output/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(output/'predictions.npz',**{f'{m}_{s}':v for (m,s),v in pred.items()})
    unseen=split_results['unseen']; report=['| Split | Row | Grip MSE all / hard | Active7 P5 arm | Active7 E-VLM+H arm |','|---|---|---:|---:|---:|']
    for sn in ('learned','unseen'):
        for label,item in split_results[sn].items():
            hard=item['hard']; hardm=hard['gripper_contribution_mse_sign'] if hard else float('nan')
            if label=='History+state': a1=a2=float('nan')
            else: a1=item['active7_by_arm']['arm_p5_top1']['active7_mse'];a2=item['active7_by_arm']['arm_e_vlm_top1']['active7_mse']
            report.append(f"| {sn} | {label} | {item['all']['gripper_contribution_mse_sign']:.6f} / {hardm:.6f} | {a1:.6f} | {a2:.6f} |")
    report.extend(['| unseen reference | π0.5 | {0:.6f} | {1:.6f} | — | — |'.format(reference['pi05'],grip_ref['G3_pi05_gripper']['hard']['grip_mse_contribution']),
        '| unseen reference | classifier | {0:.6f} | {1:.6f} | — | — |'.format(reference['classifier'],grip_ref['G4_classifier']['hard']['grip_mse_contribution']),
        '| unseen reference | G1 hold | {0:.6f} | {1:.6f} | — | — |'.format(reference['G1_state_initialized_hold'],grip_ref['G1_state_initialized_hold']['hard']['grip_mse_contribution']),
        '| unseen reference | G2 hold | {0:.6f} | {1:.6f} | — | — |'.format(reference['G2_previous_hold'],grip_ref['G2_previous_hold']['hard']['grip_mse_contribution']),
        '| unseen reference | saved value | {0:.6f} | {1:.6f} | — | — |'.format(reference['stored'],grip_ref['G0_unit_gripper']['hard']['grip_mse_contribution']),
        '','### production-03/04 から何を変えたか','','- 全行へ直前10実行アクションと有効フラグを追加し、G1と同じ因果履歴を使った。',
        '- Dev task 6/7を選択に使わず、学習・Evalに使わないLIBERO-40 episodeを保留し、L90Xもタスク単位で保留した。',
        '- BCE正例重みを新Devで選んだ。前回の旧学習は1 epoch 1更新だったためproduction-03/04の潜在ヘッド数値は無効のまま。',
        '',f"π0.5={reference['pi05']:.6f}、classifier={reference['classifier']:.6f}、G1={reference['G1_state_initialized_hold']:.6f}、G2={reference['G2_previous_hold']:.6f}、保存値={reference['stored']:.6f}。",
        f"history+state gate: {selected_gate['kind']}、unseen {results['history_state_gate']['unseen_mse']:.6f}。",
        '',f"paired episode bootstrap ({bootstrap_reps:,} reps): {json.dumps(paired,ensure_ascii=False)}",''])
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    progress('production05_complete',results=str(output/'results.json'))
    return results


def _predict_joint_head(encoder,head,part,device):
    from . import key_shared_latent as key2
    encoder=encoder.to(device).eval(); head=head.to(device).eval(); pieces=[]
    with torch.no_grad():
        for lo in range(0,len(part['key']),32):
            ids=np.arange(lo,min(lo+32,len(part['key']))); batch=key2._pack_part(part,ids)
            z=key2._model_batch_encode(encoder,'e_vlm_h',batch,device)
            state=torch.as_tensor(np.asarray(part['state'])[ids,6:8],dtype=torch.float32,device=device)
            hist=torch.as_tensor(np.asarray(part['history'])[ids,:,6],dtype=torch.float32,device=device)
            valid=torch.as_tensor(np.asarray(part['history_mask'])[ids],dtype=torch.bool,device=device)
            pieces.append(head(z,state,hist,valid)['logits'].cpu().numpy())
    return np.concatenate(pieces)


def smoke(output: Path) -> dict[str, Any]:
    """Exercise training, Dev selection, and evaluation on deterministic toy rows."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing GH1 output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=(48, 5, 16)).astype(np.float32)
    state = rng.normal(size=(48, 2)).astype(np.float32)
    y = np.where((latent[:, 0, 0, None] + state[:, 0, None] > 0), 1., -1.).astype(np.float32)
    y = np.repeat(y, HORIZON, axis=1)
    train = {'latent_tokens': latent[:32], 'gripper_state': state[:32], 'target': y[:32]}
    dev = {'latent_tokens': latent[32:40], 'gripper_state': state[32:40], 'target': y[32:40]}
    eval_data = {'latent_tokens': latent[40:], 'gripper_state': state[40:], 'target': y[40:]}
    records = []
    for kind in ('mlp', 'cross_attention'):
        model = (LatentGripperMLP(16, hidden_dim=16) if kind == 'mlp' else
                 LatentGripperHead(16, hidden_dim=16, heads=4))
        model, record = fit_head(model, train, dev,
                                 config=HeadConfig(kind, hidden_dim=16, epochs=2, learning_rate=1e-2),
                                 device='cpu', num_threads=1)
        records.append({'kind': kind, 'dev_gripper_mse': record['dev_gripper_mse'],
                        'dev_bce_at_selected_checkpoint': record['dev_bce_at_selected_checkpoint'], 'model': model})
    selected = dev_select([{k: v for k, v in r.items() if k != 'model'} for r in records])
    model = next(r['model'] for r in records if r['kind'] == selected['kind'])
    with torch.no_grad(): logits = model(torch.tensor(eval_data['latent_tokens']), torch.tensor(eval_data['gripper_state']))['logits'].numpy()
    payload = {'status': 'smoke_only', 'seed': SEED, 'rows': {'train': len(train['target']), 'dev': len(dev['target']), 'eval': len(eval_data['target'])},
               'selected_by': 'Dev gripper-contribution MSE only', 'selected': selected,
               'smoke_eval': evaluate_logits(logits, eval_data['target']),
               'production_evaluation': 'not run; pending review'}
    (output / 'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'report.md').write_text('# GH1 smoke\n\n合成データ上で head fit、Dev 選択、Eval 採点を通過。これは本番データの結果ではない。\n\n'
                                     f"- 選択: {selected['kind']}（Dev MSE {selected['dev_gripper_mse']:.6f}）\n"
                                     f"- smoke Eval 符号化 MSE: {payload['smoke_eval']['gripper_contribution_mse_sign']:.6f}\n"
                                     '- 本番 LIBERO 評価: 未実行（レビュー後）\n', encoding='utf-8')
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--data-smoke', action='store_true', help='run frozen E on actual KEY2-F LIBERO-40 memory and Dev only')
    p.add_argument('--production', action='store_true', help='run the locked GH1 matrix after review approval')
    p.add_argument('--production05', action='store_true', help='run fair held-out GH1 production-05')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if args.smoke:
        print(json.dumps(smoke(args.output or OUT / 'synthetic-smoke'), indent=2, ensure_ascii=False))
    elif args.data_smoke:
        print(json.dumps(real_data_smoke(args.output or OUT / 'real-data-smoke', checkpoint=args.checkpoint, device=args.device or 'cpu'),
                         indent=2, ensure_ascii=False))
    elif args.production05:
        production_output=args.output or OUT/'production-05'
        try:
            print(json.dumps(run_production05(production_output,device=args.device or 'cuda:0',
                num_threads=int(os.environ.get('OMP_NUM_THREADS','8'))),indent=2,ensure_ascii=False))
        except BaseException as exc:
            progress_path=Path(production_output)/'progress.log'
            if progress_path.parent.exists():
                with progress_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'stage':'production05_exception','error_type':type(exc).__name__,
                                        'error':str(exc)},ensure_ascii=False)+'\n');f.flush()
            raise
    elif args.production:
        production_output = args.output or OUT / 'production-01'
        try:
            print(json.dumps(run_production(production_output, device=args.device or 'cuda:0',
                                            num_threads=int(os.environ.get('OMP_NUM_THREADS', '8'))),
                             indent=2, ensure_ascii=False))
        except BaseException as exc:
            progress_path = Path(production_output) / 'progress.log'
            if progress_path.parent.exists():
                with progress_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'stage': 'production_exception', 'error_type': type(exc).__name__,
                                        'error': str(exc)}, ensure_ascii=False) + '\n')
                    f.flush()
            raise
    else:
        raise SystemExit('choose --smoke, --data-smoke, --production05, or --production')


if __name__ == '__main__':
    main()
