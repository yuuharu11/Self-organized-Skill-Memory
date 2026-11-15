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
        self.query = nn.Parameter(torch.randn(1, horizon, hidden_dim) * .02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.logit = nn.Linear(hidden_dim, 1)
        self.switch_horizon = nn.Linear(hidden_dim, 11) if auxiliary_switch else None

    def forward(self, latent_tokens: torch.Tensor, gripper_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_tokens.ndim != 3:
            raise ValueError('latent_tokens must have shape [B,K,D]')
        if gripper_state.shape != (latent_tokens.shape[0], self.state_dim):
            raise ValueError(f'gripper_state must have shape [B,{self.state_dim}]')
        memory = self.latent_proj(latent_tokens.float())
        state = self.state_proj(gripper_state.float()).unsqueeze(1)
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
        self.net = nn.Sequential(nn.Linear(latent_dim + state_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon))

    def forward(self, latent_tokens: torch.Tensor, gripper_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_tokens.ndim not in (2, 3) or gripper_state.shape != (len(latent_tokens), self.state_dim):
            raise ValueError('expected latent features [B,D] or [B,K,D] and gripper_state [B,state_dim]')
        pooled = latent_tokens.float().mean(dim=1) if latent_tokens.ndim == 3 else latent_tokens.float()
        return {'logits': self.net(torch.cat((pooled, gripper_state.float()), dim=-1))}


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
                                gripper_target: torch.Tensor, *, gripper_weight: float) -> torch.Tensor:
    """G-c objective term: L_ret + λ L_grip, with BCE on future gripper signs."""
    if gripper_logits.shape != gripper_target.shape or gripper_logits.ndim != 2 or gripper_logits.shape[1] != HORIZON:
        raise ValueError('gripper logits and targets must have matching [B,10] shapes')
    if float(gripper_weight) < 0:
        raise ValueError('gripper_weight must be non-negative')
    grip_loss = F.binary_cross_entropy_with_logits(gripper_logits, (gripper_target > 0).float())
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
             num_threads: int | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Fit one head with BCE and retain its minimum Dev BCE checkpoint."""
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
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_dev_mse, best_state, best_dev_bce = float('inf'), None, float('inf')
    for _ in range(int(config.epochs)):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(x, s)
        loss = F.binary_cross_entropy_with_logits(output['logits'], y)
        if config.auxiliary_switch and 'switch_horizon_logits' in output:
            labels = torch.as_tensor(switch_horizon_labels(train['target']), dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(output['switch_horizon_logits'], labels)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_logits = model(dx, ds)['logits']
            dev_bce = float(F.binary_cross_entropy_with_logits(dev_logits, dy).item())
            dev_mse = float(gripper_contribution_mse(
                np.where(dev_logits.detach().cpu().numpy() >= 0, 1., -1.), dev['target']).mean())
        if dev_mse < best_dev_mse:
            best_dev_mse, best_dev_bce = dev_mse, dev_bce
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        dev_logits = model(dx, ds)['logits'].cpu().numpy()
    return model, {'dev_gripper_mse': best_dev_mse, 'dev_bce_at_selected_checkpoint': best_dev_bce,
                   'dev_metrics': evaluate_logits(dev_logits, dev['target']),
                   'epochs': int(config.epochs), 'learning_rate': float(config.learning_rate),
                   'seed': int(seed), 'config': config.name}


def encode_frozen_key2_e(checkpoint: Path, parts: Mapping[str, Mapping[str, np.ndarray]], *,
                         mode: str = 'e_vlm_h', device: str = 'cpu', batch_size: int = 32) -> dict[str, dict[str, np.ndarray]]:
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
            z_all = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 5, 256), np.float32)
            state2 = np.asarray(part['state'], np.float32)[:, 6:8]
            record = {'latent_tokens': z_all, 'gripper_state': state2}
            validate_model_inputs(record)
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
        record = {'latent_tokens': pooled.astype(np.float32), 'gripper_state': state2}
        if 'actions' in part:
            record['target'] = np.asarray(part['actions'], np.float32)[:, :, 6]
        for key in ('task', 'episode', 'hard', 'row', 'key'):
            if key in part:
                record[key] = np.asarray(part[key])
        validate_model_inputs({k: record[k] for k in ('latent_tokens', 'gripper_state')})
        result[split_name] = record
    return result


def encode_l90x_frozen_e(checkpoint: Path, row_cache: Path, pooled_features: Path, *,
                         row_limit: int | None = None, device='cpu', batch_size=32) -> dict[str, np.ndarray]:
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
    if n < 1 or pooled.shape[1:] != (2048,):
        raise ValueError(f'invalid L90X pooled-prefix features: {pooled.shape}')
    # Re-assert task IDs on the exact rows being consumed, in addition to the
    # complete source manifest assertion above.
    tasks = np.asarray(rows['task'][:n], np.int32)
    forbidden = set(DEV_TASKS) | set(EVAL_TASKS)
    if set(map(int, np.unique(tasks))) & forbidden:
        raise AssertionError('filtered L90X smoke rows contain Dev/Eval task IDs')
    part = {'key': np.stack([rows['task'][:n], rows['episode'][:n], rows['frame'][:n]], axis=1).astype(np.int32),
            'tokens': np.asarray(pooled[:n], np.float32)[:, None, :],
            'groups': np.zeros((n, 1), np.int8), 'history': np.asarray(rows['hist'][:n], np.float32),
            'history_mask': np.repeat(np.asarray(rows['has_hist'][:n], bool).reshape(n, 1), 10, axis=1),
            'state': np.asarray(rows['state'][:n, :8], np.float32),
            'actions': np.asarray(rows['actions'][:n, :10, :7], np.float32),
            'task': tasks, 'episode': np.asarray(rows['episode'][:n], np.int32)}
    latent = encode_frozen_key2_e(checkpoint, {'l90x': part}, mode='e_vlm_h', device=device,
                                  batch_size=batch_size)['l90x']
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
            for key in ('latent_tokens', 'gripper_state', 'target')}


def fit_joint_e_grid(pretrained_checkpoint: Path, fit_bank: Mapping[str, np.ndarray],
                     fit_bank_input: Mapping[str, np.ndarray], train: Mapping[str, np.ndarray],
                     dev: Mapping[str, np.ndarray], *, lambdas=(0.1, 0.3, 1.0),
                     learning_rates=(1e-4, 3e-4), max_epochs=(4, 8, 16),
                     seed=SEED, device='cpu', batch_size=8,
                     num_threads: int | None = None) -> tuple[nn.Module, nn.Module, dict[str, Any], list[dict[str, Any]]]:
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
    records: list[dict[str, Any]] = []
    models: dict[tuple[float, float, int], tuple[nn.Module, nn.Module]] = {}
    train_n = len(train['key']); bank_n = len(fit_bank['length'])
    if train_n == 0 or bank_n == 0:
        raise ValueError('joint training requires nonempty train queries and memory candidates')
    if train_n > bank_n or not np.array_equal(np.asarray(train['key']), np.asarray(fit_bank_input['key'])[:train_n]):
        raise AssertionError('memory queries must align with the prefix of the fit candidate bank')
    for lam in lambdas:
        for lr in learning_rates:
            for epochs in max_epochs:
                torch.manual_seed(int(seed)); np.random.seed(int(seed))
                encoder = key2._new_model('e_vlm_h').to(device)
                encoder.load_state_dict(initial_state)
                head = LatentGripperHead(256, hidden_dim=64, heads=4).to(device)
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                              lr=float(lr), weight_decay=1e-4)
                rng = np.random.default_rng(int(seed) + 37)
                best_mse, best_epoch, best_state = float('inf'), None, None
                for epoch in range(int(epochs)):
                    qids = np.arange(train_n); rng.shuffle(qids)
                    encoder.train(); head.train()
                    for start in range(0, train_n, int(batch_size)):
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
                        grip_logits = head(qrep, grip_state)['logits']
                        loss = retrieval_plus_gripper_loss(retrieval, grip_logits, grip_target, gripper_weight=float(lam))
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.)
                        optimizer.step(); key2._clamp_logit_scale_(encoder)
                    dev_rep, _ = key2._query_representations(encoder, 'e_vlm_h', dev, device)
                    dev_state = torch.as_tensor(np.asarray(dev['state'])[:, 6:8], dtype=torch.float32, device=device)
                    head.eval(); encoder.eval()
                    with torch.no_grad():
                        logits = head(torch.as_tensor(dev_rep, dtype=torch.float32, device=device), dev_state)['logits'].cpu().numpy()
                    dev_mse = float(gripper_contribution_mse(np.where(logits >= 0, 1., -1.), dev['actions'][:, :, 6]).mean())
                    if dev_mse < best_mse:
                        best_mse, best_epoch = dev_mse, epoch + 1
                        best_state = ({k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                                      {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                if best_state is None:
                    raise RuntimeError('joint E/head training produced no Dev checkpoint')
                record = {'lambda': float(lam), 'learning_rate': float(lr), 'max_epochs': int(epochs),
                          'selected_epoch': int(best_epoch), 'dev_gripper_mse': float(best_mse),
                          'pretrained_dev_arm6_top1_mse': baseline_arm_mse}
                records.append(record)
                fitted_encoder = key2._new_model('e_vlm_h').to(device); fitted_encoder.load_state_dict(best_state[0])
                fitted_head = LatentGripperHead(256, hidden_dim=64, heads=4).to(device); fitted_head.load_state_dict(best_state[1])
                models[(float(lam), float(lr), int(epochs))] = (fitted_encoder, fitted_head)
    selected = dev_select(records)
    model_pair = models[(selected['lambda'], selected['learning_rate'], selected['max_epochs'])]
    selected_encoder, selected_head = model_pair
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
                 num_threads: int | None = None) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
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
                                            num_threads=num_threads)
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
                             vjepa_eval_cache: Path | None = None) -> tuple[
                                 dict[str, dict[str, np.ndarray]], Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Join exact KEY2-F locked rows with frozen-E inputs and saved top-1 arms."""
    from . import key_shared_latent as key2
    selected_path = KEY2_ROOT / 'selected_configs.json'
    selected = json.loads(selected_path.read_text())
    checkpoint = Path(checkpoint or selected['selected']['e_vlm_h']['checkpoint'])
    vjepa_root = KEY2_ROOT.parent / 'vjepa'
    vjepa_eval_cache = Path(vjepa_eval_cache or KEY2_ROOT / 'vjepa/libero90_eval')
    bank, bank_input, fit_bank, fit_bank_input, train, learned, unseen, dev = key2._build_key2_queries(
        run_root=KEY2_RUN, missing_root=KEY2_MISSING, seen_stride5=KEY2_SEEN_CACHE,
        unseen_stride5=KEY2_UNSEEN_CACHE, vjepa_eval_cache=vjepa_eval_cache)
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
        # Permit only the extractor's own resumable partial directory. Any
        # other existing artifact means this invocation could overwrite a
        # checkpoint or result, so fail closed.
        unexpected = [p.name for p in output.iterdir() if p.name != 'l90x_features']
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
    parts, e_a_checkpoint, fit_bank, fit_bank_input = load_key2_frozen_e_parts()
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

    configs = {'kinds': ('mlp', 'cross_attention', 'cross_attention_aux'),
               'hidden_dims': (64, 128), 'learning_rates': (3e-4, 1e-3),
               'epochs_grid': (8, 16, 32)}
    selected: dict[str, Any] = {}
    models: dict[str, nn.Module] = {}
    enc_checkpoints: dict[str, Path] = {'G-a': e_a_checkpoint, 'G-b': Path(
        json.loads((KEY2_ROOT / 'selected_configs.json').read_text())['selected']['e_vj_h']['checkpoint'])}

    # G-a frozen KEY2-F E-VLM+H.
    enc_a = encode_frozen_key2_e(e_a_checkpoint, {'train': train, 'dev': dev}, mode='e_vlm_h', device=device)
    model, cfg, grid = fit_dev_grid(enc_a['train'], enc_a['dev'], device=device, num_threads=threads, **configs)
    models['G-a'] = model; selected['G-a'] = {'config': cfg, 'grid': grid}
    del enc_a

    # G-b frozen KEY2-F E-VJ+H.
    enc_b = encode_frozen_key2_e(enc_checkpoints['G-b'], {'train': train, 'dev': dev}, mode='e_vj_h', device=device)
    model, cfg, grid = fit_dev_grid(enc_b['train'], enc_b['dev'], device=device, num_threads=threads, **configs)
    models['G-b'] = model; selected['G-b'] = {'config': cfg, 'grid': grid}
    del enc_b

    # G-d π0.5 final-prefix mean plus current gripper state.
    pooled = pooled_pi05_features({'train': train, 'dev': dev})
    model, cfg, grid = fit_dev_grid(pooled['train'], pooled['dev'], kinds=('mlp',),
                                    hidden_dims=configs['hidden_dims'], learning_rates=configs['learning_rates'],
                                    epochs_grid=configs['epochs_grid'], device=device, num_threads=threads)
    models['G-d'] = model; selected['G-d'] = {'config': cfg, 'grid': grid}
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
        la2.extract_l90x_pooled(l90x_row_cache, l90x_feature_dir, batch_size=2,
                                device=device, memory_fraction=.08)
    l90x_data = encode_l90x_frozen_e(e_a_checkpoint, l90x_row_cache, l90x_feature_path, device=device)
    train_a = encode_frozen_key2_e(e_a_checkpoint, {'train': train}, mode='e_vlm_h', device=device)['train']
    train_e = append_l90x_training_data(train_a, l90x_data, l90x_data['manifest'])
    dev_a = encode_frozen_key2_e(e_a_checkpoint, {'dev': dev}, mode='e_vlm_h', device=device)['dev']
    model, cfg, grid = fit_dev_grid(train_e, dev_a, device=device, num_threads=threads, **configs)
    models['G-e'] = model; selected['G-e'] = {'config': cfg, 'grid': grid,
                                              'l90x_rows': int(len(l90x_data['target'])),
                                              'l90x_manifest': str(l90x_row_cache / 'manifest.json')}
    # G-e augments only the head-fitting data; its observation encoder remains
    # the same frozen KEY2-F E-VLM+H used by G-a.
    enc_checkpoints['G-e'] = e_a_checkpoint
    del train_a, train_e, dev_a, l90x_data

    # G-c joint E/head optimization; λ/lr/epoch are selected with Dev MSE.
    model_c, head_c, cfg_c, grid_c = fit_joint_e_grid(
        e_a_checkpoint, fit_bank, fit_bank_input, train, dev,
        lambdas=(.03, .1, .3, 1.0), learning_rates=(1e-4, 3e-4),
        max_epochs=(4, 8, 16), seed=SEED, device=device, batch_size=8, num_threads=threads)
    joint_encoder_path = output / 'g_c_joint_e.pt'
    torch.save(model_c.state_dict(), joint_encoder_path)
    torch.save(head_c.state_dict(), output / 'g_c_head.pt')
    models['G-c'] = head_c; enc_checkpoints['G-c'] = joint_encoder_path
    selected['G-c'] = {'config': cfg_c, 'grid': grid_c}
    for label, model0 in models.items():
        if label != 'G-c':
            torch.save(model0.state_dict(), output / f'{label.lower()}_head.pt')

    # Eval is touched only after all Dev selections and checkpoints have been saved.
    evaluation_parts = {name: parts[name] for name in ('learned', 'unseen')}
    eval_latents: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for label, mode in (('G-a', 'e_vlm_h'), ('G-b', 'e_vj_h'), ('G-c', 'e_vlm_h'), ('G-e', 'e_vlm_h')):
        eval_latents[label] = encode_frozen_key2_e(enc_checkpoints[label], evaluation_parts,
                                                   mode=mode, device=device)
    pooled_eval = pooled_pi05_features(evaluation_parts)
    prediction_logits: dict[str, dict[str, np.ndarray]] = {k: {} for k in models}
    split_results: dict[str, dict[str, Any]] = {}
    for split_name in ('learned', 'unseen'):
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
                            'seed': SEED, 'bootstrap_reps': int(bootstrap_reps)}}
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
    report.extend(['', 'Settings selected by Dev only; Eval was scored once after selection. Paired intervals use 5,000 episode bootstrap replicates.'])
    (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return results


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
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if args.smoke:
        print(json.dumps(smoke(args.output or OUT / 'synthetic-smoke'), indent=2, ensure_ascii=False))
    elif args.data_smoke:
        print(json.dumps(real_data_smoke(args.output or OUT / 'real-data-smoke', checkpoint=args.checkpoint, device=args.device or 'cpu'),
                         indent=2, ensure_ascii=False))
    elif args.production:
        print(json.dumps(run_production(args.output or OUT / 'production-01', device=args.device or 'cuda:0',
                                        num_threads=int(os.environ.get('OMP_NUM_THREADS', '8'))),
                         indent=2, ensure_ascii=False))
    else:
        raise SystemExit('choose --smoke, --data-smoke, or --production (after review approval)')


if __name__ == '__main__':
    main()
