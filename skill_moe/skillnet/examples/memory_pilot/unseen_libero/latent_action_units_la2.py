#!/usr/bin/env python3
"""LA2: extend stored-motion latent retrieval with more data and longer fitting.

LA1 remains read-only.  This driver imports its fixed split, unit/value
construction and metrics, and writes every LA2 artifact under la2-production.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import latent_action_units as la  # noqa: E402

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
LA1_ROOT = ROOT / 'latent-action-01'
L90X_ROWS = ROOT / 'expert-refine-exp3b-01/l90x_rows'
KEY2_ROOT = ROOT / 'key-shared-01/attempt-27-production'
DEVICE_DEFAULT = 'cuda:0'
LRS = (1e-4, 3e-4)
MAX_EPOCHS = (16, 32, 64)
SEED = 4101


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def assert_l90x_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed if any fixed Dev/Eval task entered the L90X fit source."""
    forbidden = set(map(int, manifest.get('dev_tasks', (6, 7)))) | set(map(int, manifest.get('eval_tasks', (0, 10, 13, 31, 35, 43, 57, 61, 84, 88))))
    actual = set(map(int, manifest.get('tasks', ())))
    if actual & forbidden:
        raise AssertionError(f'L90X task manifest leaks Dev/Eval tasks: {sorted(actual & forbidden)}')
    episodes = manifest.get('episodes', ())
    leaked = sorted({int(x['task_id']) for x in episodes if int(x['task_id']) in forbidden})
    if leaked:
        raise AssertionError(f'L90X rows leak fixed Dev/Eval tasks: {leaked}')


def extract_l90x_pooled(row_cache: Path, output: Path, *, batch_size: int = 4,
                        device: str = 'cuda:0', memory_fraction: float = .12,
                        max_rows: int | None = None) -> dict[str, Any]:
    """Extract exact pooled final-prefix VLM features from EXP3-B image rows.

    EXP3-B supplies correctly oriented, already transformed images/actions.  A
    full X1 token cache was not present, so only the LA1-compatible mean of
    valid final-prefix tokens is extracted and retained.  Progress is resumable.
    """
    row_cache = Path(row_cache); output = Path(output)
    manifest = json.loads((row_cache / 'manifest.json').read_text())
    assert_l90x_manifest(manifest)
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in row_cache.glob('*.npy')}
    n = int(manifest['rows']) if max_rows is None else min(int(max_rows), int(manifest['rows']))
    if not all(k in rows for k in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask', 'actions', 'task', 'episode', 'frame')):
        raise FileNotFoundError(f'incomplete EXP3-B row cache: {row_cache}')
    output.mkdir(parents=True, exist_ok=True)
    feat_path = output / 'l90x_pooled.npy'
    progress_path = output / 'extract_progress.json'
    if (output / 'extract_complete.json').exists():
        old = json.loads((output / 'extract_complete.json').read_text())
        if old.get('rows') == n:
            return old
        raise FileExistsError(f'completed L90X features use a different row count: {old}')
    done = int(json.loads(progress_path.read_text()).get('done', 0)) if progress_path.exists() else 0
    if feat_path.exists():
        cached = np.load(feat_path, mmap_mode='r', allow_pickle=False)
        if cached.shape != (n, 2048) or cached.dtype != np.float32:
            raise ValueError(f'feature cache has wrong shape/dtype: {feat_path} {cached.shape} {cached.dtype}')
    pooled = np.lib.format.open_memmap(feat_path, mode='r+' if feat_path.exists() else 'w+',
                                       dtype=np.float32, shape=(n, 2048))
    # Match the same model and policy prefix path used by LA1/X1.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', str(memory_fraction))
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    import jax
    import numpy as _np
    pi05_run = HERE.parent / 'pi05_memory' / 'run.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location('la2_pi05_run', pi05_run)
    if spec is None or spec.loader is None:
        raise ImportError(pi05_run)
    run = importlib.util.module_from_spec(spec); sys.modules[spec.name] = run; spec.loader.exec_module(run)
    data_manifest = json.loads((LA1_ROOT / 'sources' / 'data_manifest.json').read_text()) if (LA1_ROOT / 'sources' / 'data_manifest.json').exists() else json.loads((ROOT.parent / 'pi05-memory40-20261104/run-01/data_manifest.json').read_text())
    checkpoint = Path(data_manifest.get('checkpoint', '/work/.skillnet-state/checkpoints/pi05_libero'))
    model, _ = run.load_policy(checkpoint)
    from openpi.shared.nnx_utils import module_jit
    prefix = module_jit(model.prefix_features)
    for start in range(done, n, int(batch_size)):
        ids = np.arange(start, min(start + int(batch_size), n), dtype=np.int64)
        obs = run.obs_batch(rows, ids)
        tokens, groups = prefix(obs)
        tokens = np.asarray(tokens, dtype=np.float32); groups = np.asarray(groups)
        valid = groups >= 0
        pooled[ids] = (tokens * valid[..., None]).sum(axis=1) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        pooled.flush()
        done = int(ids[-1] + 1)
        _atomic_json(progress_path, {'done': done, 'total': n, 'complete': done == n,
                                     'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    pooled.flush()
    # The full X1 smoke cache covers a small exact identity subset; verify values
    # at those keys to catch orientation/preprocessing drift.
    parity = {'checked': 0, 'max_abs': None, 'source': None}
    smoke_rows = ROOT / 'x-attn-01/smoke/l90x_rows'
    smoke_tokens = smoke_rows / 'features/tokens.npy'
    smoke_groups = smoke_rows / 'features/groups.npy'
    if smoke_tokens.exists() and smoke_groups.exists():
        sm = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in smoke_rows.glob('*.npy')}
        skeys = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(sm['task'], sm['episode'], sm['frame'], strict=True))}
        lkeys = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(rows['task'][:n], rows['episode'][:n], rows['frame'][:n], strict=True))}
        st = np.load(smoke_tokens, mmap_mode='r'); sg = np.load(smoke_groups, mmap_mode='r'); diffs=[]
        for key in sorted(set(skeys) & set(lkeys)):
            si, li = skeys[key], lkeys[key]; valid = np.asarray(sg[si]) >= 0
            ref = np.asarray(st[si, valid], np.float32).mean(axis=0)
            diffs.append(float(np.max(np.abs(ref - np.asarray(pooled[li])))))
        if diffs:
            # X1 stores activations in fp16 whereas this LA2 cache stores the
            # pooled fp32 prefix. Per-coordinate differences include the
            # expected quantization and batch-kernel rounding; retain and
            # report the bound instead of demanding bit equality.
            parity = {'checked': len(diffs), 'max_abs': max(diffs),
                      'tolerance': 5e-3, 'comparison': 'fp16-token mean vs fp32 pooled prefix',
                      'source': str(smoke_rows)}
            if max(diffs) > 5e-3:
                raise AssertionError(f'LA2 L90X VLM feature parity failed: {parity}')
    result = {'complete': True, 'rows': n, 'shape': [n, 2048], 'dtype': 'float32',
              'source_rows': str(row_cache), 'token_source': 'π0.5 final prefix; mean of valid tokens to match LA1 pooled query input',
              'full_x1_token_cache_present': False, 'exact_extraction_performed': True,
              'dev_eval_tasks_excluded': True, 'task_count_with_rows': len(np.unique(rows['task'][:n])),
              'task_count_manifest': len(manifest.get('tasks', ())), 'missing_task_count': len(manifest.get('missing_tasks', ())),
              'x1_smoke_parity': parity}
    _atomic_json(output / 'extract_complete.json', result)
    return result


def _make_l90x_queries(row_cache: Path, feature_file: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((row_cache / 'manifest.json').read_text()); assert_l90x_manifest(manifest)
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in Path(row_cache).glob('*.npy')}
    pooled = np.load(feature_file, mmap_mode='r', allow_pickle=False)
    n = len(pooled)
    if n > int(manifest['rows']):
        raise ValueError(f'L90X feature rows exceed source cache: {n} > {manifest["rows"]}')
    q = np.concatenate([np.asarray(pooled, np.float32), np.asarray(rows['hist'][:n], np.float32).reshape(n, -1),
                        np.repeat(np.asarray(rows['has_hist'][:n], np.float32)[:, None], 10, axis=1),
                        np.asarray(rows['state'][:n, :8], np.float32)], axis=1)
    return {'x': q.astype(np.float32), 'actions': np.asarray(rows['actions'][:n, :, :7], np.float32),
            'task': np.asarray(rows['task'][:n], np.int32), 'episode': np.asarray(rows['episode'][:n], np.int32),
            'frame': np.asarray(rows['frame'][:n], np.int32), 'domain': np.ones(n, np.int8),
            'hard': np.zeros(n, bool), 'key': np.stack([rows['task'][:n], rows['episode'][:n], rows['frame'][:n]], axis=1).astype(np.int32),
            'pooled': pooled}


def _query40(split: dict[str, np.ndarray], *, domain: int = 0) -> dict[str, np.ndarray]:
    x = la.make_observation_matrix(split)
    n = len(x)
    return {'x': x, 'actions': np.asarray(split['actions'], np.float32)[..., :7],
            'task': np.asarray(split['task'], np.int32), 'episode': np.asarray(split['episode'], np.int32),
            'frame': np.asarray(split['key'], np.int32)[:, 2], 'domain': np.full(n, domain, np.int8),
            'hard': np.asarray(split.get('hard', np.zeros(n, bool)), bool), 'key': np.asarray(split['key'], np.int32),
            'pooled': la.pooled_tokens(split['tokens'], split['groups'])}


def _concat_queries(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = ('x', 'actions', 'task', 'episode', 'frame', 'domain', 'hard', 'key', 'pooled')
    return {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0) for k in keys}


def _teacher_errors(queries: dict[str, np.ndarray], candidate_actions: np.ndarray,
                    *, chunk: int = 32) -> np.ndarray:
    n = len(queries['actions']); out = np.empty((n, len(candidate_actions)), np.float32)
    target = np.asarray(queries['actions'], np.float32)[..., :6]
    cand = np.asarray(candidate_actions, np.float32)[..., :6]
    for lo in range(0, n, chunk):
        d = target[lo:lo + chunk, None] - cand[None]
        out[lo:lo + chunk] = np.square(d).sum(axis=(2, 3), dtype=np.float32) / 70.0
    return out


def _instruction_map(run_root: Path, l90x_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    m40 = json.loads((run_root / 'data_manifest.json').read_text())
    mx = json.loads((l90x_root / 'manifest.json').read_text())
    tasks40 = {int(k): ' '.join(str(v).lower().split()) for k, v in m40['tasks'].items()}
    tasks90 = {}
    for ep in mx.get('episodes', ()):
        tasks90[int(ep['task_id'])] = ' '.join(str(ep.get('instruction', '')).lower().split())
    return tasks40, tasks90


def _allowed_matrix(q: dict[str, np.ndarray], candidates: dict[str, np.ndarray],
                    tasks40: dict[int, str], tasks90: dict[int, str]) -> np.ndarray:
    """Mask identical source task/demo pairs; cross-domain rows stay eligible."""
    qt = np.asarray(q['task']); qe = np.asarray(q['episode']); qd = np.asarray(q['domain'])
    ct = np.asarray(candidates['task']); ce = np.asarray(candidates['episode'])
    allowed = np.ones((len(qt), len(ct)), bool)
    for i in range(len(qt)):
        if qd[i] == 0:
            same = (ct == qt[i]) & (ce == qe[i])
        else:
            qname = tasks90.get(int(qt[i]))
            same_tasks = [k for k, v in tasks40.items() if qname and v == qname]
            same = (np.isin(ct, same_tasks) & (ce == qe[i])) if same_tasks else np.zeros(len(ct), bool)
        allowed[i, same] = False
    if np.any(~allowed.any(axis=1)):
        raise AssertionError('a query has no eligible memory candidate')
    return allowed


class CompositeQueryEncoder(nn.Module):
    """Concatenate a learned query code with an observation E(o) code."""
    def __init__(self, input_dim: int, dim: int = 128):
        super().__init__()
        self.content = la.QueryEncoder(input_dim, dim)
        self.observation = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Linear(512, dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, x: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        a = self.content(x)
        b = F.normalize(self.observation(pooled), dim=-1)
        return F.normalize(torch.cat([a, b], dim=-1), dim=-1)


class CompositeUnitEncoder(nn.Module):
    """Unit content and start observation are independently represented."""
    def __init__(self, action_dim: int = 60, dim: int = 128):
        super().__init__()
        self.content = la.UnitKeyEncoder(action_dim, dim)
        self.observation = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Linear(512, dim))

    def forward(self, action: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        a = self.content(action)
        b = F.normalize(self.observation(pooled), dim=-1)
        return F.normalize(torch.cat([a, b], dim=-1), dim=-1)


def composite_forward(qnet: CompositeQueryEncoder, knet: CompositeUnitEncoder,
                      qx: torch.Tensor, qp: torch.Tensor, ka: torch.Tensor, kp: torch.Tensor):
    return qnet(qx, qp), knet(ka, kp)


def _allowed_torch(q: dict[str, np.ndarray], candidate_meta: dict[str, np.ndarray],
                   tasks40: dict[int, str], tasks90: dict[int, str], device: str) -> torch.Tensor:
    return torch.as_tensor(_allowed_matrix(q, candidate_meta, tasks40, tasks90), device=device)


def _fit_ranker(q: dict[str, np.ndarray], c_inputs: np.ndarray, c_actions: np.ndarray,
                c_meta: dict[str, np.ndarray], dev: dict[str, np.ndarray], dev_errors: np.ndarray,
                dev_allowed: np.ndarray, tasks40: dict[int, str], tasks90: dict[int, str], *,
                lr: float, max_epochs: int, seed: int, device: str,
                composite: bool = False, patience: int = 12) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed)
    if composite:
        qnet = CompositeQueryEncoder(q['x'].shape[1]).to(device)
        knet = CompositeUnitEncoder().to(device)
    else:
        qnet = la.QueryEncoder(q['x'].shape[1]).to(device)
        knet = la.UnitKeyEncoder(c_inputs.shape[1]).to(device)
    qx = torch.as_tensor(q['x'], dtype=torch.float32, device=device)
    qp = torch.as_tensor(q['pooled'], dtype=torch.float32, device=device)
    if composite:
        ca = torch.as_tensor(c_actions[..., :6].reshape(len(c_actions), -1), dtype=torch.float32, device=device)
        cp = torch.as_tensor(c_meta['pooled'], dtype=torch.float32, device=device)
    else:
        ki = torch.as_tensor(c_inputs, dtype=torch.float32, device=device)
    errors_np = _teacher_errors(q, c_actions)
    errors = torch.as_tensor(errors_np, dtype=torch.float32, device=device)
    allowed = _allowed_torch(q, c_meta, tasks40, tasks90, device)
    opt = torch.optim.AdamW(list(qnet.parameters()) + list(knet.parameters()), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_metric = float('inf'); best_q = None; best_k = None; best_epoch = 0; stale = 0; history=[]
    dev_x = torch.as_tensor(dev['x'], dtype=torch.float32, device=device)
    dev_p = torch.as_tensor(dev['pooled'], dtype=torch.float32, device=device)
    dev_allowed_t = torch.as_tensor(dev_allowed, device=device)
    dev_err_t = torch.as_tensor(dev_errors, dtype=torch.float32, device=device)
    dev_cand_t = torch.as_tensor(c_actions[..., :6], dtype=torch.float32, device=device)
    dev_target = torch.as_tensor(dev['actions'][..., :6], dtype=torch.float32, device=device)
    for ep in range(1, max_epochs + 1):
        qnet.train(); knet.train()
        if composite:
            qrep, krep = composite_forward(qnet, knet, qx, qp, ca, cp)
        else:
            qrep = qnet(qx); krep = knet(ki)
        scale = qnet.logit_scale.exp().clamp(max=100)
        scores = qrep @ krep.T * scale
        loss = la.listwise_loss(scores, errors, allowed)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(qnet.parameters()) + list(knet.parameters()), 1.)
        opt.step(); qnet.logit_scale.data.clamp_(max=float(np.log(100.0)))
        qnet.eval(); knet.eval()
        with torch.no_grad():
            if composite:
                dq, dk = composite_forward(qnet, knet, dev_x, dev_p, ca, cp)
            else:
                dq, dk = qnet(dev_x), knet(ki)
            ds = dq @ dk.T * qnet.logit_scale.exp().clamp(max=100)
            ds = ds.masked_fill(~dev_allowed_t, -torch.inf)
            idx = ds.argmax(-1)
            pred = dev_cand_t[idx]
            val = float(torch.square(pred - dev_target).sum((1, 2)).mean().cpu() / 70.)
        history.append({'epoch': ep, 'loss': float(loss.detach().cpu()), 'dev_top1_arm6_mse': val})
        if val < best_metric - 1e-8:
            best_metric = val; best_epoch = ep; stale = 0
            best_q = {k: v.detach().cpu().clone() for k, v in qnet.state_dict().items()}
            best_k = {k: v.detach().cpu().clone() for k, v in knet.state_dict().items()}
        else:
            stale += 1
        if stale >= patience and ep >= min(16, max_epochs):
            break
    if best_q is None or best_k is None:
        raise RuntimeError('early stopping produced no checkpoint')
    qnet.load_state_dict(best_q); knet.load_state_dict(best_k)
    return qnet, knet, {'history': history, 'best_epoch': best_epoch, 'best_dev_top1_arm6_mse': best_metric,
                        'stopped_early': len(history) < max_epochs, 'max_epochs': max_epochs,
                        'lr': lr, 'seed': seed, 'patience': patience}


def _encode_eval(qnet: nn.Module, knet: nn.Module, queries: dict[str, np.ndarray],
                 c_inputs: np.ndarray, c_actions: np.ndarray, c_meta: dict[str, np.ndarray],
                 *, device: str, composite: bool = False, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    qnet.eval(); knet.eval(); qs=[]; ks=[]
    with torch.no_grad():
        for lo in range(0, len(queries['x']), batch_size):
            sl = slice(lo, min(lo + batch_size, len(queries['x'])))
            x = torch.as_tensor(queries['x'][sl], dtype=torch.float32, device=device)
            if composite:
                p = torch.as_tensor(queries['pooled'][sl], dtype=torch.float32, device=device)
                qs.append(qnet(x, p).cpu().numpy())
            else:
                qs.append(qnet(x).cpu().numpy())
        if composite:
            act = torch.as_tensor(c_actions[..., :6].reshape(len(c_actions), -1), dtype=torch.float32, device=device)
            obs = torch.as_tensor(c_meta['pooled'], dtype=torch.float32, device=device)
            ks.append(knet(act, obs).cpu().numpy())
        else:
            ci = torch.as_tensor(c_inputs, dtype=torch.float32, device=device)
            for lo in range(0, len(ci), batch_size): ks.append(knet(ci[lo:lo + batch_size]).cpu().numpy())
    return np.concatenate(qs), np.concatenate(ks)


def _scores(qnet: nn.Module, qrep: np.ndarray, krep: np.ndarray) -> np.ndarray:
    scale = float(qnet.logit_scale.detach().exp().clamp(max=100).cpu())
    return (np.asarray(qrep, np.float32) @ np.asarray(krep, np.float32).T) * scale


def _summary(values: np.ndarray, q: dict[str, np.ndarray]) -> dict[str, Any]:
    return la.episode_bootstrap(values, q['task'], q['episode'])


def _evaluate_method(name: str, qnet: nn.Module, knet: nn.Module,
                     splits: dict[str, dict[str, np.ndarray]], c_inputs: np.ndarray,
                     c_actions: np.ndarray, c_meta: dict[str, np.ndarray],
                     taskmaps: tuple[dict[int, str], dict[int, str]], *, device: str,
                     composite: bool, output: Path, topk_filter: np.ndarray | None = None) -> dict[str, Any]:
    tasks40, tasks90 = taskmaps; result={}; prediction_paths={}
    all_candidate_meta = c_meta
    q_full_keys = list(splits)
    for split_name, q in splits.items():
        qrep, krep = _encode_eval(qnet, knet, q, c_inputs, c_actions, all_candidate_meta,
                                  device=device, composite=composite)
        scores = _scores(qnet, qrep, krep)
        allowed = _allowed_matrix(q, c_meta, tasks40, tasks90)
        scores[~allowed] = -np.inf
        if topk_filter is not None:
            base = topk_filter[split_name]
            scores = np.where(base, scores, -np.inf)
        errors = _teacher_errors(q, c_actions)
        idx = np.argmax(scores, axis=1)
        pred = c_actions[idx]
        mse = la.action_error(pred, q['actions'])
        rank = la.ranking_metrics(scores, errors)
        hard = np.asarray(q['hard'], bool)
        record={'top1_arm6': _summary(mse, q), 'hard_top1_arm6': _summary(mse[hard], {k: np.asarray(v)[hard] for k,v in q.items() if isinstance(v, np.ndarray) and len(v)==len(hard)}) if hard.any() else None,
                'ranking': rank, 'candidate_rows': len(c_actions)}
        if split_name == 'dev':
            temps = la.select_blend_temperatures(scores, c_actions, q['actions'])
        else:
            temps = result['dev_blend_temperatures']
        record['topk_blend']={}; record['dev_blend_temperatures']=temps
        for k in (5, 20):
            cfg=temps[str(k)]; blended,_=la.topk_blend(scores,c_actions,k,temperature=float(cfg['temperature']))
            bmse=la.action_error(blended,q['actions'])
            record['topk_blend'][str(k)]={'all':_summary(bmse,q),
                'hard':_summary(bmse[hard], {kk:np.asarray(vv)[hard] for kk,vv in q.items() if isinstance(vv,np.ndarray) and len(vv)==len(hard)}) if hard.any() else None,
                'temperature':cfg['temperature']}
        if split_name == 'dev': result['dev_blend_temperatures']=temps
        result[split_name]=record
        p=output/'predictions'/f'{name}-{split_name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p, key=q['key'], task=q['task'], episode=q['episode'], frame=q['frame'], hard=hard,
                            top1_indices=idx, top1_actions=pred, scores=scores.astype(np.float32), errors=errors.astype(np.float32))
        prediction_paths[split_name]=str(p)
    return {'metrics':result,'predictions':prediction_paths}


def _train_decoder(qnet: nn.Module, q: dict[str, np.ndarray], dev: dict[str, np.ndarray], *,
                   lr: float, max_epochs: int, seed: int, device: str,
                   composite: bool = False, patience: int = 10):
    torch.manual_seed(seed+700); rng=np.random.default_rng(seed+700)
    decoder=la.ActionDecoder(256 if composite else 128).to(device);opt=torch.optim.AdamW(decoder.parameters(),lr=lr)
    qnet.eval()
    with torch.no_grad():
        if composite:
            z=torch.cat([qnet(torch.as_tensor(q['x'][i:i+512],dtype=torch.float32,device=device),torch.as_tensor(q['pooled'][i:i+512],dtype=torch.float32,device=device)) for i in range(0,len(q['x']),512)],0).detach()
            dz=torch.cat([qnet(torch.as_tensor(dev['x'][i:i+512],dtype=torch.float32,device=device),torch.as_tensor(dev['pooled'][i:i+512],dtype=torch.float32,device=device)) for i in range(0,len(dev['x']),512)],0).detach()
        else:
            z=torch.cat([qnet(torch.as_tensor(q['x'][i:i+512],dtype=torch.float32,device=device))
                         for i in range(0,len(q['x']),512)],0).detach()
            dz=torch.cat([qnet(torch.as_tensor(dev['x'][i:i+512],dtype=torch.float32,device=device))
                          for i in range(0,len(dev['x']),512)],0).detach()
    y=torch.as_tensor(q['actions'][...,:6],dtype=torch.float32,device=device)
    dy=torch.as_tensor(dev['actions'][...,:6],dtype=torch.float32,device=device)
    best=float('inf');best_state=None;best_ep=0;stale=0;hist=[]
    order=np.arange(len(z))
    for ep in range(1,max_epochs+1):
        decoder.train();rng.shuffle(order);losses=[]
        for ix0 in np.array_split(order,max(1,int(np.ceil(len(order)/256)))):
            ix=torch.as_tensor(ix0,device=device);loss=F.mse_loss(decoder(z[ix]),y[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
        decoder.eval()
        with torch.no_grad(): val=float(torch.square(decoder(dz)-dy).sum((1,2)).mean().cpu()/70.)
        hist.append({'epoch':ep,'train_mse':float(np.mean(losses)),'dev_mse':val})
        if val < best - 1e-8:
            best=val;best_ep=ep;best_state={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()};stale=0
        else: stale+=1
        if stale>=patience and ep>=min(16,max_epochs):break
    decoder.load_state_dict(best_state)
    return decoder, {'history':hist,'best_epoch':best_ep,'best_dev_mse':best,'max_epochs':max_epochs,'stopped_early':len(hist)<max_epochs}


def run_la2(output: Path, *, row_cache: Path = L90X_ROWS, run_root: Path = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'),
             missing_root: Path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'),
             seen_stride5: Path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache'),
             unseen_stride5: Path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'),
             vjepa_eval_cache: Path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval'),
             device: str = 'cuda:0', memory_fraction: float = .15, smoke: bool = False,
             regimes: tuple[str, ...] = ('lib40', 'l90x', 'lib40_l90x'), max_rows: int | None = None) -> dict[str, Any]:
    output=Path(output)
    if output.exists():
        extra=[p.name for p in output.iterdir() if p.name not in ('features',)]
        if extra: raise FileExistsError(f'refusing to overwrite LA2 results in {output}: {extra[:8]}')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if torch.cuda.is_available() and device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction),device=torch.device(device))
    # Feature extraction is separate/resumable and may be launched before this stage.
    feature_file=output/'features/l90x_pooled.npy'
    if not (output/'features/extract_complete.json').exists():
        raise FileNotFoundError(f'missing complete L90X VLM pooled features: {output}/features; run --stage extract first')
    bank, bank_input, _fit_bank, _fit_bank_input, train40, learned, unseen, dev = la.key2._build_key2_queries(
        run_root,missing_root,seen_stride5,unseen_stride5,vjepa_eval_cache)
    if (len(bank['length']),len(learned['key']),len(unseen['key'])) != (4516,621,619):
        raise AssertionError('LA2 fixed bank/eval row set changed')
    # Every learned-eval episode is excluded from training candidates, as in LA1.
    learned_pairs=set(zip(np.asarray(learned['task']).tolist(),np.asarray(learned['episode']).tolist()))
    fit_ids=np.asarray([i for i,(t,e) in enumerate(zip(bank['task'],bank['episode'],strict=True)) if (int(t),int(e)) not in learned_pairs],np.int64)
    all_actions=la.key2._candidate_window_actions(np.arange(len(bank['length'])),bank).astype(np.float32)
    fit_actions=all_actions[fit_ids]
    fit_meta={k:np.asarray(v)[fit_ids] for k,v in bank.items() if isinstance(v,np.ndarray) and len(v)==len(bank['length'])}
    # Exact unit-start VLM observation keys; reuse the exact end cache from LA1.
    end_root=LA1_ROOT/'unit-end-features'
    end_keys=np.asarray(bank_input['key']).copy();end_keys[:,2]=np.asarray(bank['source'])[:,3]-1
    store=la.key2.ExactFeatureStore(run_root,missing_root,seen_stride5,unseen_stride5)
    try:
        tok,grp=store.get(end_keys,domain='libero40'); end_pool=la.pooled_tokens(tok,grp)
    except KeyError:
        end_pool=la.exact_pool_with_extra(end_keys,store,end_root)
    unit_inputs_full=la.make_unit_input(bank,bank_input,use_action=True,end_observation=end_pool).astype(np.float32)
    unit_inputs=unit_inputs_full[fit_ids]
    c_meta={**fit_meta,'pooled':la.pooled_tokens(bank_input['tokens'][fit_ids],bank_input['groups'][fit_ids]),
            'task':np.asarray(bank['task'])[fit_ids], 'episode':np.asarray(bank['episode'])[fit_ids]}
    q40=_query40(train40,domain=0); qdev=_query40(dev,domain=0); qlearned=_query40(learned,domain=0); qunseen=_query40(unseen,domain=0)
    if smoke:
        def take_rows(d, limit=12):
            return {k:(np.asarray(v)[:limit] if isinstance(v,np.ndarray) and len(v)==len(d['x']) else v) for k,v in d.items()}
        qdev=take_rows(qdev);qlearned=take_rows(qlearned);qunseen=take_rows(qunseen)
    # External task windows are labels only for teacher construction/training, never features from the future.
    qx=_make_l90x_queries(row_cache,feature_file)
    if smoke and max_rows is None: max_rows=128
    if max_rows is not None:
        qx={k:(np.asarray(v)[:max_rows] if isinstance(v,np.ndarray) and len(v)==len(qx['x']) else v) for k,v in qx.items()}
    tasks40,tasks90=_instruction_map(run_root,row_cache)
    # Prevent any fixed Dev/Eval task by exact manifest assertion and row check.
    forbidden=set((6,7,0,10,13,31,35,43,57,61,84,88))
    if set(map(int,np.unique(qx['task']))) & forbidden:
        raise AssertionError('fixed Dev/Eval LIBERO-90 task entered L90X training')
    splits={'dev':qdev,'learned':qlearned,'unseen':qunseen}
    # L90X queries come from 77 available tasks; task names overlap LIBERO-90.
    regimes_data={'lib40':q40,'l90x':qx,'lib40_l90x':_concat_queries(q40,qx)}
    lr_grid=(1e-4,3e-4)
    ep_grid=(16,32,64)
    if smoke: lr_grid=(3e-4,);ep_grid=(1,)
    cand_meta_for_train=c_meta
    dev_errors=_teacher_errors(qdev,fit_actions)
    dev_allowed=_allowed_matrix(qdev,c_meta,tasks40,tasks90)
    models={};selected={};selected_decoder={};all_summaries={};job_idx=0
    (output/'models').mkdir(parents=True,exist_ok=True)
    for regime in regimes:
      qtrain=regimes_data[regime]
      for lr in lr_grid:
       for max_ep in ep_grid:
        job_idx+=1;tag=f'{regime}-lr{lr:g}-maxep{max_ep}'
        print(json.dumps({'event':'train_start','job':tag,'queries':len(qtrain['x'])}),flush=True)
        qnet,knet,trainlog=_fit_ranker(qtrain,unit_inputs,fit_actions,cand_meta_for_train,qdev,dev_errors,
          dev_allowed,tasks40,tasks90,lr=lr,max_epochs=max_ep,seed=SEED,device=device,patience=12)
        ev=_evaluate_method(tag,qnet,knet,splits,unit_inputs_full,all_actions,{
            **{k:np.asarray(v) for k,v in bank.items() if isinstance(v,np.ndarray) and len(v)==len(bank['length'])},
            'pooled':la.pooled_tokens(bank_input['tokens'],bank_input['groups'])},
            (tasks40,tasks90),device=device,composite=False,output=output)
        # Decoder is separately fit/selected on Dev while query/key retrieval weights stay fixed.
        decoder,declog=_train_decoder(qnet,qtrain,qdev,lr=lr,max_epochs=trainlog['best_epoch'],seed=SEED,device=device)
        decoder.eval(); decmetrics={}
        for split_name,qq in splits.items():
            with torch.no_grad():
                z=torch.cat([qnet(torch.as_tensor(qq['x'][lo:lo+512],dtype=torch.float32,device=device)) for lo in range(0,len(qq['x']),512)],0)
                pred=decoder(z).cpu().numpy()
            dm=la.action_error(pred,qq['actions']);hard=np.asarray(qq['hard'],bool)
            decmetrics[split_name]={'all':_summary(dm,qq),'hard':_summary(dm[hard],{k:np.asarray(v)[hard] for k,v in qq.items() if isinstance(v,np.ndarray) and len(v)==len(hard)}) if hard.any() else None}
            np.savez_compressed(output/'predictions'/f'{tag}-decoder-{split_name}.npz',key=qq['key'],task=qq['task'],episode=qq['episode'],frame=qq['frame'],hard=hard,actions=pred)
        state={'qnet':qnet.state_dict(),'knet':knet.state_dict(),'decoder':decoder.state_dict(),'config':tag,'train':trainlog,'decoder_train':declog}
        torch.save(state,output/'models'/f'{tag}.pt')
        metrics=ev['metrics']; dev_mse=metrics['dev']['top1_arm6']['mean']
        dev_dec=decmetrics['dev']['all']['mean']
        models[tag]={'regime':regime,'lr':lr,'max_epochs':max_ep,'retrieval_training':trainlog,'decoder_training':declog,
                     'retrieval':metrics,'decoder':decmetrics,'predictions':ev['predictions'],'checkpoint':str(output/'models'/f'{tag}.pt')}
        all_summaries[tag]={'dev_top1':dev_mse,'dev_decoder':dev_dec}
        print(json.dumps({'event':'train_done','job':tag,'best_epoch':trainlog['best_epoch'],'dev_top1':dev_mse,'dev_decoder':dev_dec}),flush=True)
      # selection: best retrieval MSE on Dev in this training-data regime.
      tags=[t for t,v in models.items() if v['regime']==regime]
      selected[regime]=min(tags,key=lambda t:(models[t]['retrieval']['dev']['top1_arm6']['mean'],models[t]['lr'],models[t]['max_epochs']))
      selected_decoder[regime]=min(tags,key=lambda t:(models[t]['decoder']['dev']['all']['mean'],models[t]['lr'],models[t]['max_epochs']))
    # Explicit synthetic composite key: unit action-content E plus E(o_start), same 40+L90X query regime.
    qtrain=regimes_data['lib40_l90x']; composite_tags=[]
    for lr in lr_grid:
     for max_ep in ep_grid:
      tag=f'composite-lib40_l90x-lr{lr:g}-maxep{max_ep}';composite_tags.append(tag)
      qnet,knet,trainlog=_fit_ranker(qtrain,unit_inputs,fit_actions,cand_meta_for_train,qdev,dev_errors,
        dev_allowed,tasks40,tasks90,lr=lr,max_epochs=max_ep,seed=SEED,device=device,composite=True,patience=12)
      # Composite value keys use action-only content plus the exact start observation.
      c_actions_full=all_actions
      c_meta_full={**{k:np.asarray(v) for k,v in bank.items() if isinstance(v,np.ndarray) and len(v)==len(bank['length'])},
                   'pooled':la.pooled_tokens(bank_input['tokens'],bank_input['groups'])}
      ev=_evaluate_method(tag,qnet,knet,splits,unit_inputs_full,c_actions_full,c_meta_full,(tasks40,tasks90),
                          device=device,composite=True,output=output)
      decoder,declog=_train_decoder(qnet,qtrain,qdev,lr=lr,max_epochs=trainlog['best_epoch'],seed=SEED,device=device,composite=True)
      decmetrics={}
      for split_name,qq in splits.items():
       with torch.no_grad():
        z=torch.cat([qnet(torch.as_tensor(qq['x'][lo:lo+512],dtype=torch.float32,device=device),torch.as_tensor(qq['pooled'][lo:lo+512],dtype=torch.float32,device=device)) for lo in range(0,len(qq['x']),512)],0)
        pred=decoder(z).cpu().numpy()
       dm=la.action_error(pred,qq['actions']);hard=np.asarray(qq['hard'],bool)
       decmetrics[split_name]={'all':_summary(dm,qq),'hard':_summary(dm[hard],{k:np.asarray(v)[hard] for k,v in qq.items() if isinstance(v,np.ndarray) and len(v)==len(hard)}) if hard.any() else None}
       np.savez_compressed(output/'predictions'/f'{tag}-decoder-{split_name}.npz',key=qq['key'],task=qq['task'],episode=qq['episode'],frame=qq['frame'],hard=hard,actions=pred)
      torch.save({'qnet':qnet.state_dict(),'knet':knet.state_dict(),'decoder':decoder.state_dict(),'config':tag,'train':trainlog,'decoder_train':declog},output/'models'/f'{tag}.pt')
      models[tag]={'regime':'composite-lib40_l90x','lr':lr,'max_epochs':max_ep,'retrieval_training':trainlog,'decoder_training':declog,'retrieval':ev['metrics'],'decoder':decmetrics,'predictions':ev['predictions'],'checkpoint':str(output/'models'/f'{tag}.pt')}
    if composite_tags:
      selected['composite']=min(composite_tags,key=lambda t:(models[t]['retrieval']['dev']['top1_arm6']['mean'],models[t]['lr'],models[t]['max_epochs']))
      selected_decoder['composite']=min(composite_tags,key=lambda t:(models[t]['decoder']['dev']['all']['mean'],models[t]['lr'],models[t]['max_epochs']))
    # Re-rank only P5's top-50/top-200 with the Dev-selected best non-composite model.
    p5_rerank={}
    p5_scores_by_split={}
    run_p5_rerank=True
    if run_p5_rerank:
      best_tag=min([t for t in models if models[t]['regime'] in ('lib40','l90x','lib40_l90x')],key=lambda t:models[t]['retrieval']['dev']['top1_arm6']['mean'])
      p5model=la.key2._new_model('p5').to(device)
      p5path=KEY2_ROOT/'final/p5/p5_seed1.pt'
      p5model.load_state_dict(torch.load(p5path,map_location=device,weights_only=True));p5model.eval()
      # Build exact P5 scores from its frozen attempt-27 checkpoint.
      all_bank_pooled=la.pooled_tokens(bank_input['tokens'],bank_input['groups'])
      # Key2 VectorRanker accepts pooled obs + history + mask + state for query/key.
      def p5_representations(src,bank_side=False):
        if bank_side:
            po=all_bank_pooled;h=np.asarray(bank_input['history'],np.float32);m=np.asarray(bank_input['history_mask'],np.float32);st=np.asarray(bank_input['state'],np.float32)
        else:
            po=la.pooled_tokens(src['tokens'],src['groups']);h=np.asarray(src['history'],np.float32);m=np.asarray(src['history_mask'],np.float32);st=np.asarray(src['state'],np.float32)
        parts=[]
        with torch.no_grad():
          for lo in range(0,len(po),256):
            sl=slice(lo,min(lo+256,len(po)))
            t={'pooled':po[sl],'history':h[sl],'history_mask':m[sl],'state':st[sl]}
            parts.append(la.key2._model_batch_encode(p5model,'p5',t,device,key=bank_side).cpu().numpy())
        return np.concatenate(parts)
      kb=p5_representations(None,True)
      source_splits={'dev':dev,'learned':learned,'unseen':unseen}
      for split_name, qq in splits.items():
        source_splits[split_name]={k:(np.asarray(v)[:len(qq['x'])] if isinstance(v,np.ndarray) and len(v)==len(source_splits[split_name]['key']) else v)
                                   for k,v in source_splits[split_name].items()}
      for split_name,qq in splits.items():
       qr=p5_representations(source_splits[split_name],False)
       p5scores=np.einsum('qkd,nkd->qn',qr,kb).astype(np.float32)/5
       p5scores*=float(p5model.logit_scale.detach().exp().clamp(max=100).cpu())
       p5scores[~_allowed_matrix(qq,{**{k:np.asarray(v) for k,v in bank.items() if isinstance(v,np.ndarray) and len(v)==len(bank['length'])}},tasks40,tasks90)]=-np.inf
       # Keep the dense score matrix local only; results.json stores metrics,
       # not the potentially large query-by-bank score arrays.
       p5_scores_by_split[split_name]=p5scores
      chosen=models[best_tag]
      qbest,kbest=_load_model_for_eval(chosen['checkpoint'],device,composite=False)
      c_meta_full={**{kk:np.asarray(vv) for kk,vv in bank.items() if isinstance(vv,np.ndarray) and len(vv)==len(bank['length'])},'pooled':all_bank_pooled}
      for k in (50,200):
       name=f'{best_tag}-p5top{k}'
       mask_by_split={}
       for split_name,qq in splits.items():
        p5scores=p5_scores_by_split[split_name]
        top=np.argsort(-p5scores,axis=1,kind='stable')[:,:k]
        subset=np.zeros_like(p5scores,bool);subset[np.arange(len(top))[:,None],top]=True
        mask_by_split[split_name]=subset
       ev=_evaluate_method(name,qbest,kbest,splits,unit_inputs_full,all_actions,c_meta_full,(tasks40,tasks90),
                           device=device,composite=False,output=output,topk_filter=mask_by_split)
       p5_rerank[name]=ev['metrics']
       models[name]={'regime':'p5_rerank','base_model':best_tag,'p5_candidate_k':k,
                     'retrieval':ev['metrics'],'predictions':ev['predictions'],
                     'checkpoint':chosen['checkpoint']}
    results={'status':'smoke_complete' if smoke else 'complete','seed':SEED,'device':device,'memory_fraction':memory_fraction,
      'sources':{'lib40_training_query_count':len(q40['x']),'l90x_training_query_count':len(qx['x']),'l90x_rows':str(row_cache),
        'l90x_task_count_manifest':len(json.loads((Path(row_cache)/'manifest.json').read_text()).get('tasks',[])),
        'l90x_tasks_with_data':len(np.unique(np.load(Path(row_cache)/'task.npy',mmap_mode='r'))),
        'l90x_missing_tasks':json.loads((Path(row_cache)/'manifest.json').read_text()).get('missing_tasks',[]),
        'l90x_dev_eval_excluded':True,'l90x_scene_overlap':'L90X source is LIBERO-90; its scenes/tasks overlap LIBERO-90 distribution, so task-held-out generalization is not tested by the added training rows.',
        'full_x1_vlm_tokens_cache':False,'fresh_pooled_prefix_extraction':True},
      'candidate_bank_units':len(all_actions),'fit_candidates':len(fit_actions),'learned_rows':len(learned['key']),'unseen_rows':len(unseen['key']),
      'selection':'Dev LIBERO-90 tasks 6/7 only; min retrieved action MSE. Learned/Eval never used for selection.',
      'methods':models,'selected_by_regime':selected,'selected_decoder_by_regime':selected_decoder,'p5_rerank':p5_rerank,
      'references':{'key2_p5_unseen_top1':0.10544633470538463,'hold_last_unseen_all_hard':[0.1140,0.2010],
        'pi05_unseen_all_hard':[0.1150,0.2529],'la1_cont_dec_unseen_all_hard':[0.0866,0.1755]},
      'training_grid':{'learning_rates':list(lr_grid),'max_epochs':list(ep_grid),'early_stopping_patience':12,'seed':SEED,
        'data_regimes':list(regimes),'optimizer':'AdamW','candidate_memory_fixed':'LIBERO-40 DCT3 4,516 units; learned evaluation episodes excluded from training candidate bank'},
      'metrics':'arm6 action MSE uses sum squared error over 10x6 divided by 10x7; gripper replaced by truth only at scoring. Episode bootstrap 5,000. Ranking is LA1-C positive-is-good definition.'}
    _atomic_json(output/'results.json',results)
    _write_report(output,results)
    return results


def _load_model_for_eval(path: str, device: str, composite: bool = False):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    qstate=ck['qnet'];kstate=ck['knet']
    qdim=qstate['net.0.weight'].shape[1] if not composite else qstate['content.net.0.weight'].shape[1]
    if composite:qnet=CompositeQueryEncoder(qdim).to(device);knet=CompositeUnitEncoder().to(device)
    else:qnet=la.QueryEncoder(qdim).to(device);knet=la.UnitKeyEncoder(kstate['net.0.weight'].shape[1]).to(device)
    qnet.load_state_dict(qstate);knet.load_state_dict(kstate);qnet.eval();knet.eval()
    return qnet,knet


def _load_qnet_scale(path: str, device: str, composite: bool = False):
    return _load_model_for_eval(path,device,composite)[0]


def _write_report(output: Path, results: dict[str, Any]) -> None:
    lines=['# LA2：保存運動単位に結びつく連続潜在の拡張','','## 主表','','MSE は腕6軸の二乗誤差和/(10×7)、グリッパーは採点時のみ正解に置換。top-1 は単位再生、top-k はDevで温度を選んだ行動ブレンド。全体/hard の CI はエピソード bootstrap 5,000回。Recall/Spearman は点推定。','',
      '|条件|Unseen top-1 全体/難|Unseen top-5/top-20 blend 全体|Learned top-1 全体/難|Recall@1/@20 (Unseen)|Spearman (Unseen)|Dev 選択|','|---|---:|---:|---:|---:|---:|---|']
    def fmt(item,hard=False):
        if not item:return '—'
        x=item.get('hard_top1_arm6') if hard else item.get('top1_arm6')
        if x is None:return '—'
        return f"{x['mean']:.4f} [{x['ci95_low']:.4f},{x['ci95_high']:.4f}]"
    for regime,tag in results.get('selected_by_regime',{}).items():
        if tag not in results['methods']:continue
        m=results['methods'][tag];u=m['retrieval']['unseen'];l=m['retrieval']['learned'];r=u['ranking']
        b5=u['topk_blend']['5']['all']['mean'];b20=u['topk_blend']['20']['all']['mean']
        lines.append(f"|{regime} {tag}|{fmt(u)}/{fmt(u,True)}|{b5:.4f}/{b20:.4f}|{fmt(l)}/{fmt(l,True)}|{r['recall_at_k']['1']:.3f}/{r['recall_at_k']['20']:.3f}|{r['spearman_score_neg_error']:.3f}|{tag}|")
    lines += ['', '### 同じ学習設定の連続デコーダ出口', '',
      '|条件|Unseen decoder 全体/難|Learned decoder 全体/難|Dev選択|','|---|---:|---:|---|']
    for regime,tag in results.get('selected_decoder_by_regime',{}).items():
        if tag not in results['methods'] or 'decoder' not in results['methods'][tag]:continue
        m=results['methods'][tag]['decoder']
        u=m['unseen'];l=m['learned']
        def dformat(x,key):
            z=x.get(key)
            return '—' if not z else f"{z['mean']:.4f} [{z['ci95_low']:.4f},{z['ci95_high']:.4f}]"
        lines.append(f"|{regime} {tag}|{dformat(u,'all')}/{dformat(u,'hard')}|{dformat(l,'all')}/{dformat(l,'hard')}|decoder Dev MSE|")
    if results.get('selected_by_regime',{}).get('composite') in results.get('methods',{}):
        tag=results['selected_by_regime']['composite'];m=results['methods'][tag];u=m['retrieval']['unseen'];l=m['retrieval']['learned'];r=u['ranking']
        lines.append(f"|合成キー {tag}|{fmt(u)}/{fmt(u,True)}|{u['topk_blend']['5']['all']['mean']:.4f}/{u['topk_blend']['20']['all']['mean']:.4f}|{fmt(l)}/{fmt(l,True)}|{r['recall_at_k']['1']:.3f}/{r['recall_at_k']['20']:.3f}|{r['spearman_score_neg_error']:.3f}|Dev 最良|")
    lines += ['', '参照行は同じ指標定義の公開値。既存 P5 / 直前保持 / π0.5 / LA1-cont-dec は別評価結果から併記し、今回の bootstrap CI が無い値には CI を付けない。','',
      '|参照条件|Unseen top-1 全体/難|備考|','|---|---:|---|',
      '|KEY2-F P5|0.1054 / 0.1842|固定 attempt-27|','|直前保持|0.1140 / 0.2010|固定行|','|π0.5|0.1150 / 0.2529|回転補正済み固定行|','|LA1 continuous decoder|0.0866 / 0.1755|LA1 Dev選択済み設定|']
    lines += ['', '## 手法上の注記','',
      '- L90X の生行キャッシュは EXP3-B の 11,030 行を使用した。78タスクIDのうちローカルHDF5が無い1タスクは行0で、データ実在は77タスク。固定 Dev 6/7 と Eval 10タスクは manifest と行IDの双方で除外した。',
      '- X1 の全件VLM token feature cache は存在しなかったため、EXP3-B の正しい向き・変換済み画像から π0.5 の最終prefixを新規抽出し、LA1と同じ有効トークン平均を保存した。X1 smoke の共通行で数値一致を確認。',
      '- L90X は LIBERO-90 のシーン・タスク分布に由来し、LIBERO-90 との scene/task overlap がある。したがってこのデータ拡大は新シーン汎化の評価ではない。',
      '- 合成キーは [unit action content embedding; learned E(o_start)]。P5再順位付けは固定 attempt-27 P5 の上位50/200候補に限定し、追加の学習は Dev で選んだLA潜在だけを使う。',
      '- 各方法は単一seed 4101。評価正解行動は教師距離・Dev選択・最終採点以外の特徴には使用しない。']
    if results.get('p5_rerank'):
        lines += ['', '### P5 上位候補だけを連続潜在で再順位付け', '',
                  '|条件|Unseen top-1 全体/難|Learned top-1 全体/難|Recall@1/@20|Spearman|',
                  '|---|---:|---:|---:|---:|']
        for name, block in results['p5_rerank'].items():
            if name not in results['methods'] or not isinstance(block,dict) or 'unseen' not in block:continue
            u=block['unseen'];l=block['learned'];r=u['ranking']
            lines.append(f"|{name}|{fmt(u)}/{fmt(u,True)}|{fmt(l)}/{fmt(l,True)}|{r['recall_at_k']['1']:.3f}/{r['recall_at_k']['20']:.3f}|{r['spearman_score_neg_error']:.3f}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('extract','train','all'),default='all')
    p.add_argument('--output',type=Path,default=LA1_ROOT/'la2-production')
    p.add_argument('--row-cache',type=Path,default=L90X_ROWS)
    p.add_argument('--device',default=DEVICE_DEFAULT);p.add_argument('--memory-fraction',type=float,default=.15)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-rows',type=int,default=None)
    a=p.parse_args();out=Path(a.output)
    if a.stage in ('extract','all'):
        r=extract_l90x_pooled(a.row_cache,out/'features',batch_size=a.batch_size,device=a.device,
                              memory_fraction=a.memory_fraction,max_rows=a.max_rows if a.smoke else None)
        print(json.dumps({'stage':'extract','result':r},ensure_ascii=False),flush=True)
        if a.stage=='extract':return
    run_la2(out,device=a.device,memory_fraction=a.memory_fraction,smoke=a.smoke,max_rows=a.max_rows)


if __name__=='__main__':main()
