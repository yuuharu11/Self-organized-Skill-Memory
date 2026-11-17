#!/usr/bin/env python3
"""EX2: key-independent action Expert with an error-conditioned bridge time.

The Expert is trained only from saved DCT3 motion units sampled by action-space
rank. Retrieval keys are used only for the EX2 evaluation and runtime draft
selection. Small, framework-independent helpers live here so the scheduling and
confidence rules can be tested without loading the policy model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
EX1 = ROOT / 'expert-ex1-01/production-03/scope-change-01'
KEYGRID = ROOT / 'key-grid-01/production-04'
OUT = ROOT / 'expert-ex2-01'


def geometric_rank_sample(topk_errors: np.ndarray, rng: np.random.Generator, *,
                          p: float = .16, random_fraction: float = .10) -> tuple[np.ndarray, np.ndarray]:
    """Sample 1-based action-error rank with a truncated geometric law.

    ``topk_errors`` contains the eligible memory candidates' ordered MSEs.
    Ten percent of rows instead receive a uniformly sampled candidate ID (rank
    is -1 for those rows); all other ranks follow P(r)=p(1-p)^(r-1), truncated
    and renormalized over the available columns.
    """
    e = np.asarray(topk_errors)
    if e.ndim != 2 or e.shape[1] < 1 or not np.isfinite(e).all():
        raise ValueError(f'expected finite [N,K] errors, got {e.shape}')
    if not 0 < p <= 1 or not 0 <= random_fraction <= 1:
        raise ValueError('invalid geometric/random probabilities')
    n, k = e.shape
    probs = p * (1 - p) ** np.arange(k, dtype=np.float64)
    probs /= probs.sum()
    ranks = rng.choice(k, size=n, p=probs)
    random_mask = rng.random(n) < random_fraction
    ranks[random_mask] = -1
    return ranks.astype(np.int32), random_mask


def rank_stored_actions(targets: np.ndarray, candidates: np.ndarray, candidate_task: np.ndarray,
                        *, query_task: np.ndarray, candidate_episode: np.ndarray | None=None,
                        query_episode: np.ndarray | None=None, k: int=50) -> tuple[np.ndarray,np.ndarray]:
    """Rank unchanged stored 10-step actions by arm-six MSE with leakage masks.

    Candidate indices are bank row IDs. Every candidate from the query task is
    excluded; episode vectors additionally exclude the query episode explicitly.
    """
    y=np.asarray(targets,np.float32);c=np.asarray(candidates,np.float32);ct=np.asarray(candidate_task,np.int64);qt=np.asarray(query_task,np.int64)
    if y.ndim!=3 or c.ndim!=3 or y.shape[1:]!=(10,7) or c.shape[1:]!=(10,7) or ct.shape!=(len(c),) or qt.shape!=(len(y),):
        raise ValueError(f'invalid rank arrays: {y.shape}/{c.shape}/{ct.shape}/{qt.shape}')
    ce=None if candidate_episode is None else np.asarray(candidate_episode,np.int64)
    qe=None if query_episode is None else np.asarray(query_episode,np.int64)
    if (ce is None)!=(qe is None) or (ce is not None and (ce.shape!=(len(c),) or qe.shape!=(len(y),))):raise ValueError('episode vectors must be aligned')
    if not 1<=int(k)<=len(c):raise ValueError('invalid top-k')
    cf=c[:,:,:6].reshape(len(c),-1).astype(np.float64);yf=y[:,:,:6].reshape(len(y),-1).astype(np.float64);cn=np.square(cf).sum(1)
    ids=np.empty((len(y),int(k)),np.int32);errors=np.empty((len(y),int(k)),np.float32)
    for lo in range(0,len(y),128):
        hi=min(lo+128,len(y));yy=yf[lo:hi];d=np.square(yy).sum(1,keepdims=True)+cn[None]-2*yy@cf.T
        mask=ct[None,:]==qt[lo:hi,None]
        d[mask]=np.inf;order=np.argsort(d,axis=1,kind='stable')[:,:int(k)]
        if not np.isfinite(np.take_along_axis(d,order,axis=1)).all():raise ValueError('too few eligible stored units')
        ids[lo:hi]=order.astype(np.int32);errors[lo:hi]=(np.maximum(0,np.take_along_axis(d,order,axis=1))/yf.shape[1]).astype(np.float32)
    return ids,errors


def select_t0(error: np.ndarray | float, scale: float, *, jitter: np.ndarray | float = 1.0,
              low: float = .05, high: float = .60) -> np.ndarray:
    """Monotone EX2 bridge schedule ``clip(c * sqrt(error) * jitter)``."""
    e = np.asarray(error, np.float64)
    j = np.asarray(jitter, np.float64)
    if not np.isfinite(e).all() or not np.isfinite(j).all() or np.any(e < 0):
        raise ValueError('error and jitter must be finite; error must be nonnegative')
    lo=np.asarray(low,np.float64)
    if not math.isfinite(float(scale)) or scale <= 0 or not np.isfinite(lo).all() or np.any(lo<0) or np.any(lo>=high) or not 0 < high <= 1:
        raise ValueError('invalid schedule scale or bounds')
    return np.minimum(np.maximum(float(scale) * np.sqrt(e) * j, lo), high).astype(np.float32)


def single_float(value: Any, *, name: str = 'value') -> float:
    """Convert a one-item scalar/array result without assuming its rank."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f'{name} must contain one value, got shape {arr.shape}')
    return float(arr.reshape(-1).item())


def confidence_features(draft: np.ndarray, pi05: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Build inference-available error predictors from retrieval and π0.5.

    Features are log(arm6 draft/reference MSE), top-1/top-2 score margin,
    normalized score entropy, and log arm6 norm ratio. No target actions,
    task labels, episode IDs, or future observations are accepted.
    """
    d = np.asarray(draft, np.float32)
    r = np.asarray(pi05, np.float32)
    s = np.asarray(scores, np.float64)
    if d.ndim != 3 or r.ndim != 3 or r.shape[:2] != d.shape[:2] or d.shape[1] != 10 or d.shape[-1] < 6 or r.shape[-1] < 6:
        raise ValueError(f'bad action shapes: {d.shape}, {r.shape}')
    if s.ndim != 2 or len(s) != len(d) or s.shape[1] < 2:
        raise ValueError(f'need at least two candidate scores per query, got {s.shape}')
    if not np.isfinite(d).all() or not np.isfinite(r).all() or np.isnan(s).any() or np.isposinf(s).any():
        raise ValueError('nonfinite draft/reference or invalid candidate score')
    if np.any(np.isfinite(s).sum(axis=1) < 2):
        raise ValueError('each query needs two unmasked candidate scores')
    order = np.argsort(-s, axis=1, kind='stable')
    top = np.take_along_axis(s, order[:, :2], axis=1)
    z = s - s.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    entropy = -(prob * np.log(np.maximum(prob, 1e-30))).sum(axis=1)
    entropy /= np.log(max(s.shape[1], 2))
    diff_mse = np.square(d[:, :, :6] - r[:, :, :6]).mean(axis=(1, 2))
    dn = np.linalg.norm(d[:, :, :6].reshape(len(d), -1), axis=1)
    rn = np.linalg.norm(r[:, :, :6].reshape(len(r), -1), axis=1)
    ratio = dn / np.maximum(rn, 1e-6)
    return np.stack([np.log(diff_mse + 1e-6), top[:, 0] - top[:, 1], entropy,
                     np.log(ratio + 1e-6)], axis=1).astype(np.float32)


def _new_confidence_model():
    import torch
    from torch import nn
    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 16),
                                     nn.GELU(), nn.Linear(16, 1))
            self.register_buffer('x_mean', torch.zeros(4))
            self.register_buffer('x_std', torch.ones(4))
        def forward(self, z):
            return self.net((z - self.x_mean) / self.x_std)
    return Regressor()


def fit_confidence_regressor(features: np.ndarray, errors: np.ndarray, *, seed: int = 20261104,
                             steps: int = 600, learning_rate: float = 1e-3,
                             device: str = 'cpu'):
    """Fit a compact torch MLP to log draft error on held-out memory episodes."""
    import torch
    x = np.asarray(features, np.float32)
    y = np.asarray(errors, np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) != len(y) or len(x) < 16:
        raise ValueError(f'bad trust-regression data shapes: {x.shape}, {y.shape}')
    if np.any(y < 0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('trust-regression data must be finite and errors nonnegative')
    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(x, device=device)
    y_t = torch.as_tensor(np.log(y + 1e-6), device=device)[:, None]
    model = _new_confidence_model().to(device)
    # Normalize using only the memory-held-out training fold.
    model.register_buffer('x_mean', x_t.mean(0))
    model.register_buffer('x_std', x_t.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(int(steps)):
        pred = model(x_t)
        loss = torch.nn.functional.smooth_l1_loss(pred, y_t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


def predict_error(model, features: np.ndarray, *, device: str = 'cpu') -> np.ndarray:
    """Predict positive action error without exposing targets to the predictor."""
    import torch
    x = torch.as_tensor(np.asarray(features, np.float32), device=device)
    with torch.no_grad():
        loge = model(x).squeeze(-1)
    return np.maximum(np.exp(np.clip(loge.cpu().numpy(), -20., 5.)) - 1e-6, 0).astype(np.float32)


def choose_schedule_on_dev(errors: np.ndarray, features: np.ndarray, *, scales=(.5, 1., 1.5, 2., 3.),
                           thresholds=(None, .05, .10, .15, .20, .30),
                           pi05_mse: np.ndarray | None = None) -> dict[str, Any]:
    """Choose c and optional π0.5 fallback threshold on Dev arm6 MSE only.

    The function accepts predictions from the fitted error regressor and
    deterministic t0-independent proxy errors for the proposed schedule. A
    caller supplies per-row validation MSE curves (generated from the frozen
    Expert) to ``dev_curves`` in :func:`select_dev_policy` below.
    """
    e = np.asarray(errors, np.float32).reshape(-1)
    if not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError('invalid predicted Dev errors')
    return {'scales': [float(c) for c in scales],
            'fallback_thresholds': [None if t is None else float(t) for t in thresholds],
            'predicted_error_median': float(np.median(e)),
            'fallback_reference_mse': None if pi05_mse is None else float(np.mean(pi05_mse)),
            'selection_metric': 'Dev arm6 MSE; caller evaluates frozen Expert outputs for each c/threshold'}


def select_dev_policy(dev_curves: dict[tuple[float, float | None], np.ndarray],
                      pi05_mse: np.ndarray) -> dict[str, Any]:
    """Select the scale/fallback pair with minimum paired Dev arm6 MSE."""
    baseline = np.asarray(pi05_mse, np.float64).reshape(-1)
    if not dev_curves or not np.isfinite(baseline).all():
        raise ValueError('empty/nonfinite Dev curves')
    rows = []
    for key, values in dev_curves.items():
        scale,threshold,*extras=key
        low_floor=float(extras[0]) if extras else .05
        v = np.asarray(values, np.float64).reshape(-1)
        if v.shape != baseline.shape or not np.isfinite(v).all():
            raise ValueError('Dev predictions must align with π0.5 rows and be finite')
        rows.append({'scale': float(scale), 'threshold': None if threshold is None else float(threshold),
                     'high_confidence_t0_floor':low_floor,
                     'mse6': float(v.mean())})
    best = min(rows, key=lambda x: (x['mse6'], x['scale'], float('inf') if x['threshold'] is None else x['threshold'],x['high_confidence_t0_floor']))
    return {'selected': best, 'pi05_mse6': float(baseline.mean()), 'grid': rows,
            'selection_rows': len(baseline), 'uses_eval': False}


def _write(path: Path, payload: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def _progress(output: Path, stage: str, status: str, **fields) -> None:
    record={'utc':__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ',__import__('time').gmtime()),
            'stage':stage,'status':status,**fields}
    path=Path(output)/'progress.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    print(json.dumps(record,ensure_ascii=False),flush=True)


def build_retrieval_cache(output: Path, *, device: str = 'cuda:0', threads: int = 8,
                          mem_fraction: float = .25, smoke: bool = False) -> dict[str, Any]:
    """Score D0/D1/D3/D5 over fixed EX2 splits and save ranked unit drafts."""
    import sys
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
    import key_shared_latent as key2
    import key_grid
    import expert_ex1 as ex1
    import skill_expert_refine as exp3
    run_root = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
    root = ROOT
    bank, bi, _fit_bank, _fit_bi, train, learned, unseen, dev = key2._build_key2_queries(
        run_root, root / 'key-shared-01/attempt-02/prefix-missing', root / 'seg-context-01/seg5-cache',
        root / 'seg-context-01/unseen-stride5', root / 'key-shared-01/attempt-27-production/vjepa/libero90_eval')
    if smoke:
        # Keep the complete candidate bank so candidate-ID contracts are tested;
        # choose spread rows rather than a contiguous prefix, preserving multiple
        # tasks/episodes for the confidence holdout and Dev smoke checks.
        def spread(part,n):
            idx=np.unique(np.linspace(0,len(part['key'])-1,min(n,len(part['key'])),dtype=np.int64))
            return {k:v[idx] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(part['key']) else v for k,v in part.items()}
        train=spread(train,16);dev=spread(dev,8);learned=spread(learned,8);unseen=spread(unseen,8)
    candidate_actions = key2._candidate_window_actions(np.arange(len(bank['length'])), bank).astype(np.float32)
    # KEYGRID ranker was selected on Dev. It consumes exactly the history+state
    # vector used by its reported top-1 row.
    grid_results = json.loads((KEYGRID / 'results.json').read_text())
    d5_rec = grid_results['new_models']['history_state']['variants']['two_tower_mlp']
    d5 = key_grid.PairMLPRanker(88, shared=False).to(device)
    d5.load_state_dict(torch.load(d5_rec['checkpoint'], map_location=device, weights_only=True)); d5.eval()
    bank_prepared = key_grid._prepare_vector_sources(bi)
    bank_vec = key_grid.make_vector_features(bank_prepared, 'history_state')
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    np.save(output/'candidate_history_state.npy',bank_vec.astype(np.float32))
    np.savez_compressed(output/'candidate_metadata.npz',source=np.asarray(bank['source'],np.int32),
                        unit_actions=np.asarray(candidate_actions,np.float32))
    key_cfg = ex1.load_locked_key2_checkpoints()
    loaded = {}
    for mode in ('p0', 'p5', 'e_vlm_h'):
        model = key2._new_model(mode, token_count=5).to(device)
        model.load_state_dict(torch.load(key_cfg[mode]['checkpoint'], map_location=device, weights_only=True)); model.eval()
        loaded[mode] = model

    def score_split(split: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
        # Score definitions and fixed weights are the same KEY2-F/KEYGRID
        # checkpoints used by EX1; this function adds no learned key model.
        prepared = key_grid._prepare_vector_sources(split)
        vectors = key_grid.make_vector_features(prepared, 'history_state')
        if 'tokens' in split:
            pooled = key2._pool_tokens(split['tokens'], split['groups'])
        else:
            pooled = None
        part = dict(split)
        if pooled is not None: part['pooled'] = pooled
        score_by_mode = {}
        for label, mode in (('D0_P0_top1', 'p0'), ('D1_P5_top1', 'p5'), ('D3_E_VLM_H_top1', 'e_vlm_h')):
            sc, _ = key2._score_model(loaded[mode], mode, bank, bi, part, device)
            score_by_mode[label] = sc
        score_by_mode['D5_HS_top1'] = key_grid._score_vectors(d5, vectors, bank_vec, device=device)
        # Exclude every candidate from the query's task from every ranker.
        source = np.asarray(bank['source'], np.int64)
        qkey = np.asarray(split['key'], np.int64)
        for i, key in enumerate(qkey):
            own_task = source[:, 0] == key[0]
            for sc in score_by_mode.values(): sc[i, own_task] = -np.inf
        target = np.asarray(split['actions'], np.float32)
        rows = {'key': qkey, 'target': target, 'task': np.asarray(split['task'], np.int32),
                'episode': np.asarray(split['episode'], np.int64),
                'key2_row_ids': np.asarray(split.get('row', np.full(len(qkey), -1)), np.int64),
                'row_ids': np.asarray(split.get('row', np.full(len(qkey), -1)), np.int64)}
        for label, sc in score_by_mode.items():
            ids = np.argmax(sc, axis=1).astype(np.int32)
            rows[label + '_scores'] = sc.astype(np.float32)
            rows[label + '_unit_ids'] = ids
            rows[label + '_draft'] = candidate_actions[ids].astype(np.float32)
        return rows

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = {}
    cached_splits = {}
    for label, split in (('memory', train), ('dev', dev), ('learned', learned), ('unseen', unseen)):
        data = score_split(split, label)
        cached_splits[label] = data
        np.savez_compressed(output / f'{label}_retrieval.npz', **data)
        records[label] = {'queries': int(len(data['key'])),
                          'key_modes': ['D0_P0_top1', 'D1_P5_top1', 'D3_E_VLM_H_top1', 'D5_HS_top1']}
    # Pi0.5 action outputs are needed only as the fallback/gripper and for
    # confidence features. Reuse fixed PI0REF output for Eval; generate Dev and
    # memory anchors from their exact cached source rows where available.
    learned_ref = np.load(root / 'pi05-reference-01/predictions/learned.npz', allow_pickle=False)
    unseen_ref = np.load(root / 'pi05-reference-01/predictions/unseen_cached.npz', allow_pickle=False)
    reference_selection=np.load(root/'pi05-reference-01/selection.npz',allow_pickle=False)
    for split_name,reference in (('learned',learned_ref),('unseen',unseen_ref)):
        reference_keys=np.stack([reference_selection[f'{split_name}_{name}']
                                 for name in ('task','episode','frame')],axis=1)
        cached_splits[split_name]['pi05']=join_reference_by_key(
            cached_splits[split_name]['key'],cached_splits[split_name]['target'],reference_keys,
            reference['target'],reference['actions'])
    # All candidate drafts presented to confidence or action-Expert paths use
    # the same π0.5 gripper as training and runtime.
    for split_name in ('learned','unseen'):
        normal=np.asarray(cached_splits[split_name]['pi05'],np.float32)
        for name in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):
            cached_splits[split_name][name+'_draft'][:,:,6]=normal[:,:,6]
    base_run = exp3._module(HERE.parent / 'pi05_memory/run.py', 'ex2_query_policy')
    policy, _ = base_run.load_policy(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    rows40 = base_run.load_rows(run_root)
    rows90 = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
              for p in (root / 'unseen-rows-02').glob('*.npy')}
    # Match unseen Dev query features to their exact row IDs, then use fixed
    # noise to get the π0.5 fallback chunks.
    def causal_source_ids(source_rows, keys):
        grouped={}
        for i,(t,e,f) in enumerate(zip(source_rows['task'],source_rows['episode'],source_rows['frame'],strict=True)):
            grouped.setdefault((int(t),int(e)),[]).append((int(f),i))
        out=[]
        for key in np.asarray(keys,np.int64):
            options=[(f,i) for f,i in grouped.get((int(key[0]),int(key[1])),[]) if f<=int(key[2])]
            if not options:raise KeyError(f'no causal source row at/before {tuple(map(int,key))}')
            out.append(max(options)[1])
        return np.asarray(out,np.int64)
    # Exact-frame image/state observations are reconstructed from source demos;
    # stride-10 floor-mapping is forbidden for stride-5 Dev and sparse memory keys.
    dev_exact=_exact_episode_observations(cached_splits['dev']['key'],domain='libero90',source_rows=rows90)
    dev_ids=np.arange(len(cached_splits['dev']['key']),dtype=np.int64)
    cached_splits['dev']['row_ids']=dev_ids
    cached_splits['dev']['pi05']=_normal_for_rows(base_run,policy,dev_exact,dev_ids,70_000_000+dev_ids)
    for name in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):
        cached_splits['dev'][name+'_draft'][:,:,6]=cached_splits['dev']['pi05'][:,:,6]
    np.savez_compressed(output/'dev_exact_observations.npz',**dev_exact)
    memory_exact=_exact_episode_observations(cached_splits['memory']['key'],domain='libero40',source_rows=rows40)
    mem_ids=np.arange(len(cached_splits['memory']['key']),dtype=np.int64)
    cached_splits['memory']['row_ids']=mem_ids
    cached_splits['memory']['pi05']=_normal_for_rows(base_run,policy,memory_exact,mem_ids,80_000_000+mem_ids)
    for name in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):
        cached_splits['memory'][name+'_draft'][:,:,6]=cached_splits['memory']['pi05'][:,:,6]
    cached_splits['unseen']['row_ids']=causal_source_ids(rows90,cached_splits['unseen']['key'])
    for label in ('memory', 'dev', 'learned', 'unseen'):
        np.savez_compressed(output / f'{label}_retrieval.npz', **cached_splits[label])
    manifest = {'status': 'complete_smoke' if smoke else 'complete', 'candidate_count': int(len(candidate_actions)),
                'candidate_contract': 'full KEY2/CMP3 ordered bank; every same-task candidate excluded per query',
                'dev_and_memory_pi05_observations': 'source demonstration exact frame; no causal stride-10 floor',
                'D5_checkpoint': str(d5_rec['checkpoint']), 'D5_dev_top1_arm6_mse': float(d5_rec['dev_top1_arm6_mse']),
                'fixed_KEY2_checkpoints': {k: v['checkpoint'] for k, v in key_cfg.items()}, 'splits': records}
    _write(output / 'retrieval_manifest.json', manifest)
    return manifest


def _normal_for_rows(base, policy, rows, row_ids, row_keys):
    """Fixed-noise frozen π0.5 chunks for confidence features / fallback."""
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    row_ids = np.asarray(row_ids, np.int64); row_keys = np.asarray(row_keys, np.int64)
    fn = module_jit(policy.sample_actions, static_argnames=('num_steps',))
    noise = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101), int(k)), (10, 32)))
                      for k in row_keys]).astype(np.float32)
    out=[]
    for start in range(0, len(row_ids), 8):
        stop=min(start+8,len(row_ids));ids=row_ids[start:stop]
        out.append(np.asarray(fn(jax.random.key(0),base.obs_batch(rows,ids),num_steps=10,
                                 noise=jnp.asarray(noise[start:stop])),np.float32))
        if (start//8)%64==0 or stop==len(row_ids):
            print(json.dumps({'stage':'pi05_exact_rows','done':stop,'total':len(row_ids)}),flush=True)
    return np.concatenate(out) if out else np.empty((0,10,32),np.float32)


def _exact_episode_observations(keys, *, domain, source_rows, root=ROOT):
    """Load exact-frame images/state for sparse memory or stride-5 Dev keys.

    Query caches hold latent/action windows at stride 5, while the generic
    observation cache is stride 10. Reconstruct observations from the original
    demonstration files so π0.5 sees the query frame itself.
    """
    import io, json
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.shared.normalize import load
    checkpoint=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
    normfiles=list((checkpoint/'assets').rglob('norm_stats.json'))
    if len(normfiles)!=1:raise ValueError(f'expected one normalization file, got {normfiles}')
    processor=[LiberoInputs(ModelType.PI05),transforms.Normalize(load(normfiles[0].parent),use_quantiles=True),
               transforms.ResizeImages(224,224),transforms.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),
               transforms.PadStatesAndActions(32)]
    keys=np.asarray(keys,np.int64)
    if keys.ndim!=2 or keys.shape[1]!=3:raise ValueError(keys.shape)
    if domain=='libero40':
        manifest=json.loads((Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')/'data_manifest.json').read_text())
        recs={(int(x['task']),int(x['episode'])):x for x in manifest['episodes']}
        tasknames={int(k):str(v) for k,v in manifest['tasks'].items()}
    else:
        manifest=json.loads((root/'unseen-rows-02/manifest.json').read_text())
        recs={(int(x['task_id']),int(x['episode'])):x for x in manifest['source_records']}
    parquet_cache={};hdf_cache={};result={k:[] for k in ('image_base','image_wrist','state','prompt','prompt_mask')}
    for task,episode,frame in keys:
        rec=recs[(int(task),int(episode))]
        if domain=='libero40':
            path=Path(rec['path'])
            if path not in parquet_cache:parquet_cache[path]=pq.read_table(path,columns=['image','wrist_image','state'])
            table=parquet_cache[path];idx=int(frame)
            imgs=[]
            for col in ('image','wrist_image'):
                obj=table[col][idx].as_py()
                with Image.open(io.BytesIO(obj['bytes'])) as im:imgs.append(np.asarray(im.convert('RGB')))
            state=np.asarray(table['state'][idx].as_py(),np.float32);prompt=tasknames[int(task)]
        else:
            import h5py
            path=Path(rec['source_hdf5'])
            if path not in hdf_cache:hdf_cache[path]=h5py.File(path,'r')
            obs=hdf_cache[path]['data'][rec['source_demo']]['obs'];idx=min(int(frame)*2,len(obs['agentview_rgb'])-1)
            imgs=[np.asarray(obs['agentview_rgb'][idx],np.uint8)[::-1,::-1],
                  np.asarray(obs['eye_in_hand_rgb'][idx],np.uint8)[::-1,::-1]]
            state=np.concatenate([np.asarray(obs['ee_states'][idx],np.float32),np.asarray(obs['gripper_states'][idx],np.float32)])
            prompt=str(rec.get('instruction',rec.get('task_name','')))
        item={'observation/image':imgs[0],'observation/wrist_image':imgs[1],'observation/state':state,'prompt':prompt}
        for transform in processor:item=transform(item)
        result['image_base'].append(np.asarray(item['image']['base_0_rgb'],np.uint8))
        result['image_wrist'].append(np.asarray(item['image']['left_wrist_0_rgb'],np.uint8))
        result['state'].append(np.asarray(item['state'],np.float32))
        result['prompt'].append(np.asarray(item['tokenized_prompt'],np.int32))
        result['prompt_mask'].append(np.asarray(item['tokenized_prompt_mask'],bool))
        if (len(result['state'])%64)==0 or len(result['state'])==len(keys):
            print(json.dumps({'stage':f'exact_observations_{domain}','done':len(result['state']),'total':len(keys)}),flush=True)
    for f in hdf_cache.values():f.close()
    result['row_keys']=keys.copy()
    exact={k:np.asarray(v) for k,v in result.items()}
    if len(keys) and not np.isfinite(exact['state']).all():raise ValueError('nonfinite exact observation state')
    return exact


_BRIDGE_FN_CACHE={}


def _bridge_with_row_t0(model, obs, drafts, noise, t0, *, batch=8):
    """Evaluate the fixed 10-step bridge with one continuous t0 per row."""
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    key=id(model)
    fn=_BRIDGE_FN_CACHE.get(key)
    if fn is None:
        fn = module_jit(model.sample_bridge, static_argnames=('num_steps',));_BRIDGE_FN_CACHE[key]=fn
    d=np.asarray(drafts,np.float32);n=np.asarray(noise,np.float32);ts=np.asarray(t0,np.float32).reshape(-1)
    if d.ndim!=3 or d.shape[1:]!=(10,32) or n.shape!=d.shape or ts.shape!=(len(d),):
        raise ValueError(f'bridge rows are misaligned: {d.shape}/{n.shape}/{ts.shape}')
    out=[]
    for start in range(0,len(d),int(batch)):
        stop=min(start+int(batch),len(d));sl=slice(start,stop)
        val=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(d[sl]),jnp.asarray(ts[sl]),
               num_steps=10,noise=jnp.asarray(n[sl]))
        out.append(np.asarray(val,np.float32))
    return np.concatenate(out) if out else np.empty_like(d)


def fit_memory_confidence(retrieval_dir: Path, output: Path, *, seed: int = 20261104,
                          threads: int = 8, smoke: bool=False) -> dict[str, Any]:
    """Fit ê only from memory-side held-out episodes; Dev/Eval labels are absent."""
    import torch
    torch.set_num_threads(int(threads))
    retrieval_dir=Path(retrieval_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(retrieval_dir/'memory_retrieval.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    task=np.asarray(data['task'],np.int64);episode=np.asarray(data['episode'],np.int64)
    fold=np.mod(task*1_000_003+episode*97,5)
    fit=fold!=0;held=fold==0
    if smoke and (fit.sum()<8 or held.sum()<2):
        # Tiny smoke caches can contain too few episodes for a five-fold split;
        # still keep whole episodes disjoint and label this diagnostic clearly.
        pairs=np.asarray([f'{t}:{e}' for t,e in zip(task,episode,strict=True)])
        uniq=np.unique(pairs);held_pairs=set(uniq[::max(2,len(uniq)//2 or 2)][:max(1,len(uniq)//4)].tolist())
        held=np.asarray([x in held_pairs for x in pairs]);fit=~held
    if fit.sum()<(8 if smoke else 100) or held.sum()<(2 if smoke else 20):raise ValueError('insufficient disjoint memory episodes for confidence regression')
    x=[];y=[]
    for mode in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):
        draft=np.asarray(data[mode+'_draft'],np.float32)
        pi=np.asarray(data['pi05'],np.float32)
        scores=np.asarray(data[mode+'_scores'],np.float32)
        features=confidence_features(draft,pi,scores)
        error=np.square(draft[:,:,:6]-data['target'][:,:,:6]).sum((1,2))/70.
        x.append(features[fit]);y.append(error[fit])
    xfit=np.concatenate(x);yfit=np.concatenate(y)
    model=fit_confidence_regressor(xfit,yfit,seed=seed,steps=40 if smoke else 800,device='cpu')
    # Held-out fold is never used to update weights; it measures calibration
    # generalization across memory episodes.
    xv=[];yv=[]
    for mode in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):
        draft=np.asarray(data[mode+'_draft'],np.float32);pi=np.asarray(data['pi05'],np.float32)
        xv.append(confidence_features(draft,pi,data[mode+'_scores'])[held])
        yv.append(np.square(draft[held,:,:6]-data['target'][held,:,:6]).sum((1,2))/70.)
    xv=np.concatenate(xv);yv=np.concatenate(yv);pred=predict_error(model,xv)
    def corr(a,b):
        ra=np.argsort(np.argsort(a,kind='stable'),kind='stable');rb=np.argsort(np.argsort(b,kind='stable'),kind='stable')
        return float(np.corrcoef(ra,rb)[0,1]) if len(a)>1 else 0.
    checkpoint=output/'confidence_head.pt'
    torch.save({'state_dict':model.state_dict(),'feature_names':['log_draft_pi05_mse6','score_margin12','normalized_score_entropy','log_arm_norm_ratio'],
                'target':'log(arm6 MSE + 1e-6)','heldout_fold':'(task*1000003 + episode*97) mod 5 == 0'},checkpoint)
    result={'status':'complete_smoke' if smoke else 'complete','fit_rows_per_mode':int(fit.sum()),'heldout_rows_per_mode':int(held.sum()),
        'fit_episodes':int(len(set(zip(task[fit].tolist(),episode[fit].tolist())))),
        'heldout_episodes':int(len(set(zip(task[held].tolist(),episode[held].tolist())))),
        'heldout_modes':4,'heldout_spearman':corr(yv,pred),'heldout_log_error_mae':float(np.mean(np.abs(np.log(yv+1e-6)-np.log(pred+1e-6)))),
        'heldout_true_error_median':float(np.median(yv)),'heldout_pred_error_median':float(np.median(pred)),
        'checkpoint':str(checkpoint),'uses_dev_or_eval_labels':False}
    _write(output/'confidence_fit.json',result)
    return result


def write_dev_selection_cache(retrieval_dir: Path, output: Path) -> dict[str, Any]:
    """Freeze the actual D1 P5 top-1 Dev draft for Expert checkpoint selection."""
    source=Path(retrieval_dir)/'dev_retrieval.npz';output=Path(output)
    if output.exists():raise FileExistsError(output)
    with np.load(source,allow_pickle=False) as z:
        keys=np.asarray(z['key'],np.int64);draft=np.asarray(z['D1_P5_top1_draft'],np.float32)
        target=np.asarray(z['target'],np.float32);pi=np.asarray(z['pi05'],np.float32)
        row_ids=np.asarray(z['row_ids'],np.int64)
    if row_ids.shape!=(len(keys),) or np.any(row_ids<0):raise ValueError('Dev source row IDs are missing')
    np.savez_compressed(output,D1_P5_top1=draft,target=target,pi05=pi,row_ids=row_ids,row_keys=keys)
    return {'status':'complete','rows':len(keys),'tasks':sorted(set(keys[:,0].tolist())),
            'draft':'locked actual KEY2-F P5 top-1','target_used_for_selection_only':True,
            'output':str(output)}


def _load_expert_for_eval(base, checkpoint: Path):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    model,_=base.load_policy(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    with ocp.PyTreeCheckpointer() as loader:state=loader.restore(Path(checkpoint))
    live=nnx.state(model,filt);live.replace_by_pure_dict(jax.tree.map(jnp.asarray,state));nnx.update(model,live)
    return model,base,filt


def _full_actions(actions7):
    a=np.asarray(actions7,np.float32)
    if a.shape[1:]!=(10,7):raise ValueError(a.shape)
    out=np.zeros((len(a),10,32),np.float32);out[:,:,:7]=a
    return out


def join_reference_by_key(query_keys: np.ndarray, query_target: np.ndarray,
                          reference_keys: np.ndarray, reference_target: np.ndarray,
                          reference_actions: np.ndarray) -> np.ndarray:
    """Join fixed π0.5 predictions by (task, episode, frame), never cache row ID."""
    qk=np.asarray(query_keys,np.int64);rk=np.asarray(reference_keys,np.int64)
    qt=np.asarray(query_target,np.float32);rt=np.asarray(reference_target,np.float32)
    ra=np.asarray(reference_actions,np.float32)
    if qk.ndim!=2 or qk.shape[1]!=3 or rk.ndim!=2 or rk.shape[1]!=3:
        raise ValueError(f'expected [N,task/episode/frame] keys, got {qk.shape}/{rk.shape}')
    if qt.shape!=(len(qk),10,7) or rt.shape!=(len(rk),10,7) or ra.shape!=(len(rk),10,7):
        raise ValueError(f'reference action shapes do not align: {qt.shape}/{rt.shape}/{ra.shape}')
    rmap={tuple(map(int,k)):i for i,k in enumerate(rk)}
    if len(rmap)!=len(rk):raise ValueError('duplicate key in π0.5 reference cache')
    try:ix=np.asarray([rmap[tuple(map(int,k))] for k in qk],np.int64)
    except KeyError as exc:raise AssertionError(f'query key missing from π0.5 reference cache: {exc}') from exc
    if not np.allclose(qt,rt[ix],atol=1e-6,rtol=0):
        raise AssertionError('π0.5 reference targets differ from query targets')
    return ra[ix].copy()


def select_confidence_policy(retrieval_dir: Path, train_dir: Path, confidence_dir: Path,
                             output: Path, *, threads: int=8, mem_fraction: float=.25,
                             smoke: bool=False) -> dict[str,Any]:
    """Select Expert A/B, c and fallback threshold using only fixed Dev rows."""
    import sys, torch, jax
    torch.set_num_threads(int(threads));torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    sys.path[:0]=[str(HERE),str(HERE.parent/'pi05_memory')]
    import skill_expert_refine as exp3
    import expert_ex1 as ex1
    base=exp3._module(HERE.parent/'pi05_memory/run.py','ex2_select_base')
    with np.load(Path(retrieval_dir)/'dev_retrieval.npz',allow_pickle=False) as z:dev={k:z[k] for k in z.files}
    if smoke:
        n=min(8,len(dev['key']));dev={k:v[:n] if isinstance(v,np.ndarray) and len(v)==len(dev['key']) else v for k,v in dev.items()}
    train=json.loads((Path(train_dir)/'train.json').read_text())
    # One checkpoint is selected before inspecting any Eval row. A/B is the
    # only pair of candidates, and D1 actual KEY2-F retrieval is the Dev task.
    candidates=[]
    for condition in ('A','B'):
        best=train['conditions'][condition]['selected']
        candidates.append({'condition':condition,**best})
    selected=min(candidates,key=lambda x:(x['mse6'],x['condition']))
    conf=torch.load(Path(confidence_dir)/'confidence_head.pt',map_location='cpu',weights_only=True)
    reg=_new_confidence_model();reg.load_state_dict(conf['state_dict']);reg.eval()
    normal=np.asarray(dev['pi05'],np.float32)
    drafts_dev={m:np.asarray(dev[m+'_draft'],np.float32) for m in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1')}
    error_hat={m:predict_error(reg,confidence_features(d,normal,dev[m+'_scores'])) for m,d in drafts_dev.items()}
    ids=np.asarray(dev['row_ids'],np.int64)
    if ids.shape!=(len(dev['key']),) or np.any(ids<0):raise AssertionError('Dev source row IDs missing')
    with np.load(Path(retrieval_dir)/'dev_exact_observations.npz',allow_pickle=False) as z:
        dev_obs_rows={k:z[k] for k in z.files}
    if len(dev_obs_rows['state'])!=len(dev['key']) or not np.array_equal(ids,np.arange(len(ids))) or not np.array_equal(dev_obs_rows['row_keys'],dev['key']):
        raise AssertionError('exact-frame Dev observation rows are misaligned')
    if not np.array_equal(np.asarray(dev['key'])[:,0],np.asarray(dev['task'])):
        raise AssertionError('Dev query identities disagree with exact-frame observations')
    model,_,_=_load_expert_for_eval(base,Path(selected['checkpoint']))
    obs=base.obs_batch(dev_obs_rows,ids)
    noise=exp3._seed_noise(ids+70_000_000)
    # To keep the selection stable and relevant to all closed-loop retrievals,
    # optimize the equal-weight D1/D3/D5 Dev MSE for each (c, fallback q).
    target=np.asarray(dev['target'],np.float32)
    all_e=np.concatenate([error_hat[m] for m in ('D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1')])
    thresholds=[None]+[float(np.quantile(all_e,q)) for q in (.5,.7,.8,.9,.95)]
    curves={}
    for c in (.5,1.,1.5,2.,3.):
        for mode,draft7 in (('D1',drafts_dev['D1_P5_top1']),('D3',drafts_dev['D3_E_VLM_H_top1']),('D5',drafts_dev['D5_HS_top1'])):
            keymode={'D1':'D1_P5_top1','D3':'D3_E_VLM_H_top1','D5':'D5_HS_top1'}[mode]
            eh=error_hat[keymode]
            for threshold in thresholds:
                fallback=np.zeros(len(eh),bool) if threshold is None else eh>threshold
                for low_floor in (.05,0.):
                    row_low=np.full(len(eh),.05,np.float32)
                    row_low[(eh<=threshold) if threshold is not None else np.ones(len(eh),bool)]=low_floor
                    t=select_t0(eh,c,low=row_low)
                    raw=_bridge_with_row_t0(model,obs,_full_actions(draft7),noise,t,batch=8)
                    raw=ex1.restore_pi05_gripper(raw,normal)
                    routed=raw.copy();routed[fallback]=normal[fallback]
                    mse=np.square(routed[:,:,:6]-target[:,:,:6]).sum((1,2))/70.
                    curves[(float(c),threshold,float(low_floor),mode)]=mse
    # Aggregate the 3 intended modes without giving any mode a validation-only advantage.
    grid={}
    for c in (.5,1.,1.5,2.,3.):
        for threshold in thresholds:
            for low_floor in (.05,0.):
                merged=np.concatenate([curves[(float(c),threshold,float(low_floor),m)] for m in ('D1','D3','D5')])
                grid[(float(c),threshold,float(low_floor))]=merged
    selected_grid=select_dev_policy(grid,np.tile(np.square(normal[:,:,:6]-target[:,:,:6]).sum((1,2))/70.,3))
    sel=selected_grid['selected'];threshold=sel['threshold'];c=sel['scale'];low_floor=sel['high_confidence_t0_floor']
    fallbacks={m:(np.zeros(len(error_hat[m]),bool) if threshold is None else error_hat[m]>threshold) for m in error_hat}
    dev_spearman={}
    for mode,d in drafts_dev.items():
        tr=np.square(d[:,:,:6]-target[:,:,:6]).sum((1,2))/70.;pr=error_hat[mode]
        rt=np.argsort(np.argsort(tr,kind='stable'),kind='stable');rp=np.argsort(np.argsort(pr,kind='stable'),kind='stable')
        dev_spearman[mode]=float(np.corrcoef(rt,rp)[0,1]) if len(rt)>1 else 0.
    result={'status':'complete_smoke' if smoke else 'complete','expert':selected,
        'schedule':{'scale_c':c,'fallback_error_threshold':threshold,'clip':[.05,.6],
                    'high_confidence_t0_floor':low_floor,
                    'rule':'high confidence (predicted error <= threshold) uses clip(c*sqrt(error), selected floor, .6); otherwise floor .05; if predicted error > threshold use π0.5'},
        'dev_modes':['D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'],
        'dev_rows':len(dev['key']),'dev_fallback_fraction_by_mode':{m:float(f.mean()) for m,f in fallbacks.items()},
        'dev_predicted_error_median_by_mode':{m:float(np.median(e)) for m,e in error_hat.items()},
        'dev_confidence_error_spearman':dev_spearman,
        'dev_pi05_arm6_mse':selected_grid['pi05_mse6'],'grid':selected_grid['grid'],
        'checkpoint_selection':'min D1 fixed actual-retrieval Dev arm6 MSE across A/B; Eval unused'}
    _write(output,result);return result


def _bootstrap_mean(values, episodes, *, seed=4101, replicates=5000):
    x=np.asarray(values,np.float64);ep=np.asarray(episodes)
    unique,inv=np.unique(ep,return_inverse=True)
    sums=np.bincount(inv,weights=x);counts=np.bincount(inv)
    rng=np.random.default_rng(seed);draw=rng.integers(len(unique),size=(replicates,len(unique)))
    means=sums[draw].sum(1)/counts[draw].sum(1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def evaluate_offline(retrieval_dir: Path, train_dir: Path, confidence_dir: Path,
                     schedule_path: Path, output: Path, *, threads=8,
                     mem_fraction=.25, batch=8, smoke=False) -> dict[str,Any]:
    """Offline EX2 evaluation, with Dev-locked schedule and fixed Eval rows."""
    import sys,torch,jax
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    torch.set_num_threads(int(threads));torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    sys.path[:0]=[str(HERE),str(HERE.parent/'pi05_memory')]
    import skill_expert_refine as exp3
    import expert_ex1 as ex1
    base=exp3._module(HERE.parent/'pi05_memory/run.py','ex2_offline_base')
    schedule=json.loads(Path(schedule_path).read_text())
    condition=schedule['expert']['condition'];chosen=schedule['expert']
    checkpoint=Path(chosen['checkpoint'])
    conf=torch.load(Path(confidence_dir)/'confidence_head.pt',map_location='cpu',weights_only=True)
    reg=_new_confidence_model();reg.load_state_dict(conf['state_dict']);reg.eval()
    learned_rows=exp3._module(HERE.parent/'pi05_memory/run.py','ex2_offline_rows').load_rows(exp3.RUN)
    unseen_rows=exp3._load_rows(exp3.UNSEEN)
    srcs={'learned':learned_rows,'unseen':unseen_rows}
    train=json.loads((Path(train_dir)/'train.json').read_text())
    model,_,_=_load_expert_for_eval(base,checkpoint)
    # EX1 helper defines exactly the action-Expert parameter filter.
    filt=nnx.All(nnx.Param, __import__('openpi.shared.nnx_utils',fromlist=['PathRegex']).PathRegex(
        r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    root=ROOT/'expert-ex1-01/production-03/scope-change-01/eval-drafts'
    result={'status':'complete_smoke' if smoke else 'complete','selected_expert':chosen,
            'schedule':schedule['schedule'],'splits':{},'selection_uses_eval':False,
            'metric':'arm6 MSE = sum over 10x6 squared normalized error / 70; active7 adds π0.5 gripper'}
    for split_name in ('learned','unseen'):
        path=Path(retrieval_dir)/f'{split_name}_retrieval.npz'
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        old_path=root/f'{split_name}_drafts.npz'
        with np.load(old_path,allow_pickle=False) as z:old={k:z[k] for k in z.files}
        old_keymap={tuple(map(int,k)):i for i,k in enumerate(old['row_keys'])}
        try:
            old_ix=np.asarray([old_keymap[tuple(map(int,k))] for k in data['key']],np.int64)
        except KeyError as exc:
            raise AssertionError(f'{split_name} contains a row outside the frozen EX1 evaluation set: {exc}') from exc
        if not np.array_equal(data['target'],old['target'][old_ix]):
            raise AssertionError(f'{split_name} target rows differ from frozen EX1 selection')
        data['hard']=np.asarray(old['hard'][old_ix],bool)
        if smoke:
            n=min(8,len(data['key']))
            data={k:(v[:n] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(old['row_keys']) else v) for k,v in data.items()}
        ids=np.asarray(data['row_ids'],np.int64);source=srcs[split_name]
        obs=base.obs_batch(source,ids);target=np.asarray(data['target'],np.float32);normal=np.asarray(data['pi05'],np.float32)
        if normal.shape[-1]==32:normal=normal[:,:,:7]
        if target.shape!=(len(ids),10,7) or normal.shape!=target.shape:raise AssertionError('action row shape/order mismatch')
        scores_by_mode={m:np.asarray(data[m+'_scores'],np.float32) for m in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1')}
        drafts={m:np.asarray(data[m+'_draft'],np.float32).copy() for m in scores_by_mode}
        for d7 in drafts.values():d7[:,:,6]=normal[:,:,6]
        error_hat={m:predict_error(reg,confidence_features(drafts[m],normal,scores_by_mode[m])) for m in scores_by_mode}
        rowkeys=np.asarray(data['key'],np.int64);noise=exp3._seed_noise(ids+(0 if split_name=='learned' else 50_000_000))
        predictions={'pi05':normal}
        for mode,d7 in drafts.items():
            predictions[mode+'_draft']=d7
            d32=np.zeros((len(ids),10,32),np.float32);d32[:,:,:7]=d7
            for fixed_t in (.1,.2):
                p=_bridge_with_row_t0(model,obs,d32,noise,np.full(len(ids),fixed_t,np.float32),batch=batch)
                predictions[f'{mode}_expert_t0_{fixed_t:.1f}']=ex1.restore_pi05_gripper(p,normal)
            feat=confidence_features(d7,normal,scores_by_mode[mode]);eh=predict_error(reg,feat)
            threshold=schedule['schedule']['fallback_error_threshold']
            low_floor=float(schedule['schedule'].get('high_confidence_t0_floor',.05))
            low=np.full(len(eh),.05,np.float32)
            low[(eh<=float(threshold)) if threshold is not None else np.ones(len(eh),bool)]=low_floor
            tt=select_t0(eh,float(schedule['schedule']['scale_c']),low=low)
            p=_bridge_with_row_t0(model,obs,d32,noise,tt,batch=batch)
            predictions[mode+'_expert_confidence']=ex1.restore_pi05_gripper(p,normal)
            fallback=schedule['schedule']['fallback_error_threshold']
            if fallback is not None:p[eh>float(fallback)]=_full_actions(normal)[eh>float(fallback)]
            predictions[mode+'_expert_confidence_fallback']=ex1.restore_pi05_gripper(p,normal)
        # References: hold-last and privileged oracle use only for reporting.
        hold=np.zeros_like(normal)
        hist=np.asarray(old['history'][old_ix],np.float32);hm=np.asarray(old['history_mask'][old_ix],bool)
        for i in range(len(hold)):
            prev=hist[i,hm[i]]
            if len(prev):hold[i]=np.repeat(prev[-1][None,:7],10,axis=0)
        predictions['hold_last']=hold
        bank=exp3.make_memory_bank(learned_rows,np.asarray(np.load(exp3.RUN/'memory_row_ids.npy',allow_pickle=False),np.int64))
        units=exp3.unit_window(bank['actions'])
        oracle_ids,_=exp3.nearest_unit_ids(target,units,bank['task'])
        oracle=np.zeros_like(normal);oracle[:,:,:6]=units[oracle_ids,:,:6];oracle[:,:,6]=normal[:,:,6]
        predictions['oracle_draft']=oracle
        od=np.zeros((len(ids),10,32),np.float32);od[:,:,:7]=oracle
        # Use the fixed selected Expert and the Dev-selected t0 rule for oracle bound.
        err=np.square(oracle[:,:,:6]-target[:,:,:6]).sum((1,2))/70.
        op=_bridge_with_row_t0(model,obs,od,noise,select_t0(err,float(schedule['schedule']['scale_c'])),batch=batch)
        predictions['oracle_expert']=ex1.restore_pi05_gripper(op,normal)
        split_ep=rowkeys[:,0]*10_000_000+rowkeys[:,1];hard=np.asarray(data['hard'],bool)
        metrics={}
        for name,pred in predictions.items():
            p=np.asarray(pred,np.float32)
            arm=np.square(p[:,:,:6]-target[:,:,:6]).sum((1,2))/70.
            active=np.square(p[:,:,:7]-target[:,:,:7]).sum((1,2))/70.
            metrics[name]={}
            for label,mask in (('all',np.ones(len(hard),bool)),('hard',hard)):
                ix=np.flatnonzero(mask)
                if len(ix):metrics[name][label]={'arm6_mse':float(arm[ix].mean()),'arm6_ci95':_bootstrap_mean(arm[ix],split_ep[ix]),'active7_mse':float(active[ix].mean()),'rows':len(ix)}
        paired={}
        for mode in drafts:
            for variant in ('expert_t0_0.1','expert_t0_0.2','expert_confidence','expert_confidence_fallback'):
                name=f'{mode}_{variant}'
                if name not in predictions:continue
                arm=np.square(predictions[name][:,:,:6]-target[:,:,:6]).sum((1,2))/70.
                for refname in (mode+'_draft','pi05'):
                    ref=np.square(predictions[refname][:,:,:6]-target[:,:,:6]).sum((1,2))/70.
                    delta=arm-ref;paired[f'{name} - {refname}']={'all_mean':float(delta.mean()),'all_ci95':_bootstrap_mean(delta,split_ep),
                        'hard_mean':float(delta[hard].mean()) if np.any(hard) else None,
                        'hard_ci95':_bootstrap_mean(delta[hard],split_ep[hard]) if np.any(hard) else None}
        correlations={}
        for mode,d7 in drafts.items():
            truth=np.square(d7[:,:,:6]-target[:,:,:6]).sum((1,2))/70.;eh=error_hat[mode]
            rx=np.argsort(np.argsort(truth,kind='stable'),kind='stable');ry=np.argsort(np.argsort(eh,kind='stable'),kind='stable')
            correlations[mode]={'spearman':float(np.corrcoef(rx,ry)[0,1]),'true_error_median':float(np.median(truth)),'pred_error_median':float(np.median(eh))}
        predfile=Path(output)/f'{split_name}_predictions.npz';predfile.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(predfile,**predictions)
        result['splits'][split_name]={'rows':len(ids),'hard_rows':int(hard.sum()),'metrics':metrics,'paired_differences':paired,
             'confidence_error_spearman':correlations,'fallback_fraction':{m:float(np.mean(error_hat[m]>schedule['schedule']['fallback_error_threshold'])) if schedule['schedule']['fallback_error_threshold'] is not None else 0. for m in drafts},
             'prediction_file':str(predfile)}
    _write(Path(output)/'offline_results.json',result)
    return result


def repair_dev_checkpoint_selection(output: Path, *, threads: int = 8,
                                    mem_fraction: float = .25) -> dict[str, Any]:
    """Re-score every saved A/B checkpoint on fixed Dev with live NNX state.

    The first production training run exposed that ``module_jit`` freezes a
    module's parameters when constructed. Training used it for periodic Dev
    scoring, so its score curve reflected only the first evaluated state.
    This post-hoc selector restores each saved checkpoint into one live NNX
    module and evaluates it through ``nnx.jit`` with the module as an input.
    It only reads the fixed Dev rows and is run before any Eval scoring.
    """
    import sys, torch, jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    torch.set_num_threads(int(threads)); torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
    import skill_expert_refine as exp3
    base = exp3._module(HERE.parent / 'pi05_memory/run.py', 'ex2_select_repaired_base')
    with np.load(Path(output)/'dev-selection.npz', allow_pickle=False) as z:
        draft7 = np.asarray(z['D1_P5_top1'], np.float32)
        target = np.asarray(z['target'], np.float32)
        ids = np.asarray(z['row_ids'], np.int64)
    with np.load(Path(output)/'dev-selection.npz', allow_pickle=False) as z:
        dev_keys = np.asarray(z['row_keys'], np.int64)
    if len(ids) != len(draft7) or not np.array_equal(ids,np.arange(len(ids))):
        raise AssertionError('fixed Dev row identity mismatch during checkpoint selection')
    model, _ = base.load_policy(exp3.CHECKPOINT)
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    checkpointer = ocp.PyTreeCheckpointer()
    with np.load(Path(output)/'retrieval/dev_exact_observations.npz',allow_pickle=False) as z:
        dev_obs_rows={k:z[k] for k in z.files}
    if len(dev_obs_rows['state'])!=len(dev_keys) or not np.array_equal(dev_obs_rows['row_keys'],dev_keys):raise AssertionError('exact Dev observation identity mismatch')
    obs = base.obs_batch(dev_obs_rows, ids)
    d32 = np.zeros((len(ids), 10, 32), np.float32); d32[:, :, :7] = draft7
    noise = exp3._seed_noise(ids + 60_000_000)

    @nnx.jit
    def evaluate_live(current, ob, draft, starts, eps):
        return current.sample_bridge(jax.random.key(0), ob, draft, starts,
                                     num_steps=10, noise=eps)

    all_scores = {}
    for condition in ('A', 'B'):
        meta = json.loads((Path(output)/f'train-{condition}/train.json').read_text())
        if meta.get('status') != 'complete':
            raise RuntimeError(f'{condition} training has not completed')
        for item in meta['history']:
            ck = Path(output)/f'train-{condition}/checkpoints/{int(item["step"]):06d}'
            restored = checkpointer.restore(ck)
            state = nnx.state(model, filt)
            state.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored))
            nnx.update(model, state)
            curves = {}
            for c in (.5, 1., 1.5, 2., 3.):
                # Match EX1's fixed Dev selection rule: t0 from the actual
                # stored P5 draft's arm error, with no confidence fallback.
                t0 = select_t0(np.square(draft7[:, :, :6]-target[:, :, :6]).mean((1, 2)), c)
                pieces = []
                for start in range(0, len(ids), 8):
                    stop = min(start+8, len(ids)); sl = slice(start, stop)
                    pred = evaluate_live(model, jax.tree.map(lambda x: x[sl], obs),
                        jnp.asarray(d32[sl]), jnp.asarray(t0[sl]), jnp.asarray(noise[sl]))
                    pieces.append(np.asarray(pred, np.float32))
                pred = np.concatenate(pieces)
                curves[str(float(c))] = float(np.square(pred[:, :, :6]-target[:, :, :6]).sum((1, 2)).mean()/70.)
            all_scores[f'{condition}:{int(item["step"])}'] = {
                'condition': condition, 'step': int(item['step']), 'checkpoint': str(ck),
                'dev_mse6_by_scale': curves}
    winners = {}
    for condition in ('A', 'B'):
        eligible = [x for x in all_scores.values() if x['condition'] == condition]
        winner = min(eligible, key=lambda x: min(x['dev_mse6_by_scale'].values()))
        winner['scale'] = float(min(winner['dev_mse6_by_scale'], key=winner['dev_mse6_by_scale'].get))
        winner['mse6'] = float(winner['dev_mse6_by_scale'][str(winner['scale'])])
        winners[condition] = winner
    winner = min(winners.values(), key=lambda x: (x['mse6'], x['condition']))
    result = {'status': 'complete', 'selection_split': 'fixed LIBERO-90 Dev only',
        'selection_metric': 'actual P5 top-1 draft; arm6 MSE after bridge; no confidence fallback',
        'reason': 're-score all saved checkpoints because module_jit freezes the state at construction',
        'checkpoint_scores': all_scores, 'selected_by_condition': winners, 'selected': winner}
    _write(Path(output)/'dev-checkpoint-selection-repaired.json', result)
    return result


def aggregate_training(output: Path) -> dict[str,Any]:
    """Collect A/B training and corrected Dev-only checkpoint selection."""
    output=Path(output);conditions={}
    for name in ('A','B'):
        path=output/f'train-{name}/train.json'
        value=json.loads(path.read_text())
        if value.get('status')!='complete':raise RuntimeError(f'{name} training has not completed')
    repaired=repair_dev_checkpoint_selection(output)
    for name in ('A','B'):
        value=json.loads((output/f'train-{name}/train.json').read_text())
        conditions[name]={'selected':repaired['selected_by_condition'][name],
                          'training_selected_stale_module_jit_curve':value['selected'],
                          'history':value['history'],'fit_domains':value['fit_domains'],'rows':value['rows']}
    result={'status':'complete','conditions':conditions,'repaired_dev_selection':repaired['selected'],
            'checkpoint_selection':'all checkpoints re-scored on fixed LIBERO-90 Dev with live NNX state; no Eval rows'}
    _write(output/'train.json',result);return result


def run_smoke(output: Path, *, threads=8, mem_fraction=.10) -> dict[str,Any]:
    """End-to-end small retrieval→training→confidence→selection→offline smoke."""
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'smoke output is not empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    l90x=ROOT/'expert-refine-exp3b-01/l90x_rows'
    retrieval=output/'retrieval';samples=output/'rank-samples';confidence=output/'confidence'
    build_retrieval_cache(retrieval,device='cuda:0',threads=threads,mem_fraction=mem_fraction,smoke=True)
    write_dev_selection_cache(retrieval,output/'dev-selection.npz')
    build_rank_sample_cache(samples,l90x_dir=l90x,smoke=True,threads=threads)
    fit_memory_confidence(retrieval,confidence,threads=threads,smoke=True)
    for cond in ('A','B'):
        train_ex2(output,condition=cond,updates=12,batch=2,eval_every=6,threads=threads,
                  mem_fraction=mem_fraction,smoke=True)
    aggregate_training(output)
    schedule=select_confidence_policy(retrieval,output,confidence,output/'selected-schedule.json',
                                      threads=threads,mem_fraction=mem_fraction,smoke=True)
    offline=evaluate_offline(retrieval,output,confidence,output/'selected-schedule.json',output/'offline',
                             threads=threads,mem_fraction=mem_fraction,batch=2,smoke=True)
    result={'status':'complete','retrieval_queries':{k:v['queries'] for k,v in json.loads((retrieval/'retrieval_manifest.json').read_text())['splits'].items()},
            'rank_sample_manifest':str(samples/'rank_sample_manifest.json'),
            'confidence':json.loads((confidence/'confidence_fit.json').read_text()),
            'selected_schedule':schedule,'offline_status':offline['status']}
    _write(output/'smoke_results.json',result);return result


def render_report(results_path: Path, output: Path) -> str:
    """Render a concise EX2 report from production results and closed-loop JSON."""
    results_path=Path(results_path);output=Path(output)
    results=json.loads(results_path.read_text()) if results_path.exists() else {}
    lines=['# EX2：キーに依存しない運動 Expert と誤差条件付きデノイズ','',
        '## 原因と修正','',
        'production-04 は仕様不一致のため無効な比較値として保持する。学習候補の16.2%が同タスク別episode由来、Dev 217行中104行が5 step古いstride-10観測、memory信頼度学習も4,516行中3,715行で観測時点がずれていた。さらに学習下書きはgripper=0、オフラインは保存単位のgripper、閉ループはπ0.5 gripperだった。production-05では同タスク全除外、元デモからのquery frame完全一致観測、全段階のπ0.5 gripper統一を適用した。','']
    old=results.get('pre_fix_production04',{})
    if old:
        lines += ['production-04 の値（修正前、無効。比較のため保存）:','',
            '| 行 | 未学習 arm6 MSE | LIBERO-90 closed loop | LIBERO-10 closed loop |','|---|---:|---:|---:|']
        lines.append(f"| D5 draft | {old.get('d5_draft_mse','—')} | — | — |")
        lines.append(f"| D5 + Expert | {old.get('d5_expert_mse','—')} | {old.get('libero90_d5','—')} | {old.get('libero10_d5','—')} |")
        lines.append('')
    offline=results.get('offline',{})
    lines += ['## オフライン主表','', '| 学習分割 | 行 | 条件 | Arm6 MSE all [episode bootstrap 95%] | Arm6 MSE hard | Active7 MSE all |','|---|---:|---|---:|---:|---:|']
    for split,record in offline.get('splits',{}).items():
        for name in ('D0_P0_top1_draft','D0_P0_top1_expert_t0_0.1','D0_P0_top1_expert_t0_0.2','D0_P0_top1_expert_confidence','D0_P0_top1_expert_confidence_fallback',
                     'D1_P5_top1_draft','D1_P5_top1_expert_t0_0.1','D1_P5_top1_expert_t0_0.2','D1_P5_top1_expert_confidence','D1_P5_top1_expert_confidence_fallback',
                     'D3_E_VLM_H_top1_draft','D3_E_VLM_H_top1_expert_t0_0.1','D3_E_VLM_H_top1_expert_t0_0.2','D3_E_VLM_H_top1_expert_confidence','D3_E_VLM_H_top1_expert_confidence_fallback',
                     'D5_HS_top1_draft','D5_HS_top1_expert_t0_0.1','D5_HS_top1_expert_t0_0.2','D5_HS_top1_expert_confidence','D5_HS_top1_expert_confidence_fallback',
                     'pi05','hold_last','oracle_expert'):
            if name not in record.get('metrics',{}):continue
            m=record['metrics'][name].get('all',{});h=record['metrics'][name].get('hard',{})
            lines.append(f"| {split} | {record['rows']} | {name} | {m.get('arm6_mse',float('nan')):.4f} [{m.get('arm6_ci95','—')}] | {h.get('arm6_mse','—')} | {m.get('active7_mse',float('nan')):.4f} |")
    lines += ['', '対比較（Expert 条件−参照、arm6 MSE、episode bootstrap 95%）','']
    for split,record in offline.get('splits',{}).items():
        for name,value in record.get('paired_differences',{}).items():
            lines.append(f"- {split}/{name}: all {value['all_mean']:.4f} [{value['all_ci95'][0]:.4f}, {value['all_ci95'][1]:.4f}], hard {value['hard_mean'] if value['hard_mean'] is not None else '—'} [{value['hard_ci95'] if value['hard_ci95'] is not None else '—'}].")
    prior=results.get('prior_ex1_reference',{})
    if prior:
        lines += ['', '参考：EX1 の Expert A/B（旧比較の D1/P5 top-1）','', '| 分割 | 条件 | Arm6 MSE all | hard |','|---|---|---:|---:|']
        for split,rows in prior.items():
            for name,item in rows.items():
                allv=item.get('all',{}).get('arm6_mse',{}).get('mean');hardv=item.get('hard',{}).get('arm6_mse',{}).get('mean')
                if allv is not None:lines.append(f"| {split} | {name} | {allv:.4f} | {hardv:.4f} |")
    lines += ['', '## 閉ループ','', '未学習 LIBERO-90（10×10）・学習済み LIBERO-10（10×5）の試行別結果と π0.5 対応 McNemar を下表に記載。', '',
              '| スイート | 条件 | 成功 | π0.5 | 両方 | Expertのみ | π0.5のみ | McNemar p |','|---|---|---:|---:|---:|---:|---:|---:|']
    for suite,record in results.get('closed_loop',{}).items():
        for cond,v in record.get('conditions',{}).items():
            m=v.get('mcnemar',{});lines.append(f"| {suite} | {cond} | {v.get('successes',0)}/{v.get('completed',0)} | {v.get('baseline_successes','—')} | {m.get('both_success','—')} | {m.get('expert_only',m.get('memory_only','—'))} | {m.get('baseline_only','—')} | {m.get('p_exact','—')} |")
    lines += ['', '## 信頼度診断','']
    for split,record in offline.get('splits',{}).items():
        for mode,value in record.get('confidence_error_spearman',{}).items():lines.append(f"- {split}/{mode}: 真の下書き誤差との Spearman {value['spearman']:.3f}。")
    lines += ['', '## 条件と制約','',
        'Expert は保存済み可変長 DCT3 単位から腕6軸の action flow を学習し、検索キーを学習入力に使わない。π0.5 はグリッパーと信頼度特徴・fallback にのみ使う。信頼度器は LIBERO-40 メモリ内のエピソード holdout で学習し、t0 と π0.5 fallback は Dev だけで選択した。Eval 行動は採点にのみ使用。閉ループは π0.5 の既知方策上でのオフライン環境評価であり、独立した方策・多 seed の評価ではない。', '']
    text='\n'.join(lines);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text);return text


def finalize_results(run_dir: Path) -> dict[str,Any]:
    """Merge offline and suite results, with prior EX1 A/B as explicit references."""
    run_dir=Path(run_dir);offline_path=run_dir/'offline/offline_results.json'
    offline=json.loads(offline_path.read_text()) if offline_path.exists() else {}
    closed={}
    for suite in ('libero90','libero10'):
        path=run_dir/f'closed-loop/{suite}/results.json'
        if path.exists():closed[suite]=json.loads(path.read_text())
    old_path=OUT/'production-04/results.json';old_result={}
    if old_path.exists():
        old=json.loads(old_path.read_text());old_unseen=old.get('offline',{}).get('splits',{}).get('unseen',{}).get('metrics',{})
        old90=old.get('closed_loop',{}).get('libero90',{}).get('conditions',{}).get('EX2_D5_HS_top1_confidence',{})
        old10=old.get('closed_loop',{}).get('libero10',{}).get('conditions',{}).get('EX2_D5_HS_top1_confidence',{})
        old_result={'source':str(old_path),
            'd5_draft_mse':old_unseen.get('D5_HS_top1_draft',{}).get('all',{}).get('arm6_mse'),
            'd5_expert_mse':old_unseen.get('D5_HS_top1_expert_confidence',{}).get('all',{}).get('arm6_mse'),
            'libero90_d5':f"{old90.get('successes','—')}/{old90.get('completed','—')}",
            'libero10_d5':f"{old10.get('successes','—')}/{old10.get('completed','—')}"}
    prior={}
    prior_path=ROOT/'expert-ex1-01/production-03/scope-change-01/offline/offline_results.json'
    if prior_path.exists():
        prev=json.loads(prior_path.read_text())
        for split,key in (('learned','learned_eval'),('unseen','unseen_eval')):
            s=prev.get(key,{})
            prior[split]={}
            for name in ('D1_P5_top1_expert_A','D1_P5_top1_expert_B'):
                prior[split][name]=s.get('metrics',{}).get(name,{})
    result={'status':'complete' if offline.get('status')=='complete' and len(closed)==2 and all(x.get('status')=='complete' for x in closed.values()) else 'partial',
        'protocol':{'training':'action-rank sampled stored DCT3 units; no retrieval key input; arm6 flow loss; frozen VLM',
          'confidence':'memory episode holdout only; features from draft-vs-π0.5 distance, score margin, entropy, arm norm ratio',
          'closed_loop_seed':'same EX1/lib-closed rule: base 7; task initial-state index; replan 5; max 520',
          'subtask_generation':False,'l90x_missing_task_51':True},
        'offline':offline,'closed_loop':closed,'prior_ex1_reference':prior,'pre_fix_production04':old_result}
    _write(run_dir/'results.json',result);render_report(run_dir/'results.json',run_dir/'report.md');return result


def build_rank_sample_cache(output: Path, *, l90x_dir: Path, seed: int = 20261104,
                            smoke: bool = False, threads: int = 8) -> dict[str, Any]:
    """Materialize A/B training tuples from rank-sampled DCT3 memory drafts."""
    import sys
    import torch
    torch.set_num_threads(int(threads))
    sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
    import skill_expert_refine as exp3
    data = exp3._load_data(smoke=smoke, l90x_dir=l90x_dir, n_train=64, n_l90x=64, n_dev=32)
    bank = data['unit_bank']; windows = exp3.unit_window(bank['actions'])
    rng = np.random.default_rng(int(seed))
    lmap = exp3.l90x_to_lib40_task_map(data['manifest_l90x'])

    base = exp3._module(HERE.parent/'pi05_memory/run.py','ex2_rank_sample_base')
    policy, _ = base.load_policy(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    def make(name, rows, ids, target, query_tasks, source_episode=None, *, exclude_episode=False):
        ids = np.asarray(ids, np.int64); target = np.asarray(target, np.float32)
        qtask = np.asarray(query_tasks, np.int32)
        # Rank candidates by the stored 10-step action (no pose/scale fitting).
        # Same-task units are excluded for LIBERO-40 and exact instruction
        # matches in L90X; unmatched L90X tasks retain the full bank.
        k=min(50,len(windows))
        rank_target=target[:,:,:7] if target.shape[-1]>=7 else target
        top_ids,top_err=rank_stored_actions(rank_target,windows,bank['task'],query_task=qtask,
            candidate_episode=bank['episode'] if exclude_episode else None,
            query_episode=np.asarray(source_episode) if exclude_episode and source_episode is not None else None,k=k)
        ranks, random_mask = geometric_rank_sample(top_err, rng, p=.16, random_fraction=.10)
        picked = top_ids[np.arange(len(ids)), np.maximum(ranks, 0)]
        if random_mask.any():
            for j in np.flatnonzero(random_mask):
                if qtask[j] >= 0:
                    exclude=bank['task']==qtask[j]
                    eligible=np.flatnonzero(~exclude)
                else:eligible=np.arange(len(windows))
                if not len(eligible): raise ValueError('no eligible random memory unit')
                picked[j] = rng.choice(eligible)
        drafts = windows[picked].copy()
        pre = np.square(drafts[:, :, :6] - target[:, :, :6]).sum((1, 2)) / 70.
        # Train, offline and closed-loop all provide π0.5's current gripper
        # channel to the Expert. Generate it at the exact source observation.
        normal=_normal_for_rows(base,policy,rows,ids,90_000_000+ids)
        packed_target = np.zeros((len(target), 10, 32), np.float32); packed_target[:, :, :7] = target[:, :, :7]
        packed_target[:,:,6]=normal[:,:,6]
        packed_draft = np.zeros_like(packed_target); packed_draft[:, :, :6] = drafts[:, :, :6]
        packed_draft[:,:,6]=normal[:,:,6]
        out = Path(output) / f'{name}.npz'
        np.savez_compressed(out, row_ids=ids, target=packed_target, draft=packed_draft,
            selected_unit=picked.astype(np.int32), sampled_rank=ranks.astype(np.int32),
            random_branch=random_mask.astype(bool), pre_mse6=pre.astype(np.float32),
            query_task=qtask.astype(np.int32), pi05_gripper=normal[:,:,6].astype(np.float32))
        return {'rows': int(len(ids)), 'random_fraction_actual': float(random_mask.mean()),
                'pre_mse6_median': float(np.median(pre)), 'output': str(out)}

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows40 = data['rows']; train_ids = np.asarray(data['train_ids'], np.int64)
    train40 = make('train_a', rows40, train_ids, data['target_a'], data['task_a'], rows40['episode'][train_ids],exclude_episode=True)
    lr = data['l90x_rows']; li = np.asarray(data['l90x_ids'], np.int64)
    lq = np.asarray([lmap.get(int(t), -1) for t in lr['task'][li]], np.int32)
    # Unmatched LIBERO-90 task IDs are unrelated to LIBERO-40 IDs. Same-episode
    # exclusion is meaningful only for an exact instruction match.
    lep = lr['episode'][li]
    train90 = make('train_l90x', lr, li, data['target_b'], lq, lep,exclude_episode=False)
    meta = {'status': 'complete_smoke' if smoke else 'complete', 'seed': int(seed),
            'memory_units': int(len(windows)), 'unit_definition': 'CMP3 DCT3 raw decoded natural-length unit; first 10 frames, endpoint hold',
            'geometric_rank_p': .16, 'random_fraction': .10,
            'A': train40, 'B_additional_L90X': train90,
            'L90X_missing_tasks': data['manifest_l90x'].get('missing_tasks', []),
            'fit_rows_exclude_memory_rows': True, 'learned_eval_episodes_excluded': data['learned_eval_holdout'],
            'same_task_units_excluded': True,'training_gripper':'π0.5 exact observation output on source row'}
    _write(output / 'rank_sample_manifest.json', meta)
    return meta


def train_ex2(output: Path, *, condition: str, updates: int = 5000, batch: int = 8,
              eval_every: int = 500, microbatch: int=1, threads: int = 8, mem_fraction: float = .30,
              train_scale: float = 2.0, seed: int = 20261104, smoke: bool = False) -> dict[str, Any]:
    """Train one action Expert from rank-sampled stored motion; no retrieval key."""
    import sys
    import torch
    torch.set_num_threads(int(threads))
    torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
    import skill_expert_refine as exp3
    prep = Path(output) / 'rank-samples'
    paths = [prep / 'train_a.npz'] if condition == 'A' else [prep / 'train_a.npz', prep / 'train_l90x.npz']
    if condition not in ('A', 'B') or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f'missing EX2 rank-sample files for {condition}: {paths}')
    if int(microbatch)<1 or int(microbatch)>int(batch):raise ValueError('microbatch must be in [1,batch]')
    batches = [dict(np.load(p, allow_pickle=False)) for p in paths]
    l90x_dir = ROOT / 'expert-refine-exp3b-01/l90x_rows'
    data = exp3._load_data(smoke=smoke, l90x_dir=l90x_dir, n_train=64, n_l90x=64, n_dev=32)
    base = data['base']; rows40 = data['rows']; rows90 = data['l90x_rows']
    sources = {'A': rows40, 'B': rows90}
    source_rows = [rows40] if condition == 'A' else [rows40, rows90]
    offsets = np.cumsum([0] + [len(x['row_ids']) for x in batches])
    targets = np.concatenate([x['target'] for x in batches])
    drafts = np.concatenate([x['draft'] for x in batches])
    row_ids = np.concatenate([x['row_ids'] for x in batches])
    pre = np.concatenate([x['pre_mse6'] for x in batches])
    domains = np.concatenate([np.full(len(x['row_ids']), i, np.int8) for i, x in enumerate(batches)])
    model, _ = base.load_policy(exp3.CHECKPOINT)
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    state = nnx.state(model, filt)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    opt_state = optimizer.init(state)

    @nnx.jit
    def grad_step(current, obs, action, draft, t0, u, eps):
        source = (1. - t0[:, None, None]) * draft + t0[:, None, None] * eps
        x = (1. - u[:, None, None]) * action + u[:, None, None] * source
        target_v = (source - action) / jnp.maximum(t0[:, None, None], 1e-3)
        times = t0 * u
        def loss_fn(m):
            pred = m.velocity(obs, x, times)
            return jnp.mean(jnp.square(pred[:, :, :6] - target_v[:, :, :6]))
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(current)

    train_meta = {'status': 'running', 'condition': condition, 'updates': updates, 'rows': len(row_ids),
                  'sample_rule': 'action-space top-50; truncated geometric p=.16; 10% uniform eligible unit',
                  't0_rule_train': f'clip({train_scale}*sqrt(pre_mse6)*Uniform(.9,1.1),.05,.6)',
                  'seed': int(seed), 'history': []}
    ckroot = Path(output) / f'train-{condition}' / 'checkpoints'
    ckroot.mkdir(parents=True, exist_ok=smoke)
    rng = jax.random.key(int(seed))
    # Same sampling stream and initialization for A/B; only the training-domain
    # mixture differs.
    np_rng = np.random.default_rng(int(seed))
    dev_path = Path(output) / 'dev-selection.npz'
    with np.load(dev_path, allow_pickle=False) as z:
        dev_draft = np.asarray(z['D1_P5_top1'], np.float32)
        dev_target = np.asarray(z['target'], np.float32)
        dev_pi = np.asarray(z['pi05'], np.float32)
        dev_row_ids = np.asarray(z['row_ids'], np.int64)
        dev_row_keys = np.asarray(z['row_keys'],np.int64)
    with np.load(Path(output)/'retrieval/dev_exact_observations.npz',allow_pickle=False) as z:
        dev_obs_rows={k:z[k] for k in z.files}
    if not np.array_equal(dev_row_ids,np.arange(len(dev_row_ids))) or len(dev_obs_rows['state'])!=len(dev_row_ids) or not np.array_equal(dev_obs_rows['row_keys'],dev_row_keys):
        raise AssertionError('training Dev rows are not exact-frame aligned')
    dev_obs = base.obs_batch(dev_obs_rows, dev_row_ids)
    noise = exp3._seed_noise(dev_row_ids + 60_000_000)
    selected = {'mse6': float('inf')}
    for step in range(1, int(updates) + 1):
        ids = np_rng.integers(0, len(row_ids), size=int(batch))
        bdomains = domains[ids]
        # The sampled stored unit is independent of P0/P5/D3/D5 retrieval.
        action = jnp.asarray(targets[ids]); draft = jnp.asarray(drafts[ids])
        jitter = np_rng.uniform(.9, 1.1, size=len(ids)).astype(np.float32)
        t0 = jnp.asarray(select_t0(pre[ids], train_scale, jitter=jitter))
        rng, ukey, ekey = jax.random.split(rng, 3)
        u = jax.random.uniform(ukey, (len(ids),), minval=.01, maxval=1.)
        eps = jax.random.normal(ekey, action.shape)
        obs_parts = []
        # Grouping by source row collection avoids constructing a mixed pyTree.
        loss_accum = None; grad_accum = None
        for domain in np.unique(bdomains):
            pos = np.flatnonzero(bdomains == domain)
            for mblo in range(0,len(pos),int(microbatch)):
                mb=pos[mblo:mblo+int(microbatch)]
                obs = base.obs_batch(source_rows[int(domain)], row_ids[ids[mb]])
                val, g = grad_step(model, obs, action[mb], draft[mb], t0[mb], u[mb], eps[mb])
                weight = len(mb) / float(batch)
                loss_accum = float(val) * weight if loss_accum is None else loss_accum + float(val) * weight
                g = jax.tree.map(lambda x: x * weight, g)
                grad_accum = g if grad_accum is None else jax.tree.map(lambda a, b: a + b, grad_accum, g)
        state = nnx.state(model, filt)
        updates_tree, opt_state = optimizer.update(grad_accum, opt_state, state)
        nnx.update(model, optax.apply_updates(state, updates_tree))
        if step % int(eval_every) == 0 or step == int(updates):
            # Dev c selection uses actual KEY2-F P5 top-1 drafts; the Dev rows
            # were excluded from model fit and are never used by the confidence fit.
            row_scores = {}
            curves = {}
            for c in (.5, 1., 1.5, 2., 3.):
                t = select_t0(np.maximum(np.square(dev_draft[:, :, :6] - dev_target[:, :, :6]).mean((1, 2)), 0), c)
                dev_draft32=np.zeros((len(dev_draft),10,32),np.float32);dev_draft32[:,:,:7]=dev_draft
                pred = _bridge_with_row_t0(model, dev_obs, dev_draft32, noise, t, batch=8)
                pred = np.asarray(pred, np.float32)
                curves[float(c)] = float(np.square(pred[:, :, :6] - dev_target[:, :, :6]).sum((1, 2)).mean() / 70.)
            c_best = min(curves, key=curves.get); mse = curves[c_best]
            ck = ckroot / f'{step:06d}'
            exp3._save_state(model, filt, ck, ocp)
            if mse < selected['mse6']:
                selected = {'step': step, 'scale': c_best, 'mse6': mse, 'grid': curves, 'checkpoint': str(ck)}
            train_meta['history'].append({'step': step, 'loss': loss_accum, 'dev_mse6_by_scale': curves,
                                          'selected_scale': c_best, 'selected_mse6': mse})
            train_meta['selected'] = selected
            _write(Path(output) / f'train-{condition}' / 'train.json', train_meta)
    train_meta.update({'status': 'complete', 'selected': selected, 'rows': len(row_ids),
                       'fit_domains': ['LIBERO-40'] if condition == 'A' else ['LIBERO-40', 'L90X (77 tasks)']})
    _write(Path(output) / f'train-{condition}' / 'train.json', train_meta)
    return train_meta


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    q=sp.add_parser('status');q.add_argument('--output',type=Path,default=OUT)
    q=sp.add_parser('prepare');q.add_argument('--output',type=Path,required=True);q.add_argument('--threads',type=int,default=8);q.add_argument('--memory-fraction',type=float,default=.20);q.add_argument('--device',default='cuda:0')
    q=sp.add_parser('rank-samples');q.add_argument('--output',type=Path,required=True);q.add_argument('--l90x-dir',type=Path,default=ROOT/'expert-refine-exp3b-01/l90x_rows');q.add_argument('--threads',type=int,default=8)
    q=sp.add_parser('train');q.add_argument('--output',type=Path,required=True);q.add_argument('--condition',choices=('A','B'),required=True);q.add_argument('--updates',type=int,default=5000);q.add_argument('--batch',type=int,default=8);q.add_argument('--microbatch',type=int,default=1);q.add_argument('--eval-every',type=int,default=500);q.add_argument('--threads',type=int,default=8);q.add_argument('--memory-fraction',type=float,default=.20);q.add_argument('--train-scale',type=float,default=2.);q.add_argument('--smoke',action='store_true')
    q=sp.add_parser('aggregate');q.add_argument('--output',type=Path,required=True)
    q=sp.add_parser('confidence-fit');q.add_argument('--retrieval',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--threads',type=int,default=8);q.add_argument('--smoke',action='store_true')
    q=sp.add_parser('select');q.add_argument('--retrieval',type=Path,required=True);q.add_argument('--train-dir',type=Path,required=True);q.add_argument('--confidence',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--threads',type=int,default=8);q.add_argument('--memory-fraction',type=float,default=.20);q.add_argument('--smoke',action='store_true')
    q=sp.add_parser('offline');q.add_argument('--retrieval',type=Path,required=True);q.add_argument('--train-dir',type=Path,required=True);q.add_argument('--confidence',type=Path,required=True);q.add_argument('--schedule',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--threads',type=int,default=8);q.add_argument('--memory-fraction',type=float,default=.20);q.add_argument('--batch',type=int,default=8);q.add_argument('--smoke',action='store_true')
    q=sp.add_parser('smoke');q.add_argument('--output',type=Path,required=True);q.add_argument('--threads',type=int,default=8);q.add_argument('--memory-fraction',type=float,default=.10)
    q=sp.add_parser('report');q.add_argument('--results',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sp.add_parser('finalize');q.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    if a.command=='status':payload={'output':str(a.output),'ex1':str(EX1),'keygrid':str(KEYGRID),'gpu_preallocate':'must be false'}
    elif a.command=='prepare':
        a.output.mkdir(parents=True,exist_ok=True);ret=a.output/'retrieval';samples=a.output/'rank-samples';conf=a.output/'confidence'
        payload={}
        for name,fn in (
            ('retrieval_and_exact_observations',lambda:build_retrieval_cache(ret,device=a.device,threads=a.threads,mem_fraction=a.memory_fraction)),
            ('dev_selection_cache',lambda:write_dev_selection_cache(ret,a.output/'dev-selection.npz')),
            ('rank_samples_and_pi05_gripper',lambda:build_rank_sample_cache(samples,l90x_dir=ROOT/'expert-refine-exp3b-01/l90x_rows',threads=a.threads)),
            ('memory_confidence_fit',lambda:fit_memory_confidence(ret,conf,threads=a.threads))):
            _progress(a.output,name,'running');payload[name]=fn();_progress(a.output,name,'complete')
    elif a.command=='rank-samples':payload=build_rank_sample_cache(a.output,l90x_dir=a.l90x_dir,threads=a.threads)
    elif a.command=='train':payload=train_ex2(a.output,condition=a.condition,updates=a.updates,batch=a.batch,microbatch=a.microbatch,eval_every=a.eval_every,threads=a.threads,mem_fraction=a.memory_fraction,train_scale=a.train_scale,smoke=a.smoke)
    elif a.command=='aggregate':payload=aggregate_training(a.output)
    elif a.command=='confidence-fit':payload=fit_memory_confidence(a.retrieval,a.output,threads=a.threads,smoke=a.smoke)
    elif a.command=='select':payload=select_confidence_policy(a.retrieval,a.train_dir,a.confidence,a.output,threads=a.threads,mem_fraction=a.memory_fraction,smoke=a.smoke)
    elif a.command=='offline':payload=evaluate_offline(a.retrieval,a.train_dir,a.confidence,a.schedule,a.output,threads=a.threads,mem_fraction=a.memory_fraction,batch=a.batch,smoke=a.smoke)
    elif a.command=='smoke':payload=run_smoke(a.output,threads=a.threads,mem_fraction=a.memory_fraction)
    elif a.command=='finalize':payload=finalize_results(a.run_dir)
    else:payload={'report':render_report(a.results,a.output)}
    print(json.dumps(payload,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))


if __name__ == '__main__':
    main()
