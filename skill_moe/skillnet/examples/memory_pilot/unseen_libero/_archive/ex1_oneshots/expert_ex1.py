#!/usr/bin/env python3
"""EX1 draft-mixture utilities and locked KEY2-F checkpoint manifest.

The expert optimizer and bridge loss remain the EXP3 implementation in
``skill_expert_refine.py``.  This module owns the EX1-specific draft policy:
oracle-nearest unit most of the time, ranks 2/3 occasionally, and frozen
KEY2-F retrieval for the remaining training examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
STATE = Path('/work/.skillnet-state')
KEY2 = STATE / 'outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production'
SELECTION = KEY2 / 'selected_configs.json'
OUT = STATE / 'outputs/memory-pilot/unseen-libero/expert-ex1-01'
L90X_SMOKE = STATE / 'outputs/memory-pilot/unseen-libero/x-attn-01/smoke/l90x_rows'
KEY2_MISSING = STATE / 'outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'
KEY2_SEEN = STATE / 'outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache'
KEY2_UNSEEN = STATE / 'outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'
KEY2_VJEPA_EVAL = STATE / 'outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval'
MODES = {'D1': ('p5', 1), 'D2': ('p5', 20), 'D3': ('e_vlm_h', 1), 'D4': ('e_vj_h', 20)}


def load_locked_key2_checkpoints(path: Path = SELECTION) -> dict[str, dict[str, Any]]:
    """Read, validate, and freeze the selected KEY2-F models/configuration."""
    value = json.loads(Path(path).read_text())
    selected = value.get('selected', {})
    result = {}
    for label, mode in (('p0', 'p0'), ('p5', 'p5'), ('e_vlm_h', 'e_vlm_h'), ('e_vj_h', 'e_vj_h')):
        item = selected.get(mode)
        if not item:
            raise KeyError(f'KEY2-F selection lacks {mode}')
        checkpoint = Path(item['checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if item.get('mode') != mode:
            raise ValueError(f'{mode} selection points at {item.get("mode")}')
        result[label] = {
            'mode': mode, 'checkpoint': str(checkpoint),
            'selected_k': int(item['selected_blend']['k']),
            'temperature': float(item['selected_blend']['temperature']),
            'learning_rate': float(item['learning_rate']), 'epochs': int(item['epochs']),
        }
    if result['p5']['selected_k'] != 20:
        raise ValueError('KEY2-F P5 top-20 blend selection changed')
    if result['e_vlm_h']['selected_k'] != 20 or result['e_vj_h']['selected_k'] != 20:
        raise ValueError('KEY2-F cross-attention top-20 selection changed')
    return result


def unit_drafts(unit_windows: np.ndarray, unit_ids: np.ndarray,
                normal: np.ndarray) -> np.ndarray:
    """Copy selected stored 10-step values into [N,10,32], keeping π0.5 grip."""
    windows = np.asarray(unit_windows, np.float32)
    ids = np.asarray(unit_ids, np.int64)
    pi = np.asarray(normal, np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (10, 7):
        raise ValueError(f'unit windows must be [M,10,7], got {windows.shape}')
    if pi.ndim != 3 or pi.shape[1:] != (10, 32) or ids.shape != (len(pi),):
        raise ValueError(f'bad indices/normal shapes: {ids.shape}/{pi.shape}')
    if np.any((ids < 0) | (ids >= len(windows))):
        raise IndexError('candidate IDs must be ordered memory-bank row IDs')
    result = np.zeros_like(pi)
    result[:, :, :6] = windows[ids, :, :6]
    result[:, :, 6] = pi[:, :, 6]
    return result


def blend_unit_windows(scores: np.ndarray, unit_windows: np.ndarray, *, k: int,
                       temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Softmax blend already-natural 10-step windows; no pose/scale adjustment."""
    s = np.asarray(scores, np.float64)
    units = np.asarray(unit_windows, np.float32)
    if s.ndim != 2 or units.ndim != 3 or units.shape[1:] != (10, 7):
        raise ValueError((s.shape, units.shape))
    if s.shape[1] != len(units) or not 1 <= int(k) <= len(units) or temperature <= 0:
        raise ValueError('invalid candidate count, k, or temperature')
    ids = np.argsort(-s, axis=1, kind='stable')[:, :int(k)]
    logits = np.take_along_axis(s, ids, axis=1) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    selected = units[ids]
    return np.einsum('nk,nktd->ntd', weights.astype(np.float32), selected), ids


def make_training_draft_pool(target: np.ndarray, normal: np.ndarray,
                             unit_windows: np.ndarray, unit_task: np.ndarray,
                             query_task: np.ndarray | None, p5_unit_ids: np.ndarray,
                             *, batch: int = 128) -> dict[str, np.ndarray]:
    """Prepare the fixed per-row candidates sampled during Expert training."""
    top3, errors = nearest_unit_topk_ids(target, unit_windows, unit_task,
                                          query_task=query_task, k=3, batch=batch)
    n, _, d = np.asarray(normal).shape
    oracle = unit_drafts(unit_windows, top3.reshape(-1), np.repeat(normal, 3, axis=0))
    oracle = oracle.reshape(n, 3, 10, d)
    p5 = unit_drafts(unit_windows, p5_unit_ids, normal)
    return {'nearest_top3': oracle.astype(np.float32), 'p5_top1': p5.astype(np.float32),
            'nearest_top3_unit_ids': top3.astype(np.int32),
            'nearest_top3_arm6_mse': errors.astype(np.float32)}


def sample_training_drafts(nearest_top3: np.ndarray, p5_top1: np.ndarray,
                           rng: np.random.Generator, *, probabilities=(.60, .15, .25)):
    """Public deterministic helper implementing the EXP3 mixture branches."""
    near=np.asarray(nearest_top3,np.float32);p5=np.asarray(p5_top1,np.float32)
    probs=np.asarray(probabilities,np.float64)
    if near.ndim!=4 or near.shape[1:]!=(3,10,32) or p5.shape!=(len(near),10,32):
        raise ValueError(f'bad shapes {near.shape}/{p5.shape}')
    if probs.shape!=(3,) or np.any(probs<0) or not np.isclose(probs.sum(),1):raise ValueError(probs)
    branch=rng.choice(3,size=len(near),p=probs);out=np.empty_like(p5)
    for i,b in enumerate(branch):out[i]=p5[i] if b==2 else near[i, int(rng.integers(1,3)) if b==1 else 0]
    return out,branch


def nearest_unit_topk_ids(targets, candidates, candidate_task, *, query_task=None, k=3, batch=128):
    """Stable top-k by arm-six MSE; IDs refer to candidate-bank rows."""
    y=np.asarray(targets,np.float32);c=np.asarray(candidates,np.float32);ct=np.asarray(candidate_task,np.int32)
    if y.ndim!=3 or c.ndim!=3 or y.shape[1]!=10 or c.shape[1]!=10 or y.shape[-1]<6 or c.shape[-1]<6:raise ValueError((y.shape,c.shape))
    if not 1<=int(k)<=len(c):raise ValueError(k)
    qt=None if query_task is None else np.asarray(query_task,np.int32)
    if qt is not None and qt.shape!=(len(y),):raise ValueError(qt.shape)
    yf=y[:,:,:6].reshape(len(y),-1).astype(np.float64);cf=c[:,:,:6].reshape(len(c),-1).astype(np.float64)
    cn=np.square(cf).sum(1);ids=np.empty((len(y),int(k)),np.int64);err=np.empty((len(y),int(k)),np.float32)
    for lo in range(0,len(y),int(batch)):
        hi=min(lo+int(batch),len(y));yy=yf[lo:hi]
        dist=np.square(yy).sum(1,keepdims=True)+cn[None]-2*yy@cf.T
        if qt is not None:dist[ct[None,:]==qt[lo:hi,None]]=np.inf
        order=np.argsort(dist,axis=1,kind='stable')[:,:int(k)]
        vals=np.take_along_axis(dist,order,axis=1)
        if not np.isfinite(vals).all():raise ValueError('insufficient candidates after task exclusion')
        ids[lo:hi]=order;err[lo:hi]=np.maximum(vals,0)/yf.shape[1]
    return ids,err


def unit_source_and_digests(exp3_bank: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Make exact source keys and 1e-6-quantized arm-action digests.

    KEY2-F and EXP3 decode the same DCT-3 units through separate NumPy paths;
    measured arm-channel round-off is below 6e-8. The gripper is not part of
    an EX1 draft (it comes from π0.5), and the two decoders differ by up to
    4e-4 there, so the digest intentionally covers only the six drafted arm
    channels. Hashing values rounded to six decimals permits that decode noise
    while rejecting meaningful action mismatches.
    """
    required = ('task', 'episode', 'start_frame', 'end_frame', 'length', 'actions')
    missing = [name for name in required if name not in exp3_bank]
    if missing:
        raise KeyError(f'EXP3 memory bank lacks provenance fields: {missing}')
    n = len(exp3_bank['length'])
    source = np.stack([np.asarray(exp3_bank[k], np.int64) for k in
                       ('task', 'episode', 'start_frame', 'end_frame')], axis=1)
    if source.shape != (n, 4) or len({tuple(map(int, x)) for x in source}) != n:
        raise ValueError('EXP3 unit source keys are malformed or duplicated')
    digests = []
    for i, (action, length) in enumerate(zip(exp3_bank['actions'], exp3_bank['length'], strict=True)):
        length = int(length)
        value = np.asarray(action, np.float32)
        if value.ndim != 2 or value.shape[0] < length or value.shape[1] < 7 or length < 1:
            raise ValueError(f'bad EXP3 unit action/length at row {i}: {value.shape}/{length}')
        h = hashlib.sha256()
        h.update(np.asarray([length, 6], np.int64).tobytes())
        canonical = np.round(value[:length, :6].astype(np.float64), decimals=6).astype('<f4')
        h.update(np.ascontiguousarray(canonical).tobytes())
        digests.append(h.hexdigest())
    return source, np.asarray(digests, dtype='U64')


def candidate_bank_fingerprint(source: np.ndarray, action_digests: np.ndarray) -> str:
    """Hash the ordered KEY2-F candidate bank identity and values."""
    src = np.asarray(source, np.int64)
    dig = np.asarray(action_digests).astype('U64')
    if src.ndim != 2 or src.shape[1] != 4 or dig.shape != (len(src),):
        raise ValueError(f'bad candidate identity arrays: {src.shape}, {dig.shape}')
    h = hashlib.sha256()
    h.update(src.tobytes(order='C'))
    for item in dig.tolist():
        h.update(item.encode('ascii'))
    return h.hexdigest()


def key2_to_exp3_candidate_map(key2_source: np.ndarray, key2_action_digests: np.ndarray,
                               exp3_bank: dict[str, Any], *,
                               key2_arm_actions: list[np.ndarray] | None = None,
                               roundoff_atol: float = 5e-7) -> np.ndarray:
    """Map KEY2-F candidate positions to EXP3 row IDs by exact unit provenance.

    The source tuple is (task, episode, absolute_start_frame, exclusive_end_frame).
    Action hashes prevent a same-span unit from silently matching different values.
    """
    source = np.asarray(key2_source, np.int64)
    digests = np.asarray(key2_action_digests).astype('U64')
    exp_source, exp_digest = unit_source_and_digests(exp3_bank)
    if source.ndim != 2 or source.shape[1] != 4 or digests.shape != (len(source),):
        raise ValueError(f'bad KEY2-F candidate identity arrays: {source.shape}, {digests.shape}')
    if len({tuple(map(int, x)) for x in source}) != len(source):
        raise ValueError('KEY2-F candidate source keys are duplicated')
    exp_map = {tuple(map(int, key)): i for i, key in enumerate(exp_source)}
    if len(exp_map) != len(exp_source):
        raise ValueError('EXP3 candidate source keys are duplicated')
    mapped = np.empty(len(source), np.int64)
    for i, key in enumerate(source):
        row = exp_map.get(tuple(map(int, key)))
        if row is None:
            raise ValueError(f'KEY2-F candidate has no EXP3 provenance match: {tuple(map(int, key))}')
        if key2_arm_actions is None:
            if digests[i] != exp_digest[row]:
                raise ValueError(f'KEY2-F/EXP3 unit action fingerprint mismatch at source {tuple(map(int,key))}')
        else:
            if len(key2_arm_actions) != len(source):
                raise ValueError('KEY2-F arm action arrays do not align with its ordered bank')
            length = int(exp3_bank['length'][row])
            left = np.asarray(key2_arm_actions[i], np.float32)
            right = np.asarray(exp3_bank['actions'][row], np.float32)[:length, :6]
            if left.shape != right.shape or not np.allclose(left, right, rtol=0., atol=roundoff_atol):
                delta = float(np.max(np.abs(left - right))) if left.shape == right.shape else float('inf')
                raise ValueError(f'KEY2-F/EXP3 arm action mismatch at source {tuple(map(int,key))}: max_abs={delta}')
        mapped[i] = row
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError('candidate mapping is not one-to-one')
    return mapped


def reorder_candidate_windows_to_key2(exp3_windows: np.ndarray, key2_to_exp3: np.ndarray) -> np.ndarray:
    """Return playback windows in the exact order used by KEY2 candidate scores."""
    windows = np.asarray(exp3_windows, np.float32)
    order = np.asarray(key2_to_exp3, np.int64)
    if windows.ndim != 3 or windows.shape[1:] != (10, 7):
        raise ValueError(f'expected EXP3 windows [N,10,7], got {windows.shape}')
    if order.shape != (len(windows),) or not np.array_equal(np.sort(order), np.arange(len(windows))):
        raise ValueError('KEY2-to-EXP3 order must be a complete candidate permutation')
    return np.ascontiguousarray(windows[order])


def drafts_from_cached_candidate_ids(candidate_windows, candidate_ids, weights, pi05):
    """Replay selected natural-length arm windows, retaining π0.5 gripper values."""
    values = np.asarray(candidate_windows, np.float32)
    ids = np.asarray(candidate_ids, np.int64)
    w = np.asarray(weights, np.float32)
    normal = np.asarray(pi05, np.float32)
    if values.ndim != 3 or values.shape[1:] != (10, 7):
        raise ValueError(f'bad cached candidate window shape {values.shape}')
    if ids.ndim != 2 or w.shape != ids.shape or ids.shape[0] != len(normal):
        raise ValueError(f'bad cached selection shapes: ids={ids.shape}, weights={w.shape}, pi05={normal.shape}')
    if np.any(ids < 0) or np.any(ids >= len(values)) or not np.isfinite(w).all():
        raise ValueError('candidate IDs or blend weights are invalid')
    if not np.allclose(w.sum(axis=1), 1., rtol=0., atol=2e-6):
        raise ValueError('candidate blend weights must sum to one')
    arms = np.sum(values[ids, :, :6] * w[:, :, None, None], axis=1)
    result = np.zeros_like(normal, dtype=np.float32)
    result[:, :, :6] = arms
    result[:, :, 6] = normal[:, :, 6]
    return result


def restore_pi05_gripper(actions, pi05_actions):
    """Keep the frozen π0.5 gripper channel after Expert bridge inference."""
    result=np.asarray(actions,np.float32).copy()
    normal=np.asarray(pi05_actions,np.float32)
    if result.ndim!=3 or normal.ndim!=3 or result.shape[:2]!=normal.shape[:2] or result.shape[2]<7 or normal.shape[2]<7:
        raise ValueError(f'bad Expert/π0.5 action shapes: {result.shape}/{normal.shape}')
    result[:,:,6]=normal[:,:,6]
    return result


def top1_indices_with_task_exclusion(scores: np.ndarray, candidate_task: np.ndarray,
                                     query_task: np.ndarray) -> np.ndarray:
    """Choose P5 top-1 after source-task exclusion; -1 means no matching task."""
    score = np.asarray(scores, np.float32).copy()
    ct, qt = np.asarray(candidate_task, np.int32), np.asarray(query_task, np.int32)
    if score.ndim != 2 or score.shape[1] != len(ct) or score.shape[0] != len(qt):
        raise ValueError(f'P5 scores/tasks do not align: {score.shape}, {ct.shape}, {qt.shape}')
    if not np.isfinite(score).all():
        raise ValueError('P5 score matrix contains nonfinite values')
    score[ct[None, :] == qt[:, None]] = -np.inf
    if not np.isfinite(score.max(axis=1)).all():
        raise ValueError('same-task exclusion leaves a query without candidates')
    picked = np.argmax(score, axis=1).astype(np.int32)
    if np.any((qt >= 0) & (ct[picked] == qt)):
        raise AssertionError('P5 top-1 violates same-task exclusion')
    return picked


def make_p5_index_cache(row_ids: np.ndarray, scores: np.ndarray,
                        candidate_source: np.ndarray, candidate_action_digests: np.ndarray,
                        query_task: np.ndarray, *, candidate_arm_actions: np.ndarray | None = None,
                        candidate_lengths: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Serialize a KEY2-F P5 index with explicit ordering and exclusion contract."""
    source = np.asarray(candidate_source, np.int64)
    digests = np.asarray(candidate_action_digests).astype('U64')
    if source.ndim != 2 or source.shape[1] != 4 or digests.shape != (len(source),):
        raise ValueError('malformed KEY2-F candidate manifest')
    selected = top1_indices_with_task_exclusion(scores, source[:, 0], query_task)
    result = {'row_ids': np.asarray(row_ids, np.int64), 'selected_unit': selected,
            'candidate_source': source, 'candidate_action_digests': digests,
            'candidate_fingerprint_sha256': np.asarray(candidate_bank_fingerprint(source, digests)),
            'query_task': np.asarray(query_task, np.int32),
            'exclusion_rule': np.asarray('same_source_task_before_top1')}
    if candidate_arm_actions is not None or candidate_lengths is not None:
        actions = np.asarray(candidate_arm_actions, np.float32)
        lengths = np.asarray(candidate_lengths, np.int32)
        if actions.ndim != 3 or actions.shape[0] != len(source) or actions.shape[-1] != 6:
            raise ValueError(f'candidate arm actions must be [N,T,6], got {actions.shape}')
        if lengths.shape != (len(source),) or np.any(lengths < 1) or np.any(lengths > actions.shape[1]):
            raise ValueError(f'candidate lengths do not align with action cache: {lengths.shape}')
        result['candidate_arm_actions'] = actions
        result['candidate_lengths'] = lengths
    return result


def write_checkpoint_manifest(path: Path, *, l90x_manifest: Path,
                              prepared: Path | None = None) -> dict[str, Any]:
    """Record the accepted 77-task L90X pool, explicitly preserving absent task 51."""
    source = json.loads(Path(l90x_manifest).read_text())
    tasks = source.get('tasks', [])
    missing = [int(x['task_id']) for x in source.get('missing_tasks', [])]
    zeros = [int(k) for k, v in source.get('task_row_counts', {}).items() if int(v) == 0]
    if len(tasks) != 78 or missing != [51] or zeros != [51]:
        raise ValueError(f'EX1 only accepts exactly task 51 absent: tasks={len(tasks)}, missing={missing}, zero={zeros}')
    meta = {'status': 'prepared', 'l90x_manifest_tasks': 78, 'l90x_tasks_with_rows': 77,
            'missing_l90x_task_ids': [51], 'missing_task_policy': 'keep absent and proceed with other 77 task row pools',
            'key2f_checkpoints': load_locked_key2_checkpoints(),
            'drafts': {'D1': 'P5 top-1', 'D2': 'P5 top-20 softmax blend, selected k/temp from KEY2-F',
                       'D3': 'E-VLM+H top-1', 'D4': 'E-VJ+H top-20 softmax blend'},
            'expert_training_mix': {'nearest_top1': .60, 'nearest_rank2_or3': .15, 'p5_top1': .25},
            'gripper_source': 'ordinary pi05 action chunk; expert loss/update acts on arm6 only',
            'prepared_dir': str(prepared) if prepared is not None else None}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    return meta


def run_training_smoke(output: Path, *, threads=8, updates=12, mem_fraction=.10):
    """Exercise the EXP3 trainer's mixed-draft branch on a tiny cache.

    The smoke uses a documented nearest-unit proxy in the P5 slot, solely to
    validate tensor plumbing and optimizer updates. It does not claim to test
    KEY2 retrieval quality; production preparation must replace that field by
    the locked P5 top-1 index cache.
    """
    import skill_expert_refine as exp3
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    prep=out/'prepared'
    data=exp3.prepare(prep,smoke=True,threads=threads,n_train=16,n_l90x=16,n_dev=8,
                      l90x_dir=L90X_SMOKE,batch=4,mem_fraction=mem_fraction)
    bank=exp3.unit_window(data['unit_bank']['actions'])
    for name,target,qtask in (
        ('train_a',data['target_a'],data['task_a']),
        ('train_l90x',data['target_b'],None)):
        path=prep/f'{name}.npz';old=exp3._load_npz(path)
        normal=np.asarray(old['normal'],np.float32)
        top3,_=exp3.nearest_unit_topk_ids(target,bank,data['unit_bank']['task'],query_task=qtask,k=3)
        pools=exp3.draft_from_unit_ids(np.repeat(normal,3,axis=0),data['unit_bank'],top3.reshape(-1)).reshape(len(normal),3,10,32)
        # Placeholder only for smoke: production requires the immutable actual P5 index cache.
        exp3._write_npz(path,**old,nearest_top3=pools,p5_top1=pools[:,0].copy())
    smoke_result=exp3.train_conditions(prep,threads=threads,batch=2,updates=updates,
                                       eval_every=max(1,updates//2),mem_fraction=mem_fraction,
                                       conditions=('A','B'),artifact_dir=out/'train',
                                       allow_oracle_dev_smoke=True)
    result={'status':'complete','updates':updates,'conditions':smoke_result,
            'draft_mix_branch_exercised':True,
            'p5_smoke_source':'oracle-top1 tensor placeholder; production must use KEY2-F P5 top-1 cache',
            'full_closed_loop':'not run by this smoke'}
    (out/'smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def attach_training_mixture(prepared: Path, p5_index_dir: Path, *, run_root=None,
                            l90x_manifest=None, threads=8):
    """Attach oracle rank-1/rank-2/3 and frozen P5 candidates to EXP3 caches.

    P5 index archives are immutable inputs with ``row_ids`` and ``selected_unit``
    in exact source-row / EXP3 memory-bank order. This API intentionally fails
    on any row-order or bank fingerprint ambiguity.
    """
    import skill_expert_refine as exp3
    root=Path(run_root or exp3.RUN);prepared=Path(prepared);p5_index_dir=Path(p5_index_dir)
    rows=exp3._module(exp3.PI05/'run.py','ex1_attach_base').load_rows(root)
    memory_ids=np.asarray(np.load(root/'memory_row_ids.npy',allow_pickle=False),np.int64)
    bank=exp3.make_memory_bank(rows,memory_ids);windows=exp3.unit_window(bank['actions'])
    for name in ('train_a','train_l90x'):
        path=prepared/f'{name}.npz';cache=exp3._load_npz(path)
        ranks=exp3._load_npz(p5_index_dir/f'{name}.npz')
        if not np.array_equal(ranks['row_ids'],cache['row_ids']):
            raise ValueError(f'{name}: P5 index cache row order differs from prepared Expert rows')
        required={'candidate_source','candidate_action_digests','candidate_fingerprint_sha256','candidate_arm_actions',
                  'candidate_lengths','query_task','exclusion_rule'}
        if not required <= set(ranks):
            raise ValueError(f'{name}: strict KEY2-F candidate manifest fields missing: {sorted(required-set(ranks))}')
        key2_source=np.asarray(ranks['candidate_source'],np.int64)
        key2_digests=np.asarray(ranks['candidate_action_digests']).astype('U64')
        actual_fp=candidate_bank_fingerprint(key2_source,key2_digests)
        if str(np.asarray(ranks['candidate_fingerprint_sha256']).item())!=actual_fp:
            raise ValueError(f'{name}: KEY2-F ordered candidate fingerprint is corrupt')
        key2_actions=np.asarray(ranks['candidate_arm_actions'],np.float32)
        key2_lengths=np.asarray(ranks['candidate_lengths'],np.int32)
        if key2_lengths.shape!=(len(key2_source),) or key2_actions.shape[0]!=len(key2_source):
            raise ValueError(f'{name}: KEY2-F candidate action values/lengths do not align')
        key2_to_exp3=key2_to_exp3_candidate_map(
            key2_source,key2_digests,bank,
            key2_arm_actions=[key2_actions[i,:int(n)] for i,n in enumerate(key2_lengths)],
            roundoff_atol=5e-7)
        if len(key2_to_exp3)!=len(windows):
            raise ValueError(f'{name}: candidate bank is incomplete ({len(key2_to_exp3)}/{len(windows)})')
        selected_key2=np.asarray(ranks['selected_unit'],np.int64)
        if selected_key2.shape!=(len(cache['row_ids']),):raise ValueError((name,selected_key2.shape))
        if np.any((selected_key2<0)|(selected_key2>=len(key2_to_exp3))):
            raise ValueError(f'{name}: selected candidate ID outside KEY2-F ordered bank')
        p5_ids=key2_to_exp3[selected_key2]
        if name=='train_a':qtask=np.asarray(cache['task'],np.int32)
        else:
            if l90x_manifest is None:raise ValueError('L90X exact-instruction exclusion requires its manifest')
            lm=json.loads(Path(l90x_manifest).read_text())
            mapping=exp3.l90x_to_lib40_task_map(lm)
            source=exp3._load_rows(lm['_cache_path']) if '_cache_path' in lm else None
            if source is None:
                # Row task IDs are carried by the L90X cache; the path is explicit in prepare.json.
                prepmeta=json.loads((prepared/'prepare.json').read_text())
                source=exp3._load_rows(prepmeta['l90x_source'])
            qtask=np.asarray([mapping.get(int(source['task'][int(i)]),-1) for i in cache['row_ids']],np.int32)
        cache_qtask=np.asarray(ranks['query_task'],np.int32)
        if cache_qtask.shape!=qtask.shape or not np.array_equal(cache_qtask,qtask):
            raise ValueError(f'{name}: P5 query task identities do not match prepared source rows')
        rule=str(np.asarray(ranks['exclusion_rule']).item())
        if rule!='same_source_task_before_top1':
            raise ValueError(f'{name}: P5 cache lacks pre-ranking same-task exclusion')
        matching=qtask>=0
        if matching.any() and np.any(bank['task'][p5_ids[matching]]==qtask[matching]):
            raise AssertionError(f'{name}: P5 retrieval index violates same-task exclusion')
        top3,_=exp3.nearest_unit_topk_ids(cache['target'],windows,bank['task'],query_task=qtask,k=3)
        normal=np.asarray(cache['normal'],np.float32)
        nearest=exp3.draft_from_unit_ids(np.repeat(normal,3,axis=0),bank,top3.reshape(-1)).reshape(len(normal),3,10,32)
        p5=exp3.draft_from_unit_ids(normal,bank,p5_ids)
        exp3._write_npz(path,**cache,nearest_top3=nearest,p5_top1=p5,
                        nearest_top3_unit_ids=top3.astype(np.int32),p5_unit_ids=p5_ids.astype(np.int32))
    return {'status':'complete','train_a_rows':len(exp3._load_npz(prepared/'train_a.npz')['row_ids']),
            'train_l90x_rows':len(exp3._load_npz(prepared/'train_l90x.npz')['row_ids']),
            'drafts':'nearest rank1 / rank2-3 / actual KEY2-F P5 top1','gripper':'ordinary pi05 copy'}


def run_actual_retrieval_smoke(output: Path, *, threads=8, mem_fraction=.20,
                               train_queries=16, dev_queries=8, eval_queries=8):
    """Exercise locked P5 (D1/D2) retrieval and strict candidate mapping on real rows.

    This is a retrieval/data-contract smoke, not EX1 Expert evaluation. It
    computes real P5 top-1 scores for a small fixed sample; only the A
    training sample uses same-task exclusion. Dev/Eval IDs and targets are
    never passed to the scorer.
    """
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('actual KEY2 retrieval smoke requires the available CUDA model path')
    import key_shared_latent as key2
    import skill_expert_refine as exp3

    bank, bank_input, _, _, _, learned, unseen, dev = key2._build_key2_queries(
        exp3.RUN, KEY2_MISSING, KEY2_SEEN, KEY2_UNSEEN, KEY2_VJEPA_EVAL)
    exp3_rows = exp3._module(exp3.PI05 / 'run.py', 'ex1_retrieval_base').load_rows(exp3.RUN)
    fit_ids, holdout = exp3.fit_rows_without_learned_eval(exp3_rows)
    rng = np.random.default_rng(4101)
    train_ids = np.sort(rng.choice(fit_ids, min(int(train_queries), len(fit_ids)), replace=False))
    dev_ids = np.arange(min(int(dev_queries), len(dev['key'])), dtype=np.int64)
    eval_ids = np.arange(min(int(eval_queries), len(unseen['key'])), dtype=np.int64)
    source, digests = unit_source_and_digests({
        'task': bank['task'], 'episode': bank['episode'], 'start_frame': bank['source'][:, 2],
        'end_frame': bank['source'][:, 3], 'length': bank['length'],
        'actions': [bank['natural_actions'][i, :int(n)] for i, n in enumerate(bank['length'])]})
    exp3_bank = exp3.make_memory_bank(
        exp3_rows, np.asarray(np.load(exp3.RUN / 'memory_row_ids.npy', allow_pickle=False), np.int64))
    key2_arm_actions = [np.asarray(bank['natural_actions'][i, :int(n), :6], np.float32)
                        for i, n in enumerate(bank['length'])]
    mapped = key2_to_exp3_candidate_map(source, digests, exp3_bank,
                                        key2_arm_actions=key2_arm_actions, roundoff_atol=5e-7)
    if len(mapped) != len(bank['length']) or len(np.unique(mapped)) != len(mapped):
        raise AssertionError('KEY2-F and EXP3 candidate mapping is not complete and one-to-one')

    # Query rows for the Expert A fit smoke are arbitrary non-memory data rows;
    # their exact cached VLM tokens and causal previous-ten-action histories
    # are used, rather than memory-start or oracle query features.
    run_tokens = np.load(exp3.RUN / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    run_groups = np.load(exp3.RUN / 'groups.npy', mmap_mode='r', allow_pickle=False)
    train_hist, train_mask, train_state = [], [], []
    for row_id in train_ids:
        t, e, f = (int(exp3_rows[k][row_id]) for k in ('task', 'episode', 'frame'))
        hist, mask, state, _ = key2._episode_history(exp3_rows, t, e, f)
        train_hist.append(hist); train_mask.append(mask); train_state.append(state)
    train_part = {
        'pooled': key2._pool_tokens(run_tokens[train_ids], run_groups[train_ids]),
        'history': np.asarray(train_hist, np.float32),
        'history_mask': np.asarray(train_mask, bool),
        'state': np.asarray(train_state, np.float32),
    }
    train_task = np.asarray(exp3_rows['task'][train_ids], np.int32)
    configs = load_locked_key2_checkpoints()
    results = {'status': 'complete', 'device': str(device), 'bank_units': len(bank['length']),
               'candidate_mapping': {'key2_ordered_fingerprint': candidate_bank_fingerprint(source, digests),
                                     'key2_to_exp3_rows': len(mapped), 'one_to_one': True},
               'learned_eval_exclusion': holdout, 'train_rows': train_ids.tolist(),
               'dev_rows': dev_ids.tolist(), 'eval_rows': eval_ids.tolist(), 'modes': {}}
    training_index_cache = None
    # The first real retrieval gate is P5 (D1/D2), the expensive and critical
    # Expert-training input. D3/D4 are evaluated later by the production runner.
    for label, mode in (('D1_P5_D2_P5', 'p5'),):
        config_label = 'p5'
        checkpoint = Path(configs[config_label]['checkpoint'])
        model = key2._new_model(mode).to(device)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict); model.eval()
        with torch.no_grad():
            bank_reps, _, _ = key2._encode_candidates(model, mode, bank_input, bank, device, batch=16)
            bank_tensor = torch.as_tensor(bank_reps, device=device, dtype=torch.float32)
            scale = key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
            def score_subset(split, ids):
                subset = {k: (v[ids] if isinstance(v, np.ndarray) and len(v) == len(split['key']) else v)
                          for k, v in split.items()}
                qrep, _ = key2._query_representations(model, mode, subset, device, batch=8)
                qrep = torch.as_tensor(qrep, device=device, dtype=torch.float32)
                return (torch.einsum('qkd,nkd->qnk', qrep, bank_tensor).mean(-1) * scale).cpu().numpy()
            dev_scores = score_subset(dev, dev_ids)
            eval_scores = score_subset(unseen, eval_ids)
            query_reps = key2._model_batch_encode(model, mode, train_part, device)
            qscore = (torch.einsum('qkd,nkd->qnk', query_reps, bank_tensor).mean(-1) * scale).cpu().numpy()
            task = np.asarray(bank['task'], np.int32)
            qscore[task[None, :] == train_task[:, None]] = -np.inf
            train_top = qscore.argmax(axis=1)
            if np.any(task[train_top] == train_task):
                raise AssertionError('actual P5 top-1 training retrieval did not exclude same task')
            candidate_actions = np.asarray(bank['natural_actions'], np.float32)[..., :6]
            training_index_cache = make_p5_index_cache(
                train_ids, qscore, source, digests, train_task,
                candidate_arm_actions=candidate_actions,
                candidate_lengths=np.asarray(bank['length'], np.int32))
            if not np.array_equal(training_index_cache['selected_unit'], train_top):
                raise AssertionError('serialized actual P5 indices differ from the scored top-1 IDs')
            dev_top = np.argmax(dev_scores, axis=1)
            eval_top = np.argmax(eval_scores, axis=1)
            if not np.isfinite(dev_scores).all() or not np.isfinite(eval_scores).all():
                raise AssertionError(f'{label} returned nonfinite retrieval scores')
            results['modes'][label] = {
                'checkpoint': str(checkpoint), 'train_top1_exp3_row_ids': mapped[train_top].tolist() if train_top is not None else None,
                'dev_top1_key2_ids': dev_top.tolist(), 'eval_top1_key2_ids': eval_top.tolist(),
                'dev_score_shape': list(dev_scores.shape), 'eval_score_shape': list(eval_scores.shape),
                'top_scores_finite': True,
            }
        del model
        torch.cuda.empty_cache()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'actual_retrieval_smoke.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    if training_index_cache is not None:
        cache_path = output / 'p5_index' / 'train_a.npz'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            raise FileExistsError(cache_path)
        np.savez_compressed(cache_path, **training_index_cache)
    return results


def attach_actual_p5_smoke(prepared_dir: Path, output: Path, *, threads=8, mem_fraction=.20,
                           batch=16):
    """Replace oracle-smoke P5/Dev drafts with the locked KEY2-F P5 retrieval.

    This is intentionally a small A-only integration smoke. It uses real
    candidate-bank encodings and real P5 scores, while preserving the fixed
    smoke row IDs and the EXP3 memory-bank row-ID contract.
    """
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('actual P5 integration smoke requires CUDA')
    prepared_dir = Path(prepared_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'retrieval smoke output is not empty: {output}')

    import key_shared_latent as key2
    import skill_expert_refine as exp3

    prep_meta = json.loads((prepared_dir / 'prepare.json').read_text())
    if not prep_meta.get('smoke'):
        raise ValueError('this helper only edits a fresh smoke preparation')
    train = exp3._load_npz(prepared_dir / 'train_a.npz')
    dev = exp3._load_npz(prepared_dir / 'dev.npz')
    bank, bank_input, _, _, _, _, _, key2_dev = key2._build_key2_queries(
        exp3.RUN, KEY2_MISSING, KEY2_SEEN, KEY2_UNSEEN, KEY2_VJEPA_EVAL)

    # Map KEY2-F's exact ordered bank onto the EXP3 unit rows before using IDs.
    source, digests = unit_source_and_digests({
        'task': bank['task'], 'episode': bank['episode'], 'start_frame': bank['source'][:, 2],
        'end_frame': bank['source'][:, 3], 'length': bank['length'],
        'actions': [bank['natural_actions'][i, :int(n)] for i, n in enumerate(bank['length'])]})
    exp3_rows = exp3._module(exp3.PI05 / 'run.py', 'ex1_actual_p5_smoke_base').load_rows(exp3.RUN)
    memory_ids = np.asarray(np.load(exp3.RUN / 'memory_row_ids.npy', allow_pickle=False), np.int64)
    exp3_bank = exp3.make_memory_bank(exp3_rows, memory_ids)
    arm_values = [np.asarray(bank['natural_actions'][i, :int(n), :6], np.float32)
                  for i, n in enumerate(bank['length'])]
    key2_to_exp3 = key2_to_exp3_candidate_map(source, digests, exp3_bank,
                                               key2_arm_actions=arm_values, roundoff_atol=5e-7)
    if len(key2_to_exp3) != len(bank['length']) or len(np.unique(key2_to_exp3)) != len(bank['length']):
        raise AssertionError('candidate bank join is incomplete or not one-to-one')

    selected = load_locked_key2_checkpoints()['p5']
    model = key2._new_model('p5').to(device)
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        candidate_reps, _, _ = key2._encode_candidates(model, 'p5', bank_input, bank, device, batch=batch)
        candidate_reps = torch.as_tensor(candidate_reps, device=device, dtype=torch.float32)
        scale = key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)

        # A-training query inputs: cached VLM mean + exact causal previous-10
        # executed actions and state. No target action is passed to the scorer.
        run_tokens = np.load(exp3.RUN / 'tokens.npy', mmap_mode='r', allow_pickle=False)
        run_groups = np.load(exp3.RUN / 'groups.npy', mmap_mode='r', allow_pickle=False)
        train_hist, train_mask, train_state = [], [], []
        for rid in train['row_ids']:
            t, e, f = (int(exp3_rows[k][int(rid)]) for k in ('task', 'episode', 'frame'))
            h, m, st, _ = key2._episode_history(exp3_rows, t, e, f)
            train_hist.append(h); train_mask.append(m); train_state.append(st)
        train_part = {'pooled': key2._pool_tokens(run_tokens[train['row_ids']], run_groups[train['row_ids']]),
                      'history': np.asarray(train_hist, np.float32),
                      'history_mask': np.asarray(train_mask, bool),
                      'state': np.asarray(train_state, np.float32)}
        qtrain = key2._model_batch_encode(model, 'p5', train_part, device)
        train_scores = (torch.einsum('qkd,nkd->qn', qtrain, candidate_reps) / qtrain.shape[1] * scale).cpu().numpy().astype(np.float32)
        train_task = np.asarray(exp3_rows['task'][train['row_ids']], np.int32)
        train_rank_scores = train_scores.copy()
        train_rank_scores[bank['task'][None, :] == train_task[:, None]] = -np.inf
        train_key2_ids = np.argmax(train_rank_scores, axis=1).astype(np.int64)
        if np.any(bank['task'][train_key2_ids] == train_task):
            raise AssertionError('real P5 retrieval failed same-task exclusion on A-training rows')

        # Join smoke Dev rows by their immutable (task, episode, frame) key.
        unseen_rows = exp3._load_rows(exp3.UNSEEN)
        dev_keys = [tuple(map(int, (unseen_rows['task'][int(i)], unseen_rows['episode'][int(i)],
                                    unseen_rows['frame'][int(i)]))) for i in dev['row_ids']]
        key2_lookup = {tuple(map(int, k)): i for i, k in enumerate(np.asarray(key2_dev['key']))}
        if any(k not in key2_lookup for k in dev_keys):
            missing = [k for k in dev_keys if k not in key2_lookup]
            raise ValueError(f'prepared Dev smoke rows are not in the fixed KEY2-F Dev cache: {missing[:3]}')
        dev_ix = np.asarray([key2_lookup[k] for k in dev_keys], np.int64)
        qdev, _ = key2._query_representations(model, 'p5', key2_dev, device, batch=batch)
        qdev = torch.as_tensor(qdev[dev_ix], device=device, dtype=torch.float32)
        dev_scores = (torch.einsum('qkd,nkd->qn', qdev, candidate_reps) / qdev.shape[1] * scale).cpu().numpy().astype(np.float32)
        dev_key2_ids = np.argmax(dev_scores, axis=1).astype(np.int64)
        if not np.isfinite(dev_scores).all():
            raise AssertionError('nonfinite real P5 Dev scores')

    unit_windows = exp3.unit_window(exp3_bank['actions'])
    train_exp3_ids = key2_to_exp3[train_key2_ids]
    dev_exp3_ids = key2_to_exp3[dev_key2_ids]
    train_drafts = exp3.draft_from_unit_ids(train['normal'], exp3_bank, train_exp3_ids)
    dev_drafts = exp3.draft_from_unit_ids(dev['normal'], exp3_bank, dev_exp3_ids)
    revised_train = dict(train, p5_top1=train_drafts, p5_unit_ids=train_exp3_ids.astype(np.int32))
    revised_dev = dict(dev, draft=dev_drafts, selected_unit=dev_exp3_ids.astype(np.int32),
                       pre_mse6=exp3._mse6(dev_drafts, dev['target']).astype(np.float32))
    exp3._write_npz(prepared_dir / 'train_a.npz', **revised_train)
    exp3._write_npz(prepared_dir / 'dev.npz', **revised_dev)
    prep_meta.update({'dev_draft_selection': 'actual KEY2-F P5 top-1 retrieval; no Dev target used for ranking',
                      'dev_draft_selection_method': 'key2f_retrieval',
                      'p5_retrieval_smoke': True})
    exp3._write_json(prepared_dir / 'prepare.json', prep_meta)

    candidate_actions = np.asarray(bank['natural_actions'], np.float32)[..., :6]
    index = make_p5_index_cache(train['row_ids'], train_scores, source, digests, train_task,
                                candidate_arm_actions=candidate_actions,
                                candidate_lengths=np.asarray(bank['length'], np.int32))
    index['selected_exp3_unit'] = train_exp3_ids.astype(np.int32)
    np.savez_compressed(output / 'train_a_actual_p5_index.npz', **index)
    result = {'status': 'complete', 'device': str(device), 'checkpoint': selected['checkpoint'],
              'bank_units': int(len(bank['length'])), 'key2_to_exp3_join': int(len(key2_to_exp3)),
              'train_rows': int(len(train['row_ids'])), 'dev_rows': int(len(dev['row_ids'])),
              'train_same_task_excluded': True, 'dev_ranked_without_target': True,
              'train_key2_ids': train_key2_ids.tolist(), 'train_exp3_ids': train_exp3_ids.tolist(),
              'dev_key2_ids': dev_key2_ids.tolist(), 'dev_exp3_ids': dev_exp3_ids.tolist(),
              'scores_finite': True, 'saved_index': str(output / 'train_a_actual_p5_index.npz')}
    (output / 'actual_p5_training_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_d1_d4_retrieval_smoke(output: Path, *, threads=8, mem_fraction=.20,
                              dev_queries=8, eval_queries=8, batch=16):
    """Run the locked actual D1-D4 retrieval heads on small Dev/Eval slices."""
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('D1-D4 retrieval smoke requires CUDA')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'D1-D4 retrieval smoke output is not empty: {output}')
    import key_shared_latent as key2
    import skill_expert_refine as exp3
    bank, bank_input, _, _, _, _, unseen, dev = key2._build_key2_queries(
        exp3.RUN, KEY2_MISSING, KEY2_SEEN, KEY2_UNSEEN, KEY2_VJEPA_EVAL)
    selected = load_locked_key2_checkpoints()
    dev_ix = np.arange(min(int(dev_queries), len(dev['key'])), dtype=np.int64)
    eval_ix = np.arange(min(int(eval_queries), len(unseen['key'])), dtype=np.int64)
    modes = [('D1_P5_top1', 'p5'), ('D2_P5_top20', 'p5'),
             ('D3_EVLMH_top1', 'e_vlm_h'), ('D4_EVJH_top20', 'e_vj_h')]
    candidate_natural = np.asarray(bank['natural_actions'], np.float32)
    # Convert natural units to the exact 10-action playback windows used by EX1.
    candidate_windows = exp3.unit_window([candidate_natural[i, :int(n)]
                                           for i, n in enumerate(bank['length'])])
    results = {'status': 'complete', 'bank_units': int(len(bank['length'])),
               'dev_queries': len(dev_ix), 'eval_queries': len(eval_ix), 'modes': {}}
    for label, mode in modes:
        item = selected['p5' if mode == 'p5' else mode]
        model = key2._new_model(mode).to(device)
        model.load_state_dict(torch.load(item['checkpoint'], map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            candidate_reps, _, _ = key2._encode_candidates(model, mode, bank_input, bank, device, batch=batch)
            candidate_reps = torch.as_tensor(candidate_reps, device=device, dtype=torch.float32)
            scale = key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
            dev_reps, _ = key2._query_representations(model, mode, dev, device, batch=batch)
            eval_reps, _ = key2._query_representations(model, mode, unseen, device, batch=batch)
            dev_reps = torch.as_tensor(dev_reps[dev_ix], device=device, dtype=torch.float32)
            eval_reps = torch.as_tensor(eval_reps[eval_ix], device=device, dtype=torch.float32)
            def scores(q):
                return (torch.einsum('qkd,nkd->qnk', q, candidate_reps).mean(-1) * scale).cpu().numpy()
            ds, es = scores(dev_reps), scores(eval_reps)
        k = 20 if label.endswith('top20') else 1
        temp = float(item['temperature'])
        dev_blend, dev_top = blend_unit_windows(ds, candidate_windows, k=k, temperature=temp)
        eval_blend, eval_top = blend_unit_windows(es, candidate_windows, k=k, temperature=temp)
        dev_best = ds.argmax(1); eval_best = es.argmax(1)
        dev_target = np.asarray(dev['actions'][dev_ix], np.float32)
        eval_target = np.asarray(unseen['actions'][eval_ix], np.float32)
        results['modes'][label] = {
            'checkpoint': item['checkpoint'], 'k': k, 'temperature': temp,
            'dev_score_shape': list(ds.shape), 'eval_score_shape': list(es.shape),
            'dev_top1_arm6_mse': float(exp3._mse6(candidate_windows[dev_best], dev_target).mean()),
            'dev_blend_arm6_mse': float(exp3._mse6(dev_blend, dev_target).mean()),
            'eval_top1_arm6_mse': float(exp3._mse6(candidate_windows[eval_best], eval_target).mean()),
            'eval_blend_arm6_mse': float(exp3._mse6(eval_blend, eval_target).mean()),
            'dev_top_unit_ids': dev_top.tolist(), 'eval_top_unit_ids': eval_top.tolist(),
            'scores_finite': bool(np.isfinite(ds).all() and np.isfinite(es).all()),
            'targets_used_after_ranking_only': True}
        if not results['modes'][label]['scores_finite']:
            raise AssertionError(f'{label} produced nonfinite scores')
        del model
        torch.cuda.empty_cache()
    (output / 'd1_d4_retrieval_smoke.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def run_production_p5_retrieval(prepared_dir: Path, output: Path, *, threads=8,
                                mem_fraction=.20, batch=32):
    """Run the locked KEY2-F P5 search over full EX1 A/B fit rows and Dev.

    All outputs are keyed to the exact prepared row IDs.  L90X pooled VLM
    features come from the exact frozen-π0.5 extraction cache; the cache
    manifest is checked against the 77-task row source before use.
    """
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('production P5 retrieval requires CUDA')
    prepared_dir,output=Path(prepared_dir),Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    import key_shared_latent as key2
    import skill_expert_refine as exp3
    meta=json.loads((prepared_dir/'prepare.json').read_text())
    if meta.get('smoke') or meta.get('dev_draft_selection_method')!='oracle_pending_key2f_retrieval':
        raise ValueError('expected full preparation awaiting actual P5 retrieval')

    cfg=load_locked_key2_checkpoints()['p5']
    rows=exp3._module(exp3.PI05/'run.py','ex1_prod_p5_rows').load_rows(exp3.RUN)
    memory_ids=np.asarray(np.load(exp3.RUN/'memory_row_ids.npy',allow_pickle=False),np.int64)
    expbank=exp3.make_memory_bank(rows,memory_ids)
    source,digests=unit_source_and_digests(expbank)
    bank,bank_input,_,_,_,_,_,key2_dev=key2._build_key2_queries(
        exp3.RUN,KEY2_MISSING,KEY2_SEEN,KEY2_UNSEEN,KEY2_VJEPA_EVAL)
    key2_source,key2_digests=unit_source_and_digests({
        'task':bank['task'],'episode':bank['episode'],'start_frame':bank['source'][:,2],
        'end_frame':bank['source'][:,3],'length':bank['length'],
        'actions':[np.asarray(bank['natural_actions'][i,:int(n)],np.float32)
                   for i,n in enumerate(bank['length'])]})
    bank_map=key2_to_exp3_candidate_map(
        key2_source,key2_digests,expbank,
        key2_arm_actions=[np.asarray(bank['natural_actions'][i,:int(n),:6],np.float32)
                          for i,n in enumerate(bank['length'])])
    if len(bank_map)!=len(source) or len(np.unique(bank_map))!=len(source):
        raise ValueError('KEY2-F/EXP3 unit mapping is not one-to-one')
    # key2's ordered source keys and values must match the expert memory bank.
    for i,n in enumerate(bank['length']):
        exp_action=np.asarray(expbank['actions'][int(bank_map[i])],np.float32)
        if not np.allclose(bank['natural_actions'][i,:int(n),:6],exp_action[:int(n),:6],rtol=0,atol=5e-7):
            raise ValueError(f'candidate arm action mismatch at KEY2-F row {i}')

    model=key2._new_model('p5').to(device)
    model.load_state_dict(torch.load(cfg['checkpoint'],map_location=device,weights_only=True));model.eval()
    with torch.no_grad():
        kreps,_,_=key2._encode_candidates(model,'p5',bank_input,bank,device,batch=batch)
        kreps=torch.as_tensor(kreps,device=device,dtype=torch.float32)
        scale=key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
        def scores_for(part):
            values=[]
            for st in range(0,len(part['key']),batch):
                ix=np.arange(st,min(len(part['key']),st+batch))
                pp={k:(np.asarray(v)[ix] if isinstance(v,np.ndarray) and len(v)==len(part['key']) else v)
                    for k,v in part.items()}
                q=key2._model_batch_encode(model,'p5',pp,device)
                values.append((torch.einsum('qkd,nkd->qnk',q,kreps).mean(-1)*scale).cpu().numpy())
            return np.concatenate(values,axis=0)
        # A fit-query inputs: exact token rows and causal previous-ten actions.
        a=exp3._load_npz(prepared_dir/'train_a.npz'); a_ids=np.asarray(a['row_ids'],np.int64)
        rt=np.load(exp3.RUN/'tokens.npy',mmap_mode='r',allow_pickle=False)
        rg=np.load(exp3.RUN/'groups.npy',mmap_mode='r',allow_pickle=False)
        apool=np.empty((len(a_ids),2048),np.float32)
        for st in range(0,len(a_ids),64):
            ix=a_ids[st:st+64];apool[st:st+len(ix)]=key2._pool_tokens(rt[ix],rg[ix])
        ah=[];am=[];ast=[]
        for rid in a_ids:
            t,e,f=(int(rows[k][rid]) for k in ('task','episode','frame'))
            h,m,s,_=key2._episode_history(rows,t,e,f);ah.append(h);am.append(m);ast.append(s)
        apart={'key':a_ids,'pooled':apool,'history':np.asarray(ah,np.float32),
               'history_mask':np.asarray(am,bool),'state':np.asarray(ast,np.float32)}
        # B fit-query inputs: cached full L90X rows and exact pooled VLM outputs.
        l=exp3._load_npz(prepared_dir/'train_l90x.npz'); lid=np.asarray(l['row_ids'],np.int64)
        lroot=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/gripper-latent-01/production-01/l90x_features')
        lpool=np.load(lroot/'l90x_pooled.npy',mmap_mode='r',allow_pickle=False)
        lrows=exp3._load_rows(meta['l90x_source'])
        lm=json.loads((Path(meta['l90x_source'])/'manifest.json').read_text())
        if len(lpool)!=len(lrows['task']):raise ValueError('L90X pooled feature row order/length mismatch')
        lpart={'key':lid,'pooled':np.asarray(lpool[lid],np.float32),
               'history':np.asarray(lrows['hist'][lid],np.float32),
               # EXP3b stores only an episode-level has-history flag; the
               # frozen KEY2-F P5 input expects one validity bit per history
               # slot, matching the prior cached L90X key-input adapter.
               'history_mask':np.repeat(np.asarray(lrows['has_hist'][lid],bool)[:,None],10,axis=1),
               'state':np.asarray(lrows['state'][lid,:8],np.float32)}
        # Dev row identity join uses the fixed query keys from the KEY2-F builder.
        d=exp3._load_npz(prepared_dir/'dev.npz')
        ut=np.load(exp3.UNSEEN/'task.npy',mmap_mode='r',allow_pickle=False)
        ue=np.load(exp3.UNSEEN/'episode.npy',mmap_mode='r',allow_pickle=False)
        uf=np.load(exp3.UNSEEN/'frame.npy',mmap_mode='r',allow_pickle=False)
        keymap={tuple(map(int,k)):i for i,k in enumerate(key2_dev['key'])}
        dkeys=[(int(ut[i]),int(ue[i]),int(uf[i])) for i in np.asarray(d['row_ids'],np.int64)]
        dix=np.asarray([keymap[k] for k in dkeys],np.int64)
        dpart={k:(np.asarray(v)[dix] if isinstance(v,np.ndarray) and len(v)==len(key2_dev['key']) else v)
               for k,v in key2_dev.items()}
        dpart['key']=np.asarray(d['row_ids'],np.int64)
        dpart['pooled']=key2._pool_tokens(dpart['tokens'],dpart['groups'])
        sa,sl,sd=scores_for(apart),scores_for(lpart),scores_for(dpart)
    del model;torch.cuda.empty_cache()

    # The pre-ranking exclusion is the EXP3b contract: same LIBERO-40 task,
    # or exact-instruction-mapped source task for L90X. Dev uses no task label.
    at=np.asarray(a['task'],np.int32); lmap=exp3.l90x_to_lib40_task_map(lm)
    lt=np.asarray([lmap.get(int(t),-1) for t in lrows['task'][lid]],np.int32)
    for sc,qt in ((sa,at),(sl,lt)):
        sc[np.asarray(bank['task'])[None,:]==qt[:,None]]=-np.inf
        if not np.isfinite(sc.max(1)).all():raise ValueError('task exclusion leaves an empty P5 candidate set')
    def top20(sc):
        ids=np.argsort(-sc,axis=1,kind='stable')[:,:20]
        vals=np.take_along_axis(sc,ids,axis=1)
        return ids.astype(np.int32),vals.astype(np.float32)
    ai,av=top20(sa);li,lv=top20(sl);di,dv=top20(sd)
    if np.any(bank['task'][ai[:,0]]==at) or np.any((lt>=0)&(bank['task'][li[:,0]]==lt)):
        raise AssertionError('serialized P5 top-1 violates source-task exclusion')
    np.savez_compressed(output/'train_a.npz',row_ids=a_ids,selected_unit=ai[:,0],top20_unit_ids=ai,top20_scores=av,
        query_task=at,candidate_source=key2_source,candidate_action_digests=key2_digests,
        candidate_fingerprint_sha256=np.asarray(candidate_bank_fingerprint(key2_source,key2_digests)),
        candidate_arm_actions=np.asarray(bank['natural_actions'],np.float32)[...,:6],candidate_lengths=bank['length'],
        exclusion_rule=np.asarray('same_source_task_before_top1'))
    np.savez_compressed(output/'train_l90x.npz',row_ids=lid,selected_unit=li[:,0],top20_unit_ids=li,top20_scores=lv,
        query_task=lt,candidate_source=key2_source,candidate_action_digests=key2_digests,
        candidate_fingerprint_sha256=np.asarray(candidate_bank_fingerprint(key2_source,key2_digests)),
        candidate_arm_actions=np.asarray(bank['natural_actions'],np.float32)[...,:6],candidate_lengths=bank['length'],
        exclusion_rule=np.asarray('same_source_task_before_top1'))
    np.savez_compressed(output/'dev.npz',row_ids=np.asarray(d['row_ids']),selected_unit=di[:,0],top20_unit_ids=di,top20_scores=dv)
    result={'status':'complete','checkpoint':cfg['checkpoint'],'candidate_count':len(bank['length']),
        'train_a_rows':len(ai),'train_l90x_rows':len(li),'dev_rows':len(di),'top_k':20,
        'a_same_task_excluded':True,'l90x_exact_instruction_source_task_excluded':True,'dev_task_filter':'none',
        'feature_cache':str(lroot),'l90x_task_51_missing':True}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def attach_full_p5_drafts(prepared_dir: Path, index_dir: Path, *, threads=8):
    """Attach actual P5 train/Dev drafts and write Dev-selected D1/D2 tensors."""
    import skill_expert_refine as exp3
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.20)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('full D1-D4 Dev retrieval requires CUDA')
    prepared_dir,index_dir=Path(prepared_dir),Path(index_dir)
    if json.loads((prepared_dir/'prepare.json').read_text()).get('smoke'):
        raise ValueError('full attachment refuses smoke cache')
    # This performs exact unit mapping and prepares the .60/.15/.25 train mix;
    # P5 scores are from the locked KEY2-F model, not the oracle placeholder.
    attach_training_mixture(prepared_dir,index_dir,threads=threads,
        l90x_manifest=Path(json.loads((prepared_dir/'prepare.json').read_text())['l90x_source'])/'manifest.json')
    train_index=exp3._load_npz(index_dir/'train_a.npz')
    dev_index=exp3._load_npz(index_dir/'dev.npz')
    exp3_rows=exp3._module(exp3.PI05/'run.py','ex1_attach_full_p5_run').load_rows(exp3.RUN)
    bank=exp3.make_memory_bank(exp3_rows,np.asarray(np.load(exp3.RUN/'memory_row_ids.npy',allow_pickle=False),np.int64))
    source=np.asarray(train_index['candidate_source'],np.int64)
    dig=np.asarray(train_index['candidate_action_digests']).astype('U64')
    actions=np.asarray(train_index['candidate_arm_actions'],np.float32)
    lengths=np.asarray(train_index['candidate_lengths'],np.int32)
    ordered_map=key2_to_exp3_candidate_map(source,dig,bank,
        key2_arm_actions=[actions[i,:int(n)] for i,n in enumerate(lengths)],roundoff_atol=5e-7)
    ids=np.asarray(dev_index['top20_unit_ids'],np.int64)
    mapped=ordered_map[ids]
    dev=exp3._load_npz(prepared_dir/'dev.npz')
    if not np.array_equal(dev['row_ids'],dev_index['row_ids']):raise ValueError('Dev P5 index row order changed')
    windows=exp3.unit_window(bank['actions'])
    top1=mapped[:,0]
    d1=exp3.draft_from_unit_ids(dev['normal'],bank,top1)
    cfg=load_locked_key2_checkpoints()['p5'];temp=float(cfg['temperature'])
    logits=np.asarray(dev_index['top20_scores'],np.float64)/temp
    logits-=logits.max(axis=1,keepdims=True);weights=np.exp(logits);weights/=weights.sum(axis=1,keepdims=True)
    soft_windows=np.sum(windows[mapped]*weights[:,:,None,None],axis=1).astype(np.float32)
    d2=np.zeros_like(dev['normal']);d2[:,:,:6]=soft_windows[:,:,:6];d2[:,:,6]=dev['normal'][:,:,6]
    # D3/D4 use the frozen KEY2-F rerankers. Recreate the same exact feature
    # join and do not expose Dev action targets to their score functions.
    import key_shared_latent as key2
    bank2,bank_input,_,_,_,_,_,dev2=key2._build_key2_queries(
        exp3.RUN,KEY2_MISSING,KEY2_SEEN,KEY2_UNSEEN,KEY2_VJEPA_EVAL)
    unseen_t=np.load(exp3.UNSEEN/'task.npy',mmap_mode='r',allow_pickle=False)
    unseen_e=np.load(exp3.UNSEEN/'episode.npy',mmap_mode='r',allow_pickle=False)
    unseen_f=np.load(exp3.UNSEEN/'frame.npy',mmap_mode='r',allow_pickle=False)
    keymap={tuple(map(int,k)):i for i,k in enumerate(dev2['key'])}
    dkeys=[(int(unseen_t[i]),int(unseen_e[i]),int(unseen_f[i])) for i in np.asarray(dev['row_ids'],np.int64)]
    dix=np.asarray([keymap[k] for k in dkeys],np.int64)
    dpart={k:(np.asarray(v)[dix] if isinstance(v,np.ndarray) and len(v)==len(dev2['key']) else v)
           for k,v in dev2.items()}
    configs=load_locked_key2_checkpoints()
    mode_drafts={'D1_P5_top1':d1,'D2_P5_top20':d2}
    # P0 is the frozen VLM-mean-only VectorRanker from KEY2-F. It is a new
    # top-1 comparison; no blend selection is used for this arm.
    cfg0=configs['p0'];model0=key2._new_model('p0').to(device)
    model0.load_state_dict(torch.load(cfg0['checkpoint'],map_location=device,weights_only=True));model0.eval()
    with torch.no_grad():scores0,_=key2._score_model(model0,'p0',bank2,bank_input,dpart,device)
    idx0=np.argsort(-scores0,axis=1,kind='stable')[:,0]
    mode_drafts['D0_P0_top1']=exp3.draft_from_unit_ids(dev['normal'],bank,ordered_map[idx0])
    del model0;torch.cuda.empty_cache()
    for label,mode,k in (('D3_E_VLM_H_top1','e_vlm_h',1),('D4_E_VJ_H_top20','e_vj_h',20)):
        cfgm=configs[mode];retriever=key2._new_model(mode).to(device)
        retriever.load_state_dict(torch.load(cfgm['checkpoint'],map_location=device,weights_only=True));retriever.eval()
        with torch.no_grad():scores,_=key2._score_model(retriever,mode,bank2,bank_input,dpart,device)
        del retriever;torch.cuda.empty_cache()
        idx=np.argsort(-scores,axis=1,kind='stable')[:,:k]
        mapped_ids=ordered_map[idx]
        if k==1:
            mode_drafts[label]=exp3.draft_from_unit_ids(dev['normal'],bank,mapped_ids[:,0])
        else:
            temp=float(cfgm['temperature']);logits=np.take_along_axis(scores,idx,axis=1)/temp
            logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
            arr=np.sum(windows[mapped_ids]*w[:,:,None,None],axis=1).astype(np.float32)
            arr32=np.zeros_like(dev['normal']);arr32[:,:,:6]=arr[:,:,:6];arr32[:,:,6]=dev['normal'][:,:,6]
            mode_drafts[label]=arr32
    # D1 drives the production train checkpoint-selection interface; the other
    # modes select step/t0 independently using saved snapshots after training.
    dev['selected_unit']=top1.astype(np.int32)
    dev['draft']=d1
    dev['pre_mse6']=exp3._mse6(d1,dev['target']).astype(np.float32)
    exp3._write_npz(prepared_dir/'dev.npz',**dev)
    exp3._write_npz(prepared_dir/'dev_drafts.npz',**mode_drafts)
    p=prepared_dir/'prepare.json';meta=json.loads(p.read_text())
    meta.update({'dev_draft_selection_method':'key2f_retrieval',
      'dev_draft_selection':'actual locked KEY2-F P5 top-1; no target used by scorer',
      'dev_draft_modes':list(mode_drafts),
      'actual_p5_index':str(index_dir),'oracle_placeholder_replaced':True})
    exp3._write_json(p,meta)
    return {'status':'complete','train_a_rows':len(train_index['row_ids']),
      'train_l90x_rows':len(exp3._load_npz(prepared_dir/'train_l90x.npz')['row_ids']),
      'dev_rows':len(dev['row_ids']),'dev_modes':list(mode_drafts),
      'dev_oracle_placeholder_replaced':True}


def retrieve_eval_drafts(output: Path, *, threads=8, mem_fraction=.20, batch=16):
    """Build the exact 621/619 D1-D4 evaluation drafts from locked KEY2-F."""
    import torch
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('EX1 full retrieval requires CUDA')
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    import key_shared_latent as key2
    import skill_expert_refine as exp3
    from openpi import transforms  # noqa: F401 -- assures the activated openpi install is used
    bank,bank_input,_,_,_,learned,unseen,dev=key2._build_key2_queries(
        exp3.RUN,KEY2_MISSING,KEY2_SEEN,KEY2_UNSEEN,KEY2_VJEPA_EVAL)
    rows=exp3._module(exp3.PI05/'run.py','ex1_eval_retrieval_rows').load_rows(exp3.RUN)
    memory_ids=np.asarray(np.load(exp3.RUN/'memory_row_ids.npy',allow_pickle=False),np.int64)
    expbank=exp3.make_memory_bank(rows,memory_ids)
    source,digests=unit_source_and_digests(expbank)
    key2_source,key2_digests=unit_source_and_digests({
        'task':bank['task'],'episode':bank['episode'],'start_frame':bank['source'][:,2],
        'end_frame':bank['source'][:,3],'length':bank['length'],
        'actions':[np.asarray(bank['natural_actions'][i,:int(n)],np.float32)
                   for i,n in enumerate(bank['length'])]})
    ordered_map=key2_to_exp3_candidate_map(key2_source,key2_digests,expbank,
        key2_arm_actions=[np.asarray(bank['natural_actions'][i,:int(n),:6],np.float32)
                          for i,n in enumerate(bank['length'])],roundoff_atol=5e-7)
    configs=load_locked_key2_checkpoints()
    def score_only(model,mode,part):
        # Do not call key2._score_model here: that convenience helper also
        # computes action-distance labels. Evaluation targets must not enter
        # the retrieval/scoring path at all.
        key_reps,_,_=key2._encode_candidates(model,mode,bank_input,bank,device,batch=batch)
        q_reps,_=key2._query_representations(model,mode,part,device,batch=batch)
        if q_reps.ndim==3:values=np.einsum('qkd,nkd->qn',q_reps,key_reps,optimize=True)/q_reps.shape[1]
        else:values=q_reps@key_reps.T
        scale=float(key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
        return (values*scale).astype(np.float32)
    # Persist the closed-loop P5 candidate tensors once. Serving then needs
    # only exact image/state query inference, not a second full 92-GB row scan.
    p5cfg=configs['p5'];p5model=key2._new_model('p5').to(device)
    p5model.load_state_dict(torch.load(p5cfg['checkpoint'],map_location=device,weights_only=True));p5model.eval()
    with torch.no_grad(): candidate_reps,_,_=key2._encode_candidates(
        p5model,'p5',bank_input,bank,device,batch=batch)
    closedloop_scale=float(key2._logit_scale_parameter(p5model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    del p5model;torch.cuda.empty_cache()
    p0cfg=configs['p0'];p0model=key2._new_model('p0').to(device)
    p0model.load_state_dict(torch.load(p0cfg['checkpoint'],map_location=device,weights_only=True));p0model.eval()
    with torch.no_grad(): candidate_reps_p0,_,_=key2._encode_candidates(
        p0model,'p0',bank_input,bank,device,batch=batch)
    p0_scale=float(key2._logit_scale_parameter(p0model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    del p0model;torch.cuda.empty_cache()
    e0cfg=configs['e_vlm_h'];e0model=key2._new_model('e_vlm_h').to(device)
    e0model.load_state_dict(torch.load(e0cfg['checkpoint'],map_location=device,weights_only=True));e0model.eval()
    with torch.no_grad(): candidate_reps_e_vlm_h,_,_=key2._encode_candidates(
        e0model,'e_vlm_h',bank_input,bank,device,batch=batch)
    e0_scale=float(key2._logit_scale_parameter(e0model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    del e0model;torch.cuda.empty_cache()
    unit_windows_exp3=exp3.unit_window(expbank['actions']).astype(np.float32)
    # candidate_reps and all score columns use KEY2 order. Reorder playback
    # values to exactly that order before persisting the closed-loop bank.
    unit_windows_key2=reorder_candidate_windows_to_key2(unit_windows_exp3,ordered_map)
    mode_selections={}
    np.savez_compressed(output/'closedloop_p5_bank.npz',candidate_reps=np.asarray(candidate_reps,np.float32),
        candidate_reps_p0=np.asarray(candidate_reps_p0,np.float32),
        candidate_reps_e_vlm_h=np.asarray(candidate_reps_e_vlm_h,np.float32),
        unit_windows=unit_windows_key2,task=np.asarray(bank['task'],np.int32),source=np.asarray(key2_source,np.int32),
        temperature=np.asarray(p5cfg['temperature'],np.float32),logit_scale=np.asarray(closedloop_scale,np.float32),
        p0_logit_scale=np.asarray(p0_scale,np.float32),e_vlm_h_logit_scale=np.asarray(e0_scale,np.float32))
    all_drafts={}
    for split_name,part in (('learned',learned),('unseen',unseen)):
        split_drafts={}
        for label,mode,k in (('D0_P0_top1','p0',1),('D1_P5_top1','p5',1),('D2_P5_top20','p5',20),
                             ('D3_E_VLM_H_top1','e_vlm_h',1),('D4_E_VJ_H_top20','e_vj_h',20)):
            cfg=configs[mode]
            model=key2._new_model(mode).to(device)
            model.load_state_dict(torch.load(cfg['checkpoint'],map_location=device,weights_only=True));model.eval()
            with torch.no_grad(): scores=score_only(model,mode,part)
            del model;torch.cuda.empty_cache()
            idx=np.argsort(-np.asarray(scores),axis=1,kind='stable')[:,:k]
            if k==1:
                weights=np.ones((len(idx),1),np.float32)
            else:
                temp=float(cfg['temperature']);logits=np.take_along_axis(scores,idx,axis=1)/temp
                logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
                weights=w.astype(np.float32)
            pi=np.asarray(part['pi05'],np.float32)
            draft=drafts_from_cached_candidate_ids(unit_windows_key2,idx,weights,pi)
            split_drafts[label]=draft
            mode_selections.setdefault(split_name,{})[label]={
                'candidate_ids':np.asarray(idx,np.int32),'weights':weights}
        all_drafts[split_name]=split_drafts
    for name,split in (('learned',learned),('unseen',unseen)):
        selection_fields={}
        for label,selection in mode_selections[name].items():
            selection_fields[f'{label}_candidate_ids']=selection['candidate_ids']
            selection_fields[f'{label}_weights']=selection['weights']
        np.savez_compressed(output/f'{name}_drafts.npz',**all_drafts[name],**selection_fields,
            row_keys=np.asarray(split['key'],np.int32),target=np.asarray(split['actions'],np.float32),
            pi05=np.asarray(split['pi05'],np.float32),hard=np.asarray(split['hard'],bool),
            task=np.asarray(split['task'],np.int32),episode=np.asarray(split['episode'],np.int64),
            history=np.asarray(split['history'],np.float32),history_mask=np.asarray(split['history_mask'],bool))
    # Independent parity gate: reconstruct D1-D4 from the saved closed-loop
    # candidate cache and per-row selected candidate IDs/weights. This catches
    # any divergence between offline evaluation and runtime playback ordering.
    with np.load(output/'closedloop_p5_bank.npz',allow_pickle=False) as z:
        saved_windows=np.asarray(z['unit_windows'],np.float32)
        saved_source=np.asarray(z['source'],np.int32)
        saved_reps=np.asarray(z['candidate_reps'])
    if not np.array_equal(saved_source,np.asarray(key2_source,np.int32)):
        raise AssertionError('closed-loop source order differs from KEY2 score order')
    if saved_reps.shape[0]!=len(saved_windows) or saved_windows.shape[0]!=len(key2_source):
        raise AssertionError('closed-loop key/value candidate count mismatch')
    parity_max=0.0
    for split_name,split in (('learned',learned),('unseen',unseen)):
        normal=np.asarray(split['pi05'],np.float32)
        with np.load(output/f'{split_name}_drafts.npz',allow_pickle=False) as z:
            for label in ('D0_P0_top1','D1_P5_top1','D2_P5_top20','D3_E_VLM_H_top1','D4_E_VJ_H_top20'):
                replay=drafts_from_cached_candidate_ids(saved_windows,z[f'{label}_candidate_ids'],
                    z[f'{label}_weights'],normal)
                offline=np.asarray(z[label],np.float32)
                delta=float(np.max(np.abs(replay-offline)))
                parity_max=max(parity_max,delta)
                if delta>1e-7:
                    raise AssertionError(f'{split_name}/{label} closed-loop cache playback differs from offline draft: {delta}')
                target=np.asarray(split['actions'],np.float32)
                arm_mse=np.square(replay[:,:,:6]-target[:,:,:6]).sum((1,2))/70.
                offline_arm_mse=np.square(offline[:,:,:6]-target[:,:,:6]).sum((1,2))/70.
                if not np.allclose(arm_mse,offline_arm_mse,rtol=0.,atol=1e-8):
                    raise AssertionError(f'{split_name}/{label} cache playback MSE differs from offline')
    result={'status':'complete','learned_rows':len(learned['key']),'learned_hard':int(np.asarray(learned['hard']).sum()),
        'unseen_rows':len(unseen['key']),'unseen_hard':int(np.asarray(unseen['hard']).sum()),
        'candidate_units':len(bank['length']),'modes':list(all_drafts['learned']),
        'checkpoints':configs,'row_order':'fixed PI0REF/CMP3 identity order','task51_missing':True,
        'closedloop_candidate_value_order':'KEY2 candidate order; joined by source and arm-action digest',
        'p0_checkpoint':configs['p0'],
        'closedloop_offline_draft_parity_max_abs':parity_max,
        'closedloop_candidate_cache':str(output/'closedloop_p5_bank.npz')}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def _episode_bootstrap_delta(a,b,episodes,*,seed=4101,replicates=5000):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);ep=np.asarray(episodes)
    unique,inverse=np.unique(ep,return_inverse=True)
    delta=a-b
    sums=np.bincount(inverse,weights=delta,minlength=len(unique))
    counts=np.bincount(inverse,minlength=len(unique)).astype(np.float64)
    rng=np.random.default_rng(seed);values=np.empty(replicates,np.float64)
    for r in range(replicates):
        chosen=rng.integers(0,len(unique),size=len(unique))
        values[r]=sums[chosen].sum()/counts[chosen].sum()
    return {'mean_delta':float(np.mean(a-b)),'ci95':[float(x) for x in np.quantile(values,[.025,.975])],
            'replicates':replicates,'unit':'episode'}


def evaluate_ex1_offline(train_dir: Path, eval_drafts_dir: Path, output: Path, *,
                         threads=8, mem_fraction=.20, batch=8):
    """Score complete learned/unseen offline EX1 rows for draft and experts."""
    import torch
    import jax
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    import skill_expert_refine as exp3
    from compressed_denoise import _predict_action
    train_meta=json.loads((Path(train_dir)/'train.json').read_text())
    rows40=exp3._module(exp3.PI05/'run.py','ex1_offline_base').load_rows(exp3.RUN)
    rows90=exp3._load_rows(exp3.UNSEEN)
    rows_by_split={'learned':rows40,'unseen':rows90}
    base=exp3._module(exp3.PI05/'run.py','ex1_offline_policy')
    model,_=base.load_policy(exp3.CHECKPOINT)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    if not nnx.state(model,filt).flat_state(): raise AssertionError('empty action Expert filter')
    unit_bank=exp3.make_memory_bank(rows40,np.asarray(np.load(exp3.RUN/'memory_row_ids.npy',allow_pickle=False),np.int64))
    units=exp3.unit_window(unit_bank['actions'])
    all_results={}
    for split_name in ('learned','unseen'):
        with np.load(Path(eval_drafts_dir)/f'{split_name}_drafts.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        keys=np.asarray(data['row_keys'],np.int64);task=np.asarray(data['task'],np.int32);episode=np.asarray(data['episode'],np.int64)
        src=rows_by_split[split_name]
        keymap={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(src['task'],src['episode'],src['frame']))}
        try:rowids=np.asarray([keymap[tuple(map(int,k))] for k in keys],np.int64)
        except KeyError as exc:raise ValueError(f'{split_name} fixed row identity not in source arrays: {exc}') from exc
        target=np.asarray(data['target'],np.float32);normal=np.asarray(data['pi05'],np.float32)
        hard=np.asarray(data['hard'],bool)
        if target.shape!=(len(keys),10,7) or normal.shape!=(len(keys),10,32):raise ValueError(f'{split_name} eval shapes mismatch')
        # Full privileged upper bound: choose the memory unit by the same
        # stored-action arm-six MSE and natural playback used by EX1.
        oracle_ids,_=exp3.nearest_unit_ids(target,units,unit_bank['task'])
        oracle=np.zeros_like(normal);oracle[:,:,:6]=units[oracle_ids,:,:6];oracle[:,:,6]=normal[:,:,6]
        nearest_pi05_ids,_=exp3.nearest_unit_ids(normal,units,unit_bank['task'])
        pi05_nearest=np.zeros_like(normal);pi05_nearest[:,:,:6]=units[nearest_pi05_ids,:,:6]
        pi05_nearest[:,:,6]=normal[:,:,6]
        history=np.asarray(data['history'],np.float32);hmask=np.asarray(data['history_mask'],bool)
        held=np.zeros_like(normal)
        for i in range(len(held)):
            prev=history[i,hmask[i]]
            if len(prev):held[i,:,:7]=prev[-1]
        noise_keys=np.arange(len(keys),dtype=np.int64)+(0 if split_name=='learned' else 50_000_000)
        noise=exp3._seed_noise(noise_keys)
        split_result={'rows':len(keys),'hard_rows':int(hard.sum()),'task_ids':sorted(set(task.tolist()),key=str),
                      'row_keys':keys.tolist(),'metrics':{}}
        targets32=np.zeros_like(normal);targets32[:,:,:7]=target
        modes=('D0_P0_top1','D1_P5_top1','D2_P5_top20','D3_E_VLM_H_top1','D4_E_VJ_H_top20')
        zero=np.zeros_like(normal);zero[:,:,6]=normal[:,:,6]
        pred_map={'pi05':normal,'hold_last':held,'zero_draft':zero,
                  'pi05_nearest_draft':pi05_nearest,'oracle_draft':oracle}
        for mode in modes:
            draft=np.asarray(data[mode],np.float32)
            pred_map[f'{mode}_draft']=draft
            for condition in ('A','B'):
                best=train_meta['conditions'][condition]['best_by_draft'].get(mode)
                if best is None and mode=='D0_P0_top1':
                    # Training was already underway on the locked D1-D4 Dev
                    # bank when P0 was added to the primary comparison. Reuse
                    # the same condition's D1 Dev-selected checkpoint/t0.
                    best=train_meta['conditions'][condition]['best_by_draft']['D1_P5_top1']
                checkpoint=Path(best['checkpoint'])
                state=ocp.PyTreeCheckpointer().restore(checkpoint)
                live=nnx.state(model,filt);live.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray,state));nnx.update(model,live)
                pred=_predict_action(model,base.obs_batch(src,rowids),draft,noise,float(best['t0']),batch=batch)
                pred=restore_pi05_gripper(pred,normal)
                pred_map[f'{mode}_expert_{condition}']=np.asarray(pred,np.float32)
                split_result.setdefault('selection',{})[f'{mode}_{condition}']={'checkpoint':str(checkpoint),'step':int(best['step']),'t0':float(best['t0']),'dev_mse6':float(best['mse6'])}
        # Oracle plus each independently trained Expert is an explicit bound
        # diagnostic; it is never used to select a checkpoint or a method.
        for condition in ('A','B'):
            mode='D1_P5_top1';best=train_meta['conditions'][condition]['best_by_draft'][mode]
            state=ocp.PyTreeCheckpointer().restore(Path(best['checkpoint']))
            live=nnx.state(model,filt);live.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray,state));nnx.update(model,live)
            oracle_pred=_predict_action(model,base.obs_batch(src,rowids),oracle,noise,float(best['t0']),batch=batch)
            pred_map[f'oracle_draft_expert_{condition}']=restore_pi05_gripper(oracle_pred,normal)
        def score(pred):
            p=np.asarray(pred,np.float64);y=targets32.astype(np.float64)
            return {'arm6_mse':(np.square(p[:,:,:6]-y[:,:,:6]).sum((1,2))/70).astype(np.float32),
                    'active7_mse':(np.square(p[:,:,:7]-y[:,:,:7]).sum((1,2))/70).astype(np.float32)}
        for name,pred in pred_map.items():
            vals=score(pred);split_result['metrics'][name]={}
            for maskname,mask in (('all',np.ones(len(hard),bool)),('hard',hard)):
                ix=np.flatnonzero(mask)
                if not len(ix):continue
                split_result['metrics'][name][maskname]={
                    k:{'mean':float(v[ix].mean()),'median':float(np.median(v[ix]))}
                    for k,v in vals.items()}
        pred_path=output/f'{split_name}_predictions.npz'
        np.savez_compressed(pred_path,**{k:np.asarray(v,np.float32) for k,v in pred_map.items()})
        split_result['prediction_file']=str(pred_path)
        all_results[split_name]=split_result
    # Episode-paired intervals for the principal comparisons.
    for split_name,split in all_results.items():
        row_keys=np.asarray(split['row_keys'],np.int64)
        eps=row_keys[:,0]*10_000_000+row_keys[:,1]
        with np.load(split['prediction_file'],allow_pickle=False) as z:
            predictions={k:z[k] for k in z.files}
        target=np.asarray(np.load(Path(eval_drafts_dir)/f'{split_name}_drafts.npz')['target'],np.float32)
        def mse(pred):return np.square(np.asarray(pred)[:,:,:6]-target[:,:,:6]).sum((1,2))/70
        comparisons={}
        for mode in ('D1_P5_top1','D2_P5_top20','D3_E_VLM_H_top1','D4_E_VJ_H_top20'):
            for condition in ('A','B'):
                a=f'{mode}_expert_{condition}'
                for b in (f'{mode}_draft','pi05'):
                    comparisons[f'{a} - {b}']=_episode_bootstrap_delta(mse(predictions[a]),mse(predictions[b]),eps)
        for condition in ('A','B'):
            a=f'D2_P5_top20_expert_{condition}';b=f'D1_P5_top1_expert_{condition}'
            comparisons[f'{a} - {b}']=_episode_bootstrap_delta(mse(predictions[a]),mse(predictions[b]),eps)
        split['paired_arm6_bootstrap']=comparisons
    result={'status':'complete','train_dir':str(train_dir),'eval_drafts':str(eval_drafts_dir),
            'model':'D1-D4 draft × draft/A/B, pi05, hold_last, oracle draft + A/B',
            'learned_eval':all_results['learned'],'unseen_eval':all_results['unseen'],
            'selection_rule':'Per draft, A/B checkpoint and t0 selected only on locked Dev rows; Eval used once here.'}
    (output/'offline_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    modes_smoke = sub.add_parser('retrieval-modes-smoke', help='score locked D1-D4 retrieval heads on Dev/Eval slices')
    modes_smoke.add_argument('--out', type=Path, required=True)
    modes_smoke.add_argument('--threads', type=int, default=8)
    modes_smoke.add_argument('--mem-fraction', type=float, default=.20)
    modes_smoke.add_argument('--dev-queries', type=int, default=8)
    modes_smoke.add_argument('--eval-queries', type=int, default=8)
    modes_smoke.add_argument('--batch', type=int, default=16)
    retrieval = sub.add_parser('retrieval-smoke', help='score real P5 train/Dev retrieval on a smoke preparation')
    retrieval.add_argument('--prepared-dir', type=Path, required=True)
    retrieval.add_argument('--out', type=Path, required=True)
    retrieval.add_argument('--threads', type=int, default=8)
    retrieval.add_argument('--mem-fraction', type=float, default=.20)
    retrieval.add_argument('--batch', type=int, default=16)
    full=sub.add_parser('production-p5',help='run locked P5 over all full-preparation rows')
    full.add_argument('--prepared-dir',type=Path,required=True);full.add_argument('--out',type=Path,required=True)
    full.add_argument('--threads',type=int,default=8);full.add_argument('--mem-fraction',type=float,default=.20)
    full.add_argument('--batch',type=int,default=32)
    attach=sub.add_parser('attach-production-p5',help='attach actual P5 indices and Dev D1/D2 drafts')
    attach.add_argument('--prepared-dir',type=Path,required=True);attach.add_argument('--index-dir',type=Path,required=True)
    attach.add_argument('--threads',type=int,default=8)
    ev=sub.add_parser('production-eval-drafts',help='materialize all fixed learned/unseen D1-D4 drafts')
    ev.add_argument('--out',type=Path,required=True);ev.add_argument('--threads',type=int,default=8)
    ev.add_argument('--mem-fraction',type=float,default=.20);ev.add_argument('--batch',type=int,default=16)
    oe=sub.add_parser('production-offline-eval',help='run full fixed 621/619 EX1 offline evaluation')
    oe.add_argument('--train-dir',type=Path,required=True);oe.add_argument('--drafts-dir',type=Path,required=True)
    oe.add_argument('--out',type=Path,required=True);oe.add_argument('--threads',type=int,default=8)
    oe.add_argument('--mem-fraction',type=float,default=.20);oe.add_argument('--batch',type=int,default=8)
    parser.add_argument('--check-key2', action='store_true', help='validate locked KEY2-F models')
    parser.add_argument('--l90x-manifest', type=Path)
    parser.add_argument('--smoke', action='store_true', help='run the short EXP3 mixed-draft optimizer smoke')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--updates', type=int, default=12)
    parser.add_argument('--output', type=Path, default=OUT / 'attempt-01' / 'prepared-manifest.json')
    args = parser.parse_args()
    if args.command == 'retrieval-modes-smoke':
        print(json.dumps(run_d1_d4_retrieval_smoke(args.out, threads=args.threads,
              mem_fraction=args.mem_fraction, dev_queries=args.dev_queries,
              eval_queries=args.eval_queries, batch=args.batch), indent=2, ensure_ascii=False))
        return
    if args.command == 'retrieval-smoke':
        print(json.dumps(attach_actual_p5_smoke(args.prepared_dir, args.out, threads=args.threads,
                                                mem_fraction=args.mem_fraction, batch=args.batch),
                         indent=2, ensure_ascii=False))
        return
    if args.command == 'production-p5':
        print(json.dumps(run_production_p5_retrieval(args.prepared_dir,args.out,
              threads=args.threads,mem_fraction=args.mem_fraction,batch=args.batch),indent=2,ensure_ascii=False))
        return
    if args.command == 'attach-production-p5':
        print(json.dumps(attach_full_p5_drafts(args.prepared_dir,args.index_dir,threads=args.threads),indent=2,ensure_ascii=False))
        return
    if args.command == 'production-eval-drafts':
        print(json.dumps(retrieve_eval_drafts(args.out,threads=args.threads,mem_fraction=args.mem_fraction,batch=args.batch),indent=2,ensure_ascii=False))
        return
    if args.command == 'production-offline-eval':
        print(json.dumps(evaluate_ex1_offline(args.train_dir,args.drafts_dir,args.out,threads=args.threads,
              mem_fraction=args.mem_fraction,batch=args.batch),indent=2,ensure_ascii=False))
        return
    if args.check_key2:
        print(json.dumps(load_locked_key2_checkpoints(), indent=2))
    if args.l90x_manifest:
        print(json.dumps(write_checkpoint_manifest(args.output, l90x_manifest=args.l90x_manifest), indent=2))
    if args.smoke:
        print(json.dumps(run_training_smoke(args.output,threads=args.threads,updates=args.updates),indent=2))
    if not args.check_key2 and not args.l90x_manifest and not args.smoke:
        parser.error('choose --check-key2 or --l90x-manifest')


if __name__ == '__main__':
    main()
