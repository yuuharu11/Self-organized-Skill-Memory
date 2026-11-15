#!/usr/bin/env python3
"""EXP3: learn a frozen-VLM/action-Expert refinement from retrieved motion units.

Draft generation is replaceable through ``draft_factory``. The current training
selector is an explicitly privileged arm-six nearest-unit oracle, excluding the
query task for LIBERO-40 rows. Smoke runs only; production training is opt-in.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent / 'pi05_memory'
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
UNSEEN = ROOT / 'unseen-rows-02'
L90X_SMOKE = ROOT / 'x-attn-01/smoke/l90x_rows'
CHECKPOINT = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
OUT = ROOT / 'expert-refine-exp3-01'
SEED = 20261104
NOISE_SEED = 4101
T0_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
LEARNED_EVAL_SELECTION = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/compress-seen-cuts-01/production-control/evaluation-selection.json')

sys.path[:0] = [str(HERE), str(PI05)]
import compression_seen_cuts as cmp3
import compressed_denoise as cd


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def make_memory_bank(rows, memory_ids):
    """Build natural-length anchor units and DCT-3 reconstructions from memory rows."""
    task = np.asarray(rows['task'], np.int32)
    episode = np.asarray(rows['episode'], np.int32)
    frame = np.asarray(rows['frame'], np.int32)
    actions = np.asarray(rows['actions'], np.float32)
    traces = {};base_frames={}
    ids = np.asarray(memory_ids, np.int64)
    keys = np.stack([task[ids], episode[ids]], axis=1)
    for k in np.unique(keys, axis=0):
        ix = ids[(task[ids] == k[0]) & (episode[ids] == k[1])]
        ix = ix[np.argsort(frame[ix])]
        fs = frame[ix]
        if len(fs) > 1 and not np.array_equal(fs, fs[0] + 10 * np.arange(len(fs))):
            raise ValueError(f'noncontiguous memory episode {tuple(map(int, k))}')
        traces[(int(k[0]), int(k[1]))] = np.concatenate(actions[ix, :, :7], axis=0)
        base_frames[(int(k[0]),int(k[1]))]=int(fs[0])
    units, lengths = cmp3.spans(traces, {'mode': 'all', 'min_len': 5, 'width': 3})
    reconstructed = cmp3.dct3(units)
    if len(units) != len(lengths) or len(units) == 0:
        raise AssertionError('empty or misaligned variable-unit pool')
    unit_task, unit_episode, unit_start, unit_end = [], [], [], []
    for key in traces:
        n = sum(1 for _ in ())
        # Re-run the deterministic cut to keep each unit's provenance explicit.
        spans = cmp3.cut(traces[key], mode='all', min_len=5, width=3)
        unit_task.extend([key[0]] * len(spans))
        unit_episode.extend([key[1]] * len(spans))
        # Frame traces are in units of ten actions, while cut boundaries are
        # action frames. `base_frame` is the first saved window's source frame.
        base_frame=base_frames[key]
        for start,end in spans:
            unit_start.append(base_frame+int(start));unit_end.append(base_frame+int(end))
    if len(unit_task) != len(units):
        raise AssertionError((len(unit_task), len(units)))
    return {'actions': reconstructed, 'length': np.asarray(lengths, np.int32),
            'task': np.asarray(unit_task, np.int32), 'episode': np.asarray(unit_episode, np.int32),
            'start_frame': np.asarray(unit_start,np.int32),'end_frame':np.asarray(unit_end,np.int32),
            'raw_units': int(len(units)), 'traces': traces}


def unit_window(units, horizon=10):
    """Play each unit from its beginning at natural frame rate; hold its end if short."""
    seqs = [np.asarray(x, np.float32) for x in units]
    if not seqs or any(x.ndim != 2 or x.shape[1] < 6 or len(x) == 0 for x in seqs):
        raise ValueError([x.shape for x in seqs[:3]])
    out = np.empty((len(seqs), horizon, 7), np.float32)
    for i, seq in enumerate(seqs):
        n = min(horizon, len(seq))
        out[i, :n] = seq[:n, :7]
        if n < horizon:
            out[i, n:] = seq[n - 1, :7]
    return out


def nearest_unit_ids(targets, candidate_actions, candidate_task, *, query_task=None,
                     global_candidate_ids=None, batch=128):
    """Nearest stored unit by unscaled active arm-six MSE with optional task exclusion."""
    y = np.asarray(targets, np.float32)
    c = np.asarray(candidate_actions, np.float32)
    ct = np.asarray(candidate_task, np.int32)
    if (y.ndim != 3 or y.shape[1] != 10 or y.shape[2] < 6 or
            c.ndim != 3 or c.shape[1] != 10 or c.shape[2] < 6):
        raise ValueError((y.shape, c.shape))
    qtask = None if query_task is None else np.asarray(query_task, np.int32)
    if qtask is not None and qtask.shape != (len(y),):
        raise ValueError(qtask.shape)
    flat_c = c[:, :, :6].reshape(len(c), -1).astype(np.float64)
    norm_c = np.square(flat_c).sum(1)
    flat_y = y[:, :, :6].reshape(len(y), -1).astype(np.float64)
    ids = np.arange(len(c), dtype=np.int64) if global_candidate_ids is None else np.asarray(global_candidate_ids, np.int64)
    if ids.shape != (len(c),):
        raise ValueError(ids.shape)
    picked, errors = np.empty(len(y), np.int64), np.empty(len(y), np.float64)
    for lo in range(0, len(y), batch):
        hi = min(lo + batch, len(y))
        yy = flat_y[lo:hi]
        dist = np.square(yy).sum(1, keepdims=True) + norm_c[None] - 2 * yy @ flat_c.T
        if qtask is not None:
            dist[ct[None, :] == qtask[lo:hi, None]] = np.inf
        ix = np.argmin(dist, axis=1)
        if not np.isfinite(dist[np.arange(len(ix)), ix]).all():
            raise ValueError('no eligible memory unit after task exclusion')
        picked[lo:hi] = ids[ix]
        errors[lo:hi] = np.maximum(0, dist[np.arange(len(ix)), ix]) / flat_y.shape[1]
    return picked, errors


def nearest_unit_topk_ids(targets, candidate_actions, candidate_task, *, query_task=None,
                          k=3, batch=64):
    """Return arm-six nearest eligible units, preserving stable bank row IDs."""
    y=np.asarray(targets,np.float32);c=np.asarray(candidate_actions,np.float32)
    ct=np.asarray(candidate_task,np.int32)
    if y.ndim!=3 or c.ndim!=3 or y.shape[1]!=10 or c.shape[1]!=10 or y.shape[2]<6 or c.shape[2]<6:
        raise ValueError(f"expected [N,10,>=6] arrays, got {y.shape}/{c.shape}")
    if not 1<=int(k)<=len(c):raise ValueError(f"invalid k={k} for {len(c)} candidates")
    qt=None if query_task is None else np.asarray(query_task,np.int32)
    if qt is not None and qt.shape!=(len(y),):raise ValueError(qt.shape)
    cf=c[:,:,:6].reshape(len(c),-1).astype(np.float64);yf=y[:,:,:6].reshape(len(y),-1).astype(np.float64)
    cn=np.square(cf).sum(1);ids=np.arange(len(c),dtype=np.int64)
    picked=np.empty((len(y),int(k)),np.int64);errors=np.empty((len(y),int(k)),np.float32)
    for lo in range(0,len(y),int(batch)):
        hi=min(lo+int(batch),len(y));yy=yf[lo:hi]
        dist=np.square(yy).sum(1,keepdims=True)+cn[None]-2.*yy@cf.T
        if qt is not None:dist[ct[None,:]==qt[lo:hi,None]]=np.inf
        # Stable sort makes ties deterministic and preserves the bank-row-ID contract.
        order=np.argsort(dist,axis=1,kind='stable')[:,:int(k)]
        if not np.isfinite(np.take_along_axis(dist,order,axis=1)).all():
            raise ValueError('fewer eligible units than requested after task exclusion')
        picked[lo:hi]=ids[order]
        errors[lo:hi]=np.maximum(0.,np.take_along_axis(dist,order,axis=1))/yf.shape[1]
    return picked,errors


def sample_ex1_drafts(nearest_top3, p5_top1, rng, *, probabilities=(.60,.15,.25)):
    """Sample EXP1 drafts: oracle rank1, rank2/3, or runtime P5 top1.

    ``nearest_top3`` is [B,3,10,32] and ``p5_top1`` is [B,10,32].
    The function deliberately returns unmodified stored unit values except for
    using the caller-provided π0.5 gripper channel already present in each draft.
    """
    near=np.asarray(nearest_top3,np.float32);p5=np.asarray(p5_top1,np.float32)
    probs=np.asarray(probabilities,np.float64)
    if near.ndim!=4 or near.shape[1]<3 or near.shape[2:]!=(10,32) or p5.shape!=(len(near),10,32):
        raise ValueError(f"bad EXP1 draft pool shapes {near.shape}/{p5.shape}")
    if probs.shape!=(3,) or np.any(probs<0) or not np.isclose(probs.sum(),1):raise ValueError(probs)
    branch=rng.choice(3,size=len(near),p=probs)
    result=np.empty_like(p5)
    for i,b in enumerate(branch):
        rank=int(rng.integers(1,3)) if b==1 else 0
        result[i]=p5[i] if b==2 else near[i,rank]
    return result,branch


def draft_factory(target, normal, unit_bank, *, query_task=None, row_ids=None):
    """Replaceable training interface: choose raw unit, no scale/pose fit, pi05 grip."""
    candidates = unit_window(unit_bank['actions'])
    selected, error = nearest_unit_ids(target, candidates, unit_bank['task'], query_task=query_task)
    if query_task is not None and np.any(unit_bank['task'][selected] == np.asarray(query_task)):
        raise AssertionError('LIBERO-40 training draft selected a same-task unit')
    return draft_from_unit_ids(normal, unit_bank, selected), selected, error


def draft_from_unit_ids(normal, unit_bank, selected):
    """Materialize cached unit IDs into a bridge draft, retaining π0.5 grip."""
    candidates = unit_window(unit_bank['actions'])
    selected = np.asarray(selected, np.int64)
    if selected.shape != (len(normal),) or np.any((selected < 0) | (selected >= len(candidates))):
        raise ValueError(f'invalid selected unit ids shape/range: {selected.shape}')
    draft = np.zeros((len(normal), 10, 32), np.float32)
    draft[:, :, :6] = candidates[selected, :, :6]
    draft[:, :, 6] = np.asarray(normal, np.float32)[:, :, 6]
    return draft


def _load_rows(path):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in Path(path).glob('*.npy')}


def _unit_bank_fingerprint(bank):
    h = hashlib.sha256()
    for key in ('actions', 'length', 'task', 'episode'):
        h.update(key.encode())
        if key == 'actions':
            for seq in bank[key]:
                a = np.asarray(seq, np.float32)
                h.update(np.asarray(a.shape, np.int64).tobytes())
                h.update(a.tobytes(order='C'))
        else:
            a = np.asarray(bank[key])
            h.update(str(a.shape).encode())
            h.update(a.tobytes(order='C'))
    return h.hexdigest()


def l90x_to_lib40_task_map(l90x_manifest):
    """Map exact task-language matches across suites for source-task exclusion.

    LIBERO-90 task IDs are not LIBERO-40 IDs. Exact normalized instruction
    matches are the only cross-suite task identity used here; unmatched tasks
    retain the full bank. This is conservative and avoids treating numeric IDs
    from different suites as equivalent.
    """
    source = json.loads((RUN / 'data_manifest.json').read_text())
    return map_instruction_tasks(source['tasks'], l90x_manifest)


def map_instruction_tasks(lib40_tasks, l90x_manifest):
    """Pure mapping helper, exposed for tests and metadata validation."""
    lib40 = {int(k): ' '.join(v.strip().lower().split()) for k, v in lib40_tasks.items()}
    inverse = {v: k for k, v in lib40.items()}
    l90x_text = {}
    for ep in l90x_manifest.get('episodes', []):
        l90x_text.setdefault(int(ep['task_id']), ' '.join(ep['instruction'].strip().lower().split()))
    return {task: inverse.get(text, -1) for task, text in l90x_text.items()}


def learned_eval_episode_pairs(selection_path=LEARNED_EVAL_SELECTION):
    """Return the fixed 40 LIBERO-40 demos held out for learned evaluation."""
    selection = json.loads(Path(selection_path).read_text())
    pairs = {tuple(map(int, pair)) for pair in selection.get('seen_demo_pairs', [])}
    row_ids = np.asarray(selection.get('seen_row_ids', []), np.int64)
    if len(pairs) != 40 or len(row_ids) != 621 or len(np.unique(row_ids)) != 621:
        raise ValueError(f'fixed learned holdout changed: episodes={len(pairs)}, rows={len(row_ids)}')
    return pairs


def fit_rows_without_learned_eval(rows, *, selection_path=LEARNED_EVAL_SELECTION):
    """Fit rows exclude every complete held-out evaluation episode."""
    task = np.asarray(rows['task'])
    episode = np.asarray(rows['episode'])
    memory = np.asarray(rows['memory'], bool)
    held = learned_eval_episode_pairs(selection_path)
    holdout_mask = np.fromiter(((int(t), int(e)) in held for t, e in zip(task, episode, strict=True)),
                               dtype=bool, count=len(task))
    selected_rows = np.asarray(json.loads(Path(selection_path).read_text())['seen_row_ids'], np.int64)
    if np.any(memory[selected_rows]) or not np.all(holdout_mask[selected_rows]):
        raise AssertionError('fixed learned Eval rows do not map to held-out non-memory episodes')
    fit_ids = np.flatnonzero(~memory & ~holdout_mask).astype(np.int64)
    if np.intersect1d(fit_ids, selected_rows).size:
        raise AssertionError('learned Eval row IDs leaked into Expert fit rows')
    fit_pairs = set(zip(task[fit_ids].astype(int).tolist(), episode[fit_ids].astype(int).tolist(), strict=True))
    if fit_pairs & held:
        raise AssertionError('a complete learned Eval episode leaked into Expert fit rows')
    return fit_ids, {'fixed_eval_episodes': len(held), 'fixed_eval_rows': len(selected_rows),
                     'excluded_episode_rows': int(np.count_nonzero(holdout_mask)),
                     'remaining_fit_rows': len(fit_ids)}


def assert_dev_retrieval_for_training(prep_meta, *, allow_oracle_smoke=False):
    """Prevent oracle-selected Dev drafts from choosing production weights/t0."""
    if prep_meta.get('dev_draft_selection_method') != 'key2f_retrieval' and not allow_oracle_smoke:
        raise ValueError('production checkpoint/t0 selection requires actual KEY2-F-retrieved Dev drafts; oracle Dev is smoke-only')


def _seed_noise(row_keys):
    import jax
    return np.stack([np.asarray(jax.random.normal(
        jax.random.fold_in(jax.random.key(NOISE_SEED), int(r)), (10, 32))) for r in row_keys]).astype(np.float32)


def normal_predictions(model, base, rows, row_ids, row_keys, *, batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    noise = _seed_noise(row_keys)
    result = []
    for start in range(0, len(row_ids), batch):
        ids = np.asarray(row_ids[start:start + batch], np.int64)
        result.append(np.asarray(fn(jax.random.key(0), base.obs_batch(rows, ids), num_steps=10,
                                    noise=jnp.asarray(noise[start:start + len(ids)])), np.float32))
    return np.concatenate(result, axis=0) if result else np.empty((0, 10, 32), np.float32)


def _pack_actions(actions):
    a = np.asarray(actions, np.float32)
    if a.ndim != 3 or a.shape[1] != 10 or a.shape[-1] not in (7, 32):
        raise ValueError(a.shape)
    if a.shape[-1] == 32:
        return a.copy()
    out = np.zeros((len(a), 10, 32), np.float32)
    out[:, :, :7] = a
    return out


def _load_data(smoke=False, l90x_dir=L90X_SMOKE, n_train=12, n_l90x=12, n_dev=16, seed=SEED):
    base = _module(PI05 / 'run.py', 'exp3_base_run')
    rows = base.load_rows(RUN)
    memory_ids = np.asarray(np.load(RUN / 'memory_row_ids.npy', allow_pickle=False), np.int64)
    unit_bank = make_memory_bank(rows, memory_ids)
    train_pool, learned_holdout = fit_rows_without_learned_eval(rows)
    if np.intersect1d(train_pool, memory_ids).size:
        raise AssertionError('memory rows leaked into EXP3 train pool')
    rng = np.random.default_rng(seed)
    if smoke:
        # Use different tasks across A training windows and retain representative task exclusion.
        tasks = np.unique(np.asarray(rows['task'])[train_pool])
        chosen = tasks[np.linspace(0, len(tasks) - 1, min(4, len(tasks)), dtype=int)]
        chosen_ids = train_pool[np.isin(np.asarray(rows['task'])[train_pool], chosen)]
        train_ids = np.sort(rng.choice(chosen_ids, min(n_train, len(chosen_ids)), replace=False))
    else:
        train_ids = train_pool
    target_a = np.asarray(rows['actions'][train_ids], np.float32)
    qtask_a = np.asarray(rows['task'][train_ids], np.int32)

    # Full L90X data is not present in this workspace. The existing 4-task X1
    # smoke cache is used for implementation validation only; production accepts
    # a complete l90x_rows cache via --l90x-dir and excludes its selection tasks.
    lrows = _load_rows(l90x_dir)
    manifest = json.loads((Path(l90x_dir) / 'manifest.json').read_text())
    fit_mask = ~np.asarray(lrows['is_select'], bool)
    lfit_ids = np.flatnonzero(fit_mask)
    if smoke:
        l90x_ids = np.sort(rng.choice(lfit_ids, min(n_l90x, len(lfit_ids)), replace=False))
    else:
        if manifest.get('smoke') or len(manifest.get('tasks', [])) != 78:
            raise FileNotFoundError('EXP3-B production requires the full existing L90X 78-task cache; smoke cache is not accepted')
        missing_ids={int(x['task_id']) for x in manifest.get('missing_tasks', [])}
        zero_ids={int(k) for k,v in manifest.get('task_row_counts', {}).items() if int(v)==0}
        if missing_ids != {51} or zero_ids != {51}:
            raise FileNotFoundError(f'EXP3-B accepts only the documented absent task 51; missing={sorted(missing_ids)}, zero={sorted(zero_ids)}')
        l90x_ids = lfit_ids
    target_b = np.asarray(lrows['actions'][l90x_ids], np.float32)
    # Suite task IDs are not comparable across LIBERO-40 / LIBERO-90. For
    # exact normalized instruction matches, exclude the corresponding LIBERO-40
    # source task. Other tasks use the full bank.

    unseen = _load_rows(UNSEEN)
    dev_candidates = np.flatnonzero(np.asarray(unseen['is_dev'], bool))
    dev_tasks = np.asarray(unseen['task'])[dev_candidates]
    if set(np.unique(dev_tasks).tolist()) != {6, 7}:
        raise AssertionError('selection must use only LIBERO-90 Dev tasks 6 and 7')
    if smoke:
        dev_ids = []
        for t in (6, 7):
            ix = dev_candidates[dev_tasks == t]
            dev_ids.extend(ix[:max(1, n_dev // 2)].tolist())
        dev_ids = np.asarray(dev_ids, np.int64)
    else:
        dev_ids = dev_candidates.astype(np.int64)
    return {'base': base, 'rows': rows, 'memory_ids': memory_ids, 'unit_bank': unit_bank,
            'train_pool': train_pool, 'train_ids': train_ids, 'target_a': target_a, 'task_a': qtask_a,
            'l90x_rows': lrows, 'l90x_ids': l90x_ids, 'target_b': target_b,
            'unseen_rows': unseen, 'dev_ids': dev_ids,
            'dev_target': np.asarray(unseen['actions'][dev_ids], np.float32),
            'manifest_l90x': manifest, 'learned_eval_holdout': learned_holdout,
            'smoke': bool(smoke)}


def precompute_unit_selections(out, *, l90x_dir, threads=8, batch=128):
    """CPU-only nearest-unit oracle selections for A, B, and Dev.

    The cache keys include source row ids and a fingerprint of the ordered unit
    bank so prepare() can reject stale or misaligned selections.
    """
    import torch
    torch.set_num_threads(int(threads))
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    base = _module(PI05 / 'run.py', 'exp3_select_base')
    rows = base.load_rows(RUN)
    memory_ids = np.asarray(np.load(RUN / 'memory_row_ids.npy', allow_pickle=False), np.int64)
    bank = make_memory_bank(rows, memory_ids)
    candidates = unit_window(bank['actions'])
    train_a_ids, learned_holdout = fit_rows_without_learned_eval(rows)
    if np.intersect1d(train_a_ids, np.asarray(json.loads(LEARNED_EVAL_SELECTION.read_text())['seen_row_ids'], np.int64)).size:
        raise AssertionError('learned Eval row IDs entered unit-selection fit queries')
    unseen = _load_rows(UNSEEN)
    dev_ids = np.flatnonzero(np.asarray(unseen['is_dev'], bool)).astype(np.int64)
    if set(np.unique(np.asarray(unseen['task'])[dev_ids]).tolist()) != {6, 7}:
        raise AssertionError('Dev selection cache may only include LIBERO-90 tasks 6/7')
    lrows = _load_rows(l90x_dir)
    manifest = json.loads((Path(l90x_dir) / 'manifest.json').read_text())
    if manifest.get('smoke') or len(manifest.get('tasks', [])) != 78:
        raise ValueError('selection precompute requires a full 78-task L90X row cache')
    missing_ids={int(x['task_id']) for x in manifest.get('missing_tasks', [])}
    zero_ids={int(k) for k,v in manifest.get('task_row_counts', {}).items() if int(v)==0}
    if missing_ids!={51} or zero_ids!={51}:
        raise ValueError(f'only absent L90X task 51 is permitted; missing={sorted(missing_ids)}, zero={sorted(zero_ids)}')
    lfit_ids = np.flatnonzero(~np.asarray(lrows['is_select'], bool)).astype(np.int64)

    l90x_task_map = l90x_to_lib40_task_map(manifest)

    def select(name, source_rows, ids, exclude_task=False, mapped_task=None):
        y = np.asarray(source_rows['actions'][ids], np.float32)
        if mapped_task is not None:
            qtask = np.asarray([mapped_task.get(int(t), -1) for t in source_rows['task'][ids]], np.int32)
        else:
            qtask = np.asarray(source_rows['task'][ids], np.int32) if exclude_task else None
        picked, err = nearest_unit_ids(y, candidates, bank['task'], query_task=qtask, batch=batch)
        return name, {'row_ids': ids, 'selected_unit': picked.astype(np.int32), 'arm6_mse': err.astype(np.float32)}

    jobs = [
        ('train_a', rows, train_a_ids, True),
        ('train_l90x', lrows, lfit_ids, False),
        # Per request this is an upper-bound validation selection using Dev truth.
        ('dev_oracle', unseen, dev_ids, False),
    ]
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(select, name, src, ids, excl,
                               l90x_task_map if name == 'train_l90x' else None)
                   for name, src, ids, excl in jobs]
        selected = dict(f.result() for f in futures)
    for name, arrays in selected.items():
        _write_npz(out / f'{name}.npz', **arrays)
    result = {'status': 'complete', 'selection_seconds': time.monotonic() - t0,
        'cpu_threads_per_worker': int(threads), 'parallel_workers': 3,
        'unit_bank_fingerprint_sha256': _unit_bank_fingerprint(bank),
        'memory_units': int(len(bank['actions'])), 'train_a_rows': len(train_a_ids),
        'train_l90x_rows': len(lfit_ids), 'dev_rows': len(dev_ids),
        'dev_task_ids': sorted(map(int, np.unique(np.asarray(unseen['task'])[dev_ids]))),
        'train_a_task_exclusion': 'query LIBERO-40 task excluded',
        'learned_eval_episode_exclusion': learned_holdout,
        'train_l90x_exclusion': 'exclude any LIBERO-40 task with an exact normalized instruction match; nonmatching L90X task IDs map to -1 and use all memory units',
        'l90x_to_lib40_exact_instruction_map': {str(k): int(v) for k, v in l90x_task_map.items()},
        'dev_selection': 'oracle nearest unit by Dev target arm-six MSE, upper-bound selection as requested',
        'l90x_missing_tasks': manifest.get('missing_tasks', []),
        'l90x_row_cache': str(l90x_dir)}
    _write_json(out / 'selection_manifest.json', result)
    return result


def load_unit_selections(selection_dir, unit_bank, row_ids_by_split):
    """Validate and load selection IDs for train A, train L90X, and Dev oracle."""
    selection_dir = Path(selection_dir)
    meta = json.loads((selection_dir / 'selection_manifest.json').read_text())
    if meta['unit_bank_fingerprint_sha256'] != _unit_bank_fingerprint(unit_bank):
        raise ValueError('precomputed unit selections use a different ordered memory bank')
    selected = {}
    for name, expected in row_ids_by_split.items():
        values = _load_npz(selection_dir / f'{name}.npz')
        if not np.array_equal(values['row_ids'], np.asarray(expected, np.int64)):
            raise ValueError(f'{name} precomputed selections are misaligned with source rows')
        if np.any((values['selected_unit'] < 0) | (values['selected_unit'] >= len(unit_bank['actions']))):
            raise ValueError(f'{name} selected unit id outside current memory bank')
        selected[name] = values
    return meta, selected


def prepare(out, *, smoke=False, threads=8, n_train=12, n_l90x=12, n_dev=8,
            l90x_dir=L90X_SMOKE, batch=8, mem_fraction=.15, selection_dir=None):
    """Create immutable oracle-draft caches for A/B smoke or full production."""
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    data = _load_data(smoke=smoke, l90x_dir=l90x_dir, n_train=n_train,
                      n_l90x=n_l90x, n_dev=n_dev)
    out.mkdir(parents=True)
    selection_meta = None
    selected = {}
    if selection_dir is not None:
        selection_meta, selected = load_unit_selections(selection_dir, data['unit_bank'], {
            'train_a': data['train_ids'], 'train_l90x': data['l90x_ids'], 'dev_oracle': data['dev_ids']})
        if np.any(data['unit_bank']['task'][selected['train_a']['selected_unit']] == data['task_a']):
            raise AssertionError('precomputed A selections contain same-task memory units')
        l90x_map = l90x_to_lib40_task_map(data['manifest_l90x'])
        l_query_task = np.asarray([l90x_map.get(int(t), -1)
                                   for t in data['l90x_rows']['task'][data['l90x_ids']]], np.int32)
        l_selected_task = data['unit_bank']['task'][selected['train_l90x']['selected_unit']]
        if np.any((l_query_task >= 0) & (l_selected_task == l_query_task)):
            raise AssertionError('precomputed L90X selection contains exact-instruction source-task units')
    model, _ = data['base'].load_policy(CHECKPOINT)
    a_ids = data['train_ids']
    a_normal = normal_predictions(model, data['base'], data['rows'], a_ids, a_ids, batch=batch)
    if selection_meta is None:
        a_draft, a_unit, a_pre = draft_factory(data['target_a'], a_normal, data['unit_bank'],
                                               query_task=data['task_a'], row_ids=a_ids)
    else:
        a_unit = selected['train_a']['selected_unit']
        a_pre = selected['train_a']['arm6_mse']
        a_draft = draft_from_unit_ids(a_normal, data['unit_bank'], a_unit)
        if np.any(data['unit_bank']['task'][a_unit] == data['task_a']):
            raise AssertionError('precomputed A selections contain same-task memory units')
    li = data['l90x_ids']
    l_normal = normal_predictions(model, data['base'], data['l90x_rows'], li,
                                  np.asarray(li, np.int64) + 40_000_000, batch=batch)
    if selection_meta is None:
        l_draft, l_unit, l_pre = draft_factory(data['target_b'], l_normal, data['unit_bank'])
    else:
        l_unit = selected['train_l90x']['selected_unit']
        l_pre = selected['train_l90x']['arm6_mse']
        l_draft = draft_from_unit_ids(l_normal, data['unit_bank'], l_unit)
    d_ids = data['dev_ids']
    d_normal = normal_predictions(model, data['base'], data['unseen_rows'], d_ids,
                                  d_ids + 50_000_000, batch=batch)
    # EXP3-B uses Dev truth to match the requested oracle-draft evaluation
    # upper bound. A separate π0.5-nearest draft is retained for later reporting.
    if selection_meta is None:
        d_draft, d_unit, d_pre = draft_factory(data['dev_target'], d_normal, data['unit_bank'])
    else:
        d_unit = selected['dev_oracle']['selected_unit']
        d_pre = selected['dev_oracle']['arm6_mse']
        d_draft = draft_from_unit_ids(d_normal, data['unit_bank'], d_unit)
    d_pi05_draft, d_pi05_unit, d_pi05_pre = draft_factory(d_normal, d_normal, data['unit_bank'])
    train_a = {'row_ids': a_ids, 'row_keys': a_ids, 'domain': np.zeros(len(a_ids), np.int8), 'task': data['task_a'],
               'target': _pack_actions(data['target_a']), 'draft': a_draft,
               'normal': _pack_actions(a_normal), 'selected_unit': a_unit, 'pre_mse6': a_pre}
    train_l = {'row_ids': li, 'row_keys': li + 40_000_000, 'domain': np.ones(len(li), np.int8),
               'task': np.asarray(data['l90x_rows']['task'][li], np.int32),
               'target': _pack_actions(data['target_b']), 'draft': l_draft,
               'normal': _pack_actions(l_normal), 'selected_unit': l_unit, 'pre_mse6': l_pre}
    dev = {'row_ids': d_ids, 'row_keys': d_ids + 50_000_000,
           'task': np.asarray(data['unseen_rows']['task'][d_ids], np.int32),
           'target': _pack_actions(data['dev_target']), 'draft': d_draft,
           'normal': _pack_actions(d_normal), 'selected_unit': d_unit, 'pre_mse6': d_pre,
           'pi05_nearest_draft': d_pi05_draft, 'pi05_nearest_unit': d_pi05_unit,
           'pi05_nearest_query_mse6': d_pi05_pre}
    _write_npz(out / 'train_a.npz', **train_a)
    _write_npz(out / 'train_l90x.npz', **train_l)
    _write_npz(out / 'dev.npz', **dev)
    _write_json(out / 'prepare.json', {'status': 'complete', 'smoke': smoke,
        'condition_a_rows': len(a_ids), 'condition_b_l90x_fit_rows': len(li),
        'condition_b_total_rows': len(a_ids) + len(li), 'dev_rows': len(d_ids),
        'dev_tasks': sorted(set(map(int, dev['task']))), 'memory_units': data['unit_bank']['raw_units'],
        'dct3_units': len(data['unit_bank']['actions']),
        'memory_task_exclusion': 'LIBERO-40 source task excluded for every A training query',
        'l90x_task_exclusion': 'exact normalized instruction matches map to LIBERO-40 source tasks; matching source task excluded; unmatched tasks use full memory bank',
        'l90x_to_lib40_exact_instruction_map': ({str(k): int(v) for k, v in
                                                  l90x_to_lib40_task_map(data['manifest_l90x']).items()}
                                                 if selection_meta is not None else None),
        'l90x_source': str(l90x_dir), 'l90x_manifest_smoke': data['manifest_l90x'].get('smoke'),
        'l90x_fit_tasks': data['manifest_l90x'].get('fit_tasks'),
        'l90x_select_tasks_excluded': data['manifest_l90x'].get('selection_tasks'),
        'l90x_missing_tasks': data['manifest_l90x'].get('missing_tasks', []),
        'selection_cache': str(selection_dir) if selection_dir is not None else None,
        'learned_eval_episode_exclusion': data['learned_eval_holdout'],
        'draft_factory': 'replaceable selector -> natural-length DCT3 unit, arm6 raw MSE, no pose/scale fit; pi05 grip copied',
        'unit_playback': 'first 10 frames at natural rate; short units hold final frame',
        'dev_draft_selection': 'oracle nearest unit by Dev target arm-six MSE; smoke/upper-bound only, forbidden for production Expert selection until actual KEY2-F retrieval is attached',
        'dev_draft_selection_method': 'oracle_smoke_only' if smoke else 'oracle_pending_key2f_retrieval',
        'ground_truth_use': 'A/B train target selects unit and supervises; Dev target selects upper-bound draft/checkpoint/t0; no Eval used',
        'learned_eval_exclusion_assertion': 'all rows from 40 fixed learned-evaluation episodes excluded from A fit; selected 621 fixed rows audited'} )
    return data


def _grad_fn(filt):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    @nnx.jit
    def grad(model, obs, actions, draft, t0, u, eps):
        source = (1 - t0[:, None, None]) * draft + t0[:, None, None] * eps
        x = (1 - u[:, None, None]) * actions + u[:, None, None] * source
        velocity_target = (source - actions) / jnp.maximum(t0[:, None, None], 1e-3)
        time = t0 * u

        def loss_fn(current):
            pred = current.velocity(obs, x, time)
            return jnp.mean(jnp.square(pred[:, :, :6] - velocity_target[:, :, :6]))
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(model)
    return grad


def _sample_flow_inputs(key, action_shape):
    """Sample one full effective batch before optional memory-saving accumulation."""
    import jax
    k0, k1, k2 = jax.random.split(key, 3)
    t0 = jax.random.uniform(k0, (action_shape[0],), minval=0.0, maxval=.3)
    u = jax.random.uniform(k1, (action_shape[0],), minval=.01, maxval=1.)
    eps = jax.random.normal(k2, action_shape)
    return t0, u, eps


def _mse6(pred, target):
    p, y = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    return np.square(p[:, :, :6] - y[:, :, :6]).sum(axis=(1, 2)) / 70.0


def _save_state(model, filt, path, ocp):
    import flax.nnx as nnx
    state = nnx.state(model, filt).to_pure_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(path, state)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def train_conditions(out, *, threads=8, batch=2, updates=24, eval_every=12,
                     mem_fraction=.15, conditions=('A', 'B'), artifact_dir=None,
                     allow_oracle_dev_smoke=False, microbatch=None):
    """Train selected EXP3 conditions; all checkpoint and t0 choices use Dev only."""
    import torch
    torch.set_num_threads(threads)
    microbatch = int(batch if microbatch is None else microbatch)
    if microbatch < 1 or microbatch > int(batch):
        raise ValueError(f'microbatch must be in [1,batch], got {microbatch}/{batch}')
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex

    out = Path(out)
    artifacts = Path(artifact_dir) if artifact_dir is not None else out
    if artifact_dir is not None:
        if artifacts.exists():
            raise FileExistsError(artifacts)
        artifacts.mkdir(parents=True)
    if not (out / 'prepare.json').exists():
        raise FileNotFoundError(f'run prepare first: {out}')
    base = _module(PI05 / 'run.py', 'exp3_train_base')
    run_rows = base.load_rows(RUN)
    unseen_rows = _load_rows(UNSEEN)
    train_a = _load_npz(out / 'train_a.npz')
    train_l = _load_npz(out / 'train_l90x.npz')
    dev = _load_npz(out / 'dev.npz')
    dev_modes_path=out/'dev_drafts.npz'
    if dev_modes_path.exists():
        with np.load(dev_modes_path,allow_pickle=False) as z:
            dev_modes={k:np.asarray(z[k],np.float32) for k in z.files}
        if any(v.shape!=dev['draft'].shape for v in dev_modes.values()):
            raise ValueError('Dev draft mode tensors must align exactly with dev.npz')
    else:
        dev_modes={'D1_P5_top1':np.asarray(dev['draft'],np.float32)}
    prep_meta = json.loads((out / 'prepare.json').read_text())
    assert_dev_retrieval_for_training(prep_meta, allow_oracle_smoke=allow_oracle_dev_smoke)
    held_pairs = learned_eval_episode_pairs()
    a_pairs = set(zip(np.asarray(run_rows['task'])[train_a['row_ids']].astype(int).tolist(),
                      np.asarray(run_rows['episode'])[train_a['row_ids']].astype(int).tolist(), strict=True))
    if a_pairs & held_pairs:
        raise AssertionError(f'learned-evaluation episodes leaked into Expert training: {sorted(a_pairs & held_pairs)[:5]}')
    fixed_eval_ids = np.asarray(json.loads(LEARNED_EVAL_SELECTION.read_text())['seen_row_ids'], np.int64)
    if np.intersect1d(train_a['row_ids'], fixed_eval_ids).size:
        raise AssertionError('fixed 621 learned-Eval rows leaked into Expert training')
    l90x_rows = _load_rows(prep_meta['l90x_source'])
    dev_obs = base.obs_batch(unseen_rows, dev['row_ids'])
    dev_noise = _seed_noise(dev['row_keys'])
    records = {}
    for condition in conditions:
        mixture_rng=np.random.default_rng(SEED+311)
        if condition == 'A':
            cache = train_a
            source_rows = {0: run_rows}
        else:
            cache = {k: np.concatenate([train_a[k], train_l[k]], axis=0) for k in train_a}
            source_rows = {0: run_rows, 1: l90x_rows}
        model, _ = base.load_policy(CHECKPOINT)
        filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        state = nnx.state(model, filt)
        if not state.flat_state():
            raise AssertionError('EXP3 action expert trainable filter is empty')
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
        opt_state = optimizer.init(state)
        grad = _grad_fn(filt)
        # Keep the requested initialization/batch RNG seed fixed across A/B;
        # only the row pool differs between the two conditions.
        rng = jax.random.key(SEED)
        history = []
        best_by_draft={name:{'mse6':None} for name in dev_modes}
        n = len(cache['row_ids'])
        for step in range(1, updates + 1):
            rng, batch_key = jax.random.split(rng)
            ids = np.asarray(jax.random.choice(batch_key, n, (batch,), replace=n < batch))
            split = batch_key
            keys = jax.random.split(split, batch)
            losses = []
            grad_sum = None
            domains = np.asarray(cache.get('domain', np.zeros(n, np.int8)))[ids]
            for domain in np.unique(domains):
                pos = np.flatnonzero(domains == domain)
                item_ids = ids[pos]
                obs = base.obs_batch(source_rows[int(domain)],
                                     np.asarray(cache['row_ids'][item_ids], np.int64))
                action = jnp.asarray(cache['target'][item_ids])
                if 'nearest_top3' in cache and 'p5_top1' in cache:
                    mixed, _branch = sample_ex1_drafts(cache['nearest_top3'][item_ids],
                                                       cache['p5_top1'][item_ids], mixture_rng)
                    draft = jnp.asarray(mixed)
                else:
                    # EXP3 compatibility path; EX1 preparation must attach the
                    # explicit mixture arrays before training.
                    draft = jnp.asarray(cache['draft'][item_ids])
                # Keep the effective batch and optimizer update identical while
                # splitting activation memory. Draw t/u/eps at each domain
                # group's full shape so microbatching preserves seeded samples.
                t0_all, u_all, eps_all = _sample_flow_inputs(
                    keys[int(pos[0])], tuple(action.shape))
                for micro_start in range(0, len(item_ids), microbatch):
                    micro_end = min(micro_start + microbatch, len(item_ids))
                    micro_obs = jax.tree.map(lambda x: x[micro_start:micro_end], obs)
                    loss_i, grads_i = grad(
                        model, micro_obs, action[micro_start:micro_end],
                        draft[micro_start:micro_end], t0_all[micro_start:micro_end],
                        u_all[micro_start:micro_end], eps_all[micro_start:micro_end])
                    weight = (micro_end - micro_start) / batch
                    losses.append(float(loss_i) * weight)
                    weighted = jax.tree.map(lambda x: x * weight, grads_i)
                    grad_sum = weighted if grad_sum is None else jax.tree.map(
                        lambda a, b: a + b, grad_sum, weighted)
            state = nnx.state(model, filt)
            updates_tree, opt_state = optimizer.update(grad_sum, opt_state, state)
            nnx.update(model, optax.apply_updates(state, updates_tree))
            rec = {'step': step, 'train_loss': float(np.sum(losses))}
            if step % eval_every == 0 or step == updates:
                mode_scores={}
                # Keep every validation snapshot so D1-D4 can independently
                # choose (step,t0) on Dev, while never inspecting Eval.
                for mode_name,mode_draft in dev_modes.items():
                    dev_scores={}
                    for t0 in T0_GRID:
                        pred=(mode_draft if t0==0.0 else
                              cd._predict_action(model,dev_obs,mode_draft,dev_noise,t0,batch=batch))
                        dev_scores[str(t0)]=float(_mse6(pred,dev['target']).mean())
                    selected_t0=min(T0_GRID,key=lambda x:dev_scores[str(x)])
                    pred=(mode_draft if selected_t0==0.0 else
                          cd._predict_action(model,dev_obs,mode_draft,dev_noise,selected_t0,batch=batch))
                    score=float(_mse6(pred,dev['target']).mean())
                    mode_scores[mode_name]={'dev_mse6_by_t0':dev_scores,'selected_t0':float(selected_t0),
                        'dev_mse6':score,'dev_draft_mse6':float(_mse6(mode_draft,dev['target']).mean()),
                        'dev_pi05_mse6':float(_mse6(dev['normal'],dev['target']).mean()),
                        'output_to_draft_mse6':float(_mse6(pred,mode_draft).mean()),
                        'output_to_pi05_mse6':float(_mse6(pred,dev['normal']).mean())}
                    ckpt_dir=artifacts/f'checkpoint_{condition}'/f'{step:06d}'
                    if not ckpt_dir.exists():_save_state(model,filt,ckpt_dir,ocp)
                    old=best_by_draft[mode_name]
                    if old['mse6'] is None or score<old['mse6']:
                        best_by_draft[mode_name]={'mse6':score,'step':step,'t0':float(selected_t0),
                            'dev_mse6_by_t0':dev_scores,'checkpoint':str(ckpt_dir)}
                primary=mode_scores.get('D1_P5_top1',next(iter(mode_scores.values())))
                rec.update({'dev_modes':mode_scores,'selected_t0':primary['selected_t0'],
                            'dev_mse6':primary['dev_mse6'],'dev_mse6_by_t0':primary['dev_mse6_by_t0']})
            history.append(rec)
            if step == 1 or step % eval_every == 0 or step == updates:
                _write_json(artifacts / f'progress_{condition}.json', {'status': 'running', 'steps_done': step,
                    'loss_first': history[0]['train_loss'], 'loss_latest': history[-1]['train_loss'],
                    'history': history, 'best_dev_by_draft': best_by_draft})
        if any(not np.isfinite(v['mse6']) for v in best_by_draft.values()):
            raise AssertionError(f'{condition} never produced a finite Dev score for every draft mode')
        records[condition] = {'status': 'complete', 'train_rows': len(cache['row_ids']),
                              'updates': updates, 'batch': batch,
                              'gradient_accumulation': int(np.ceil(batch / microbatch)),
                              'microbatch': microbatch,
                              'best_by_draft':best_by_draft,'best':best_by_draft.get('D1_P5_top1'),
                              'history': history}
        _write_json(artifacts / f'progress_{condition}.json', {'status': 'complete', 'steps_done': updates,
            'loss_first': history[0]['train_loss'], 'loss_latest': history[-1]['train_loss'],
            'history': history, 'best_dev_by_draft':best_by_draft})
    _write_json(artifacts / 'train.json', {'status': 'complete', 'conditions': records,
        'dev_tasks': sorted(set(map(int, dev['task']))), 'dev_rows': len(dev['row_ids']),
        'selection': 'minimum live-weight Dev arm6 MSE over checkpoint steps and t0 grid; no Eval rows used',
        'training_t0': 'Uniform[0,.3), denominator floor 1e-3 only for numerical stability',
        'trainable': 'public pi05 action Expert suffix filter; VLM prefix frozen',
        'loss': 'arm six channels only; gripper drafts and final gripper use ordinary pi05',
        'artifact_dir': str(artifacts),'dev_draft_modes':list(dev_modes),
        'checkpoint_policy':'save each Dev snapshot for per-draft step/t0 selection'})
    return records


def train_smoke(out, *, threads=8, batch=2, updates=24, eval_every=12, mem_fraction=.15):
    return train_conditions(out, threads=threads, batch=batch, updates=updates,
                            eval_every=eval_every, mem_fraction=mem_fraction,
                            allow_oracle_dev_smoke=True)


def render_report(prep, smoke_result, out):
    is_smoke = smoke_result.get('status') == 'complete'
    heading = '実装smoke' if is_smoke else '本番学習'
    if prep.get('l90x_manifest_smoke'):
        l90x_note = 'このsmokeはX1の4タスクキャッシュでの実装確認に限る。'
    else:
        l90x_note = (f"L90X行キャッシュは{len(prep.get('l90x_fit_tasks') or []) + len(prep.get('l90x_select_tasks_excluded') or [])}タスクをmanifestに持つ。"
                     f"手元に無いHDF5タスク: {prep.get('l90x_missing_tasks', [])}。"
                     'Bの学習行では選択用16タスクを除き、Dev/Evalタスクは含めない。')
    lines = [f'# EXP3：取り出した単位から直すExpert（{heading}）', '',
        ('この実行は実装確認用の短いsmokeで、本番学習・評価ではない。各窓の正解は学習時の単位選択と教師、およびDevでのチェックポイント/t0選択にだけ使用。Evalは未使用。' if is_smoke else
         'この実行は本番学習のみで、各窓の正解は学習時の単位選択と教師、およびDevでのチェックポイント/t0選択に使用。Evalは未使用。'),
        '単位はLIBERO-40 memory行から speed-minimum + gripper-switch、最短5 frameで区切り、DCT-3復元した自然長の列。推論窓は単位の先頭を自然速度で再生し、10 frame未満は末尾値保持。倍率・姿勢合わせなし。',
        f"Aはmemory外LIBERO-40行、BはA + L90X fit行。{l90x_note} L90Xは場面がLIBERO-90と重なる。", '',
        '| 条件 | train窓 | Dev窓 / task | 更新 | 最良step/t0 | Dev draft MSE6 | Dev PI0.5 MSE6 | Dev Expert MSE6 | 出力−draft | 出力−PI0.5 | loss 初/終 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in ('A', 'B'):
        x = smoke_result['conditions'][cond]
        best = x['best']
        ev = next(h for h in reversed(x['history']) if 'dev_mse6' in h and h['step'] == best['step'])
        lines.append(f"|{cond}|{x['train_rows']}|{smoke_result['dev_rows']} / {smoke_result['dev_tasks']}|{x['updates']}|{best['step']} / {best['t0']:.2f}|{ev['dev_draft_mse6']:.5f}|{ev['dev_pi05_mse6']:.5f}|{ev['dev_mse6']:.5f}|{ev['output_to_draft_mse6']:.5f}|{ev['output_to_pi05_mse6']:.5f}|{x['history'][0]['train_loss']:.5f}/{x['history'][-1]['train_loss']:.5f}|")
    lines += ['', '## 本番設定と起動', '',
        '本番ではAを19,998 memory外LIBERO-40行、BをA + L90X 78-task fit分（select tasksは除外）で学習する。DEVはLIBERO-90 task 6/7のみでcheckpointとt0を選び、Eval 619行は選択後に一度だけ評価する。',
        '推奨本番: A/Bを各5,000更新、実効batch 8、t0~Uniform[0,.3)。H200で逐次なら概算A 1.0–1.5 GPU h、B 1.2–1.8 GPU h（画像prefixを毎stepに計算、prepareは別途約0.5–1 GPU h）。先にCPUでunit selectionを作る時間はtask exclusion付き約0.5–2 hを見込む。実測smokeからの外挿であり、本番速度は未確認。',
        'Full L90X行キャッシュを用意してから以下を実行（CPUで単位選択、低fractionでprepare後、A/Bは同一GPU上でfraction .30ずつ並列）：',
        '```bash',
        'cd /work && setsid -f skill_moe/skillnet/examples/memory_pilot/unseen_libero/run-exp3b-parallel.sh',
        '```', '', 'この確認では通常π0.5 bridge出力が小規模Devで予測値として使われたが、長期学習・Eval・閉ループ試行は実行していない。']
    Path(out, 'report.md').write_text('\n'.join(lines) + '\n')


def run_smoke(out, *, threads=8, batch=2, updates=24, eval_every=12, mem_fraction=.15,
              n_train=12, n_l90x=12, n_dev=8, l90x_dir=L90X_SMOKE):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    prep_dir = out / 'prepared'
    prepare(prep_dir, smoke=True, threads=threads, n_train=n_train, n_l90x=n_l90x,
            n_dev=n_dev, l90x_dir=l90x_dir, mem_fraction=mem_fraction)
    prep = json.loads((prep_dir / 'prepare.json').read_text())
    result = train_smoke(prep_dir, threads=threads, batch=batch, updates=updates,
                         eval_every=eval_every, mem_fraction=mem_fraction)
    write = {'status': 'complete', 'conditions': result,
             'train_rows_a': prep['condition_a_rows'], 'train_rows_l90x': prep['condition_b_l90x_fit_rows'],
             'dev_rows': prep['dev_rows'], 'dev_tasks': prep['dev_tasks']}
    _write_json(out / 'smoke.json', write)
    render_report(prep, write, out)
    return write


def run_production(out, *, steps=5000, threads=8, batch=8, l90x_dir=None, selection_dir=None):
    """Prepare full A/B caches and train both conditions, without using Eval."""
    if l90x_dir is None:
        raise ValueError('production requires --l90x-dir containing all 78 L90X tasks')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if selection_dir is None:
        selection_dir = out / 'selections'
        precompute_unit_selections(selection_dir, l90x_dir=l90x_dir, threads=threads)
    elif not (Path(selection_dir) / 'selection_manifest.json').is_file():
        raise FileNotFoundError(f'incomplete precomputed selection cache: {selection_dir}')
    prep_dir = out / 'prepared'
    prepare(prep_dir, smoke=False, threads=threads, l90x_dir=l90x_dir, mem_fraction=.20,
            selection_dir=selection_dir)
    prep = json.loads((prep_dir / 'prepare.json').read_text())
    train = train_conditions(prep_dir, threads=threads, batch=batch, updates=steps,
                             eval_every=500, mem_fraction=.20, allow_oracle_dev_smoke=False)
    result = {'status': 'training_complete', 'conditions': train,
              'train_rows_a': prep['condition_a_rows'],
              'train_rows_l90x': prep['condition_b_l90x_fit_rows'],
              'dev_rows': prep['dev_rows'], 'dev_tasks': prep['dev_tasks']}
    _write_json(out / 'result.json', result)
    render_report(prep, result, out)
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('smoke')
    s.add_argument('--out', type=Path, default=OUT / 'smoke-01')
    s.add_argument('--threads', type=int, default=8)
    s.add_argument('--batch', type=int, default=2)
    s.add_argument('--updates', type=int, default=24)
    s.add_argument('--eval-every', type=int, default=12)
    s.add_argument('--mem-fraction', type=float, default=.15)
    s.add_argument('--n-train', type=int, default=12)
    s.add_argument('--n-l90x', type=int, default=12)
    s.add_argument('--n-dev', type=int, default=8)
    s.add_argument('--l90x-dir', type=Path, default=L90X_SMOKE)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--l90x-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-train', type=int, default=12)
    p.add_argument('--n-l90x', type=int, default=12)
    p.add_argument('--n-dev', type=int, default=8)
    p.add_argument('--mem-fraction', type=float, default=.15)
    p.add_argument('--selection-dir', type=Path)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--l90x-dir', type=Path, required=True)
    r.add_argument('--steps', type=int, default=5000)
    r.add_argument('--threads', type=int, default=8)
    r.add_argument('--batch', type=int, default=8)
    r.add_argument('--selection-dir', type=Path)
    t = sub.add_parser('train')
    t.add_argument('--prepared-dir', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--condition', choices=('A', 'B'), required=True)
    t.add_argument('--steps', type=int, default=5000)
    t.add_argument('--threads', type=int, default=8)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--microbatch', type=int)
    t.add_argument('--mem-fraction', type=float, default=.30)
    a = ap.parse_args()
    if a.command == 'smoke':
        result = run_smoke(a.out, threads=a.threads, batch=a.batch, updates=a.updates,
                           eval_every=a.eval_every, mem_fraction=a.mem_fraction,
                           n_train=a.n_train, n_l90x=a.n_l90x, n_dev=a.n_dev, l90x_dir=a.l90x_dir)
    elif a.command == 'prepare':
        result = prepare(a.out, smoke=a.smoke, threads=a.threads, n_train=a.n_train,
                         n_l90x=a.n_l90x, n_dev=a.n_dev, l90x_dir=a.l90x_dir,
                         mem_fraction=a.mem_fraction, selection_dir=a.selection_dir)
    else:
        if a.command == 'run':
            result = run_production(a.out, steps=a.steps, threads=a.threads, batch=a.batch,
                                    l90x_dir=a.l90x_dir, selection_dir=a.selection_dir)
        else:
            result = train_conditions(a.prepared_dir, artifact_dir=a.out, threads=a.threads,
                                      batch=a.batch, updates=a.steps, eval_every=500,
                                      mem_fraction=a.mem_fraction, conditions=(a.condition,),
                                      microbatch=a.microbatch)
    print(json.dumps({'status': result.get('status'), 'out': str(getattr(a, 'out', ''))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
