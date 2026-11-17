#!/usr/bin/env python3
"""KEY3: KEYGRID rankers trained with LIBERO-40 memory rows plus L90X.

L90X is deliberately treated as a distinct task/episode namespace.  Its
actions supervise only soft arm-MSE ranking against the unchanged 4,516-unit
LIBERO-40 memory bank; no L90X unit is added to memory.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import key_grid as kg  # noqa: E402
import key_shared_latent as k2  # noqa: E402
import latent_action_units_la2 as la2  # noqa: E402

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
DEFAULT_OUT = ROOT / 'key-grid-l90x-01'
L90X_ROWS = ROOT / 'expert-refine-exp3b-01/l90x_rows'
L90X_VLM = ROOT / 'latent-action-01/la2-production/features/l90x_pooled.npy'
VJEPA_MODEL = ROOT / 'latent-probe-01/data/vjepa2-vitl-fpc64-256'
RUN_ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
KEYGRID_RESULT = ROOT / 'key-grid-01/results.json'
KEY2_RESULT = ROOT / 'key-shared-01/attempt-27-production/results.json'
SEED = 1


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_l90x_rows(row_dir: Path = L90X_ROWS) -> dict[str, Any]:
    manifest = json.loads((Path(row_dir) / 'manifest.json').read_text())
    la2.assert_l90x_manifest(manifest)
    tasks = tuple(map(int, manifest['tasks']))
    expected = set(range(90)) - {6, 7, 0, 10, 13, 31, 35, 43, 57, 61, 84, 88}
    missing = {int(x['task_id']) if isinstance(x,dict) else int(x) for x in manifest.get('missing_tasks', [])}
    # This source's established omission is task 51 (no HDF5), while the
    # task manifest still records the requested task universe.
    if missing and missing != {51}:
        raise AssertionError(f'unexpected unavailable L90X tasks: {sorted(missing)}')
    if set(tasks) != expected:
        raise AssertionError(f'L90X task list differs from fixed 77-task set: missing={sorted(expected-set(tasks))}, extra={sorted(set(tasks)-expected)}')
    forbidden = {0, 6, 7, 10, 13, 31, 35, 43, 57, 61, 84, 88}
    if set(tasks) & forbidden:
        raise AssertionError('fixed Dev/Eval tasks leaked into L90X manifest')
    task = np.load(Path(row_dir) / 'task.npy', mmap_mode='r', allow_pickle=False)
    populated = set(map(int, np.unique(task)))
    if populated != set(tasks) - {51}:
        raise AssertionError(f'populated task set mismatch: {sorted(populated ^ (set(tasks)-{51}))}')
    return {'manifest_tasks': len(tasks), 'populated_tasks': len(populated), 'missing_tasks': sorted(missing),
            'rows': int(manifest['rows']), 'excluded_dev_eval_tasks': sorted(forbidden)}


def extract_l90x_vjepa(row_dir: Path, output: Path, *, device: str = 'cuda:0',
                        memory_fraction: float = .10, batch_size: int = 8,
                        max_rows: int | None = None) -> dict[str, Any]:
    """Extract exact two-camera V-JEPA tokens at L90X query frames.

    The two input frames are source frame t-1 and t (never t+1). Images are
    loaded from the already orientation-corrected HDF5 source and resized in
    the same way as key_inputs_vjepa's KEYGRID cache builder.
    """
    from PIL import Image
    import h5py
    import key_inputs_vjepa as kvj
    from unseen_paths import orient_libero_frame
    row_dir, output = Path(row_dir), Path(output)
    contract = validate_l90x_rows(row_dir)
    man = json.loads((row_dir / 'manifest.json').read_text())
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in row_dir.glob('*.npy')}
    n = int(man['rows']) if max_rows is None else min(int(max_rows), int(man['rows']))
    output.mkdir(parents=True, exist_ok=True)
    keys_path = output / 'keys.npy'
    keys = np.stack([rows['task'][:n], rows['episode'][:n], rows['frame'][:n]], axis=1).astype(np.int32)
    if keys_path.exists():
        if not np.array_equal(np.load(keys_path, mmap_mode='r', allow_pickle=False), keys):
            raise ValueError('existing L90X V-JEPA cache uses a different row-key order')
    else:
        np.save(keys_path, keys, allow_pickle=False)
    bpath, wpath, donepath = output / 'base.npy', output / 'wrist.npy', output / 'completed.npy'
    shape = (n, 64, 1024)
    base = np.lib.format.open_memmap(bpath, mode='r+' if bpath.exists() else 'w+', dtype=np.float16, shape=shape)
    wrist = np.lib.format.open_memmap(wpath, mode='r+' if wpath.exists() else 'w+', dtype=np.float16, shape=shape)
    done = np.lib.format.open_memmap(donepath, mode='r+' if donepath.exists() else 'w+', dtype=np.bool_, shape=(n,))
    if not donepath.exists(): done[:] = False; done.flush()
    if (output / 'manifest.json').exists() and json.loads((output / 'manifest.json').read_text()).get('complete'):
        if bool(np.asarray(done).all()):
            return json.loads((output / 'manifest.json').read_text())
        raise ValueError('manifest marked complete but resume bitmap has gaps')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    model, processor = kvj._load_model(VJEPA_MODEL, device=device, memory_fraction=memory_fraction)
    episodes = {(int(e['task_id']), int(e['episode'])): e for e in man['episodes']}
    pending = np.flatnonzero(~np.asarray(done))
    # Process small batches grouped by episode so HDF5 files are opened once.
    groups: dict[tuple[int, int], list[int]] = {}
    for i in pending:
        groups.setdefault((int(rows['task'][i]), int(rows['episode'][i])), []).append(int(i))
    completed = 0
    for (task, episode), ids0 in groups.items():
        rec = episodes[(task, episode)]
        with h5py.File(rec['hdf5'], 'r') as f:
            obs = f['data'][rec['demo']]['obs']
            bsrc = obs['agentview_rgb']; wsrc = obs['eye_in_hand_rgb']
            for off in range(0, len(ids0), int(batch_size)):
                ids = ids0[off:off + int(batch_size)]
                videos = {'base': [], 'wrist': []}
                for idx in ids:
                    frame = int(rows['frame'][idx])
                    raw_t = min(2 * frame, len(bsrc) - 1)
                    raw_prev = max(0, raw_t - 1)
                    for cam, src in (('base', bsrc), ('wrist', wsrc)):
                        pair = []
                        for fr in (raw_prev, raw_t):
                            image = orient_libero_frame(np.asarray(src[fr], dtype=np.uint8))
                            pil = Image.fromarray(image).convert('RGB').resize((128, 128), resample=Image.Resampling.BICUBIC)
                            pair.append(pil)
                        videos[cam].append(pair)
                with torch.inference_mode():
                    for cam, dest in (('base', base), ('wrist', wrist)):
                        pix = processor(videos=videos[cam], return_tensors='pt')['pixel_values_videos'].to(device)
                        feats = model(pixel_values_videos=pix).last_hidden_state
                        if feats.ndim != 3 or feats.shape[1:] != (256, 1024):
                            raise ValueError(f'unexpected V-JEPA output {tuple(feats.shape)}')
                        grid = feats.reshape(len(ids), 16, 16, 1024)
                        pooled = grid.reshape(len(ids), 8, 2, 8, 2, 1024).mean((2, 4))
                        dest[ids] = pooled.reshape(len(ids), 64, 1024).float().cpu().numpy().astype(np.float16)
                done[ids] = True; completed += len(ids)
                if completed % 256 < len(ids):
                    base.flush(); wrist.flush(); done.flush()
                    _atomic_json(output / 'progress.json', {'done': int(np.asarray(done).sum()), 'total': n,
                        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    base.flush(); wrist.flush(); done.flush()
    if not bool(np.asarray(done).all()): raise RuntimeError('V-JEPA feature extraction incomplete')
    result = {'contract': 'key3-l90x-vjepa-v1', 'complete': True, 'rows': n, 'smoke_prefix': max_rows is not None,
        'shape_per_camera': list(shape[1:]), 'dtype': 'float16', 'model_id': kvj.MODEL_ID,
        'source_manifest': str(row_dir / 'manifest.json'), 'source_rows': contract,
        'key_fields': ['task', 'episode', 'frame'], 'temporal_order': ['source frame t-1', 'source frame t'],
        'orientation': 'orient_libero_frame, same corrected orientation as L90X row cache',
        'resize': 'PIL bicubic 128x128; V-JEPA processor', 'spatial_pool': '16x16 to 8x8, 2x2 mean',
        'extracted_rows': n}
    _atomic_json(output / 'manifest.json', result)
    del model, processor, base, wrist, done
    return result


def extract_l90x_vlm(row_dir: Path, output: Path, *, batch_size: int = 8,
                     max_rows: int | None = None) -> dict[str, Any]:
    """Save KEYGRID-compatible final prefix token rows, resumably, from L90X."""
    row_dir, output = Path(row_dir), Path(output)
    validate_l90x_rows(row_dir)
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in row_dir.glob('*.npy')}
    n = len(rows['task']) if max_rows is None else min(int(max_rows),len(rows['task'])); output.mkdir(parents=True, exist_ok=True)
    keys = np.stack([rows['task'][:n], rows['episode'][:n], rows['frame'][:n]], axis=1).astype(np.int32)
    kp, tp, gp, dp = (output / name for name in ('keys.npy','tokens.npy','groups.npy','completed.npy'))
    if kp.exists() and not np.array_equal(np.load(kp,mmap_mode='r',allow_pickle=False),keys):
        raise ValueError('L90X VLM cache row order differs from source manifest')
    if not kp.exists(): np.save(kp, keys, allow_pickle=False)
    tok = np.lib.format.open_memmap(tp, mode='r+' if tp.exists() else 'w+', dtype=np.float16, shape=(n,712,2048))
    grp = np.lib.format.open_memmap(gp, mode='r+' if gp.exists() else 'w+', dtype=np.int8, shape=(n,712))
    done = np.lib.format.open_memmap(dp, mode='r+' if dp.exists() else 'w+', dtype=np.bool_, shape=(n,))
    if not dp.exists(): done[:] = False; done.flush()
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text()).get('complete'):
        if bool(np.asarray(done).all()): return json.loads((output/'manifest.json').read_text())
        raise ValueError('VLM manifest complete but cache has unfinished rows')
    # Use the same policy checkpoint, prefix tap and transformed row cache as KEYGRID.
    import theta_regression as tr
    runmod = tr._module(HERE.parent/'pi05_memory'/'run.py', 'key3_pi05_run')
    dm = json.loads((RUN_ROOT/'data_manifest.json').read_text())
    policy, _ = runmod.load_policy(Path(dm.get('checkpoint','/work/.skillnet-state/checkpoints/pi05_libero')))
    from openpi.shared.nnx_utils import module_jit
    prefix = module_jit(policy.prefix_features)
    ids = np.flatnonzero(~np.asarray(done))
    for st in range(0,len(ids),int(batch_size)):
        ix=ids[st:st+int(batch_size)]
        obs=runmod.obs_batch(rows,ix)
        features,groups=prefix(obs)
        tok[ix]=np.asarray(features,np.float16);grp[ix]=np.asarray(groups,np.int8);done[ix]=True
        tok.flush();grp.flush();done.flush()
        if (st+len(ix))%256 < len(ix):
            _atomic_json(output/'progress.json',{'done':int(np.asarray(done).sum()),'total':n,
                'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if not bool(np.asarray(done).all()): raise RuntimeError('incomplete L90X VLM token extraction')
    result={'contract':'key3-l90x-vlm-v1','complete':True,'rows':n,'smoke_prefix':max_rows is not None,'tokens_shape':[712,2048],
        'groups_shape':[712],'dtype':'float16/int8','source_rows':str(row_dir),'checkpoint':str(dm.get('checkpoint')),
        'method':'same pi05 final prefix_features and group mask as KEYGRID; correctly oriented L90X row cache',
        'target_actions_used':False}
    _atomic_json(output/'manifest.json',result)
    return result


def _load_l90x(row_dir: Path, vlm_dir: Path, vlm_file: Path, vjepa_dir: Path) -> dict[str, np.ndarray]:
    validate_l90x_rows(row_dir)
    m = json.loads((Path(row_dir) / 'manifest.json').read_text())
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in Path(row_dir).glob('*.npy')}
    pooled_all = np.load(vlm_file, mmap_mode='r', allow_pickle=False)
    if pooled_all.ndim!=2 or pooled_all.shape[1:]!=(2048,) or len(pooled_all)>int(m['rows']):
        raise ValueError(f'bad exact VLM feature shape {pooled_all.shape}')
    pooled = pooled_all
    n=len(pooled)
    if n<1:raise ValueError('empty L90X pooled feature cache')
    vman = json.loads((Path(vjepa_dir) / 'manifest.json').read_text())
    if vman.get('complete') is not True or vman.get('rows') != len(pooled): raise ValueError('incomplete V-JEPA cache')
    keys = np.stack([rows['task'][:n], rows['episode'][:n], rows['frame'][:n]], axis=1).astype(np.int32)
    vman0=json.loads((Path(vlm_dir)/'manifest.json').read_text())
    if vman0.get('complete') is not True or vman0.get('rows')!=len(keys): raise ValueError('incomplete L90X VLM token cache')
    tkeys=np.load(Path(vlm_dir)/'keys.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(keys,tkeys): raise AssertionError('L90X VLM row-key alignment changed')
    tokens=np.load(Path(vlm_dir)/'tokens.npy',mmap_mode='r',allow_pickle=False)
    groups=np.load(Path(vlm_dir)/'groups.npy',mmap_mode='r',allow_pickle=False)
    if tokens.shape!=(len(keys),712,2048) or groups.shape!=(len(keys),712): raise ValueError('bad VLM token cache shapes')
    ckeys = np.load(Path(vjepa_dir) / 'keys.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(keys, ckeys): raise AssertionError('L90X V-JEPA row-key alignment changed')
    b = np.load(Path(vjepa_dir) / 'base.npy', mmap_mode='r', allow_pickle=False)
    w = np.load(Path(vjepa_dir) / 'wrist.npy', mmap_mode='r', allow_pickle=False)
    vtokens = np.empty((len(keys), 128, 1024), np.float16)
    vtokens[:, :64] = b; vtokens[:, 64:] = w
    vgroups = np.tile(np.r_[np.zeros(64, np.int64), np.ones(64, np.int64)], (len(keys), 1))
    hist = np.asarray(rows['hist'], np.float32)
    has = np.asarray(rows['has_hist'], bool)
    return {'key': keys, 'pooled': np.asarray(pooled, np.float32), 'tokens': tokens, 'groups': groups,
        'vjepa_tokens': vtokens,
        'vjepa_groups': vgroups, 'history': hist, 'history_mask': np.repeat(has[:, None], 10, axis=1),
        'state': np.asarray(rows['state'][:, :8], np.float32),
        'actions': np.asarray(rows['actions'][:, :, :7], np.float32),
        # Domains are disjoint: numeric task IDs are never compared across suites.
        'task': np.asarray(rows['task'], np.int32) + 1000,
        'episode': np.asarray(rows['episode'], np.int64) + 1_000_000,
        'source_task': np.asarray(rows['task'], np.int32), 'source_episode': np.asarray(rows['episode'], np.int32)}


def _merge_q_parts(mem: dict[str, np.ndarray], ext: dict[str, np.ndarray],
                   mem_candidate_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    keys = ('key', 'history', 'history_mask', 'state', 'actions', 'task', 'episode')
    train = {k: np.concatenate([np.asarray(mem[k]), np.asarray(ext[k])], axis=0) for k in keys}
    train['key'] = np.concatenate([np.asarray(mem['key']),
        np.column_stack([ext['source_task'], ext['source_episode'], np.asarray(ext['key'])[:, 2]])], axis=0).astype(np.int32)
    ids = np.r_[np.asarray(mem_candidate_ids, np.int64), np.full(len(ext['key']), -1, np.int64)]
    train['domain'] = np.r_[np.zeros(len(mem['key']), np.int8), np.ones(len(ext['key']), np.int8)]
    return train, ids


def same_episode_allowed(query_task: np.ndarray, query_episode: np.ndarray,
                         candidate_task: np.ndarray, candidate_episode: np.ndarray) -> np.ndarray:
    """Same-episode exclusion, after suite IDs have been namespaced."""
    qt,qe,ct,ce=map(np.asarray,(query_task,query_episode,candidate_task,candidate_episode))
    if qt.ndim!=1 or qe.shape!=qt.shape or ct.ndim!=1 or ce.shape!=ct.shape:
        raise ValueError('task and episode arrays must be aligned vectors')
    return ~((qt[:,None]==ct[None,:])&(qe[:,None]==ce[None,:]))


def _external_part(ext: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.asarray(v)[indices] for k, v in ext.items()
            if isinstance(v, np.ndarray) and len(v) == len(ext['key'])}


def _get_train_part(mem: dict[str, np.ndarray], ext: dict[str, np.ndarray], qids: np.ndarray,
                    mem_count: int) -> dict[str, np.ndarray]:
    """Gather a mixed minibatch without concatenating multi-GB token arrays."""
    ismem = qids < mem_count
    out: dict[str, np.ndarray] = {}
    mem_ix, ext_ix = qids[ismem], qids[~ismem] - mem_count
    small = ('history', 'history_mask', 'state', 'actions', 'task', 'episode')
    for k in small:
        out[k] = np.concatenate([np.asarray(mem[k])[mem_ix], np.asarray(ext[k])[ext_ix]], axis=0)
    out['key'] = np.concatenate([np.asarray(mem['key'])[mem_ix],
        np.column_stack([ext['source_task'][ext_ix], ext['source_episode'][ext_ix], ext['key'][ext_ix, 2]])], axis=0).astype(np.int32)
    out['tokens'] = np.concatenate([np.asarray(mem['tokens'])[mem_ix], np.asarray(ext['tokens'])[ext_ix]], axis=0)
    out['groups'] = np.concatenate([np.asarray(mem['groups'])[mem_ix], np.asarray(ext['groups'])[ext_ix]], axis=0)
    out['vjepa_tokens'] = np.concatenate([np.asarray(mem['vjepa_tokens'])[mem_ix], np.asarray(ext['vjepa_tokens'])[ext_ix]], axis=0)
    out['vjepa_groups'] = np.concatenate([np.asarray(mem['vjepa_groups'])[mem_ix], np.asarray(ext['vjepa_groups'])[ext_ix]], axis=0)
    return out


def _train_mlp(input_name: str, query_vectors: np.ndarray, fit_k: np.ndarray, train: dict[str, np.ndarray],
               ext: dict[str, np.ndarray], fit_actions: np.ndarray, fit_task: np.ndarray, fit_episode: np.ndarray,
               mem_candidate_ids: np.ndarray, *,
               seed: int, lr: float, epochs: int, tau: float, device: str, smoke: bool):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 101)
    model = kg.PairMLPRanker(fit_q.shape[1], shared=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    m = len(train['key']); n = m + len(ext['key']); qids = np.arange(n); hist=[]
    q_mem_ids = np.asarray(mem_candidate_ids, np.int64)
    for ep in range(epochs):
        rng.shuffle(qids); model.train(); losses=[]
        for s in range(0, n, 8):
            qi = qids[s:s+8]; in_mem = qi < m
            qmat = np.asarray(query_vectors[qi], np.float32)
            rand = rng.choice(len(fit_k), size=min(256 if not smoke else 32, len(fit_k)), replace=False)
            cids = np.unique(np.r_[q_mem_ids[qi[in_mem]], rand]); cids = cids[cids >= 0]
            # Random candidates are always retained; this protects all-external minibatches.
            cids = np.unique(np.r_[cids, rand])
            tp = _get_train_part(train, ext, qi, m)
            allowed = same_episode_allowed(tp['task'],tp['episode'],np.asarray(fit_task)[cids],np.asarray(fit_episode)[cids])
            q = torch.as_tensor(qmat, device=device, dtype=torch.float32)
            k = torch.as_tensor(fit_k[cids], device=device, dtype=torch.float32)
            score = model(q, k) * model.logit_scale.detach().exp().clamp(max=k2.LOGIT_SCALE_MAX)
            errs = k2._arm6_candidate_mse(tp['actions'], fit_actions[cids])
            loss = k2.contrastive_teacher_loss(score, torch.as_tensor(errs, device=device),
                torch.as_tensor(allowed, device=device), objective='soft', teacher_temperature=tau)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            k2._clamp_logit_scale_(model); losses.append(float(loss.detach()))
        hist.append({'epoch': ep+1, 'loss': float(np.mean(losses))})
    return model, hist


def _concat_split_parts(mem: dict[str, np.ndarray], ext: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Lightweight merged E inputs for evaluation; token data stays under 1GB per split."""
    return {k: np.concatenate([np.asarray(mem[k]), np.asarray(ext[k])], axis=0)
            for k in ('key','history','history_mask','state','actions','task','episode','tokens','groups','vjepa_tokens','vjepa_groups')}


def _train_e(mode: str, mem: dict[str, np.ndarray], ext: dict[str, np.ndarray],
             bank_input: dict[str, np.ndarray], fit_bank_input: dict[str, np.ndarray],
             fit_actions: np.ndarray, fit_task: np.ndarray, fit_episode: np.ndarray,
             mem_candidate_ids: np.ndarray, *, seed: int, lr: float,
             epochs: int, tau: float, device: str, smoke: bool):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 101)
    model = k2._new_model(mode, token_count=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    m = len(mem['key']); n = m + len(ext['key']); qids = np.arange(n); logs=[]
    for ep in range(epochs):
        rng.shuffle(qids); model.train(); losses=[]
        for s in range(0, n, 8):
            qi=qids[s:s+8]; tp=_get_train_part(mem,ext,qi,m)
            rand=rng.choice(len(fit_actions),size=min(256 if not smoke else 32,len(fit_actions)),replace=False)
            positives=np.asarray(mem_candidate_ids)[qi[qi<m]]
            cids=np.unique(np.r_[positives,rand]); cids=cids[cids>=0]
            allowed=same_episode_allowed(tp['task'],tp['episode'],np.asarray(fit_task)[cids],np.asarray(fit_episode)[cids])
            qpart={k:tp[k] for k in ('tokens','groups','history','history_mask','state')}
            kpart={k:np.asarray(fit_bank_input[k])[cids] for k in ('tokens','groups','history','history_mask','state')}
            qr=k2._model_batch_encode(model,mode,qpart,device)
            kr=k2._model_batch_encode(model,mode,kpart,device,key=True)
            scores=(qr[:,None]*kr[None]).sum(-1).mean(-1)*model.logit_scale.detach().exp().clamp(max=k2.LOGIT_SCALE_MAX)
            errors=k2._arm6_candidate_mse(tp['actions'],fit_actions[cids])
            loss=k2.contrastive_teacher_loss(scores,torch.as_tensor(errors,device=device),
                torch.as_tensor(allowed,device=device),objective='soft',teacher_temperature=tau)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            k2._clamp_logit_scale_(model);losses.append(float(loss.detach()))
        logs.append({'epoch':ep+1,'loss':float(np.mean(losses))})
    return model,logs


def _metrics_for_model(model: nn.Module, mode: str, bank: dict[str,np.ndarray], bi: dict[str,np.ndarray],
                       split: dict[str,np.ndarray], bank_vectors: np.ndarray | None, split_vectors: np.ndarray | None,
                       candidate_actions: np.ndarray, device: str) -> tuple[dict[str,Any],np.ndarray]:
    if mode in ('history_state','vlm_history_state'):
        scores=kg._score_vectors(model,split_vectors,bank_vectors,device=device)
        errors=k2._arm6_candidate_mse(split['actions'],candidate_actions)
    else:
        scores,errors=k2._score_model(model,mode,bank,bi,split,device)
    result=kg._split_eval(scores,errors,split)
    return result,np.asarray(result['top1_per_query'],np.float32)


def _baseline_keygrid_rows(results: dict[str,Any]) -> dict[str,dict[str,Any]]:
    rows={}
    # The 40-only row is the KEYGRID checkpoint selected on its fixed Dev.
    h=results['new_models']['history_state']['variants']
    rows['history_state']=min(h.values(),key=lambda v:float(v['dev_top1_arm6_mse']))
    # P5 is the explicit VLM+history+state two-tower concatenation baseline.
    rows['p5']=results['new_models']['vlm_history_state']['variants']['two_tower_mlp']
    for key,mode in (('e_vlm_h','vlm_history_state'),('e_vj_h','vjepa_history_state'),
                     ('e_vj_vlm_h','vjepa_vlm_history_state')):
        rows[key]=results['existing_reselected'][mode]['selected_by_dev_top1']
    return rows


def _load_40_checkpoint(name: str, row: dict[str,Any], input_name: str,
                        input_mode: str, bank_vectors: np.ndarray, *, device: str) -> nn.Module:
    ckpt=Path(row['checkpoint'])
    if not ckpt.exists(): raise FileNotFoundError(f'KEYGRID baseline checkpoint missing: {ckpt}')
    if name in ('history_state','p5'):
        model=kg.PairMLPRanker(bank_vectors.shape[-1],shared=False).to(device)
    else:
        model=k2._new_model(input_mode,token_count=5).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True));model.eval()
    return model


def run_keygrid_l90x(output: Path, *, row_dir: Path=L90X_ROWS, vlm_dir: Path|None=None,
                     vjepa_dir: Path|None=None, vlm_file: Path=L90X_VLM, device: str='cuda:0',
                     memory_fraction: float=.10, seed: int=SEED, learning_rates=(1e-4,3e-4),
                     epochs_grid=(4,8,16), tau_grid=(.002,.005,.01,.02), smoke: bool=False) -> dict[str,Any]:
    output=Path(output);vlm_dir=vlm_dir or output/'features/l90x_vlm';vjepa_dir=vjepa_dir or output/'features/l90x_vjepa'
    if output.exists() and any(p.name!='features' for p in output.iterdir()) and not (output/'run_manifest.json').exists():
        raise FileExistsError(f'refusing unmanifested KEY3 output: {output}')
    output.mkdir(parents=True,exist_ok=True)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction),torch.device(device).index or 0)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    contract=validate_l90x_rows(row_dir)
    if not (Path(vlm_dir)/'manifest.json').exists():raise FileNotFoundError(f'missing L90X VLM token cache {vlm_dir}')
    if not Path(vlm_file).exists():raise FileNotFoundError(f'missing pooled VLM features {vlm_file}')
    if not (Path(vjepa_dir)/'manifest.json').exists():raise FileNotFoundError(f'missing L90X V-JEPA cache {vjepa_dir}')
    # Keep the builder identical to KEYGRID: exact 4,516 DCT3 units and fixed split rows.
    def progress(stage,**fields):
        with (output/'progress.jsonl').open('a') as f:
            f.write(json.dumps({'stage':stage,**fields})+'\n')
    runroot=RUN_ROOT
    bank,bi,fit_bank,fbi,train40,learned,unseen,dev=k2._build_key2_queries(
        runroot,ROOT/'key-shared-01/attempt-02/prefix-missing',ROOT/'seg-context-01/seg5-cache',
        ROOT/'seg-context-01/unseen-stride5',
        ROOT/'key-shared-01/attempt-27-production/vjepa/libero90_eval',
        progress_callback=progress)
    if len(bank['length'])!=4516 or len(dev['key'])!=217 or len(learned['key'])!=621 or len(unseen['key'])!=619:
        raise AssertionError('KEYGRID fixed bank/split changed')
    ext=_load_l90x(row_dir,vlm_dir,vlm_file,vjepa_dir)
    if smoke:
        if len(ext['key'])<1:raise AssertionError('empty L90X smoke feature rows')
    elif len(ext['key'])!=11030:raise AssertionError(f'expected complete L90X rows=11030, got {len(ext["key"])}')
    if set(np.unique(ext['source_task'])) & {6,7,0,10,13,31,35,43,57,61,84,88}:
        raise AssertionError('L90X forbidden Dev/Eval task in fit data')
    # Train only against memory units not sourced from fixed learned-eval episodes.
    fit_map={tuple(map(int,k)):i for i,k in enumerate(fbi['key'])}
    mem_pos=np.asarray([fit_map[tuple(map(int,k))] for k in train40['key']],np.int64)
    all_actions=k2._candidate_window_actions(np.arange(len(bank['length'])),bank)
    full_map={tuple(map(int,k)):i for i,k in enumerate(bi['key'])}
    fit_full=np.asarray([full_map[tuple(map(int,k))] for k in fbi['key']],np.int64)
    fit_actions=all_actions[fit_full]
    fit_task=np.asarray(fbi['task'],np.int32);fit_episode=np.asarray(fbi['episode'],np.int64)
    # Memory fit rows already carry the exact prefix/cache feature rows.
    mem={k:np.asarray(v) for k,v in train40.items()}
    train_bank_ids=np.asarray([full_map[tuple(map(int,k))] for k in train40['key']],np.int64)
    for k in ('tokens','groups','vjepa_tokens','vjepa_groups'):
        mem[k]=np.asarray(bi[k])[train_bank_ids]
    train_meta, candidate_ids=_merge_q_parts(mem,ext,mem_pos)
    train_vectors={}
    prepared_bank=kg._prepare_vector_sources(bi)
    prepared_train=kg._prepare_vector_sources(mem)
    for inp in ('history_state','vlm_history_state','vjepa_history_state','vjepa_vlm_history_state'):
        bank_vec=kg.make_vector_features(prepared_bank,inp)
        fit_vec=bank_vec[fit_full]
        mem_vec=kg.make_vector_features(prepared_train,inp)
        ext_vec=kg.make_vector_features(ext,inp)
        train_vectors[inp]=np.concatenate([mem_vec,ext_vec],axis=0)
        if inp=='history_state':
            bank_vectors_h,fit_vectors_h=bank_vec,fit_vec
        elif inp=='vlm_history_state':
            bank_vectors_p5,fit_vectors_p5=bank_vec,fit_vec
    if smoke:
        # Small but real-data end-to-end verification. It still checks all five model paths.
        ext={k:(np.asarray(v)[:64] if isinstance(v,np.ndarray) and len(v)==len(ext['key']) else v) for k,v in ext.items()}
        train_meta,candidate_ids=_merge_q_parts(mem,ext,mem_pos)
        for k in list(train_vectors):
            mv=kg.make_vector_features(prepared_train,k);ev=kg.make_vector_features(ext,k)
            train_vectors[k]=np.concatenate([mv,ev],0)
        learning_rates=(3e-4,);epochs_grid=(1,);tau_grid=(.02,)
    result={'status':'running','seed':seed,'smoke':smoke,'l90x':contract,
        'L90X_fit_rows':int(len(ext['key'])),'memory_fit_rows':int(len(train40['key'])),
        'bank_units':int(len(bank['length'])),'split_rows':{'dev':len(dev['key']),'learned':len(learned['key']),'unseen':len(unseen['key'])},
        'grid':{'learning_rates':list(learning_rates),'epochs':list(epochs_grid),'teacher_tau':list(tau_grid)},
        'methods':{},'paired_episode_bootstrap_5000':{}}
    if not (output/'run_manifest.json').exists():
        _atomic_json(output/'run_manifest.json',{'contract':'key3-l90x-v1','row_dir':str(row_dir),
            'l90x_vlm':str(vlm_dir),'l90x_vjepa':str(vjepa_dir),'grid':result['grid'],'seed':seed})
    # The vector MLP methods and all three shared-latent E modes use the same
    # candidate sampling, teacher, optimizer, data split and Dev selection.
    method_specs=[('history_state','mlp','history_state'),('p5','mlp','vlm_history_state'),
        ('e_vlm_h','e','e_vlm_h'),('e_vj_h','e','e_vj_h'),('e_vj_vlm_h','e','e_vj_vlm_h')]
    grids={}
    deverr=k2._arm6_candidate_mse(dev['actions'],all_actions)
    rng_eval=np.random.default_rng(20261116)
    for method,typ,mode in method_specs:
        inp=mode if typ=='mlp' else None
        cfgs=[];tagroot=output/'tuning'/method;tagroot.mkdir(parents=True,exist_ok=True)
        for lr in learning_rates:
          for ep in epochs_grid:
            for tau in tau_grid:
              tag=f'lr{lr:g}-ep{ep}-tau{tau:g}';sub=tagroot/tag;sub.mkdir(parents=True,exist_ok=True)
              record=sub/'record.json';ckpt=sub/'model.pt'
              if record.exists() and ckpt.exists():
                  row=json.loads(record.read_text());cfgs.append(row);continue
              if typ=='mlp':
                  query_vectors=train_vectors[inp];fit_vec=fit_vectors_h if method=='history_state' else fit_vectors_p5
                  model,history=_train_mlp(inp,query_vectors,fit_vec,train_meta,ext,fit_actions,fit_task,fit_episode,
                      candidate_ids,seed=seed,lr=float(lr),epochs=int(ep),tau=float(tau),device=device,smoke=smoke)
                  dev_vectors=kg.make_vector_features(kg._prepare_vector_sources(dev),inp)
                  scores=kg._score_vectors(model,dev_vectors,bank_vectors_h if method=='history_state' else bank_vectors_p5,device=device)
                  err=deverr
              else:
                  model,history=_train_e(mode,mem,ext,bi,fbi,fit_actions,fit_task,fit_episode,candidate_ids,
                      seed=seed,lr=float(lr),epochs=int(ep),tau=float(tau),device=device,smoke=smoke)
                  scores,err=k2._score_model(model,mode,bank,bi,dev,device)
                  if smoke:
                      # keep the smoke-sized candidate prefix and labels aligned
                      scores=scores[:,:128];err=err[:,:128]
              if smoke and typ=='mlp':
                  scores=scores[:,:128];err=err[:,:128]
              met=kg._rank_metrics(scores,err)
              torch.save(model.state_dict(),ckpt)
              row={'method':method,'mode':mode,'learning_rate':float(lr),'epochs':int(ep),
                   'teacher_temperature':float(tau),'dev_top1_arm6_mse':float(met['top1_arm6_mse']),
                   'dev_metrics':met,'train_history':history,'checkpoint':str(ckpt)}
              _atomic_json(record,row);cfgs.append(row)
              with (output/'progress.jsonl').open('a') as f:f.write(json.dumps({'tuning':method,'config':tag,'dev_top1':row['dev_top1_arm6_mse']})+'\n')
              del model
        best=min(cfgs,key=lambda r:(r['dev_top1_arm6_mse'],r['learning_rate'],r['epochs'],r['teacher_temperature']))
        model=(kg.PairMLPRanker((fit_vectors_h if method=='history_state' else fit_vectors_p5).shape[-1],shared=False).to(device)
               if typ=='mlp' else k2._new_model(mode,token_count=5).to(device))
        model.load_state_dict(torch.load(best['checkpoint'],map_location=device,weights_only=True));model.eval()
        ev={};per={}
        for sname,split in (('learned',learned),('unseen',unseen)):
            if typ=='mlp':
                qv=kg.make_vector_features(kg._prepare_vector_sources(split),inp)
                bv=bank_vectors_h if method=='history_state' else bank_vectors_p5
                scores=kg._score_vectors(model,qv,bv,device=device)
                errors=k2._arm6_candidate_mse(split['actions'],all_actions)
                rowev=kg._split_eval(scores,errors,split)
            else:
                scores,errors=k2._score_model(model,mode,bank,bi,split,device)
                rowev=kg._split_eval(scores,errors,split)
            if smoke:
                # smoke models were tuned on prefix candidates; report smoke only.
                scores=scores[:,:128];errors=errors[:,:128];rowev=kg._split_eval(scores,errors,split)
            ev[sname]=rowev['metrics'];per[sname]=rowev['top1_per_query']
        result['methods'][method]={'selected':best,'candidate_grid':len(cfgs),'evaluation':ev}
        grids[method]={'per':per,'model':model,'typ':typ,'mode':mode,'input':inp}
        del model
    # Re-evaluate original KEYGRID selected models on the identical fixed rows
    # to retain rowwise paired deltas while reporting the published values too.
    old=json.loads(Path(KEYGRID_RESULT).read_text())
    base_rows=_baseline_keygrid_rows(old)
    base_per={};base_metrics={}
    for method,typ,mode in method_specs:
        inp=mode if typ=='mlp' else None
        oldrow=base_rows[method]
        if typ=='mlp':
            bv=bank_vectors_h if method=='history_state' else bank_vectors_p5
            model=kg.PairMLPRanker(bv.shape[-1],shared=False).to(device)
        else:model=k2._new_model(mode,token_count=5).to(device)
        model.load_state_dict(torch.load(oldrow['checkpoint'],map_location=device,weights_only=True));model.eval()
        base_per[method]={};base_metrics[method]={}
        for sname,split in (('learned',learned),('unseen',unseen)):
            if typ=='mlp':
                qv=kg.make_vector_features(kg._prepare_vector_sources(split),inp)
                scores=kg._score_vectors(model,qv,bv,device=device);errors=k2._arm6_candidate_mse(split['actions'],all_actions)
            else:scores,errors=k2._score_model(model,mode,bank,bi,split,device)
            evrow=kg._split_eval(scores,errors,split)
            base_per[method][sname]=evrow['top1_per_query'];base_metrics[method][sname]=evrow['metrics']
        del model
    result['libero40_only_reference']={m:{'selected':base_rows[m], 'evaluation':base_metrics[m]} for m in base_rows}
    result['paired_episode_bootstrap_5000']={}
    for method,_,_ in method_specs:
        result['paired_episode_bootstrap_5000'][method]={}
        for sname,split in (('learned',learned),('unseen',unseen)):
            diff=grids[method]['per'][sname]-base_per[method][sname]
            result['paired_episode_bootstrap_5000'][method][sname]=kg._top1_boot(diff,split,seed+73)
    result['e_minus_history_state_mlp']={}
    for method in ('e_vlm_h','e_vj_h','e_vj_vlm_h'):
        result['e_minus_history_state_mlp'][method]={}
        for sname,split in (('learned',learned),('unseen',unseen)):
            result['e_minus_history_state_mlp'][method][sname]=kg._top1_boot(
                grids[method]['per'][sname]-grids['history_state']['per'][sname],split,seed+91)
    result['reference_rows']={'keygrid_40_only':old.get('key2_reference_rows',{}),
        'la2_unseen_top1_arm_mse':.077}
    result['feature_provenance']={'L90X VLM pooled':str(vlm_file),'L90X VLM tokens':str(vlm_dir),
        'L90X V-JEPA tokens':str(vjepa_dir),'VLM token extraction':json.loads((Path(vlm_dir)/'manifest.json').read_text()),
        'V-JEPA extraction':json.loads((Path(vjepa_dir)/'manifest.json').read_text())}
    result['status']='complete_smoke' if smoke else 'complete'
    _atomic_json(output/'results.json',result);_write_report(output,result)
    return result


def _fmt(m):
    return f"{m['top1_arm6_mse']:.5f}/{m.get('top1_arm6_mse_hard',float('nan')):.5f}"


def _write_report(output: Path, r: dict[str,Any]) -> None:
    names={'history_state':'履歴＋状態 MLP','p5':'P5: VLM＋履歴＋状態','e_vlm_h':'E-VLM＋H',
        'e_vj_h':'E-V-JEPA＋H','e_vj_vlm_h':'E-V-JEPA＋VLM＋H'}
    lines=['# KEY3：L90X を学習に足したキー（参考、場面既知）','',
      '**注意：L90X は LIBERO-90 Eval と場面・物体が重なる。以下は「見た場面での新しいタスク」の参考値であり、LIBERO-40 のみで学習した本線とは分けて読む。**','',
      '## 未学習 LIBERO-90 Eval（参考：場面既知、新しいタスク）','',
      '指標は top-1 腕6軸 MSE（6軸二乗和/70）、全体/固定hard。学習設定は Dev 217行の top-1 腕MSEで選び、Eval 619行を一度だけ採点した。','',
      '|方式|LIBERO-40のみ (全体/難)|LIBERO-40＋L90X (全体/難)|差 L90X−40 [episode bootstrap 95%]|Spearman / Recall@1 / 回収率 (L90X)|',
      '|---|---:|---:|---:|---:|']
    for method,label in names.items():
        b=r['libero40_only_reference'][method]['evaluation']['unseen'];n=r['methods'][method]['evaluation']['unseen']
        ci=r['paired_episode_bootstrap_5000'][method]['unseen']
        lines.append(f"|{label}|{_fmt(b)}|{_fmt(n)}|{ci['mean']:.5f} [{ci['ci95_low']:.5f}, {ci['ci95_high']:.5f}]|{n['spearman_positive']:.4f} / {n['recall@1']:.4f} / {n['recovery']:.4f}|")
    lines += ['', '## 学習済み LIBERO-40 評価（621窓）','',
      '|方式|LIBERO-40のみ (全体/難)|LIBERO-40＋L90X (全体/難)|差 L90X−40 [episode bootstrap 95%]|Spearman / Recall@1 / 回収率 (L90X)|','|---|---:|---:|---:|---:|']
    for method,label in names.items():
        b=r['libero40_only_reference'][method]['evaluation']['learned'];n=r['methods'][method]['evaluation']['learned']
        ci=r['paired_episode_bootstrap_5000'][method]['learned']
        lines.append(f"|{label}|{_fmt(b)}|{_fmt(n)}|{ci['mean']:.5f} [{ci['ci95_low']:.5f}, {ci['ci95_high']:.5f}]|{n['spearman_positive']:.4f} / {n['recall@1']:.4f} / {n['recovery']:.4f}|")
    lines += ['', '## L90X 条件で E と履歴＋状態 MLP の対比較','',
      '|E方式 − 履歴＋状態 MLP|未学習 Eval|学習済み Eval|','|---|---:|---:|']
    for method,label in names.items():
        if method=='history_state':continue
        if method=='p5':continue
        a=r['e_minus_history_state_mlp'][method]['unseen'];b=r['e_minus_history_state_mlp'][method]['learned']
        lines.append(f"|{label}|{a['mean']:.5f} [{a['ci95_low']:.5f}, {a['ci95_high']:.5f}]|{b['mean']:.5f} [{b['ci95_low']:.5f}, {b['ci95_high']:.5f}]|")
    lines += ['', '## 参考値','',
      'KEYGRID 固定行（oracle、直前保持、π0.5、random）と KEYGRID の学習済み条件は `results.json:reference_rows` に収録。LA2 の未学習 top-1 参考値は .077。LA2 は潜在行動単位の評価で、ここでは学習候補単位を増やさず検索キーだけを比較しているため、同一手法の直接比較ではない。','',
      '## プロトコル','',
      f"- 4,516個の LIBERO-40 DCT3 単位を固定候補として使用。L90X は {r['L90X_fit_rows']} query 行、候補単位には追加しない。LIBERO-40 fit queries は {r['memory_fit_rows']} 行。",
      '- L90X は task 6/7 と固定 Eval 10タスクを除外し、task 51 はローカル HDF5 不在で0行。L90X の行動は、同じ4,516候補への腕6軸 soft MSE 教師にのみ使う。',
      '- L90X と LIBERO-40 の task/episode ID を別名前空間にして、無関係な数値IDの一致で誤って候補を除外しない。同一episode除外は LIBERO-40 query と同一 episode の候補だけに適用。',
      '- VLM は KEYGRID と同じ pi05 final-prefix pooled representation、V-JEPA は同じ frozen ViT-L / corrected images / previous-current frames / 2×2 spatial pooling。',
      '- Grid: lr={0.0001,0.0003}, epochs={4,8,16}, teacher τ={0.002,0.005,0.01,0.02}, seed=1。各手法で Dev top-1 腕 MSE により選択。',
      '- L90X は未学習 Eval とシーン・物体が重なるため、汎化の結論には使わない。', '',
      '## 実行再現','',
      '```bash',
      'cd /work && source env/activate.sh',
      'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false',
      'taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/key_grid_l90x.py --stage all --output /work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-grid-l90x-01/production',
      '```']
    (Path(output)/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('extract-vlm','extract-vjepa','run','all'),default='all')
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT/'production')
    ap.add_argument('--row-dir',type=Path,default=L90X_ROWS)
    ap.add_argument('--vlm-dir',type=Path)
    ap.add_argument('--vjepa-dir',type=Path)
    ap.add_argument('--vlm-file',type=Path,default=L90X_VLM)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--memory-fraction',type=float,default=.10)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    vlm_dir=a.vlm_dir or a.output/'features/l90x_vlm';vjepa_dir=a.vjepa_dir or a.output/'features/l90x_vjepa'
    vlm_file=a.vlm_file
    if a.smoke and a.stage in ('run','all') and vlm_file==L90X_VLM:
        vlm_file=a.output/'features/l90x_pooled_smoke.npy';vlm_file.parent.mkdir(parents=True,exist_ok=True)
        if not vlm_file.exists():
            src=np.load(L90X_VLM,mmap_mode='r',allow_pickle=False)
            np.save(vlm_file,np.asarray(src[:8],np.float32),allow_pickle=False)
    if a.stage in ('extract-vlm','all'):extract_l90x_vlm(a.row_dir,vlm_dir,batch_size=8,max_rows=8 if a.smoke else None)
    if a.stage in ('extract-vjepa','all'):extract_l90x_vjepa(a.row_dir,vjepa_dir,device=a.device,memory_fraction=a.memory_fraction,max_rows=8 if a.smoke else None)
    if a.stage in ('run','all'):
        run_keygrid_l90x(a.output,row_dir=a.row_dir,vlm_dir=vlm_dir,vjepa_dir=vjepa_dir,vlm_file=vlm_file,
            device=a.device,memory_fraction=a.memory_fraction,smoke=a.smoke)


if __name__=='__main__':main()

