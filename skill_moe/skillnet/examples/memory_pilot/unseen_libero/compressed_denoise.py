#!/usr/bin/env python3
"""Train/evaluate denoisers for variable-length DCT motion-unit drafts.

The oracle choice is intentionally privileged: target actions may select a single
stored unit/sequence, but are never used to cut, encode, or train the memory.  This
is an offline representation upper-bound diagnostic, not a retrieval policy.

Stages: prepare, train, evaluate, smoke, run.  All artifacts are kept under the
fixed compressed-denoise-01 output directory; per-condition artifacts are resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.25')

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent / 'pi05_memory'
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
RUN = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser() / 'run-01'
UNSEEN = ROOT / 'oracle-01'
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROWS  # corrected rows (image orientation fix)
OUT = ROOT / 'compressed-denoise-01'
CHECKPOINT = Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser()
SEED = 20261104
NOISE_SEED = 4101
STEPS = 4000
BATCH = 8
T0_EVAL = (.05, .1, .2, .3, .4, .6, 1.)
CONDS = ('A-raw', 'A-vm-none', 'A-vm-08', 'A-vm-08-hi', 'A-fx-08',
         'C-vm-none', 'C-vm-08', 'C-fx-08', 'C-rand-08', 'B-vm-08')
ACTION_CONDS = ('A-raw', 'A-vm-none', 'A-vm-08', 'A-vm-08-hi', 'A-fx-08', 'B-vm-08')
THETA_CONDS = ('C-vm-none', 'C-vm-08', 'C-fx-08', 'C-rand-08')
THETA_FILTER = r'head/(kernel|bias)'


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def _write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


def _pad_actions(actions):
    a = np.asarray(actions, np.float32)
    if a.ndim != 3 or a.shape[1:] not in ((10, 7), (10, 32)):
        raise ValueError(f'expected [N,10,7|32], got {a.shape}')
    if a.shape[-1] == 32:
        return a.copy()
    out = np.zeros((len(a), 10, 32), np.float32)
    out[:, :, :7] = a
    return out


def _stitch(actions, tasks, episodes, frames, row_ids):
    """Concatenate stride-10 windows into episode traces and index every row."""
    actions = np.asarray(actions)
    tasks, episodes, frames, row_ids = map(lambda x: np.asarray(x, np.int64),
                                           (tasks, episodes, frames, row_ids))
    traces, row_at = {}, {}
    keys = np.stack([tasks[row_ids], episodes[row_ids]], axis=1)
    for key in np.unique(keys, axis=0):
        loc = np.flatnonzero((keys == key).all(1))
        loc = loc[np.argsort(frames[row_ids[loc]])]
        ids = row_ids[loc]
        f = frames[ids]
        if len(f) and not np.array_equal(f, f[0] + 10 * np.arange(len(f))):
            raise ValueError(f'gap/overlap in episode {tuple(key.tolist())}')
        trace = np.concatenate(np.asarray(actions[ids])[:, :, :7], axis=0).astype(np.float32)
        epkey = (int(key[0]), int(key[1]))
        traces[epkey] = trace
        for row, frame in zip(ids, f):
            row_at[int(row)] = (epkey, int(frame - f[0]))
    return traces, row_at


def _anchor_spans(trace):
    """Use the reviewed speed-minimum + gripper-switch cut (minimum 5 frames)."""
    sys.path.insert(0, str(HERE))
    import motion_core_eval as motion
    return motion.anchor_spans(np.asarray(trace, np.float32), min_length=5)


def _action_rows(rows, ids):
    return np.asarray(rows['actions'][ids], np.float32)[:, :, :7]


def _episode_keys(task, episode):
    return np.asarray(task, np.int64) * 100000 + np.asarray(episode, np.int64)


def _bootstrap_diff(a, b, episode, *, reps=5000, seed=NOISE_SEED):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ep = np.asarray(episode)
    uniq, inv = np.unique(ep, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    draws = sums[ix].sum(1) / counts[ix].sum(1)
    return {'mean': float(d.mean()), 'ci95': [float(np.quantile(draws, .025)),
                                               float(np.quantile(draws, .975))],
            'episodes': int(len(uniq)), 'rows': int(len(d)), 'reps': int(reps)}


def _bootstrap_all_and_hard(a,b,episode,hard):
    """Apply the same subset to paired errors and their episode IDs."""
    a=np.asarray(a);b=np.asarray(b);episode=np.asarray(episode);hard=np.asarray(hard,bool)
    if not(len(a)==len(b)==len(episode)==len(hard)):raise ValueError('paired bootstrap row mismatch')
    return {'all':_bootstrap_diff(a,b,episode),
            'hard':_bootstrap_diff(a[hard],b[hard],episode[hard]) if hard.any() else None}


def _procrustes(shape, target):
    """Best shared SO(3) rotation + one scale for translation and rotation axes."""
    from scipy.spatial.transform import Rotation
    x = np.asarray(target, np.float64)
    y = np.asarray(shape, np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 6:
        raise ValueError((x.shape, y.shape))
    cross = y[:, :3].T @ x[:, :3] + y[:, 3:6].T @ x[:, 3:6]
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    corr = np.eye(3)
    corr[-1, -1] = np.linalg.det(u @ vt)
    r = u @ corr @ vt
    yrot = np.concatenate([y[:, :3] @ r, y[:, 3:6] @ r], axis=1)
    scale = max(float(np.sum(yrot * x) / max(np.sum(y * y), 1e-12)), 1e-6)
    theta = np.r_[Rotation.from_matrix(r).as_rotvec(), np.log(scale)].astype(np.float32)
    aligned = yrot * scale
    err = float(np.mean((aligned - x) ** 2))
    return theta, aligned, err


def _apply_theta(shape, theta, vs):
    from scipy.spatial.transform import Rotation
    z = np.asarray(shape, np.float64)
    th = np.asarray(theta, np.float64)
    r = Rotation.from_rotvec(th[:3]).as_matrix()
    size = float(np.exp(np.clip(th[3], -5., 5.)))
    phys = np.concatenate([z[:, :3] @ r, z[:, 3:6] @ r], axis=1) * size
    return vs.to_normalized(phys)


def _bank_from_memory(rows, mem_ids, *, mode, form, tau):
    """Build actual DCT-3 variable or fixed windows and optional medoid merge."""
    if str(HERE) not in sys.path:sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    import dct_tolerance_merge as merge
    from dct_skill_draft import SkillMemory

    tasks = np.asarray(rows['task'], np.int32)
    eps = np.asarray(rows['episode'], np.int32)
    frames = np.asarray(rows['frame'], np.int32)
    actions = np.load(RUN / 'rows' / 'actions.npy', mmap_mode='r', allow_pickle=False)
    traces, _ = _stitch(actions, tasks, eps, frames, mem_ids)
    values, lengths, unit_task, unit_ep, unit_start, source_rows, matrices = [], [], [], [], [], [], []
    unit = 0
    # Reconstruct per-episode memory demos with labels only for task exclusion/metadata.
    mem_by_ep = {}
    for rid in np.asarray(mem_ids, np.int64):
        mem_by_ep.setdefault((int(tasks[rid]), int(eps[rid])), []).append(int(rid))
    for key in sorted(mem_by_ep):
        trace = traces[key]
        spans = _anchor_spans(trace) if mode == 'variable' else [(s, min(s + 10, len(trace))) for s in range(0, len(trace), 10)]
        grip_k = 1 if mode == 'variable' else 3
        for s, e in spans:
            if e <= s:
                continue
            values.append(vs.encode(trace, s, e, form, grip_k))
            lengths.append(e - s); unit_task.append(key[0]); unit_ep.append(key[1]); unit_start.append(s)
            rowpos = min(len(mem_by_ep[key]) - 1, s // 10)
            source_rows.append(mem_by_ep[key][rowpos])
            matrices.append(merge.pose_matrix(trace[s:e, :6], form))
            unit += 1
    vals = np.asarray(values, np.float32)
    lens = np.asarray(lengths, np.int32)
    task_arr = np.asarray(unit_task, np.int32); ep_arr = np.asarray(unit_ep, np.int32)
    if tau is not None:
        err = merge.rebuild_errors(vals, np.asarray(matrices), 1 if mode == 'variable' else 3)
        selected, _ = merge.greedy_cover(err <= float(tau))
    else:
        selected = np.arange(len(vals), dtype=np.int64)
    selected = np.asarray(selected, np.int64)
    bank = SkillMemory(vals[selected], lens[selected], 1 if mode == 'variable' else 3, form)
    return {'memory': bank, 'task': task_arr[selected], 'episode': ep_arr[selected],
            'start': np.asarray(unit_start, np.int32)[selected],
            'source_row': np.asarray(source_rows, np.int64)[selected],
            'raw_unit_count': int(len(vals)), 'unit_count': int(len(selected)),
            'mode': mode, 'form': form, 'tau': tau}


def _bank_decode(bank):
    """Predecode each selected unit once; returns variable-length 7-D chunks."""
    mem = bank['memory']
    out = []
    for i in range(len(mem.lengths)):
        shape, grip = mem.shape(i)
        out.append(np.concatenate([shape, grip[:, None]], axis=1).astype(np.float32))
    return out


def _candidate_errors(candidates,target,form):
    """Vectorized scores, retaining scalar Procrustes's float64 convention."""
    import variable_shape_memory as vs
    candidates=np.asarray(candidates)
    if form!='rms':return np.mean((candidates-target[None])**2,axis=(1,2))
    y=np.asarray(candidates[:,:,:6],np.float64);x=vs.to_physical(target[:,:6])
    cross=y[:,:,:3].transpose(0,2,1)@x[:,:3]+y[:,:,3:6].transpose(0,2,1)@x[:,3:6]
    u,_,vt=np.linalg.svd(cross);corr=np.broadcast_to(np.eye(3),cross.shape).copy()
    corr[:,-1,-1]=np.linalg.det(u@vt);r=u@corr@vt
    yr=np.concatenate([y[:,:,:3]@r,y[:,:,3:6]@r],axis=2)
    scale=np.maximum(np.sum(yr*x[None],axis=(1,2))/np.maximum(np.sum(y*y,axis=(1,2)),1e-12),1e-6)
    aligned=vs.to_normalized(yr*scale[:,None,None])
    pred=np.concatenate([aligned,candidates[:,:,6:7]],axis=2)
    return np.mean((pred-target[None])**2,axis=(1,2))

def _plan_episode(bank, decoded, trace, query_task, *, form,
                  exclude_task=True, random_pick=False, seed=0):
    """Choose a whole-episode unit sequence; unit boundaries never reset at queries."""
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    import variable_shape_memory as vs
    lengths=np.asarray(bank['memory'].lengths);tasks=np.asarray(bank['task'])
    allowed=np.flatnonzero(tasks!=int(query_task)) if exclude_task else np.arange(len(tasks))
    if not len(allowed):raise ValueError('no task-excluded memory units')
    total=len(trace);out=np.zeros((total,7),np.float32);shape_out=np.zeros_like(out)
    unit_ids=np.full(total,-1,np.int32);src_theta=np.zeros((total,4),np.float32);target_theta=np.zeros_like(src_theta)
    groups=bank.get('_decoded_groups')
    if groups is None:
        groups=[(int(n),np.flatnonzero(lengths==n),np.stack([decoded[k] for k in np.flatnonzero(lengths==n)])) for n in np.unique(lengths)]
        bank['_decoded_groups']=groups
    rng=np.random.default_rng(seed);pos=0;slot=0;records=[]
    while pos<total:
        if random_pick:
            pick=int(rng.choice(allowed));n=min(int(lengths[pick]),total-pos)
            candidate=np.asarray(decoded[pick][:n],np.float32);target=np.asarray(trace[pos:pos+n,:7],np.float32)
            score=float(_candidate_errors(candidate[None],target,form)[0])
        else:
            best=(float('inf'),-1)
            for length,all_ids,all_cand in groups:
                valid=tasks[all_ids]!=int(query_task) if exclude_task else np.ones(len(all_ids),bool)
                if not valid.any():continue
                n=min(length,total-pos);ids=all_ids[valid];cand=all_cand[valid,:n]
                scores=_candidate_errors(cand,np.asarray(trace[pos:pos+n,:7],np.float32),form);j=int(np.argmin(scores))
                if float(scores[j])<best[0]:best=(float(scores[j]),int(ids[j]))
            score,pick=best
            if pick<0:raise RuntimeError('no unit selected')
            n=min(int(lengths[pick]),total-pos)
        candidate=np.asarray(decoded[pick][:n],np.float32);target=np.asarray(trace[pos:pos+n,:7],np.float32)
        src=np.asarray(bank.get('source_theta',np.zeros((len(lengths),4),np.float32))[pick],np.float32)
        if form=='rms':
            th,_,_=_procrustes(candidate[:,:6],vs.to_physical(target[:,:6]))
            action=np.concatenate([_apply_theta(candidate,src,vs),candidate[:,6:7]],axis=1)
        else:th=src;action=candidate
        end=pos+n;out[pos:end]=action;shape_out[pos:end]=candidate;unit_ids[pos:end]=slot
        src_theta[pos:end]=src;target_theta[pos:end]=th
        records.append({'unit':int(pick),'source_row':int(bank['source_row'][pick]),'source_task':int(tasks[pick]),
            'episode_start':int(pos),'unit_length':int(lengths[pick]),'length':int(n),
            'selection_error':float(score),'source_theta':src.tolist(),'target_theta':np.asarray(th).tolist()})
        pos=end;slot+=1
    return {'draft':out,'shape':shape_out,'unit_ids':unit_ids,'theta_src':src_theta,'theta_target':target_theta,'records':records}


def _select_sequence(bank, decoded, trace, offset, query_task, *, form,
                     exclude_task=True, random_pick=False, seed=0):
    """Slice a 10-frame query from the persistent episode plan, including phase."""
    from collections import OrderedDict
    cache=bank.setdefault('_episode_plans',OrderedDict())
    key=(id(trace),int(query_task),form,bool(exclude_task),bool(random_pick),int(seed))
    plan=cache.get(key)
    if plan is None:
        plan=_plan_episode(bank,decoded,trace,query_task,form=form,exclude_task=exclude_task,random_pick=random_pick,seed=seed)
        cache[key]=plan
        while len(cache)>8:cache.popitem(last=False)
    stop=int(offset)+10
    result={}
    for k in ('draft','shape','unit_ids','theta_src','theta_target'):
        value=plan[k][int(offset):stop].copy()
        if len(value)<10:value=np.pad(value,[(0,10-len(value))]+[(0,0)]*(value.ndim-1),mode='edge')
        result[k]=value
    # Per-window IDs are local slots, preserving the same selected unit/phase.
    used=np.unique(result['unit_ids']);result['unit_ids']=np.searchsorted(used,result['unit_ids']).astype(np.int32)
    records=[]
    for rec in plan['records']:
        first=rec['episode_start'];last=first+rec['length'];lo=max(int(offset),first);hi=min(stop,last)
        if hi<=lo:continue
        records.append({**rec,'offset':int(lo-offset),'phase':int(lo-first),'length':int(hi-lo)})
    result['records']=records
    return result


def _raw_oracle(actions, qids, qtask, mem_ids, tasks, *, exclude_task=True):
    src=np.asarray(actions[mem_ids,:,:7],np.float32).reshape(len(mem_ids),70)
    q=np.asarray(actions[qids,:,:7],np.float32).reshape(len(qids),70)
    cn=np.einsum('ij,ij->i',src,src)
    picked=np.empty(len(qids),np.int64);draft=np.empty((len(qids),10,7),np.float32);err=np.empty(len(qids),np.float32)
    for st in range(0,len(qids),64):
        sl=slice(st,min(st+64,len(qids)));x=q[sl];xn=np.einsum('ij,ij->i',x,x)
        d=np.maximum(xn[:,None]+cn[None,:]-2*x@src.T,0.)/70.
        if exclude_task:
            d[np.asarray(tasks[mem_ids])[None,:]==np.asarray(qtask[sl])[:,None]]=np.inf
        ix=d.argmin(1);picked[sl]=mem_ids[ix];draft[sl]=src[ix].reshape(-1,10,7);err[sl]=d[np.arange(len(ix)),ix]
    return picked,draft,err


def _build_condition_banks(rows, mem_ids):
    specs={
      'A-vm-none':('variable','as_is',None), 'A-vm-08':('variable','as_is',.08),
      'A-vm-08-hi':('variable','as_is',.08), 'A-fx-08':('fixed10','as_is',.08),
      'C-vm-none':('variable','rms',None), 'C-vm-08':('variable','rms',.08),
      'C-fx-08':('fixed10','rms',.08), 'C-rand-08':('variable','rms',.08)}
    banks={}
    for name,(mode,form,tau) in specs.items():
        key=f'{mode}_{form}_'+('none' if tau is None else 'tau08')
        if key not in banks:
            bank=_bank_from_memory(rows,mem_ids,mode=mode,form=form,tau=tau)
            # Store source orientation and magnitude for each unit as the theta start.
            import variable_shape_memory as vs
            source=[]
            all_actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
            task=np.asarray(rows['task']);ep=np.asarray(rows['episode']);frame=np.asarray(rows['frame'])
            mtraces,_=_stitch(all_actions,task,ep,frame,mem_ids)
            for i in range(bank['unit_count']):
                key_ep=(int(bank['task'][i]),int(bank['episode'][i]));trace=mtraces[key_ep]
                s=int(bank['start'][i]);n=int(bank['memory'].lengths[i]);chunk=trace[s:min(s+n,len(trace)),:6]
                if len(chunk)<n:chunk=np.pad(chunk,((0,n-len(chunk)),(0,0)),mode='edge')
                rot,scale=vs.pose_of(chunk,'rms')
                from scipy.spatial.transform import Rotation
                source.append(np.r_[Rotation.from_matrix(rot).as_rotvec(),np.log(max(float(scale),1e-6))])
            bank['source_theta']=np.asarray(source,np.float32)
            bank['decoded']=_bank_decode(bank)
            banks[key]=bank
        banks[name]=banks[key]
    return banks


def prepare(*, smoke=False, limit_train=16, limit_dev=8, limit_eval=8):
    """Create frozen split/draft caches. Smoke limits affect only the test cache."""
    if not OUT.exists():
        OUT.mkdir(parents=True)
    cache_root=OUT/('smoke_drafts' if smoke else 'drafts')
    marker=OUT/('prepare_smoke_status.json' if smoke else 'prepare_status.json')
    if marker.exists() and json.loads(marker.read_text()).get('complete'):
        return json.loads(marker.read_text())
    base=_module(PI05/'run.py','compressed_denoise_base')
    rows=base.load_rows(RUN)
    all_actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    task=np.asarray(rows['task'],np.int32);episode=np.asarray(rows['episode'],np.int32);frame=np.asarray(rows['frame'],np.int32)
    mem_ids=np.asarray(np.load(RUN/'memory_row_ids.npy',allow_pickle=False),np.int64)
    if len(mem_ids)!=6583:raise AssertionError(f'expected 6583 LIBERO-40 memory rows, got {len(mem_ids)}')
    if smoke:
        # Three source tasks keep the smoke's task-exclusion path realistic while
        # avoiding the quadratic merge over the full memory bank.
        mem_ids=mem_ids[np.isin(task[mem_ids],np.array([0,1,2],np.int32))]
    train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    if len(train)!=23304 or np.any(task[train]>=40):raise AssertionError('Expert training rows are not LIBERO-40 train only')
    un={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}
    dev=np.flatnonzero(np.asarray(un['is_dev'],bool)).astype(np.int64);ev=np.flatnonzero(~np.asarray(un['is_dev'],bool)).astype(np.int64)
    raw=np.load(UNSEEN/'drafts/raw_none_eval.npz',allow_pickle=False)
    if len(dev)!=113 or len(ev)!=619 or len(np.unique(un['task'][ev]))!=10:raise AssertionError('fixed unseen split changed')
    expected_eval_ids=len(all_actions)+ev
    if not np.array_equal(np.asarray(raw['query_ids'],np.int64),expected_eval_ids):
        raise AssertionError('raw draft rows do not align to unseen Eval row IDs')
    hard_cut=float(json.loads((UNSEEN/'hard_threshold.json').read_text())['threshold'])
    split_ids={'train':train,'dev':dev,'eval':ev}
    if smoke:
        split_ids={'train':train[:limit_train], 'dev':dev[:limit_dev], 'eval':ev[:limit_eval]}
    # Build the task-excluded candidate banks from the 400 memory demos only.
    banks=_build_condition_banks(rows,mem_ids)
    manifest={'status':'building','smoke':bool(smoke),'train_rows':len(split_ids['train']),
      'dev_rows':len(split_ids['dev']),'eval_rows':len(split_ids['eval']),'memory_rows':len(mem_ids),
      'hard_threshold':hard_cut,'condition_names':list(CONDS),'unseen_eval_used_for_training':False}
    _write_json(marker,manifest)
    for split,ids in split_ids.items():
        is_unseen=split!='train'
        source_actions=np.asarray(un['actions'] if is_unseen else all_actions,np.float32)
        source_task=np.asarray(un['task'] if is_unseen else task,np.int32)
        source_ep=np.asarray(un['episode'] if is_unseen else episode,np.int32)
        source_frame=np.asarray(un['frame'] if is_unseen else frame,np.int32)
        source_valid=np.arange(len(source_actions),dtype=np.int64)
        traces,row_at=_stitch(source_actions,source_task,source_ep,source_frame,source_valid)
        qtask=source_task[ids];qep=source_ep[ids]
        # Train IDs are global run row IDs; unseen IDs are their separate row indices.
        raw_pick,raw_draft,raw_pre=_raw_oracle(source_actions,ids,qtask,mem_ids,task) if not is_unseen else _raw_oracle(
            np.concatenate([np.asarray(all_actions),source_actions],axis=0),
            len(all_actions)+ids,qtask,mem_ids,task,exclude_task=False)
        payloads={'A-raw':{'draft':raw_draft,'selected_rows':raw_pick,'pre_mse':raw_pre}}
        # Train must exclude the query task. Unseen tasks have no matching LIBERO-40 ID.
        for name in ('A-vm-none','A-vm-08','A-vm-08-hi','A-fx-08','C-vm-none','C-vm-08','C-fx-08','C-rand-08'):
            cached=cache_root/f'{name.replace("-","_")}_{split}.npz'
            if cached.exists():
                with np.load(cached,allow_pickle=False) as prior:
                    if not np.array_equal(prior['row_ids'],ids):raise ValueError('cached row alignment changed')
                    payloads[name]={k:prior[k].copy() for k in prior.files if k not in ('row_ids','task','episode','hard')}
                continue
            if name=='A-vm-08-hi':
                payloads[name]=dict(payloads['A-vm-08']);continue
            bank=banks[name];decoded=bank['decoded'];form=bank['memory'].form
            pieces=[]
            seedbase=SEED+{'train':1,'dev':2,'eval':3}[split]
            def select_row(item):
                j,rid=item
                epkey,offset=row_at[int(rid)]
                return _select_sequence(bank,decoded,traces[epkey],offset,int(qtask[j]),form=form,
                    exclude_task=not is_unseen,random_pick=(name=='C-rand-08'),seed=seedbase+int(epkey[0])*100000+int(epkey[1]))
            # Prime immutable length groups before starting parallel row work.
            pieces.append(select_row((0,ids[0])))
            if form=='rms':
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing as mp
                import compressed_prepare_cpu as cpu
                cpu_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONPATH')}
                for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
                os.environ['CUDA_VISIBLE_DEVICES']=''
                os.environ['PYTHONPATH']=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH','')
                try:
                    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=cpu.initialize,
                        initargs=(bank,decoded,traces,row_at,qtask,form,not is_unseen,name=='C-rand-08',seedbase)) as pool:
                        for j,result in enumerate(pool.map(cpu.select_row,enumerate(ids[1:],start=1),chunksize=32),start=1):
                            pieces.append(result)
                            if j%250==0:_write_json(OUT/'prepare_progress.json',{'split':split,'condition':name,'rows':j,'total':len(ids)})
                finally:
                    for k,v in cpu_env.items():
                        if v is None:os.environ.pop(k,None)
                        else:os.environ[k]=v
            else:
                for j,rid in enumerate(ids[1:],start=1):
                    pieces.append(select_row((j,rid)))
                    if j%250==0:_write_json(OUT/'prepare_progress.json',{'split':split,'condition':name,'rows':j,'total':len(ids)})
            draft=np.stack([x['draft'] for x in pieces]).astype(np.float32)
            shape=np.stack([x['shape'] for x in pieces]).astype(np.float32)
            units=np.stack([x['unit_ids'] for x in pieces]).astype(np.int32)
            ts=np.stack([x['theta_src'] for x in pieces]).astype(np.float32)
            tt=np.stack([x['theta_target'] for x in pieces]).astype(np.float32)
            # Candidate source tasks must never equal a LIBERO-40 training query task.
            if not is_unseen:
                for p in pieces:
                    if any(int(z['source_task'])==int(p['records'][0]['source_task']) for z in []):pass
            payloads[name]={'draft':draft,'shape':shape,'unit_ids':units,'theta_src':ts,'theta_target':tt,
            'pre_mse':np.mean((draft-np.asarray(source_actions[ids],np.float32)[:,:,:7])**2,axis=(1,2)),
                            'records':np.asarray([json.dumps(x['records'],separators=(',',':')) for x in pieces])}
        # B's cache is replaced by C checkpoint predictions before B training;
        # this placeholder is the selected unit's stored-source-pose draft.
        payloads['B-vm-08']=dict(payloads['C-vm-08'])
        if split=='eval':
            rowpos=np.searchsorted(np.asarray(raw['query_ids'],np.int64),len(all_actions)+ids)
            if np.any(rowpos>=len(raw['query_ids'])) or not np.array_equal(raw['query_ids'][rowpos],len(all_actions)+ids):
                raise AssertionError('failed to join hard labels by global row ID')
            hard=np.asarray(raw['pre_mse'][rowpos])>=hard_cut
        else:hard=np.zeros(len(ids),bool)
        for name,p in payloads.items():
            path=cache_root/f'{name.replace("-","_")}_{split}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            data={'row_ids':ids,'task':qtask,'episode':qep,'hard':hard,'draft':_pad_actions(p['draft'])}
            for key in ('selected_rows','pre_mse','shape','unit_ids','theta_src','theta_target','records'):
                if key in p:data[key]=p[key]
            if path.exists():
                old=np.load(path,allow_pickle=False)
                if not np.array_equal(old['row_ids'],ids):raise FileExistsError(f'cache id mismatch: {path}')
            else:_write_npz(path,**data)
    manifest.update({'status':'complete','complete':True,'memory_units':{k:int(v['unit_count']) for k,v in banks.items() if '-' in k},
                     'train_rows':int(len(split_ids['train'])),'dev_rows':int(len(split_ids['dev'])),'eval_rows':int(len(split_ids['eval'])),
                     'hard_rows_eval':int(np.sum(np.load(OUT/'drafts/A_raw_eval.npz',allow_pickle=False)['hard'])) if not smoke else None})
    _write_json(marker,manifest)
    return manifest


def _make_action_step(filt,tx,t0_range):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    lo,hi=t0_range
    @nnx.jit
    def step(model,opt,obs,actions,draft,rng):
        r0,r1,r2=jax.random.split(rng,3)
        t0=jax.random.uniform(r0,(len(actions),),minval=lo,maxval=hi)
        u=jax.random.uniform(r1,(len(actions),),minval=.01,maxval=1.)
        eps=jax.random.normal(r2,actions.shape)
        source=(1-t0[:,None,None])*draft+t0[:,None,None]*eps
        x=(1-u[:,None,None])*actions+u[:,None,None]*source
        target=(source-actions)/t0[:,None,None]
        def loss_fn(current):
            v=current.velocity(obs,x,t0*u)
            return jnp.mean(jnp.square(v[:,:,:7]-target[:,:,:7]))
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(model)
        state=nnx.state(model,filt);updates,opt=tx.update(grads,opt,state)
        nnx.update(model,optax.apply_updates(state,updates))
        return loss,opt
    return step


def _rotation_jax(v, jnp):
    theta=jnp.linalg.norm(v,axis=-1,keepdims=True)
    x,y,z=jnp.moveaxis(v,-1,0)
    zero=jnp.zeros_like(x)
    k=jnp.stack([jnp.stack([zero,-z,y],-1),jnp.stack([z,zero,-x],-1),jnp.stack([-y,x,zero],-1)],-2)
    eye=jnp.broadcast_to(jnp.eye(3,dtype=v.dtype),k.shape)
    a=jnp.where(theta>1e-5,jnp.sin(theta)/jnp.maximum(theta,1e-5),1.)[...,None]
    b=jnp.where(theta>1e-5,(1-jnp.cos(theta))/jnp.maximum(theta**2,1e-10),.5)[...,None]
    return eye+a*k+b*(k@k)


def _theta_model_type(nnx):
  class ThetaActionModel(nnx.Module):
    """Frozen Pi0 suffix path plus a zero-init per-token theta projection."""
    def __init__(self, base_model, head):
        self.base=base_model;self.head=head
        import jax.numpy as jnp
        self.theta_mean=nnx.Variable(jnp.zeros(4));self.theta_std=nnx.Variable(jnp.ones(4))

    def theta_prefix(self,obs):
        import jax.numpy as jnp
        from openpi.models.pi0 import make_attn_mask
        prefix,mask,ar=self.base.embed_prefix(obs)
        _,cache=self.base.PaliGemma.llm([prefix,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
        return mask,cache

    def theta_velocity(self,obs,shape,unit_ids,theta_t,time,*,prefix_context=None):
        import jax.numpy as jnp
        from openpi.models.pi0 import make_attn_mask
        base=self.base
        # shape is canonical physical motion [B,10,7]; apply one shared 4-D
        # Procrustes pose to every token belonging to each selected unit.
        physical_theta=theta_t*self.theta_std.value+self.theta_mean.value
        r=_rotation_jax(physical_theta[:,:,:3],jnp)
        scale=jnp.exp(jnp.clip(physical_theta[:,:,3:4],-5.,5.))
        p=jnp.einsum('bti,btij->btj',shape[:,:,:3],r)*scale
        rot=jnp.einsum('bti,btij->btj',shape[:,:,3:6],r)*scale
        from variable_shape_memory import SPAN,LOW
        sp=jnp.asarray(SPAN,jnp.float32);lo=jnp.asarray(LOW,jnp.float32)
        motion=jnp.concatenate([p,rot],-1)
        motion=(motion-lo)/(sp+1e-6)*2-1
        x=jnp.zeros((len(shape),10,32),jnp.float32).at[:,:,:6].set(motion).at[:,:,6].set(shape[:,:,6])
        if prefix_context is None:mask,cache=self.theta_prefix(obs)
        else:mask,cache=prefix_context
        suffix,smask,sar,cond=base.embed_suffix(obs,x,time)
        attn=jnp.concatenate([jnp.broadcast_to(mask[:,None,:],(len(x),suffix.shape[1],mask.shape[1])),
            make_attn_mask(smask,sar)],axis=-1)
        pos=jnp.sum(mask,axis=1)[:,None]+jnp.cumsum(smask,axis=1)-1
        (_,hidden),_=base.PaliGemma.llm([None,suffix],mask=attn,positions=pos,kv_cache=cache,adarms_cond=[None,cond])
        return self.head(hidden[:,-10:])
  return ThetaActionModel


def _attach_theta_head(model):
    import flax.nnx as nnx
    width=int(model.action_out_proj.kernel.value.shape[0])
    head=nnx.Linear(width,4,kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),rngs=nnx.Rngs(731))
    return _theta_model_type(nnx)(model,head)


def _theta_group_mean(values,unit_ids):
    import jax.numpy as jnp
    slots=jnp.arange(10)[None,:,None]
    eq=(unit_ids[:,:,None]==unit_ids[:,None,:]) & (unit_ids[:,:,None]>=0)
    # For each token, average values of tokens carrying its unit id.
    count=jnp.maximum(eq.sum(-1,keepdims=True),1)
    return jnp.einsum('bij,bjd->bid',eq.astype(values.dtype),values)/count


def _theta_unit_noise(values,unit_ids):
    import jax.numpy as jnp
    equal=(unit_ids[:,:,None]==unit_ids[:,None,:])
    first=jnp.argmax(equal,axis=-1)
    return jnp.take_along_axis(values,first[...,None],axis=1)

def _make_theta_step(filt,tx,t0_range):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    lo,hi=t0_range
    @nnx.jit
    def step(wrapper,opt,obs,shape,unit_ids,theta_src,theta_target,rng):
        r0,r1,r2=jax.random.split(rng,3)
        t0=jnp.maximum(jax.random.uniform(r0,(len(shape),),minval=lo,maxval=hi),1e-6)
        u=jax.random.uniform(r1,(len(shape),),minval=.01,maxval=1.)
        eps=_theta_unit_noise(jax.random.normal(r2,theta_src.shape),unit_ids)
        source=(1-t0[:,None,None])*theta_src+t0[:,None,None]*eps
        theta_t=(1-u[:,None,None])*theta_target+u[:,None,None]*source
        theta_t=_theta_group_mean(theta_t,unit_ids)
        target=_theta_group_mean((source-theta_target)/t0[:,None,None],unit_ids)
        def loss_fn(current):
            v=current.theta_velocity(obs,shape,unit_ids,theta_t,t0*u)
            v=_theta_group_mean(v,unit_ids)
            mask=(unit_ids>=0).astype(v.dtype)[...,None]
            return jnp.sum(jnp.square(v-target)*mask)/jnp.maximum(jnp.sum(mask)*4,1.)
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(wrapper)
        state=nnx.state(wrapper,filt);updates,opt=tx.update(grads,opt,state)
        nnx.update(wrapper,optax.apply_updates(state,updates))
        return loss,opt
    return step


def _t0_range(condition):
    if condition in THETA_CONDS:return (0.,1.)
    return (.2,.6) if condition=='A-vm-08-hi' else (.05,.4)


def _theta_stats(theta_src,theta_target,unit_ids):
    valid=np.asarray(unit_ids)>=0
    target=np.asarray(theta_target,np.float64)[valid]
    if target.size==0:raise ValueError('no valid theta tokens for train-only normalization')
    mean=target.mean(0).astype(np.float32)
    std=target.std(0).astype(np.float32)
    std=np.where(std<1e-5,1.,std).astype(np.float32)
    return mean,std


def _theta_sample_batch(wrapper,obs,cache,t0,row_global_ids,*,theta_mean=None,theta_std=None,num_steps=10,velocity_fn=None,prefix_fn=None):
    """Integrate theta from a stored source transform, then decode its shape."""
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn=velocity_fn if velocity_fn is not None else module_jit(wrapper.theta_velocity)
    shape=jnp.asarray(cache['shape'][:,:,:7],jnp.float32)
    unit=jnp.asarray(cache['unit_ids'],jnp.int32)
    src_np=np.asarray(cache['theta_src'],np.float32)
    mu=np.zeros(4,np.float32) if theta_mean is None else np.asarray(theta_mean,np.float32)
    sd=np.ones(4,np.float32) if theta_std is None else np.asarray(theta_std,np.float32)
    src=jnp.asarray((src_np-mu)/sd,jnp.float32)
    eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,4)))
                  for r in row_global_ids]).astype(np.float32)
    eps=_theta_unit_noise(jnp.asarray(eps),unit)
    starts=jnp.full((len(shape),),float(t0),jnp.float32)
    x=_theta_group_mean((1-starts[:,None,None])*src+starts[:,None,None]*eps,unit)
    context=(prefix_fn or module_jit(wrapper.theta_prefix))(obs)
    for i in range(num_steps):
        time=starts*(1-i/num_steps)
        v=_theta_group_mean(fn(obs,shape,unit,x,time,prefix_context=context),unit)
        x=x-starts[:,None,None]/num_steps*v
        x=_theta_group_mean(x,unit)
    shape_np=np.asarray(shape);theta_np=np.asarray(x)*sd+mu;unit_np=np.asarray(unit)
    if str(HERE) not in sys.path:sys.path.insert(0, str(HERE))
    import variable_shape_memory as vs
    from scipy.spatial.transform import Rotation
    out=np.empty((len(shape_np),10,7),np.float32)
    for b in range(len(out)):
        for t in range(10):
            if unit_np[b,t]<0:
                out[b,t]=0.;continue
            th=theta_np[b,t]
            r=Rotation.from_rotvec(th[:3]).as_matrix();size=float(np.exp(np.clip(th[3],-5,5)))
            motion=np.r_[shape_np[b,t,:3]@r,shape_np[b,t,3:6]@r]*size
            out[b,t,:6]=vs.to_normalized(motion)
            out[b,t,6]=shape_np[b,t,6]
    return out


def _theta_sample(wrapper,obs,cache,t0,row_global_ids,*,theta_mean=None,theta_std=None,num_steps=10,velocity_fn=None,prefix_fn=None):
    """Bound VLM activation memory while preserving per-row explicit noise."""
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    mu=np.zeros(4,np.float32) if theta_mean is None else np.asarray(theta_mean,np.float32)
    sd=np.ones(4,np.float32) if theta_std is None else np.asarray(theta_std,np.float32)
    wrapper.theta_mean.value=jnp.asarray(mu);wrapper.theta_std.value=jnp.asarray(sd)
    fn=velocity_fn if velocity_fn is not None else module_jit(wrapper.theta_velocity)
    prefix_fn=prefix_fn if prefix_fn is not None else module_jit(wrapper.theta_prefix)
    predictions=[]
    for start in range(0,len(row_global_ids),BATCH):
        sl=slice(start,min(start+BATCH,len(row_global_ids)))
        sub={k:np.asarray(cache[k][sl]) for k in ('shape','unit_ids','theta_src')}
        predictions.append(_theta_sample_batch(wrapper,jax.tree.map(lambda x:x[sl],obs),sub,t0,
            np.asarray(row_global_ids)[sl],theta_mean=mu,theta_std=sd,num_steps=num_steps,velocity_fn=fn,prefix_fn=prefix_fn))
    return np.concatenate(predictions)

def _decode_theta_tokens(shape,theta,vs):
    """Restore each token with its own selected unit's pose (not the first unit)."""
    shape=np.asarray(shape);theta=np.asarray(theta)
    return np.stack([np.r_[_apply_theta(shape[t:t+1],theta[t],vs)[0],shape[t,6]]
                     for t in range(len(shape))]).astype(np.float32)


def _predict_action(model,obs,drafts,noise,t0,*,batch=BATCH):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn=module_jit(model.sample_bridge,static_argnames=('num_steps',))
    out=[]
    for st in range(0,len(drafts),batch):
        sl=slice(st,min(st+batch,len(drafts)))
        y=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(drafts[sl]),
             jnp.full((sl.stop-sl.start,),float(t0)),num_steps=10,noise=jnp.asarray(noise[sl]))
        out.append(np.asarray(y))
    return np.concatenate(out)


def train_condition(condition, *, steps=STEPS, threads=8, smoke=False):
    """Train one expert; complete artifacts are skipped for safe resumption."""
    if condition not in CONDS:raise ValueError(condition)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    import torch;torch.set_num_threads(threads)
    base=_module(PI05/'run.py','compressed_train_base')
    rows=base.load_rows(RUN)
    train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    cache_root=(OUT/'smoke_predicted_b_drafts' if smoke and condition=='B-vm-08' else
                OUT/'smoke_drafts' if smoke else
                OUT/'predicted_b_drafts' if condition=='B-vm-08' else OUT/'drafts')
    cache=np.load(cache_root/f'{condition.replace("-","_")}_train.npz',allow_pickle=False)
    if not np.array_equal(cache['row_ids'],train if not smoke else cache['row_ids']):raise AssertionError('train row IDs changed')
    train_ids=np.asarray(cache['row_ids'],np.int64)
    if not smoke and (len(train_ids)!=23304 or np.any(np.asarray(rows['task'])[train_ids]>=40)):
        raise AssertionError('train rows leaked outside LIBERO-40 or validation')
    if condition=='B-vm-08' and not (OUT/'experts/C-vm-08/trained.json').exists() and not smoke:
        raise FileNotFoundError('B-vm-08 requires trained C-vm-08 predictions first')
    expdir=OUT/'experts'/condition
    done=expdir/'trained.json'
    if done.exists():return json.loads(done.read_text())
    expdir.mkdir(parents=True,exist_ok=True);(expdir/'bridge').mkdir(exist_ok=True)
    model,_=base.load_policy(CHECKPOINT)
    is_theta=condition in THETA_CONDS
    if is_theta:
        wrapper=_attach_theta_head(model)
        filt=nnx.All(nnx.Param,PathRegex(THETA_FILTER))
        train_model=wrapper
    else:
        wrapper=None;filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        train_model=model
    selected=nnx.state(train_model,filt)
    if not selected.flat_state():raise AssertionError(f'empty trainable state for {condition}')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5));opt=tx.init(selected)
    if is_theta:step=_make_theta_step(filt,tx,_t0_range(condition))
    else:step=_make_action_step(filt,tx,_t0_range(condition))
    target=np.asarray(rows['actions'][train_ids],np.float32)
    draft=jnp.asarray(cache['draft'],jnp.float32)
    if is_theta:
        shape=jnp.asarray(cache['shape'][:,:,:7],jnp.float32)
        unit_ids=jnp.asarray(cache['unit_ids'],jnp.int32)
        theta_mean,theta_std=_theta_stats(cache['theta_src'],cache['theta_target'],cache['unit_ids'])
        wrapper.theta_mean.value=jnp.asarray(theta_mean);wrapper.theta_std.value=jnp.asarray(theta_std)
        theta_src=jnp.asarray((cache['theta_src']-theta_mean)/theta_std,jnp.float32)
        theta_target=jnp.asarray((cache['theta_target']-theta_mean)/theta_std,jnp.float32)
    batches= _module(PI05/'fair_expert'/'run.py','compressed_fair').make_batch_indices(train_ids,steps=steps,batch=BATCH,seed=SEED)
    loc={int(r):i for i,r in enumerate(train_ids)}
    key=jax.random.key(SEED);logs=[];started=time.monotonic();best=float('inf');best_meta=None;start_step=0
    nstep=min(2,steps) if smoke else steps
    progress_path=expdir/'progress.json'
    if not smoke and progress_path.exists():
        progress=json.loads(progress_path.read_text());latest=progress.get('latest',{})
        step0=int(latest.get('steps',0));resume_ckpt=expdir/'resume'/f'{step0:06d}'
        if step0 and resume_ckpt.exists():
            resume=ocp.PyTreeCheckpointer().restore(resume_ckpt)
            state=nnx.state(train_model,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,resume['model']));nnx.update(train_model,state)
            opt=jax.tree.map(jnp.asarray,resume['opt']);start_step=step0
            logs=list(progress.get('logs',[]));best_meta=progress.get('best')
            best=float(best_meta['dev_active7_mse']) if best_meta else float('inf')
            for _ in range(start_step):key,_sub=jax.random.split(key)
    devcache=np.load(cache_root/f'{condition.replace("-","_")}_dev.npz',allow_pickle=False)
    devids=np.asarray(devcache['row_ids'],np.int64)
    unseen={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}
    if len(devids):
        dev_obs=base.obs_batch(unseen,devids)
        dev_target=np.asarray(unseen['actions'][devids],np.float32)
    else:dev_obs=None;dev_target=None
    dev_noise=np.stack([np.asarray(jax.random.normal(
        jax.random.fold_in(jax.random.key(NOISE_SEED),int(len(rows['actions'])+rid)),(10,32)))
        for rid in devids]).astype(np.float32) if len(devids) else None
    for i in range(start_step,nstep):
        ids=batches[i];ix=np.asarray([loc[int(r)] for r in ids],np.int64)
        key,sub=jax.random.split(key)
        if is_theta:
            loss,opt=step(wrapper,opt,base.obs_batch(rows,ids),shape[ix],unit_ids[ix],theta_src[ix],theta_target[ix],sub)
        else:
            loss,opt=step(model,opt,base.obs_batch(rows,ids),jnp.asarray(target[ix]),draft[ix],sub)
        value=float(loss)
        if not np.isfinite(value):raise FloatingPointError(f'{condition} nonfinite at step {i+1}')
        logs.append({'step':i+1,'loss':value,'elapsed_s':time.monotonic()-started})
        if not smoke and ((i+1)%500==0 or i+1==steps):
            ckpt=expdir/'bridge'/f'{i+1:06d}';ocp.PyTreeCheckpointer().save(ckpt,nnx.state(train_model,filt).to_pure_dict())
            resume_dir=expdir/'resume'/f'{i+1:06d}';resume_dir.parent.mkdir(parents=True,exist_ok=True)
            ocp.PyTreeCheckpointer().save(resume_dir,{'model':nnx.state(train_model,filt).to_pure_dict(),'opt':opt})
            dev_curve={}
            if is_theta:
                for val_t in T0_EVAL:
                    pred=_theta_sample(wrapper,dev_obs,devcache,val_t,len(rows['actions'])+devids,
                                       theta_mean=theta_mean,theta_std=theta_std)
                    dev_curve[str(val_t)]=float(np.mean((pred[:,:,:7]-dev_target[:,:,:7])**2))
            else:
                for val_t in T0_EVAL:
                    pred=_predict_action(model,dev_obs,devcache['draft'],dev_noise,val_t)
                    dev_curve[str(val_t)]=float(np.mean((pred[:,:,:7]-dev_target[:,:,:7])**2))
            selected_t=min(T0_EVAL,key=lambda t:dev_curve[str(t)])
            selected_mse=dev_curve[str(selected_t)]
            if selected_mse<best:
                best=selected_mse
                best_meta={'path':str(ckpt),'steps':i+1,'t0':float(selected_t),
                           'dev_active7_mse':best,'selection':'fixed Dev rows only; Eval rows excluded'}
                _write_json(expdir/'best_bridge.json',best_meta)
            meta={'path':str(ckpt),'steps':i+1,'dev_mse_by_t0':dev_curve,
                  'dev_best_t0':float(selected_t),'dev_best_mse':selected_mse}
            logs[-1].update({'dev_mse_by_t0':dev_curve,'best_checkpoint':best_meta})
            _write_json(expdir/'progress.json',{'status':'running','condition':condition,'logs':logs,'latest':meta,'best':best_meta})
        if smoke and i+1==nstep:
            _write_json(expdir/'smoke.json',{'condition':condition,'steps':nstep,'loss':value,'trainable_leaves':len(selected.flat_state())})
    if smoke:
        if is_theta:
            smoke_pred=_theta_sample(wrapper,dev_obs,devcache,.2,len(rows['actions'])+devids,
                                     theta_mean=theta_mean,theta_std=theta_std)
        else:
            smoke_pred=_predict_action(model,dev_obs,devcache['draft'],dev_noise,.2)
        smoke_mse=float(np.mean((smoke_pred[:,:,:7]-dev_target[:,:,:7])**2))
        smoke_ckpt=expdir/'smoke_bridge_04'
        if not smoke_ckpt.exists():ocp.PyTreeCheckpointer().save(smoke_ckpt,nnx.state(train_model,filt).to_pure_dict())
        result={'condition':condition,'steps':nstep,'loss':logs[-1]['loss'],
                'trainable_leaves':len(selected.flat_state()),'dev_generation_rows':len(devids),
                'dev_generation_mse_t0_0.2':smoke_mse,'checkpoint':str(smoke_ckpt)}
        if is_theta:result.update({'theta_mean':theta_mean.tolist(),'theta_std':theta_std.tolist()})
        return result
    # Finish with latest checkpoint as a provisional default; evaluate() chooses
    # checkpoint/t0 strictly on the fixed 113-row Dev split.
    meta={'condition':condition,'steps':steps,'seed':SEED,'batch_size':BATCH,
          'train_rows':len(train_ids),'dev_rows':len(devids),'training_seconds':time.monotonic()-started,
          'trainable_filter':'new theta head only' if is_theta else 'llm.*_1|action_in_proj|time_mlp|action_out_proj',
          'optimizer':'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5 clip=1','loss':'active7 bridge velocity' if not is_theta else 'theta-space bridge velocity',
          'checkpoints':list(range(500,steps+1,500)),'best':best_meta}
    if is_theta:meta.update({'theta_mean':theta_mean.tolist(),'theta_std':theta_std.tolist(),
                             'theta_normalization_source':'training rows only'})
    _write_json(done,meta);return meta


def smoke():
    """Small real-data and GPU update smoke across all ten named conditions."""
    prep=prepare(smoke=True)
    results=[]
    cache_root=OUT/'smoke_drafts'
    # All representation paths/caches are checked; one real action Expert update
    # and one theta-adapter update exercise both differentiable model branches.
    for name in CONDS:
        p=cache_root/f'{name.replace("-","_")}_train.npz'
        z=np.load(p,allow_pickle=False)
        if len(z['row_ids'])!=16 or z['draft'].shape!=(16,10,32):raise AssertionError(f'{name} cache shape')
        if name in ('A-raw','A-vm-none','A-vm-08-hi','C-vm-none','C-vm-08','C-fx-08','C-rand-08'):
            results.append({'condition':name,'cache_rows':len(z['row_ids']),'draft_shape':list(z['draft'].shape),
                            'pre_mse':float(np.mean(z['pre_mse']))})
    # GPU model branch smokes: exact condition-specific data, two optimizer steps.
    for name in ('A-vm-08','C-vm-08'):
        old=OUT/'experts'/name/'smoke_attempt04.json'
        if old.exists():
            results.append(json.loads(old.read_text()))
        else:
            result=train_condition(name,steps=2,smoke=True,threads=8)
            _write_json(old,result);results.append(result)
    materialized=materialize_b_drafts(smoke=True)
    results.append({'stage':'B draft materialization','rows_by_split':materialized['rows_by_split'],
                    'checkpoint':materialized['checkpoint']})
    results.append(train_condition('B-vm-08',steps=2,smoke=True,threads=8))
    _write_json(OUT/'smoke-04.json',{'status':'complete','conditions':results,'cache_condition_count':len(CONDS),
        'train_rows':prep['train_rows'],'dev_rows':prep['dev_rows'],'eval_rows':prep['eval_rows'],
        'eval_used_for_training':False})
    return results


def evaluate_condition(condition, *, threads=8):
    """Evaluate a trained checkpoint over the frozen Eval rows and t0 grid."""
    if condition not in CONDS:raise ValueError(condition)
    import torch;torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import module_jit,PathRegex
    base=_module(PI05/'run.py','compressed_eval_base');skill=_module(PI05/'skill_expert'/'run.py','compressed_eval_skill')
    trained=json.loads((OUT/'experts'/condition/'trained.json').read_text())
    rows=base.load_rows(RUN)
    unseen={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}
    cache_root=OUT/('predicted_b_drafts' if condition=='B-vm-08' else 'drafts')
    cache=np.load(cache_root/f'{condition.replace("-","_")}_eval.npz',allow_pickle=False)
    ids=np.asarray(cache['row_ids'],np.int64);target=np.asarray(unseen['actions'][ids],np.float32)
    model,_=base.load_policy(CHECKPOINT);is_theta=condition in THETA_CONDS
    if is_theta:
        wrapper=_attach_theta_head(model);train_model=wrapper;filt=nnx.All(nnx.Param,PathRegex(THETA_FILTER))
    else:train_model=model;filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    ckpt=Path(trained['best']['path'] if 'best' in trained else trained['checkpoints_path'])
    with ocp.PyTreeCheckpointer() as c:part=c.restore(ckpt)
    state=nnx.state(train_model,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,part));nnx.update(train_model,state)
    obs=base.obs_batch(unseen,ids);eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(i)),(10,32))) for i in len(rows['actions'])+ids])
    target7=target[:,:,:7]
    pred={}
    if is_theta:
        for t0 in T0_EVAL:
            pred[t0]=_theta_sample(wrapper,obs,cache,t0,len(rows['actions'])+ids,
                theta_mean=trained['theta_mean'],theta_std=trained['theta_std'])[:,:,:7]
    else:
        for t0 in T0_EVAL:
            pred[t0]=_predict_action(model,obs,cache['draft'],eps,t0)[:,:,:7]
    e={str(t):np.mean((pred[t]-target7)**2,axis=(1,2)) for t in T0_EVAL}
    key=_episode_keys(cache['task'],cache['episode'])
    save=OUT/'predictions'/condition;save.mkdir(parents=True,exist_ok=True)
    payload={'row_ids':ids,'task':cache['task'],'episode':cache['episode'],'target':target7,'hard':cache['hard'],
             'draft':cache['draft'][:,:,:7],**{f'pred_t{t}':pred[t] for t in T0_EVAL},
             **{f'err_t{t}':e[str(t)] for t in T0_EVAL},
             **{f'err_xyz_t{t}':np.mean((pred[t][:,:,:3]-target7[:,:,:3])**2,axis=(1,2)) for t in T0_EVAL},
             **{f'err_rotation_t{t}':np.mean((pred[t][:,:,3:6]-target7[:,:,3:6])**2,axis=(1,2)) for t in T0_EVAL},
             **{f'err_grip_t{t}':np.mean((pred[t][:,:,6]-target7[:,:,6])**2,axis=1) for t in T0_EVAL}}
    report={'condition':condition,'rows':len(ids),'hard_rows':int(np.sum(cache['hard'])),'selected_checkpoint':trained['best'],'training_steps':trained['steps'],'trainable_filter':trained['trainable_filter'],'t0':{},
        'selection':'fixed Eval; no tuning on Eval'}
    for t in T0_EVAL:
        d=pred[t]-target7
        draft_err=np.mean((cache['draft'][:,:,:7]-target7)**2,axis=(1,2))
        hm=np.asarray(cache['hard'],bool)
        paired=_bootstrap_all_and_hard(e[str(t)],draft_err,key,hm)
        report['t0'][str(t)]={'mse':float(e[str(t)].mean()),'hard_mse':float(e[str(t)][cache['hard']].mean()),
            'components':{'xyz':float(np.mean(d[:,:,:3]**2)),'rotation':float(np.mean(d[:,:,3:6]**2)),
                          'gripper':float(np.mean(d[:,:,6]**2))},
            'hard_components':{'xyz':float(np.mean(d[hm,:,:3]**2)),'rotation':float(np.mean(d[hm,:,3:6]**2)),
                               'gripper':float(np.mean(d[hm,:,6]**2))},
            'draft_mse':float(draft_err.mean()),'hard_draft_mse':float(draft_err[hm].mean()),
            'vs_draft':paired['all'],
            'vs_draft_hard':paired['hard']}
    if is_theta:
        sys.path.insert(0,str(HERE));import variable_shape_memory as vs
        src=[];oracle=[]
        for i in range(len(ids)):
            s=np.asarray(cache['shape'][i,:,:7],np.float32)
            src.append(_decode_theta_tokens(s,cache['theta_src'][i],vs))
            oracle.append(_decode_theta_tokens(s,cache['theta_target'][i],vs))
        src=np.asarray(src);oracle=np.asarray(oracle)
        payload['theta_src_reconstruction']=src;payload['theta_target_upper_bound']=oracle
        report['theta_diagnostics']={'theta_src_mse':float(np.mean((src-target7)**2)),
                                     'oracle_theta_mse':float(np.mean((oracle-target7)**2)),
                                     'theta_src_hard_mse':float(np.mean((src[np.asarray(cache['hard'],bool)]-target7[np.asarray(cache['hard'],bool)])**2)),
                                     'oracle_theta_hard_mse':float(np.mean((oracle[np.asarray(cache['hard'],bool)]-target7[np.asarray(cache['hard'],bool)])**2)),
                                     'note':'oracle theta is target-derived upper-bound diagnostic'}
    # Frozen legacy oracle-01 action Expert, evaluated on this condition's same
    # draft and its own Dev-selected start noise. This is a reference only.
    old_meta=json.loads((UNSEEN/'experts/raw_none/trained.json').read_text())
    skill.load_expert(model,Path(old_meta['best']['path']))
    old_t0=float(old_meta['best']['t0'])
    old_pred=_predict_action(model,obs,cache['draft'],eps,old_t0)[:,:,:7]
    old_err=np.mean((old_pred-target7)**2,axis=(1,2))
    payload['oracle01_pred_selected_t0']=old_pred
    report['oracle01_expert_same_draft']={'t0':old_t0,'mse':float(old_err.mean()),
        'hard_mse':float(old_err[cache['hard']].mean()),
        'vs_new_expert_at_same_t0':_bootstrap_diff(old_err,e[str(old_t0)],key)}
    official=np.load(UNSEEN/'official_normal_eval.npz',allow_pickle=False)
    loc={int(r):i for i,r in enumerate(official['row_ids'])}
    global_ids=len(rows['actions'])+ids
    if not np.all(np.isin(global_ids,official['row_ids'])):raise AssertionError('official normal Eval row IDs mismatch')
    oi=np.asarray([loc[int(r)] for r in global_ids],np.int64)
    normal_err=np.mean((official['prediction'][oi]-target7)**2,axis=(1,2))
    payload['official_pi05_prediction']=official['prediction'][oi]
    report['official_pi05']={'mse':float(normal_err.mean()),'hard_mse':float(normal_err[cache['hard']].mean())}
    _write_npz(save/'eval_rows.npz',**payload)
    _write_json(save/'result.json',report);return report


def materialize_b_drafts(*,threads=8,batch=8,smoke=False):
    """Use the trained C-vm-08 generator to make B training/query drafts."""
    import torch;torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex,module_jit
    if smoke:
        trained_path=OUT/'experts/C-vm-08/smoke_attempt04.json'
        if not trained_path.exists():raise FileNotFoundError('C-vm-08 smoke must run first')
        trained=json.loads(trained_path.read_text())
        best={'path':trained['checkpoint'],'t0':.2}
    else:
        trained_path=OUT/'experts/C-vm-08/trained.json'
        if not trained_path.exists():raise FileNotFoundError('C-vm-08 must be trained first')
        trained=json.loads(trained_path.read_text());best=trained.get('best')
        if not best:raise ValueError('C-vm-08 lacks a Dev-selected checkpoint')
    base=_module(PI05/'run.py','compressed_bdraft_base');model,_=base.load_policy(CHECKPOINT)
    wrapper=_attach_theta_head(model);filt=nnx.All(nnx.Param,PathRegex(THETA_FILTER))
    with ocp.PyTreeCheckpointer() as c:part=c.restore(Path(best['path']))
    state=nnx.state(wrapper,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,part));nnx.update(wrapper,state)
    # Frozen checkpoint and train-only theta normalization are constant across
    # all rows. Reuse compiled methods, not predictions or observation caches.
    wrapper.theta_mean.value=jnp.asarray(trained['theta_mean'],jnp.float32)
    wrapper.theta_std.value=jnp.asarray(trained['theta_std'],jnp.float32)
    velocity_fn=module_jit(wrapper.theta_velocity);prefix_fn=module_jit(wrapper.theta_prefix)
    rows=base.load_rows(RUN)
    unseen={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}
    srcroot=OUT/('smoke_drafts' if smoke else 'drafts')
    dstroot=OUT/('smoke_predicted_b_drafts' if smoke else 'predicted_b_drafts');dstroot.mkdir(parents=True,exist_ok=True)
    for split,data in [('train',rows),('dev',unseen),('eval',unseen)]:
        # NPZ indexing decompresses the whole field each time. Load once per
        # split, and never read theta_target for the model's draft generation.
        with np.load(srcroot/f'C_vm_08_{split}.npz',allow_pickle=False) as stored:
            ccache={k:stored[k] for k in ('row_ids','task','episode','hard','shape','unit_ids','theta_src','records')}
        ids=np.asarray(ccache['row_ids'],np.int64);chunks=[]
        global_ids=ids if split=='train' else len(rows['actions'])+ids
        for st in range(0,len(ids),batch):
            sl=slice(st,min(st+batch,len(ids)))
            obs=base.obs_batch(data,ids[sl])
            chunks.append(_theta_sample(wrapper,obs,{k:ccache[k][sl] for k in ('shape','unit_ids','theta_src')},
                best['t0'],global_ids[sl],theta_mean=trained['theta_mean'],theta_std=trained['theta_std'],
                velocity_fn=velocity_fn,prefix_fn=prefix_fn))
        actions=np.concatenate(chunks,axis=0)
        out={'row_ids':ids,'task':ccache['task'],'episode':ccache['episode'],'hard':ccache['hard'],
             'draft':_pad_actions(actions),'pre_mse':np.mean((actions[:,:,:7]-np.asarray(data['actions'][ids,:,:7]))**2,axis=(1,2)),
             'records':ccache['records']}
        path=dstroot/f'B_vm_08_{split}.npz'
        if path.exists():
            old=np.load(path,allow_pickle=False)
            if not np.array_equal(old['row_ids'],ids):raise FileExistsError(f'B draft cache mismatch: {path}')
        else:_write_npz(path,**out)
    info={'source':'C-vm-08 theta Expert','checkpoint':best,'dev_selected_t0':best['t0'],
          'eval_used_for_selection':False,'rows_by_split':{s:int(len(np.load(dstroot/f'B_vm_08_{s}.npz',allow_pickle=False)['row_ids'])) for s in ('train','dev','eval')}}
    _write_json(dstroot/'materialized.json',info)
    return info


def run_all(*,smoke_mode=False):
    prepare(smoke=smoke_mode)
    if smoke_mode:return smoke()
    # Full production is intentionally explicit and resumable; B follows C-vm-08.
    for c in CONDS:
        if c=='B-vm-08':materialize_b_drafts()
        train_condition(c)
        if not (OUT/'predictions'/c/'result.json').exists():evaluate_condition(c)
    normal=np.load(UNSEEN/'official_normal_eval.npz',allow_pickle=False)
    normal_mse=float(np.mean((normal['prediction']-normal['target'])**2))
    entries=[]
    for c in CONDS:
        result=json.loads((OUT/'predictions'/c/'result.json').read_text())
        best=json.loads((OUT/'experts'/c/'trained.json').read_text())['best']
        entries.append({'condition':c,'selected_checkpoint_step':best['steps'],'selected_t0':best['t0'],
            'dev_active7_mse':best['dev_active7_mse'],'eval_selected_t0_mse':result['t0'][str(best['t0'])]['mse'],
            'eval_hard_mse':result['t0'][str(best['t0'])]['hard_mse'],
            'draft_mse':result['t0'][str(best['t0'])]['draft_mse'],
            'oracle01_same_draft_mse':result['oracle01_expert_same_draft']['mse']})
    summary={'status':'complete','train_rows':23304,'dev_rows':113,'eval_rows':619,
        'hard_eval_rows':int(np.sum(np.load(OUT/'drafts/A_raw_eval.npz',allow_pickle=False)['hard'])),
        'official_pi05_eval_mse':normal_mse,'conditions':entries,
        'evaluation':'fixed Eval only after Dev checkpoint/t0 selection; oracle-best candidate selection is diagnostic upper bound; no search'}
    _write_json(OUT/'summary.json',summary)
    lines=['# Compressed motion denoising', '',
      '未学習の評価デモの正解は候補単位の選択と採点にだけ使用するオフライン上限診断であり、検索器や閉ループ性能の評価ではない。',
      f"固定分割: train {summary['train_rows']} / dev {summary['dev_rows']} / eval {summary['eval_rows']} (hard {summary['hard_eval_rows']}); 通常π0.5 Eval MSE={normal_mse:.6f}。", '',
      '| 条件 | 選択step | Dev選択t0 | Eval MSE | hard MSE | 下書きMSE | oracle-01 Expert同下書き |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for row in entries:
        lines.append(f"| {row['condition']} | {row['selected_checkpoint_step']} | {row['selected_t0']:.2f} | {row['eval_selected_t0_mse']:.6f} | {row['eval_hard_mse']:.6f} | {row['draft_mse']:.6f} | {row['oracle01_same_draft_mse']:.6f} |")
    lines += ['', '全t0と成分別・行別の結果は `predictions/<condition>/` を参照。標準誤差区間は行動誤差の episode bootstrap を記録する。',
      '制約: 正解を見て単位を選ぶ上限、評価デモが選択に入る、既知LIBERO-40メモリ、単一seed、オフライン評価。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','train','evaluate','run'])
    p.add_argument('--condition',choices=CONDS)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--output',type=Path,default=None,
                   help='output directory (default: compressed-denoise-01/production, read by later steps)')
    a=p.parse_args()
    OUT=a.output or OUT/'production'
    if a.stage=='prepare':print(json.dumps(prepare(smoke=a.smoke),ensure_ascii=False))
    elif a.stage=='smoke':print(json.dumps(smoke(),ensure_ascii=False))
    elif a.stage=='train':
        if not a.condition:raise SystemExit('--condition is required')
        print(json.dumps(train_condition(a.condition,steps=a.steps,threads=a.threads,smoke=a.smoke),ensure_ascii=False))
    elif a.stage=='evaluate':
        if not a.condition:raise SystemExit('--condition is required')
        print(json.dumps(evaluate_condition(a.condition,threads=a.threads),ensure_ascii=False))
    else:print(json.dumps(run_all(smoke_mode=a.smoke),ensure_ascii=False))


if __name__=='__main__':main()
