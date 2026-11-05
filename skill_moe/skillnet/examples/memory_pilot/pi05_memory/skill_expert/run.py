#!/usr/bin/env python3
"""Skill-content bridge Expert training and offline checks for pi05_memory.

Inputs under run-01 are immutable. This module only writes beneath --output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = HERE.parents[4]
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN_DEFAULT = _PI05_ROOT / 'run-01'
CHECKPOINT_DEFAULT = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
sys.path.insert(0, str(PI05))

from run import (calibration_episodes, load_policy, load_rows, obs_batch, restore_codec,
                 write_json, log)  # noqa: E402
from bprior.serve import retrieve_b  # noqa: E402


def fixed_dev_ids(rows):
    dev_episodes, eval_episodes = calibration_episodes(rows)
    ids = []
    for task in np.unique(rows['task']):
        candidates = np.flatnonzero((rows['task'] == task) & np.isin(rows['episode'], dev_episodes))
        if len(candidates):
            ids.append(int(candidates[len(candidates)//2]))
    ids = np.asarray(ids, np.int64)
    if set(rows['episode'][ids].tolist()) & set(eval_episodes.tolist()):
        raise AssertionError('Dev/Eval episode leakage')
    return ids


def mix_categories(rng, size):
    """Categorical sampling with the predeclared .60/.15/.25 proportions."""
    labels = rng.choice(3, size=size, p=[.60, .15, .25])
    counts = np.bincount(labels, minlength=3)
    return labels.astype(np.int8), counts


def confidence_edges(scores):
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Dev retrieval scores must be finite and one dimensional')
    q = np.quantile(scores, [.2, .6, .8])
    return {'q20': float(q[0]), 'q60': float(q[1]), 'q80': float(q[2]),
            'n_dev': int(len(scores)), 'source': 'fixed Dev rows; no success labels'}


def confidence_mode(score, edges, condition):
    if condition not in {'C', 'A', 'B', 'D'}:
        raise ValueError(condition)
    if condition == 'D':
        return {'fallback': False, 't0': .6, 'draft': 'zero'}
    if condition == 'A':
        return {'fallback': False, 't0': .6, 'draft': 'bprior_retrieve'}
    if score < edges['q20'] and condition == 'C':
        return {'fallback': True, 't0': None, 'draft': 'normal'}
    if score < edges['q60']:
        return {'fallback': False, 't0': .6, 'draft': 'bprior_retrieve'}
    if score < edges['q80']:
        return {'fallback': False, 't0': .4, 'draft': 'bprior_retrieve'}
    return {'fallback': False, 't0': .2, 'draft': 'bprior_retrieve'}


def seed_values(init_state_id: int, query_number: int, seed_base: int = 7):
    if seed_base != 7:
        raise ValueError('this experiment fixes seed_base=7')
    env_seed = seed_base + int(init_state_id)
    return env_seed, env_seed * 100000 + int(query_number)


def make_drafts(args):
    import torch
    torch.set_num_threads(int(args.threads))
    if not torch.cuda.is_available():
        raise RuntimeError('draft search requires CUDA')
    out, run = args.output, args.run
    if any((out/name).exists() for name in ['draft_sources.npz','draft_nearest.npy','draft_rank23.npy','draft_bprior.npy']):
        raise FileExistsError(f'draft artifacts already exist under {out}; refusing to overwrite')
    rows = load_rows(run)
    codec = restore_codec(run)
    actions = np.load(run/'rows/actions.npy', mmap_mode='r')
    states = np.load(run/'rows/state.npy', mmap_mode='r')
    features = np.load(run/'pooled.npy', mmap_mode='r')
    bases = np.load(run/'bases.npy', mmap_mode='r')
    mem_ids = np.flatnonzero(rows['memory']).astype(np.int64)
    if len(mem_ids) != 6583:
        raise ValueError(f'memory inventory changed: expected 6583, got {len(mem_ids)}')
    query_ids = np.arange(len(actions), dtype=np.int64)
    # Prior-mapped retrieval is exactly bprior/serve.retrieve_b: memory-only
    # standardization and cosine Top-1. Query residual is decoded in context.
    query_b, query_r = codec.prior(np.asarray(features), np.asarray(states))
    memory_b, _ = codec.prior(np.asarray(features[mem_ids]), np.asarray(states[mem_ids]))
    memory_b = np.asarray(memory_b, np.float32)
    bmean, bstd = memory_b.mean(0), memory_b.std(0)
    bstd = np.where(bstd < 1e-5, 1., bstd).astype(np.float32)
    unit_bank = (memory_b-bmean)/bstd
    unit_bank /= np.maximum(np.linalg.norm(unit_bank, axis=1, keepdims=True), 1e-8)
    retrieve_ids = np.empty(len(actions), np.int64)
    retrieve_local = np.empty(len(actions), np.int64)
    retrieve_score = np.empty(len(actions), np.float32)
    # Exact active7 action nearest neighbors, excluding same episode. Batched
    # matrix multiplication avoids an [query,memory,10,7] temporary.
    mem_action = torch.as_tensor(np.asarray(actions[mem_ids, :, :7]).reshape(len(mem_ids), -1),
                                 device='cuda', dtype=torch.float32)
    mem_sq = (mem_action*mem_action).sum(-1)
    mem_episode = np.asarray(rows['episode'][mem_ids])
    near_ids = np.empty((len(actions), 3), np.int64)
    near_mse = np.empty((len(actions), 3), np.float32)
    second_ids=np.empty(len(actions),np.int64)
    started = time.monotonic()
    for start in range(0, len(actions), args.search_batch):
        end = min(start+args.search_batch, len(actions))
        qids = np.arange(start,end)
        q_action = torch.as_tensor(np.asarray(actions[qids, :, :7]).reshape(len(qids),-1).copy(),
                                   device='cuda', dtype=torch.float32)
        q_sq = (q_action*q_action).sum(-1,keepdim=True)
        dist = (q_sq + mem_sq[None,:] - 2*q_action@mem_action.T).clamp_min_(0) / q_action.shape[-1]
        same = torch.as_tensor(mem_episode[None,:] == np.asarray(rows['episode'][qids])[:,None],
                               device='cuda')
        dist.masked_fill_(same, torch.inf)
        vals, idxs = torch.topk(dist, k=3, largest=False, sorted=True)
        local = idxs.cpu().numpy()
        near_ids[qids] = mem_ids[local]
        near_mse[qids] = vals.cpu().numpy()
        second_ids[qids]=mem_ids[local[np.arange(len(qids)),np.where(qids%2==0,1,2)]]
        qnorm=(np.asarray(query_b[qids],np.float32)-bmean)/bstd
        qnorm/=np.maximum(np.linalg.norm(qnorm,axis=1,keepdims=True),1e-8)
        bscores=qnorm@unit_bank.T
        bsame=mem_episode[None,:]==np.asarray(rows['episode'][qids])[:,None]
        bscores[bsame]=-np.inf
        best=np.argmax(bscores,axis=1)
        retrieve_local[qids]=best;retrieve_ids[qids]=mem_ids[best];retrieve_score[qids]=bscores[np.arange(len(qids)),best]
        if np.any(np.asarray(rows['episode'][retrieve_ids[qids]])==np.asarray(rows['episode'][qids])):
            raise AssertionError('same-episode b-prior retrieval')
        if (start // args.search_batch) % 10 == 0 or end == len(actions):
            log(stage='drafts', rows=end,total=len(actions), elapsed=time.monotonic()-started)
    bprior_draft = np.lib.format.open_memmap(out/'draft_bprior.npy', mode='w+', dtype='float32',shape=(len(actions),10,32))
    nearest_draft = np.lib.format.open_memmap(out/'draft_nearest.npy', mode='w+', dtype='float32',shape=(len(actions),10,32))
    rank23_draft = np.lib.format.open_memmap(out/'draft_rank23.npy', mode='w+', dtype='float32',shape=(len(actions),10,32))
    for start in range(0,len(actions),args.search_batch):
        qids=np.arange(start,min(start+args.search_batch,len(actions)))
        feat=np.asarray(features[qids],np.float32);state=np.asarray(states[qids],np.float32);rhat=np.asarray(query_r[qids],np.float32)
        bprior_draft[qids]=codec.decode(memory_b[retrieve_local[qids]],rhat,feat,state)
        nearest_draft[qids]=codec.decode(np.asarray(bases[near_ids[qids,0]]),rhat,feat,state)
        rank23_draft[qids]=codec.decode(np.asarray(bases[second_ids[qids]]),rhat,feat,state)
        if start% (args.search_batch*10)==0 or qids[-1]==len(actions)-1:
            for a in (bprior_draft,nearest_draft,rank23_draft):a.flush()
            log(stage='decode_drafts',rows=int(qids[-1])+1,total=len(actions),elapsed=time.monotonic()-started)
    draft_errors = {}
    for name, arr in [('nearest',nearest_draft),('rank23',rank23_draft),('bprior',bprior_draft)]:
        draft_errors[name] = np.mean((np.asarray(arr[:,:, :7])-np.asarray(actions[:,:, :7]))**2,axis=(1,2))
    np.savez(out/'draft_sources.npz', query_ids=query_ids, memory_ids=mem_ids,
             nearest_ids=near_ids, nearest_mse=near_mse, retrieve_ids=retrieve_ids,
             retrieve_score=retrieve_score, draft_mse=np.stack([draft_errors[k] for k in ['nearest','rank23','bprior']],axis=1))
    write_json(out/'drafts.json', {'rows':len(actions),'memory_rows':len(mem_ids),
        'candidate_exclusion':'same episode for nearest and bprior retrieval',
        'bprior_rule':'bprior/serve.retrieve_b; memory mean/std standardized cosine Top-1',
        'mixture_probabilities':{'nearest':.60,'rank23_uniform_second_or_third':.15,'bprior':.25},
        'active7_draft_mse_mean':{k:float(v.mean()) for k,v in draft_errors.items()},
        'seconds':time.monotonic()-started})


def train(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    rows = load_rows(args.run)
    draft_dir=args.draft_dir
    if (args.output/'best_bridge.json').exists() or (args.output/'training_progress.json').exists():
        raise FileExistsError(f'training artifacts already exist under {args.output}; choose a fresh output')
    nearest = np.load(draft_dir/'draft_nearest.npy',mmap_mode='r')
    rank23 = np.load(draft_dir/'draft_rank23.npy',mmap_mode='r')
    bprior = np.load(draft_dir/'draft_bprior.npy',mmap_mode='r')
    dev_ids = fixed_dev_ids(rows)
    train_ids = np.flatnonzero(~rows['validation'])
    if np.any(np.isin(rows['episode'][train_ids], calibration_episodes(rows)[1])):
        raise AssertionError('Eval episode entered training')
    model,_ = load_policy(args.checkpoint)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected=nnx.state(model,filt)
    if sum(x.value.size for x in selected.flat_state().values())<100_000_000:
        raise ValueError('action expert filter did not select the dense tower')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt=tx.init(selected)
    @nnx.jit
    def step(current,opt_state,obs,actions,draft,rng):
        r1,r2,r3=jax.random.split(rng,3)
        t0=jax.random.uniform(r1,(actions.shape[0],),minval=.2,maxval=.6)
        progress=jax.random.uniform(r2,(actions.shape[0],),minval=.01,maxval=1.)
        eps=jax.random.normal(r3,actions.shape)
        source=(1-t0[:,None,None])*draft+t0[:,None,None]*eps
        x=(1-progress[:,None,None])*actions+progress[:,None,None]*source
        target=(source-actions)/t0[:,None,None]
        def loss_fn(m):
            v=m.velocity(obs,x,t0*progress)
            return jnp.mean(jnp.square(v[:,:,:7]-target[:,:,:7]))
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(current)
        params=nnx.state(current,filt);updates,opt_state=tx.update(grads,opt_state,params)
        nnx.update(current,optax.apply_updates(params,updates))
        return loss,opt_state
    @nnx.jit
    def validate(current,obs,actions,draft,noise):
        t0=.4; source=.6*draft+.4*noise; x=.5*actions+.5*source
        v=current.velocity(obs,x,jnp.full((len(actions),),.2))
        return jnp.mean(jnp.square(v[:,:,:7]-((source-actions)/t0)[:,:,:7]))
    np_rng=np.random.default_rng(args.seed)
    mix_rng=np.random.default_rng(args.seed ^ 0x5EED)
    key=jax.random.key(args.seed)
    total_steps=args.steps
    best=float('inf'); best_info=None; records=[]; started=time.monotonic()
    (args.output/'bridge').mkdir(exist_ok=True)
    for i in range(total_steps):
        ids=np_rng.choice(train_ids,args.batch,replace=False)
        labels,counts=mix_categories(mix_rng,len(ids))
        draft=np.empty((len(ids),10,32),np.float32)
        for k,arr in enumerate((nearest,rank23,bprior)):
            select=np.flatnonzero(labels==k)
            if len(select): draft[select]=np.asarray(arr[ids[select]])
        key,k=jax.random.split(key)
        loss,opt=step(model,opt,obs_batch(rows,ids),jnp.asarray(rows['actions'][ids]),jnp.asarray(draft),k)
        loss=float(loss)
        if not np.isfinite(loss): raise FloatingPointError('nonfinite loss')
        rec={'step':i+1,'train_loss':loss,'draft_counts':counts.tolist(), 'elapsed_s':time.monotonic()-started}
        if i % 20 == 0 or i+1 == total_steps: log(stage='train',**rec)
        if (i+1)%args.save_every==0 or i+1==total_steps:
            path=args.output/'bridge'/f'{i+1:06d}'
            ocp.PyTreeCheckpointer().save(path,nnx.state(model,filt).to_pure_dict())
            vals=[]
            for start in range(0,len(dev_ids),args.batch):
                ids_v=dev_ids[start:start+args.batch]
                eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(77123),int(row)),(10,32))) for row in ids_v])
                vals.append(float(validate(model,obs_batch(rows,ids_v),jnp.asarray(rows['actions'][ids_v]),
                                           jnp.asarray(bprior[ids_v]),jnp.asarray(eps))))
            vloss=float(np.mean(vals));rec['dev_bprior_t04_velocity_mse']=vloss
            if vloss < best:
                best=vloss;best_info={'path':str(path),'steps':i+1,'validation_velocity_mse':best,
                    'selection':'fixed Dev rows; bprior retrieval draft; t0=.4; no Eval episodes'}
                write_json(args.output/'best_bridge.json',best_info)
            write_json(args.output/'training_progress.json',{'status':'running','target_steps':total_steps,
                'logs':records+[rec],'best':best_info})
        records.append(rec)
    write_json(args.output/'trained.json',{'steps':total_steps,'seed':args.seed,
        'seconds':time.monotonic()-started,'best_checkpoint':best_info,
        'trainable_filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj',
        'batch_size':args.batch,'optimizer':'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5; clip=1',
        'draft_mix':{'nearest':.60,'rank23':.15,'bprior':.25},
        'row_sampling_rng_seed':args.seed,'batch_order_matches_run01':True,
        'training_rows':int(len(train_ids)),'dev_rows':int(len(dev_ids)),
        'training_logs':records})


def load_expert(model, checkpoint_path):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    with ocp.PyTreeCheckpointer() as ckpt: partial=ckpt.restore(checkpoint_path)
    state=nnx.state(model,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial));nnx.update(model,state)


def offline(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    if (args.output/'offline_rows.csv').exists():
        raise FileExistsError(f'offline results already exist under {args.output}')
    rows=load_rows(args.run);src=np.load(args.draft_dir/'draft_sources.npz')
    eval_episodes=calibration_episodes(rows)[1]
    eval_ids=np.flatnonzero(rows['validation'] & np.isin(rows['episode'],eval_episodes))
    if args.limit: eval_ids=eval_ids[:args.limit]
    if np.intersect1d(rows['episode'][eval_ids],calibration_episodes(rows)[0]).size:
        raise AssertionError('Eval/Dev overlap')
    # Use the established 1,318-row split and hard-row threshold, never a new
    # post-hoc split. Long is LIBERO-10 task indices 0..9 in run-01.
    gap_path=RUN_DEFAULT.parent/'memory-gap-01'/'predictions_and_errors.npz'
    long_mask=np.asarray(rows['task'][eval_ids])<10
    hard_mask=np.zeros(len(eval_ids),bool)
    hard_path=RUN_DEFAULT.parent/'retrieval-4b-01'/'raw_draft_final_mse.npz'
    if gap_path.exists() and hard_path.exists() and not args.limit:
        gap=np.load(gap_path,allow_pickle=False);hard=np.load(hard_path,allow_pickle=False)
        if not np.array_equal(gap['row_ids'],eval_ids):
            raise AssertionError('memory-gap Eval rows do not match fixed Eval split')
        long_mask=np.asarray(rows['task'][eval_ids])<10
        if int(long_mask.sum())!=548:raise AssertionError(f'expected 548 Long rows, got {long_mask.sum()}')
        if not np.array_equal(hard['eval_rows'],eval_ids):raise AssertionError('hard mask Eval row alignment changed')
        if not np.isclose(float(hard['hard_threshold']),.01549410422449,atol=1e-10,rtol=0):
            raise AssertionError(f'hard-row cutoff changed: {hard["hard_threshold"]}')
        hard_mask=np.asarray(hard['hard_mask'],bool)[long_mask]
        if int(hard_mask.sum())!=63:raise AssertionError(f'expected 63 hard Long rows, got {hard_mask.sum()}')
    else:
        hard_mask=np.zeros(int(long_mask.sum()),bool)
    rows_for_long=eval_ids[long_mask]
    rows_for_hard=rows_for_long[hard_mask] if len(hard_mask)==int(long_mask.sum()) else np.asarray([],np.int64)
    near_ids=np.asarray(src['nearest_ids']);rid=np.asarray(src['retrieve_ids'])
    near=np.load(args.draft_dir/'draft_nearest.npy',mmap_mode='r');retr=np.load(args.draft_dir/'draft_bprior.npy',mmap_mode='r')
    result=[]
    # Official π0.5 baseline, evaluated before any Expert overlay.
    official,_=load_policy(args.checkpoint);official_fn=module_jit(official.sample_actions,static_argnames=('num_steps',))
    for start in range(0,len(eval_ids),args.batch):
        ids=eval_ids[start:start+args.batch];obs=obs_batch(rows,ids)
        eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(i)),(10,32))) for i in ids])
        pred=np.asarray(official_fn(jax.random.key(0),obs,num_steps=10,noise=jnp.asarray(eps)))[:,:,:7]
        errors=np.mean((pred-np.asarray(rows['actions'][ids])[:,:,:7])**2,axis=(1,2))
        result.extend({'expert':'official','draft':'normal','t0':1.,'row_id':int(row),'mse':float(e)} for row,e in zip(ids,errors,strict=True))
    del official
    # Load independently to compare old vs new expert while preserving normal output.
    for expert_name, expert_path in [('old',Path(json.loads((args.run/'best_bridge.json').read_text())['path'])),
                                     ('new',Path(json.loads((args.expert_dir/'best_bridge.json').read_text())['path']))]:
        model,_=load_policy(args.checkpoint)
        load_expert(model,expert_path);bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',))
        for start in range(0,len(eval_ids),args.batch):
            ids=eval_ids[start:start+args.batch];obs=obs_batch(rows,ids);target=np.asarray(rows['actions'][ids])[:,:,:7]
            eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(i)),(10,32))) for i in ids])
            for draft_name,draft in [('nearest',near[ids]),('bprior_retrieve',retr[ids]),('zero',np.zeros((len(ids),10,32),np.float32))]:
                for t0 in [.2,.4,.6]:
                    pred=np.asarray(bridge(jax.random.key(0),obs,jnp.asarray(draft),jnp.full((len(ids),),t0),num_steps=10,noise=jnp.asarray(eps)))[:,:,:7]
                    err=np.mean((pred-target)**2,axis=(1,2))
                    result.extend({'expert':expert_name,'draft':draft_name,'t0':t0,'row_id':int(row),'mse':float(e)} for row,e in zip(ids,err,strict=True))
            log(stage='offline',expert=expert_name,rows=min(start+len(ids),len(eval_ids)),total=len(eval_ids))
        del model
    import csv
    with (args.output/'offline_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['expert','draft','t0','row_id','mse']);w.writeheader();w.writerows(result)
    # Long/hard IDs are identified from the fixed reference metadata when present;
    # otherwise retain explicit row split and report the unavailable subset.
    row_split={int(row):('Long' if row in set(rows_for_long.tolist()) else 'all') for row in eval_ids}
    row_split.update({int(row):'hard' for row in rows_for_hard})
    summaries={}
    for expert in sorted({x['expert'] for x in result}):
        for draft in sorted({x['draft'] for x in result if x['expert']==expert}):
            for t0 in sorted({x['t0'] for x in result if x['expert']==expert and x['draft']==draft}):
                for subset in ['all','Long','hard']:
                    vals=[x['mse'] for x in result if x['expert']==expert and x['draft']==draft and x['t0']==t0 and
                          (subset=='all' or row_split.get(x['row_id'])==subset or (subset=='Long' and row_split.get(x['row_id'])=='hard'))]
                    summaries[f'{expert}/{draft}/t0={t0}/{subset}']={'n':len(vals),'mean_active7_mse':float(np.mean(vals)) if vals else None}
    write_json(args.output/'offline_summary.json',{'rows':len(eval_ids),'results':summaries,
        'noise':'jax.random.fold_in(key(4101), row_id), shared across conditions',
        'subset_note':'Full calibration Eval split unless --limit set; Long/hard manifests are added when matching row list is supplied.'})


def calibrate(args):
    if (args.output/'confidence_edges.json').exists():
        raise FileExistsError(f'confidence edges already fixed under {args.output}')
    rows=load_rows(args.run);src=np.load(args.draft_dir/'draft_sources.npz')
    ids=fixed_dev_ids(rows)
    # only fixed Dev rows and prior retrieval cosine are used
    edges=confidence_edges(src['retrieve_score'][ids])
    write_json(args.output/'confidence_edges.json',edges)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['drafts','train','offline','calibrate'])
    p.add_argument('--run',type=Path,default=RUN_DEFAULT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT_DEFAULT)
    p.add_argument('--seed',type=int,default=20261104);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--search-batch',type=int,default=256);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--draft-dir',type=Path)
    p.add_argument('--expert-dir',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.run=a.run.resolve()
    a.draft_dir=(a.draft_dir or a.output).resolve();a.expert_dir=(a.expert_dir or a.output).resolve()
    if not a.output.exists(): a.output.mkdir(parents=True)
    if a.stage in {'train','calibrate'} and not (a.draft_dir/'draft_sources.npz').exists():
        raise FileNotFoundError(a.draft_dir/'draft_sources.npz')
    if a.stage=='offline' and not (a.expert_dir/'best_bridge.json').exists():
        raise FileNotFoundError(a.expert_dir/'best_bridge.json')
    {'drafts':make_drafts,'train':train,'offline':offline,'calibrate':calibrate}[a.stage](a)

if __name__=='__main__': main()
