"""Full RM-EXP2 bridge training on the fixed memory-side P5 drafts."""
from __future__ import annotations

import json, os, pickle, time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import expert
import expert2_prep as prep

STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
ROOT = Path(os.environ.get('ROBOCASA_EXPERT2_ROOT', str(STATE / 'outputs/robocasa-memory/expert-02')))
SOURCE = expert.SOURCE
CHECKPOINT = expert.CHECKPOINT
PREP = Path(os.environ.get('ROBOCASA_EXPERT2_PREP', str(ROOT / 'prep/attempt-10')))
OUT = Path(os.environ.get('ROBOCASA_EXPERT2_OUT', str(ROOT / 'train/attempt-01')))
T0 = 0.2
SEED = 20261111


def _json(path, x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=expert.ex._json_default)+'\n'); tmp.replace(path)


def _val_sample(data, per_task=50, seed=20261112):
    valid=np.flatnonzero(np.asarray(data['validation'],bool)); tasks=np.asarray(data['task']); eps=np.asarray(data['episode']); frames=np.asarray(data['frame'])
    rng=np.random.default_rng(seed); keep=[]
    for t in sorted(np.unique(tasks[valid]).tolist()):
        ids=valid[tasks[valid]==t]
        by={int(e):ids[eps[ids]==e] for e in np.unique(eps[ids])}
        epids=sorted(by); rng.shuffle(epids)
        # Round-robin over episodes, then over frame order, to keep all reserved episodes represented.
        buckets=[list(sorted(by[e].tolist(),key=lambda i: int(frames[i]))) for e in epids]
        picked=[]
        while len(picked)<per_task and any(buckets):
            for b in buckets:
                if b and len(picked)<per_task:
                    j=int(rng.integers(len(b))); picked.append(b.pop(j))
        keep.extend(picked)
    return np.asarray(sorted(keep),np.int64)


def prepare_full_cache(out=OUT, *, threads=8):
    """Cache all 9,088 fit rows plus a fixed 300-row episode-stratified validation sample.

    It reuses matching rows from prep/attempt-10's immutable 300-per-task cache,
    then extracts only missing observations. Per-task files are atomically replaced.
    """
    import jax
    import torch
    torch.set_num_threads(threads)
    out=Path(out); data=np.load(PREP/'drafts/memory_dynamic.npz',allow_pickle=False)
    train_ids=np.flatnonzero(~np.asarray(data['validation'],bool)); val_ids=_val_sample(data)
    selected=np.unique(np.r_[train_ids,val_ids]); tasks=np.asarray(data['task']); eps=np.asarray(data['episode']); frames=np.asarray(data['frame'])
    manifest=json.loads((SOURCE/'manifest.json').read_text()); info={(int(r['task_id']),int(r['episode'])):r for r in manifest['demos']}
    cache_dir=out/'obs_cache'; cache_dir.mkdir(parents=True,exist_ok=True)
    reuse_dirs=[ROOT/'train/attempt-03/obs_cache',ROOT/'train/attempt-02/obs_cache',PREP/'train/obs_cache']
    report={'train_rows':len(train_ids),'validation_rows':len(val_ids),'validation_rows_per_task':50,'reused_rows':0,'extracted_rows':0,'tasks':{}}
    for task in sorted(np.unique(tasks[selected]).tolist()):
        ids=selected[tasks[selected]==task]
        dest=cache_dir/f'task_{task}.pkl'
        if dest.exists():
            old=pickle.load(dest.open('rb'))
            if np.array_equal(np.sort(old['row_ids']),np.sort(ids)):
                report['tasks'][str(task)]={'rows':len(ids),'reused_existing':True}; continue
            raise FileExistsError(f'partial cache needs resume-aware merge, refusing overwrite: {dest}')
        old=None; old_map={}; old_tree=None
        for reuse_dir in reuse_dirs:
            old_path=reuse_dir/f'task_{task}.pkl'
            if not old_path.exists(): continue
            candidate=pickle.load(old_path.open('rb'))
            if old_tree is None: old_tree=candidate['tree_def']
            elif candidate['tree_def']!=old_tree: raise AssertionError('reused observation tree mismatch')
            for pos,row in enumerate(candidate['row_ids']):
                if int(row) not in old_map: old_map[int(row)]=[np.asarray(x[pos]) for x in candidate['leaves']]
            old=candidate
        row_leaves={}; row_targets={}
        for row in ids:
            if int(row) in old_map:
                row_leaves[int(row)]=old_map[int(row)]; continue
            obs,target,_,_=expert._build_obs_and_target(info[(int(task),int(eps[row]))],int(frames[row]),SOURCE,CHECKPOINT)
            leaves,td=jax.tree.flatten(obs)
            if 'tree_def' not in locals(): tree_def=td
            if td != tree_def: raise AssertionError('observation tree differs across rows')
            vals=[]
            for leaf in leaves:
                a=np.asarray(jax.device_get(leaf)); vals.append(a[0] if a.shape[:1]==(1,) else a)
            row_leaves[int(row)]=vals; row_targets[int(row)]=np.asarray(target,np.float32)
            report['extracted_rows']+=1
            if report['extracted_rows']%100==0:
                print(json.dumps({'stage':'obs_cache','extracted':report['extracted_rows'],'task':int(task)}),flush=True)
        # Recover cached targets for reused rows from the fixed draft dataset; observation cache targets are not used.
        ordered=sorted(map(int,ids)); leaves_stacked=[np.stack([row_leaves[row][j] for row in ordered]) for j in range(len(row_leaves[ordered[0]]))]
        # Save only the observation leaves and row mapping. Targets/drafts are aligned in memory_dynamic.npz.
        payload={'row_ids':np.asarray(ordered,np.int64),'tree_def':(old_tree if old_tree is not None else td),'leaves':leaves_stacked}
        tmp=dest.with_suffix('.pkl.tmp')
        with tmp.open('wb') as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dest)
        report['reused_rows']+=len(set(ordered)&set(old_map)); report['tasks'][str(task)]={'rows':len(ids),'reused':len(set(ordered)&set(old_map)),'extracted':len(ids)-len(set(ordered)&set(old_map))}
    _json(out/'cache_manifest.json',report)
    return train_ids,val_ids,selected


def _fixed_grad(filt):
    import jax, jax.numpy as jnp
    import flax.nnx as nnx
    @nnx.jit
    def step(model, obs, target, draft, key):
        kt,kp,ke=jax.random.split(key,3); b=target.shape[0]
        t0=jnp.full((b,),T0,jnp.float32); progress=jax.random.uniform(kp,(b,),minval=.01,maxval=1.)
        noise=jax.random.normal(ke,target.shape)
        xt,t,vel=expert._bridge_training_sample(target,draft,noise,t0,progress)
        weight=jnp.asarray([1.]*7+[.1]*5+[0.]*20,jnp.float32)
        def loss_fn(m):
            pred=m.velocity(obs,xt,t)
            return jnp.sum((pred-vel)**2*weight[None,None,:])/(jnp.sum(weight)*target.shape[1]*b)
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(model)
        return loss,grads
    return step


def train(out=OUT, *, steps=3000, effective_batch=32, microbatch=8, threads=8, memory_fraction=.45):
    import jax, jax.numpy as jnp
    import flax.nnx as nnx
    import optax, torch
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex
    torch.set_num_threads(threads)
    if effective_batch%microbatch: raise ValueError('effective batch must divide by microbatch')
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    out=Path(out); dpath=PREP/'drafts/memory_dynamic.npz'; data=np.load(dpath,allow_pickle=False)
    train_ids,val_ids,_=prepare_full_cache(out,threads=threads)
    cache_map={}; tree_def=None
    for task in sorted(np.unique(data['task'][np.r_[train_ids,val_ids]]).tolist()):
        payload=pickle.load((out/'obs_cache'/f'task_{task}.pkl').open('rb'))
        if tree_def is None: tree_def=payload['tree_def']
        if payload['tree_def']!=tree_def: raise AssertionError('cached observation PyTrees differ')
        for pos,row in enumerate(payload['row_ids']): cache_map[int(row)]=[np.asarray(x[pos]) for x in payload['leaves']]
    if not set(map(int,np.r_[train_ids,val_ids])).issubset(cache_map): raise AssertionError('full observation cache incomplete')
    print(json.dumps({'stage':'cache_loaded','train_rows':len(train_ids),'validation_rows':len(val_ids),'cached_rows':len(cache_map)}),flush=True)
    stats=__import__('openpi.shared.normalize',fromlist=['load']).load(CHECKPOINT/'assets')['actions']
    # Only drafts used for fitting are transformed; reserve validation is the 300-row fixed sample.
    # NpzFile.__getitem__ inflates the entire compressed member at each access;
    # materialize once before the row-wise conversion.
    raw_drafts=np.asarray(data['draft'],np.float32)
    raw_targets=np.asarray(data['target'],np.float32)
    unit_lengths_all=np.asarray(data['first_unit_length'],np.int32)
    draft=np.stack([expert._draft_to_model(row,stats) for row in raw_drafts]).astype(np.float32)
    target=np.stack([expert._draft_to_model(row,stats) for row in raw_targets]).astype(np.float32)
    print(json.dumps({'stage':'draft_arrays_ready','rows':len(draft),'shape':list(draft.shape)}),flush=True)
    model,_=expert._load_expert_model(); filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    print(json.dumps({'stage':'model_loaded'}),flush=True)
    state=nnx.state(model,filt); tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5)); opt=tx.init(state); grad=_fixed_grad(filt)
    fit=np.asarray(train_ids,np.int64); rng=np.random.default_rng(SEED); logs=[]; best=float('inf'); best_step=0
    expdir=out/'expert'; expdir.mkdir(parents=True,exist_ok=True)
    # A fixed update order makes restart/evaluation comparisons reproducible.
    sample_path=out/'train_row_order.npy'
    if sample_path.exists(): order=np.load(sample_path,mmap_mode='r')
    else:
        order=rng.choice(fit,(steps,effective_batch),replace=True).astype(np.int32); np.save(sample_path,order)
    print(json.dumps({'stage':'optimizer_ready','steps':steps,'effective_batch':effective_batch,'microbatch':microbatch}),flush=True)
    # Do not use module_jit here: it freezes the NNX model state when the
    # wrapper is created, so every validation checkpoint would be evaluated
    # with the initial weights. Passing the model through nnx.jit keeps the
    # current trainable state dynamic.
    @nnx.jit(static_argnames=('num_steps',))
    def eval_fn(m, rng, obs, dn, starts, noise, *, num_steps=10):
        return m.sample_bridge(rng, obs, dn, starts, num_steps=num_steps, noise=noise)
    def make_batch(ids):
        leaves=[jnp.asarray(np.stack([cache_map[int(i)][j] for i in ids])) for j in range(len(cache_map[int(ids[0])]))]
        return jax.tree.unflatten(tree_def,leaves)
    training_started=time.monotonic(); actual_steps=0; hit_budget=False
    for step0 in range(steps):
        ids=np.asarray(order[step0]); accum=None; losses=[]
        for start in range(0,effective_batch,microbatch):
            sub=ids[start:start+microbatch]; obs=make_batch(sub)
            loss,grads=grad(model,obs,jnp.asarray(target[sub]),jnp.asarray(draft[sub]),jax.random.fold_in(jax.random.key(SEED),step0*17+start))
            accum=grads if accum is None else jax.tree.map(lambda a,b:a+b,accum,grads); losses.append(float(loss))
        accum=jax.tree.map(lambda g:g/(effective_batch//microbatch),accum)
        state=nnx.state(model,filt); updates,opt=tx.update(accum,opt,state); nnx.update(model,optax.apply_updates(state,updates))
        actual_steps=step0+1
        rec={'step':step0+1,'train_loss':float(np.mean(losses)),'t0':T0,'microbatch':microbatch,'effective_batch':effective_batch}; logs.append(rec)
        if (step0+1)%500==0 or step0+1==steps:
            preds=[]; truth=[]; unit_lengths=[]
            for st in range(0,len(val_ids),4):
                bid=np.asarray(val_ids[st:st+4],np.int64); obs=make_batch(bid)
                dn=jnp.asarray(draft[bid]); eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(i)),(1,50,32)))[0] for i in bid])
                pr=np.asarray(eval_fn(model,jax.random.key(0),obs,dn,jnp.full((len(bid),),T0,jnp.float32),jnp.asarray(eps),num_steps=10))
                preds.extend(pr); truth.extend(target[bid]); unit_lengths.extend(np.clip(unit_lengths_all[bid],1,50).tolist())
            pred=np.asarray(preds); tru=np.asarray(truth); mse5=float(np.mean((pred[:,:5,:6]-tru[:,:5,:6])**2))
            errs=[]
            for p,t,l in zip(pred,tru,unit_lengths,strict=True): errs.append(float(np.mean((p[:int(l),:6]-t[:int(l),:6])**2)))
            unit=float(np.mean(errs)); metric=mse5
            rec.update({'val_rows':len(val_ids),'val_first5_mse6':mse5,'val_unit_mse6':unit})
            cp=expdir/f'expert_{step0+1:06d}.msgpack'
            if not cp.exists(): cp.write_bytes(serialization.to_bytes(nnx.state(model,filt).to_pure_dict()))
            _json(expdir/'progress.json',{'status':'running','latest_step':step0+1,'steps':steps,'best_step':best_step,'best_val_first5_mse6':best,'logs':logs,'checkpoint':cp.name,'t0':T0,'VLM_frozen':True,'trainable_filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj'})
            if metric<best: best,best_step=metric,step0+1
            print(json.dumps(rec),flush=True)
            del preds,truth,pred,tru,obs,eps,pr; import gc; gc.collect()
        elif (step0+1)%100==0:
            _json(expdir/'progress.json',{'status':'running','latest_step':step0+1,'steps':steps,'best_step':best_step,'best_val_first5_mse6':best,'logs':logs,'t0':T0})
            print(json.dumps(rec),flush=True)
        if time.monotonic()-training_started >= 3600:
            hit_budget=True
            print(json.dumps({'stage':'train','status':'one_hour_budget_reached','latest_step':actual_steps}),flush=True)
            break
    if actual_steps%500 and actual_steps>0:
        last=expdir/f'expert_{actual_steps:06d}.msgpack'
        if not last.exists(): last.write_bytes(serialization.to_bytes(nnx.state(model,filt).to_pure_dict()))
    status='time_budget_reached' if hit_budget else 'complete'
    _json(expdir/'best.json',{'checkpoint':f'expert_{best_step:06d}.msgpack','steps':best_step,'t0':T0,'val_first5_mse6':best,'selection':'fixed 300 memory validation rows, first 5 steps MSE6'})
    _json(expdir/'progress.json',{'status':status,'latest_step':actual_steps,'steps_requested':steps,'best_step':best_step,'best_val_first5_mse6':best,'logs':logs,'t0':T0,'training_seconds':time.monotonic()-training_started})
    return expdir/f'expert_{best_step:06d}.msgpack'


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT); p.add_argument('--steps',type=int,default=3000); p.add_argument('--microbatch',type=int,default=8); p.add_argument('--effective-batch',type=int,default=32); p.add_argument('--threads',type=int,default=8); p.add_argument('--fraction',type=float,default=.45)
    a=p.parse_args(); train(a.out,steps=a.steps,effective_batch=a.effective_batch,microbatch=a.microbatch,threads=a.threads,memory_fraction=a.fraction)
