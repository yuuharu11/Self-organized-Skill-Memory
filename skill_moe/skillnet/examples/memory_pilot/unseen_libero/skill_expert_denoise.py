#!/usr/bin/env python3
"""Train/evaluate the DN2 action Expert from pool_unit_08 memory drafts.

Stages are resumable: prepare (LIBERO-40 train/validation and unseen Dev only),
train (4k updates with live-weight Dev selection), evaluate (the frozen Eval split
once).  The query/draft factory is deliberately isolated for later replacement.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
ROOT = STATE / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'expert-skill-01'
RUN = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE / 'outputs/memory-pilot/pi05-memory40-20261104')) / 'run-01'
POOL = ROOT / 'cluster-pools-01/pool_unit_08.npz'
QUERY = ROOT / 'cluster-pools-01/query_unit_08.pt'
CHECKPOINT = Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE / 'checkpoints/openpi-assets/checkpoints/pi05_libero'))
UNSEEN = ROOT / 'unseen-rows-02'
EVAL_BASE = ROOT / 'oracle-01/official_normal_eval.npz'
TRAIN_SEED = 20261104
NOISE_SEED = 4101
STEPS = 4000
BATCH = 8
MICRO = 1
T0_GRID = (.2, .3, .4)
os.environ['MEM_POOL'] = str(POOL)
os.environ['MEM_CKPT'] = str(QUERY)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pi05_memory'))

def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

def _json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + '\n'); tmp.replace(path)

def _npz(path, **kw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **kw)
    tmp.replace(path)

def _load_unseen():
    from unseen_paths import UNSEEN_ROWS
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}

def _history(actions, task, episode, frame, ids):
    """Previous observed 10-action window, per episode; zero at episode start."""
    lookup = {(int(task[i]), int(episode[i]), int(frame[i])): int(i) for i in range(len(task))}
    out = np.zeros((len(ids), 10, 7), np.float32)
    for j, rid in enumerate(ids):
        prev = lookup.get((int(task[rid]), int(episode[rid]), int(frame[rid])-10))
        if prev is not None: out[j] = np.asarray(actions[prev], np.float32)[:, :7]
    return out

def draft_factory(selector, history, state, normal, task=None):
    """Replaceable current-query interface: P5(history,state)->unit->theta_pi05 pose."""
    excluded = None if task is None else np.asarray(task, np.int32)
    picked = np.empty(len(normal), np.int64)
    # group by exclusion task while preserving row order
    if excluded is None:
        picked[:] = selector.pick(history, state)
    else:
        for t in np.unique(excluded):
            ix = np.flatnonzero(excluded == t)
            picked[ix] = selector.pick(history[ix], state[ix], int(t))
        if np.any(np.asarray(selector.src_task)[picked] == excluded):
            raise AssertionError('draft factory selected a same-task memory unit')
    posed = selector.pose_to(picked, np.asarray(normal)[:, :, :7])
    out = np.zeros((len(posed), 10, 32), np.float32); out[:, :, :7] = posed
    # Memory closed-loop policy copies only the six posed motion channels; pi05 gripper is retained.
    out[:, :, 6] = np.asarray(normal)[:, :, 6]
    return out, picked

def _normal_chunks(model, base, rows, ids, batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    chunks=[]
    for start in range(0, len(ids), batch):
        ix=np.asarray(ids[start:start+batch],np.int64)
        # Explicit row-keyed Gaussian noise, independent of batching and order.
        noise=np.stack([np.asarray(jax.random.normal(
            jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in ix]).astype(np.float32)
        pred=fn(jax.random.key(0), base.obs_batch(rows, ix), num_steps=10, noise=jnp.asarray(noise))
        chunks.append(np.asarray(pred,np.float32))
        if start % 512 == 0: print(json.dumps({'stage':'normal_chunks','done':start+len(ix),'total':len(ids)}),flush=True)
    return np.concatenate(chunks) if chunks else np.empty((0,10,32),np.float32)

def prepare(*, threads=8, include_eval=False):
    import torch
    torch.set_num_threads(threads)
    import jax
    from memory_closed_loop import MemorySelector
    base=_module(HERE.parent/'pi05_memory'/'run.py','dn2_memory_base')
    rows=base.load_rows(RUN)
    selector=MemorySelector()
    model,_=base.load_policy(CHECKPOINT)
    unseen=_load_unseen()
    sets={
      'train':np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64),
      'lib40_validation':np.flatnonzero(np.asarray(rows['validation'],bool)).astype(np.int64),
      'dev':np.flatnonzero(np.asarray(unseen['is_dev'],bool)).astype(np.int64),
    }
    if include_eval: sets['eval']=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    for split,ids in sets.items():
        path=OUT/'drafts'/f'{split}.npz'
        if path.exists():
            z=np.load(path,allow_pickle=False)
            if not np.array_equal(z['row_ids'],ids): raise RuntimeError(f'cached row IDs differ: {split}')
            print(f'cached {split}: {len(ids)}',flush=True); continue
        is_unseen=split in ('dev','eval')
        src=unseen if is_unseen else rows
        n_global=len(rows['actions']) if is_unseen else 0
        target=np.asarray(src['actions'][ids],np.float32)
        task=np.asarray(src['task'][ids],np.int32); episode=np.asarray(src['episode'][ids],np.int32)
        if split=='train':
            h=_history(rows['actions'],rows['task'],rows['episode'],rows['frame'],ids)
        elif split=='lib40_validation':
            h=_history(rows['actions'],rows['task'],rows['episode'],rows['frame'],ids)
        else:
            h=_history(unseen['actions'],unseen['task'],unseen['episode'],unseen['frame'],ids)
        state=np.asarray(src['state'][ids],np.float32)[:,:8]
        # Always do inference from the actual configured pi05 checkpoint.
        normal=_normal_chunks(model,base,src,ids,batch=8)
        draft,pick=draft_factory(selector,h,state,normal,task if not is_unseen else None)
        _npz(path,row_ids=ids,global_ids=ids+n_global,task=task,episode=episode,
             history=h,state=state,target=target,draft=draft,normal=normal,selected_unit=pick)
        _json(OUT/'prepare_progress.json',{'status':'running','split':split,'rows':len(ids),'completed':True})
    _json(OUT/'prepare_progress.json',{'status':'complete','splits':{k:len(v) for k,v in sets.items()},
        'train_task_exclusion':'pool source_task != query task; entire task excluded','query':'P5 query_unit_08; replaceable draft_factory',
        'eval_prepared':bool(include_eval)})

def _pred(model, base, cache, t0, batch=8):
    import compressed_denoise as cd
    import jax
    noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32)))
                    for r in cache['global_ids']]).astype(np.float32)
    obs=base.obs_batch(cache, np.arange(len(cache['row_ids'])))
    return cd._predict_action(model,obs,cache['draft'],noise,t0,batch=batch)

def _mse(pred,target):
    p=np.asarray(pred)[:,:,:7]; y=np.asarray(target)[:,:,:7]
    return {'active7':np.mean((p-y)**2,axis=(1,2)), 'mse6':np.mean((p[:,:,:6]-y[:,:,:6])**2,axis=(1,2))*(6./7.),
            'gripper':np.mean((p[:,:,6]-y[:,:,6])**2,axis=1)}

def _make_action_grad(filt,t0_range):
    """Per-microbatch gradients; caller accumulates before applying an update."""
    import flax.nnx as nnx
    import jax, jax.numpy as jnp
    lo,hi=t0_range
    @nnx.jit
    def grad(model,obs,actions,draft,rng):
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
        return nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(model)
    return grad

def _bootstrap(v,episode,seed=4101,reps=5000):
    x=np.asarray(v,np.float64);ep=np.asarray(episode);uniq,inv=np.unique(ep,return_inverse=True)
    sums=np.bincount(inv,weights=x);cnt=np.bincount(inv);rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(uniq),(reps,len(uniq)));d=sums[ix].sum(1)/cnt[ix].sum(1)
    return {'mean':float(x.mean()),'ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))], 'episodes':len(uniq),'rows':len(x)}

def train(*,steps=STEPS,threads=8):
    import torch; torch.set_num_threads(threads)
    import jax, jax.numpy as jnp, flax.nnx as nnx, optax, orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    import compressed_denoise as cd
    base=_module(HERE.parent/'pi05_memory'/'run.py','dn2_train_base')
    rows=base.load_rows(RUN)
    cache=np.load(OUT/'drafts/train.npz',allow_pickle=False)
    dev=np.load(OUT/'drafts/dev.npz',allow_pickle=False)
    held=np.load(OUT/'drafts/lib40_validation.npz',allow_pickle=False)
    model,_=base.load_policy(CHECKPOINT)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    state=nnx.state(model,filt); assert state.flat_state()
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5)); opt=tx.init(state)
    grad_micro=_make_action_grad(filt,(.2,.5))
    train_ids=np.asarray(cache['row_ids'],np.int64); loc={int(r):i for i,r in enumerate(train_ids)}
    batches=_module(HERE.parent/'pi05_memory'/'fair_expert'/'run.py','dn2_batches').make_batch_indices(train_ids,steps=steps,batch=BATCH,seed=TRAIN_SEED)
    progress=OUT/'train_progress.json';start=0;logs=[];best={'dev_active7_mse':float('inf')}
    if progress.exists():
        info=json.loads(progress.read_text()); start=int(info.get('latest_step',0)); logs=info.get('logs',[])
        if start:
            cp=OUT/'resume'/f'{start:06d}';r=ocp.PyTreeCheckpointer().restore(cp)
            s=nnx.state(model,filt);s.replace_by_pure_dict(jax.tree.map(jnp.asarray,r['model']));nnx.update(model,s)
            # NNX/Orbax round-trips Optax named states as plain dict/list on this
            # runtime; restore the model exactly and reinitialize Adam moments.
            # This keeps restart safe without relying on an incompatible state schema.
            opt=tx.init(nnx.state(model,filt));best=info.get('best',best)
    # Dev is the only selector. LIBERO-40 validation tasks are descriptive diagnostics.
    unseen=_load_unseen()
    dev_obs=base.obs_batch(unseen,np.asarray(dev['row_ids'],np.int64))
    devnoise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in dev['global_ids']]).astype(np.float32)
    # Descriptive second validation set: fixed 300 rows stratified by task; it does not choose t0/checkpoint.
    rng_hold=np.random.default_rng(4101); hold_ids=[]
    for t in np.unique(held['task']):
        candidates=np.flatnonzero(np.asarray(held['task'])==t)
        hold_ids.extend(rng_hold.choice(candidates,min(8,len(candidates)),replace=False).tolist())
    hold_ids=np.asarray(sorted(hold_ids),np.int64)
    hold_sub={k:np.asarray(held[k])[hold_ids] for k in held.files}
    hold_obs=base.obs_batch(rows,np.asarray(hold_sub['row_ids'],np.int64))
    holdnoise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in hold_sub['global_ids']]).astype(np.float32)
    started=time.monotonic();key=jax.random.key(TRAIN_SEED)
    for _ in range(start): key,_=jax.random.split(key)
    for i in range(start,steps):
        ids=batches[i];ix=np.asarray([loc[int(r)] for r in ids])
        key,step_key=jax.random.split(key);micro_keys=jax.random.split(step_key,BATCH//MICRO)
        grad_acc=None;loss_sum=0.
        for m,st in enumerate(range(0,BATCH,MICRO)):
            sl=slice(st,min(st+MICRO,BATCH));mid=ids[sl]
            loss_i,grads_i=grad_micro(model,base.obs_batch(rows,mid),jnp.asarray(cache['target'][ix[sl]]),
                                     jnp.asarray(cache['draft'][ix[sl]]),micro_keys[m])
            loss_sum+=loss_i/(BATCH//MICRO)
            if grad_acc is None: grad_acc=jax.tree.map(lambda x:x/(BATCH//MICRO),grads_i)
            else: grad_acc=jax.tree.map(lambda a,b:a+b/(BATCH//MICRO),grad_acc,grads_i)
        state=nnx.state(model,filt);updates,opt=tx.update(grad_acc,opt,state)
        nnx.update(model,optax.apply_updates(state,updates))
        lv=float(loss_sum)
        if not np.isfinite(lv): raise FloatingPointError(f'nonfinite loss at step {i+1}')
        logs.append({'step':i+1,'loss':lv,'elapsed_s':time.monotonic()-started})
        if (i+1)%500==0 or i+1==steps:
            partial=nnx.state(model,filt).to_pure_dict()
            cp=OUT/'bridge'/f'{i+1:06d}'; cp.parent.mkdir(parents=True,exist_ok=True)
            ocp.PyTreeCheckpointer().save(cp,partial)
            resume=OUT/'resume'/f'{i+1:06d}';ocp.PyTreeCheckpointer().save(resume,{'model':partial,'opt':opt})
            scores={}
            for t0 in T0_GRID:
                pred=cd._predict_action(model,dev_obs,dev['draft'],devnoise,t0,batch=8)
                scores[str(t0)]=float(np.mean((pred[:,:,:7]-dev['target'][:,:,:7])**2))
            tsel=min(T0_GRID,key=lambda t:scores[str(t)])
            # Validation means the selected checkpoint/t0; record distance to both the draft and pi05.
            pred=cd._predict_action(model,dev_obs,dev['draft'],devnoise,tsel,batch=8)
            mm=_mse(pred,dev['target']); db=_mse(dev['draft'],dev['target']); pi=_mse(dev['normal'],dev['target'])
            hpred=cd._predict_action(model,hold_obs,hold_sub['draft'],holdnoise,tsel,batch=8)
            hmm=_mse(hpred,hold_sub['target'])
            val=float(np.mean(mm['active7']))
            event={'step':i+1,'dev_active7_mse_by_t0':scores,'selected_t0':tsel,'dev_active7_mse':val,
                'dev_mse6':float(np.mean(mm['mse6'])),'draft_active7_mse':float(np.mean(db['active7'])),'pi05_active7_mse':float(np.mean(pi['active7'])),
                'output_minus_draft_active7':float(np.mean((pred[:,:,:7]-dev['draft'][:,:,:7])**2)),
                'output_minus_pi05_active7':float(np.mean((pred[:,:,:7]-dev['normal'][:,:,:7])**2)),
                'lib40_validation_sample_rows':len(hold_ids),'lib40_validation_active7_mse':float(np.mean(hmm['active7'])),
                'lib40_validation_mse6':float(np.mean(hmm['mse6']))}
            logs[-1].update({'validation':event})
            if val<best['dev_active7_mse']:
                best={'path':str(cp),'steps':i+1,'t0':float(tsel),'dev_active7_mse':val,**event}
                _json(OUT/'best_bridge.json',best)
            _json(progress,{'status':'running','latest_step':i+1,'logs':logs,'best':best,'seed':TRAIN_SEED})
            print(json.dumps({'stage':'train','step':i+1,'loss':lv,'best':best},allow_nan=True),flush=True)
    _json(progress,{'status':'complete','latest_step':steps,'logs':logs,'best':best,'seed':TRAIN_SEED})
    _json(OUT/'trained.json',{'status':'complete','steps':steps,'seed':TRAIN_SEED,'train_rows':len(train_ids),
        'validation_rows':len(dev['row_ids']),'filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj',
        't0_range':[.2,.5],'optimizer':'AdamW 2.5e-5 clip=1 wd=1e-5',
        'resume_policy':'model checkpoint is exact; Adam moments restart if resuming after interruption',
        'best':best,'logs':logs})

def evaluate(*,threads=8):
    import torch;torch.set_num_threads(threads)
    import jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    import compressed_denoise as cd
    from memory_closed_loop import MemorySelector
    base=_module(HERE.parent/'pi05_memory'/'run.py','dn2_eval_base')
    rows=base.load_rows(RUN);unseen=_load_unseen();ids=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    path=OUT/'drafts/eval.npz'
    # Eval is read for the first/only time only after selected checkpoint and t0 are frozen.
    if not path.exists():
        selection=json.loads((OUT/'best_bridge.json').read_text());selector=MemorySelector()
        official=np.load(EVAL_BASE,allow_pickle=False); expected=len(rows['actions'])+ids
        take=np.searchsorted(np.asarray(official['row_ids'],np.int64),expected)
        if np.any(take>=len(official['row_ids'])) or not np.array_equal(official['row_ids'][take],expected):raise RuntimeError('official pi05 Eval rows not aligned')
        normal=np.asarray(official['prediction'][take],np.float32)
        task=np.asarray(unseen['task'][ids],np.int32);episode=np.asarray(unseen['episode'][ids],np.int32)
        h=_history(unseen['actions'],unseen['task'],unseen['episode'],unseen['frame'],ids)
        state=np.asarray(unseen['state'][ids],np.float32)[:,:8]
        draft,pick=draft_factory(selector,h,state,normal,None)
        _npz(path,row_ids=ids,global_ids=expected,task=task,episode=episode,target=np.asarray(unseen['actions'][ids],np.float32),
             history=h,state=state,draft=draft,normal=normal,selected_unit=pick)
    ev=np.load(path,allow_pickle=False); selection=json.loads((OUT/'best_bridge.json').read_text())
    model,_=base.load_policy(CHECKPOINT);filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    part=ocp.PyTreeCheckpointer().restore(Path(selection['path']));st=nnx.state(model,filt);st.replace_by_pure_dict(jax.tree.map(jnp.asarray,part));nnx.update(model,st)
    obs=base.obs_batch(unseen,ids);noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in ev['global_ids']]).astype(np.float32)
    pred=cd._predict_action(model,obs,ev['draft'],noise,float(selection['t0']),batch=8)
    # Oracle type draft: allowed only in this post-selection scoring diagnostic.
    from denoise_eval import pose_oracle_types, pad_actions
    import variable_shape_memory as vs
    pool=np.load(POOL,allow_pickle=False)
    oi,_,od7=pose_oracle_types(ev['target'][:,:,:7],ev['normal'][:,:,:7],pool['shape'],vs,cond='S-shared')
    od=pad_actions(od7)
    # Predictions for the main draft, oracle draft, and exact-action upper bound use identical noise.
    main=pred
    oracle_pred=cd._predict_action(model,obs,od,noise,float(selection['t0']),batch=8)
    gt=np.zeros_like(ev['draft']);gt[:,:,:7]=ev['target'][:,:,:7]
    gt_pred=cd._predict_action(model,obs,gt,noise,float(selection['t0']),batch=8)
    target=ev['target'][:,:,:7];hard=np.asarray(np.load(ROOT/'oracle-01/drafts/raw_none_eval.npz',allow_pickle=False)['pre_mse'])>=float(json.loads((ROOT/'oracle-01/hard_threshold.json').read_text())['threshold'])
    # join hard by global row id, never positional assumptions
    raw=np.load(ROOT/'oracle-01/drafts/raw_none_eval.npz',allow_pickle=False);hp=np.searchsorted(raw['query_ids'],ev['global_ids']);hard=np.asarray(raw['pre_mse'][hp])>=float(json.loads((ROOT/'oracle-01/hard_threshold.json').read_text())['threshold'])
    examples={'skill_expert_bridge':main,'oracle_type_bridge':oracle_pred,'exact_action_draft_bridge':gt_pred}
    report={'selected':selection,'rows':len(ids),'hard_rows':int(hard.sum()),'eval_used_for_selection':False,'t0_grid_all':list(T0_GRID),'conditions':{}}
    boot_episode=np.asarray(ev['task'],np.int64)*100000+np.asarray(ev['episode'],np.int64)
    payload={'row_ids':ids,'global_ids':ev['global_ids'],'task':ev['task'],'episode':ev['episode'],'hard':hard,'target':target,'draft':ev['draft'][:,:,:7],'normal':ev['normal'][:,:,:7]}
    for name,p in examples.items():
        m=_mse(p,target);d=_mse(ev['draft'],target)
        for subset,mask in [('all',np.ones(len(ids),bool)),('hard',hard)]:
            report['conditions'].setdefault(name,{})[subset]={}
            for metric,v in m.items():report['conditions'][name][subset][metric]=_bootstrap(v[mask],boot_episode[mask])
        report['conditions'][name]['output_minus_draft_active7']=float(np.mean((p[:,:,:7]-ev['draft'][:,:,:7])**2))
        report['conditions'][name]['output_minus_pi05_active7']=float(np.mean((p[:,:,:7]-ev['normal'][:,:,:7])**2))
        payload['pred_'+name]=p[:,:,:7]
        payload['dist_draft_'+name]=np.mean((p[:,:,:7]-ev['draft'][:,:,:7])**2,axis=(1,2))
        payload['dist_pi05_'+name]=np.mean((p[:,:,:7]-ev['normal'][:,:,:7])**2,axis=(1,2))
    for name,p in [('draft',ev['draft']),('normal_pi05',ev['normal']),('hold_last',np.repeat(ev['history'][:,-1:, :],10,axis=1))]:
        m=_mse(p,target)
        report['conditions'][name]={s:{k:_bootstrap(v[mask],boot_episode[mask]) for k,v in m.items()} for s,mask in [('all',np.ones(len(ids),bool)),('hard',hard)]}
    report['dn1_comparison']={'public_expert_t0_0.6_mse6_all_hard':[.11467,.25100],
        'compressed_A_vm_08_t0_0.4_mse6_all_hard':[.11419,.24796],
        'metric':'MSE6=sum(squared error over motion six axes)/7; active7=mean over 7 axes'}
    _npz(OUT/'eval_predictions.npz',**payload);_json(OUT/'eval_summary.json',report)
    trained=json.loads((OUT/'trained.json').read_text())
    curve=[x for x in trained.get('logs',[]) if 'validation' in x]
    lines=['# DN2: Skill Expert デノイズ（LIBERO）','',
        '評価は選択用Devでcheckpoint/t0を固定した後に一度だけ実行。unit_08のP5下書きは通常π0.5のチャンクへSO(3)+共通倍率で合わせ、gripperはπ0.5からコピー。',
        'MSE6は6軸二乗誤差の平均×6/7、active7は7軸の二乗誤差平均。CIはepisode単位bootstrap 5,000回。','',
        f"選択: step {selection['steps']}、t0={selection['t0']:.2f}、Dev active7 MSE={selection['dev_active7_mse']:.6f}。",'',
        '| 条件 | subset | MSE6 [95% CI] | active7 [95% CI] | gripper [95% CI] | 出力−draft active7 | 出力−π0.5 active7 |',
        '|---|---|---:|---:|---:|---:|---:|']
    lines+=['','## 学習曲線（各500更新時、live weights）','',
        '| step | train loss | Dev t0=.2 | Dev t0=.3 | Dev t0=.4 | selected t0 | Dev active7 | LIBERO-40 holdout sample active7 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in curve:
        v=x['validation'];sc=v['dev_active7_mse_by_t0']
        lines.append(f"| {x['step']} | {x['loss']:.6f} | {sc['0.2']:.6f} | {sc['0.3']:.6f} | {sc['0.4']:.6f} | {v['selected_t0']:.1f} | {v['dev_active7_mse']:.6f} | {v['lib40_validation_active7_mse']:.6f} |")
    lines += ['', 'LIBERO-40 holdout diagnosticはtaskごと8行（最大320行）を固定seedで層化抽出し、checkpoint/t0選択には使っていない。']
    fmt=lambda d:f"{d['mean']:.6f} [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]"
    for name,subsets in report['conditions'].items():
        for subset in ('all','hard'):
            if subset not in subsets: continue
            v=subsets[subset]
            lines.append(f"| {name} | {subset} | {fmt(v['mse6'])} | {fmt(v['active7'])} | {fmt(v['gripper'])} | {subsets.get('output_minus_draft_active7',float('nan')):.6f} | {subsets.get('output_minus_pi05_active7',float('nan')):.6f} |")
    lines+=['','## DN1 comparator','',
        '| DN1 public Expert t0=.6 MSE6 all/hard | DN1 compressed A-vm-08 t0=.4 MSE6 all/hard |',
        '|---:|---:|',
        '| .11467 / .25100 | .11419 / .24796 |','',
        'このオフライン比較は保留デモを検索評価に使う上限診断で、閉ループ成功率の代用ではない。正解行動下書きはデノイザーの診断対照。','']
    (OUT/'report.md').write_text('\n'.join(lines))
    return report

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('prepare','train','evaluate'):
        s=sp.add_parser(cmd);s.add_argument('--threads',type=int,default=8)
        if cmd=='train':s.add_argument('--steps',type=int,default=STEPS)
        if cmd=='prepare':s.add_argument('--include-eval',action='store_true')
    a=p.parse_args()
    if a.cmd=='prepare':prepare(threads=a.threads,include_eval=a.include_eval)
    elif a.cmd=='train':train(steps=a.steps,threads=a.threads)
    else:evaluate(threads=a.threads)
if __name__=='__main__':main()
