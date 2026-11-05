"""Representation 4-A: offline compression/coordinate-frame comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN=_PI05_ROOT / 'run-01'
GAP=_PI05_ROOT / 'memory-gap-01'
OUT=_PI05_ROOT / 'representation-4a-01'
OFFICIAL=_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
NOISE_SEEDS=(4101,4102)
T0S=(0.2,0.4)
BATCH=4
sys.path.insert(0,str(PI05))


def rows_load(path):
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (path/'rows').glob('*.npy')}


def _quantile_stats():
    from openpi.shared.normalize import load
    return load(OFFICIAL/'assets/physical-intelligence/libero')


def _unnorm(x, stats):
    x=np.asarray(x,np.float32); q01=np.asarray(stats.q01,np.float32)[...,:x.shape[-1]]; q99=np.asarray(stats.q99,np.float32)[...,:x.shape[-1]]
    return (x+1.)*.5*(q99-q01+1e-6)+q01


def _norm(x, stats):
    x=np.asarray(x,np.float32); q01=np.asarray(stats.q01,np.float32)[...,:x.shape[-1]]; q99=np.asarray(stats.q99,np.float32)[...,:x.shape[-1]]
    return (x-q01)/(q99-q01+1e-6)*2.-1.


def _axis_angle_R(v):
    v=np.asarray(v,np.float64); theta=np.linalg.norm(v,axis=-1)
    a=v/np.maximum(theta[...,None],1e-12)
    x,y,z=np.moveaxis(a,-1,0)
    K=np.zeros(v.shape[:-1]+(3,3),np.float64)
    K[...,0,1]=-z; K[...,0,2]=y; K[...,1,0]=z; K[...,1,2]=-x; K[...,2,0]=-y; K[...,2,1]=x
    I=np.broadcast_to(np.eye(3),K.shape)
    R=I+np.sin(theta)[...,None,None]*K+(1-np.cos(theta))[...,None,None]*(K@K)
    small=theta<1e-10
    if np.any(small): R[small]=np.eye(3)
    return R.astype(np.float32)


def _state_R(state_norm, state_stats):
    physical=_unnorm(np.asarray(state_norm)[...,:6],state_stats)
    # LIBERO policy state is EEF xyz followed by axis-angle orientation.
    return _axis_angle_R(physical[...,3:6])


def to_eef_relative(actions_norm, state_norm, action_stats, state_stats):
    a=np.asarray(actions_norm,np.float32).copy()
    physical=_unnorm(a[...,:7],action_stats)
    R=_state_R(state_norm,state_stats)
    Rt=np.swapaxes(R,-1,-2)
    # R is [N,3,3]; deltas are expressed in the world basis and rotate into the initial EEF basis.
    physical[...,:3]=np.einsum('bij,btj->bti',Rt,physical[...,:3])
    physical[...,3:6]=np.einsum('bij,btj->bti',Rt,physical[...,3:6])
    a[...,:7]=_norm(physical,action_stats)
    return a


def from_eef_relative(actions_rel_norm, query_state_norm, action_stats, state_stats):
    a=np.asarray(actions_rel_norm,np.float32).copy()
    physical=_unnorm(a[...,:7],action_stats)
    R=_state_R(query_state_norm,state_stats)
    physical[...,:3]=np.einsum('bij,btj->bti',R,physical[...,:3])
    physical[...,3:6]=np.einsum('bij,btj->bti',R,physical[...,3:6])
    a[...,:7]=_norm(physical,action_stats)
    return a


def _cluster(task,episode):
    return np.asarray([f'{int(t)}:{int(e)}' for t,e in zip(task,episode,strict=True)])


def _ridge_fit(state, actions, alpha=1.0):
    x=np.asarray(state,np.float64); y=np.asarray(actions,np.float64).reshape(len(actions),-1)
    mean=x.mean(0); std=np.maximum(x.std(0),1e-3)
    xn=(x-mean)/std
    xa=np.concatenate([np.ones((len(x),1)),xn],axis=1)
    reg=np.eye(xa.shape[1])*float(alpha); reg[0,0]=0
    coef=np.linalg.solve(xa.T@xa+reg,xa.T@y)
    return {'mean':mean.astype(np.float32),'std':std.astype(np.float32),'coef':coef.astype(np.float32),'alpha':float(alpha)}


def _ridge_predict(model, state):
    x=(np.asarray(state,np.float32)-model['mean'])/model['std']
    xa=np.concatenate([np.ones((len(x),1),np.float32),x],axis=1)
    return (xa@model['coef']).reshape(len(x),10,7)


def _fit_codec(rows, features, train_ids, *, base_dim, seed=20261104, actions_override=None, tag=''):
    import torch
    sys.path.insert(0,str(PI05.parents[1]))
    from emergent.light.methods.m1 import M1
    torch.set_num_threads(8)
    model=M1(epochs=30,batch_size=128,seed_value=seed,base_dim=base_dim,residual_dim=4,device='cpu',
             independence_weight=.01,beta_r=1.,base_use_weight=5.,base_use_margin=.05,width=128,decoder_obs=False)
    A=np.asarray(rows['actions'][train_ids] if actions_override is None else actions_override[train_ids],np.float32)
    S=np.asarray(rows['state'][train_ids],np.float32); F=np.asarray(features[train_ids],np.float32)
    task=np.asarray(rows['task'])[train_ids]; epi=np.asarray(rows['episode'])[train_ids]
    model.fit({'actions':A,'state':S,'feature':F,'key':F,
               'episode_id':np.asarray([f'{t}:{e}' for t,e in zip(task,epi,strict=True)]),
               'init_state_id':np.asarray([f'{t}:{e}' for t,e in zip(task,epi,strict=True)]),
               'seed':seed})
    return model


def _noise(jax,seed,row_id):
    return np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(seed),int(row_id)),(10,32)),np.float32)


def _infer_new(run, ids, rows, drafts_by_key, *, output):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex,module_jit
    from run import load_policy,obs_batch
    best=json.loads((run/'best_bridge.json').read_text())
    best_path=Path(best['path'])
    if best_path!=run/'bridge/001500' or int(best['steps'])!=1500:
        raise ValueError(f'not the requested best bridge checkpoint: {best}')
    model,_=load_policy(OFFICIAL)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    with ocp.PyTreeCheckpointer() as ckpt: partial=ckpt.restore(best_path)
    state=nnx.state(model,filt); state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial)); nnx.update(model,state)
    bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',))
    names=list(drafts_by_key)
    preds=np.empty((len(names),len(NOISE_SEEDS),len(ids),10,7),np.float32)
    started=time.monotonic(); batches=0
    for si,seed in enumerate(NOISE_SEEDS):
        for start in range(0,len(ids),BATCH):
            end=min(start+BATCH,len(ids)); qids=ids[start:end]
            obs=obs_batch(rows,qids); eps=jnp.asarray(np.stack([_noise(jax,seed,i) for i in qids]))
            key=jax.random.fold_in(jax.random.key(seed),start)
            for ci,name in enumerate(names):
                t0,draft=drafts_by_key[name]
                y=bridge(key,obs,jnp.asarray(draft[start:end]),jnp.full((end-start,),t0),num_steps=10,noise=eps)
                preds[ci,si,start:end]=np.asarray(y)[:,:,:7]
            batches+=1
            if batches%20==0 or end==len(ids):
                with (output/'progress.jsonl').open('a') as f:
                    f.write(json.dumps({'event':'infer_new','seed':seed,'rows_done':end,'total':len(ids),
                                        'batches':batches,'elapsed_s':time.monotonic()-started,
                                        'gpu_used_mib':_gpu_mib()})+'\n')
    return names,preds,time.monotonic()-started


def _gpu_mib():
    import subprocess
    try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0])
    except Exception:return None


def _boot_ci(diff, clusters, mask, *, seed=20261104, reps=5000):
    ix=np.flatnonzero(mask)
    if not len(ix):return [float('nan'),float('nan')]
    cl=clusters[ix]; values=np.asarray(diff,np.float64); uniq=np.unique(cl)
    blocks=[ix[cl==u] for u in uniq]; rng=np.random.default_rng(seed); out=np.empty(reps)
    for i in range(reps):
        pick=rng.integers(0,len(blocks),len(blocks)); sample=np.concatenate([blocks[j] for j in pick]); out[i]=values[sample].mean()
    return [float(x) for x in np.quantile(out,[.025,.975])]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,default=RUN); ap.add_argument('--gap',type=Path,default=GAP); ap.add_argument('--output',type=Path,default=OUT); ap.add_argument('--row-limit',type=int,default=None)
    args=ap.parse_args(); out=args.output
    if out.exists() and any(not (p.suffix in {'.log','.sh'} or p.name.startswith('pid')) for p in out.iterdir()):
        raise FileExistsError(f'output contains prior experiment artifacts: {out}')
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false'); os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.20')
    rows=rows_load(args.run); feat=np.load(args.run/'pooled.npy',mmap_mode='r',allow_pickle=False)
    gap=np.load(args.gap/'predictions_and_errors.npz',allow_pickle=False)
    ids=np.asarray(gap['row_ids'],np.int64)
    if len(ids)!=1318:raise AssertionError(f'expected 1318 fixed Eval rows, got {len(ids)}')
    if args.row_limit is not None: ids=ids[:args.row_limit]
    mem_ids=np.flatnonzero(rows['memory']); train_ids=np.flatnonzero(~rows['validation'] & ~rows['memory'])
    if len(mem_ids)!=6583 or len(train_ids)!=16721:raise AssertionError((len(mem_ids),len(train_ids)))
    old_row_map={int(r):i for i,r in enumerate(gap['row_ids'])}; gap_ix=np.asarray([old_row_map[int(r)] for r in ids])
    task=np.asarray(rows['task'])[ids]; episode=np.asarray(rows['episode'])[ids]; clusters=_cluster(task,episode)
    candidate_global={'episode':gap['chosen_candidate_rows'][0,gap_ix].astype(np.int64),
                      'task':gap['chosen_candidate_rows'][1,gap_ix].astype(np.int64)}
    if np.any(~np.asarray(rows['memory'])[candidate_global['episode']]) or np.any(~np.asarray(rows['memory'])[candidate_global['task']]):
        raise AssertionError('stage2 candidate row is not in the memory')
    for exclusion,candidates in candidate_global.items():
        ct=np.asarray(rows['task'])[candidates]; ce=np.asarray(rows['episode'])[candidates]
        if exclusion=='episode' and np.any((ct==task)&(ce==episode)):
            raise AssertionError('same-episode oracle candidate leaked into retrieval')
        if exclusion=='task' and np.any(ct==task):
            raise AssertionError('same-task oracle candidate leaked into retrieval')

    # Source distributions and normalization transform are verified before any learned codec fit.
    norm=_quantile_stats(); action_stats=norm['actions']; state_stats=norm['state']
    probe=np.asarray(rows['actions'][ids[:64]],np.float32); stprobe=np.asarray(rows['state'][ids[:64]],np.float32)
    relprobe=to_eef_relative(probe,stprobe,action_stats,state_stats)
    roundtrip=from_eef_relative(relprobe,stprobe,action_stats,state_stats)
    rel_roundtrip_max=float(np.max(np.abs(roundtrip[:,:,:7]-probe[:,:,:7])))
    if rel_roundtrip_max>3e-5:raise AssertionError(f'EEF transform round-trip error {rel_roundtrip_max}')

    # Fit variants only on training episodes outside the memory and validation split.
    action_all=np.load(args.run/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    state_all=np.load(args.run/'rows/state.npy',mmap_mode='r',allow_pickle=False)
    action_rel=to_eef_relative(np.asarray(action_all),np.asarray(state_all),action_stats,state_stats)
    del action_all
    features=np.asarray(feat,np.float32)
    with (out/'progress.jsonl').open('w') as logf:
        logf.write(json.dumps({'event':'prepared','eval_rows':len(ids),'memory_rows':len(mem_ids),'memory_excluded_fit_rows':len(train_ids),'hard_threshold_source':str(GAP/'summary_corrected.json')})+'\n')
    variants={}; training_log=[]; fit_start=time.monotonic()
    for key,bdim,override in [('fresh_b12r4',12,None),('b24r4',24,None),('b48r4',48,None),('eef_relative_b12r4',12,action_rel)]:
        started=time.monotonic(); codec=_fit_codec(rows,features,train_ids,base_dim=bdim,actions_override=override,tag=key)
        variants[key]=codec
        training_log.append({'variant':key,'base_dim':bdim,'residual_dim':4,'epochs_run':codec.epochs_run,
                             'fit_rows':len(train_ids),'fit_seconds':time.monotonic()-started,'device':'cpu','torch_threads':8})
        import torch
        torch.save({'base_dim':bdim,'residual_dim':4,'stats':{k:torch.tensor(getattr(codec,k)) for k in ['am','ass','sm','ss','fm','fs']},
                    'nets':{k:getattr(codec,k).state_dict() for k in ['encoder','prior_net','base_decoder','residual_decoder']}},out/f'{key}.pt')
    training_log.append({'variant':'ridge_state_residual','fit_rows':len(train_ids),'alpha':1.0,'fit_seconds':None})
    ridge_started=time.monotonic()
    ridge=_ridge_fit(np.asarray(state_all[train_ids]),np.asarray(rows['actions'][train_ids,:,:7]),alpha=1.0)
    training_log[-1]['fit_seconds']=time.monotonic()-ridge_started
    np.savez_compressed(out/'ridge_state_residual.npz',**ridge)
    total_fit_s=time.monotonic()-fit_start

    # Self-reconstruction error on memory rows not used to fit the new variants.
    rec_errors={}
    raw_mem=np.asarray(rows['actions'][mem_ids],np.float32); state_mem=np.asarray(rows['state'][mem_ids],np.float32); feat_mem=features[mem_ids]
    rec_errors['raw_exact']={'mean_mse':0.0,'median_mse':0.0}
    rec_errors['ridge_state_residual_exact']={'mean_mse':0.0,'median_mse':0.0}
    current_module=__import__('run',fromlist=['restore_codec']); current=current_module.restore_codec(args.run)
    bcur,rcur=current.encode(raw_mem,state_mem); rcur_pred=current.decode(bcur,rcur,feat_mem,state_mem)
    cur_err=np.mean((rcur_pred[:,:,:7]-raw_mem[:,:,:7])**2,axis=(1,2))
    rec_errors['current_b12r4_memory_self']={'mean_mse':float(cur_err.mean()),'median_mse':float(np.median(cur_err)),'n':len(cur_err),'fit_included_memory':True}
    rec_errors['fresh_b12r4']={}; rec_errors['b24r4']={}; rec_errors['b48r4']={}; rec_errors['eef_relative_b12r4']={}
    for name,codec in variants.items():
        amem=action_rel[mem_ids] if name=='eef_relative_b12r4' else raw_mem
        b,r=codec.encode(amem,state_mem); own=codec.decode(b,r,feat_mem,state_mem)
        if name=='eef_relative_b12r4':own=from_eef_relative(own,state_mem,action_stats,state_stats)
        e=np.mean((own[:,:,:7]-raw_mem[:,:,:7])**2,axis=(1,2))
        rec_errors[name]={'mean_mse':float(e.mean()),'median_mse':float(np.median(e)),'n':len(e),'fit_included_memory':False}

    # Draft generation for every fixed raw-distance oracle candidate.
    qstate=np.asarray(rows['state'][ids],np.float32); qfeat=features[ids]
    drafts_by_exclusion={}
    new_rep_names=list(variants)+['ridge_state_residual']
    drafts={}
    for exclusion in ('episode','task'):
        cand=candidate_global[exclusion]
        ac=np.asarray(rows['actions'][cand],np.float32); sc=np.asarray(rows['state'][cand],np.float32); fc=features[cand]
        drafts[f'raw_{exclusion}']=ac
        # Reuse the current saved base and its prior residual; this is exactly stage2 D.
        bases=np.load(args.run/'bases.npy',mmap_mode='r',allow_pickle=False)
        pred_r=np.load(args.run/'predicted_residual.npy',mmap_mode='r',allow_pickle=False)
        drafts[f'current_b12r4_{exclusion}']=np.asarray(current.decode(np.asarray(bases[cand]),np.asarray(pred_r[ids]),qfeat,qstate),np.float32)
        for name,codec in variants.items():
            Aall=action_rel if name=='eef_relative_b12r4' else np.asarray(rows['actions'])
            bc,_=codec.encode(np.asarray(Aall[cand],np.float32),sc)
            _,rq=codec.prior(qfeat,qstate)
            rel=codec.decode(bc,rq,qfeat,qstate)
            if name=='eef_relative_b12r4': rel=from_eef_relative(rel,qstate,action_stats,state_stats)
            drafts[f'{name}_{exclusion}']=np.asarray(rel,np.float32)
        # State-conditioned component is a train-only ridge; store residual actions and add query prediction.
        pred_candidate=_ridge_predict(ridge,sc)
        residual=ac[:,:,:7]-pred_candidate
        reconstructed=residual+_ridge_predict(ridge,qstate)
        d=np.zeros_like(ac,np.float32);d[:,:,:7]=reconstructed
        drafts[f'ridge_state_residual_{exclusion}']=d
        for name in [f'{n}_{exclusion}' for n in ['raw','current_b12r4']+new_rep_names]:
            if drafts[name].shape!=(len(ids),10,32) or not np.isfinite(drafts[name]).all():raise ValueError(f'invalid draft {name}: {drafts[name].shape}')
        drafts_by_exclusion[exclusion]=None

    # Load frozen correction expert and evaluate newly fit representations; old current/raw predictions are reused byte-for-byte from stage2.
    infer_inputs={}
    for exclusion in ('episode','task'):
        for rep in new_rep_names:
            for t0 in T0S:
                infer_inputs[f'{rep}_{exclusion}_t0_{t0:.1f}']=(t0,drafts[f'{rep}_{exclusion}'])
    names_new,preds_new,infer_s=_infer_new(args.run,ids,rows,infer_inputs,output=out)
    actions=np.asarray(rows['actions'][ids,:,:7],np.float32)
    all_names=[]; preds=[]; row_errors=[]
    # Reuse stage2 conditions for raw/current with the exact candidate policy and fixed noises.
    old_names=gap['condition_names'].tolist()
    for exclusion in ('episode','task'):
        for rep,old_rep in [('current_b12r4','D'),('raw','Draw')]:
            for t0 in T0S:
                name=f'{rep}_{exclusion}_t0_{t0:.1f}';src=f'{old_rep}_{exclusion}_t0_{t0:.1f}'
                oldidx=old_names.index(src); pred=gap['predictions'][oldidx][:,gap_ix]
                all_names.append(name);preds.append(pred)
    for i,name in enumerate(names_new):all_names.append(name);preds.append(preds_new[i])
    predictions=np.stack(preds).astype(np.float32)
    row_mse=np.mean((predictions-actions[None,None])**2,axis=(-1,-2))
    # Group by the same frozen difficult-row threshold as stage2.
    old_summary=json.load(open(args.gap/'summary_corrected.json'))
    threshold=float(old_summary['threshold']); normal=gap['row_mse'][old_names.index('A_normal')][:,gap_ix].mean(0)
    hard=normal>=threshold
    groups={'all':np.ones(len(ids),bool),'hard':hard}
    metrics={'rows':len(ids),'hard_threshold':threshold,'hard_rows':int(hard.sum()),'groups':{},'paired_deltas':{},
             'inference_seconds':infer_s,'fit_seconds':total_fit_s,'training':training_log,'reconstruction':rec_errors,
             'relative_roundtrip_max_abs_normalized':rel_roundtrip_max,'fallback_checkpoint':best_path_record(args.run),
             'candidate_choice_source':'memory_gap-01 chosen_candidate_rows, row selected by raw action MSE',
             'storage_values_per_item':{'raw':70,'current_b12r4':12,'fresh_b12r4':12,'b24r4':24,'b48r4':48,
                                        'eef_relative_b12r4':12,'ridge_state_residual':70},
             'storage_note':'M1 residual r is predicted from the current query by prior(feature,state), not stored per memory item; capacity counts stored base b only.'}
    for grp,mask in groups.items():
        metrics['groups'][grp]={'rows':int(mask.sum()),'conditions':{}}
        for i,name in enumerate(all_names):
            means=row_mse[i].mean(0);v=means[mask]
            metrics['groups'][grp]['conditions'][name]={'mean_mse':float(v.mean()),'median_mse':float(np.median(v)),
                                                        'rows':int(mask.sum())}
        # Pair all newly fitted representations against current and raw of same exclusion/t0.
        for exclusion in ('episode','task'):
            for t0 in T0S:
                raw_name=f'raw_{exclusion}_t0_{t0:.1f}';cur_name=f'current_b12r4_{exclusion}_t0_{t0:.1f}'
                for rep in new_rep_names:
                    name=f'{rep}_{exclusion}_t0_{t0:.1f}';idx=all_names.index(name)
                    for ref in (raw_name,cur_name):
                        refidx=all_names.index(ref); diff=row_mse[idx].mean(0)-row_mse[refidx].mean(0)
                        key=f'{name} - {ref}'
                        metrics['paired_deltas'].setdefault(grp,{})[key]={'mean':float(diff[mask].mean()),
                            'ci95':_boot_ci(diff,clusters,mask,seed=20261104),'episodes':int(len(np.unique(clusters[mask])))}
    # Save data and artifacts without touching source runs.
    np.savez_compressed(out/'predictions_and_errors.npz',row_ids=ids,task_ids=task,episode_ids=episode,condition_names=np.asarray(all_names),
                        noise_seeds=np.asarray(NOISE_SEEDS),actions=actions,predictions=predictions,row_mse=row_mse,
                        candidate_episode=candidate_global['episode'],candidate_task=candidate_global['task'],
                        draft_names=np.asarray(list(drafts)),drafts=np.stack(list(drafts.values())))
    protocol={'source_run':str(args.run),'stage2':str(args.gap),'eval_rows':ids.tolist(),'candidate_rows':{k:v.tolist() for k,v in candidate_global.items()},
              'excluded_fit_rows':{'validation_or_memory':int(len(rows['actions'])-len(train_ids)),'fit_rows':int(len(train_ids)),
                                   'memory_rows':int(len(mem_ids))},'codec_settings':training_log,
              'noise_seeds':list(NOISE_SEEDS),'t0':list(T0S),'metric':'normalized active7 MSE averaged over 10 steps and 2 fixed noise seeds',
              'xla_preallocate':os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE'),'xla_mem_fraction':os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION'),
              'candidate_source':'raw action-distance oracle; fixed from stage2', 'relative_coordinate_rule':'world delta vectors transformed by initial EEF R.T; inverse query R',
              'ridge_state_residual_alpha':1.0,'hard_threshold':threshold}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({'rows':len(ids),'conditions':len(all_names),'fit_seconds':total_fit_s,'infer_seconds':infer_s,
                      'hard_rows':int(hard.sum()),'relative_roundtrip_max':rel_roundtrip_max},indent=2))


def best_path_record(run):
    return json.loads((run/'best_bridge.json').read_text())

if __name__=='__main__':main()
