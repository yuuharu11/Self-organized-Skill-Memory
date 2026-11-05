#!/usr/bin/env python3
"""Dimension-matched extension of the reviewed compress-vs-raw experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[2]))  # skillnet/examples, for emergent.light methods
from common import (FAIR, MEMORY, ROOT, RUN, T0S, base_module, fair_module,
                    require_new_artifacts)

REP = ROOT / 'representation-4a-01'
REVIEW = ROOT / 'compress-vs-raw-review-20261105-01'
OUT = ROOT / 'compress-vs-raw-01/dimension-extension-01'
DIMS = {'b12m': 12, 'b24': 24, 'b48': 48, 'b96': 96}
SEED = 20261104


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def codec_fit_ids(rows):
    return np.flatnonzero(~np.asarray(rows['validation'], bool) & ~np.asarray(rows['memory'], bool)).astype(np.int64)


def nonlong_holdout_ids(rows):
    return np.flatnonzero(np.asarray(rows['validation'], bool) & (np.asarray(rows['task']) >= 10)).astype(np.int64)


def load_codec(payload, base_dim):
    import torch
    from torch import nn
    from emergent.light.methods.m1 import M1
    m = M1(epochs=30, batch_size=128, seed_value=SEED, base_dim=base_dim, residual_dim=4,
           device='cpu', independence_weight=.01, beta_r=1., base_use_weight=5.,
           base_use_margin=.05, width=128, decoder_obs=False)
    m.action_shape=(10,32); m.a_dim=320; m.state_dim=32; m.feature_dim=2048
    for k,v in payload['stats'].items(): setattr(m,k,np.asarray(v))
    for name, input_dim, output_dim in [
        ('encoder',352,base_dim+8),('prior_net',2080,base_dim+4),
        ('base_decoder',base_dim+32,320),('residual_decoder',4,320)]:
        net=nn.Sequential(nn.Linear(input_dim,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,output_dim))
        net.load_state_dict(payload['nets'][name]); net.eval(); setattr(m,name,net)
    from emergent.light.methods._common import AdditiveActionDecoder
    m.decoder=AdditiveActionDecoder(m.base_decoder,m.residual_decoder,base_dim=base_dim,residual_dim=4,
                                    state_dim=32,feature_dim=2048,decoder_obs=False,residual_uses_context=False)
    m.decoder.eval()
    return m


def codec_payload(model):
    import torch
    return {'base_dim':model.base_dim,'residual_dim':4,
            'stats':{k:torch.tensor(getattr(model,k)) for k in ('am','ass','sm','ss','fm','fs')},
            'nets':{k:getattr(model,k).state_dict() for k in ('encoder','prior_net','base_decoder','residual_decoder')}}


def fit_one(rows, features, fit_ids, dim, *, no_early_stop=False):
    import torch
    from emergent.light.methods.m1 import M1
    torch.set_num_threads(8)
    model=M1(epochs=30,batch_size=128,seed_value=SEED,base_dim=dim,residual_dim=4,device='cpu',
             independence_weight=.01,beta_r=1.,base_use_weight=5.,base_use_margin=.05,width=128,decoder_obs=False)
    task=np.asarray(rows['task'][fit_ids]); epi=np.asarray(rows['episode'][fit_ids])
    train={'actions':np.asarray(rows['actions'][fit_ids],np.float32),'state':np.asarray(rows['state'][fit_ids],np.float32),
           'feature':np.asarray(features[fit_ids],np.float32),'key':np.asarray(features[fit_ids],np.float32),
           'episode_id':np.asarray([f'{t}:{e}' for t,e in zip(task,epi,strict=True)]),
           'init_state_id':np.asarray([f'{t}:{e}' for t,e in zip(task,epi,strict=True)]),'seed':SEED}
    if no_early_stop: train['early_stopping_patience']=1000
    model.fit(train)
    return model


def own_reconstruction(codec, rows, features, ids, batch=512):
    actions=np.asarray(rows['actions'][ids],np.float32); states=np.asarray(rows['state'][ids],np.float32); feat=np.asarray(features[ids],np.float32)
    b,r=codec.encode(actions,states); pred=np.empty_like(actions)
    for lo in range(0,len(ids),batch):
        sl=slice(lo,min(lo+batch,len(ids)))
        pred[sl]=codec.decode(b[sl],r[sl],feat[sl],states[sl])
    return np.mean((pred[:,:,:7]-actions[:,:,:7])**2,axis=(1,2)).astype(np.float32)


def encode_candidate_b(codec, rows, actions, candidate_ids):
    ids=np.asarray(candidate_ids,np.int64)
    unique, inverse=np.unique(ids.reshape(-1),return_inverse=True)
    if unique.size and (unique.min()<0 or unique.max()>=len(actions)): raise IndexError('candidate row ID out of range')
    b,_=codec.encode(np.asarray(actions[unique],np.float32),np.asarray(rows['state'][unique],np.float32))
    return np.asarray(b[inverse].reshape(*ids.shape,-1),np.float32)


def decode_for_queries(codec, rows, features, actions, query_ids, candidate_ids, batch=128):
    q=np.asarray(query_ids,np.int64); c=np.asarray(candidate_ids,np.int64)
    if c.ndim==1: c=c[:,None]
    if c.shape[0]!=len(q): raise ValueError('query/candidate rows are misaligned')
    b=encode_candidate_b(codec,rows,actions,c)
    out=np.empty((len(q),c.shape[1],10,32),np.float32)
    for lo in range(0,len(q),batch):
        sl=slice(lo,min(lo+batch,len(q))); qpart=q[sl]
        _,r=codec.prior(np.asarray(features[qpart],np.float32),np.asarray(rows['state'][qpart],np.float32))
        rr=np.repeat(r,c.shape[1],axis=0)
        ff=np.repeat(np.asarray(features[qpart],np.float32),c.shape[1],axis=0)
        ss=np.repeat(np.asarray(rows['state'][qpart],np.float32),c.shape[1],axis=0)
        pred=codec.decode(b[sl].reshape(-1,b.shape[-1]),rr,ff,ss)
        out[sl]=np.asarray(pred,np.float32).reshape(len(qpart),c.shape[1],10,32)
    return out


def _check_source(rows):
    prep=np.load(REVIEW/'drafts_prepared.npz',allow_pickle=False)
    exp=np.load(REVIEW/'predictions_and_errors.npz',allow_pickle=False)
    if not np.array_equal(prep['eval_query_rows'],exp['row_ids']) or not np.array_equal(prep['eval_target'],exp['target']):
        raise AssertionError('reviewed base rows/targets are inconsistent')
    long=prep['eval_query_rows']
    if len(long)!=548 or np.any(np.asarray(rows['task'][long])>=10) or int(prep['hard_mask'].sum())!=63:
        raise AssertionError('fixed Long/hard split changed')
    m2=prep['m2_candidate_rows']; sh=prep['shuffle_candidate_rows']; oracle=prep['oracle_candidate_rows']
    if not np.array_equal(np.sort(sh),np.sort(m2)) or np.any(np.asarray(rows['task'][sh])==np.asarray(rows['task'][long])):
        raise AssertionError('reviewed shuffle is not cross-task Top-1 permutation')
    pool=np.asarray(np.load(MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'],np.int64)
    if not np.all(np.isin(np.r_[m2,sh,oracle],pool)): raise AssertionError('candidate outside fixed noLong memory')
    return prep,exp


def fit_codecs(out=OUT):
    require_new_artifacts((out/'codec_metrics.json',out/'prepared.json',out/'drafts_b12m.npz',out/'drafts_b24.npz',
                          out/'drafts_b48.npz',out/'drafts_b96.npz',out/'codecs/b12m.pt',out/'codecs/b24.pt',
                          out/'codecs/b48.pt',out/'codecs/b96.pt'))
    import torch
    torch.set_num_threads(8)
    base=base_module(); rows=base.load_rows(RUN)
    fit=codec_fit_ids(rows); hold=nonlong_holdout_ids(rows)
    if len(fit)!=16721 or len(hold)==0 or not np.any(np.asarray(rows['task'][fit])<10): raise AssertionError('codec fit split changed or Long train rows missing')
    features=np.load(RUN/'pooled.npy',mmap_mode='r',allow_pickle=False)
    actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    source,review_preds=_check_source(rows)
    train_meta=json.loads((FAIR/'E-M2/trained.json').read_text())
    batch_ids=np.load(FAIR/'batch_row_ids.npy',allow_pickle=False)
    if hashlib.sha256(batch_ids.tobytes()).hexdigest()!=train_meta['batch_row_ids_sha256']:
        raise AssertionError('shared Expert batch order changed')
    em2=np.load(FAIR/'E-M2/training_drafts.npz',allow_pickle=False)
    train_q=np.asarray(em2['query_rows'],np.int64); train_c=np.asarray(em2['selected_candidate_rows'],np.int64)
    fair=fair_module(); pool=np.asarray(np.load(MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'],np.int64)
    fair.validate_candidates(train_c[:,None],train_q,rows,pool)
    dev_q=fair.validation_rows(rows); lists=np.load(FAIR/'candidate_lists.npz',allow_pickle=False)
    mapq={int(r):i for i,r in enumerate(lists['query_rows'])}
    dev_lists=np.stack([lists['candidate_rows'][mapq[int(r)]] for r in dev_q])
    dev_c,_=fair.sample_m2_training_draft(dev_lists,np.random.default_rng(fair.SEED+1902))
    hold_mse={}; dim_meta={}; chosen_models={}
    # Exact matched codec artifacts from representation-4a are reused when their protocol agrees.
    repmeta=json.load(open(REP/'protocol.json'))
    settings={x['variant']:x for x in repmeta['codec_settings']}
    for name,dim in DIMS.items():
        t0=time.monotonic(); variants=[]
        if name in ('b12m','b24','b48'):
            key={'b12m':'fresh_b12r4','b24':'b24r4','b48':'b48r4'}[name]
            sm=settings[key]
            if sm['fit_rows']!=len(fit) or sm['base_dim']!=dim or sm['residual_dim']!=4:
                raise AssertionError(f'{key} is not fit on matched rows/config')
            path=REP/f'{key}.pt'; payload=torch.load(path,map_location='cpu',weights_only=False)
            codec=load_codec(payload,dim); variants.append((codec,int(sm['epochs_run']),str(path),'reused'))
        else:
            codec=fit_one(rows,features,fit,dim,no_early_stop=False)
            variants.append((codec,int(codec.epochs_run),'new-early','fit'))
        # If a fit stopped early, compare a full-epoch counterpart on held-out
        # nonLong validation demonstrations; this selection never inspects Long Eval.
        if name=='b48' or (variants[0][1]<30):
            full=fit_one(rows,features,fit,dim,no_early_stop=True)
            variants.append((full,int(full.epochs_run),'new-no-early-stop','fit'))
        variant_rows=[]
        for idx,(codec,epochs,label,origin) in enumerate(variants):
            val=own_reconstruction(codec,rows,features,hold)
            mem=own_reconstruction(codec,rows,features,memory_ids)
            variant_rows.append({'codec':codec,'epochs':epochs,'label':label,'origin':origin,'hold_mse':val,'memory_mse':mem})
        selected=min(variant_rows,key=lambda x:float(x['hold_mse'].mean()))
        chosen_models[name]=selected['codec']
        # Save selected full payload under new output; reused source artifacts remain read-only.
        ddir=out/'codecs'; ddir.mkdir(parents=True,exist_ok=True)
        torch.save(codec_payload(selected['codec']),ddir/f'{name}.pt')
        dim_meta[name]={'base_dim':dim,'residual_dim':4,'fit_rows':int(len(fit)),
            'fit_split':'~validation & ~memory; Long included','selected_epochs':selected['epochs'],
            'selected_origin':selected['origin'],'selection_rule':'lowest self-reconstruction active7 MSE on validation & task>=10 rows',
            'holdout_rows':int(len(hold)),'selected_holdout_mean_mse':float(selected['hold_mse'].mean()),
            'selected_memory_self_mean_mse':float(selected['memory_mse'].mean()),
            'candidates':[{'epochs':x['epochs'],'label':x['label'],'origin':x['origin'],
                           'holdout_mean_mse':float(x['hold_mse'].mean()),'holdout_median_mse':float(np.median(x['hold_mse'])),
                           'memory_self_mean_mse':float(x['memory_mse'].mean())} for x in variant_rows],
            'elapsed_s':time.monotonic()-t0}
        # Materialize train, fixed Dev, and fixed Long drafts using each
        # codec's encode(candidate action) and query prior residual.
        tr=decode_for_queries(codec,rows,features,actions,train_q,train_c)
        dv=decode_for_queries(codec,rows,features,actions,dev_q,dev_c)
        eval_ids=np.asarray(source['eval_query_rows'],np.int64)
        candidate_ids=np.stack([source['m2_candidate_rows'],source['oracle_candidate_rows'],source['shuffle_candidate_rows']],axis=1)
        ev=decode_for_queries(codec,rows,features,actions,eval_ids,candidate_ids)
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/f'drafts_{name}.npz',train_query_rows=train_q,train_candidate_rows=train_c,
            train_drafts=tr[:,0],dev_query_rows=dev_q,dev_candidate_rows=dev_c,dev_drafts=dv[:,0],
            eval_query_rows=eval_ids,eval_candidate_rows=candidate_ids,eval_decoded_drafts=ev,
            eval_target=source['eval_target'],eval_noise=source['eval_noise'],hard_mask=source['hard_mask'],
            batch_row_ids_sha256=np.asarray(hashlib.sha256(batch_ids.tobytes()).hexdigest()))
        print(json.dumps({'stage':'codec','dimension':name,'meta':dim_meta[name]},ensure_ascii=False),flush=True)
    write_json(out/'codec_metrics.json',{'codec_fit_rows':len(fit),'nonlong_holdout_rows':len(hold),
        'fit_split':'~validation & ~memory (includes Long tasks; no Long Eval used for selection)',
        'dimensions':dim_meta,'validation_definition':'rows.validation & task>=10',
        'reconstruction_metric':'M1 encode true action/state, then decode encoded base+residual; normalized active7 MSE'})
    write_json(out/'prepared.json',{'train_rows':len(train_q),'dev_rows':len(dev_q),'long_rows':len(source['eval_query_rows']),
        'long_hard_rows':int(source['hard_mask'].sum()),'candidate_source':'reviewed compress-vs-raw Top-1, oracle, exact Top-1 shuffle',
        'candidate_identity':'dataset row IDs','train_selected_candidate_rows_exactly_E-M2':True,
        'batch_row_ids_sha256':hashlib.sha256(batch_ids.tobytes()).hexdigest()})


import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
def train_one(name, out=OUT, *, threads=8):
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex,module_jit
    fair=fair_module(); base=base_module(); rows=base.load_rows(RUN)
    if name not in DIMS: raise ValueError(name)
    prep=np.load(out/f'drafts_{name}.npz',allow_pickle=False)
    trainq=np.asarray(prep['train_query_rows'],np.int64); trainc=np.asarray(prep['train_candidate_rows'],np.int64)
    em2=np.load(FAIR/'E-M2/training_drafts.npz',allow_pickle=False)
    if not np.array_equal(trainq,em2['query_rows']) or not np.array_equal(trainc,em2['selected_candidate_rows']):
        raise AssertionError('dimension Expert candidate rows differ from E-M2')
    batch_ids=np.load(FAIR/'batch_row_ids.npy',allow_pickle=False)
    if str(prep['batch_row_ids_sha256'].item())!=hashlib.sha256(batch_ids.tobytes()).hexdigest():
        raise AssertionError('dimension Expert batch hash mismatch')
    pos={int(r):i for i,r in enumerate(trainq)}
    devq=np.asarray(prep['dev_query_rows'],np.int64); devdraft=np.asarray(prep['dev_drafts'],np.float32)
    if np.any(np.asarray(rows['task'][trainq])<10): raise AssertionError('Long rows entered Expert training')
    if np.any(np.asarray(rows['task'][devq])<10): raise AssertionError('Long entered Expert Dev')
    model,_=base.load_policy(_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected=nnx.state(model,filt)
    if sum(x.value.size for x in selected.flat_state().values())<100_000_000: raise ValueError('trainable filter mismatch')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt=tx.init(selected); stepfn=base.make_bridge_step(filt,tx); key=jax.random.key(SEED)
    exp=out/f'E-{name}'
    if exp.exists(): raise FileExistsError(exp)
    (exp/'bridge').mkdir(parents=True)
    val_target=np.asarray(rows['actions'][devq],np.float32)[:,:,:7]
    val_noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(88261),int(r)),(10,32))) for r in devq])
    best=float('inf'); logs=[]; start=time.monotonic()
    for step,ids in enumerate(batch_ids,start=1):
        draft=np.asarray(prep['train_drafts'][[pos[int(r)] for r in ids]],np.float32)
        key,sub=jax.random.split(key)
        loss,opt=stepfn(model,opt,base.obs_batch(rows,ids),jnp.asarray(rows['actions'][ids]),jnp.asarray(draft),sub)
        loss=float(loss)
        if not np.isfinite(loss): raise FloatingPointError((name,step,loss))
        if step%500==0:
            ckpt=exp/'bridge'/f'{step:06d}'
            ocp.PyTreeCheckpointer().save(ckpt,nnx.state(model,filt).to_pure_dict())
            bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',)); valobs=base.obs_batch(rows,devq); vals={}
            for t0 in T0S:
                preds=[]
                for lo in range(0,len(devq),8):
                    sl=slice(lo,min(lo+8,len(devq)))
                    p=bridge(jax.random.key(0),jax.tree.map(lambda x:x[sl],valobs),jnp.asarray(devdraft[sl]),
                        jnp.full((len(devq[sl]),),t0),num_steps=10,noise=jnp.asarray(val_noise[sl]))
                    preds.append(np.asarray(p)[:,:,:7])
                vals[f'{t0:.1f}']=float(np.mean((np.concatenate(preds)-val_target)**2))
            chosen=min(vals,key=vals.get); vmse=vals[chosen]
            logs.append({'step':step,'loss':loss,'dev_mse':vals,'t0':float(chosen),'elapsed_s':time.monotonic()-start})
            if vmse<best:
                best=vmse; fair._write_json(exp/'best_bridge.json',{'path':str(ckpt),'steps':step,'t0':float(chosen),
                    'validation_generated_active7_mse':vmse,'selection':'same fixed E-M2 Dev candidates/noise; 6 heldout tasks; no Long'})
            fair._write_json(exp/'training_progress.json',{'status':'running','steps_done':step,'logs':logs,'best_mse':best})
            print(json.dumps({'stage':'train','dimension':name,'step':step,'loss':loss,'dev':vals,'elapsed_s':time.monotonic()-start}),flush=True)
    fair._write_json(exp/'trained.json',{'expert':f'E-{name}','steps':4000,'seed':SEED,'batch_size':8,
        'train_rows':len(trainq),'batch_row_ids_sha256':hashlib.sha256(batch_ids.tobytes()).hexdigest(),
        'best':json.loads((exp/'best_bridge.json').read_text()),'loss':'same fair-expert MSE velocity objective',
        'optimizer':'AdamW lr2.5e-5 b1.9 b2.95 wd1e-5 clip1','seconds':time.monotonic()-start,'logs':logs})
    fair._write_json(exp/'training_progress.json',{'status':'complete','steps_done':4000,'logs':logs,'best_mse':best})


def evaluate_one(name,out=OUT,*,batch=8,threads=8):
    require_new_artifacts((out/f'predictions_{name}.npz',out/f'eval_{name}.json'))
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fair=fair_module(); base=base_module(); rows=base.load_rows(RUN)
    prep=np.load(out/f'drafts_{name}.npz',allow_pickle=False)
    if not (out/f'E-{name}/best_bridge.json').exists(): raise FileNotFoundError(f'need train first: {name}')
    q=np.asarray(prep['eval_query_rows'],np.int64); target=np.asarray(prep['eval_target'],np.float32); noise=np.asarray(prep['eval_noise'],np.float32)
    hard=np.asarray(prep['hard_mask'],bool); episode=fair._base_run().np if False else (np.asarray(rows['task'][q],np.int64)*100000+np.asarray(rows['episode'][q],np.int64))
    if len(q)!=548 or hard.sum()!=63: raise AssertionError('Long/hard eval split changed')
    spec=json.loads((out/f'E-{name}/best_bridge.json').read_text())
    policy,_=base.load_policy(_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
    fair._skill_run().load_expert(policy,Path(spec['path']))
    fn=module_jit(policy.sample_bridge,static_argnames=('num_steps',)); obs=base.obs_batch(rows,q)
    drafts=np.asarray(prep['eval_decoded_drafts'],np.float32); kinds=('m2','oracle','shuffle')
    pred={}; mse={}
    for ki,k in enumerate(kinds):
        for t0 in T0S:
            pp=[]
            for lo in range(0,len(q),batch):
                sl=slice(lo,min(lo+batch,len(q)))
                y=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(drafts[sl,ki]),
                    jnp.full((len(q[sl]),),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
                pp.append(np.asarray(y)[:,:,:7])
            pred[f'{k}_t0{t0:.1f}']=np.concatenate(pp).astype(np.float32)
    zero=np.zeros((len(q),10,32),np.float32)
    for t0 in T0S:
        pp=[]
        for lo in range(0,len(q),batch):
            sl=slice(lo,min(lo+batch,len(q)))
            y=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(zero[sl]),jnp.full((len(q[sl]),),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
            pp.append(np.asarray(y)[:,:,:7])
        pred[f'zero_t0{t0:.1f}']=np.concatenate(pp).astype(np.float32)
    for k,v in pred.items(): mse[k]=np.mean((v-target)**2,axis=(1,2)).astype(np.float32)
    # Episode bootstrap contrasts against fixed, reviewed E-M2 raw predictions.
    ref=np.load(REVIEW/'predictions_and_errors.npz',allow_pickle=False)
    if not np.array_equal(ref['row_ids'],q) or not np.array_equal(ref['target'],target): raise AssertionError('review baseline alignment mismatch')
    raw_m2=np.asarray(ref['mse_E-M2_m2_t0_0.6'],np.float32)
    raw_oracle=np.asarray(ref['mse_E-M2_oracle_t0_0.6'],np.float32)
    raw_t0=.6; chosen_t0=float(spec['t0'])
    diffs={}
    for sub,mask in [('all',np.ones(len(q),bool)),('hard',hard)]:
        diffs[f'C1_{sub}']=fair._bootstrap_diff(raw_m2[mask],mse[f'm2_t0{chosen_t0:.1f}'][mask],episode[mask],reps=5000,seed=20261105)
        diffs[f'C2_{sub}']=fair._bootstrap_diff(raw_oracle[mask],mse[f'oracle_t0{chosen_t0:.1f}'][mask],episode[mask],reps=5000,seed=20261105)
        diffs[f'C3_{sub}']=fair._bootstrap_diff(mse[f'm2_t0{chosen_t0:.1f}'][mask],mse[f'shuffle_t0{chosen_t0:.1f}'][mask],episode[mask],reps=5000,seed=20261105)
    np.savez_compressed(out/f'predictions_{name}.npz',row_ids=q,target=target,hard_mask=hard,episode_cluster=episode,
        decoded_drafts=drafts,**{f'pred_{k}':v for k,v in pred.items()},**{f'mse_{k}':v for k,v in mse.items()})
    write_json(out/f'eval_{name}.json',{'dimension':name,'selected':spec,'metrics':{k:{'all_mean':float(v.mean()),'hard_mean':float(v[hard].mean())} for k,v in mse.items()},
        'paired_bootstrap':diffs,'noise':'same reviewed fixed noise bank','rows':len(q),'hard_rows':int(hard.sum())})
    print(json.dumps({'stage':'evaluate','dimension':name,'selected_t0':chosen_t0,'m2_all':float(mse[f'm2_t0{chosen_t0:.1f}'].mean())},flush=True))


def _summ(x, mask=None):
    a=np.asarray(x,np.float64); a=a if mask is None else a[np.asarray(mask,bool)]
    return {'mean':float(a.mean()),'median':float(np.median(a)),'n':int(len(a))}


def _direction(ci):
    if ci['ci95'][1]<0:return 'raw_better'
    if ci['ci95'][0]>0:return 'compressed_better'
    return 'no_detected_difference'


def _cluster_signflip_p(diff,clusters,mask,reps=5000,seed=20261106):
    d=np.asarray(diff,np.float64)[np.asarray(mask,bool)]; c=np.asarray(clusters)[np.asarray(mask,bool)]
    uniq=np.unique(c); sums=np.asarray([d[c==u].sum() for u in uniq],np.float64); n=len(d)
    obs=float(d.mean()); rng=np.random.default_rng(seed); count=0
    for start in range(0,reps,250):
        k=min(250,reps-start); signs=rng.choice(np.asarray([-1.,1.]),size=(k,len(uniq)))
        stats=(signs@sums)/n; count+=int(np.sum(np.abs(stats)>=abs(obs)))
    return float((count+1)/(reps+1))


def _holm(pvalues):
    keys=list(pvalues); order=sorted(keys,key=lambda k:pvalues[k]); adjusted={}; running=0.
    m=len(order)
    for rank,key in enumerate(order):
        running=max(running,min(1.,(m-rank)*float(pvalues[key])))
        adjusted[key]=running
    return adjusted


def aggregate(out=OUT):
    require_new_artifacts((out/'summary.json',out/'decision.json',out/'predictions_and_errors.npz'))
    fair=fair_module()
    rows=base_module().load_rows(RUN)
    review=np.load(REVIEW/'predictions_and_errors.npz',allow_pickle=False)
    base_drafts=np.load(REVIEW/'drafts_prepared.npz',allow_pickle=False)
    q=np.asarray(review['row_ids'],np.int64); target=np.asarray(review['target'],np.float32); hard=np.asarray(review['hard_mask'],bool)
    cluster=np.asarray(review['episode_cluster'],np.int64)
    if len(q)!=548 or hard.sum()!=63: raise AssertionError('wrong fixed Long/hard rows')
    names=['E-M2','E-b12']+[f'E-{n}' for n in DIMS]
    all_mse={}; all_pred={}; chosen={}; all_drafts={}
    # Reviewed E-M2 and E-b12 are immutable baseline controls; their corrected
    # Top-1 shuffle output is read from review-20261105-01.
    for expert in ('E-M2','E-b12'):
        for key in ('m2','oracle','shuffle','zero'):
            for t0 in T0S:
                label=f'{expert}/{key}/t0={t0:.1f}'; source_key=f'mse_{expert}_{key}_t0_{t0:.1f}'
                pred_key=f'pred_{expert}_{key}_t0_{t0:.1f}'
                all_mse[label]=np.asarray(review[source_key],np.float32); all_pred[label]=np.asarray(review[pred_key],np.float32)
        if expert=='E-M2': chosen[expert]={'steps':3000,'t0':.6,'source':'fair-expert-01, reproduced in reviewed output'}
        else:
            src=ROOT/'compress-vs-raw-review-20261105-01/E-b12/best_bridge.json'
            chosen[expert]=json.loads(src.read_text())
        all_drafts[expert]={'raw':np.asarray(base_drafts['eval_raw_drafts'],np.float32),
                            'decoded':np.asarray(base_drafts['eval_decoded_drafts'],np.float32)}
    # Official and E-zero are non-dimensional context controls.
    for key in ('official_normal',):
        all_mse[key]=np.asarray(review['mse_'+key],np.float32); all_pred[key]=np.asarray(review['pred_'+key],np.float32)
    for t0 in T0S:
        key=f'E-zero/zero/t0={t0:.1f}'; suffix=f'E-zero_zero_t0_{t0:.1f}'
        all_mse[key]=np.asarray(review[f'mse_{suffix}'],np.float32); all_pred[key]=np.asarray(review[f'pred_{suffix}'],np.float32)
    dimsmeta=json.load(open(out/'codec_metrics.json'))['dimensions']
    dimension_decisions={}; differences={}; raw_top=np.asarray(review['raw_drafts'][:,0],np.float32); raw_oracle=np.asarray(review['raw_drafts'][:,1],np.float32)
    for name in DIMS:
        ev=json.load(open(out/f'eval_{name}.json')); spec=ev['selected']; chosen[f'E-{name}']=spec
        arr=np.load(out/f'predictions_{name}.npz',allow_pickle=False)
        for key in ('m2','oracle','shuffle','zero'):
            for t0 in T0S:
                label=f'E-{name}/{key}/t0={t0:.1f}'
                all_mse[label]=np.asarray(arr[f'mse_{key}_t0{t0:.1f}'],np.float32)
                all_pred[label]=np.asarray(arr[f'pred_{key}_t0{t0:.1f}'],np.float32)
        dec=np.asarray(arr['decoded_drafts'],np.float32)
        all_drafts[f'E-{name}']={'decoded':dec}
        t=float(spec['t0']); c1=all_mse[f'E-M2/m2/t0=0.6']-all_mse[f'E-{name}/m2/t0={t:.1f}']
        c2=all_mse[f'E-M2/oracle/t0=0.6']-all_mse[f'E-{name}/oracle/t0={t:.1f}']
        c3=all_mse[f'E-{name}/m2/t0={t:.1f}']-all_mse[f'E-{name}/shuffle/t0={t:.1f}']
        contrasts={'C1':c1,'C2':c2,'C3':c3}; differences[name]={}
        for ckey,d in contrasts.items():
            for subset,mask in [('all',np.ones(len(q),bool)),('hard',hard)]:
                ci=fair._bootstrap_diff(np.asarray(d[mask]),np.zeros(int(mask.sum()),np.float32),cluster[mask],reps=5000,seed=20261105)
                # fair helper returns paired CI over `a-b`; using d vs zero retains d's sign.
                p=_cluster_signflip_p(d,cluster,mask,reps=5000,seed=20261106)
                differences[name][f'{ckey}_{subset}']={'mean':float(d[mask].mean()),'ci95':ci['ci95'],'p_signflip':p}
        # generation-before-error for the same retrieved/oracle row values
        raw_actions=np.asarray(base_drafts['eval_raw_drafts'][:,:,:,:7],np.float32)
        tar=np.asarray(base_drafts['eval_target'],np.float32)
        dec7=dec[:,:,:,:7]
        raw_mse=np.mean((raw_actions[:,0]-tar)**2,axis=(1,2)); oracle_raw=np.mean((raw_actions[:,1]-tar)**2,axis=(1,2))
        decoded_mse=np.mean((dec7[:,0]-tar)**2,axis=(1,2)); oracle_dec=np.mean((dec7[:,1]-tar)**2,axis=(1,2))
        compression=np.mean((dec7[:,:2]-raw_actions[:,:2])**2,axis=(2,3))
        sensitivity={}
        for t0 in T0S:
            out_dist=np.sqrt(np.mean((all_pred[f'E-{name}/m2/t0={t0:.1f}']-all_pred[f'E-{name}/shuffle/t0={t0:.1f}'])**2,axis=(1,2)))
            draft_dist=np.sqrt(np.mean((dec7[:,0]-dec7[:,2])**2,axis=(1,2)))
            sensitivity[str(t0)]={'mean_output_distance':float(out_dist.mean()),'mean_draft_distance':float(draft_dist.mean()),
                                  'mean_ratio':float(np.mean(out_dist/np.maximum(draft_dist,1e-8)))}
        dimension_decisions[name]={'selected':spec,'self_reconstruction':dimsmeta[name],
            'pre_generation':{'m2_raw_mse':_summ(raw_mse),'m2_decoded_mse':_summ(decoded_mse),
                              'oracle_raw_mse':_summ(oracle_raw),'oracle_decoded_mse':_summ(oracle_dec),
                              'compression_delta_mse':{'m2':_summ(np.mean((dec7[:,0]-raw_actions[:,0])**2,axis=(1,2))),
                                                       'oracle':_summ(np.mean((dec7[:,1]-raw_actions[:,1])**2,axis=(1,2)))},
                              'raw_vs_decoded_value_mse':{'m2':float(compression[:,0].mean()),'oracle':float(compression[:,1].mean())}},
            'output_sensitivity':sensitivity,'contrasts':differences[name]}
    # Holm correction is applied separately to the four dimensions within each
    # contrast and subset, and exposed alongside unadjusted episode-bootstrap CIs.
    for ckey in ('C1','C2','C3'):
        for subset in ('all','hard'):
            p={n:differences[n][f'{ckey}_{subset}']['p_signflip'] for n in DIMS}
            adj=_holm(p)
            for n in DIMS:
                differences[n][f'{ckey}_{subset}']['p_holm_4dims']=adj[n]
                differences[n][f'{ckey}_{subset}']['holm_significant_05']=bool(adj[n]<.05)
                differences[n][f'{ckey}_{subset}']['direction_unadjusted_ci']=_direction(differences[n][f'{ckey}_{subset}'])
                dimension_decisions[n]['contrasts'][f'{ckey}_{subset}']=differences[n][f'{ckey}_{subset}']
    metrics={k:{'all':_summ(v),'hard':_summ(v,hard)} for k,v in all_mse.items()}
    baseline_old=json.load(open(REVIEW/'summary.json'))
    stor={name:{'values_per_segment':DIMS[name],'float32_bytes_3857':3857*DIMS[name]*4} for name in DIMS}
    stor.update({'raw_active7':{'values_per_segment':70,'float32_bytes_3857':3857*70*4},
                 'raw_stored_10x32':{'values_per_segment':320,'float32_bytes_3857':3857*320*4},
                 'b12_run01':{'values_per_segment':12,'float32_bytes_3857':3857*12*4},
                 'tokens_npy_bytes':(RUN/'tokens.npy').stat().st_size})
    result={'rows':548,'hard_rows':63,'selected':chosen,'metrics':metrics,'dimension_results':dimension_decisions,
        'baseline_review_summary':baseline_old,'storage':stor,'holm_method':'episode-cluster sign-flip permutation p-values, 5,000 draws; Holm within 4 dimensions for each C and subset',
        'ci_method':'episode-cluster bootstrap 5,000 draws; unadjusted 95% percentile intervals',
        'limitations':['one Expert training seed per dimension','all M1 codecs fit rows ~validation & ~memory and therefore include Long task training rows','offline known-task Long action MSE, not closed-loop success','run-01 b12 control codec itself fit on all task rows']}
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions_and_errors.npz',row_ids=q,target=target,hard_mask=hard,episode_cluster=cluster,
        **{f'mse_{k.replace("/","_").replace("=","_")}':v for k,v in all_mse.items()},
        **{f'pred_{k.replace("/","_").replace("=","_")}':v for k,v in all_pred.items()})
    write_json(out/'summary.json',result)
    decision={n:{'C1':{s:differences[n][f'C1_{s}']['direction_unadjusted_ci'],'holm_significant':differences[n][f'C1_{s}']['holm_significant_05']}
                    for s in ('all','hard') } for n in DIMS}
    for n in DIMS:
        decision[n]['C2']={s:{'direction':differences[n][f'C2_{s}']['direction_unadjusted_ci'],'holm_significant':differences[n][f'C2_{s}']['holm_significant_05']} for s in ('all','hard')}
        decision[n]['C3']={s:{'direction':differences[n][f'C3_{s}']['direction_unadjusted_ci'],'holm_significant':differences[n][f'C3_{s}']['holm_significant_05']} for s in ('all','hard')}
        decision[n]['selected']=chosen[f'E-{n}']
    write_json(out/'decision.json',decision)
    print(json.dumps({'stage':'aggregate','dimensions':list(DIMS),'decision':decision},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['codecs','train','evaluate','aggregate']); p.add_argument('--dim',choices=list(DIMS)); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--threads',type=int,default=8)
    a=p.parse_args(); out=a.output.resolve()
    if not 1<=a.threads<=16: raise ValueError('threads must be 1..16')
    if a.stage=='codecs': fit_codecs(out)
    elif a.stage=='train':
        if not a.dim: raise ValueError('--dim required')
        train_one(a.dim,out,threads=a.threads)
    elif a.stage=='evaluate':
        if not a.dim: raise ValueError('--dim required')
        evaluate_one(a.dim,out,threads=a.threads)
    else: aggregate(out)

if __name__=='__main__': main()
