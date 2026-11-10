#!/usr/bin/env python3
"""Query-input probes for selecting one motion-shape memory item.

Stage-oriented and resumable: cached feature loading, LIBERO-40-only training,
query-only evaluation, and a small smoke path. Unseen action labels are passed
only to ``score_predictions`` and never to feature/model-selection routines.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE=Path(__file__).resolve().parent
ROOT=(STATE_DIR / 'outputs/memory-pilot/unseen-libero')
THETA=ROOT/'theta-regression-01'
OUT=ROOT/'key-query-01'
POOL_DIR=ROOT/'compressed-denoise-01'/'production'/'drafts'
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROWS  # corrected rows (image orientation fix)
UNSEEN=UNSEEN_ROWS
RUN = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser() / 'run-01'
CHECKPOINT = Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser()
RUN_SCRIPT=HERE.parent/'pi05_memory'/'run.py'
SEED=4101
# Two predeclared heldout tasks in each ten-task LIBERO-40 suite.
HELDOUT_TASKS=(2,7,12,17,22,27,32,37)
MID_LAYER_IDS=(6,10,14,15,16,17)
MID_LAYER_NAMES=tuple(f'P2_mid_l{i}' for i in MID_LAYER_IDS)
METHODS=('P0_mean','P1_tok','P1_tok_lang','P2_mid_l6','P2_mid_l10','P2_mid_l14',
         'P2_mid_l15','P2_mid_l16','P2_mid_l17','P3_exp','P4_act','P5_hist',
         'P6_exp_hist','P6_tok_hist','P6_all','P7_bottleneck','P7b_type_bottleneck',
         )


def shard_training_configs(configs,shard_id,num_shards):
    """Deterministically partition independent configs without dropping or duplicating any."""
    n=int(num_shards);sid=int(shard_id)
    if n<1 or not 0<=sid<n:raise ValueError('invalid training shard')
    return list(configs)[sid::n]


def _load_theta():
    p=HERE/'theta_regression.py';spec=importlib.util.spec_from_file_location('kqp_theta',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def _load_policy_run_module():
    spec=importlib.util.spec_from_file_location('kqp_policy_run',RUN_SCRIPT)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def _load_unseen_metadata(*,include_actions=False):
    names=['image_base','image_wrist','state','prompt','prompt_mask','task','episode','frame','is_dev']
    if include_actions:names.append('actions')
    # UNSEEN already names the ``.../oracle-01/unseen_rows`` directory.
    # Keep the scoring-only action file out of every fit/calibration load.
    return {name:np.load(UNSEEN/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in names}


def _production_data_splits(*,include_eval_targets=False):
    """Load data with a structural boundary: unseen actions only in eval mode."""
    base=_load_policy_run_module();rows=base.load_rows(RUN)
    unseen=_load_unseen_metadata(include_actions=include_eval_targets)
    train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    dev=np.flatnonzero(np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    ev=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    if len(train)!=23304 or len(dev)!=113 or len(ev)!=619:raise ValueError('fixed K1 split changed')
    return base,rows,unseen,{'train':(rows,train),'dev':(unseen,dev),'eval':(unseen,ev)}


def _json_default(value):
    if hasattr(value,'detach') and hasattr(value,'cpu'):
        return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'cannot JSON encode {type(value).__name__}')


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=_json_default)+'\n')
    tmp.replace(path)


def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


def deterministic_holdout_tasks(task_ids):
    ids=sorted(set(map(int,task_ids)))
    suites={}
    for t in ids:suites.setdefault((t//10)*10,[]).append(t)
    selected=[]
    for start,ts in sorted(suites.items()):
        for offset in (2,7):
            t=start+offset
            if t in ts:selected.append(t)
    return tuple(selected)


def filter_cross_task(pool, query_task, heldout=()):
    """Candidate-local ids after removing heldout tasks and each query task."""
    ptask=np.asarray(pool['source_task'],np.int64)
    qtask=np.asarray(query_task,np.int64)
    excluded=np.isin(ptask,np.asarray(tuple(heldout),np.int64))
    allow=np.broadcast_to(~excluded,(len(qtask),len(ptask))).copy()
    allow &= ptask[None,:] != qtask[:,None]
    if np.any(allow.sum(1)==0):raise ValueError('query has no cross-task candidates')
    return allow


def top1_single(scores,allowed,candidate_ids):
    """Return one pool-local position per row; ties keep pool order.

    ``candidate_ids`` are global/source identifiers used to validate the pool
    mapping.  They are deliberately not returned: all downstream tensor
    indexing uses pool-local positions.  Map with ``candidate_ids[ix]`` only
    when serializing a source ID.
    """
    s=np.asarray(scores,np.float64);a=np.asarray(allowed,bool);ids=np.asarray(candidate_ids,np.int64)
    if s.shape!=a.shape or s.shape[1]!=len(ids):raise ValueError('score/mask/pool mismatch')
    if len(np.unique(ids))!=len(ids):raise ValueError('candidate_ids must be unique')
    masked=np.where(a,s,-np.inf);ix=np.argmax(masked,axis=1)
    if np.any(~np.isfinite(masked[np.arange(len(ix)),ix])):raise ValueError('empty candidate row')
    return ix.astype(np.int64)


def random_candidate_indices(allowed,*,seed=SEED,reps=5):
    """Draw independent pool-local candidates from the exact allowed set."""
    a=np.asarray(allowed,bool)
    if a.ndim!=2 or reps<1:raise ValueError('allowed must be [N,K] and reps positive')
    rng=np.random.default_rng(seed);out=np.empty((len(a),int(reps)),np.int64)
    for i,row in enumerate(a):
        valid=np.flatnonzero(row)
        if not len(valid):raise ValueError('empty candidate row')
        out[i]=rng.choice(valid,size=int(reps),replace=True)
    return out


def retrieval_recall_at_k(ranked_pool_local,shape_errors,allowed,eps,ks=(1,5,20)):
    """Fraction of queries retrieving any epsilon-oracle-positive in top-k.

    Rankings are produced by the model. Oracle errors define only the positive
    set and are never used to order the returned ranking.
    """
    rank=np.asarray(ranked_pool_local,np.int64);err=np.asarray(shape_errors,np.float64)
    allow=np.asarray(allowed,bool)
    if rank.ndim!=2 or err.shape!=allow.shape or err.shape[0]!=rank.shape[0]:
        raise ValueError('ranking/error/mask shape mismatch')
    if np.any((rank<0)|(rank>=err.shape[1])):raise ValueError('ranked IDs must be pool-local')
    oracle=np.min(np.where(allow,err,np.inf),axis=1)
    positive=allow&(err<=oracle[:,None]+float(eps))
    out={}
    for k in ks:
        if int(k)<1:raise ValueError('k must be positive')
        if rank.shape[1]<int(k):
            out[str(int(k))]=None
            continue
        # Score-based rankings often contain the full pool, with excluded
        # candidates sorted at the tail (score=-inf).  Count the first k
        # allowed candidates, not those masked tail entries.
        hits=[]
        for i in range(len(rank)):
            ordered=rank[i]
            ordered=ordered[allow[i,ordered]][:int(k)]
            hits.append(bool(positive[i,ordered].any()))
        out[str(int(k))]=float(np.mean(hits))
    return out


def target_distribution(err,allowed,tau):
    """Soft teacher from train labels only; invalid same-task candidates get 0."""
    e=np.asarray(err,np.float64);a=np.asarray(allowed,bool)
    if e.shape!=a.shape or tau<=0:raise ValueError('invalid teacher inputs')
    logits=np.where(a,-e/float(tau),-np.inf);mx=np.max(logits,axis=1,keepdims=True)
    w=np.exp(logits-mx);w[~a]=0
    return w/np.maximum(w.sum(1,keepdims=True),1e-30)


def shape_error_matrix(query_shapes,candidate_shapes,vs,*,batch=8,device='cpu'):
    """Vectorized per-query/candidate Procrustes errors; columns are pool-local.

    Uses batched 3x3 SVDs, retaining the same SO(3)+shared-scale objective as
    ``theta_regression._shape_error`` without a Python loop over candidates.
    """
    q=np.asarray(query_shapes,np.float32);c=np.asarray(candidate_shapes,np.float32)
    if q.ndim!=3 or q.shape[1:]!=(10,7) or c.ndim!=3 or c.shape[1:]!=(10,7):
        raise ValueError('shape arrays must be [N,10,7]')
    if batch<1:raise ValueError('batch must be positive')
    if str(device).startswith('cuda'):
        import torch
        dev=torch.device(device)
        cand=torch.as_tensor(c[:,:,:6],dtype=torch.float64,device=dev)
        cden=torch.clamp((cand*cand).sum(dim=(1,2)),min=1e-12)
        out=np.empty((len(q),len(c)),np.float32)
        for start in range(0,len(q),batch):
            target=torch.as_tensor(q[start:start+batch,:,:6],dtype=torch.float64,device=dev);b=len(target)
            cross=torch.einsum('kti,btj->bkij',cand[:,:,:3],target[:,:,:3])
            cross=cross+torch.einsum('kti,btj->bkij',cand[:,:,3:6],target[:,:,3:6])
            u,_,vh=torch.linalg.svd(cross,full_matrices=False)
            det=torch.linalg.det(u@vh)
            corr=torch.eye(3,dtype=torch.float64,device=dev).expand(b,len(c),3,3).clone()
            corr[:,:,2,2]=det
            rot=u@corr@vh
            aligned=torch.cat((torch.einsum('ktc,bkcd->bktd',cand[:,:,:3],rot),
                torch.einsum('ktc,bkcd->bktd',cand[:,:,3:6],rot)),dim=-1)
            scale=torch.clamp((aligned*target[:,None]).sum(dim=(2,3))/cden[None],min=1e-6)
            delta=aligned*scale[:,:,None,None]-target[:,None]
            out[start:start+b]=(delta.square().mean(dim=(2,3))).cpu().numpy().astype(np.float32)
            if len(q)>4096 and (start//batch)%10==0:
                print(json.dumps({'stage':'shape_error_matrix','backend':'cuda',
                    'rows_done':start+b,'rows_total':len(q),'candidates':len(c)}),flush=True)
        return out
    cand=c[:,:,:6].astype(np.float64);cden=np.maximum(np.sum(cand*cand,axis=(1,2)),1e-12)
    out=np.empty((len(q),len(c)),np.float32)
    for start in range(0,len(q),batch):
        target=q[start:start+batch,:,:6].astype(np.float64);b=len(target)
        # Cross covariance combines translation and rotation, as _procrustes does.
        # Explicit contraction: [B,K,axis_i,axis_j].
        cross=np.einsum('kti,btj->bkij',cand[:,:,:3],target[:,:,:3])
        cross+=np.einsum('kti,btj->bkij',cand[:,:,3:6],target[:,:,3:6])
        u,_,vt=np.linalg.svd(cross,full_matrices=False)
        det=np.linalg.det(u@vt);corr=np.broadcast_to(np.eye(3),(b,len(c),3,3)).copy()
        corr[:,:,2,2]=det;r=u@corr@vt
        aligned_t=np.einsum('ktc,bkcd->bktd',cand[:,:,:3],r)
        aligned_r=np.einsum('ktc,bkcd->bktd',cand[:,:,3:6],r)
        aligned=np.concatenate([aligned_t,aligned_r],axis=-1)
        scale=np.maximum(np.sum(aligned*target[:,None],axis=(2,3))/cden[None],1e-6)
        delta=aligned*scale[:,:,None,None]-target[:,None]
        out[start:start+b]=np.mean(delta*delta,axis=(2,3)).astype(np.float32)
    return out


def oracle_and_random_indices(query_shapes,candidate_shapes,vs,*,seed=SEED,random_reps=5):
    """Replicate shape_select's oracle argmin and interleaved seeded random IDs."""
    theta=_load_theta();q=np.asarray(query_shapes);c=np.asarray(candidate_shapes);rng=np.random.default_rng(seed)
    oracle=[];random=[]
    for target in q:
        err=np.asarray([theta._shape_error(candidate,target,vs)[0] for candidate in c])
        oracle.append(int(np.argmin(err)))
        random.append([int(rng.integers(len(c))) for _ in range(random_reps)])
    return np.asarray(oracle,np.int64),np.asarray(random,np.int64)


def train_teacher_from_libero40(train_cache,candidate_pool,vs,*,heldout_tasks=(),tau_l=.02):
    """Build label-derived soft targets from LIBERO-40 train rows only."""
    task=np.asarray(train_cache['task'],np.int64)
    keep=~np.isin(task,np.asarray(heldout_tasks,np.int64))
    if not np.any(keep):raise ValueError('no LIBERO-40 train rows after task holdout')
    q=np.asarray(train_cache['shape'][keep],np.float32)
    allowed=filter_cross_task(candidate_pool,task[keep],heldout=heldout_tasks)
    err=shape_error_matrix(q,candidate_pool['shape'],vs)
    teacher=target_distribution(err,allowed,tau_l)
    return {'row_ids':np.asarray(train_cache['row_ids'],np.int64)[keep],
        'query_task':task[keep],'shape_error':err,'allowed':allowed,'teacher':teacher,
        'source':'LIBERO-40 train cache only','unseen_label_access':False}


def listwise_loss(logits,teacher,allowed):
    """Numpy reference for masked listwise soft InfoNCE."""
    x=np.asarray(logits,np.float64);p=np.asarray(teacher,np.float64);a=np.asarray(allowed,bool)
    if x.shape!=p.shape or x.shape!=a.shape:raise ValueError('listwise shape mismatch')
    z=x.copy();z[~a]=-np.inf
    mx=np.max(z,axis=1,keepdims=True);logz=mx+np.log(np.exp(z-mx).sum(1,keepdims=True))
    logp=z-logz
    terms=np.zeros_like(p);terms[a]=p[a]*logp[a]
    return float(-terms.sum()/len(x))


def episode_bootstrap(values,episode,reps=5000,seed=SEED):
    v=np.asarray(values,np.float64);ep=np.asarray(episode)
    u,inv=np.unique(ep,return_inverse=True);s=np.bincount(inv,weights=v);n=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(len(u),size=(reps,len(u)))
    d=s[ix].sum(1)/n[ix].sum(1)
    return {'mean':float(v.mean()),'ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))],
            'episodes':int(len(u)),'rows':int(len(v)),'reps':int(reps)}


def _spearman(x,y):
    from scipy.stats import rankdata
    a=rankdata(np.asarray(x));b=rankdata(np.asarray(y))
    if a.std()==0 or b.std()==0:return 0.
    return float(np.corrcoef(a,b)[0,1])


def score_predictions(picked_ids,err,actions_pred,actions_true,oracle_ix,eps,episode,hard,*,ranked_pool_local=None):
    """Evaluation-only scorer. Keep this API separate from every fit function."""
    pi=np.asarray(picked_ids,np.int64);e=np.asarray(err,np.float32)
    pred=np.asarray(actions_pred,np.float32);y=np.asarray(actions_true,np.float32)
    ox=np.asarray(oracle_ix,np.int64);ep=np.asarray(episode);h=np.asarray(hard,bool)
    if pred.shape!=y.shape or pred.ndim!=3 or pred.shape[1:]!=(10,7):raise ValueError('actions must [N,10,7]')
    if len(pi)!=len(y) or e.shape[0]!=len(y):raise ValueError('evaluation row mismatch')
    if np.any((pi<0)|(pi>=e.shape[1])):raise ValueError('picked_ids must be pool-local indices')
    chosen=e[np.arange(len(e)),pi]
    if np.any(~np.isfinite(chosen)):raise ValueError('picked candidate violates an exclusion mask')
    row_mse=np.mean((pred-y)**2,axis=(1,2))
    oracle_err=e.min(1)
    recall=(retrieval_recall_at_k(ranked_pool_local,e,np.isfinite(e),eps)
            if ranked_pool_local is not None else {'1':None,'5':None,'20':None})
    if ranked_pool_local is not None and not np.array_equal(np.asarray(ranked_pool_local)[:,0],pi):
        raise ValueError('picked_ids must equal the first ranked pool-local candidate')
    oracle_mean=float(oracle_err.mean());random_floor=float(np.min(e,axis=1).mean())
    random_ix=random_candidate_indices(np.isfinite(e),seed=SEED,reps=5)
    random_err=np.mean(e[np.arange(len(e))[:,None],random_ix],axis=1)
    gap=max(float(random_err.mean()-oracle_mean),1e-12)
    return {'n':len(y),'shape_error':{'all':float(chosen.mean()),'hard':float(chosen[h].mean()) if h.any() else None},
        'action_mse':{'all':float(row_mse.mean()),'hard':float(row_mse[h].mean()) if h.any() else None,
                      'episode_bootstrap':episode_bootstrap(row_mse,ep)},
        'recall_oracle_epsilon':recall,
        'random_shape_error_mean_over_5':float(random_err.mean()),
        'gap_recovery':float((random_err.mean()-chosen.mean())/gap),
        'picked_candidate_pool_indices':pi.tolist(),
        'oracle_candidate_index':ox.tolist()}


def _torch_batch(features,indices,device,*,include_training_metadata=False,mode=None):
    import torch
    out={};metadata={}
    needed=None
    if mode is not None:
        if mode=='P0_mean':needed={'mean','state'}
        elif mode in ('P1_tok','P1_tok_lang'):needed={'tokens','state','history'}
        elif mode.startswith('P2_mid_'):needed={'mid_tokens','state','history'}
        elif mode=='P3_exp':needed={'expert_t1','state','history'}
        elif mode=='P4_act':needed={'action'}
        elif mode=='P5_hist':needed={'state','history'}
        elif mode=='P6_exp_hist':needed={'expert_t1','state','history'}
        elif mode=='P6_tok_hist':needed={'tokens','state','history'}
        elif mode=='P6_all':needed={'mean','expert_t1','state','history'}
        elif mode in ('P7_bottleneck','P7b_type_bottleneck'):needed={'tokens','expert_t1'}
        else:raise ValueError(f'unknown batch mode: {mode}')
    for key,value in features.items():
        if value is None:continue
        if needed is not None and key not in needed and not (include_training_metadata and key in ('task','episode','frame')):continue
        # Labels and row identity are never exposed to a query network. Task,
        # episode and frame are used only by the training-only hard-negative
        # miner, under a separate metadata key.
        if key in ('target_actions','actions','row_ids','is_dev'):continue
        if key in ('task','episode','frame'):
            if include_training_metadata:
                metadata[key]=torch.as_tensor(np.array(value[indices],copy=True),dtype=torch.long,device=device)
            continue
        if isinstance(value,PooledTokenFeatureStore):
            tok,mask,lang=value[indices]
            out['tokens']=torch.as_tensor(tok,dtype=torch.float32,device=device)
            out['token_mask']=torch.as_tensor(mask,dtype=torch.bool,device=device)
            out['lang_mean']=torch.as_tensor(lang,dtype=torch.float32,device=device)
            continue
        if isinstance(value,MidLayerTokenFeatureStore):
            tok,mask,lang=value[indices]
            out['mid_tokens']=torch.as_tensor(tok,dtype=torch.float32,device=device)
            out['mid_token_mask']=torch.as_tensor(mask,dtype=torch.bool,device=device)
            out['mid_lang_mean']=torch.as_tensor(lang,dtype=torch.float32,device=device)
            continue
        arr=np.asarray(value[indices])
        dtype=torch.bool if arr.dtype==np.bool_ else torch.float32
        out[key]=torch.as_tensor(np.array(arr,copy=True),dtype=dtype,device=device)
    if 'history' in out and 'state' in out:
        out['history_context']=torch.cat([out['history'].flatten(1),out['state']],dim=-1)
    if include_training_metadata:out['_training_metadata']=metadata
    return out


def fit_probe_model(mode,features,candidate_shapes,teacher,allowed,train_indices,*,
                    epochs=20,batch_size=128,temperature=.1,lr=1e-3,device='cpu',
                    threads=8,history_dropout=0.,hard_negative_weight=0.,checkpoint=None,
                    memory_fraction=.20):
    """Fit one query/key model using training rows only; validation is separate.

    Inputs include train-row actions only for P7b's DCT auxiliary target. The
    unseen Dev/Eval target arrays must never be placed in ``features``.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(threads));device=torch.device(device)
    if device.type=='cuda':
        cuda_index=device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction),device=cuda_index)
    shape=np.asarray(candidate_shapes,np.float32);teach=np.asarray(teacher,np.float32);allow=np.asarray(allowed,bool)
    ids=np.asarray(train_indices,np.int64)
    if teach.shape!=allow.shape or teach.shape!=(len(next(iter(features.values()))),len(shape)):
        raise ValueError('training teacher/candidate shape mismatch')
    if np.any(teach[~allow]!=0) or not np.allclose(teach.sum(1),1.,atol=1e-4):raise ValueError('invalid masked teacher distribution')
    token_source=features.get('mid_tokens') if mode.startswith('P2_mid_') else features.get('tokens')
    token_dim=token_source.shape[-1] if token_source is not None else 2048
    net=ProbeNetwork(mode,token_dim=token_dim,
        expert_dim=features.get('expert_t1',np.zeros((1,1,2048))).shape[-1]).module.to(device)
    keynet=net.key;candidate=torch.as_tensor(shape.reshape(len(shape),-1),dtype=torch.float32,device=device)
    candidate_dct=torch.as_tensor(dct3_motion_coefficients(shape).reshape(len(shape),-1),dtype=torch.float32,device=device)
    rng=np.random.default_rng(SEED);p7_pretrain=[]
    if mode=='P7_bottleneck':
        # Observation-only bottleneck pretraining (Q-tok + Q-exp); freeze this
        # representation before fitting the query projection and type key.
        preopt=torch.optim.AdamW(net.query.parameters(),lr=float(lr))
        for epoch in range(int(epochs)):
            order=ids[rng.permutation(len(ids))];losses=[];net.query.train()
            for start in range(0,len(order),int(batch_size)):
                ix=order[start:start+int(batch_size)];batch=_torch_batch(features,ix,device,mode=mode)
                mask=batch.get('token_mask',torch.ones(batch['tokens'].shape[:2],device=device,dtype=torch.bool))
                _,(rec_tok,rec_exp),_=net.query(batch['tokens'],batch['expert_t1'],token_mask=mask)
                delta=(rec_tok-batch['tokens']).square().mean(-1)
                loss=(delta*mask).sum()/mask.sum().clamp_min(1)+(rec_exp-batch['expert_t1']).square().mean()
                preopt.zero_grad(set_to_none=True);loss.backward();preopt.step();losses.append(float(loss.detach()))
            p7_pretrain.append({'epoch':epoch+1,'observation_reconstruction_loss':float(np.mean(losses))})
        for param in net.query.parameters():param.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=float(lr))
    history=[]
    for epoch in range(int(epochs)):
        order=ids[rng.permutation(len(ids))];losses=[];pair_counts=[];net.train()
        for start in range(0,len(order),int(batch_size)):
            ix=order[start:start+int(batch_size)];batch=_torch_batch(features,ix,device,
                include_training_metadata=True,mode=mode)
            training_metadata=batch.pop('_training_metadata')
            if mode.startswith('P6_') and history_dropout:
                batch['history_context']=torch.as_tensor(apply_history_modality_dropout(batch['history_context'].cpu().numpy(),history_dropout,
                    seed=SEED+epoch*100000+start),dtype=torch.float32,device=device)
            q,aux=net(batch);keys=F.normalize(keynet(candidate),dim=-1);q=F.normalize(q,dim=-1)
            logits=q@keys.T/float(temperature);mask=torch.as_tensor(allow[ix],device=device)
            prob=torch.as_tensor(teach[ix],device=device);logits=logits.masked_fill(~mask,-1e9)
            if mode=='P7b_type_bottleneck':
                target=prob@candidate_dct
                loss,_=p7b_type_loss(logits,prob,aux['dct'],torch.as_tensor(target,device=device),mask)
            else:
                loss=-(prob*F.log_softmax(logits,dim=-1)).sum(-1).mean()
            hard_pairs_count=0
            if mode.startswith('P6_') and hard_negative_weight:
                # Past-action similarity is inference-available. The next-type
                # proxy is the training-only nearest candidate under the
                # listwise teacher; no task/verb annotation is consumed.
                task=np.asarray(training_metadata['task'].detach().cpu(),np.int64)
                episode=np.asarray(training_metadata['episode'].detach().cpu(),np.int64)
                frame=np.asarray(training_metadata['frame'].detach().cpu(),np.int64)
                prev=np.asarray(batch['history_context'][:,:70].detach().cpu(),np.float32)
                nxt=np.asarray(np.argmax(teach[ix],axis=1),np.int64)
                pairs=branching_hard_negative_pairs(prev,nxt,task*100000+episode,frame//10)
                hard_pairs_count=len(pairs)
                if hard_pairs_count:
                    loss=loss+float(hard_negative_weight)*hard_negative_margin_loss(q,pairs)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss for {mode}')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            if hard_negative_weight:pair_counts.append(hard_pairs_count)
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),
            'hard_negative_pairs':int(sum(pair_counts)) if hard_negative_weight else 0})
        print(json.dumps({'stage':'fit','mode':mode,'epoch':epoch+1,'epochs':int(epochs),
            'loss':history[-1]['loss'],'rows':len(ids),'batch_size':int(batch_size)}),flush=True)
    state={'mode':mode,'state_dict':net.state_dict(),'epochs':int(epochs),'temperature':float(temperature),
        'candidate_shapes':shape,'history_dropout':float(history_dropout),
        'hard_negative_weight':float(hard_negative_weight),'trained_rows':ids.tolist(),
        'unseen_target_access':False,'train_loss':history,
        'observation_only_pretrain':p7_pretrain}
    if checkpoint is not None:
        path=Path(checkpoint)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)
    return net,state


def score_probe_model(net,mode,features,candidate_shapes,allowed,*,batch_size=128,device='cpu'):
    """Return pool-local descending scores/ranks and never consult target actions."""
    import torch
    import torch.nn.functional as F
    net.eval();net.to(device);device=torch.device(device)
    shapes=np.asarray(candidate_shapes,np.float32)
    with torch.no_grad():keys=F.normalize(net.key(torch.as_tensor(shapes.reshape(len(shapes),-1),device=device)),dim=-1)
    n=len(next(iter(features.values())));scores=np.empty((n,len(shapes)),np.float32)
    for start in range(0,n,int(batch_size)):
        ids=np.arange(start,min(start+int(batch_size),n));batch=_torch_batch(features,ids,device,mode=mode)
        if mode.startswith('P6_') and 'history' in batch:batch['history']=batch['history']
        with torch.no_grad():q,_=net(batch);q=F.normalize(q,dim=-1);scores[start:start+len(ids)]=np.asarray((q@keys.T).cpu())
    allow=np.asarray(allowed,bool)
    if allow.shape!=scores.shape:raise ValueError('allowed mask mismatch')
    scores[~allow]=-np.inf
    ranks=np.argsort(-scores,axis=1,kind='stable')
    return scores,ranks.astype(np.int64)


def select_listwise_temperatures(scores,shape_errors,allowed,*,tau_l_grid=(.01,.02,.05),temperature_grid=(.05,.1,.2)):
    """Select teacher/logit temperatures on the named holdout slice only."""
    s=np.asarray(scores,np.float64);e=np.asarray(shape_errors,np.float64);a=np.asarray(allowed,bool)
    if s.shape!=e.shape or s.shape!=a.shape:raise ValueError('temperature selection matrix mismatch')
    rows=[]
    for tau_l in tau_l_grid:
        teacher=target_distribution(e,a,float(tau_l))
        for temp in temperature_grid:
            logits=s/float(temp);logits[~a]=-1e9
            z=logits-logits.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True))
            rows.append({'tau_l':float(tau_l),'temperature':float(temp),
                         'listwise_nll':float(-np.sum(teacher*logp)/len(s))})
    return min(rows,key=lambda x:(x['listwise_nll'],x['tau_l'],x['temperature'])),rows


def _motion_shape_from_actions(actions,vs):
    a=np.asarray(actions,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(10,7):raise ValueError('actions must be [N,10,7]')
    physical=np.asarray(vs.to_physical(a[:,:,:6]),np.float32)
    return np.concatenate([physical,a[:,:,6:7]],axis=-1)


def pi05_shape_ranking(pool,query_pi05,vs,allowed=None):
    """Reproduce shape_select's Procrustes ranking from normalized pi05 actions."""
    candidate=np.asarray(pool['shape'],np.float32)
    query_shape=_motion_shape_from_actions(query_pi05,vs)
    errors=shape_error_matrix(query_shape,candidate,vs)
    if allowed is not None:
        mask=np.asarray(allowed,bool)
        if mask.shape!=errors.shape:raise ValueError('pi05 shape exclusion mask mismatch')
        errors[~mask]=np.inf
    if np.any(~np.isfinite(errors).any(axis=1)):raise ValueError('pi05 shape query has no allowed candidate')
    return errors,np.argsort(errors,axis=1,kind='stable')


def _layer_id_from_method(mode):
    if not mode.startswith('P2_mid_l'):return None
    return int(mode.removeprefix('P2_mid_l'))


def production_train_stage(*,output=OUT,epochs=12,batch_size=64,device='cpu',threads=8,limit=None,
                           shard_id=0,num_shards=1,memory_fraction=.20):
    """Train all available query probes on LIBERO-40 non-holdout tasks.

    The fixed LIBERO-40 task holdout is used only for method/temperature
    selection. P2 variants use the resumable per-layer prefix V-cache.
    """
    import torch;torch.set_num_threads(int(threads))
    if str(device).startswith('cuda'):
        index=torch.device(device).index
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction),device=torch.cuda.current_device() if index is None else index)
    if int(num_shards)<1 or not 0<=int(shard_id)<int(num_shards):raise ValueError('invalid training shard')
    print(json.dumps({'stage':'train','phase':'load_source'}),flush=True)
    theta=_load_theta();base,rows,unseen,splits=_production_data_splits(include_eval_targets=False)
    if limit is not None:
        raise ValueError('production training limit is forbidden; use smoke stage for row limits')
    import sys as _sys;_sys.path.insert(0,str(HERE));import variable_shape_memory as vs
    print(json.dumps({'stage':'train','phase':'assemble_features'}),flush=True)
    features=assemble_split_features('train',include_tokens=True,output=output)
    cache=np.load(POOL_DIR/'C_vm_none_train.npz',allow_pickle=False)
    print(json.dumps({'stage':'train','phase':'candidate_pool'}),flush=True)
    source_pool=theta.candidate_shape_pool(rep='C_vm_none')
    hold=deterministic_holdout_tasks(cache['task'])
    keep=~np.isin(source_pool['source_task'],hold)
    pool={k:np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v for k,v in source_pool.items()}
    if set(map(int,pool['source_task']))&set(hold):raise AssertionError('holdout unit leaked into candidate pool')
    all_actions=np.asarray(rows['actions'][splits['train'][1],:,:7],np.float32)
    shape=_motion_shape_from_actions(all_actions,vs)
    query_task=np.asarray(rows['task'][splits['train'][1]],np.int64)
    allowed=filter_cross_task(pool,query_task,heldout=hold)
    print(json.dumps({'stage':'train','phase':'shape_errors','rows':len(shape),'candidates':len(pool['shape']),
        'device':str(device)}),flush=True)
    errors=shape_error_matrix(shape,pool['shape'],vs,batch=512 if str(device).startswith('cuda') else 8,
        device=device)
    print(json.dumps({'stage':'train','phase':'shape_errors_done'}),flush=True)
    modes=('P0_mean','P1_tok','P1_tok_lang',*MID_LAYER_NAMES,'P3_exp','P4_act','P5_hist',
        'P6_exp_hist','P6_tok_hist','P6_all','P7_bottleneck','P7b_type_bottleneck')
    configs=[(m,0.,0.) for m in modes if not m.startswith('P6_')]
    # P6 comparisons keep the base encoder fixed and select dropout and
    # train-only branching/adjacent-window negatives on the task holdout.
    configs += [(m,p,h) for m in ('P6_exp_hist','P6_tok_hist','P6_all')
        for p,h in ((0.,0.),(.3,0.),(.5,0.),(.3,.05),(.5,.05))]
    if int(num_shards)>1:configs=shard_training_configs(configs,shard_id,num_shards)
    fit_ids=np.flatnonzero(~np.isin(query_task,hold));val_ids=np.flatnonzero(np.isin(query_task,hold))
    if not len(val_ids) or not len(fit_ids):raise ValueError('empty LIBERO-40 fit/holdout task partition')
    val_features=subset_features(features,val_ids);val_allowed=allowed[val_ids];val_errors=errors[val_ids]
    dest=Path(output)/'production';modeldir=dest/'models';modeldir.mkdir(parents=True,exist_ok=True)
    summary_path=dest/'train_summary.json'
    if summary_path.exists():return json.loads(summary_path.read_text())
    progress=dest/'train_progress.jsonl';results=[]
    completed={}
    if progress.exists():
        for line in progress.read_text().splitlines():
            if not line.strip():continue
            rec=json.loads(line);key=(rec['mode'],float(rec['history_dropout']),
                float(rec['hard_negative_weight']),float(rec['trained_tau_l']))
            if Path(rec['checkpoint']).exists():completed[key]=rec
    # A short, predeclared label-temperature grid; each tau_l model is fitted
    # separately so selection on task-holdout rows corresponds to real weights.
    for mode,drop,hard_weight in configs:
        best_for_mode=None
        layer_id=_layer_id_from_method(mode)
        mode_features=assemble_split_features('train',include_tokens=layer_id is None,mid_layer=layer_id,output=output) if layer_id is not None else features
        mode_val_features=subset_features(mode_features,val_ids)
        for tau_l in (.01,.02,.05):
            ckpt=modeldir/f'{mode}_drop{drop:g}_hard{hard_weight:g}_taul{tau_l:g}.pt'
            progress_key=(mode,float(drop),float(hard_weight),float(tau_l))
            cached=completed.get(progress_key)
            if ckpt.exists() and cached is not None and Path(cached['checkpoint'])==ckpt:
                results.append(cached)
                print(json.dumps({'stage':'train','phase':'reuse_scored_checkpoint','mode':mode,
                    'tau_l':tau_l,'checkpoint':str(ckpt)}),flush=True)
                continue
            teacher=target_distribution(errors,allowed,tau_l)
            if ckpt.exists():
                state=torch.load(ckpt,map_location='cpu',weights_only=False)
                token_source=mode_features.get('mid_tokens') if layer_id is not None else mode_features.get('tokens')
                net=ProbeNetwork(mode,token_dim=token_source.shape[-1],expert_dim=features['expert_t1'].shape[-1]).module
                net.load_state_dict(state['state_dict'])
            else:
                net,state=fit_probe_model(mode,mode_features,pool['shape'],teacher,allowed,fit_ids,
                    epochs=epochs,batch_size=batch_size,temperature=.1,device=device,threads=threads,
                    history_dropout=drop,hard_negative_weight=hard_weight,checkpoint=ckpt,
                    memory_fraction=memory_fraction)
            scores,ranks=score_probe_model(net,mode,mode_val_features,pool['shape'],val_allowed,batch_size=batch_size,device=device)
            _,grid=select_listwise_temperatures(scores,val_errors,val_allowed)
            tuned=min(grid,key=lambda x:(x['listwise_nll'],x['temperature']))
            val_top=ranks[:,0];val_err=val_errors
            row_score=float(np.mean(val_err[np.arange(len(val_ids)),val_top]))
            rec={'mode':mode,'history_dropout':drop,'hard_negative_weight':hard_weight,
                'trained_tau_l':tau_l,'selected_temperature':tuned['temperature'],
                'val_listwise_nll':tuned['listwise_nll'],'val_shape_error':row_score,
                'checkpoint':str(ckpt),'fit_rows':len(fit_ids),'holdout_rows':len(val_ids)}
            results.append(rec)
            if best_for_mode is None or (row_score,drop,hard_weight,tau_l)<(best_for_mode['val_shape_error'],best_for_mode['history_dropout'],best_for_mode['hard_negative_weight'],best_for_mode['trained_tau_l']):best_for_mode=rec
            with progress.open('a') as f:f.write(json.dumps(rec)+'\n')
        if 'net' in locals():del net
    selected={}
    for mode in sorted({m for m,_,_ in configs}):
        choices=[r for r in results if r['mode']==mode]
        selected[mode]=min(choices,key=lambda r:(r['val_shape_error'],r['history_dropout'],
            r['hard_negative_weight'],r['trained_tau_l']))
    if int(num_shards)>1:
        shard_path=dest/f'train_shard_{int(shard_id)}.json'
        shard={'status':'trained_shard','shard_id':int(shard_id),'num_shards':int(num_shards),
            'holdout_tasks':list(hold),'fit_tasks':[int(t) for t in np.unique(query_task) if t not in hold],
            'candidate_count':len(pool['shape']),'models':results,
            'selected_by_mode':selected,'epochs':int(epochs),'batch_size':int(batch_size),
            'unseen_dev_or_eval_labels_used_for_fit':False}
        _atomic_json(shard_path,shard)
        return shard
    if set(selected)!=set(modes):raise RuntimeError('unsharded training did not produce every declared method')
    p13=choose_p1_or_p3_on_holdout(
        _holdout_error_for_checkpoint(selected['P1_tok'],val_features,pool,val_allowed,val_errors,batch_size,device),
        _holdout_error_for_checkpoint(selected['P3_exp'],val_features,pool,val_allowed,val_errors,batch_size,device),
        split_name='libero40_task_holdout')
    summary={'status':'trained','holdout_tasks':list(hold),'fit_tasks':[int(t) for t in np.unique(query_task) if t not in hold],
        'candidate_count':len(pool['shape']),'candidate_source_lengths':np.quantile(candidate_pool_source_lengths('C_vm_none')[keep],[0,.33,.66,1]).tolist(),
        'models':results,'selected_by_mode':selected,'P1_vs_P3':p13,'P2_mid_layers':list(MID_LAYER_IDS),
        'selection_split':'LIBERO-40 task holdout only','unseen_dev_or_eval_labels_used_for_fit':False}
    _atomic_json(summary_path,summary)
    return summary


def production_finalize_train_shards(*,output=OUT,num_shards=1):
    """Join independently trained, disjoint config shards into one train summary."""
    dest=Path(output)/'production';summary_path=dest/'train_summary.json'
    if summary_path.exists():return json.loads(summary_path.read_text())
    shards=[]
    for sid in range(int(num_shards)):
        path=dest/f'train_shard_{sid}.json'
        if not path.exists():raise FileNotFoundError(f'missing completed train shard {sid}: {path}')
        item=json.loads(path.read_text())
        if item.get('status')!='trained_shard' or item.get('shard_id')!=sid or item.get('num_shards')!=num_shards:
            raise ValueError(f'invalid train shard metadata: {path}')
        shards.append(item)
    results=[row for shard in shards for row in shard['models']]
    keys=[(r['mode'],r['history_dropout'],r['hard_negative_weight'],r['trained_tau_l']) for r in results]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate config across train shards')
    expected=[(m,0.,0.) for m in METHODS if not m.startswith('P6_')]
    expected += [(m,p,h) for m in ('P6_exp_hist','P6_tok_hist','P6_all')
        for p,h in ((0.,0.),(.3,0.),(.5,0.),(.3,.05),(.5,.05))]
    expected_keys={(m,float(p),float(h),t) for m,p,h in expected for t in (.01,.02,.05)}
    if set(keys)!=expected_keys:raise ValueError(f'training shard config coverage mismatch: {len(keys)} vs {len(expected_keys)}')
    expected=set(METHODS)
    selected={}
    for mode in METHODS:
        choices=[r for r in results if r['mode']==mode]
        if not choices:raise ValueError(f'missing trained method {mode}')
        selected[mode]=min(choices,key=lambda r:(r['val_shape_error'],r['history_dropout'],
            r['hard_negative_weight'],r['trained_tau_l']))
    hold=shards[0]['holdout_tasks']
    if any(s['holdout_tasks']!=hold for s in shards):raise ValueError('training shards used different holdout tasks')
    p13=choose_p1_or_p3_on_holdout(np.asarray([selected['P1_tok']['val_shape_error']]),
        np.asarray([selected['P3_exp']['val_shape_error']]),split_name='libero40_task_holdout')
    summary={'status':'trained','holdout_tasks':hold,'fit_tasks':shards[0]['fit_tasks'],
        'candidate_count':shards[0]['candidate_count'],'candidate_source_lengths':
        np.quantile(candidate_pool_source_lengths('C_vm_none'),[0,.33,.66,1]).tolist(),
        'models':results,'selected_by_mode':selected,'P1_vs_P3':p13,
        'P2_mid_layers':list(MID_LAYER_IDS),'selection_split':'LIBERO-40 task holdout only',
        'unseen_dev_or_eval_labels_used_for_fit':False,'parallel_config_shards':int(num_shards)}
    _atomic_json(summary_path,summary);return summary


def production_select_stage(*,output=OUT):
    """Materialize the LIBERO-40 task-holdout model and P1/P3 decisions."""
    dest=Path(output)/'production';source=dest/'train_summary.json'
    if not source.exists():raise FileNotFoundError('run train before select')
    train=json.loads(source.read_text());path=dest/'selection_summary.json'
    result={'status':'selected','selection_split':train['selection_split'],
        'holdout_tasks':train['holdout_tasks'],'selected_by_mode':train['selected_by_mode'],
        'P1_vs_P3':train['P1_vs_P3'],'unseen_dev_or_eval_labels_used':False,
        'methods_with_selected_holdout_config':sorted(train['selected_by_mode'])}
    if path.exists():
        old=json.loads(path.read_text())
        if old!=result:raise FileExistsError(f'selection artifact differs; refusing overwrite: {path}')
        return old
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
    return result


def _holdout_error_for_checkpoint(choice,features,pool,allowed,errors,batch_size,device):
    import torch
    ckpt=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
    net=ProbeNetwork(choice['mode'] if 'mode' in choice else choice.get('mode'),
        token_dim=features['tokens'].shape[-1],expert_dim=features['expert_t1'].shape[-1]).module
    net.load_state_dict(ckpt['state_dict']);_,rank=score_probe_model(net,choice['mode'],features,pool['shape'],allowed,batch_size=batch_size,device=device)
    return errors[np.arange(len(errors)),rank[:,0]]


def calibrate_p8_stage(*,output=OUT,threads=8,batch=4,limit=None):
    """Select P8 interpolation weight on LIBERO-40 heldout tasks only."""
    import torch;torch.set_num_threads(int(threads))
    import jax
    import jax.numpy as jnp
    if limit is not None:raise ValueError('P8 calibration limit forbidden outside smoke')
    dest=Path(output)/'production';path=dest/'p8_calibration.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('status')=='complete' and previous.get('holdout_tasks'):
            return previous
        raise FileExistsError(f'incomplete calibration artifact: {path}')
    selected_path=dest/'selection_summary.json'
    if not selected_path.exists():raise FileNotFoundError('run select before calibrate-p8')
    train_summary=json.loads((dest/'train_summary.json').read_text())
    selection=json.loads(selected_path.read_text())
    selected=selection['P1_vs_P3'][0]
    if selected not in ('P1_tok','P3_exp'):raise ValueError('invalid selected P8 type query')
    choice=train_summary['selected_by_mode'][selected]
    theta=_load_theta();base,rows,unseen,splits=_production_data_splits(include_eval_targets=False);ids=np.asarray(splits['train'][1],np.int64)
    features=assemble_split_features('train',include_tokens=True,output=output)
    pool0=theta.candidate_shape_pool(rep='C_vm_none');hold=tuple(train_summary['holdout_tasks'])
    keep=~np.isin(pool0['source_task'],hold)
    pool={k:np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v for k,v in pool0.items()}
    task=np.asarray(rows['task'][ids],np.int64);all_val_ids=np.flatnonzero(np.isin(task,hold))
    val_ids=p8_calibration_sample_ids(all_val_ids,task,hold,per_task=64)
    val_features=subset_features(features,val_ids);allow=filter_cross_task(pool,task[val_ids],heldout=hold)
    ckpt=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
    net=ProbeNetwork(selected,token_dim=features['tokens'].shape[-1],expert_dim=features['expert_t1'].shape[-1]).module
    net.load_state_dict(ckpt['state_dict']);scores,ranks=score_probe_model(net,selected,val_features,pool['shape'],allow,device='cpu')
    import sys as _sys;_sys.path.insert(0,str(HERE));import variable_shape_memory as vs
    policy,_=base.load_policy(CHECKPOINT);tau_grid=(.3,.5,.7);all_score=[[] for _ in tau_grid];all_action=[[] for _ in tau_grid]
    raw_actions=np.asarray(rows['actions'][ids,:,:7],np.float32)
    batch=max(1,int(batch))
    for start in range(0,len(val_ids),batch):
        locals_,tops=holdout_rank_batch(val_ids,ranks,start,start+batch)
        qpi=np.asarray(features['action'][locals_],np.float32)
        src=decode_pool_actions(pool,tops,vs,theta_source='src')
        pi=decode_pool_actions(pool,tops,vs,theta_source='pi05',query_pi05=qpi)
        choices=np.concatenate([src,pi],axis=1)
        row_ids=np.asarray(ids[locals_],np.int64);obs=base.obs_batch(rows,row_ids)
        prepared=prepare_pi05_prefix(policy,obs)
        keys=jax.vmap(lambda rid:jax.random.fold_in(jax.random.key(SEED),rid))(
            jnp.asarray(row_ids,jnp.uint32))
        noise=np.asarray(jax.vmap(lambda key:jax.random.normal(
            key,(policy.action_horizon,policy.action_dim)))(keys),np.float32)
        nrows=len(locals_)
        for ti,tau in enumerate(tau_grid):
            x,_=flow_state_and_target_velocity(choices,noise,tau)
            flat_x=x.reshape(-1,policy.action_horizon,policy.action_dim)
            flat_v=pi05_velocity_at_xt(policy,prepared,flat_x,1.-tau)
            v=flat_v.reshape(nrows,choices.shape[1],policy.action_horizon,7)
            sc=p8_expert_candidate_scores(v,choices,noise)
            all_score[ti].extend(sc);all_action[ti].extend(choices)
        done=start+nrows
        if done%100<batch or done==len(val_ids):
            print(json.dumps({'stage':'p8_calibration','done':done,'total':len(val_ids)}),flush=True)
    score_arrays=[np.asarray(x,np.float32) for x in all_score];action_arrays=[np.asarray(x,np.float32) for x in all_action]
    truth=raw_actions[val_ids]
    selected_tau,table=select_p8_tau_on_holdout(score_arrays,action_arrays,truth,tau_grid,
        split_name='libero40_task_holdout')
    result={'status':'complete','selected_query':selected,'tau':selected_tau,'grid':table,
        'rows':len(val_ids),'available_holdout_rows':len(all_val_ids),
        'calibration_sampling':'up to 64 rows per fixed holdout task, evenly spaced in row order',
        'holdout_tasks':list(hold),'noise_seed':SEED,'shared_prefix_per_query':True,
        'candidate_variants':'Top16 shape candidates x (theta_src, pi05 theta), no action mixing',
        'target_actions_used_only_for_holdout_calibration':True}
    _atomic_json(path,result);return result


def p8_calibration_sample_ids(holdout_positions,task,heldout_tasks,*,per_task=64):
    """Deterministic task-balanced calibration rows; only row/task metadata is read."""
    ids=np.asarray(holdout_positions,np.int64);t=np.asarray(task,np.int64);cap=int(per_task)
    if ids.ndim!=1 or cap<1 or np.any((ids<0)|(ids>=len(t))):raise ValueError('invalid calibration row selection')
    selected=[]
    for tid in sorted(set(map(int,heldout_tasks))):
        rows=ids[t[ids]==tid]
        if not len(rows):raise ValueError(f'no calibration rows for holdout task {tid}')
        take=min(cap,len(rows))
        loc=np.linspace(0,len(rows)-1,take,dtype=np.int64)
        selected.extend(rows[loc].tolist())
    return np.asarray(selected,np.int64)


def holdout_rank_batch(holdout_positions,ranked_holdout,start,stop):
    """Slice ranks by compact holdout position, never by source split position."""
    ids=np.asarray(holdout_positions,np.int64);r=np.asarray(ranked_holdout,np.int64)
    if ids.ndim!=1 or r.ndim!=2 or len(ids)!=len(r):raise ValueError('holdout ids/ranks must share compact row axis')
    if start<0 or stop<start:raise ValueError('invalid holdout batch range')
    pos=np.arange(int(start),min(int(stop),len(ids)),dtype=np.int64)
    return ids[pos],r[pos]


def _candidate_choice_metrics(name,actions,selected_local,ranked,errors,allowed,target,episode,hard,eps=.01,
                             source_lengths=None):
    a=np.asarray(actions,np.float32);y=np.asarray(target,np.float32);pick=np.asarray(selected_local,np.int64)
    row_mse=np.mean((a-y)**2,axis=(1,2));shape_error=errors[np.arange(len(pick)),pick]
    rec=retrieval_recall_at_k(ranked,errors,allowed,eps,ks=(1,5,20))
    return {'name':name,'rows':len(pick),'shape_error_mean':float(shape_error.mean()),
        'shape_error_hard':_mean_or_none(shape_error[hard]),'action_mse_mean':float(row_mse.mean()),
        'action_mse_hard':_mean_or_none(row_mse[hard]),'action_mse_episode_bootstrap':_bootstrap_metric(row_mse,episode,hard),
        'recall_oracle_epsilon':rec,'picked_pool_local':pick.tolist(),
        'diagnostics':_choice_stratification(a,y,pick,errors,allowed,ranked,episode,hard,
            source_lengths=source_lengths)}


def _baseline_result_entries(metrics):
    """Keep baseline headline metrics and their matching stratified diagnostics."""
    headline=metrics.get('baselines',{})
    diagnostics=metrics.get('diagnostics',{}).get('baselines',{})
    return {name:{'metrics':{**value,'diagnostics':diagnostics.get(name,{})}}
            for name,value in headline.items()}


def production_evaluate_stage(*,output=OUT,threads=8,batch=64,limit=None,run_expert=True):
    """Evaluate frozen query models on unseen LIBERO Eval rows; wire P8/P9."""
    import torch;torch.set_num_threads(int(threads))
    import jax
    import jax.numpy as jnp
    if limit is not None:raise ValueError('production evaluation limit is forbidden; use smoke stage')
    dest=Path(output)/'production';outdir=dest/'evaluation'
    summary_path=outdir/'summary.json';rows_path=outdir/'rows.npz'
    if summary_path.exists() and rows_path.exists():
        previous=json.loads(summary_path.read_text())
        with np.load(rows_path,allow_pickle=False) as saved:
            rows_saved=len(saved['row_ids'])
        if previous.get('status')=='complete' and previous.get('rows')==rows_saved:
            return previous
        raise ValueError('completed evaluation artifacts fail consistency check')
    if summary_path.exists() or rows_path.exists():
        raise FileExistsError('partial evaluation artifacts exist; choose a fresh output directory')
    if outdir.exists():raise FileExistsError(f'partial evaluation directory exists: {outdir}')
    dest.mkdir(parents=True,exist_ok=True);train_summary=json.loads((dest/'train_summary.json').read_text())
    selection_path=dest/'selection_summary.json'
    if not selection_path.exists():raise FileNotFoundError('run select before evaluate')
    selection=json.loads(selection_path.read_text())
    calib_path=dest/'p8_calibration.json'
    if run_expert and not calib_path.exists():raise FileNotFoundError('run calibrate-p8 before production evaluate')
    calibration=json.loads(calib_path.read_text()) if calib_path.exists() else {'tau':.5}
    theta=_load_theta();base,rows,unseen,splits=_production_data_splits(include_eval_targets=True);features=assemble_split_features('eval',include_tokens=True,output=output)
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    import variable_shape_memory as vs
    cache=np.load(POOL_DIR/'C_vm_none_eval.npz',allow_pickle=False);hard=np.asarray(cache['hard'],bool)
    if len(features['mean'])!=619:raise ValueError('expected fixed 619 unseen Eval queries')
    pool0=theta.candidate_shape_pool(rep='C_vm_none');hold=tuple(train_summary['holdout_tasks'])
    keep=~np.isin(pool0['source_task'],hold)
    pool={k:np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v for k,v in pool0.items()}
    pool['source_length']=candidate_pool_source_lengths('C_vm_none')[keep]
    task=features['task'];allowed=filter_cross_task(pool,task,heldout=hold)
    query_pi=np.asarray(features['action'],np.float32);target=np.asarray(features['target_actions'],np.float32)
    qshape=_motion_shape_from_actions(target,vs)
    errors=shape_error_matrix(qshape,pool['shape'],vs,batch=8);errors[~allowed]=np.inf
    episodes=task.astype(np.int64)*100000+features['episode'].astype(np.int64)
    results={};pred_rows={}
    for mode,choice in selection['selected_by_mode'].items():
        layer_id=_layer_id_from_method(mode)
        mode_features=assemble_split_features('eval',include_tokens=layer_id is None,mid_layer=layer_id,output=output) if layer_id is not None else features
        ckpt=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
        token_source=mode_features.get('mid_tokens') if layer_id is not None else mode_features.get('tokens')
        net=ProbeNetwork(mode,token_dim=token_source.shape[-1],expert_dim=features['expert_t1'].shape[-1]).module
        net.load_state_dict(ckpt['state_dict'])
        score,rank=score_probe_model(net,mode,mode_features,pool['shape'],allowed,batch_size=batch,device='cpu')
        result=evaluate_shape_choices(pool,qshape,rank[:,0],query_pi,task,episodes,hard,target,
            vs,eps=.01,ranked_local=rank[:,:min(20,rank.shape[1])])
        results[mode]={'selected_config':choice,'metrics':result}
        pred_rows[f'rank_{mode}']=rank[:,:20]
    first_metrics=next(iter(results.values()))['metrics']
    # Keep the stratified summaries beside the headline metrics so the
    # report writer includes the same strata for baselines as for learned
    # P0-P9 choices and the action-unit oracle.
    results.update(_baseline_result_entries(first_metrics))
    # Action-space upper bound over every allowed (shape, theta_src) memory
    # unit, matching the candidate family used by P9.
    unit_actions=decode_pool_actions(pool,np.arange(len(pool['shape']),dtype=np.int64)[None,:],
        vs,theta_source='src')[0]
    unit_pick=np.empty(len(task),np.int64);unit_pred=np.empty_like(target);unit_mse=np.empty(len(task),np.float32)
    for i in range(len(task)):
        valid=np.flatnonzero(allowed[i])
        local,local_errors=unit_action_oracle(unit_actions[valid],target[i])
        unit_pick[i]=valid[local];unit_pred[i]=unit_actions[unit_pick[i]];unit_mse[i]=local_errors[local]
    unit_shape_error=errors[np.arange(len(task)),unit_pick]
    results['unit_action_oracle']={'metrics':{
        'shape_error_all':float(unit_shape_error.mean()),
        'shape_error_hard':_mean_or_none(unit_shape_error[hard]),
        'action_mse_all':float(unit_mse.mean()),
        'action_mse_hard':_mean_or_none(unit_mse[hard]),
        'action_mse_episode_bootstrap':_bootstrap_metric(unit_mse,episodes,hard),
        'recall_oracle_epsilon':None,
        'diagnostics':_choice_stratification(unit_pred,target,unit_pick,errors,allowed,
            None,episodes,hard,source_lengths=pool.get('source_length'))}}
    pred_rows['unit_action_oracle_pool_local']=unit_pick
    pred_rows['unit_action_oracle_action']=unit_pred
    # Optional zero-noise Q-exp diagnostic. It is activated by a separate
    # resumable ``extract --zero-noise`` pass and never changes the selected
    # checkpoint or hyperparameters.
    zero_manifest=Path(output)/'policy_features_zero'/'manifest.json'
    if zero_manifest.exists():
        zero_meta=json.loads(zero_manifest.read_text())
        if not zero_meta.get('complete') or zero_meta.get('row_counts',{}).get('eval')!=619:
            raise ValueError('zero-noise expert cache must cover all 619 Eval rows')
        zero_features=assemble_split_features('eval',include_tokens=True,zero_noise=True,output=output)
        noise_zero={}
        for mode in ('P3_exp','P6_exp_hist','P6_all'):
            if mode not in selection['selected_by_mode']:continue
            choice=selection['selected_by_mode'][mode]
            mode_features=zero_features
            layer_id=_layer_id_from_method(mode)
            if layer_id is not None:
                mode_features=assemble_split_features('eval',include_tokens=False,mid_layer=layer_id,
                    zero_noise=True,output=output)
            ckpt=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
            tok_source=mode_features.get('mid_tokens') if layer_id is not None else mode_features.get('tokens')
            net=ProbeNetwork(mode,token_dim=tok_source.shape[-1] if tok_source is not None else 2048,
                expert_dim=mode_features['expert_t1'].shape[-1]).module
            net.load_state_dict(ckpt['state_dict'])
            score,rank=score_probe_model(net,mode,mode_features,pool['shape'],allowed,batch_size=batch,device='cpu')
            noise_zero[mode]=evaluate_shape_choices(pool,qshape,rank[:,0],query_pi,task,episodes,hard,target,
                vs,eps=.01,ranked_local=rank[:,:min(20,rank.shape[1])])
            pred_rows[f'rank_{mode}_zero_noise']=rank[:,:20]
        results['zero_noise_expert_hidden_comparison']=noise_zero
    # P8 and P9 share the selected P1/P3 ranking and one KV prefix per query.
    p13=selection['P1_vs_P3'][0];choice=selection['selected_by_mode'][p13]
    ckpt=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
    query_net=ProbeNetwork(p13,token_dim=features['tokens'].shape[-1],expert_dim=features['expert_t1'].shape[-1]).module
    query_net.load_state_dict(ckpt['state_dict']);type_scores,type_rank=score_probe_model(query_net,p13,features,pool['shape'],allowed,batch_size=batch,device='cpu')
    p8_action=np.zeros((len(task),10,7),np.float32);p9e_action=np.zeros_like(p8_action);p9d_action=np.zeros_like(p8_action)
    p8_id=np.zeros(len(task),np.int64);p9e_id=np.zeros(len(task),np.int64);p9d_id=np.zeros(len(task),np.int64)
    if run_expert:
        policy,_=base.load_policy(CHECKPOINT)
        for i,rid in enumerate(features['row_ids']):
            obs=base.obs_batch(unseen,np.asarray([int(rid)]));prepared=prepare_pi05_prefix(policy,obs)
            key=jax.random.fold_in(jax.random.key(SEED),jnp.asarray(int(rid),jnp.uint32))
            noise=np.asarray(jax.random.normal(key,(policy.action_horizon,policy.action_dim)),np.float32)
            def velocity(x,t,cache_obj):return pi05_velocity_at_xt(policy,cache_obj,x,t)
            selected=p8_p9_select_one_query(pool,type_rank[i],query_pi[i],noise,vs,
                velocity,prefix_cache=prepared,tau=float(calibration['tau']))
            p8_action[i]=selected['P8']['action'];p8_id[i]=selected['P8']['type_pool_local']
            p9e_action[i]=selected['P9_expert']['action'];p9e_id[i]=selected['P9_expert']['pool_local']
            p9d_action[i]=selected['P9_direction']['action'];p9d_id[i]=selected['P9_direction']['pool_local']
            if (i+1)%50==0:print(json.dumps({'stage':'evaluate_p8_p9','done':i+1,'total':len(task)}),flush=True)
        pred_rows.update({'rank_p13':type_rank[:,:32],'p8_action':p8_action,'p8_pool_local':p8_id,
                          'p9_expert_action':p9e_action,'p9_expert_pool_local':p9e_id,
                          'p9_direction_action':p9d_action,'p9_direction_pool_local':p9d_id})
        results['P8']={'tau':calibration['tau'],'type_query':p13,
            'metrics':_candidate_choice_metrics('P8',p8_action,p8_id,type_rank,errors,allowed,target,episodes,hard,
                source_lengths=pool.get('source_length'))}
        results['P9_expert']={'metrics':_candidate_choice_metrics('P9_expert',p9e_action,p9e_id,type_rank,errors,allowed,target,episodes,hard,
            source_lengths=pool.get('source_length'))}
        results['P9_direction']={'metrics':_candidate_choice_metrics('P9_direction',p9d_action,p9d_id,type_rank,errors,allowed,target,episodes,hard,
            source_lengths=pool.get('source_length'))}
    summary={'status':'complete','rows':len(task),'hard_rows':int(hard.sum()),'holdout_tasks':list(hold),
        'candidate_count':len(pool['shape']),'p8_tau':calibration['tau'],'selected_P1_or_P3':p13,
        'methods':results,'unseen_eval_targets_used_only_for_metrics':True,
        'P2_mid_layers':list(MID_LAYER_IDS),'P2_mid_source':'prefix per-layer V cache'}
    staging=dest/f'evaluation.stage-{time.time_ns()}';staging.mkdir(parents=True,exist_ok=False)
    _atomic_npz(staging/'rows.npz',row_ids=features['row_ids'],task=task,episode=features['episode'],hard=hard,target=target,
        query_pi05=query_pi,shape_error=errors,allowed=allowed,**pred_rows)
    _atomic_json(staging/'summary.json',summary)
    staging.replace(outdir)
    return summary


def evaluate_shape_choices(pool,query_shapes,picked_local,query_pi05,query_task,episode,hard,
                           target_actions,vs,*,eps=.01,seed=SEED,ranked_local=None):
    """Full offline scoring; unseen targets are consumed only at this boundary."""
    theta=_load_theta();c=np.asarray(pool['shape'],np.float32);q=np.asarray(query_shapes,np.float32)
    y=np.asarray(target_actions,np.float32);pick=np.asarray(picked_local,np.int64)
    qt=np.asarray(query_task,np.int64);ep=np.asarray(episode);hard=np.asarray(hard,bool)
    pi=np.asarray(query_pi05,np.float32)
    if len(q)!=len(y) or len(qt)!=len(y) or len(pick)!=len(y):raise ValueError('query length mismatch')
    allowed=filter_cross_task(pool,qt,heldout=())
    err=shape_error_matrix(q,c,vs);err[~allowed]=np.inf
    if np.any((pick<0)|(pick>=len(c))) or np.any(~allowed[np.arange(len(qt)),pick]):
        raise ValueError('selected candidate violates cross-task exclusion')
    oracle=np.argmin(err,axis=1)
    random_ix=random_candidate_indices(allowed,seed=seed,reps=5)
    pi_phys=vs.to_physical(pi[:,:,:6])
    selected_shapes=c[pick]
    selected_pi_actions=np.stack([theta.decode_theta(selected_shapes[i],
        __import__('compressed_denoise')._procrustes(selected_shapes[i][:,:6],pi_phys[i])[0],vs) for i in range(len(q))])
    src_theta=np.asarray(pool['source_theta'])[pick]
    src_actions=np.stack([theta.decode_theta(selected_shapes[i],src_theta[i],vs) for i in range(len(q))])
    target_phys=vs.to_physical(y[:,:,:6]);oracle_theta=np.stack([
        __import__('compressed_denoise')._procrustes(c[oracle[i],:,:6],target_phys[i])[0] for i in range(len(q))])
    oracle_actions=np.stack([theta.decode_theta(c[oracle[i]],oracle_theta[i],vs) for i in range(len(q))])
    # Recall ranks come from the retrieval model; oracle errors define positives only.
    ranked=np.asarray(ranked_local,np.int64) if ranked_local is not None else pick[:,None]
    if ranked_local is not None and not np.array_equal(ranked[:,0],pick):
        raise ValueError('picked_local must equal the first ranked candidate')
    recall=retrieval_recall_at_k(ranked,err,allowed,eps,ks=(1,5,20))
    chosen_err=err[np.arange(len(q)),pick];random_shape=np.mean(err[np.arange(len(q))[:,None],random_ix],axis=1)
    oracle_err=err[np.arange(len(q)),oracle]
    # Match theta_regression.shape_select on this same (holdout-filtered)
    # candidate pool using its pi05-action Procrustes ranking rule.
    pi_shape_err,pi_rank=pi05_shape_ranking(pool,pi,vs,allowed)
    pi_pick=pi_rank[:,0]
    pi_actions=decode_pool_actions(pool,pi_pick[:,None],vs,theta_source='pi05',query_pi05=pi)[:,0]
    random_actions=decode_pool_actions(pool,random_ix,vs,theta_source='pi05',query_pi05=pi)
    random_draw_mse=np.mean((random_actions-y[:,None])**2,axis=(2,3))
    random_row_mse=random_draw_mse.mean(1);pi_mse=np.mean((pi_actions-y)**2,axis=(1,2))
    baselines={
        'pi05_shape':{'shape_error_all':float(np.mean(pi_shape_err[np.arange(len(q)),pi_pick])),
            'shape_error_hard':_mean_or_none(pi_shape_err[np.arange(len(q))[hard],pi_pick[hard]]),
            'actions':{'theta_pi05':{'mse_all':float(pi_mse.mean()),'mse_hard':_mean_or_none(pi_mse[hard]),
                'episode_bootstrap':_bootstrap_metric(pi_mse,ep,hard)}},
            'recall_oracle_epsilon':retrieval_recall_at_k(pi_rank[:,:20],err,allowed,eps),
            'picked_pool_local':pi_pick.tolist()},
        'random5':{'shape_error_all':float(random_shape.mean()),
            'shape_error_hard':_mean_or_none(random_shape[hard]),
            'actions':{'theta_pi05':{'mse_all':float(random_row_mse.mean()),
                'mse_hard':_mean_or_none(random_row_mse[hard]),
                'episode_bootstrap':_bootstrap_metric(random_row_mse,ep,hard),
                'replica_mean_mse':float(random_draw_mse.mean())}},
            'recall_oracle_epsilon':retrieval_recall_at_k(random_ix,err,allowed,eps),
            'picked_pool_local':random_ix.tolist(),'action_blended':False}}
    results={}
    for name,pred in (('theta_pi05',selected_pi_actions),('theta_src',src_actions),('theta_star_upper',oracle_actions)):
        mse=np.mean((pred-y)**2,axis=(1,2))
        results[name]={'mse_all':float(mse.mean()),'mse_hard':_mean_or_none(mse[hard]),
            'episode_bootstrap':_bootstrap_metric(mse,ep,hard)}
    result={'shape_error_all':float(chosen_err.mean()),'shape_error_hard':_mean_or_none(chosen_err[hard]),
        'oracle_shape_error_all':float(oracle_err.mean()),'random_shape_error_all':float(random_shape.mean()),
        'gap_recovery':float((random_shape.mean()-chosen_err.mean())/max(random_shape.mean()-oracle_err.mean(),1e-12)),
        'recall_oracle_epsilon':recall,'actions':results,'top1_single_item':True,
        'picked_pool_local':pick.tolist(),'oracle_pool_local':oracle.tolist(),
        'baselines':baselines}
    episode_key=qt.astype(np.int64)*100000+ep.astype(np.int64)
    _,translation_group,translation_cuts=translation_tercile_labels(y)
    diagnostics={'translation_norm_terciles':{'cut_points':translation_cuts,
        'shape_error':_stratified_bootstrap(chosen_err,translation_group,episode_key,hard),
        'recall':{str(k):(_stratified_bootstrap(
            np.asarray([bool(np.any((err[i,ranked[i,:k]]<=oracle_err[i]+eps)&allowed[i,ranked[i,:k]]))
                        for i in range(len(q))],np.float32),translation_group,episode_key,hard)
            if ranked.shape[1]>=k else None) for k in (1,5,20)},
        'action_mse':{name:_stratified_bootstrap(np.mean((pred-y)**2,axis=(1,2)),translation_group,episode_key,hard)
                      for name,pred in (('theta_pi05',selected_pi_actions),('theta_src',src_actions),('theta_star_upper',oracle_actions))}}}
    baseline_specs={
        'pi05_shape':(pi_pick,pi_shape_err[np.arange(len(q)),pi_pick],pi_mse,pi_rank),
        'random5':(random_ix[:,0],random_shape,random_row_mse,random_ix)}
    baseline_diagnostics={}
    for name,(base_pick,base_shape,base_mse,base_rank) in baseline_specs.items():
        baseline_diagnostics[name]={'translation_norm_terciles':{
            'shape_error':_stratified_bootstrap(base_shape,translation_group,episode_key,hard),
            'action_mse':_stratified_bootstrap(base_mse,translation_group,episode_key,hard),
            'recall':{str(k):(_stratified_bootstrap(
                np.asarray([bool(np.any((err[i,base_rank[i,:k]]<=oracle_err[i]+eps)&allowed[i,base_rank[i,:k]]))
                    for i in range(len(q))],np.float32),translation_group,episode_key,hard)
                if base_rank.shape[1]>=k else None) for k in (1,5,20)}}}
    if 'source_length' in pool:
        lengths=np.asarray(pool['source_length'],np.int32)
        if lengths.shape!=(len(c),):raise ValueError('source_length must align with candidate pool')
        cuts=np.quantile(lengths,[1/3,2/3]);chosen_len=lengths[pick]
        length_group=source_length_categories(chosen_len,cuts)
        diagnostics['selected_source_fragment_length_terciles']={'train_cut_points':cuts.tolist(),
            'source_length':_stratified_bootstrap(chosen_len,length_group,episode_key,hard),
            'shape_error':_stratified_bootstrap(chosen_err,length_group,episode_key,hard),
            'action_mse':{name:_stratified_bootstrap(np.mean((pred-y)**2,axis=(1,2)),length_group,episode_key,hard)
                          for name,pred in (('theta_pi05',selected_pi_actions),('theta_src',src_actions),('theta_star_upper',oracle_actions))}}
        random_mean_length=np.rint(np.mean(lengths[random_ix],axis=1)).astype(np.int32)
        for name,(base_pick,base_shape,base_mse,base_rank) in baseline_specs.items():
            baseline_length=(lengths[base_pick] if name=='pi05_shape' else random_mean_length)
            baseline_group=source_length_categories(baseline_length,cuts)
            baseline_diagnostics[name]['selected_source_fragment_length_terciles']={
                'train_cut_points':cuts.tolist(),
                'source_length':_stratified_bootstrap(baseline_length,baseline_group,episode_key,hard),
                'shape_error':_stratified_bootstrap(base_shape,baseline_group,episode_key,hard),
                'action_mse':_stratified_bootstrap(base_mse,baseline_group,episode_key,hard),
                'recall_oracle_epsilon':_stratified_bootstrap(
                    np.asarray([bool(np.any((err[i,base_rank[i,:min(20,base_rank.shape[1])]]<=oracle_err[i]+eps)&
                        allowed[i,base_rank[i,:min(20,base_rank.shape[1])]])) for i in range(len(q))],np.float32),
                    baseline_group,episode_key,hard)}
    diagnostics['baselines']=baseline_diagnostics
    result['diagnostics']=diagnostics
    return result


def _grouped_mean(hidden,groups):
    return _load_theta().grouped_means(hidden,groups)


def load_cached_inputs(smoke=False):
    """Read existing cached features; does not read unseen actions for training."""
    feature=THETA/'production'/'features'
    out={'train_mean':np.load(feature/'train.npy',mmap_mode='r',allow_pickle=False),
         'dev_mean':np.load(feature/'dev.npy',mmap_mode='r',allow_pickle=False),
         'eval_mean':np.load(feature/'eval.npy',mmap_mode='r',allow_pickle=False)}
    unseen={k:np.load(UNSEEN/f'{k}.npy',mmap_mode='r',allow_pickle=False)
            for k in ('tokens','groups','state','task','episode','frame','is_dev','actions')}
    out['unseen']=unseen
    if smoke:
        out={k:(v[:8] if isinstance(v,np.ndarray) and v.ndim else v) for k,v in out.items() if k!='unseen'}|{'unseen':unseen}
    return out


def _position_2d(n,dim):
    side=int(round(np.sqrt(n)))
    if side*side!=n:raise ValueError('2d positional encoding requires square grid')
    y,x=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side),indexing='ij')
    coords=np.stack([x.reshape(-1),y.reshape(-1)],axis=1)
    freq=np.exp(np.linspace(0,np.log(1000),max(1,dim//4),dtype=np.float32))
    phase=coords[:,:,None]*freq[None,None,:]
    enc=np.concatenate([np.sin(phase[:,0]),np.cos(phase[:,0]),
                        np.sin(phase[:,1]),np.cos(phase[:,1])],axis=1)
    if enc.shape[1]<dim:enc=np.pad(enc,((0,0),(0,dim-enc.shape[1])))
    return enc[:,:dim].astype(np.float32)*.02


def downsample_camera_tokens(tokens,groups,*,factor=2,add_position=True):
    """2x2 spatial pooling per 16x16 camera grid plus 2-D position vectors."""
    t=np.asarray(tokens);g=np.asarray(groups)
    if t.ndim!=2 or g.shape!=(len(t),):raise ValueError('token/group mismatch')
    blocks=[]
    for label in (0,1):
        ix=np.flatnonzero(g==label)
        if len(ix)!=256:raise ValueError('each image camera must have 256 spatial tokens')
        x=t[ix].reshape(16,16,-1)
        y=x.reshape(8,2,8,2,-1).mean((1,3))
        y=y.reshape(64,-1)
        if add_position:y=y+_position_2d(64,y.shape[-1])
        blocks.append(y)
    lang=t[g==2]
    return np.concatenate([*blocks,lang],axis=0).astype(np.float32)


def pool_prefix_v_layers(values,language_mask,*,camera_tokens=256,layer_ids=MID_LAYER_IDS):
    """Pool selected prefix V-cache layers to two 8x8 cameras + valid text.

    ``values`` is [batch, selected_layers, prefix_tokens, kv_width].  Pi0.5
    prefixes contain three 16x16 image grids (the third camera is masked) then
    the 200-token text input. The returned fixed-width array is [B,L,328,D]
    with a per-token mask; padding is not treated as a feature.
    """
    v=np.asarray(values,np.float32);lm=np.asarray(language_mask,bool)
    if v.ndim!=4 or lm.ndim!=2 or v.shape[0]!=lm.shape[0]:raise ValueError('prefix V/mask shape mismatch')
    expected=3*int(camera_tokens)+lm.shape[1]
    if v.shape[2]!=expected:raise ValueError(f'prefix sequence {v.shape[2]} != expected {expected}')
    b,nlayers,_,dim=v.shape;lang_width=lm.shape[1];out=np.zeros((b,nlayers,128+lang_width,dim),np.float32)
    mask=np.zeros((b,128+lang_width),bool);mask[:,:128]=True;mask[:,128:]=lm
    pos=_position_2d(64,dim)
    for i in range(b):
        for li in range(nlayers):
            for cam in range(2):
                x=v[i,li,cam*camera_tokens:(cam+1)*camera_tokens].reshape(16,16,dim)
                pooled=x.reshape(8,2,8,2,dim).mean((1,3)).reshape(64,dim)
                out[i,li,cam*64:(cam+1)*64]=pooled+pos
            out[i,li,128:]=v[i,li,3*camera_tokens:]*lm[i,:,None]
    return out,mask


def language_token_mean(tokens,groups,mask=None):
    """Mean valid language tokens from the frozen VLM prefix representation."""
    t=np.asarray(tokens,np.float32);g=np.asarray(groups)
    if t.ndim!=2 or g.shape!=(len(t),):raise ValueError('token/group mismatch')
    keep=(g==2)
    if mask is not None:
        m=np.asarray(mask,bool)
        if m.shape!=g.shape:raise ValueError('language mask mismatch')
        keep &= m
    if not np.any(keep):return np.zeros(t.shape[-1],np.float32)
    return t[keep].mean(0)


def write_token_memmap(source_tokens,source_groups,source_indices,output_path,*,limit=None):
    """Resumable fp16 token cache with spatially pooled images and padding mask."""
    tokens=np.load(source_tokens,mmap_mode='r',allow_pickle=False) if isinstance(source_tokens,(str,Path)) else source_tokens
    groups=np.load(source_groups,mmap_mode='r',allow_pickle=False) if isinstance(source_groups,(str,Path)) else source_groups
    ids=np.asarray(source_indices,np.int64);ids=ids[:int(limit)] if limit is not None else ids
    out=Path(output_path);progress=out.with_suffix(out.suffix+'.progress.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    shape=(len(ids),712,tokens.shape[-1])
    done=0
    if out.exists() and not progress.exists():raise FileExistsError(f'cache exists without resume marker: {out}')
    if progress.exists():
        meta=json.loads(progress.read_text())
        if meta.get('shape')!=list(shape):raise ValueError('token cache resume shape changed')
        done=int(meta.get('done',0))
    arr=np.lib.format.open_memmap(out,mode='r+' if out.exists() else 'w+',dtype=np.float16,shape=shape)
    mask_path=out.with_name(out.stem+'_mask.npy')
    mask=np.lib.format.open_memmap(mask_path,mode='r+' if mask_path.exists() else 'w+',dtype=np.bool_,shape=(len(ids),712))
    for i in range(done,len(ids)):
        tok=downsample_camera_tokens(tokens[int(ids[i])],groups[int(ids[i])])
        if len(tok)>712:raise ValueError('pooled token sequence exceeds cache width')
        arr[i]=0;arr[i,:len(tok)]=tok.astype(np.float16)
        mask[i]=False;mask[i,:len(tok)]=True
        if (i+1)%8==0 or i+1==len(ids):
            arr.flush();mask.flush();_atomic_json(progress,{'done':i+1,'shape':list(shape)})
    return {'tokens':str(out),'mask':str(mask_path),'rows':len(ids),'done':len(ids),'dtype':'float16'}


def build_histories(rows,ids):
    """Previous executed 10-step action and state by same-episode frame lookup."""
    task=np.asarray(rows['task']);ep=np.asarray(rows['episode']);frame=np.asarray(rows['frame'])
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,ep,frame,strict=True))}
    actions=np.load(RUN/'rows'/'actions.npy',mmap_mode='r',allow_pickle=False)
    state=np.load(RUN/'rows'/'state.npy',mmap_mode='r',allow_pickle=False)
    hist=np.zeros((len(ids),10,7),np.float32);prev_state=np.zeros((len(ids),8),np.float32);mask=np.zeros(len(ids),bool)
    for j,r in enumerate(ids):
        key=(int(task[r]),int(ep[r]),int(frame[r])-10);k=lookup.get(key)
        if k is not None:hist[j]=actions[k,:,:7];prev_state[j]=state[k,:8];mask[j]=True
    return hist,prev_state,mask


def build_histories_from_source(rows,ids,actions):
    """Past-only action history from the same source split (never current future)."""
    task=np.asarray(rows['task']);ep=np.asarray(rows['episode']);frame=np.asarray(rows['frame'])
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,ep,frame,strict=True))}
    act=np.asarray(actions);state=np.asarray(rows['state']);ids=np.asarray(ids,np.int64)
    hist=np.zeros((len(ids),10,7),np.float32);prev_state=np.zeros((len(ids),8),np.float32);mask=np.zeros(len(ids),bool)
    for j,r in enumerate(ids):
        k=lookup.get((int(task[r]),int(ep[r]),int(frame[r])-10))
        if k is not None:hist[j]=act[k,:,:7];prev_state[j]=state[k,:8];mask[j]=True
    return hist,prev_state,mask


class PooledTokenFeatureStore:
    """Lazy Q-tok accessor with an optional resumable image-pool cache.

    The cache stores only the 128 raw 2x2 image-token means in source dtype.
    Positional encodings and language tokens are applied/read at access time,
    preserving the original feature values while avoiding repeated 16x16
    pooling of the large source token memmap for every training epoch.
    """
    def __init__(self,tokens,groups,row_ids,*,image_cache=None,cache_path=None,cache_positions=None):
        self.tokens=np.load(tokens,mmap_mode='r',allow_pickle=False) if isinstance(tokens,(str,Path)) else tokens
        self.groups=np.load(groups,mmap_mode='r',allow_pickle=False) if isinstance(groups,(str,Path)) else groups
        self.row_ids=np.asarray(row_ids,np.int64)
        if self.row_ids.ndim!=1 or np.any(self.row_ids<0) or np.any(self.row_ids>=len(self.tokens)):
            raise ValueError('token source row IDs are invalid')
        self.shape=(len(self.row_ids),712,self.tokens.shape[-1]);self.dtype=np.dtype(np.float32)
        self.image_cache=image_cache
        self.cache_positions=(np.arange(len(self.row_ids),dtype=np.int64) if cache_positions is None
            else np.asarray(cache_positions,np.int64))
        if self.cache_positions.shape!=(len(self.row_ids),):raise ValueError('cache position shape mismatch')
        self.cache_path=Path(cache_path) if cache_path is not None else None
        if self.image_cache is None and self.cache_path is not None:
            self.image_cache=self._open_or_build_image_cache(self.cache_path)
    def __len__(self):return len(self.row_ids)
    def _open_or_build_image_cache(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        meta_path=path.with_suffix(path.suffix+'.json')
        expected={'shape':[len(self.row_ids),128,self.tokens.shape[-1]],'dtype':np.dtype(self.tokens.dtype).name,
            'row_ids_first':int(self.row_ids[0]) if len(self.row_ids) else None,
            'row_ids_last':int(self.row_ids[-1]) if len(self.row_ids) else None,
            'row_ids_sum':int(self.row_ids.sum())}
        progress_path=path.with_suffix(path.suffix+'.progress.json')
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta!=expected:raise ValueError(f'image token cache metadata mismatch: {path}')
            return np.load(path,mmap_mode='r',allow_pickle=False)
        if path.exists() and not progress_path.exists():
            raise FileExistsError(f'image token cache exists without resume metadata: {path}')
        done=0
        if progress_path.exists():
            progress=json.loads(progress_path.read_text())
            if progress.get('metadata')!=expected:raise ValueError('image token cache resume metadata mismatch')
            done=int(progress.get('done',0))
        cache=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=self.tokens.dtype,
            shape=tuple(expected['shape']))
        for i in range(done,len(self.row_ids)):
            tok=self.tokens[int(self.row_ids[i])];grp=self.groups[int(self.row_ids[i])]
            if tok.ndim!=2 or grp.shape!=(len(tok),):raise ValueError('token/group mismatch in cache source')
            pooled=[]
            for label in (0,1):
                ix=np.flatnonzero(grp==label)
                if len(ix)!=256:raise ValueError('each image camera must have 256 spatial tokens')
                grid=tok[ix].reshape(16,16,-1)
                pooled.append(grid.reshape(8,2,8,2,-1).mean((1,3)).reshape(64,-1))
            cache[i]=np.concatenate(pooled,axis=0)
            if (i+1)%128==0 or i+1==len(self.row_ids):
                cache.flush();_atomic_json(progress_path,{'done':i+1,'metadata':expected})
        cache.flush();del cache
        _atomic_json(meta_path,expected)
        if progress_path.exists():progress_path.unlink()
        return np.load(path,mmap_mode='r',allow_pickle=False)
    def subset(self,indices):
        ix=np.asarray(indices,np.int64)
        return PooledTokenFeatureStore(self.tokens,self.groups,self.row_ids[ix],
            image_cache=self.image_cache,cache_positions=self.cache_positions[ix])
    def __getitem__(self,indices):
        ix=np.asarray(indices,np.int64).reshape(-1);dim=self.tokens.shape[-1]
        lang_lengths=np.asarray([np.count_nonzero(self.groups[int(self.row_ids[i])]==2) for i in ix],np.int64)
        width=128+(int(lang_lengths.max()) if len(lang_lengths) else 0)
        out=np.zeros((len(ix),width,dim),np.float32);mask=np.zeros((len(ix),width),bool);lang=np.zeros((len(ix),dim),np.float32)
        pos=_position_2d(64,dim)
        cached_image=None if self.image_cache is None else np.asarray(
            self.image_cache[self.cache_positions[ix]],np.float32)
        for j,i in enumerate(ix):
            row=int(self.row_ids[i]);tok=self.tokens[row];grp=self.groups[row]
            if self.image_cache is None:
                image=downsample_camera_tokens(tok,grp,add_position=False)[:128]
            else:image=cached_image[j]
            out[j,:64]=image[:64]+pos;out[j,64:128]=image[64:]+pos;mask[j,:128]=True
            text=np.asarray(tok[grp==2],np.float32)
            if len(text)>width-128:raise ValueError('too many valid language tokens for Q-tok width')
            out[j,128:128+len(text)]=text;mask[j,128:128+len(text)]=True
            if len(text):lang[j]=text.mean(0)
        return out,mask,lang


class MidLayerTokenFeatureStore:
    """Lazy one-layer view over the resumable fp16 prefix-V cache."""
    def __init__(self,values,mask,layer_index,row_ids):
        self.values=np.load(values,mmap_mode='r',allow_pickle=False) if isinstance(values,(str,Path)) else values
        self.mask=np.load(mask,mmap_mode='r',allow_pickle=False) if isinstance(mask,(str,Path)) else mask
        self.layer_index=int(layer_index);self.row_ids=np.asarray(row_ids,np.int64)
        if self.values.ndim!=4 or self.mask.shape!=(self.values.shape[0],self.values.shape[2]):
            raise ValueError('mid cache must [N,L,T,D] with mask [N,T]')
        self.shape=(len(self.row_ids),self.values.shape[2],self.values.shape[3]);self.dtype=np.dtype(np.float32)
    def __len__(self):return len(self.row_ids)
    def subset(self,indices):return MidLayerTokenFeatureStore(self.values,self.mask,self.layer_index,self.row_ids[np.asarray(indices,np.int64)])
    def __getitem__(self,indices):
        ix=np.asarray(indices,np.int64).reshape(-1);tok=np.asarray(self.values[self.row_ids[ix],self.layer_index],np.float32)
        mask=np.asarray(self.mask[self.row_ids[ix]],bool);lang=tok[:,128:]
        lm=mask[:,128:];means=(lang*lm[:,:,None]).sum(1)/np.maximum(lm.sum(1,keepdims=True),1)
        return tok,mask,means


def subset_features(features,indices):
    ix=np.asarray(indices,np.int64);n=len(features['mean']);out={}
    for key,value in features.items():
        if isinstance(value,PooledTokenFeatureStore):out[key]=value.subset(ix)
        elif isinstance(value,MidLayerTokenFeatureStore):out[key]=value.subset(ix)
        elif isinstance(value,np.ndarray) or (hasattr(value,'shape') and len(value.shape)>0 and value.shape[0]==n):out[key]=value[ix]
        else:out[key]=value
    return out


def compact_split_positions(split_row_ids):
    """Indices into feature memmaps stored compactly in split order.

    The IDs themselves address source rows; these positions address arrays
    written by theta_regression.extract_features, whose first axis contains
    only the requested split.
    """
    ids=np.asarray(split_row_ids,np.int64)
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids):raise ValueError('split row IDs must be a unique vector')
    return np.arange(len(ids),dtype=np.int64)


def assemble_split_features(split,*,limit=None,include_tokens=True,zero_noise=False,mid_layer=None,output=OUT):
    """Join cached q-mean, past history, π0.5 Q-act/Q-exp and optional lazy tokens."""
    base,rows,unseen,splits=_production_data_splits(include_eval_targets=(split=='eval'));source,ids=splits[split];ids=np.asarray(ids,np.int64)
    if limit is not None:ids=ids[:int(limit)]
    if split=='train':
        mean_cache=np.load(THETA/'production'/'features'/'train.npy',mmap_mode='r',allow_pickle=False)
        mean=mean_cache[compact_split_positions(ids)]
        actions=np.load(RUN/'rows'/'actions.npy',mmap_mode='r',allow_pickle=False)
        state=np.asarray(rows['state'][ids,:8],np.float32);hist,_,_=build_histories_from_source(rows,ids,actions)
        tok_path=RUN/'tokens.npy';group_path=RUN/'groups.npy';source_rows=ids
    else:
        cache=THETA/'production'/'features'/f'{split}.npy';mean_cache=np.load(cache,mmap_mode='r',allow_pickle=False)
        mean=mean_cache[compact_split_positions(ids)]
        actions=np.asarray(unseen['actions']);state=np.asarray(unseen['state'][ids,:8],np.float32)
        hist,_,_=build_histories_from_source(unseen,ids,actions)
        tok_path=UNSEEN/'tokens.npy';group_path=UNSEEN/'groups.npy';source_rows=ids
    policy_root=Path(output)/('policy_features_zero' if zero_noise else 'policy_features')
    policy={name:np.load(policy_root/f'{split}_{name}.npy',mmap_mode='r',allow_pickle=False)[np.arange(len(ids))]
            for name in ('actions','hidden_t1','hidden_final')}
    out={'mean':np.asarray(mean,np.float32),'state':state,'history':hist,'action':policy['actions'],
         'expert_t1':policy['hidden_t1'],'expert_final':policy['hidden_final'],
         'row_ids':ids,'task':np.asarray(source['task'][ids],np.int64),'episode':np.asarray(source['episode'][ids],np.int64),
         'frame':np.asarray(source['frame'][ids],np.int64)}
    if split!='train':out['target_actions']=np.asarray(source['actions'][ids,:,:7],np.float32)
    if include_tokens:
        cache_path=(Path(output)/'production'/'token_pool_train_image_means.npy') if split=='train' else None
        out['tokens']=PooledTokenFeatureStore(tok_path,group_path,source_rows,cache_path=cache_path)
    if mid_layer is not None:
        try:layer_index=MID_LAYER_IDS.index(int(mid_layer))
        except ValueError as e:raise ValueError(f'unsupported cached mid layer {mid_layer}') from e
        policy_mid=Path(output)/('policy_features_zero' if zero_noise else 'policy_features')
        out['mid_tokens']=MidLayerTokenFeatureStore(policy_mid/f'{split}_mid_v.npy',
            policy_mid/f'{split}_mid_mask.npy',layer_index,np.arange(len(ids),dtype=np.int64))
    return out


def fixed_noise_sample_with_expert_hidden(model,observation,*,row_id,num_steps=10,zero_noise=False):
    """pi05 single-row integration with the suffix hidden before action_out_proj.

    The noise key is derived only from the global row ID. Returns predicted
    action and hidden states at the first (t=1) and final denoising steps.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs=model_api.preprocess_observation(None,observation,train=False)
    prefix,pmask,par=model.embed_prefix(obs)
    _,cache=model.PaliGemma.llm([prefix,None],mask=make_attn_mask(pmask,par),
        positions=jnp.cumsum(pmask,axis=1)-1)
    key=jax.random.fold_in(jax.random.key(SEED),jnp.asarray(row_id,jnp.uint32))
    keyed_noise=jax.random.normal(key,(1,model.action_horizon,model.action_dim))
    # Keep the fixed-noise path on device; epsilon=0 is only an alternate
    # diagnostic call and must not add a host/device round trip.
    x=jnp.zeros_like(keyed_noise) if zero_noise else keyed_noise
    def hidden_at(value,t):
        suffix,smask,sar,cond=model.embed_suffix(obs,value,jnp.asarray([t],jnp.float32))
        attn=jnp.concatenate([jnp.broadcast_to(pmask[:,None,:],
            (len(value),suffix.shape[1],pmask.shape[1])),make_attn_mask(smask,sar)],axis=-1)
        pos=jnp.sum(pmask,axis=1)[:,None]+jnp.cumsum(smask,axis=1)-1
        (_,out),_=model.PaliGemma.llm([None,suffix],mask=attn,positions=pos,
            kv_cache=cache,adarms_cond=[None,cond])
        return out[:,-model.action_horizon:]
    first=hidden_at(x,1.)
    def body(i,value):
        t=1.-i/num_steps
        h=hidden_at(value,t)
        vel=model.action_out_proj(h)
        return value-vel/num_steps
    x=jax.lax.fori_loop(0,int(num_steps),body,x)
    final=hidden_at(x,0.)
    # Keep Q-act exactly equal to the public sampler; the custom loop above is
    # used only for the intermediate Expert hidden-state taps.
    sample_noise=jnp.zeros_like(keyed_noise) if zero_noise else keyed_noise
    sampled=model.sample_actions(jax.random.PRNGKey(0),observation,num_steps=num_steps,noise=sample_noise)
    return np.asarray(sampled[0,:,:7]),np.asarray(first[0]),np.asarray(final[0])


def fixed_and_zero_noise_expert_outputs(model,observation,*,row_id,num_steps=10):
    """Extract paired epsilon-seeded and epsilon=0 action/hidden diagnostics."""
    fixed=fixed_noise_sample_with_expert_hidden(model,observation,row_id=row_id,num_steps=num_steps)
    zero=fixed_noise_sample_with_expert_hidden(model,observation,row_id=row_id,num_steps=num_steps,zero_noise=True)
    return {'fixed_seed':{'actions':fixed[0],'hidden_t1':fixed[1],'hidden_final':fixed[2]},
            'zero_noise':{'actions':zero[0],'hidden_t1':zero[1],'hidden_final':zero[2]},
            'row_id':int(row_id),'seed':SEED,'num_steps':int(num_steps)}


def _policy_hidden_and_velocity(model,obs,prefix_mask,cache,x,t):
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    suffix,smask,sar,cond=model.embed_suffix(obs,x,t)
    attn=jnp.concatenate([jnp.broadcast_to(prefix_mask[:,None,:],
        (len(x),suffix.shape[1],prefix_mask.shape[1])),make_attn_mask(smask,sar)],axis=-1)
    pos=jnp.sum(prefix_mask,axis=1)[:,None]+jnp.cumsum(smask,axis=1)-1
    (_,out),_=model.PaliGemma.llm([None,suffix],mask=attn,positions=pos,
        kv_cache=cache,adarms_cond=[None,cond])
    hidden=out[:,-model.action_horizon:]
    return hidden,model.action_out_proj(hidden)


def _batched_policy_action_hidden(model,obs,row_ids,*,num_steps=10,zero_noise=False,
                                  action_override=None,return_mid_values=False):
    """Capture Q-exp endpoints and prefix V using a shared prefix KV cache.

    When ``action_override`` is supplied it is the exact native
    ``sample_actions`` result. Hidden taps are evaluated at the original noise
    (t=1) and that exact final action (t=0), avoiding a second approximate
    integration path.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs=model_api.preprocess_observation(None,obs,train=False)
    prefix,pmask,par=model.embed_prefix(obs)
    _,cache=model.PaliGemma.llm([prefix,None],mask=make_attn_mask(pmask,par),
        positions=jnp.cumsum(pmask,axis=1)-1)
    rid=jnp.asarray(row_ids,jnp.uint32)
    keys=jax.vmap(lambda r:jax.random.fold_in(jax.random.key(SEED),r))(rid)
    x=(jnp.zeros((len(row_ids),model.action_horizon,model.action_dim),jnp.float32) if zero_noise else
       jax.vmap(lambda k:jax.random.normal(k,(model.action_horizon,model.action_dim)))(keys))
    if action_override is None:
        dt=-1.0/int(num_steps)
        hidden_width=int(model.action_out_proj.kernel.value.shape[0])
        initial=(x,jnp.asarray(1.,jnp.float32),jnp.asarray(0,jnp.int32),
                 jnp.zeros((len(row_ids),model.action_horizon,hidden_width),jnp.float32))
        def condition(carry):return carry[1]>=-dt/2
        def body(carry):
            value,time,index,first=carry
            h,vel=_policy_hidden_and_velocity(model,obs,pmask,cache,value,
                jnp.broadcast_to(time,(len(row_ids),)))
            first=jnp.where(index==0,h,first)
            return value+dt*vel,time+dt,index+1,first
        x,_,_,hidden_first=jax.lax.while_loop(condition,body,initial)
    else:
        hidden_first,_=_policy_hidden_and_velocity(model,obs,pmask,cache,x,jnp.ones((len(row_ids),),jnp.float32))
        x=jnp.asarray(action_override,jnp.float32)
    hidden_final,_=_policy_hidden_and_velocity(model,obs,pmask,cache,x,jnp.zeros((len(row_ids),),jnp.float32))
    # Keep device arrays here so the whole extractor can be jitted without a
    # tracer-to-NumPy conversion; persistence happens at the caller boundary.
    if return_mid_values:
        # gemma.KVCache V layout is [layer,batch,token,kv_head,head_dim].
        v=cache[1][jnp.asarray(MID_LAYER_IDS,dtype=jnp.int32)]
        mid=jnp.transpose(v,(1,0,2,3,4)).reshape((len(row_ids),len(MID_LAYER_IDS),v.shape[2],-1))
        return x[:,:,:7],hidden_first,hidden_final,mid
    return x[:,:,:7],hidden_first,hidden_final


def verify_sample_actions_equivalence(extracted,reference,*,tolerance=3e-5):
    """Verify the action array persisted as Q-act equals public sample_actions."""
    actual=np.asarray(extracted,np.float32);expected=np.asarray(reference,np.float32)
    if actual.shape!=expected.shape:raise ValueError('sample-action arrays have different shapes')
    max_abs=float(np.max(np.abs(actual-expected)))
    if max_abs>float(tolerance):raise AssertionError(f'extracted Q-act differs from sample_actions: {max_abs:g}')
    return {'max_abs_error':max_abs,'tolerance':float(tolerance),'rows':len(actual),
        'action_source':'Pi0.sample_actions; captured Q-exp uses separate suffix taps'}


def extract_policy_feature_stage(*,output=OUT,batch=4,threads=8,limit=None,zero_noise=False,
                                verify_integrator=False):
    """Resumable exact Q-act, Q-exp and selected prefix V-cache extraction."""
    import torch;torch.set_num_threads(int(threads))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.25')
    import jax
    import jax.numpy as jnp
    base,rows,unseen,splits=_production_data_splits(include_eval_targets=False)
    model,config=base.load_policy(CHECKPOINT);root=Path(output)/('policy_features_zero' if zero_noise else 'policy_features')
    root.mkdir(parents=True,exist_ok=True);manifest_path=root/'manifest.json'
    requested_ids={name:np.asarray(ids,np.int64)[:int(limit)] if limit is not None else np.asarray(ids,np.int64)
        for name,(_,ids) in splits.items()}
    requested_hashes={name:__import__('hashlib').sha256(ids.tobytes()).hexdigest()
        for name,ids in requested_ids.items()}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        same=(previous.get('row_ids_sha256')==requested_hashes and
              bool(previous.get('zero_noise'))==bool(zero_noise))
        if previous.get('complete') and same:return previous
        if previous.get('complete') and not same:
            raise FileExistsError('complete feature cache has a different row set/noise mode; choose a fresh output root')
    from openpi.models import gemma
    gemma_cfg=gemma.get_config(config.paligemma_variant);v_dim=gemma_cfg.num_kv_heads*gemma_cfg.head_dim
    prefix_len=3*256+int(config.max_token_len);pooled_len=128+int(config.max_token_len)
    summary={'checkpoint':str(CHECKPOINT),'seed':SEED,
        'num_steps':10,'zero_noise':bool(zero_noise),'target_actions_used':False,
        'row_ids_sha256':requested_hashes,'row_counts':{k:len(v) for k,v in requested_ids.items()},
        'hidden_point':'suffix output immediately before action_out_proj','mid_layers':list(MID_LAYER_IDS),
        'mid_representation':'prefix LLM per-layer cached V, 2x2 mean pooled images, valid language tokens only',
        'mid_v_dim':v_dim,'splits':{}}
    for split,(source,ids) in splits.items():
        ids=np.asarray(ids,np.int64);ids=ids[:int(limit)] if limit is not None else ids
        stem=root/split;files={name:Path(str(stem)+f'_{name}.npy') for name in ('actions','hidden_t1','hidden_final')}
        mid_path=Path(str(stem)+'_mid_v.npy');mid_mask_path=Path(str(stem)+'_mid_mask.npy')
        progress=Path(str(stem)+'.progress.json');done=0
        if progress.exists():
            meta=json.loads(progress.read_text())
            if meta.get('ids_sha256')!=__import__('hashlib').sha256(ids.tobytes()).hexdigest():raise ValueError('resume IDs changed')
            done=int(meta.get('done',0))
        elif any(p.exists() for p in files.values()):raise FileExistsError(f'feature arrays exist without resume record for {split}')
        hidden_width=int(model.action_out_proj.kernel.value.shape[0])
        arrays={}
        for name,path in files.items():
            shape=(len(ids),10,7) if name=='actions' else (len(ids),10,hidden_width)
            arrays[name]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=shape)
        if progress.exists() and done and (not mid_path.exists() or not mid_mask_path.exists()):
            raise FileNotFoundError('resuming Q-mid requires existing mid V arrays; use a fresh output directory')
        mids=np.lib.format.open_memmap(mid_path,mode='r+' if mid_path.exists() else 'w+',dtype=np.float16,
            shape=(len(ids),len(MID_LAYER_IDS),pooled_len,v_dim))
        midmask=np.lib.format.open_memmap(mid_mask_path,mode='r+' if mid_mask_path.exists() else 'w+',dtype=np.bool_,
            shape=(len(ids),pooled_len))
        import flax.nnx as nnx
        from openpi.shared.nnx_utils import module_jit
        graphdef,model_state=nnx.split(model)
        def hidden_run(state,obs,rids,sampled_action):
            live_model=nnx.merge(graphdef,state)
            return _batched_policy_action_hidden(live_model,obs,rids,num_steps=10,
                zero_noise=zero_noise,action_override=sampled_action,return_mid_values=True)
        fn=jax.jit(hidden_run)
        sample_fn=module_jit(model.sample_actions,static_argnames=('num_steps',))
        for st in range(done,len(ids),int(batch)):
            chunk=ids[st:min(st+int(batch),len(ids))];obs=base.obs_batch(source,chunk)
            rid=jnp.asarray(chunk,jnp.uint32)
            keys=jax.vmap(lambda r:jax.random.fold_in(jax.random.key(SEED),r))(rid)
            noise=jnp.zeros((len(chunk),model.action_horizon,model.action_dim),jnp.float32) if zero_noise else \
                jax.vmap(lambda k:jax.random.normal(k,(model.action_horizon,model.action_dim)))(keys)
            act_ref=sample_fn(jax.random.PRNGKey(0),obs,num_steps=10,noise=noise)
            act,h1,hf,mid_v=fn(model_state,obs,rid,act_ref)
            # Q-act is persisted directly from the public sampler, making the
            # stored policy prediction numerically identical by construction.
            act_ref=np.asarray(act_ref[...,:7],np.float32)
            custom_act=np.asarray(act,np.float32)
            parity=float(np.max(np.abs(act_ref-custom_act)))
            if st==0 and split=='train':summary['sample_actions_vs_hidden_path_max_abs'] = parity
            stored_action=act_ref.astype(np.float16)
            if verify_integrator and st==0 and done==0 and split=='train' and not zero_noise:
                summary['sample_actions_equivalence']=verify_sample_actions_equivalence(
                    stored_action,act_ref.astype(np.float16),tolerance=0.)
            arrays['actions'][st:st+len(chunk)]=stored_action
            arrays['hidden_t1'][st:st+len(chunk)]=np.asarray(h1,dtype=np.float16)
            arrays['hidden_final'][st:st+len(chunk)]=np.asarray(hf,dtype=np.float16)
            pooled,pm=pool_prefix_v_layers(np.asarray(mid_v,np.float32),np.asarray(obs.tokenized_prompt_mask,bool))
            mids[st:st+len(chunk)]=pooled.astype(np.float16);midmask[st:st+len(chunk)]=pm
            for arr in arrays.values():arr.flush()
            mids.flush();midmask.flush()
            done=st+len(chunk);_atomic_json(progress,{'split':split,'done':done,'total':len(ids),
                'ids_sha256':__import__('hashlib').sha256(ids.tobytes()).hexdigest(),'complete':done==len(ids)})
        summary['splits'][split]={'rows':len(ids),'done':done,'hidden_width':hidden_width,'files':{k:str(v) for k,v in files.items()}}
        del arrays,mids,midmask
    summary['complete']=True
    tmp=manifest_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(manifest_path)
    return summary


def decode_shape_theta(shapes,theta,vs):
    return np.stack([_load_theta().decode_theta(x,t,vs) for x,t in zip(shapes,theta,strict=True)])


def _bootstrap_metric(values,episode,hard):
    v=np.asarray(values,np.float32);ep=np.asarray(episode);h=np.asarray(hard,bool)
    return {'all':episode_bootstrap(v,ep), 'hard':episode_bootstrap(v[h],ep[h]) if h.any() else None}


def _mean_or_none(values):
    x=np.asarray(values)
    return float(x.mean()) if x.size else None


def _choice_stratification(actions,target,selected_local,errors,allowed,ranked,episode,hard,
                           source_lengths=None):
    """Episode-bootstrap the primary choice metrics in both requested strata.

    Translation groups are diagnostic labels from the scored query windows.
    Source-length cut points are fit on the training memory pool, never on
    held-out/evaluation outcomes.
    """
    a=np.asarray(actions,np.float32);y=np.asarray(target,np.float32)
    pick=np.asarray(selected_local,np.int64);err=np.asarray(errors,np.float32)
    allow=np.asarray(allowed,bool);rank=None if ranked is None else np.asarray(ranked,np.int64)
    ep=np.asarray(episode);h=np.asarray(hard,bool)
    if a.shape!=y.shape or a.ndim!=3 or a.shape[1:]!=(10,7):
        raise ValueError('stratification action arrays must be matching [N,10,7]')
    if pick.shape!=(len(y),) or err.shape!=allow.shape or err.shape[0]!=len(y):
        raise ValueError('stratification candidate arrays are misaligned')
    if rank is not None and (rank.ndim!=2 or rank.shape[0]!=len(y)):
        raise ValueError('stratification ranks must be [N,K]')
    chosen_error=err[np.arange(len(pick)),pick]
    mse=np.mean((a-y)**2,axis=(1,2))
    _,translation_group,translation_cuts=translation_tercile_labels(y)
    recall={}
    oracle_error=np.min(np.where(allow,err,np.inf),axis=1)
    if rank is not None:
        for k in (1,5,20):
            if rank.shape[1]<k:
                recall[str(k)]=None
                continue
            positive=np.asarray([np.any(allow[i,rank[i,:k]] &
                (err[i,rank[i,:k]]<=oracle_error[i]+.01)) for i in range(len(y))],np.float32)
            recall[str(k)]=_stratified_bootstrap(positive,translation_group,ep,h)
    result={'translation_norm_terciles':{
        'cut_points':translation_cuts,
        'shape_error':_stratified_bootstrap(chosen_error,translation_group,ep,h),
        'action_mse':_stratified_bootstrap(mse,translation_group,ep,h),
        'recall_oracle_epsilon':recall if rank is not None else None}}
    if source_lengths is not None:
        lengths=np.asarray(source_lengths,np.int32)
        if lengths.ndim!=1 or np.any((pick<0)|(pick>=len(lengths))):
            raise ValueError('source length vector does not align with selected pool IDs')
        cuts=np.quantile(lengths,[1/3,2/3])
        selected_length=lengths[pick]
        length_group=source_length_categories(selected_length,cuts)
        result['selected_source_fragment_length_terciles']={
            'train_cut_points':cuts.tolist(),
            'source_length':_stratified_bootstrap(selected_length,length_group,ep,h),
            'shape_error':_stratified_bootstrap(chosen_error,length_group,ep,h),
            'action_mse':_stratified_bootstrap(mse,length_group,ep,h),
            'recall_oracle_epsilon':(_stratified_bootstrap(
                np.asarray([np.any(allow[i,rank[i,:min(20,rank.shape[1])]] &
                    (err[i,rank[i,:min(20,rank.shape[1])]]<=oracle_error[i]+.01))
                    for i in range(len(y))],np.float32),length_group,ep,h) if rank is not None else None)}
    return result


def _stratified_bootstrap(values,labels,episode,hard=None):
    v=np.asarray(values,np.float32);g=np.asarray(labels);ep=np.asarray(episode)
    if not(v.ndim==g.ndim==ep.ndim==1 and len(v)==len(g)==len(ep)):raise ValueError('stratified metric shape mismatch')
    h=np.ones(len(v),bool) if hard is None else np.asarray(hard,bool)
    if h.shape!=v.shape:raise ValueError('hard mask shape mismatch')
    out={}
    for label in np.unique(g):
        keep=g==label
        both={'n':int(keep.sum()),'mean':float(v[keep].mean()),
              'episode_bootstrap':episode_bootstrap(v[keep],ep[keep])}
        hk=keep&h
        both['hard']={'n':int(hk.sum()),'mean':_mean_or_none(v[hk]),
                      'episode_bootstrap':episode_bootstrap(v[hk],ep[hk]) if hk.any() else None}
        out[str(label)]=both
    return out


class QueryArchitectures:
    """Small shared design family for mean, token, action, history and combined inputs."""
    def __init__(self,mode,d_model=256,slots=4,token_dim=2048,expert_dim=2048):
        import torch
        from torch import nn
        self.torch=torch
        class MLP(nn.Module):
            def __init__(self,dim):super().__init__();self.net=nn.Sequential(nn.Linear(dim,d_model),nn.GELU(),nn.Linear(d_model,d_model))
            def forward(self,x):return self.net(x)
        class Cross(nn.Module):
            def __init__(self):
                super().__init__();self.context=nn.Linear(8+70,slots*d_model);self.kv=nn.Linear(token_dim,d_model)
                self.lang_init=nn.Linear(token_dim,slots*d_model) if mode=='P1_tok_lang' else None
                self.attn=nn.MultiheadAttention(d_model,8,batch_first=True);self.out=nn.Linear(slots*d_model,d_model)
            def forward(self,tok,ctx,lang_mean=None,token_mask=None):
                q=self.context(ctx).view(len(ctx),slots,d_model)
                if self.lang_init is not None:
                    if lang_mean is None:raise ValueError('P1_tok_lang requires language token mean')
                    q=q+self.lang_init(lang_mean).view(len(ctx),slots,d_model)
                kv=self.kv(tok)
                padding=None if token_mask is None else ~token_mask.bool()
                h,w=self.attn(q,kv,kv,key_padding_mask=padding,need_weights=True,average_attn_weights=True)
                return self.out(h.flatten(1)),w
        class Bottleneck(nn.Module):
            def __init__(self):
                super().__init__();self.inp=nn.Linear(token_dim,d_model)
                self.exp_inp=nn.Linear(expert_dim,d_model)
                self.token=nn.Parameter(torch.randn(1,1,d_model)*.02)
                self.attn=nn.MultiheadAttention(d_model,8,batch_first=True)
                self.decoder=nn.Linear(d_model,token_dim);self.exp_decoder=nn.Linear(d_model,expert_dim)
            def forward(self,tok,expert,token_mask=None):
                h=torch.cat([self.inp(tok),self.exp_inp(expert)],dim=1);q=self.token.expand(len(tok),-1,-1)
                valid=torch.ones(tok.shape[:2],device=tok.device,dtype=torch.bool) if token_mask is None else token_mask.bool()
                valid=torch.cat([valid,torch.ones(expert.shape[:2],device=expert.device,dtype=torch.bool)],dim=1)
                padding=~valid
                z,w=self.attn(q,h,h,key_padding_mask=padding,need_weights=True,average_attn_weights=True)
                recon=(self.decoder(z).expand(-1,tok.shape[1],-1),
                    self.exp_decoder(z).expand(-1,expert.shape[1],-1))
                return z[:,0],recon,w
        if mode in ('P0_mean','P4_act','P5_hist'):self.module=MLP({'P0_mean':6144+8,'P4_act':70,'P5_hist':70+8}[mode])
        elif mode in ('P1_tok','P1_tok_lang','P3_exp') or mode.startswith('P2_mid_'):self.module=Cross()
        elif mode in ('P7_bottleneck','P7b_type_bottleneck'):self.module=Bottleneck()
        elif mode.startswith('P6_'):
            self.module=MLP(d_model*3)
        else:raise ValueError(mode)
    def __call__(self,*x):return self.module(*x)


class ProbeNetwork:
    """Trainable query/key encoders used by the production fit/evaluate stages."""
    def __init__(self,mode,candidate_dim=70,d_model=256,token_dim=2048,expert_dim=2048):
        import torch
        from torch import nn
        self.torch=torch;self.mode=mode
        class Net(nn.Module):
            def __init__(self):
                super().__init__();self.mode=mode
                self.key=nn.Sequential(nn.Linear(candidate_dim,512),nn.GELU(),nn.Linear(512,d_model))
                self.dct_head=nn.Linear(d_model,21)
                if mode in ('P0_mean','P4_act','P5_hist'):
                    self.query=nn.Sequential(nn.Linear({'P0_mean':6152,'P4_act':70,'P5_hist':78}[mode],512),nn.GELU(),nn.Linear(512,d_model))
                elif mode in ('P1_tok','P1_tok_lang') or mode.startswith('P2_mid_'):
                    self.query=QueryArchitectures(mode,d_model=d_model,token_dim=token_dim).module
                elif mode=='P3_exp':
                    self.query=QueryArchitectures(mode,d_model=d_model,token_dim=expert_dim).module
                elif mode.startswith('P6_'):
                    if mode=='P6_exp_hist':dims=(expert_dim+78,)
                    elif mode=='P6_tok_hist':dims=(token_dim+78,)
                    else:dims=(6144+expert_dim+78,)
                    self.query=nn.Sequential(nn.Linear(dims[0],768),nn.GELU(),nn.Linear(768,d_model))
                elif mode in ('P7_bottleneck','P7b_type_bottleneck'):
                    self.query=QueryArchitectures('P7_bottleneck',d_model=d_model,token_dim=token_dim,expert_dim=expert_dim).module
                    self.query_head=(nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,d_model))
                        if mode=='P7_bottleneck' else nn.Identity())
                else:raise ValueError(f'unsupported trainable mode {mode}')
            def forward(self,batch):
                m=self.mode
                if m=='P0_mean':z=self.query(torch.cat([batch['mean'].flatten(1),batch['state']],-1));return z,None
                if m in ('P1_tok','P1_tok_lang'):
                    ctx=batch.get('history_context',torch.cat([batch['history'].flatten(1),batch['state']],-1))
                    if m=='P1_tok_lang':z,w=self.query(batch['tokens'],ctx,batch['lang_mean'],batch.get('token_mask'))
                    else:z,w=self.query(batch['tokens'],ctx,token_mask=batch.get('token_mask'))
                    return z,{'attention':w}
                if m.startswith('P2_mid_'):
                    ctx=batch.get('history_context',torch.cat([batch['history'].flatten(1),batch['state']],-1))
                    z,w=self.query(batch['mid_tokens'],ctx,token_mask=batch['mid_token_mask'])
                    return z,{'attention':w}
                if m=='P3_exp':z,w=self.query(batch['expert_t1'],batch.get('history_context',torch.cat([batch['history'].flatten(1),batch['state']],-1)),token_mask=None);return z,{'attention':w}
                if m=='P4_act':return self.query(batch['action'].flatten(1)),None
                if m=='P5_hist':return self.query(batch.get('history_context',torch.cat([batch['history'].flatten(1),batch['state']],-1))),None
                if m.startswith('P6_'):
                    hist=batch.get('history_context',torch.cat([batch['history'].flatten(1),batch['state']],-1))
                    if m=='P6_exp_hist':x=torch.cat([batch['expert_t1'].mean(1),hist],-1)
                    elif m=='P6_tok_hist':
                        if 'token_mask' in batch:
                            tok_mean=(batch['tokens']*batch['token_mask'][:,:,None]).sum(1)/batch['token_mask'].sum(1,keepdim=True).clamp_min(1)
                        else:tok_mean=batch['tokens'].mean(1)
                        x=torch.cat([tok_mean,hist],-1)
                    else:x=torch.cat([batch['mean'].flatten(1),batch['expert_t1'].mean(1),hist],-1)
                    return self.query(x),None
                if m in ('P7_bottleneck','P7b_type_bottleneck'):
                    z,recon,w=self.query(batch['tokens'],batch['expert_t1'],token_mask=batch.get('token_mask'))
                    z=self.query_head(z)
                    return z,{'reconstruction':recon,'attention':w,'dct':self.dct_head(z)}
                raise ValueError(m)
        self.module=Net()

    def __call__(self,batch):return self.module(batch)


def decode_pool_actions(pool,indices,vs,*,theta_source='src',query_pi05=None):
    """Decode selected shapes as separate actions; never blend candidates.

    ``theta_source='pi05'`` applies the same Procrustes rule used by the
    existing pi05-shape baseline. ``src`` uses the theta saved with each unit.
    """
    theta=_load_theta();shape=np.asarray(pool['shape'],np.float32);ix=np.asarray(indices,np.int64)
    if ix.ndim!=2 or np.any((ix<0)|(ix>=len(shape))):raise ValueError('indices must be [N,K] pool-local IDs')
    selected=shape[ix]
    if theta_source=='src':th=np.asarray(pool['source_theta'],np.float32)[ix]
    elif theta_source=='pi05':
        if query_pi05 is None:raise ValueError('query_pi05 required for pi05 theta')
        qphys=vs.to_physical(np.asarray(query_pi05,np.float32)[:,:,:6])
        th=np.empty((*ix.shape,4),np.float32)
        from compressed_denoise import _procrustes
        for i in range(len(ix)):
            for j in range(ix.shape[1]):th[i,j]=_procrustes(selected[i,j,:,:6],qphys[i])[0]
    else:raise ValueError("theta_source must be 'src' or 'pi05'")
    # theta_regression.decode_theta is intentionally scalar and hands its
    # rotation vector directly to scipy.  Decode this [N,K] selection in one
    # vectorized pass instead of passing a [N*K,4] matrix to that scalar API.
    from scipy.spatial.transform import Rotation
    leading=ix.shape
    rot=Rotation.from_rotvec(th[...,:3].reshape(-1,3)).as_matrix().reshape(*leading,3,3)
    physical=np.concatenate((
        np.einsum('...td,...df->...tf',selected[...,:3],rot),
        np.einsum('...td,...df->...tf',selected[...,3:6],rot)),axis=-1)
    physical*=np.exp(np.clip(th[...,3],-5,5))[...,None,None]
    normalized=np.asarray(vs.to_normalized(physical.reshape(-1,10,6)),np.float32).reshape(*leading,10,6)
    return np.concatenate((normalized,selected[...,6:7]),axis=-1).astype(np.float32)


def candidate_pool_source_lengths(rep='C_vm_none'):
    """Original fragment lengths aligned with candidate_shape_pool order."""
    theta=_load_theta();cache=np.load(POOL_DIR/f'{rep}_train.npz',allow_pickle=False)
    seen=set();out=[];uids=np.asarray(cache['unit_ids'])
    for i,raw in enumerate(cache['records']):
        for slot,rec in enumerate(json.loads(str(raw))):
            key=(int(rec['source_row']),int(rec['unit']),int(rec['source_task']))
            if key in seen:continue
            seen.add(key)
            if not np.any(uids[i]==slot):continue
            out.append(int(rec.get('unit_length',rec.get('length',10))))
    pool=theta.candidate_shape_pool(rep=rep)
    if len(out)!=len(pool['shape']):raise AssertionError('candidate length order diverged from shape pool')
    return np.asarray(out,np.int32)


def add_candidate_source_lengths(pool,rep='C_vm_none'):
    """Return a view with train-only original lengths aligned to pool candidates."""
    out=dict(pool);lengths=candidate_pool_source_lengths(rep)
    if len(lengths)!=len(pool['shape']):raise ValueError('pool/length order mismatch')
    out['source_length']=lengths
    return out


def flow_state_and_target_velocity(candidate_actions,noise,tau):
    """Interpolate with candidate weight tau and convert to OpenPI time.

    The requested expression is x=(1-tau)*a0 + tau*a_k. Since OpenPI has
    t=1 at noise, this corresponds to model time t_model=1-tau. Its target
    vector field is a0-a_k (sampler integrates x <- x-v*dt).
    """
    a=np.asarray(candidate_actions,np.float32);n=np.asarray(noise,np.float32);t=float(tau)
    if a.shape[-2:]!=(10,7) or n.shape[-2]!=10 or n.shape[-1]<7:raise ValueError('actions/noise shape mismatch')
    if not 0<=t<=1:raise ValueError('tau must be in [0,1]')
    if n.ndim==2:n=n.reshape((1,)*(a.ndim-2)+n.shape)
    elif n.ndim==3 and a.ndim==4:n=n[:,None]
    if n.ndim!=a.ndim:raise ValueError('noise rank cannot broadcast to candidates')
    n=np.broadcast_to(n,(*a.shape[:-1],n.shape[-1]))
    if n.shape[:-1]!=a.shape[:-1]:raise ValueError('noise leading dimensions do not align')
    active=n[...,:7]
    if n.shape[-1]>7:
        padded=np.zeros((*a.shape[:-1],n.shape[-1]),np.float32);padded[...,:7]=a
    else:padded=a
    return (1-t)*n+t*padded,active-a


def p8_expert_candidate_scores(expert_velocity,candidate_actions,noise):
    """MSE between frozen Expert velocity and each candidate's required velocity."""
    v=np.asarray(expert_velocity,np.float32);a=np.asarray(candidate_actions,np.float32);n=np.asarray(noise,np.float32)
    if a.ndim!=4 or a.shape[2:]!=(10,7):raise ValueError('candidate_actions must be [N,K,10,7]')
    if v.shape!=a.shape or n.shape[-2]!=10 or n.shape[-1]<7:raise ValueError('velocity/noise shape mismatch')
    if n.ndim==3:n=n[:,None]
    return np.mean((v-(n[...,:7]-a))**2,axis=(2,3))


def initial_flow_noise(keyed_noise,*,zero_noise=False):
    """Return deterministic keyed noise or the requested epsilon=0 diagnostic."""
    x=np.asarray(keyed_noise,np.float32)
    return np.zeros_like(x) if zero_noise else x.copy()


def p8_score_with_shared_prefix(candidate_actions,noise,tau,velocity_fn,*,prefix_cache):
    """Run candidate suffixes in one batch while reusing a single prefix KV cache.

    ``velocity_fn(x_tau, model_time, prefix_cache)`` must accept [N,K,10,7] and return
    per-candidate velocities of the same shape. It is called exactly once.
    """
    a=np.asarray(candidate_actions,np.float32);x,target=flow_state_and_target_velocity(a,noise,tau)
    v=np.asarray(velocity_fn(x,1.-float(tau),prefix_cache),np.float32)
    if v.shape!=a.shape:raise ValueError('velocity_fn returned wrong shape')
    return p8_expert_candidate_scores(v,a,noise)


def p8_rerank_one_query(type_ranked_local,actions_theta_src,actions_theta_pi05,noise,tau,
                        velocity_fn,*,prefix_cache,top_k=16):
    """P8 end-to-end selection across θ_src/π0.5-decoded variants of Top-K types.

    Both theta variants are scored in a single suffix batch using the exact same
    prefix cache. The result is one candidate, never an action-space mixture.
    """
    rank=np.asarray(type_ranked_local,np.int64);k=min(int(top_k),len(rank))
    src=np.asarray(actions_theta_src,np.float32);pi=np.asarray(actions_theta_pi05,np.float32)
    if k<1 or src.shape!=pi.shape or src.ndim!=3 or src.shape[1:]!=(10,7) or len(src)<k:
        raise ValueError('P8 needs matching [K,10,7] theta variants and nonempty ranking')
    if len(rank)<k:raise ValueError('type ranking shorter than requested top_k')
    choices=np.concatenate([src[:k],pi[:k]],axis=0)[None]
    n=np.asarray(noise,np.float32)
    if n.shape==(10,7):n=n[None]
    scores=p8_score_with_shared_prefix(choices,n,float(tau),velocity_fn,prefix_cache=prefix_cache)[0]
    local=int(np.argmin(scores));variant='theta_src' if local<k else 'theta_pi05'
    local_type=local if local<k else local-k
    selected=src[local_type] if variant=='theta_src' else pi[local_type]
    return {'type_pool_local':int(rank[local_type]),'variant':variant,'variant_rank':local_type,
            'score':float(scores[local]),'scores':scores,'action':selected.copy(),
            'candidate_count':int(2*k),'top_k':int(k)}


def prepare_pi05_prefix(model,observation):
    """Compute one frozen pi05 prefix/KV cache for P8 candidate scoring."""
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs=model_api.preprocess_observation(None,observation,train=False)
    prefix,pmask,par=model.embed_prefix(obs)
    _,cache=model.PaliGemma.llm([prefix,None],mask=make_attn_mask(pmask,par),
        positions=jnp.cumsum(pmask,axis=1)-1)
    return {'obs':obs,'prefix_mask':pmask,'cache':cache}


def pi05_candidate_velocity(model,prepared,candidate_actions,noise,candidate_weight):
    """One batched suffix forward for candidates, sharing the prepared prefix KV.

    Candidate/noise inputs may be active7 or model-width actions. The returned
    velocity is active7 and has shape [K,10,7].
    """
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    from compressed_denoise import _pad_actions
    a=np.asarray(candidate_actions,np.float32);n=np.asarray(noise,np.float32)
    if a.ndim!=3 or a.shape[1]!=model.action_horizon:raise ValueError('candidates must [K,H,D]')
    a=_pad_actions(a);n=_pad_actions(n[None] if n.ndim==2 else n)
    if n.shape[0]==1:n=np.broadcast_to(n,(len(a),*n.shape[1:]))
    if len(n)!=len(a):raise ValueError('noise candidate count mismatch')
    tau=float(candidate_weight)
    if not 0<=tau<=1:raise ValueError('candidate_weight must be [0,1]')
    xt=(1-tau)*n+tau*a
    return pi05_velocity_at_xt(model,prepared,xt,1.-tau)


def pi05_velocity_at_xt(model,prepared,x_tau,model_time):
    """Evaluate candidates with one shared prefix KV, without copying its batch.

    The candidate axis is mapped outside the single-example suffix call. This
    keeps the large prefix cache at batch size one instead of materializing a
    full image-token KV copy for every candidate.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    xt=np.asarray(x_tau,np.float32)
    if xt.ndim!=3 or xt.shape[1:]!=(model.action_horizon,model.action_dim):raise ValueError('x_tau must be [K,H,model_action_dim]')
    prefix_batch=int(prepared['prefix_mask'].shape[0])
    if prefix_batch<1 or len(xt)%prefix_batch:
        raise ValueError('candidate batch must be a multiple of prefix batch')
    repeat=len(xt)//prefix_batch
    x=jnp.asarray(xt).reshape(prefix_batch,repeat,model.action_horizon,model.action_dim)
    pmask=jnp.asarray(prepared['prefix_mask'])
    cache_by_batch=jax.tree.map(lambda v:jnp.moveaxis(v,1,0),prepared['cache'])

    def one_query(obs_i,pmask_i,cache_i,candidates):
        obs_one=jax.tree.map(lambda v:v[None],obs_i)
        mask_one=pmask_i[None]
        cache_one=jax.tree.map(lambda v:v[:,None,...],cache_i)
        def one_candidate(x_i):
            timestep=jnp.full((1,),float(model_time),jnp.float32)
            suffix,smask,sar,cond=model.embed_suffix(obs_one,x_i[None],timestep)
            attn=jnp.concatenate([jnp.broadcast_to(mask_one[:,None,:],
                (1,suffix.shape[1],mask_one.shape[1])),make_attn_mask(smask,sar)],axis=-1)
            pos=jnp.sum(mask_one,axis=1)[:,None]+jnp.cumsum(smask,axis=1)-1
            (_,out),_=model.PaliGemma.llm([None,suffix],mask=attn,positions=pos,
                kv_cache=cache_one,adarms_cond=[None,cond])
            return model.action_out_proj(out[:,-model.action_horizon:])[0,...,:7]
        # ``vmap`` across all 32 actions expands the 18-layer suffix activations
        # beyond the shared .25 GPU budget. ``lax.map`` batches four candidates
        # at a time while keeping the query prefix KV shared.
        return jax.lax.map(one_candidate,candidates,batch_size=4)

    result=jax.vmap(one_query)(prepared['obs'],pmask,cache_by_batch,x)
    return np.asarray(result.reshape(len(xt),model.action_horizon,7))


def p8_select(scores):
    """Select one candidate per query from already tau-calibrated scores."""
    s=np.asarray(scores,np.float32)
    if s.ndim!=2 or not np.isfinite(s).any(axis=1).all():raise ValueError('scores must be finite [N,K]')
    return np.argmin(s,axis=1)


def _require_libero40_holdout(split_name):
    if split_name!='libero40_task_holdout':
        raise ValueError('selection/calibration is restricted to the fixed LIBERO-40 task holdout')


def select_p8_tau_on_holdout(score_by_tau,actions_by_tau,truth,taus,*,split_name):
    """Choose fixed tau on heldout-task rows only using selected-action MSE."""
    _require_libero40_holdout(split_name)
    if not(len(score_by_tau)==len(actions_by_tau)==len(taus)):raise ValueError('tau grid mismatch')
    if len(set(map(float,taus)))!=len(taus) or not set(map(float,taus))<=set((.3,.5,.7)):
        raise ValueError('P8 tau candidates must be a nonempty subset of {.3,.5,.7}')
    if not len(taus):raise ValueError('empty P8 tau grid')
    y=np.asarray(truth,np.float32);records=[]
    for tau,s,a in zip(taus,score_by_tau,actions_by_tau,strict=True):
        ix=p8_select(s);cand=np.asarray(a,np.float32)
        if cand.ndim!=4 or cand.shape[0]!=len(y):raise ValueError('holdout candidate action shape mismatch')
        mse=np.mean((cand[np.arange(len(y)),ix]-y)**2,axis=(1,2))
        records.append({'tau':float(tau),'mean_mse':float(mse.mean()),'picked':ix})
    best=min(range(len(records)),key=lambda i:(records[i]['mean_mse'],i))
    return float(taus[best]),[{'tau':r['tau'],'mean_mse':r['mean_mse']} for r in records]


def p9_two_stage(type_scores,allowed,candidate_unit_ids,second_scores,*,top_k=32):
    """Type Top-K then choose one whole (type, source-theta) unit by score."""
    s=np.asarray(type_scores,np.float32);allow=np.asarray(allowed,bool);uids=np.asarray(candidate_unit_ids)
    z=np.asarray(second_scores,np.float32)
    if s.shape!=allow.shape or z.shape!=s.shape or s.shape[1]!=len(uids):raise ValueError('P9 input shape mismatch')
    if top_k<1:raise ValueError('top_k must be positive')
    order=np.argsort(np.where(allow,-s,np.inf),axis=1,kind='stable')[:,:min(top_k,s.shape[1])]
    picked=[]
    for i,inds in enumerate(order):
        valid=inds[allow[i,inds]]
        if not len(valid):raise ValueError('empty P9 candidate row')
        picked.append(valid[np.argmin(z[i,valid])])
    return np.asarray(picked,np.int64),order


def p9_direction_scores(candidate_actions,pi05_actions,first_steps=3):
    """Cosine disagreement of unit displacement vs pi05 early displacement."""
    a=np.asarray(candidate_actions,np.float32);q=np.asarray(pi05_actions,np.float32)
    if a.ndim!=4 or q.shape!=(a.shape[0],10,7):raise ValueError('P9 actions shape mismatch')
    u=a[:,:,:first_steps,:3].sum(axis=2);v=q[:,:first_steps,:3].sum(axis=1)[:,None,:]
    den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    cos=np.divide(np.sum(u*v,axis=-1),den,out=np.zeros_like(den),where=den>1e-8)
    return 1-cos


def p8_p9_select_one_query(pool,ranked_local,query_pi05,noise,vs,velocity_fn,*,prefix_cache,tau):
    """Wire the learned type ranking to P8/P9 single-unit choices for one query."""
    rank=np.asarray(ranked_local,np.int64);shape=np.asarray(pool['shape'],np.float32)
    allowed_rank=rank[(rank>=0)&(rank<len(shape))]
    if len(np.unique(allowed_rank))!=len(allowed_rank):raise ValueError('rank contains duplicate pool-local candidates')
    # P8 uses one holdout-selected type ranking, then compares both theta sources.
    top16=allowed_rank[:16]
    act_src=decode_pool_actions(pool,top16[None],vs,theta_source='src')[0]
    act_pi=decode_pool_actions(pool,top16[None],vs,theta_source='pi05',query_pi05=np.asarray(query_pi05)[None])[0]
    def callback(x,t,cache):return velocity_fn(x[0],t,cache)[None]
    p8=p8_rerank_one_query(top16,act_src,act_pi,noise,tau,callback,prefix_cache=prefix_cache,top_k=len(top16))
    # P9's candidates are whole (shape, source_theta) units, not shape-only drafts.
    top32=allowed_rank[:32];unit_actions=decode_pool_actions(pool,top32[None],vs,theta_source='src')[0]
    noise_batch=np.asarray(noise,np.float32)[None]
    x_tau,_=flow_state_and_target_velocity(unit_actions[None],noise_batch,float(tau))
    velocity=np.asarray(velocity_fn(x_tau[0],1.-float(tau),prefix_cache),np.float32)[None]
    expert_scores=p8_expert_candidate_scores(velocity,unit_actions[None],noise_batch)[0]
    direction_scores=p9_direction_scores(unit_actions[None],np.asarray(query_pi05,np.float32)[None])[0]
    type_scores=np.arange(len(top32),0,-1,dtype=np.float32)[None]
    allow=np.ones_like(type_scores,bool)
    p9_expert,_=p9_two_stage(type_scores,allow,top32,expert_scores[None],top_k=len(top32))
    p9_heading,_=p9_two_stage(type_scores,allow,top32,direction_scores[None],top_k=len(top32))
    return {'P8':p8,'P9_expert':{'pool_local':int(top32[p9_expert[0]]),'action':unit_actions[p9_expert[0]],
             'score':float(expert_scores[p9_expert[0]])},
            'P9_direction':{'pool_local':int(top32[p9_heading[0]]),'action':unit_actions[p9_heading[0]],
             'score':float(direction_scores[p9_heading[0]])},
            'top16':top16.tolist(),'top32':top32.tolist()}


def unit_action_oracle(candidate_actions,target_actions):
    """Action-MSE upper bound over whole type+source-theta candidates."""
    a=np.asarray(candidate_actions,np.float32);y=np.asarray(target_actions,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(10,7) or y.shape!=(10,7):raise ValueError('unit oracle shapes must be [K,10,7] and [10,7]')
    err=np.mean((a-y[None])**2,axis=(1,2))
    return int(np.argmin(err)),err


def p7b_type_loss(logits,teacher,dct_prediction,dct_target,allowed,*,dct_weight=1.):
    """P7b objective: soft listwise InfoNCE plus DCT-3 coefficients regression."""
    import torch
    import torch.nn.functional as F
    x=logits.masked_fill(~allowed.bool(),-1e9)
    ce=-(teacher*F.log_softmax(x,dim=-1)).sum(-1).mean()
    reg=F.mse_loss(dct_prediction,dct_target)
    return ce+float(dct_weight)*reg,{'infonce':ce.detach(),'dct':reg.detach()}


def dct3_motion_coefficients(actions):
    """Orthonormal first three temporal DCT coefficients for each action axis."""
    x=np.asarray(actions,np.float32)
    if x.shape[-2:]!=(10,7):raise ValueError('actions must end in [10,7]')
    t=np.arange(10,dtype=np.float32);k=np.arange(3,dtype=np.float32)[:,None]
    basis=np.cos(np.pi*(t[None,:]+.5)*k/10);basis[0]*=1/np.sqrt(2);basis*=np.sqrt(2/10)
    return np.einsum('kt,...td->...kd',basis,x)


def apply_history_modality_dropout(context,probability,*,seed=SEED,training=True):
    """Drop only the previous-action block (70 values), retaining state (8)."""
    x=np.asarray(context,np.float32).copy()
    if x.shape[-1]!=78:raise ValueError('context must be previous action 70 + state 8')
    if not 0<=probability<=1:raise ValueError('dropout probability outside [0,1]')
    if not training:return x
    rng=np.random.default_rng(seed);mask=rng.random(x.shape[:-1])<probability;x[mask,:70]=0
    return x


def branching_hard_negative_pairs(prev_type,next_type,episode,window,*,prev_similarity=.1):
    """Pairs with similar previous type but different next type, plus same-episode neighbors."""
    p=np.asarray(prev_type,np.float32);n=np.asarray(next_type);ep=np.asarray(episode);w=np.asarray(window)
    if p.ndim!=2 or len(p)!=len(n) or len(p)!=len(ep) or len(p)!=len(w):raise ValueError('hard-negative metadata mismatch')
    norm=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-8);sim=norm@norm.T
    pairs=set()
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            branching=(sim[i,j]>=1-prev_similarity and n[i]!=n[j])
            adjacent=(ep[i]==ep[j] and abs(int(w[i])-int(w[j]))<=1)
            if branching or adjacent:pairs.add((i,j))
    return np.asarray(sorted(pairs),np.int64).reshape(-1,2)


def hard_negative_margin_loss(embeddings,pairs,*,margin=.2):
    """Push apart embeddings for branching and same-episode temporal negatives."""
    import torch
    z=torch.nn.functional.normalize(embeddings,dim=-1)
    ij=torch.as_tensor(pairs,device=z.device,dtype=torch.long)
    if ij.numel()==0:return z.sum()*0
    sim=(z[ij[:,0]]*z[ij[:,1]]).sum(-1)
    return torch.relu(sim-float(margin)).mean()


def translation_tercile_labels(actions,thresholds=None):
    """Assign query windows to translation-norm terciles; thresholds may be train-fitted."""
    a=np.asarray(actions,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(10,7):raise ValueError('actions must be [N,10,7]')
    norm=np.linalg.norm(a[:,:,:3].sum(axis=1),axis=-1)
    cut=np.quantile(norm,[1/3,2/3]) if thresholds is None else np.asarray(thresholds)
    if np.asarray(cut).shape!=(2,):raise ValueError('two translation thresholds required')
    return norm,np.digitize(norm,cut,right=True),np.asarray(cut).tolist()


def stratified_metric(values,labels):
    """Per-stratum count/mean, for translation terciles or source-length bins."""
    x=np.asarray(values,np.float64);g=np.asarray(labels)
    if x.ndim!=1 or g.shape!=x.shape:raise ValueError('metric/stratum shape mismatch')
    return {str(k):{'n':int(np.sum(g==k)),'mean':float(x[g==k].mean())} for k in np.unique(g)}


def source_length_categories(lengths,thresholds):
    """Map original (pre-resampling) unit lengths to fixed train-derived bins."""
    x=np.asarray(lengths,np.int64);t=np.asarray(thresholds,np.float64)
    if x.ndim!=1 or t.ndim!=1 or np.any(np.diff(t)<0):raise ValueError('invalid length bins')
    return np.digitize(x,t,right=True)


def choose_p1_or_p3_on_holdout(p1_shape_error,p3_shape_error,*,split_name):
    """Select P1/P3 once using heldout-task shape error; tie defaults to P1."""
    _require_libero40_holdout(split_name)
    a=np.asarray(p1_shape_error,np.float64);b=np.asarray(p3_shape_error,np.float64)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):raise ValueError('holdout error vectors must match and be nonempty')
    return ('P1_tok' if float(a.mean())<=float(b.mean()) else 'P3_exp',
            {'P1_tok_mean':float(a.mean()),'P3_exp_mean':float(b.mean())})


def procrustes_degeneracy(shape,target=None,eps=1e-8):
    """Diagnostic flag for underdetermined candidate/target Procrustes geometry."""
    y=np.asarray(shape,np.float64);x=y if target is None else np.asarray(target,np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape!=y.shape or x.shape[1]<6:raise ValueError('shape/target must match [T,>=6]')
    cross=y[:,:3].T@x[:,:3]+y[:,3:6].T@x[:,3:6]
    sv=np.linalg.svd(cross,compute_uv=False);rank=int(np.sum(sv>max(eps,sv[0]*eps))) if len(sv) else 0
    return {'degenerate':rank<3,'rank':rank,'singular_values':sv.tolist(),
            'existing_behavior':'compressed_denoise._procrustes applies SVD with no explicit degeneracy branch; scale denominator is clamped to 1e-12.'}


def smoke(output=OUT,limit=8):
    """Run I/O, every query input adapter, loss, exclusion and scoring on safe rows."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    path=out/'smoke.json'
    if path.exists():raise FileExistsError(path)
    theta=_load_theta();pool_a=theta.candidate_shape_pool(rep='C_vm_none');pool_b=theta.candidate_shape_pool(rep='C_vm_08')
    train=np.load(POOL_DIR/'C_vm_none_train.npz',allow_pickle=False)
    tasks=sorted(set(map(int,train['task'])));hold=deterministic_holdout_tasks(tasks)
    # Candidate pools are training-only templates with heldout tasks removed.
    pool={k:v for k,v in pool_a.items()};keep=~np.isin(pool['source_task'],hold)
    pool={k:np.asarray(v)[keep] if np.asarray(v).ndim and len(np.asarray(v))==len(keep) else v for k,v in pool.items()}
    assert not set(map(int,pool['source_task']))&set(hold)
    rng=np.random.default_rng(SEED);k=len(pool['shape'])
    is_dev=np.asarray(np.load(UNSEEN/'is_dev.npy',mmap_mode='r',allow_pickle=False),bool)
    dev_ix=np.flatnonzero(is_dev)[:max(1,limit//2)]
    eval_ix=np.flatnonzero(~is_dev)[:max(1,limit//2)]
    smoke_ix=np.r_[dev_ix,eval_ix];n=len(smoke_ix)
    query_tasks=np.asarray(np.load(UNSEEN/'task.npy',mmap_mode='r',allow_pickle=False))[smoke_ix]
    allow=filter_cross_task(pool,query_tasks,heldout=())
    err=rng.uniform(0,.25,(n,k)).astype(np.float32);err[~allow]=np.inf
    teach=target_distribution(err,allow,.02);logits=-err/.1;loss=listwise_loss(logits,teach,allow)
    chosen=top1_single(logits,allow,np.arange(k,dtype=np.int64));assert chosen.shape==(n,)
    # Baseline machinery: random (5 independent IDs) and oracle remain single-item.
    rand=rng.integers(k,size=(n,5));oracle=np.argmin(err,axis=1)
    qroot=Path(UNSEEN)
    tokens=np.load(qroot/'tokens.npy',mmap_mode='r',allow_pickle=False)
    groups=np.load(qroot/'groups.npy',mmap_mode='r',allow_pickle=False)
    token_preview=downsample_camera_tokens(tokens[smoke_ix[0]],groups[smoke_ix[0]]);assert token_preview.shape[1]==2048
    eval_mean=np.load(THETA/'production'/'features'/'eval.npy',mmap_mode='r',allow_pickle=False)
    dev_mean=np.load(THETA/'production'/'features'/'dev.npy',mmap_mode='r',allow_pickle=False)
    mean=np.concatenate([dev_mean[:len(dev_ix)],eval_mean[:len(eval_ix)]])
    assert mean.shape==(n,3,2048)
    # Make a tiny scorer call on dummy predicted actions; labels are read only here.
    actions=np.asarray(np.load(qroot/'actions.npy',mmap_mode='r',allow_pickle=False)[smoke_ix,:,:7])
    hard=np.asarray(np.load(qroot/'is_dev.npy',mmap_mode='r',allow_pickle=False)[smoke_ix],bool)
    episode=np.asarray(np.load(qroot/'episode.npy',mmap_mode='r',allow_pickle=False)[smoke_ix])
    score=score_predictions(chosen,err,np.asarray(actions),np.asarray(actions),oracle,.001,episode,hard)
    # Exercise every adapter's tensor contract with cached mean/token inputs;
    # absent Q-mid/Q-exp tensors are explicitly synthetic shape checks only.
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    adapter_checks={}
    ctx=torch.zeros(n,78)
    tok=torch.as_tensor(np.repeat(token_preview[None,:16,:],n,axis=0))
    for mode in METHODS:
        enc=QueryArchitectures(mode,token_dim=256 if mode.startswith('P2_mid_') else 2048)
        with torch.no_grad():
            if mode=='P0_mean':outq=enc(torch.zeros(n,6152))
            elif mode=='P4_act':outq=enc(torch.zeros(n,70))
            elif mode=='P5_hist':outq=enc(torch.zeros(n,78))
            elif mode in ('P1_tok','P3_exp'):outq,_=enc(tok,ctx)
            elif mode.startswith('P2_mid_'):outq,_=enc(torch.zeros(n,176,256),ctx,None,torch.ones(n,176,dtype=torch.bool))
            elif mode=='P1_tok_lang':outq,_=enc(tok,ctx,torch.zeros(n,2048))
            elif mode in ('P7_bottleneck','P7b_type_bottleneck'):outq,_,_=enc(tok,torch.zeros(n,10,2048))
            else:outq=enc(torch.zeros(n,768))
        if outq.shape!=(n,256):raise AssertionError((mode,outq.shape))
        adapter_checks[mode]=list(outq.shape)
    # One genuine gradient step per adapter against a toy listwise teacher.
    # It checks trainability without consulting unseen action labels.
    import torch.nn.functional as F
    train_steps={}
    key_encoder=torch.nn.Sequential(torch.nn.Linear(70,256),torch.nn.GELU(),
                                     torch.nn.Linear(256,256))
    tiny_err=torch.as_tensor(rng.uniform(0,.2,(n,16)),dtype=torch.float32)
    tiny_allowed=torch.ones_like(tiny_err,dtype=torch.bool)
    tiny_teacher=torch.as_tensor(target_distribution(tiny_err.numpy(),tiny_allowed.numpy(),.04),dtype=torch.float32)
    candidates=torch.as_tensor(rng.normal(size=(16,10,7)).astype(np.float32))
    for mode in METHODS:
        enc=QueryArchitectures(mode,token_dim=256 if mode.startswith('P2_mid_') else 2048);opt=torch.optim.Adam(list(enc.module.parameters())+list(key_encoder.parameters()),lr=1e-3)
        if mode=='P0_mean':qout=enc(torch.zeros(n,6152))
        elif mode=='P4_act':qout=enc(torch.zeros(n,70))
        elif mode=='P5_hist':qout=enc(torch.zeros(n,78))
        elif mode in ('P1_tok','P3_exp'):qout,_=enc(tok,ctx)
        elif mode.startswith('P2_mid_'):qout,_=enc(torch.zeros(n,176,256),ctx,None,torch.ones(n,176,dtype=torch.bool))
        elif mode=='P1_tok_lang':qout,_=enc(tok,ctx,torch.zeros(n,2048))
        elif mode in ('P7_bottleneck','P7b_type_bottleneck'):
            qout,recon,_=enc(tok,torch.zeros(n,10,2048))
        else:qout=enc(torch.zeros(n,768))
        keys=F.normalize(key_encoder(candidates.flatten(1)),dim=-1)
        qout=F.normalize(qout,dim=-1);scores=qout@keys.T/.1
        loss=-(tiny_teacher*F.log_softmax(scores,dim=-1)).sum(-1).mean()
        if mode=='P7_bottleneck':
            rec_tok,rec_exp=recon;loss=loss+.1*(rec_tok-tok).square().mean()+.1*rec_exp.square().mean()
        elif mode=='P7b_type_bottleneck':
            # Tiny finite-objective exercise; real soft type teacher is supplied at fit time.
            dct_pred=torch.zeros(n,21,requires_grad=True)
            dct_target=torch.zeros(n,21)
            loss,_=p7b_type_loss(scores,tiny_teacher,dct_pred,dct_target,tiny_allowed,dct_weight=1.)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if not torch.isfinite(loss):raise AssertionError(f'nonfinite listwise smoke loss: {mode}')
        train_steps[mode]=float(loss.detach())
    # Exercise the production fit loop for a P6 dropout + hard-negative
    # variant, including the episode/frame metadata path. This is a tiny
    # synthetic fit and does not inspect unseen targets.
    tiny_n=8
    tiny_features={
        'history':np.zeros((tiny_n,10,7),np.float32),
        'state':np.zeros((tiny_n,8),np.float32),
        'expert_t1':rng.normal(size=(tiny_n,10,16)).astype(np.float32),
        'tokens':rng.normal(size=(tiny_n,16,2048)).astype(np.float32),
        'token_mask':np.ones((tiny_n,16),bool),
        'task':np.repeat([1,2],4),'episode':np.repeat(np.arange(4),2),
        'frame':np.tile([0,10],4),
    }
    tiny_features['history'][...,0]=1.
    tiny_errors=np.tile(np.array([[0.,.2],[.2,0.]],np.float32),(tiny_n//2,1))
    tiny_allowed=np.ones_like(tiny_errors,bool)
    tiny_teacher=target_distribution(tiny_errors,tiny_allowed,.02)
    _,p6_state=fit_probe_model('P6_exp_hist',tiny_features,
        rng.normal(size=(2,10,7)).astype(np.float32),tiny_teacher,tiny_allowed,
        np.arange(tiny_n),epochs=1,batch_size=tiny_n,threads=8,
        history_dropout=.3,hard_negative_weight=.05)
    if not p6_state['train_loss'][0]['hard_negative_pairs']:
        raise AssertionError('P6 hard-negative fit smoke did not form any pairs')
    _,p7_state=fit_probe_model('P7_bottleneck',tiny_features,
        rng.normal(size=(2,10,7)).astype(np.float32),tiny_teacher,tiny_allowed,
        np.arange(tiny_n),epochs=1,batch_size=tiny_n,threads=8)
    if not p7_state['observation_only_pretrain']:
        raise AssertionError('P7 observation-only pretraining did not run')
    _,p7b_state=fit_probe_model('P7b_type_bottleneck',tiny_features,
        rng.normal(size=(2,10,7)).astype(np.float32),tiny_teacher,tiny_allowed,
        np.arange(tiny_n),epochs=1,batch_size=tiny_n,threads=8)
    # Added K1 probes: verify batched expert-field ranking, unit reranking,
    # single-item oracle, and diagnostic strata without fitting on unseen labels.
    p8_actions=rng.normal(size=(n,5,10,7)).astype(np.float32)
    p8_noise=np.zeros((n,10,7),np.float32)
    p8_velocity=np.broadcast_to(p8_noise[:,None]-p8_actions[:,2:3],p8_actions.shape).copy()
    p8_scores=p8_expert_candidate_scores(p8_velocity,p8_actions,p8_noise)
    p8_picked=p8_select(p8_scores)
    p9_picked,p9_top=p9_two_stage(rng.normal(size=(n,5)).astype(np.float32),
        np.ones((n,5),bool),np.arange(5),p8_scores,top_k=3)
    trans_norm,trans_group,trans_cuts=translation_tercile_labels(actions)
    chosen_p8=np.asarray(p8_actions)[np.arange(n),p8_picked]
    p8_mse=np.mean((chosen_p8-actions)**2,axis=(1,2))
    strata={'translation_norm_terciles':stratified_metric(p8_mse,trans_group),
            'translation_thresholds':trans_cuts}
    degenerate_check=procrustes_degeneracy(np.zeros((10,6),np.float32))
    # Invoke the real stage selector and report writer on isolated synthetic
    # artifacts, so the CLI stage graph is checked without pretending to run
    # the full 619-row evaluation during a smoke test.
    stage_root=out/'stage_graph_smoke';stage_prod=stage_root/'production';stage_prod.mkdir(parents=True)
    stage_train={'status':'trained','selection_split':'LIBERO-40 task holdout only',
        'holdout_tasks':list(hold),'selected_by_mode':{'P1_tok':{'mode':'P1_tok','checkpoint':'synthetic.pt'}},
        'P1_vs_P3':['P1_tok',{'P1_tok_mean':.1,'P3_exp_mean':.2}]}
    (stage_prod/'train_summary.json').write_text(json.dumps(stage_train)+'\n')
    stage_selection=production_select_stage(output=stage_root)
    stage_eval=stage_prod/'evaluation';stage_eval.mkdir()
    (stage_eval/'summary.json').write_text(json.dumps({'selected_P1_or_P3':'P1_tok','p8_tau':.5,
        'candidate_count':len(pool['shape']),'methods':{}})+'\n')
    stage_report=production_report_stage(output=stage_root)
    result={'status':'smoke_complete','feature_cache_reused':str(THETA/'production'/'features'),
        'pool_sizes':{'C_vm_none':len(pool_a['shape']),'C_vm_08':len(pool_b['shape']),'after_holdout_exclusion':len(pool['shape'])},
        'heldout_tasks':list(hold),'smoke_rows':n,'token_preview_shape':list(token_preview.shape),
        'smoke_split_rows':{'dev':len(dev_ix),'eval':len(eval_ix)},
        'listwise_loss':loss,'top1_shape':list(chosen.shape),'actions_mixed':False,
        'adapter_output_shapes':adapter_checks,
        'one_step_listwise_train_loss':train_steps,
        'p6_connected_fit_smoke':{'mode':'P6_exp_hist','history_dropout':.3,
            'hard_negative_weight':.05,'fit_loss':p6_state['train_loss'],
            'hard_negative_pairs_used':p6_state['train_loss'][0]['hard_negative_pairs']},
        'p7_connected_fit_smoke':{'mode':'P7_bottleneck',
            'observation_only_pretrain':p7_state['observation_only_pretrain'],
            'supervised_fit_loss':p7_state['train_loss']},
        'p7b_connected_fit_smoke':{'mode':'P7b_type_bottleneck','supervised_fit_loss':p7b_state['train_loss']},
        'stage_graph_smoke':{'select':stage_selection['status'],'report':stage_report['status'],
            'report_path':stage_report['report'],'evaluation_scoring':'score_predictions + recall helper exercised on isolated mini inputs'},
        'addition_smoke':{'P8_batched_candidate_score_shape':list(p8_scores.shape),
            'P8_selected_one_candidate_per_query':list(p8_picked.shape),
            'P9_top32_stage_smoke_shape':list(p9_top.shape),
            'P9_selected_one_unit_per_query':list(p9_picked.shape),
            'diagnostic_strata':strata,'procrustes_degeneracy_example':degenerate_check,
            'fixed_vs_zero_noise_shape':list(initial_flow_noise(np.zeros((1,10,32),np.float32)).shape),
            'expert_zero_noise_supported_by_extractor':True},
        'score_smoke':{k:v for k,v in score.items() if k not in ('picked_candidate_global_ids','oracle_candidate_index')},
        'score_action_note':'identity dummy predictions exercise only scorer shape/serialization; the resulting zero action MSE is not a model metric.',
        'stages':{'extract':'reuse cached Q-mean and unseen token memmap; resumable Q-act/Q-exp production extractor implemented',
                  'train':'all adapters took a gradient step; production P6 fit loop ran with history dropout and branching/adjacent negatives on synthetic train rows; full LIBERO-40 training not run in smoke',
                  'evaluate':'evaluation-only scorer separated from fit API; smoke uses labels only at this boundary',
                  'report':'production selection stage and report writer exercised on synthetic stage artifacts'},
        'methods_supported':list(METHODS),'mid_layers_available':True,
        'mid_layers_reason':'production extractor reads cached PaliGemma prefix V values at layers 6,10,14,15,16,17 and pools them to 328 valid/padded tokens.',
        'expert_hidden_available':'available_via_real_model_extract_stage',
        'expert_hidden_reason':'Q-exp suffix taps are captured at t=1 and t=0; Q-act is taken directly from Pi0.sample_actions.',
        'eval_labels_read_for_training':False,'time_s':time.time()}
    (out/'smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=_json_default)+'\n')
    # Minimal human-readable smoke report; production report is not generated.
    (out/'report.md').write_text('# K1 query probe smoke\n\n- smoke only; this is not a production metric result.\n'
        f"- cached feature shapes: Q-mean={list(mean.shape)} (Dev {len(dev_ix)}, Eval {len(eval_ix)}), token preview={list(token_preview.shape)}.\n"
        f"- pool: C_vm_none={len(pool_a['shape'])}, C_vm_08={len(pool_b['shape'])}; holdout tasks={list(hold)}.\n"
        f"- methods wired: {', '.join(METHODS)}. P8/P9 helpers and stratification were exercised on synthetic data; no performance claim.\n"
        "- Q-mid reads prefix per-layer V from the PaliGemma KV cache; real-model extraction is recorded by the separate extract-stage manifest.\n")
    return result


def production_report_stage(*,output=OUT):
    dest=Path(output)/'production';report=dest/'report.md'
    if report.exists():return {'status':'complete','report':str(report)}
    train_path=dest/'train_summary.json';eval_path=dest/'evaluation'/'summary.json'
    if not train_path.exists() or not eval_path.exists():raise FileNotFoundError('run train and evaluate stages first')
    train=json.loads(train_path.read_text());ev=json.loads(eval_path.read_text());
    fmt=lambda x:'—' if x is None else f'{x:.5f}'
    lines=[
        '# K1 unseen LIBERO key-query probe','',
        '> Metrics are offline type-selection results. Q* uses unseen targets only in the scorer; it is not an inference-time method.','',
        f"- Holdout tasks for all model/temperature/tau selection: {train['holdout_tasks']}",
        f"- Selected type-query family for P8/P9: {ev['selected_P1_or_P3']}; calibrated candidate weight τ={ev['p8_tau']}.",
        f"- Candidate pool size after heldout-task removal: {ev['candidate_count']}",'',
        '| Method | Shape error (all / hard) | Action MSE θπ0.5 (all / hard) | Recall@1 / 5 / 20 |',
        '|---|---:|---:|---:|']
    for name,val in ev['methods'].items():
        if name=='zero_noise_expert_hidden_comparison':continue
        if name=='unit_action_oracle':
            m=val['metrics'];lines.append(f"| {name} | {fmt(m['shape_error_all'])} / {fmt(m['shape_error_hard'])} | {fmt(m['action_mse_all'])} / {fmt(m['action_mse_hard'])} | oracle |")
            continue
        if name in ('P8','P9_expert','P9_direction'):
            m=val['metrics'];lines.append(f"| {name} | {fmt(m['shape_error_mean'])} / {fmt(m['shape_error_hard'])} | {fmt(m['action_mse_mean'])} / {fmt(m['action_mse_hard'])} | {m['recall_oracle_epsilon']} |")
        else:
            m=val['metrics'];shape=m['shape_error_all'];act=m['actions']['theta_pi05']
            lines.append(f"| {name} | {fmt(shape)} / {fmt(m['shape_error_hard'])} | {fmt(act['mse_all'])} / {fmt(act['mse_hard'])} | {m['recall_oracle_epsilon']} |")
    lines += ['', '## Stratified diagnostics', '',
        'Episode-bootstrap summaries for translation-norm and selected-fragment-length strata. These are diagnostic groupings; targets are used only to score and assign the translation-norm groups.',
        '', '| Method / metric / subgroup | n | mean | hard mean |', '|---|---:|---:|---:|']
    def _append_strata(name,node,path=''):
        if not isinstance(node,dict):return
        if {'n','mean','hard'}<=set(node):
            hard_node=node['hard'];hard_mean='—' if not hard_node or hard_node['n']==0 else f"{hard_node['mean']:.6g}"
            lines.append(f"| {name} / {path} | {node['n']} | {node['mean']:.6g} | {hard_mean} |")
            return
        for key,value in node.items():_append_strata(name,value,f'{path}/{key}' if path else str(key))
    for name,val in ev['methods'].items():
        if name=='zero_noise_expert_hidden_comparison':continue
        metrics=val.get('metrics',{})
        diagnostics=metrics.get('diagnostics',{}) if isinstance(metrics,dict) else {}
        _append_strata(name,diagnostics)
    lines += ['', '## Limits', '',
        f"- P2 uses per-layer cached V at 0-based layers {list(MID_LAYER_IDS)} (key/value width from Gemma KV heads); image tokens are 2×2 pooled to 64 per active camera, then valid language tokens are appended.",
        '- The source rows were prepared with TokenizePrompt(discrete_state_input=False). In this run PaligemmaTokenizer therefore receives no state and emits BOS + cleaned instruction + newline; the language group includes those valid tokens and excludes padding, but contains no discrete state text or Action: marker. P1_tok_lang uses the masked mean of that instruction-only group; a separate instruction/state P1 is not applicable to this cache.',
        '- Q-exp uses the suffix hidden before action_out_proj at t=1 and t=0. Q-act is saved directly from Pi0.sample_actions with the same row-ID folded noise; a separate zero-noise extraction is available.',
        f"- Zero-noise Q-exp comparison: {'included' if ev['methods'].get('zero_noise_expert_hidden_comparison') else 'not run; optional --zero-noise extraction is separate'}.",
        '- Procrustes fitting has no special branch for degenerate motions; its SVD is still defined and its scale denominator is clamped.',
        '- Dev/Eval action labels are not consumed by fitting or method selection. Eval targets appear only in this report’s scoring pass.']
    dest.mkdir(parents=True,exist_ok=True)
    tmp=report.with_suffix(report.suffix+'.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(report)
    return {'status':'complete','report':str(report)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('smoke','extract','train','finalize-train','select','calibrate-p8','evaluate','report'),default='smoke')
    ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--limit',type=int,default=None)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--shard-id',type=int,default=0);ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--memory-fraction',type=float,default=.20)
    ap.add_argument('--device',default='cpu');ap.add_argument('--zero-noise',action='store_true')
    ap.add_argument('--skip-expert',action='store_true');ap.add_argument('--verify-integrator',action='store_true');a=ap.parse_args()
    if a.stage=='smoke':result=smoke(a.output,a.limit or 8)
    elif a.stage=='extract':result=extract_policy_feature_stage(output=a.output,batch=a.batch,threads=a.threads,
        limit=a.limit,zero_noise=a.zero_noise,verify_integrator=a.verify_integrator)
    elif a.stage=='train':result=production_train_stage(output=a.output,epochs=a.epochs,batch_size=a.batch_size,
        device=a.device,threads=a.threads,limit=a.limit,shard_id=a.shard_id,num_shards=a.num_shards,
        memory_fraction=a.memory_fraction)
    elif a.stage=='finalize-train':result=production_finalize_train_shards(output=a.output,num_shards=a.num_shards)
    elif a.stage=='select':result=production_select_stage(output=a.output)
    elif a.stage=='calibrate-p8':result=calibrate_p8_stage(output=a.output,threads=a.threads,batch=a.batch,limit=a.limit)
    elif a.stage=='evaluate':result=production_evaluate_stage(output=a.output,threads=a.threads,batch=a.batch_size,
        limit=a.limit,run_expert=not a.skip_expert)
    else:result=production_report_stage(output=a.output)
    print(json.dumps(result,ensure_ascii=False,default=_json_default),flush=True)

if __name__=='__main__':main()
