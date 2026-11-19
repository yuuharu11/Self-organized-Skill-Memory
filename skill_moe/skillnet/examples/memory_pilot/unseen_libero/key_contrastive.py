#!/usr/bin/env python3
"""Full-candidate contrastive retrieval training for fixed MK1 motion keys.

CT1 deliberately imports (and does not modify) key_motion.py.  Query-side
training rows are causal; target actions are used only to build soft teacher
distributions and score held-out rows.
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
import key_motion as km  # noqa: E402, read-only dependency
import key_shared_latent as k2  # noqa: E402

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'key-contrastive-01'
RUN_ROOT = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
MOTION_ROOT = ROOT / 'key-motion-01'
MISSING_ROOT = ROOT / 'key-shared-01/attempt-02/prefix-missing'
SEEN5 = ROOT / 'seg-context-01/seg5-cache'
UNSEEN5 = ROOT / 'seg-context-01/unseen-stride5'
VJEPA_EVAL = ROOT / 'key-shared-01/attempt-27-production/vjepa/libero90_eval'
X1_CKPT = MOTION_ROOT / 'arch/x1_absolute_dist_tau0.01.pt'
KEY2_ROOT = ROOT / 'key-shared-01/attempt-27-production'


def _log(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'stage': event, **fields}, allow_nan=False) + '\n')


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _load_data(out: Path, *, cache_dir:Path|None=None, force_build: bool = False):
    cache = Path(cache_dir) if cache_dir is not None else out / 'data'
    if (cache / 'manifest.json').exists() and not force_build:
        z = np.load(cache / 'aligned.npz', allow_pickle=False)
        payload = {k: z[k] for k in z.files}
        for name in ('tokens','groups'):
            p=cache/f'vlmh_{name}.npy'
            if p.exists():payload[f'vlmh__{name}']=np.load(p,mmap_mode='r',allow_pickle=False)
        manifest = json.loads((cache / 'manifest.json').read_text())
        return payload, manifest
    cache.mkdir(parents=True, exist_ok=True)
    def cb(stage, **kw): _log(out / 'progress.log', stage, **kw)
    bank, bi, fit_bank, fi, train0, learned, unseen, dev0 = k2._build_key2_queries(
        RUN_ROOT, MISSING_ROOT, SEEN5, UNSEEN5, VJEPA_EVAL, progress_callback=cb)
    # Bank query action windows are the fixed memory-unit labels used by the
    # teacher. Training candidates exclude held learned-evaluation episodes.
    bank_actions = k2._candidate_window_actions(np.arange(len(bank['length'])), bank)[..., :7].astype(np.float32)
    bank_task = np.asarray(bank['task'], np.int32); bank_ep = np.asarray(bank['episode'], np.int32)
    x1 = km.ArchMotionEncoder('x1', 'absolute')
    x1.load_state_dict(torch.load(X1_CKPT, map_location='cpu', weights_only=True)); x1.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x1.to(device)
    actions = torch.as_tensor(bank['natural_actions'][..., :6], device=device, dtype=torch.float32)
    lengths = torch.as_tensor(bank['length'], device=device, dtype=torch.long)
    valid = torch.arange(actions.shape[1], device=device)[None] < lengths[:, None]
    x1_keys = km._encode_motion(x1, actions, valid, batch=64).numpy().astype(np.float32)
    validate_fixed_key_cache(x1_keys)
    # DCT3 is fixed and standardized only with the training-eligible bank.
    coeff = np.asarray(bank['coefficients'], np.float32).reshape(len(bank_task), 18)
    # Use fit_bank's exact candidate membership (authoritative exclusion).
    fit_keys = {tuple(map(int, x[:2])) for x in fit_bank['source']}
    eligible = np.asarray([(int(t), int(e)) in fit_keys for t, e in zip(bank_task, bank_ep)], bool)
    dmean = coeff[eligible].mean(0); dstd = coeff[eligible].std(0).clip(1e-4)
    dct_keys = ((coeff - dmean) / dstd).astype(np.float32)
    # All-frame E-H queries: every five frames in eligible memory episodes.
    rows = k2._load_key2_rows(RUN_ROOT)
    excluded = set(zip(np.asarray(learned['task'], int).tolist(),
                       (np.asarray(learned['episode'], int) % 1_000_000).tolist()))
    # Group dense action/state traces from the canonical run rows.
    grouped: dict[tuple[int, int], list[int]] = {}
    for i, pair in enumerate(zip(rows['task'], rows['episode'], strict=True)):
        grouped.setdefault((int(pair[0]), int(pair[1])), []).append(i)
    dense: dict[str, list[np.ndarray]] = {k: [] for k in ('key','history','history_mask','state','actions','task','episode')}
    for (t, e), ids0 in grouped.items():
        ids = np.asarray(ids0, np.int64); ids = ids[np.argsort(rows['frame'][ids])]
        if not bool(rows['memory'][ids[0]]) or (t, e) in excluded: continue
        trace = np.asarray(rows['actions'][ids, :, :7], np.float32).reshape(-1, 7)
        base = int(rows['frame'][ids[0]])
        for f in range(0, len(trace) - 9, 5):
            h = np.zeros((10, 7), np.float32); m = np.zeros(10, bool)
            n = min(10, f); h[-n:] = trace[f-n:f] if n else h[-n:]; m[-n:] = True if n else m[-n:]
            state_i = min(len(ids)-1, f // 10)
            for k, v in (('key', (t,e,base+f)), ('history',h), ('history_mask',m),
                         ('state',np.asarray(rows['state'][ids[state_i],:8],np.float32)),
                         ('actions',trace[f:f+10]), ('task',t), ('episode',e)):
                dense[k].append(np.asarray(v))
    dense = {k: np.asarray(v) for k,v in dense.items()}
    dense['key'] = dense['key'].astype(np.int32); dense['task']=dense['task'].astype(np.int32); dense['episode']=dense['episode'].astype(np.int32)
    # E-VLM+H uses all exact cached rows available for memory demonstrations
    # (the cache cadence is the existing stride-5 extraction, typically 8k).
    cache_manifest = json.loads((SEEN5/'manifest.json').read_text())
    keys5 = np.asarray(cache_manifest['keys'], np.int32)
    tok5=np.load(SEEN5/'tokens.npy',mmap_mode='r',allow_pickle=False); grp5=np.load(SEEN5/'groups.npy',mmap_mode='r',allow_pickle=False)
    maprows=k2._load_key2_rows(RUN_ROOT); rowmap={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(maprows['task'],maprows['episode'],maprows['frame'],strict=True))}
    token_rows=[]; visual={k:[] for k in ('key','history','history_mask','state','actions','task','episode','tokens','groups')}
    # Use row frame metadata/chunks; cache actions and state are exactly keyed.
    cache_actions=np.load(SEEN5/'actions.npy',mmap_mode='r',allow_pickle=False); cache_state=np.load(SEEN5/'state.npy',mmap_mode='r',allow_pickle=False)
    trace_by_pair={}
    for (t,e), ids0 in grouped.items():
        ids=np.asarray(ids0,np.int64);ids=ids[np.argsort(rows['frame'][ids])]
        trace_by_pair[(t,e)]=(ids,np.asarray(rows['actions'][ids,:,:7],np.float32).reshape(-1,7),int(rows['frame'][ids[0]]))
    for i,(t,e,f) in enumerate(keys5):
        t,e,f=int(t),int(e),int(f)
        if (t,e) not in trace_by_pair or not bool(rows['memory'][trace_by_pair[(t,e)][0][0]]) or (t,e) in excluded: continue
        ids,trace,base=trace_by_pair[(t,e)]; off=f-base
        if off < 0 or off+10 > len(trace): continue
        h=np.zeros((10,7),np.float32);m=np.zeros(10,bool);n=min(10,off)
        if n: h[-n:]=trace[off-n:off];m[-n:]=True
        visual['key'].append((t,e,f));visual['history'].append(h);visual['history_mask'].append(m)
        visual['state'].append(np.asarray(cache_state[i],np.float32)[:8]);visual['actions'].append(np.asarray(cache_actions[i],np.float32)[:10,:7])
        visual['task'].append(t);visual['episode'].append(e);visual['tokens'].append(np.asarray(tok5[i]));visual['groups'].append(np.asarray(grp5[i]))
    visual={k:np.asarray(v) for k,v in visual.items()}
    visual['key']=visual['key'].astype(np.int32);visual['task']=visual['task'].astype(np.int32);visual['episode']=visual['episode'].astype(np.int32)
    # Dev rows already have exact features/actions and are selection-only.
    dev = {k:np.asarray(dev0[k]) for k in ('key','history','history_mask','state','actions','task','episode','tokens','groups','hard')}
    # Row-level data compressed into a portable npz. Keep bank tensors and
    # query features; fixed key cache is persisted separately and immutable.
    payload={}
    start_train={k:np.asarray(train0[k]) for k in ('key','history','history_mask','state','actions','task','episode')}
    start_map={tuple(map(int,k)):i for i,k in enumerate(bank['source'][:,:3])}
    start_train['candidate_id']=np.asarray([start_map[tuple(map(int,k))] for k in start_train['key']],np.int64)
    assert_training_eval_disjoint(start_train['key'],learned['key'])
    assert_training_eval_disjoint(dense['key'],learned['key'])
    assert_training_eval_disjoint(visual['key'],learned['key'])
    for prefix,part in (('bank',bank),('learned',learned),('unseen',unseen),('dev',dev),('eh',dense),('vlmh',visual),('start',start_train)):
        for k,v in part.items():
            if isinstance(v,np.ndarray) and not (prefix=='vlmh' and k in ('tokens','groups')): payload[prefix+'__'+k]=v
    np.save(cache/'vlmh_tokens.npy',visual['tokens'],allow_pickle=False)
    np.save(cache/'vlmh_groups.npy',visual['groups'],allow_pickle=False)
    payload['vlmh__tokens']=np.load(cache/'vlmh_tokens.npy',mmap_mode='r',allow_pickle=False)
    payload['vlmh__groups']=np.load(cache/'vlmh_groups.npy',mmap_mode='r',allow_pickle=False)
    payload.update(bank_actions=bank_actions,bank_task=bank_task,bank_episode=bank_ep,
                   x1_keys=x1_keys,dct_keys=dct_keys,dct_mean=dmean,dct_std=dstd,
                   fit_candidate=eligible.astype(np.uint8))
    np.savez_compressed(cache/'aligned.npz',**payload)
    manifest={'complete':True,'bank_units':len(bank_task),'eligible_candidate_units':int(eligible.sum()),
              'dense_eh_queries_stride5':len(dense['key']),'vlm_h_queries_cached':len(visual['key']),
              'dev_rows':len(dev['key']),'learned_eval_rows':len(learned['key']),'unseen_eval_rows':len(unseen['key']),
              'fixed_key_checkpoint':str(X1_CKPT),'fixed_keys':True,'eval_excluded_from_training':True,
              'visual_cache_manifest':str(SEEN5/'manifest.json')}
    _write_json(cache/'manifest.json',manifest)
    return payload,manifest


def _tensor_part(p:dict[str,np.ndarray],prefix:str,device):
    out={k:p[prefix+'__'+k] for k in p if k.startswith(prefix+'__')}
    return {k[len(prefix)+2:]:v for k,v in out.items()}


def symmetric_soft_loss(scores:torch.Tensor, errors:torch.Tensor, allowed:torch.Tensor,
                        tau:float, *, reverse_scores:torch.Tensor|None=None,
                        reverse_errors:torch.Tensor|None=None, reverse_allowed:torch.Tensor|None=None):
    """Bidirectional soft listwise loss. Shapes are validated and both axes masked."""
    if scores.shape != errors.shape or scores.shape != allowed.shape or scores.ndim != 2:
        raise ValueError('forward scores/errors/mask must align [B,N]')
    forward=k2.soft_listwise_loss(scores,errors,allowed,tau)
    if reverse_scores is None: return forward
    if reverse_errors is None or reverse_allowed is None:
        raise ValueError('reverse loss requires errors and allowed mask')
    if reverse_scores.shape != reverse_errors.shape or reverse_scores.shape != reverse_allowed.shape:
        raise ValueError('reverse scores/errors/mask must align')
    return .5*(forward+k2.soft_listwise_loss(reverse_scores,reverse_errors,reverse_allowed,tau))


def assert_training_eval_disjoint(train_keys:np.ndarray, eval_keys:np.ndarray) -> None:
    """Fail closed if any immutable (task,episode,frame) Eval key enters fit."""
    tr={tuple(map(int,x)) for x in np.asarray(train_keys)}
    ev={tuple(map(int,x)) for x in np.asarray(eval_keys)}
    overlap=tr&ev
    if overlap:raise AssertionError(f'Eval query identities leaked into training: {sorted(overlap)[:3]}')


def validate_fixed_key_cache(keys:np.ndarray, expected_rows:int=4516) -> None:
    """Validate the immutable, full memory candidate cache before training."""
    a=np.asarray(keys)
    if a.ndim!=2 or len(a)!=expected_rows or not np.isfinite(a).all():
        raise AssertionError(f'expected {expected_rows} finite cached keys, got {a.shape}')


def _pack_q(part,ids,input_mode,device,*,augment=False,rng=None):
    ix=np.asarray(ids,np.int64)
    def t(k,dtype):return torch.as_tensor(part[k][ix],device=device,dtype=dtype)
    hist=t('history',torch.float32);mask=t('history_mask',torch.bool);tokens=t('tokens',torch.float32) if input_mode=='e_vlm_h' else None
    groups=t('groups',torch.long) if input_mode=='e_vlm_h' else None
    if augment and tokens is not None:
        assert rng is not None
        # Drop 30% valid visual tokens and add low-amplitude feature noise.
        keep=torch.as_tensor(rng.random(tuple(groups.shape))>=.30,device=device)
        groups=torch.where((groups>=0)&keep,groups,torch.full_like(groups,-1))
        tokens=tokens+torch.randn_like(tokens)*.01
        hist_np,mask_np=k2._history_corrupt(hist.cpu().numpy(),mask.cpu().numpy(),rng)
        hist=torch.as_tensor(hist_np,device=device);mask=torch.as_tensor(mask_np,device=device)
    elif augment:
        hist_np,mask_np=k2._history_corrupt(hist.cpu().numpy(),mask.cpu().numpy(),rng)
        hist=torch.as_tensor(hist_np,device=device);mask=torch.as_tensor(mask_np,device=device)
    return dict(history=hist,history_mask=mask,state=t('state',torch.float32),tokens=tokens,groups=groups)


def _top_metric(scores:np.ndarray, errors:np.ndarray):
    from scipy.stats import spearmanr
    top=scores.argmax(1);rows=np.arange(len(top));te=errors[rows,top];best=errors.argmin(1)
    rho=[spearmanr(scores[i],-errors[i]).statistic for i in range(len(scores))]
    order=np.argsort(-scores,axis=1);oracle=np.argsort(errors,axis=1)
    recall={str(k):float(np.mean([len(set(order[i,:k])&set(oracle[i,:k]))/k for i in range(len(order))])) for k in (1,5,20)}
    return {'top1_arm6_mse':float(te.mean()),'top1_per_query':te.astype(np.float32),
            'spearman_positive':float(np.nanmean(rho)),'recall_at_k':recall,
            'top1_candidate':top,'oracle_candidate':best}


def _teacher_cross_task_mass(errors, allowed, query_task, bank_task, tau):
    ee=np.asarray(errors,np.float64);aa=np.asarray(allowed,bool)
    logits=np.where(aa,-ee/float(tau),-1e9);logits-=logits.max(1,keepdims=True)
    p=np.exp(logits)*aa;p/=np.maximum(p.sum(1,keepdims=True),1e-30)
    cross=np.asarray(query_task)[:,None]!=np.asarray(bank_task)[None,:]
    return p[cross].sum()/max(len(p),1)


def _history_mlp_top1(split, bank_source, bank_task, bank_ep, bank_actions, rows, device):
    """Per-query KeyGrid H+state MLP evaluation for paired bootstrap."""
    import key_grid as kg
    result_path=ROOT/'key-grid-01/production-04/results.json'
    saved=json.loads(result_path.read_text())
    meta=saved['new_models']['history_state']['variants']['two_tower_mlp']
    ck=Path(meta['checkpoint'])
    model=kg.PairMLPRanker(88,shared=False).to(device)
    model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval()
    # Build exact, causal memory-start history/state from aligned run traces.
    grouped={}
    for i,(t,e) in enumerate(zip(rows['task'],rows['episode'],strict=True)):
        grouped.setdefault((int(t),int(e)),[]).append(i)
    traces={}
    for pair,ids0 in grouped.items():
        ii=np.asarray(ids0,np.int64);ii=ii[np.argsort(rows['frame'][ii])]
        traces[pair]=(ii,np.asarray(rows['actions'][ii,:,:7],np.float32).reshape(-1,7),int(rows['frame'][ii[0]]))
    hbank=np.zeros((len(bank_source),10,7),np.float32);mbank=np.zeros((len(bank_source),10),bool)
    sbank=np.zeros((len(bank_source),8),np.float32)
    bank_keys=np.asarray(bank_source)[:,:3]
    # Use the same exact normalized state lookup as the canonical bank query builder.
    sbank[:]=k2._exact_start_states(bank_keys,rows,RUN_ROOT,MISSING_ROOT)
    for i,(t,e,f,_end) in enumerate(np.asarray(bank_source)):
        ii,tr,base=traces[(int(t),int(e))];off=int(f)-base;n=min(10,off)
        if n:hbank[i,-n:]=tr[off-n:off];mbank[i,-n:]=True
    # Query and candidate vectors match key_grid.make_vector_features(history_state).
    qx=np.concatenate([split['history'].reshape(len(split['key']),-1),split['history_mask'].astype(np.float32),split['state']],-1).astype(np.float32)
    kx=np.concatenate([hbank.reshape(len(hbank),-1),mbank.astype(np.float32),sbank],-1).astype(np.float32)
    scores=kg._score_vectors(model,qx,kx,device=str(device),batch=128)
    errors=k2._arm6_candidate_mse(split['actions'],bank_actions)
    allowed=~((split['task'][:,None]==bank_task[None,:])&(split['episode'][:,None]==bank_ep[None,:]))
    scores[~allowed]=-1e9;top=scores.argmax(1)
    return errors[np.arange(len(top)),top], {'checkpoint':str(ck),'saved_keygrid_unseen':meta['evaluation']['unseen']['metrics']['top1_arm6_mse']}


def _query_vector(model,qin):
    q=model(**qin)
    return model.dct_projection(q[:,0]) if model.key_mode=='dct' else q[:,0]


def _score(model,part,keys,errors,input_mode,key_mode,device,batch=64):
    model.eval();out=[]
    with torch.no_grad():
        for s in range(0,len(part['key']),batch):
            ids=np.arange(s,min(s+batch,len(part['key'])))
            qin=_pack_q(part,ids,input_mode,device)
            q=model(**qin)
            if key_mode=='dct':q=model.dct_projection(q[:,0])
            else:q=q[:,0]
            if key_mode=='x1':
                k=torch.as_tensor(keys,device=device,dtype=torch.float32)
                sc=(q@k.T)*model.logit_scale.detach().exp().clamp(max=100)
            else:
                k=torch.as_tensor(keys,device=device,dtype=torch.float32)
                sc=-torch.cdist(q.float(),k.float()).square()*model.logit_scale.detach().exp().clamp(max=100)
            out.append(sc.cpu().numpy())
    scores=np.concatenate(out)
    allowed=~((part['task'][:,None]==part['bank_task'][None,:])&(part['episode'][:,None]==part['bank_episode'][None,:]))
    scores=scores.astype(np.float32);scores[~allowed]=-1e9
    return _top_metric(scores,errors),scores


class CTQueryModel(nn.Module):
    """The requested MK1 MotionQueryEncoder q1 with optional DCT-space head."""
    def __init__(self,input_mode,key_mode):
        super().__init__();self.key_mode=key_mode
        self.encoder=km.MotionQueryEncoder(input_mode,'q1')
        self.dct_projection=nn.Linear(256,18) if key_mode=='dct' else nn.Identity()
    @property
    def logit_scale(self):return self.encoder.logit_scale
    def forward(self,**kwargs):return self.encoder(**kwargs)


def _make_model(input_mode,key_mode,device):
    return CTQueryModel(input_mode,key_mode).to(device)


def _train_config(name,train,dev,bank_actions,bank_keys,bank_task,bank_ep,*,input_mode,key_mode,
                  lr,tau,epochs,device,seed,batch_size=256,ablation=None,baseline=False,progress=None,
                  dct_stats=None):
    rng=np.random.default_rng(seed);torch.manual_seed(seed);model=_make_model(input_mode,key_mode,device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    key_encoder=None
    if key_mode=='x1' and not baseline:
        key_encoder=km.ArchMotionEncoder('x1','absolute').to(device)
        key_encoder.load_state_dict(torch.load(X1_CKPT,map_location=device,weights_only=True));key_encoder.eval()
        for p in key_encoder.parameters():p.requires_grad_(False)
    qn=len(train['key']);ids=np.arange(qn);history=[];best=float('inf');best_epoch=0;best_state=None
    steps=max(1,int(np.ceil(qn/batch_size)))
    for ep in range(1,epochs+1):
        rng.shuffle(ids);model.train();losses=[]
        for bi,start in enumerate(range(0,qn,batch_size)):
            qi=ids[start:start+batch_size]
            if baseline:
                # MK1c-style start-only / batch8 / 256 random candidates.
                rand=rng.choice(len(bank_actions),size=min(256,len(bank_actions)),replace=False)
                cids=np.unique(np.r_[train['candidate_id'][qi],rand])
            elif ablation=='minus_all_candidates':
                qcid=train['candidate_id'][qi]
                qcid=qcid[qcid>=0]
                oracle_ids=np.argmin(k2._arm6_candidate_mse(train['actions'][qi],bank_actions),axis=1)
                cids=np.unique(np.r_[qcid,oracle_ids,rng.choice(len(bank_actions),size=min(256,len(bank_actions)),replace=False)])
            else:cids=np.arange(len(bank_actions))
            allowed=~((train['task'][qi,None]==bank_task[cids][None,:])&(train['episode'][qi,None]==bank_ep[cids][None,:]))
            qin=_pack_q(train,qi,input_mode,device,augment=(not baseline and ablation!='minus_augmentation'),rng=rng)
            rawq=model(**qin);q=rawq[:,0]
            if key_mode=='dct':q=model.dct_projection(q)
            kt=torch.as_tensor(bank_keys[cids],device=device,dtype=torch.float32)
            if key_mode=='x1':scores=(q@kt.T)*model.logit_scale.exp().clamp(max=100)
            else:scores=-torch.cdist(q.float(),kt.float()).square()*model.logit_scale.exp().clamp(max=100)
            err=k2._arm6_candidate_mse(train['actions'][qi],bank_actions[cids])
            et=torch.as_tensor(err,device=device);am=torch.as_tensor(allowed,device=device)
            forward=k2.soft_listwise_loss(scores,et,am,tau)
            if baseline or ablation=='minus_symmetric':loss=forward
            else:
                # The reverse direction treats each query target as a fixed
                # motion key and retrieves its matching causal query in batch.
                qactions=torch.as_tensor(train['actions'][qi,:,:6],device=device,dtype=torch.float32)
                if key_mode=='x1':
                    with torch.no_grad():
                        qvalid=torch.ones(qactions.shape[:2],device=device,dtype=torch.bool)
                        qkey=key_encoder(qactions,qvalid)[:,0]
                    rev_scores=(qkey@q.T)*model.logit_scale.exp().clamp(max=100)
                else:
                    qcoeff=km._dct3_from_batch(qactions,torch.ones(qactions.shape[:2],device=device,dtype=torch.bool))
                    if dct_stats is not None:
                        qcoeff=(qcoeff-torch.as_tensor(dct_stats[0],device=device))/torch.as_tensor(dct_stats[1],device=device)
                    rev_scores=-torch.cdist(qcoeff.float(),q.float()).square()*model.logit_scale.exp().clamp(max=100)
                rev_err=k2._arm6_candidate_mse(train['actions'][qi],train['actions'][qi])
                rev_allow=~((train['task'][qi,None]==train['task'][qi][None,:]) &
                            (train['episode'][qi,None]==train['episode'][qi][None,:]))
                loss=symmetric_soft_loss(scores,et,am,tau,reverse_scores=rev_scores,
                                         reverse_errors=torch.as_tensor(rev_err,device=device),
                                         reverse_allowed=torch.as_tensor(rev_allow,device=device))
            if not baseline and ablation!='minus_same_scene_negative':
                # Encourage ranking of the best same-task alternative above
                # a hard same-scene negative (largest teacher MSE in-task).
                local=np.flatnonzero(bank_task[cids][None,:]==train['task'][qi,None] & (bank_ep[cids][None,:]!=train['episode'][qi,None])) if False else None
                same=(bank_task[cids][None,:]==train['task'][qi,None]) & (bank_ep[cids][None,:]!=train['episode'][qi,None])
                sm=torch.as_tensor(same,device=device)&am
                negerr=et.masked_fill(~sm,-torch.inf);negid=negerr.argmax(-1)
                has=sm.any(-1)
                posid=et.masked_fill(~am,torch.inf).argmin(-1)
                if has.any():
                    pos=scores.gather(1,posid[:,None]).squeeze(1);neg=scores.gather(1,negid[:,None]).squeeze(1)
                    loss=loss+.1*F.softplus(neg[has]-pos[has]+.1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();k2._clamp_logit_scale_(model)
            losses.append(float(loss.detach()))
        rec={'epoch':ep,'loss':float(np.mean(losses)),'steps':len(losses)}
        if ep%2==0 or ep==epochs:
            dm,_=_score(model,dev,bank_keys,dev['errors'],input_mode,key_mode,device)
            rec['dev_top1_arm6_mse']=dm['top1_arm6_mse']
            if rec['dev_top1_arm6_mse']<best:
                best=rec['dev_top1_arm6_mse'];best_epoch=ep
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(rec)
        if progress:progress(name=name,epoch=ep,loss=rec['loss'],dev=rec.get('dev_top1_arm6_mse'))
    if best_state:model.load_state_dict(best_state)
    return model,history,best_epoch,best


def _build_query_split(part, prefix, bank_actions, bank_task,bank_ep, *, dense=False):
    out={k:part[prefix+'__'+k] for k in ('key','history','history_mask','state','actions','task','episode')}
    if prefix=='vlmh':out.update(tokens=part['vlmh__tokens'],groups=part['vlmh__groups'])
    else:
        out['tokens']=np.zeros((len(out['key']),1,2048),np.float16);out['groups']=np.full((len(out['key']),1),-1,np.int8)
    cmap={tuple(map(int,k)):i for i,k in enumerate(part['bank__source'][:,:3])}
    out['candidate_id']=np.asarray([cmap.get(tuple(map(int,k)),-1) for k in out['key']],np.int64)
    if prefix+'__hard' in part:out['hard']=np.asarray(part[prefix+'__hard'],bool)
    errors=k2._arm6_candidate_mse(out['actions'],bank_actions)
    out['errors']=errors;out['bank_task']=bank_task;out['bank_episode']=bank_ep
    return out


def _derive_eval_input(split, bank, keys, key_mode, input_mode, device):
    # key2's fixed datasets have exact row identities and cached features.
    out={k:np.asarray(split[k]) for k in ('key','history','history_mask','state','actions','task','episode','hard')}
    out['tokens']=np.asarray(split['tokens']);out['groups']=np.asarray(split['groups'])
    out['bank_task']=bank['task'];out['bank_episode']=bank['episode']
    out['errors']=k2._arm6_candidate_mse(out['actions'],bank['actions'])
    return out


def _eval_experiment(out:Path,data,manifest,*,device='cuda:0',smoke=False,force=False):
    # This function is restartable: completed named rows are retained in results.json.
    result_path=out/'results.json'
    if result_path.exists() and not force:
        saved=json.loads(result_path.read_text())
        if saved.get('status')=='complete':return saved
        result=saved
    else: result={'status':'running','metadata':manifest,'rows':{},'references':{}}
    device=torch.device(device if torch.cuda.is_available() else 'cpu')
    bank={k:data['bank__'+k] for k in ('source','task','episode','natural_actions','length','coefficients')}
    bank_actions=data['bank_actions'];bt=data['bank_task'];be=data['bank_episode'];fit=data['fit_candidate'].astype(bool)
    xkeys=data['x1_keys'];dkeys=data['dct_keys']
    # dev selection data is feature-keyed to available held-out task demos.
    dev_native=_load_key2_queries_for_dev() if False else None
    train_eh=_build_query_split(data,'eh',bank_actions,bt,be,dense=True)
    train_v=_build_query_split(data,'vlmh',bank_actions,bt,be)
    train_start=_build_query_split(data,'start',bank_actions,bt,be)
    if 'train_teacher_cross_task_mass' not in result['metadata']:
        # Fixed evenly spaced sample avoids tying the diagnostic to a run's
        # shuffled training minibatches while covering all memory episodes.
        sample=np.linspace(0,len(train_eh['key'])-1,min(1024,len(train_eh['key']))).astype(np.int64)
        allow=~((train_eh['task'][sample,None]==bt[None,:])&(train_eh['episode'][sample,None]==be[None,:]))
        result['metadata']['train_teacher_cross_task_mass']=float(_teacher_cross_task_mass(
            train_eh['errors'][sample],allow,train_eh['task'][sample],bt,
            float(result.get('metadata',{}).get('selected_tau',.005))))
        result['metadata']['train_teacher_mass_sample_rows']=len(sample)
    # Remove any query whose own episode was excluded from fixed candidate support.
    train_eh={k:v for k,v in train_eh.items()};train_v={k:v for k,v in train_v.items()}
    # Candidate id lookup is by immutable identity; all memory train examples
    # are verified disjoint from learned-evaluation demos in data construction.
    learned=k2._load_eval_cached_for_ct(data,'learned') if hasattr(k2,'_load_eval_cached_for_ct') else None
    # The data NPZ stores eval query arrays from the helper; no action/labels in train.
    def part(prefix):return _build_query_split(data,prefix,bank_actions,bt,be)
    dev=part('dev');learned=part('learned');unseen=part('unseen')
    # Set hard flags to report subset only; carry split identities for bootstrap.
    configs=[]
    configs.append(('baseline_x1_e_h','e_h','x1',None,True))
    configs.append(('ctfull_x1_e_h','e_h','x1',None,False))
    for a in ('minus_all_candidates','minus_all_frames','minus_symmetric','minus_same_scene_negative','minus_augmentation'):
        configs.append((a+'_x1_e_h','e_h','x1',a,False))
    for kmode,imode in (('x1','e_vlm_h'),('dct','e_h'),('dct','e_vlm_h')):
        configs.append((f'ctfull_{kmode}_{imode}',imode,kmode,None,False))
        configs.append((f'baseline_{kmode}_{imode}',imode,kmode,None,True))
    if smoke:configs=configs[:2]
    if 'selected_lr' not in result['metadata'] and not smoke:
        # Small, explicit Dev-only grid. Use a fixed subset of memory queries
        # for tuning cost; full training rows are used after selection.
        grid_train={k:(v[:2048] if isinstance(v,np.ndarray) and len(v)==len(train_eh['key']) else v)
                    for k,v in train_eh.items()}
        tuning=[]
        for lr0 in (1e-4,3e-4):
            for tau0 in (.002,.005):
                tm,th,te,td=_train_config(f'tune-lr{lr0}-tau{tau0}',grid_train,dev,bank_actions,xkeys,bt,be,
                    input_mode='e_h',key_mode='x1',lr=lr0,tau=tau0,epochs=4,device=device,seed=4101,
                    batch_size=256,dct_stats=(data['dct_mean'],data['dct_std']))
                tuning.append({'lr':lr0,'tau':tau0,'selected_epoch':te,'dev_top1_arm6_mse':td,'history':th})
                del tm
                if torch.cuda.is_available():torch.cuda.empty_cache()
        chosen=min(tuning,key=lambda x:x['dev_top1_arm6_mse'])
        result['metadata'].update(selected_lr=chosen['lr'],selected_tau=chosen['tau'],tuning=tuning,
                                  tuning_selection='Dev top-1 arm MSE on first 2048 EH memory stride-5 rows')
        _write_json(result_path,result)
    for name,imode,kmode,ablation,baseline in configs:
        if name in result['rows']:continue
        source=train_eh if imode=='e_h' else train_v
        if ablation=='minus_all_frames':source=train_start
        if baseline:
            # Explicit start-only subset, batch 8 and random 256 candidate set.
            starts=np.arange(len(train_start['key']))
            source={k:(v[starts] if isinstance(v,np.ndarray) and len(v)==len(train_start['key']) else v) for k,v in train_start.items()}
            if imode=='e_vlm_h':
                vmap={tuple(map(int,k)):i for i,k in enumerate(train_v['key'])}
                keep=np.asarray([i for i,k in enumerate(source['key']) if tuple(map(int,k)) in vmap],np.int64)
                source={k:v[keep] for k,v in source.items()}
                visix=np.asarray([vmap[tuple(map(int,k))] for k in source['key']],np.int64)
                source['tokens']=train_v['tokens'][visix];source['groups']=train_v['groups'][visix]
        key=xkeys if kmode=='x1' else dkeys
        _log(out/'progress.log','train_start',name=name,train_rows=len(source['key']),input=imode,key=kmode)
        model,hist,ep,devm=_train_config(name,source,dev,bank_actions, key,bt,be,input_mode=imode,key_mode=kmode,
            lr=float(result.get('metadata',{}).get('selected_lr',1e-4)),tau=float(result.get('metadata',{}).get('selected_tau',.002)),
            epochs=2 if smoke else (16 if baseline else 12),device=device,seed=4101,batch_size=8 if baseline else (32 if smoke else 256),
            ablation=ablation,baseline=baseline,progress=lambda **kw:_log(out/'progress.log','epoch',**kw))
        ck=out/'checkpoints'/f'{name}.pt';ck.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),ck)
        evs={}
        for splitname,sp in (('learned',learned),('unseen',unseen)):
            met,scores=_score(model,sp,key,sp['errors'],imode,kmode,device)
            hard=np.asarray(sp.get('hard',np.zeros(len(sp['key']),bool)),bool);per=met.pop('top1_per_query');chosen=met.pop('top1_candidate');oracle=met.pop('oracle_candidate')
            met['top1_arm6_mse_hard']=float(per[hard].mean()) if hard.any() else None
            if hard.any():
                hm=_top_metric(scores[hard],sp['errors'][hard])
                for k0,v0 in hm.items():
                    if k0 not in ('top1_per_query','top1_candidate','oracle_candidate'):met[k0+'_hard']=v0
            met['rows']=len(per);met['hard_rows']=int(hard.sum());met['per_query_top1']=per.tolist()
            met['top1_from_other_task_fraction']=float(np.mean(bt[chosen]!=sp['task']))
            allowed=~((sp['task'][:,None]==bt[None,:])&(sp['episode'][:,None]==be[None,:]))
            met['teacher_cross_task_mass']=float(_teacher_cross_task_mass(sp['errors'],allowed,sp['task'],bt,
                float(result.get('metadata',{}).get('selected_tau',.002))))
            evs[splitname]=met
        result['rows'][name]={'key_mode':kmode,'input_mode':imode,'ablation':ablation,'baseline':baseline,
            'selected_epoch':ep,'dev_top1_arm6_mse':devm,'history':hist,'evaluation':evs,'checkpoint':str(ck)}
        result['metadata'].update({'selected_lr':result['metadata'].get('selected_lr',1e-4),'selected_tau':result['metadata'].get('selected_tau',.002)})
        _write_json(result_path,result);_log(out/'progress.log','train_complete',name=name,dev=devm)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not smoke:
        rows0=k2._load_key2_rows(RUN_ROOT)
        refs_path=ROOT/'key-grid-01/production-04/results.json';kg=json.loads(refs_path.read_text())
        key2res=json.loads((KEY2_ROOT/'results.json').read_text())
        result['references']={
          'history_state_mlp':{'unseen':kg['new_models']['history_state']['variants']['two_tower_mlp']['evaluation']['unseen']['metrics']['top1_arm6_mse'],
                               'learned':kg['new_models']['history_state']['variants']['two_tower_mlp']['evaluation']['learned']['metrics']['top1_arm6_mse']},
          'key2_separate_unseen':.1187,
          'p5_unseen':kg['existing_reselected']['p5']['evaluation']['unseen']['metrics']['top1_arm6_mse'],
          'e_vlm_h_unseen':kg['existing_reselected']['e_vlm_h']['evaluation']['unseen']['metrics']['top1_arm6_mse'],
          'key2_fixed_baselines':key2res['methods']['unseen']['fixed_baselines']}
        for splitname,sp in (('learned',learned),('unseen',unseen)):
            base,_=_history_mlp_top1(sp,data['bank__source'],bt,be,bank_actions,rows0,device)
            for name in ('baseline_x1_e_h','ctfull_x1_e_h'):
                if name not in result['rows']:continue
                per=np.asarray(result['rows'][name]['evaluation'][splitname]['per_query_top1'],np.float64)
                result.setdefault('paired_episode_bootstrap_5000',{}).setdefault(splitname,{})[name+'_minus_history_state_mlp']=k2.episode_bootstrap_ci(
                    per-base,sp['task'],sp['episode'],reps=5000,seed=6117)
            if 'ctfull_x1_e_h' in result['rows'] and 'baseline_x1_e_h' in result['rows']:
                a=np.asarray(result['rows']['ctfull_x1_e_h']['evaluation'][splitname]['per_query_top1'],np.float64)
                b=np.asarray(result['rows']['baseline_x1_e_h']['evaluation'][splitname]['per_query_top1'],np.float64)
                result['paired_episode_bootstrap_5000'][splitname]['ctfull_minus_baseline']=k2.episode_bootstrap_ci(
                    a-b,sp['task'],sp['episode'],reps=5000,seed=6118)
    result['status']='complete_smoke' if smoke else 'complete'
    _write_json(result_path,result);_write_report(out,result)
    return result


def _load_key2_queries_for_dev():
    return None


def _write_report(out,r):
    lines=['# CT1：固定運動キーでの対照学習','','状態：'+str(r['status']),'',
           f"選択した学習率={r.get('metadata',{}).get('selected_lr','—')}、教師温度={r.get('metadata',{}).get('selected_tau','—')}。設定は Dev 217 行の top-1 腕 MSE で選択。",'',
           f"候補={r.get('metadata',{}).get('bank_units')}、E-H 学習 query={r.get('metadata',{}).get('dense_eh_queries_stride5')}（stride 5）、E-VLM+H 学習 query={r.get('metadata',{}).get('vlm_h_queries_cached')}（既存 cache）、Dev={r.get('metadata',{}).get('dev_rows')}、Eval={r.get('metadata',{}).get('learned_eval_rows')}/{r.get('metadata',{}).get('unseen_eval_rows')}。",'',
           '## 主表：Eval top-1 腕 MSE','','|条件|未学習 all/hard|学習済み all/hard|他タスク単位率（未学習）|選択 epoch|','|---|---:|---:|---:|---:|']
    for name,row in r.get('rows',{}).items():
        u=row['evaluation']['unseen'];l=row['evaluation']['learned']
        hard=lambda x:('—' if x is None else f'{x:.5f}')
        lines.append(f"|{name}|{u['top1_arm6_mse']:.5f}/{hard(u['top1_arm6_mse_hard'])}|{l['top1_arm6_mse']:.5f}/{hard(l['top1_arm6_mse_hard'])}|{u['top1_from_other_task_fraction']:.3f}|{row['selected_epoch']}|")
    lines += ['', '## 検索指標：Spearman・Recall@k・回収率', '', '回収率は (random − 条件)/(random − oracle)。各欄は all/hard。Recall@k は正解誤差上位 k との重なり割合。', '']
    for split,label in (('unseen','未学習'),('learned','学習済み')):
        lines += [f'### {label}', '', '|条件|Spearman all/hard|R@1 all/hard|R@5 all/hard|R@20 all/hard|回収率 all/hard|', '|---|---:|---:|---:|---:|---:|']
        refs_split=r.get('references',{}).get(split,{})
        fixed=refs_split.get('fixed_baselines',{})
        rand=fixed.get('random',{});oracle=fixed.get('oracle',{})
        den_all=rand.get('all',0)-oracle.get('all',0);den_hard=rand.get('hard',0)-oracle.get('hard',0)
        for name,row in r.get('rows',{}).items():
            m=row['evaluation'][split]
            def pair(a,b): return f'{a:.4f}/{b:.4f}'
            rec=m['recall_at_k'];rech=m.get('recall_at_k_hard',{})
            recpairs=[pair(rec[str(k)],rech.get(str(k),float('nan'))) for k in (1,5,20)]
            recov_all=(rand.get('all',0)-m['top1_arm6_mse'])/den_all if den_all else float('nan')
            mh=m.get('top1_arm6_mse_hard')
            recov_hard=(rand.get('hard',0)-mh)/den_hard if den_hard and mh is not None else float('nan')
            lines.append(f"|{name}|{pair(m['spearman_positive'],m.get('spearman_positive_hard',float('nan')))}|{recpairs[0]}|{recpairs[1]}|{recpairs[2]}|{pair(recov_all,recov_hard)}|")
    lines += ['', '## 既存比較・参考値（top-1 腕 MSE、全体／難）', '', '|行|未学習 all/hard|学習済み all/hard|出典|','|---|---:|---:|---|']
    refs=r.get('references',{})
    labels={'history_state_mlp':'履歴＋状態 MLP','key2_separate':'KEY2 別キー','p5':'P5（KEYGRID）','e_vlm_h':'E-VLM+H（KEYGRID）'}
    for key,label in labels.items():
        u=refs.get('unseen',{}).get(key);l=refs.get('learned',{}).get(key)
        if u and l:lines.append(f"|{label}|{u['all']:.5f}/{u['hard']:.5f}|{l['all']:.5f}/{l['hard']:.5f}|既存 locked result|")
    for key,label in (('hold_last','直前保持'),('oracle','oracle'),('random','random')):
        u=refs.get('unseen',{}).get('fixed_baselines',{}).get(key);l=refs.get('learned',{}).get('fixed_baselines',{}).get(key)
        if u and l:lines.append(f"|{label}|{u['all']:.5f}/{u['hard']:.5f}|{l['all']:.5f}/{l['hard']:.5f}|KEY2-F fixed baseline|")
    if 'train_teacher_cross_task_mass' in r.get('metadata',{}):
        lines += ['', '### 教師分布の別タスク質量', '',
          f"固定した学習 query {r['metadata'].get('train_teacher_mass_sample_rows')} 行において、τ={r['metadata'].get('selected_tau')} の教師分布の平均 **{r['metadata']['train_teacher_cross_task_mass']:.4f}** が別タスク候補へ割り当てられた。"]
    lines += ['', '## CT-full / 簡易版 / 履歴＋状態 MLP の対比較（episode bootstrap 95%、5,000回）', '', '|Split|Contrast|Mean Δ|95% CI|','|---|---|---:|---:|']
    for split,items in r.get('paired_episode_bootstrap_5000',{}).items():
        for key,label in (('ctfull_minus_baseline','CT-full − 簡易版'),('ctfull_x1_e_h_minus_history_state_mlp','CT-full − H+state MLP'),('baseline_x1_e_h_minus_history_state_mlp','簡易版 − H+state MLP')):
            d=items.get(key)
            if d:lines.append(f"|{split}|{label}|{d['mean']:.5f}|[{d['ci95_low']:.5f}, {d['ci95_high']:.5f}]|")
    lines += ['', '## 学習済み／未学習差（合わせ込みの記述）', '', '|行|学習済み MSE − 未学習 MSE|','|---|---:|']
    for name,row in r.get('rows',{}).items():
        lines.append(f"|{name}|{row['evaluation']['learned']['top1_arm6_mse']-row['evaluation']['unseen']['top1_arm6_mse']:+.5f}|")
    lines+=['','## 指標・制約','','- 腕 MSE は各窓の10×6二乗和を10×7で割る。Recall@k は予測上位kと正解誤差上位kの重なり割合。Spearman は score と負の行動誤差の順位相関（正が良い）。',
            '- キーは固定。K-x1 は MK1b `arch/x1_absolute_dist_tau0.01.pt`、K-dct はメモリ学習候補の DCT3 係数を標準化した固定ベクトル。',
            '- E-H はメモリデモの5フレーム刻み、E-VLM+H は既存 exact VLM cache にある窓だけを使う。',
            '- baseline は start-only / batch 8 / random 256 / 片方向の再実装。MK1c 本番値が存在しないため直接照合できない。',
            '- Eval の行動は採点にだけ使用。タスクIDは行動検索モデル入力に含めず、候補マスク・事後集計だけに使った。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def _finalize_cached_results(out,data,*,device='cuda:0'):
    """Re-score the locked Eval sets for hard-subset metrics and finalize references."""
    out=Path(out);rp=out/'results.json';r=json.loads(rp.read_text())
    dev=torch.device(device if torch.cuda.is_available() else 'cpu')
    bank_actions=data['bank_actions'];bt=data['bank_task'];be=data['bank_episode']
    splits={n:_build_query_split(data,n,bank_actions,bt,be) for n in ('learned','unseen')}
    for name,row in r['rows'].items():
        model=_make_model(row['input_mode'],row['key_mode'],dev)
        model.load_state_dict(torch.load(row['checkpoint'],map_location=dev,weights_only=True));model.eval()
        keys=data['x1_keys'] if row['key_mode']=='x1' else data['dct_keys']
        for splitname,sp in splits.items():
            hard=np.asarray(sp['hard'],bool)
            metrics,scores=_score(model,sp,keys,sp['errors'],row['input_mode'],row['key_mode'],dev)
            hm=_top_metric(scores[hard],sp['errors'][hard])
            row['evaluation'][splitname]['top1_arm6_mse_hard']=hm['top1_arm6_mse']
            row['evaluation'][splitname]['spearman_positive_hard']=hm['spearman_positive']
            row['evaluation'][splitname]['recall_at_k_hard']=hm['recall_at_k']
            row['evaluation'][splitname]['hard_rows']=int(hard.sum())
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Reference values come from the already locked KEYGRID / KEY2-F result files.
    kg=json.loads((ROOT/'key-grid-01/production-04/results.json').read_text())
    k2r=json.loads((KEY2_ROOT/'results.json').read_text())
    refs={}
    for key,label in (('history_state_mlp','history_state'),('p5_unseen','p5'),('e_vlm_h_unseen','e_vlm_h')):
        pass
    for splitname in ('learned','unseen'):
        kg_split={}
        for m in ('history_state','p5','e_vlm_h'):
            if m=='history_state':x=kg['new_models'][m]['variants']['two_tower_mlp']['evaluation'][splitname]['metrics']
            else:x=kg['existing_reselected'][m]['evaluation'][splitname]['metrics']
            kg_split[m]={'all':float(x['top1_arm6_mse']),'hard':float(x.get('top1_arm6_mse_hard') or 0.)}
        km=k2r['methods'][splitname]
        bas=km['fixed_baselines']['baselines'];bash=km['fixed_baselines'].get('baselines_hard',{})
        fixed={k:{'all':float(v['arm6']),'hard':float(bash.get(k,{}).get('arm6') or 0.)}
                for k,v in bas.items()}
        fixed['oracle']={'all':float(km['fixed_baselines']['oracle_playback_all']['arm6']),
                         'hard':float(km['fixed_baselines'].get('oracle_playback_hard',{}).get('arm6') or 0.)}
        fixed['random']={'all':float(km['fixed_baselines']['random_rank']['top1_arm6_mse']),
                         'hard':float(km['fixed_baselines']['random_rank_hard']['top1_arm6_mse'])}
        refs[splitname]={'history_state_mlp':kg_split['history_state'],'p5':kg_split['p5'],
                         'e_vlm_h':kg_split['e_vlm_h'],'fixed_baselines':fixed,
                         'key2_separate':{'all':float(km['separate']['rank_all']['top1_arm6_mse']),
                                          'hard':float(km['separate']['rank_hard']['top1_arm6_mse'])}}
    r['references']=refs
    # Teacher mass on different-task candidates is estimated on a fixed
    # evenly spaced memory-query sample; no Eval labels influence training.
    eh=_build_query_split(data,'eh',bank_actions,bt,be,dense=True)
    sample=np.linspace(0,len(eh['key'])-1,min(1024,len(eh['key']))).astype(np.int64)
    allow=~((eh['task'][sample,None]==bt[None,:])&(eh['episode'][sample,None]==be[None,:]))
    r['metadata']['train_teacher_cross_task_mass']=float(_teacher_cross_task_mass(
        eh['errors'][sample],allow,eh['task'][sample],bt,float(r['metadata']['selected_tau'])))
    r['metadata']['train_teacher_mass_sample_rows']=len(sample)
    _write_json(rp,r);_write_report(out,r)
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--data-cache',type=Path,default=None);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--force-build',action='store_true');ap.add_argument('--force-run',action='store_true');ap.add_argument('--finalize',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION','.25')))
    data,manifest=_load_data(a.output,cache_dir=a.data_cache,force_build=a.force_build)
    if a.finalize:return _finalize_cached_results(a.output,data,device=a.device)
    if a.smoke:
        # Slice all arrays consistently for a fast end-to-end structural pass.
        for key in list(data):
            if key.startswith(('eh__','vlmh__','dev__','learned__','unseen__')):
                arr=data[key]
                if arr.ndim: data[key]=arr[:min(len(arr),128)]
        manifest={**manifest,'smoke':True}
    return _eval_experiment(a.output,data,manifest,device=a.device,smoke=a.smoke,force=a.force_run)


if __name__=='__main__': main()
