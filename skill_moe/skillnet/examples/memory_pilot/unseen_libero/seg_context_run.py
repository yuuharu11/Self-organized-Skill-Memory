#!/usr/bin/env python3
"""Resumable SEG2 experiment driver.

Stages: bank -> train -> offline -> closed-loop. Large prefix extraction is
handled by seg_context_extract.py and is a separate process so JAX and Torch
do not retain accelerator allocations together. All stage products are written
under the supplied output root and are validated before reuse.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/work/.skillnet-state/outputs/memory-pilot')
RUN=ROOT/'pi05-memory40-20261104/run-01'
UNSEEN=ROOT/'unseen-libero/unseen-rows-02'
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import seg_context_retrieval as sc


def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');tmp.replace(p)

def log(out,stage,**kw):
    rec={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,**kw}
    line=json.dumps(rec,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))
    print(line,flush=True)
    with (Path(out)/'progress.jsonl').open('a') as f:f.write(line+'\n')


class FrameStore:
    """Read-only join of old stride-10 tokens and exact supplemental frame cache."""
    def __init__(self,run_root,extra_root):
        self.root=Path(run_root);self.extra=Path(extra_root)
        self.rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (self.root/'rows').glob('*.npy')}
        self.tok=np.load(self.root/'tokens.npy',mmap_mode='r',allow_pickle=False)
        self.grp=np.load(self.root/'groups.npy',mmap_mode='r',allow_pickle=False)
        self.row_at={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(self.rows['task'],self.rows['episode'],self.rows['frame']))}
        mp=self.extra/'manifest.json'
        if not mp.exists():raise FileNotFoundError(f'extraction incomplete: {mp}')
        self.manifest=json.loads(mp.read_text());self.extra_keys={tuple(map(int,k)):i for i,k in enumerate(self.manifest['keys'])}
        self.xt=np.load(self.extra/'tokens.npy',mmap_mode='r',allow_pickle=False)
        self.xg=np.load(self.extra/'groups.npy',mmap_mode='r',allow_pickle=False)
        self.xstate=np.load(self.extra/'state.npy',mmap_mode='r',allow_pickle=False)
        self.xactions=np.load(self.extra/'actions.npy',mmap_mode='r',allow_pickle=False)
        self.state_map={};self.action_map={}
    def frame(self,key):
        key=tuple(map(int,key))
        if key in self.row_at:
            i=self.row_at[key]
            return self.tok[i],self.grp[i],self.rows['state'][i],self.rows['actions'][i]
        if key not in self.extra_keys:raise KeyError(f'missing exact prefix/action frame: {key}')
        i=self.extra_keys[key]
        return self.xt[i],self.xg[i],self.xstate[i],self.xactions[i]
    def meta(self,key):
        key=tuple(map(int,key))
        if key in self.row_at:
            i=self.row_at[key];return self.rows['state'][i],self.rows['actions'][i]
        if key not in self.extra_keys:raise KeyError(f'missing exact metadata frame: {key}')
        i=self.extra_keys[key];return self.xstate[i],self.xactions[i]
    def tokens(self,keys):
        t=np.zeros((len(keys),712,2048),np.float16);m=np.zeros((len(keys),712),bool)
        for j,key in enumerate(keys):
            x,g,_,_=self.frame(key);n=min(712,len(x));t[j,:n]=x[:n];m[j,:n]=np.asarray(g[:n])>=0
        return t,m


class UnseenStore:
    def __init__(self,root):
        self.root=Path(root);self.arr={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in self.root.glob('*.npy')}
    def tokens(self,keys):
        ids=np.asarray([int(k[0]) for k in keys],np.int64)
        return self.arr['tokens'][ids],self.arr['groups'][ids]>=0


def build_unseen_queries(root,*,split,official_stride10=False):
    store=UnseenStore(root);a=store.arr;use=(a['is_dev'] if split=='dev' else ~a['is_dev'])
    if split=='eval' and official_stride10:
        use=use & ((a['frame'] % 10)==0)
    byep={}
    for i,(t,e) in enumerate(zip(a['task'],a['episode'])):byep.setdefault((int(t),int(e)),[]).append(i)
    out={k:[] for k in ('key','task','episode','actions','state','previous_velocity','history_actions','history_states','history_mask',
                        'history_stream_actions','history_stream_states','history_stream_mask')}
    for (task,ep),ids in sorted(byep.items()):
        ids=sorted(ids,key=lambda i:int(a['frame'][i]))
        # Rebuild the unique action trace from overlapping query windows. A
        # stride-5 row contributes only its last five new actions; stride-10
        # rows contribute all ten. This keeps history causal and avoids twice
        # counting the overlap.
        first=int(a['frame'][ids[0]]);parts=[np.asarray(a['actions'][ids[0],:10,:7],np.float32)]
        prev=int(a['frame'][ids[0]])
        for i in ids[1:]:
            fnow=int(a['frame'][i]);delta=fnow-prev;chunk=np.asarray(a['actions'][i,:10,:7],np.float32)
            if not 0<delta<=10:raise ValueError(f'unseen query frame gap {delta}')
            parts.append(chunk[delta:])
            prev=fnow
        trace=np.concatenate(parts,axis=0)
        st=np.repeat(np.asarray(a['state'][ids[0],:8],np.float32)[None],len(trace),axis=0)
        for i in ids:
            f=int(a['frame'][i])-first;st[f:min(f+5,len(st))]=np.asarray(a['state'][i,:8],np.float32)
        for i in ids:
            if not use[i]:continue
            f=int(a['frame'][i])-first;hist=sc.causal_history_at(trace,st,f);ha,hs,hm=_pad_history(hist)
            out['key'].append((int(i),));out['task'].append(task);out['episode'].append(1_000_000+ep)
            out['actions'].append(np.asarray(a['actions'][i,:10,:7],np.float32));out['state'].append(np.asarray(a['state'][i,:8],np.float32))
            out['previous_velocity'].append(sc._physical6(trace[f-1:f,:6])[0,:3] if f>0 else np.zeros(3,np.float32))
            out['history_actions'].append(ha);out['history_states'].append(hs);out['history_mask'].append(hm)
            # The policy detects completed boundaries on the full causal
            # prefix and only then clips the selected segment to 40 steps.
            # Preserve the complete causal prefix: the segmenter can depend on
            # boundaries before the encoder's 40-step history cap.
            sa,ss,sm=_pad_history((trace[:f],st[:f],None),maxlen=2048)
            out['history_stream_actions'].append(sa);out['history_stream_states'].append(ss);out['history_stream_mask'].append(sm)
    return store,{k:np.asarray(v) for k,v in out.items()}


def evaluate_selector(model,store,bank,mem_inputs,queries,device='cuda',*,same_task_exclusion=False,query_store=None):
    import torch
    model.eval();mkey=_encode_items(model,store,bank,mem_inputs,device)
    qstore=query_store or store
    qkeys=[]
    # Singleton query inference mirrors the websocket server and avoids
    # batch-shape-dependent CUDA kernel rounding in pose prediction.
    for start in range(0,len(queries['key'])):
        ix=np.asarray([start]);tok,mask=qstore.tokens(queries['key'][ix].tolist())
        with torch.no_grad():
            q=model.encoder(torch.as_tensor(tok,device=device,dtype=torch.float32),torch.as_tensor(mask,device=device),
                torch.as_tensor(queries['history_actions'][ix],device=device),torch.as_tensor(queries['history_states'][ix],device=device),
                torch.as_tensor(queries['state'][ix],device=device),torch.as_tensor(queries['history_mask'][ix],device=device))
        qkeys.append(q.cpu().numpy())
    qkey=np.concatenate(qkeys);score_scale=float(model.encoder.logit_scale.detach().exp().clamp(max=100.).cpu())
    if model.encoder.config.continuity_weight=='learned':
        scale=float(torch.nn.functional.softplus(model.encoder.learned_continuity_scale).clamp(max=10.).detach().cpu())
    else:scale=float(model.encoder.config.continuity_weight)
    # Use the same function as the online policy (including normalization,
    # logit scale, continuity ordering, and exclusion masks) for parity.
    qten=torch.as_tensor(qkey,device=device,dtype=torch.float32)
    mten=torch.as_tensor(mkey,device=device,dtype=torch.float32)
    continuity=None
    if scale:
        prev=torch.as_tensor(queries['previous_velocity'],device=device,dtype=torch.float32)
        start=torch.as_tensor(bank['start_velocity'],device=device,dtype=torch.float32)
        continuity=sc.continuity_bonus(start[None,:,:],prev[:,None,:])
    score_t,allowed_t=sc.retrieval_scores(qten,mten,query_episode=queries['episode'],memory_episode=bank['episode'],
        query_task=queries['task'],memory_task=bank['task'],exclude_same_task=same_task_exclusion,
        continuity=continuity,continuity_weight=scale,score_scale=score_scale)
    scores=score_t.cpu().numpy();allowed=allowed_t.cpu().numpy()
    errors=sc.s_shared_type_error(queries['actions'][:,:,:6],bank['value_actions10'][:,:,:6])
    top=np.argmax(scores,axis=1);row=np.arange(len(top));sel=errors[row,top]
    from scipy.stats import spearmanr
    rho=np.asarray([spearmanr(scores[i],errors[i]).statistic for i in range(len(top))],np.float64)
    # Action-space score reports selected shape as a reference before pose.
    mse=np.mean((queries['actions'][:,:,:6]-bank['value_actions10'][top,:,:6])**2,axis=(1,2))*6/7
    return {'rows':len(top),'type_error':float(sel.mean()),'spearman':float(np.nanmean(rho)),
        'recovery_at_1':float(np.mean(sel<=errors.min(1)+.01)),'selected_raw_action_mse6_7':float(mse.mean()),
        'score':scores,'type_errors':errors,'top':top,'query_keys':qkey,'memory_keys':mkey}


def prepare_bank(out,run_root=RUN):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(run_root)/'rows').glob('*.npy')}
    ids=np.load(Path(run_root)/'memory_row_ids.npy',allow_pickle=False)
    bank=sc.build_memory_segments(rows,ids,holdout_tasks=sc.HELDOUT_TASKS,min_length=10)
    if len(bank['length'])==0:raise ValueError('empty memory bank')
    path=out/'memory_bank.npz'
    tmp=out/'memory_bank.tmp.npz';np.savez(tmp,**bank);tmp.replace(path)
    meta={'units':len(bank['length']),'task_count':len(np.unique(bank['task'])),'heldout_tasks':list(sc.HELDOUT_TASKS),
          'source_memory_rows':int(len(ids)),'first_source':bank['source'][:5].tolist(),'value_space':'physical DCT xyz+rotation; raw value_actions10 normalized active7',
          'value_head_mode_options':['dct','split']}
    atomic_json(out/'bank.json',meta);return bank,meta


def audit_inputs(out,run_root=RUN,unseen_root=UNSEEN):
    out=Path(out);run_manifest=json.loads((Path(run_root)/'data_manifest.json').read_text())
    unseen=json.loads((Path(unseen_root)/'manifest.json').read_text())
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(run_root)/'rows').glob('*.npy')}
    bank=np.load(out/'memory_bank.npz',allow_pickle=False)
    return {'run_rows':len(rows['task']),'run_stride':int(run_manifest['stride']),'unseen_rows':unseen['rows'],
            'unseen_stride_description':unseen['windows'],'unseen_eval_tasks':unseen['eval_tasks'],
            'unseen_dev_tasks':unseen['dev_tasks'],'memory_units':len(bank['length']),
            'heldout_train_tasks':list(sc.HELDOUT_TASKS),'eval_features_existing_stride10_only':True,
            'raw_hdf5_root':'/work/.skillnet-state/data/libero/datasets/libero_90',
            'warning':'LIBERO-90 cache currently has stride-10 windows; stride-5 exact-frame extraction is not yet wired in this stage.'}


def _pad_history(history,maxlen=40):
    aa=np.zeros((maxlen,7),np.float32);ss=np.zeros((maxlen,8),np.float32);mask=np.zeros(maxlen,bool)
    if history is not None:
        a,s,_=history;a=np.asarray(a,np.float32)[-maxlen:];s=np.asarray(s,np.float32)[-maxlen:]
        aa[:len(a)]=a[:,:7];ss[:len(s)]=s[:,:8];mask[:len(a)]=True
    return aa,ss,mask


def build_fit_queries(store,*,heldout_tasks=sc.HELDOUT_TASKS):
    """Assemble all non-memory LIBERO-40 stride-5 query windows, causally."""
    manifest=json.loads((store.root/'data_manifest.json').read_text());rows=store.rows
    by_episode={}
    for rec in manifest['episodes']:
        task,ep=int(rec['task']),int(rec['episode'])
        ids=np.flatnonzero((rows['task']==task)&(rows['episode']==ep))
        if not len(ids):continue
        ids=ids[np.argsort(rows['frame'][ids])]
        trace=np.concatenate(np.asarray(rows['actions'][ids])[:,:,:7],axis=0).astype(np.float32)
        state_trace=np.repeat(np.asarray(rows['state'][ids,:8],np.float32),10,axis=0)[:len(trace)]
        # Exact metadata at intermediate cached prefixes improves state history.
        for f in range(5,len(trace),10):
            key=(task,ep,f)
            if key in store.extra_keys:state_trace[f,:8]=store.xstate[store.extra_keys[key],:8]
        by_episode[(task,ep)]=(rec,trace,state_trace)
    out={k:[] for k in ('key','task','episode','actions','state','previous_velocity','history_actions','history_states','history_mask')}
    for (task,ep),(rec,trace,state_trace) in sorted(by_episode.items()):
        if rec.get('memory',False) or task in heldout_tasks:continue
        for f in range(0,max(0,len(trace)-9),5):
            key=(task,ep,f)
            if key not in store.row_at and key not in store.extra_keys:continue
            state,actions=store.meta(key)
            if f%10==0:future=np.asarray(actions,np.float32)[:10,:7]
            else:future=np.asarray(store.xactions[store.extra_keys[key]],np.float32)[:10,:7]
            hist=sc.causal_history_at(trace,state_trace,f)
            ha,hs,hm=_pad_history(hist)
            out['key'].append(key);out['task'].append(task);out['episode'].append(ep)
            out['actions'].append(future);out['state'].append(np.asarray(state,np.float32)[:8])
            out['previous_velocity'].append(sc._physical6(trace[f-1:f,:6])[0,:3] if f>0 else np.zeros(3,np.float32))
            out['history_actions'].append(ha);out['history_states'].append(hs);out['history_mask'].append(hm)
    if not out['key']:raise ValueError('no eligible stride-5 fit queries assembled')
    return {k:np.asarray(v) for k,v in out.items()}


def build_memory_inputs(store,bank):
    """Join each variable segment with its exact start-frame prefix and history."""
    manifest=json.loads((store.root/'data_manifest.json').read_text());rows=store.rows
    traces={};state_traces={}
    for t,e in sorted(set(zip(rows['task'].tolist(),rows['episode'].tolist()))):
        ids=np.flatnonzero((rows['task']==t)&(rows['episode']==e));ids=ids[np.argsort(rows['frame'][ids])]
        traces[(int(t),int(e))]=np.concatenate(np.asarray(rows['actions'][ids])[:,:,:7],axis=0).astype(np.float32)
        state_traces[(int(t),int(e))]=np.repeat(np.asarray(rows['state'][ids,:8],np.float32),10,axis=0)
    n=len(bank['length']);out={k:[] for k in ('key','state','history_actions','history_states','history_mask')}
    for t,e,f,_ in bank['source']:
        trace=traces[(int(t),int(e))];states=state_traces[(int(t),int(e))]
        hist=sc.causal_history_at(trace,states,int(f))
        ha,hs,hm=_pad_history(hist)
        key=(int(t),int(e),int(f));state,_=store.meta(key)
        out['key'].append(key);out['state'].append(np.asarray(state,np.float32)[:8])
        out['history_actions'].append(ha);out['history_states'].append(hs);out['history_mask'].append(hm)
    return {k:np.asarray(v) for k,v in out.items()}


def _positive_indices(queries,bank,*,block=128):
    best=np.empty(len(queries['actions']),np.int64);besterr=np.empty(len(best),np.float32)
    mem=bank['value_actions10'][:,:,:6]
    for start in range(0,len(best),block):
        end=min(len(best),start+block)
        err=sc.s_shared_type_error(queries['actions'][start:end,:,:6],mem)
        allowed=(queries['task'][start:end,None]!=bank['task'][None,:]) & (queries['episode'][start:end,None]!=bank['episode'][None,:])
        err[~allowed]=np.inf
        best[start:end]=np.argmin(err,axis=1);besterr[start:end]=err[np.arange(end-start),best[start:end]]
    if not np.isfinite(besterr).all():raise ValueError('some fit queries have no eligible positive segment')
    return best,besterr


def _torch_batch(model,store,bank,mi,qi,positive_local,queries,mem_inputs,device,*,same_task=True,rng=None):
    import torch
    from scipy.spatial.distance import cdist
    qkeys=queries['key'][qi].tolist();mkeys=mem_inputs['key'][mi].tolist()
    qt,qm=store.tokens(qkeys);mt,mm=store.tokens(mkeys)
    qact=queries['actions'][qi];mact=bank['value_actions10'][mi]
    te=sc.s_shared_type_error(qact[:,:,:6],mact[:,:,:6]);ad=sc.future_action_mse(qact,mact)
    ha=queries['history_actions'][qi];hs=queries['history_states'][qi];hm=queries['history_mask'][qi]
    mha=mem_inputs['history_actions'][mi];mhs=mem_inputs['history_states'][mi];mhm=mem_inputs['history_mask'][mi]
    qv=np.asarray(queries['previous_velocity'][qi],np.float32)
    continuity=-np.linalg.norm(qv[:,None,:]-bank['start_velocity'][mi][None,:,:],axis=-1)
    positive_value=model.encode_value(torch.as_tensor(mact[positive_local],device=device),
        torch.as_tensor(bank['coefficients'][mi[positive_local]],device=device))
    return {
      'query_tokens':torch.as_tensor(qt,device=device,dtype=torch.float32),'query_token_mask':torch.as_tensor(qm,device=device),
      'history_actions':torch.as_tensor(ha,device=device),'history_states':torch.as_tensor(hs,device=device),'history_mask':torch.as_tensor(hm,device=device),
      'query_state':torch.as_tensor(queries['state'][qi],device=device),
      'memory_tokens':torch.as_tensor(mt,device=device,dtype=torch.float32),'memory_token_mask':torch.as_tensor(mm,device=device),
      'memory_history_actions':torch.as_tensor(mha,device=device),'memory_history_states':torch.as_tensor(mhs,device=device),'memory_history_mask':torch.as_tensor(mhm,device=device),
      'memory_state':torch.as_tensor(mem_inputs['state'][mi],device=device),
      'query_episode':torch.as_tensor(queries['episode'][qi],device=device),'memory_episode':torch.as_tensor(bank['episode'][mi],device=device),
      'query_task':torch.as_tensor(queries['task'][qi],device=device),'memory_task':torch.as_tensor(bank['task'][mi],device=device),
      'type_error':torch.as_tensor(te,device=device),'action_distance':torch.as_tensor(ad,device=device),
      'positive_memory_index':torch.as_tensor(positive_local,device=device),
      'positive_coefficients':torch.as_tensor(bank['coefficients'][mi[positive_local]],device=device),
      'positive_value_features':positive_value,
      'pose_target':torch.as_tensor(np.stack([sc.fit_pose_scale(bank['value_actions10'][mi[positive_local[j]],:,:6],qact[j,:,:6]) for j in range(len(qi))]),device=device),
      'continuity_features':torch.as_tensor(continuity,device=device),
    }


def _encode_items(model,store,bank,inputs,device,batch=8):
    import torch
    enc=model.encoder.eval();vec=[]
    for start in range(0,len(bank['length']),batch):
        ix=np.arange(start,min(len(bank['length']),start+batch));tok,mask=store.tokens(inputs['key'][ix].tolist())
        with torch.no_grad():
            z=enc(torch.as_tensor(tok,device=device,dtype=torch.float32),torch.as_tensor(mask,device=device),
                torch.as_tensor(inputs['history_actions'][ix],device=device),torch.as_tensor(inputs['history_states'][ix],device=device),
                torch.as_tensor(inputs['state'][ix],device=device),torch.as_tensor(inputs['history_mask'][ix],device=device))
        vec.append(z.cpu().numpy())
    return np.concatenate(vec)


def train_one(out,store,bank,queries,mem_inputs,dev_store,dev_queries,*,seed,config,max_epochs=8,steps_per_epoch=128,batch_q=16,device='cuda'):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    torch.manual_seed(int(seed));np_rng=np.random.default_rng(int(seed))
    cfg=sc.SegmentConfig(token_dim=2048,width=256,heads=4,cross_layers=int(config['layers']),
        continuity_weight=config['continuity'],exclude_same_task=True,value_head_mode=config.get('value_head_mode','dct'))
    model=sc.SegmentContextModel(cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    positives,perr=_positive_indices(queries,bank)
    gen=torch.Generator(device=device).manual_seed(int(seed)+1)
    history=[];best=None;best_epoch=0;best_state=None;stale=0
    for epoch in range(1,int(max_epochs)+1):
        model.train();losses=[];order=np_rng.permutation(len(queries['key']))
        for step in range(int(steps_per_epoch)):
            qi=order[(step*batch_q)%len(order):min((step+1)*batch_q,len(order))]
            if len(qi)<batch_q:qi=order[:batch_q]
            pos=positives[qi];mi=list(dict.fromkeys(pos.tolist()))
            while len(mi)<max(batch_q*2,32):
                cand=int(np_rng.integers(len(bank['length'])))
                if cand not in mi:mi.append(cand)
            lookup={v:i for i,v in enumerate(mi)};p_local=np.asarray([lookup[x] for x in pos],np.int64)
            batch=_torch_batch(model,store,bank,np.asarray(mi),qi,p_local,queries,mem_inputs,device)
            opt.zero_grad(set_to_none=True)
            res=sc.training_objective(model.encoder,model.pose_head,batch,tau=float(config['tau']),
                tau_action=.03,same_task_exclusion=True,generator=gen)
            res['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append([float(res[k].detach()) for k in ('loss','contrastive','pose')])
        mean=np.asarray(losses).mean(0)
        record={'epoch':epoch,'loss':mean.tolist(),'positive_type_error_mean':float(perr.mean())}
        history.append(record)
        # Early stop / checkpoint selection uses only LIBERO-90 Dev. Eval rows
        # are not constructed until the frozen checkpoints are scored.
        val=evaluate_selector(model,store,bank,mem_inputs,dev_queries,device,query_store=dev_store)
        record['dev']={k:val[k] for k in ('type_error','spearman','recovery_at_1','selected_raw_action_mse6_7')}
        score=(float(val['type_error']),-float(val['spearman']))
        if best is None or score < best:
            import copy
            best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
            torch.save({'state_dict':best_state,'config':config,'seed':seed,'epoch':epoch,'dev':record['dev']},Path(out)/f'model_seed{seed}_{config["name"]}.pt')
        else:stale+=1
        if stale>=3:break
    atomic_json(Path(out)/f'train_seed{seed}_{config["name"]}.json',{'seed':seed,'config':config,'epochs':history,'selected_epoch':best_epoch,
        'criterion':'LIBERO-90 Dev type error, Spearman tie-break; Eval unseen labels are not read','query_count':len(queries['key']),'memory_count':len(bank['length'])})
    if best_state is None:raise RuntimeError('no checkpoint selected on Dev')
    model.load_state_dict(best_state)
    return model,history


def train_pipeline(out,*,run_root=RUN,unseen_root=UNSEEN,extra_root=None,device='cuda',max_epochs=6,steps_per_epoch=64):
    """Dev-only 8-way config selection, then three locked final seeds."""
    import torch
    out=Path(out);extra_root=Path(extra_root or out/'seg5-cache')
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.40')))
    if not torch.cuda.is_available() and str(device).startswith('cuda'):raise RuntimeError('GPU requested but torch.cuda is unavailable')
    bank=np.load(out/'memory_bank.npz',allow_pickle=False)
    store=FrameStore(run_root,extra_root);mem_inputs=build_memory_inputs(store,bank)
    queries=build_fit_queries(store)
    dev_store,dev=build_unseen_queries(unseen_root,split='dev')
    configs=[]
    for tau in (.01,.03):
        for layers in (1,2):
            for continuity in (0.,'learned'):
                configs.append({'name':f't{tau:g}_l{layers}_c{"learn" if continuity=="learned" else "0"}',
                    'tau':tau,'layers':layers,'continuity':continuity,'value_head_mode':'dct'})
    tune=[];tune_root=out/'tuning';tune_root.mkdir(exist_ok=True)
    for cfg in configs:
        sub=tune_root/cfg['name'];sub.mkdir(exist_ok=True)
        model,hist=train_one(sub,store,bank,queries,mem_inputs,dev_store,dev,seed=4101,config=cfg,
            max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,device=device)
        met=evaluate_selector(model,store,bank,mem_inputs,dev,device,query_store=dev_store)
        tune.append({'config':cfg,'epochs':len(hist),'dev':{k:met[k] for k in ('type_error','spearman','recovery_at_1','selected_raw_action_mse6_7')}})
        log(out,'tune_config',name=cfg['name'],**tune[-1]['dev'])
    chosen=min(tune,key=lambda x:(x['dev']['type_error'],-x['dev']['spearman']))
    atomic_json(out/'selected_config.json',{'selection_split':'LIBERO-90 Dev tasks only','selected':chosen,'all_candidates':tune,
        'selection_metric':'mean S-shared type error; Spearman tie-break','eval_touched':False})
    final_root=out/'models';final_root.mkdir(exist_ok=True);final=[]
    for seed in (4101,4102,4103):
        sub=final_root/f'seed{seed}';sub.mkdir(exist_ok=True)
        model,hist=train_one(sub,store,bank,queries,mem_inputs,dev_store,dev,seed=seed,config=chosen['config'],
            max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,device=device)
        ck=sub/f'model_seed{seed}_{chosen["config"]["name"]}.pt';best_ck=torch.load(ck,map_location='cpu',weights_only=False)
        final.append({'seed':seed,'epochs':len(hist),'selected_epoch':best_ck['epoch'],'checkpoint':str(ck),
                      'dev':best_ck.get('dev')})
        log(out,'final_seed',seed=seed,epochs=len(hist),selected_epoch=best_ck['epoch'],dev=best_ck.get('dev'))
    atomic_json(out/'training_summary.json',{'selected':chosen['config'],'final':final,'query_count':len(queries['key']),
        'memory_units':len(bank['length']),'fit_tasks':sorted(map(int,np.unique(queries['task']))),'eval_labels_used':False})
    return chosen,final


def _load_model(out,seed,bank):
    import torch
    sel=json.loads((Path(out)/'selected_config.json').read_text())['selected']['config']
    ck=Path(out)/'models'/f'seed{seed}'/f'model_seed{seed}_{sel["name"]}.pt'
    payload=torch.load(ck,map_location='cpu',weights_only=False)
    model=sc.SegmentContextModel(sc.SegmentConfig(token_dim=2048,width=256,heads=4,cross_layers=sel['layers'],
        continuity_weight=sel['continuity'],value_head_mode=sel.get('value_head_mode','dct')))
    model.load_state_dict(payload['state_dict']);model.eval();return model,sel,ck


def _normal_predictions(unseen_root,output_path,*,checkpoint=CHECKPOINT,batch_size=4,num_steps=10):
    """Frozen π0.5 actions for the cached unseen query windows, never labels."""
    import importlib.util, jax, numpy as np
    from openpi.shared.nnx_utils import module_jit
    run_path=HERE.parent/'pi05_memory'/'run.py';spec=importlib.util.spec_from_file_location('seg_context_pi05_run',run_path)
    run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)
    model,_=run.load_policy(checkpoint);sample=module_jit(model.sample_actions,static_argnames=('num_steps',))
    store=UnseenStore(unseen_root);arr=store.arr;n=len(arr['task']);output_path=Path(output_path)
    if output_path.exists():raise FileExistsError(f'refusing to overwrite normal predictions: {output_path}')
    pred=np.lib.format.open_memmap(output_path,mode='w+',dtype=np.float32,shape=(n,10,7))
    for st in range(0,n,batch_size):
        ix=np.arange(st,min(n,st+batch_size));obsdata={'image_base':arr['image_base'][ix],'image_wrist':arr['image_wrist'][ix],
            'state':arr['state'][ix],'prompt':arr['prompt'][ix],'prompt_mask':arr['prompt_mask'][ix]}
        obs=run.obs_batch(obsdata,np.arange(len(ix)));key=jax.random.PRNGKey(4101+st)
        act=np.asarray(sample(key,obs,num_steps=num_steps))[:,:,:7];pred[ix]=act
    pred.flush();del pred
    return np.load(output_path,mmap_mode='r',allow_pickle=False)


def offline_evaluate(out,*,unseen_root=UNSEEN,device='cuda',checkpoint=CHECKPOINT):
    """Locked Eval report: selected retrieval plus independent ablations."""
    import torch
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.40')))
    out=Path(out);bank=np.load(out/'memory_bank.npz',allow_pickle=False)
    run_root=RUN;store=FrameStore(run_root,out/'seg5-cache');mem_inputs=build_memory_inputs(store,bank)
    qstore,q=build_unseen_queries(unseen_root,split='eval',official_stride10=True)
    if len(q['key'])!=619:raise AssertionError(f'expected fixed 619 Eval windows, got {len(q["key"])}')
    normal_path=out/'normal_actions_stride10.npy'
    if not normal_path.exists():raise FileNotFoundError('run the JAX-only normal stage first')
    normal=np.load(normal_path,mmap_mode='r',allow_pickle=False)
    # map held-out rows in the feature-cache order to the Eval subset
    unseen=UnseenStore(unseen_root);all_eval=np.asarray([int(k[0]) for k in q['key']],np.int64)
    old=UnseenStore(UNSEEN);old_lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(old.arr['task'],old.arr['episode'],old.arr['frame']))}
    normal_rows=[]
    for j,key in enumerate(q['key']):
        idx=int(key[0]);frame=int(unseen.arr['frame'][idx]);episode=int(q['episode'][j]-1_000_000);task=int(q['task'][j])
        oldkey=(task,episode,frame)
        if oldkey not in old_lookup:raise AssertionError(f'no original π0.5 row for fixed Eval key {oldkey}')
        normal_rows.append(old_lookup[oldkey])
    normal_rows=np.asarray(normal_rows,np.int64)
    if len(all_eval)!=619:raise AssertionError('Eval row count changed')
    target=q['actions'];normal_action=np.asarray(normal[normal_rows])
    refpath=ROOT/'unseen-libero/lib-closed-01/eval_rows.npz'
    hard=np.zeros(len(all_eval),bool);reference={}
    if refpath.exists():
        ref=np.load(refpath,allow_pickle=False)
        expected_ref_episode=(q['episode']-1_000_000)+q['task']*100_000
        if not (np.array_equal(ref['task'],q['task']) and np.array_equal(ref['episode'],expected_ref_episode) and
                np.allclose(ref['target'],target[:,:,:6],atol=1e-6)):
            raise AssertionError('SEG2 Eval rows do not align with fixed K1 Eval rows')
        hard=np.asarray(ref['hard'],bool)
        reference={'lib1_R5_mse6_7_mean':float(np.mean(ref['R5_row_mse6_over7'][0])),
                   'lib1_R5_hard_mse6_7_mean':float(np.mean(ref['R5_row_mse6_over7'][0,hard]))}
    bootstrap_episode=q['task'].astype(np.int64)*100000+q['episode'].astype(np.int64)
    # Zero-padding baseline and one-step hold use the prior target action only
    # as the held-out scoring reference; no target enters model inputs.
    hold=np.zeros_like(target)
    frames=np.asarray(unseen.arr['frame'])[all_eval]
    for key in sorted(set(zip(q['task'].tolist(),q['episode'].tolist()))):
        loc=np.flatnonzero((q['task']==key[0])&(q['episode']==key[1]));loc=loc[np.argsort(frames[loc])]
        for j,pos in enumerate(loc[1:],1):
            delta=int(frames[pos]-frames[loc[j-1]])
            prev_action=target[loc[j-1],min(delta-1,9),:]
            hold[pos]=prev_action[None,:]
    reports=[];per_seed=[];ablation_rows=[]
    for seed in (4101,4102,4103):
        model,sel,ck=_load_model(out,seed,bank);model.to(device).eval()
        met=evaluate_selector(model,store,bank,mem_inputs,q,device,query_store=qstore)
        scores=met['score'];top=met['top'];qkey=met['query_keys']
        dev=device;theta=[];draft=[]
        with torch.no_grad():
            for i,ix in enumerate(top):
                qi=torch.as_tensor(qkey[i:i+1],device=dev,dtype=torch.float32)
                ki=torch.as_tensor(met['memory_keys'][ix:ix+1],device=dev,dtype=torch.float32)
                coeff=torch.as_tensor(bank['coefficients'][ix:ix+1],device=dev,dtype=torch.float32)
                theta.append(model.pose_head(qi,ki,coeff).cpu().numpy()[0])
        theta=np.asarray(theta,np.float32)
        for i,ix in enumerate(top):
            draft.append(sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta[i],
                grip_actions=normal_action[i,:,6])[0])
        draft=np.asarray(draft);base_err=np.mean((normal_action[:,:,:6]-target[:,:,:6])**2,axis=(1,2))*6/7
        selected_err=np.mean((draft[:,:,:6]-target[:,:,:6])**2,axis=(1,2))*6/7
        selected_first=np.mean((draft[:,:5,:6]-target[:,:5,:6])**2,axis=(1,2))*6/7
        selected_cont=np.mean((draft[:,5:10,:6]-target[:,5:10,:6])**2,axis=(1,2))*6/7
        def action_metrics(pred):
            e=np.mean((np.asarray(pred)[:,:,:6]-target[:,:,:6])**2,axis=(1,2))*6/7
            return {'mse6_over7_all':float(e.mean()),'mse6_over7_hard':float(e[hard].mean()) if hard.any() else None,
                'first5':float(np.mean((np.asarray(pred)[:,:5,:6]-target[:,:5,:6])**2)*6/7),
                'continuation5':float(np.mean((np.asarray(pred)[:,5:10,:6]-target[:,5:10,:6])**2)*6/7)}
        # Required one-at-a-time diagnostics on the frozen selected checkpoint.
        # No-pose uses the same π0.5 Procrustes fitting path as SEG1.
        no_pose=[]
        for i,ix in enumerate(top):
            raw=sc.segment_value_decode(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],out_length=10)
            theta0=sc.fit_pose_scale(sc._normalized6(raw),normal_action[i,:,:6])
            no_pose.append(sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta0,
                grip_actions=normal_action[i,:,6])[0])
        no_pose=np.asarray(no_pose)
        # History ablation zeros only the GRU inputs for query and memory.
        q0={k:np.array(v,copy=True) for k,v in q.items()}
        mi0={k:np.array(v,copy=True) for k,v in mem_inputs.items()}
        for data in (q0,mi0):
            for name in ('history_actions','history_states','history_mask'):data[name][...] = 0
        nohist=evaluate_selector(model,store,bank,mi0,q0,device,query_store=qstore)
        nohist_top=nohist['top'];nohist_drafts=[]
        with torch.no_grad():
            for i,ix in enumerate(nohist_top):
                theta_i=model.pose_head(torch.as_tensor(nohist['query_keys'][i:i+1],device=device,dtype=torch.float32),
                    torch.as_tensor(nohist['memory_keys'][ix:ix+1],device=device,dtype=torch.float32),
                    torch.as_tensor(bank['coefficients'][ix:ix+1],device=device,dtype=torch.float32)).cpu().numpy()[0]
                nohist_drafts.append(sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta_i,
                    grip_actions=normal_action[i,:,6])[0])
        # Remove only the continuity score term; embeddings remain identical.
        base_scores=met['query_keys']@met['memory_keys'].T*np.exp(float(model.encoder.logit_scale.detach().cpu()))
        allowed=q['episode'][:,None]!=bank['episode'][None,:]
        base_scores=np.where(allowed,base_scores,-np.inf)
        no_cont_top=np.argmax(base_scores,axis=1)
        no_cont_drafts=[]
        with torch.no_grad():
            for i,ix in enumerate(no_cont_top):
                theta_i=model.pose_head(torch.as_tensor(met['query_keys'][i:i+1],device=device,dtype=torch.float32),
                    torch.as_tensor(met['memory_keys'][ix:ix+1],device=device,dtype=torch.float32),
                    torch.as_tensor(bank['coefficients'][ix:ix+1],device=device,dtype=torch.float32)).cpu().numpy()[0]
                no_cont_drafts.append(sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta_i,
                    grip_actions=normal_action[i,:,6])[0])
        # Sequentially remove the immediately previous retrieved segment for
        # each query episode, without consulting target actions.
        no_prev_top=np.asarray(top).copy();frame_by_row={int(x[0]):int(unseen.arr['frame'][int(x[0])]) for x in q['key']}
        for epkey in sorted(set(zip(q['task'].tolist(),q['episode'].tolist()))):
            loc=np.flatnonzero((q['task']==epkey[0])&(q['episode']==epkey[1]))
            loc=loc[np.argsort([frame_by_row[int(q['key'][j][0])] for j in loc])]
            previous=None
            for j in loc:
                candidates=met['score'][j].copy()
                if previous is not None:candidates[previous]=-np.inf
                no_prev_top[j]=int(np.argmax(candidates));previous=int(no_prev_top[j])
        no_prev_drafts=[]
        with torch.no_grad():
            for i,ix in enumerate(no_prev_top):
                theta_i=model.pose_head(torch.as_tensor(met['query_keys'][i:i+1],device=device,dtype=torch.float32),
                    torch.as_tensor(met['memory_keys'][ix:ix+1],device=device,dtype=torch.float32),
                    torch.as_tensor(bank['coefficients'][ix:ix+1],device=device,dtype=torch.float32)).cpu().numpy()[0]
                no_prev_drafts.append(sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta_i,
                    grip_actions=normal_action[i,:,6])[0])
        ablation_rows.append({'seed':seed,'selected':action_metrics(draft),'pose_head_off_procrustes':action_metrics(no_pose),
            'gru_history_off':action_metrics(nohist_drafts),'continuity_score_off':action_metrics(no_cont_drafts),
            'exclude_previous_segment':action_metrics(no_prev_drafts)})
        per_seed.append({'seed':seed,'checkpoint':str(ck),'retrieval':{k:met[k] for k in ('type_error','spearman','recovery_at_1')},
            'mse6_over7_all':float(selected_err.mean()),'mse6_over7_first5':float(selected_first.mean()),
            'mse6_over7_continuation5':float(selected_cont.mean()),'normal_mse6_over7':float(base_err.mean()),
            'zero_mse6_over7':float(np.mean((target[:,:,:6])**2)*6/7),
            'hold_last_mse6_over7':float(np.mean((hold[:,:,:6]-target[:,:,:6])**2)*6/7),
            'hard_mse6_over7':float(selected_err[hard].mean()) if hard.any() else None,
            'normal_hard_mse6_over7':float(base_err[hard].mean()) if hard.any() else None,
            'paired_vs_normal_all':sc.paired_episode_bootstrap(selected_err,base_err,bootstrap_episode),
            'paired_vs_normal_hard':sc.paired_episode_bootstrap(selected_err[hard],base_err[hard],bootstrap_episode[hard]) if hard.any() else None,
            'paired_vs_hold_all':sc.paired_episode_bootstrap(selected_err,np.mean((hold[:,:,:6]-target[:,:,:6])**2,axis=(1,2))*6/7,bootstrap_episode),
            'paired_vs_hold_hard':sc.paired_episode_bootstrap(selected_err[hard],np.mean((hold[:,:,:6]-target[:,:,:6])**2,axis=(1,2))[hard]*6/7,bootstrap_episode[hard]) if hard.any() else None,
            'predicted_log_scale':theta[:,3].tolist(),'predicted_rotation_rad':np.linalg.norm(theta[:,:3],axis=1).tolist(),
            'fallback_fraction':float(np.mean(np.asarray([not np.any(x['history_mask']) for x in []]))) if False else None})
        np.savez_compressed(out/f'eval_seed{seed}.npz',rows=all_eval,top=top,scores=scores,theta=theta,drafts=draft,
                            target=target,normal=normal_action,type_errors=met['type_errors'])
    atomic_json(out/'offline_summary.json',{'selection':'frozen Dev-only selected config','eval_rows':len(q['key']),
        'hard_rows':int(hard.sum()),'reference':reference,'seeds':per_seed,
        'caveat':'offline selected segment actions use π0.5 gripper; evaluate split is sourced from stride-5 HDF5 rows when requested; see input_audit.'})
    atomic_json(out/'offline_ablation_summary.json',{'selection':'same Dev-selected configuration, one change at a time','rows':ablation_rows,
        'meaning':'pose-head-off uses π0.5 Procrustes; GRU off zeros history tensors; continuity off removes score term; previous-segment exclusion is sequential by episode with no target action use'})
    return per_seed


def prepare_online_bank(out,*,run_root=RUN,extra_root=None,device='cuda'):
    """Freeze the train-only bank embeddings used by the websocket adapter."""
    import torch
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.40')))
    out=Path(out);path=out/'online_bank.npz'
    if path.exists():raise FileExistsError(f'refusing to overwrite {path}')
    bank=np.load(out/'memory_bank.npz',allow_pickle=False);store=FrameStore(run_root,extra_root or out/'seg5-cache')
    mem=build_memory_inputs(store,bank);model,selected,ck=_load_model(out,4101,bank);model.to(device)
    keys=_encode_items(model,store,bank,mem,device)
    payload={k:bank[k] for k in bank.files};payload.update({'keys':keys,'memory_history_actions':mem['history_actions'],
        'memory_history_states':mem['history_states'],'memory_history_mask':mem['history_mask'],'memory_state':mem['state'],
        'unit':np.arange(len(bank['length']),dtype=np.int32)})
    tmp=out/'online_bank.tmp.npz';np.savez(tmp,**payload);tmp.replace(path)
    atomic_json(out/'online_bank.json',{'units':len(keys),'width':keys.shape[1],'selected_config':selected,
        'encoder_seed':4101,'checkpoint':str(ck),'source':'LIBERO-40 train memory only; heldout tasks excluded'})
    return {'units':len(keys),'width':keys.shape[1],'checkpoint':str(ck)}


def parity_audit(out,*,unseen_root,device='cuda',count=10):
    """Compare the offline selected draft with the exact online policy path."""
    import torch
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.40')))
    out=Path(out);bank=np.load(out/'memory_bank.npz',allow_pickle=False)
    online_file=np.load(out/'online_bank.npz',allow_pickle=False)
    online={k:online_file[k] for k in online_file.files}
    online['keys']=torch.as_tensor(np.asarray(online['keys']),device=device,dtype=torch.float32)
    store=FrameStore(RUN,out/'seg5-cache');mem_inputs=build_memory_inputs(store,bank)
    qstore,qall=build_unseen_queries(unseen_root,split='dev')
    unseen=UnseenStore(unseen_root);keep=[]
    for j,key in enumerate(qall['key']):
        row=int(key[0])
        if int(unseen.arr['frame'][row])%10==0 and np.any(qall['history_mask'][j]):keep.append(j)
    if len(keep)<count:raise ValueError(f'need {count} Dev windows with history; found {len(keep)}')
    keep=np.asarray(keep[:count],np.int64);q={k:v[keep] for k,v in qall.items()}
    model,selected,ck=_load_model(out,4101,bank);model.to(device).eval()
    met=evaluate_selector(model,store,bank,mem_inputs,q,device,query_store=qstore)
    old=UnseenStore(UNSEEN);old_lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(old.arr['task'],old.arr['episode'],old.arr['frame']))}
    normal=np.load(out/'normal_actions_stride10.npy',mmap_mode='r')
    from seg_context_retrieval import SegmentContextPolicy
    local_by_cache_row={int(q['key'][j][0]):j for j in range(len(q['key']))}
    cweight=selected['continuity']
    if cweight=='learned':cweight=float(torch.nn.functional.softplus(model.encoder.learned_continuity_scale).detach().cpu())
    cfg=sc.SegmentConfig(token_dim=2048,width=256,heads=4,cross_layers=int(selected['layers']),
        continuity_weight=cweight,exclude_same_task=False,boundary_blend=False,
        value_head_mode=selected.get('value_head_mode','dct'))
    def prefix_fn(row):
        x,m=qstore.tokens([(int(row),)]);return x,m
    def pi_fn(row):
        j=local_by_cache_row[int(row)]
        return normal[old_lookup[(int(q['task'][j]),int(q['episode'][j]-1_000_000),
                                  int(unseen.arr['frame'][int(q['key'][j][0])]))]]
    policy=SegmentContextPolicy(model.encoder,online,prefix_fn=prefix_fn,pi05_fn=pi_fn,
        config=cfg,pose_head=model.pose_head)
    diffs=[];selected_units=[]
    with torch.no_grad():
        for i in range(len(keep)):
            policy.reset();valid=np.flatnonzero(q['history_stream_mask'][i])
            policy.previous_actions=[x.copy() for x in q['history_stream_actions'][i,valid]]
            policy.previous_states=[x.copy() for x in q['history_stream_states'][i,valid]]
            policy.previous_velocity=q['previous_velocity'][i].copy()
            policy.previous_unit=None
            # prefix_fn expects the actual unseen-cache row ID, not the
            # compact index inside this ten-row audit subset.
            online_action,meta=policy.act(int(q['key'][i][0]),q['state'][i],
                task_id=int(q['task'][i]),episode_id=int(q['episode'][i]))
            ix=int(met['top'][i]);qi=torch.as_tensor(met['query_keys'][i:i+1],device=device,dtype=torch.float32)
            ki=torch.as_tensor(met['memory_keys'][ix:ix+1],device=device,dtype=torch.float32)
            coeff=torch.as_tensor(bank['coefficients'][ix:ix+1],device=device,dtype=torch.float32)
            theta=model.pose_head(qi,ki,coeff).cpu().numpy()[0]
            oldi=old_lookup[(int(q['task'][i]),int(q['episode'][i]-1_000_000),int(unseen.arr['frame'][int(q['key'][i][0])]))]
            offline_action=sc.offline_draft(bank['coefficients'][ix],bank['magnitude'][ix],bank['length'][ix],theta,
                grip_actions=normal[oldi,:,6])[0]
            delta=float(np.max(np.abs(np.asarray(online_action)[:,:6]-offline_action[:,:6])))
            diffs.append(delta)
            selected_units.append({'offline':ix,'online':meta.get('unit_id'),'fallback':meta.get('fallback'),
                'draft_max_abs_6axis_delta':delta})
    result={'rows':len(diffs),'checkpoint':str(ck),'selected_config':selected,'max_abs_6axis_difference':max(diffs),
        'threshold':1e-5,'passed':bool(max(diffs)<=1e-5),'cases':selected_units,
        'note':'identical cached prefix tensors, action gripper source, selected bank, pose head, and canonical offline decoder'}
    atomic_json(out/'server_offline_parity.json',result)
    if not result['passed']:raise AssertionError(f'online/offline draft mismatch {result["max_abs_6axis_difference"]}')
    return result


def prepare_run(args):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError(f'nonempty experiment output: {out}; pass --resume')
    out.mkdir(parents=True,exist_ok=True)
    bank,meta=prepare_bank(out,args.run_root)
    audit=audit_inputs(out,args.run_root,args.unseen_root);atomic_json(out/'input_audit.json',audit)
    log(out,'bank',**meta)
    print(json.dumps({'bank':meta,'input_audit':audit},indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'unseen-libero/seg-context-01')
    p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--unseen-root',type=Path,default=UNSEEN)
    p.add_argument('--resume',action='store_true')
    sub=p.add_subparsers(dest='stage',required=True)
    sub.add_parser('bank',help='build and audit variable-length memory bank')
    tr=sub.add_parser('train',help='Dev-select 8 settings and train 3 frozen seeds');tr.add_argument('--device',default='cuda');tr.add_argument('--max-epochs',type=int,default=6);tr.add_argument('--steps-per-epoch',type=int,default=64)
    nr=sub.add_parser('normal',help='precompute frozen pi05 actions in an isolated JAX process');nr.add_argument('--checkpoint',type=Path,default=CHECKPOINT);nr.add_argument('--batch-size',type=int,default=4)
    ev=sub.add_parser('offline',help='score frozen Eval rows');ev.add_argument('--device',default='cuda');ev.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    ob=sub.add_parser('online-bank',help='encode final memory bank for serving');ob.add_argument('--device',default='cuda')
    pa=sub.add_parser('parity',help='check ten Dev offline/online drafts');pa.add_argument('--device',default='cuda');pa.add_argument('--count',type=int,default=10)
    sub.add_parser('status',help='print current stage inventory')
    a=p.parse_args(argv)
    if a.stage=='bank':prepare_run(a)
    elif a.stage=='train':
        train_pipeline(a.output,run_root=a.run_root,unseen_root=a.unseen_root,device=a.device,max_epochs=a.max_epochs,steps_per_epoch=a.steps_per_epoch)
    elif a.stage=='normal':
        x=_normal_predictions(UNSEEN,a.output/'normal_actions_stride10.npy',checkpoint=a.checkpoint,batch_size=a.batch_size)
        atomic_json(a.output/'normal_actions_stride10.json',{'rows':len(x),'checkpoint':str(a.checkpoint),'noise_seed_rule':'PRNGKey(4101+batch_start), 10 flow steps','source_rows':str(UNSEEN)})
        log(a.output,'normal',rows=len(x))
    elif a.stage=='offline':offline_evaluate(a.output,unseen_root=a.unseen_root,device=a.device,checkpoint=a.checkpoint)
    elif a.stage=='online-bank':
        result=prepare_online_bank(a.output,run_root=a.run_root,device=a.device);log(a.output,'online_bank',**result);print(json.dumps(result,indent=2))
    elif a.stage=='parity':
        result=parity_audit(a.output,unseen_root=a.unseen_root,device=a.device,count=a.count);log(a.output,'parity',**{k:v for k,v in result.items() if k not in ('cases','note')});print(json.dumps(result,indent=2))
    else:
        out=a.output;print(json.dumps({'output':str(out),'files':sorted(x.name for x in out.iterdir()) if out.exists() else [],
             'extract_progress':str(out/'seg5-cache/progress.jsonl'),'input_audit':str(out/'input_audit.json')},indent=2))

if __name__=='__main__':main()
