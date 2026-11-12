"""Offline action-aware reranking over the existing Context+history top-K.

Training candidates are generated from the fixed Context+history ranker and
magnitude/gripper heads. The reranker is validated on episode-disjoint memory
queries. Task identifiers are never features. This is an offline diagnostic;
its ranker is not wired into the live server.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

STATE=Path('/work/.skillnet-state');ROOT=STATE/'outputs/robocasa-memory';WT=STATE/'worktrees/robocasa-memory'
SRC=ROOT/'memory-01/policy-order-v2';CTX=ROOT/'ctx-01';KEY=ROOT/'key-lite-01';OUT=ROOT/'rerank-01'
EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa';SERV=WT/'skill_moe/skillnet/examples/robocasa'
sys.path[:0]=[str(EXDIR),str(SERV),str(KEY)]
import experiment as ex, expert
from serve_memonly_policy import _network_classes, unit_metadata, materialize_unit

THREADS=int(os.environ.get('OMP_NUM_THREADS','8'));torch.set_num_threads(THREADS)

class ActionReranker(nn.Module):
    def __init__(self,qdim,cdim=38):
        super().__init__()
        self.q=nn.Sequential(nn.Linear(qdim,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,64))
        self.c=nn.Sequential(nn.Linear(cdim,64),nn.GELU(),nn.Linear(64,64))
        self.bias=nn.Linear(1,1)
    def forward(self,q,c):
        qe=self.q(q)
        ce=self.c(c)
        logits=(qe[:,None,:]*ce).sum(-1)/8.0+self.bias(c[...,-1:]).squeeze(-1)
        return logits

def _load():
    z=np.load(CTX/'cache/rows.npz',allow_pickle=False);pool=dict(np.load(CTX/'cache/pool.npz',allow_pickle=False))
    feat=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    rck=torch.load(KEY/'context_history.pt',map_location='cpu',weights_only=False)
    gck=torch.load(CTX/'context_model.pt',map_location='cpu',weights_only=False)
    mck=torch.load(KEY/'magnitude_v2/magnitude_head.pt',map_location='cpu',weights_only=False)
    Rank,Context,Mag=_network_classes()
    rank=Rank(rck['q_dim'],rck['k_dim']);rank.load_state_dict(rck['state']);rank.eval()
    grip=Context(int(gck['input_dim']));grip.load_state_dict(gck['state']);grip.eval()
    mag=Mag(int(mck['input_dim']));mag.load_state_dict(mck['state']);mag.eval()
    return z,pool,feat,rck,gck,mck,rank,grip,mag

def _norm_context(z,feat,rck,split):
    mean,std,sm,ss=[np.asarray(x,np.float32) for x in rck['norm']]
    ci=np.asarray(z[f'{split}_context'],np.int64);state=np.asarray(z[f'{split}_state'],np.float32)
    qctx=np.concatenate(((feat[ci]-mean)/std,(state-sm)/ss),axis=1).astype(np.float32)
    return qctx

def _hist_features(raw_hist,frame,rck):
    hist=ex.action_storage_to_policy(np.asarray(raw_hist,np.float32));mask=np.zeros((len(hist),10),bool)
    for i,f in enumerate(np.asarray(frame)):
        n=min(max(int(f),0),10)
        if n:mask[i,10-n:]=True
    hn=((hist-np.asarray(rck['hist_norm']['mean'],np.float32))/np.asarray(rck['hist_norm']['std'],np.float32))*mask[:,:,None]
    return np.concatenate([hn.reshape(len(hist),-1),mask.astype(np.float32)],axis=1).astype(np.float32),hist,mask

def _rank_scores(qfeat,rank,rck,pool,batch=128):
    mean,std,sm,ss=[np.asarray(x,np.float32) for x in rck['norm']]
    # qfeat already contains normalized query context and normalized history.
    candidates=np.load(CTX/'cache/rows.npz',allow_pickle=False)
    cand_ctx=np.asarray(candidates['candidate_context'],np.int64);feat=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    state=np.asarray(candidates['candidate_state'],np.float32)
    keys=np.concatenate(((feat[cand_ctx]-mean)/std,(state-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad():
        kt=torch.as_tensor(keys); out=[]
        for s in range(0,len(qfeat),batch):out.append(rank(torch.as_tensor(qfeat[s:s+batch]),kt).numpy())
    return np.concatenate(out)

def _top_order(scores,episodes,pool,k):
    masked=np.asarray(scores,np.float32).copy();masked[np.asarray(episodes)[:,None]==np.asarray(pool['global_episode'])[None,:]]=-np.inf
    top=np.argpartition(-masked,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(masked,top,axis=1);ix=np.argsort(-vals,axis=1,kind='stable');top=np.take_along_axis(top,ix,axis=1)
    return top,np.take_along_axis(masked,top,axis=1)

def _candidate_batch(qctx, ids, score, pool, meta, mag, grip_sign, actions5, stats, *, batch=8192):
    """Return post-magnitude, post-grip action features and selected scales."""
    ids=np.asarray(ids,np.int64);n=len(ids);scales=np.empty(n,np.float32)
    for st in range(0,n,batch):
        sl=slice(st,min(st+batch,n)); qbatch=np.asarray(qctx,np.float32); qpart=(qbatch[sl] if qbatch.ndim==2 else np.broadcast_to(qbatch,(sl.stop-sl.start,len(qbatch)))); x=np.concatenate((qpart,meta[ids[sl]]),axis=1).astype(np.float32)
        x=(x-np.asarray(mag._rerank_mean,np.float32))/np.asarray(mag._rerank_std,np.float32)
        with torch.no_grad(): scales[sl]=torch.exp(mag(torch.as_tensor(x))).clamp(.25,4.).numpy().reshape(-1)
    a=actions5[ids].copy();a[:,:,:6]*=scales[:,None,None];a[:,:,6]=np.asarray(grip_sign,np.float32)[:,:5]
    # Fork/policy order and checkpoint normalization; only first five active7 enter the feature/label.
    st=np.asarray(stats.mean,np.float32)[:7];ss=np.asarray(stats.std,np.float32)[:7]
    model=((a[:,:,:7]-st)/(ss+1e-6)).astype(np.float32)
    length=np.asarray(pool['length'])[ids].astype(np.float32)
    # Candidate vector: actual first-five action sequence, length, grip summary, current key score.
    cf=np.concatenate((model.reshape(n,35),np.log1p(length)[:,None],a[:,:,6].mean(1,keepdims=True),np.asarray(score,np.float32).reshape(n,1)),axis=1).astype(np.float32)
    return cf,scales,model

def _materialized_first5(qctx,top,scores,grip,meta,mag,pool,actions5,stats,k):
    ids=top[:,:k].reshape(-1); sc=scores[:,:k].reshape(-1)
    cf,scale,act=_candidate_batch(qctx,ids,sc,pool,meta,mag,grip,actions5,stats)
    return cf.reshape(len(top),k,-1),scale.reshape(len(top),k),act.reshape(len(top),k,5,7)

def _episode_split(task,episode,seed=6021):
    eps=np.asarray(episode,np.int64);task=np.asarray(task,np.int64);rng=np.random.default_rng(seed);val=set()
    for t in np.unique(task):
        e=np.unique(eps[task==t]);rng.shuffle(e);val.update(map(int,e[:max(1,int(np.ceil(.2*len(e))))]))
    return np.asarray([int(e) in val for e in eps],bool),sorted(val)

def _corrupt_queries(q, idx, rng, *, context_dim=2064):
    out=q[idx].copy();h=out[:,context_dim:].copy();u=rng.random(len(idx))
    zero=u<.25;replace=(u>=.25)&(u<.50);flip=(u>=.50)&(u<.65)
    h[zero]=0
    if np.any(replace):h[replace]=q[rng.integers(0,len(q),int(replace.sum())),context_dim:]
    # Reverse 1-2 translation components in the normalized action history.
    for i in np.flatnonzero(flip):
        n=int(rng.integers(1,3));h[i,:120].reshape(10,12)[:,:n]*=-1
    out[:,context_dim:]=h
    return out

def _fit_one(q,c,y,train_rows,val_rows,*,temp,epochs,seed=6021):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ActionReranker(q.shape[-1],c.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    qmean=q[train_rows].mean(0);qstd=np.maximum(q[train_rows].std(0),1e-3)
    cm=c[train_rows].reshape(-1,c.shape[-1]).mean(0);cs=np.maximum(c[train_rows].reshape(-1,c.shape[-1]).std(0),1e-3)
    qn=((q-qmean)/qstd).astype(np.float32);cn=((c-cm)/cs).astype(np.float32)
    yt=torch.as_tensor(y,dtype=torch.float32);qt=torch.as_tensor(qn);ct=torch.as_tensor(cn)
    best=(float('inf'),None,None);curve=[];batch=128
    for ep in range(1,epochs+1):
        model.train();order=rng.permutation(train_rows);losses=[]
        for s in range(0,len(order),batch):
            ids=order[s:s+batch];qq=_corrupt_queries(qn,ids,rng)
            logits=model(torch.as_tensor(qq),ct[ids]);teacher=torch.softmax(-yt[ids]/float(temp),dim=-1)
            loss=-(teacher*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            preds=[]
            for s in range(0,len(val_rows),batch):
                ids=val_rows[s:s+batch];logits=model(qt[ids],ct[ids]);pick=logits.argmax(-1);preds.extend(yt[ids].gather(1,pick[:,None]).squeeze(1).tolist())
            val=float(np.mean(preds))
        curve.append({'epoch':ep,'train_listwise_loss':float(np.mean(losses)),'val_selected_mse':val})
        if val<best[0]:best=(val,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep)
    return {'state':best[1],'epoch':best[2],'val_mse':best[0],'curve':curve,'qmean':qmean,'qstd':qstd,'cmean':cm,'cstd':cs,'temp':float(temp)}

def _fit_final(q,c,y,*,temp,epochs,seed=6021):
    ids=np.arange(len(q));torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ActionReranker(q.shape[-1],c.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    qm=q.mean(0);qs=np.maximum(q.std(0),1e-3);cm=c.reshape(-1,c.shape[-1]).mean(0);cs=np.maximum(c.reshape(-1,c.shape[-1]).std(0),1e-3)
    qn=((q-qm)/qs).astype(np.float32);cn=((c-cm)/cs).astype(np.float32);qt=torch.as_tensor(qn);ct=torch.as_tensor(cn);yt=torch.as_tensor(y,dtype=torch.float32)
    for ep in range(epochs):
        order=rng.permutation(ids);model.train()
        for s in range(0,len(order),128):
            ix=order[s:s+128];qq=_corrupt_queries(qn,ix,rng);logits=model(torch.as_tensor(qq),ct[ix]);teacher=torch.softmax(-yt[ix]/temp,dim=1);loss=-(teacher*torch.log_softmax(logits,dim=1)).sum(-1).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval();return {'state':model.state_dict(),'qmean':qm,'qstd':qs,'cmean':cm,'cstd':cs,'temp':float(temp),'epochs':int(epochs)}

def _predict(rerank_ck,q,c):
    model=ActionReranker(q.shape[-1],c.shape[-1]);model.load_state_dict(rerank_ck['state']);model.eval()
    qn=(q-rerank_ck['qmean'])/rerank_ck['qstd'];cn=(c-rerank_ck['cmean'])/rerank_ck['cstd']
    with torch.no_grad():return model(torch.as_tensor(qn),torch.as_tensor(cn)).numpy()

def _bootstrap(values,task,episode,seed=6021):
    values=np.asarray(values,np.float64);ids=np.asarray(task,np.int64)*1000+np.asarray(episode,np.int64);eps=np.unique(ids);m={e:np.flatnonzero(ids==e) for e in eps};rng=np.random.default_rng(seed);b=np.empty(5000)
    for i in range(len(b)):
        ix=np.concatenate([m[e] for e in rng.choice(eps,len(eps),replace=True)]);b[i]=values[ix].mean()
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def run():
    if OUT.exists() and (OUT/'report.md').exists():raise FileExistsError(f'refusing to overwrite {OUT}/report.md')
    OUT.mkdir(parents=True,exist_ok=True)
    z,pool,feat,rck,gck,mck,rank,gripnet,mag=_load()
    # Preserve M1 statistics on the module so candidate batches use exact normalization.
    mag._rerank_mean=np.asarray(mck['feature_mean'],np.float32);mag._rerank_std=np.asarray(mck['feature_std'],np.float32)
    mem_q=dict(np.load(SRC/'queries/U_memory.npz',allow_pickle=False));mem_context=_norm_context(z,feat,rck,'memory');hist_data=__import__('run_key_lite').build_history_rows(z,'memory')
    hfeat,_,_=_hist_features(z['memory_hist'],z['memory_frame'],rck);qfeat=np.concatenate((mem_context,hfeat),axis=1).astype(np.float32)
    rkmod=__import__('run_key_lite');rkmod.DEVICE='cpu'
    scores=rkmod.predict_ranker(rck,z,feat,pool,split='memory',hist_data=hist_data,exclude_episode=True)
    top50,top50score=_top_order(scores,z['memory_episode'],pool,50)
    meta,_=unit_metadata(pool);stats=__import__('openpi.shared.normalize',fromlist=['load']).load(expert.CHECKPOINT/'assets')['actions']
    # Query grip head is context-only and used exactly as in runtime.
    feature_mean=np.asarray(gck['norm']['feature_mean'],np.float32);feature_std=np.asarray(gck['norm']['feature_std'],np.float32);state_mean=np.asarray(gck['norm']['state_mean'],np.float32);state_std=np.asarray(gck['norm']['state_std'],np.float32)
    pre=np.asarray(feat[np.asarray(z['memory_context'],np.int64)],np.float32);st=np.asarray(z['memory_state'],np.float32)
    gx=np.concatenate(((pre-feature_mean)/feature_std,(st-state_mean)/state_std),axis=1).astype(np.float32)
    with torch.no_grad():glog=gripnet.grip(torch.as_tensor(gx)).numpy()
    gr=np.where(glog[:,:5]>=0,1.,-1.).astype(np.float32)
    flat=ex.action_storage_to_policy(np.asarray(pool['actions_flat'],np.float32));offs=np.asarray(pool['action_offsets'],np.int64);lens=np.diff(offs)
    actions5=np.empty((len(pool['length']),5,12),np.float32)
    for u in range(len(actions5)):
        a=flat[offs[u]:offs[u+1]]
        actions5[u]=ex.resample_rows(a,5)
    qn=len(qfeat);candidate_feat=np.empty((qn,50,38),np.float32);candidate_action=np.empty((qn,50,5,7),np.float32);candidate_scale=np.empty((qn,50),np.float32);labels=np.empty((qn,50),np.float32)
    targetp=ex.action_storage_to_policy(np.asarray(z['memory_target'],np.float32));mean=np.asarray(stats.mean,np.float32)[:7];std=np.asarray(stats.std,np.float32)[:7]
    targetn=((targetp[:,:5,:7]-mean)/(std+1e-6)).astype(np.float32)
    for start in range(0,qn,64):
        sl=slice(start,min(start+64,qn));ids=top50[sl].reshape(-1);sc=top50score[sl].reshape(-1)
        cf,scale,act=_candidate_batch(np.repeat(mem_context[sl],50,axis=0),ids,sc,pool,meta,mag,gr[sl].repeat(50,axis=0),actions5,stats)
        b=sl.stop-sl.start;candidate_feat[sl]=cf.reshape(b,50,-1);candidate_scale[sl]=scale.reshape(b,50);candidate_action[sl]=act.reshape(b,50,5,7)
        labels[sl]=np.mean((candidate_action[sl]-targetn[sl,None])**2,axis=(2,3))
        if sl.stop%1024<64:print(json.dumps({'stage':'memory_features','rows':sl.stop,'total':qn}),flush=True)
    # Task-stratified episode split; all candidates for an episode stay on one side.
    valmask,val_eps=_episode_split(z['memory_task'],z['memory_episode']);trainids=np.flatnonzero(~valmask);valids=np.flatnonzero(valmask)
    np.savez_compressed(OUT/'memory_features_k50.npz',query=qfeat,candidate=candidate_feat,label=labels,top_ids=top50,top_scores=top50score,task=z['memory_task'],episode=z['memory_episode'],frame=z['memory_frame'],valmask=valmask)
    chosen={};cv={}
    for k in (20,50):
        best=None
        for temp in (.02,.05,.1):
            fit=_fit_one(qfeat,candidate_feat[:,:k],labels[:,:k],trainids,valids,temp=temp,epochs=12,seed=6021+k)
            rec={'temp':temp,'epoch':fit['epoch'],'val_mse':fit['val_mse'],'curve':fit['curve']}
            cv.setdefault(str(k),[]).append(rec)
            if best is None or fit['val_mse']<best['val_mse']:best=rec
        model=_fit_final(qfeat,candidate_feat[:,:k],labels[:,:k],temp=best['temp'],epochs=best['epoch'],seed=6021+k)
        ck={**model,'input_dim':int(qfeat.shape[1]),'candidate_dim':int(candidate_feat.shape[-1]),'top_k':k,'validation':best,'fit_episode_split':{'val_global_episodes':val_eps,'train_rows':len(trainids),'validation_rows':len(valids)}}
        torch.save(ck,OUT/f'ranker_k{k}.pt');chosen[str(k)]={'temp':best['temp'],'epochs':best['epoch'],'validation_mse':best['val_mse']}
    # Held-out 210 rows from the fixed key-diag run.
    held_result=np.load(ROOT/'expert-02/key-diag/keydiag_candidates.npz',allow_pickle=False)
    rowids=np.asarray(held_result['row_ids'],np.int64);qheld=dict(np.load(SRC/'queries/U_heldout.npz',allow_pickle=False))
    hr=np.load(CTX/'cache/rows.npz',allow_pickle=False);lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(hr['heldout_task'],hr['heldout_local_episode'],hr['heldout_frame']))}
    pre=[];states=[];rawhist=[];hframes=[]
    for qi in rowids:
        ix=lookup[(int(qheld['task'][qi]),int(qheld['episode'][qi]),int(qheld['frame'][qi]))]
        pre.append(feat[int(hr['heldout_context'][ix])]);states.append(hr['heldout_state'][ix]);rawhist.append(qheld['hist'][qi]);hframes.append(qheld['frame'][qi])
    pre=np.asarray(pre,np.float32);states=np.asarray(states,np.float32);qctx=np.concatenate(((pre-rck['norm'][0])/rck['norm'][1],(states-rck['norm'][2])/rck['norm'][3]),axis=1).astype(np.float32)
    hh,polhist,hmask=_hist_features(np.asarray(rawhist),np.asarray(hframes),rck);qhold=np.concatenate((qctx,hh),axis=1).astype(np.float32)
    ranknet,_=__import__('run_key_lite').RankNet, None
    # query scores as in policy-order-v2 runtime, excluding the source episode.
    Rank,Context,Mag=_network_classes();rnet=Rank(rck['q_dim'],rck['k_dim']);rnet.load_state_dict(rck['state']);rnet.eval()
    cctx=np.asarray(z['candidate_context'],np.int64);cx=np.concatenate(((feat[cctx]-rck['norm'][0])/rck['norm'][1],(np.asarray(z['candidate_state'])-rck['norm'][2])/rck['norm'][3]),axis=1).astype(np.float32)
    with torch.no_grad():kemb=torch.nn.functional.normalize(rnet.key(torch.as_tensor(cx)),dim=-1).numpy()
    scores_h=np.empty((len(rowids),len(pool['length'])),np.float32)
    for i in range(len(rowids)):
        hn=(polhist[i]-np.asarray(rck['hist_norm']['mean'],np.float32))/np.asarray(rck['hist_norm']['std'],np.float32);hn*=hmask[i,:,None]
        qr=np.concatenate((qctx[i],hn.reshape(-1),hmask[i].astype(np.float32))).astype(np.float32)
        with torch.no_grad():qe=torch.nn.functional.normalize(rnet.query(torch.as_tensor(qr[None])),dim=-1).numpy()[0]
        scores_h[i]=kemb@qe;scores_h[i,np.asarray(pool['global_episode'])==int(qheld['global_episode'][rowids[i]])]=-np.inf
    top_h,score_h=_top_order(scores_h,np.asarray(qheld['global_episode'])[rowids],pool,50)
    pre_hold=np.asarray(pre,np.float32);gxh=np.concatenate(((pre_hold-feature_mean)/feature_std,(states-state_mean)/state_std),axis=1).astype(np.float32)
    with torch.no_grad():grip_h=np.where(gripnet.grip(torch.as_tensor(gxh)).numpy()[:,:5]>=0,1.,-1.).astype(np.float32)
    holdcf=np.empty((len(rowids),50,38),np.float32);holdact=np.empty((len(rowids),50,5,7),np.float32)
    for i in range(len(rowids)):
        cf,sc,ac=_candidate_batch(qctx[i],top_h[i],score_h[i],pool,meta,mag,np.broadcast_to(grip_h[i],(len(top_h[i]),5)),actions5,stats)
        holdcf[i]=cf;holdact[i]=ac
    holdtarget=held_result['target'][:,:5,:7]
    holderrors=np.mean((holdact-holdtarget[:,None])**2,axis=(2,3))
    evalres={};ranker_ck={}
    for k in (20,50):
        ck=torch.load(OUT/f'ranker_k{k}.pt',map_location='cpu',weights_only=False);ranker_ck[k]=ck
        logits=_predict(ck,qhold,holdcf[:,:k]);pick=logits.argmax(1)
        sel=np.arange(len(rowids));first=holdact[sel,pick]
        evalres[str(k)]={'reranked_first5':first,'reranked_mse':holderrors[sel,pick],'scores':logits,'pick_local':pick}
    # Build 50-step P5 and reranked chains from the same query starts.
    def chain_one(i,k=None):
        hist=polhist[i].copy();mask=hmask[i].copy();cursor=0;outa=np.zeros((50,12),np.float32);used=[]
        while cursor<50:
            hn=((hist-np.asarray(rck['hist_norm']['mean'],np.float32))/np.asarray(rck['hist_norm']['std'],np.float32))*mask[:,None]
            qr=np.concatenate((qctx[i],hn.reshape(-1),mask.astype(np.float32))).astype(np.float32)
            with torch.no_grad():qe=torch.nn.functional.normalize(rnet.query(torch.as_tensor(qr[None])),dim=-1).numpy()[0]
            sc=kemb@qe;sc[np.asarray(pool['global_episode'])==int(qheld['global_episode'][rowids[i]])]=-np.inf
            o=np.argsort(-sc,kind='stable')[:50];sco=sc[o]
            cfs,ss,acs=_candidate_batch(qctx[i],o,sco,pool,meta,mag,np.broadcast_to(grip_h[i],(len(o),5)),actions5,stats)
            u=int(o[0])
            if k is not None:
                ck=ranker_ck[k];cf=cfs[:k];qn=(qhold[i]-ck['qmean'])/ck['qstd'];cn=(cf-ck['cmean'])/ck['cstd'];model=ActionReranker(qhold.shape[-1],cf.shape[-1]);model.load_state_dict(ck['state']);model.eval()
                with torch.no_grad():u=int(o[int(model(torch.as_tensor(qn[None]),torch.as_tensor(cn[None])).argmax(1)[0])])
            ix=int(np.flatnonzero(o==u)[0]);scale=float(ss[ix]);raw=materialize_unit(flat[offs[u]:offs[u+1]].copy(),scale,grip_h[i]);take=min(len(raw),50-cursor)
            outa[cursor:cursor+take]=raw[:take];used.append(u);cursor+=take
            hist=np.concatenate((hist,raw[:take]),axis=0)[-10:];mask[:]=True
        return outa,used
    # Chain outcomes; action norm matches model normalization used by key-diag.
    target50=np.asarray(held_result['target'],np.float32)[:,:,:7];mean7=np.asarray(stats.mean,np.float32)[:7];std7=np.asarray(stats.std,np.float32)[:7]
    chains={}
    for tag,k in [('key_top1',None),('rerank_k20',20),('rerank_k50',50)]:
        pred=[];unitids=[]
        for i in range(len(rowids)):
            a,uu=chain_one(i,k);pred.append(((a[:,:7]-mean7)/(std7+1e-6)).astype(np.float32));unitids.append(uu)
        pred=np.stack(pred);err=(pred-target50)**2
        chains[tag]={'pred':pred,'ids':unitids,'first5_mse':err[:,:5].mean(axis=(1,2)),'horizon50_mse':err.mean(axis=(1,2))}
    # bootstrap and task summaries
    task=np.asarray(qheld['task'])[rowids];episode=np.asarray(qheld['episode'])[rowids]
    baselines={
      'random':{'first5':np.mean((held_result['random_draft'][:,:5,:7]-target50[:,:5])**2,axis=(1,2)),'horizon50':np.mean((held_result['random_draft'][:,:,:7]-target50)**2,axis=(1,2))},
      'top20_action_oracle_first5':{'first5':np.asarray(held_result['top20_best_action_mse'])},
      'pi05_normal':{'first5':np.full(len(rowids),.4635819),'horizon50':np.full(len(rowids),1.0395)},
    }
    def summary(v):return {'mean':float(np.mean(v)),'episode_bootstrap95':_bootstrap(v,task,episode)}
    metrics={}
    for tag,dc in chains.items():metrics[tag]={'first5':summary(dc['first5_mse']),'horizon50':summary(dc['horizon50_mse'])}
    for name,dic in baselines.items():metrics[name]={key:summary(v) for key,v in dic.items()}
    # top-1 candidate MSE vs chosen reranker candidate
    for k in (20,50):
        er=evalres[str(k)]['reranked_mse'];metrics[f'rerank_k{k}_direct_first5_candidate']=summary(er)
        metrics[f'key_top1_direct_first5_candidate']=summary(holderrors[:,0]) if k==20 else metrics['key_top1_direct_first5_candidate']
    # Save detailed results; action oracle top20 is the value from key-diag, no evaluation labels used in fit.
    np.savez_compressed(OUT/'heldout_rerank.npz',row_ids=rowids,target=target50,top_ids=top_h,top_scores=score_h,top_candidate_mse=holderrors,
      **{f'chain_{k}':v['pred'] for k,v in chains.items()},
      **{f'chain_first5_{k}':v['first5_mse'] for k,v in chains.items()},**{f'chain_horizon50_{k}':v['horizon50_mse'] for k,v in chains.items()},
      **{f'rerank_logits_k{k}':evalres[str(k)]['scores'] for k in (20,50)})
    result={'status':'complete','rows':len(rowids),'memory_rows':qn,'episode_split':'task-stratified 80/20 episode split; reranker validation episodes never enter reranker fit; current base key/heads were frozen global models, so their query training exposure is not cross-fit','selected_hyperparameters':chosen,'validation_search':cv,'metrics':metrics,'task_metrics':{}}
    for tag,dc in chains.items():
      result['task_metrics'][tag]={}
      for t in np.unique(task):
        ix=np.flatnonzero(task==t);result['task_metrics'][tag][str(int(t))]={'n':len(ix),'first5':float(dc['first5_mse'][ix].mean()),'horizon50':float(dc['horizon50_mse'][ix].mean())}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'complete','metrics':metrics,'selected':chosen},indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    run()
