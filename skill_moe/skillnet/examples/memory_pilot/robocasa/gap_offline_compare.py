from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np

STATE=Path('/work/.skillnet-state')
ROOT=STATE/'outputs/robocasa-memory'
WT=STATE/'worktrees/robocasa-memory'
SRC=ROOT/'memory-01/policy-order-v2'
CTX=ROOT/'ctx-01'
OUT=ROOT/'expert-02/gap'
EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa'
SERV=WT/'skill_moe/skillnet/examples/robocasa'
sys.path.insert(0,str(EXDIR));sys.path.insert(0,str(SERV))
import expert, experiment as ex
from serve_memonly_policy import _network_classes, unit_metadata, materialize_unit
TASKS=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']


def main():
    import jax, jax.numpy as jnp, torch
    import flax.nnx as nnx
    from flax import serialization
    from openpi.shared.nnx_utils import module_jit
    from openpi.shared import normalize
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.20)
    t0=time.time(); print(json.dumps({'stage':'start','utc':time.strftime('%FT%TZ',time.gmtime())}),flush=True)
    q=np.load(SRC/'queries/U_heldout.npz',allow_pickle=False)
    pool=dict(np.load(SRC/'pools/U_memory.npz',allow_pickle=False))
    err=np.load(SRC/'cache/errors_U_heldout.npy',mmap_mode='r')
    assert err.shape==(len(q['task']),len(pool['length']))
    ctx_rows=np.load(CTX/'cache/rows.npz',allow_pickle=False)
    feats=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    context_ck=torch.load(ROOT/'ctx-01/context_model.pt',map_location='cpu',weights_only=False)
    key_ck=torch.load(ROOT/'key-lite-01/context_history.pt',map_location='cpu',weights_only=False)
    mag_ck=torch.load(ROOT/'key-lite-01/magnitude_v2/magnitude_head.pt',map_location='cpu',weights_only=False)
    RankNet,ContextNet,MagnitudeNet=_network_classes()
    rank=RankNet(int(key_ck['q_dim']),int(key_ck['k_dim'])); rank.load_state_dict(key_ck['state']); rank.eval()
    grip_net=ContextNet(int(context_ck['input_dim']));grip_net.load_state_dict(context_ck['state']);grip_net.eval()
    mag=MagnitudeNet(int(mag_ck['input_dim']));mag.load_state_dict(mag_ck['state']);mag.eval()
    # Match the serving ranker's query and candidate normalization exactly.
    fm,fs,sm,ss=[np.asarray(x,np.float32) for x in key_ck['norm']]
    cand_context=np.asarray(ctx_rows['candidate_context'],np.int64)
    cand_state=np.asarray(ctx_rows['candidate_state'],np.float32)
    cand_in=np.concatenate(((feats[cand_context]-fm)/fs,(cand_state-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad():
        cand_emb=torch.nn.functional.normalize(rank.key(torch.as_tensor(cand_in)),dim=-1)
    mem_meta,unit_mag=unit_metadata(pool)
    hm=np.asarray(key_ck['hist_norm']['mean'],np.float32);hs=np.asarray(key_ck['hist_norm']['std'],np.float32)
    source_manifest=json.loads((SRC/'manifest.json').read_text())
    infos={(int(d['task_id']),int(d['episode'])):d for d in source_manifest['demos'] if d['split']=='heldout'}
    # Use only exact frame-key overlaps with the precomputed Context cache; choose a deterministic, episode-balanced 35 rows/task.
    feat_lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ctx_rows['heldout_task'],ctx_rows['heldout_local_episode'],ctx_rows['heldout_frame']))}
    chosen=[]; rng=np.random.default_rng(20261113)
    for task in range(6):
        all_ids=[i for i,(tt,ee,ff) in enumerate(zip(q['task'],q['episode'],q['frame'])) if int(tt)==task and (task,int(ee),int(ff)) in feat_lookup]
        by={}
        for i in all_ids: by.setdefault(int(q['episode'][i]),[]).append(i)
        eps=sorted(by); rng.shuffle(eps); buckets=[list(by[e]) for e in eps]
        for b in buckets: rng.shuffle(b)
        picked=[]
        while len(picked)<min(35,len(all_ids)) and any(buckets):
            for b in buckets:
                if b and len(picked)<min(35,len(all_ids)): picked.append(b.pop())
        if len(picked)<25: raise RuntimeError(f'too few cached rows for task {task}: {len(picked)}')
        chosen.extend(picked)
    chosen=np.asarray(sorted(chosen,key=lambda i:(int(q['task'][i]),int(q['episode'][i]),int(q['frame'][i]))),np.int64)
    # Build exact Context + history runtime retrieval, with same-episode exclusion. The oracle swaps only the first selected unit and then follows normal retrieval.
    def q_features(qi, ctxi):
        pref=feats[int(ctx_rows['heldout_context'][ctxi])]
        state=np.asarray(ctx_rows['heldout_state'][ctxi],np.float32)
        qctx=np.concatenate(((pref-fm)/fs,(state-sm)/ss)).astype(np.float32)
        hist_storage=np.asarray(q['hist'][qi],np.float32)
        hist=ex.action_storage_to_policy(hist_storage)
        frame=int(q['frame'][qi]); valid=min(max(frame,0),10)
        mask=np.zeros(10,bool)
        if valid:mask[10-valid:]=True
        return pref,state,qctx,hist,mask
    def rank_select(qctx,hist,mask,global_ep):
        hnorm=((hist-hm)/hs)*mask[:,None]
        rq=np.concatenate((qctx,hnorm.reshape(-1),mask.astype(np.float32))).astype(np.float32)
        with torch.no_grad():
            qe=torch.nn.functional.normalize(rank.query(torch.as_tensor(rq[None])),dim=-1)
            scores=(qe@cand_emb.T)[0]*rank.log_scale.exp()
        scores_np=scores.numpy();scores_np[np.asarray(pool['global_episode'])==int(global_ep)]=-np.inf
        return int(np.argmax(scores_np)),scores_np
    def scale_unit(qctx,selected):
        x=np.concatenate((qctx,mem_meta[selected])).astype(np.float32)
        x=(x-np.asarray(mag_ck['feature_mean'],np.float32))/np.asarray(mag_ck['feature_std'],np.float32)
        with torch.no_grad(): s=float(torch.exp(mag(torch.as_tensor(x[None],dtype=torch.float32)))[0,0].item())
        return float(np.clip(s,.25,4.))
    def grip5(pref,state):
        v=np.concatenate(((pref-np.asarray(context_ck['norm']['feature_mean'],np.float32))/np.asarray(context_ck['norm']['feature_std'],np.float32),
                          (state-np.asarray(context_ck['norm']['state_mean'],np.float32))/np.asarray(context_ck['norm']['state_std'],np.float32))).astype(np.float32)
        with torch.no_grad(): z=grip_net.grip(torch.as_tensor(v[None],dtype=torch.float32))[0]
        return z[:5].ge(0).numpy().astype(np.float32)*2-1
    def emit_unit(qctx,pref,state,hist,mask,u,used):
        lo,hi=int(pool['action_offsets'][u]),int(pool['action_offsets'][u+1])
        stored=np.asarray(pool['actions_flat'][lo:hi],np.float32)
        pol=ex.action_storage_to_policy(stored)
        pol=materialize_unit(pol,scale_unit(qctx,u),grip5(pref,state))
        return pol[:min(len(pol),50-used)].copy()
    def make_draft(qi,ctxi,force=None):
        pref,state,qctx,hist,mask=q_features(qi,ctxi)
        global_ep=int(q['global_episode'][qi])
        chunks=[];used=0;first=-1;first_len=0
        while used<50:
            if used==0 and force is not None: u=int(force)
            else: u,_=rank_select(qctx,hist,mask,global_ep)
            seg=emit_unit(qctx,pref,state,hist,mask,u,used)
            if len(seg)==0: raise RuntimeError('empty selected memory unit')
            if used==0:first=u;first_len=int(pool['length'][u])
            chunks.append(seg);used+=len(seg)
            hist=np.concatenate((hist,seg),axis=0)[-10:]; mask[:]=True
        return np.concatenate(chunks,axis=0)[:50],first,first_len
    # Candidate shape score matrix row index is exactly query row index. Exclude all memory units from the query's own episode.
    per_task={}; obs_records=[]; target_norms=[]; drafts=[]; oracle_drafts=[]; metadata=[]
    from experiment import state_storage_to_policy
    for j,qi0 in enumerate(chosen):
        qi=int(qi0); task=int(q['task'][qi]); ep=int(q['episode'][qi]); frame=int(q['frame'][qi])
        ctxi=feat_lookup[(task,ep,frame)]
        info=infos[(task,ep)]
        obs,target,raw_target,state_raw=expert._build_obs_and_target(info,frame,SRC,expert.CHECKPOINT)
        leaves,td=jax.tree.flatten(obs)
        leafvals=[np.asarray(jax.device_get(x))[0] for x in leaves]
        if j and td!=tree_def: raise AssertionError('Observation tree changed')
        if not j: tree_def=td
        obs_records.append(leafvals); target_norms.append(np.asarray(target,np.float32))
        qctx_data=q_features(qi,ctxi)
        retrieved,ri,rl=make_draft(qi,ctxi)
        rowerr=np.asarray(err[qi],np.float32).copy()
        rowerr[np.asarray(pool['global_episode'])==int(q['global_episode'][qi])]=np.inf
        ou=int(np.argmin(rowerr))
        oracle,oi,ol=make_draft(qi,ctxi,force=ou)
        drafts.append(retrieved);oracle_drafts.append(oracle)
        metadata.append({'row':qi,'task':task,'episode':ep,'frame':frame,'retrieved_unit':ri,'oracle_unit':oi,
                         'retrieved_length':rl,'oracle_length':ol,'oracle_type_error':float(rowerr[ou]),
                         'retrieved_task':int(pool['task'][ri]),'oracle_task':int(pool['task'][oi])})
        if (j+1)%50==0: print(json.dumps({'stage':'prepared','rows':j+1,'total':len(chosen)}),flush=True)
    # Stack observations in batches; each leaf is unbatched at collection time.
    batch_obs=[]
    for start in range(0,len(chosen),4):
        chunk=obs_records[start:start+4]
        leaves=[jnp.asarray(np.stack([rec[k] for rec in chunk])) for k in range(len(chunk[0]))]
        batch_obs.append((np.arange(start,min(start+4,len(chosen))),jax.tree.unflatten(tree_def,leaves)))
    stats=normalize.load(expert.CHECKPOINT/'assets')['actions']
    draft_norm=np.stack([expert._draft_to_model(x,stats) for x in drafts]).astype(np.float32)
    oracle_norm=np.stack([expert._draft_to_model(x,stats) for x in oracle_drafts]).astype(np.float32)
    target_norm=np.stack(target_norms).astype(np.float32)
    model,_=expert._load_expert_model()
    cp=ROOT/'expert-02/train/attempt-04/expert/expert_001000.msgpack'
    @nnx.jit(static_argnames=('num_steps',))
    def bridge(m,rng,obs,dn,t0,noise,*,num_steps=10):
        return m.sample_bridge(rng,obs,dn,t0,num_steps=num_steps,noise=noise)
    @nnx.jit(static_argnames=('num_steps',))
    def normal(m,rng,obs,noise,*,num_steps=10):
        return m.sample_actions(rng,obs,num_steps=num_steps,noise=noise)
    predictions={x:np.zeros((len(chosen),50,32),np.float32) for x in ('retrieved','oracle','pi05')}
    # One model instance only: do the plain pi0.5 pass first, then overlay the trained Expert.
    for bid,obs in batch_obs:
        noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(chosen[i])),(1,50,32)))[0] for i in bid])
        predictions['pi05'][bid]=np.asarray(normal(model,jax.random.key(0),obs,jnp.asarray(noise),num_steps=10))
    expert._overlay_expert(model,cp)
    for bid,obs in batch_obs:
        noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(chosen[i])),(1,50,32)))[0] for i in bid])
        eps=jnp.asarray(noise); ts=jnp.full((len(bid),),.2,jnp.float32)
        for name,dn in [('retrieved',draft_norm[bid]),('oracle',oracle_norm[bid])]:
            predictions[name][bid]=np.asarray(bridge(model,jax.random.key(0),obs,jnp.asarray(dn),ts,eps,num_steps=10))
        if (int(bid[-1])+1)%40==0: print(json.dumps({'stage':'inferred_expert','rows':int(bid[-1])+1}),flush=True)
    # Metrics are normalized action-space EEF6 MSE. Unit-length window is min(50, actual remaining length in the source's fixed variable-length segmentation).
    def seg_length(m):
        task,ep,fr=int(m['task']),int(m['episode']),int(m['frame'])
        cand=np.flatnonzero((pool['task']==task)&(pool['episode']==ep)&(pool['frame']<=fr)&(pool['frame']+pool['length']>fr))
        if not len(cand): return 10
        u=int(cand[np.argmax(pool['frame'][cand])]); return int(np.clip(pool['frame'][u]+pool['length'][u]-fr,1,50))
    L=np.asarray([seg_length(m) for m in metadata],np.int32)
    result_rows=[]
    for i,m in enumerate(metadata):
        rec=dict(m); rec['eval_unit_length']=int(L[i])
        for name,pred in predictions.items():
            d=(pred[i,:,:6]-target_norm[i,:,:6])**2
            rec[name+'_first5_mse6']=float(d[:5].mean())
            rec[name+'_unit_mse6']=float(d[:int(L[i])].mean())
            if name=='retrieved': rec['retrieved_axis_first5_mse']=d[:5].mean(axis=0).tolist();rec['retrieved_axis_unit_mse']=d[:int(L[i])].mean(axis=0).tolist()
            if name=='oracle': rec['oracle_axis_first5_mse']=d[:5].mean(axis=0).tolist();rec['oracle_axis_unit_mse']=d[:int(L[i])].mean(axis=0).tolist()
            if name=='pi05': rec['pi05_axis_first5_mse']=d[:5].mean(axis=0).tolist();rec['pi05_axis_unit_mse']=d[:int(L[i])].mean(axis=0).tolist()
        result_rows.append(rec)
    summary={}
    for name in predictions:
        summary[name]={k:float(np.mean([r[name+'_'+k] for r in result_rows])) for k in ('first5_mse6','unit_mse6')}
        summary[name]['by_task']={TASKS[t]:{k:float(np.mean([r[name+'_'+k] for r in result_rows if r['task']==t])) for k in ('first5_mse6','unit_mse6')} for t in range(6)}
    sink_rows=[r for r in result_rows if r['task']==4]
    axis_summary={}
    for name in ('retrieved','oracle','pi05'):
        axis_summary[name]={'first5':np.mean([r[name+'_axis_first5_mse'] for r in sink_rows],axis=0).tolist(),
                            'unit':np.mean([r[name+'_axis_unit_mse'] for r in sink_rows],axis=0).tolist()}
    payload={'status':'complete','n_rows':len(chosen),'per_task':{TASKS[t]:sum(m['task']==t for m in metadata) for t in range(6)},
             'selection':'episode-balanced deterministic up to 35/task from actual heldout U queries with exact cached context frame; same-episode memory units excluded for search/oracle',
             'expert_checkpoint':str(cp),'bridge_t0':.2,'bridge_steps':10,'noise':'fold_in(key(4101), query_row_id), shared across conditions',
             'oracle':'minimum cached S-shared type error for heldout query, same-episode excluded; magnitude-v2 and subsequent retrieval chain are runtime, only first unit selection uses truth',
             'metric':'checkpoint-normalized fork action EEF six axes, first5 and remaining length of true segmentation interval',
             'summary':summary,'sinkfaucet_axis_mse6':axis_summary,'rows':result_rows,'elapsed_seconds':time.time()-t0}
    p=OUT/'offline_comparison.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(p)
    np.savez_compressed(OUT/'offline_predictions.npz',row_ids=chosen, target=target_norm, retrieved=draft_norm, oracle=oracle_norm,
                        pred_retrieved=predictions['retrieved'],pred_oracle=predictions['oracle'],pred_pi05=predictions['pi05'],length=L)
    print(json.dumps({'stage':'done','n_rows':len(chosen),'summary':summary,'sink_axis':axis_summary,'elapsed':time.time()-t0}),flush=True)

if __name__=='__main__':main()
