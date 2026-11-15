"""Bridge comparison for RM-RERANK; emits additional bridge_outputs.npz only."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
STATE=Path('/work/.skillnet-state');ROOT=STATE/'outputs/robocasa-memory';WT=STATE/'worktrees/robocasa-memory'
OUT=ROOT/'rerank-01';SRC=ROOT/'memory-01/policy-order-v2';CTX=ROOT/'ctx-01';KEY=ROOT/'key-lite-01'
EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa';SERV=WT/'skill_moe/skillnet/examples/robocasa'
sys.path[:0]=[str(EXDIR),str(SERV),str(KEY)]
import experiment as ex, expert, rerank
from serve_memonly_policy import _network_classes, unit_metadata, materialize_unit

def run():
    if (OUT/'bridge_outputs.npz').exists(): raise FileExistsError('refusing to overwrite bridge_outputs.npz')
    torch.set_num_threads(8)
    qheld=dict(np.load(SRC/'queries/U_heldout.npz',allow_pickle=False)); old=np.load(ROOT/'expert-02/key-diag/keydiag_candidates.npz',allow_pickle=False); rowids=np.asarray(old['row_ids'],np.int64)
    z,pool,feat,rck,gck,mck,rank,gripnet,mag=rerank._load()
    mag._rerank_mean=np.asarray(mck['feature_mean'],np.float32);mag._rerank_std=np.asarray(mck['feature_std'],np.float32)
    meta,_=unit_metadata(pool);stats=__import__('openpi.shared.normalize',fromlist=['load']).load(expert.CHECKPOINT/'assets')['actions']
    flat=ex.action_storage_to_policy(np.asarray(pool['actions_flat'],np.float32));offs=np.asarray(pool['action_offsets'],np.int64);length=np.diff(offs)
    acts5=np.stack([ex.resample_rows(flat[offs[i]:offs[i+1]],5) for i in range(len(length))]).astype(np.float32)
    ctxrows=np.load(CTX/'cache/rows.npz',allow_pickle=False); featlookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ctxrows['heldout_task'],ctxrows['heldout_local_episode'],ctxrows['heldout_frame']))}
    pref=[];state=[];hist=[];frames=[]
    for qi in rowids:
        ix=featlookup[(int(qheld['task'][qi]),int(qheld['episode'][qi]),int(qheld['frame'][qi]))]
        pref.append(feat[int(ctxrows['heldout_context'][ix])]);state.append(ctxrows['heldout_state'][ix]);hist.append(qheld['hist'][qi]);frames.append(qheld['frame'][qi])
    pref=np.asarray(pref,np.float32);state=np.asarray(state,np.float32);frames=np.asarray(frames);hist=ex.action_storage_to_policy(np.asarray(hist,np.float32))
    fm,fs,sm,ss=[np.asarray(x,np.float32) for x in rck['norm']];qctx=np.concatenate(((pref-fm)/fs,(state-sm)/ss),axis=1).astype(np.float32)
    # Load query and candidate projections of the frozen P5 key.
    Rank,Context,Mag= _network_classes();rnet=Rank(rck['q_dim'],rck['k_dim']);rnet.load_state_dict(rck['state']);rnet.eval()
    cctx=np.asarray(z['candidate_context'],np.int64);cx=np.concatenate(((feat[cctx]-fm)/fs,(np.asarray(z['candidate_state'])-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad():kemb=torch.nn.functional.normalize(rnet.key(torch.as_tensor(cx)),dim=-1).numpy()
    hmean=np.asarray(rck['hist_norm']['mean'],np.float32);hstd=np.asarray(rck['hist_norm']['std'],np.float32);gfm=np.asarray(gck['norm']['feature_mean'],np.float32);gfs=np.asarray(gck['norm']['feature_std'],np.float32);gsm=np.asarray(gck['norm']['state_mean'],np.float32);gss=np.asarray(gck['norm']['state_std'],np.float32)
    with torch.no_grad():grips=np.where(gripnet.grip(torch.as_tensor(np.concatenate(((pref-gfm)/gfs,(state-gsm)/gss),axis=1))).numpy()[:,:5]>=0,1.,-1.).astype(np.float32)
    checkpoints={k:torch.load(OUT/f'ranker_k{k}.pt',map_location='cpu',weights_only=False) for k in (20,50)}
    q0,h0,mask0=rerank._hist_features(np.asarray(qheld['hist'])[rowids],frames,rck);qall=np.concatenate((qctx,q0),axis=1).astype(np.float32)
    chains={}
    for k in (20,50):
        ck=checkpoints[k];model=rerank.ActionReranker(qall.shape[-1],38);model.load_state_dict(ck['state']);model.eval();rawchains=[];unitchains=[]
        for i in range(len(rowids)):
            h=hist[i].copy();m=mask0[i].copy();cursor=0;out=np.zeros((50,12),np.float32);ids_used=[]
            while cursor<50:
                hn=((h-hmean)/hstd)*m[:,None];rq=np.concatenate((qctx[i],hn.reshape(-1),m.astype(np.float32))).astype(np.float32)
                with torch.no_grad():qe=torch.nn.functional.normalize(rnet.query(torch.as_tensor(rq[None])),dim=-1).numpy()[0]
                scr=kemb@qe;ge=int(qheld['global_episode'][rowids[i]]);scr[np.asarray(pool['global_episode'])==ge]=-np.inf
                top=np.argsort(-scr,kind='stable')[:50];scores=scr[top]
                cf,scales,_=rerank._candidate_batch(qctx[i],top,scores,pool,meta,mag,np.broadcast_to(grips[i],(len(top),5)),acts5,stats)
                qn=(qall[i]-ck['qmean'])/ck['qstd'];cn=(cf[:k]-ck['cmean'])/ck['cstd']
                with torch.no_grad():j=int(model(torch.as_tensor(qn[None]),torch.as_tensor(cn[None])).argmax(1)[0])
                uid=int(top[j]);scale=float(scales[j]);a=materialize_unit(flat[offs[uid]:offs[uid+1]].copy(),scale,grips[i]);take=min(50-cursor,len(a));out[cursor:cursor+take]=a[:take];ids_used.append(uid);cursor+=take
                h=np.concatenate((h,a[:take]),axis=0)[-10:];m[:]=True
            rawchains.append(out);unitchains.append(ids_used)
            if (i+1)%25==0:print(json.dumps({'stage':'rebuild_raw_chain','k':k,'rows':i+1,'total':len(rowids)}),flush=True)
        chains[k]=(np.stack(rawchains),unitchains)
    # Construct exact transformed observations and targets from the same heldout rows.
    manifest=json.loads((SRC/'manifest.json').read_text());infos={(int(d['task_id']),int(d['episode'])):d for d in manifest['demos'] if d['split']=='heldout'}
    obs=[];target=[];episode=[];task=[]
    for j,qi in enumerate(rowids):
        t,e,f=int(qheld['task'][qi]),int(qheld['episode'][qi]),int(qheld['frame'][qi]);info=infos[(t,e)]
        ob,targ,_,_=expert._build_obs_and_target(info,f,SRC,expert.CHECKPOINT);obs.append(ob);target.append(targ);episode.append(e);task.append(t)
        if (j+1)%25==0:print(json.dumps({'stage':'observations','rows':j+1,'total':len(rowids)}),flush=True)
    target=np.asarray(target,np.float32)
    import jax, jax.numpy as jnp, flax.nnx as nnx
    from openpi.shared.nnx_utils import module_jit
    from openpi.shared import normalize
    from act_oracle import _load_model_cpu_init
    model,_=_load_model_cpu_init()
    @nnx.jit(static_argnames=('num_steps',))
    def bridge(m,rng,o,d,t,eps,*,num_steps=10):return m.sample_bridge(rng,o,d,t,num_steps=num_steps,noise=eps)
    st=normalize.load(expert.CHECKPOINT/'assets')['actions'];means=np.asarray(st.mean,np.float32);stds=np.asarray(st.std,np.float32)
    # Same row-id folded eps for pretrained and Expert03; model-space drafts use full fork-order raw unit actions.
    epsall=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(r)),(1,50,32)))[0] for r in rowids])
    outputs={}
    for k in (20,50):
        dp=np.stack([ex.action_policy_to_storage(a) for a in chains[k][0]])
        dn=np.stack([expert._draft_to_model(a,st) for a in dp])
        outputs[f'draft_k{k}']=dn;outputs[f'pred_pretrained_k{k}']=np.empty_like(dn);outputs[f'pred_expert03_k{k}']=np.empty_like(dn)
        for i in range(len(rowids)):
            o=obs[i];noise=jnp.asarray(epsall[i:i+1]);d=jnp.asarray(dn[i:i+1]);tv=jnp.asarray([.2],jnp.float32)
            outputs[f'pred_pretrained_k{k}'][i]=np.asarray(bridge(model,jax.random.key(4102),o,d,tv,noise,num_steps=10))[0]
        cp=ROOT/'expert-03/train/expert/expert_003000.msgpack';expert._overlay_expert(model,cp)
        for i in range(len(rowids)):
            o=obs[i];noise=jnp.asarray(epsall[i:i+1]);d=jnp.asarray(dn[i:i+1]);tv=jnp.asarray([.2],jnp.float32)
            outputs[f'pred_expert03_k{k}'][i]=np.asarray(bridge(model,jax.random.key(4102),o,d,tv,noise,num_steps=10))[0]
        # Restore only trainable leaves to base for the next K comparison.
        if k==20:
            model,_=_load_model_cpu_init()
    # Metrics against transformed target, normalized active7; per-task records.
    metrics={}
    for k in (20,50):
        for kind,key in [('pretrained',f'pred_pretrained_k{k}'),('expert03',f'pred_expert03_k{k}')]:
            d=(outputs[key][:,:,:7]-target[:,:,:7])**2
            metrics[f'{kind}_k{k}']={'first5':float(d[:,:5].mean()),'horizon50':float(d.mean())}
    for k in (20,50):
        # Draft MSE on the same active7 transformed scale.
        d=(outputs[f'draft_k{k}'][:,:,:7]-target[:,:,:7])**2
        metrics[f'draft_k{k}']={'first5':float(d[:,:5].mean()),'horizon50':float(d.mean())}
    OUT.joinpath('bridge_summary.json').write_text(json.dumps({'status':'complete','rows':len(rowids),'noise':'fold_in(key(4101), row_id)','t0':.2,'metrics':metrics},indent=2)+'\n')
    np.savez_compressed(OUT/'bridge_outputs.npz',row_ids=rowids,target=target,task=np.asarray(task),episode=np.asarray(episode),**outputs)
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':run()
