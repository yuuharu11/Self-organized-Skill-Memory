"""RM-OOF heldout continuation and bridge evaluation (210 fixed windows)."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import experiment as ex, expert, gripper
import expert4_oof as oof
progress=oof.progress

ROOT=Path('/work/.skillnet-state/outputs/robocasa-memory')
SOURCE=ROOT/'memory-01/policy-order-v2'; CTX=ROOT/'ctx-01'; GAP=ROOT/'expert-02/gap'
OUT=ROOT/'expert-04/prep'; EXP3=ROOT/'expert-03/train/expert/expert_003000.msgpack'


class ContextNet:
    """Shape-compatible accessor for saved Context model (only .grip used)."""
    def __new__(cls, dim):
        import torch
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.query=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256))
                self.key=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,256))
                self.log_scale=nn.Parameter(torch.tensor(float(np.log(20.))))
                self.grip=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,6))
            def forward(self,q,k):
                return nn.functional.normalize(self.query(q),dim=-1) @ nn.functional.normalize(self.key(k),dim=-1).T * self.log_scale.exp()
        return Net()


def _metrics(pred,target):
    d=(np.asarray(pred,np.float64)[...,:7]-np.asarray(target,np.float64)[...,:7])**2
    return {'first5_active7':float(d[:,:5].mean()),'horizon50_active7':float(d.mean()),
            'first5_position':float(d[:,:5,:3].mean()),'horizon50_position':float(d[:,:,:3].mean()),
            'first5_rotation':float(d[:,:5,3:6].mean()),'horizon50_rotation':float(d[:,:,3:6].mean()),
            'first5_grip':float(d[:,:5,6].mean()),'horizon50_grip':float(d[:,:,6].mean())}


def _switch_summary(pred,target,initial):
    rows=[]
    for p,t,i in zip(pred,target,initial,strict=True):
        rows.append(gripper._metrics(np.asarray(p)[:,6],np.asarray(t)[:,6],float(i)))
    true=sum(x['switch_count_true'] for x in rows); missed=sum(x['missed'] for x in rows); false=sum(x['false'] for x in rows)
    times=[v for x in rows for v in x['timing_abs_errors']]
    return {'rows_with_true_switch':sum(x['switch_count_true']>0 for x in rows),'true_switches':true,
            'detected':true-missed,'missed':missed,'false':false,
            'mean_abs_timing_error_steps':float(np.mean(times)) if times else None}


def _generate(rows, pool, feat, qctx, cctx, ginput, rank_ck, rank_net, mag_ck, mag_net, grip_net,
              *, continuation=False):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    n=len(rows['memory_task']); out=np.zeros((n,50,12),np.float32)
    candidate_ep=np.asarray(pool['global_episode'],np.int64)
    meta,_=__import__('magnitude_head')._unit_metadata(pool)
    fmean=np.asarray(mag_ck['feature_mean'],np.float32); fstd=np.asarray(mag_ck['feature_std'],np.float32)
    hmean=np.asarray(rank_ck['hist_norm']['mean'],np.float32); hstd=np.asarray(rank_ck['hist_norm']['std'],np.float32)
    cemb=torch.nn.functional.normalize(rank_net.key(torch.as_tensor(cctx,dtype=torch.float32)),dim=-1)
    off=np.asarray(pool['action_offsets'],np.int64); flat=np.asarray(pool['actions_flat'],np.float32)
    starts={(int(e),int(f)):i for i,(e,f) in enumerate(zip(pool['global_episode'],pool['frame'],strict=True))}
    next_idx=np.full(len(candidate_ep),-1,np.int32)
    for u in range(len(candidate_ep)):
        next_idx[u]=starts.get((int(candidate_ep[u]),int(pool['frame'][u]+pool['length'][u])),-1)
    hist0=ex.action_storage_to_policy(np.asarray(rows['memory_hist'],np.float32))
    frame=np.asarray(rows['memory_frame'],np.int64); global_ep=np.asarray(rows['memory_episode'],np.int64)
    mask0=np.zeros((n,10),bool)
    for i,f in enumerate(frame):
        v=min(max(int(f),0),10)
        if v:mask0[i,-v:]=True
    with torch.no_grad():
        gpred=np.where(grip_net.grip(torch.as_tensor(ginput,dtype=torch.float32)).numpy()[:,:5]>=0,1.,-1.).astype(np.float32)
    for ri in range(n):
        hist=hist0[ri].copy(); mask=mask0[ri].copy(); cursor=0; previous=-1; first_unit=True
        while cursor<50:
            u=-1
            if continuation and previous>=0:
                nu=int(next_idx[previous])
                if nu>=0: u=nu
            if u<0:
                hn=((hist-hmean)/hstd)*mask[:,None]
                q=np.concatenate([qctx[ri],hn.reshape(-1),mask.astype(np.float32)]).astype(np.float32)
                with torch.no_grad():
                    qe=torch.nn.functional.normalize(rank_net.query(torch.as_tensor(q[None],dtype=torch.float32)),dim=-1)
                    s=(qe@cemb.T)[0].numpy()
                s[candidate_ep==int(global_ep[ri])]=-np.inf
                u=int(np.argmax(s))
            raw=flat[off[u]:off[u+1]].copy()
            mx=np.concatenate([qctx[ri],meta[u]]).astype(np.float32)
            mx=torch.as_tensor(((mx-fmean)/fstd)[None],dtype=torch.float32)
            with torch.no_grad(): scale=float(torch.exp(mag_net(mx)).clamp(.25,4.).item())
            raw[:,5:8]*=scale
            if first_unit:
                raw[:min(5,len(raw)),11]=gpred[ri,:min(5,len(raw))]
                first_unit=False
            emit=raw[:min(len(raw),50-cursor)]
            out[ri,cursor:cursor+len(emit)]=emit
            hist=np.concatenate([hist,ex.action_storage_to_policy(emit)],axis=0)[-10:];mask[:]=True
            cursor+=len(emit);previous=u
    return out


def _batch_obs(items):
    import jax
    return jax.tree.map(lambda *xs: np.concatenate([np.asarray(x) for x in xs],axis=0),*items)


def bridge_eval(draft_np, target, row_ids, infos, query, stats):
    import torch, jax, jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.20)
    model,_=expert._load_expert_model()
    fn=module_jit(model.sample_bridge,static_argnames=('num_steps',))
    # Cache each observation once; heldout truth remains scoring-only.
    items=[]
    for i in row_ids:
        t,e,f=map(int,(query['task'][i],query['episode'][i],query['frame'][i]))
        obs,truth,_,_=expert._build_obs_and_target(infos[(t,e)],f,SOURCE,expert.CHECKPOINT)
        items.append(obs)
    results={}
    for label in ('pretrained','expert03'):
        if label=='expert03': expert._overlay_expert(model,EXP3)
        pcur=[];pfollow=[]
        for st in range(0,len(row_ids),8):
            ids=np.arange(st,min(st+8,len(row_ids)))
            ob=_batch_obs([items[i] for i in ids])
            for name,drafts,store in (('current',draft_np[0],pcur),('follow',draft_np[1],pfollow)):
                dn=np.stack([expert._draft_to_model(drafts[i],stats) for i in ids]).astype(np.float32)
                eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(row_ids[i])),(1,50,32)))[0] for i in ids])
                pred=np.asarray(fn(jax.random.key(0),ob,jnp.asarray(dn),jnp.full((len(ids),),.2,jnp.float32),num_steps=10,noise=jnp.asarray(eps)),np.float32)
                store.extend(pred)
        results[f'{label}_current']=np.asarray(pcur,np.float32)
        results[f'{label}_follow']=np.asarray(pfollow,np.float32)
    return results


def run(*, drafts_only=False):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'offline_210.npz').exists(): raise FileExistsError(OUT/'offline_210.npz')
    rk,z,pool,_,feat,_=oof.load_inputs()
    old=np.load(GAP/'offline_predictions.npz',allow_pickle=False); row_ids=np.asarray(old['row_ids'],np.int64); target=np.asarray(old['target'],np.float32)
    qsrc=np.load(SOURCE/'queries/U_heldout.npz',allow_pickle=False)
    qlookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(z['heldout_task'],z['heldout_local_episode'],z['heldout_frame'],strict=True))}
    qctx_ids=np.asarray([qlookup[(int(qsrc['task'][i]),int(qsrc['episode'][i]),int(qsrc['frame'][i]))] for i in row_ids],np.int64)
    cids=np.arange(len(pool['task']),dtype=np.int64)
    zheld=oof.subset_z(z,qctx_ids,cids)
    rank_ck=torch.load(ROOT/'key-lite-01/context_history.pt',map_location='cpu',weights_only=False)
    rank_net=rk.RankNet(rank_ck['q_dim'],rank_ck['k_dim']);rank_net.load_state_dict(rank_ck['state']);rank_net.eval()
    qctx,cctx=rk.assemble_context(zheld,feat,rank_ck['norm'],'memory')
    zheld['candidate_context']=z['candidate_context']
    zheld['candidate_state']=z['candidate_state']
    mag_ck=torch.load(ROOT/'key-lite-01/magnitude_v2/magnitude_head.pt',map_location='cpu',weights_only=False)
    mag_net=__import__('magnitude_head')._MLP.create(int(mag_ck['input_dim']));mag_net.load_state_dict(mag_ck['state']);mag_net.eval()
    ctx_ck=torch.load(CTX/'context_model.pt',map_location='cpu',weights_only=False)
    ctx_net=ContextNet(int(ctx_ck['input_dim']));ctx_net.load_state_dict(ctx_ck['state']);ctx_net.eval()
    state=np.asarray(z['heldout_storage_state'])[qctx_ids]
    pref=np.asarray(feat)[np.asarray(z['heldout_context'])[qctx_ids]]
    cn=ctx_ck['norm'];ginput=np.concatenate([(pref-np.asarray(cn['feature_mean']))/np.asarray(cn['feature_std']),
        (state-np.asarray(cn['state_mean']))/np.asarray(cn['state_std'])],axis=1).astype(np.float32)
    # Candidate table must exactly match the immutable U-memory row order.
    srcpool=np.load(SOURCE/'pools/U_memory.npz',allow_pickle=False)
    assert len(srcpool['task'])==len(pool['task']) and np.array_equal(srcpool['global_episode'],pool['global_episode']) and np.array_equal(srcpool['frame'],pool['frame'])
    draft_file=OUT/'offline_drafts.npz'
    if draft_file.exists():
        dd=np.load(draft_file,allow_pickle=False)
        if not np.array_equal(dd['row_ids'],row_ids): raise AssertionError('saved draft row order mismatch')
        current=np.asarray(dd['draft_current'],np.float32); follow=np.asarray(dd['draft_follow'],np.float32)
    else:
        current=_generate(zheld,pool,feat,qctx,cctx,ginput,rank_ck,rank_net,mag_ck,mag_net,ctx_net)
        follow=_generate(zheld,pool,feat,qctx,cctx,ginput,rank_ck,rank_net,mag_ck,mag_net,ctx_net,continuation=True)
    stats=__import__('openpi.shared.normalize',fromlist=['load']).load(expert.CHECKPOINT/'assets')['actions']
    # Reuse exact target/row order saved by the accepted gap analysis.
    draft_norm=[]
    for a in (current,follow): draft_norm.append(np.stack([expert._draft_to_model(x,stats) for x in a]))
    draft_metrics={name:_metrics(draft_norm[i],target) for i,name in enumerate(('current_research_chain','follow_source_units'))}
    initial=np.asarray([gripper._initial_grip(np.asarray(qsrc['hist'][i],np.float32),np.asarray(qsrc['state'][i],np.float32)) for i in row_ids])
    grip_metrics={name:_switch_summary(draft_norm[i],target,initial) for i,name in enumerate(('current_research_chain','follow_source_units'))}
    progress('offline_drafts_complete',windows=len(row_ids),draft_metrics=draft_metrics,gripper=grip_metrics)
    if not draft_file.exists():
        with draft_file.open('wb') as f: np.savez_compressed(f,row_ids=row_ids,draft_current=current,draft_follow=follow,target=target)
    oof.write_json(OUT/'offline_drafts_summary.json',{'windows':len(row_ids),'draft_metrics':draft_metrics,'gripper_switch':grip_metrics,
        'rules':'current: search each boundary; follow: after first retrieved unit follow its source-episode successors until demo end, then resume search. FIXGR grip and position-only magnitude are applied to both.'})
    if drafts_only: return {'windows':len(row_ids),'draft_metrics':draft_metrics,'gripper_switch':grip_metrics}
    manifest=json.loads((SOURCE/'manifest.json').read_text()); infos={(int(d['task_id']),int(d['episode'])):d for d in manifest['demos'] if d['split']=='heldout'}
    bridge=bridge_eval((current,follow),target,row_ids,infos,qsrc,stats)
    bridge_metrics={k:_metrics(v,target) for k,v in bridge.items()}
    with (OUT/'offline_210.npz').open('wb') as f:
        np.savez_compressed(f,row_ids=row_ids,target=target,draft_current=current,draft_follow=follow,
                            pred_pretrained_current=bridge['pretrained_current'],pred_pretrained_follow=bridge['pretrained_follow'],
                            pred_expert03_current=bridge['expert03_current'],pred_expert03_follow=bridge['expert03_follow'])
    write={'windows':len(row_ids),'draft_metrics':draft_metrics,'gripper_switch':grip_metrics,'bridge_metrics':bridge_metrics,
           'bridge':'pretrained π0.5 bridge and expert-03 step3000; t0=.2, 10 Euler steps; noise fold_in(4101,row_id)',
           'heldout_truth_usage':'scoring only; retrieval, scale, grip and continuation use no heldout action values'}
    oof.write_json(OUT/'offline_summary.json',write)
    return write


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--drafts-only',action='store_true');args=ap.parse_args()
    r=run(drafts_only=args.drafts_only);print(json.dumps(r,ensure_ascii=False,indent=2))
