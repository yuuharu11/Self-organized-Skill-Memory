"""CPU diagnostic of top-K action error for the saved M2/Context key candidates."""
import argparse
import os,sys,json,time
from pathlib import Path
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('MKL_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=None,help='new output directory; defaults under SKILLNET_STATE_DIR')
    args=parser.parse_args()
    STATE=Path(os.environ.get('SKILLNET_STATE_DIR', Path.home()/'.skillnet-state'));ROOT=STATE/'outputs/robocasa-memory';WT=Path(os.environ.get('SKILLNET_REPO_ROOT', Path(__file__).resolve().parents[5]))
    SRC=ROOT/'memory-01/policy-order-v2';CTX=ROOT/'ctx-01';ACT=ROOT/'expert-02/act-oracle';OUT=Path(args.out) if args.out else Path(os.environ.get('ROBOCASA_KEYDIAG_OUT', ROOT/'expert-02/key-diag'))
    EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa';SERV=WT/'skill_moe/skillnet/examples/robocasa'
    sys.path[:0]=[str(EXDIR),str(SERV)]
    import torch
    import experiment as ex, expert
    from serve_memonly_policy import _network_classes,unit_metadata,materialize_unit
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'keydiag_candidates.npz').exists() or (OUT/'cpu_summary.json').exists():
        raise FileExistsError(f'refusing to overwrite keydiag outputs in {OUT}')
    q=dict(np.load(SRC/'queries/U_heldout.npz',allow_pickle=False));pool=dict(np.load(SRC/'pools/U_memory.npz',allow_pickle=False));errs=np.load(SRC/'cache/errors_U_heldout.npy',mmap_mode='r')
    old=np.load(ACT/'predictions.npz',allow_pickle=False);row_ids=np.asarray(old['row_ids']);target=np.asarray(old['target'],np.float32)
    assert len(row_ids)==210
    ctx=np.load(CTX/'cache/rows.npz',allow_pickle=False);feat=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    ck=torch.load(ROOT/'key-lite-01/context_history.pt',map_location='cpu',weights_only=False);gck=torch.load(ROOT/'ctx-01/context_model.pt',map_location='cpu',weights_only=False);mck=torch.load(ROOT/'key-lite-01/magnitude_v2/magnitude_head.pt',map_location='cpu',weights_only=False)
    Rank,Context,Mag=_network_classes();rank=Rank(ck['q_dim'],ck['k_dim']);rank.load_state_dict(ck['state']);rank.eval();gripnet=Context(int(gck['input_dim']));gripnet.load_state_dict(gck['state']);gripnet.eval();mag=Mag(int(mck['input_dim']));mag.load_state_dict(mck['state']);mag.eval()
    fm,fs,sm,ss=[np.asarray(x,np.float32) for x in ck['norm']];candcontext=np.asarray(ctx['candidate_context'],np.int64);candstate=np.asarray(ctx['candidate_state'],np.float32)
    candin=np.concatenate(((feat[candcontext]-fm)/fs,(candstate-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad(): kemb=torch.nn.functional.normalize(rank.key(torch.as_tensor(candin)),dim=-1).numpy()
    meta,_=unit_metadata(pool);offset=np.asarray(pool['action_offsets'],np.int64);lens=np.diff(offset);glob=np.asarray(pool['global_episode'],np.int64);allpol=ex.action_storage_to_policy(np.asarray(pool['actions_flat'],np.float32)).astype(np.float32)
    hm=np.asarray(ck['hist_norm']['mean'],np.float32);hs=np.asarray(ck['hist_norm']['std'],np.float32)
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ctx['heldout_task'],ctx['heldout_local_episode'],ctx['heldout_frame']))}
    stats=__import__('openpi.shared.normalize',fromlist=['load']).load(expert.CHECKPOINT/'assets')['actions']
    def model_action(a):return expert._draft_to_model(ex.action_policy_to_storage(np.asarray(a,np.float32)),stats)
    def row_context(qi):
     t,e,f=map(int,(q['task'][qi],q['episode'][qi],q['frame'][qi]));ri=lookup[(t,e,f)]
     pref=feat[int(ctx['heldout_context'][ri])];state=np.asarray(ctx['heldout_state'][ri],np.float32)
     qctx=np.r_[(pref-fm)/fs,(state-sm)/ss].astype(np.float32)
     hist=ex.action_storage_to_policy(np.asarray(q['hist'][qi],np.float32));n=min(max(f,0),10);mask=np.zeros(10,bool)
     if n:mask[-n:]=True
     return pref,state,qctx,hist,mask

    def candidates_scores(qctx,hist,mask,excluded):
     rq=np.r_[qctx,(((hist-hm)/hs)*mask[:,None]).reshape(-1),mask.astype(np.float32)].astype(np.float32)
     with torch.no_grad(): qe=torch.nn.functional.normalize(rank.query(torch.as_tensor(rq[None])),dim=-1).numpy()[0]
     scores=kemb@qe; scores[glob==excluded]=-np.inf
     return np.argsort(-scores),scores

    def predict_grip(pref,state):
     x=np.r_[(pref-np.asarray(gck['norm']['feature_mean'],np.float32))/np.asarray(gck['norm']['feature_std'],np.float32),(state-np.asarray(gck['norm']['state_mean'],np.float32))/np.asarray(gck['norm']['state_std'],np.float32)].astype(np.float32)
     with torch.no_grad(): v=gripnet.grip(torch.as_tensor(x[None]))[0].numpy()
     return np.where(v[:5]>=0,1.,-1.).astype(np.float32)
    def scale_top(qctx,ids):
     x=np.concatenate([np.broadcast_to(qctx,(len(ids),len(qctx))),meta[ids]],axis=1).astype(np.float32);x=(x-mck['feature_mean'])/mck['feature_std']
     with torch.no_grad():return torch.exp(mag(torch.as_tensor(x))).clamp(.25,4.).numpy()
    def act_err(qtarget,a):
     return float(np.mean((a[:5,:7]-qtarget[:5,:7])**2))
    rec=[];random_drafts=[];top_tables=[];rho=[];type_top1=[];t0=time.time()
    rng_global=np.random.default_rng(20261112)
    for j,qi0 in enumerate(row_ids):
     qi=int(qi0);pref,state,qctx,hist,mask=row_context(qi);excluded=int(q['global_episode'][qi]);grip=predict_grip(pref,state);order,scores=candidates_scores(qctx,hist,mask,excluded)
     top=order[:200]; scales=scale_top(qctx,top)
     acts=[];aerrs=[]
     for u,s in zip(top,scales,strict=True):
      a=materialize_unit(allpol[offset[u]:offset[u+1]].copy(),float(np.asarray(s).reshape(-1)[0]),grip)
      a5=ex.resample_rows(a,5)
      acts.append(a5);aerrs.append(float(np.mean((model_action(a5)[:,:7]-target[j,:5,:7])**2)))
     aerrs=np.asarray(aerrs,np.float64); types=np.asarray(errs[qi,top],np.float64)
     from scipy.stats import spearmanr
     sr=spearmanr(types,aerrs).statistic
     rho.append(float(sr) if np.isfinite(sr) else 0.)
     limits={k:float(np.min(aerrs[:k])) for k in (1,5,20,100)}
     chosen0=int(order[0]); type_top1.append(float(errs[qi,chosen0]))
     # One random 50-step chain, selecting randomly among same-episode-excluded units at every boundary.
     rg=np.random.default_rng(4111+qi); draft=np.zeros((50,12),np.float32);cursor=0;ids=[]
     while cursor<50:
      allowed=np.flatnonzero(glob!=excluded);u=int(rg.choice(allowed));scale=float(scale_top(qctx,np.asarray([u]))[0]);a=materialize_unit(allpol[offset[u]:offset[u+1]].copy(),scale,grip)
      take=min(len(a),50-cursor);draft[cursor:cursor+take]=a[:take];ids.append(u);cursor+=take
      if take==0: raise RuntimeError('empty unit')
     md=model_action(draft); random_drafts.append(md)
     rec.append({'query_row':qi,'task':int(q['task'][qi]),'episode':int(q['episode'][qi]),'frame':int(q['frame'][qi]),'type_top1':float(errs[qi,chosen0]),'action_mse_top1':float(aerrs[0]),'best_action_mse_topk':limits,'top200_type_action_spearman':rho[-1],'top200_rows':len(top),'top_candidate_ids':top.astype(int).tolist(),'top_candidate_type_error':types.tolist(),'top_candidate_action_mse':aerrs.tolist(),'random_unit_ids':ids})
    print('rank_done',time.time()-t0,flush=True)
    random_drafts=np.stack(random_drafts)
    def mse_stats(pred):
     d=(pred[:,:,:7]-target[:,:,:7])**2
     return {'first5_active7':float(d[:,:5].mean()),'horizon50_active7':float(d[:,:50].mean()),'first5_position3':float(d[:,:5,:3].mean()),'first5_rotation3':float(d[:,:5,3:6].mean()),'first5_gripper':float(d[:,:5,6].mean()),'horizon50_position3':float(d[:,:50,:3].mean()),'horizon50_rotation3':float(d[:,:50,3:6].mean()),'horizon50_gripper':float(d[:,:50,6].mean())}
    rank_summary={str(k):float(np.mean([r['best_action_mse_topk'][k] for r in rec])) for k in (1,5,20,100)}
    existing={}
    for key in ['draft_a_retrieved','draft_b_action_oracle_scale_head','draft_d_greedy_action_oracle_truth','draft_e_truth_draft']:
     existing[key]=mse_stats(np.asarray(old[key]))
    res={'random_unit_with_head':'random same-episode-excluded unit at every 50-step chain boundary; M1 magnitude head and Context grip output; fixed rng 4111+query row','random_metrics':mse_stats(random_drafts),'current_rank_topk_best_first5_active7':rank_summary,'act_oracle_saved_drafts':existing,'type_action_spearman_top200':{'mean':float(np.mean(rho)),'median':float(np.median(rho)),'q25':float(np.quantile(rho,.25)),'q75':float(np.quantile(rho,.75)),'positive_fraction':float(np.mean(np.asarray(rho)>0))},'type_error_top1_mean':float(np.mean(type_top1)),'n_queries':len(rec),'row_ids':row_ids.tolist(),'query_rows':rec}
    np.savez_compressed(OUT/'keydiag_candidates.npz',row_ids=row_ids,random_draft=random_drafts, target=target,top1_action_mse=np.asarray([r['best_action_mse_topk'][1] for r in rec]),top5_best_action_mse=np.asarray([r['best_action_mse_topk'][5] for r in rec]),top20_best_action_mse=np.asarray([r['best_action_mse_topk'][20] for r in rec]),top100_best_action_mse=np.asarray([r['best_action_mse_topk'][100] for r in rec]),type_action_spearman=np.asarray(rho))
    (OUT/'cpu_summary.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in res.items() if k!='query_rows'},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
