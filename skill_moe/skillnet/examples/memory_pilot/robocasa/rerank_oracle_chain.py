"""Per-boundary top-20 action-oracle chain upper bound for RM-RERANK."""
import json,sys,os
from pathlib import Path
import numpy as np, torch
STATE=Path('/work/.skillnet-state');ROOT=STATE/'outputs/robocasa-memory';OUT=ROOT/'rerank-01';SRC=ROOT/'memory-01/policy-order-v2';CTX=ROOT/'ctx-01';KEY=ROOT/'key-lite-01';WT=STATE/'worktrees/robocasa-memory';EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa';SERV=WT/'skill_moe/skillnet/examples/robocasa';sys.path[:0]=[str(EXDIR),str(SERV),str(KEY)]
import experiment as ex, rerank
from serve_memonly_policy import _network_classes,unit_metadata,materialize_unit

def run():
    if (OUT/'oracle_chain.npz').exists():raise FileExistsError('no overwrite')
    torch.set_num_threads(8);q=dict(np.load(SRC/'queries/U_heldout.npz',allow_pickle=False));old=np.load(ROOT/'expert-02/key-diag/keydiag_candidates.npz',allow_pickle=False);rows=np.asarray(old['row_ids'],np.int64);target=np.asarray(old['target'],np.float32)[:,:,:7]
    z,pool,feat,rck,gck,mck,rank,gripnet,mag=rerank._load();mag._rerank_mean=np.asarray(mck['feature_mean'],np.float32);mag._rerank_std=np.asarray(mck['feature_std'],np.float32)
    meta,_=unit_metadata(pool);stats=__import__('openpi.shared.normalize',fromlist=['load']).load(STATE/'checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/assets')['actions']
    flat=ex.action_storage_to_policy(np.asarray(pool['actions_flat'],np.float32));offs=np.asarray(pool['action_offsets'],np.int64);length=np.diff(offs);acts5=np.stack([ex.resample_rows(flat[offs[j]:offs[j+1]],5) for j in range(len(length))])
    rr=np.load(CTX/'cache/rows.npz',allow_pickle=False);lk={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rr['heldout_task'],rr['heldout_local_episode'],rr['heldout_frame']))};pref=[];st=[]
    for qi in rows:
        ix=lk[(int(q['task'][qi]),int(q['episode'][qi]),int(q['frame'][qi]))];pref.append(feat[int(rr['heldout_context'][ix])]);st.append(rr['heldout_state'][ix])
    pref=np.asarray(pref,np.float32);st=np.asarray(st,np.float32);fm,fs,sm,ss=[np.asarray(x,np.float32) for x in rck['norm']];qctx=np.concatenate(((pref-fm)/fs,(st-sm)/ss),axis=1).astype(np.float32)
    Rank,_,_=_network_classes();rnet=Rank(rck['q_dim'],rck['k_dim']);rnet.load_state_dict(rck['state']);rnet.eval();cctx=np.asarray(z['candidate_context'],np.int64);cx=np.concatenate(((feat[cctx]-fm)/fs,(np.asarray(z['candidate_state'])-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad():kemb=torch.nn.functional.normalize(rnet.key(torch.as_tensor(cx)),dim=-1).numpy()
    gf=np.asarray(gck['norm']['feature_mean'],np.float32);gs=np.asarray(gck['norm']['feature_std'],np.float32);sf=np.asarray(gck['norm']['state_mean'],np.float32);sstd=np.asarray(gck['norm']['state_std'],np.float32)
    with torch.no_grad():grips=np.where(gripnet.grip(torch.as_tensor(np.concatenate(((pref-gf)/gs,(st-sf)/sstd),axis=1))).numpy()[:,:5]>=0,1.,-1.).astype(np.float32)
    means=np.asarray(stats.mean,np.float32)[:7];stds=np.asarray(stats.std,np.float32)[:7];pred=[]
    for ii,qi in enumerate(rows):
        hist=ex.action_storage_to_policy(np.asarray(q['hist'][qi],np.float32));mask=np.zeros(10,bool);n=min(max(int(q['frame'][qi]),0),10);mask[10-n:]=True;cursor=0;out=np.zeros((50,7),np.float32);ge=int(q['global_episode'][qi])
        while cursor<50:
            hn=((hist-np.asarray(rck['hist_norm']['mean'],np.float32))/np.asarray(rck['hist_norm']['std'],np.float32))*mask[:,None];rq=np.concatenate((qctx[ii],hn.reshape(-1),mask.astype(np.float32))).astype(np.float32)
            with torch.no_grad():qe=torch.nn.functional.normalize(rnet.query(torch.as_tensor(rq[None])),dim=-1).numpy()[0]
            sc=kemb@qe;sc[np.asarray(pool['global_episode'])==ge]=-np.inf;top=np.argsort(-sc,kind='stable')[:20]
            cf,scales,acs=rerank._candidate_batch(qctx[ii],top,sc[top],pool,meta,mag,np.broadcast_to(grips[ii],(20,5)),acts5,stats)
            truth=target[ii,cursor:min(cursor+5,50)]; compare=acs[:,:len(truth)];err=np.mean((compare-truth[None])**2,axis=(1,2));j=int(np.argmin(err));u=int(top[j]);raw=materialize_unit(flat[offs[u]:offs[u+1]].copy(),float(scales[j]),grips[ii]);take=min(50-cursor,len(raw));norm=(raw[:take,:7]-means)/(stds+1e-6);out[cursor:cursor+take]=norm;cursor+=take;hist=np.concatenate((hist,raw[:take]),axis=0)[-10:];mask[:]=True
        pred.append(out)
        if (ii+1)%35==0:print(json.dumps({'rows':ii+1,'total':len(rows)}),flush=True)
    pred=np.stack(pred);err=np.mean((pred-target)**2,axis=(1,2));first=np.mean((pred[:,:5]-target[:,:5])**2,axis=(1,2))
    np.savez_compressed(OUT/'oracle_chain.npz',row_ids=rows,pred=pred,first5_mse=first,horizon50_mse=err)
    print(json.dumps({'first5_mean':float(first.mean()),'horizon50_mean':float(err.mean())},indent=2))
if __name__=='__main__':run()
