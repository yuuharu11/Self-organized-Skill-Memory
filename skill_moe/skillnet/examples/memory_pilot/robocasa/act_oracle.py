"""Action-space oracle upper bound for RM-ACTORACLE, offline only."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

STATE=Path('/work/.skillnet-state')
ROOT=STATE/'outputs/robocasa-memory'
WT=STATE/'worktrees/robocasa-memory'
SRC=ROOT/'memory-01/policy-order-v2'
CTX=ROOT/'ctx-01'
GAP=ROOT/'expert-02/gap'
OUT=ROOT/'expert-02/act-oracle'
EXDIR=WT/'skill_moe/skillnet/examples/memory_pilot/robocasa'
SERV=WT/'skill_moe/skillnet/examples/robocasa'
sys.path.insert(0,str(EXDIR));sys.path.insert(0,str(SERV))
import expert, experiment as ex
from serve_memonly_policy import _network_classes, unit_metadata, materialize_unit
TASKS=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']
COND=['a_retrieved','b_action_oracle_scale_head','c_action_oracle_scale_truth','d_greedy_action_oracle_truth','e_truth_draft']


def _load_model_cpu_init():
    """Build/restore the model on CPU, then move only the final weights to GPU.

    This avoids holding a random-initialized model and the checkpoint copy on
    the shared GPU at the same time.
    """
    import jax
    import flax.nnx as nnx
    from openpi.models.model import restore_params
    core_dir=Path(expert.__file__).resolve().parents[1]/'pi05_memory'
    if str(core_dir) not in sys.path: sys.path.insert(0,str(core_dir))
    import core
    cpu=jax.devices('cpu')[0]
    with jax.default_device(cpu):
        model,cfg=core.make_model(action_horizon=50,discrete_state_input=True,robocasa=True)
        params=restore_params(expert.CHECKPOINT/'params',restore_type=np.ndarray)
        state=nnx.state(model); expected=state.to_pure_dict()
        a,ta=jax.tree.flatten(expected); b,tb=jax.tree.flatten(params)
        if ta!=tb or any(x.shape!=y.shape for x,y in zip(a,b,strict=True)):
            raise ValueError('RoboCasa π0.5 checkpoint tree mismatch')
        state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray,params))
        nnx.update(model,state)
    del params,expected,a,b,state
    import gc; gc.collect()
    gpu=jax.devices('gpu')[0]
    state=nnx.state(model)
    gpu_pure=jax.tree.map(lambda x:jax.device_put(x,gpu),state.to_pure_dict())
    state.replace_by_pure_dict(gpu_pure);nnx.update(model,state)
    return model,cfg


def _metric(pred,target):
    d=(np.asarray(pred,np.float64)[...,:7]-np.asarray(target,np.float64)[...,:7])**2
    def fields(x):
        return {'active7':float(x.mean()),'position_xyz':float(x[...,:3].mean()),
                'rotation':float(x[...,3:6].mean()),'gripper':float(x[...,6:7].mean())}
    return {'first5':fields(d[:5]),'horizon50':fields(d[:50])}


def run():
    import torch, jax, jax.numpy as jnp, flax.nnx as nnx
    from openpi.shared import normalize
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.20)
    OUT.mkdir(parents=True,exist_ok=True)
    if any((OUT/name).exists() for name in ('summary.json','report.md','predictions.npz')):
        raise FileExistsError(f"refusing to overwrite completed output in {OUT}")
    q=dict(np.load(SRC/'queries/U_heldout.npz',allow_pickle=False))
    pool=dict(np.load(SRC/'pools/U_memory.npz',allow_pickle=False))
    errs=np.load(SRC/'cache/errors_U_heldout.npy',mmap_mode='r')
    old=np.load(GAP/'offline_predictions.npz',allow_pickle=False)
    chosen=np.asarray(old['row_ids'],np.int64)
    result_old=json.loads((GAP/'offline_comparison.json').read_text())
    if len(result_old['rows'])!=210 or not np.array_equal(chosen,np.asarray([r['row'] for r in result_old['rows']],np.int64)):
        raise AssertionError('offline comparison rows do not match the reused 210 row IDs')
    target=np.asarray(old['target'],np.float32)
    retrieved_existing=np.asarray(old['retrieved'],np.float32)
    ctx_rows=np.load(CTX/'cache/rows.npz',allow_pickle=False)
    feats=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    context_ck=torch.load(ROOT/'ctx-01/context_model.pt',map_location='cpu',weights_only=False)
    key_ck=torch.load(ROOT/'key-lite-01/context_history.pt',map_location='cpu',weights_only=False)
    mag_ck=torch.load(ROOT/'key-lite-01/magnitude_v2/magnitude_head.pt',map_location='cpu',weights_only=False)
    RankNet,ContextNet,MagnitudeNet=_network_classes()
    rank=RankNet(int(key_ck['q_dim']),int(key_ck['k_dim']));rank.load_state_dict(key_ck['state']);rank.eval()
    grip_net=ContextNet(int(context_ck['input_dim']));grip_net.load_state_dict(context_ck['state']);grip_net.eval()
    mag=MagnitudeNet(int(mag_ck['input_dim']));mag.load_state_dict(mag_ck['state']);mag.eval()
    fm,fs,sm,ss=[np.asarray(x,np.float32) for x in key_ck['norm']]
    cand_context=np.asarray(ctx_rows['candidate_context'],np.int64)
    cand_state=np.asarray(ctx_rows['candidate_state'],np.float32)
    cand_in=np.concatenate(((feats[cand_context]-fm)/fs,(cand_state-sm)/ss),axis=1).astype(np.float32)
    with torch.no_grad(): cand_emb=torch.nn.functional.normalize(rank.key(torch.as_tensor(cand_in)),dim=-1)
    mem_meta,_=unit_metadata(pool)
    hm=np.asarray(key_ck['hist_norm']['mean'],np.float32);hs=np.asarray(key_ck['hist_norm']['std'],np.float32)
    manifest=json.loads((SRC/'manifest.json').read_text())
    infos={(int(d['task_id']),int(d['episode'])):d for d in manifest['demos'] if d['split']=='heldout'}
    feat_lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ctx_rows['heldout_task'],ctx_rows['heldout_local_episode'],ctx_rows['heldout_frame']))}
    if any((int(q['task'][i]),int(q['episode'][i]),int(q['frame'][i])) not in feat_lookup for i in chosen):
        raise AssertionError('selected rows are not in the ctx-01 feature cache')
    all_policy=ex.action_storage_to_policy(np.asarray(pool['actions_flat'],np.float32)).astype(np.float32)
    offsets=np.asarray(pool['action_offsets'],np.int64); lengths=np.diff(offsets)
    global_eps=np.asarray(pool['global_episode'],np.int64)

    def query(i):
        t,e,f=int(q['task'][i]),int(q['episode'][i]),int(q['frame'][i]); ci=feat_lookup[(t,e,f)]
        pref=feats[int(ctx_rows['heldout_context'][ci])]
        state=np.asarray(ctx_rows['heldout_state'][ci],np.float32)
        qctx=np.concatenate(((pref-fm)/fs,(state-sm)/ss)).astype(np.float32)
        hist=ex.action_storage_to_policy(np.asarray(q['hist'][i],np.float32))
        nvalid=min(max(f,0),10);mask=np.zeros(10,bool)
        if nvalid:mask[-nvalid:]=True
        return pref,state,qctx,hist,mask

    def rank_select(qctx,hist,mask,ge):
        hnorm=((hist-hm)/hs)*mask[:,None]
        rq=np.concatenate((qctx,hnorm.reshape(-1),mask.astype(np.float32))).astype(np.float32)
        with torch.no_grad():
            qe=torch.nn.functional.normalize(rank.query(torch.as_tensor(rq[None])),dim=-1)
            scores=(qe@cand_emb.T)[0]
        s=scores.numpy();s[global_eps==int(ge)]=-np.inf
        return int(np.argmax(s))

    def scales_head(qctx):
        x=np.concatenate((np.broadcast_to(qctx,(len(pool['length']),len(qctx))),mem_meta),axis=1).astype(np.float32)
        x=(x-np.asarray(mag_ck['feature_mean'],np.float32))/np.asarray(mag_ck['feature_std'],np.float32)
        with torch.no_grad():
            return np.clip(torch.exp(mag(torch.as_tensor(x))).cpu().numpy().reshape(-1),.25,4.).astype(np.float32)

    def grip_pred(pref,state):
        x=np.concatenate(((pref-np.asarray(context_ck['norm']['feature_mean'],np.float32))/np.asarray(context_ck['norm']['feature_std'],np.float32),
                          (state-np.asarray(context_ck['norm']['state_mean'],np.float32))/np.asarray(context_ck['norm']['state_std'],np.float32))).astype(np.float32)
        with torch.no_grad(): v=grip_net.grip(torch.as_tensor(x[None]))[0]
        return v[:5].ge(0).numpy().astype(np.float32)*2-1

    def unit_actions(u,scale,grip):
        a=all_policy[offsets[u]:offsets[u+1]].copy()
        return materialize_unit(a,scale,grip)

    def head_scales_for_query(qctx): return scales_head(qctx)

    def first_action_oracle(qctx,grip,target_pol,excluded,scale_table):
        best=(float('inf'),-1)
        # All RoboCasa U units are at least five frames. Compare the same first-five-step
        # active-7 MSE so variable segment duration cannot change the denominator.
        valid=np.flatnonzero(global_eps!=int(excluded))
        for u in valid:
            a=unit_actions(int(u),float(scale_table[u]),grip)[:,:7]
            # Compare a fixed five-step view for all variable-duration units.
            # The stored unit itself is still emitted at its native duration.
            a5=ex.resample_rows(a,5)
            er=float(np.mean((a5-target_pol[:5,:7])**2))
            if er<best[0]: best=(er,int(u))
        if best[1]<0: raise RuntimeError('empty action oracle candidate set')
        return best[1],best[0]

    def oracle_scale(u,target_seg):
        src=all_policy[offsets[u]:offsets[u+1],:3]
        n=min(len(src),len(target_seg))
        den=float(np.sum(src[:n].astype(np.float64)**2))
        s=float(np.sum(src[:n].astype(np.float64)*target_seg[:n,:3].astype(np.float64))/max(den,1e-12))
        return float(np.clip(s,.25,4.))

    def emit_draft(i, style, target_pol, pref,state,qctx,hist,mask,head_table):
        ge=int(q['global_episode'][i]); chunks=[]; used=0; first=-1; ids=[]
        if style=='d':
            while used<50:
                remaining=50-used; allowed=np.flatnonzero(global_eps!=ge)
                best_err=np.full(len(pool['length']),np.inf,np.float64); best_scale=np.ones(len(pool['length']),np.float32)
                cand_lens=np.minimum(lengths,remaining)
                for L in np.unique(cand_lens[allowed]):
                    L=int(L); group=allowed[cand_lens[allowed]==L]
                    src=all_policy[offsets[group,None]+np.arange(L)[None,:],:7].copy()
                    tgt=target_pol[used:used+L,:7].astype(np.float32)
                    # Oracle position norm ratio is chosen per candidate and boundary,
                    # then applied to all six motion channels exactly as the stored draft path.
                    pos=src[:,:,:3].astype(np.float64); tp=tgt[None,:,:3].astype(np.float64)
                    den=np.sum(pos*pos,axis=(1,2)); num=np.sum(pos*tp,axis=(1,2))
                    sc=np.clip(num/np.maximum(den,1e-12),.25,4.).astype(np.float32)
                    src[:,:,:6]*=sc[:,None,None]
                    ng=min(5,L);src[:,:ng,6]=grip[None,:ng]
                    errs=np.mean((src-tgt[None])**2,axis=(1,2))
                    best_err[group]=errs;best_scale[group]=sc
                u=int(np.argmin(best_err));L=int(cand_lens[u]);a=all_policy[offsets[u]:offsets[u]+L].copy()
                a[:,:6]*=best_scale[u];a[:min(5,L),6]=grip[:min(5,L)]
                if first<0:first=u
                ids.append(u);chunks.append(a);used+=L
                hist=np.concatenate([hist,a],axis=0)[-10:];mask[:]=True
            return np.concatenate(chunks,axis=0)[:50],first,ids
        # a is reused verbatim from gap; b/c use a truth-selected first unit, then
        # use the original P5 runtime chain for the remaining units.
        if style in ('b','c'):
            u,_=first_action_oracle(qctx,grip,target_pol,ge,head_table)
            forced_scale=float(head_table[u]) if style=='b' else oracle_scale(u,target_pol[:min(5,int(lengths[u]))])
        else: u=-1;forced_scale=None
        while used<50:
            if used==0:
                selected=u if u>=0 else rank_select(qctx,hist,mask,ge)
            else:selected=rank_select(qctx,hist,mask,ge)
            scale=float(forced_scale if used==0 and forced_scale is not None else head_table[selected])
            a=unit_actions(selected,scale,grip)
            take=min(len(a),50-used);segment=a[:take].copy()
            if first<0:first=selected
            ids.append(selected);chunks.append(segment);used+=take
            hist=np.concatenate([hist,segment],axis=0)[-10:];mask[:]=True
        return np.concatenate(chunks,axis=0)[:50],first,ids

    # Prepare observations and all draft variants using the same 210 rows and
    # the exact query-row noise keys used by gap/offline_compare.py.
    stats=normalize.load(expert.CHECKPOINT/'assets')['actions']
    started=time.monotonic(); obs_records=[]; raw_targets=[]; drafts={c:[] for c in COND}; meta=[]
    for j,qi0 in enumerate(chosen):
        i=int(qi0);task=int(q['task'][i]);ep=int(q['episode'][i]);frame=int(q['frame'][i])
        info=infos[(task,ep)]
        obs,targ,raw_target,state_raw=expert._build_obs_and_target(info,frame,SRC,expert.CHECKPOINT)
        leaves,td=jax.tree.flatten(obs)
        if j and td!=tree_def: raise AssertionError('observation pytree structure changed')
        if not j:tree_def=td
        obs_records.append([np.asarray(jax.device_get(x))[0] for x in leaves])
        raw=np.asarray(raw_target,np.float32)
        if len(raw)<50:raw=np.pad(raw,((0,50-len(raw)),(0,0)))
        target_pol=ex.action_storage_to_policy(raw[:50]).astype(np.float32)
        raw_targets.append(np.asarray(targ,np.float32))
        pref,state,qctx,hist,mask=query(i);grip=grip_pred(pref,state);head_table=head_scales_for_query(qctx)
        # Existing A prediction is carried from gap, not recomputed with altered code.
        drafts['a_retrieved'].append(retrieved_existing[j])
        diag={}
        for style,name in [('b','b_action_oracle_scale_head'),('c','c_action_oracle_scale_truth'),('d','d_greedy_action_oracle_truth')]:
            draft_pol,first,ids=emit_draft(i,style,target_pol,pref,state,qctx,hist.copy(),mask.copy(),head_table)
            storage=ex.action_policy_to_storage(draft_pol)
            drafts[name].append(expert._draft_to_model(storage,stats))
            diag[style]=(first,ids)
            if style=='b': b_oracle_err=float(np.mean((draft_pol[:5,:7]-target_pol[:5,:7])**2))
            if style=='c': c_oracle_scale=float(oracle_scale(first,target_pol[:min(5,int(lengths[first]))]))
        drafts['e_truth_draft'].append(np.asarray(targ,np.float32))
        meta.append({'row_id':i,'task_id':task,'task':TASKS[task],'episode':ep,'frame':frame,
                     'b_first_unit':int(diag['b'][0]),'b_first_action_mse':b_oracle_err,
                     'c_first_unit':int(diag['c'][0]),'c_scale_oracle':c_oracle_scale,
                     'd_units':diag['d'][1]})
        if (j+1)%25==0:print(json.dumps({'stage':'prepare','rows':j+1,'total':len(chosen)}),flush=True)

    drafts_np={k:np.stack(v).astype(np.float32) for k,v in drafts.items()}
    # Make sure exact-truth draft has zero normalized active-7 error by construction.
    target_np=np.stack(raw_targets).astype(np.float32)
    if drafts_np['e_truth_draft'].shape!=target_np.shape:raise AssertionError('truth draft target shape mismatch')
    infer_batch=1
    for start in range(0,len(chosen),infer_batch):
        chunk=obs_records[start:start+infer_batch]
        leaves=[jnp.asarray(np.stack([rec[k] for rec in chunk])) for k in range(len(chunk[0]))]
        obs_records[start:start+infer_batch]=[None]*len(chunk)
        # convert the stacked batch to a single observation pytree
        if start==0: obs_batches=[]
        obs_batches.append((np.arange(start,min(start+infer_batch,len(chosen))),jax.tree.unflatten(tree_def,leaves)))

    # Model and compiled sampling functions. Fractions are constrained by env and torch/JAX process settings.
    model,_=_load_model_cpu_init()
    @nnx.jit(static_argnames=('num_steps',))
    def bridge(m,rng,obs,dn,t0,noise,*,num_steps=10):
        return m.sample_bridge(rng,obs,dn,t0,num_steps=num_steps,noise=noise)
    ncond=len(COND); predictions={name:{c:np.empty((len(chosen),50,32),np.float32) for c in COND} for name in ('pretrained','expert1000')}
    noise_all=[]
    rootkey=jax.random.key(4101)
    for i in chosen:noise_all.append(np.asarray(jax.random.normal(jax.random.fold_in(rootkey,int(i)),(1,50,32)))[0])
    noise_all=np.stack(noise_all).astype(np.float32)
    dn_all=np.stack([drafts_np[c] for c in COND],axis=1)
    ts=jnp.full((ncond*infer_batch,),.2,jnp.float32)
    for bid,obs in obs_batches:
        b=len(bid);rep=jax.tree.map(lambda x:jnp.repeat(x,ncond,axis=0),obs)
        dn=jnp.asarray(dn_all[bid].reshape(b*ncond,50,32));eps=jnp.asarray(np.repeat(noise_all[bid],ncond,axis=0))
        tvec=jnp.full((b*ncond,),.2,jnp.float32)
        out=np.asarray(bridge(model,jax.random.key(0),rep,dn,tvec,eps,num_steps=10)).reshape(b,ncond,50,32)
        for cidx,c in enumerate(COND):predictions['pretrained'][c][bid]=out[:,cidx]
    cp=ROOT/'expert-02/train/attempt-04/expert/expert_001000.msgpack'
    expert._overlay_expert(model,cp)
    for bid,obs in obs_batches:
        b=len(bid);rep=jax.tree.map(lambda x:jnp.repeat(x,ncond,axis=0),obs)
        dn=jnp.asarray(dn_all[bid].reshape(b*ncond,50,32));eps=jnp.asarray(np.repeat(noise_all[bid],ncond,axis=0))
        tvec=jnp.full((b*ncond,),.2,jnp.float32)
        out=np.asarray(bridge(model,jax.random.key(0),rep,dn,tvec,eps,num_steps=10)).reshape(b,ncond,50,32)
        for cidx,c in enumerate(COND):predictions['expert1000'][c][bid]=out[:,cidx]
    metrics={}
    for model_name in predictions:
        metrics[model_name]={}
        for c in COND:
            rows=[]
            for j in range(len(chosen)):
                row={'row_id':int(chosen[j]),'task':meta[j]['task'],'episode':meta[j]['episode'],'frame':meta[j]['frame']}
                row['draft']=_metric(drafts_np[c][j],target_np[j])
                row['generated']=_metric(predictions[model_name][c][j],target_np[j])
                rows.append(row)
            metrics[model_name][c]={'summary':{phase:{key:float(np.mean([r['generated'][phase][key] for r in rows])) for key in ('active7','position_xyz','rotation','gripper')} for phase in ('first5','horizon50')},
                                    'draft_summary':{phase:{key:float(np.mean([r['draft'][phase][key] for r in rows])) for key in ('active7','position_xyz','rotation','gripper')} for phase in ('first5','horizon50')},
                                    'by_task':{TASKS[t]:{phase:{key:float(np.mean([r['generated'][phase][key] for r in rows if next(m['task_id'] for m in meta if m['row_id']==r['row_id'])==t])) for key in ('active7','position_xyz','rotation','gripper')} for phase in ('first5','horizon50')} for t in range(6)},
                                    'rows':rows}
    payload={'status':'complete','n_rows':len(chosen),'per_task':{t:35 for t in TASKS},
      'conditions':{'a_retrieved':'existing gap runtime P5 draft, reused from offline_predictions.npz','b_action_oracle_scale_head':'first unit minimizes first-five active7 action MSE over all U memory units except same episode, magnitude-v2 scale and Context grip head; candidates are temporally resampled to five points for fair ranking, emitted at native duration; later units use normal runtime P5 chain','c_action_oracle_scale_truth':'same first-unit action oracle selection, first unit scale from positive clipped position XYZ least-squares ratio to true first five; later units normal runtime P5 chain','d_greedy_action_oracle_truth':'at each unit boundary choose the same-episode-excluded unit with minimum active7 action MSE over its own duration (clipped to remaining horizon), oracle position norm ratio; P5 is not used','e_truth_draft':'exact checkpoint-normalized target actions, padded/truncated to 50'},
      'expert_checkpoint':str(cp),'pretrained_checkpoint':str(expert.CHECKPOINT),'t0':.2,'integration_steps':10,
      'noise':'jax.random.normal(fold_in(key(4101), query_row_id), [50,32]), shared across conditions and models',
      'metric':'checkpoint-normalized active7 MSE, first five and full 50-step horizon; XYZ, rotation, gripper groups; rows are same episode-balanced 35/task from gap/offline_compare.py',
      'action_oracle_selection_uses_truth':'yes, this is an intentional upper-bound diagnosis, not a deployable search method',
      'summary':metrics,'elapsed_seconds':time.monotonic()-started}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(OUT/'predictions.npz',row_ids=chosen,target=target_np,
       **{f'draft_{k}':v for k,v in drafts_np.items()},
       **{f'pred_{model}_{k}':v for model,ds in predictions.items() for k,v in ds.items()})
    lines=['実装: 完了','','# RM-ACTORACLE: 行動誤差によるメモリ上限診断','',
      'オフライン診断。保留210窓（6タスク×35、gap/offline_compare.py と同一行）、Expert attempt-04 step1000、t0=.2、10積分。同じ query_row_id 由来のノイズを全下書き・両モデルで共有。行動MSEを用いる条件は正解を見て候補を選ぶ意図的な上限であり、検索性能や汎化を主張しない。指標はチェックポイント正規化済み active7。','',
      '|下書き|下書き first5 / 50|学習前π0.5 bridge first5 / 50|Expert step1000 first5 / 50|',
      '|---|---:|---:|---:|']
    for c in COND:
        d=metrics['pretrained'][c]['draft_summary']; p=metrics['pretrained'][c]['summary']; e=metrics['expert1000'][c]['summary']
        lines.append(f"|{c}|{d['first5']['active7']:.4f} / {d['horizon50']['active7']:.4f}|{p['first5']['active7']:.4f} / {p['horizon50']['active7']:.4f}|{e['first5']['active7']:.4f} / {e['horizon50']['active7']:.4f}|")
    lines += ['', '## 成分別（Expert step1000）','', '|下書き|範囲|active7|位置XYZ|回転|gripper|','|---|---|---:|---:|---:|---:|']
    for c in COND:
        s=metrics['expert1000'][c]['summary']
        for ph,label in [('first5','先頭5'),('horizon50','50 step')]:
            v=s[ph]; lines.append(f"|{c}|{label}|{v['active7']:.4f}|{v['position_xyz']:.4f}|{v['rotation']:.4f}|{v['gripper']:.4f}|")
    lines += ['', '## 判定','', '基準π0.5の同一ノイズ通常生成との比較とともに、(d) 全区間の行動オラクルと (e) 正解下書き後も Expert の誤差がπ0.5より大きいかを確認する。改善が残る場合は検索・倍率で到達できる余地があるが、行動oracleでも成功する保証ではない。', '', '行ごとの値と予測は `summary.json` / `predictions.npz` に保存。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','n_rows':len(chosen),'metrics':{m:{c:metrics[m][c]['summary']['first5']['active7'] for c in COND} for m in predictions},'out':str(OUT)},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
