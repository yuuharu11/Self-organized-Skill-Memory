#!/usr/bin/env python3
"""UQ1: train the fixed GRU × VLM cross-attention retriever/gripper query, then XF2 Expert."""
from __future__ import annotations
import argparse,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import xf2_run as X
import kc1_run as K1
import kc2_run as K2
import kq2cl_common as C

ROOT=X.ROOT;KC2=X.KC2;XF1=X.XF1;XF2=X.OUT
REPORT_OUT=ROOT/'uq1-01';OUT=REPORT_OUT/'production'
SEEDS=(1,2,3);TAUS=(.03,.1);LAMBDAS=(.3,1.,3.);SWITCH_WEIGHTS=(5.,20.)
EPOCHS=16;BATCH=256;RANDOM=256

class UQ1Net(K2.KC2Net):
    """Distance query heads with KQ2 iii_G1X2F's two cross-attention layers."""
    def __init__(self):
        super().__init__('iii','distance')
        from kq2_run import QueryNet
        self.query=QueryNet('gru',1,2,True,history='w10',dct=False)

def log(event,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**fields}
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,default=float)+'\n')
    print(json.dumps(row,ensure_ascii=False,default=float),flush=True)

def _assets(bank):
    # KC2 supplies the shared standardized coefficient space and base context.
    return X._kc2_assets(bank)

def _switch_info(sp):
    hist=np.asarray(sp['history'],np.float32);mask=np.asarray(sp['history_mask'],bool)
    last=hist[:,9,6];valid=mask.any(1)&(last!=0);sign=np.where(last>0,1.,-1.)
    y=(np.asarray(sp['actions'],np.float32)[...,6]>0).astype(np.float32)
    change=(np.where(y>0,1.,-1.)!=sign[:,None]) & valid[:,None]
    first=np.where(change.any(1),change.argmax(1),-1)
    return y,sign,valid,first

def _switch_f1(pred,target,last,valid):
    p=np.where(np.asarray(pred)>0,1.,-1.);t=np.where(np.asarray(target)>0,1.,-1.);g0=np.where(np.asarray(last)>0,1.,-1.)
    valid=np.asarray(valid,bool)&(g0!=0)
    def first(a):
        m=(a!=g0[:,None])&valid[:,None]
        return np.where(m.any(1),m.argmax(1),-1)
    pi,ti=first(p),first(t);tp=fp=fn=0;timing=[]
    for i in np.flatnonzero(valid):
        if ti[i]>=0:
            if pi[i]>=0 and abs(int(pi[i]-ti[i]))<=2:tp+=1;timing.append(abs(int(pi[i]-ti[i])))
            else:fn+=1
        elif pi[i]>=0:fp+=1
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*precision*recall/max(precision+recall,1e-12)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
            'timing_mae_steps':float(np.mean(timing)) if timing else None,
            'gt_switches':int((ti>=0).sum()),'valid_rows':int(valid.sum())}

def spread10(keys, top_ids):
    """Mean coefficient-wise variance across the ten retrieved keys."""
    k=np.asarray(keys,np.float32);ids=np.asarray(top_ids,np.int64)
    if ids.ndim!=2 or ids.shape[1]!=10 or k.ndim!=2 or np.any(ids<0) or np.any(ids>=len(k)):
        raise ValueError('spread10 expects [N,10] valid key IDs and [M,D] keys')
    return np.var(k[ids],axis=1).mean(axis=1).astype(np.float32)

def _metric(model,gs,sp,target_raw,bank,keys_np,threshold,device,batch=256):
    model.eval();tops=[];probs=[]
    with torch.inference_mode():
        kt=torch.as_tensor(keys_np,dtype=torch.float32,device=device)
        for st in range(0,gs.n,batch):
            ix=torch.arange(st,min(st+batch,gs.n),device=device);q=model(gs,ix)
            tops.append(model.score(q,kt).float().cpu().numpy())
            probs.append(torch.sigmoid(model.grip_logits(q)).float().cpu().numpy())
    score=np.concatenate(tops);prob=np.concatenate(probs)
    allow=K1.candidate_mask(sp['task'],sp['episode'],bank['task'],bank['episode']);score[~allow]=-np.inf
    picked=np.argmax(score,axis=1)
    coeff=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    err=(np.asarray(target_raw,np.float32)-coeff[picked])**2
    err=err.sum(1)
    y,last,valid,first=_switch_info(sp);decoded=C.decode_gripper(prob,last,threshold)
    return {'coef_sq':float(err.mean()),'coef_per_row':err.astype(np.float32),
            'grip':_switch_f1(decoded,y,last,valid),'decoded_gripper':decoded,
            'prob_close':prob.astype(np.float32),'scores':score.astype(np.float32),'top_ids':picked.astype(np.int32)}

def _target_loss_arrays(sp,weight):
    y,last,valid,first=_switch_info(sp);n=len(y)
    pos=np.arange(10)[None]
    timestep=np.ones_like(y,np.float32)
    for i,k in enumerate(first):
        if k>=0:timestep[i,np.abs(pos[0]-k)<=2]=float(weight)
    sample=np.where(first>=0,4.,1.).astype(np.float64);sample/=sample.sum()
    return y.astype(np.float32),timestep,first,sample

def _fit_query(cfg,ctx,targets,bank,keys_np,mean,std,*,fold=None,save_dir=None,device='cuda:0'):
    """Train one KC2Net(iii,distance); optional episode fold excludes fit and memory."""
    from kc2_run import KC2Net
    sp=ctx.splits['eh'];assign=X._split_fold_assignment(sp,bank)
    if fold is None:fit=np.arange(len(sp['key']));held=None;mem=np.arange(len(bank['length']))
    else:fit,held,mem=X.fold_partition(sp['task'],sp['episode'],bank['task'],bank['episode'],assign,int(fold))
    memtask=np.asarray(bank['task'])[mem];memep=np.asarray(bank['episode'])[mem];mkeys=keys_np[mem]
    raw=np.asarray(targets['eh'][0],np.float32);norm=(raw-mean)/std
    near=K1.nearest_allowed_coefficients(norm[fit],mkeys,sp['task'][fit],sp['episode'][fit],memtask,memep)
    y_all,tweight,first,sample_prob=_target_loss_arrays(sp,float(cfg['switch_weight']))
    gs=K1._as_gpusplit(ctx.kr,sp,torch.device(device),True)
    devsp=ctx.splits['dev2'];dgs=K1._as_gpusplit(ctx.kr,devsp,torch.device(device),True)
    init_seed=int(cfg['seed']*100000+int(round(cfg['tau']*1000))*100+int(round(cfg['lambda']*10))*10+int(cfg['switch_weight']))
    dest=Path(save_dir) if save_dir else OUT/'models'/('full' if fold is None else f'fold{fold}')
    dest.mkdir(parents=True,exist_ok=True);modelpath=dest/'selected.pt';histpath=dest/'train.json'
    if modelpath.exists() and histpath.exists():
        blob=torch.load(modelpath,map_location=device,weights_only=False);model=UQ1Net().to(device);model.load_state_dict(blob['state_dict']);model.eval()
        h=json.loads(histpath.read_text())
        return model,{'fit_rows':len(fit),'held_rows':None if held is None else len(held),'memory_units':len(mem),'checkpoint':str(modelpath),
          'history':h.get('history',[]),'dev_threshold':h.get('dev_threshold'),'dev_coef_sq':h.get('dev_coef_sq'),'dev_switch':h.get('dev_switch')}
    torch.manual_seed(init_seed);rng=np.random.default_rng(init_seed)
    model=UQ1Net().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    steps=math.ceil(len(fit)/BATCH);total=steps*EPOCHS
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda st:min(1.,(st+1)/steps)*.5*(1+math.cos(math.pi*min(1.,st/max(total,1)))))
    key_t=torch.as_tensor(mkeys,dtype=torch.float32,device=device);labels=torch.as_tensor(y_all,dtype=torch.float32,device=device)
    tw=torch.as_tensor(tweight,dtype=torch.float32,device=device)
    history=[];start_epoch=0
    prior=sorted(dest.glob('epoch*.pt'))
    if prior:
        last=prior[-1];blob=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(blob['state_dict']);opt.load_state_dict(blob['optimizer']);start_epoch=int(blob['epoch'])
        hpath=dest/'train.partial.json'
        if hpath.exists():history=json.loads(hpath.read_text()).get('history',[])
    fit_lookup=np.full(len(sp['key']),-1,np.int64);fit_lookup[fit]=np.arange(len(fit))
    for epoch in range(start_epoch+1,EPOCHS+1):
        model.train();order=rng.choice(fit,size=len(fit),replace=True,p=sample_prob[fit]/sample_prob[fit].sum());losses=[]
        for st in range(0,len(order),BATCH):
            rows=order[st:st+BATCH];loc=fit_lookup[rows]
            ix=torch.as_tensor(rows,dtype=torch.long,device=device);q=model(gs,ix)
            near_rows=near[loc];rand=rng.choice(len(mem),size=min(RANDOM,len(mem)),replace=False)
            cand=np.unique(np.r_[near_rows,rand]).astype(np.int64)
            allowed=K1.candidate_mask(sp['task'][rows],sp['episode'][rows],memtask[cand],memep[cand])
            rawt=torch.as_tensor(norm[rows],dtype=torch.float32,device=device)
            dd=torch.square(rawt[:,None]-key_t[cand][None]).sum(-1)
            sc=model.score(q,key_t[cand]);am=torch.as_tensor(allowed,dtype=torch.bool,device=device)
            soft=torch.softmax((-dd/float(cfg['tau'])).masked_fill(~am,-torch.inf),-1)
            logp=torch.log_softmax(sc.masked_fill(~am,-torch.inf),-1)
            retr=-(soft*torch.nan_to_num(logp,nan=0.,neginf=0.)).sum(-1).mean()
            gl=model.grip_logits(q);bce=F.binary_cross_entropy_with_logits(gl,labels[ix],reduction='none')
            grip=(bce*tw[ix]).sum(-1).div(tw[ix].sum(-1).clamp_min(1)).mean()
            loss=retr+float(cfg['lambda'])*grip
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step()
            with torch.no_grad():model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append((float(retr.detach()),float(grip.detach())))
        row={'epoch':epoch,'retrieval_loss':float(np.mean([v[0] for v in losses])),'gripper_bce':float(np.mean([v[1] for v in losses]))}
        history.append(row)
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'epoch':epoch,'fold':fold},dest/f'epoch{epoch:02d}.pt')
        (dest/'train.partial.json').write_text(json.dumps({'config':cfg,'history':history},indent=2,default=float)+'\n')
        log('query_train_epoch',fold=fold,**cfg,**row)
    model.eval();best_threshold=None;best_metric=None
    for threshold in (.3,.4,.5,.6,.7,.8,.9):
        m=_metric(model,dgs,devsp,targets['dev2'][0],bank,keys_np,threshold,device)
        g=m['grip'];key=(g['f1'],-abs(threshold-.5),-m['coef_sq'])
        if best_metric is None or key>best_metric:best_metric=key;best_threshold=threshold;devm=m
    blob={'state_dict':model.state_dict(),'config':cfg,'epoch':EPOCHS,'fold':fold,
          'dev_selected_threshold':float(best_threshold),'dev_coef_sq':float(devm['coef_sq']),'dev_switch':devm['grip']}
    tmp=modelpath.with_suffix('.tmp.pt');torch.save(blob,tmp);tmp.replace(modelpath)
    histpath.write_text(json.dumps({'config':cfg,'history':history,'dev_threshold':best_threshold,
        'dev_coef_sq':devm['coef_sq'],'dev_switch':devm['grip'],'fit_rows':len(fit),'held_rows':None if held is None else len(held),
        'memory_units':len(mem)},indent=2,ensure_ascii=False,default=float)+'\n')
    return model,{'fit_rows':len(fit),'held_rows':None if held is None else len(held),'memory_units':len(mem),
        'checkpoint':str(modelpath),'history':history,'dev_threshold':best_threshold,'dev_coef_sq':devm['coef_sq'],'dev_switch':devm['grip']}

def _score_rows(model,gs,sp,target_raw,bank,keys_np,row_ids,*,threshold,memory_ids=None,batch=256):
    mids=np.arange(len(bank['length']),dtype=np.int64) if memory_ids is None else np.asarray(memory_ids,np.int64)
    part=np.asarray(keys_np[mids],np.float32);kt=torch.as_tensor(part,dtype=torch.float32,device=next(model.parameters()).device)
    scores=[];probs=[]
    model.eval()
    with torch.inference_mode():
        for st in range(0,len(row_ids),batch):
            rows=np.asarray(row_ids[st:st+batch],np.int64);ix=torch.as_tensor(rows,dtype=torch.long,device=kt.device)
            q=model(gs,ix);scores.append(model.score(q,kt).float().cpu().numpy())
            probs.append(torch.sigmoid(model.grip_logits(q)).float().cpu().numpy())
    score=np.concatenate(scores);prob=np.concatenate(probs)
    allowed=K1.candidate_mask(sp['task'][row_ids],sp['episode'][row_ids],np.asarray(bank['task'])[mids],np.asarray(bank['episode'])[mids])
    score[~allowed]=-np.inf;order=np.argsort(-score,axis=1,kind='stable')[:,:10];top=mids[order]
    top_scores=np.take_along_axis(score,order,axis=1)
    spread=spread10(keys_np,top)
    _,last,valid,_=_switch_info({k:(np.asarray(v)[row_ids] if isinstance(v,np.ndarray) and len(v)==len(sp['key']) else v) for k,v in sp.items()})
    # history is aligned in `sp`; decoder only sees model probabilities and last executed sign.
    prev=np.where(valid,last,-1.).astype(np.float32);decoded=C.decode_gripper(prob,prev,float(threshold))
    draft=np.asarray(bank['value_actions10'])[top[:,0]].copy();draft[:,:,6]=decoded
    target=np.asarray(sp['actions'])[row_ids]
    errors=C.arm6(draft,target)
    raw_target=np.asarray(target_raw,np.float32)[row_ids]
    coeff=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    coef_sq=np.square(raw_target-coeff[top[:,0]]).sum(1)
    return {'top_ids':top.astype(np.int32),'top_scores':top_scores.astype(np.float32),'spread10':spread.astype(np.float32),
       'gripper_probs':prob.astype(np.float32),'decoded_gripper':decoded.astype(np.float32),'draft':draft.astype(np.float32),
       'e_arm':errors.astype(np.float32),'coef_sq':coef_sq.astype(np.float32)}

def _split_summary(rr,sp):
    y,last,valid,_=_switch_info(sp);g=_switch_f1(rr['decoded_gripper'],y,last,valid)
    return {'rows':int(len(y)),'coefficient_error_sq_mean':float(np.mean(rr['coef_sq'])),
      'arm6_mse_mean':float(np.mean(rr['e_arm'])),'switch':g}

def query_offline_metrics(ctx,targets,bank,rows_splits):
    result={'UQ1':{s:_split_summary(rows_splits[s],ctx.splits[s]) for s in ('dev2','learned','unseen20')},'KC2':{}}
    kjson=json.loads((KC2/'offline_results.json').read_text())['final_retrieval_and_joint_gripper']['joint']['splits']
    for s in ('dev2','learned','unseen20'):
        v=kjson[s];decoded=np.asarray(v['decoded'],np.float32);sp=ctx.splits[s];y,last,valid,_=_switch_info(sp)
        g=_switch_f1(decoded,y,last,valid)
        result['KC2'][s]={'rows':int(v['rows']),'coefficient_error_sq_mean':float(v['top1_coeff_sq']),
           'arm6_mse_mean':None,'switch':g,'source':'kc2-01/offline_results.json'}
    # XF3 is an independently trained gripper decoder. It has no coefficient
    # retrieval metric, so retain its measured transition metrics and mark that
    # field inapplicable instead of fabricating a query score.
    xf3=ROOT/'xf3-01'/'offline_results.json'
    if xf3.exists():
        xd=json.loads(xf3.read_text());best=xd['best_decoder'];per=xd['per_model'][best['name']]
        result['XF3']={'status':xd['status'],'selected':best['name'],'source':str(xf3),
            **{s:{'rows':int(per[s]['rows']),'coefficient_error_sq_mean':None,'arm6_mse_mean':None,
                  'switch':{k:per[s][k] for k in ('tp_within2','fp_within2','fn_within2','precision_within2',
                    'recall_within2','f1_within2','timing_mae_steps','gt_switch_rows')}}
              for s in ('dev2','learned','unseen20')}}
    else:
        result['XF3']={'status':'in_progress','source':'xf3-01/offline_results.json not yet present'}
    return result

def _all_switch_metrics(model,gs,sp,target_raw,bank,keys_np,thresholds,device):
    vals={t:_metric(model,gs,sp,target_raw,bank,keys_np,t,device) for t in thresholds}
    return vals

def fit_queries():
    """Grid-select UQ1 jointly; then rebuild five leakage-free folds and full production query."""
    if not (KC2/'DONE').is_file():raise RuntimeError('KC2 DONE gate is required')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'QUERY_DONE').exists():raise FileExistsError('QUERY_DONE exists; refusing to overwrite')
    mem,total=map(int,__import__('subprocess').run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().split(',')[:2])
    if mem>8192:raise RuntimeError(f'GPU memory busy at UQ1 query start: {mem}/{total} MiB')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    kr,ctx,targets=X._load_kc2_context();ctx.kr=kr;bank=K1.memory_bank();assignment=X._split_fold_assignment(ctx.splits['eh'],bank)
    chosen,fullck,keys_np,mean,std=_assets(bank);device='cuda:0'
    gs={s:K1._as_gpusplit(kr,ctx.splits[s],torch.device(device),True) for s in ('eh','dev2','learned','unseen20')}
    cfgs=[{'q_name':'iii','score_mode':'distance','xattn':2,'tau':tau,'lambda':lam,'switch_weight':w,'seed':seed}
      for tau in TAUS for lam in LAMBDAS for w in SWITCH_WEIGHTS for seed in SEEDS]
    threshold_grid=(.3,.4,.5,.6,.7,.8,.9);grid=[]
    for j,cfg in enumerate(cfgs):
        name=f"tau{cfg['tau']:g}_lam{cfg['lambda']:g}_w{cfg['switch_weight']:g}_s{cfg['seed']}"
        model,info=_fit_query(cfg,ctx,targets,bank,keys_np,mean,std,save_dir=OUT/'models'/'grid'/name,device=device)
        if info.get('dev_coef_sq') is None:
            measures=_all_switch_metrics(model,gs['dev2'],ctx.splits['dev2'],targets['dev2'][0],bank,keys_np,threshold_grid,device)
            best_th=max(threshold_grid,key=lambda t:(measures[t]['grip']['f1'],-abs(t-.5),-measures[t]['coef_sq']))
            info.update(dev_threshold=best_th,dev_coef_sq=measures[best_th]['coef_sq'],dev_switch=measures[best_th]['grip'])
        grid.append({'name':name,'config':cfg,'checkpoint':info['checkpoint'],'dev_threshold':info['dev_threshold'],
            'dev_coef_sq':info['dev_coef_sq'],'dev_switch':info['dev_switch'],'fit_rows':info['fit_rows'],
            'epochs':EPOCHS,'history':info['history']})
        log('query_grid_complete',index=j+1,total=len(cfgs),name=name,dev_coef_sq=info['dev_coef_sq'],
            dev_f1=info['dev_switch']['f1'],threshold=info['dev_threshold'])
    best_coef=min(float(r['dev_coef_sq']) for r in grid);eligible=[r for r in grid if r['dev_coef_sq']<=best_coef*1.05]
    selected=max(eligible,key=lambda r:(r['dev_switch']['f1'],-r['dev_coef_sq'],-r['config']['lambda'],r['config']['seed']))
    (OUT/'query_grid.json').write_text(json.dumps({'grid':grid,'best_dev_coef_sq':best_coef,'eligible_count':len(eligible),
      'selected':selected,'selection':'among configs with dev coefficient error <= 1.05 * global best, maximize Dev switch F1 (±2), then lower coefficient error'},indent=2,ensure_ascii=False,default=float)+'\n')
    cfg=selected['config'];th=float(selected['dev_threshold']);blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    if not (OUT/'query_full.pt').exists():torch.save(blob,OUT/'query_full.pt')
    from kc2_run import KC2Net
    fullmodel=UQ1Net().to(device);fullmodel.load_state_dict(blob['state_dict']);fullmodel.eval()
    # Full-memory production query at all common evaluation splits.
    rows_splits={}
    for split in ('dev2','learned','unseen20'):
        sp=ctx.splits[split];rr=_score_rows(fullmodel,gs[split],sp,targets[split][0],bank,keys_np,np.arange(len(sp['key'])),threshold=th)
        rr.update(task=np.asarray(sp['task']),episode=np.asarray(sp['episode']),frame=np.asarray(sp['key'])[:,2])
        rows_splits[split]=rr;np.savez_compressed(OUT/f'rows_{split}.npz',**rr)
    offmetrics=query_offline_metrics(ctx,targets,bank,rows_splits)
    (OUT/'query_offline_metrics.json').write_text(json.dumps(offmetrics,indent=2,ensure_ascii=False,default=float)+'\n')
    # Selected hyperparameters are refit in each XF1 episode fold; both query-fit and memory exclude held episodes.
    fold_rows=[]
    for f in range(5):
        fcfg=dict(cfg);model,info=_fit_query(fcfg,ctx,targets,bank,keys_np,mean,std,fold=f,save_dir=OUT/'models'/'folds'/f'fold{f}',device=device)
        fit,held,mids=X.fold_partition(ctx.splits['eh']['task'],ctx.splits['eh']['episode'],bank['task'],bank['episode'],assignment,f)
        rr=_score_rows(model,gs['eh'],ctx.splits['eh'],targets['eh'][0],bank,keys_np,held,threshold=th,memory_ids=mids)
        rr.update(task=np.asarray(ctx.splits['eh']['task'])[held],episode=np.asarray(ctx.splits['eh']['episode'])[held],
            frame=np.asarray(ctx.splits['eh']['key'])[held,2],row_id=held.astype(np.int32),fold=np.full(len(held),f,np.int8))
        path=OUT/f'rows_fold{f}.npz';np.savez_compressed(path,**rr);fold_rows.append(rr)
        log('query_fold_rows_done',fold=f,fit_rows=len(fit),held_rows=len(held),memory_units=len(mids),mean_e_arm=float(rr['e_arm'].mean()))
    cv={k:np.concatenate([r[k] for r in fold_rows],axis=0) for k in fold_rows[0] if isinstance(fold_rows[0][k],np.ndarray)}
    order=np.argsort(cv['row_id']);cv={k:v[order] for k,v in cv.items()}
    ehat,iso=X.fit_oof_isotonic(cv['spread10'],cv['e_arm'],cv['fold']);cv['ehat_oof']=ehat
    np.savez_compressed(OUT/'rows_eh_crossfit.npz',**cv)
    meta={'x':np.asarray(iso.X_thresholds_,float).tolist(),'y':np.asarray(iso.y_thresholds_,float).tolist(),'out_of_bounds':'clip'}
    (OUT/'isotonic.json').write_text(json.dumps(meta,indent=2)+'\n')
    dev_ehat=X.apply_isotonic(meta,rows_splits['dev2']['spread10'])
    fb=X.calibration_threshold(dev_ehat,.204,spread=rows_splits['dev2']['spread10'])
    calibration={'isotonic':meta,'fallback':fb,'p_b':.204,'threshold_source':'Dev v2 only; spread tie-break for isotonic plateaus',
      'query_config':selected['name'],'gripper_threshold':th}
    (OUT/'calibration.json').write_text(json.dumps(calibration,indent=2,ensure_ascii=False)+'\n')
    (OUT/'calibration_v2.json').write_text(json.dumps(calibration,indent=2,ensure_ascii=False)+'\n')
    eh=ctx.splits['eh'];train={'row_id':cv['row_id'],'fold':cv['fold'],'task':cv['task'],'episode':cv['episode'],
       'draft':cv['draft'][:,:,:7],'target':np.asarray(eh['actions'])[cv['row_id'],:,:7].astype(np.float32),
       'ehat_oof':ehat,'spread10':cv['spread10'],'e_arm':cv['e_arm']}
    np.savez_compressed(OUT/'expert_train_rows.npz',**train)
    # Reuse the exact immutable observation tensors prepared by XF2; verify row-key identity.
    src=XF2/'obs_eh'
    if not src.exists():raise FileNotFoundError('XF2 verified EH observations are not prepared')
    if not (OUT/'obs_eh').exists():(OUT/'obs_eh').symlink_to(src,target_is_directory=True)
    if not np.array_equal(np.load(OUT/'obs_eh'/'key.npy'),np.asarray(eh['key'],np.int32)):raise AssertionError('EH observation key order mismatch')
    manifest={'status':'query_complete','row_counts':{'eh_crossfit':len(cv['row_id']),'dev2':len(rows_splits['dev2']['spread10']),
        'learned':len(rows_splits['learned']['spread10']),'unseen20':len(rows_splits['unseen20']['spread10'])},
        'selected_config':selected,'grid_size':len(grid),'folds':5,'fold_assignment':str(XF1/'manifest.json'),
        'query_architecture':'UQ1Net (KC2Net distance heads): KQ2 iii_G1X2F, GRU h(256) queries 34x2048 VLM K/V via two cross-attention layers; no VLM concatenation',
        'VLM_rows_20Hz':True,'all_fit_and_memory_fold_exclusion':True,'Gdec_threshold':th,
        'isotonic_out_of_fold':True,'fallback':fb,'memory_units':len(bank['length'])}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=float)+'\n')
    (OUT/'query_selection.json').write_text(json.dumps({'selected':selected,'best_dev_coef_sq':best_coef,'eligible_count':len(eligible)},indent=2,ensure_ascii=False,default=float)+'\n')
    log('query_done',selected=selected['name'],dev_coef_sq=selected['dev_coef_sq'],dev_f1=selected['dev_switch']['f1'],
        fallback_realized=fb['realized_rate'],rows=len(cv['row_id']))
    (OUT/'QUERY_DONE').write_text('UQ1 query/fold/crossfit complete\n')
    write_report(None)
    return manifest

def train_expert(updates=5000):
    if not (OUT/'QUERY_DONE').is_file():raise RuntimeError('UQ1 query crossfit must complete before Expert training')
    X.OUT=OUT
    # XF2's shared trainer calls the save hook on `expert_ex3`, while the
    # exported filter-aware hook lives in `skill_expert_refine`.
    import expert_ex3 as EX
    import skill_expert_refine as E3
    EX._save_state=E3._save_state
    # Eight-sample effective batch, microbatch one to fit the shared accelerator.
    return X.train_expert(updates=updates,batch=8,microbatch=1,save_every=500)

def evaluate_offline():
    if not (OUT/'expert/train-XF2-K1/checkpoints/005000').exists():raise FileNotFoundError('UQ1 Expert checkpoint missing')
    X.OUT=OUT
    result=X.offline()
    kr,ctx,targets=X._load_kc2_context();ctx.kr=kr;bank=K1.memory_bank()
    rows_splits={s:{k:v for k,v in np.load(OUT/f'rows_{s}.npz',allow_pickle=False).items()}
                 for s in ('dev2','learned','unseen20')}
    qm=query_offline_metrics(ctx,targets,bank,rows_splits)
    (OUT/'query_offline_metrics.json').write_text(json.dumps(qm,indent=2,ensure_ascii=False,default=float)+'\n')
    marker=REPORT_OUT/'OFFLINE_DONE'
    if not marker.exists():marker.write_text('UQ1 offline selection complete\n')
    write_report(None)
    return result

def _success_map(path):
    out={};p=Path(path)
    if not p.exists():return out
    for line in p.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('status')=='complete':out[(int(r['task_id']),int(r['trial_index']))]=bool(r['success'])
    return out

def _references(suite):
    from t0x_report import baseline_for_suite
    t5=ROOT/'t0x-01'/'t5'/'closed-loop'/'closed-loop'/suite/'AtopkGdec'/'trials.jsonl'
    kc=KC2/'closed-loop'/suite/'KC2-dot-Gdec'/'trials.jsonl'
    _,pi=baseline_for_suite(suite)
    return {'KC2':_success_map(kc),'T5_A-topk-Gdec':_success_map(t5),
       'pi05':{(int(k[0]),int(k[1])):bool(v['success']) for k,v in pi.items() if v.get('status')=='complete'}}

def run_closed(*,fraction=.20,workers=8,smoke=False):
    """Run V/F serially through the standard CL1 runner; resume per-trial files."""
    import subprocess
    import kq2cl_closed as CL
    if not (OUT/'OFFLINE_DONE').is_file():raise RuntimeError('offline selection must finish before closed loop')
    if (REPORT_OUT/'DONE').exists():raise FileExistsError('UQ1 DONE already exists; preserving outputs')
    used,total=map(int,subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().split(',')[:2])
    if used>min(8192,int(total*fraction*.65)):raise RuntimeError(f'GPU memory too busy before UQ1 servers: {used}/{total} MiB')
    X.OUT=OUT;CL.C.OUT=OUT
    offline=json.loads((OUT/'offline_results.json').read_text());sel=json.loads((OUT/'query_selection.json').read_text())['selected']
    query_ck=OUT/'query_full.pt';expert=OUT/'expert/train-XF2-K1/checkpoints/005000';cal=OUT/'calibration_v2.json'
    spec={name:{'mode':'memory','query':'iii','expert':str(expert),'bundle':str(query_ck),
      'tpol':{'strategy':name,'calibration':str(cal),'query_checkpoint':str(query_ck),
              'gripper_threshold':float(sel['dev_threshold'])}} for name in ('UQ1-V','UQ1-F')}
    specpath=OUT/'closed-spec-smoke.json' if smoke else OUT/'closed-spec-production.json'
    if specpath.exists() and json.loads(specpath.read_text())!=spec:raise FileExistsError(f'conflicting existing spec {specpath}')
    if not specpath.exists():C.write_json(specpath,spec)
    refs={s:_references(s) for s in ('libero90','libero10')}
    real_popen=CL.subprocess.Popen
    def launch(args,*a,**kw):
        if isinstance(args,(list,tuple)):
            args=list(args)
            for i,v in enumerate(args):
                if str(v).endswith('/kq2cl_serve.py'):args[i]=str(HERE/'uq1_serve.py');break
        return real_popen(args,*a,**kw)
    CL.subprocess.Popen=launch
    oldenv={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    suites={}
    try:
        for suite in ('libero90','libero10'):
            CL.PLAN[suite]=dict(CL.PLAN[suite])
            for name in ('UQ1-V','UQ1-F'):CL.PLAN[suite][name]=range(2) if smoke else range(10)
            for name in ('UQ1-V','UQ1-F'):
                log('closed_start',suite=suite,condition=name,gpu_used_mib=used,gpu_total_mib=total,fraction=fraction,smoke=smoke)
                CL.run([name],workers=1 if smoke else workers,fraction=fraction,smoke=smoke,cpus=['24-39'],spec_path=str(specpath),suite=suite)
            stats={}
            trial_root=OUT/(('closed-loop-smoke' if suite=='libero90' else 'closed-loop-smoke-libero10') if smoke else f'closed-loop/{suite}')
            for name in ('UQ1-V','UQ1-F'):
                tr=_success_map(trial_root/name/'trials.jsonl')
                expected=2 if smoke else 100
                if len(tr)!=expected:raise RuntimeError(f'{suite}/{name} has {len(tr)}/{expected} complete trials')
                comparison={ref:X.paired_counts(tr,v) for ref,v in refs[suite].items()}
                stats[name]={'completed':len(tr),'successes':sum(tr.values()),'paired':comparison,
                    'references':{r:{'completed':len(v),'successes':sum(v.values())} for r,v in refs[suite].items()}}
            stats['UQ1-V_vs_UQ1-F']=X.paired_counts(_success_map(trial_root/'UQ1-V'/'trials.jsonl'),
                _success_map(trial_root/'UQ1-F'/'trials.jsonl'))
            suites[suite]=stats
    finally:
        CL.subprocess.Popen=real_popen;CL.C.OUT=ROOT/'kq2-cl-01'
        for k,v in oldenv.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    if smoke:return {'status':'smoke_complete','suites':suites}
    result={'status':'complete','suites':suites,'paired_protocol':'CL1: task/trial paired; seed_base=7, wait=10, replan=5, max_steps=520',
       'fallback':json.loads(cal.read_text())['fallback'],'query_selection':sel,'gpu_before':{'used_mib':used,'total_mib':total,'fraction':fraction}}
    path=REPORT_OUT/'closed_loop_results.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    write_report(result)
    (REPORT_OUT/'DONE').write_text('UQ1 complete\n')
    log('closed_done',success={s:{n:v['successes'] for n,v in d.items() if n.startswith('UQ1-')} for s,d in suites.items()})
    return result

def write_report(closed=None):
    q=json.loads((OUT/'query_selection.json').read_text());sel=q['selected']
    off=json.loads((OUT/'offline_results.json').read_text()) if (OUT/'offline_results.json').exists() else None
    train=np.load(OUT/'expert_train_rows.npz',allow_pickle=False) if (OUT/'expert_train_rows.npz').exists() else None
    lines=['# UQ1：GRU×VLM クエリと信頼度連動 Expert', '',
       f"作成時刻: {time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())} / {time.strftime('%Y-%m-%d %H:%M:%S JST',time.localtime(time.time()+9*3600))}", '',
       '## 結論と主表', '',
       '| Suite | 条件 | 成功数 | 対KC2 (UQ1のみ/対照のみ; p) | 対T5 (同; p) | 対π0.5 (同; p) | V/F (Vのみ/Fのみ; p) |',
       '|---|---|---|---|---|---|---|']
    if closed:
        for suite,d in closed['suites'].items():
            for cond in ('UQ1-V','UQ1-F'):
                x=d[cond];vf=d['UQ1-V_vs_UQ1-F']
                pair=lambda name: f"{x['paired'][name]['xf2_only']}/{x['paired'][name]['reference_only']}; {x['paired'][name]['mcnemar_p_exact']:.4g}"
                lines.append(f"| {suite} | {cond} | {x['successes']}/100 | {pair('KC2')} | {pair('T5_A-topk-Gdec')} | {pair('pi05')} | {vf['xf2_only']}/{vf['reference_only']}; {vf['mcnemar_p_exact']:.4g} |")
    else:lines.append('| 閉ループ未完了 | — | — | — | — | — | — |')
    lines += ['', '選択クエリ：GRU×VLM cross-attention、tau={tau}, lambda={lam}, switch-weight={w}, seed={seed}; Dev Gdec threshold={th:.2f}。選択基準は Dev v2 係数誤差の全体最良から5%以内で切替 F1 (±2) 最大。'.format(
       tau=sel['config']['tau'],lam=sel['config']['lambda'],w=sel['config']['switch_weight'],seed=sel['config']['seed'],th=sel['dev_threshold']), '']
    if off:
        lines += ['## オフライン Expert 比較（腕6軸 MSE）','', '| split | 条件 | 平均 | p50 | p90 |','|---|---|---:|---:|---:|']
        for split,metrics in off['metrics'].items():
            for name,m in metrics.items():lines.append(f"| {split} | {name} | {m['mean_arm6']:.6f} | {m['p50']:.6f} | {m['p90']:.6f} |")
        lines += ['', f"Dev 選択: 可変 `{off['dev_selection']['best_variable']}`, 固定 `{off['dev_selection']['best_fixed']}`。Dev fallback率 {off['dev_selection']['fallback']['realized_rate']:.3f} (target .204).", '']
    qm=OUT/'query_offline_metrics.json'
    if qm.exists():
        metrics=json.loads(qm.read_text())
        lines += ['## クエリ・グリッパーのオフライン比較','',
          '| split | モデル | 行数 | 係数二乗誤差 | 腕6軸MSE | 切替P/R/F1 | ±2ステップ誤差 | GT切替数 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
        for name in ('UQ1','KC2','XF3'):
            for split in ('dev2','learned','unseen20'):
                m=metrics.get(name,{}).get(split)
                if not m:continue
                g=m['switch'];coef='—' if m.get('coefficient_error_sq_mean') is None else f"{m['coefficient_error_sq_mean']:.5f}"
                arm='—' if m.get('arm6_mse_mean') is None else f"{m['arm6_mse_mean']:.5f}"
                p=g.get('precision',g.get('precision_within2'));r=g.get('recall',g.get('recall_within2'));f1=g.get('f1',g.get('f1_within2'))
                timing=g.get('timing_mae_steps');gt=g.get('gt_switches',g.get('gt_switch_rows','—'))
                lines.append(f"| {split} | {name} | {m['rows']} | {coef} | {arm} | {p:.3f}/{r:.3f}/{f1:.3f} | {'—' if timing is None else f'{timing:.3f}'} | {gt} |")
        lines += ['', '係数誤差は KC1 標準化済み保存係数空間での top-1 二乗誤差。腕6軸のオフライン指標は選んだ型の下書きに対するもの。XF3 は独立したグリッパー読出しで係数検索を持たないため係数誤差・腕誤差は適用外。切替は±2ステップ照合。', '']
    if train is not None:
        e=train['e_arm'];eh=train['ehat_oof']
        lines += ['## Expert 学習用クロスフィット下書き','',
          f"行数 {len(e)}; retrieved draft arm MSE 平均 / p50 / p90 = {e.mean():.6f} / {np.median(e):.6f} / {np.quantile(e,.9):.6f}; OOF ê = {eh.mean():.6f} / {np.median(eh):.6f} / {np.quantile(eh,.9):.6f}.",
          '各 fold は query fit と memory の双方から heldout episodes を除外。グリッパー下書きも fold モデルの OOF 出力。', '']
    lines += ['## 指標・選択の注記','',
       '- 係数誤差は保存 DCT 係数の生空間の二乗誤差。切替 F1 は最後の実行グリッパーからの最初の符号切替を±2ステップで照合。',
       '- 同点成功列は「UQ1のみ / 対照のみ」。`paired_counts` が全ペアの成功差と不一致 discordant 差を assert する。',
       '- t0 は ê=E[e_arm|spread10] isotonic の後 `clip(k√ê/(1+k√ê), .05, .6)`。Dev の isotonic plateau 同点だけ spread10 で切替率を校正。',
       '- GPU OOM 試行数は progress.log に記録。', '']
    (REPORT_OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('query','expert','offline','smoke','closed','all'))
    ap.add_argument('--updates',type=int,default=5000);ap.add_argument('--fraction',type=float,default=.20);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    if a.stage in ('query','all'):fit_queries()
    if a.stage in ('expert','all'):train_expert(a.updates)
    if a.stage in ('offline','all'):evaluate_offline()
    if a.stage=='smoke':print(json.dumps(run_closed(fraction=a.fraction,workers=a.workers,smoke=True),indent=2,default=float))
    if a.stage in ('closed','all'):print(json.dumps(run_closed(fraction=a.fraction,workers=a.workers),indent=2,ensure_ascii=False,default=float))
if __name__=='__main__':main()


