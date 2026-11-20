#!/usr/bin/env python3
"""Dev-only model/threshold selection and final KC2 offline evaluation."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kc2_run as K
import kc1_run as K1


class QOnlyHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,10))
    def forward(self,x):return self.net(x)


def model_from(row,device='cpu'):
    blob=torch.load(row['checkpoint'],map_location=device,weights_only=False)
    cfg=blob['config'];m=K.KC2Net(cfg['q_name'],cfg['score_mode']).to(device)
    m.load_state_dict(blob['state_dict']);m.eval();return m,cfg


def build_context():
    return K.load_ctx()


def split_arrays(sp,split):
    d=K.OUT/'data'/split
    h=np.load(d/'history.npy');hm=np.load(d/'history_mask.npy');state=np.load(d/'state.npy')
    last=h[:,9,6];valid=hm.any(1)&(last!=0)
    import kq2cl_common as C
    gx=C.grip_extra(last,valid,state[:,6:8])
    target=np.where(np.asarray(sp['actions'])[...,6]>0,1.,-1.).astype(np.float32)
    return gx,last,valid,target


def features_for(model,kr,sp,split,keys,key_task,key_episode):
    # KC1's feature builder is reused for the explicitly requested extra-input control.
    import kc1_head as H
    gx,_,_,_ = split_arrays(sp,split)
    gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),model.q_name=='iii')
    feat=H.retrieval_features(model,gs,sp,keys,key_task,key_episode)
    return feat,gx


def fit_head(kind,x,y,seed):
    torch.manual_seed(seed);torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if kind=='q':m=QOnlyHead()
    else:
        import kc1_head as H
        m=H.KC1Head(x.shape[1])
    xt=torch.as_tensor(x,dtype=torch.float32);yt=torch.as_tensor(y,dtype=torch.float32)
    if kind=='extra':
        m.mu.copy_(xt.mean(0));m.sd.copy_(xt.std(0).clamp_min(1e-4))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);rng=np.random.default_rng(seed)
    for _ in range(30):
        m.train()
        for _st in range(0,len(xt),256):
            ix=torch.as_tensor(rng.integers(0,len(xt),min(256,len(xt))))
            loss=F.binary_cross_entropy_with_logits(m(xt[ix]),yt[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m.cpu().eval()


def choose_threshold(probs,target,last,valid):
    import kc1_head as H
    import kq2cl_common as C
    target_sign=np.where(np.asarray(target)>0,1.,-1.)
    def metrics_at(t):
        m=H._metrics(probs,target_sign,last,valid,t)
        decoded=C.decode_gripper(probs,np.where(valid,np.sign(last),-1.),t)
        m['exact_step_accuracy']=float((decoded==target_sign).mean())
        return m
    # Dense enough to expose early/weak opposite-side probabilities.  Selection
    # remains Dev-only and uses the same switch-row error count as KC1.
    metrics={str(t):metrics_at(float(t)) for t in np.arange(.05,1.,.05)}
    th=min(metrics,key=lambda z:(metrics[z]['errors'],-float(z)))
    return float(th),metrics[th],metrics


def kc1_gripper_reference():
    """Re-run frozen KC1 post-hoc head on the same offline splits."""
    import kc1_head as H
    old=K1.OUT;K1.OUT=K.K1_OUT
    try:
        kr,ctx,targets=K1._prepare_context();bank=K1.memory_bank()
        row=json.loads((K.K1_OUT/'offline_corrected.json').read_text())['selected']
        ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False);cfg=ck['cfg']
        model=K1._load_model(kr,row['query'],'cpu',row['seed']);model.load_state_dict(ck['query']);model.eval()
        kn,_,_=K1.coefficient_keys(np.asarray(bank['coefficients']),standardize=row['standardize'],
            mean=np.asarray(cfg['mean']),std=np.asarray(cfg['std']))
        keys=torch.as_tensor(kn,dtype=torch.float32)
        b=torch.load(K.K1_OUT/'heads/bundle.pt',map_location='cpu',weights_only=False)
        head=H.KC1Head(int(b['input_dim']));head.load_state_dict(b['state_dict']);head.eval()
        out={}
        for split in ('dev2','learned','unseen20'):
            sp=ctx.splits[split];gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),row['query']=='iii')
            feat=H.retrieval_features(model,gs,sp,keys,np.asarray(bank['task']),np.asarray(bank['episode']))
            # KC2 materializes one aligned history/state file per split; use it
            # here so learned and unseen20 rows never fall through to Dev's
            # legacy KC1 fallback path.
            gx,last,valid,target=split_arrays(sp,split)
            x=H.head_input(feat,gx,b['kind'])
            with torch.no_grad():probs=torch.sigmoid(head(torch.as_tensor(x))).numpy()
            m=H._metrics(probs,target,last,valid,float(b['threshold']))
            # Reconstruct decoded actions for direct 10-step agreement.
            import kq2cl_common as C
            decoded=C.decode_gripper(probs,np.where(valid,np.sign(last),-1.),float(b['threshold']))
            signed=np.where(target>0,1.,-1.)
            m['exact_step_accuracy']=float((decoded==signed).mean())
            out[split]=m
        return out
    finally:K1.OUT=old


def _evaluate_retrieval(model,gs,sp,target_raw,raw_keys,keyspace,bank,labels,last,valid,threshold):
    model.eval();scores=[];pr=[]
    with torch.no_grad():
        for st in range(0,gs.n,128):
            ix=torch.arange(st,min(st+128,gs.n));q=model(gs,ix)
            scores.append(model.score(q,keyspace).cpu().numpy());pr.append(torch.sigmoid(model.grip_logits(q)).cpu().numpy())
    scores=np.concatenate(scores);probs=np.concatenate(pr)
    allowed=K1.candidate_mask(sp['task'],sp['episode'],bank['task'],bank['episode']);scores[~allowed]=-np.inf
    top=np.argmax(scores,1);raw=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    ce=np.square(np.asarray(target_raw,np.float32)-raw[top]).sum(-1)
    import kq2cl_common as C
    dec=C.decode_gripper(probs,np.where(valid,np.sign(last),-1.),threshold);tg=np.where(labels>0,1.,-1.)
    gm=C.switch_metrics(dec,tg,last,valid);gm.update(errors=int(gm['missed_rows']+gm['false_switch_rows']),
        exact_step_accuracy=float((dec==tg).mean()))
    return {'rows':len(top),'top1_coeff_sq':float(ce.mean()),'per_row_coeff_sq':ce.tolist(),
            'gripper':gm,'decoded':dec.tolist(),'prob_close':probs.tolist()}


def _fit_posthoc(model,kr,ctx,split_targets,bank,keyspace,kind,tag,seed):
    feats={};extras={}
    for split in ('eh','dev2','learned','unseen20'):
        feat,gx=features_for(model,kr,ctx.splits[split],split,keyspace,np.asarray(bank['task']),np.asarray(bank['episode']))
        feats[split]=feat;extras[split]=gx
    train_target=(np.asarray(ctx.splits['eh']['actions'])[...,6]>0).astype(np.float32)
    if kind=='q':xtr=feats['eh']['q'];xdev=feats['dev2']['q']
    else:
        import kc1_head as H
        xtr=H.head_input(feats['eh'],extras['eh'],'full');xdev=H.head_input(feats['dev2'],extras['dev2'],'full')
    head=fit_head(kind,xtr,train_target,seed)
    with torch.no_grad():pdev=torch.sigmoid(head(torch.as_tensor(xdev))).numpy()
    gx,last,valid,tdev=split_arrays(ctx.splits['dev2'],'dev2')
    th,metrics,_=choose_threshold(pdev,tdev,last,valid)
    hdir=K.OUT/'posthoc';hdir.mkdir(exist_ok=True)
    hp=hdir/f'{tag}_{kind}.pt'
    if not hp.exists():
        torch.save({'kind':kind,'state_dict':head.state_dict(),'threshold':th,'seed':seed,
            'input_dim':xtr.shape[1],'tag':tag},hp)
    result={'threshold':th,'dev_errors':metrics['errors'],'dev_exact_step_accuracy':metrics['exact_step_accuracy'],'splits':{}}
    for split in ('dev2','learned','unseen20'):
        if kind=='q':x=feats[split]['q']
        else:
            import kc1_head as H
            x=H.head_input(feats[split],extras[split],'full')
        with torch.no_grad():p=torch.sigmoid(head(torch.as_tensor(x))).numpy()
        gx,last,valid,target=split_arrays(ctx.splits[split],split)
        import kc1_head as H
        result['splits'][split]=H._metrics(p,target,last,valid,th)
        import kq2cl_common as C
        decoded=C.decode_gripper(p,np.where(valid,np.sign(last),-1.),th)
        result['splits'][split]['exact_step_accuracy']=float((decoded==np.where(target>0,1.,-1.)).mean())
    return result


def evaluate_all():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    dest=K.OUT/'offline_results.json'
    if dest.exists():raise FileExistsError(dest)
    model_rows=[]
    for jp in sorted((K.OUT/'models').glob('seed*/*.json')):
        row=json.loads(jp.read_text());model_rows.append(row)
    if len(model_rows)!=144:raise AssertionError(f'expected 144 checkpoints, found {len(model_rows)}')
    eligible=[r for r in model_rows if r['lambda']>0 and K.within_retrieval_constraint(r['dev_raw_coeff_sq'])]
    if not eligible:raise RuntimeError('no joint model within KC1 Dev coefficient error +5%')
    # Re-evaluate the gripper threshold densely on Dev for all retrieval-eligible
    # joint candidates, avoiding the coarse training-time threshold grid.
    kr,ctx,targets=build_context();bank=K1.memory_bank()
    dev=ctx.splits['dev2'];gd=K1._as_gpusplit(kr,dev,torch.device('cpu'),False)
    ddev=K.OUT/'data'/'dev2';dh=np.load(ddev/'history.npy');dm=np.load(ddev/'history_mask.npy')
    dlast=dh[:,9,6];dvalid=dm.any(1)&(dlast!=0)
    dlabels=(np.asarray(dev['actions'])[...,6]>0).astype(np.float32)
    import kc1_head as H
    for row in eligible:
        model,_=model_from(row)
        gd=K1._as_gpusplit(kr,dev,torch.device('cpu'),row['q_name']=='iii')
        with torch.no_grad():
            pp=[]
            for st in range(0,gd.n,128):
                ix=torch.arange(st,min(st+128,gd.n));pp.append(torch.sigmoid(model.grip_logits(model(gd,ix))).numpy())
        probs=np.concatenate(pp)
        th,metric,metrics=choose_threshold(probs,dlabels,dlast,dvalid)
        row['dev_gripper_errors']=int(metric['errors']);row['dev_threshold']=float(th)
        row['dev_gripper_exact_accuracy']=float(metric['exact_step_accuracy'])
        row['dev_threshold_metrics']={k:{'errors':v['errors'],'exact_step_accuracy':v['exact_step_accuracy']} for k,v in metrics.items()}
    selected=K.select_joint_candidate(eligible)
    lam0name=K._config_name(selected['q_name'],selected['score_mode'],selected['tau'],0.,selected['seed'])
    lambda0=next(r for r in model_rows if r['name']==lam0name)
    # Build the stable data context once for all splits and calibration below.
    kc1meta=json.loads((K.K1_OUT/'offline_corrected.json').read_text())['selected'];kc1cfg=torch.load(kc1meta['checkpoint'],map_location='cpu',weights_only=False)['cfg']
    raw=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    keyspace,mean,std=K1.coefficient_keys(raw,standardize=True,mean=np.asarray(kc1cfg['mean']),std=np.asarray(kc1cfg['std']))
    keys=torch.as_tensor(keyspace,dtype=torch.float32)
    dev_records={}
    for row in eligible:
        dev_records[row['name']]={'lambda':row['lambda'],'q_name':row['q_name'],'score_mode':row['score_mode'],
            'tau':row['tau'],'seed':row['seed'],'dev_raw_coeff_sq':row['dev_raw_coeff_sq'],
            'dev_gripper_errors':row['dev_gripper_errors'],'within_kc1_5pct':row['within_kc1_5pct']}
    modelset={}
    for key,row in [('joint',selected),('lambda0',lambda0)]:modelset[key]=model_from(row)[0]
    final={}
    for tag,row in [('joint',selected),('lambda0',lambda0)]:
        model=modelset[tag];final[tag]={'config':{k:row[k] for k in ('name','q_name','score_mode','tau','lambda','seed','selected_epoch','dev_raw_coeff_sq','dev_gripper_errors','dev_threshold')},'splits':{}}
        for split in ('dev2','learned','unseen20'):
            sp=ctx.splits[split];gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),row['q_name']=='iii')
            gx,last,valid,sgn=split_arrays(sp,split);labels=(np.asarray(sp['actions'])[...,6]>0).astype(np.float32)
            threshold=float(row['dev_threshold']) if tag=='joint' else .5
            final[tag]['splits'][split]=_evaluate_retrieval(model,gs,sp,targets[split][0],raw,keys,bank,labels,last,valid,threshold)
            # The lambda=0 head is not trained; its head metrics are intentionally omitted.
    posthoc={}
    for tag,row in [('joint',selected),('lambda0',lambda0)]:
        posthoc[f'{tag}_q_only']=_fit_posthoc(modelset[tag],kr,ctx,targets,bank,keys,'q',tag,seed=17+row['seed'])
        posthoc[f'{tag}_kc1_extra']=_fit_posthoc(modelset[tag],kr,ctx,targets,bank,keys,'extra',tag,seed=37+row['seed'])
    kc1ref=kc1_gripper_reference()
    # Match T5's Dev fallback mass with KC2's own top-10 disagreement, then use
    # the same t0 mapping online.  No Eval rows enter this calibration.
    import kq2cl_common as C
    from t0x_common import topk_disagreement,calibrate_disagreement
    model=modelset['joint'];gs=K1._as_gpusplit(kr,ctx.splits['dev2'],torch.device('cpu'),selected['q_name']=='iii')
    sc=[]
    with torch.no_grad():
        for st in range(0,gs.n,128):
            ix=torch.arange(st,min(st+128,gs.n));sc.append(model.score(model(gs,ix),keys).numpy())
    scores=np.concatenate(sc);sp=ctx.splits['dev2']
    allowed=K1.candidate_mask(sp['task'],sp['episode'],bank['task'],bank['episode'])
    summ=C.summarize_scores(scores,allowed,k=10)
    units=np.asarray(bank['value_actions10'],np.float32)
    D,_=topk_disagreement(units[np.asarray(summ['ids'],np.int64)],np.asarray(summ['logits']),k=10)
    basecal=json.loads((K.ROOT/'t0x-01/t5/calibration.json').read_text())
    oldconf=np.load(K.ROOT/'kq2-cl-01/heads/pred_iii.npz')['dev2__ehat']
    calibration=calibrate_disagreement(D,oldconf,float(basecal['b']))
    calibration.update({'query':selected['q_name'],'score_mode':selected['score_mode'],
        'dev_D_quantiles_p05_p50_p95':np.quantile(D,[.05,.5,.95]).tolist(),
        'reference_rule':'match T5 Dev v2 ehat>b fallback rate; KC2 top-10 action disagreement; T5 t0 formula'})
    cal_path=K.OUT/'calibration.json'
    if cal_path.exists():
        existing=json.loads(cal_path.read_text())
        if existing!=calibration:raise FileExistsError(f'KC2 calibration differs from existing immutable artifact: {cal_path}')
    else:
        cal_path.write_text(json.dumps(calibration,indent=2,ensure_ascii=False)+'\n')
    out={'status':'offline_complete','selection_rule':f'min joint switch errors among Dev retrieval coeff MSE <= KC1 Dev ({K.RAW_DEV_KC1:.6f}) × 1.05',
        'kc1_dev_raw_coeff_sq':K.RAW_DEV_KC1,'dev_limit':K.DEV_LIMIT,'candidate_count':len(model_rows),
        'eligible_joint_candidates':len(eligible),'selected':selected,'lambda0_comparator':lambda0,
        'dev_candidate_metrics':dev_records,'final_retrieval_and_joint_gripper':final,'calibration':calibration,
        'posthoc_same_q_heads':posthoc,'kc1_posthoc_reference':kc1ref,
        'key_source':'KC1 standardization (LIBERO-40 mean/std) applied to immutable saved bank__coefficients [4516,3,6]',
        'inference_inputs':'observed state/action history and (iii only) current VLM tokens; no future target, task ID, pi0.5 output, or gripper labels'}
    dest.write_text(json.dumps(out,indent=2,ensure_ascii=False,default=float)+'\n')
    done=K.OUT/'OFFLINE_DONE'
    if not done.exists():done.write_text('offline evaluation complete\n')
    K.log('offline_complete',selected=selected['name'],lambda0=lambda0['name'],dev_coeff=selected['dev_raw_coeff_sq'],
        dev_grip_errors=selected['dev_gripper_errors'])
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=('evaluate',));a=ap.parse_args()
    print(json.dumps(evaluate_all(),ensure_ascii=False))
