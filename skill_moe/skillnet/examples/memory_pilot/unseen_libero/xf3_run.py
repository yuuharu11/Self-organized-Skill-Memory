#!/usr/bin/env python3
"""XF3: transition-focused gripper decoders over fixed KC2 and VLM features."""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kc2_run as K2
import kc2_eval as E2
import kc1_run as K1

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'xf3-01'
KC2_OUT=ROOT/'kc2-01'
KC1_OUT=ROOT/'kc1-01'
SPLITS=('eh','dev2','learned','unseen20')
INPUTS=('G-q','G-qv','G-v')
LOSSES=('L-bce','L-w5','L-w20','L-haz')
SEEDS=(1,2,3)
EPOCHS=16
BATCH=256
THREADS=int(os.environ.get('OMP_NUM_THREADS','8'))


def log(event,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    row={'utc':utc,'jst':jst,'event':event,**kw}
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,default=float)+'\n')
    print(json.dumps(row,ensure_ascii=False,default=float),flush=True)


def close_sign(actions):
    """LIBERO action convention: positive gripper action means closed (+1)."""
    a=np.asarray(actions)
    if a.ndim!=3 or a.shape[1]<1 or a.shape[2]<7:raise ValueError(f'bad action shape {a.shape}')
    return np.where(a[...,6]>0,1.,-1.).astype(np.float32)


def switch_class(target_sign,last_sign):
    """First future sign change from last executed sign, 10 means no change."""
    y=np.asarray(target_sign,np.float32);g=np.asarray(last_sign,np.float32).reshape(-1)
    if y.shape!=(len(g),10):raise ValueError(f'action labels must be [N,10], got {y.shape}')
    d=y!=np.where(g>=0,1.,-1.)[:,None]
    return np.where(d.any(1),d.argmax(1),10).astype(np.int64)


def transition_metrics(decoded,target_sign,last_sign,valid,switch_at=None):
    """Timing-aware ±2 metrics plus the legacy switch-row and exact-step measures."""
    import kq2cl_common as C
    pred=np.asarray(decoded,np.float32);target=np.asarray(target_sign,np.float32)
    last=np.asarray(last_sign,np.float32).reshape(-1);ok=np.asarray(valid,bool)&(last!=0)
    gtclass=switch_class(target,last);pred_diff=pred!=np.where(last>=0,1.,-1.)[:,None]
    pk=np.where(pred_diff.any(1),pred_diff.argmax(1),10)
    gt=ok&(gtclass<10);pr=ok&(pk<10)
    match=gt&pr&(np.abs(pk-gtclass)<=2)
    tp=int(match.sum());fp=int(pr.sum())-tp;fn=int(gt.sum())-tp
    prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1)
    f1=2*prec*rec/max(prec+rec,1e-12)
    timing=np.abs(pk[match]-gtclass[match])
    legacy=C.switch_metrics(pred,target,last,ok)
    return {'rows':len(pred),'valid_rows':int(ok.sum()),'gt_switch_rows':int(gt.sum()),
        'pred_switch_rows':int(pr.sum()),'tp_within2':tp,'fp_within2':fp,'fn_within2':fn,
        'recall_within2':float(rec),'precision_within2':float(prec),'f1_within2':float(f1),
        'timing_mae_steps':float(timing.mean()) if len(timing) else None,
        'legacy_missed_plus_false':int(legacy['missed_rows']+legacy['false_switch_rows']),
        'legacy_missed_rows':int(legacy['missed_rows']),'legacy_false_switch_rows':int(legacy['false_switch_rows']),
        'exact_step_accuracy':float((pred==target).mean()),'legacy':legacy}


class XF3Decoder(nn.Module):
    """Inputs are query/gripper context and optional VLM tokens; never labels."""
    def __init__(self,input_mode:str,output_mode:str):
        super().__init__()
        if input_mode not in INPUTS:raise ValueError(input_mode)
        if output_mode not in ('bce','hazard'):raise ValueError(output_mode)
        self.input_mode=input_mode;self.output_mode=output_mode
        self.q_norm=nn.LayerNorm(256)
        self.g_norm=nn.LayerNorm(4)
        if input_mode in ('G-qv','G-v'):
            self.vlm_proj=nn.Linear(2048,256)
            self.attn=nn.MultiheadAttention(256,8,batch_first=True)
            self.attn_norm=nn.LayerNorm(256)
            self.pool_query=nn.Parameter(torch.zeros(1,1,256)) if input_mode=='G-v' else None
        else:
            self.vlm_proj=None;self.attn=None;self.attn_norm=None;self.register_parameter('pool_query',None)
        repdim={'G-q':260,'G-qv':516,'G-v':260}[input_mode]
        self.body=nn.Sequential(nn.Linear(repdim,256),nn.GELU(),nn.LayerNorm(256),
                                nn.Linear(256,256),nn.GELU())
        self.output=nn.Linear(256,10 if output_mode=='bce' else 11)

    def forward(self,q,grip_extra,vlm=None):
        """q:[B,256], grip_extra:[B,4], optional VLM:[B,34,2048]."""
        g=self.g_norm(grip_extra.float())
        if self.input_mode=='G-q':
            if q is None or vlm is not None:raise ValueError('G-q requires q and excludes VLM tokens')
            rep=torch.cat((self.q_norm(q.float()),g),-1)
        elif self.input_mode=='G-qv':
            if q is None or vlm is None:raise ValueError('G-qv requires q and VLM tokens')
            qn=self.q_norm(q.float()).unsqueeze(1)
            mem=self.vlm_proj(vlm.float())
            att,_=self.attn(qn,mem,mem,need_weights=False)
            pooled=self.attn_norm(qn+att).squeeze(1)
            rep=torch.cat((self.q_norm(q.float()),pooled,g),-1)
        else:
            if q is not None or vlm is None:raise ValueError('G-v requires VLM tokens and excludes q')
            mem=self.vlm_proj(vlm.float())
            query=self.pool_query.expand(len(mem),-1,-1)
            att,_=self.attn(query,mem,mem,need_weights=False)
            pooled=self.attn_norm(query+att).squeeze(1)
            rep=torch.cat((pooled,g),-1)
        return self.output(self.body(rep))


def _np(path):return np.load(path,mmap_mode='r')


def prepare_features():
    """Extract the fixed KC2-i query and aligned KC2 VLM/gripper features."""
    OUT.mkdir(parents=True,exist_ok=True)
    feature_dir=OUT/'data';feature_dir.mkdir(exist_ok=True)
    done=feature_dir/'FEATURES_DONE'
    if done.exists():return json.loads((feature_dir/'features.json').read_text())
    kr,ctx,_=K2.load_ctx()
    offline=json.loads((KC2_OUT/'offline_results.json').read_text())
    selected=offline['selected']
    if selected['q_name']!='i':raise AssertionError('KC2 selection must be the fixed GRU-only q=i checkpoint')
    model,_=E2.model_from(selected,'cpu');model.eval();torch.set_num_threads(THREADS)
    meta={'source_query_checkpoint':selected['checkpoint'],'q_name':'i','q_dim':256,
          'vlm_source':'KC2 data/*.vlm.npy materialized by kq2_data.build_vlm; 32 pooled image + 2 prompt tokens',
          'rows':{},'inference_inputs':'q, grip_extra, optional VLM; no target/actions/task ID/pi05'}
    for split in SPLITS:
        d=KC2_OUT/'data'/split;vd=_np(d/'vlm.npy')
        sp=ctx.splits[split]
        if not (d/'VLM_DONE').exists():raise FileNotFoundError(f'{split}: VLM_DONE missing')
        if vd.shape!=(len(sp['key']),34,2048):raise AssertionError(f'{split} VLM shape/order: {vd.shape}')
        if not np.array_equal(np.load(d/'key.npy'),sp['key']):raise AssertionError(f'{split} VLM rows are misaligned')
        gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),False)
        qrows=[]
        with torch.no_grad():
            for st in range(0,gs.n,128):
                ix=torch.arange(st,min(st+128,gs.n));qrows.append(model(gs,ix).float().cpu().numpy())
        q=np.concatenate(qrows).astype(np.float32)
        gx,last,valid,target=E2.split_arrays(sp,split)
        labels=close_sign(sp['actions']);classes=switch_class(labels,last)
        if len(q)!=len(vd) or len(gx)!=len(q):raise AssertionError(f'{split}: feature row counts differ')
        cache={'q':q,'grip_extra':np.asarray(gx,np.float32),'last_sign':np.where(np.asarray(gx)[:,0]>=0,1.,-1.).astype(np.float32),
            'valid':np.asarray(valid,bool),'target_sign':labels,'target_close':(labels>0).astype(np.float32),'switch_class':classes,
            'key':np.asarray(sp['key'])}
        sd=feature_dir/split;sd.mkdir(exist_ok=True)
        for name,arr in cache.items():
            p=sd/f'{name}.npy'
            if p.exists():raise FileExistsError(p)
            np.save(p,arr)
        meta['rows'][split]={'rows':len(q),'vlm_tokens':int(vd.shape[1]),'gt_switch_rows':int(((classes<10)&np.asarray(valid,bool)).sum()),
            'valid_history_rows':int(np.asarray(valid,bool).sum()),'source_key_count':len(np.load(d/'key.npy'))}
        log('features_split_complete',split=split,**meta['rows'][split])
    if meta['rows']['dev2']['gt_switch_rows']!=54:raise AssertionError('Dev v2 switch-row total must match KC2 (54)')
    meta['rows']['unseen20']['hz']='20Hz'
    (feature_dir/'features.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    done.write_text('done\n')
    return meta


def load_features(split):
    d=OUT/'data'/split
    return {n:_np(d/f'{n}.npy') for n in ('q','grip_extra','last_sign','valid','target_sign','target_close','switch_class','key')}|{'vlm':_np(KC2_OUT/'data'/split/'vlm.npy')}


def model_input(model,features,idx,device='cpu'):
    ix=np.asarray(idx,np.int64)
    q=None if model.input_mode=='G-v' else torch.as_tensor(np.asarray(features['q'][ix]),device=device,dtype=torch.float32)
    gx=torch.as_tensor(np.asarray(features['grip_extra'][ix]),device=device,dtype=torch.float32)
    vlm=None if model.input_mode=='G-q' else torch.as_tensor(np.asarray(features['vlm'][ix]),device=device,dtype=torch.float32)
    return q,gx,vlm


def predict_logits(model,features,batch=128):
    model.eval();all_logits=[]
    with torch.no_grad():
        for st in range(0,len(features['q']),batch):
            ix=np.arange(st,min(st+batch,len(features['q'])))
            all_logits.append(model(*model_input(model,features,ix)))
    return torch.cat(all_logits,0).cpu().numpy()


def hazard_decode(prob,last_sign,th):
    p=np.asarray(prob,np.float64)
    best=np.argmax(p,axis=1);switch_conf=p[:,:10].sum(1)
    cls=np.where(switch_conf>float(th),best,10)
    cls=np.where(cls>=10,10,cls)
    last=np.where(np.asarray(last_sign)>=0,1.,-1.)
    out=np.repeat(last[:,None],10,axis=1)
    for i,k in enumerate(cls):
        if k<10:out[i,int(k):]=-last[i]
    return out.astype(np.float32),cls.astype(np.int64)


def decode_logits(logits,loss_name,last_sign,threshold):
    if loss_name=='L-haz':return hazard_decode(torch.softmax(torch.as_tensor(logits),-1).numpy(),last_sign,threshold)
    p=1/(1+np.exp(-np.clip(np.asarray(logits,np.float64),-40,40)))
    import kq2cl_common as C
    decoded=C.decode_gripper(p,np.asarray(last_sign),float(threshold))
    return decoded,np.where(np.asarray(decoded)!=np.where(np.asarray(last_sign)[:,None]>=0,1.,-1.),
        np.arange(10)[None],10).min(1)


def metrics_for_logits(logits,features,loss_name,threshold):
    decoded,_=decode_logits(logits,loss_name,features['last_sign'],threshold)
    return transition_metrics(decoded,features['target_sign'],features['last_sign'],features['valid'])


def threshold_grid(loss_name):
    if loss_name=='L-haz':return tuple(np.arange(0.,1.,.05))
    return tuple(np.arange(.05,1.,.05))


def choose_threshold(logits,features,loss_name):
    options=[]
    for th in threshold_grid(loss_name):
        metric=metrics_for_logits(logits,features,loss_name,float(th))
        options.append((float(th),metric))
    th,metric=min(options,key=lambda x:(-x[1]['f1_within2'],-x[1]['recall_within2'],
        -x[1]['precision_within2'],x[1]['timing_mae_steps'] if x[1]['timing_mae_steps'] is not None else 99,
        x[1]['legacy_missed_plus_false'],-x[0]))
    return th,metric,{str(t):m for t,m in options}


def _state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _weighted_switch_rows(order,classes):
    sw=np.asarray(classes)[order]<10
    return np.repeat(order,np.where(sw,4,1))


def _step_weights(classes,nsteps=10,focus_weight=1.):
    w=np.ones((len(classes),nsteps),np.float32)
    for i,k in enumerate(np.asarray(classes)):
        if k<10:w[i,max(0,int(k)-2):min(nsteps,int(k)+3)]=float(focus_weight)
    return w


def _hazard_class_weights(classes):
    count=np.bincount(np.asarray(classes,np.int64),minlength=11).astype(np.float32)
    present=count>0;weights=np.zeros(11,np.float32);weights[present]=1/np.sqrt(count[present])
    weights[present]/=weights[present].mean()
    return weights,count.astype(int).tolist()


def _metric_key(metric):
    return (-metric['f1_within2'],-metric['recall_within2'],-metric['precision_within2'],
            metric['timing_mae_steps'] if metric['timing_mae_steps'] is not None else 99,
            metric['legacy_missed_plus_false'])


def config_name(mode,loss,seed):return f'{mode}_{loss}_seed{seed}'


def train_one(mode,loss_name,seed,*,epochs=EPOCHS):
    torch.set_num_threads(THREADS)
    if loss_name not in LOSSES or mode not in INPUTS:raise ValueError((mode,loss_name))
    d=OUT/'models'/mode;d.mkdir(parents=True,exist_ok=True)
    name=config_name(mode,loss_name,seed);cp=d/f'{name}.pt';jp=d/f'{name}.json'
    if cp.exists() and jp.exists():return json.loads(jp.read_text())
    if cp.exists() or jp.exists():raise FileExistsError(f'incomplete output {name}')
    torch.manual_seed(int(seed)*1009+INPUTS.index(mode)*101+LOSSES.index(loss_name))
    rng=np.random.default_rng(int(seed)*1009+INPUTS.index(mode)*101+LOSSES.index(loss_name))
    tr=load_features('eh');dev=load_features('dev2')
    output='hazard' if loss_name=='L-haz' else 'bce';model=XF3Decoder(mode,output).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    all_target=torch.as_tensor(np.asarray(tr['target_close'],np.float32));classes=np.asarray(tr['switch_class'],np.int64)
    class_weight,class_freq=_hazard_class_weights(classes)
    class_weight_t=torch.as_tensor(class_weight,dtype=torch.float32)
    step_weights=_step_weights(classes,focus_weight=5 if loss_name=='L-w5' else 20 if loss_name=='L-w20' else 1)
    best=None;history=[];started=time.time()
    log('model_start',name=name,mode=mode,loss=loss_name,seed=seed,train_rows=len(classes),epochs=epochs,
        train_switch_rows=int((classes<10).sum()),hazard_class_freq=class_freq if loss_name=='L-haz' else None)
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(classes))
        if loss_name in ('L-w5','L-w20'):order=_weighted_switch_rows(order,classes)
        epochloss=[]
        for st in range(0,len(order),BATCH):
            ix=order[st:st+BATCH];q,gx,vlm=model_input(model,tr,ix)
            logits=model(q,gx,vlm)
            if loss_name=='L-haz':
                cls=torch.as_tensor(classes[ix],dtype=torch.long)
                loss=F.cross_entropy(logits,cls,weight=class_weight_t)
            else:
                y=all_target[ix]
                bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
                if loss_name in ('L-w5','L-w20'):
                    weights=torch.as_tensor(step_weights[ix],dtype=torch.float32)
                    loss=(bce*weights).mean()
                else:loss=bce.mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            epochloss.append(float(loss.detach()))
        devlogits=predict_logits(model,dev);th,metric,thresholds=choose_threshold(devlogits,dev,loss_name)
        rec={'epoch':epoch,'loss':float(np.mean(epochloss)),'dev_threshold':th,
             'dev_f1_within2':metric['f1_within2'],'dev_recall_within2':metric['recall_within2'],
             'dev_precision_within2':metric['precision_within2'],'dev_timing_mae_steps':metric['timing_mae_steps'],
             'dev_gt_switch_rows':metric['gt_switch_rows'],'dev_pred_switch_rows':metric['pred_switch_rows'],
             'dev_legacy_errors':metric['legacy_missed_plus_false'],'dev_exact_step_accuracy':metric['exact_step_accuracy'],
             'seconds':time.time()-started}
        history.append(rec)
        key=_metric_key(metric)
        if best is None or key<best['key']:
            best={'key':key,'state':_state_cpu(model),'rec':rec,'metric':metric,'thresholds':thresholds}
        if epoch in (1,4,8,12,16) or epoch==epochs:log('train_epoch',name=name,**rec)
    torch.save({'format':'kc2_gripper_decoder_v1','state_dict':best['state'],
        'config':{'input_mode':mode,'loss':loss_name,'seed':seed,'output_mode':output,'epochs':epochs,
            'q_checkpoint':json.loads((KC2_OUT/'offline_results.json').read_text())['selected']['checkpoint'],
            'vlm_shape':[34,2048],'attention_dim':256,'attention_heads':8,'fixed_q':True},
        'selected_epoch':best['rec']['epoch'],'threshold':best['rec']['dev_threshold']},cp)
    result={'name':name,'input_mode':mode,'loss':loss_name,'seed':seed,'checkpoint':str(cp),
        'selected_epoch':best['rec']['epoch'],'threshold':best['rec']['dev_threshold'],'dev':best['metric'],
        'dev_threshold_grid':best['thresholds'],'history':history,'train_seconds':time.time()-started,
        'hazard_class_freq_train':class_freq if loss_name=='L-haz' else None}
    jp.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    log('model_complete',name=name,selected_epoch=result['selected_epoch'],dev_f1=best['metric']['f1_within2'],
        dev_recall=best['metric']['recall_within2'],dev_precision=best['metric']['precision_within2'],threshold=result['threshold'])
    return result


def _load_model(row):
    ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    m=XF3Decoder(ck['config']['input_mode'],ck['config']['output_mode'])
    m.load_state_dict(ck['state_dict']);m.eval();return m,ck


def _eval_row(row):
    m,ck=_load_model(row);loss=row['loss'];threshold=float(row['threshold']);out={}
    for split in ('dev2','learned','unseen20'):
        feat=load_features(split);logits=predict_logits(m,feat);out[split]=metrics_for_logits(logits,feat,loss,threshold)
        if loss=='L-haz':out[split+'_argmax']=metrics_for_logits(logits,feat,loss,0.)
    return out


def _kc2_joint_references():
    off=json.loads((KC2_OUT/'offline_results.json').read_text());res={}
    for split in ('dev2','learned','unseen20'):
        f=load_features(split);r=off['final_retrieval_and_joint_gripper']['joint']['splits'][split]
        res[split]=transition_metrics(np.asarray(r['decoded']),f['target_sign'],f['last_sign'],f['valid'])
    return res


def _kc1_reference():
    import kc1_head as H
    old=K1.OUT;K1.OUT=KC1_OUT
    try:
        kr,ctx,_=K1._prepare_context();bank=K1.memory_bank()
        meta=json.loads((KC1_OUT/'offline_corrected.json').read_text())['selected']
        blob=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False);cfg=blob['cfg']
        qmodel=K1._load_model(kr,meta['query'],'cpu',meta['seed']);qmodel.load_state_dict(blob['query']);qmodel.eval()
        kn,_,_=K1.coefficient_keys(np.asarray(bank['coefficients']),standardize=meta['standardize'],
            mean=np.asarray(cfg['mean']),std=np.asarray(cfg['std']))
        keys=torch.as_tensor(kn,dtype=torch.float32)
        hb=torch.load(KC1_OUT/'heads/bundle.pt',map_location='cpu',weights_only=False)
        head=H.KC1Head(int(hb['input_dim']));head.load_state_dict(hb['state_dict']);head.eval()
        result={}
        for split in ('dev2','learned','unseen20'):
            sp=ctx.splits[split];gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),meta['query']=='iii')
            feat=H.retrieval_features(qmodel,gs,sp,keys,np.asarray(bank['task']),np.asarray(bank['episode']))
            gx=load_features(split)['grip_extra'];f=load_features(split)
            x=H.head_input(feat,gx,hb['kind'])
            with torch.no_grad():prob=torch.sigmoid(head(torch.as_tensor(x,dtype=torch.float32))).numpy()
            import kq2cl_common as C
            dec=C.decode_gripper(prob,f['last_sign'],float(hb['threshold']))
            result[split]=transition_metrics(dec,f['target_sign'],f['last_sign'],f['valid'])
        return result
    finally:K1.OUT=old


def _always_hold():
    out={}
    for s in ('dev2','learned','unseen20'):
        f=load_features(s);dec=np.repeat(np.asarray(f['last_sign'])[:,None],10,axis=1)
        out[s]=transition_metrics(dec,f['target_sign'],f['last_sign'],f['valid'])
    return out


def evaluate_all():
    torch.set_num_threads(THREADS)
    dest=OUT/'offline_results.json'
    if dest.exists():raise FileExistsError(dest)
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'models').glob('*/*.json'))]
    if len(rows)!=len(INPUTS)*len(LOSSES)*len(SEEDS):raise AssertionError(f'expected 36 trained models, got {len(rows)}')
    per_model={r['name']:_eval_row(r) for r in rows}
    cells={}
    for mode in INPUTS:
        for loss in LOSSES:
            rr=[r for r in rows if r['input_mode']==mode and r['loss']==loss]
            cells[f'{mode}/{loss}']={'seeds':{},'mean':{},'std':{}}
            for r in rr:cells[f'{mode}/{loss}']['seeds'][str(r['seed'])]=per_model[r['name']]
            for split in ('dev2','unseen20'):
                fields=('recall_within2','precision_within2','f1_within2','timing_mae_steps',
                        'legacy_missed_plus_false','exact_step_accuracy','gt_switch_rows')
                vals={field:[] for field in fields}
                for r in rr:
                    m=per_model[r['name']][split]
                    for field in fields:
                        if m[field] is not None:vals[field].append(float(m[field]))
                cells[f'{mode}/{loss}']['mean'][split]={k:float(np.mean(v)) for k,v in vals.items()}
                cells[f'{mode}/{loss}']['std'][split]={k:float(np.std(v)) for k,v in vals.items()}
            cells[f'{mode}/{loss}']['threshold_by_seed']={str(r['seed']):r['threshold'] for r in rr}
            cells[f'{mode}/{loss}']['selected_epoch_by_seed']={str(r['seed']):r['selected_epoch'] for r in rr}
    # Dev chooses the operating decoder and any threshold; unseen20/learned do not
    # influence model, epoch, weighting, or threshold selection.
    bestrow=min(rows,key=lambda r:_metric_key(per_model[r['name']]['dev2']))
    bestmetrics=per_model[bestrow['name']]['dev2']
    bundlepath=OUT/'bundle.pt'
    if bundlepath.exists():raise FileExistsError(bundlepath)
    ck=torch.load(bestrow['checkpoint'],map_location='cpu',weights_only=False)
    torch.save({'format':'kc2_gripper_decoder_v1','state_dict':ck['state_dict'],
        'config':ck['config']|{'selected_model':bestrow['name']},'input_mode':bestrow['input_mode'],
        'loss':bestrow['loss'],'threshold':bestrow['threshold'],'selected_epoch':bestrow['selected_epoch'],
        'threshold_rule':'Dev v2 maximum ±2-step F1; ties prefer recall, precision, timing MAE, legacy errors',
        'input_contract':['fixed KC2 q=i 256-D' if bestrow['input_mode']!='G-v' else 'no q',
            'grip_extra: last executed sign, has-history, qpos[6:8]',
            'current-time VLM [34,2048]' if bestrow['input_mode']!='G-q' else 'no VLM'],
        'q_checkpoint':ck['config']['q_checkpoint'],'decoder_checkpoint':bestrow['checkpoint']},bundlepath)
    refs={'KC1_posthoc':_kc1_reference(),'KC2_joint':_kc2_joint_references(),'always_hold':_always_hold()}
    result={'status':'offline_complete','selection':'Dev v2 ±2-step F1; ties by recall, precision, timing error, legacy errors; unseen20/learned are report-only',
        'source':{'KC2_query':'KC2 selected q=i fixed 256D checkpoint','VLM':'KC2 data/*.vlm.npy built by kq2_data.build_vlm, 20Hz rows for dev2/unseen20',
            'training_split':'LIBERO-40 eh rows only','inference_inputs':'q, grip_extra, optional current VLM; no labels/future/task ID/pi05'},
        'grid':{'inputs':list(INPUTS),'losses':list(LOSSES),'weighted_bce_focus_weights':[5,20],
            'switch_row_sampling_multiplier':4,'seeds':list(SEEDS),'epochs_max':EPOCHS,'train_rows':int(len(load_features('eh')['q']))},
        'split_rows':{s:{'rows':len(load_features(s)['q']),'gt_switch_rows':int(((np.asarray(load_features(s)['switch_class'])<10)&np.asarray(load_features(s)['valid'],bool)).sum())} for s in ('dev2','learned','unseen20')},
        'cells':cells,'per_model':per_model,'best_decoder':{'name':bestrow['name'],'input_mode':bestrow['input_mode'],'loss':bestrow['loss'],
            'seed':bestrow['seed'],'selected_epoch':bestrow['selected_epoch'],'threshold':bestrow['threshold'],'dev':bestmetrics,
            'bundle':str(bundlepath)},'references':refs}
    result['split_rows']['dev2']['gt_switch_rows']==54 or (_ for _ in ()).throw(AssertionError('Dev switch total changed'))
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    done=OUT/'OFFLINE_DONE'
    if not done.exists():done.write_text('offline evaluation complete\n')
    log('offline_complete',best=bestrow['name'],dev_f1=bestmetrics['f1_within2'],dev_recall=bestmetrics['recall_within2'],
        dev_precision=bestmetrics['precision_within2'],threshold=bestrow['threshold'])
    return result


def smoke():
    torch.set_num_threads(THREADS);rng=np.random.default_rng(3);res=[]
    q=torch.randn(3,256);gx=torch.randn(3,4);vlm=torch.randn(3,34,2048)
    for mode in INPUTS:
        for out in ('bce','hazard'):
            m=XF3Decoder(mode,out)
            args=(q,gx,None) if mode=='G-q' else (q,gx,vlm) if mode=='G-qv' else (None,gx,vlm)
            logits=m(*args);loss=logits.square().mean();loss.backward()
            assert logits.shape==(3,10 if out=='bce' else 11)
            res.append({'mode':mode,'output':out,'shape':list(logits.shape)})
    # Single known close switch at step 4 must have the correct position/direction.
    y=np.full((1,10),-1.,np.float32);y[:,4:]=1.
    assert switch_class(y,np.asarray([-1.])).tolist()==[4]
    hd,cls=hazard_decode(np.eye(11,dtype=np.float32)[[4]],np.asarray([-1.]),0.)
    assert cls.tolist()==[4] and np.array_equal(hd[0,:4],-np.ones(4)) and np.array_equal(hd[0,4:],np.ones(6))
    log('smoke_complete',models=res,label_class=4)
    return {'models':res,'known_switch_class':4}


def run_all():
    prepare_features()
    smoke()
    for mode in INPUTS:
        for loss in LOSSES:
            for seed in SEEDS:train_one(mode,loss,seed)
    return evaluate_all()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=('prepare','smoke','train','evaluate','all'))
    ap.add_argument('--mode',choices=INPUTS);ap.add_argument('--loss',choices=LOSSES);ap.add_argument('--seed',type=int,choices=SEEDS)
    a=ap.parse_args()
    if a.cmd=='prepare':print(json.dumps(prepare_features(),indent=2,ensure_ascii=False))
    elif a.cmd=='smoke':print(json.dumps(smoke(),indent=2,ensure_ascii=False))
    elif a.cmd=='train':
        if not (a.mode and a.loss and a.seed):raise SystemExit('--mode --loss --seed required')
        print(json.dumps(train_one(a.mode,a.loss,a.seed),indent=2,ensure_ascii=False,default=float))
    elif a.cmd=='evaluate':print(json.dumps(evaluate_all(),indent=2,ensure_ascii=False,default=float))
    else:print(json.dumps(run_all(),indent=2,ensure_ascii=False,default=float))

if __name__=='__main__':main()
