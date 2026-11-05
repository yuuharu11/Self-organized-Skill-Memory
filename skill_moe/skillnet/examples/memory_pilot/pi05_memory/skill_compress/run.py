#!/usr/bin/env python3
"""CPU-only compression comparison: variable M6 units plus small per-window parameters."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN=_PI05_ROOT / 'run-01'
MEMORIES=_PI05_ROOT / 'skill-memories-01'
GAP=_PI05_ROOT / 'memory-gap-01'
TOPK=_PI05_ROOT / 'topk-long-01/production'
RETRIEVAL=_PI05_ROOT / 'retrieval-4b-01'
OUT=_PI05_ROOT / 'skill-compress-01'
LONG_TASKS=set(range(10))
KAPPAS=('k0p05','k0p1','k0p2')
TIME_GRID=tuple((s,d) for s in (.65,.8,1.,1.25,1.5) for d in (-2.,-1.,0.,1.,2.))
RIDGE_ALPHA=10.
PCA_DIM=64


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(mod);return mod


def timewarp(action,speed,offset):
    x=np.asarray(action,np.float32)
    if x.shape[-2:]!=(10,7):raise ValueError(f'expected [...,10,7], got {x.shape}')
    t=np.arange(10,dtype=np.float32);src=(t-4.5)*float(speed)+4.5+float(offset)
    flat=x.reshape(-1,10,7);out=np.empty_like(flat)
    for i,row in enumerate(flat):
        for d in range(7):out[i,:,d]=np.interp(src,t,row[:,d])
    return out.reshape(x.shape)


def fit_template(target,template,mode='timewarp'):
    """Small fixed grid for speed/start offset, optionally fit seven positive channel scales."""
    y=np.asarray(target,np.float32);base=np.asarray(template,np.float32)
    if y.shape!=(10,7) or base.shape!=(10,7):raise ValueError('template and target must be [10,7]')
    use_scale=mode in ('axis','rotation')
    best=(float('inf'),None,None)
    for speed,offset in TIME_GRID:
        warped=timewarp(base,speed,offset)
        if use_scale:
            den=np.sum(warped*warped,axis=0)
            scale=np.divide(np.sum(warped*y,axis=0),den,out=np.ones(7,np.float32),where=den>1e-10)
            scale=np.clip(scale,0.,3.)
            pred=warped*scale
        else:
            scale=np.ones(7,np.float32);pred=warped
        err=float(np.mean((pred-y)**2))
        if err<best[0]:best=(err,np.r_[speed,offset,scale if use_scale else np.empty(0)].astype(np.float32),pred.astype(np.float32))
    return best[1],best[2]


def rotvec_matrix(rotvec):
    """SO(3) matrix exponential for LIBERO's xyz axis-angle state channels."""
    v=np.asarray(rotvec,np.float64);theta=float(np.linalg.norm(v))
    x,y,z=v
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    if theta<1e-10:return np.eye(3,dtype=np.float32)
    return (np.eye(3)+np.sin(theta)/theta*skew+(1.-np.cos(theta))/(theta*theta)*(skew@skew)).astype(np.float32)


def normalize_action(action,q01,q99):
    a=np.asarray(action,np.float32)
    return (2.*(a-q01)/(q99-q01+1e-6)-1.).astype(np.float32)


def denormalize_action(action,q01,q99):
    a=np.asarray(action,np.float32)
    return ((a+1.)*.5*(q99-q01+1e-6)+q01).astype(np.float32)


def pose_align_template(template,query_state,representative_state,*,action_q01=None,action_q99=None,state_q01=None,state_q99=None):
    """Rotate representative delta xyz/rot vectors into query EEF orientation coordinates."""
    t=np.asarray(template,np.float32)
    qstate=np.asarray(query_state,np.float32)[:8];rstate=np.asarray(representative_state,np.float32)[:8]
    if state_q01 is not None:
        qstate=denormalize_action(qstate[:7][None,None,:],state_q01[:7],state_q99[:7]).reshape(-1)[:7]
        rstate=denormalize_action(rstate[:7][None,None,:],state_q01[:7],state_q99[:7]).reshape(-1)[:7]
    rq=rotvec_matrix(qstate[3:6]);rr=rotvec_matrix(rstate[3:6]);rotation=rq@rr.T
    if action_q01 is None:
        a=t.copy();a[:,:3]=a[:,:3]@rotation.T;a[:,3:6]=a[:,3:6]@rotation.T
        return a
    physical=denormalize_action(t,action_q01,action_q99)
    physical[:,:3]=physical[:,:3]@rotation.T
    physical[:,3:6]=physical[:,3:6]@rotation.T
    return normalize_action(physical,action_q01,action_q99)


def assign_pending_nearest(labels,actions,centers):
    lab=np.asarray(labels,np.int32).copy();a=np.asarray(actions,np.float32);c=np.asarray(centers,np.float32)
    pending=lab<0
    if pending.any():
        delta=a[pending,None]-c[None]
        lab[pending]=np.argmin(np.mean(delta*delta,axis=(-1,-2)),axis=1).astype(np.int32)
    return lab,pending


def _load_data():
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (RUN/'rows').glob('*.npy')}
    pool=np.flatnonzero(np.asarray(rows['memory'],bool)&~np.isin(np.asarray(rows['task']),list(LONG_TASKS)))
    if len(pool)!=3857 or np.any(np.isin(np.asarray(rows['task'])[pool],list(LONG_TASKS))):
        raise AssertionError('noLong pool must be the 3,857 memory rows from tasks 10..39')
    gap=np.load(GAP/'predictions_and_errors.npz',allow_pickle=False)
    gap_summary=json.loads((GAP/'summary_corrected.json').read_text())
    if not np.isclose(float(gap_summary['threshold']),.01549410422449,atol=1e-10,rtol=0):
        raise AssertionError('fixed memory-gap Dev q80 threshold changed')
    top=np.load(TOPK/'predictions_and_errors.npz',allow_pickle=False)
    q=gap['row_ids'][np.asarray(rows['task'][gap['row_ids']])<10]
    if len(q)!=548 or not np.array_equal(q,top['query_rows']) or int(top['hard_mask'].sum())!=63:
        raise AssertionError('Long Eval or hard split does not match the fixed 548/63 rows')
    if np.intersect1d(pool,q).size:raise AssertionError('Long Eval rows leaked into noLong pool')
    return rows,pool,np.asarray(q,np.int64),np.asarray(top['hard_mask'],bool),top


def _actions(rows,ids):return np.asarray(rows['actions'][ids,:,:7],np.float32)


def _mse(x,y):return np.mean((np.asarray(x,np.float32)-np.asarray(y,np.float32))**2,axis=(-1,-2))


def _nearest_action_units(actions,centers):
    a=np.asarray(actions,np.float32).reshape(len(actions),70);c=np.asarray(centers,np.float32).reshape(len(centers),70)
    cn=np.sum(c*c,axis=1);out=np.empty(len(a),np.int32)
    for s in range(0,len(a),256):
        x=a[s:s+256];d=np.sum(x*x,axis=1)[:,None]+cn[None]-2*x@c.T
        out[s:s+len(x)]=np.argmin(np.maximum(d,0),axis=1)
    return out


def _raw_oracle(targets,mem_actions):
    q=np.asarray(targets,np.float32).reshape(len(targets),70);m=np.asarray(mem_actions,np.float32).reshape(len(mem_actions),70)
    mn=np.sum(m*m,axis=1);out=np.empty(len(q),np.float32)
    for s in range(0,len(q),32):
        x=q[s:s+32];d=(np.sum(x*x,axis=1)[:,None]+mn[None]-2*x@m.T)/70.
        out[s:s+len(x)]=np.min(np.maximum(d,0),axis=1)
    return out


def _fit_row_parameters(actions,units,templates,mode,states=None,rep_states=None,pose_kwargs=None):
    n=len(actions);params=[];recon=np.empty_like(actions)
    for i in range(n):
        u=int(units[i]);template=templates[u]
        if mode=='rep':
            params.append(np.zeros(0,np.float32));recon[i]=template
            continue
        if mode=='rotation':
            template=pose_align_template(template,states[i],rep_states[u],**(pose_kwargs or {}))
        theta,p=fit_template(actions[i],template,mode=mode)
        params.append(theta);recon[i]=p
    dim=max((len(x) for x in params),default=0)
    return (np.stack([np.pad(x,(0,dim-len(x))) for x in params]).astype(np.float32) if dim else np.zeros((n,0),np.float32)),recon


def _decode(template,theta,mode,query_state=None,rep_state=None,pose_kwargs=None):
    t=np.asarray(template,np.float32)
    if mode=='rotation':t=pose_align_template(t,query_state,rep_state,**(pose_kwargs or {}))
    if mode=='rep':return t
    speed=float(np.clip(theta[0],.5,1.5));offset=float(np.clip(theta[1],-2,2))
    p=timewarp(t,speed,offset)
    if mode in ('axis','rotation'):
        scales=np.clip(np.asarray(theta[2:9],np.float32),0.,3.)
        p=p*scales
    return p.astype(np.float32)


def _fit_feature_map(raw_features,train_ix,n_components=PCA_DIM):
    """NoLong-only standardization and randomized PCA for a compact ridge input."""
    x=np.asarray(raw_features,np.float32)
    mean=x[train_ix].mean(0,dtype=np.float64).astype(np.float32)
    std=x[train_ix].std(0,dtype=np.float64).astype(np.float32);std[std<1e-5]=1.
    xs=(x-mean)/std
    train=xs[train_ix]
    d=min(n_components,train.shape[1],len(train)-1)
    rng=np.random.default_rng(20261104);omega=rng.normal(size=(train.shape[1],d+12)).astype(np.float32)
    y=train@omega
    for _ in range(2):y=train@(train.T@y)
    q,_=np.linalg.qr(y,mode='reduced')
    _,_,vt=np.linalg.svd(q.T@train,full_matrices=False)
    components=vt[:d].T.astype(np.float32)
    return (xs@components).astype(np.float32),{'mean':mean,'std':std,'components':components}


def _ridge_fit(features,unit_ids,targets,n_units,alpha=RIDGE_ALPHA):
    z=np.asarray(features,np.float64);u=np.asarray(unit_ids,np.int64);y=np.asarray(targets,np.float64)
    if y.ndim==1:y=y[:,None]
    ymean=y.mean(0);ystd=y.std(0);ystd[ystd<1e-6]=1.
    ys=(y-ymean)/ystd
    one=np.eye(n_units,dtype=np.float64)[u]
    x=np.concatenate([z,one],axis=1);xm=x.mean(0);xc=x-xm;ym=ys.mean(0);yc=ys-ym
    gram=xc.T@xc;gram.flat[::len(gram)+1]+=float(alpha)
    coef=np.linalg.solve(gram,xc.T@yc)
    intercept=ym-xm@coef
    return {'coef':coef.astype(np.float32),'intercept':intercept.astype(np.float32),
            'ymean':ymean.astype(np.float32),'ystd':ystd.astype(np.float32),'feature_dim':z.shape[1],
            'alpha':float(alpha),'weight_values':int(coef.size+intercept.size+len(ymean)+len(ystd))}


def _ridge_predict_all_units(features,model,n_units):
    z=np.asarray(features,np.float32);f=model['feature_dim'];coef=model['coef'];inter=model['intercept']
    raw=z@coef[:f]+inter[None]
    raw=raw[:,None,:]+coef[f:f+n_units][None,:,:]
    return raw*model['ystd'][None,None,:]+model['ymean'][None,None,:]


def _fit_pca_members(actions,units,representatives,n_units,kmax=8):
    bases=[];all_coeff=np.zeros((len(actions),kmax),np.float32);recons={k:np.empty_like(actions) for k in (1,2,4,8)}
    for u in range(n_units):
        ix=np.flatnonzero(units==u)
        residual=(actions[ix]-representatives[u]).reshape(len(ix),70)
        _,_,vt=np.linalg.svd(residual,full_matrices=False)
        comp=np.zeros((kmax,70),np.float32);comp[:min(kmax,len(vt))]=vt[:kmax]
        bases.append(comp.reshape(kmax,10,7))
        co=residual.reshape(len(ix),70)@comp.T
        all_coeff[ix]=co.astype(np.float32)
        for k in (1,2,4,8):
            p=representatives[u].reshape(70)+co[:,:k]@comp[:k]
            recons[k][ix]=p.reshape(-1,10,7)
    return np.stack(bases),all_coeff,recons


def _residual_vq(actions,base_labels,base_reps,pool_tasks,pool_eps,kappa,builder,stages):
    labels=[np.asarray(base_labels,np.int32)];centers=[np.asarray(base_reps,np.float32)]
    recon=centers[0][labels[0]].copy()
    fits=[]
    for s in range(1,stages):
        residual=(actions-recon).reshape(len(actions),70)
        fit=builder.m6_fit(residual,np.arange(len(actions),dtype=np.int64),pool_tasks,pool_eps,
                           builder.RawFunctional(),kappa=kappa,seed=builder.SEED+1100*s)
        c=np.asarray(fit['centers'],np.float32).reshape(-1,10,7)
        if not len(c):raise RuntimeError(f'residual VQ stage {s+1} produced no supported codes')
        lab,pending=assign_pending_nearest(fit['row_units'],residual.reshape(-1,10,7),c)
        labels.append(lab);centers.append(c);fits.append({'pending':pending,'fit':fit})
        recon=recon+c[lab]
    return labels,centers,recon,fits


def _vq_decode_indices(actions,base_unit,stage_centers):
    residual=np.asarray(actions,np.float32)-stage_centers[0][base_unit]
    chosen=[];recon=stage_centers[0][base_unit].copy()
    for centers in stage_centers[1:]:
        ids=_nearest_action_units(residual,centers)
        chosen.append(ids);recon+=centers[ids];residual-=centers[ids]
    return chosen,recon


def _metric_record(row, name, kappa, pending_mode, self_errors, difficulty, oracle_errors,
                   pred_oracle, pred_key, unit_count, pending_fraction, values_assigned,
                   values_average, representatives, model_values, all_mask, hard_mask,
                   controls):
    def bysubset(v):return {'all_long':float(np.mean(v[all_mask])),'hard_long':float(np.mean(v[hard_mask]))}
    row.append({'kappa':kappa,'representation':name,'pending':pending_mode,'unit_count':int(unit_count),
       'pending_fraction':float(pending_fraction),'values_per_assigned_window':float(values_assigned),
       'average_values_per_window':float(values_average),'representative_floats':int(representatives),
       'model_floats':int(model_values),'self_mse_all':float(self_errors.mean()),
       'self_mse_by_difficulty':difficulty,'long_oracle_mse':bysubset(oracle_errors),
       'long_oracle_unit_pred_theta_mse':bysubset(pred_oracle),
       'long_key_unit_pred_theta_mse':bysubset(pred_key),
       'normal_policy_mse':controls['normal'],'zero_t0_02_mse':controls['zero02'],
       'zero_t0_04_mse':controls['zero04']} )


def run(out=OUT,threads=8):
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing output overwrite: {out}')
    import os
    os.environ['OMP_NUM_THREADS']=str(threads);os.environ['MKL_NUM_THREADS']=str(threads);os.environ['OPENBLAS_NUM_THREADS']=str(threads)
    rows,pool,long_ids,hard,top=_load_data()
    builder=load_module('skill_memories_build_for_compress',PI05/'skill_memories'/'build.py')
    actions=_actions(rows,pool);long_actions=_actions(rows,long_ids);n=len(pool)
    task=np.asarray(rows['task'][pool],np.int32);episode=np.asarray(rows['episode'][pool],np.int32)
    assert np.all(task>=10) and not np.intersect1d(pool,long_ids).size
    if actions.shape!=(3857,10,7):raise AssertionError(actions.shape)
    norm_file=next(_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero/assets'.rglob('norm_stats.json'))
    norm=json.loads(norm_file.read_text())['norm_stats'];aq01=np.asarray(norm['actions']['q01'][:7],np.float32);aq99=np.asarray(norm['actions']['q99'][:7],np.float32)
    sq01=np.asarray(norm['state']['q01'][:8],np.float32);sq99=np.asarray(norm['state']['q99'][:8],np.float32)
    pose_kwargs={'action_q01':aq01,'action_q99':aq99,'state_q01':sq01,'state_q99':sq99}
    memfile=np.load(RUN/'rows'/'memory.npy',mmap_mode='r')
    wrist=np.load(RETRIEVAL/'wrist_token_mean.npy',mmap_mode='r')
    rawkey=np.concatenate([np.asarray(wrist[pool],np.float32),np.asarray(rows['state'][pool,:8],np.float32)],axis=1)
    key_train_ix=np.arange(n)
    latent,feat_spec=_fit_feature_map(rawkey,key_train_ix)
    long_rawkey=np.concatenate([np.asarray(wrist[long_ids],np.float32),np.asarray(rows['state'][long_ids,:8],np.float32)],axis=1)
    long_feat=((long_rawkey-feat_spec['mean'])/feat_spec['std'])@feat_spec['components']
    kappa_value={'k0p05':.05,'k0p1':.1,'k0p2':.2}
    input_rows=[];model_summaries={};all_report=[];raw_pending_oracles={}
    # Fixed references from the same row set and completed Long diagnostic.
    controls={'normal':float(np.mean(top['normal_single_mse'])),
      'zero02':float(np.mean(top['zero_mse'][0,0])),'zero04':float(np.mean(top['zero_mse'][1,0]))}
    controls['normal_hard']=float(np.mean(top['normal_single_mse'][hard]));controls['zero02_hard']=float(np.mean(top['zero_mse'][0,0,hard]));controls['zero04_hard']=float(np.mean(top['zero_mse'][1,0,hard]))
    codec_errors=np.load(RUN/'codec_errors.npz',allow_pickle=False)
    own_codec=np.asarray(codec_errors['own'][pool],np.float32)
    controls['b12_self_noLong']=float(own_codec.mean())
    controls['difficulty_thresholds']=np.quantile(np.sqrt(np.mean(actions**2,axis=(1,2))),[.25,.5,.75]).tolist()
    diff=np.sqrt(np.mean(actions**2,axis=(1,2)));thresholds=np.quantile(diff,[.25,.5,.75]);difficult=np.digitize(diff,thresholds)
    if np.any(np.isin(task,list(LONG_TASKS))):raise AssertionError('Long task leaked into compression fit')

    # Normalize and assert fixed benchmark split before any representation-specific fit.
    out.mkdir(parents=True,exist_ok=True)
    protocol={'pool_rows':int(n),'pool_task_ids':sorted(map(int,np.unique(task))),
      'excluded_tasks':sorted(LONG_TASKS),'long_eval_rows':int(len(long_ids)),'long_hard_rows':int(hard.sum()),
      'hard_threshold':.01549410422449,'long_rows_used_in_fit':False,'long_rows_used_in_hparam_selection':False,
      'source_m6':str(MEMORIES),'source_topk':str(TOPK),'source_feature':str(RETRIEVAL/'wrist_token_mean.npy'),
      'input_action_space':'normalized active7, 10x7', 'pending_modes':['nearest_unit','raw_exception'],
      'ridge_alpha_fixed':RIDGE_ALPHA,'feature_transform':'noLong-only standardization + 64D randomized PCA; fit solely on noLong memory rows',
      'time_grid':list(TIME_GRID),'rotation_interpretation':'LIBERO state first 8: xyz, axis-angle rotvec xyz, two gripper qpos; orientation is unnormalized with pi05 quantile stats then converted by Rodrigues. xyz and action rotation-vector deltas rotate by R_query R_rep^T',
      'difficulty_bins':'quartiles of noLong row action RMS magnitude; cutpoints fixed from noLong only',
      'controls':controls,'threads':threads,'expert_learning':False,'gpu_inference':False}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (out/'progress.jsonl').write_text('')
    started=time.monotonic()
    # Unit key representation is directly the saved representative wrist mean + state[:8].
    all_wrist=np.asarray(wrist[pool],np.float32);pool_state=np.asarray(rows['state'][pool,:8],np.float32)
    for kname in KAPPAS:
      kappa=kappa_value[kname];src=MEMORIES/f'raw-m6-noLong-{kname}'
      z=np.load(src/'units.npz',allow_pickle=False);keyz=np.load(src/'keys.npz',allow_pickle=False)
      # Input row order is checked rather than assumed.
      if not np.array_equal(z['pool_row_ids'],pool):raise AssertionError(f'{kname} M6 pool rows are misaligned')
      reps=np.asarray(z['representative_action'],np.float32)[:,:,:7]
      labels_raw=np.asarray(z['pool_unit_ids'],np.int32);n_units=len(reps)
      labels,pending=assign_pending_nearest(labels_raw,actions,reps)
      rep_rows=np.asarray(z['representative_row_ids'],np.int64)
      rep_states=np.asarray(rows['state'][rep_rows,:8],np.float32)
      # Representative keys standardized using only the noLong memory pool.
      fmean=rawkey.mean(0,dtype=np.float64).astype(np.float32);fstd=rawkey.std(0,dtype=np.float64).astype(np.float32);fstd[fstd<1e-5]=1.
      repkey=np.concatenate([np.asarray(keyz['representative_wrist'],np.float32),np.asarray(keyz['representative_state'],np.float32)],axis=1)
      repkey_std=(repkey-fmean)/fstd;querykey_std=(long_rawkey-fmean)/fstd
      key_unit=np.argmin(np.sum((querykey_std[:,None]-repkey_std[None])**2,axis=-1),axis=1).astype(np.int32)
      rowrec=[]

      # Main codebook-backed continuous representations.
      modes=[('representative','rep',0),('timewarp','timewarp',2),('axis_scale','axis',9),('eef_rotation','rotation',9)]
      model_data={}
      for name,mode,theta_dim in modes:
        self_theta,self_recon=_fit_row_parameters(actions,labels,reps,mode,
          states=np.asarray(rows['state'][pool,:8],np.float32) if mode=='rotation' else None,
          rep_states=rep_states if mode=='rotation' else None,pose_kwargs=pose_kwargs)
        # Fit theta predictor on noLong M6 rows only; pending rows use the prescribed nearest-unit assignment.
        reg=None
        if theta_dim:
          reg=_ridge_fit(latent,labels,self_theta,n_units,alpha=RIDGE_ALPHA)
        pred_all=_ridge_predict_all_units(long_feat,reg,n_units) if reg else np.zeros((len(long_ids),n_units,0),np.float32)
        # Oracle unit with predicted theta and retrieval-selected representative unit with predicted theta.
        long_dists=np.empty((len(long_ids),n_units),np.float32)
        pred_oracle_recon=np.empty_like(long_actions);pred_key_recon=np.empty_like(long_actions)
        for qi in range(len(long_ids)):
          y=long_actions[qi];stateq=np.asarray(rows['state'][long_ids[qi],:8],np.float32)
          for u in range(n_units):
            template=reps[u]
            if mode=='rotation':template=pose_align_template(template,stateq,rep_states[u],**pose_kwargs)
            if mode=='rep':
              theta=np.zeros(0,np.float32);pred=template
            else:
              theta,pred=fit_template(y,template,mode=mode)
            long_dists[qi,u]=float(np.mean((pred-y)**2))
          oracle_u=int(np.argmin(long_dists[qi]));key_u=int(key_unit[qi])
          for dest,u in ((pred_oracle_recon,oracle_u),(pred_key_recon,key_u)):
            theta=pred_all[qi,u] if theta_dim else np.zeros(0,np.float32)
            dest[qi]=_decode(reps[u],theta,mode,stateq,rep_states[u],pose_kwargs)
        oracle_mse=long_dists.min(axis=1)
        pred_oracle_mse=_mse(pred_oracle_recon,long_actions)
        pred_key_mse=_mse(pred_key_recon,long_actions)
        self_err=_mse(self_recon,actions)
        difficulty={f'q{i+1}':float(self_err[difficult==i].mean()) for i in range(4)}
        assigned_values=1+theta_dim
        predictor_values=reg['weight_values'] if reg else 0
        model_values=int(predictor_values)
        model_data[name]={'theta':self_theta,'recon':self_recon,'reg':reg,'pred_oracle':pred_oracle_mse,'pred_key':pred_key_mse}
        for pending_mode in ('nearest_unit','raw_exception'):
          e=self_err.copy()
          mode_oracle=oracle_mse.copy()
          if pending_mode=='raw_exception':
            e[pending]=0.
            cache_key=np.packbits(pending).tobytes()
            if cache_key not in raw_pending_oracles:raw_pending_oracles[cache_key]=_raw_oracle(long_actions,actions[pending])
            mode_oracle=np.minimum(mode_oracle,raw_pending_oracles[cache_key])
          avg=float((np.sum(~pending)*assigned_values+np.sum(pending)*70)/n) if pending_mode=='raw_exception' else float(assigned_values)
          rec={'name':name,'mode':mode,'theta_dim':theta_dim,'self_errors':e,
               'oracle':mode_oracle,'pred_oracle':pred_oracle_mse.copy(),'pred_key':pred_key_mse.copy(),
               'pending_fraction':float(pending.mean()),'assigned_values':assigned_values,'average_values':avg,
               'rep_values':int(n_units*70),'model_values':model_values,'pending_mask':pending}
          model_data[name,pending_mode]=rec
          _metric_record(rowrec,name,kname,pending_mode,e,difficulty,rec['oracle'],rec['pred_oracle'],rec['pred_key'],
             n_units,pending.mean(),assigned_values,avg,n_units*70,model_values,np.ones(len(long_ids),bool),hard,controls)
          rowrec[-1]['unit_counts_by_stage']=[int(n_units)]
      # Per-unit PCA residuals. The mean/representative remains a separately counted stored value.
      bases,coeff,recons=_fit_pca_members(actions,labels,reps,n_units,8)
      for pc in (1,2,4,8):
        theta=coeff[:,:pc]
        reg=_ridge_fit(latent,labels,theta,n_units,alpha=RIDGE_ALPHA)
        pred_all=_ridge_predict_all_units(long_feat,reg,n_units)[:,:,:pc]
        oracle_d=np.empty((len(long_ids),n_units),np.float32);oracle_rec=np.empty_like(long_actions);key_rec=np.empty_like(long_actions)
        for qi,y in enumerate(long_actions):
          for u in range(n_units):
            co=(y-reps[u]).reshape(70)@bases[u,:pc].reshape(pc,70).T
            dec=reps[u].reshape(70)+co@bases[u,:pc].reshape(pc,70)
            oracle_d[qi,u]=np.mean((dec.reshape(10,7)-y)**2)
          ou=int(np.argmin(oracle_d[qi]));ku=int(key_unit[qi])
          oracle_rec[qi]=(reps[ou].reshape(70)+pred_all[qi,ou]@bases[ou,:pc].reshape(pc,70)).reshape(10,7)
          key_rec[qi]=(reps[ku].reshape(70)+pred_all[qi,ku]@bases[ku,:pc].reshape(pc,70)).reshape(10,7)
        self_err=_mse(recons[pc],actions);oracle_mse=oracle_d.min(axis=1);po=_mse(oracle_rec,long_actions);pk=_mse(key_rec,long_actions)
        name=f'unit_pca_{pc}';difficulty={f'q{i+1}':float(self_err[difficult==i].mean()) for i in range(4)}
        for pm in ('nearest_unit','raw_exception'):
          e=self_err.copy()
          mode_oracle=oracle_mse.copy()
          if pm=='raw_exception':
            e[pending]=0.
            cache_key=np.packbits(pending).tobytes()
            if cache_key not in raw_pending_oracles:raw_pending_oracles[cache_key]=_raw_oracle(long_actions,actions[pending])
            mode_oracle=np.minimum(mode_oracle,raw_pending_oracles[cache_key])
          avg=(np.sum(~pending)*(1+pc)+np.sum(pending)*70)/n if pm=='raw_exception' else 1+pc
          _metric_record(rowrec,name,kname,pm,e,difficulty,mode_oracle,po,pk,n_units,pending.mean(),1+pc,avg,
             n_units*70,n_units*pc*70+reg['weight_values'],np.ones(len(long_ids),bool),hard,controls)
          rowrec[-1]['unit_counts_by_stage']=[int(n_units)]

      # Residual VQ: M6 codebooks at each residual stage, no preset cluster count.
      vq_counts_by_stage={}
      for stages in (2,3):
        stage_labels,stage_centers,vq_self,stage_fits=_residual_vq(actions,labels,reps,task,episode,kappa,builder,stages)
        first_pending=labels_raw<0
        any_pending=first_pending.copy()
        for f in stage_fits:any_pending|=f['pending']
        # VQ C pathway: ridge-predict each continuous residual from noLong wrist+state and assigned code history,
        # then quantize to the next M6 residual center. This is the predicted-theta analogue for discrete codes.
        vq_models=[];previous=stage_centers[0][labels].copy()
        prev_code_ids=[]
        for si in range(1,stages):
          targets=(actions-previous).reshape(n,70)
          # Condition on the first-stage unit; later stage predictors also see prior residual-code IDs.
          cond_parts=[np.eye(n_units,dtype=np.float32)[labels]]
          for prior_ids,prior_count in prev_code_ids:
            cond_parts.append(np.eye(prior_count,dtype=np.float32)[prior_ids])
          zaug=np.concatenate([latent,*cond_parts],axis=1)
          # Put categorical codes in the feature input; use a single intercept-only ridge via a sentinel unit.
          reg=_ridge_fit(zaug,np.zeros(n,np.int32),targets,1,alpha=RIDGE_ALPHA)
          # Custom direct ridge predictor over zaug (single category column is constant).
          vq_models.append((reg,zaug.shape[1]))
          if si < stages-1:
            c=stage_centers[si]
            ids=stage_labels[si]
            previous+=c[ids];prev_code_ids.append((ids,len(c)))
        # The VQ C estimates use the selected/oracle first-stage unit; predict residual reconstruction iteratively.
        def vq_predict(qfeatures,base_u):
          out=np.empty((len(base_u),10,7),np.float32)
          for qi,u in enumerate(base_u):
            zz=qfeatures[qi:qi+1]
            cond=[np.eye(n_units,dtype=np.float32)[[int(u)]]]
            recon=reps[int(u)].copy();res_pred=None
            for si,(reg,fdim) in enumerate(vq_models):
              zaug=np.concatenate([zz,*cond],axis=1)
              # weight model was fit with one constant dummy unit and accepts the corresponding exact width.
              x=np.concatenate([zaug,np.ones((1,1),np.float32)],axis=1)
              # Ridge representation stores feature weights followed by one dummy unit coefficient.
              pred=(x[:,:reg['feature_dim']]@reg['coef'][:reg['feature_dim']]+reg['intercept']+
                    reg['coef'][reg['feature_dim']:reg['feature_dim']+1])
              residual=pred.reshape(10,7)
              center_ids=_nearest_action_units(residual[None],stage_centers[si+1])[0]
              center=stage_centers[si+1][center_ids]
              recon+=center;res_pred=residual-center
              if si+1<stages-1:cond.append(np.eye(len(stage_centers[si+1]),dtype=np.float32)[[center_ids]])
            out[qi]=recon
          return out
        # Exact per-query oracle: all base codes, greedy residual code assignment from the true remainder.
        vq_oracle=np.empty((len(long_ids),n_units),np.float32);long_pred_oracle=np.empty_like(long_actions);long_pred_key=np.empty_like(long_actions)
        long_oracle_units=np.empty(len(long_ids),np.int32)
        for qi,y in enumerate(long_actions):
          for u in range(n_units):
            _,dec=_vq_decode_indices(y[None],np.asarray([u]),stage_centers)
            vq_oracle[qi,u]=float(np.mean((dec[0]-y)**2))
          ou=int(np.argmin(vq_oracle[qi]));ku=int(key_unit[qi]);long_oracle_units[qi]=ou
          long_pred_oracle[qi]=vq_predict(long_feat[qi:qi+1],np.asarray([ou]))[0]
          long_pred_key[qi]=vq_predict(long_feat[qi:qi+1],np.asarray([ku]))[0]
        oracle_mse=vq_oracle.min(axis=1);po=_mse(long_pred_oracle,long_actions);pk=_mse(long_pred_key,long_actions)
        self_err=_mse(vq_self,actions);name=f'residual_vq_{stages}stage';differr={f'q{i+1}':float(self_err[difficult==i].mean()) for i in range(4)}
        residual_pending=any_pending
        rep_values=int(len(stage_centers[0])*70)
        vq_counts_by_stage[f'{stages}stage']=[int(len(c)) for c in stage_centers]
        # Estimate per-code predictor weights plus the M6 codebook storage; count stored code IDs separately.
        model_values=int(sum(reg['weight_values'] for reg,_ in vq_models))
        for pm in ('nearest_unit','raw_exception'):
          e=self_err.copy()
          mode_oracle=oracle_mse.copy()
          if pm=='raw_exception':
            e[residual_pending]=0.
            cache_key=np.packbits(residual_pending).tobytes()
            if cache_key not in raw_pending_oracles:raw_pending_oracles[cache_key]=_raw_oracle(long_actions,actions[residual_pending])
            mode_oracle=np.minimum(mode_oracle,raw_pending_oracles[cache_key])
          pending_fraction=float(residual_pending.mean())
          assigned_values=float(stages)
          avg=(np.sum(~residual_pending)*stages+np.sum(residual_pending)*70)/n if pm=='raw_exception' else stages
          _metric_record(rowrec,name,kname,pm,e,differr,mode_oracle,po,pk,n_units,pending_fraction,
             assigned_values,avg,rep_values,model_values+int(sum(len(c)*70 for c in stage_centers[1:])),np.ones(len(long_ids),bool),hard,controls)
          rowrec[-1]['unit_counts_by_stage']=[int(len(c)) for c in stage_centers]
      model_summaries[kname]={'unit_count':n_units,'pending_rows':int(pending.sum()),'epsilon':float(json.loads((src/'manifest.json').read_text())['epsilon']),
         'unit_sizes':z['unit_member_count'].tolist(),'residual_vq_unit_counts':vq_counts_by_stage}
      all_report.extend(rowrec)
      with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'kappa':kname,'done':True,'rows':n,'units':n_units,'elapsed_s':time.monotonic()-started})+'\n')

    # Uncompressed row memory and b12 self-reconstruction are references, not learned methods.
    raw_oracle=_raw_oracle(long_actions,actions)
    # Exact action-space oracle; this differs from the prior report's `nearest`
    # statistic because its raw_diagnostics helper reports partition[:, 0],
    # not the minimum across all candidates.
    if abs(float(raw_oracle[hard].mean())-.03512223)>.003:
      raise AssertionError(f'raw noLong action-oracle hard baseline changed: {float(raw_oracle[hard].mean())}')
    primary_rep=next(r for r in all_report if r['kappa']=='k0p1' and r['representation']=='representative' and r['pending']=='nearest_unit')
    if abs(primary_rep['long_oracle_mse']['hard_long']-.08957296)>.01:
      raise AssertionError('raw M6 kappa .1 exhaustive representative oracle changed: '
                           f'{primary_rep["long_oracle_mse"]["hard_long"]}')
    mem_summary=json.loads((MEMORIES/'summary.json').read_text())
    raw_key_ref=mem_summary['memories']['raw-rows-noLong']
    references={'raw_noLong_action_oracle':{'all_long':float(raw_oracle.mean()),'hard_long':float(raw_oracle[hard].mean())},
      'prior_report_raw_nearest_statistic':{'all_long':raw_key_ref['long_eval_nearest_mean_mse'],'hard_long':raw_key_ref['long_hard_nearest_mean_mse']},
      'b12_codec_self_noLong_mse':controls['b12_self_noLong'],
      'normal_pi05_single_long_mse':{'all_long':controls['normal'],'hard_long':controls['normal_hard']},
      'zero_draft_t0_0.2_long_mse':{'all_long':controls['zero02'],'hard_long':controls['zero02_hard']},
      'zero_draft_t0_0.4_long_mse':{'all_long':controls['zero04'],'hard_long':controls['zero04_hard']}}
    all_report.append({'kappa':'reference','representation':'raw_uncompressed','pending':'none','unit_count':int(n),
      'unit_counts_by_stage':[int(n)],'pending_fraction':0.,'values_per_assigned_window':70.,'average_values_per_window':70.,
      'representative_floats':0,'model_floats':0,'self_mse_all':0.,'self_mse_by_difficulty':{f'q{i+1}':0. for i in range(4)},
      'long_oracle_mse':{'all_long':float(raw_oracle.mean()),'hard_long':float(raw_oracle[hard].mean())},
      'long_oracle_unit_pred_theta_mse':{'all_long':None,'hard_long':None},
      'long_key_unit_pred_theta_mse':{'all_long':None,'hard_long':None},
      'normal_policy_mse':controls['normal'],'zero_t0_02_mse':controls['zero02'],'zero_t0_04_mse':controls['zero04']})
    all_report.append({'kappa':'reference','representation':'b12_codec_self_reference','pending':'n/a','unit_count':0,
      'unit_counts_by_stage':[],'pending_fraction':0.,'values_per_assigned_window':16.,'average_values_per_window':16.,
      'representative_floats':0,'model_floats':0,'self_mse_all':controls['b12_self_noLong'],
      'self_mse_by_difficulty':{},'long_oracle_mse':{'all_long':None,'hard_long':None},
      'long_oracle_unit_pred_theta_mse':{'all_long':None,'hard_long':None},
      'long_key_unit_pred_theta_mse':{'all_long':None,'hard_long':None},
      'normal_policy_mse':controls['normal'],'zero_t0_02_mse':controls['zero02'],'zero_t0_04_mse':controls['zero04']})
    # CSV has one row per representation and pending policy; JSON retains difficulty-bin detail.
    import csv
    csvfile=out/'comparison.csv'
    fields=['kappa','representation','pending','unit_count','unit_counts_by_stage','pending_fraction','values_per_assigned_window','average_values_per_window','representative_floats','model_floats','self_mse_all','oracle_hard','oracle_all','pred_theta_oracle_hard','pred_theta_key_hard','pred_theta_oracle_all','pred_theta_key_all']
    with csvfile.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
      for r in all_report:
        w.writerow({'kappa':r['kappa'],'representation':r['representation'],'pending':r['pending'],'unit_count':r['unit_count'],'unit_counts_by_stage':r.get('unit_counts_by_stage',[]),
          'pending_fraction':r['pending_fraction'],'values_per_assigned_window':r['values_per_assigned_window'],'average_values_per_window':r['average_values_per_window'],
          'representative_floats':r['representative_floats'],'model_floats':r['model_floats'],'self_mse_all':r['self_mse_all'],
          'oracle_hard':r['long_oracle_mse']['hard_long'],'oracle_all':r['long_oracle_mse']['all_long'],
          'pred_theta_oracle_hard':r['long_oracle_unit_pred_theta_mse']['hard_long'],'pred_theta_key_hard':r['long_key_unit_pred_theta_mse']['hard_long'],
          'pred_theta_oracle_all':r['long_oracle_unit_pred_theta_mse']['all_long'],'pred_theta_key_all':r['long_key_unit_pred_theta_mse']['all_long']})
    np.savez_compressed(out/'long_predictions.npz',query_rows=long_ids,hard_mask=hard,truth=long_actions,
      raw_noLong_oracle_mse=raw_oracle,normal_single_mse=top['normal_single_mse'],zero02_seed4101_mse=top['zero_mse'][0,0],zero04_seed4101_mse=top['zero_mse'][1,0])
    summary={'protocol':protocol,'models':model_summaries,'references':references,'rows':all_report,
             'seconds':time.monotonic()-started,'output':str(out),'cpu_only':True}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--threads',type=int,default=8)
    a=ap.parse_args()
    if a.threads<1 or a.threads>8:raise ValueError('CPU thread limit must be 1..8')
    result=run(a.output,a.threads)
    print(json.dumps({'seconds':result['seconds'],'rows':len(result['rows']),'output':result['output']},ensure_ascii=False))


if __name__=='__main__':main()
