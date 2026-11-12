#!/usr/bin/env python3
"""Lightweight observation-to-pose and shape-choice diagnostics for unseen LIBERO."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.35')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
PI05=HERE.parent/'pi05_memory'
ROOT=(STATE_DIR / 'outputs/memory-pilot/unseen-libero')
RUN = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser() / 'run-01'
SOURCE=ROOT/'compressed-denoise-01'/'production'
UNSEEN=ROOT/'oracle-01'
from unseen_paths import UNSEEN_ROWS  # corrected rows (image orientation fix)
OUT=ROOT/'theta-regression-01'
CHECKPOINT = Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser()
SPLITS=('train','dev','eval')
CACHES=('C_vm_none','C_vm_08')
VARIANTS=('R-aa-theta','R-6d-theta','R-6d-act','R-6d-act-nowrist','R-6d-act-wristonly')


def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def _atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=True)+'\n');tmp.replace(path)

def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def grouped_means(hidden,groups):
    """Return means for third-person camera, wrist camera, language in that order."""
    h=np.asarray(hidden,np.float32);g=np.asarray(groups)
    if h.ndim!=2 or g.shape!=(len(h),):raise ValueError('hidden/groups shape mismatch')
    out=[]
    for label in (0,1,2):
        keep=g==label
        out.append(h[keep].mean(0) if np.any(keep) else np.zeros(h.shape[-1],np.float32))
    return np.stack(out)

def rotation6(rotvec):
    from scipy.spatial.transform import Rotation
    r=Rotation.from_rotvec(np.asarray(rotvec,np.float64)).as_matrix()
    return np.concatenate([r[:,0],r[:,1]]).astype(np.float32)

def rot6_matrix(v):
    v=np.asarray(v,np.float64);a=v[:3];b=v[3:]
    x=a/max(np.linalg.norm(a),1e-9);y0=b-x*np.dot(x,b);y=y0/max(np.linalg.norm(y0),1e-9);z=np.cross(x,y)
    return np.stack([x,y,z],axis=1)

def decode_theta(shape,theta,vs):
    from scipy.spatial.transform import Rotation
    z=np.asarray(shape,np.float64);th=np.asarray(theta,np.float64)
    r=Rotation.from_rotvec(th[:3]).as_matrix();scale=np.exp(np.clip(th[3],-5,5))
    phys=np.concatenate([z[:,:3]@r,z[:,3:6]@r],axis=-1)*scale
    return np.concatenate([vs.to_normalized(phys),z[:,6:7]],axis=-1).astype(np.float32)

def _splits():
    base=_module(PI05/'run.py','theta_regression_data');rows=base.load_rows(RUN)
    unseen={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in UNSEEN_ROWS.glob('*.npy')}
    train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    dev=np.flatnonzero(np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    ev=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    if len(train)!=23304 or len(dev)!=113 or len(ev)!=619:raise ValueError('fixed split sizes changed')
    for split,ids in [('train',train),('dev',dev),('eval',ev)]:
        for rep in CACHES:
            z=np.load(SOURCE/'drafts'/f'{rep}_{split}.npz',allow_pickle=False)
            expected=ids
            if not np.array_equal(z['row_ids'],expected):raise ValueError(f'{rep}/{split} cache row IDs misaligned')
    return base,rows,unseen,{'train':(rows,train),'dev':(unseen,dev),'eval':(unseen,ev)}

def extract_features(*,batch=8,threads=8,smoke=False,limit=8):
    """Resumable frozen VLM features; labels/target actions are never read here."""
    import torch;torch.set_num_threads(threads)
    import jax, jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    base,rows,unseen,splits=_splits()
    featdir=OUT/('smoke_features' if smoke else 'features');featdir.mkdir(parents=True,exist_ok=True)
    manifest_path=featdir/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('complete') and bool(old.get('smoke'))==bool(smoke):return old
    model,_=base.load_policy(CHECKPOINT)
    fn=module_jit(model.prefix_features)
    manifest={'status':'running','source':'frozen pi05_libero prefix_features','groups':['base_camera','wrist_camera','language'],
              'train_rows':23304,'dev_rows':113,'eval_rows':619,'target_actions_used':False}
    for split,(source,ids) in splits.items():
        ids=ids[:min(len(ids),limit if smoke else len(ids))]
        dest=featdir/f'{split}.npy';statefile=featdir/f'{split}.progress.json'
        if dest.exists() and statefile.exists() and json.loads(statefile.read_text()).get('complete'):continue
        if dest.exists() and not statefile.exists():raise FileExistsError(f'feature output exists without resume marker: {dest}')
        done=0
        if statefile.exists():done=min(int(json.loads(statefile.read_text()).get('done',0)),len(ids))
        arr=np.lib.format.open_memmap(dest,mode='r+' if dest.exists() else 'w+',dtype='float32',shape=(len(ids),3,2048))
        for st in range(done,len(ids),batch):
            q=ids[st:min(st+batch,len(ids))];obs=base.obs_batch(source,q)
            hidden,groups=fn(obs);hidden=np.asarray(hidden);groups=np.asarray(groups)
            if hidden.shape[-1]!=2048:raise ValueError(f'unexpected feature width {hidden.shape}')
            for k in range(len(q)):arr[st+k]=grouped_means(hidden[k],groups[k])
            arr.flush();done=st+len(q)
            _atomic_json(statefile,{'split':split,'done':done,'total':len(ids),'complete':done==len(ids)})
        del arr
    manifest['status']='complete';manifest['complete']=True;manifest['smoke']=bool(smoke)
    _atomic_json(featdir/'manifest.json',manifest)
    return manifest

def _unit_examples(cache,features,states):
    """Make one example per unit occurrence using only VLM/state/shape metadata."""
    from scipy.fft import dct
    shape=np.asarray(cache['shape'],np.float32);uids=np.asarray(cache['unit_ids']);ts=np.asarray(cache['theta_src']);tt=np.asarray(cache['theta_target'])
    examples=[]
    records=np.asarray(cache['records'])
    for i in range(len(shape)):
        for uid in np.unique(uids[i]):
            if uid<0:continue
            pos=np.flatnonzero(uids[i]==uid);z=shape[i,pos]
            coeff0=dct(z[:,:6],type=2,norm='ortho',axis=0)[:3]
            coeff=np.zeros((3,6),np.float32);coeff[:len(coeff0)]=coeff0
            grip0=dct(z[:,6],type=2,norm='ortho')[:3]
            grip=np.zeros(3,np.float32);grip[:len(grip0)]=grip0
            recs=json.loads(str(records[i]));rec=recs[int(uid)] if int(uid)<len(recs) else {}
            length=float(rec.get('length',len(pos)))
            unitfeat=np.r_[features[i].reshape(-1),np.asarray(states[i],np.float32),coeff.ravel(),grip,np.log1p(length),float(pos.mean()/9.)]
            examples.append((unitfeat,tt[i,pos[0]],ts[i,pos[0]],z,pos,i,uid))
    if not examples:raise ValueError('cache has no selected units')
    return examples

def _target_theta_repr(theta,kind):
    theta=np.asarray(theta,np.float32)
    if kind=='aa':return theta
    return np.r_[rotation6(theta[:3]),theta[3]].astype(np.float32)

def _pred_to_aa(pred,kind):
    if kind=='aa':return np.asarray(pred,np.float32)
    from scipy.spatial.transform import Rotation
    return np.r_[Rotation.from_matrix(rot6_matrix(pred[:6])).as_rotvec(),pred[6]].astype(np.float32)

def _pose_parameter_error(predicted_aa, target_aa, kind):
    """Compare both poses in the same output representation for diagnostics."""
    p=_target_theta_repr(predicted_aa,kind)
    t=_target_theta_repr(target_aa,kind)
    return float(np.mean((p-t)**2))

def _make_data(cache,features,states,kind,cam_mode='all',actions=None):
    ex=_unit_examples(cache,features,states);x=[];y=[];meta=[];action_targets=[]
    for feat,target,src,shape,pos,row,uid in ex:
        feat=feat.copy(); # grouped camera means occupy leading 6144 columns
        if cam_mode=='nowrist':feat[2048:4096]=0
        elif cam_mode=='wristonly':feat[:2048]=0;feat[4096:6144]=0
        x.append(feat);y.append(_target_theta_repr(target,kind));meta.append((row,uid,shape,pos,target,src))
        if actions is not None:action_targets.append(np.asarray(actions[row,pos,:7],np.float32))
    return np.asarray(x,np.float32),np.asarray(y,np.float32),meta,(action_targets if actions is not None else None)

def _model(input_dim,width,kind):
    import torch
    from torch import nn
    out=4 if kind=='aa' else 7
    return nn.Sequential(nn.Linear(input_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,out))

def _torch_decode6(shape,pred,vs):
    import torch
    from torch.nn import functional as F
    a=pred[:,:3];b=pred[:,3:6]
    x=F.normalize(a,dim=-1,eps=1e-8);y=F.normalize(b-(x*b).sum(-1,keepdim=True)*x,dim=-1,eps=1e-8);z=torch.cross(x,y,dim=-1)
    r=torch.stack([x,y,z],dim=-1)
    p=torch.einsum('ni,nij->nj',shape[:,:3],r);rot=torch.einsum('ni,nij->nj',shape[:,3:6],r)
    scale=pred[:,6:7].clamp(-5,5).exp();motion=torch.cat([p,rot],-1)*scale
    lo=torch.as_tensor(vs.LOW,device=pred.device,dtype=pred.dtype);span=torch.as_tensor(vs.SPAN,device=pred.device,dtype=pred.dtype)
    norm=(motion-lo)/(span+1e-6)*2-1
    return torch.cat([norm,shape[:,6:7]],-1)

def _torch_decode6_batch(shape,pred,vs):
    import torch
    b,l,_=shape.shape
    expanded=pred[:,None,:].expand(-1,l,-1).reshape(-1,pred.shape[-1])
    return _torch_decode6(shape.reshape(-1,7),expanded,vs).reshape(b,l,7)

def train_small(condition,*,epochs=12,width=256,lr=1e-3,threads=8,device='cuda',smoke=False):
    """Small optimizer diagnostic. Split holdout is LIBERO-40 train episodes only."""
    import torch;torch.set_num_threads(threads)
    if device=='cuda' and torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.35)
    else:device='cpu'
    rep,variant=condition.split('/')
    resultdir=OUT/('smoke_models_v2' if smoke else 'models')/rep/variant
    if smoke and (resultdir/'train.json').exists() and (resultdir/'model.pt').exists():
        return json.loads((resultdir/'train.json').read_text())
    kind='aa' if variant=='R-aa-theta' else '6d'
    cam='nowrist' if variant.endswith('nowrist') else 'wristonly' if variant.endswith('wristonly') else 'all'
    caches={s:np.load(SOURCE/'drafts'/f'{rep}_{s}.npz',allow_pickle=False) for s in SPLITS}
    featroot=OUT/('smoke_features' if smoke else 'features')
    feats={s:np.load(featroot/f'{s}.npy',mmap_mode='r') for s in SPLITS}
    base,rows,unseen,sourced=_splits();train_states=np.asarray(rows['state'][caches['train']['row_ids']],np.float32)
    if smoke:
        caches['train']=dict((k,caches['train'][k][:8]) for k in caches['train'].files)
        train_states=train_states[:8]
        feats['train']=feats['train'][:8]
    train_actions=np.asarray(rows['actions'][caches['train']['row_ids']],np.float32)[:,:,:7]
    if smoke:train_actions=train_actions[:8]
    x,y,meta,act_target=_make_data(caches['train'],feats['train'],train_states,kind,cam,train_actions)
    # Deterministic held-out training episodes only; no unseen action labels
    # enter early stopping or hyperparameter selection.
    rng=np.random.default_rng(20261105)
    ex_episode=np.asarray([caches['train']['episode'][m[0]] for m in meta])
    unique_ep=np.unique(ex_episode)
    if len(unique_ep)>1:
        hold_ep=set(rng.choice(unique_ep,size=max(1,int(np.ceil(.1*len(unique_ep)))),replace=False).tolist())
        hold=np.flatnonzero(np.isin(ex_episode,list(hold_ep)));fit=np.flatnonzero(~np.isin(ex_episode,list(hold_ep)))
    else:
        perm=rng.permutation(len(x));cut=max(1,int(.9*len(x)));fit,hold=perm[:cut],perm[cut:]
    if not len(fit):raise ValueError('training episode holdout left no fit examples')
    mean=x[fit].mean(0);std=x[fit].std(0);std=np.where(std<1e-5,1,std)
    ym=y[fit].mean(0);ys=y[fit].std(0);ys=np.where(ys<1e-5,1,ys)
    X=torch.tensor((x-mean)/std,device=device);Y=torch.tensor((y-ym)/ys,device=device)
    import sys as _sys;_sys.path.insert(0,str(HERE));import variable_shape_memory as vs
    shape_rows=np.zeros((len(meta),10,7),np.float32);target_rows=np.zeros((len(meta),10,7),np.float32);valid_rows=np.zeros((len(meta),10,1),np.float32)
    for j,m in enumerate(meta):
        _,_,sh,pos,_,_=m;n=min(10,len(pos));shape_rows[j,:n]=sh[:n]
        if act_target is not None:target_rows[j,:n]=act_target[j][:n]
        valid_rows[j,:n]=1.
    SH=torch.tensor(shape_rows,device=device);ATflat=torch.tensor(target_rows,device=device);MASK=torch.tensor(valid_rows,device=device)
    candidates=[(width,lr)] if smoke else [(128,3e-4),(256,1e-3)]
    best=float('inf');best_state=None;best_model=None;best_width=None;best_lr=None;all_history=[]
    for candidate_width,candidate_lr in candidates:
        torch.manual_seed(20261105)
        model=_model(X.shape[1],candidate_width,kind).to(device);opt=torch.optim.AdamW(model.parameters(),lr=candidate_lr)
        candidate_best=float('inf');candidate_state=None;history=[]
        for epoch in range(epochs):
            model.train()
            for batch_ids in np.array_split(rng.permutation(fit),max(1,(len(fit)+255)//256)):
                bix=torch.as_tensor(batch_ids,device=device);opt.zero_grad(set_to_none=True);p=model(X[bix])
                if variant.startswith('R-6d-act'):
                    pred=_torch_decode6_batch(SH[bix],p*torch.tensor(ys,device=device)+torch.tensor(ym,device=device),vs)
                    loss=(((pred-ATflat[bix])**2)*MASK[bix]).sum()/MASK[bix].sum().clamp_min(1.)/7.
                else:loss=((p-Y[bix])**2).mean()
                loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                if len(hold):
                    ph=model(X[hold])
                    if variant.startswith('R-6d-act'):
                        pv=_torch_decode6_batch(SH[hold],ph*torch.tensor(ys,device=device)+torch.tensor(ym,device=device),vs)
                        val=float(((((pv-ATflat[hold])**2)*MASK[hold]).sum()/MASK[hold].sum().clamp_min(1.)/7.).cpu())
                    else:val=float(((ph-Y[hold])**2).mean())
                else:val=float(loss)
            history.append({'epoch':epoch+1,'train_loss':float(loss.detach().cpu()),'holdout_loss':val})
            if val<candidate_best:candidate_best=val;candidate_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        all_history.append({'width':candidate_width,'lr':candidate_lr,'best_holdout_loss':candidate_best,'history':history})
        if candidate_best<best:
            best=candidate_best;best_state=candidate_state;best_width=candidate_width;best_lr=candidate_lr;best_model=model
    model=best_model;model.load_state_dict(best_state)
    resultdir.mkdir(parents=True,exist_ok=True)
    modelpath=resultdir/'model.pt';tmpmodel=modelpath.with_suffix('.pt.tmp')
    torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'ymean':ym,'ystd':ys,'width':best_width,'kind':kind,'cam':cam},tmpmodel)
    tmpmodel.replace(modelpath)
    _atomic_json(resultdir/'train.json',{'condition':condition,'epochs':epochs,'best_holdout_loss':best,'candidate_runs':all_history,
        'train_source':'LIBERO-40 ~validation only','unseen_dev_labels_used':False,'width':best_width,'lr':best_lr,
        'hparam_selection':'episode-held-out subset of LIBERO-40 train; unseen Dev/Eval action labels are not used'})
    return {'condition':condition,'best_holdout_loss':best,'epochs':epochs,'examples':len(x),'smoke':smoke,
            'selected_width':best_width,'selected_lr':best_lr}

def _predict_model(model_path,x):
    import torch
    z=torch.load(model_path,map_location='cpu',weights_only=False);kind=z['kind'];m=_model(x.shape[1],z['width'],kind);m.load_state_dict(z['state_dict']);m.eval()
    with torch.no_grad():p=m(torch.tensor((x-z['mean'])/z['std'],dtype=torch.float32)).numpy()*z['ystd']+z['ymean']
    return np.stack([_pred_to_aa(v,kind) for v in p])

def evaluate_condition(condition,*,threads=8):
    import torch;torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    import sys as _sys;_sys.path.insert(0,str(HERE));import variable_shape_memory as vs
    rep,variant=condition.split('/');kind='aa' if variant=='R-aa-theta' else '6d'
    existing=OUT/'predictions'/rep/variant/'result.json'
    if existing.exists():return json.loads(existing.read_text())
    cam='nowrist' if variant.endswith('nowrist') else 'wristonly' if variant.endswith('wristonly') else 'all'
    base,rows,unseen,splits=_splits();cache=np.load(SOURCE/'drafts'/f'{rep}_eval.npz',allow_pickle=False)
    x,y,meta,_=_make_data(cache,np.load(OUT/'features/eval.npy',mmap_mode='r'),np.asarray(unseen['state'][cache['row_ids']]),kind,cam)
    model_path=OUT/'models'/rep/variant/'model.pt';predtheta=_predict_model(model_path,x)
    n=len(cache['row_ids']);pred=np.zeros((n,10,7),np.float32);src=np.zeros_like(pred);oracle=np.zeros_like(pred)
    for j,(row,uid,shape,pos,target_theta,src_theta) in enumerate(meta):
        pred[row,pos]=decode_theta(shape[:len(pos)],predtheta[j],vs)
        src[row,pos]=decode_theta(shape[:len(pos)],src_theta,vs)
        oracle[row,pos]=decode_theta(shape[:len(pos)],target_theta,vs)
    target=np.asarray(unseen['actions'][cache['row_ids']],np.float32)[:,:,:7]
    # π0.5 reference on Eval is the saved official normal output; Dev normal is
    # generated only when Dev selection diagnostics are requested.
    official=np.load(UNSEEN/'official_normal_eval.npz',allow_pickle=False)
    loc={int(r):i for i,r in enumerate(official['row_ids'])};oi=np.asarray([loc[int(len(rows['task'])+i)] for i in cache['row_ids']])
    normal=np.asarray(official['prediction'][oi],np.float32)
    # Predict a constant training pose; statistics use train rows only.
    tr=np.load(SOURCE/'drafts'/f'{rep}_train.npz',allow_pickle=False)
    theta_const=np.asarray(tr['theta_target'],np.float32)[np.asarray(tr['unit_ids'])>=0].reshape(-1,4).mean(0)
    const=np.zeros_like(pred)
    for j,(row,uid,shape,pos,_,_) in enumerate(meta):const[row,pos]=decode_theta(shape[:len(pos)],theta_const,vs)
    normal_pose=np.zeros_like(pred)
    from compressed_denoise import _procrustes
    for row,uid,shape,pos,_,_ in meta:
        theta_n,_,_=_procrustes(shape[:len(pos),:6],vs.to_physical(normal[row,pos,:6]))
        normal_pose[row,pos]=decode_theta(shape[:len(pos)],theta_n,vs)
    methods={'predicted_theta':pred,'theta_src':src,'train_constant_theta':const,
             'pi05_pose':normal_pose,'theta_star_upper':oracle,'official_pi05':normal}
    hard=np.asarray(cache['hard'],bool);ep=np.asarray(cache['episode']);ep_key=np.asarray(cache['task'],np.int64)*100000+ep
    outdir=OUT/'predictions'/rep/variant;outdir.mkdir(parents=True,exist_ok=True)
    metrics={};row_errors={}
    for name,a in methods.items():
        d=a-target;err=np.mean(d*d,axis=(1,2));row_errors[name]=err
        move_pred=vs.to_physical(a[:,:,:6])[:,:,:3].sum(1);move_true=vs.to_physical(target[:,:,:6])[:,:,:3].sum(1)
        npred=np.linalg.norm(move_pred,axis=1);ntrue=np.linalg.norm(move_true,axis=1);good=ntrue>1e-6
        cos=np.sum(move_pred*move_true,axis=1)/np.maximum(npred*ntrue,1e-8)
        angle=np.degrees(np.arccos(np.clip(cos[good],-1,1))) if good.any() else np.asarray([np.nan])
        metrics[name]={'mse':float(err.mean()),'hard_mse':float(err[hard].mean()) if hard.any() else None,
          'episode_bootstrap_ci':bootstrap_episode(err,ep_key),
          'xyz_mse':float(np.mean(d[:,:,:3]**2)),'rotation_mse':float(np.mean(d[:,:,3:6]**2)),
          'gripper_mse':float(np.mean(d[:,:,6]**2)),
          'hard_xyz_mse':float(np.mean(d[hard,:,:3]**2)),'hard_rotation_mse':float(np.mean(d[hard,:,3:6]**2)),
          'hard_gripper_mse':float(np.mean(d[hard,:,6]**2)),
          'translation_direction_error_degrees':float(np.nanmean(angle)),
          'translation_magnitude_ratio_median':float(np.median(npred[good]/ntrue[good])) if good.any() else None}
    # Unit-level pose discrepancy; target θ is used strictly for scoring.
    theta_err=np.asarray([_pose_parameter_error(predtheta[j],m[4],kind) for j,m in enumerate(meta)])
    theta_src_err=np.asarray([_pose_parameter_error(m[5],m[4],kind) for m in meta]);theta_ep=np.asarray([int(cache['task'][m[0]])*100000+int(ep[m[0]]) for m in meta])
    theta_stats={'predicted_vs_theta_star_mse':float(theta_err.mean()),'theta_src_vs_theta_star_mse':float(theta_src_err.mean()),
        'predicted_theta_ci_vs_star':bootstrap_episode(theta_err,theta_ep),'source_theta_ci_vs_star':bootstrap_episode(theta_src_err,theta_ep)}
    theta_stats['fraction_theta_gap_closed']=float(1-theta_err.mean()/max(theta_src_err.mean(),1e-12))
    theta_from_src=np.asarray([_pose_parameter_error(predtheta[j],m[5],kind) for j,m in enumerate(meta)])
    theta_stats['predicted_vs_theta_src_mse']=float(theta_from_src.mean())
    theta_stats['predicted_vs_src_ci']=bootstrap_episode(theta_from_src,theta_ep)
    changes=np.sign(target[:,:,6]);switch=np.zeros_like(changes,dtype=bool);switch[:,1:]=changes[:,1:]!=changes[:,:-1]
    local=np.zeros(n,bool)
    for i in range(n):
        ix=np.flatnonzero(switch[i]);local[i]=bool(len(ix) and np.any(np.abs(ix[:,None]-np.arange(10)[None,:])<=5))
    summary={'condition':condition,'rows':n,'hard_rows':int(hard.sum()),'methods':metrics,'theta_diagnostics':theta_stats,
      'switch_neighborhood_mse':{name:float(np.mean((a[local,:,6]-target[local,:,6])**2)) if local.any() else None for name,a in methods.items()},
      'selection':'No Eval targets used to fit features, theta, model, or hyperparameters; target used only for oracle unit selection and scoring.'}
    rowpath=outdir/'eval_rows.npz'
    if rowpath.exists():raise FileExistsError(f'incomplete evaluation output exists: {rowpath}')
    _atomic_npz(rowpath,row_ids=cache['row_ids'],episode=ep,task=cache['task'],hard=hard,target=target,
        theta_pred=predtheta,predicted=pred,theta_src=src,theta_star_upper=oracle,pi05_pose=normal_pose,official_pi05=normal,
        **{f'err_{k}':v for k,v in row_errors.items()})
    _atomic_json(outdir/'result.json',summary);return summary

def bootstrap_episode(values,episodes,*,reps=5000,seed=4101):
    values=np.asarray(values,np.float64);ep=np.asarray(episodes);u,inv=np.unique(ep,return_inverse=True);s=np.bincount(inv,weights=values);n=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(u),size=(reps,len(u)));d=s[ix].sum(1)/n[ix].sum(1)
    return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]

def _resample_window(values,n=10):
    a=np.asarray(values,np.float32)
    if len(a)==n:return a.copy()
    if len(a)<1:raise ValueError('empty motion template')
    src=np.linspace(0.,1.,len(a));dst=np.linspace(0.,1.,n)
    return np.stack([np.interp(dst,src,a[:,j]) for j in range(a.shape[1])],axis=1).astype(np.float32)

def _shape_error(candidate,target,vs):
    # Both are physical motion arrays; the shared Procrustes rotation/scale is
    # used only to score shape similarity.
    from compressed_denoise import _procrustes
    _,aligned,err=_procrustes(np.asarray(candidate)[:,:6],np.asarray(target)[:,:6])
    return err,aligned

def _replica_mean_errors(predictions, target):
    """Mean per-replica row errors; never average candidate actions together."""
    p=np.asarray(predictions,np.float64);y=np.asarray(target,np.float64)
    if p.ndim!=4 or p.shape[0]!=len(y) or p.shape[2:]!=y.shape[1:]:
        raise ValueError((p.shape,y.shape))
    return np.mean(np.mean((p-y[:,None])**2,axis=(2,3)),axis=1)

def candidate_shape_pool(*,rep='C_vm_none'):
    """Deduplicate cached training-only unit templates; never accesses Dev/Eval labels."""
    cache=np.load(SOURCE/'drafts'/f'{rep}_train.npz',allow_pickle=False)
    seen=set();shapes=[];source_rows=[];source_thetas=[];source_tasks=[];source_units=[]
    uids=np.asarray(cache['unit_ids']);shape=np.asarray(cache['shape'])
    for i,raw in enumerate(cache['records']):
        records=json.loads(str(raw))
        for slot,rec in enumerate(records):
            sr=int(rec['source_row']);uid=int(rec['unit']);key=(sr,uid,int(rec['source_task']))
            if key in seen:continue
            seen.add(key);pos=np.flatnonzero(uids[i]==slot)
            if not len(pos):continue
            s=_resample_window(shape[i,pos],10)
            shapes.append(s);source_rows.append(sr);source_tasks.append(int(rec['source_task']))
            source_units.append(uid);source_thetas.append(np.asarray(rec['source_theta'],np.float32))
    if not shapes:raise ValueError('no candidate templates found in train cache')
    return {'shape':np.asarray(shapes,np.float32),'source_row':np.asarray(source_rows,np.int64),'source_unit':np.asarray(source_units,np.int32),
            'source_task':np.asarray(source_tasks,np.int32),'source_theta':np.asarray(source_thetas,np.float32)}

def _normal_eval_predictions(base,source,ids,*,global_ids,batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    model,_=base.load_policy(CHECKPOINT);fn=module_jit(model.sample_actions,static_argnames=('num_steps',));out=[]
    for st in range(0,len(ids),batch):
        sl=slice(st,min(st+batch,len(ids)));obs=base.obs_batch(source,ids[sl])
        seeds=jnp.asarray(global_ids[sl],jnp.uint32)
        keys=jax.vmap(lambda r:jax.random.fold_in(jax.random.key(4101),r))(seeds)
        noise=jax.vmap(lambda k:jax.random.normal(k,(model.action_horizon,model.action_dim)))(keys)
        out.append(np.asarray(fn(jax.random.PRNGKey(0),obs,num_steps=10,noise=noise)))
    return np.concatenate(out)[:,:,:7]

def shape_select(split='eval',*,rep='C_vm_none',seed=4101,limit=None,smoke=False):
    """Compare π0.5-selected, target-oracle and random cached motion shapes."""
    shape_root=OUT/('smoke_shape_selection' if smoke else 'shape-selection')
    result_path=shape_root/rep/split/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    import sys as _sys;_sys.path.insert(0,str(HERE));import variable_shape_memory as vs
    base,rows,unseen,splits=_splits();source,ids=splits[split]
    if split=='train':raise ValueError('shape selection is for held-out Dev/Eval only')
    cache=np.load(SOURCE/'drafts'/f'{rep}_{split}.npz',allow_pickle=False)
    if limit is not None:
        ids=ids[:int(limit)]
        cache={k:cache[k][:len(ids)] for k in cache.files}
    if split=='eval':
        official=np.load(UNSEEN/'official_normal_eval.npz',allow_pickle=False);loc={int(r):i for i,r in enumerate(official['row_ids'])}
        normal=np.asarray(official['prediction'][[loc[int(len(rows['task'])+i)] for i in ids]],np.float32)
    else:
        normal=_normal_eval_predictions(base,source,ids,global_ids=len(rows['actions'])+ids)
    target=np.asarray(source['actions'][ids],np.float32)[:,:,:7]
    candidates=candidate_shape_pool(rep=rep);K=len(candidates['shape']);n=len(ids)
    # Keep five random trials as five separate candidate predictions.  The
    # random baseline averages their errors, never their actions.
    picked=np.zeros((n,7),np.int64);preds=np.zeros((n,7,10,7),np.float32);score=np.zeros((n,K),np.float32)
    pi_target_pose=np.zeros((n,10,7),np.float32);oracle_pi_pose=np.zeros_like(pi_target_pose)
    random_target_pose=np.zeros((n,5,10,7),np.float32)
    rng=np.random.default_rng(seed);normal_phys=vs.to_physical(normal[:,:,:6]);target_phys=vs.to_physical(target[:,:,:6])
    for i in range(n):
        for j,c in enumerate(candidates['shape']):score[i,j]=_shape_error(c,normal_phys[i],vs)[0]
        pi=int(np.argmin(score[i]));oracle=int(np.argmin([_shape_error(c,target_phys[i],vs)[0] for c in candidates['shape']]))
        rand=[int(rng.integers(K)) for _ in range(5)]
        for mode,ci in enumerate([pi,oracle]):
            c=candidates['shape'][ci];_,pn=_shape_error(c,normal_phys[i],vs);_,pt=_shape_error(c,target_phys[i],vs)
            # Recover theta parameters with the same shared-rotation/scale rule.
            from compressed_denoise import _procrustes
            theta_n,_,_=_procrustes(c[:,:6],normal_phys[i])
            theta_t=_procrustes(c[:,:6],target_phys[i])[0]
            theta=theta_t if mode==1 else theta_n
            preds[i,mode]=decode_theta(c,theta,vs);picked[i,mode]=ci
            if mode==0:pi_target_pose[i]=decode_theta(c,theta_t,vs)
            else:oracle_pi_pose[i]=decode_theta(c,theta_n,vs)
        # Five independent random templates, each restored using π0.5-derived θ.
        for rep_ix,ci in enumerate(rand):
            c=candidates['shape'][ci];theta,_,_=_procrustes(c[:,:6],normal_phys[i])
            preds[i,2+rep_ix]=decode_theta(c,theta,vs);picked[i,2+rep_ix]=ci
            theta_t,_,_=_procrustes(c[:,:6],target_phys[i])
            random_target_pose[i,rep_ix]=decode_theta(c,theta_t,vs)
    names=('pi05_shape','oracle_shape_upper','random5_mean');metrics={}
    hard=np.asarray(cache['hard'],bool)
    episode_keys=np.asarray(cache['task'])*100000+np.asarray(cache['episode'])
    for j,name in enumerate(names):
        if name=='random5_mean':
            d=preds[:,2:]-target[:,None]
            err=_replica_mean_errors(preds[:,2:],target)
            xyz=np.mean(np.mean(d[:,:,:,:3]**2,axis=(2,3)),axis=1)
            rot=np.mean(np.mean(d[:,:,:,3:6]**2,axis=(2,3)),axis=1)
            grip=np.mean(np.mean(d[:,:,:,6]**2,axis=2),axis=1)
        else:
            d=preds[:,j]-target;err=np.mean(d*d,axis=(1,2))
            xyz=np.mean(d[:,:,:3]**2,axis=(1,2));rot=np.mean(d[:,:,3:6]**2,axis=(1,2));grip=np.mean(d[:,:,6]**2,axis=1)
        metrics[name]={'mse':float(err.mean()),'hard_mse':float(err[hard].mean()),
            'xyz_mse':float(xyz.mean()),'rotation_mse':float(rot.mean()),'gripper_mse':float(grip.mean()),
            'episode_bootstrap_ci':bootstrap_episode(err,episode_keys)}
    def add_pose_metric(name, arr, replicas=False):
        if replicas:
            delta=np.asarray(arr)-target[:,None]
            err=_replica_mean_errors(arr,target)
            xyz=np.mean(np.mean(delta[:,:,:,:3]**2,axis=(2,3)),axis=1)
            rot=np.mean(np.mean(delta[:,:,:,3:6]**2,axis=(2,3)),axis=1)
            grip=np.mean(np.mean(delta[:,:,:,6]**2,axis=2),axis=1)
        else:
            delta=np.asarray(arr)-target;err=np.mean(delta**2,axis=(1,2))
            xyz=np.mean(delta[:,:,:3]**2,axis=(1,2));rot=np.mean(delta[:,:,3:6]**2,axis=(1,2));grip=np.mean(delta[:,:,6]**2,axis=1)
        metrics[name]={'mse':float(err.mean()),'hard_mse':float(err[hard].mean()),
            'xyz_mse':float(xyz.mean()),'rotation_mse':float(rot.mean()),'gripper_mse':float(grip.mean()),
            'episode_bootstrap_ci':bootstrap_episode(err,episode_keys),
            'theta_restore':'target-derived diagnostic' if 'theta_star' in name else 'pi05-derived'}
    add_pose_metric('pi05_shape_theta_star',pi_target_pose)
    add_pose_metric('oracle_shape_pi05_pose',oracle_pi_pose)
    add_pose_metric('random5_theta_star_mean',random_target_pose,replicas=True)
    out=shape_root/rep/split;out.mkdir(parents=True,exist_ok=True)
    rowpath=out/'rows.npz'
    if rowpath.exists():raise FileExistsError(f'incomplete shape-selection output exists: {rowpath}')
    _atomic_npz(rowpath,row_ids=ids,episode=cache['episode'],task=cache['task'],candidate_source_rows=candidates['source_row'],
        selected_candidate=picked,prediction=preds,pi05_shape_theta_star=pi_target_pose,
        oracle_shape_pi05_pose=oracle_pi_pose,random5_theta_star=random_target_pose,
        target=target,normal=normal,hard=cache['hard'])
    result={'split':split,'representation':rep,'rows':n,'candidate_templates':K,'metrics':metrics,
        'note':'Candidates are deduplicated only from cached LIBERO-40 train rows; this is a cache-limited proxy. Window selection does not continue a unit for its full duration. random5_mean is the mean metric over five independently selected random candidates; candidate actions are never blended.'}
    _atomic_json(out/'result.json',result);return result

def run_smoke():
    """CPU synthetic interface smoke; never reads Eval action targets for training."""
    smoke_path=OUT/'smoke-v3.json'
    if smoke_path.exists():return json.loads(smoke_path.read_text())
    rng=np.random.default_rng(17);h=rng.normal(size=(13,2048)).astype(np.float32);g=np.repeat([0,1,2],[4,5,4])
    m=grouped_means(h,g)
    assert m.shape==(3,2048) and np.isfinite(m).all()
    from scipy.spatial.transform import Rotation
    theta=np.r_[Rotation.from_rotvec([.2,.1,-.3]).as_rotvec(),np.log(1.4)]
    theta6=np.r_[rotation6(theta[:3]),theta[3]]
    np.testing.assert_allclose(Rotation.from_matrix(rot6_matrix(theta6[:6])).as_rotvec(),theta[:3],atol=1e-5)
    # Check cache discovery and split IDs without reading any unseen target actions.
    _,rows,unseen,splits=_splits();assert len(splits['train'][1])==23304 and len(splits['dev'][1])==113 and len(splits['eval'][1])==619
    for rep in CACHES:
        for split in SPLITS:
            z=np.load(SOURCE/'drafts'/f'{rep}_{split}.npz',allow_pickle=False)
            expected=len(splits[split][1]);assert len(z['row_ids'])==expected
            assert z['shape'].shape==(expected,10,7) and z['unit_ids'].shape==(expected,10)
    # Real, tiny frozen-feature pass and a CPU MLP update use only 8 train rows.
    feat=extract_features(batch=4,threads=8,smoke=True,limit=8)
    fit=train_small('C_vm_none/R-6d-theta',epochs=2,width=64,lr=1e-3,threads=8,device='cpu',smoke=True)
    shape_smoke=[shape_select('eval',rep=rep,limit=4,smoke=True) for rep in CACHES]
    result={'status':'complete','split_rows':{'train':23304,'dev':113,'eval':619},'cache_sources':CACHES,
            'feature_extraction':feat,'smoke_fit':fit,
            'shape_selection_smoke':[{'rep':r['representation'],'candidate_templates':r['candidate_templates'],
                'pi05_shape_mse':r['metrics']['pi05_shape']['mse'],
                'oracle_shape_mse':r['metrics']['oracle_shape_upper']['mse'],
                'random_mse':r['metrics']['random5_mean']['mse']} for r in shape_smoke],
            'target_actions_used_for_theta_or_hparam':False}
    _atomic_json(smoke_path,result);return result

def run_all(*,threads=8,device='cuda'):
    if (OUT/'summary.json').exists():return json.loads((OUT/'summary.json').read_text())
    feat=OUT/'features'/'manifest.json'
    if not feat.exists():extract_features(batch=8,threads=threads)
    conditions=[f'{rep}/{variant}' for rep in CACHES for variant in VARIANTS]
    train_results=[];eval_results=[]
    for condition in conditions:
        rep,variant=condition.split('/');mdir=OUT/'models'/rep/variant
        if (mdir/'train.json').exists() and (mdir/'model.pt').exists():
            trained=json.loads((mdir/'train.json').read_text())
        else:trained=train_small(condition,threads=threads,device=device)
        train_results.append(trained)
        rpath=OUT/'predictions'/rep/variant/'result.json'
        eval_results.append(json.loads(rpath.read_text()) if rpath.exists() else evaluate_condition(condition,threads=threads))
    shape_results=[]
    for rep in CACHES:
        for split in ('dev','eval'):
            rpath=OUT/'shape-selection'/rep/split/'result.json'
            shape_results.append(json.loads(rpath.read_text()) if rpath.exists() else shape_select(split,rep=rep))
    summary={'status':'complete','conditions':conditions,'training':train_results,
      'evaluation':eval_results,'shape_selection':shape_results,
      'splits':{'train_rows':23304,'dev_rows':113,'eval_rows':619},
      'data_policy':'Only LIBERO-40 train rows train models/hyperparameters. Unseen Dev/Eval actions are read only for per-row unit selection and scoring; no labels enter θ, model fitting, or hyperparameter search.'}
    _atomic_json(OUT/'summary.json',summary)
    # Compact human-readable outcome table.
    lines=['# Observation-derived pose and shape selection','',
      '正解行動を使う上限選択の診断。θ回帰・学習・ハイパーパラメータ選択に未学習Dev/Evalの行動ラベルは使用しない。',
      '学習・内部holdoutはLIBERO-40 train内で行い、Dev/Evalは型選択と採点だけに使う。', '',
      '| 条件 | θ予測MSE | θ_srcとの差 | Eval action MSE | hard MSE | π0.5 pose MSE | θ*上限 MSE |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for res in eval_results:
        m=res['methods'];t=res['theta_diagnostics']
        lines.append(f"| {res['condition']} | {t['predicted_vs_theta_star_mse']:.5f} | {t['predicted_vs_theta_src_mse']:.5f} | {m['predicted_theta']['mse']:.5f} | {m['predicted_theta']['hard_mse']:.5f} | {m['pi05_pose']['mse']:.5f} | {m['theta_star_upper']['mse']:.5f} |")
    lines+=['','Shape selection (窓ごとの選択で、型を元の長さいっぱい続けない):', '',
      '| 表現 / split | 候補数 | π0.5 shape MSE | oracle shape MSE | random MSE |', '|---|---:|---:|---:|---:|']
    for res in shape_results:
        m=res['metrics'];lines.append(f"| {res['representation']} / {res['split']} | {res['candidate_templates']} | {m['pi05_shape']['mse']:.5f} | {m['oracle_shape_upper']['mse']:.5f} | {m['random5_mean']['mse']:.5f} |")
    lines+=['','候補単位プールは既存のC-vm trainキャッシュからメモリ行で重複除去したもの。全単位の網羅性は保証せず、この限定的な比較として解釈する。',
      '未学習Evalは619行・hard 134行。単一seed、オフライン、正解での型選択を含むため閉ループの効果とは解釈しない。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    return {'status':'complete','conditions':len(conditions),'shape_runs':len(shape_results)}

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','smoke','train','evaluate','shape-select','run']);p.add_argument('--condition');p.add_argument('--split',choices=('dev','eval'),default='eval');p.add_argument('--threads',type=int,default=8);p.add_argument('--batch',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=None,help='output directory (default: theta-regression-01/production, read by later steps)')
    a=p.parse_args()
    OUT=a.output or OUT/'production'
    if a.stage=='features':print(json.dumps(extract_features(batch=a.batch,threads=a.threads,smoke=a.smoke)))
    elif a.stage=='smoke':print(json.dumps(run_smoke()))
    elif a.stage=='evaluate':
        if not a.condition:raise SystemExit('--condition rep/variant required')
        print(json.dumps(evaluate_condition(a.condition,threads=a.threads)))
    elif a.stage=='shape-select':print(json.dumps(shape_select(a.split)))
    elif a.stage=='run':print(json.dumps(run_all(threads=a.threads,device=a.device)))
    else:
        if not a.condition:raise SystemExit('--condition rep/variant required')
        print(json.dumps(train_small(a.condition,threads=a.threads,device=a.device)))
if __name__=='__main__':main()
