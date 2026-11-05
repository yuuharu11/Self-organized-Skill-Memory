#!/usr/bin/env python3
"""Offline local-observation key audit for raw LIBERO action memories."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104')
RUN=ROOT/'run-01'; MEMORY=ROOT/'skill-memories-01'/'raw-rows-noLong'
GAP=ROOT/'memory-gap-01'; TOPK=ROOT/'topk-long-01'/'production'
OUT=ROOT/'local-key-02'; Q80=0.01549410422449
TASK_RANGES=((0,10,'libero_10'),(10,20,'libero_goal'),(20,30,'libero_object'),(30,40,'libero_spatial'))
REGIONS={'wrist_center4':(slice(6,10),slice(6,10)),
         'wrist_center8':(slice(4,12),slice(4,12)),
         'wrist_lowerhalf':(slice(8,16),slice(0,16))}


def _load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(mod);return mod


def _l2_normalize(x):
    x=np.asarray(x,np.float32);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def _standardize(train,other):
    tr=np.asarray(train,np.float32);x=np.asarray(other,np.float32)
    mu=tr.mean(0,dtype=np.float64).astype(np.float32);sd=tr.std(0,dtype=np.float64).astype(np.float32);sd[sd<1e-5]=1.
    return ((tr-mu)/sd).astype(np.float32),((x-mu)/sd).astype(np.float32),{'mean':mu,'std':sd}


def _fit_key_block(train, other):
    """Train-only standardize and row-normalize one visual/state/history block."""
    tr, x, stats = _standardize(train, other)
    return _l2_normalize(tr), _l2_normalize(x), stats


def _join_key_blocks(visual, state=None, history=None, weights=(1., 0., 0.)):
    """Weighted concat of independently normalized blocks, then normalize once more."""
    blocks=[]
    for block, weight in zip((visual, state, history), weights, strict=True):
        if weight <= 0 or block is None:
            continue
        blocks.append(_l2_normalize(np.asarray(block, np.float32)) * np.float32(np.sqrt(weight)))
    if not blocks:
        raise ValueError('at least one key block must have positive weight')
    return _l2_normalize(np.concatenate(blocks, axis=-1))


def _state_denorm(s,q01,q99):return ((np.asarray(s,np.float32)+1.)*.5*(q99-q01)+q01).astype(np.float32)


def project_world_to_pixel(point,transform,*,camera_size=256,image_size=224,rotate180=True):
    p=np.r_[np.asarray(point,np.float64)[:3],1.]
    c=np.asarray(transform,np.float64)@p
    if c[2]<=1e-7:return np.array([np.nan,np.nan],np.float32)
    uv=c[:2]/c[2]
    scale=float(image_size)/float(camera_size)
    if rotate180:uv=np.array([camera_size-1-uv[0],camera_size-1-uv[1]])
    return (uv*scale).astype(np.float32)


def patch_ids_for_point(pixel,patch=3,image_size=224,grid=16):
    uv=np.asarray(pixel,np.float32);cell=image_size/grid
    if not np.isfinite(uv).all():return np.empty(0,np.int64)
    col,row=int(np.floor(uv[0]/cell)),int(np.floor(uv[1]/cell));radius=patch//2
    return np.asarray([r*grid+c for r in range(max(0,row-radius),min(grid,row+radius+1))
                       for c in range(max(0,col-radius),min(grid,col+radius+1))],np.int64)


def _history(rows):
    n=len(rows['frame']);lookup={(int(e),int(f)):i for i,(e,f) in enumerate(zip(rows['episode'],rows['frame']))}
    out=np.zeros((n,71),np.float32);acts=np.asarray(rows['actions'][:,:,:7],np.float32)
    for i,(e,f) in enumerate(zip(rows['episode'],rows['frame'])):
        j=lookup.get((int(e),int(f)-10))
        if j is not None:out[i,:70]=acts[j].reshape(-1);out[i,70]=1.
    return out


def _camera_transforms(task_texts,cache_path=None):
    """Read each task's agentview world-to-pixel matrix from a CPU LIBERO env."""
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    text_to_id={str(v).strip().lower():int(k) for k,v in task_texts.items()}
    if cache_path is not None and Path(cache_path).exists():
        z=np.load(cache_path,allow_pickle=False);ids=np.asarray(z['task_ids'],np.int32);mats=np.asarray(z['matrices'],np.float32)
        cached={int(i):m for i,m in zip(ids,mats,strict=True)}
        if set(cached)==set(range(40)):return cached
        raise AssertionError('camera matrix cache is incomplete')
    result={};benchmark_dict=benchmark.get_benchmark_dict()
    for suite_name in ('libero_10','libero_goal','libero_object','libero_spatial'):
        suite=benchmark_dict[suite_name]()
        for local in range(suite.n_tasks):
            task=suite.get_task(local)
            key=task.language.strip().lower()
            if key not in text_to_id:raise AssertionError(f'LIBERO task text missing from recorded task table: {key}')
            task_id=text_to_id[key]
            if task_id in result:raise AssertionError(f'duplicate camera mapping for task {task_id}')
            bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
            try:
                env.reset();env.sim.forward()
                result[task_id]=np.asarray(get_camera_transform_matrix(env.sim,'agentview',256,256),np.float32)
            finally:env.close()
    if set(result)!=set(range(40)):raise AssertionError('camera calibration missing a task')
    if cache_path is not None:
        np.savez_compressed(cache_path,task_ids=np.asarray(sorted(result),np.int32),matrices=np.stack([result[i] for i in sorted(result)]))
    return result


def _extract_keys(rows,needed,cameras,pose_q01,pose_q99,*,tokens_path=None,block=12):
    tokens=np.load(tokens_path or RUN/'tokens.npy',mmap_mode='r',allow_pickle=False)
    groups=np.load(RUN/'groups.npy',mmap_mode='r',allow_pickle=False)
    if tokens.shape!=(26581,712,2048) or tokens.dtype!=np.float16:raise AssertionError((tokens.shape,tokens.dtype))
    if groups.shape!=(len(tokens),712):raise AssertionError(groups.shape)
    # Verify the fixed VLM layout: row-major base patches, wrist patches, then text.
    if not (np.all(groups[0,:256]==0) and np.all(groups[0,256:512]==1)):
        raise AssertionError('unexpected base/wrist token order')
    output={k:np.empty((len(needed),2048),np.float32) for k in ('G','W',*REGIONS,'base_ee3','base_ee5')}
    index={int(r):i for i,r in enumerate(needed)}
    points={}
    for start in range(0,len(needed),block):
        ids=np.asarray(needed[start:start+block],np.int64);tok=np.asarray(tokens[ids,:512],np.float32).reshape(len(ids),2,256,2048)
        output['G'][start:start+len(ids)]=tok.mean(axis=(1,2))
        output['W'][start:start+len(ids)]=tok[:,1].mean(axis=1)
        for name,(rs,cs) in REGIONS.items():
            patch_ix=np.asarray([r*16+c for r in range(*rs.indices(16)) for c in range(*cs.indices(16))],np.int64)
            output[name][start:start+len(ids)]=tok[:,1,patch_ix].mean(axis=1)
        states=_state_denorm(np.asarray(rows['state'][ids,:8],np.float32),pose_q01,pose_q99)
        for j,row in enumerate(ids):
            mat=cameras[int(rows['task'][row])]
            pixel=project_world_to_pixel(states[j,:3],mat)
            points[int(row)]=pixel
            for name,p in [('base_ee3',3),('base_ee5',5)]:
                pi=patch_ids_for_point(pixel,p)
                if not len(pi):output[name][start+j]=tok[j,0].mean(0)
                else:output[name][start+j]=tok[j,0,pi].mean(0)
    return output,points


def _row_mse(q,mem):
    q=np.asarray(q,np.float32).reshape(len(q),70);m=np.asarray(mem,np.float32).reshape(len(mem),70)
    mn=np.sum(m*m,axis=1);qn=np.sum(q*q,axis=1);d=(qn[:,None]+mn[None,:]-2*q@m.T)/70.
    return np.maximum(d,0.).astype(np.float32)


def _candidate_mse(query_actions,candidate_actions):
    q=np.asarray(query_actions,np.float32);c=np.asarray(candidate_actions,np.float32)
    if c.ndim!=4 or c.shape[0]!=len(q):raise ValueError(f'candidates must be [Q,K,10,7], got {c.shape}')
    return np.mean((q[:,None]-c)**2,axis=(-1,-2),dtype=np.float32)


def _stable_rank(score,row_ids,k):return np.lexsort((np.asarray(row_ids,np.int64),-np.asarray(score,np.float32)))[:k]


def _retrieve(qfeat,mfeat,qids,mids,rows,actions,*,metric='cos',exclusion='task',k=64):
    q=np.asarray(qfeat,np.float32);m=np.asarray(mfeat,np.float32)
    if metric=='cos':q=_l2_normalize(q);m=_l2_normalize(m);mn=None
    elif metric=='l2':mn=np.sum(m*m,axis=1)
    else:raise ValueError(metric)
    cand=np.empty((len(qids),k),np.int64);scores=np.empty((len(qids),k),np.float32);mses=np.empty((len(qids),k),np.float32)
    mt=np.asarray(rows['task'][mids]);me=np.asarray(rows['episode'][mids]);actm=actions
    for s in range(0,len(qids),32):
        e=min(s+32,len(qids));dot=q[s:e]@m.T
        if metric=='cos':allscore=dot
        else:allscore=-(np.sum(q[s:e]*q[s:e],axis=1)[:,None]+mn[None]-2*dot)
        for j,row in enumerate(qids[s:e]):
            ok=(mt!=int(rows['task'][row])) if exclusion=='task' else (me!=int(rows['episode'][row]))
            vals=np.asarray(allscore[j],np.float32).copy();vals[~ok]=-np.inf
            ix=_stable_rank(vals,mids,k);cand[s+j]=mids[ix];scores[s+j]=vals[ix]
            # ix indexes the compact memory pool; convert it back to dataset row IDs.
            mses[s+j]=_row_mse(actions[row:row+1],actm[mids[ix]])[0]
    return cand,scores,mses


def _spearman(x,y):
    from scipy.stats import spearmanr
    if np.std(x)<1e-12 or np.std(y)<1e-12:return float('nan')
    return float(spearmanr(x,y).statistic)


def _metrics(name,cand,score,cand_mse,all_mse,qids,rows):
    if all_mse is None:
        recall=np.full(len(qids),np.nan,np.float32);eligible=np.zeros(len(qids),bool)
    else:
        relevant=all_mse<=.01;den=relevant.sum(1);eligible=den>0
        # The candidate arrays are ranked Top-64; Recall@8 must count only the
        # first eight retrieved rows, not every relevant item in the rerank set.
        recall=np.divide((cand_mse[:,:8]<=.01).sum(1),den,out=np.zeros(len(qids),np.float32),where=den>0)
    return {'name':name,'top1_mse':cand_mse[:,0].tolist(),'top5_best_mse':cand_mse[:,:5].min(1).tolist(),
      'recall8':recall.tolist(),'query_positive':eligible.tolist(),
      'spearman_top64':[_spearman(score[i],cand_mse[i]) for i in range(len(qids))],
      'top1_row':cand[:,0].astype(int).tolist(),
      'candidate_rows':np.asarray(cand,np.int64).tolist()}


def _episode_bootstrap(diff,episodes,seed=20261105,reps=5000):
    d=np.asarray(diff,np.float64);ep=np.asarray(episodes,np.int64);unique=np.unique(ep)
    rng=np.random.default_rng(seed);means=np.empty(reps,np.float64)
    groups={int(e):np.flatnonzero(ep==e) for e in unique}
    for b in range(reps):
        sampled=rng.choice(unique,size=len(unique),replace=True);ix=np.concatenate([groups[int(e)] for e in sampled])
        means[b]=d[ix].mean()
    return {'mean':float(d.mean()),'ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
            'episodes':int(len(unique)),'rows':int(len(d)),'bootstrap_reps':reps}


def _assert_no_long(train_ids,select_ids,rows):
    for name,ids in [('fit',train_ids),('selection',select_ids)]:
        if np.any(np.asarray(rows['task'])[ids]<10):raise AssertionError(f'Long rows leaked into {name}')


def _plot_grid(out,rows,example_rows,points):
    from PIL import Image,ImageDraw
    tiles=[]
    for row in example_rows:
        img=Image.fromarray(np.asarray(rows['image_base'][row],np.uint8)).resize((224,224));d=ImageDraw.Draw(img)
        for i in range(0,225,14):d.line((i,0,i,223),fill=(70,70,70),width=1);d.line((0,i,223,i),fill=(70,70,70),width=1)
        p=np.asarray(points[int(row)],np.float32)
        if np.isfinite(p).all():d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=(255,0,0),outline=(255,255,255),width=2)
        d.rectangle((0,0,223,18),fill=(0,0,0));d.text((4,3),f"task={int(rows['task'][row])} row={int(row)} px={p[0]:.1f},{p[1]:.1f}",fill='white')
        tiles.append(img)
    canvas=Image.new('RGB',(448,448),'white')
    for i,img in enumerate(tiles[:4]):canvas.paste(img,((i%2)*224,(i//2)*224))
    canvas.save(out/'projection_overlay.png',optimize=True)


def _camera_preview(out,rows,cameras):
    from PIL import Image,ImageDraw
    ids=[]
    for t in [0,3,6,9]:
        ix=np.flatnonzero(np.asarray(rows['task'])==t)
        if len(ix):ids.append(int(ix[len(ix)//2]))
    # Preview grid demonstrates that 16x16 ViT tokens are mapped row-major onto
    # the post-180-degree, resized camera image used by the tokenizer.
    _plot_grid(out,rows,ids,{r:project_world_to_pixel(_state_denorm(np.asarray(rows['state'][r,:8]),
        np.asarray([-0.35244689,-0.26824865,0.04083746,1.53176537,-2.715233,-1.076538,0,0]),
        np.asarray([0.13891278,0.32519916,1.25689629,3.26276857,2.4437233,0.56384698,1,1])),cameras[int(rows['task'][r])]) for r in ids})


def _fit_direct(train_x,train_y,dev_x,dev_y,threads=8):
    import torch
    from torch import nn
    torch.set_num_threads(threads);torch.manual_seed(442)
    mu=train_x.mean(0);sd=np.maximum(train_x.std(0),1e-5);x=(train_x-mu)/sd;dv=(dev_x-mu)/sd
    model=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,70))
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4);best=None;bestloss=float('inf');patience=0
    ds=torch.utils.data.TensorDataset(torch.as_tensor(x),torch.as_tensor(train_y.reshape(-1,70)))
    loader=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True,num_workers=0)
    for epoch in range(20):
        model.train()
        for bx,by in loader:
            opt.zero_grad(set_to_none=True);loss=((model(bx)-by)**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():vl=float(((model(torch.as_tensor(dv))-torch.as_tensor(dev_y.reshape(-1,70)))**2).mean())
        if vl<bestloss-1e-5:bestloss=vl;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
        else:patience+=1
        if patience>=3:break
    if best is not None:model.load_state_dict(best)
    model.eval()
    return model,{'epochs':epoch+1,'dev_mse':bestloss,'mean':np.asarray(mu,np.float32),'std':np.asarray(sd,np.float32)}


def _predict_direct(model,x,mean,std):
    import torch
    model.eval()
    with torch.no_grad():return model(torch.as_tensor((x-mean)/std)).cpu().numpy().reshape(-1,10,7)


def _fit_projection(train_x,train_actions,train_task,dev_x,dev_actions,dev_task,threads=8):
    """Small supervised-contrastive local key projection, labels are only action-distance pairs."""
    import torch
    from torch import nn
    torch.set_num_threads(threads);torch.manual_seed(904)
    rng=np.random.default_rng(904); n=len(train_x)
    # Pick tau on Dev by a predeclared density rule: cross-task positive fraction closest to 10%.
    pair_i=rng.integers(0,len(dev_x),size=50000);pair_j=rng.integers(0,len(dev_x),size=50000)
    cross=dev_task[pair_i]!=dev_task[pair_j]
    dm=np.mean((dev_actions[pair_i]-dev_actions[pair_j])**2,axis=(1,2));rates={str(t):float(np.mean(dm[cross]<=t)) for t in (.005,.01,.02)}
    tau=min((.005,.01,.02),key=lambda t:(abs(rates[str(t)]-.10),t))
    mu=train_x.mean(0);sd=train_x.std(0).clip(1e-5);x=((train_x-mu)/sd).astype(np.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,128));opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    ids=np.arange(n)
    for epoch in range(8):
        rng.shuffle(ids)
        for start in range(0,n,128):
            ix=ids[start:start+128]
            if len(ix)<4:continue
            z=nn.functional.normalize(model(torch.as_tensor(x[ix])),dim=1)
            sim=z@z.T/.12;act=torch.as_tensor(train_actions[ix].reshape(len(ix),70),dtype=torch.float32)
            diff=((act[:,None]-act[None])**2).mean(-1)
            task=torch.as_tensor(train_task[ix],dtype=torch.int64);same=task[:,None]==task[None,:]
            eye=torch.eye(len(ix),dtype=torch.bool);pos=(diff<=tau)&~same&~eye;valid=pos.any(1)
            if not valid.any():continue
            sim=sim.masked_fill(eye,-1e4);logden=torch.logsumexp(sim,dim=1);logpos=torch.logsumexp(sim.masked_fill(~pos,-1e4),dim=1)
            loss=(logden[valid]-logpos[valid]).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    model.eval()
    return model,{'tau':tau,'dev_cross_task_positive_rates':rates,'epochs':8,'feature_mean':mu,'feature_std':sd}


def _project_features(model,feat,mean,std):
    import torch
    model.eval()
    with torch.no_grad():return torch.nn.functional.normalize(model(torch.as_tensor((feat-mean)/std)),dim=-1).cpu().numpy()


def run(out=OUT,threads=8,skip_camera=False,feature_cache_source=None,camera_cache_source=None):
    allowed_partial={'projection_overlay.png','token_grid_order.png','camera_matrices.npz','local_features.npz'}
    if out.exists() and any(p.name not in allowed_partial for p in out.iterdir()):raise FileExistsError(f'refusing output overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['OMP_NUM_THREADS']=str(threads);os.environ['MKL_NUM_THREADS']=str(threads);os.environ['OPENBLAS_NUM_THREADS']=str(threads)
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (RUN/'rows').glob('*.npy')}
    ids=np.arange(len(rows['task']));pool=np.flatnonzero(np.asarray(rows['memory'],bool)&(np.asarray(rows['task'])>=10))
    gap=np.load(GAP/'predictions_and_errors.npz',allow_pickle=False);top=np.load(TOPK/'predictions_and_errors.npz',allow_pickle=False)
    long_ids=gap['row_ids'][np.asarray(rows['task'][gap['row_ids']])<10];hard=top['hard_mask']
    if len(pool)!=3857 or len(long_ids)!=548 or int(hard.sum())!=63 or not np.array_equal(long_ids,top['query_rows']):raise AssertionError('fixed pool/Long split mismatch')
    if abs(float(json.loads((GAP/'summary_corrected.json').read_text())['threshold'])-Q80)>1e-10:raise AssertionError('fixed q80 changed')
    memids=np.load(MEMORY/'units.npz',allow_pickle=False)['representative_row_ids']
    if not np.array_equal(pool,memids):raise AssertionError('memory row IDs differ from raw-rows-noLong')
    row_train=np.flatnonzero((np.asarray(rows['task'])>=10)&~np.asarray(rows['memory'],bool)&~np.asarray(rows['validation'],bool))
    val_no_long=np.flatnonzero((np.asarray(rows['task'])>=10)&np.asarray(rows['validation'],bool))
    policy=_load_module('local_key_pi05_run',Path(__file__).resolve().parents[1]/'run.py')
    dev_eps,_=policy.calibration_episodes(rows)
    dev_ids=val_no_long[np.isin(np.asarray(rows['episode'][val_no_long]),dev_eps)]
    _assert_no_long(np.r_[pool,row_train],dev_ids,rows)
    all_required=np.unique(np.r_[pool,row_train,dev_ids,long_ids])
    norm=json.loads(next(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero/assets').rglob('norm_stats.json')).read_text())['norm_stats']
    state_q01=np.asarray(norm['state']['q01'][:8],np.float32);state_q99=np.asarray(norm['state']['q99'][:8],np.float32)
    if skip_camera:raise NotImplementedError('camera projection is required for the final run')
    task_texts=json.loads((RUN/'data_manifest.json').read_text())['tasks']
    camera_source=Path(camera_cache_source) if camera_cache_source else out/'camera_matrices.npz'
    cameras=_camera_transforms(task_texts,camera_source)
    feature_cache=Path(feature_cache_source) if feature_cache_source else out/'local_features.npz'
    if feature_cache.exists():
        zcache=np.load(feature_cache,allow_pickle=False)
        if not np.array_equal(zcache['row_ids'],all_required):raise AssertionError('local feature cache row IDs differ')
        feats={k:np.asarray(zcache[f'feat_{k}']) for k in ('G','W',*REGIONS,'base_ee3','base_ee5')}
        points={int(r):p for r,p in zip(zcache['point_rows'],zcache['points'],strict=True)}
    else:
        feats,points=_extract_keys(rows,all_required,cameras,state_q01,state_q99)
        point_rows=np.asarray(sorted(points),np.int64)
        np.savez(feature_cache,row_ids=all_required,point_rows=point_rows,points=np.stack([points[int(r)] for r in point_rows]),
                 **{f'feat_{k}':v for k,v in feats.items()})
    where={int(r):i for i,r in enumerate(all_required)}
    def take(name,ix):return feats[name][np.asarray([where[int(r)] for r in ix])]
    _plot_grid(out,rows,long_ids[np.unique(np.linspace(0,len(long_ids)-1,min(4,len(long_ids))).astype(int))],points)
    # Draw numbered 16x16 row-major grid as an explicit patch/image alignment check.
    from PIL import Image,ImageDraw
    preview=Image.fromarray(np.asarray(rows['image_wrist'][int(pool[0])],np.uint8)).resize((448,448));dr=ImageDraw.Draw(preview)
    for i in range(17):
        x=i*28;dr.line((x,0,x,447),fill=(220,40,40),width=1);dr.line((0,x,447,x),fill=(220,40,40),width=1)
    preview.save(out/'token_grid_order.png',optimize=True)
    actions=np.asarray(rows['actions'][:,:,:7],np.float32)
    pool_actions=actions[pool];long_actions=actions[long_ids]
    state=np.asarray(rows['state'][:,:8],np.float32);hist=_history(rows)
    train_features={};mem_features={};dev_features={};long_features={}
    # Baseline current VLM token MaxSim from the shared score-audit cache.
    current=np.load(ROOT/'score-audit-01'/'vlm-full'/'keys'/'current_task.npz',allow_pickle=False)
    crows=current['query_rows'];lookup={int(r):i for i,r in enumerate(crows)};cix=np.asarray([lookup[int(r)] for r in long_ids])
    cur_cand=current['cand_rows'][cix,:64];cur_score=current['score'][cix,:64]
    cur_mse=_candidate_mse(long_actions,actions[cur_cand])
    current_metric=_metrics('current_vlm_maxsim_full40_reference',cur_cand,cur_score,cur_mse,None,long_ids,rows)
    # The fixed visual keys are pooled and then cosine-ranked. Select W-local scale on Dev only.
    candidates=['wrist_center4','wrist_center8','wrist_lowerhalf','base_ee3','base_ee5']
    dev_action_mse=_row_mse(actions[dev_ids],pool_actions);dev_local={}
    mem_task=np.asarray(rows['task'][pool]);dev_task=np.asarray(rows['task'][dev_ids])
    for name in candidates:
        qf=take(name,dev_ids);mf=take(name,pool);ranked=_l2_normalize(qf)@_l2_normalize(mf).T
        for i,t in enumerate(dev_task):ranked[i,mem_task==t]=-np.inf
        idx=np.argmax(ranked,axis=1);dev_local[name]=float(np.mean(dev_action_mse[np.arange(len(dev_ids)),idx]))
    selected_local=min(('wrist_center4','wrist_center8','wrist_lowerhalf'),key=lambda n:dev_local[n])
    selected_visual=min(candidates,key=lambda n:dev_local[n])
    # Standardize and L2-normalize visual, state and history blocks independently.
    # Dev selects block weights only after applying the same-task exclusion used at Eval.
    base_names=['G','W',selected_local,'base_ee3','base_ee5']
    state_parts={};hist_parts={}
    state_tr,state_other,state_stats=_fit_key_block(state[row_train],np.concatenate([state[dev_ids],state[pool],state[long_ids]]))
    hist_tr,hist_other,hist_stats=_fit_key_block(hist[row_train],np.concatenate([hist[dev_ids],hist[pool],hist[long_ids]]))
    nd,nm=len(dev_ids),len(pool)
    state_parts={'train':state_tr,'dev':state_other[:nd],'mem':state_other[nd:nd+nm],'query':state_other[nd+nm:]}
    hist_parts={'train':hist_tr,'dev':hist_other[:nd],'mem':hist_other[nd:nd+nm],'query':hist_other[nd+nm:]}
    results={};allmse=_row_mse(long_actions,pool_actions);dev_configs={}
    # Candidate weights are predeclared and tuned only on nonLong Dev.
    weight_grid=[('visual',(1.,0.,0.)),
                 ('state',(0.75,0.25,0.)),('state',(0.50,0.50,0.)),('state',(0.25,0.75,0.)),
                 ('prev',(0.75,0.,0.25)),('prev',(0.50,0.,0.50)),('prev',(0.25,0.,0.75)),
                 ('state_prev',(0.60,0.20,0.20)),('state_prev',(0.40,0.30,0.30)),('state_prev',(0.20,0.40,0.40))]
    composed={}
    for name in base_names:
        raw_parts={'train':_l2_normalize(take(name,row_train)),
                   'dev':_l2_normalize(take(name,dev_ids)),
                   'mem':_l2_normalize(take(name,pool)),
                   'query':_l2_normalize(take(name,long_ids))}
        vtr,vother,_=_fit_key_block(take(name,row_train),np.concatenate([take(name,dev_ids),take(name,pool),take(name,long_ids)]))
        standardized_parts={'train':vtr,'dev':vother[:nd],'mem':vother[nd:nd+nm],'query':vother[nd+nm:]}
        for suffix,w in weight_grid:
            keyname=name if suffix=='visual' else f'{name}+{suffix}'
            allparts=(raw_parts if suffix=='visual' else
                      {role:_join_key_blocks(standardized_parts[role],state_parts[role],hist_parts[role],w)
                       for role in ('train','dev','mem','query')})
            # Explicitly same-task-excluded Dev selection for every candidate weighting.
            sims=allparts['dev']@allparts['mem'].T
            for i,t in enumerate(dev_task):sims[i,mem_task==t]=-np.inf
            ix=np.argmax(sims,axis=1)
            score=float(dev_action_mse[np.arange(nd),ix].mean())
            dev_configs[(name,suffix,w)]=score
            if suffix=='visual' or keyname not in composed or score<composed[keyname]['dev_mse']:
                composed[keyname]={'parts':allparts,'weights':w,'dev_mse':score,'visual':name}
    # Select the best local region from the visual-only Dev scores, as before.
    local_visual_names={selected_local,'base_ee3','base_ee5'}
    selected_key=min((n for n,cfg in composed.items() if cfg['visual'] in local_visual_names),
                     key=lambda n:composed[n]['dev_mse'])
    # Compare per visual block and augmentation, using each variant's Dev-selected weights.
    results['current_vlm_maxsim_full40_reference']={**current_metric,'pool_note':'cached current_task key uses the original full memory bank; reference only, unlike 3,857-row conditions'}
    for vname,cfg in composed.items():
        part=cfg['parts'];cand,score,cmse=_retrieve(part['query'],part['mem'],long_ids,pool,rows,actions,metric='cos',exclusion='task',k=64)
        met=_metrics(vname,cand,score,cmse,allmse,long_ids,rows)
        met['block_weights']={'visual':cfg['weights'][0],'state':cfg['weights'][1],'history':cfg['weights'][2]}
        met['dev_top1_mse']=cfg['dev_mse'];results[vname]=met
    # The selected local retrieval feature and direct regressor receive identical block composition.
    selected_config=composed[selected_key];best_input=selected_key
    tx=selected_config['parts']['train'];dx=selected_config['parts']['dev'];mx=selected_config['parts']['mem'];qx=selected_config['parts']['query']
    selected_visual=selected_config['visual'];train_vis=take(selected_visual,row_train);dev_vis=take(selected_visual,dev_ids)
    mem_vis=take(selected_visual,pool);long_vis=take(selected_visual,long_ids)
    import torch
    torch.set_num_threads(threads)
    direct,dm=_fit_direct(tx,actions[row_train],dx,actions[dev_ids],threads)
    pred=_predict_direct(direct,qx,dm['mean'],dm['std']);direct_mse=np.mean((pred-long_actions)**2,axis=(1,2))
    # Train a supervised-contrastive projection from the same local visual key. Labels are action distances only.
    proj,pm=_fit_projection(train_vis,actions[row_train],np.asarray(rows['task'][row_train]),dev_vis,actions[dev_ids],dev_task,threads)
    train_mu=pm.pop('feature_mean');train_sd=pm.pop('feature_std')
    p_mem=_project_features(proj,mem_vis,train_mu,train_sd);p_long=_project_features(proj,long_vis,train_mu,train_sd)
    pcand,pscore,pmse=_retrieve(p_long,p_mem,long_ids,pool,rows,actions,metric='cos',exclusion='task',k=64)
    results['invariant_projected_local']={**_metrics('invariant_projected_local',pcand,pscore,pmse,allmse,long_ids,rows),'tau':pm['tau'],'dev_positive_rates':pm['dev_cross_task_positive_rates']}
    rank_parity={}
    for name in ('G','W','wrist_lowerhalf','invariant_projected_local'):
        old_path=ROOT/'local-key-01'/f'ranks_{name}.npz'
        if old_path.exists():
            old=np.load(old_path,allow_pickle=False)['cand_rows']
            now=np.asarray(results[name]['candidate_rows'],np.int64)[:,0]
            if old.ndim==2:old=old[:,0]
            equal=bool(np.array_equal(old,now))
            if not equal:raise AssertionError(f'candidate ranking changed for {name}; old local-key-01 parity failed')
            rank_parity[name]={'matches_local_key_01_top1_rows':True,
                               'corrected_top1_mse_mean':float(np.mean(results[name]['top1_mse']))}
    results['direct_regression']={'name':'direct_regression','input':best_input,'mse':direct_mse.tolist(),'dev_fit':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in dm.items()}}
    # Baselines and per-query outputs.
    oracle=allmse.min(1);zero=np.asarray(top['zero_mse'][0,0],np.float32);normal=np.asarray(top['normal_single_mse'],np.float32)
    if len(normal)!=len(long_ids):raise AssertionError('normal baseline query alignment mismatch')
    all_metrics=[]
    for name,met in results.items():
        if name=='direct_regression':
            vals=np.asarray(met['mse']);metric_row={'name':name,'top1_mse':vals.tolist()}
        else:metric_row=met
        top1=np.asarray(metric_row['top1_mse'],np.float32)
        all_metrics.append({'name':name,'all_mean':float(top1.mean()),'hard_mean':float(top1[hard].mean()),
          'all_top5_best_mean':float(np.mean(np.asarray(met.get('top5_best_mse',top1)))) if name!='direct_regression' else None,
          'hard_top5_best_mean':float(np.mean(np.asarray(met.get('top5_best_mse',top1))[hard])) if name!='direct_regression' else None,
          'all_recall8':float(np.mean(np.asarray(met.get('recall8',np.zeros(len(long_ids))))[np.asarray(met.get('query_positive',np.zeros(len(long_ids),bool))) ])) if name!='direct_regression' and np.any(np.asarray(met.get('query_positive',np.zeros(len(long_ids),bool)))) else None,
          'hard_recall8':float(np.mean(np.asarray(met.get('recall8',np.zeros(len(long_ids))))[hard & np.asarray(met.get('query_positive',np.zeros(len(long_ids),bool))) ])) if name!='direct_regression' and np.any(hard & np.asarray(met.get('query_positive',np.zeros(len(long_ids),bool)))) else None,
          'all_spearman':float(np.nanmean(np.asarray(met.get('spearman_top64',np.full(len(long_ids),np.nan))))) if name!='direct_regression' else None,
          'hard_spearman':float(np.nanmean(np.asarray(met.get('spearman_top64',np.full(len(long_ids),np.nan)))[hard])) if name!='direct_regression' else None,
          'block_weights':met.get('block_weights'),'dev_top1_mse':met.get('dev_top1_mse')})
    # Episode bootstrap paired differences: select the Dev-chosen local key and matched direct regressor.
    primary_name=selected_key
    primary=np.asarray(results[primary_name]['top1_mse'],np.float32)
    local_minus_direct=primary-direct_mse
    comparison={'L1_local_minus_G_all':_episode_bootstrap(primary-np.asarray(results['G']['top1_mse']),rows['episode'][long_ids]),
      'L1_local_minus_G_hard':_episode_bootstrap((primary-np.asarray(results['G']['top1_mse']))[hard],rows['episode'][long_ids][hard]),
      'L2_local_minus_direct_all':_episode_bootstrap(local_minus_direct,rows['episode'][long_ids]),
      'L2_local_minus_direct_hard':_episode_bootstrap(local_minus_direct[hard],rows['episode'][long_ids][hard])}
    local_key_cis={}
    for lname,lmet in results.items():
        if lname.startswith(('wrist_','base_ee')) or lname=='invariant_projected_local':
            delta=np.asarray(lmet['top1_mse'],np.float32)-np.asarray(results['G']['top1_mse'],np.float32)
            local_key_cis[lname]={'all':_episode_bootstrap(delta,rows['episode'][long_ids]),
                                  'hard':_episode_bootstrap(delta[hard],rows['episode'][long_ids][hard])}
    comparison['L1_all_local_candidates_vs_G']=local_key_cis
    # All-task oracle, policy, and zero controls.
    all_metrics += [{'name':'action_oracle_all_memory','all_mean':float(oracle.mean()),'hard_mean':float(oracle[hard].mean())},
      {'name':'zero_draft','all_mean':float(zero.mean()),'hard_mean':float(zero[hard].mean())},
      {'name':'normal_pi05','all_mean':float(normal.mean()),'hard_mean':float(normal[hard].mean())}]
    # Persist small arrays, learned weights and complete per-query rank metrics.
    np.savez_compressed(out/'predictions.npz',query_rows=long_ids,hard_mask=hard,oracle_mse=oracle,zero_mse=zero,normal_mse=normal,
      direct_mse=direct_mse,local_top1=primary,local_name=np.asarray(primary_name),episode=np.asarray(rows['episode'][long_ids]),
      direct_action=pred.astype(np.float32))
    torch.save({'direct':direct.state_dict(),'direct_input':best_input,'direct_fit':dm,'projection':proj.state_dict(),
                'projection_fit':{'feature_mean':train_mu,'feature_std':train_sd,**pm}},out/'models.pt')
    for name,met in results.items():
        if name=='direct_regression':continue
        np.savez_compressed(out/f'ranks_{name.replace("+","_").replace("[","_").replace("]","").replace(",","_")}.npz',cand_rows=np.asarray(met['candidate_rows'],np.int64),
          top1_mse=np.asarray(met['top1_mse'],np.float32),top5_best_mse=np.asarray(met['top5_best_mse'],np.float32),
          recall8=np.asarray(met['recall8'],np.float32),spearman=np.asarray(met['spearman_top64'],np.float32))
    protocol={'pool_rows':len(pool),'pool_source':str(MEMORY),'pool_matches_existing_raw_rows_noLong':True,
      'pool_task_ids':list(range(10,40)),'long_eval_rows':len(long_ids),'long_hard_rows':int(hard.sum()),'hard_threshold':Q80,
      'train_rows':len(row_train),'train_tasks':[int(x) for x in np.unique(rows['task'][row_train])],
      'dev_rows':len(dev_ids),'dev_episode_count':len(np.unique(rows['episode'][dev_ids])),
      'long_rows_used_for_fit_or_selection':False,'long_task_excluded_from_memory':True,
      'fixed_key_dev_scores':dev_local,'selected_wrist_region':selected_local,'selected_local_for_learning':selected_visual,'selected_direct_input':best_input,
      'projected_key':pm,'key_layout':'tokens first 256 base row-major 16x16; next 256 wrist row-major 16x16; text starts at 512',
      'projection':'LIBERO agentview world-to-pixel matrix per task; EEF state denormalized from pi05 q01/q99, pixel coordinate rotated 180 degrees then resized 256->224',
      'attention_key':'deferred: obtaining per-patch action-expert cross-attention is not exposed by saved token cache; no new VLM forward was run',
      'normal_metric':'mean active7 normalized action MSE; Top-1, Top-5 best, Recall@8 positives MSE<=0.01, per-query Spearman over 64; task exclusion',
      'baseline_current':'reused current_task Top-64 score-audit key cache; full40 bank so reference only and excluded from L1/L2',
      'bootstrap':5000,'seed':20261105,
      'CPU_threads':threads,'GPU_used':False,'long_policy_was_training_seen':True,
      'feature_cache_source':str(feature_cache),'camera_cache_source':str(camera_source),
      'key_block_normalization':'fit z-score on nonLong train rows, L2 normalize each visual/state/history block, sqrt(weight) concatenate, final L2 normalize',
      'weight_grid':{suffix:[list(w) for ss,w in weight_grid if ss==suffix] for suffix in sorted({x[0] for x in weight_grid})},
      'dev_exclusion':'same task excluded for region/weight/input selection, matching Eval',
      'selected_local_variant':selected_key,'selected_local_weights':composed[selected_key]['weights']}
    protocol['legacy_top1_rank_parity']=rank_parity
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    detail={'protocol':protocol,'metrics':all_metrics,'comparisons':comparison,'per_method':results,
      'oracle_mse':oracle.tolist(),'dev_region_selection':dev_local,'direct_regressor':results['direct_regression']['dev_fit']}
    (out/'summary.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    return detail


def postprocess_summary(out=OUT):
    """Recompute saved retrieval metrics against dataset row IDs and fix recall@8."""
    p=Path(out)/'summary.json';x=json.loads(p.read_text());pred=np.load(Path(out)/'predictions.npz',allow_pickle=False)
    hard=np.asarray(pred['hard_mask'],bool);episodes=np.asarray(pred['episode'],np.int64);per=x['per_method']
    rows={f.stem:np.load(f,mmap_mode='r',allow_pickle=False) for f in (RUN/'rows').glob('*.npy')}
    actions=np.asarray(rows['actions'][:,:,:7],np.float32);qids=np.asarray(pred['query_rows'],np.int64)
    pool=np.flatnonzero(np.asarray(rows['memory'],bool)&(np.asarray(rows['task'])>=10))
    all_mse=_row_mse(actions[qids],actions[pool]);eligible=(all_mse<=.01).any(1)
    metric_map={r['name']:r for r in x['metrics']}
    for name,rec in per.items():
        if name not in metric_map or 'candidate_rows' not in rec:continue
        cand=np.asarray(rec['candidate_rows'],np.int64)
        cand_mse=_candidate_mse(actions[qids],actions[cand])
        den=(all_mse<=.01).sum(1)
        recall=np.divide((cand_mse[:,:8]<=.01).sum(1),den,out=np.zeros(len(qids),np.float32),where=den>0)
        rec['recall8']=recall.tolist();rec['query_positive']=eligible.tolist()
        rec['top1_mse']=cand_mse[:,0].tolist();rec['top5_best_mse']=cand_mse[:,:5].min(1).tolist()
        metric_map[name]['all_recall8']=float(recall[eligible].mean()) if np.any(eligible) else None
        metric_map[name]['hard_recall8']=float(recall[hard & eligible].mean()) if np.any(hard & eligible) else None
        metric_map[name]['all_mean']=float(cand_mse[:,0].mean())
        metric_map[name]['hard_mean']=float(cand_mse[hard,0].mean())
        metric_map[name]['all_top5_best_mean']=float(cand_mse[:,:5].min(1).mean())
        metric_map[name]['hard_top5_best_mean']=float(cand_mse[hard,:5].min(1).mean())
        file=Path(out)/f"ranks_{name.replace('+','_').replace('[','_').replace(']','').replace(',','_')}.npz"
        np.savez_compressed(file,cand_rows=cand,top1_mse=cand_mse[:,0],top5_best_mse=cand_mse[:,:5].min(1),
                            recall8=recall,spearman=np.asarray(rec['spearman_top64'],np.float32))
    g=np.asarray(per['G']['top1_mse'],np.float32);local={}
    for name,rec in per.items():
        if name.startswith(('wrist_','base_ee')) or name=='invariant_projected_local':
            delta=np.asarray(rec['top1_mse'],np.float32)-g
            local[name]={'all':_episode_bootstrap(delta,episodes),'hard':_episode_bootstrap(delta[hard],episodes[hard])}
    x['comparisons']['L1_all_local_candidates_vs_G']=local
    x['protocol']['recall8_aggregation']='Recall@8: relevant retrieved rows among first 8 ranked candidates divided by all relevant rows in the 3,857-row pool; mean over queries with at least one relevant item (action MSE<=0.01)'
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--postprocess',action='store_true')
    ap.add_argument('--feature-cache-source',type=Path);ap.add_argument('--camera-cache-source',type=Path)
    a=ap.parse_args()
    if a.threads<1 or a.threads>8:raise ValueError('threads must be 1..8')
    if a.postprocess:
        result=postprocess_summary(a.output);print(json.dumps({'summary':str(a.output/'summary.json'),'metrics':len(result['metrics'])},ensure_ascii=False));return
    result=run(a.output,a.threads,feature_cache_source=a.feature_cache_source,camera_cache_source=a.camera_cache_source);print(json.dumps({'output':str(a.output),'methods':len(result['metrics'])},ensure_ascii=False))

if __name__=='__main__':main()
