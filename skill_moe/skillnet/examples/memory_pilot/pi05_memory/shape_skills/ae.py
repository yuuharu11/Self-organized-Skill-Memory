#!/usr/bin/env python3
"""Masked variable-length autoencoders for shape-skills A (R5).

Padding is represented only by a boolean mask. Sequence length is also passed
as an explicit scalar to both encoder and decoder. Bundles follow FORMAT.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as shape_data
import representations as shape_repr

RUN_DIR = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
ROOT_OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/shape-skills-01')
VARLEN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/varlen-01/final-04/segments.npz')
VQ_REP = ROOT_OUT / 'representations'
GAP = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/compress-vs-raw-review-20261105-01/drafts_prepared.npz')
HELDOUT = set(shape_data.HELDOUT_TASKS)
LONG = set(shape_data.LONG_TASKS)
LATENT_DIMS = (8, 16, 24, 32)
SEGMENTATIONS = ('S1_greedy_e0p01', 'S1_dp_e0p01', 'S2_grip_velocity')
VARIANTS = ('raw', 'procrustes_shape', 'raw_invariant')
KAPPAS = (.025, .05, .1, .2, .4, .8)
SEED = 20261106


def _torch_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MaskedAE(nn.Module):
        def __init__(self, input_dim=7, output_dim=7, latent_dim=24, width=128):
            super().__init__()
            self.input_dim, self.output_dim, self.latent_dim = input_dim, output_dim, latent_dim
            self.frame = nn.Sequential(nn.Linear(input_dim + 1, width), nn.GELU(),
                                       nn.Linear(width, width), nn.GELU())
            self.length_encoder = nn.Sequential(nn.Linear(1, 16), nn.GELU())
            self.to_z = nn.Sequential(nn.Linear(width + 16, width), nn.GELU(),
                                      nn.Linear(width, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim + 2, width), nn.GELU(),
                                         nn.Linear(width, width), nn.GELU(),
                                         nn.Linear(width, output_dim))

        def encode(self, x, phase, mask, length_feature):
            h = self.frame(torch.cat([x, phase.unsqueeze(-1)], dim=-1))
            m = mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.)
            le = self.length_encoder(length_feature.reshape(-1, 1))
            return self.to_z(torch.cat([pooled, le], dim=-1))

        def decode(self, z, phase, length_feature):
            n, t = phase.shape
            zz = z[:, None, :].expand(n, t, -1)
            ll = length_feature.reshape(n, 1, 1).expand(n, t, 1)
            return self.decoder(torch.cat([zz, phase.unsqueeze(-1), ll], dim=-1))

        def forward(self, x, phase, mask, length_feature):
            z = self.encode(x, phase, mask, length_feature)
            return self.decode(z, phase, length_feature), z

    return torch, nn, F, MaskedAE


def _pad_batch(xseq, yseq, xmean, xstd, ymean, ystd, lmax, device):
    """Pad a batch with zeros after standardization; return mask/phase/true length."""
    import torch
    n = len(xseq)
    used = [min(len(x), lmax) for x in xseq]
    # Every tensor uses the train-selected global ceiling. The padded suffix
    # is explicit zero and is excluded by mask in both pooling and loss.
    tmax = int(lmax)
    x = np.zeros((n, tmax, xseq[0].shape[1]), np.float32)
    y = np.zeros((n, tmax, yseq[0].shape[1]), np.float32)
    phase = np.zeros((n, tmax), np.float32)
    mask = np.zeros((n, tmax), bool)
    lf = np.zeros((n, 1), np.float32)
    for i, (xi, yi, nu) in enumerate(zip(xseq, yseq, used, strict=True)):
        if len(xi) > lmax:
            xi = shape_repr.resample(xi, lmax)
            yi = shape_repr.resample(yi, lmax)
        x[i, :nu] = (np.asarray(xi[:nu], np.float32) - xmean) / xstd
        y[i, :nu] = (np.asarray(yi[:nu], np.float32) - ymean) / ystd
        phase[i, :nu] = np.linspace(0., 1., nu, dtype=np.float32) if nu > 1 else 0.
        mask[i, :nu] = True
        lf[i, 0] = np.log1p(len(xseq[i])) / np.log1p(lmax)
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device),
            torch.as_tensor(phase, device=device), torch.as_tensor(mask, device=device),
            torch.as_tensor(lf, device=device))


def masked_mse(pred, target, mask):
    """Per-item mean squared error over valid timesteps and channels only."""
    import torch
    m = mask.to(pred.dtype).unsqueeze(-1)
    return (((pred - target).square() * m).sum((1, 2)) /
            (m.sum((1, 2)) * pred.shape[-1]).clamp_min(1.))


def choose_lmax(interval_lengths, *, percentile=.99, minimum=10):
    lengths=np.asarray(interval_lengths,np.int64)
    if lengths.ndim!=1 or not len(lengths) or np.any(lengths<1):
        raise ValueError('interval lengths must be a nonempty positive vector')
    p=float(np.quantile(lengths,float(percentile)))
    return max(int(minimum),int(math.ceil(p-1e-9))),p


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q.astype(np.float32)


def augment_physical(seq, rng, *, scale_range=(.7, 1.3)):
    """Rotate position/rotation deltas in physical space and scale; grip is fixed."""
    a = np.asarray(seq, np.float32).copy()
    R = _random_rotation(rng)
    scale = float(rng.uniform(*scale_range))
    a[:, :3] = (a[:, :3] @ R.T) * scale
    a[:, 3:6] = (a[:, 3:6] @ R.T) * scale
    return a, {'rotation': R, 'scale': scale}


def _load_splits_and_segments():
    """Build every segment split with the reviewed, rotation-corrected A code."""
    checkpoint = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
    episodes, rows, source_meta = shape_data.load_source(RUN_DIR, checkpoint)
    import run as shape_run
    segout = Path('/tmp/shape-skills-ae-segmentation-progress.jsonl')
    if segout.exists():
        segout = segout.with_name(f'{segout.stem}-{os.getpid()}{segout.suffix}')
    raw_sets, stats, thresholds = shape_run.make_segment_sets(episodes, rows, segout)
    segs = {}
    for name, values in raw_sets.items():
        segs[name] = [dict(task=int(s['task']), episode=int(s['episode']),
                            start=int(s['start']), end=int(s['start'] + s['length']),
                            row_id=int(s['row_id']), meta=s['meta']) for s in values]
    segment_meta = {'segmentation_stats': stats, 'thresholds_selected_on_heldout_dev': thresholds,
                    'source': 'shape_skills.run.make_segment_sets; shared reviewed A segmentation'}
    return episodes, rows, source_meta, segs, segment_meta


def _is_train(seg, episodes):
    if seg['task'] in LONG or seg['task'] in HELDOUT:
        return False
    ep = episodes[(seg['task'], seg['episode'])]
    return bool(ep['meta'].get('present',False) and not ep['meta']['memory'] and not ep['meta']['validation'])


def _is_memory(seg, episodes):
    if seg['task'] in LONG or seg['task'] in HELDOUT:
        return False
    ep=episodes[(seg['task'],seg['episode'])]
    return bool(ep['meta'].get('present',False) and ep['meta']['memory'])


def _seq(seg, episodes):
    ep=episodes[(seg['task'],seg['episode'])]
    return np.asarray(ep['actions_phys'][seg['start']:seg['end'],:7],np.float32)


def canonicalize_7(actions_phys, reference):
    """Canonicalize physical xyz/rotation and preserve gripper unchanged."""
    a=np.asarray(actions_phys,np.float32)
    can,pose=shape_repr.canonicalize(a,procrustes_ref=reference)
    grip=shape_repr.resample(a[:,6:7],len(a))
    return np.concatenate([can,grip],axis=1),pose


def _make_procrustes_reference(train_segments, episodes):
    paths=[]
    for s in train_segments:
        a=_seq(s,episodes)
        can,_=shape_repr.canonicalize(a)
        p=shape_repr.resample(np.cumsum(can[:,:3],axis=0),shape_repr.PHASE)
        paths.append(p-p.mean(0,keepdims=True))
    if not paths:raise ValueError('no train segments for Procrustes reference')
    ref=np.mean(paths,axis=0).astype(np.float32)
    ref-=ref.mean(0,keepdims=True)
    return ref


def _prepare(segs, episodes, variant, q01, q99, reference, lmax):
    """Build physical source/target sequences and per-segment pose sidecars."""
    xs,ys,poses=[],[],[]
    for s in segs:
        a=_seq(s,episodes)
        if len(a)<1:continue
        if variant=='raw':
            x=shape_data.normalize_physical(a,q01[:7],q99[:7]); y=x.copy()
            pose={'scale':1.,'rotation':np.eye(3,dtype=np.float32),'length':len(a)}
        elif variant=='procrustes_shape':
            y,pose=canonicalize_7(a,reference)
            x=y.copy()
        elif variant=='raw_invariant':
            x=a.copy();y,pose=canonicalize_7(a,reference)
        else:raise ValueError(variant)
        xs.append(np.asarray(x,np.float32));ys.append(np.asarray(y,np.float32));poses.append(pose)
    return xs,ys,poses


def _valid_stats(sequences):
    joined=np.concatenate([np.asarray(s,np.float32) for s in sequences],axis=0)
    return joined.mean(0).astype(np.float32),np.maximum(joined.std(0),1e-5).astype(np.float32)


def fit_ae(xseq, yseq, *, latent_dim, lmax, epochs=30, batch_size=128, seed=SEED,
           device='cuda', threads=8, invariant=False, inv_weight=.1, progress=None):
    torch,nn,F,Model=_torch_classes();torch.set_num_threads(int(threads))
    if str(device).startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.20,0)
    if len(xseq)<2:raise ValueError('AE fit requires at least two segments')
    np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    xmean,xstd=_valid_stats(xseq);ymean,ystd=_valid_stats(yseq)
    model=Model(7,7,latent_dim).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    model.train();logs=[];start=time.monotonic()
    for ep in range(int(epochs)):
        order=rng.permutation(len(xseq));ep_loss=[]
        for lo in range(0,len(order),batch_size):
            ids=order[lo:lo+batch_size]
            xb,yb,phase,mask,lf=_pad_batch([xseq[i] for i in ids],[yseq[i] for i in ids],
                xmean,xstd,ymean,ystd,lmax,device)
            opt.zero_grad(set_to_none=True)
            if invariant:
                # Augment raw physical observations. Target remains the
                # Procrustes shape, so z is asked to discard scale and pose.
                aug=[]
                for i in ids:
                    xa,_=augment_physical(xseq[i],rng)
                    aug.append(xa)
                x2,_,phase2,mask2,lf2=_pad_batch(aug,[yseq[i] for i in ids],
                    xmean,xstd,ymean,ystd,lmax,device)
                p1,z1=model(xb,phase,mask,lf);p2,z2=model(x2,phase2,mask2,lf2)
                rec=.5*(masked_mse(p1,yb,mask).mean()+masked_mse(p2,yb,mask2).mean())
                consistency=(F.normalize(z1,dim=-1)-F.normalize(z2,dim=-1)).square().mean()
                loss=rec+float(inv_weight)*consistency
            else:
                pred,z=model(xb,phase,mask,lf)
                rec=masked_mse(pred,yb,mask).mean();consistency=rec.new_zeros(())
                loss=rec
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            ep_loss.append((float(rec.detach().cpu()),float(consistency.detach().cpu())))
        log={'epoch':ep+1,'reconstruction':float(np.mean([v[0] for v in ep_loss])),
             'invariance':float(np.mean([v[1] for v in ep_loss])),'elapsed_s':time.monotonic()-start}
        logs.append(log)
        if progress and ((ep+1)%5==0 or ep+1==epochs):
            with Path(progress).open('a') as f:f.write(json.dumps(log)+'\n')
    model.eval()
    return model,{'xmean':xmean,'xstd':xstd,'ymean':ymean,'ystd':ystd,'logs':logs,
                  'fit_seconds':time.monotonic()-start,'device':str(device)}


def _encode(model, seqs, xmean,xstd,lmax,device='cuda',batch_size=128):
    torch,_,_,_=_torch_classes();model.eval();outs=[]
    with torch.no_grad():
        for lo in range(0,len(seqs),batch_size):
            ss=seqs[lo:lo+batch_size]
            # dummy target is shape-compatible; unused by encode.
            xb,_,ph,mask,lf=_pad_batch(ss,ss,xmean,xstd,xmean,xstd,lmax,device)
            z=model.encode(xb,ph,mask,lf);outs.append(z.cpu().numpy())
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,model.latent_dim),np.float32)


def _decode(model,z,lengths,poses,variant,stats,q01,q99,lmax,device='cuda',batch_size=128):
    torch,_,_,_=_torch_classes();model.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(z),batch_size):
            zz=torch.as_tensor(np.asarray(z[lo:lo+batch_size],np.float32),device=device)
            lens=np.asarray(lengths[lo:lo+batch_size],np.int64);n=len(lens);t=int(lens.max())
            ph=torch.as_tensor(np.tile(np.linspace(0.,1.,t,dtype=np.float32),(n,1)),device=device)
            lf=torch.as_tensor((np.log1p(lens)/np.log1p(lmax)).astype(np.float32)[:,None],device=device)
            pred=model.decode(zz,ph,lf).cpu().numpy()
            for j in range(n):
                y=pred[j,:t]*stats['ystd']+stats['ymean']
                y=shape_repr.resample(y,int(lens[j]))
                if variant=='raw':
                    decoded=y
                else:
                    decoded=shape_repr.apply_pose(y,poses[lo+j])
                    decoded=shape_data.normalize_physical(decoded,q01[:7],q99[:7])
                out.append(np.asarray(decoded,np.float32))
    return out


def _pca_reference(train_y, query_y, query_poses, variant, q01,q99, lengths):
    def phase_vec(seq):return shape_repr.resample(seq,shape_repr.PHASE).reshape(-1)
    tr=np.stack([phase_vec(x) for x in train_y]);pca=shape_repr.pca_fit(tr,24)
    vals=[]
    for i,seq in enumerate(query_y):
        code=shape_repr.pca_encode(pca,phase_vec(seq));decoded=shape_repr.pca_decode(pca,code).reshape(shape_repr.PHASE,7)
        decoded=shape_repr.resample(decoded,int(lengths[i]))
        if variant!='raw':
            decoded=shape_repr.apply_pose(decoded,query_poses[i]);decoded=shape_data.normalize_physical(decoded,q01[:7],q99[:7])
        vals.append(decoded)
    return vals


def _save_model_bundle(folder,model,stats,config,codes,segments,poses,q01,q99,fit_tasks,lmax):
    folder=Path(folder)
    if folder.exists() and any(folder.iterdir()):raise FileExistsError(f'refusing to overwrite {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'codes.npy',np.asarray(codes,np.float32))
    np.savez_compressed(folder/'segments.npz',task=np.asarray([s['task'] for s in segments],np.int32),
        episode=np.asarray([s['episode'] for s in segments],np.int32),start=np.asarray([s['start'] for s in segments],np.int32),
        length=np.asarray([s['end']-s['start'] for s in segments],np.int32),row_ids=np.asarray([s['row_id'] for s in segments],np.int64))
    np.savez_compressed(folder/'pose.npz',scale=np.asarray([p['scale'] for p in poses],np.float32),
        rotation=np.asarray([p['rotation'] for p in poses],np.float32),length=np.asarray([p['length'] for p in poses],np.int32))
    state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    cfg={**config,'kind':'masked_variable_length_ae','version':1,'latent_dim':int(config['latent_dim']),
        'input_dim':7,'output_dim':7,'lmax':int(lmax),'fit_tasks':fit_tasks,
        'fit_excludes_tasks':sorted(LONG|HELDOUT),'length_input':'log1p(original_frames)/log1p(Lmax)',
        'padding':'zero after standardization; masked from encoder pooling and loss',
        'nucleus_definition':'continuous AE latent; not an observed segment',
        'q01':np.asarray(q01[:7]).tolist(),'q99':np.asarray(q99[:7]).tolist()}
    np.savez_compressed(folder/'model.npz',**{'weight__'+k:v for k,v in state.items()},
        xmean=stats['xmean'],xstd=stats['xstd'],ymean=stats['ymean'],ystd=stats['ystd'],
        config_json=np.asarray(json.dumps(cfg,ensure_ascii=False)))
    (folder/'manifest.json').write_text(json.dumps({**cfg,'segments':len(segments),'code_dim':int(config['latent_dim']),
        'normalization':'run-01 official q01/q99 then per-channel train-only standardization'},ensure_ascii=False,indent=2)+'\n')
    (folder/'reconstruct.py').write_text('''from pathlib import Path\nfrom ae import decode_bundle\nBUNDLE=Path(__file__).resolve().parent\ndef decode(codes,pose,length): return decode_bundle(BUNDLE,codes,pose,length)\n''')


def decode_bundle(bundle_dir,codes,pose,length):
    torch,_,_,Model=_torch_classes();d=Path(bundle_dir)
    f=np.load(d/'model.npz',allow_pickle=False);cfg=json.loads(str(f['config_json'].item()))
    model=Model(7,7,int(cfg['latent_dim']))
    model.load_state_dict({k:torch.as_tensor(f['weight__'+k]) for k in model.state_dict()});model.eval()
    stats={k:np.asarray(f[k],np.float32) for k in ('xmean','xstd','ymean','ystd')}
    c=np.asarray(codes,np.float32);c=c[None] if c.ndim==1 else c
    p={k:np.asarray(v) for k,v in pose.items()}
    n=len(c);lens=np.full(n,int(length),np.int64) if np.asarray(length).ndim==0 else np.asarray(length,np.int64)
    q01=np.asarray(cfg['q01'],np.float32);q99=np.asarray(cfg['q99'],np.float32);out=[]
    with torch.no_grad():
        for i in range(n):
            l=max(1,int(lens[i]));phase=torch.as_tensor(np.linspace(0,1,l,dtype=np.float32)[None])
            lf=torch.as_tensor([[np.log1p(l)/np.log1p(cfg['lmax'])]],dtype=torch.float32)
            ci=torch.as_tensor(np.array(c[i:i+1],copy=True),dtype=torch.float32)
            y=model.decode(ci,phase,lf).numpy()[0]*stats['ystd']+stats['ymean']
            if cfg['variant']=='raw':decoded=y
            else:
                pi={k:v[i] for k,v in p.items()};decoded=shape_repr.apply_pose(y,pi);decoded=shape_data.normalize_physical(decoded,q01,q99)
            out.append(decoded.astype(np.float32))
    if len(set(lens.tolist()))==1:return np.stack(out)
    return out


def _evaluate(model,stats,variant,lmax,query_x,query_y,query_poses,query_lengths,targets,
              mem_z,mem_segments,device,q01,q99,hard,seed=SEED):
    qz=_encode(model,query_x,stats['xmean'],stats['xstd'],lmax,device=device)
    self_pred=_decode(model,qz,query_lengths,query_poses,variant,stats,q01,q99,lmax,device=device)
    self_mse=np.asarray([np.mean((a-b)**2) for a,b in zip(self_pred,targets,strict=True)],np.float32)
    m6_curve=[]
    mt=np.asarray([s['task'] for s in mem_segments],np.int32);me=np.asarray([s['episode'] for s in mem_segments],np.int32)
    for j,kappa in enumerate(KAPPAS):
        m6=shape_repr.m6_fit_generic(mem_z,mt,me,kappa=kappa,seed=SEED+j,min_support=10,min_episodes=3)
        centers=m6['centers'];members=m6['members_local']
        if not len(centers):
            m6_curve.append({'kappa':kappa,'units':0,'mse_all':None,'mse_hard':None,'shared_win':None});continue
        # Select a real memory segment medoid for every M6 unit.
        mids=np.asarray([shape_repr.medoid_generic(v,mem_z) for v in members],np.int64)
        proto=mem_z[mids];dist=((qz[:,None,:]-centers[None,:,:])**2).mean(-1);assign=np.argmin(dist,axis=1)
        selected=proto[assign]
        med_pred=_decode(model,selected,query_lengths,query_poses,variant,stats,q01,q99,lmax,device=device)
        med_err=np.asarray([np.mean((a-b)**2) for a,b in zip(med_pred,targets,strict=True)],np.float32)
        rng=np.random.default_rng(seed+j);randix=rng.integers(len(proto),size=(8,len(qz)))
        random_errors=[]
        for ri in randix:
            rp=_decode(model,proto[ri],query_lengths,query_poses,variant,stats,q01,q99,lmax,device=device)
            random_errors.append([np.mean((a-b)**2) for a,b in zip(rp,targets,strict=True)])
        random_errors=np.asarray(random_errors,np.float32)
        # A unit is shared across tasks if its M6 memory members include >=2 tasks.
        multitask=np.asarray([len(set(mt[members[u]].tolist()))>=2 for u in range(len(members))],bool)
        m6_curve.append({'kappa':float(kappa),'units':int(len(centers)),'mse_all':float(med_err.mean()),
            'mse_hard':float(med_err[hard].mean()),'coverage_004':float(np.mean(med_err<=.04)),
            'coverage_002':float(np.mean(med_err<=.02)),'random_mse_all':float(random_errors.mean()),
            'random_mse_hard':float(random_errors[:,hard].mean()),
            'nearest_beats_random_fraction':float(np.mean(med_err<random_errors.mean(0))),
            'multi_task_unit_fraction':float(multitask.mean()),
            'long_queries_using_multitask_unit':float(multitask[assign].mean()),
            'unit_task_counts':[len(set(mt[m].tolist())) for m in members]})
    return {'query_z':qz,'self_pred':self_pred,'self_mse':self_mse,'m6_curve':m6_curve}


def run_pca24_m6(*,output=ROOT_OUT,threads=8):
    """PCA24 control with the same M6 medoid/random sharing diagnostics."""
    torch,_,_,_=_torch_classes();torch.set_num_threads(int(threads))
    out=Path(output);dst=out/'ae_pca24_m6.json'
    if dst.exists():raise FileExistsError(dst)
    episodes,rows,smeta,segsets,segment_meta=_load_splits_and_segments()
    q01=np.asarray(smeta['q01'],np.float32);q99=np.asarray(smeta['q99'],np.float32)
    qprep=np.load(GAP,allow_pickle=False);qids=np.asarray(qprep['eval_query_rows'],np.int64)
    hard=np.asarray(qprep['hard_mask'],bool)
    qsegs=[{'task':int(rows['task'][r]),'episode':int(rows['episode'][r]),
        'start':int(rows['frame'][r]),'end':int(rows['frame'][r])+10,'row_id':int(r)} for r in qids]
    targets=[np.asarray(rows['actions'][r,:10,:7],np.float32) for r in qids]
    outputs=[]
    configs=[(s,v) for s in SEGMENTATIONS for v in VARIANTS]+[('S0_fixed10','raw')]
    for segname,variant in configs:
        segments=[s for s in segsets[segname] if s['end']>s['start']]
        train=[s for s in segments if _is_train(s,episodes)]
        memory=[s for s in segments if _is_memory(s,episodes)]
        if any(s['task'] in LONG|HELDOUT for s in train+memory):raise AssertionError('PCA split leakage')
        ref=_make_procrustes_reference(train,episodes) if variant!='raw' else None
        train_x,train_y,_=_prepare(train,episodes,variant,q01,q99,ref,158)
        mem_x,mem_y,_=_prepare(memory,episodes,variant,q01,q99,ref,158)
        qx,qy,qposes=_prepare(qsegs,episodes,variant,q01,q99,ref,158)
        if variant=='raw':
            train_y=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in train]
            mem_y=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in memory]
            qy=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in qsegs]
        phase_vec=lambda x:shape_repr.resample(x,shape_repr.PHASE).reshape(-1)
        pca=shape_repr.pca_fit(np.stack([phase_vec(x) for x in train_y]),24)
        mz=np.stack([shape_repr.pca_encode(pca,phase_vec(x)) for x in mem_y])
        qz=np.stack([shape_repr.pca_encode(pca,phase_vec(x)) for x in qy])
        mt=np.asarray([s['task'] for s in memory],np.int32)
        me=np.asarray([s['episode'] for s in memory],np.int32)
        curve=[]
        for ix,kappa in enumerate(KAPPAS):
            m6=shape_repr.m6_fit_generic(mz,mt,me,kappa=kappa,seed=SEED+ix,min_support=10,min_episodes=3)
            centers=m6['centers'];members=m6['members_local']
            if not len(centers):
                curve.append({'kappa':kappa,'units':0,'mse_all':None,'mse_hard':None});continue
            mids=np.asarray([shape_repr.medoid_generic(v,mz) for v in members],np.int64)
            assign=np.argmin(((qz[:,None,:]-centers[None,:,:])**2).mean(-1),axis=1)
            chosen=mz[mids[assign]]
            def decode_codes(codes):
                err=[]
                for i,z in enumerate(codes):
                    canonical=shape_repr.pca_decode(pca,z).reshape(shape_repr.PHASE,7)
                    seq=shape_repr.resample(canonical,10)
                    if variant!='raw':
                        seq=shape_repr.apply_pose(seq,qposes[i])
                        seq=shape_data.normalize_physical(seq,q01[:7],q99[:7])
                    err.append(np.mean((seq-targets[i])**2))
                return np.asarray(err,np.float32)
            mse=decode_codes(chosen);rng=np.random.default_rng(SEED+ix)
            random=np.stack([decode_codes(mz[mids[rng.integers(len(mids),size=len(qids))]])
                             for _ in range(8)])
            multitask=np.asarray([len(set(mt[members[u]].tolist()))>=2 for u in range(len(members))])
            curve.append({'kappa':float(kappa),'units':int(len(centers)),
                'mse_all':float(mse.mean()),'mse_hard':float(mse[hard].mean()),
                'coverage_004':float(np.mean(mse<=.04)),'coverage_002':float(np.mean(mse<=.02)),
                'random_mse_all':float(random.mean()),'random_mse_hard':float(random[:,hard].mean()),
                'nearest_beats_random_fraction':float(np.mean(mse<random.mean(0))),
                'multi_task_unit_fraction':float(multitask.mean()),
                'unit_task_counts':[len(set(mt[m].tolist())) for m in members]})
        outputs.append({'segmentation':segname,'variant':variant,'code_dim':24,
            'fit_segments':len(train),'memory_segments':len(memory),'fit_excludes_tasks':sorted(LONG|HELDOUT),
            'm6_curve':curve})
    result={'method':'train-fit PCA24 latent with same M6 protocol as AE',
        'long_rows':len(qids),'hard_rows':int(hard.sum()),'seed':SEED,
        'segment_metadata':segment_meta.get('S2_grip_velocity'),'results':outputs}
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def run(*,output=ROOT_OUT,epochs=30,batch_size=128,threads=8,device=None,smoke=False,resume=False):
    import torch
    torch.set_num_threads(int(threads))
    if device is None:device='cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.20,0)
    out=Path(output);repdir=out/'representations'
    if (out/'ae_summary.json').exists():raise FileExistsError(f'refusing to overwrite {out}/ae_summary.json')
    out.mkdir(parents=True,exist_ok=True);repdir.mkdir(parents=True,exist_ok=True)
    episodes,rows,smeta,segsets,segment_meta=_load_splits_and_segments()
    q01=np.asarray(smeta['q01'],np.float32);q99=np.asarray(smeta['q99'],np.float32)
    # Lmax is the ceiling of p99 over all selected non-Long/non-heldout S1/S2
    # intervals. No Long or heldout length contributes to this choice.
    all_lengths=[]
    for name in SEGMENTATIONS:
        all_lengths.extend([s['end']-s['start'] for s in segsets[name]
                            if s['task'] not in LONG|HELDOUT and s['end']>s['start']])
    lmax,p99=choose_lmax(all_lengths,percentile=.99,minimum=10)
    qprep=np.load(GAP,allow_pickle=False);qids=np.asarray(qprep['eval_query_rows'],np.int64);hard=np.asarray(qprep['hard_mask'],bool)
    if len(qids)!=548 or hard.sum()!=63:raise AssertionError('Long eval rows/hard mask mismatch')
    if np.any(np.asarray(rows['task'])[qids]>=10):raise AssertionError('non-Long leaked into Long evaluation')
    train_tasks=sorted(set(int(t) for t in rows['task'] if int(t) not in LONG|HELDOUT))
    if set(train_tasks)&(LONG|HELDOUT):raise AssertionError('Long/heldout task leakage')
    configs=[]
    for segname in SEGMENTATIONS:
        for variant in VARIANTS:
            for dim in LATENT_DIMS:configs.append((segname,variant,dim))
    configs.append(('S0_fixed10','raw',24))
    if smoke:configs=configs[:1]
    result=[];t0=time.monotonic()
    for segname,variant,dim in configs:
        name=f'ae_{segname}_{variant}_d{dim}';folder=repdir/name
        if (folder/'metrics.json').exists():
            if not resume:raise FileExistsError(f'refusing to overwrite completed bundle {folder}')
            result.append(json.loads((folder/'metrics.json').read_text()))
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'incomplete bundle exists; refusing to overwrite {folder}')
        segments=[s for s in segsets[segname] if (s['task'],s['episode']) in episodes and s['end']>s['start']]
        train=[s for s in segments if _is_train(s,episodes)]
        memory=[s for s in segments if _is_memory(s,episodes)]
        if not train or not memory:raise ValueError(f'{segname}: missing train or memory segments')
        if any(s['task'] in LONG|HELDOUT for s in train+memory):raise AssertionError('split leakage')
        ref=_make_procrustes_reference(train,episodes) if variant!='raw' else None
        train_x,train_y,_=_prepare(train,episodes,variant,q01,q99,ref,lmax)
        mem_x,mem_y,_=_prepare(memory,episodes,variant,q01,q99,ref,lmax)
        qsegs=[]
        for rid in qids:
            t,e,st=map(int,(rows['task'][rid],rows['episode'][rid],rows['frame'][rid]))
            qsegs.append({'task':t,'episode':e,'start':st,'end':st+10,'row_id':int(rid)})
        qx,qy,qpose=_prepare(qsegs,episodes,variant,q01,q99,ref,lmax)
        if variant=='raw':
            # qstats already puts raw values in official run-01 normalized space.
            train_x=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in train]
            mem_x=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in memory]
            qx=[shape_data.normalize_physical(_seq(s,episodes),q01[:7],q99[:7]) for s in qsegs]
        stats_x=train_x;stats_y=train_y
        progress=out/f'ae_{segname}_{variant}_d{dim}.progress.jsonl'
        if progress.exists():raise FileExistsError(progress)
        model,stats=fit_ae(train_x,train_y,latent_dim=dim,lmax=lmax,epochs=(2 if smoke else epochs),
            batch_size=batch_size,seed=SEED,device=device,threads=threads,invariant=(variant=='raw_invariant'),
            progress=progress)
        mem_z=_encode(model,mem_x,stats['xmean'],stats['xstd'],lmax,device=device,batch_size=batch_size)
        q_y_target=qy
        targets=[np.asarray(rows['actions'][rid,:10,:7],np.float32) for rid in qids]
        # Train-fit PCA24 baseline in the target representation, evaluated on
        # the same query windows and restored pose as the AE.
        qtarget=[]
        for i in range(len(qids)):
            if variant=='raw':qtarget.append(qx[i])
            else:qtarget.append(qy[i])
        pca_pred=_pca_reference(train_y,qtarget,qpose,variant,q01,q99,[10]*len(qids))
        ev=_evaluate(model,stats,variant,lmax,qx,qy,qpose,[10]*len(qids),targets,
            mem_z,memory,device,q01,q99,hard)
        ev['pca24_pred']=pca_pred
        pca_mse=np.asarray([np.mean((a-b)**2) for a,b in zip(pca_pred,targets,strict=True)],np.float32)
        self_mse=ev['self_mse']
        # Store codes and sidecars for every segment in the FORMAT array order.
        allx,ally,allpose=_prepare(segments,episodes,variant,q01,q99,ref,lmax)
        allz=_encode(model,allx,stats['xmean'],stats['xstd'],lmax,device=device,batch_size=batch_size)
        _save_model_bundle(folder,model,stats,{'representation':name,'segmentation':segname,'variant':variant,
            'latent_dim':dim,'invariance_loss_weight':.1 if variant=='raw_invariant' else 0.,
            'augmentation':'SO(3) rotation + scale U[.7,1.3] in physical action space' if variant=='raw_invariant' else 'none',
            'reference_train_only':ref is not None},allz,segments,allpose,q01,q99,train_tasks,lmax)
        # Evaluate M6 on memory segments and self-reconstruction on Long windows.
        self_mse=np.asarray(ev['self_mse'],np.float32)
        rec={'representation':name,'segmentation':segname,'variant':variant,'latent_dim':dim,
            'fit_segments':len(train),'memory_segments':len(memory),'fit_tasks':train_tasks,
            'active_memory_latents':int(len(mem_z)),'fit_seconds':stats['fit_seconds'],
            'train_loss_last':stats['logs'][-1]['reconstruction'],'invariance_loss_last':stats['logs'][-1]['invariance'],
            'long_mse_all':float(self_mse.mean()),'long_mse_hard':float(self_mse[hard].mean()),
            'long_coverage_004':float(np.mean(self_mse<=.04)),'long_coverage_002':float(np.mean(self_mse<=.02)),
            'pca24_mse_all':float(pca_mse.mean()),'pca24_mse_hard':float(pca_mse[hard].mean()),
            'm6_curve':ev['m6_curve'],'output_dir':str(folder)}
        np.savez_compressed(folder/'eval_long.npz',query_rows=qids,hard_mask=hard,self_mse=self_mse,
            pca24_mse=pca_mse,query_latent=ev['query_z'],self_prediction=np.asarray(ev['self_pred'],np.float32))
        (folder/'metrics.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
        result.append(rec)
        with (out/'ae_progress.jsonl').open('a') as f:f.write(json.dumps({'event':'complete','name':name,
            'mse_all':rec['long_mse_all'],'mse_hard':rec['long_mse_hard'],'seconds':rec['fit_seconds']})+'\n')
    summary={'method':'masked variable-length AE; physical Procrustes and latent-invariance variants',
        'seed':SEED,'device':str(device),'epochs':2 if smoke else epochs,'batch_size':batch_size,
        'lmax':lmax,'nonlong_nonheldout_interval_length_p99':p99,'max_interval_length_before_resampling':int(max(all_lengths)),
        'interval_sources':{'S1_greedy':'recomputed with shape_skills.run.make_segment_sets from current run-01 rows',
            'S1_dp':'recomputed with shape_skills.run.make_segment_sets from current run-01 rows',
        'S2_grip_velocity':next(x for x in segment_meta['segmentation_stats']
                                if x['name']=='S2_grip_velocity'),
            'S0_fixed10':'non-overlap 10-frame windows from Train episodes'},
        'excluded_tasks':sorted(LONG|HELDOUT),'fit_tasks':train_tasks,'long_rows':548,'hard_rows':63,
        'elapsed_s':time.monotonic()-t0,'results':result}
    (out/'ae_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT_OUT)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--threads',type=int,default=8);ap.add_argument('--device',default=None)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume-existing',action='store_true')
    ap.add_argument('--pca24-m6-only',action='store_true');a=ap.parse_args()
    if not 1<=a.threads<=16:raise ValueError('threads must be in 1..16')
    if a.pca24_m6_only:
        r=run_pca24_m6(output=a.output,threads=a.threads)
        print(json.dumps({'output':str(a.output),'pca_configs':len(r['results'])}),flush=True);return
    r=run(output=a.output,epochs=a.epochs,batch_size=a.batch_size,threads=a.threads,device=a.device,
          smoke=a.smoke,resume=a.resume_existing)
    print(json.dumps({'output':str(a.output),'models':len(r['results']),'lmax':r['lmax'],'elapsed_s':r['elapsed_s']}),flush=True)


if __name__=='__main__':main()
