#!/usr/bin/env python3
"""Masked variable-length EMA VQ-VAE for shape-skills A.

The discrete nucleus is an EMA codebook vector, not a stored demonstration.
This file also has a standalone runner that writes FORMAT.md bundles under
shape-skills-01/representations/vq_*.
"""
from __future__ import annotations

import argparse
import json
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
GAP = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/compress-vs-raw-review-20261105-01/drafts_prepared.npz')
K_VALUES = (256, 1024)
RESIDUAL_DIMS = (0, 4, 8)
SEGMENT_CONFIGS = (
    ('S0_fixed10', 'fixed10', None),
    ('S1_greedy_e0p01', 'greedy', .01),
    ('S1_dp_e0p01', 'dp', .01),
    ('S2_grip', 'grip', None),
    ('S2_grip_velocity', 'gripvel', 'dev'),
)
SEED = 20261106


def pad_sequences(sequences, *, mean=None, std=None, device='cpu'):
    """Pad [T,7] sequences and return a valid-frame mask; padding is not data."""
    import torch
    if not sequences:
        raise ValueError('need at least one sequence')
    lengths = np.asarray([len(x) for x in sequences], np.int64)
    if np.any(lengths < 1):
        raise ValueError('empty sequences are not valid examples')
    dim = int(np.asarray(sequences[0]).shape[1])
    if any(np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != dim for x in sequences):
        raise ValueError('all sequences must be [T,D] with the same D')
    tmax = int(lengths.max())
    values = np.zeros((len(sequences), tmax, dim), np.float32)
    phase = np.zeros((len(sequences), tmax), np.float32)
    mask = np.zeros((len(sequences), tmax), bool)
    for i, seq in enumerate(sequences):
        x = np.asarray(seq, np.float32)
        if mean is not None:
            x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        n = lengths[i]
        values[i, :n] = x
        phase[i, :n] = np.linspace(0., 1., n, dtype=np.float32) if n > 1 else 0.
        mask[i, :n] = True
    return (torch.as_tensor(values, device=device), torch.as_tensor(phase, device=device),
            torch.as_tensor(mask, device=device), torch.as_tensor(lengths, device=device))


def _masked_reconstruction_loss(pred, target, mask):
    """Per-example MSE over valid time/channel entries only."""
    import torch
    weight = mask.to(dtype=pred.dtype).unsqueeze(-1)
    return (((pred - target) ** 2) * weight).sum((1, 2)) / (weight.sum((1, 2)) * pred.shape[-1]).clamp_min(1.)


def _torch_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class EMACodebook(nn.Module):
        def __init__(self, k, dim, decay=.99, commitment=.25, dead_after=200):
            super().__init__()
            self.k, self.dim = int(k), int(dim)
            self.decay, self.commitment, self.dead_after = float(decay), float(commitment), int(dead_after)
            self.register_buffer('embedding', torch.zeros(k, dim))
            self.register_buffer('ema_count', torch.zeros(k))
            self.register_buffer('ema_sum', torch.zeros(k, dim))
            self.register_buffer('dead_age', torch.zeros(k, dtype=torch.long))
            self.register_buffer('initialized', torch.tensor(False))
            self.reinitializations = 0

        @torch.no_grad()
        def _initialize(self, z):
            ix = torch.randint(len(z), (self.k,), device=z.device)
            v = z[ix].detach()
            self.embedding.copy_(v); self.ema_sum.copy_(v); self.ema_count.fill_(1.)
            self.initialized.fill_(True)

        @torch.no_grad()
        def _ema(self, z, ids):
            counts = torch.bincount(ids, minlength=self.k).to(z.dtype)
            sums = torch.zeros_like(self.ema_sum)
            sums.index_add_(0, ids, z.detach())
            self.ema_count.mul_(self.decay).add_(counts, alpha=1-self.decay)
            self.ema_sum.mul_(self.decay).add_(sums, alpha=1-self.decay)
            total = self.ema_count.sum()
            smooth = (self.ema_count + 1e-5) / (total + self.k*1e-5) * total.clamp_min(1.)
            self.embedding.copy_(self.ema_sum / smooth.unsqueeze(-1).clamp_min(1e-5))
            self.dead_age.copy_(torch.where(counts > 0, torch.zeros_like(self.dead_age), self.dead_age+1))
            dead = torch.nonzero(self.dead_age >= self.dead_after, as_tuple=False).flatten()
            if len(dead):
                ix = torch.randint(len(z), (len(dead),), device=z.device)
                repl = z[ix].detach()
                self.embedding[dead] = repl
                self.ema_sum[dead] = repl
                self.ema_count[dead] = 1.
                self.dead_age[dead] = 0
                self.reinitializations += int(len(dead))

        def forward(self, z, update=True):
            if not bool(self.initialized): self._initialize(z)
            zf = z.float()
            emb = self.embedding.float()
            dist = zf.square().sum(1, keepdim=True) + emb.square().sum(1)[None] - 2*zf@emb.T
            ids = torch.argmin(dist, dim=1)
            q = self.embedding[ids]
            if self.training and update: self._ema(zf, ids)
            loss = self.commitment * F.mse_loss(zf, q.detach())
            q_st = z + (q.to(z.dtype)-z).detach()
            return q_st, ids, loss

        @torch.no_grad()
        def compact(self, ids):
            used = torch.unique(ids, sorted=True)
            if not len(used): raise ValueError('quantizer assigned no codes')
            return self.embedding[used].detach().clone(), used.detach().clone()

    class VQVAE(nn.Module):
        def __init__(self, input_dim=7, latent_dim=32, k=256, residual_dims=(4, 8)):
            super().__init__()
            self.input_dim, self.latent_dim, self.k = input_dim, latent_dim, k
            self.residual_dims = tuple(residual_dims)
            self.encoder = nn.Sequential(nn.Linear(input_dim+1, 128), nn.GELU(), nn.Linear(128,128), nn.GELU())
            self.to_latent = nn.Linear(128, latent_dim)
            self.quantizer = EMACodebook(k, latent_dim)
            self.res_enc = nn.ModuleDict({str(r): nn.Linear(latent_dim, r) for r in self.residual_dims})
            self.res_proj = nn.ModuleDict({str(r): nn.Linear(r, latent_dim, bias=False) for r in self.residual_dims})
            self.decoders = nn.ModuleDict()
            for r in (0,) + self.residual_dims:
                self.decoders[str(r)] = nn.Sequential(nn.Linear(latent_dim+1,128), nn.GELU(),
                    nn.Linear(128,128), nn.GELU(), nn.Linear(128,input_dim))

        def encode_quantized(self, x, phase, mask, update=False):
            zstep = self.encoder(torch.cat([x, phase.unsqueeze(-1)], dim=-1))
            pooled = (zstep * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            z = self.to_latent(pooled)
            q, ids, qloss = self.quantizer(z, update=update)
            residual = {str(r): self.res_enc[str(r)](z) for r in self.residual_dims}
            return z, q, ids, qloss, residual

        def decode(self, q, residual, phase, rdim):
            if rdim:
                latent = q + self.res_proj[str(rdim)](residual)
            else: latent = q
            tiled = latent[:,None,:].expand(-1,phase.shape[1],-1)
            return self.decoders[str(rdim)](torch.cat([tiled,phase.unsqueeze(-1)],dim=-1))

        def forward(self, x, phase, mask, update=True):
            z,q,ids,qloss,res = self.encode_quantized(x,phase,mask,update)
            pred={'0':self.decode(q,None,phase,0)}
            reg=z.new_zeros(())
            for r in self.residual_dims:
                pred[str(r)]=self.decode(q,res[str(r)],phase,r)
                reg=reg+res[str(r)].square().mean()
            return pred, ids, qloss, .001*reg

    return torch, nn, F, EMACodebook, VQVAE


def fit_vq(sequences, *, k=256, residual_dims=(4,8), epochs=40, batch_size=128,
           seed=SEED, device=None, threads=8, progress=None):
    """Fit masked sequence VQ-VAE. Channel statistics use valid Train frames only."""
    torch, nn, F, _, VQVAE = _torch_classes()
    torch.set_num_threads(int(threads))
    np.random.seed(seed); torch.manual_seed(seed)
    if device is None: device='cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.20, 0)
    xseq=[np.asarray(x,np.float32) for x in sequences]
    if len(xseq)<2: raise ValueError('fit requires at least two training segments')
    if any(x.ndim!=2 or x.shape[1]!=7 for x in xseq): raise ValueError('sequences must be [T,7]')
    joined=np.concatenate(xseq,axis=0)
    mean=joined.mean(0).astype(np.float32); std=np.maximum(joined.std(0),1e-4).astype(np.float32)
    model=VQVAE(7,32,k,residual_dims).to(device)
    optim=torch.optim.AdamW([p for n,p in model.named_parameters() if 'quantizer.' not in n],lr=8e-4,weight_decay=1e-5)
    rng=np.random.default_rng(seed); logs=[]; start=time.monotonic(); model.train()
    for epoch in range(int(epochs)):
        order=rng.permutation(len(xseq)); ep_loss=[]
        for lo in range(0,len(order),int(batch_size)):
            inds=order[lo:lo+int(batch_size)]
            x,phase,mask,_=pad_sequences([xseq[i] for i in inds],mean=mean,std=std,device=device)
            pred,ids,qloss,reg=model(x,phase,mask,update=True)
            losses=[_masked_reconstruction_loss(pred[str(r)],x,mask).mean() for r in (0,)+tuple(residual_dims)]
            loss=torch.stack(losses).mean()+qloss+reg
            optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optim.step()
            ep_loss.append(float(loss.detach().cpu()))
        log={'epoch':epoch+1,'loss':float(np.mean(ep_loss)),'elapsed_s':time.monotonic()-start,
             'code_reinitializations':int(model.quantizer.reinitializations)}
        logs.append(log)
        if progress and ((epoch+1)%5==0 or epoch+1==epochs):
            Path(progress).parent.mkdir(parents=True,exist_ok=True)
            with Path(progress).open('a') as f:f.write(json.dumps(log)+'\n')
    model.eval()
    return model, {'mean':mean,'std':std,'logs':logs,'seconds':time.monotonic()-start,'device':str(device)}


def _encode_sequences(model, sequences, stats, *, batch_size=128, device='cpu'):
    torch,_,_,_,_=_torch_classes(); model.eval(); out=[]
    with torch.no_grad():
        for lo in range(0,len(sequences),batch_size):
            seq=sequences[lo:lo+batch_size]
            x,phase,mask,lengths=pad_sequences(seq,mean=stats['mean'],std=stats['std'],device=device)
            z,q,ids,_,res=model.encode_quantized(x,phase,mask,update=False)
            out.append({'z':z.cpu().numpy(),'q':q.cpu().numpy(),'ids':ids.cpu().numpy(),
                        'res':{k:v.cpu().numpy() for k,v in res.items()},'lengths':lengths.cpu().numpy()})
    return {k:np.concatenate([b[k] for b in out]) for k in ('z','q','ids','lengths')} | {
        'res':{r:np.concatenate([b['res'][r] for b in out]) for r in out[0]['res']}}


def _seq_input(action_phys, mode, q01, q99, reference=None):
    a=np.asarray(action_phys,np.float32)
    if mode=='raw':
        return shape_data.normalize_physical(a[:,:7],q01[:7],q99[:7]), {'scale':1.,'rotation':np.eye(3,dtype=np.float32),'length':len(a)}
    if reference is None:
        can,pose=shape_repr.canonicalize(a[:,:7])
    else:
        # Kabsch requires matching phase grids; align a 32-phase cumulative
        # path, then apply the same proper rotation to position and rotation.
        base,pose=shape_repr.canonicalize(a[:,:7])
        path=np.cumsum(a[:,:3],axis=0)
        path=shape_repr.resample(path,shape_repr.PHASE)
        R=shape_repr.procrustes_rotation(path,reference).T
        can=a[:,:6].copy()
        can[:,:3]=a[:,:3]@R.T/float(pose['scale'])
        can[:,3:6]=a[:,3:6]@R.T/float(pose['scale'])
        pose['rotation']=R
    x=np.column_stack([can,a[:,6]]).astype(np.float32)
    pose['length']=len(a)
    return x,pose


def _restore_sequence(x, mode, pose, q01, q99):
    a=np.asarray(x,np.float32)
    if mode=='raw': return a
    phys=a.copy(); phys[:,:6]=shape_repr.apply_pose(phys[:,:6],pose)
    return shape_data.normalize_physical(phys,q01[:7],q99[:7])


def _state_to_np(model):
    return {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}


def _load_model_npz(path, *, device='cpu'):
    torch,_,_,_,VQVAE=_torch_classes()
    z=np.load(path,allow_pickle=False); meta=json.loads(str(z['config_json'].item()))
    model=VQVAE(int(meta['input_dim']),int(meta['latent_dim']),int(meta['codebook_size']),tuple(meta['residual_dims']))
    state={k:torch.as_tensor(z['weight__'+k]) for k in model.state_dict()}
    model.load_state_dict(state,strict=True); model.to(device).eval()
    return model,meta,{k:z[k] for k in z.files if not k.startswith('weight__') and k!='config_json'}


def decode_bundle(bundle_dir, codes, pose, length):
    """FORMAT.md decode(codes, pose, length); output normalized active7 [N,T,7]."""
    torch,_,_,_,_=_torch_classes()
    d=Path(bundle_dir); model,meta,extras=_load_model_npz(d/'model.npz')
    c=np.asarray(codes,np.float32); n=len(c)
    lens=np.full(n,int(length),np.int64) if np.asarray(length).ndim==0 else np.asarray(length,np.int64)
    p={k:np.asarray(v) for k,v in pose.items()}; mode=meta['input_mode']; rdim=int(meta['residual_dim'])
    q01=np.asarray(meta['q01'],np.float32);q99=np.asarray(meta['q99'],np.float32)
    mean=np.asarray(extras['feature_mean'],np.float32);std=np.asarray(extras['feature_std'],np.float32)
    out=[]
    with torch.no_grad():
        for i in range(n):
            l=max(1,int(lens[i])); phase=torch.as_tensor(np.linspace(0,1,l,dtype=np.float32)[None],device='cpu')
            # FORMAT readers commonly memory-map read-only arrays. Copy the
            # small code vectors before exposing them to torch tensors.
            q=torch.as_tensor(np.array(c[i:i+1,:32],copy=True),dtype=torch.float32)
            r=torch.as_tensor(np.array(c[i:i+1,32:],copy=True),dtype=torch.float32) if rdim else None
            y=model.decode(q,r,phase,rdim).cpu().numpy()[0]*std+mean
            pi={k:v[i] for k,v in p.items()}
            out.append(_restore_sequence(y,mode,pi,q01,q99))
    maxlen=max(map(len,out)); ret=np.zeros((n,maxlen,7),np.float32)
    for i,x in enumerate(out):ret[i,:len(x)]=x
    return ret


def _bundle_save(folder, model, stats, config, code_vectors, codes, code_ids, segments, poses, q01, q99, usage):
    folder=Path(folder)
    if folder.exists() and any(folder.iterdir()): raise FileExistsError(f'refusing to overwrite {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    rdim=int(config['residual_dim'])
    columns=[code_vectors]
    if rdim: columns.append(codes['res'][str(rdim)])
    allcodes=np.concatenate(columns,axis=1).astype(np.float32)
    np.save(folder/'codes.npy',allcodes)
    np.save(folder/'code_ids.npy',np.asarray(code_ids,np.int32))
    np.savez_compressed(folder/'segments.npz',task=np.asarray([s['task'] for s in segments],np.int32),
        episode=np.asarray([s['episode'] for s in segments],np.int32),start=np.asarray([s['start'] for s in segments],np.int32),
        length=np.asarray([s['length'] for s in segments],np.int32),row_ids=np.asarray([s['row_id'] for s in segments],np.int64))
    if config['input_mode']=='procrustes':
        np.savez_compressed(folder/'pose.npz',scale=np.asarray([p['scale'] for p in poses],np.float32),
            rotation=np.asarray([p['rotation'] for p in poses],np.float32),length=np.asarray([p['length'] for p in poses],np.int32))
    else:
        np.savez_compressed(folder/'pose.npz',scale=np.ones(len(segments),np.float32),
            rotation=np.broadcast_to(np.eye(3,dtype=np.float32),(len(segments),3,3)).copy(),
            length=np.asarray([s['length'] for s in segments],np.int32))
    cfg={**config,'kind':'vqvae_ema','version':1,'input_dim':7,'latent_dim':32,
         'codebook_size':int(usage['active_codes']),
         'residual_dims':[4,8],'q01':np.asarray(q01[:7]).tolist(),'q99':np.asarray(q99[:7]).tolist(),
         'decoder_input':'codebook nucleus embedding + optional continuous residual + normalized phase',
         'nucleus_is_real_segment':False,'nucleus_definition':'EMA codebook vector; not an observed demonstration',
         'masking':'valid frames only in encoder pooling and reconstruction loss; zero padding excluded',
         'pose_units':'physical position/rotation delta scale and proper rotation; gripper is unrotated'}
    weights=_state_to_np(model)
    compact=np.asarray(usage['compact_vectors'],np.float32)
    # The portable bundle physically removes dead entries from all codebook buffers.
    weights['quantizer.embedding']=compact
    weights['quantizer.ema_sum']=compact.copy()
    weights['quantizer.ema_count']=np.ones(len(compact),np.float32)
    weights['quantizer.dead_age']=np.zeros(len(compact),np.int64)
    np.savez_compressed(folder/'model.npz',**{'weight__'+k:v for k,v in weights.items()},
        feature_mean=stats['mean'],feature_std=stats['std'],code_vectors=usage['compact_vectors'],
        config_json=np.asarray(json.dumps(cfg,ensure_ascii=False)))
    (folder/'manifest.json').write_text(json.dumps({**cfg,'segments':len(segments),'code_dim':32+rdim,
        'used_codes':usage['active_codes'],'dead_codes_removed':usage['dead_codes_removed'],
        'fit_split':config['fit_split'],'normalization':'run-01 q01/q99; per-feature z-score fit on valid Train frames only',
        'seed':SEED},ensure_ascii=False,indent=2)+'\n')
    (folder/'reconstruct.py').write_text('''from pathlib import Path\nfrom vq import decode_bundle\nBUNDLE = Path(__file__).resolve().parent\ndef decode(codes, pose, length):\n    return decode_bundle(BUNDLE, codes, pose, length)\n''')
    public_usage={k:v for k,v in usage.items() if k!='compact_vectors'}
    (folder/'metrics.json').write_text(json.dumps(public_usage,ensure_ascii=False,indent=2)+'\n')


def _build_segments(episodes, rows, vthr):
    configs=list(SEGMENT_CONFIGS)
    out={}
    row_lookup={}
    for rid,(t,e,f) in enumerate(zip(rows['task'],rows['episode'],rows['frame'],strict=True)):
        row_lookup.setdefault((int(t),int(e),int(f)),rid)
    for name,kind,param in configs:
        segs=[]
        for key,seq in episodes.items():
            if kind=='fixed10': bounds=shape_data.fixed10(seq)
            elif kind in ('greedy','dp'): bounds=shape_data.threshold_segments(seq,float(param),kind)
            elif kind=='grip': bounds=shape_data.grip_segments(seq)
            elif kind=='gripvel': bounds=shape_data.grip_velocity_segments(seq,float(vthr))
            else:raise ValueError(kind)
            for a,b in bounds:
                if b-a<10:continue
                start=int(seq['frame'][a]);rid=row_lookup.get((key[0],key[1],start),-1)
                segs.append({'task':key[0],'episode':key[1],'start':start,'length':b-a,'row_id':rid,
                             'action_phys':np.asarray(seq['actions_phys'][a:b],np.float32),
                             'meta':seq['meta']})
        out[name]=segs
    return out


def _training_indices(segments):
    excluded=set(shape_data.LONG_TASKS)|set(shape_data.HELDOUT_TASKS)
    return np.asarray([i for i,s in enumerate(segments) if s['task'] not in excluded],np.int64)


def _load_source_robust(run_dir, checkpoint):
    """Same source join as shape_skills.data.load_source, robust to Arrow fixed lists."""
    import pyarrow.parquet as pq
    run_dir=Path(run_dir); manifest=json.loads((run_dir/'data_manifest.json').read_text())
    source=Path(manifest['source']); q01,q99=shape_data.qstats(checkpoint)
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (run_dir/'rows').glob('*.npy')}
    pair_to_ix={}
    for i,(t,e) in enumerate(zip(rows['task'],rows['episode'],strict=True)):
        pair_to_ix.setdefault((int(t),int(e)),[]).append(i)
    episode_meta={}
    for key,ix0 in pair_to_ix.items():
        ix=np.asarray(ix0,np.int64);mem=np.asarray(rows['memory'][ix],bool);val=np.asarray(rows['validation'][ix],bool)
        if np.any(mem!=mem[0]) or np.any(val!=val[0]):raise AssertionError(f'inconsistent episode split: {key}')
        episode_meta[key]={'memory':bool(mem[0]),'validation':bool(val[0])}
    episodes={}
    for path in sorted(source.glob('data/chunk-*/episode_*.parquet')):
        tab=pq.read_table(path,columns=['actions','state','frame_index','task_index','episode_index'])
        ac=tab['actions'].combine_chunks();st=tab['state'].combine_chunks()
        a=np.asarray(ac.values,np.float32).reshape(len(tab),-1)
        state=np.asarray(st.values,np.float32).reshape(len(tab),-1)
        frame=np.asarray(tab['frame_index'].to_numpy(),np.int64)
        task=int(tab['task_index'][0].as_py());episode=int(tab['episode_index'][0].as_py())
        order=np.argsort(frame,kind='stable');frame,a,state=frame[order],a[order],state[order]
        if not np.array_equal(frame,np.arange(len(frame),dtype=np.int64)):raise AssertionError(f'frame gaps in {path}')
        # run-01 actions were produced by the affine q01/q99 transform without
        # clipping saturated gripper endpoints (they can be about 4e-4 > 1).
        norm=(a-q01[:a.shape[1]])/(q99[:a.shape[1]]-q01[:a.shape[1]]+1e-6)*2-1
        episodes[(task,episode)]={'task':task,'episode':episode,'frame':frame,'actions_phys':a,
            'actions_norm':norm.astype(np.float32),'state_phys':state,
            'meta':episode_meta.get((task,episode),{'memory':False,'validation':False})}
    if not episodes:raise FileNotFoundError(f'no source episodes under {source}')
    ra=np.load(run_dir/'rows/actions.npy',mmap_mode='r',allow_pickle=False);checked=0
    for rid,(t,e,start) in enumerate(zip(rows['task'],rows['episode'],rows['frame'],strict=True)):
        seq=episodes.get((int(t),int(e)))
        if seq is not None and int(start)+10<=len(seq['frame']) and checked<64:
            np.testing.assert_allclose(seq['actions_norm'][int(start):int(start)+10],ra[rid,:,:7],atol=4e-5,rtol=0)
            checked+=1
    if checked<16:raise AssertionError(f'insufficient row parity validation: {checked}')
    return episodes,rows,{'source':str(source),'parquet_episodes':len(episodes),'source_row_parity_checked':checked,
        'q01':q01.tolist(),'q99':q99.tolist()}


def _metric(pred, target):
    return np.mean((np.asarray(pred,np.float32)-np.asarray(target,np.float32))**2,axis=(1,2))


def _pca_m6_reference(segments, fit_ix, all_x, fitmeta, input_mode, reference, q01, q99, rows, episodes):
    """PCA-24 and label-free M6-on-PCA unit-count/reconstruction curve."""
    prep=np.load(GAP,allow_pickle=False);qids=np.asarray(prep['eval_query_rows'],np.int64)
    hard=np.asarray(prep['hard_mask'],bool)
    target=np.asarray(rows['actions'][qids,:10,:7],np.float32)
    mean,std=fitmeta['mean'],fitmeta['std']
    def vec(x):return ((shape_repr.resample(x,shape_repr.PHASE)-mean)/std).reshape(-1)
    train=np.stack([vec(all_x[i]) for i in fit_ix])
    pca=shape_repr.pca_fit(train,24)
    qx=[];qpose=[]
    for rid in qids:
        t,e,start=map(int,(rows['task'][rid],rows['episode'][rid],rows['frame'][rid]))
        # The query segment is a fixed 10-step window; no future labels or task text enters the key.
        source_ep=episodes[(t,e)]['actions_phys'][start:start+10]
        x,p=_seq_input(source_ep,input_mode,q01,q99,reference)
        qx.append(vec(x));qpose.append(p)
    qx=np.stack(qx);qcode=shape_repr.pca_encode(pca,qx)
    pca_pred=[]
    for i,z in enumerate(qcode):
        dec=shape_repr.pca_decode(pca,z).reshape(shape_repr.PHASE,7)*std+mean
        pca_pred.append(_restore_sequence(shape_repr.resample(dec,10),input_mode,qpose[i],q01,q99))
    pca_mse=_metric(np.asarray(pca_pred),target)
    tarr=np.asarray([segments[i]['task'] for i in fit_ix],np.int32)
    earr=np.asarray([segments[i]['episode'] for i in fit_ix],np.int32)
    curve=[]
    for j,kappa in enumerate((.01,.025,.05,.1,.2,.4,.8,1.6)):
        m6=shape_repr.m6_fit_generic(qcode_train:=shape_repr.pca_encode(pca,train),tarr,earr,
            kappa=kappa,seed=SEED+j,min_support=10,min_episodes=3)
        centers=np.asarray(m6['centers'],np.float32)
        if not len(centers):
            curve.append({'kappa':kappa,'unit_count':0,'mse_all':None,'mse_hard':None,'active_task_counts':[]});continue
        dist=((qcode[:,None,:]-centers[None])**2).mean(-1);assign=np.argmin(dist,axis=1)
        pred=[]
        for qi,cid in enumerate(assign):
            dcode=shape_repr.pca_decode(pca,centers[cid]).reshape(shape_repr.PHASE,7)*std+mean
            pred.append(_restore_sequence(shape_repr.resample(dcode,10),input_mode,qpose[qi],q01,q99))
        err=_metric(np.asarray(pred),target)
        active_tasks=[len(s) for s in m6['episode_sets']]
        curve.append({'kappa':kappa,'unit_count':int(len(centers)),'mse_all':float(err.mean()),
                      'mse_hard':float(err[hard].mean()),'coverage_004':float(np.mean(err<=.04)),
                      'coverage_002':float(np.mean(err<=.02)),'active_task_counts':active_tasks,
                      'pending_train_segments':int(np.sum(m6['row_units']<0))})
    return {'pca24_all_mse':float(pca_mse.mean()),'pca24_hard_mse':float(pca_mse[hard].mean()),
            'pca24_coverage_004':float(np.mean(pca_mse<=.04)),'pca24_coverage_002':float(np.mean(pca_mse<=.02)),
            'm6_pca24_curve':curve}


def _evaluate_long(bundle, model, meta, stats, segments, all_inputs, poses, q01, q99, rows, *, seed=SEED):
    """Encode Long Eval 10-step windows, compare assigned vs random nuclei."""
    torch,_,_,_,_=_torch_classes()
    prep=np.load(GAP,allow_pickle=False)
    qids=np.asarray(prep['eval_query_rows'] if 'eval_query_rows' in prep else prep['eval_query_rows'],np.int64)
    # The corrected compress-vs-raw row file is the fixed Long Eval contract.
    hard=np.asarray(prep['hard_mask'],bool)
    if len(qids)!=548 or hard.sum()!=63:raise AssertionError('fixed Long Eval/hard mask mismatch')
    rowtask=np.asarray(rows['task'][qids])
    if np.any(rowtask>=10):raise AssertionError('non-Long query entered evaluation')
    seqs=[];qposes=[];targets=[]
    for rid in qids:
        task=int(rows['task'][rid]);epi=int(rows['episode'][rid]);start=int(rows['frame'][rid])
        ep=all_inputs['episodes'][(task,epi)]['actions_phys'][start:start+10]
        x,p=_seq_input(ep,meta['input_mode'],q01,q99,all_inputs['reference'])
        seqs.append(x);qposes.append(p);targets.append(np.asarray(rows['actions'][rid,:10,:7],np.float32))
    target=np.asarray(targets,np.float32)
    encoded=_encode_sequences(model,seqs,stats,batch_size=128,device=meta['device'])
    codebook=np.asarray(meta['compact_vectors'],np.float32)
    # Assignments are nearest codebook nucleus; code vectors themselves were
    # formed only from non-Long, non-heldout training segments.
    d=((encoded['z'][:,None,:]-codebook[None])**2).mean(-1)
    nearest=np.argmin(d,axis=1)
    rng=np.random.default_rng(seed); random_ix=rng.integers(len(codebook),size=(20,len(qids)))
    def decode_ids(ix, residual=True):
        result=[]
        with torch.no_grad():
            for start in range(0,len(qids),64):
                sl=slice(start,min(start+64,len(qids))); ids=ix[sl]
                q=torch.as_tensor(codebook[ids],device=meta['device'])
                lens=np.asarray([10]*len(ids),np.int64);phase=torch.linspace(0,1,10,device=meta['device'])[None].repeat(len(ids),1)
                if residual and int(meta['residual_dim']):
                    rv=encoded['res'][str(meta['residual_dim'])][sl]
                    r=torch.as_tensor(rv,device=meta['device'])
                else:r=None
                y=model.decode(q,r,phase,int(meta['residual_dim']) if residual else 0).cpu().numpy()
                for j in range(len(y)):
                    yj=y[j]*stats['std']+stats['mean']
                    result.append(_restore_sequence(yj,meta['input_mode'],qposes[start+j],q01,q99))
        return np.asarray(result,np.float32)
    pred=decode_ids(nearest,True); mse=_metric(pred,target)
    random_mses=[]
    for rix in random_ix:
        random_mses.append(_metric(decode_ids(rix,True),target))
    random_mses=np.asarray(random_mses)
    shared=(mse<random_mses.mean(0)).astype(np.float32)
    return {'query_rows':qids,'hard_mask':hard,'nearest_mse':mse,'random_mse_mean':random_mses.mean(0),
            'random_mse_p05_p95':np.quantile(random_mses,[.05,.95],axis=0).T,
            'nearest_code_id':nearest,'shared_vs_random_win_rate':shared,
            'coverage_004':float(np.mean(mse<=.04)),'coverage_002':float(np.mean(mse<=.02)),
            'nearest_all_mean':float(mse.mean()),'nearest_all_median':float(np.median(mse)),
            'nearest_hard_mean':float(mse[hard].mean()),'nearest_hard_median':float(np.median(mse[hard])),
            'random_all_mean':float(random_mses.mean()),'random_hard_mean':float(random_mses[:,hard].mean()),
            'shared_win_rate':float(shared.mean()),'shared_win_rate_hard':float(shared[hard].mean())}


def run(*, output=ROOT_OUT, epochs=40, batch_size=128, threads=8, device=None, smoke=False):
    import torch
    torch.set_num_threads(int(threads))
    if device is None:device='cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.20,0)
    out=Path(output); reps_dir=out/'representations'
    if out.exists() and (out/'vq_run_manifest.json').exists():raise FileExistsError(f'output already exists: {out}')
    out.mkdir(parents=True,exist_ok=True);reps_dir.mkdir(parents=True,exist_ok=True)
    episodes,rows,srcmeta=_load_source_robust(RUN_DIR,Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    dev_keys=[k for k,v in episodes.items() if k[0] in shape_data.HELDOUT_TASKS and v['meta']['validation']]
    vthr,vrec=shape_data.choose_velocity_threshold(episodes,dev_keys)
    segsets=_build_segments(episodes,rows,vthr)
    configs=list(SEGMENT_CONFIGS)
    if smoke:configs=configs[:1]
    fit_tasks=sorted(set(s['task'] for segs in segsets.values() for s in segs)-set(shape_data.LONG_TASKS)-set(shape_data.HELDOUT_TASKS))
    q01=np.asarray(srcmeta['q01'],np.float32);q99=np.asarray(srcmeta['q99'],np.float32)
    eval_records=[];started=time.monotonic()
    for seg_name,_,_ in configs:
      segments=segsets[seg_name];fit_ix=_training_indices(segments)
      train_segs=[segments[i] for i in fit_ix]
      for input_mode in ('raw','procrustes'):
        # Procrustes reference and all feature transforms are training-only.
        can_train=[]
        if input_mode=='procrustes':
            for s in train_segs:
                can,_=shape_repr.canonicalize(s['action_phys'][:,:7]);can_train.append(np.cumsum(can,axis=0)[:,:3])
            ref=np.mean([shape_repr.resample(x,shape_repr.PHASE) for x in can_train],axis=0)
            ref=ref-ref.mean(0,keepdims=True)
        else:ref=None
        all_x=[];all_pose=[]
        for s in segments:
            x,p=_seq_input(s['action_phys'],input_mode,q01,q99,ref);all_x.append(x);all_pose.append(p)
        train_x=[all_x[i] for i in fit_ix]
        if not fit_tasks or set(fit_tasks)&(set(shape_data.LONG_TASKS)|set(shape_data.HELDOUT_TASKS)):
            raise AssertionError('Long/heldout task contamination in VQ fit')
        base_info={'segment_name':seg_name,'input_mode':input_mode,'fit_split':'all segments from tasks excluding Long 0-9 and heldout tasks [17,19,28,29,35,38]',
                   'fit_tasks':fit_tasks,'fit_segments':int(len(fit_ix)),'total_segments':len(segments),
                   'velocity_threshold':float(vthr),'procrustes_reference_train_only':input_mode=='procrustes'}
        refcurve=None
        for k in K_VALUES:
            key=f'{seg_name}_{input_mode}_K{k}'
            fitlog=out/f'{key}.progress.jsonl'
            if fitlog.exists():raise FileExistsError(f'refusing to append existing training log: {fitlog}')
            model,fitmeta=fit_vq(train_x,k=k,residual_dims=(4,8),epochs=(2 if smoke else epochs),
                batch_size=batch_size,seed=SEED,device=device,threads=threads,progress=fitlog)
            if refcurve is None:
                refcurve=_pca_m6_reference(segments,fit_ix,all_x,fitmeta,input_mode,ref,q01,q99,rows,episodes)
            enc=_encode_sequences(model,all_x,fitmeta,batch_size=batch_size,device=device)
            train_ids=np.asarray(enc['ids'][fit_ix],np.int64)
            used,counts=np.unique(train_ids,return_counts=True)
            # Remove codes never assigned to a fit segment, and retain an
            # explicit old-to-compact mapping for auditability.
            remap=np.full(k,-1,np.int32);remap[used]=np.arange(len(used),dtype=np.int32)
            compact=np.asarray(model.quantizer.embedding.detach().cpu())[used]
            usage_by_task={}
            for local,global_i in enumerate(fit_ix):
                cid=int(remap[train_ids[local]]);task=int(segments[global_i]['task'])
                usage_by_task.setdefault(cid,set()).add(task)
            task_counts=np.asarray([len(usage_by_task.get(i,set())) for i in range(len(used))],np.int32)
            prob=counts/counts.sum();perplexity=float(np.exp(-np.sum(prob*np.log(np.maximum(prob,1e-12)))))
            code_stats={'initial_codes':int(k),'active_codes':int(len(used)),'dead_codes_removed':int(k-len(used)),
                'active_fraction':float(len(used)/k),'perplexity':perplexity,'counts_min_median_max':[int(counts.min()),float(np.median(counts)),int(counts.max())],
                'tasks_per_code_min_median_max':[int(task_counts.min()),float(np.median(task_counts)),int(task_counts.max())],
                'code_task_counts':task_counts.tolist(),'assignment_counts':counts.tolist(),
                'reinitializations_during_ema':int(model.quantizer.reinitializations)}
            base_dir=reps_dir/f'vq_{key}'
            # Reconstruct after codebook pruning: the retained vector is the
            # actual discrete code/nucleus. Residual dimensions share the fit.
            meta_common={**base_info,'initial_k':k,'active_codes':int(len(used)),'input_mode':input_mode,
                'device':str(device),'compact_vectors':compact,'residual_dim':0}
            eval_items={}
            # Serialize separate FORMAT bundles for r=0,4,8. The encoder,
            # EMA codebook and decoder heads are joint-fit; only decoder branch differs.
            for rdim in RESIDUAL_DIMS:
                folder=base_dir/f'r{rdim}'
                codes={**enc,'res':enc['res']}
                compact_ids=remap[enc['ids']]
                missing=compact_ids<0
                if np.any(missing):
                    # Codes unused by fit are removed. Eval-only assignments map
                    # to the nearest retained training nucleus without updating it.
                    zz=enc['z'][missing]
                    dist=((zz[:,None,:]-compact[None])**2).mean(-1)
                    compact_ids[missing]=np.argmin(dist,axis=1).astype(np.int32)
                usage={**code_stats,'compact_vectors':compact,'active_codes':len(used),
                       'eval_assignments_mapped_from_removed_codes':int(missing.sum())}
                config={**base_info,'input_mode':input_mode,'initial_k':k,'residual_dim':rdim,
                        'fit_tasks':fit_tasks,'fit_split':base_info['fit_split']}
                code_vec=compact[compact_ids]
                _bundle_save(folder,model,fitmeta,config,code_vec,codes,compact_ids,segments,all_pose,q01,q99,usage)
                # A temporary Python decode facade for the same trained object.
                model_meta={**meta_common,'residual_dim':rdim}
                meval=_evaluate_long(folder,model,model_meta,fitmeta,segments,
                    {'episodes':episodes,'reference':ref},all_pose,q01,q99,rows)
                np.savez_compressed(folder/'eval_long.npz',**meval)
                eval_items[f'r{rdim}']={k0:v for k0,v in meval.items() if np.asarray(v).ndim==0}
                # Include all-segment type counts; task labels are metric-only.
                eval_items[f'r{rdim}']['usage']=code_stats
            m6counts=np.asarray([x['unit_count'] for x in refcurve['m6_pca24_curve'] if x['unit_count']>0],np.int32)
            closest_m6=int(m6counts[np.argmin(np.abs(m6counts-len(used)))]) if len(m6counts) else 0
            m6match=min((x for x in refcurve['m6_pca24_curve'] if x['unit_count']==closest_m6),
                        key=lambda x:abs(x['unit_count']-len(used)),default=None)
            if m6match is not None:
                m6match={**m6match,'matches_vq_active_code_count':bool(closest_m6==len(used))}
            eval_records.append({'representation':key,'segment_name':seg_name,'input_mode':input_mode,
                'initial_k':k,'active_codes':int(len(used)),'fit_segments':int(len(fit_ix)),
                'fit_seconds':float(fitmeta['seconds']),'residuals':eval_items,
                'pca24_m6_curve_reference':refcurve,
                'closest_m6_curve_point_to_vq_active_count':m6match})
            with (out/'vq_progress.jsonl').open('a') as f:f.write(json.dumps({'event':'model_complete','representation':key,
                'active_codes':int(len(used)),'seconds':float(fitmeta['seconds'])})+'\n')
    summary={'method':'EMA VQ-VAE over masked variable-length segments','seed':SEED,'device':str(device),
        'epochs':2 if smoke else epochs,'batch_size':batch_size,'fit_excludes_tasks':sorted(set(shape_data.LONG_TASKS)|set(shape_data.HELDOUT_TASKS)),
        'fit_tasks':fit_tasks,'input_statistics_fit_on':'valid frames in fit segments only','residual_dims':RESIDUAL_DIMS,
        'nucleus_is_real_segment':False,'nucleus_definition':'EMA codebook vector; not an observed segment',
        'masking':'padding excluded from encoder pooling and reconstruction loss','velocity_threshold_selection':vrec,
        'long_eval_rows':548,'long_hard_rows':63,'representations':eval_records,'elapsed_s':time.monotonic()-started}
    (out/'vq_run_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'vq_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT_OUT)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--threads',type=int,default=8);ap.add_argument('--device',default=None)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if not 1<=a.threads<=16:raise ValueError('threads must be 1..16')
    result=run(output=a.output,epochs=a.epochs,batch_size=a.batch_size,threads=a.threads,device=a.device,smoke=a.smoke)
    print(json.dumps({'output':str(a.output),'representations':len(result['representations']),'elapsed_s':result['elapsed_s']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
