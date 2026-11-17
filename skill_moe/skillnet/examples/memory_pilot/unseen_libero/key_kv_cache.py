#!/usr/bin/env python3
"""Read π0.5 prefix KV caches for retrieval (KV1).

The large frozen cache is stored by exact (domain, task, episode, frame)
identity.  Training/evaluation data construction and ranking metrics are
delegated to ``key_shared_latent``; this file implements the KV reader models
and resumable cache extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
RUN_ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
L90X_ROOT=ROOT/'expert-refine-exp3b-01/l90x_rows'
DEV_TASKS=(6,7)
EVAL_TASKS=(0,10,13,31,35,43,57,61,84,88)
KV_LAYERS=18
KV_TOKENS=532
KV_WIDTH=256
INPUT_MODES=('kv_l','kv_h','kv_lh')
LRS=(1e-4,3e-4)
EPOCHS=(4,8,16)
TEACHER_TAUS=(.002,.005,.01,.02)


def row_map(keys: np.ndarray) -> dict[tuple[int,int,int],int]:
    result={}
    for i,k in enumerate(np.asarray(keys)):
        key=tuple(map(int,k))
        if len(key)!=3 or key in result:raise ValueError(f'duplicate/invalid row identity {key}')
        result[key]=i
    return result


def compact_prefix_positions(length: int, *, image_tokens_per_camera: int=256) -> np.ndarray:
    """Select base, wrist and text from base/wrist/absent-right/text ordering."""
    image=int(image_tokens_per_camera)
    if length<3*image:raise ValueError(f'prefix sequence lacks the right-camera slot: {length}')
    selected=np.r_[0:2*image,3*image:length].astype(np.int64)
    return selected


def pack_prefix_kv(key_cache, value_cache, valid_mask, *, token_limit: int=KV_TOKENS):
    """Pack RoPE-applied prefix K and V from Gemma's [L,B,T,H,D] caches.

    Returns fp16 ``[B,L,2,T,D]`` and a compact boolean mask.  The unused
    right-camera placeholders are dropped, matching ``MemoryPi05.prefix_features``.
    """
    k=np.asarray(key_cache);v=np.asarray(value_cache);mask=np.asarray(valid_mask,bool)
    if k.ndim!=5 or v.shape!=k.shape or k.shape[0]!=KV_LAYERS or k.shape[3]!=1 or k.shape[4]!=KV_WIDTH:
        raise ValueError(f'expected Gemma cache [18,B,T,1,256], got K{k.shape} V{v.shape}')
    if mask.shape!=(k.shape[1],k.shape[2]):raise ValueError(f'prefix mask shape {mask.shape} does not match cache')
    positions=compact_prefix_positions(k.shape[2])
    # Layer-major cache -> row-major, with K/V explicit and the single KV head folded into D.
    kk=np.transpose(k[:,:,positions,0,:],(1,0,2,3))
    vv=np.transpose(v[:,:,positions,0,:],(1,0,2,3))
    compact_mask=mask[:,positions]
    out=np.zeros((len(mask),KV_LAYERS,2,token_limit,KV_WIDTH),np.float16)
    out_mask=np.zeros((len(mask),token_limit),bool)
    for i in range(len(mask)):
        ids=np.flatnonzero(compact_mask[i])
        if len(ids)>token_limit:raise ValueError(f'{len(ids)} valid KV tokens exceed token_limit={token_limit}')
        n=len(ids);out[i,:,0,:n]=kk[i][:,ids,:];out[i,:,1,:n]=vv[i][:,ids,:];out_mask[i,:n]=True
    if out.shape[2:]!=(2,token_limit,KV_WIDTH):raise AssertionError(out.shape)
    return out,out_mask


def final_tokens_from_prefix(hidden, valid_mask, *, image_tokens_per_camera: int=256):
    """Match normalized final-layer output and group order from core.prefix_features."""
    h=np.asarray(hidden,np.float32);m=np.asarray(valid_mask,bool)
    if h.ndim!=3 or h.shape[1]!=m.shape[1]:raise ValueError('hidden/mask shape mismatch')
    pos=compact_prefix_positions(h.shape[1],image_tokens_per_camera=image_tokens_per_camera)
    selected=h[:,pos];selected_mask=m[:,pos]
    out=np.zeros((len(h),KV_TOKENS,h.shape[-1]),np.float16);g=np.full((len(h),KV_TOKENS),-1,np.int8)
    for i in range(len(h)):
        ids=np.flatnonzero(selected_mask[i]);n=len(ids)
        if n>KV_TOKENS:raise ValueError(f'{n} valid hidden tokens exceed {KV_TOKENS}')
        row=selected[i,ids];row=row/np.maximum(np.linalg.norm(row,axis=-1,keepdims=True),1e-8)
        out[i,:n]=row.astype(np.float16)
        groups=np.r_[np.zeros(256,np.int8),np.ones(256,np.int8),np.full(max(0,n-512),2,np.int8)]
        g[i,:n]=groups[:n]
    return out,g


def exact_join_rows(wanted_keys, cache_keys, values):
    wanted=np.asarray(wanted_keys);arr=np.asarray(values)
    if arr.shape[0]!=len(cache_keys):raise ValueError('cache rows do not match cache keys')
    mapping=row_map(cache_keys)
    missing=[tuple(map(int,k)) for k in wanted if tuple(map(int,k)) not in mapping]
    if missing:raise KeyError(f'{len(missing)} exact rows absent; examples={missing[:3]}')
    return arr[np.asarray([mapping[tuple(map(int,k))] for k in wanted],np.int64)]


def prefix_parity_metrics(reference, candidate, *, eps: float = 1e-6) -> dict[str, float]:
    """Compare aligned normalized token rows without letting near-zero values dominate.

    Relative error is summarized over reference entries with magnitude >= eps;
    token cosine and cosine of mean-pooled vectors are both retained because a
    single outlier token can be hidden by pooling.
    """
    a=np.asarray(reference,np.float32);b=np.asarray(candidate,np.float32)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError(f'expected aligned [T,D] arrays, got {a.shape}, {b.shape}')
    delta=np.abs(a-b);denom=np.abs(a);keep=denom>=float(eps)
    rel=delta[keep]/denom[keep] if np.any(keep) else np.asarray([0.],np.float32)
    dot=np.sum(a*b,axis=-1);norm=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    token_cos=dot/np.maximum(norm,1e-12)
    am=a.mean(0);bm=b.mean(0)
    pooled_cos=float(np.dot(am,bm)/max(float(np.linalg.norm(am)*np.linalg.norm(bm)),1e-12))
    return {'relative_median':float(np.median(rel)),
            'relative_p95':float(np.quantile(rel,.95)),
            'relative_coverage':float(np.mean(keep)),
            'token_cosine_median':float(np.median(token_cos)),
            'token_cosine_min':float(np.min(token_cos)),
            'mean_pooled_cosine':pooled_cos,
            'max_abs':float(np.max(delta,initial=0.))}


def parity_passes(metrics: dict[str, float]) -> bool:
    return (metrics['relative_median']<.01 and metrics['token_cosine_median']>=.999
            and metrics['mean_pooled_cosine']>=.999)


def training_bank_indices(task: np.ndarray, episode: np.ndarray,
                          excluded_pairs) -> np.ndarray:
    """Candidate-row mask for fitting; evaluation episodes cannot be positives."""
    t=np.asarray(task);e=np.asarray(episode)
    if t.shape!=e.shape or t.ndim!=1:raise ValueError('task/episode vectors must align')
    excluded={(int(a),int(b)) for a,b in excluded_pairs}
    keep=np.fromiter(((int(a),int(b)) not in excluded for a,b in zip(t,e,strict=True)),
                     dtype=bool,count=len(t))
    return np.flatnonzero(keep)


def source_grouped_batches(target_keys, source_keys: dict[str,np.ndarray], *,
                           source_priority: dict[int,tuple[str,...]], batch_size: int=4):
    """Assign each requested row to its existing cache source and exact batch group.

    Numerical prefix parity depends on the original accelerator batch grouping.
    Replaying a row inside an arbitrary batch can change bf16 GEMM rounding even
    when its observation is identical. This preserves source order and groups.
    """
    targets=np.asarray(target_keys,np.int64)
    mappings={}
    for name,keys in source_keys.items():
        mapping={}
        for i,key in enumerate(np.asarray(keys)):
            k=tuple(map(int,key))
            if k in mapping:raise ValueError(f'duplicate source identity in {name}: {k}')
            mapping[k]=i
        mappings[name]=mapping
    assigned={}
    for ti,row in enumerate(targets):
        domain=int(row[0]);key=tuple(map(int,row[1:]));found=None
        for name in source_priority.get(domain,()):
            if key in mappings.get(name,{}):found=(name,mappings[name][key]);break
        if found is None:raise KeyError(f'no source cache row for requested identity {tuple(map(int,row))}')
        assigned[ti]=found
    grouped={}
    for ti,(name,source_i) in assigned.items():grouped.setdefault((name,source_i//int(batch_size)),[]).append(ti)
    out=[]
    for (name,group),tids in grouped.items():
        source_batch=np.arange(group*int(batch_size),min((group+1)*int(batch_size),len(source_keys[name])),dtype=np.int64)
        positions=[]
        source_to_pos={int(v):p for p,v in enumerate(source_batch)}
        for ti in tids:
            _,source_i=assigned[ti]
            positions.append(source_to_pos[int(source_i)])
        out.append({'source':name,'source_indices':source_batch,'target_indices':np.asarray(tids,np.int64),
                    'target_positions':np.asarray(positions,np.int64)})
    return out


def validate_l90x_manifest(manifest: dict[str,Any]) -> None:
    """Reuse the LA2 leak guard and also require the fixed 78-task inventory."""
    sys.path.insert(0,str(HERE))
    import latent_action_units_la2 as la2
    la2.assert_l90x_manifest(manifest)
    tasks=set(map(int,manifest.get('tasks',())))
    forbidden=set(DEV_TASKS)|set(EVAL_TASKS)
    if tasks&forbidden:raise AssertionError(f'L90X includes held-out tasks: {sorted(tasks&forbidden)}')
    if len(tasks)!=78:raise AssertionError(f'expected 78 L90X tasks, got {len(tasks)}')


class KVReader(nn.Module):
    """Shared query/key encoder with learnable latent and/or causal history tokens."""
    def __init__(self, mode: str, *, layer_count: int=KV_LAYERS, context_dim: int=256,
                 token_count: int=5, selected_layers: tuple[int,...]|None=None,
                 layerwise: bool=False):
        super().__init__()
        if mode not in INPUT_MODES:raise ValueError(f'unknown mode {mode}')
        self.mode=mode;self.layer_count=int(layer_count);self.context_dim=int(context_dim)
        self.selected_layers=tuple(range(layer_count)) if selected_layers is None else tuple(map(int,selected_layers))
        if not self.selected_layers or min(self.selected_layers)<0 or max(self.selected_layers)>=layer_count:
            raise ValueError('invalid selected layer IDs')
        self.layerwise=bool(layerwise)
        self.latent=nn.Parameter(torch.randn(1,token_count,context_dim)*.02)
        self.layer_embedding=nn.Parameter(torch.randn(layer_count,context_dim)*.01)
        self.key_projection=nn.Linear(KV_WIDTH,context_dim)
        self.value_projection=nn.Linear(KV_WIDTH,context_dim)
        self.history_projection=nn.Linear(8,context_dim)
        self.history_position=nn.Parameter(torch.zeros(1,10,context_dim))
        self.state_projection=nn.Sequential(nn.Linear(8,context_dim),nn.GELU(),nn.Linear(context_dim,context_dim))
        self.state_type=nn.Parameter(torch.zeros(1,1,context_dim))
        nlayer=len(self.selected_layers) if self.layerwise else 1
        self.cross=nn.ModuleList([nn.MultiheadAttention(context_dim,8,batch_first=True) for _ in range(nlayer)])
        self.self_attn=nn.MultiheadAttention(context_dim,8,batch_first=True)
        self.norm=nn.LayerNorm(context_dim);self.ff=nn.Sequential(nn.Linear(context_dim,context_dim*4),nn.GELU(),nn.Linear(context_dim*4,context_dim))
        self.logit_scale=nn.Parameter(torch.tensor(float(np.log(10.))))

    @property
    def query_count(self):
        return self.latent.shape[1] + (11 if self.mode in ('kv_h','kv_lh') else 0)

    def _queries(self, batch, history, history_mask, state):
        parts=[]
        if self.mode in ('kv_l','kv_lh'):parts.append(self.latent.expand(batch,-1,-1))
        if self.mode in ('kv_h','kv_lh'):
            if history.shape!=(batch,10,7) or history_mask.shape!=(batch,10) or state.shape!=(batch,8):
                raise ValueError('history/state shapes do not match KV-H contract')
            h=torch.cat([history,history_mask.float().unsqueeze(-1)],-1)
            parts.extend([self.history_projection(h)+self.history_position,
                          self.state_projection(state).unsqueeze(1)+self.state_type])
        return torch.cat(parts,1)

    def forward(self, kv, valid, history, history_mask, state, *, selected_layers=None):
        if kv.ndim!=5 or kv.shape[1:3]!=(self.layer_count,2) or kv.shape[-1]!=KV_WIDTH:
            raise ValueError(f'expected KV [B,{self.layer_count},2,T,256], got {tuple(kv.shape)}')
        b,_,_,t,_=kv.shape
        if valid.shape!=(b,t):raise ValueError('KV validity mask mismatch')
        layers=self.selected_layers if selected_layers is None else tuple(map(int,selected_layers))
        if not layers or min(layers)<0 or max(layers)>=self.layer_count:raise ValueError('bad layer subset')
        k=kv[:,layers,0].float();v=kv[:,layers,1].float()
        emb=self.layer_embedding[torch.as_tensor(layers,device=kv.device)].view(1,len(layers),1,-1)
        key=self.key_projection(k)+emb;value=self.value_projection(v)+emb
        q=self._queries(b,history,history_mask,state)
        padding=(~valid.bool())[:,None,:].expand(-1,len(layers),-1).reshape(b,-1)
        if self.layerwise:
            outputs=[]
            for i in range(len(layers)):
                block=self.cross[i]
                y=block(q,key[:,i],value[:,i],key_padding_mask=~valid.bool(),need_weights=False)[0]
                outputs.append(y)
            q=torch.stack(outputs).mean(0)
        else:
            q=self.cross[0](q,key.flatten(1,2),value.flatten(1,2),key_padding_mask=padding,need_weights=False)[0]
        q=q+self.self_attn(self.norm(q),self.norm(q),self.norm(q),need_weights=False)[0]
        q=q+self.ff(self.norm(q))
        return F.normalize(q,dim=-1)


def retrieval_scores(model: KVReader, query_z: torch.Tensor, candidate_z: torch.Tensor) -> torch.Tensor:
    if query_z.ndim!=3 or candidate_z.ndim!=3 or query_z.shape[1:]!=candidate_z.shape[1:]:
        raise ValueError('query/key embeddings must be aligned [N,Q,D] tensors')
    return torch.einsum('bqd,nqd->bn',query_z,candidate_z)/query_z.shape[1] * model.logit_scale.exp().clamp(max=100.)


def build_l90x_key_rows(path: Path=L90X_ROOT):
    p=Path(path);manifest=json.loads((p/'manifest.json').read_text());validate_l90x_manifest(manifest)
    r={x.stem:np.load(x,mmap_mode='r',allow_pickle=False) for x in p.glob('*.npy')}
    n=int(manifest['rows'])
    keys=np.stack([r['task'][:n],r['episode'][:n],r['frame'][:n]],axis=1).astype(np.int32)
    if len(set(map(tuple,keys.tolist())))!=n:raise ValueError('duplicate L90X key rows')
    return manifest,r,keys


def _read_transforms(checkpoint: Path):
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    norms=list((Path(checkpoint)/'assets').rglob('norm_stats.json'))
    if len(norms)!=1:raise FileNotFoundError(f'expected one pi05 norm_stats.json, got {norms}')
    trans=[LiberoInputs(ModelType.PI05),Normalize(load(norms[0].parent),use_quantiles=True),
           ResizeImages(224,224),TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    return trans


def _load_pi05_run():
    run_path=HERE.parent/'pi05_memory'/'run.py'
    spec=importlib.util.spec_from_file_location('kv1_pi05_run',run_path)
    if spec is None or spec.loader is None:raise ImportError(run_path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def extract_kv_cache(output: Path, *, checkpoint: Path, batch_size: int=4,
                     max_rows: int|None=None, device_fraction: float=.12,
                     selected_keys: np.ndarray|None=None) -> dict[str,Any]:
    """Create/restart the exact union KV cache for memory40, Eval/Dev and L90X."""
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION',str(device_fraction))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from key_latent_sources import lt1_fixed_row_keys,_lt1_source_records,_lt1_read_observation
    from collections import OrderedDict
    import key_shared_latent as key2
    sets=lt1_fixed_row_keys(RUN_ROOT)
    l90_manifest,l90_rows,l90_keys=build_l90x_key_rows()
    l90_lookup={tuple(map(int,k)):i for i,k in enumerate(l90_keys)}
    domain_rows={
      0:np.unique(np.concatenate([sets['memory40'],sets['learned40']],axis=0),axis=0).astype(np.int32),
      1:np.unique(np.concatenate([sets['unseen90'],sets['dev90']],axis=0),axis=0).astype(np.int32),
      2:l90_keys,
    }
    if selected_keys is not None:
        selected=np.asarray(selected_keys,np.int32)
        if selected.ndim!=2 or selected.shape[1]!=4:raise ValueError('selected_keys must be [N,domain+task+episode+frame]')
        domain_rows={d:selected[selected[:,0]==d,1:] for d in (0,1,2)}
        for d,keys in domain_rows.items():
            if len(set(map(tuple,keys.tolist())))!=len(keys):raise ValueError(f'duplicate selected domain-{d} rows')
    elif max_rows is not None:domain_rows={d:k[:max_rows] for d,k in domain_rows.items()}
    all_keys=np.concatenate([np.column_stack([np.full(len(k),d,np.int8),k]) for d,k in domain_rows.items()]).astype(np.int32)
    digest=hashlib.sha256(all_keys.tobytes()).hexdigest()
    feature_dir=output/'features';feature_dir.mkdir(exist_ok=True)
    manifest_path=feature_dir/'manifest.json';kv_path=feature_dir/'kv.npy';mask_path=feature_dir/'valid.npy';done_path=feature_dir/'done.npy'
    shape=(len(all_keys),KV_LAYERS,2,KV_TOKENS,KV_WIDTH)
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('row_key_sha256')!=digest or tuple(old.get('shape',()))!=shape:
            raise ValueError('existing KV cache metadata does not match exact requested row keys')
        kv=np.lib.format.open_memmap(kv_path,mode='r+',dtype=np.float16,shape=shape)
        mask=np.lib.format.open_memmap(mask_path,mode='r+',dtype=np.bool_,shape=(len(all_keys),KV_TOKENS))
        done=np.lib.format.open_memmap(done_path,mode='r+',dtype=np.bool_,shape=(len(all_keys),))
    else:
        if any(feature_dir.iterdir()):raise FileExistsError(f'unrecognized partial KV directory {feature_dir}')
        kv=np.lib.format.open_memmap(kv_path,mode='w+',dtype=np.float16,shape=shape)
        mask=np.lib.format.open_memmap(mask_path,mode='w+',dtype=np.bool_,shape=(len(all_keys),KV_TOKENS))
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype=np.bool_,shape=(len(all_keys),))
        # Rows with done=False are never read. Avoid zero-filling the 150GB+
        # mmap on shared storage; initialize only the small validity/status maps.
        mask[:]=False;done[:]=False
        np.save(feature_dir/'row_keys.npy',all_keys,allow_pickle=False)
        manifest_path.write_text(json.dumps({'contract':'pi05-prefix-kv-v1','shape':list(shape),'dtype':'float16',
          'row_key_sha256':digest,'domains':{str(d):len(k) for d,k in domain_rows.items()},
          'token_contract':'valid base[0:256]+wrist[0:256]+prompt; compact <=532',
          'keys_are_exact_task_episode_frame_plus_domain':True,'complete':False},indent=2)+'\n')
    run=_load_pi05_run();model,_=run.load_policy(Path(checkpoint))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    import types
    from openpi.shared.nnx_utils import module_jit
    # One pass returns normalized final tokens and exact RoPE-applied KV arrays.
    def prefix_with_kv(live,obs):
        obs=model_api.preprocess_observation(None,obs,train=False)
        tokens,valid,ar=live.embed_prefix(obs)
        (hidden,_),cache=live.PaliGemma.llm([tokens,None],mask=make_attn_mask(valid,ar),
                                           positions=jnp.cumsum(valid,axis=1)-1)
        return hidden,valid,cache
    # Use OpenPI's frozen-state module_jit wrapper. This preserves the same
    # NNX variable handling as the public prefix_features call while avoiding
    # returning a mutable module state from the compiled function.
    encode=module_jit(types.MethodType(prefix_with_kv,model))
    public_prefix=module_jit(model.prefix_features)
    _,task_names,hdf5_records=_lt1_source_records(RUN_ROOT)
    key2_rows=key2._load_key2_rows(RUN_ROOT)
    record_manifest=json.loads((RUN_ROOT/'data_manifest.json').read_text())
    records={(int(x['task']),int(x['episode'])):x for x in record_manifest['episodes']}
    transforms=_read_transforms(Path(checkpoint));parquet_cache=OrderedDict();todo=np.flatnonzero(~np.asarray(done,bool))
    from key_shared_latent import ExactFeatureStore
    token_store=ExactFeatureStore(RUN_ROOT,ROOT/'key-shared-01/attempt-02/prefix-missing',
        ROOT/'seg-context-01/seg5-cache',ROOT/'seg-context-01/unseen-stride5')
    run_rows=run.load_rows(RUN_ROOT)
    source_keys={
      'run':np.stack([run_rows['task'],run_rows['episode'],run_rows['frame']],axis=1).astype(np.int32),
      'seen':np.asarray(json.loads((ROOT/'seg-context-01/seg5-cache/manifest.json').read_text())['keys'],np.int32),
      'missing':np.asarray(json.loads((ROOT/'key-shared-01/attempt-02/prefix-missing/manifest.json').read_text())['keys'],np.int32),
      'unseen':np.asarray(json.loads((ROOT/'seg-context-01/unseen-stride5/manifest.json').read_text())['keys'],np.int32),
      'l90':l90_keys,
    }
    priority={0:('run','seen','missing'),1:('unseen',),2:('l90',)}
    if int(batch_size)!=4:raise ValueError('parity-preserving extraction requires batch_size=4')
    pending_keys=all_keys[todo]
    batches=source_grouped_batches(pending_keys,source_keys,source_priority=priority,batch_size=4)
    for bi,batch in enumerate(batches):
        src=batch['source'];src_ids=batch['source_indices']
        ids=todo[batch['target_indices']];target_pos=batch['target_positions']
        if src=='run':
            # run.extract pads its last partial batch by repeating the final
            # row; retain that behavior because batch shape is part of parity.
            model_src_ids=src_ids
            if len(model_src_ids)<4 and int(model_src_ids[-1])==len(source_keys['run'])-1:
                model_src_ids=np.r_[model_src_ids,np.repeat(model_src_ids[-1],4-len(model_src_ids))]
            obs=run.obs_batch(run_rows,model_src_ids)
            packed={'prompt':np.asarray(run_rows['prompt'])[model_src_ids]}
        else:
            items=[]
            for src_i in src_ids:
                t,e,f=map(int,source_keys[src][int(src_i)])
                if src=='l90':
                    li=l90_lookup[(t,e,f)]
                    row={'image':{'base_0_rgb':np.asarray(l90_rows['image_base'][li]),
                                  'left_wrist_0_rgb':np.asarray(l90_rows['image_wrist'][li])},
                         'state':np.asarray(l90_rows['state'][li]),'tokenized_prompt':np.asarray(l90_rows['prompt'][li]),
                         'tokenized_prompt_mask':np.asarray(l90_rows['prompt_mask'][li])}
                else:
                    domain_name='libero40' if src in ('seen','missing') else 'libero90'
                    raw=_lt1_read_observation((t,e,f),domain=domain_name,run_root=RUN_ROOT,records=records,
                        task_names=task_names,hdf5_records=hdf5_records,parquet_cache=parquet_cache)
                    for tr in transforms:raw=tr(raw)
                    row={'image':{'base_0_rgb':raw['image']['base_0_rgb'],'left_wrist_0_rgb':raw['image']['left_wrist_0_rgb']},
                         'state':raw['state'],'tokenized_prompt':raw['tokenized_prompt'],'tokenized_prompt_mask':raw['tokenized_prompt_mask']}
                items.append(row)
            packed={'image_base':np.stack([x['image']['base_0_rgb'] for x in items]),
                    'image_wrist':np.stack([x['image']['left_wrist_0_rgb'] for x in items]),
                    'state':np.stack([x['state'] for x in items]),'prompt':np.stack([x['tokenized_prompt'] for x in items]),
                    'prompt_mask':np.stack([x['tokenized_prompt_mask'] for x in items])}
            obs=run.obs_batch(packed,np.arange(len(src_ids)))
        hidden,valid,cache=encode(obs)
        keys=np.asarray(cache[0]);values=np.asarray(cache[1]);valid=np.asarray(valid,bool)
        model_batch_size=int(valid.shape[0])
        # Match core.MemoryPi05.prefix_features, which excludes tokenizer
        # sentinel IDs (PAD/BOS/EOS) from its language-token validity mask.
        text=np.asarray(packed['prompt'])>2
        valid=valid.copy();valid[:,3*256:3*256+text.shape[1]] &= text
        # core.prefix_features drops absent-right-camera positions before text.
        h,g=final_tokens_from_prefix(hidden,valid)
        kvrow,validrow=pack_prefix_kv(keys,values,valid)
        if kvrow.shape!=(model_batch_size,KV_LAYERS,2,KV_TOKENS,KV_WIDTH):raise AssertionError(kvrow.shape)
        core_metrics=[];core_vectors={}
        if int(np.asarray(done).sum())==0:
            core_h,core_g=public_prefix(obs)
            core_h=np.asarray(core_h);core_g=np.asarray(core_g)
            for j,ix in zip(target_pos,ids,strict=True):
                n=int(np.sum(core_g[j]>=0));compact_core=core_h[j,core_g[j]>=0].astype(np.float32)
                core_metrics.append({'identity':all_keys[int(ix)].tolist(),
                                     **prefix_parity_metrics(compact_core,h[j,:n])})
                core_vectors[int(ix)]=compact_core
                if len(core_vectors)==1:
                    np.savez(feature_dir/'parity_vectors.npz', identity=all_keys[int(ix)],
                             extracted=h[j,:n].astype(np.float16), core=compact_core.astype(np.float16),
                             groups=core_g[j,core_g[j]>=0])
        kv[ids]=kvrow[target_pos];mask[ids]=validrow[target_pos];done[ids]=True
        if bi%16==0:
            kv.flush();mask.flush();done.flush()
        # On exact non-L90X cache rows, verify parity against the existing final-token cache.
        check_ids=[int(ix) for ix in ids if int(all_keys[ix,0]) in (0,1)]
        parity_rows=[]
        if check_ids:
            for ix in check_ids[:2]:
                d,t,e,f=map(int,all_keys[ix]);ref,grp=token_store.get(np.asarray([[t,e,f]],np.int32),domain='libero40' if d==0 else 'libero90')
                n=int((grp[0]>=0).sum());ours=h[target_pos[np.where(ids==ix)[0][0]],:n]
                row_metrics=prefix_parity_metrics(ref[0,:n],ours)
                core_match=core_vectors.get(ix)
                core_row_metrics=prefix_parity_metrics(ref[0,:n],core_match) if core_match is not None else None
                parity_rows.append({'identity':[d,t,e,f],**row_metrics,
                                    'public_core_metrics':core_row_metrics,
                                    'passed':parity_passes(core_row_metrics or row_metrics)})
        diagnostics={'core_prefix_method':core_metrics,'existing_cache':parity_rows}
        with (feature_dir/'parity_rows.jsonl').open('a',encoding='utf-8') as parity_log:
            for row in parity_rows:
                parity_log.write(json.dumps(row)+'\n')
        (feature_dir/'last_parity_check.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        if bi==0 or not (feature_dir/'first_parity_check.json').exists():
            (feature_dir/'first_parity_check.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        if parity_rows and not all(x['passed'] for x in parity_rows):
            raise AssertionError(f'final-prefix parity failed: {json.dumps(diagnostics)}')
        if bi%32==0:
            (feature_dir/'progress.json').write_text(json.dumps({'completed':int(np.asarray(done).sum()),'total':len(done),'updated':time.time()}))
    kv.flush();mask.flush();done.flush()
    if not np.asarray(done).all():raise RuntimeError(f'KV cache incomplete: {int((~np.asarray(done)).sum())}')
    final=json.loads(manifest_path.read_text());final.update(complete=True,final_layer_token_parity_checked=True,
      rows_completed=int(len(done)),max_prefix_tokens=KV_TOKENS,checkpoint=str(checkpoint))
    manifest_path.write_text(json.dumps(final,indent=2)+'\n')
    return final


class KVFeatureVault:
    """Exact-key, mmap-backed access to the large immutable KV feature array."""
    def __init__(self, feature_dir: Path):
        self.root=Path(feature_dir)
        self.meta=json.loads((self.root/'manifest.json').read_text())
        if self.meta.get('complete') is not True:raise ValueError(f'incomplete KV cache: {self.root}')
        self.keys=np.load(self.root/'row_keys.npy',mmap_mode='r',allow_pickle=False)
        self.kv=np.load(self.root/'kv.npy',mmap_mode='r',allow_pickle=False)
        self.valid=np.load(self.root/'valid.npy',mmap_mode='r',allow_pickle=False)
        if self.kv.shape!=(len(self.keys),KV_LAYERS,2,KV_TOKENS,KV_WIDTH):raise ValueError('bad KV cache shape')
        self.lookup={tuple(map(int,k)):i for i,k in enumerate(self.keys)}
    def indices(self, keys, domain: int):
        out=[];missing=[]
        for k in np.asarray(keys):
            identity=(int(domain),*map(int,k))
            if identity not in self.lookup:missing.append(identity)
            else:out.append(self.lookup[identity])
        if missing:raise KeyError(f'{len(missing)} rows missing from KV cache; examples={missing[:3]}')
        return np.asarray(out,np.int64)
    def take(self, ids):
        ids=np.asarray(ids,np.int64)
        return np.asarray(self.kv[ids]),np.asarray(self.valid[ids])


def _l90x_training_rows(vault: KVFeatureVault):
    manifest,rows,keys=build_l90x_key_rows()
    validate_l90x_manifest(manifest)
    n=len(keys);hist=np.asarray(rows['hist'],np.float32)
    if hist.shape!=(n,10,7):raise ValueError(f'L90X hist must be [N,10,7], got {hist.shape}')
    state=np.asarray(rows['state'],np.float32)
    actions=np.asarray(rows['actions'],np.float32)
    if actions.ndim!=3 or actions.shape[1]!=10 or actions.shape[2]<7:raise ValueError('L90X actions must include 10 active7 steps')
    select=np.asarray(rows['is_select'],bool)
    # KV1 uses the external LIBERO-90 task-6/7 Dev for every model choice.
    # Therefore all allowed L90X rows are fit data; the old X1 selection-task
    # annotation is recorded but must not silently discard those 16 tasks.
    fit=np.arange(n,dtype=np.int64);held=np.flatnonzero(select)
    if not len(fit):raise ValueError('L90X fit rows unexpectedly empty')
    # A preceding history exists for every non-initial stride-10 row; frame 0
    # gets a fully masked zero history. No future action enters this input.
    hmask=np.broadcast_to((np.asarray(rows['has_hist'],bool)[:,None]),(n,10)).copy()
    return {'fit':{'keys':keys[fit],'kv_ids':vault.indices(keys[fit],2),'history':hist[fit],
                   'history_mask':hmask[fit],'state':state[fit,:8],'actions':actions[fit,:,:7],
                   'task':np.asarray(rows['task'])[fit],'episode':np.asarray(rows['episode'])[fit]},
            'legacy_selection_diagnostic':{'keys':keys[held],'kv_ids':vault.indices(keys[held],2),'history':hist[held],
                   'history_mask':hmask[held],'state':state[held,:8],'actions':actions[held,:,:7],
                   'task':np.asarray(rows['task'])[held],'episode':np.asarray(rows['episode'])[held]},
            'manifest':{'fit_rows':len(fit),'legacy_selection_rows_included':len(held),
                        'fit_tasks':manifest['tasks'],'legacy_selection_tasks':manifest['selection_tasks'],
                        'missing_source_tasks':manifest.get('missing_tasks',[]),
                        'model_selection_dev':'external LIBERO-90 tasks 6/7'}}


def _reader_batch(model, vault, split, indices, device):
    import torch
    ids=np.asarray(split['kv_ids'])[np.asarray(indices,np.int64)]
    kv,valid=vault.take(ids)
    # Copy mmap slices before torch conversion to avoid read-only tensor aliasing.
    return (torch.tensor(kv,device=device,dtype=torch.float16),
            torch.tensor(valid,device=device,dtype=torch.bool),
            torch.as_tensor(np.asarray(split['history'])[indices].copy(),device=device,dtype=torch.float32),
            torch.as_tensor(np.asarray(split['history_mask'])[indices].copy(),device=device,dtype=torch.bool),
            torch.as_tensor(np.asarray(split['state'])[indices].copy(),device=device,dtype=torch.float32))


def _encode_split(model, vault, split, device, *, batch=2, selected_layers=None):
    import torch
    model.eval();parts=[]
    with torch.no_grad():
        for s in range(0,len(split['kv_ids']),batch):
            ids=np.arange(s,min(s+batch,len(split['kv_ids'])))
            kv,valid,h,hm,st=_reader_batch(model,vault,split,ids,device)
            parts.append(model(kv,valid,h,hm,st,selected_layers=selected_layers).cpu().numpy())
    return np.concatenate(parts,axis=0)


def _retrieval_eval(model, vault, bank, bank_input, bank_kv_ids, split, device, *, batch=2,selected_layers=None):
    import torch
    model.eval();key_parts=[]
    with torch.no_grad():
        for s in range(0,len(bank_kv_ids),batch):
            ids=np.arange(s,min(s+batch,len(bank_kv_ids)))
            block=dict(bank_input)
            block['kv_ids']=bank_kv_ids
            kv,valid,h,hm,st=_reader_batch(model,vault,block,ids,device)
            key_parts.append(model(kv,valid,h,hm,st,selected_layers=selected_layers).cpu().numpy())
    key_z=np.concatenate(key_parts,axis=0)
    query_z=_encode_split(model,vault,split,device,batch=batch,selected_layers=selected_layers)
    scores=np.einsum('bqd,nqd->bn',query_z,key_z,optimize=True)/query_z.shape[1]
    candidates=np.asarray(bank['value_actions10'],np.float32)
    errors=np.empty((len(split['actions']),len(candidates)),np.float32)
    import key_shared_latent as key2
    errors[:]=key2._arm6_candidate_mse(np.asarray(split['actions']),candidates)
    order=np.argsort(-scores,axis=1,kind='stable')
    first=order[:,0];top=errors[np.arange(len(errors)),first];best=errors.min(1)
    from scipy.stats import spearmanr
    rho=np.asarray([spearmanr(scores[i],-errors[i]).statistic for i in range(len(errors))],np.float64)
    hard=np.asarray(split.get('hard',np.zeros(len(top),bool)),bool)
    selected_actions=candidates[first]
    action_metrics=key2.action_errors(selected_actions,np.asarray(split['actions']))
    ci=key2.episode_bootstrap_ci(top,np.asarray(split['task']),np.asarray(split['episode']),reps=5000,seed=4101)
    hard_ci=(key2.episode_bootstrap_ci(top[hard],np.asarray(split['task'])[hard],np.asarray(split['episode'])[hard],reps=5000,seed=4102)
             if hard.any() else None)
    return {'scores':scores,'errors':errors,'top1_indices':first,'top1_errors':top,
            'top1_actions':selected_actions,'candidate_z':key_z,
            'metrics':{'top1_arm_mse':float(top.mean()),'oracle_arm_mse':float(best.mean()),
                       'top1_active7_mse':float(action_metrics['active7'].mean()),
                       'top1_active7_mse_hard':float(action_metrics['active7'][hard].mean()) if hard.any() else None,
                       'top1_arm_mse_all':float(top.mean()),
                       'top1_arm_mse_hard':float(top[hard].mean()) if hard.any() else None,
                       'oracle_arm_mse_hard':float(best[hard].mean()) if hard.any() else None,
                       'top1_arm_mse_ci95':ci,
                       'top1_arm_mse_hard_ci95':hard_ci,
                       'spearman_positive':float(np.nanmean(rho)),
                       'recall_at_1':float(np.mean(first==np.argmin(errors,axis=1))),
                       'recovery':float(np.mean((errors.mean(1)-top)/np.maximum(errors.mean(1)-best,1e-12)))} }


def _paired_delta_ci(a, b, split, *, seed=4101):
    """Episode-paired CI for scalar per-query errors; reports a minus b."""
    import key_shared_latent as key2
    d=np.asarray(a,np.float32)-np.asarray(b,np.float32)
    return key2.episode_bootstrap_ci(d,np.asarray(split['task']),np.asarray(split['episode']),reps=5000,seed=seed)


def _reference_error(reference_path: Path, split):
    """Load an existing paired prediction only after exact query-key alignment."""
    z=np.load(reference_path,allow_pickle=False)
    refkeys=np.asarray(z['key']);lookup={tuple(map(int,k)):i for i,k in enumerate(refkeys)}
    keys=np.asarray(split['key']);
    if any(tuple(map(int,k)) not in lookup for k in keys):raise KeyError(f'reference rows missing from {reference_path}')
    ix=np.asarray([lookup[tuple(map(int,k))] for k in keys],np.int64)
    pred=np.asarray(z['top1_actions'][ix],np.float32);target=np.asarray(split['actions'],np.float32)
    if not np.array_equal(np.asarray(z['target_actions'][ix]),target):raise AssertionError('reference target actions do not match fixed query rows')
    return np.square(pred[...,:6]-target[...,:6]).sum((1,2))/(10*7.)


def _pi05_nn_baseline(bank, split, split_name: str, prediction_dir: Path):
    """Exact-row π0.5 action-nearest unit baseline on the fixed query split."""
    import key_shared_latent as key2
    selection=np.load(ROOT/'pi05-reference-01/selection.npz',allow_pickle=False)
    if split_name=='unseen':
        z=np.load(ROOT/'pi05-reference-01/predictions/unseen_cached.npz',allow_pickle=False)
        pred=np.asarray(z['actions'],np.float32);row=np.asarray(z['row'],np.int64)
    else:
        parts=[]
        for path in sorted((ROOT/'pi05-reference-01/predictions/learned_parts').glob('batch-*.npz')):
            q=np.load(path,allow_pickle=False);parts.append((np.asarray(q['row']),np.asarray(q['actions'],np.float32)))
        row=np.concatenate([x[0] for x in parts]);pred=np.concatenate([x[1] for x in parts])
    if len(set(map(int,row)))!=len(row):raise ValueError('duplicate π0.5 reference row IDs')
    if split_name=='unseen':
        source={int(r):(int(t),int(e),int(f)) for r,t,e,f in zip(selection['unseen_row'],selection['unseen_task'],selection['unseen_episode'],selection['unseen_frame'],strict=True)}
    else:
        source={int(r):(int(t),int(e),int(f)) for r,t,e,f in zip(selection['learned_row'],selection['learned_task'],selection['learned_episode'],selection['learned_frame'],strict=True)}
    pred_map={int(r):a for r,a in zip(row,pred,strict=True)}
    inverse={v:k for k,v in source.items()}
    keys=[tuple(map(int,k)) for k in np.asarray(split['key'])]
    missing=[k for k in keys if k not in inverse]
    if missing:raise KeyError(f'π0.5 reference key missing in {split_name}: {missing[:3]}')
    rows=np.asarray([inverse[k] for k in keys],np.int64)
    absent=[int(r) for r in rows if int(r) not in pred_map]
    if absent:raise KeyError(f'π0.5 predictions missing row IDs: {absent[:3]}')
    qpred=np.stack([pred_map[int(r)] for r in rows])
    errors=key2._arm6_candidate_mse(qpred,np.asarray(bank['value_actions10'],np.float32))
    chosen=np.argmin(errors,axis=1);actions=np.asarray(bank['value_actions10'],np.float32)[chosen]
    eval_errors=key2._arm6_candidate_mse(np.asarray(split['actions']),np.asarray(bank['value_actions10']))
    picked=eval_errors[np.arange(len(chosen)),chosen];hard=np.asarray(split['hard'],bool)
    active=key2.action_errors(actions,np.asarray(split['actions']))['active7']
    np.savez_compressed(prediction_dir/f'pi05_nn_{split_name}.npz',key=np.asarray(split['key']),task=np.asarray(split['task']),
        episode=np.asarray(split['episode']),hard=hard,nearest_indices=chosen,top1_errors=picked,
        top1_actions=actions,target_actions=np.asarray(split['actions']),query_pi05=qpred)
    return {'metrics':{'top1_arm_mse':float(picked.mean()),'hard_top1_arm_mse':float(picked[hard].mean()),
                       'top1_active7_mse':float(active.mean()),'top1_active7_mse_hard':float(active[hard].mean()),
                       'rows':len(picked),'hard_rows':int(hard.sum()),
                       'query_source':'official corrected-orientation pi05-reference-01 exact-row predictions'},
            'errors':picked}


def _evaluate_keygrid_history(bank, bank_input, split, device, prediction_path: Path):
    """Re-run the saved Dev-selected KEYGRID H-MLP on exact aligned rows."""
    import key_grid
    import key_shared_latent as key2
    meta=json.loads((ROOT/'key-grid-01/results.json').read_text())
    variants=meta['new_models']['history_state']['variants']
    chosen=min(variants.values(),key=lambda x:float(x['dev_top1_arm6_mse']))
    qfeat=key_grid.make_vector_features(split,'history_state')
    kfeat=key_grid.make_vector_features(bank_input,'history_state')
    model=key_grid.PairMLPRanker(qfeat.shape[1],shared=chosen['fusion']=='shared_mlp').to(device)
    model.load_state_dict(torch.load(chosen['checkpoint'],map_location=device,weights_only=True))
    scores=key_grid._score_vectors(model,qfeat,kfeat,device=str(device),batch=128)
    errors=key2._arm6_candidate_mse(np.asarray(split['actions']),np.asarray(bank['value_actions10']))
    first=np.argmax(scores,axis=1);pred=np.asarray(bank['value_actions10'])[first]
    err=errors[np.arange(len(first)),first]
    hard=np.asarray(split['hard'],bool)
    active=key2.action_errors(pred,np.asarray(split['actions']))['active7']
    np.savez_compressed(prediction_path,key=np.asarray(split['key']),task=np.asarray(split['task']),
        episode=np.asarray(split['episode']),hard=hard,top1_actions=pred,target_actions=np.asarray(split['actions']),
        top1_errors=err,scores=scores)
    return {'metrics':{'top1_arm_mse':float(err.mean()),'hard_top1_arm_mse':float(err[hard].mean()),
            'top1_active7_mse':float(active.mean()),'top1_active7_mse_hard':float(active[hard].mean()),
            'dev_selection':{'fusion':chosen['fusion'],'learning_rate':chosen['learning_rate'],
                'epochs':chosen['epochs'],'teacher_tau':chosen['teacher_temperature'],
                'dev_top1_arm_mse':chosen['dev_top1_arm6_mse']},
            **key2.ranking_metrics(scores,errors)},'errors':err}


def _vector_history_eval(model, key_grid, query_features, candidate_features, bank, split, device):
    import key_shared_latent as key2
    scores=key_grid._score_vectors(model,query_features,candidate_features,device=str(device),batch=128)
    errors=key2._arm6_candidate_mse(np.asarray(split['actions']),np.asarray(bank['value_actions10']))
    first=np.argmax(scores,axis=1);pred=np.asarray(bank['value_actions10'])[first]
    selected=errors[np.arange(len(first)),first];hard=np.asarray(split['hard'],bool)
    action=key2.action_errors(pred,np.asarray(split['actions']))['active7']
    rank=key2.ranking_metrics(scores,errors)
    return {'scores':scores,'errors':selected,'prediction':pred,
        'metrics':rank|{'top1_arm_mse':float(selected.mean()),'oracle_arm_mse':rank['querywise_oracle_arm6_mse'],
          'top1_active7_mse':float(action.mean()),'top1_active7_mse_hard':float(action[hard].mean()),
          'hard_top1_arm_mse':float(selected[hard].mean()),'rows':len(first),'hard_rows':int(hard.sum())}}


def _train_history_vector_baselines(output, bank, bank_input, fit_bank, fit_bank_input,
                                    train40, learned, unseen, dev, l90, *, device,
                                    epochs=8, learning_rate=3e-4, teacher_tau=.005,
                                    batch_size=8, random_candidates=128):
    """Retrain matched KEYGRID history/state MLP baselines for both regimes."""
    import key_grid, key_shared_latent as key2
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fit_ids=training_bank_indices(bank['task'],bank['episode'],
        set(zip(np.asarray(learned['task']).astype(int).tolist(),np.asarray(learned['episode']).astype(int).tolist())))
    # Align the stricter fit bank used by KV1; never assume array order.
    fit_lookup={tuple(map(int,k)):i for i,k in enumerate(fit_bank_input['key'])}
    bank_fit_ids=np.asarray([fit_lookup[tuple(map(int,k))] for k in bank_input['key'][fit_ids]],np.int64)
    if not np.array_equal(np.asarray(bank_input['key'])[fit_ids],np.asarray(fit_bank_input['key'])):
        raise AssertionError('history MLP fit bank does not match KV1 episode-exclusion bank')
    fit_part={k:(np.asarray(v)[fit_ids] if isinstance(v,np.ndarray) and len(v)==len(bank['length']) else v)
               for k,v in bank_input.items()}
    full_features=key_grid.make_vector_features(bank_input,'history_state')
    fit_features=full_features[fit_ids]
    candidate_features=key_grid.make_vector_features(fit_part,'history_state')
    cand_actions=np.asarray(fit_bank['value_actions10'],np.float32)
    if len(candidate_features)!=len(cand_actions):raise AssertionError('fit history keys/actions mismatch')
    split_features={name:key_grid.make_vector_features(part,'history_state') for name,part in
        (('train40',train40),('learned',learned),('unseen',unseen),('dev',dev))}
    qsets={'libero40':train40}
    qsets['libero40+l90x']={k:np.concatenate([np.asarray(train40[k]),np.asarray(l90['fit'][k])],axis=0)
        for k in ('keys','history','history_mask','state','actions','task','episode')}
    qsets['libero40+l90x']['bank_ids']=np.r_[np.asarray(train40['bank_ids']),np.full(len(l90['fit']['keys']),-1,np.int64)]
    l90_features=key_grid.make_vector_features(l90['fit'],'history_state')
    split_features['train40+l90x']=np.concatenate([split_features['train40'],l90_features],axis=0)
    results={};paired={};ckpt_dir=output/'checkpoints';ckpt_dir.mkdir(exist_ok=True)
    for regime,qdata in qsets.items():
        torch.manual_seed(1);torch.cuda.manual_seed_all(1) if str(device).startswith('cuda') else None
        model=key_grid.PairMLPRanker(full_features.shape[1],shared=False).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=float(learning_rate),weight_decay=1e-4)
        rng=np.random.default_rng(20261104);indices=np.arange(len(qdata['keys']));best=None;history=[]
        qfeatures=split_features['train40' if regime=='libero40' else 'train40+l90x']
        for ep in range(int(epochs)):
            model.train();rng.shuffle(indices);losses=[]
            for start in range(0,len(indices),int(batch_size)):
                qi=indices[start:start+int(batch_size)]
                random_ids=rng.choice(len(fit_features),size=min(int(random_candidates),len(fit_features)),replace=False)
                positives=np.asarray(qdata['bank_ids'])[qi];positives=positives[positives>=0]
                cids=np.unique(np.r_[positives,random_ids])
                qt=np.asarray(qdata['task'])[qi];qe=np.asarray(qdata['episode'])[qi]
                allowed=~((qt[:,None]==np.asarray(fit_bank['task'])[cids][None,:])&
                           (qe[:,None]==np.asarray(fit_bank['episode'])[cids][None,:]))
                if regime=='libero40+l90x':allowed[qi>=len(train40['keys'])]=True
                q=torch.as_tensor(qfeatures[qi],device=device,dtype=torch.float32)
                k=torch.as_tensor(candidate_features[cids],device=device,dtype=torch.float32)
                scores=model(q,k)*model.logit_scale.exp().clamp(max=key2.LOGIT_SCALE_MAX)
                errs=key2._arm6_candidate_mse(np.asarray(qdata['actions'])[qi],cand_actions[cids])
                loss=key2.contrastive_teacher_loss(scores,torch.as_tensor(errs,device=device),
                    torch.as_tensor(allowed,device=device),objective='soft',teacher_temperature=float(teacher_tau))
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();key2._clamp_logit_scale_(model);losses.append(float(loss.detach().cpu()))
            devout=_vector_history_eval(model,key_grid,split_features['dev'],full_features,bank,dev,device)
            metric=devout['metrics']['top1_arm6_mse'];history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),
                'dev_top1_arm_mse':metric})
            if best is None or metric<best['dev_top1_arm_mse']:
                best={'epoch':ep+1,'dev_top1_arm_mse':metric,
                      'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
                torch.save(best['state'],ckpt_dir/f'history_state_{regime.replace("+","_")}.pt')
            with (output/'progress.jsonl').open('a') as log:
                log.write(json.dumps({'model':regime,'epoch':ep+1,'train_loss':history[-1]['train_loss'],
                                      'dev_top1_arm_mse':metric})+'\n')
        model.load_state_dict(best['state']);model.eval();splits={}
        for name,split in (('learned',learned),('unseen',unseen)):
            block=_vector_history_eval(model,key_grid,split_features[name],full_features,bank,split,device)
            splits[name]=block['metrics'];paired[(regime,name)]=block['errors']
            np.savez_compressed(output/f'history_state_{regime.replace("+","_")}_{name}.npz',
                key=np.asarray(split['key']),task=np.asarray(split['task']),episode=np.asarray(split['episode']),
                hard=np.asarray(split['hard']),top1_errors=block['errors'],top1_actions=block['prediction'],
                target_actions=np.asarray(split['actions']),scores=block['scores'])
        results[regime]={'selected_epoch':best['epoch'],'dev_top1_arm_mse':best['dev_top1_arm_mse'],
                         'history':history,'splits':splits,'checkpoint':str(ckpt_dir/f'history_state_{regime.replace("+","_")}.pt'),
                         'training_rows':len(qdata['keys']),'fit_candidates':len(fit_features)}
        del model,opt
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return {'models':results,'paired_errors':paired,'fit_candidate_rows':len(fit_features)}


def _task_id_linear_probe(z: np.ndarray, task: np.ndarray, episode: np.ndarray) -> dict[str,Any]:
    """Five-fold episode-disjoint linear probe of task identity (diagnostic only)."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GroupKFold
    x=np.asarray(z,np.float32).reshape(len(z),-1);y=np.asarray(task);ep=np.asarray(episode)
    groups=np.asarray([f'{int(t)}:{int(e)}' for t,e in zip(y,ep,strict=True)])
    scores=[]
    for tr,va in GroupKFold(n_splits=5).split(x,y,groups):
        clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=500,tol=1e-3,random_state=4101)
        clf.fit(x[tr],y[tr]);scores.append(float(np.mean(clf.predict(x[va])==y[va])))
    return {'fold_accuracy':scores,'mean_accuracy':float(np.mean(scores)),
            'chance_by_class_count':float(1/len(np.unique(y))),'rows':len(y),'episode_groups':len(np.unique(groups))}


def write_kv1_report(results: dict[str,Any], output_root: Path) -> Path:
    """Render the primary top-1/active7 table and record unavailable rows plainly."""
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    lines=['# KV1: π0.5 全層 prefix KV の読み出し','',
           '主指標は 10-step 行動の腕6軸二乗誤差和/(10×7)。Eval の行動は採点のみ、モデル・epoch は Dev top-1 arm MSE で選択した。','',
           '## 主表：全学習済みの top-1','',
           '| 条件 | split | top-1 腕MSE (全体 / hard) | active7 (全体 / hard) | oracle 腕MSE | Spearman | recall@1 | recovery |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for tag,entry in results.get('models',{}).items():
        for split,metrics in entry.get('splits',{}).items():
            lines.append(f"| {tag} | {split} | {metrics.get('top1_arm_mse',float('nan')):.5f} / {metrics.get('hard_top1_arm_mse',float('nan')):.5f} | {metrics.get('top1_active7_mse',float('nan')):.5f} / {metrics.get('top1_active7_mse_hard',float('nan')):.5f} | {metrics.get('oracle_arm_mse',float('nan')):.5f} | {metrics.get('spearman_positive',float('nan')):.4f} | {metrics.get('recall_at_1',float('nan')):.4f} | {metrics.get('recovery',float('nan')):.4f} |")
    for split,rows in results.get('reference_rows',{}).items():
        for tag,metrics in rows.items():
            lines.append(f"| {tag} (existing reference) | {split} | {metrics.get('top1_arm_mse',float('nan')):.5f} / {metrics.get('hard_top1_arm_mse',float('nan')):.5f} | {metrics.get('top1_active7_mse',float('nan')):.5f} / {metrics.get('top1_active7_mse_hard',float('nan')):.5f} | — | — | — | — |")
    lines += ['', '## 対応付き差（episode bootstrap 95% CI）','',
      '| 比較（左−右） | 平均差 | 95% CI | episodes |', '|---|---:|---:|---:|']
    for label,ci in results.get('paired_episode_bootstrap_5000',{}).items():
        lines.append(f"| {label} | {ci['mean']:.5f} | [{ci['ci95_low']:.5f}, {ci['ci95_high']:.5f}] | {ci['episodes']} |")
    lines += ['', '## 検証・比較','',
      f"cache domain counts: {json.dumps(results.get('fixed_counts',{}),ensure_ascii=False)}。",'',
      'Dev は LIBERO-90 task 6/7 の固定窓。最終 Eval は learned 621窓・unseen 619窓を各 checkpoint で一度採点した。hard subset の定義は KEY2 固定行を継承。',
      '', '## 範囲と制限','',
      '実装・評価したのは KV-L / KV-H / KV-LH、LIBERO-40 のみと LIBERO-40+L90X、および KV-L 全層 checkpoint の事前指定3層組合せから Dev 選択した post-hoc KV-L-sel、KV-L の layerwise cross-attention である。',
      '比較用の履歴+状態 MLP は両 regime で同じ学習行・soft teacher・Dev top-1 選択により再学習した。既存 KEY2-F の E/P5 と固定 baseline は、同じ query keys を確認した参照として results.json に保持した。',
      'AX-final は既存 LT1 cache が [N,3,3,10,256] の suffix V taps で、仕様の [N,10,1024] final action-expert output と同一でないため代用せず未実施。代わりに layerwise KV cross-attention を評価した。',
      f"L90X cache manifest は適格78 task IDs、実データ77 task IDs / {results.get('fixed_counts',{}).get('l90x_fit',0)} fit rows。task 51 は HDF5 がなく行データなし。Dev/Eval task は除外済み。",
      '選択・評価は単位再生時に保存された10-step action valuesを対象とした。KV-L-sel の3層候補は事前指定の組を Dev top-1 arm MSE で選んだ。']
    path=output_root/'report.md';path.write_text('\n'.join(lines)+'\n')
    return path


def train_and_evaluate(output: Path, *, cache_dir: Path, epochs: int=8,
                       batch_size: int=8, random_candidates: int=128,
                       learning_rate: float=3e-4, teacher_tau: float=.005,
                       memory_fraction: float=.22, device='cuda:0'):
    """Train KV-L/H/LH on memory-only and memory+L90X; select by Dev top-1 arm MSE."""
    import torch
    import key_shared_latent as key2
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    device=torch.device(device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(float(memory_fraction),device.index or 0)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    vault=KVFeatureVault(Path(cache_dir)/'features')
    # Reuse KEY2's locked units, exclusions, exact query identities and action evaluation sets.
    bank,bank_input,fit_bank,fit_bank_input,train40,learned,unseen,dev=key2._build_key2_queries(
        RUN_ROOT,ROOT/'key-shared-01/attempt-02/prefix-missing',ROOT/'seg-context-01/seg5-cache',
        ROOT/'seg-context-01/unseen-stride5',ROOT/'key-shared-01/attempt-27-production/vjepa/libero90_eval')
    if len(bank['length'])!=4516 or len(learned['key'])!=621 or len(unseen['key'])!=619:
        raise AssertionError('fixed KEY2 row/unit counts changed')
    bank_ids=vault.indices(bank_input['key'],0)
    bank_actions=np.asarray(bank['value_actions10'],np.float32)
    bank_task=np.asarray(bank['task']);bank_ep=np.asarray(bank['episode'])
    train40=dict(keys=train40['key'],kv_ids=vault.indices(train40['key'],0),
        history=np.asarray(train40['history'],np.float32),history_mask=np.asarray(train40['history_mask'],bool),
        state=np.asarray(train40['state'],np.float32),actions=np.asarray(train40['actions'],np.float32),
        task=np.asarray(train40['task']),episode=np.asarray(train40['episode']),
        bank_ids=np.asarray([vault.lookup[(0,*map(int,k))] for k in train40['key']],np.int64))
    learned_split={k:v for k,v in learned.items() if isinstance(v,np.ndarray)}
    learned_split.update(kv_ids=vault.indices(learned['key'],0))
    unseen_split={k:v for k,v in unseen.items() if isinstance(v,np.ndarray)}
    unseen_split.update(kv_ids=vault.indices(unseen['key'],1))
    dev_split={k:v for k,v in dev.items() if isinstance(v,np.ndarray)}
    dev_split.update(kv_ids=vault.indices(dev['key'],1))
    l90=_l90x_training_rows(vault)
    # Normalize L90X source structure to the same trainer contract.
    qsets={'libero40':train40,'libero40+l90x':{k:np.concatenate([train40[k],l90['fit'][k]],axis=0)
        for k in ('keys','kv_ids','history','history_mask','state','actions','task','episode')}}
    # The training teacher bank excludes every fixed learned-evaluation
    # episode, for both LIBERO-40 and L90X queries. Eval and Dev use the full
    # frozen 4,516-unit bank only at scoring time.
    excluded_pairs=set(zip(np.asarray(learned['task']).astype(int).tolist(),
                           np.asarray(learned['episode']).astype(int).tolist()))
    fit_bank_ids=training_bank_indices(bank['task'],bank['episode'],excluded_pairs)
    if not np.array_equal(np.asarray(fit_bank['source']),np.asarray(bank['source'])[fit_bank_ids]):
        raise AssertionError('KEY2 fit-bank helper and KV1 episode-exclusion mask differ')
    fit_lookup={tuple(map(int,k)):i for i,k in enumerate(fit_bank_input['key'])}
    fit_kv_ids=vault.indices(fit_bank_input['key'],0)
    qsets['libero40']['bank_ids']=np.asarray([fit_lookup[tuple(map(int,k))] for k in train40['keys']],np.int64)
    qsets['libero40+l90x']['bank_ids']=np.r_[qsets['libero40']['bank_ids'],np.full(len(l90['fit']['keys']),-1,np.int64)]
    def evaluate(model, split, selected_layers=None):
        return _retrieval_eval(model,vault,bank,bank_input,bank_ids,split,device,batch=8,selected_layers=selected_layers)
    results={'status':'running','seed':1,'seed_list':[1],'teacher_tau':teacher_tau,
        'candidate_sampling':f'in-batch memory unit starts + {random_candidates} uniform memory units; same-episode keys masked',
        'selected_by':'minimum Dev top-1 arm MSE; Eval labels are not used for selection',
        'fixed_counts':{'memory_units':len(bank['length']),'learned':len(learned['key']),'unseen':len(unseen['key']),
                        'dev':len(dev['key']),'l90x_fit':l90['manifest']['fit_rows'],
                        'l90x_legacy_selection_rows_included':l90['manifest']['legacy_selection_rows_included'],
                        'l90x_dev':0},
        'models':{}}
    parity_path=Path(cache_dir)/'features'/'first_parity_check.json'
    if parity_path.exists():results['prefix_parity_smoke']=json.loads(parity_path.read_text())
    models_dir=output/'checkpoints';models_dir.mkdir(exist_ok=True)
    predictions_dir=output/'paired_predictions';predictions_dir.mkdir(exist_ok=True)
    paired_errors={}
    for regime,qdata in qsets.items():
      configs=(('kv_l',False),('kv_h',False),('kv_lh',False),('kv_l_layerwise',True))
      for mode,layerwise in configs:
        model_mode='kv_l' if mode=='kv_l_layerwise' else mode
        tag=f'{mode}_{regime.replace("+","_")}'
        torch.manual_seed(1)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(1)
        model=KVReader(model_mode,layerwise=layerwise).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=float(learning_rate),weight_decay=1e-4)
        rng=np.random.default_rng(20261104);history=[];best=None
        for epoch in range(int(epochs)):
            model.train();perm=rng.permutation(len(qdata['keys']));losses=[]
            for start in range(0,len(perm),int(batch_size)):
                qi=perm[start:start+int(batch_size)]
                if not len(qi):continue
                inbank=np.asarray(qdata['bank_ids'])[qi];inbank=inbank[inbank>=0]
                random_ids=rng.choice(len(fit_bank['length']),size=min(int(random_candidates),len(fit_bank['length'])),replace=False)
                cids=np.unique(np.r_[inbank,random_ids])
                qt=np.asarray(qdata['task'])[qi];qe=np.asarray(qdata['episode'])[qi]
                ct=np.asarray(fit_bank['task'])[cids];ce=np.asarray(fit_bank['episode'])[cids]
                allowed=~((qt[:,None]==ct[None,:])&(qe[:,None]==ce[None,:]))
                if regime=='libero40+l90x' and np.any(qi>=len(train40['keys'])):
                    l90_rows=qi>=len(train40['keys'])
                    # Task and episode numbers are local to datasets; never
                    # treat coincident LIBERO-40/LIBERO-90 IDs as one episode.
                    allowed[l90_rows]=True
                qkv,qmask,qh,qhm,qs=_reader_batch(model,vault,qdata,qi,device)
                # Candidate observations include their own preceding action history and state.
                bkv,bmask,bh,bhm,bs=_reader_batch(model,vault,fit_bank_input|{'kv_ids':fit_kv_ids},cids,device)
                qz=model(qkv,qmask,qh,qhm,qs);kz=model(bkv,bmask,bh,bhm,bs)
                scores=torch.einsum('bqd,nqd->bn',qz,kz)/qz.shape[1]*model.logit_scale.exp().clamp(max=100.)
                errs=key2._arm6_candidate_mse(np.asarray(qdata['actions'])[qi],np.asarray(fit_bank['value_actions10'])[cids])
                et=torch.as_tensor(errs,device=device,dtype=torch.float32)
                am=torch.as_tensor(allowed,device=device,dtype=torch.bool)
                target=torch.softmax((-et/float(teacher_tau)).masked_fill(~am,-1e9),dim=-1)
                logq=torch.log_softmax(scores.masked_fill(~am,-1e9),dim=-1)
                loss=-(target*logq).sum(-1).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                with torch.no_grad():model.logit_scale.clamp_(max=float(np.log(100.)))
                losses.append(float(loss.detach().cpu()))
            devout=evaluate(model,dev_split);metric=devout['metrics']['top1_arm_mse']
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'dev_top1_arm_mse':metric})
            if best is None or metric<best['dev_top1_arm_mse']:
                best={'epoch':epoch+1,'dev_top1_arm_mse':metric,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
                torch.save(best['state'],models_dir/f'{tag}.pt')
            (output/'progress.jsonl').open('a').write(json.dumps({'model':tag,'epoch':epoch+1,'dev_top1_arm_mse':metric,'train_loss':history[-1]['train_loss']})+'\n')
        model.load_state_dict(best['state']);model.eval();splits={}
        for name,split in (('learned',learned_split),('unseen',unseen_split)):
            block=evaluate(model,split);hard=np.asarray(split['hard'],bool)
            block['hard_top1_arm_mse']=float(block['top1_errors'][hard].mean()) if hard.any() else None
            splits[name]=block['metrics']|{'hard_top1_arm_mse':block['hard_top1_arm_mse']}
            paired_errors[(tag,name)]=np.asarray(block['top1_errors'],np.float32)
            np.savez_compressed(predictions_dir/f'{tag}_{name}.npz',key=np.asarray(split['key']),
                task=np.asarray(split['task']),episode=np.asarray(split['episode']),hard=hard,
                top1_indices=np.asarray(block['top1_indices']),top1_errors=np.asarray(block['top1_errors']),
                top1_actions=np.asarray(block['top1_actions']),target_actions=np.asarray(split['actions']))
        results['models'][tag]={'history':history,'selected_epoch':best['epoch'],'dev_top1_arm_mse':best['dev_top1_arm_mse'],
                               'splits':splits,'checkpoint':str(models_dir/f'{tag}.pt')}
        if mode=='kv_l':
            results['models'][tag]['task_id_probe']={'memory_bank':_task_id_linear_probe(
                block['candidate_z'],bank['task'],bank['episode'])}
        if mode=='kv_l':
            # Small, predeclared 3-layer subset family; the held-out Dev top-1
            # arm MSE selects one subset. No Eval values enter this choice.
            subset_grid=((0,8,17),(5,11,17),(12,15,17))
            subset_scores=[]
            for subset in subset_grid:
                block=evaluate(model,dev_split,selected_layers=subset)
                subset_scores.append({'layers':list(subset),'dev_top1_arm_mse':block['metrics']['top1_arm_mse']})
            chosen=min(subset_scores,key=lambda x:x['dev_top1_arm_mse'])['layers']
            model.load_state_dict(best['state']);model.eval()
            subset_results={}
            for name,split in (('learned',learned_split),('unseen',unseen_split)):
                block=evaluate(model,split,selected_layers=tuple(chosen));hard=np.asarray(split['hard'],bool)
                subset_results[name]=block['metrics']|{'hard_top1_arm_mse':float(block['top1_errors'][hard].mean())}
                paired_errors[(f'kv_l_sel_{regime.replace("+","_")}',name)]=np.asarray(block['top1_errors'],np.float32)
                np.savez_compressed(predictions_dir/f'kv_l_sel_{regime.replace("+","_")}_{name}.npz',
                    key=np.asarray(split['key']),task=np.asarray(split['task']),episode=np.asarray(split['episode']),hard=hard,
                    top1_indices=np.asarray(block['top1_indices']),top1_errors=np.asarray(block['top1_errors']),
                    top1_actions=np.asarray(block['top1_actions']),target_actions=np.asarray(split['actions']))
            results['models'][f'kv_l_sel_{regime.replace("+","_")}']={
                'selected_layers':chosen,'layer_grid':subset_scores,
                'dev_top1_arm_mse':min(subset_scores,key=lambda x:x['dev_top1_arm_mse'])['dev_top1_arm_mse'],
                'splits':subset_results,'checkpoint':str(models_dir/f'{tag}.pt'),
                'note':'post-hoc selected 3-layer subset of the full-layer KV-L checkpoint'}
        (output/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
        del model,opt
        if device.type=='cuda':torch.cuda.empty_cache()
    history_mlp=_train_history_vector_baselines(output/'history_mlp',bank,bank_input,fit_bank,fit_bank_input,
        train40,learned,unseen,dev,l90,device=device,epochs=epochs,learning_rate=learning_rate,
        teacher_tau=teacher_tau,batch_size=batch_size,random_candidates=random_candidates)
    for regime,entry in history_mlp['models'].items():
        tag=f'history_state_mlp_{regime.replace("+","_")}'
        results['models'][tag]=entry
        for split_name in ('learned','unseen'):
            paired_errors[(tag,split_name)]=history_mlp['paired_errors'][(regime,split_name)]
    pairwise={}
    for split_name,split in (('learned',learned_split),('unseen',unseen_split)):
        for left,right,label in (
            (f'kv_l_libero40',f'kv_lh_libero40','KV-L − KV-LH'),
            (f'kv_h_libero40',f'kv_l_libero40','KV-H − KV-L'),
            (f'kv_l_libero40_l90x',f'kv_l_libero40','L90X − LIBERO-40 / KV-L'),
            (f'kv_h_libero40_l90x',f'kv_h_libero40','L90X − LIBERO-40 / KV-H'),
            (f'kv_lh_libero40_l90x',f'kv_lh_libero40','L90X − LIBERO-40 / KV-LH')):
            if (left,split_name) in paired_errors and (right,split_name) in paired_errors:
                pairwise[f'{split_name}/{label}']=_paired_delta_ci(paired_errors[(left,split_name)],paired_errors[(right,split_name)],split)
        for regime in ('libero40','libero40_l90x'):
            kv_tag='kv_h_'+regime
            mlp_tag='history_state_mlp_'+regime
            if (kv_tag,split_name) in paired_errors and (mlp_tag,split_name) in paired_errors:
                pairwise[f'{split_name}/KV-H − retrained history/state MLP ({regime})']=_paired_delta_ci(
                    paired_errors[(kv_tag,split_name)],paired_errors[(mlp_tag,split_name)],split)
    reference_dir=ROOT/'key-shared-01/attempt-27-production/final'
    h_mlp=_evaluate_keygrid_history(bank,bank_input,unseen_split,device,predictions_dir/'keygrid_history_state_unseen.npz')
    refrows={'unseen':{'KEYGRID H-MLP':h_mlp['metrics']}}
    pairwise['unseen/KV-H LIBERO-40 − KEYGRID H-MLP']=_paired_delta_ci(
        paired_errors[('kv_h_libero40','unseen')],h_mlp['errors'],unseen_split)
    refs={'E-VLM+H':reference_dir/'e_vlm_h/e_vlm_h_unseen_paired_predictions.npz',
          'E-H':reference_dir/'e_h/e_h_unseen_paired_predictions.npz',
          'P5':reference_dir/'p5/p5_unseen_paired_predictions.npz',
          'P0':reference_dir/'p0/p0_unseen_paired_predictions.npz',
          'E-AX suffix-V proxy':Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-sources-01/production/final/E_AX/E-AX_unseen_paired_predictions.npz')}
    for label,path in refs.items():
        if path.exists():
            referr=_reference_error(path,unseen_split)
            pairwise[f'unseen/KV-L LIBERO-40 − {label}']=_paired_delta_ci(paired_errors[('kv_l_libero40','unseen')],referr,unseen_split)
            if label=='E-AX suffix-V proxy':
                z=np.load(path,allow_pickle=False);lookup={tuple(map(int,k)):i for i,k in enumerate(z['key'])}
                ix=np.asarray([lookup[tuple(map(int,k))] for k in unseen_split['key']],np.int64)
                active=np.square(z['top1_actions'][ix]-unseen_split['actions']).sum((1,2))/(10*7.)
                hard=np.asarray(unseen_split['hard'],bool)
                refrows['unseen'][label]={'top1_arm_mse':float(referr.mean()),
                    'hard_top1_arm_mse':float(referr[hard].mean()),
                    'top1_active7_mse':float(active.mean()),'top1_active7_mse_hard':float(active[hard].mean()),
                    'note':'LT1 E-AX uses final suffix-V tap proxy, not KV1 requested [10,1024] final expert hidden.'}
    refjson=json.loads((ROOT/'key-shared-01/attempt-27-production/results.json').read_text())
    for split_name in ('learned','unseen'):
        refrows.setdefault(split_name,{})
        for key,label in (('e_vlm_h','E-VLM+H'),('e_h','E-H'),('p5','P5'),('p0','P0')):
            item=refjson['methods'][split_name][key]
            refrows[split_name][label]={'top1_arm_mse':item['top1_actions_all']['arm6'],
                'hard_top1_arm_mse':item['top1_actions_hard']['arm6'],
                'top1_active7_mse':item['top1_actions_all']['active7'],
                'top1_active7_mse_hard':item['top1_actions_hard']['active7'],
                'source':'KEY2-F fixed top1 action prediction; cited, not retrained in KV1'}
    pi05_nn={}
    for split_name,split in (('learned',learned_split),('unseen',unseen_split)):
        pi05_nn[split_name]=_pi05_nn_baseline(bank,split,split_name,predictions_dir)
        m=pi05_nn[split_name]['metrics'];refrows[split_name]['π0.5 action-NN'] = m
        pairwise[f'{split_name}/KV-L LIBERO-40 − π0.5 action-NN']=_paired_delta_ci(
            paired_errors[('kv_l_libero40',split_name)],pi05_nn[split_name]['errors'],
            learned_split if split_name=='learned' else unseen_split)
    # Stable KEY2-F comparison baselines on the identical fixed query rows.
    fixed=refjson['methods']['unseen']['fixed_baselines']
    refs_fixed={
      'random':fixed['random_rank']['top1_arm6_mse'],
      'oracle':fixed['oracle_rank']['top1_arm6_mse'],
      'hold_last':fixed['baselines']['hold_last']['arm6'],
      'π0.5':fixed['baselines']['pi05']['arm6'],
    }
    for name,value in refs_fixed.items():
        refrows['unseen'][name]= {'top1_arm_mse':float(value),'source':'KEY2-F same fixed Eval rows'}
    la2_path=ROOT/'latent-action-01/la2-production/results.json'
    if la2_path.exists():
        la2=json.loads(la2_path.read_text());selected=la2['selected_by_regime']['l90x'];item=la2['methods'][selected]['retrieval']['unseen']
        refrows['unseen']['LA2 L90X latent']={'top1_arm_mse':item['top1_arm6']['mean'],
            'hard_top1_arm_mse':item['hard_top1_arm6']['mean'],'source':'LA2 selected L90X-only model; same fixed Eval rows'}
    results['pi05_nn_baseline']={k:v['metrics'] for k,v in pi05_nn.items()}
    results['reference_rows']=refrows
    results['paired_episode_bootstrap_5000']=pairwise
    results['status']='complete';(output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    root=output.parents[1] if len(output.parents)>1 else output.parent
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    write_kv1_report(results,root)
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','train','all','smoke'])
    ap.add_argument('--output',type=Path,default=ROOT/'key-kv-01');ap.add_argument('--checkpoint',type=Path,
      default=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--memory-fraction',type=float,default=.22);ap.add_argument('--max-rows',type=int)
    ap.add_argument('--run-name',default='production-context-batch01')
    ap.add_argument('--selected-keys-json',type=Path)
    ap.add_argument('--train-batch-size',type=int,default=8)
    ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--random-candidates',type=int,default=128)
    ap.add_argument('--learning-rate',type=float,default=3e-4);ap.add_argument('--teacher-tau',type=float,default=.005)
    args=ap.parse_args();os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION',str(args.memory_fraction))
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    selected_keys=np.asarray(json.loads(args.selected_keys_json.read_text()),np.int32) if args.selected_keys_json else None
    if args.stage in ('extract','all','smoke'):
        target=args.output/'smoke' if args.stage=='smoke' else args.output/args.run_name
        print(json.dumps(extract_kv_cache(target,checkpoint=args.checkpoint,batch_size=args.batch_size,
          max_rows=4 if args.stage=='smoke' else args.max_rows,device_fraction=args.memory_fraction,
          selected_keys=selected_keys),indent=2),flush=True)
    if args.stage in ('train','all'):
        cache_root=args.output/args.run_name
        target=cache_root/'train'
        print(json.dumps(train_and_evaluate(target,cache_dir=cache_root,epochs=args.epochs,
             batch_size=args.train_batch_size,random_candidates=args.random_candidates,learning_rate=args.learning_rate,
             teacher_tau=args.teacher_tau,memory_fraction=args.memory_fraction,device=args.device),indent=2),flush=True)

if __name__=='__main__':main()
