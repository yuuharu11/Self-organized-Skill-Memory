#!/usr/bin/env python3
"""Resumable prefix-token extractor for SEG1's missing stride-5 timestamps.

The existing run-01 cache stores stride-10 observations. This utility accepts a
source observation callback (normally backed by the original prepared LIBERO
episode), so it never pretends cached stride-10 frames contain the missing data.
It is intentionally not run by the implementation-only SEG1 task.
"""
from __future__ import annotations

import json
import argparse
import importlib.util
import io
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def _load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def required_query_keys(task, episode, frame, *, stride=5, segment_starts=()):
    """All observed query ticks plus memory segment starts, deduplicated."""
    keys=set();task=np.asarray(task);episode=np.asarray(episode);frame=np.asarray(frame)
    for t,e in sorted(set(zip(task.astype(int).tolist(),episode.astype(int).tolist()))):
        fs=frame[(task==t)&(episode==e)]
        if len(fs):
            lo,hi=int(fs.min()),int(fs.max())
            first=((lo+int(stride)-1)//int(stride))*int(stride)
            keys.update((int(t),int(e),int(f)) for f in range(first,hi+1,int(stride)))
    keys.update((int(t),int(e),int(f)) for t,e,f in segment_starts)
    return sorted(keys)


def cached_query_keys(rows):
    """Map cache rows to stable (task, episode, frame) identities."""
    task=np.asarray(rows['task']);episode=np.asarray(rows['episode']);frame=np.asarray(rows['frame'])
    return {(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,episode,frame))}


def missing_query_keys(required, cached):
    return [key for key in required if key not in cached]


def extract_prefix_memmap(required, observe_fn: Callable, encode_fn: Callable, output_dir, *,
                          token_shape=(532,2048), batch_size=1,valid_only=True):
    """Extract missing (task,episode,frame) prefixes, resuming completed records.

    ``observe_fn(key)`` returns a preprocessed observation at that exact frame;
    ``encode_fn(batch)`` returns token and group arrays. Caller loads the frozen
    π0.5 model and controls accelerator memory. This helper never initializes GPU.
    """
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()) and not (out/'manifest.json').exists() and not (out/'progress.jsonl').exists():
        raise FileExistsError(f"refusing to reuse unmanifested output directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json';progress=out/'progress.jsonl'
    completed=set()
    if progress.exists():
        for line in progress.read_text().splitlines():
            if line.strip():
                rec=json.loads(line)
                if rec.get('status')=='done':completed.add(tuple(rec['key']))
    all_keys=list(required); todo=[k for k in all_keys if tuple(k) not in completed]
    token_path=out/'tokens.npy';group_path=out/'groups.npy'
    mode='r+' if token_path.exists() and group_path.exists() else 'w+'
    tok=np.lib.format.open_memmap(token_path,mode=mode,dtype=np.float16,shape=(len(all_keys),*token_shape))
    grp=np.lib.format.open_memmap(group_path,mode=mode,dtype=np.int8,shape=(len(all_keys),token_shape[0]))
    # Optional aligned metadata lets downstream training use exactly the frame
    # whose prefix was encoded. It is created only when observe_fn returns it.
    meta_maps={}
    slot={tuple(k):i for i,k in enumerate(all_keys)}
    with progress.open('a',encoding='utf8') as log:
        for start in range(0,len(todo),int(batch_size)):
            batch_keys=todo[start:start+int(batch_size)]
            observations=[observe_fn(k) for k in batch_keys]
            if observations and not meta_maps:
                for name in ('state','actions'):
                    if name in observations[0]:
                        value=np.asarray(observations[0][name])
                        path=out/f'{name}.npy'
                        mm=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=value.dtype,
                            shape=(len(all_keys),*value.shape))
                        meta_maps[name]=mm
            for name,mm in meta_maps.items():
                for j,item in enumerate(observations):mm[slot[tuple(batch_keys[j])]]=item[name]
            t,g=encode_fn(observations)
            t=np.asarray(t);g=np.asarray(g)
            if valid_only and t.shape[1]!=token_shape[0]:
                packed=np.zeros((len(t),*token_shape),np.float16)
                packed_groups=np.full((len(t),token_shape[0]),-1,np.int8)
                for j in range(len(t)):
                    valid=np.flatnonzero(g[j]>=0)
                    if len(valid)>token_shape[0]:raise ValueError('valid prefix token count exceeds packed capacity')
                    packed[j,:len(valid)]=t[j,valid].astype(np.float16)
                    packed_groups[j,:len(valid)]=g[j,valid]
                t,g=packed,packed_groups
            if t.shape!=(len(batch_keys),*token_shape) or g.shape!=(len(batch_keys),token_shape[0]):
                raise ValueError(f"unexpected encoder output shapes {t.shape}, {g.shape}")
            for j,key in enumerate(batch_keys):
                i=slot[tuple(key)];tok[i]=t[j].astype(np.float16);grp[i]=g[j].astype(np.int8)
                log.write(json.dumps({'key':list(key),'slot':i,'status':'done'})+'\n');log.flush()
    tok.flush();grp.flush()
    for mm in meta_maps.values():mm.flush()
    payload={'rows':len(all_keys),'token_shape':list(token_shape),'keys':all_keys,
             'source':'exact requested frames; prefix cache only','stride':5}
    tmp=manifest.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(manifest)
    return payload


def cache_audit(run_root):
    """Read-only cache inventory and stride summary, safe for the CPU smoke path."""
    root=Path(run_root)
    for rel in ('tokens.npy','groups.npy','rows/task.npy','rows/episode.npy','rows/frame.npy'):
        if not (root/rel).is_file(): raise FileNotFoundError(root/rel)
    tok=np.load(root/'tokens.npy',mmap_mode='r',allow_pickle=False)
    grp=np.load(root/'groups.npy',mmap_mode='r',allow_pickle=False)
    task=np.load(root/'rows/task.npy',mmap_mode='r',allow_pickle=False)
    ep=np.load(root/'rows/episode.npy',mmap_mode='r',allow_pickle=False)
    frame=np.load(root/'rows/frame.npy',mmap_mode='r',allow_pickle=False)
    strides=[]
    order=np.lexsort((frame,ep,task))
    t,e,f=task[order],ep[order],frame[order]
    same=(t[1:]==t[:-1])&(e[1:]==e[:-1])
    if same.any():strides=np.unique(np.diff(f)[same]).astype(int).tolist()
    groups,counts=np.unique(grp[0],return_counts=True)
    return {'tokens_shape':list(tok.shape),'tokens_dtype':str(tok.dtype),'groups_shape':list(grp.shape),
            'cached_frame_deltas':strides,'row_count':len(frame),
            'group_counts_first_row':{str(int(k)):int(v) for k,v in zip(groups,counts)},
            'stride5_missing_estimate':int(len(frame))}


def extract_run_stride5(run_root,output_dir,*,checkpoint=None,batch_size=4,limit_episodes=None,segment_starts=()):
    """Prepare exact-frame observations and extract missing stride-5 prefix tokens.

    Unlike the cache audit, this invokes the frozen JAX model and is an explicit
    experiment command. It reuses run.py's transforms, checkpoint loader, prefix
    feature function and Observation builder. It is deliberately not invoked by
    the SEG1 smoke tests.
    """
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.20')
    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    from openpi.shared.nnx_utils import module_jit

    root=Path(run_root);manifest=json.loads((root/'data_manifest.json').read_text())
    source=Path(manifest['source'])
    ckpt=Path(checkpoint or manifest['checkpoint'])
    run=_load_module(Path(__file__).resolve().parents[1]/'pi05_memory'/'run.py','seg1_pi05_run')
    rows=run.load_rows(root)
    tokens=np.load(root/'tokens.npy',mmap_mode='r',allow_pickle=False)
    groups=np.load(root/'groups.npy',mmap_mode='r',allow_pickle=False)
    cached=cached_query_keys(rows)
    records=manifest['episodes']
    if limit_episodes is not None:records=records[:int(limit_episodes)]
    starts=[]
    for rec in records:
        starts.extend((int(rec['task']),int(rec['episode']),int(f))
                      for f in range(0,max(0,int(rec['length'])-9),5))
    starts.extend(tuple(map(int,key)) for key in segment_starts)
    needed=sorted(set(starts)-set(cached))
    norm_files=list((ckpt/'assets').rglob('norm_stats.json'))
    if len(norm_files)!=1:raise ValueError(f'expected one normalization file, got {len(norm_files)}')
    tasks={int(k):v for k,v in manifest['tasks'].items()}
    transforms=[LiberoInputs(ModelType.PI05),Normalize(load(norm_files[0].parent),use_quantiles=True),
        ResizeImages(224,224),TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    model,_=run.load_policy(ckpt);encode=module_jit(model.prefix_features)
    record_map={(int(r['task']),int(r['episode'])):r for r in records}
    parquet_cache={}
    def observe(key):
        t,e,f=map(int,key);rec=record_map[(t,e)]
        if (t,e) not in parquet_cache:
            if len(parquet_cache)>=2:parquet_cache.pop(next(iter(parquet_cache)))
            parquet_cache[(t,e)]=pq.read_table(rec['path'],columns=['image','wrist_image','state','actions'])
        table=parquet_cache[(t,e)]
        images={}
        for col in ('image','wrist_image'):
            obj=table[col][f].as_py()
            with Image.open(io.BytesIO(obj['bytes'])) as im:images[col]=np.asarray(im.convert('RGB'))
        state=np.asarray(table['state'][f].as_py(),np.float32)
        act=np.asarray(table['actions'].slice(f,10).to_pylist(),np.float32)
        item={'observation/image':images['image'],'observation/wrist_image':images['wrist_image'],
              'observation/state':state,'actions':act,'prompt':tasks[t]}
        for transform in transforms:item=transform(item)
        return {'image_base':np.asarray(item['image']['base_0_rgb']),'image_wrist':np.asarray(item['image']['left_wrist_0_rgb']),
                'state':np.asarray(item['state']),'prompt':np.asarray(item['tokenized_prompt']),
                'prompt_mask':np.asarray(item['tokenized_prompt_mask']),
                'actions':np.asarray(item['actions'])}
    def encode_batch(items):
        n=len(items)
        packed={k:np.stack([x[k] for x in items]) for k in items[0]}
        batch={'image_base':packed['image_base'],'image_wrist':packed['image_wrist'],
               'state':packed['state'],'prompt':packed['prompt'],'prompt_mask':packed['prompt_mask']}
        hidden,group=encode(run.obs_batch(batch,np.arange(n)))
        return np.asarray(hidden),np.asarray(group)
    out=extract_prefix_memmap(needed,observe,encode_batch,output_dir,batch_size=batch_size)
    out.update({'cached_rows_reused':len(cached),'cached_token_shape':list(tokens.shape),
                'cached_group_shape':list(groups.shape),'missing_rows':len(needed),
                'checkpoint':str(ckpt),'dataset':str(source),'tokens_are_final_prefix_tokens':True})
    mp=Path(output_dir)/'manifest.json';tmp=mp.with_suffix('.json.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(mp)
    return out


def extract_unseen_hdf5_stride5(unseen_rows_root,output_dir,*,checkpoint=None,batch_size=4):
    """Exact stride-5 Dev/Eval prefix features from the fixed LIBERO-90 HDF5 rows.

    The original unseen-rows-02 cache is read-only and remains stride-10. This
    creates a distinct, resumable cache with overlapping 10-action targets at
    query starts every five 10Hz controls.
    """
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.20')
    import h5py, jax
    import jax.numpy as jnp
    from PIL import Image
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    from openpi.shared.nnx_utils import module_jit
    source=Path(unseen_rows_root);src_manifest=json.loads((source/'manifest.json').read_text())
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=src_manifest['source_records'];record_map={(int(r['task_id']),int(r['episode'])):r for r in records}
    keys=[];split_map={};name_map={}
    for (t,e),rec in sorted(record_map.items()):
        h5path=Path(rec['source_hdf5'])
        with h5py.File(h5path,'r') as f:n=len(f['data'][rec['source_demo']]['actions'][::2])
        for frame in range(0,max(0,n-9),5):
            key=(int(t),int(e),int(frame));keys.append(key);split_map[key]=rec['split'];name_map[key]=rec['task_name']
    ckpt=Path(checkpoint or src_manifest['actions_normalized_with']).parents[2]
    # Prefer the checkpoint recorded in the existing features manifest.
    feature_manifest=json.loads((source/'features.json').read_text()) if (source/'features.json').exists() else {}
    ckpt=Path(checkpoint or feature_manifest.get('checkpoint','/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    norm_files=list((ckpt/'assets').rglob('norm_stats.json'))
    if len(norm_files)!=1:raise ValueError(f'expected one normalization file, got {len(norm_files)}')
    transforms=[LiberoInputs(ModelType.PI05),Normalize(load(norm_files[0].parent),use_quantiles=True),
        ResizeImages(224,224),TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    run=_load_module(Path(__file__).resolve().parents[1]/'pi05_memory'/'run.py','seg2_unseen_pi05_run')
    model,_=run.load_policy(ckpt);encode=module_jit(model.prefix_features)
    h5_cache={}
    def observe(key):
        t,e,f=map(int,key);rec=record_map[(t,e)];demo=rec['source_demo']
        p=Path(rec['source_hdf5'])
        if p not in h5_cache:
            if len(h5_cache)>=1:
                _,old=h5_cache.popitem();old.close()
            h5_cache[p]=h5py.File(p,'r')
        d=h5_cache[p]['data'][demo];obs=d['obs'];src_i=2*f
        ee=np.asarray(obs['ee_pos'][src_i],np.float32);ori=np.asarray(obs['ee_ori'][src_i],np.float32)
        grip=np.asarray(obs['gripper_states'][src_i],np.float32);state=np.concatenate([ee,ori,grip])
        base=np.asarray(obs['agentview_rgb'][src_i],np.uint8)
        wrist=np.asarray(obs['eye_in_hand_rgb'][src_i],np.uint8)
        # HDF5 LIBERO frames use the same 180-degree orientation correction as
        # prepare_oracle and its rollout worker.
        base=base[::-1,::-1].copy();wrist=wrist[::-1,::-1].copy()
        acts=np.asarray(d['actions'][::2][f:f+10],np.float32)
        item={'observation/image':base,'observation/wrist_image':wrist,'observation/state':state,
              'actions':acts,'prompt':rec['instruction']}
        for tr in transforms:item=tr(item)
        return {'image_base':np.asarray(item['image']['base_0_rgb']),'image_wrist':np.asarray(item['image']['left_wrist_0_rgb']),
            'state':np.asarray(item['state']),'actions':np.asarray(item['actions']),
            'prompt':np.asarray(item['tokenized_prompt']),'prompt_mask':np.asarray(item['tokenized_prompt_mask'])}
    def encode_batch(items):
        n=len(items);packed={k:np.stack([x[k] for x in items]) for k in ('image_base','image_wrist','state','prompt','prompt_mask')}
        obs={'image_base':packed['image_base'],'image_wrist':packed['image_wrist'],'state':packed['state'],
             'prompt':packed['prompt'],'prompt_mask':packed['prompt_mask']}
        h,g=encode(run.obs_batch(obs,np.arange(n)));return np.asarray(h),np.asarray(g)
    try:
        meta=extract_prefix_memmap(keys,observe,encode_batch,out,batch_size=batch_size)
    finally:
        for f in h5_cache.values():f.close()
    for name,dtype,vals in [('task',np.int32,[k[0] for k in keys]),('episode',np.int32,[k[1] for k in keys]),
        ('frame',np.int32,[k[2] for k in keys]),('is_dev',bool,[split_map[k]=='dev' for k in keys]),
        ('task_name','U150',[name_map[k] for k in keys])]:np.save(out/f'{name}.npy',np.asarray(vals,dtype=dtype))
    meta.update({'rows':len(keys),'stride':5,'source_manifest':str(source/'manifest.json'),'checkpoint':str(ckpt),
        'orientation':'HDF5 image rotated 180 degrees before PI0.5 transforms','complete':True})
    mp=out/'manifest.json';tmp=mp.with_suffix('.json.tmp');tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(mp)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--limit-episodes',type=int)
    parser.add_argument('--segment-starts-json',type=Path,
                        help='optional JSON array of [task,episode,frame] unit starts')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--unseen-hdf5-stride5',action='store_true',help='extract fixed unseen Dev/Eval HDF5 rows to a new stride-5 cache')
    args=parser.parse_args()
    if args.unseen_hdf5_stride5:
        print(json.dumps(extract_unseen_hdf5_stride5(args.run_root,args.output,checkpoint=args.checkpoint,batch_size=args.batch_size),indent=2));return
    if args.audit_only:print(json.dumps(cache_audit(args.run_root),indent=2));return
    extra=json.loads(args.segment_starts_json.read_text()) if args.segment_starts_json else ()
    print(json.dumps(extract_run_stride5(args.run_root,args.output,checkpoint=args.checkpoint,
        batch_size=args.batch_size,limit_episodes=args.limit_episodes,segment_starts=extra),indent=2))


if __name__=='__main__':main()
