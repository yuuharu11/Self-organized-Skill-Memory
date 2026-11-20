#!/usr/bin/env python3
"""Build exact-time, 34-token VQ-B-compatible VLM features for SEG-V units.

Existing exact prefix-token caches are pooled with VQ-B's 4x4 image grid and
text mean/last rule. Only missing timestamps are sent through vq_sources' frozen
Pi0.5 extractor. No previous observation is ever substituted.
"""
from __future__ import annotations
import os
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import key_shared_latent as k2
import vq_sources as vs

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
RUN=STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
VQB=ROOT/'vq-b-01/feats'

def _pool_prefix(tok,groups):
    """Equivalent to vq_sources._vq_features pooling, from normalized final tokens."""
    tok=np.asarray(tok,np.float32);groups=np.asarray(groups)
    out=np.zeros((len(tok),34,2048),np.float32)
    for i in range(len(tok)):
        for cam,gid in enumerate((0,1)):
            pos=np.flatnonzero(groups[i]==gid)
            if len(pos)!=256:raise ValueError(f'camera group {gid}: expected 256 tokens, got {len(pos)}')
            x=tok[i,pos].reshape(16,16,2048).reshape(4,4,4,4,2048).mean((1,3))
            out[i,cam*16:(cam+1)*16]=x.reshape(16,2048)
        text=tok[i,groups[i]==2]
        if not len(text):raise ValueError('no valid prompt tokens')
        out[i,32]=text.mean(0)
        # VQ-B computes last = sum(tokenized_prompt > 2) - 1, then indexes
        # the uncompressed prompt positions (whose first valid lexical token is
        # offset by BOS). In the packed group-2 cache this is the penultimate
        # lexical token for ordinary multi-token prompts.
        out[i,33]=text[-2] if len(text)>1 else text[-1]
    return out.astype(np.float16)

def _vq_cache_map():
    maps={};arrays={}
    # VQ-B already has exact, pooled 34-token features for these rows.
    for d in sorted(VQB.iterdir()):
        if not d.is_dir() or not (d/'key.npy').exists() or not (d/'vlm_img.npy').exists():continue
        keys=np.load(d/'key.npy',mmap_mode='r',allow_pickle=False)
        arrays[d.name]={k:np.load(d/f'{k}.npy',mmap_mode='r',allow_pickle=False)
                        for k in ('vlm_img','vlm_txt_mean','vlm_txt_last')}
        for j,key in enumerate(keys):maps.setdefault(tuple(map(int,key)),(d.name,j))
    return maps,arrays

def _exact_prefix_rows(keys):
    # Same exact-key store used by KEY2; start observations with no exact row are
    # returned as misses and must be freshly extracted below.
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (RUN/'rows').glob('*.npy')}
    caches=[]
    ckeys=np.column_stack([rows['task'],rows['episode'],rows['frame']])
    caches.append(("run",{tuple(map(int,k)):i for i,k in enumerate(ckeys)},
                   np.load(RUN/'tokens.npy',mmap_mode='r',allow_pickle=False),
                   np.load(RUN/'groups.npy',mmap_mode='r',allow_pickle=False)))
    for name,path in [('missing',ROOT/'key-shared-01/attempt-02/prefix-missing'),
                      ('seen5',ROOT/'seg-context-01/seg5-cache')]:
        man=json.loads((path/'manifest.json').read_text());ks=man['keys']
        caches.append((name,{tuple(map(int,k)):i for i,k in enumerate(ks)},
                       np.load(path/'tokens.npy',mmap_mode='r',allow_pickle=False),
                       np.load(path/'groups.npy',mmap_mode='r',allow_pickle=False)))
    exact=np.zeros((len(keys),34,2048),np.float16)
    found=np.zeros(len(keys),bool);sources=np.full(len(keys),'missing',dtype='U8')
    for i,k0 in enumerate(keys):
        key=tuple(map(int,k0));hits=[]
        for name,mp,tok,grp in caches:
            if key in mp:
                j=mp[key];valid=np.asarray(grp[j])>=0;hits.append((name,np.asarray(tok[j])[valid],np.asarray(grp[j])[valid]))
        if not hits:continue
        name,t,g=hits[0]
        for _,ot,og in hits[1:]:
            if not np.array_equal(t,ot) or not np.array_equal(g,og):raise AssertionError(f'exact prefix caches disagree at {key}')
        exact[i]=_pool_prefix(t[None],g[None])[0];found[i]=True;sources[i]=name
    return exact,found,sources

def _save_json(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def finalize_existing(out:Path,reuse_res1:Path):
    """Audit an interrupted post-save run and emit non-stale reference files.

    The feature matrix is immutable: this only verifies it against the exact
    source artifacts and writes a new aligned key/source manifest.
    """
    out=Path(out);bankdir=out/'bank';keys=np.load(bankdir/'speed__unit_key.npy',allow_pickle=False)
    feats=np.load(bankdir/'speed__vlm34.npy',mmap_mode='r',allow_pickle=False)
    if feats.shape!=(len(keys),34,2048):raise AssertionError(f'bad VLM shape {feats.shape}')
    rawpath=Path(reuse_res1)
    with np.load(rawpath,allow_pickle=False) as z:
        rk=np.asarray(z['source'])[:,:3];rv=z['vlm_raw']
    rmap={tuple(map(int,k)):i for i,k in enumerate(rk)}
    qbmap,qbcache=_vq_cache_map()
    pref,have,psrc=_exact_prefix_rows(keys)
    sources=np.empty(len(keys),dtype='U20');direct_cos=[];prefix_cos=[];cached_exact=0
    fresh=[]
    for i,k in enumerate(keys):
        key=tuple(map(int,k))
        if key in rmap:
            sources[i]='res1-direct'
            ref=rv[rmap[key]].astype(np.float32);got=np.asarray(feats[i],np.float32)
            c=(got*ref).sum(-1)/(np.maximum(np.linalg.norm(got,axis=-1)*np.linalg.norm(ref,axis=-1),1e-9))
            direct_cos.extend(c.tolist())
        elif key in qbmap:
            name,j=qbmap[key];a=qbcache[name]
            ref=np.concatenate([np.asarray(a['vlm_img'][j]),np.asarray(a['vlm_txt_mean'][j])[None],np.asarray(a['vlm_txt_last'][j])[None]],0).astype(np.float32)
            if not np.array_equal(np.asarray(feats[i]),ref.astype(np.float16)):raise AssertionError(f'cached VQ-B value mismatch {key}')
            sources[i]='vq-b:'+name;cached_exact+=1
        elif have[i]:
            ref=pref[i].astype(np.float32);got=np.asarray(feats[i],np.float32)
            c=(got*ref).sum(-1)/(np.maximum(np.linalg.norm(got,axis=-1)*np.linalg.norm(ref,axis=-1),1e-9))
            if float(c.min())<.999:raise AssertionError(f'prefix pooled feature mismatch {key}: {c.min()}')
            prefix_cos.extend(c.tolist());sources[i]='prefix:'+str(psrc[i])
        else:
            fresh.append(i);sources[i]='fresh-vq-b'
    if len(fresh)!=256:raise AssertionError(f'expected 256 freshly extracted exact starts, got {len(fresh)}')
    overlap={'direct_res1_rows':len(direct_cos)//34,'direct_token_cos_min':float(min(direct_cos)),
             'direct_token_cos_mean':float(np.mean(direct_cos)),
             'prefix_rows':len(prefix_cos)//34,'prefix_token_cos_min':float(min(prefix_cos)) if prefix_cos else None,
             'cached_vq_b_rows':cached_exact}
    if overlap['direct_token_cos_min']<.999:raise AssertionError(f'direct exact-time VLM parity failed: {overlap}')
    for name,value in [('speed__vlm34_key.npy',keys),('speed__vlm34_source.npy',sources)]:
        path=bankdir/name
        if path.exists():raise FileExistsError(path)
        np.save(path,value,allow_pickle=False)
    result={'rows':len(keys),'shape':list(feats.shape),'dtype':str(feats.dtype),'unit_start_key_exact':True,
            'source_counts':{str(x):int((sources==x).sum()) for x in np.unique(sources)},
            'fresh_exact_rows':len(fresh),'fresh_exact_keys':keys[fresh].tolist(),'parity':overlap,
            'feature_path':'vq_sources.Pi05Features; exact bank start frame; 4x4 pooled base/wrist + VQ-B text mean/last; no prior-frame substitution',
            'reused_res1_archive':str(rawpath)}
    _save_json(out/'vlm_rebuild.json',result)
    with (out/'progress.log').open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':'vlm_references_regenerated',**{k:v for k,v in result.items() if k not in ('fresh_exact_keys',)}},ensure_ascii=False)+'\n')
    return result

def prepare(out:Path, *, batch_size:int=8, controls:int=24, smoke:bool=False, reuse_res1:Path|None=None):
    import torch
    torch.set_num_threads(8)
    bankdir=out/'bank';key_path=bankdir/'speed__unit_key.npy'
    if not key_path.is_file():raise FileNotFoundError(key_path)
    keys=np.load(key_path,allow_pickle=False)
    cachemap,cache=_vq_cache_map()
    pref,have_prefix,prefix_source=_exact_prefix_rows(keys)
    known=np.array([tuple(map(int,k)) in cachemap for k in keys],bool)
    res1_map={};res1_raw=None
    if reuse_res1 is not None:
        rp=Path(reuse_res1)
        if rp.suffix=='.npz':
            with np.load(rp,allow_pickle=False) as z:
                res1_keys=np.asarray(z['source'])[:,:3];res1_raw=z['vlm_raw']
        else:raise ValueError('--reuse-res1 must point to the completed bank_vlm_raw.npz archive')
        res1_map={tuple(map(int,k)):i for i,k in enumerate(res1_keys)}
    if smoke:
        n=min(12,len(keys));keys=keys[:n];pref=pref[:n]
        have_prefix=have_prefix[:n];prefix_source=prefix_source[:n];known=known[:n]
    # Build exact features from prefix caches, or reuse VQ-B rows when exact.
    features=np.empty((len(keys),34,2048),np.float16);source=np.empty(len(keys),dtype='U16')
    todo=[]
    for i,k in enumerate(keys):
        ck=tuple(map(int,k))
        if ck in res1_map:
            features[i]=res1_raw[res1_map[ck]];source[i]='res1-direct'
        elif ck in cachemap:
            name,j=cachemap[ck];a=cache[name]
            features[i,:32]=a['vlm_img'][j];features[i,32]=a['vlm_txt_mean'][j];features[i,33]=a['vlm_txt_last'][j]
            source[i]='vq-b:'+name
        elif have_prefix[i]:
            features[i]=pref[i];source[i]='prefix:'+str(prefix_source[i])
        else:todo.append(i)
    # VQ-B control rows prove prefix pooling reproduces the original feature pass.
    if res1_map:
        overlap=np.asarray([i for i,k in enumerate(keys) if have_prefix[i] and tuple(map(int,k)) in res1_map],np.int64)
    else:
        overlap=np.flatnonzero(known & have_prefix)
    if not len(overlap):raise AssertionError('no overlap rows available for extraction parity')
    cids=overlap[np.linspace(0,len(overlap)-1,min(controls,len(overlap))).astype(int)]
    pooled=pref[cids].astype(np.float32)
    if res1_map:
        direct34=np.stack([res1_raw[res1_map[tuple(map(int,keys[i]))]] for i in cids]).astype(np.float32)
    else:
        fx=vs.Pi05Features();reader=vs.ObsReader()
        raw=[reader.read(k,'libero40') for k in keys[cids]]
        direct=fx(raw)
        direct34=np.concatenate([direct['vlm_img'],direct['vlm_txt_mean'][:,None],direct['vlm_txt_last'][:,None]],axis=1).astype(np.float32)
    def per_token_cos(x,y):
        x=x.reshape(len(x),-1,x.shape[-1]);y=y.reshape(x.shape)
        return ((x*y).sum(-1)/np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1),1e-9))
    parity={'rows':int(len(cids)),'all_token_cos_min':float(per_token_cos(pooled,direct34).min()),
            'all_token_cos_mean':float(per_token_cos(pooled,direct34).mean()),
            'img_cos_min':float(per_token_cos(pooled[:,:32],direct34[:,:32]).min()),
            'text_cos_min':float(per_token_cos(pooled[:,32:],direct34[:,32:]).min()),
            'max_abs':float(np.abs(pooled-direct34).max())}
    if parity['all_token_cos_min']<.999:raise AssertionError(f'VQ-B pooling parity failed: {parity}')
    # Extract only missing exact unit starts. Inputs are the source parquet image
    # and wrist image at the exact (task, episode, frame), with normal orientation.
    if todo:
        fx=vs.Pi05Features();reader=vs.ObsReader()
        for st in range(0,len(todo),batch_size):
            ids=todo[st:st+batch_size]
            raw=[reader.read(keys[i],'libero40') for i in ids]
            r=fx(raw);features[ids,:32]=r['vlm_img'];features[ids,32]=r['vlm_txt_mean'];features[ids,33]=r['vlm_txt_last']
            source[ids]='fresh-vq-b'
            with (out/'progress.log').open('a') as f:
                f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':'vlm_extract','done':min(st+batch_size,len(todo)),'total_missing':len(todo)},ensure_ascii=False)+'\n')
    vlmout=bankdir/'speed__vlm34.npy'
    if vlmout.exists():raise FileExistsError(vlmout)
    np.save(vlmout,features,allow_pickle=False)
    if (bankdir/'speed__vlm_key.npy').exists():raise FileExistsError(bankdir/'speed__vlm_key.npy')
    np.save(bankdir/'speed__vlm_key.npy',keys,allow_pickle=False)
    if (bankdir/'speed__vlm_source.npy').exists():raise FileExistsError(bankdir/'speed__vlm_source.npy')
    np.save(bankdir/'speed__vlm_source.npy',source,allow_pickle=False)
    result={'units':int(len(keys)),'prefix_exact':int(have_prefix.sum()),'vq_b_exact':int(known.sum()),
            'res1_direct_exact':int(sum(tuple(map(int,k)) in res1_map for k in keys)),
            'fresh_exact_extracted':int(len(todo)),'fresh_keys':[keys[i].tolist() for i in todo],
            'source_counts':{str(x):int((source==x).sum()) for x in np.unique(source)},'parity':parity,
            'shape':list(features.shape),'dtype':str(features.dtype),'exact_key_alignment':True}
    _save_json(out/'vlm_rebuild.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--controls',type=int,default=24);ap.add_argument('--smoke',action='store_true');ap.add_argument('--reuse-res1',type=Path);ap.add_argument('--finalize-only',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.finalize_only:
        if a.reuse_res1 is None:ap.error('--finalize-only requires --reuse-res1')
        print(json.dumps(finalize_existing(a.out,a.reuse_res1),indent=2,ensure_ascii=False));return
    print(json.dumps(prepare(a.out,batch_size=a.batch_size,controls=a.controls,smoke=a.smoke,reuse_res1=a.reuse_res1),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
