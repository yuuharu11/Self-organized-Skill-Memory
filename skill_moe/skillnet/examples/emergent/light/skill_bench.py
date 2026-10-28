"""Evaluation harness for duck-typed skill formation methods.

Each ``methods/*.py`` module exports ``build_variants() -> {name: method}``.
Every instance implements the study's fit/encode/decode/prior/memory_add/
memory_query/describe interface. Optional ``base_features``, ``blend_base`` and
``prior_base`` adapters support discrete code IDs without a shared base class.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import importlib
import inspect
import json
import math
import os
import time
import uuid
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.metrics import MIConfig, action_motion_labels, env_motion_labels, evaluate_mi, kmeans, assign
from skill_moe.skillnet.examples.emergent.eval.memory import KeyEncoder
from .compare_keys import _fit_pca
from .diagnose_sharing import PREVIOUS, _decode_mse
from .run import DEFAULT_OUTPUTS
from .methods.m6 import M6

LIGHT=Path(__file__).resolve().parent
METHODS_DIR=LIGHT/"methods"
PREVIOUS_KEYS=Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-keys-20261026T080437Z-eedfc4")
CAMERA_CACHE=Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-keys-20261026T074213Z-9e39c3/camera_features")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_METHOD_FACTORIES={}


def discover_methods():
    found={}; _METHOD_FACTORIES.clear()
    for path in sorted(METHODS_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name=="__init__.py": continue
        module=importlib.import_module(f"skill_moe.skillnet.examples.emergent.light.methods.{path.stem}")
        factory=getattr(module,"build_variants",None)
        if factory is not None:
            for name,method in factory().items():
                if name in found: raise ValueError(f"duplicate method variant name: {name}")
                found[name]=method
                _METHOD_FACTORIES[name]=lambda f=factory,n=name: f()[n]
            continue
        # Small utility modules (e.g. geometry wrappers and training pressures)
        # are not standalone methods. Auto-register public classes that expose
        # the full duck-typed protocol and have no required constructor args.
        for attr,cls in vars(module).items():
            if attr.startswith("_") or not inspect.isclass(cls) or cls.__module__!=module.__name__: continue
            if not all(callable(getattr(cls,method,None)) for method in ("fit","encode","decode","prior","memory_add","memory_query","describe")): continue
            try: sig=inspect.signature(cls); required=[p for p in sig.parameters.values() if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]
            except (TypeError,ValueError): required=[object()]
            if required: continue
            variants=getattr(cls,"VARIANTS",None)
            if variants and "variant" in sig.parameters:
                for variant in variants:
                    name=f"{path.stem.split('_')[0].upper()}_{variant}"
                    if name in found: raise ValueError(f"duplicate method variant name: {name}")
                    found[name]=cls(variant=variant)
                    _METHOD_FACTORIES[name]=lambda c=cls,v=variant: c(variant=v)
                continue
            name=(path.stem.split("_")[0].upper() if path.stem.startswith("m") and path.stem[1:2].isdigit() else attr)
            if name in found: raise ValueError(f"duplicate method variant name: {name}")
            found[name]=cls()
            _METHOD_FACTORIES[name]=lambda c=cls: c()
    return found


def _new_method(name):
    return _METHOD_FACTORIES[name]()


def _rng_sample(rows,episode_ids,limit,seed):
    rows=np.asarray(rows,dtype=np.int64)
    if limit is None or len(rows)<=limit: return rows
    rng=np.random.default_rng(seed); groups={e:rows[episode_ids[rows]==e] for e in np.unique(episode_ids[rows])}
    selected=[]
    # Round-robin episodes so smoke runs retain query/store coverage.
    order=list(groups); rng.shuffle(order)
    while len(selected)<limit:
        changed=False
        for ep in order:
            g=groups[ep]
            available=[int(i) for i in g if int(i) not in selected]
            if available and len(selected)<limit:
                selected.append(available[int(rng.integers(len(available)))]); changed=True
        if not changed: break
    return np.asarray(sorted(selected),dtype=np.int64)


def _camera_features(cache,run):
    first=next((cache/"episodes").glob("*.npz"))
    with np.load(first,allow_pickle=False) as d: width=d["wrist"].shape[1]
    out=np.zeros((len(run),width),np.float32); seen=np.zeros(len(run),bool)
    for episode in run.episodes:
        with np.load(cache/"episodes"/f"{episode.episode_id}.npz",allow_pickle=False) as d:
            ix={int(s):j for j,s in enumerate(d["step"])}
            for r in np.flatnonzero(run.episode_id==episode.episode_id):
                j=ix.get(int(run.step[r]))
                if j is not None: out[r]=d["wrist"][j]; seen[r]=True
    if not seen[run.usable].all(): raise ValueError("camera cache does not align with all usable rows")
    return out


def _keys(run,train_rows,wrist,horizon):
    # Fit normalization/PCA only on the selected training suites.
    enc=KeyEncoder.fit({"vlm_img_nolang":run.keys["vlm_img_nolang"][train_rows],
                        "proprio":run.proprio[train_rows]},include_language=False,n_components=32)
    combined=enc.transform({"vlm_img_nolang":run.keys["vlm_img_nolang"],"proprio":run.proprio}).astype(np.float32)
    wrist_enc=KeyEncoder.fit({"vlm_img_nolang":wrist[train_rows]},include_language=False,n_components=32)
    wrist_key=wrist_enc.transform({"vlm_img_nolang":wrist}).astype(np.float32)
    return {"combined_base_wrist_proprio":combined,"wrist_only":wrist_key},wrist_enc,enc


def _split_suite_query_groups(run,rows,seed,query_fraction=.25,suites=("libero_object","libero_goal")):
    """Split each suite by whole namespaced initial-state groups."""
    rng=np.random.default_rng(seed); train=[]; query={}
    rows=np.asarray(rows,dtype=np.int64)
    for suite in suites:
        suite_rows=rows[run.suite[rows]==suite]
        groups=np.unique(run.init_state_id[suite_rows]); groups=np.asarray(groups,dtype=object)
        rng.shuffle(groups)
        n_query=min(max(len(groups)-1,0),max(1,int(np.ceil(len(groups)*query_fraction))) if len(groups)>1 else 0)
        qgroups=set(groups[:n_query].tolist())
        qrows=suite_rows[np.isin(run.init_state_id[suite_rows],list(qgroups))]
        trows=suite_rows[~np.isin(run.init_state_id[suite_rows],list(qgroups))]
        train.extend(trows.tolist()); query[suite]=qrows
        if set(run.init_state_id[trows]) & set(run.init_state_id[qrows]):
            raise AssertionError(f"{suite}: initial-state leakage in sequential split")
    return np.asarray(sorted(train),dtype=np.int64),query


def _find_libero10_root(explicit=None):
    if explicit is not None:
        root=Path(explicit)
        if (root/"feature-cache"/"manifest.json").is_file(): return root
        if (root/"manifest.json").is_file() and root.name=="feature-cache": return root.parent
        raise FileNotFoundError(f"LIBERO-10 feature cache missing under {root}")
    candidates=sorted((p for p in DEFAULT_OUTPUTS.glob("libero10-features-*")
                       if (p/"feature-cache"/"manifest.json").is_file()),key=lambda p:p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_libero10_rows(root,horizon,combined_encoder,wrist_encoder,*,max_rows_per_split=None,seed_value=0):
    if root is None: return None,"no LIBERO-10 feature cache found; sequential evaluation ends after spatial"
    from .libero10_features import read_cache
    language_camera_cache=root/"camera-language-features"
    data=read_cache(root/"feature-cache",horizon=horizon,camera_cache=root/"camera-features",
                    language_camera_cache=language_camera_cache if language_camera_cache.is_dir() else None)
    full=np.asarray(data["action_mask"],dtype=bool).all(1)&np.asarray(data["motion_mask"],dtype=bool).all(1)
    full &= np.isfinite(data["f"]).all(1)&np.isfinite(data["k"]).all(1)&np.isfinite(data["proprio"]).all(1)
    rows=np.flatnonzero(full)
    if not len(rows): return None,"LIBERO-10 feature cache has no full H-step rows"
    if max_rows_per_split is not None:
        sampled=[]
        for split_i,split_name in enumerate(("store","query")):
            split_rows=rows[np.asarray(data["split"])[rows]==split_name]
            sampled.extend(_rng_sample(split_rows,data["episode_id"],max_rows_per_split,seed_value+split_i*37).tolist())
        rows=np.asarray(sorted(sampled),dtype=np.int64)
    k=data["k"][rows].astype(np.float32); prop=data["proprio"][rows].astype(np.float32)
    keymap={"combined_base_wrist_proprio":combined_encoder.transform({"vlm_img_nolang":k,"proprio":prop}).astype(np.float32)}
    if data.get("wrist") is not None and np.isfinite(data["wrist"][rows]).all():
        keymap["wrist_only"]=wrist_encoder.transform({"vlm_img_nolang":data["wrist"][rows]}).astype(np.float32)
    language_availability={}
    for camera_name in ("base_lang","wrist_lang"):
        available=np.asarray(data.get(f"has_{camera_name}",np.zeros(len(data["step"]),dtype=bool)),dtype=bool)[rows].copy()
        values=data.get(camera_name)
        if values is not None:
            available &= np.isfinite(np.asarray(values)[rows].reshape(len(rows),-1)).all(axis=1)
        split_names=np.asarray(data["split"])[rows]
        language_availability[camera_name]={
            "available_rows":int(available.sum()),"total_rows":int(len(rows)),
            "available_fraction":float(available.mean()) if len(rows) else 0.0,
            "store_available_rows":int((available & (split_names=="store")).sum()),
            "query_available_rows":int((available & (split_names=="query")).sum()),
        }
    return {"arrays":data,"rows":rows,"keymap":keymap,
            "camera_language_availability":language_availability},None


def _meta(run,row):
    return {"episode_id":str(run.episode_id[row]),"init_state_id":str(run.init_state_id[row]),
            "task":str(run.task[row]),"suite":str(run.suite[row]),"step":int(run.step[row]),
            "segment_id":f"{run.episode_id[row]}@{int(run.step[row])}"}


def _method_decode_base(method,b):
    if isinstance(b,dict):
        if "embedding" in b: return np.asarray(b["embedding"],dtype=np.float32)
        if "base" in b and np.asarray(b["base"]).ndim>=1: return np.asarray(b["base"],dtype=np.float32)
        if "codes" in b and hasattr(method,"prototypes"): return np.asarray(method.prototypes)[np.asarray(b["codes"],dtype=np.int64)]
        if "alpha" in b and hasattr(method,"net") and method.net is not None:
            dictionary=method.net.dictionary.detach().cpu().numpy()
            dictionary=dictionary/np.maximum(np.linalg.norm(dictionary,axis=-1,keepdims=True),1e-8)
            alpha=np.asarray(b["alpha"],dtype=np.float32)
            if alpha.shape[-1]<dictionary.shape[0]: alpha=np.pad(alpha,((0,0),(0,dictionary.shape[0]-alpha.shape[-1])))
            return alpha@dictionary
        if "coarse_probs" in b and hasattr(method,"prior_base"): return np.asarray(method.prior_base(b),dtype=np.float32)
    if method.__class__.__module__.endswith(("methods.m2","methods.fsq")):
        arr=np.asarray(b)
        return method.base_features(arr) if np.issubdtype(arr.dtype,np.integer) else arr.astype(np.float32)
    return method.base_features(b) if hasattr(method,"base_features") else np.asarray(b,dtype=np.float32)


def _decoder_base(method,b):
    if isinstance(b,dict) and "alpha" in b: return {"alpha":np.asarray(b["alpha"],dtype=np.float32)}
    if isinstance(b,dict): return _method_decode_base(method,b)
    return _method_decode_base(method,b)


def _base_row(b,i):
    if isinstance(b,dict): return {k:(v[i:i+1] if hasattr(v,"__getitem__") else v) for k,v in b.items()}
    return np.asarray(b)[i] if np.asarray(b).ndim>0 else b


def _base_slice(b,selection):
    if isinstance(b,dict): return {k:v[selection] for k,v in b.items()}
    return np.asarray(b)[selection]


def _coverage_representatives(method, encoded, raw):
    """Assigned unit representative in the pre-quantization encoder space."""
    module=method.__class__.__module__; raw=np.asarray(raw,dtype=np.float32)
    if module.endswith(("methods.m0","methods.m1","methods.m6")):
        unitizer=_m6_unit_context(method)
        if unitizer is None or not len(unitizer._centers): return np.full(len(raw),-1,np.int64),np.zeros_like(raw)
        centers=np.asarray(unitizer._centers,dtype=np.float32)
        ids,_,_=_m6_ids(unitizer,raw,allow_pending=False)
        safe=np.maximum(ids,0); reps=centers[safe]
        reps[ids<0]=0
        return ids,reps
    if module.endswith(("methods.m2","methods.fsq")):
        codes=np.asarray(encoded,dtype=np.int64).reshape(-1)
        reps=method.base_features(codes)
        return codes,reps
    if module.endswith("methods.m3_rvq"):
        codes=np.asarray(encoded,dtype=np.int64).reshape(-1,2)
        with torch.no_grad():
            c=method.coarse_book(torch.as_tensor(codes[:,0],device=method._device))
            f=method.fine_book(torch.as_tensor(codes[:,1],device=method._device))
        # The encoder's pre-VQ vector is reconstructed by the RVQ sum c + f.
        reps=(c+f).cpu().numpy().astype(np.float32)
        ids=np.unique(codes,axis=0,return_inverse=True)[1]
        return ids,reps
    if module.endswith("methods.m4") and isinstance(encoded,dict):
        prototypes=np.asarray(method.prototypes,dtype=np.float32)
        ids=np.linalg.norm(raw[:,None,:]-prototypes[None,:,:],axis=-1).argmin(1)
        return ids,prototypes[ids]
    if module.endswith("methods.m5") and isinstance(encoded,dict):
        dictionary=method.net.dictionary.detach().cpu().numpy()
        dictionary=dictionary/np.maximum(np.linalg.norm(dictionary,axis=-1,keepdims=True),1e-8)
        alpha=np.asarray(encoded["alpha"],dtype=np.float32)
        if alpha.shape[-1]<len(dictionary): alpha=np.pad(alpha,((0,0),(0,len(dictionary)-alpha.shape[-1])))
        reps=alpha@dictionary
        ids=np.unique(np.round(alpha,6),axis=0,return_inverse=True)[1]
        return ids,reps
    reps=_method_decode_base(method,encoded)
    ids=np.arange(len(reps),dtype=np.int64)
    return ids,np.asarray(reps,dtype=np.float32)


def _unquantized_base(method, actions, state, encoded_base):
    """Return the continuous encoder output before any code/prototype assignment."""
    module=method.__class__.__module__
    if module.endswith(("methods.m0","methods.m1")):
        return _method_decode_base(method,encoded_base).astype(np.float32)
    if module.endswith("methods.m2"):
        from .methods._common import norm, tensor
        a=norm(np.asarray(actions,np.float32).reshape(len(actions),-1),method.am,method.ass)
        s=norm(np.asarray(state,np.float32),method.sm,method.ss)
        with torch.no_grad(): raw=method.encoder(torch.cat([tensor(a,method.device),tensor(s,method.device)],1))
        return raw[:,:method.embedding_dim].cpu().numpy().astype(np.float32)
    if module.endswith("methods.fsq"):
        from .methods._common import norm, tensor
        a=norm(np.asarray(actions,np.float32).reshape(len(actions),-1),method.am,method.ass)
        s=norm(np.asarray(state,np.float32),method.sm,method.ss)
        with torch.no_grad(): raw=method.encoder(torch.cat([tensor(a,method.device),tensor(s,method.device)],1))
        return raw[:,:method.base_dim].cpu().numpy().astype(np.float32)
    if module.endswith("methods.m3_rvq"):
        a=method._tensor(actions); s=method._tensor(state)
        flat=a.reshape(len(a),-1)
        with torch.no_grad():
            latent=method.encoder(torch.cat([method._norm_a(flat),method._norm_s(s)],dim=-1))
            z,_,_=method._split_encoder(latent)
        return z.cpu().numpy().astype(np.float32)
    if module.endswith("methods.m4") and isinstance(encoded_base,dict) and "embedding" in encoded_base:
        return np.asarray(encoded_base["embedding"],dtype=np.float32)
    if module.endswith("methods.m5") and isinstance(encoded_base,dict) and "raw_base" in encoded_base:
        return np.asarray(encoded_base["raw_base"],dtype=np.float32)
    return _method_decode_base(method,encoded_base).astype(np.float32)


def _query_parts(method,key,meta=None):
    module=method.__class__.__module__
    if module.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq","methods.m3_rvq","methods.m4","methods.m5","methods.m6")):
        memory=method.memory
        if not memory: return np.zeros((0,),np.float32),np.zeros((0,),np.float32)
        cached=getattr(method,"_bench_key_cache",None)
        if cached is None or cached[0]!=id(memory) or cached[1]!=len(memory):
            payload={}
            if module.endswith("methods.m3_rvq"): payload["codes"]=np.asarray([[e["coarse_id"],e["fine_id"]] for e in memory],dtype=np.int64)
            elif module.endswith("methods.m4"): payload["codes"]=np.asarray([e["code"] for e in memory],dtype=np.int64)
            elif module.endswith("methods.m5"):
                width=int(method.net.dictionary.shape[0]); alphas=[]
                for e in memory:
                    a=np.asarray(e["alpha"],dtype=np.float32)
                    if len(a)<width: a=np.pad(a,(0,width-len(a)))
                    alphas.append(a)
                payload["alpha"]=np.stack(alphas)
            else: payload["base"]=np.stack([np.asarray(e["base"]) for e in memory])
            keys=np.stack([np.asarray(e["key"],dtype=np.float32) for e in memory])
            if module.endswith(("methods.m3_rvq","methods.m4","methods.m5")):
                keys=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
            cached=(id(memory),len(memory),keys,
                    np.asarray([(e.get("meta") or {}).get("episode_id","") for e in memory],dtype=object),
                    np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in memory],dtype=object),payload)
            method._bench_key_cache=cached
        _,_,keys,episodes,initials,payload=cached
        q=np.asarray(key,dtype=np.float32).reshape(-1)
        eligible=np.ones(len(keys),dtype=bool)
        if meta is not None:
            eligible &= episodes!=meta.get("episode_id","")
            eligible &= initials!=meta.get("init_state_id","")
        ix=np.flatnonzero(eligible)
        if not len(ix): return np.zeros((0,),np.float32),np.zeros((0,),np.float32)
        if module.endswith(("methods.m3_rvq","methods.m4","methods.m5")):
            q=q/max(float(np.linalg.norm(q)),1e-8); similarities=keys[ix]@q
            if module.endswith("methods.m3_rvq"):
                logits=similarities/max(float(getattr(method,"retrieval_temperature",.1)),1e-6)
                probs=np.exp(logits-logits.max()); probs/=max(float(probs.sum()),1e-12)
                order=np.argsort(-probs,kind="stable")[:min(5,len(ix))]; w=probs[order]; w/=max(float(w.sum()),1e-12)
            else:
                order=np.asarray([int(np.argmax(similarities))]); w=np.ones(1,dtype=np.float32)
            chosen_ix=ix[order]
            if module.endswith("methods.m3_rvq"): return payload["codes"][chosen_ix],w.astype(np.float32)
            if module.endswith("methods.m4"): return np.asarray(method.prototypes)[payload["codes"][chosen_ix]],w.astype(np.float32)
            alpha=payload["alpha"][chosen_ix]; width=int(method.net.dictionary.shape[0])
            if alpha.shape[1]<width: alpha=np.pad(alpha,((0,0),(0,width-alpha.shape[1])))
            return {"alpha":alpha.astype(np.float32)},w.astype(np.float32)
        sq=(keys[ix]*keys[ix]).sum(1); d2=np.maximum(sq+float(q@q)-2*keys[ix]@q,0.0)
        k=int(getattr(method,"topk",5)); chosen=np.argsort(d2,kind="stable")[:min(k,len(ix))]
        chosen_ix=ix[chosen]; d=np.sqrt(d2[chosen]); temperature=float(getattr(method,"temperature",.5))
        dref=getattr(method,"d_ref",None)
        scale=(max(temperature,1e-12) if module.endswith("methods.m6") else
               max(temperature*(float(dref) if dref else max(float(d[0]),1e-12)),1e-12))
        w=np.exp(-(d-d[0])/scale); w=w/w.sum(); candidates=payload["base"][chosen_ix]
        return candidates.astype(np.float32) if not np.issubdtype(candidates.dtype,np.integer) else candidates,w.astype(np.float32)
    original=method.memory
    if meta is not None:
        method.memory=[e for e in original if (e.get("meta") or {}).get("episode_id")!=meta.get("episode_id")
                       and (e.get("meta") or {}).get("init_state_id")!=meta.get("init_state_id")]
    if method.__class__.__module__.endswith("methods.m5") and getattr(method,"net",None) is not None:
        width=int(method.net.dictionary.shape[0]); padded=[]
        for entry in method.memory:
            item=dict(entry); alpha=np.asarray(item["alpha"],dtype=np.float32)
            if len(alpha)<width: alpha=np.pad(alpha,(0,width-len(alpha)))
            item["alpha"]=alpha; padded.append(item)
        method.memory=padded
    try:
        try: ret=method.memory_query(key,meta)
        except TypeError:
            try: ret=method.memory_query(key,top_k=5)
            except TypeError: ret=method.memory_query(key)
    finally: method.memory=original
    if isinstance(ret,dict):
        weights=np.asarray(ret.get("weights",[]),dtype=np.float32)
        if "alphas" in ret: return {"alpha":np.asarray(ret["alphas"],dtype=np.float32)},weights
        if "bases" in ret: return np.asarray(ret["bases"]),weights
    if isinstance(ret,(tuple,list)) and len(ret)>=2: return ret[0],np.asarray(ret[1],dtype=np.float32)
    return np.zeros((0,),np.float32),np.zeros((0,),np.float32)


def _query_many_parts(method,keys,metas):
    """Batched equivalent of _query_parts; memory stays fixed for the batch."""
    module=method.__class__.__module__
    supported=("methods.m0","methods.m1","methods.m2","methods.fsq","methods.m3_rvq","methods.m4","methods.m5","methods.m6")
    if not module.endswith(supported):
        pairs=[_query_parts(method,k,m) for k,m in zip(keys,metas)]
        return [x[0] for x in pairs],[x[1] for x in pairs]
    memory=method.memory; n=len(keys)
    if n==0:return [],[]
    if not memory: return [np.zeros((0,),np.float32) for _ in range(n)],[np.zeros((0,),np.float32) for _ in range(n)]
    cache=getattr(method,"_bench_key_cache",None)
    if cache is None or cache[0]!=id(memory) or cache[1]!=len(memory):
        # Build once through the scalar path, which owns dtype and M5 padding rules.
        _query_parts(method,keys[0],metas[0] if n else None); cache=method._bench_key_cache
    _,_,stored,episodes,initials,payload=cache
    query=np.asarray(keys,dtype=np.float32).reshape(n,-1)
    if query.shape[1]!=stored.shape[1]: raise ValueError("query and memory key dimensions differ")
    out_b=[]; out_w=[]; block=128
    cosine=module.endswith(("methods.m3_rvq","methods.m4","methods.m5"))
    for lo in range(0,n,block):
        hi=min(n,lo+block); q=query[lo:hi]
        if cosine:
            q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8); scores=q@stored.T
        else:
            qnorm=(q*q).sum(1)[:,None]; knorm=(stored*stored).sum(1)[None,:]
            scores=np.sqrt(np.maximum(qnorm+knorm-2*q@stored.T,0.0))
        eligible=np.ones(scores.shape,dtype=bool)
        for i,meta in enumerate(metas[lo:hi]):
            if meta is not None:
                eligible[i]&=(episodes!=meta.get("episode_id",""))&(initials!=meta.get("init_state_id",""))
        for i in range(hi-lo):
            ix=np.flatnonzero(eligible[i])
            if not len(ix): out_b.append(np.zeros((0,),np.float32)); out_w.append(np.zeros((0,),np.float32)); continue
            vals=scores[i,ix]
            if module.endswith("methods.m3_rvq"):
                logits=vals/max(float(getattr(method,"retrieval_temperature",.1)),1e-6)
                prob=np.exp(logits-logits.max()); prob/=max(float(prob.sum()),1e-12)
                order=np.argsort(-prob,kind="stable")[:min(5,len(ix))]; chosen=ix[order]; w=prob[order]; w/=max(float(w.sum()),1e-12)
                out_b.append(payload["codes"][chosen]); out_w.append(w.astype(np.float32))
            elif module.endswith(("methods.m4","methods.m5")):
                chosen=np.asarray([ix[int(np.argmax(vals))]]); out_w.append(np.ones(1,np.float32))
                if module.endswith("methods.m4"): out_b.append(np.asarray(method.prototypes)[payload["codes"][chosen]])
                else: out_b.append({"alpha":payload["alpha"][chosen].astype(np.float32)})
            else:
                order=np.argsort(vals,kind="stable")[:min(int(getattr(method,"topk",5)),len(ix))]
                chosen=ix[order]; d=vals[order]; temp=float(getattr(method,"temperature",.5)); ref=getattr(method,"d_ref",None)
                scale=(max(temp,1e-12) if module.endswith("methods.m6") else
                       max(temp*(float(ref) if ref else max(float(d[0]),1e-12)),1e-12))
                w=np.exp(-(d-d[0])/scale); w/=w.sum(); out_b.append(payload["base"][chosen]); out_w.append(w.astype(np.float32))
    return out_b,out_w


def _memory_add(method,key,base,meta):
    module=method.__class__.__module__
    if module.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq")):
        key=np.asarray(key,dtype=np.float32).reshape(-1); memory=method.memory
        if not np.all(np.isfinite(key)) or not np.all(np.isfinite(np.asarray(base,dtype=float))): return False,"nonfinite"
        metadata=dict(meta or {})
        if module.endswith(("methods.m0","methods.m1")) and _m6_unit_context(method) is not None:
            unitizer=_m6_unit_context(method); feature=_method_decode_base(method,base).astype(np.float32).reshape(-1)
            if metadata.get("seed",False):
                unit_id,_distance=unitizer._nearest(feature)
                metadata.update({"skill_unit_id":unit_id,"skill_unit_added":False,"key_novel":True})
                memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":metadata})
                method._bench_key_cache=None
                return True,"seed"
            episode=str(metadata.get("episode_id",f"memory-row-{len(memory)}"))
            events_before=len(unitizer.unit_registration_events)
            unit_id,assignment,base_distance=unitizer._assign(feature,episode,training=False)
            if unitizer._last_pending_id is not None:
                for row in unitizer.pending:
                    if row["id"]==unitizer._last_pending_id:
                        row["meta"]=metadata.copy();break
            unitizer._cluster_pending()
            for registered in unitizer.unit_registration_events[events_before:]:
                for entry in memory:
                    emeta=entry.get("meta") or {}
                    if _sequence_row_identity(emeta)==_sequence_row_identity(registered):
                        emeta["skill_unit_id"]=int(registered["unit_id"]);emeta["skill_unit_added"]=True
            event=next((e for e in reversed(unitizer.unit_registration_events)
                        if _sequence_row_identity(e)==_sequence_row_identity(metadata)),None)
            if event is not None:unit_id=int(event["unit_id"])
            key_distance=_key_distance(memory,key,metadata)
            threshold=getattr(method,"tau",None); key_novel=threshold is None or key_distance>float(threshold)
            added=bool(event is not None)
            metadata.update({"key_novel":bool(key_novel),"skill_unit_added":added,"skill_unit_id":unit_id,
                "key_novelty_score":float(key_distance),"key_threshold":threshold,
                "base_novelty_score":float(base_distance),"base_threshold":float(unitizer.epsilon),
                "unit_assignment":assignment})
            method._d3_last_decision=dict(metadata)
            if key_novel or not memory:
                memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":metadata})
                method._bench_key_cache=None
                return True,"added_key_and_new_unit" if added else "added_new_key_existing_unit"
            method._d3_last_decision={**metadata,"skill_unit_added":added}
            return False,"known_key_registered_unit" if added else "known_key_pending_or_existing_unit"
        if not hasattr(method,"_d3_unit_centers"):
            _initialize_standard_units(method,np.asarray([base]))
        if metadata.get("seed",False):
            feature=_method_decode_base(method,base).astype(np.float32).reshape(1,-1)
            ds=_base_distance(feature,method._d3_unit_centers)
            decision={"key_novel":True,"skill_unit_added":False,"key_novelty_score":float("nan"),
                "key_threshold":getattr(method,"tau",None),"base_novelty_score":float(ds[0]),
                "base_threshold":getattr(method,"_d3_base_novelty_threshold",0.),
                "skill_unit_id":int(np.argmin(np.linalg.norm(feature-method._d3_unit_centers,axis=1))),"base_feature":feature[0]}
            metadata.update({k:v for k,v in decision.items() if k!="base_feature"})
            memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":metadata})
            method._bench_key_cache=None
            method._d3_last_decision=decision
            return True,"seed"
        decision=_standard_d3_decision(method,key,base,metadata)
        if not memory:
            metadata.update({k:v for k,v in decision.items() if k!="base_feature"})
            memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":metadata})
            method._d3_last_decision=decision
            return True,"added_empty"
        cached=getattr(method,"_bench_key_cache",None)
        if cached is None or cached[0]!=id(memory) or cached[1]!=len(memory):
            cached=(id(memory),len(memory),np.stack([np.asarray(e["key"],dtype=np.float32) for e in memory]),
                    np.asarray([(e.get("meta") or {}).get("episode_id","") for e in memory],dtype=object),
                    np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in memory],dtype=object),
                    {"base":np.stack([np.asarray(e["base"]) for e in memory])})
            method._bench_key_cache=cached
        _,_,stored,episodes,initials,cached_payload=cached
        eligible=(episodes!=metadata.get("episode_id",""))&(initials!=metadata.get("init_state_id",""))
        if eligible.any():
            sk=stored[eligible]; d2=np.maximum((sk*sk).sum(1)+float(key@key)-2*sk@key,0.0)
            distance=float(np.sqrt(d2.min()))
        else: distance=float("inf")
        key_threshold=getattr(method,"tau",None)
        decision["key_novelty_score"]=distance
        decision["key_novel"]=key_threshold is None or distance>float(key_threshold)
        decision["skill_unit_id"]=int(np.argmin(np.linalg.norm(decision["base_feature"][None,:]-method._d3_unit_centers,axis=1))) if len(method._d3_unit_centers) else 0
        if not (decision["key_novel"] or decision["skill_unit_added"]):
            method._d3_last_decision=decision
            return False,"known_key_known_base"
        metadata.update({k:v for k,v in decision.items() if k!="base_feature"})
        memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":metadata})
        if decision["skill_unit_added"]:
            decision["skill_unit_id"]=len(method._d3_unit_centers)
            metadata["skill_unit_id"]=decision["skill_unit_id"]
            method._d3_unit_centers=np.concatenate([method._d3_unit_centers,decision["base_feature"][None]],axis=0)
        method._d3_last_decision=decision
        if cached is not None and cached[0]==id(memory) and cached[1]==len(memory)-1:
            method._bench_key_cache=(id(memory),len(memory),np.concatenate([stored,key[None]],axis=0),
                np.append(episodes,str((meta or {}).get("episode_id",""))),
                np.append(initials,str((meta or {}).get("init_state_id",""))),
                {"base":np.concatenate([cached_payload["base"],np.asarray(base)[None]],axis=0)})
        else: method._bench_key_cache=None
        if decision["skill_unit_added"] and decision["key_novel"]: return True,"added_key_and_new_unit"
        if decision["skill_unit_added"]: return True,"added_new_unit"
        return True,"added_new_key_existing_unit"
    if module.endswith("methods.m4"):
        key=np.asarray(key,dtype=np.float32).reshape(-1); metadata=dict(meta or {})
        embedding=np.asarray(method._base_array(base),dtype=np.float32).reshape(-1,method.base_dim)[0]
        embedding=embedding/max(float(np.linalg.norm(embedding)),1e-8)
        distances=np.linalg.norm(np.asarray(method.prototypes,dtype=np.float32)-embedding[None,:],axis=-1)
        unit_id=int(np.argmin(distances)); base_distance=float(distances[unit_id]); base_threshold=float(method.novelty_distance_threshold or 0.)
        if metadata.get("seed",False):
            metadata.update({"key_novel":True,"skill_unit_added":False,"skill_unit_id":unit_id,
                             "base_novelty_score":base_distance,"base_threshold":base_threshold})
            method.memory.append({"key":key.copy(),"code":unit_id,"meta":metadata}); method._bench_key_cache=None
            return True,"seed"
        key_distance=_key_distance(method.memory,key,metadata,cosine=True); key_threshold=getattr(method,"tau",None)
        key_novel=key_threshold is None or key_distance>float(key_threshold); unit_novel=base_distance>base_threshold
        if not (key_novel or unit_novel):
            method._d3_last_decision={"key_novel":False,"skill_unit_added":False,"key_novelty_score":key_distance,
                "key_threshold":key_threshold,"base_novelty_score":base_distance,"base_threshold":base_threshold}
            return False,"known_key_known_base"
        metadata.update({"key_novel":bool(key_novel),"key_novelty_score":key_distance,"key_threshold":key_threshold,
                         "base_novelty_score":base_distance,"base_threshold":base_threshold,
                         "skill_unit_added":False,"skill_unit_id":unit_id})
        meta=metadata
    if module.endswith("methods.m5"):
        key=np.asarray(key,dtype=np.float32).reshape(-1); metadata=dict(meta or {})
        alpha=np.asarray(method._alpha(base),dtype=np.float32).reshape(-1)
        if metadata.get("seed",False):
            metadata.update({"key_novel":True,"skill_unit_added":False,"skill_unit_id":None})
            method.memory.append({"key":key.copy(),"alpha":alpha.copy(),"meta":metadata}); method._bench_key_cache=None
            return True,"seed"
        key_distance=_key_distance(method.memory,key,metadata,cosine=True); key_threshold=getattr(method,"tau",None)
        key_novel=key_threshold is None or key_distance>float(key_threshold); old_units=_unit_count(method)
        unit_score=float(np.linalg.norm(np.asarray(base.get("innovation",np.zeros((1,method.base_dim))),dtype=np.float32).reshape(-1)))
        unit_threshold=float(getattr(method,"novelty_threshold",np.inf) or np.inf)
        base_novel=unit_score>unit_threshold
        if not key_novel and not base_novel: return False,"known_key_known_base"
        metadata.update({"key_novel":bool(key_novel),"key_novelty_score":key_distance,"key_threshold":key_threshold,
                         "base_novelty_score":unit_score,"base_threshold":unit_threshold,"skill_unit_added":False})
        raw=method.memory_add(key,base,metadata)
        unit_added=bool(raw.get("added_atom",False)); metadata["skill_unit_added"]=unit_added
        metadata["skill_unit_id"]=old_units if unit_added else None
        if not key_novel and not unit_added:
            if method.memory and method.memory[-1].get("meta") is metadata: method.memory.pop()
            method._bench_key_cache=None
            return False,"known_key_pending_unit" if base_novel else "known_key_known_base"
        method._bench_key_cache=None
        return True,"added_key_and_new_unit" if key_novel and unit_added else "added_new_unit" if unit_added else "added_new_key_existing_unit"
    try:
        raw=method.memory_add(key,base,meta)
    except NameError as exc:
        # M4's documented capacity path currently references an undefined
        # `similarity` name after it has already established the nearest
        # prototype. Preserve its intended capacity-miss behavior here without
        # editing the independently-owned M4 implementation.
        if not module.endswith("methods.m4") or "similarity" not in str(exc):
            raise
        embedding=np.asarray(method._base_array(base),dtype=np.float32).reshape(-1,method.base_dim)[0]
        embedding=embedding/max(float(np.linalg.norm(embedding)),1e-8)
        distances=np.linalg.norm(np.asarray(method.prototypes)-embedding[None,:],axis=-1)
        nearest=int(np.argmin(distances)); distance=float(distances[nearest])
        method.memory.append({"key":np.asarray(key,dtype=np.float32).reshape(-1).copy(),"code":nearest,"meta":meta})
        raw={"added":True,"added_prototype":False,"code":nearest,"reason":"vigilance_miss_capacity_reached",
             "base_distance":distance,"novelty_distance_threshold":float(method.novelty_distance_threshold or 0.)}
    if module.endswith("methods.m4") and isinstance(raw,dict):
        is_new=bool(raw.get("added_prototype",False)); metadata.update({"skill_unit_added":is_new,
            "skill_unit_id":int(raw.get("code",0)),"added_reason":raw.get("reason"),
            "base_novelty_score":float(raw.get("base_distance",0.)),"base_threshold":float(raw.get("novelty_distance_threshold",0.))})
        if method.memory: method.memory[-1]["meta"]=metadata
        method._d3_last_decision=metadata
    result=_memory_add_result(raw); method._bench_key_cache=None
    return result


def _memory_add_many_standard(method,keys,bases,metas,*,block_size=256):
    """Greedy key-item and base-unit insertion with blocked key distances."""
    module=method.__class__.__module__
    if not module.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq")):
        raise TypeError("batched add is only valid for standard threshold memories")
    keys=np.asarray(keys,dtype=np.float32).reshape(len(keys),-1); accepted=np.zeros(len(keys),bool); reasons={}
    if not hasattr(method,"_d3_unit_centers"):
        _initialize_standard_units(method,np.stack([np.asarray(b) for b in bases]))
    features=_method_decode_base(method,np.stack([np.asarray(b) for b in bases])).astype(np.float32)
    tau=getattr(method,"tau",None); base_tau=float(getattr(method,"_d3_base_novelty_threshold",0.)); memory=method.memory
    for lo in range(0,len(keys),block_size):
        hi=min(len(keys),lo+block_size); q=keys[lo:hi]; metas_block=metas[lo:hi]
        if memory:
            old_keys=np.stack([np.asarray(e["key"],dtype=np.float32) for e in memory]); old_ep=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in memory],dtype=object)
            old_init=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in memory],dtype=object)
            cross=np.maximum((q*q).sum(1)[:,None]+(old_keys*old_keys).sum(1)[None,:]-2*q@old_keys.T,0.0)
        else:
            old_keys=np.zeros((0,keys.shape[1]),np.float32);old_ep=np.zeros(0,object);old_init=np.zeros(0,object);cross=np.zeros((len(q),0),np.float32)
        pair=np.maximum((q*q).sum(1)[:,None]+(q*q).sum(1)[None,:]-2*q@q.T,0.0)
        ep=np.asarray([m.get("episode_id","") for m in metas_block],dtype=object); init=np.asarray([m.get("init_state_id","") for m in metas_block],dtype=object)
        old_ok=np.ones(cross.shape,dtype=bool)
        if len(old_ep): old_ok=(ep[:,None]!=old_ep[None,:])&(init[:,None]!=old_init[None,:])
        pair_ok=(ep[:,None]!=ep[None,:])&(init[:,None]!=init[None,:])
        accepted_local=[]; append_items=[]
        for i in range(len(q)):
            bi=lo+i; feature=features[bi]
            finite=np.isfinite(q[i]).all() and np.isfinite(feature).all()
            if not finite: reasons["nonfinite"]=reasons.get("nonfinite",0)+1; continue
            key_distance=float(np.sqrt(cross[i,old_ok[i]].min())) if len(old_keys) and old_ok[i].any() else float("inf")
            if accepted_local:
                prev=np.asarray(accepted_local,dtype=np.int64); valid=pair_ok[i,prev]
                if valid.any(): key_distance=min(key_distance,float(np.sqrt(pair[i,prev[valid]].min())))
            key_new=tau is None or key_distance>tau
            centers=method._d3_unit_centers
            if len(centers):
                d2=np.maximum((centers*centers).sum(1)+(feature*feature).sum()-2*centers@feature,0.)
                unit_id=int(np.argmin(d2)); base_distance=float(np.sqrt(d2[unit_id]))
            else: unit_id=0; base_distance=float("inf")
            unit_new=base_distance>base_tau
            if not (key_new or unit_new):
                reasons["known_key_known_base"]=reasons.get("known_key_known_base",0)+1; continue
            accepted[i]=True; accepted_local.append(i)
            if unit_new:
                unit_id=len(method._d3_unit_centers)
                method._d3_unit_centers=np.concatenate([method._d3_unit_centers,feature[None]],axis=0)
                reason="added_key_and_new_unit" if key_new else "added_new_unit"
                reasons["new_skill_units"]=reasons.get("new_skill_units",0)+1
            else: reason="added_new_key_existing_unit"
            reasons[reason]=reasons.get(reason,0)+1
            reasons["new_key_items"]=reasons.get("new_key_items",0)+int(key_new)
            metadata=dict(metas_block[i]); metadata.update({"key_novel":bool(key_new),"skill_unit_added":bool(unit_new),
                "skill_unit_id":int(unit_id),"key_novelty_score":float(key_distance),"key_threshold":tau,
                "base_novelty_score":float(base_distance),"base_threshold":base_tau})
            append_items.append({"key":q[i].copy(),"base":np.array(bases[bi],copy=True),"meta":metadata})
        memory.extend(append_items)
    method._d3_last_decision={"batch_item_add_rate":float(accepted.mean()) if len(accepted) else 0.,
                              "batch_new_skill_units":int(reasons.get("new_skill_units",0))}
    method._bench_key_cache=None
    return accepted,reasons


def _memory_add_result(ret):
    if isinstance(ret,dict): return bool(ret.get("added",ret.get("added_prototype",False))),str(ret.get("reason","added"))
    if isinstance(ret,(tuple,list)) and len(ret)>=2: return bool(ret[0]),str(ret[1])
    return bool(ret),("added" if ret else "rejected")


def _decode_one(method,base,residual,feature,state):
    bb=_decoder_base(method,base)
    if isinstance(bb,dict): bb={k:(v[None] if np.asarray(v).ndim==1 else v) for k,v in bb.items()}
    else:
        bb=np.asarray(bb); bb=bb[None] if bb.ndim==1 else bb
    rr=None if residual is None else np.asarray(residual)
    return method.decode(bb,rr,feature,state)[0]


def _blend(method,candidates,weights):
    if isinstance(candidates,dict): return {k:np.asarray(weights,dtype=np.float32)@np.asarray(v,dtype=np.float32) for k,v in candidates.items()}
    if hasattr(method,"blend_base"): return np.asarray(method.blend_base(candidates,weights),dtype=np.float32)
    return np.asarray(weights,dtype=np.float32)@np.asarray(candidates,dtype=np.float32)


def _stack_base_values(values):
    if isinstance(values[0],dict): return {k:np.stack([np.asarray(v[k]).reshape(-1) for v in values]) for k in values[0]}
    return np.stack([np.asarray(v).reshape(-1) for v in values])


def _query_bases(method,key,rows,run):
    top1=[]; blend=[]; empty=0
    for r in rows:
        meta=_meta(run,r)
        cand,w=_query_parts(method,key[r],meta)
        if len(w)==0:
            empty+=1; top1.append(None); blend.append(None); continue
        top1.append(_blend(method,{k:np.asarray(v)[:1] for k,v in cand.items()} if isinstance(cand,dict) else np.asarray(cand)[:1],np.asarray([1.],np.float32)))
        blend.append(_blend(method,cand,w))
    return top1,blend,empty


def _decode_queries(method,mode,query_rows,run,*,base=None,residual=None,batches=64):
    outputs=[]
    for lo in range(0,len(query_rows),batches):
        rr=query_rows[lo:lo+batches]; bb=np.asarray(base[lo:lo+len(rr)])
        res=None if residual is None else np.asarray(residual[lo:lo+len(rr)])
        outputs.append(method.decode(bb,res,run.keys["vlm_img_lang"][rr],run.proprio[rr]))
    return np.concatenate(outputs)


def _mse(pred,target,mask):
    err=((pred-target)**2).mean(axis=2)
    return (err*mask).sum(1)/np.maximum(mask.sum(1),1)


def _rbd_hsic(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    if y.ndim==1:y=y[:,None]
    def K(z):
        d=np.maximum((z*z).sum(1)[:,None]-2*z@z.T+(z*z).sum(1)[None],0); med=np.median(d[d>0]) if np.any(d>0) else 1.
        return np.exp(-d/max(med,1e-12))
    n=len(x); h=np.eye(n)-np.ones((n,n))/n
    return float(np.sum((h@K(x)@h)*(h@K(y)@h))/max((n-1)**2,1))


def _unit_metrics(method,b_train,b_query,r_query):
    if isinstance(b_train,dict) and "codes" in b_train:
        raw_train=np.asarray(b_train["codes"]); raw_query=np.asarray(b_query["codes"])
        if raw_train.ndim>1:
            _,labels=np.unique(np.concatenate([raw_train,raw_query],axis=0),axis=0,return_inverse=True)
            train_ids=labels[:len(raw_train)]; query_ids=labels[len(raw_train):]
        else:
            _,labels=np.unique(np.concatenate([raw_train.reshape(-1),raw_query.reshape(-1)]),return_inverse=True)
            train_ids=labels[:len(raw_train)]; query_ids=labels[len(raw_train):]
        return train_ids,query_ids,np.unique(train_ids)
    if isinstance(b_train,dict) and "alpha" in b_train:
        at=np.asarray(b_train["alpha"],dtype=np.float32); aq=np.asarray(b_query["alpha"],dtype=np.float32)
        width=max(at.shape[-1],aq.shape[-1]); at=np.pad(at,((0,0),(0,width-at.shape[-1]))) if at.shape[-1]<width else at
        aq=np.pad(aq,((0,0),(0,width-aq.shape[-1]))) if aq.shape[-1]<width else aq
        signatures=np.concatenate([np.abs(at)>1e-6,np.abs(aq)>1e-6],axis=0)
        _,labels=np.unique(signatures,axis=0,return_inverse=True)
        train_ids=labels[:len(at)]; query_ids=labels[len(at):]
        return train_ids,query_ids,np.unique(train_ids)
    if isinstance(b_train,dict) and "alpha" in b_train:
        b_train=_method_decode_base(method,b_train); b_query=_method_decode_base(method,b_query)
    if np.issubdtype(np.asarray(b_train).dtype,np.integer):
        raw_train=np.asarray(b_train,dtype=np.int64); raw_query=np.asarray(b_query,dtype=np.int64)
        if raw_train.ndim>1:
            _,all_ids=np.unique(np.concatenate([raw_train,raw_query]),axis=0,return_inverse=True)
            utrain=all_ids[:len(raw_train)]; uquery=all_ids[len(raw_train):]
        else: utrain=raw_train.reshape(-1); uquery=raw_query.reshape(-1)
        units=np.unique(utrain)
        return utrain,uquery,units
    feat_tr=_method_decode_base(method,b_train); feat_q=_method_decode_base(method,b_query)
    unitizer=_m6_unit_context(method)
    if unitizer is None:
        return None,None,np.zeros(0,dtype=np.int64)
    train_ids,_,units=_m6_ids(unitizer,feat_tr,allow_pending=True)
    query_ids,_,_=_m6_ids(unitizer,feat_q,allow_pending=False)
    return train_ids,query_ids,units


def _new_m6_evaluator(backbone, bases, residual, states, features, episodes, *, seed=0,
                      kappa=.1, min_support=10, min_episodes=3, max_fit_rows=256):
    """Fit M6's functional pending/registration unitizer on an existing encoder.

    This is an evaluation adapter only: it reuses the candidate method's frozen
    decoder, while M6 owns the functional radius, pending clusters, and support
    rules. A deterministic cap bounds the common evaluator for large train sets.
    """
    bases=np.asarray(bases,dtype=np.float32); n=len(bases)
    if not n:return None
    rng=np.random.default_rng(seed+6106)
    take=np.sort(rng.choice(n,min(n,max_fit_rows),replace=False)) if n>max_fit_rows else np.arange(n)
    rs=np.zeros((len(take),1),np.float32) if residual is None else np.asarray(residual,dtype=np.float32)[take]
    ss=np.asarray(states,dtype=np.float32)[take]; ff=np.asarray(features,dtype=np.float32)[take]
    eps=np.asarray(episodes).astype(str)[take]
    unitizer=M6(kappa=kappa,min_support=min_support,min_episodes=min_episodes,
                pending_batch_size=max(len(take),1),seed=seed)
    unitizer.backbone=backbone; unitizer.base_dim=int(bases.shape[-1]); unitizer._fitted=True
    probe=min(16,len(take))
    unitizer._functional_probe={"residual":rs[:probe],"state":ss[:probe],"feature":ff[:probe]}
    pair_ix=rng.integers(0,len(take),size=(min(128,max(1,len(take)*2)),2))
    distances=[unitizer._functional_distance(bases[take[i]],bases[take[j]]) for i,j in pair_ix]
    finite=np.asarray(distances,dtype=np.float64); finite=finite[np.isfinite(finite)]
    unitizer.functional_median=float(np.median(finite)) if len(finite) else 0.0
    unitizer.epsilon=max(kappa*unitizer.functional_median,1e-12)
    unitizer._fit_in_progress=True
    for ix in range(len(take)):
        unitizer._data_seen+=1
        unitizer._assign(bases[take[ix]],eps[ix],training=True)
    unitizer._cluster_pending(force=True); unitizer._fit_in_progress=False
    return unitizer


def _m6_ids(unitizer, bases, *, allow_pending=False):
    bases=np.asarray(bases,dtype=np.float32).reshape(len(bases),-1)
    ids=np.full(len(bases),-1,dtype=np.int64); distances=np.full(len(bases),np.inf,dtype=np.float32)
    units=np.arange(len(unitizer._centers),dtype=np.int64)
    if not len(units):return ids,distances,units
    if not hasattr(unitizer,"_functional_probe"):
        for i,base in enumerate(bases):
            unit,distance=unitizer._nearest(base); distances[i]=distance
            if unit is not None and (allow_pending or distance<=unitizer.epsilon):ids[i]=unit
        return ids,distances,units
    centers=np.asarray(unitizer._centers,dtype=np.float32)
    probe=unitizer._functional_probe; n_probe=len(probe["state"]); top=min(10,len(centers))

    # Functional distance is the mean squared difference between decoder
    # outputs over a fixed probe. Decode each base and each center once, then
    # do the candidate comparisons as a matrix operation. The previous loop
    # decoded every query again for every one of its ten L2-prefiltered
    # candidates (roughly 10x redundant decoder work on full training sets).
    def embeddings(values):
        values=np.asarray(values,dtype=np.float32).reshape(len(values),-1)
        cache=getattr(unitizer,"_functional_embedding_cache",None)
        if cache is None:
            cache={}; unitizer._functional_embedding_cache=cache
        keys=[np.ascontiguousarray(row).tobytes() for row in values]
        result=[cache.get(key) for key in keys]
        missing=np.asarray([i for i,value in enumerate(result) if value is None],dtype=np.int64)
        for start in range(0,len(missing),128):
            indices=missing[start:start+128]; block=values[indices]
            repeated=np.repeat(block,n_probe,axis=0)
            residual=np.tile(np.asarray(probe["residual"],dtype=np.float32),(len(block),1))
            state=np.tile(np.asarray(probe["state"],dtype=np.float32),(len(block),1))
            feature=np.tile(np.asarray(probe["feature"],dtype=np.float32),(len(block),1))
            decoded=np.asarray(unitizer._decode(repeated,residual,feature,state),dtype=np.float32)
            signatures=decoded.reshape(len(block),-1)/np.sqrt(max(decoded[0].size*n_probe,1))
            for index,signature in zip(indices,signatures):
                cache[keys[int(index)]]=signature
                result[int(index)]=signature
        return np.stack(result)

    q_embed=embeddings(bases); center_embed=embeddings(centers)
    scale=float(q_embed.shape[1])
    center_norm=(center_embed*center_embed).sum(1)
    for lo in range(0,len(bases),512):
        hi=min(len(bases),lo+512); q=bases[lo:hi]
        l2=np.maximum((q*q).sum(1)[:,None]-2*q@centers.T+(centers*centers).sum(1)[None],0.)
        # Only the ten L2 prefilter candidates are needed for the functional
        # comparison. Full argsort over every open-ended unit was dominating
        # repeated stage metrics as M6 grew; argpartition preserves the same
        # nearest-candidate set, then stable-sorts that small set.
        ix=np.argpartition(l2,top-1,axis=1)[:,:top]
        local_l2=np.take_along_axis(l2,ix,axis=1)
        local_order=np.argsort(local_l2,axis=1,kind="stable")
        ix=np.take_along_axis(ix,local_order,axis=1)
        candidate_embed=center_embed[ix]
        qv=q_embed[lo:hi,None,:]
        functional=np.maximum((qv*qv).sum(2)-2*np.sum(qv*candidate_embed,axis=2)+center_norm[ix],0.)/scale
        local=np.argmin(functional,axis=1); rows=np.arange(len(q)); best=ix[rows,local]
        distances[lo:hi]=functional[rows,local].astype(np.float32)
        if allow_pending:ids[lo:hi]=best
        else:
            ok=distances[lo:hi]<=unitizer.epsilon
            ids[lo:hi][ok]=best[ok]
    return ids,distances,units


def _m6_unit_context(method):
    return getattr(method,"_bench_m6_unitizer",method if method.__class__.__module__.endswith("methods.m6") else None)


def _unit_centers(method, train_b):
    if isinstance(train_b,dict) and "codes" in train_b: return None
    if isinstance(train_b,dict) and "alpha" in train_b: train_b=_method_decode_base(method,train_b)
    train_b=np.asarray(train_b)
    if np.issubdtype(train_b.dtype,np.integer): return None
    unitizer=_m6_unit_context(method)
    if unitizer is not None:
        return np.asarray(unitizer._centers,dtype=np.float32).reshape(-1,unitizer.base_dim)
    return None


def _initialize_standard_units(method,train_b):
    raw=np.asarray(train_b)
    if np.issubdtype(raw.dtype,np.integer):
        centers=_method_decode_base(method,np.unique(raw))
        distances=_base_distance(_method_decode_base(method,train_b),centers)
    else:
        unitizer=_m6_unit_context(method)
        features=_method_decode_base(method,train_b).astype(np.float32)
        if unitizer is None:
            method._d3_unit_centers=np.empty((0,features.shape[-1]),np.float32)
            method._d3_base_novelty_threshold=float("inf"); method._d3_last_decision={}
            return
        centers=np.asarray(unitizer._centers,dtype=np.float32).reshape(-1,unitizer.base_dim)
        _,distances,_=_m6_ids(unitizer,features,allow_pending=False)
        distances=distances[np.isfinite(distances)]
    threshold=max(float(np.quantile(distances,.95)) if len(distances) else 0.,1e-12)
    method._d3_unit_centers=np.asarray(centers,dtype=np.float32).reshape(len(centers),-1)
    method._d3_base_novelty_threshold=threshold
    method._d3_last_decision={}


def _key_distance(memory,key,meta,*,cosine=False):
    eligible=[e for e in memory if (e.get("meta") or {}).get("episode_id")!=meta.get("episode_id")
              and (e.get("meta") or {}).get("init_state_id")!=meta.get("init_state_id")]
    if not eligible:return float("inf")
    query=np.asarray(key,dtype=np.float32).reshape(-1)
    stored=np.stack([np.asarray(e["key"],dtype=np.float32).reshape(-1) for e in eligible])
    if cosine:
        query=query/max(float(np.linalg.norm(query)),1e-8)
        stored=stored/np.maximum(np.linalg.norm(stored,axis=1,keepdims=True),1e-8)
        return float(1.0-np.max(stored@query))
    return float(np.linalg.norm(stored-query[None,:],axis=1).min())


def _calibrate_cosine_key_threshold(keys,episode_ids,initial_ids,quantile=.95,*,max_rows=4096,block=256):
    keys=np.asarray(keys,dtype=np.float32); ep=np.asarray(episode_ids).astype(str); init=np.asarray(initial_ids).astype(str)
    if len(keys)<2:return None,None
    if len(keys)>max_rows:
        ix=np.sort(np.random.default_rng(7081).choice(len(keys),max_rows,replace=False)); keys,ep,init=keys[ix],ep[ix],init[ix]
    keys=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8); nearest=np.full(len(keys),np.inf)
    for lo in range(0,len(keys),block):
        hi=min(len(keys),lo+block); scores=keys[lo:hi]@keys.T
        excluded=(ep[lo:hi,None]==ep[None,:])|(init[lo:hi,None]==init[None,:])
        scores[excluded]=-np.inf
        nearest[lo:hi]=1.0-scores.max(1)
    finite=nearest[np.isfinite(nearest)]
    return ((float(np.quantile(finite,quantile)),float(np.median(finite))) if len(finite) else (None,None))


def _standard_d3_decision(method,key,base,meta):
    feature=_method_decode_base(method,base).astype(np.float32).reshape(1,-1)
    centers=np.asarray(getattr(method,"_d3_unit_centers",np.empty((0,feature.shape[1]))),dtype=np.float32)
    if not len(centers): unit_id=0; base_distance=float("inf")
    else:
        ds=_base_distance(feature,centers); unit_id=int(np.argmin(np.linalg.norm(feature-centers,axis=1)))
        base_distance=float(ds[0])
    key=np.asarray(key,dtype=np.float32).reshape(-1); memory=method.memory
    eligible=[e for e in memory if e.get("meta",{}).get("episode_id")!=meta.get("episode_id")
              and e.get("meta",{}).get("init_state_id")!=meta.get("init_state_id")]
    key_distance=min((float(np.linalg.norm(key-np.asarray(e["key"],dtype=np.float32))) for e in eligible),default=float("inf"))
    key_threshold=getattr(method,"tau",None); base_threshold=getattr(method,"_d3_base_novelty_threshold",0.)
    is_seed=bool(meta.get("seed",False))
    key_new=bool(is_seed or key_threshold is None or key_distance>float(key_threshold))
    unit_new=bool(not is_seed and base_distance>float(base_threshold))
    return {"key_novel":key_new,"skill_unit_added":unit_new,"key_novelty_score":key_distance,
            "key_threshold":key_threshold,"base_novelty_score":base_distance,
            "base_threshold":base_threshold,"skill_unit_id":unit_id,"base_feature":feature[0]}


def _unit_ids(method, values, centers):
    if isinstance(values,dict) and "codes" in values: return np.asarray(values["codes"]).reshape(-1)
    if isinstance(values,dict) and "alpha" in values: values=_method_decode_base(method,values)
    values=np.asarray(values)
    if np.issubdtype(values.dtype,np.integer):
        if values.ndim>1: return np.unique(values,axis=0,return_inverse=True)[1]
        return values.astype(np.int64).reshape(-1)
    return assign(_method_decode_base(method,values),centers)


def _functional_unit_ablation(method,query_rows,run,key,prior_base,prior_residual,prior_pred,target,mask,train_b):
    memory=method.memory
    if not memory: return {"units":[],"redundant_unit_fraction":None,"relative_threshold":.01,"n_query":0}
    atom_mode=isinstance(memory[0],dict) and "alpha" in memory[0] and hasattr(method,"net")
    def entry_units(entry):
        if atom_mode: return [i for i,x in enumerate(np.asarray(entry.get("alpha",[]))) if x!=0]
        meta=entry.get("meta") or {}
        if "skill_unit_id" in meta:
            return [] if meta["skill_unit_id"] is None else [int(meta["skill_unit_id"])]
        if "unit_id" in entry:return [] if entry.get("unit_id") is None else [int(entry["unit_id"])]
        if "code" in entry:return [int(entry["code"])]
        if "coarse_id" in entry:return [int(entry["coarse_id"])*int(getattr(method,"fine_codes",1))+int(entry.get("fine_id",-1))+1]
        return [hash(tuple(np.asarray(entry.get("base",[])).reshape(-1).tolist()))]
    mem_units=[entry_units(e) for e in memory]
    all_units=np.asarray(sorted({u for us in mem_units for u in us}),dtype=np.int64)
    sub=np.arange(min(128,len(query_rows))); rows=query_rows[sub]
    metas=[_meta(run,row) for row in rows]
    nearest=_nearest_memory_items_many(method,key[rows],metas,top_k=5)
    usage={int(u):np.asarray([any(u in entry_units(e) for e in got) for got in nearest],dtype=bool) for u in all_units}
    counts=np.asarray([sum(int(u in us) for us in mem_units) for u in all_units])
    units=all_units
    if len(units)>32:
        order=np.argsort(counts,kind="stable"); units=np.asarray(sorted(set(units[order[:16]].tolist()+units[order[-16:]].tolist())),dtype=np.int64)
    baseline_c,baseline_w=_query_many_parts(method,key[rows],metas)
    baseline_b=[_blend(method,c,w) if len(w) else _base_row(prior_base,int(sub[i]))
                for i,(c,w) in enumerate(zip(baseline_c,baseline_w))]
    baseline_batch=_stack_base_values(baseline_b)
    baseline_pred=method.decode(baseline_batch,None if prior_residual is None else prior_residual[sub],
                                 run.keys["vlm_img_lang"][rows],run.proprio[rows])
    baseline_row=_mse(baseline_pred,target[sub],mask[sub]); reports=[]
    original=method.memory
    for unit in units:
        used=usage[int(unit)]
        if not used.any():
            reports.append({"unit_id":int(unit),"query_usage_fraction":0.0,"delta_mse_on_used_queries":None,"relative_error_increase":None,"usage_weighted_relative_contribution":0.0,"stored_items_using_unit":int(counts[np.where(all_units==unit)[0][0]])})
            continue
        if atom_mode:
            method.memory=copy.deepcopy(original)
            for entry in method.memory:
                if unit<len(entry["alpha"]): entry["alpha"][unit]=0.0
        else:
            method.memory=[e for e,us in zip(original,mem_units) if int(unit) not in us]
        loc=np.flatnonzero(used); rr=rows[loc]
        cands,weights=_query_many_parts(method,key[rr],[metas[i] for i in loc]); bases=[]
        for i,(cand,w) in enumerate(zip(cands,weights)):
            bi=int(loc[i]); b=_blend(method,cand,w) if len(w) else _base_row(prior_base,int(sub[bi]))
            if isinstance(b,dict): bases.append({k:np.asarray(v).reshape(-1) for k,v in b.items()})
            else: bases.append(np.asarray(b).reshape(-1))
        batch_base=({k:np.stack([b[k] for b in bases]) for k in bases[0]} if isinstance(bases[0],dict) else np.stack(bases))
        rr_res=None if prior_residual is None else prior_residual[sub[loc]]
        pred=method.decode(batch_base,rr_res,run.keys["vlm_img_lang"][rr],run.proprio[rr])
        after_row=_mse(pred,target[sub[loc]],mask[sub[loc]])
        base_mse=float(baseline_row[loc].mean()); removed_mse=float(after_row.mean())
        relative=(removed_mse-base_mse)/max(base_mse,1e-6); fraction=float(used.mean())
        reports.append({"unit_id":int(unit),"stored_items_using_unit":int(counts[np.where(all_units==unit)[0][0]]),
            "query_usage_fraction":fraction,"baseline_mse_on_used_queries":base_mse,"removed_mse_on_used_queries":removed_mse,
            "delta_mse_on_used_queries":removed_mse-base_mse,"relative_error_increase":relative,
            "usage_weighted_relative_contribution":fraction*relative})
    method.memory=original
    contribution=np.asarray([x["usage_weighted_relative_contribution"] for x in reports],dtype=float)
    return {"units":reports,"redundant_unit_fraction":float(np.mean(contribution<=.01)) if len(contribution) else None,
            "relative_threshold":.01,"baseline_mse":float(baseline_row.mean()),"n_query":len(sub),
            "n_units_total":int(len(all_units)),"n_units_sampled":int(len(units)),
            "unit_sampling":"top 16 and bottom 16 by memory usage when more than 32 units exist",
            "note":"each unit is ablated only on queries whose top-5 retrieved items used it; relative MSE increase is weighted by query usage"}


def _addition_order_sensitivity(method,og_memory,spatial_rows,query_rows,run,key,prior_base,prior_residual,target,mask,seed_value):
    rows=np.asarray(spatial_rows,dtype=np.int64)
    if len(rows)>512: rows=_rng_sample(rows,run.episode_id,512,seed_value+171)
    orders={"temporal":rows,"reverse":rows[::-1],"random":np.random.default_rng(seed_value+991).permutation(rows)}
    original=method.memory; post_state=_growth_state(method); pre_state=getattr(method,"_order_growth_state",{})
    sub=np.arange(min(128,len(query_rows))); qr=query_rows[sub]; results=[]
    for label,order in orders.items():
        _restore_growth_state(method,pre_state); method.memory=copy.deepcopy(og_memory); accepted=0; new_units=0; updated_units=0
        for row in order:
            units_before_row=_unit_count(method)
            ok,reason=_memory_add(method,key[row],method._order_b[row],_meta(run,row))
            accepted+=int(ok)
            new_units+=max(_unit_count(method)-units_before_row,0)
            updated_units+=int("updated" in reason or "resonance_updated" in reason)
        order_merge=None
        if getattr(method,"variant",None)=="merge" and callable(getattr(method,"memory_merge",None)):
            order_merge=method.memory_merge()
        preds=[]
        for i,row in zip(sub,qr):
            cand,w=_query_parts(method,key[row],_meta(run,row))
            b=_blend(method,cand,w) if len(w) else prior_base[i]
            residual=None if prior_residual is None else prior_residual[i:i+1]
            preds.append(_decode_one(method,b,residual,run.keys["vlm_img_lang"][row:row+1],run.proprio[row:row+1]))
        mse=float(_mse(np.stack(preds),target[sub],mask[sub]).mean()) if len(preds) else float("nan")
        results.append({"order":label,"added":accepted,"candidates":len(order),"add_rate":accepted/max(len(order),1),
                        "new_units":new_units,"updated_units":updated_units,"merge":order_merge,"spatial_mse":mse})
    _restore_growth_state(method,post_state); method.memory=original; rates=np.asarray([x["add_rate"] for x in results]); mses=np.asarray([x["spatial_mse"] for x in results])
    return {"orders":results,"add_rate_std":float(rates.std()),"spatial_mse_std":float(mses.std()),"n_query":len(sub)}


def _growth_state(method):
    state={}
    for name in ("prototypes","prototype_counts","merge_history","_prior_code_map","pending_innovations","atom_additions"):
        if hasattr(method,name): state[name]=copy.deepcopy(getattr(method,name))
    if method.__class__.__module__.endswith("methods.m5") and hasattr(method,"net"):
        state["net"]=copy.deepcopy(method.net)
    for name in ("_d3_unit_centers","_d3_last_decision"):
        if hasattr(method,name): state[name]=copy.deepcopy(getattr(method,name))
    if method.__class__.__module__.endswith(("methods.m0","methods.m1")) and getattr(method,"_bench_m6_unitizer",None) is not None:
        state["_bench_m6_unitizer"]=copy.deepcopy(method._bench_m6_unitizer)
    if method.__class__.__module__.endswith("methods.m6"):
        for name in ("_centers","prototypes","support_counts","episode_sets","pending","pending_age",
                     "unit_count_history","unit_registration_events","_pending_since_cluster","_next_pending_id",
                     "_data_seen","registered_count","merged_count","pending_expired_count"):
            if hasattr(method,name):state[name]=copy.deepcopy(getattr(method,name))
    return state


def _restore_growth_state(method,state):
    for name,value in state.items(): setattr(method,name,copy.deepcopy(value))


def _independence_interventions(method,b_query,r_query,train_b,query_rows,run,target,mask,seed_value,own_mse):
    n=min(128,len(query_rows)); rows=query_rows[:n]; b=_decoder_base(method,_base_slice(b_query,slice(0,n))); r=None if r_query is None else np.asarray(r_query[:n])
    pred=method.decode(b,r,run.keys["vlm_img_lang"][rows],run.proprio[rows])
    before=after=None
    if r is not None:
        perm=np.random.default_rng(seed_value+71).permutation(n)
        swapped=method.decode(b,r[perm],run.keys["vlm_img_lang"][rows],run.proprio[rows])
        centers=_unit_centers(method,train_b)
        ep,_=method.encode(pred,run.proprio[rows]); es,_=method.encode(swapped,run.proprio[rows])
        if centers is None and np.asarray(ep).ndim>1 and not isinstance(ep,dict):
            _,ids=np.unique(np.concatenate([ep,es]),axis=0,return_inverse=True); before=ids[:n]; after=ids[n:]
        else: before=_unit_ids(method,ep,centers); after=_unit_ids(method,es,centers)
    if isinstance(b,dict) and "alpha" in b:
        alpha=np.asarray(b["alpha"]); zero_base={"alpha":np.zeros_like(alpha)}
        tralpha=np.asarray(train_b["alpha"]); width=alpha.shape[-1]
        if tralpha.shape[-1]<width: tralpha=np.pad(tralpha,((0,0),(0,width-tralpha.shape[-1])))
        random_base={"alpha":tralpha[np.random.default_rng(seed_value+91).integers(len(tralpha),size=n)]}
    else:
        zero_base=np.zeros_like(b); train_feat=_method_decode_base(method,train_b)
        random_base=train_feat[np.random.default_rng(seed_value+91).integers(len(train_feat),size=n)]
    zero_pred=method.decode(zero_base,r,run.keys["vlm_img_lang"][rows],run.proprio[rows])
    random_pred=method.decode(random_base,r,run.keys["vlm_img_lang"][rows],run.proprio[rows])
    zero_mse=float(_mse(zero_pred,target[:n],mask[:n]).mean()); random_mse=float(_mse(random_pred,target[:n],mask[:n]).mean())
    base_mse=float(_mse(pred,target[:n],mask[:n]).mean()); zero_delta=zero_mse-base_mse; random_delta=random_mse-base_mse
    return {"available":True,"residual_available":r is not None,
            "residual_shuffle_base_type_stability":None if before is None else float(np.mean(before==after)),"n":n,
            "full_reconstruction_mse":base_mse,"zero_base_residual_only_mse":float(_mse(zero_pred,target[:n],mask[:n]).mean()),
            "random_base_with_residual_mse":random_mse,"base_zero_error_increase":zero_delta,
            "base_random_error_increase":random_delta,"base_used":bool(zero_delta>.005),
            "base_type":"re-encode intervention output and assign to train-only base unit"}


def _continuous_gaussian_kl_bits(train_values,query_values):
    tr=np.asarray(train_values,dtype=np.float64); q=np.asarray(query_values,dtype=np.float64)
    if tr.ndim==1: tr=tr[:,None]
    if q.ndim==1: q=q[:,None]
    var=np.maximum(tr.var(0),1e-3)
    # KL[N(query_mean, train-fitted diagonal covariance) || N(0,I)].
    return (.5*(q*q+var-np.log(var)-1).sum(1)/math.log(2)).clip(min=0)


def _residual_rate_bits(method,train_residual,query_residual,actions,state):
    if query_residual is None: return np.zeros(len(actions),dtype=np.float64)
    if callable(getattr(method,"residual_kl_bits",None)):
        return np.asarray(method.residual_kl_bits(actions,state),dtype=np.float64)
    return _continuous_gaussian_kl_bits(train_residual,query_residual)


def _redundancy(method,units,query_rows,run,residual,train_unit_values):
    if len(units)<2: return {"pairs":0,"redundant_pair_fraction":None,"mean_pair_action_mse":None,"threshold":0.01}
    base=[]
    unitizer=_m6_unit_context(method)
    for unit in units:
        loc=np.flatnonzero(train_unit_values[0]==unit)
        if len(loc): base.append(train_unit_values[1][loc].mean(0))
        elif unitizer is not None and 0<=int(unit)<len(unitizer._centers):
            base.append(np.asarray(unitizer._centers[int(unit)],dtype=np.float32))
    if not base: return {"pairs":0,"redundant_pair_fraction":None,"mean_pair_action_mse":None,"threshold":0.01}
    n=min(8,len(query_rows)); rows=query_rows[:n]; f=run.keys["vlm_img_lang"][rows]; s=run.proprio[rows]
    rref=None if residual is None else np.repeat(np.asarray(residual[:n]).mean(0,keepdims=True),n,axis=0)
    dec=[]
    for b in base:
        bb=np.repeat(np.asarray(b).reshape(1,-1),n,axis=0)
        pred=method.decode(bb,rref,f,s)
        dec.append(pred.reshape(n,-1).mean(0))
    dec=np.stack(dec); scale=max(float(np.var(np.concatenate([run.target_actions[rows].reshape(n,-1),dec],0))),1e-8)
    raw=((dec[:,None]-dec[None,:])**2).mean(2); d=raw/scale; upper=np.triu_indices(len(dec),1); frac=float(np.mean(d[upper]<.01)) if len(upper[0]) else None
    return {"pairs":int(len(upper[0])),"mean_pair_action_mse":float(raw[upper].mean()) if len(upper[0]) else None,
            "redundant_pair_fraction":frac,"normalized_mse_threshold":.01,"note":"pairwise decoder action MSE with shared proprioception and the same mean Prior residual"}


def _residual_stability(method,base,residual,query_rows,run,seed):
    if residual is None or len(query_rows)<2: return {"available":False,"reason":"method has no residual"}
    n=min(128,len(query_rows)); rng=np.random.default_rng(seed); p=rng.permutation(n)
    rows=query_rows[:n]; r=np.asarray(residual[:n]); rp=r[p]
    if isinstance(base,list) and base and isinstance(base[0],dict):
        b={k:np.stack([np.asarray(v[k]).reshape(-1) for v in base[:n]],axis=0) for k in base[0]}
    else: b=np.asarray(base[:n])
    a=method.decode(b,r,run.keys["vlm_img_lang"][rows],run.proprio[rows]); ap=method.decode(b,rp,run.keys["vlm_img_lang"][rows],run.proprio[rows])
    mask=np.ones((n,a.shape[1]),bool); la=action_motion_labels(a,mask); lb=action_motion_labels(ap,mask)
    return {"available":True,"same_motion_type_fraction":{k:float(np.mean(la[k]==lb[k])) for k in ("direction","rotation","gripper")},"n":n}


def _evaluate_one(method,name,variant,run,train_rows,query_rows,spatial_store_rows,all_train_rows,keys,*,seed_value,mi_permutations=20,mi_bootstrap=100,sequence_queries=None,libero10=None,libero10_note=None):
    timings={}; stage_started=time.perf_counter(); total_started=stage_started
    def lap(label):
        nonlocal stage_started
        now=time.perf_counter(); timings[label]=now-stage_started; stage_started=now
    fitrows=train_rows
    train={"actions":run.target_actions[fitrows],"state":run.proprio[fitrows],"feature":run.keys["vlm_img_lang"][fitrows],
           "key":keys[fitrows],"episode_id":run.episode_id[fitrows],"init_state_id":run.init_state_id[fitrows],
           "step":run.step[fitrows],"task":run.task[fitrows],"horizon":run.horizon,"seed":seed_value,
           "device":str(DEVICE),"epochs":variant.get("epochs"),"batch_size":variant.get("batch_size"),
           "early_stopping_patience":variant.get("early_stopping_patience",3)}
    method.fit(train)
    if method.__class__.__module__.endswith(("methods.m4","methods.m5")):
        method.tau,method.d_ref=_calibrate_cosine_key_threshold(keys[fitrows],run.episode_id[fitrows],run.init_state_id[fitrows],quantile=.95)
    method._fit_growth_state=_growth_state(method)
    lap("training")
    b_train,r_train=method.encode(run.target_actions[fitrows],run.proprio[fitrows])
    if method.__class__.__module__.endswith(("methods.m0","methods.m1")):
        method._bench_m6_unitizer=_new_m6_evaluator(method,_method_decode_base(method,b_train),r_train,
            run.proprio[fitrows],run.keys["vlm_img_lang"][fitrows],run.episode_id[fitrows],seed=seed_value)
    method._sequence_fit_coverage_base=_unquantized_base(method,run.target_actions[fitrows],run.proprio[fitrows],b_train)
    if method.__class__.__module__.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq")):
        _initialize_standard_units(method,b_train)
    b_query,r_query=method.encode(run.target_actions[query_rows],run.proprio[query_rows])
    b_spatial,r_spatial=method.encode(run.target_actions[spatial_store_rows],run.proprio[spatial_store_rows])
    lap("encoding")
    sequence_data={}
    if sequence_queries is not None:
        fit_lookup={int(row):i for i,row in enumerate(fitrows)}
        query_lookup={int(row):i for i,row in enumerate(query_rows)}
        store_lookup={int(row):i for i,row in enumerate(spatial_store_rows)}
        train_suites=getattr(method,"_train_suites",("libero_object","libero_goal"))
        new_suite=getattr(method,"_new_suite","libero_spatial")
        suite_short={"libero_object":"object","libero_goal":"goal","libero_spatial":"spatial"}
        for suite in train_suites:
            short=suite_short[suite]
            store_rows=np.asarray([r for r in fitrows if run.suite[r]==suite],dtype=np.int64)
            store_ix=np.asarray([fit_lookup[int(r)] for r in store_rows],dtype=np.int64)
            qrows=np.asarray(sequence_queries.get(suite,[]),dtype=np.int64)
            qbase,qres=method.encode(run.target_actions[qrows],run.proprio[qrows]) if len(qrows) else (None,None)
            sequence_data[short]={"store":_seq_table(run,store_rows,keys[store_rows],_base_slice(b_train,store_ix),None if r_train is None else r_train[store_ix],method._sequence_fit_coverage_base[store_ix]),
                                  "query":_seq_table(run,qrows,keys[qrows],qbase,qres,_unquantized_base(method,run.target_actions[qrows],run.proprio[qrows],qbase)) if len(qrows) else None}
        spatial_store_ix=np.asarray([store_lookup[int(r)] for r in spatial_store_rows],dtype=np.int64)
        spatial_query_ix=np.asarray([query_lookup[int(r)] for r in query_rows],dtype=np.int64)
        raw_spatial=_unquantized_base(method,run.target_actions[spatial_store_rows],run.proprio[spatial_store_rows],b_spatial)
        raw_query=_unquantized_base(method,run.target_actions[query_rows],run.proprio[query_rows],b_query)
        new_short=suite_short[new_suite]
        sequence_data[new_short]={"store":_seq_table(run,spatial_store_rows,keys[spatial_store_rows],b_spatial,r_spatial,raw_spatial),
                                  "query":_seq_table(run,query_rows,keys[query_rows],_base_slice(b_query,spatial_query_ix),None if r_query is None else r_query[spatial_query_ix],raw_query[spatial_query_ix])}
        if libero10 is not None:
            data=libero10["arrays"]; lrows=libero10["rows"]
            lkeys=libero10["keymap"].get("combined_base_wrist_proprio") if keys.ndim>1 else None
            # The active key-space is selected from the corresponding train-fitted encoder.
            key_name=getattr(method,"_active_key_name","combined_base_wrist_proprio")
            lkeys=libero10["keymap"].get(key_name,libero10["keymap"].get("combined_base_wrist_proprio"))
            actions=np.asarray(data["action_chunk"])[lrows]; states=np.asarray(data["proprio"])[lrows]
            lb,lr=method.encode(actions,states); lraw=_unquantized_base(method,actions,states,lb)
            eligible=np.asarray(data["action_mask"])[lrows].all(1)&np.asarray(data["motion_mask"])[lrows].all(1)
            store_sel=np.flatnonzero(eligible & (np.asarray(data["split"])[lrows]=="store"))
            query_sel=np.flatnonzero(eligible & (np.asarray(data["split"])[lrows]=="query"))
            query_eps=set(map(str,np.asarray(data["episode_id"])[lrows][query_sel]))
            query_init=set(map(str,np.asarray(data["init_state_id"])[lrows][query_sel]))
            store_sel=np.asarray([i for i in store_sel if str(np.asarray(data["episode_id"])[lrows][i]) not in query_eps
                                  and str(np.asarray(data["init_state_id"])[lrows][i]) not in query_init],dtype=np.int64)
            def l10_table(sel):
                return {"rows":np.asarray(sel),"key":np.asarray(lkeys)[sel],"base":_base_slice(lb,sel),"coverage_base":lraw[sel],"residual":None if lr is None else np.asarray(lr)[sel],
                    "feature":np.asarray(data["f"])[lrows][sel].astype(np.float32),"state":states[sel],"actions":actions[sel],
                    "action_mask":np.asarray(data["action_mask"])[lrows][sel],"motion":np.asarray(data["motion"])[lrows][sel],
                    "episode_id":np.asarray(data["episode_id"])[lrows][sel],"init_state_id":np.asarray(data["init_state_id"])[lrows][sel],
                    "task":np.asarray(data["task"])[lrows][sel],"suite":np.full(len(sel),"libero10",dtype=object),"step":np.asarray(data["step"])[lrows][sel]}
            sequence_data["libero10"]={"store":l10_table(store_sel),"query":l10_table(query_sel)}
    method._sequence_data=sequence_data
    method._sequence_fit_base=b_train
    method._sequence_fit_residual=r_train
    lap("sequence_table_preparation")
    method._order_b={int(row):_base_row(b_spatial,j) for j,row in enumerate(spatial_store_rows)}
    # Seed training-suite memory without novelty rejection, then snapshot it before held-out suite stream.
    for j,row in enumerate(fitrows): _memory_add(method,keys[row],_base_row(b_train,j),{**_meta(run,row),"seed":True})
    # Unit centers/prototypes may have been initialized during encoding or
    # training-memory seeding. Snapshot them now, before the ordinary held-out
    # evaluation below mutates the method, so sequential replay starts from the
    # exact train-only representation.
    method._fit_growth_state=_growth_state(method)
    lap("object_goal_memory_seed")
    mem_og=copy.deepcopy(method.memory)
    method._order_growth_state=_growth_state(method)
    og_checkrows=_rng_sample(fitrows,run.episode_id,min(len(fitrows),1024),seed_value+901)
    _,og_pre_res=method.prior(run.keys["vlm_img_lang"][og_checkrows],run.proprio[og_checkrows])
    before_cands,before_weights=_query_many_parts(method,keys[og_checkrows],[_meta(run,row) for row in og_checkrows])
    before_pred=np.zeros_like(run.target_actions[og_checkrows]); before_ix=[]; before_base=[]
    for i,(cand,w) in enumerate(zip(before_cands,before_weights)):
        if len(w): before_ix.append(i); before_base.append(_blend(method,cand,w))
    if before_ix:
        ix=np.asarray(before_ix,dtype=np.int64); b=_stack_base_values(before_base)
        rr=None if og_pre_res is None else og_pre_res[ix]
        before_pred[ix]=method.decode(b,rr,run.keys["vlm_img_lang"][og_checkrows[ix]],run.proprio[og_checkrows[ix]])
    before_mse=_mse(before_pred,run.target_actions[og_checkrows],run.target_mask[og_checkrows])
    top1_og=[]; blend_og=[]
    query_metas=[_meta(run,row) for row in query_rows]
    og_cands,og_weights=_query_many_parts(method,keys[query_rows],query_metas)
    for cand,w in zip(og_cands,og_weights):
        if not len(w): top1_og.append(None);blend_og.append(None)
        else:
            one={k:np.asarray(v)[:1] for k,v in cand.items()} if isinstance(cand,dict) else np.asarray(cand)[:1]
            top1_og.append(_blend(method,one,[1.]));blend_og.append(_blend(method,cand,w))
    q_prior,r_prior=method.prior(run.keys["vlm_img_lang"][query_rows],run.proprio[query_rows])
    q_prior_base=method.prior_base(q_prior) if hasattr(method,"prior_base") else q_prior
    q_prior_decode=_decoder_base(method,q_prior_base)
    q_own_base=_decoder_base(method,b_query)
    prior_pred=method.decode(q_prior_decode,r_prior,run.keys["vlm_img_lang"][query_rows],run.proprio[query_rows])
    own_pred=method.decode(q_own_base,r_query,run.keys["vlm_img_lang"][query_rows],run.proprio[query_rows])
    target=run.target_actions[query_rows]; mask=run.target_mask[query_rows]
    # M0 has no residual; all other methods retrieve a base and use the Prior residual.
    og1=prior_pred.copy();og5=prior_pred.copy(); valid=np.asarray([i for i,x in enumerate(top1_og) if x is not None],dtype=np.int64)
    if len(valid):
        og1[valid]=method.decode(_stack_base_values([top1_og[i] for i in valid]),None if r_prior is None else r_prior[valid],
            run.keys["vlm_img_lang"][query_rows[valid]],run.proprio[query_rows[valid]])
        og5[valid]=method.decode(_stack_base_values([blend_og[i] for i in valid]),None if r_prior is None else r_prior[valid],
            run.keys["vlm_img_lang"][query_rows[valid]],run.proprio[query_rows[valid]])
    lap("object_goal_retrieval_and_decode")
    # Add spatial memory without gradient; record reason and keep the OG snapshot.
    added=0; reasons={}; units_before_spatial=_unit_count(method)
    if method.__class__.__module__.endswith(("methods.m0","methods.m1")):
        accepted_mask=np.zeros(len(spatial_store_rows),dtype=bool)
        for j,row in enumerate(spatial_store_rows):
            ok,reason=_memory_add(method,keys[row],_base_row(b_spatial,j),_meta(run,row))
            accepted_mask[j]=ok; reasons[reason]=reasons.get(reason,0)+1
        added=int(accepted_mask.sum())
    elif method.__class__.__module__.endswith(("methods.m2","methods.fsq")):
        bases=[_base_row(b_spatial,j) for j in range(len(spatial_store_rows))]
        accepted,reasons=_memory_add_many_standard(method,keys[spatial_store_rows],bases,[_meta(run,row) for row in spatial_store_rows])
        added=int(accepted.sum())
    else:
        for j,row in enumerate(spatial_store_rows):
            ok,reason=_memory_add(method,keys[row],_base_row(b_spatial,j),_meta(run,row)); added+=int(ok); reasons[reason]=reasons.get(reason,0)+1
    merge_report=None
    if getattr(method,"variant",None)=="merge" and callable(getattr(method,"memory_merge",None)):
        merge_report=method.memory_merge()
    spatial_new_units=max(_unit_count(method)-units_before_spatial,0)
    if method.__class__.__module__.endswith("methods.m4"):
        # M4 merge can add prototypes and then merge them in the same pass;
        # the net prototype-count change is not the number of additions.
        spatial_new_units=int(reasons.get("new_prototype",0))
    lap("spatial_memory_add")
    top1_add=[]; blend_add=[]
    add_cands,add_weights=_query_many_parts(method,keys[query_rows],query_metas)
    for cand,w in zip(add_cands,add_weights):
        if not len(w): top1_add.append(None);blend_add.append(None)
        else:
            one={k:np.asarray(v)[:1] for k,v in cand.items()} if isinstance(cand,dict) else np.asarray(cand)[:1]
            top1_add.append(_blend(method,one,[1.]));blend_add.append(_blend(method,cand,w))
    add1=prior_pred.copy();add5=prior_pred.copy(); valid=np.asarray([i for i,x in enumerate(top1_add) if x is not None],dtype=np.int64)
    if len(valid):
        add1[valid]=method.decode(_stack_base_values([top1_add[i] for i in valid]),None if r_prior is None else r_prior[valid],
            run.keys["vlm_img_lang"][query_rows[valid]],run.proprio[query_rows[valid]])
        add5[valid]=method.decode(_stack_base_values([blend_add[i] for i in valid]),None if r_prior is None else r_prior[valid],
            run.keys["vlm_img_lang"][query_rows[valid]],run.proprio[query_rows[valid]])
    lap("spatial_retrieval_and_decode")
    # Re-evaluate object/goal queries after the spatial stream and any explicit merge.
    og_prior,og_res=method.prior(run.keys["vlm_img_lang"][og_checkrows],run.proprio[og_checkrows])
    if hasattr(method,"prior_base"): og_prior=method.prior_base(og_prior)
    og_prior=_decoder_base(method,og_prior)
    og_res=np.asarray(og_res) if og_res is not None else None
    post_cands,post_weights=_query_many_parts(method,keys[og_checkrows],[_meta(run,row) for row in og_checkrows])
    after_pred=np.zeros_like(run.target_actions[og_checkrows]); post_ix=[];post_base=[]
    for i,(cand,w) in enumerate(zip(post_cands,post_weights)):
        if len(w): post_ix.append(i);post_base.append(_blend(method,cand,w))
    if post_ix:
        ix=np.asarray(post_ix,dtype=np.int64); after_pred[ix]=method.decode(_stack_base_values(post_base),None if og_res is None else og_res[ix],
            run.keys["vlm_img_lang"][og_checkrows[ix]],run.proprio[og_checkrows[ix]])
    after_mse=_mse(after_pred,run.target_actions[og_checkrows],run.target_mask[og_checkrows])
    lap("object_goal_post_add_recheck")
    # Base information, conditional information, usage, redundancy and residual invariance.
    base_q=_method_decode_base(method,b_query)
    source=f"bench_{name}_{seed_value}_{run.horizon}".replace("-","_")
    original_usable=run.usable.copy(); run.usable[:]=False; run.usable[fitrows]=True;run.usable[query_rows]=True
    source_values=np.zeros((len(run),np.asarray(base_q).shape[-1]),dtype=np.float32)
    source_values[fitrows]=_method_decode_base(method,b_train).astype(np.float32)
    source_values[query_rows]=np.asarray(base_q,dtype=np.float32)
    run.z[source]=source_values
    qmask=np.zeros(len(run),bool);qmask[query_rows]=True; smask=np.zeros(len(run),bool);smask[fitrows]=True
    mi=evaluate_mi(run,qmask,smask,MIConfig(cluster_counts=(8,16,32),primary_clusters=16,n_permutations=mi_permutations,bootstrap_samples=mi_bootstrap,seed=seed_value,minimum_groups=10),sources=(source,))
    del run.z[source];run.usable[:]=original_usable
    lap("mutual_information_bootstrap")
    unit_train,unit_query,units=_unit_metrics(method,b_train,b_query,r_prior)
    qids=np.asarray(unit_query,dtype=np.int64).reshape(-1); qids=qids[qids>=0]
    counts=np.bincount(qids,minlength=int(np.max(units)+1) if len(units) else 1); p=counts[counts>0]/max(counts.sum(),1)
    usage={"n_units_used":int(np.count_nonzero(counts)),"perplexity":float(np.exp(-(p*np.log(np.maximum(p,1e-12))).sum())),"total_units":int(len(units))}
    hsic=None if r_prior is None else _rbd_hsic(base_q,r_prior)
    stability=_residual_stability(method,[v for v in blend_add if v is not None],r_prior,query_rows,run,seed_value) if r_prior is not None else {"available":False,"reason":"no residual"}
    redundancy=_redundancy(method,units,query_rows,run,r_prior,(unit_train,_method_decode_base(method,b_train)))
    mse={"own":float(_mse(own_pred,target,mask).mean()),"prior":float(_mse(prior_pred,target,mask).mean()),
         "object_goal_top1":float(_mse(og1,target,mask).mean()),"object_goal_top5":float(_mse(og5,target,mask).mean()),
         "spatial_added_top1":float(_mse(add1,target,mask).mean()),"spatial_added_top5":float(_mse(add5,target,mask).mean()),
         "object_goal_after_add":float(after_mse.mean()),"object_goal_before_add":float(before_mse.mean())}
    desc=method.describe()
    if hasattr(method,"kl_rate_bits"):
        per_window=np.asarray(method.kl_rate_bits(target,run.proprio[query_rows],run.keys["vlm_img_lang"][query_rows]))
        base_rate=None; residual_rate=None; rate_kind="conditional KL(q||p), converted to bits"
    elif isinstance(b_train,dict) and "codes" in b_train:
        used=len(np.unique(np.asarray(b_train["codes"]).reshape(len(b_train["codes"]),-1),axis=0))
        base_rate=float(math.log2(max(used,1)))
        residual_bits=_residual_rate_bits(method,r_train,r_query,target,run.proprio[query_rows])
        residual_rate=float(residual_bits.mean()) if r_train is not None else 0.0
        per_window=base_rate+residual_bits
        rate_kind="log2(train-used base codes) + residual KL(q||N(0,I)) in bits"
    elif isinstance(b_train,dict) and "alpha" in b_train:
        used=int(np.count_nonzero(np.any(np.asarray(b_train["alpha"])>0,axis=0)))
        base_rate=float(math.log2(max(used,1)))
        residual_bits=_residual_rate_bits(method,r_train,r_query,target,run.proprio[query_rows])
        residual_rate=float(residual_bits.mean()) if r_train is not None else 0.0
        per_window=base_rate+residual_bits
        rate_kind="log2(train-used dictionary atoms) + residual KL(q||N(0,I)) in bits"
    elif np.issubdtype(np.asarray(b_train).dtype,np.integer):
        train_codes=np.asarray(b_train).reshape(len(b_train),-1)
        used=len(np.unique(train_codes,axis=0)); base_rate=float(math.log2(max(used,1)))
        residual_bits=_residual_rate_bits(method,r_train,r_query,target,run.proprio[query_rows])
        residual_rate=float(residual_bits.mean()) if r_train is not None else 0.0
        per_window=base_rate+residual_bits
        rate_kind="log2(train-used discrete base units) + residual KL(q||N(0,I)) in bits"
    else:
        base_train=_method_decode_base(method,b_train); base_query=_method_decode_base(method,b_query)
        base_bits=_continuous_gaussian_kl_bits(base_train,base_query)
        residual_bits=_residual_rate_bits(method,r_train,r_query,target,run.proprio[query_rows])
        base_rate=float(base_bits.mean()); residual_rate=float(residual_bits.mean()) if r_train is not None else 0.0
        per_window=base_bits+residual_bits
        rate_kind="train-fitted diagonal Gaussian KL bits for continuous base + residual KL(q||N(0,I))"
    lap("usage_hsic_rate_distortion_core_metrics")
    rate_mse=float(np.mean(_mse(add5,target,mask)))
    rate_distortion={"estimated_bits_per_window":float(np.mean(per_window)),"base_bits":base_rate,"residual_bits":residual_rate,
                     "rate_kind":rate_kind,"spatial_added_top5_mse":rate_mse}
    # Evaluate new diagnostics on the spatial query split; no query motion labels affect updates.
    functional=_functional_unit_ablation(method,query_rows,run,keys,q_prior_decode,r_prior,add5,target,mask,b_train)
    lap("functional_unit_ablation")
    order=_addition_order_sensitivity(method,mem_og,spatial_store_rows,query_rows,run,keys,q_prior_base,r_prior,target,mask,seed_value)
    lap("addition_order_sensitivity")
    interventions=_independence_interventions(method,b_query,r_query,b_train,query_rows,run,target,mask,seed_value,float(_mse(own_pred,target,mask).mean()))
    lap("residual_interventions")
    updates={k:desc.get(k,None) for k in ("neural_weights_fixed_on_memory_add","memory_representatives_change_on_add","old_ids_change_on_add")}
    updates["neural_weights_fixed_on_memory_add"]=True if updates["neural_weights_fixed_on_memory_add"] is None else updates["neural_weights_fixed_on_memory_add"]
    if updates["memory_representatives_change_on_add"] is None:
        updates["memory_representatives_change_on_add"]=bool(getattr(method,"update_prototypes",False))
    if updates["old_ids_change_on_add"] is None:
        updates["old_ids_change_on_add"]=bool(getattr(method,"merge_history",[]))
    result={"method":name,"horizon":run.horizon,"seed":seed_value,"n_train":len(fitrows),"n_query":len(query_rows),"n_spatial_store":len(spatial_store_rows),
       "training_epochs":{"configured":getattr(method,"epochs",None),"actual":getattr(method,"epochs_run",None),
           "decoder_alignment":getattr(method,"decoder_epochs_run",None),"prior":getattr(method,"prior_epochs_run",None),
           "early_stopping_patience":variant.get("early_stopping_patience",3)},
       "description":desc,"mse":mse,"spatial_addition":{"added":added,"memory_items_added":added,"new_skill_units":spatial_new_units,
            "candidates":len(spatial_store_rows),"memory_item_rate":added/max(len(spatial_store_rows),1),
            "rate":spatial_new_units/max(len(spatial_store_rows),1),"unit_add_rate":spatial_new_units/max(len(spatial_store_rows),1),
            "reasons":reasons,"memory_size_after":len(method.memory),"merge":merge_report},
       "base_motion_mi":mi["table"][source]["16"],"base_usage":usage,"hsic_base_residual":hsic,"residual_motion_type_stability":stability,
       "redundancy":redundancy,"functional_unit_ablation":functional,"addition_order_sensitivity":order,
       "independence_interventions":interventions,"memory_update_properties":updates,"rate_distortion":rate_distortion}
    if sequence_queries is not None:
        result["sequential_protocol"]=_sequence_protocol(method,b_train,sequence_data,seed_value=seed_value,libero10_note=libero10_note)
        if libero10 is not None:
            result["libero10_camera_language_availability"]=libero10.get("camera_language_availability",{})
    lap("sequential_protocol")
    timings["total_seconds"]=time.perf_counter()-total_started
    timings["unattributed_seconds"]=max(timings["total_seconds"]-sum(v for k,v in timings.items() if k!="total_seconds"),0.0)
    result["timing_seconds"]=timings
    return result


def _seq_table(run,rows,keys,base,residual,coverage_base=None):
    rows=np.asarray(rows,dtype=np.int64)
    return {"rows":rows,"key":np.asarray(keys),"base":base,
            "coverage_base":None if coverage_base is None else np.asarray(coverage_base,dtype=np.float32),
            "residual":None if residual is None else np.asarray(residual),
            "feature":run.keys["vlm_img_lang"][rows],"state":run.proprio[rows],
            "actions":run.target_actions[rows],"action_mask":run.target_mask[rows],
            "motion":run.motion[rows],"episode_id":run.episode_id[rows],
            "init_state_id":run.init_state_id[rows],"task":run.task[rows],
            "suite":run.suite[rows],"task_index":run.task_index[rows],"step":run.step[rows]}


def _seq_meta(table,i):
    return {"episode_id":str(table["episode_id"][i]),"init_state_id":str(table["init_state_id"][i]),
            "task":str(table["task"][i]),"suite":str(table["suite"][i]),"step":int(table["step"][i]),
            "segment_id":f"{table['episode_id'][i]}@{int(table['step'][i])}"}


def _seq_predict(method,table):
    n=len(table["step"])
    prior_dist,prior_res=method.prior(table["feature"],table["state"])
    prior_base=method.prior_base(prior_dist) if hasattr(method,"prior_base") else prior_dist
    prior_decode=_decoder_base(method,prior_base)
    prior_pred=method.decode(prior_decode,prior_res,table["feature"],table["state"])
    own_pred=method.decode(_decoder_base(method,table["base"]),table["residual"],table["feature"],table["state"])
    metas=[_seq_meta(table,i) for i in range(n)]
    if method.__class__.__module__.endswith("methods.m6"):
        item_bases,skill_bases=_m6_retrieve_many(method,table,metas,prior_decode)
        item_pred=method.decode(item_bases,prior_res,table["feature"],table["state"])
        skill_pred=method.decode(skill_bases,prior_res,table["feature"],table["state"])
        return {"prior_pred":prior_pred,"own_pred":own_pred,"memory_pred":item_pred,
                "memory_item_pred":item_pred,"memory_skill_pred":skill_pred,
                "prior_base":prior_decode,"prior_residual":prior_res,
                "novelty_score":_novelty_scores_many(method,table["key"],table["base"],metas)}
    candidates,weights=_sequence_retrieve(method,table,metas); bases=[]
    for i,(cand,w) in enumerate(zip(candidates,weights)):
        bases.append(_blend(method,cand,w) if len(w) else _base_row(prior_decode,i))
    batch_base=_stack_base_values(bases)
    memory_pred=method.decode(batch_base,prior_res,table["feature"],table["state"])
    return {"prior_pred":prior_pred,"own_pred":own_pred,"memory_pred":memory_pred,
            "prior_base":prior_decode,"prior_residual":prior_res,
            "novelty_score":_novelty_scores_many(method,table["key"],table["base"],metas)}


def _m6_retrieve_many(method,table,metas,fallback_base):
    """Return the documented item and registered-skill softmax blends."""
    entries=method.memory; n=len(metas); dim=method.base_dim
    fallback=np.asarray(fallback_base,dtype=np.float32)
    item_out=fallback.copy(); skill_out=fallback.copy()
    if not entries:return item_out,skill_out
    keys=np.stack([np.asarray(e["key"],dtype=np.float32) for e in entries])
    bases=np.stack([np.asarray(e["base"],dtype=np.float32).reshape(-1) for e in entries])
    ids=np.asarray([-1 if e.get("unit_id") is None else int(e["unit_id"]) for e in entries],dtype=np.int64)
    eps=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in entries],dtype=object)
    init=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in entries],dtype=object)
    q=np.asarray(table["key"],dtype=np.float32).reshape(n,-1)
    d2=np.maximum((q*q).sum(1)[:,None]+(keys*keys).sum(1)[None,:]-2*q@keys.T,0.)
    dist=np.sqrt(d2)
    for i,meta in enumerate(metas):
        eligible=(eps!=meta.get("episode_id",""))&(init!=meta.get("init_state_id",""))
        ix=np.flatnonzero(eligible)
        if not len(ix):continue
        chosen=ix[np.argsort(dist[i,ix],kind="stable")[:min(method.topk,len(ix))]]
        w=np.exp(-dist[i,chosen]/max(float(method.temperature),1e-12)); w/=max(float(w.sum()),1e-12)
        item_out[i]=w@bases[chosen]
        registered=ids[chosen]>=0
        if registered.any():
            units=ids[chosen[registered]]; uw=w[registered]
            skill=np.zeros(dim,np.float32)
            for uid,weight in zip(units,uw):skill+=float(weight)*np.asarray(method.prototypes[int(uid)],dtype=np.float32)
            skill_out[i]=skill
    return item_out,skill_out


def _sequence_retrieve(method,table,metas):
    """Update per-query top-k incrementally as memory items are appended by suite."""
    module=method.__class__.__module__; entries=method.memory; n=len(table["step"]); size=len(entries)
    cache=getattr(method,"_sequence_retrieval_cache",None)
    key=(id(table),id(entries))
    state=cache.get(key) if cache is not None else None
    cosine=module.endswith(("methods.m3_rvq","methods.m4","methods.m5"))
    k=1 if module.endswith(("methods.m4","methods.m5")) else 5 if module.endswith("methods.m3_rvq") else int(getattr(method,"topk",5))
    query=np.asarray(table["key"],dtype=np.float32).reshape(n,-1)
    if state is None or state["seen"]>size or len(state.get("indices",[]))!=n:
        chosen=[np.zeros(0,dtype=np.int64) for _ in range(n)]; values=[np.zeros(0,dtype=np.float32) for _ in range(n)]; seen=0
    else:
        chosen=state["indices"]; values=state["values"]; seen=state["seen"]
    if size>seen:
        new_entries=entries[seen:size]; newkeys=np.stack([np.asarray(e["key"],dtype=np.float32) for e in new_entries])
        if cosine:
            kn=newkeys/np.maximum(np.linalg.norm(newkeys,axis=1,keepdims=True),1e-8)
            qn=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
            new_values=qn@kn.T
        else:
            new_values=np.sqrt(np.maximum((query*query).sum(1)[:,None]+(newkeys*newkeys).sum(1)[None,:]-2*query@newkeys.T,0.0))
        episodes=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in new_entries],dtype=object)
        initials=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in new_entries],dtype=object)
        for i,meta in enumerate(metas):
            valid=(episodes!=meta.get("episode_id",""))&(initials!=meta.get("init_state_id","")); newix=np.flatnonzero(valid)+seen
            cand_ix=np.concatenate([chosen[i],newix]); cand_val=np.concatenate([values[i],new_values[i,valid].astype(np.float32)])
            if not len(cand_ix):chosen[i]=cand_ix;values[i]=cand_val;continue
            order=np.argsort(-cand_val if cosine else cand_val,kind="stable")[:min(k,len(cand_ix))]
            chosen[i]=cand_ix[order];values[i]=cand_val[order]
        seen=size
    if cache is None: cache={};method._sequence_retrieval_cache=cache
    cache[key]={"seen":size,"indices":chosen,"values":values}
    candidates=[];weights=[]
    for ix,val in zip(chosen,values):
        if not len(ix):candidates.append(np.zeros((0,),np.float32));weights.append(np.zeros((0,),np.float32));continue
        if module.endswith("methods.m3_rvq"):
            logits=val/max(float(getattr(method,"retrieval_temperature",.1)),1e-6);w=np.exp(logits-logits.max());w/=w.sum()
            cand=np.asarray([[entries[j]["coarse_id"],entries[j]["fine_id"]] for j in ix],dtype=np.int64)
        elif module.endswith("methods.m4"):
            w=np.ones(1,dtype=np.float32);cand=np.asarray(method.prototypes)[np.asarray([entries[int(ix[0])]["code"]])]
        elif module.endswith("methods.m5"):
            w=np.ones(1,dtype=np.float32);width=int(method.net.dictionary.shape[0]);a=np.asarray(entries[int(ix[0])]["alpha"],dtype=np.float32)
            if len(a)<width:a=np.pad(a,(0,width-len(a)))
            cand={"alpha":a[None]}
        else:
            temp=float(getattr(method,"temperature",.5));ref=getattr(method,"d_ref",None);scale=max(temp*(float(ref) if ref else max(float(val[0]),1e-12)),1e-12)
            w=np.exp(-(val-val[0])/scale);w/=w.sum();cand=np.stack([np.asarray(entries[int(j)]["base"]) for j in ix])
        candidates.append(cand);weights.append(np.asarray(w,dtype=np.float32))
    return candidates,weights


def _novelty_score(method,key,base,meta=None):
    """Higher score means more novel; mirror each method's memory-add rule where available."""
    if method.__class__.__module__.endswith("methods.m5") and isinstance(base,dict) and "innovation" in base:
        return float(np.linalg.norm(np.asarray(base["innovation"])))
    if hasattr(method,"prototypes") and len(method.prototypes):
        emb=_method_decode_base(method,base).reshape(-1)
        sim=np.asarray(method.prototypes)@emb/max(float(np.linalg.norm(emb)),1e-8)
        return float(1.0-sim.max())
    q=np.asarray(key,dtype=np.float64).reshape(-1); distances=[]
    for entry in getattr(method,"memory",[]):
        emeta=entry.get("meta") or {}
        if meta and (emeta.get("episode_id")==meta.get("episode_id") or emeta.get("init_state_id")==meta.get("init_state_id")): continue
        if "key" in entry: distances.append(float(np.linalg.norm(q-np.asarray(entry["key"]).reshape(-1))))
    return float(min(distances)) if distances else float("inf")


def _novelty_scores_many(method,keys,bases,metas):
    module=method.__class__.__module__; n=len(keys)
    if module.endswith(("methods.m0","methods.m1","methods.m6")):
        unitizer=_m6_unit_context(method)
        if unitizer is not None:
            _,distances,_=_m6_ids(unitizer,_method_decode_base(method,bases),allow_pending=False)
            return distances
    if module.endswith(("methods.m2","methods.fsq")) and hasattr(method,"_d3_unit_centers"):
        features=_method_decode_base(method,bases).astype(np.float32)
        return _base_distance(features,method._d3_unit_centers)
    if module.endswith("methods.m3_rvq") and len(getattr(method,"train_base_features",[])):
        features=_method_decode_base(method,bases).astype(np.float32)
        return _base_distance(features,method.train_base_features)
    if module.endswith("methods.m5") and isinstance(bases,dict) and "innovation" in bases:
        return np.linalg.norm(np.asarray(bases["innovation"],dtype=np.float32).reshape(n,-1),axis=1)
    if module.endswith("methods.m4") and hasattr(method,"prototypes") and len(method.prototypes):
        emb=_method_decode_base(method,bases).astype(np.float32)
        return _base_distance(emb,np.asarray(method.prototypes,dtype=np.float32))
    return _memory_key_novelty_many(method,keys,metas,cosine=False)


def _memory_key_novelty_many(method,keys,metas,*,cosine=False):
    memory=getattr(method,"memory",[]); n=len(keys)
    if not memory:return np.full(n,np.inf,dtype=np.float32)
    stored=np.stack([np.asarray(e["key"],dtype=np.float32) for e in memory]); episodes=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in memory],dtype=object)
    initials=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in memory],dtype=object); q=np.asarray(keys,dtype=np.float32).reshape(n,-1)
    if cosine: stored=stored/np.maximum(np.linalg.norm(stored,axis=1,keepdims=True),1e-8)
    result=np.full(n,np.inf,dtype=np.float32)
    for lo in range(0,n,128):
        hi=min(n,lo+128); qb=q[lo:hi]
        if cosine:
            qb=qb/np.maximum(np.linalg.norm(qb,axis=1,keepdims=True),1e-8); d=1.0-qb@stored.T
        else:
            d2=np.maximum((qb*qb).sum(1)[:,None]+(stored*stored).sum(1)[None,:]-2*qb@stored.T,0.0); d=np.sqrt(d2)
        for j,meta in enumerate(metas[lo:hi]):
            if meta is not None:d[j,(episodes==meta.get("episode_id",""))|(initials==meta.get("init_state_id",""))]=np.inf
        result[lo:hi]=d.min(1)
    return result


def _auroc(new_scores,known_scores):
    a=np.asarray(new_scores,dtype=float); b=np.asarray(known_scores,dtype=float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return None
    return float((np.sum(a[:,None]>b[None,:])+.5*np.sum(a[:,None]==b[None,:]))/(len(a)*len(b)))


def _binary_auroc(scores, labels):
    scores=np.asarray(scores,dtype=float); labels=np.asarray(labels,dtype=bool)
    return _auroc(scores[labels],scores[~labels])


def _motion_novelty_reference(tables):
    motions=np.concatenate([np.asarray(t["motion"],dtype=np.float32).reshape(len(t["step"]),-1) for t in tables if len(t["step"])])
    eps=np.concatenate([np.asarray(t["episode_id"],dtype=object) for t in tables if len(t["step"])])
    initials=np.concatenate([np.asarray(t["init_state_id"],dtype=object) for t in tables if len(t["step"])])
    # Deterministic cap keeps this evaluator-only distance calculation bounded.
    if len(motions)>4096:
        take=np.linspace(0,len(motions)-1,4096,dtype=int); motions,eps,initials=motions[take],eps[take],initials[take]
    nearest=np.full(len(motions),np.inf,dtype=np.float64); norms=np.sum(motions.astype(np.float64)**2,axis=1)
    for lo in range(0,len(motions),256):
        hi=min(len(motions),lo+256)
        d2=np.maximum(norms[lo:hi,None]+norms[None,:]-2*motions[lo:hi]@motions.T,0)
        invalid=(eps[lo:hi,None]==eps[None,:])|(initials[lo:hi,None]==initials[None,:])
        d2[invalid]=np.inf; nearest[lo:hi]=np.sqrt(d2.min(axis=1))
    finite=nearest[np.isfinite(nearest)]
    quantiles={q:float(np.quantile(finite,q)) if len(finite) else float("inf") for q in (.90,.95,.99)}
    return motions,eps,initials,quantiles


def _is_new_unit_event(method, incoming, row, units_before):
    segment=_seq_meta(incoming,row)["segment_id"]
    entry=next((e for e in reversed(getattr(method,"memory",[]))
                if (e.get("meta") or {}).get("segment_id")==segment),None)
    if entry is None:return False
    meta=entry.get("meta") or {}; module=method.__class__.__module__
    if module.endswith("methods.m4"):return int(entry.get("code",-1))>=units_before
    if module.endswith("methods.m5"):
        alpha=np.asarray(entry.get("alpha",[]),dtype=np.float32)
        return bool(len(alpha)>units_before and np.any(np.abs(alpha[units_before:])>1e-8))
    if module.endswith("methods.m3_rvq"):return bool(entry.get("is_new_unit",False))
    return bool(meta.get("skill_unit_added",False))


def _seed_sequence_stage(method, incoming, indices, stage_label):
    """Seed fitted-suite memory exactly like the ordinary post-fit memory seed."""
    reasons={}; accepted=[]
    for i in indices:
        meta=_seq_meta(incoming,int(i)); meta.update({"seed":True,"skill_unit_added":False,"sequential_stage":stage_label})
        ok,reason=_memory_add(method,incoming["key"][i],_base_row(incoming["base"],int(i)),meta)
        reasons[reason]=reasons.get(reason,0)+1
        if ok: accepted.append(int(i))
    return np.asarray(accepted,dtype=np.int64),reasons


def _sequence_row_identity(meta):
    """Stable identity for matching an appended memory item to its input row."""
    return tuple(str(meta.get(key, "")) for key in ("suite", "episode_id", "init_state_id", "segment_id"))


def _sequence_unit_events(method, incoming, start_index):
    """Return unit-add flags/entries aligned to incoming rows, without ID collisions."""
    identities=[_sequence_row_identity(_seq_meta(incoming,i)) for i in range(len(incoming["step"]))]
    row_by_identity={identity:i for i,identity in enumerate(identities)}
    event_flags=np.zeros(len(identities),dtype=bool); event_entries={}
    if method.__class__.__module__.endswith("methods.m6") or method.__class__.__module__.endswith(("methods.m0","methods.m1")):
        unitizer=_m6_unit_context(method)
        for event in getattr(unitizer,"unit_registration_events",[]):
            row=row_by_identity.get(_sequence_row_identity(event))
            if row is None:continue
            event_flags[row]=True
            event_entries[row]={"meta":dict(event),"base":np.asarray(event.get("base",[]),dtype=np.float32),
                                "unit_id":event.get("unit_id")}
        return event_flags,event_entries
    for entry in method.memory[start_index:]:
        meta=entry.get("meta") or {}
        row=row_by_identity.get(_sequence_row_identity(meta))
        if row is None:
            continue
        added=bool(meta.get("skill_unit_added",entry.get("is_new_unit",False)))
        event_flags[row]=added
        if added:
            event_entries[row]=copy.deepcopy(entry)
    return event_flags,event_entries


def _should_merge_sequence_stage(method, stage_index):
    """Do not merge the fitted seed bank; M4 merge runs after held-out adds."""
    return int(stage_index)>0 and getattr(method,"variant",None)=="merge" and callable(getattr(method,"memory_merge",None))


def _motion_novelty_labels(reference, query_motion):
    motions,eps,initials,_=reference
    q=np.asarray(query_motion,dtype=np.float32).reshape(len(query_motion),-1); norms=np.sum(motions.astype(np.float64)**2,axis=1)
    qnorm=np.sum(q.astype(np.float64)**2,axis=1)
    d2=np.maximum(qnorm[:,None]+norms[None,:]-2*q@motions.T,0)
    return np.sqrt(d2.min(axis=1))


def _memory_base_features(method):
    entries=getattr(method,"memory",[])
    if not entries: return np.zeros((0,1),np.float32)
    values=[]
    for e in entries:
        if "base" in e: values.append(np.asarray(e["base"]))
        elif "code" in e and hasattr(method,"prototypes"): values.append(np.asarray(method.prototypes[e["code"]]))
        elif "coarse_id" in e: values.append(np.asarray([e["coarse_id"],e["fine_id"]],dtype=np.int64))
        elif "alpha" in e:
            al=np.asarray(e["alpha"],dtype=np.float32); d=method.net.dictionary.detach().cpu().numpy(); d=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-8)
            if len(al)<len(d): al=np.pad(al,(0,len(d)-len(al)))
            values.append(al@d)
    arr=np.stack(values)
    if np.issubdtype(arr.dtype,np.integer) and hasattr(method,"base_features"): return np.asarray(method.base_features(arr))
    return arr.astype(np.float32)


def _base_distance(query,stored):
    q=np.asarray(query,dtype=np.float64); s=np.asarray(stored,dtype=np.float64)
    if q.ndim==1:q=q[None]
    if s.ndim==1:s=s[None]
    if q.shape[1]!=s.shape[1]: return np.full(len(q),np.inf)
    d=np.maximum((q*q).sum(1)[:,None]-2*q@s.T+(s*s).sum(1)[None],0)
    return np.sqrt(d.min(1))


def _unit_count(method):
    if method.__class__.__module__.endswith(("methods.m0","methods.m1")) and getattr(method,"_bench_m6_unitizer",None) is not None:
        return int(len(method._bench_m6_unitizer._centers))
    if hasattr(method,"_d3_unit_centers"): return int(len(method._d3_unit_centers))
    if hasattr(method,"prototypes"): return int(len(method.prototypes))
    if hasattr(method,"net") and getattr(method,"net",None) is not None and hasattr(method.net,"dictionary"):
        return int(method.net.dictionary.shape[0])
    entries=getattr(method,"memory",[])
    if not entries: return 0
    signatures=[]
    for e in entries:
        if "coarse_id" in e: sig=(int(e["coarse_id"]),int(e["fine_id"]))
        elif "base" in e:
            a=np.asarray(e["base"]); sig=tuple(a.reshape(-1).tolist()) if np.issubdtype(a.dtype,np.integer) else tuple(np.round(a.reshape(-1),5).tolist())
        elif "code" in e: sig=(int(e["code"]),)
        else: sig=(len(signatures),)
        signatures.append(sig)
    return len(set(signatures))


def _added_unit_example(method,incoming,row,units_before,entry=None):
    segment=_seq_meta(incoming,row).get("segment_id")
    if entry is None:
        entry=next((e for e in reversed(getattr(method,"memory",[]))
                    if (e.get("meta") or {}).get("segment_id")==segment),None)
    if entry is None:return None
    meta=entry.get("meta") or {}; module=method.__class__.__module__
    added=bool(meta.get("skill_unit_added",False))
    unit_id=meta.get("skill_unit_id")
    if module.endswith("methods.m3_rvq"):
        added=bool(entry.get("is_new_unit",False)); unit_id=[entry.get("coarse_id"),entry.get("fine_id")]
    elif module.endswith("methods.m4"):
        unit_id=int(entry.get("code",-1)); added=bool(meta.get("skill_unit_added",unit_id>=units_before))
    elif module.endswith("methods.m5"):
        alpha=np.asarray(entry.get("alpha",[]),dtype=np.float32)
        added=bool(meta.get("skill_unit_added",False))
        unit_id=meta.get("skill_unit_id",[int(i) for i in np.flatnonzero(np.abs(alpha)>1e-8)])
    if not added:return None
    motion=np.asarray(incoming["motion"][row],dtype=np.float32)
    labels=env_motion_labels(motion[None,...])
    label={name:int(np.asarray(labels[name]).reshape(-1)[0]) for name in ("direction","rotation","gripper")}
    finite=np.isfinite(motion).all(-1)
    usable=motion[finite]
    dpos=np.nansum(usable[:,:3],axis=0).tolist() if len(usable) else None
    grip=None
    if len(usable) and usable.shape[-1]>7:
        grip=float(np.nansum(usable[-1,7:])-np.nansum(usable[0,7:]))
    return {"suite":str(incoming["suite"][row]),"task":str(incoming["task"][row]),
            "segment_id":segment,"unit_id":unit_id,"motion_labels":label,
            "motion_features":{"sum_delta_xyz":dpos,"gripper_gap_delta":grip}}


def _seq_slice(table,indices):
    out={}
    for k,v in table.items():
        if isinstance(v,dict): out[k]={name:np.asarray(value)[indices] for name,value in v.items()}
        elif isinstance(v,np.ndarray) and len(v)==len(table["step"]): out[k]=v[indices]
        else: out[k]=v
    return out


def _motion_unit_accuracy(method,store_tables,query,fit_base):
    stores=[t for t in store_tables if len(t["step"])]
    if not stores or not len(query["step"]): return None
    motions=np.concatenate([t["motion"].reshape(len(t["step"]),-1) for t in stores])
    store_b=_concat_base([t["base"] for t in stores])
    qmotion=query["motion"].reshape(len(query["step"]),-1)
    if method.__class__.__module__.endswith(("methods.m0","methods.m1","methods.m6")):
        labels=_unit_metrics(method,fit_base,store_b,None)[1]
        qlabels=_unit_metrics(method,fit_base,query["base"],None)[1]
    elif method.__class__.__module__.endswith("methods.m4"):
        labels=_coverage_representatives(method,store_b,_method_decode_base(method,store_b))[0]
        qlabels=_coverage_representatives(method,query["base"],_method_decode_base(method,query["base"]))[0]
    else:
        labels,qlabels,_=_unit_metrics(method,store_b,query["base"],None)
    good=np.isfinite(motions).all(1); qgood=np.isfinite(qmotion).all(1)
    if not good.any() or not qgood.any(): return None
    rng=np.random.default_rng(9041)
    good_ix=np.flatnonzero(good); qgood_ix=np.flatnonzero(qgood)
    if len(good_ix)>1024: good_ix=np.sort(rng.choice(good_ix,1024,replace=False))
    if len(qgood_ix)>1024: qgood_ix=np.sort(rng.choice(qgood_ix,1024,replace=False))
    good=np.zeros(len(good),bool);good[good_ix]=True;qgood=np.zeros(len(qgood),bool);qgood[qgood_ix]=True
    mean=motions[good].mean(0); scale=motions[good].std(0); scale[scale<1e-6]=1
    a=(motions[good]-mean)/scale; b=(qmotion[qgood]-mean)/scale
    dist=np.maximum((b*b).sum(1)[:,None]-2*b@a.T+(a*a).sum(1)[None],0)
    predicted=labels[good][dist.argmin(1)]
    return {"accuracy":float(np.mean(predicted==qlabels[qgood])),"chance":float(1/max(len(np.unique(labels[good])),1)),
            "n_query":int(qgood.sum()),"n_reference":int(good.sum()),"sample_cap_per_side":1024,
            "classifier":"1-nearest-neighbor in standardized environment motion feature"}


def _concat_base(parts):
    if isinstance(parts[0],dict): return {k:np.concatenate([np.asarray(p[k]) for p in parts],axis=0) for k in parts[0]}
    return np.concatenate([np.asarray(p) for p in parts],axis=0)


def _memory_decoder_base(method,entry):
    if "base" in entry: return entry["base"]
    if "code" in entry and hasattr(method,"prototypes"): return np.asarray(method.prototypes[entry["code"]])
    if "coarse_id" in entry: return np.asarray([[entry["coarse_id"],entry["fine_id"]]],dtype=np.int64)
    if "alpha" in entry: return {"alpha":np.asarray(entry["alpha"],dtype=np.float32)[None]}
    return np.zeros(1,np.float32)


def _unit_separation(method,table,prior_residual,fit_base,own_pred):
    """R2 separation and functional overlap using shared query residual/context."""
    if not len(table["step"]): return {"separation_ratio":None,"redundant_pair_fraction":None,"pairs":0}
    if method.__class__.__module__.endswith("methods.m4"):
        fit_ids,_=_coverage_representatives(method,fit_base,_method_decode_base(method,fit_base))
        q_ids,_=_coverage_representatives(method,table["base"],_method_decode_base(method,table["base"]))
        units=np.unique(fit_ids)
    elif isinstance(fit_base,dict) and "alpha" in fit_base:
        fit_ids,_,units=_unit_metrics(method,fit_base,fit_base,None)
        q_ids=_unit_metrics(method,fit_base,table["base"],prior_residual)[1]
    else:
        fit_ids,_,units=_unit_metrics(method,fit_base,fit_base,None)
        q_ids=_unit_metrics(method,fit_base,table["base"],prior_residual)[1]
    centers=_unit_centers(method,fit_base)
    base_rows=[]
    if method.__class__.__module__.endswith("methods.m4"):
        base_rows=[np.asarray(method.prototypes)[int(unit)] for unit in units]
    elif isinstance(fit_base,dict) and "alpha" in fit_base:
        for unit in units:
            idx=int(np.flatnonzero(np.asarray(fit_ids).reshape(-1)==unit)[0])
            base_rows.append(_base_row(fit_base,idx))
    else:
        for unit in units:
            if centers is not None:
                # Open-ended M6 can register evaluation-time units that have
                # no row in the original fit split; their stored functional
                # center is the authoritative representative.
                base_rows.append(centers[int(unit)])
                continue
            idx=int(np.flatnonzero(np.asarray(fit_ids).reshape(-1)==unit)[0])
            base_rows.append(_base_row(fit_base,idx))
    if len(base_rows)<2:
        return {"separation_ratio":None,"redundant_pair_fraction":None,"pairs":0,
                "between_unit_action_mse":None,"within_unit_action_mse":None}
    rng=np.random.default_rng(287)
    qn=min(24,len(table["step"])); qix=np.sort(rng.choice(len(table["step"]),qn,replace=False)) if qn<len(table["step"]) else np.arange(qn)
    un=min(32,len(base_rows)); uix=np.sort(rng.choice(len(base_rows),un,replace=False)) if un<len(base_rows) else np.arange(un)
    features=table["feature"][qix]; states=table["state"][qix]
    residual=None if prior_residual is None else np.asarray(prior_residual)[qix]
    def repeated(base):
        if isinstance(base,dict):
            return {k:(np.repeat(np.asarray(v)[:1],qn,axis=0) if np.asarray(v).ndim else np.full(qn,v)) for k,v in base.items()}
        a=np.asarray(base)
        return np.repeat(a.reshape((1,)+a.shape) if a.ndim else a.reshape(1),qn,axis=0)
    actions=[]
    for i in uix:
        pred=method.decode(repeated(base_rows[int(i)]),residual,features,states)
        actions.append(np.asarray(pred,dtype=np.float32).reshape(qn,-1))
    acts=np.stack(actions,axis=0)
    pair_dist=[]
    for i in range(len(acts)):
        for j in range(i+1,len(acts)):
            pair_dist.append(float(np.mean((acts[i]-acts[j])**2)))
    between=float(np.mean(pair_dist)) if pair_dist else float("nan")
    own=np.asarray(own_pred,dtype=np.float32).reshape(len(table["step"]),-1)[qix]
    selected=np.asarray(q_ids).reshape(-1)[qix]
    within=[]
    for unit in np.unique(selected):
        rows=np.flatnonzero(selected==unit)
        if len(rows)<2: continue
        if len(rows)>16: rows=np.sort(rng.choice(rows,16,replace=False))
        xx=own[rows]
        ii,jj=np.triu_indices(len(xx),1)
        within.extend(np.mean((xx[ii]-xx[jj])**2,axis=1).tolist())
    within_mean=float(np.mean(within)) if within else float("nan")
    if not np.isfinite(within_mean) or within_mean<=1e-12:
        ratio=None; redundant=None
    else:
        ratio=between/within_mean
        redundant=float(np.mean(np.asarray(pair_dist)<=within_mean))
    return {"separation_ratio":ratio,"redundant_pair_fraction":redundant,
            "between_unit_action_mse":between,"within_unit_action_mse":within_mean,
            "pairs":len(pair_dist),"unit_sample_cap":32,"query_context_sample_cap":24,
            "note":"mean between-unit action MSE divided by within-unit MSE; all unit pairs share the same Prior residual and query state/context"}


def _coverage_outside(method,fit_base,query_base,fit_raw,query_raw):
    unitizer=_m6_unit_context(method)
    if unitizer is not None:
        _,fit_distance,_=_m6_ids(unitizer,fit_raw,allow_pending=False)
        _,query_distance,_=_m6_ids(unitizer,query_raw,allow_pending=False)
        fit_distance=fit_distance[np.isfinite(fit_distance)]
        threshold=float(np.quantile(fit_distance,.95)) if len(fit_distance) else float("inf")
        return {"outside_rate":float(np.mean(query_distance>threshold)) if len(query_distance) else None,
                "threshold_train_q95":threshold,"mean_query_distance":float(np.mean(query_distance)) if len(query_distance) else None,
                "n_fit":len(fit_distance),"n_query":len(query_distance),
                "space":"M6 functional distance to open-ended unit; train q95 threshold"}
    _,fit_rep=_coverage_representatives(method,fit_base,fit_raw)
    _,query_rep=_coverage_representatives(method,query_base,query_raw)
    fit_distance=np.linalg.norm(np.asarray(fit_raw)-fit_rep,axis=1)
    query_distance=np.linalg.norm(np.asarray(query_raw)-query_rep,axis=1)
    threshold=float(np.quantile(fit_distance,.95)) if len(fit_distance) else 0.0
    return {"outside_rate":float(np.mean(query_distance>threshold)) if len(query_distance) else None,
            "threshold_train_q95":threshold,"mean_query_distance":float(np.mean(query_distance)) if len(query_distance) else None,
            "n_fit":len(fit_distance),"n_query":len(query_distance),
            "space":"pre-quantization encoder base to its assigned quantized/prototype representative"}


def _seq_suite_metrics(method,query,fit_base,store_tables):
    if not len(query["step"]): return {"n_query":0,"available":False}
    pred=_seq_predict(method,query); target=query["actions"]; mask=query["action_mask"]
    names=("own","prior","memory")
    if "memory_skill_pred" in pred:names=(*names,"memory_item","memory_skill")
    mse={name:float(_mse(pred.get(f"{name}_pred",pred["memory_pred"]),target,mask).mean()) for name in names}
    unit_train,unit_query,units=_unit_metrics(method,fit_base,query["base"],pred["prior_residual"])
    qids=np.asarray(unit_query,dtype=np.int64).reshape(-1);qids=qids[qids>=0]
    counts=np.bincount(qids,minlength=max(int(np.max(units))+1 if len(units) else 1,1))
    used=np.unique(unit_query,axis=0).shape[0] if np.asarray(unit_query).ndim>1 else int(np.count_nonzero(counts))
    fit_raw=np.asarray(getattr(method,"_sequence_fit_coverage_base",_method_decode_base(method,fit_base)),dtype=np.float32)
    query_raw=np.asarray(query.get("coverage_base",_method_decode_base(method,query["base"])),dtype=np.float32)
    q_feat=_method_decode_base(method,query["base"])
    coverage=_coverage_outside(method,fit_base,query["base"],fit_raw,query_raw)
    if hasattr(method,"kl_rate_bits"):
        rate=float(np.mean(method.kl_rate_bits(target,query["state"],query["feature"])))
    else:
        train_r=getattr(method,"_sequence_fit_residual",None); query_r=query.get("residual")
        residual_bits=_residual_rate_bits(method,train_r,query_r,target,query["state"])
        if isinstance(query["base"],dict) and "alpha" in query["base"]:
            alpha=np.asarray(getattr(method,"_sequence_fit_base",{}).get("alpha",[]))
            active=int(np.count_nonzero(np.any(alpha>0,axis=0))) if alpha.size else 1
            base_rate=math.log2(max(active,1))
        elif np.issubdtype(np.asarray(query["base"]).dtype,np.integer):
            codes=np.asarray(getattr(method,"_sequence_fit_base",fit_base)).reshape(len(fit_base),-1)
            base_rate=math.log2(max(len(np.unique(codes,axis=0)),1))
        else:
            base_rate=float(_continuous_gaussian_kl_bits(_method_decode_base(method,fit_base),q_feat).mean())
        rate=float(base_rate+residual_bits.mean())
    motion_acc=_motion_unit_accuracy(method,store_tables,query,fit_base)
    # Independent functional redundancy uses the same Prior residual and current active memory.
    n_diag=min(32,len(query["step"])); qdiag=_seq_slice(query,np.arange(n_diag))
    virtual=SimpleNamespace(keys={"vlm_img_lang":qdiag["feature"]},proprio=qdiag["state"],episode_id=qdiag["episode_id"],
        init_state_id=qdiag["init_state_id"],task=qdiag["task"],suite=qdiag["suite"],step=qdiag["step"])
    function=_functional_unit_ablation(method,np.arange(n_diag),virtual,qdiag["key"],pred["prior_base"][:n_diag] if not isinstance(pred["prior_base"],dict) else _base_slice(pred["prior_base"],slice(0,n_diag)),
        None if pred["prior_residual"] is None else pred["prior_residual"][:n_diag],pred["memory_pred"][:n_diag],qdiag["actions"],qdiag["action_mask"],fit_base)
    separation=_unit_separation(method,qdiag,pred["prior_residual"],fit_base,pred["own_pred"][:n_diag])
    return {"available":True,"n_query":len(query["step"]),"reconstruction_mse":mse,
            "units":{"used":int(used),"memory_units":_unit_count(method),"perplexity":float(np.exp(-(counts[counts>0]/max(counts.sum(),1)*np.log(np.maximum(counts[counts>0]/max(counts.sum(),1),1e-12))).sum()))},
            "coverage":{"outside_train_base_radius_rate":coverage["outside_rate"],"threshold_train_q95":coverage["threshold_train_q95"],"mean_query_distance":coverage["mean_query_distance"],"space":coverage["space"]},
            "rate_distortion":{"estimated_bits_per_window":rate,"memory_reconstruction_mse":mse["memory"]},
            "R2":{"unit_action_separation":separation,"redundant_unit_fraction":separation.get("redundant_pair_fraction"),"functional_ablation_redundancy_diagnostic":function.get("redundant_unit_fraction"),"motion_to_unit_accuracy":motion_acc},
            "_prediction":pred}


def _concat_seq_tables(tables):
    tables=[t for t in tables if t is not None and len(t["step"])]
    if not tables:return None
    out={}
    for key in tables[0]:
        values=[t[key] for t in tables]
        if isinstance(values[0],dict): out[key]={k:np.concatenate([v[k] for v in values],axis=0) for k in values[0]}
        elif isinstance(values[0],np.ndarray): out[key]=np.concatenate(values,axis=0)
        elif values[0] is None: out[key]=None
        else: out[key]=values[0]
    return out


def _seq_mse(method,table):
    if table is None or not len(table["step"]): return float("nan")
    p=_seq_predict(method,table)["memory_pred"]
    return float(_mse(p,table["actions"],table["action_mask"]).mean())


def _nearest_memory_items(method,key,meta,top_k=5):
    entries=getattr(method,"memory",[])
    if not entries:return []
    stored=np.stack([np.asarray(e["key"],dtype=np.float32) for e in entries]); q=np.asarray(key,dtype=np.float32).reshape(-1)
    eps=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in entries],dtype=object)
    init=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in entries],dtype=object)
    eligible=(eps!=meta.get("episode_id",""))&(init!=meta.get("init_state_id","")); ix=np.flatnonzero(eligible)
    if not len(ix):return []
    module=method.__class__.__module__
    if module.endswith(("methods.m3_rvq","methods.m4","methods.m5")):
        kn=stored[ix]/np.maximum(np.linalg.norm(stored[ix],axis=1,keepdims=True),1e-8); qn=q/max(float(np.linalg.norm(q)),1e-8)
        sim=kn@qn; order=np.argsort(-sim,kind="stable")[:min(top_k,len(ix))]
    else:
        d2=np.maximum((stored[ix]*stored[ix]).sum(1)+float(q@q)-2*stored[ix]@q,0.0); order=np.argsort(d2,kind="stable")[:min(top_k,len(ix))]
    return [entries[i] for i in ix[order]]


def _nearest_memory_items_many(method,keys,metas,top_k=5):
    entries=getattr(method,"memory",[]); n=len(keys)
    if not entries:return [[] for _ in range(n)]
    stored=np.stack([np.asarray(e["key"],dtype=np.float32) for e in entries]); eps=np.asarray([(e.get("meta") or {}).get("episode_id","") for e in entries],dtype=object)
    init=np.asarray([(e.get("meta") or {}).get("init_state_id","") for e in entries],dtype=object); q=np.asarray(keys,dtype=np.float32).reshape(n,-1)
    module=method.__class__.__module__; cosine=module.endswith(("methods.m3_rvq","methods.m4","methods.m5"))
    if cosine: stored=stored/np.maximum(np.linalg.norm(stored,axis=1,keepdims=True),1e-8)
    output=[]
    for lo in range(0,n,128):
        hi=min(n,lo+128); qb=q[lo:hi]
        if cosine: scores=(qb/np.maximum(np.linalg.norm(qb,axis=1,keepdims=True),1e-8))@stored.T
        else: scores=-np.maximum((qb*qb).sum(1)[:,None]+(stored*stored).sum(1)[None,:]-2*qb@stored.T,0.0)
        for i,meta in enumerate(metas[lo:hi]):
            eligible=(eps!=meta.get("episode_id",""))&(init!=meta.get("init_state_id","")); ix=np.flatnonzero(eligible)
            if not len(ix):output.append([]);continue
            k=1 if module.endswith(("methods.m4","methods.m5")) else min(top_k,int(getattr(method,"topk",top_k))) if not module.endswith("methods.m3_rvq") else top_k
            chosen=ix[np.argsort(-scores[i,ix],kind="stable")[:min(k,len(ix))]]
            output.append([entries[j] for j in chosen])
    return output


def _sequence_stage(method,sequence_data,fit_base,active,stage_label,order_name,seed_value,data_windows_seen=None):
    suite_metrics={};timing={}
    store_tables=[sequence_data[s]["store"] for s in active if s in sequence_data]
    for suite in active:
        table=sequence_data[suite]["query"] if suite in sequence_data else None
        if table is None: continue
        started=time.perf_counter()
        result=_seq_suite_metrics(method,table,fit_base,store_tables)
        timing[suite]=time.perf_counter()-started
        result.pop("_prediction",None)
        suite_metrics[suite]=result
    unitizer=_m6_unit_context(method)
    pending=len(getattr(unitizer,"pending",[])) if unitizer is not None else None
    seen=(int(data_windows_seen) if data_windows_seen is not None else int(getattr(unitizer,"_data_seen",0))) if unitizer is not None else None
    return {"stage":stage_label,"order":order_name,"suites_evaluated":suite_metrics,"unit_count":_unit_count(method),
            "data_seen":seen,"unit_count_per_1000_windows":None if not seen else 1000*_unit_count(method)/seen,
            "pending_count":pending,"pending_fraction":None if not seen else pending/max(seen,1),
            "suite_metric_seconds":timing}


def _sequence_protocol(method,fit_base,sequence_data,*,seed_value,libero10_note=None):
    protocol_started=time.perf_counter()
    train_short=[{"libero_object":"object","libero_goal":"goal","libero_spatial":"spatial"}[s]
                 for s in getattr(method,"_train_suites",("libero_object","libero_goal"))]
    new_short={"libero_object":"object","libero_goal":"goal","libero_spatial":"spatial"}.get(
        getattr(method,"_new_suite","libero_spatial"),"spatial")
    available=[s for s in (*train_short,new_short,"libero10") if s in sequence_data]
    if not all(s in available for s in (*train_short,new_short)):
        return {"available":False,"reason":"selected train and held-out suite tables are required","libero10_note":libero10_note}
    for suite in available:
        store=sequence_data[suite]["store"]; query=sequence_data[suite]["query"]
        if query is None: continue
        for identity in ("episode_id","init_state_id"):
            overlap=set(map(str,store[identity])) & set(map(str,query[identity]))
            if overlap: raise AssertionError(f"{suite}: sequential store/query leakage in {identity}: {len(overlap)} IDs")
    train_tables=[sequence_data[s]["store"] for s in train_short if s in sequence_data]
    motion_reference=_motion_novelty_reference(train_tables)
    _,_,_,motion_q=motion_reference
    novelty_parts=[np.asarray(_novelty_scores_many(method,t["key"],t["base"],[_seq_meta(t,i) for i in range(len(t["step"]))]),dtype=float).reshape(-1)
                   for t in train_tables if len(t["step"])]
    novelty_calibration=np.concatenate(novelty_parts) if novelty_parts else np.zeros(0)
    novelty_q={q:float(np.quantile(novelty_calibration[np.isfinite(novelty_calibration)],q))
               if np.isfinite(novelty_calibration).any() else float("inf") for q in (.90,.95,.99)}
    train_stage=tuple(train_short)
    if "libero10" in available:
        plans=[(f"{new_short}_then_libero10",[train_stage,(new_short,),("libero10",)])]
    else: plans=[(f"{new_short}_only",[train_stage,(new_short,)])]
    requested_orders=getattr(method,"_sequence_orders",None)
    if requested_orders:
        unknown=set(requested_orders)-{name for name,_ in plans}
        if unknown: raise ValueError(f"unknown sequential order(s): {sorted(unknown)}")
        plans=[plan for plan in plans if plan[0] in requested_orders]
        if not plans: raise ValueError("sequential_orders selected no available plan")
    outputs={}; fit_growth=getattr(method,"_fit_growth_state",{})
    for order_idx,(order_name,stages) in enumerate(plans):
        _restore_growth_state(method,fit_growth); method.memory=[]; method._sequence_retrieval_cache={}
        active=[]; stage_rows=[]; transition_rows=[]; previous_mse={}; forgetting_deltas=[]
        protocol_data_seen=len(fit_base)
        for stage_i,new_suites in enumerate(stages):
            transition_started=time.perf_counter(); transition_timing={}
            stage_label=f"after_{'+'.join(new_suites)}"
            incoming=_concat_seq_tables([sequence_data[s]["store"] for s in new_suites if s in sequence_data])
            if incoming is None: continue
            known_queries=[sequence_data[s]["query"] for s in active if s in sequence_data and sequence_data[s]["query"] is not None]
            known=_concat_seq_tables(known_queries)
            new_scores=_novelty_scores_many(method,incoming["key"],incoming["base"],[_seq_meta(incoming,i) for i in range(len(incoming["step"]))])
            known_scores=(_novelty_scores_many(method,known["key"],known["base"],[_seq_meta(known,i) for i in range(len(known["step"]))]) if known is not None else np.zeros(0))
            auroc=_auroc(new_scores,known_scores)
            # New-motion labels are evaluator-only: they are never passed to fit or memory_add.
            query_new=_concat_seq_tables([sequence_data[s]["query"] for s in new_suites if s in sequence_data])
            task_labels=None; segment_labels=None; segment_distances=None
            query_scores=None
            if query_new is not None:
                query_scores=_novelty_scores_many(method,query_new["key"],query_new["base"],
                    [_seq_meta(query_new,i) for i in range(len(query_new["step"]))])
                if getattr(method,"_split_name",None)=="A" and new_short=="goal" and "goal" in new_suites and "task_index" in query_new:
                    task_labels=np.isin(np.asarray(query_new["task_index"],dtype=int),[12,15,16,19])
                segment_distances=_motion_novelty_labels(motion_reference,query_new["motion"])
                segment_labels=segment_distances>motion_q[.95]
            task_auroc=_binary_auroc(query_scores,task_labels) if task_labels is not None and query_scores is not None else None
            segment_auroc=_binary_auroc(query_scores,segment_labels) if segment_labels is not None and query_scores is not None else None
            transition_timing["novelty_scores_and_auroc"]=time.perf_counter()-transition_started; tick=time.perf_counter()
            false_add=None
            if known is not None and len(known["step"]):
                clone=copy.deepcopy(method); ncheck=min(64,len(known["step"])); accepted=0
                for i in range(ncheck):
                    units_check_before=_unit_count(clone)
                    ok,_reason=_memory_add(clone,known["key"][i],_base_row(known["base"],i),_seq_meta(known,i))
                    if clone.__class__.__module__.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq")):
                        accepted+=int(bool(getattr(clone,"_d3_last_decision",{}).get("skill_unit_added",False)))
                    elif clone.__class__.__module__.endswith("methods.m3_rvq"):
                        accepted+=int(bool(clone.memory and clone.memory[-1].get("is_new_unit",False)))
                    else: accepted+=int(ok and _unit_count(clone)>units_check_before)
                false_add=accepted/max(ncheck,1)
            transition_timing["known_false_add_check"]=time.perf_counter()-tick; tick=time.perf_counter()
            before_mse=_seq_mse(method,_concat_seq_tables([sequence_data[s]["query"] for s in new_suites if s in sequence_data]))
            transition_timing["pre_add_new_suite_error"]=time.perf_counter()-tick; tick=time.perf_counter()
            base_before=_memory_base_features(method); units_before=_unit_count(method); n_before=len(method.memory)
            accepted_rows=[]; reasons={}
            indices=np.arange(len(incoming["step"]))
            if order_name=="random_mixed" and len(new_suites)>1:
                indices=np.random.default_rng(seed_value+1000+stage_i).permutation(indices)
            if stage_i==0:
                # These rows were already used to fit/calibrate the method.
                # They initialize memory as seeds and must not trigger a second
                # novelty decision or mutate the fitted unit set.
                accepted_rows,reasons=_seed_sequence_stage(method,incoming,indices,stage_label)
            elif method.__class__.__module__.endswith(("methods.m0","methods.m1")):
                for i in indices:
                    meta=_seq_meta(incoming,int(i)); meta["sequential_stage"]=stage_label
                    ok,reason=_memory_add(method,incoming["key"][i],_base_row(incoming["base"],int(i)),meta)
                    reasons[reason]=reasons.get(reason,0)+1
                    if ok:accepted_rows.append(int(i))
            elif method.__class__.__module__.endswith(("methods.m2","methods.fsq")):
                metas=[_seq_meta(incoming,i) for i in indices]
                for meta in metas: meta["sequential_stage"]=stage_label
                bases=[_base_row(incoming["base"],i) for i in indices]
                accepted_mask,reasons=_memory_add_many_standard(method,incoming["key"][indices],bases,metas)
                accepted_rows=indices[accepted_mask].astype(int).tolist()
            else:
                for i in indices:
                    meta=_seq_meta(incoming,i); meta["sequential_stage"]=stage_label
                    ok,reason=_memory_add(method,incoming["key"][i],_base_row(incoming["base"],i),meta)
                    reasons[reason]=reasons.get(reason,0)+1
                    if ok: accepted_rows.append(int(i))
            event_flags,event_entries=_sequence_unit_events(method,incoming,n_before)
            merge_report=None
            if _should_merge_sequence_stage(method,stage_i):
                merge_report=method.memory_merge()
            transition_timing["memory_add_and_merge"]=time.perf_counter()-tick; tick=time.perf_counter()
            after_mse=_seq_mse(method,_concat_seq_tables([sequence_data[s]["query"] for s in new_suites if s in sequence_data]))
            transition_timing["post_add_new_suite_error"]=time.perf_counter()-tick; tick=time.perf_counter()
            unit_delta=max(_unit_count(method)-units_before,0)
            module=method.__class__.__module__
            if module.endswith("methods.m4"):
                unit_delta=len({int((event_entries[int(i)].get("meta") or {}).get("skill_unit_id",-1))
                                for i in np.flatnonzero(event_flags) if int((event_entries[int(i)].get("meta") or {}).get("skill_unit_id",-1))>=0})
            elif module.endswith("methods.m5"):
                unit_delta=len({int((event_entries[int(i)].get("meta") or {}).get("skill_unit_id",-1))
                                for i in np.flatnonzero(event_flags) if (event_entries[int(i)].get("meta") or {}).get("skill_unit_id") is not None})
            accepted=len(accepted_rows)
            improvement=None if not np.isfinite(before_mse) or not np.isfinite(after_mse) else before_mse-after_mse
            improvement_per_unit=None if improvement is None or unit_delta==0 else improvement/unit_delta
            old_features=base_before
            duplicates=[]
            for i in np.flatnonzero(event_flags):
                item=event_entries.get(int(i))
                if item is None: continue
                meta_item=item.get("meta") or {}
                is_unit_new=bool(meta_item.get("skill_unit_added",item.get("is_new_unit",False)))
                if method.__class__.__module__.endswith("methods.m4"):
                    is_unit_new=int(item.get("code",-1))>=units_before
                if method.__class__.__module__.endswith("methods.m5"):
                    is_unit_new=bool(_unit_count(method)>units_before and any(a<len(item.get("alpha",[])) and item["alpha"][a]!=0 for a in range(units_before,_unit_count(method))))
                if not is_unit_new: continue
                candidate=_method_decode_base(method,_base_row(incoming["base"],i)).reshape(1,-1)
                duplicates.append(bool(len(old_features) and _base_distance(candidate,old_features)[0]<=1e-4))
            added_ids=set()
            if method.__class__.__module__.endswith(("methods.m0","methods.m1","methods.m2","methods.fsq")):
                for i in np.flatnonzero(event_flags):
                    item=event_entries.get(int(i))
                    if item and (item.get("meta") or {}).get("skill_unit_added"):
                        added_ids.add(int(item["meta"].get("skill_unit_id",-1)))
            else:
                if method.__class__.__module__.endswith("methods.m3_rvq"):
                    added_ids={(int(e.get("coarse_id",-1)),int(e.get("fine_id",-1))) for e in method.memory if e.get("is_new_unit",False)}
                elif method.__class__.__module__.endswith("methods.m4"):
                    added_ids={int((event_entries[int(i)].get("meta") or {}).get("skill_unit_id",-1))
                               for i in np.flatnonzero(event_flags)}
                    if merge_report and merge_report.get("old_to_new"):
                        mapping=merge_report["old_to_new"]
                        added_ids={int(mapping[i]) for i in added_ids if 0<=i<len(mapping)}
                elif method.__class__.__module__.endswith("methods.m5"):
                    added_ids={int((event_entries[int(i)].get("meta") or {}).get("skill_unit_id",-1))
                               for i in np.flatnonzero(event_flags)}
            query_new=_concat_seq_tables([sequence_data[s]["query"] for s in new_suites if s in sequence_data])
            reuse=[]
            if query_new is not None:
                if added_ids:
                    got_many=_nearest_memory_items_many(method,query_new["key"],[_seq_meta(query_new,i) for i in range(len(query_new["step"]))],top_k=5)
                    for got in got_many:
                        found=False
                        for e in got:
                            meta=e.get("meta") or {}
                            if meta.get("skill_unit_id") is not None and int(meta["skill_unit_id"]) in added_ids: found=True
                            elif "coarse_id" in e and (int(e["coarse_id"]),int(e.get("fine_id",-1))) in added_ids: found=True
                            elif "code" in e and int(e["code"]) in added_ids: found=True
                            elif "alpha" in e and any(a<len(e["alpha"]) and e["alpha"][a]!=0 for a in added_ids): found=True
                        reuse.append(found)
                else:
                    # No novel unit can be reused; avoid a full top-k pass over
                    # the large seeded training memory just to rediscover False.
                    reuse=[False]*len(query_new["step"])
            transition_timing["duplicate_and_reuse_metrics"]=time.perf_counter()-tick; tick=time.perf_counter()
            unit_event_rate=unit_delta/max(len(incoming["step"]),1)
            item_event_rate=accepted/max(len(incoming["step"]),1)
            unit_event=event_flags
            unit_examples=[]
            for row in np.flatnonzero(unit_event):
                example=_added_unit_example(method,incoming,int(row),units_before,event_entries.get(int(row)))
                if example is not None:
                    task_i=int(incoming.get("task_index",np.full(len(incoming["step"]),-1))[row])
                    example["task_motion_label"]=("new-motion" if getattr(method,"_split_name",None)=="A" and new_short=="goal" and "goal" in new_suites and task_i in (12,15,16,19) else
                                                   "known-motion" if getattr(method,"_split_name",None)=="A" and new_short=="goal" and "goal" in new_suites else "not-defined")
                    dist=_motion_novelty_labels(motion_reference,incoming["motion"][row:row+1])[0]
                    example["segment_motion_label"]="new-motion" if dist>motion_q[.95] else "known-motion"
                    example["segment_motion_distance"]=float(dist)
                if example is not None and len(unit_examples)<5: unit_examples.append(example)
            incoming_motion_distance=_motion_novelty_labels(motion_reference,incoming["motion"])
            incoming_segment_new=incoming_motion_distance>motion_q[.95]
            incoming_task_new=(np.isin(np.asarray(incoming.get("task_index",np.full(len(incoming["step"]),-1)),dtype=int),[12,15,16,19])
                               if getattr(method,"_split_name",None)=="A" and new_short=="goal" and "goal" in new_suites else None)
            add_rates={str(q):{"unit_novelty_threshold":novelty_q[q],
                "new_motion_unit_add_rate":float((new_scores>novelty_q[q])[incoming_segment_new].mean()) if np.any(incoming_segment_new) else None,
                "known_motion_unit_add_rate":float((new_scores>novelty_q[q])[~incoming_segment_new].mean()) if np.any(~incoming_segment_new) else None,
                "new_motion_unit_candidate_rate":float((new_scores>novelty_q[q])[incoming_segment_new].mean()) if np.any(incoming_segment_new) else None,
                "known_motion_unit_candidate_rate":float((new_scores>novelty_q[q])[~incoming_segment_new].mean()) if np.any(~incoming_segment_new) else None,
                "segment_label_motion_threshold_q95":motion_q[.95],
                "task_new_motion_unit_add_rate":float((new_scores>novelty_q[q])[incoming_task_new].mean()) if incoming_task_new is not None and np.any(incoming_task_new) else None,
                "task_known_motion_unit_add_rate":float((new_scores>novelty_q[q])[~incoming_task_new].mean()) if incoming_task_new is not None and np.any(~incoming_task_new) else None,
                "task_new_motion_unit_candidate_rate":float((new_scores>novelty_q[q])[incoming_task_new].mean()) if incoming_task_new is not None and np.any(incoming_task_new) else None,
                "task_known_motion_unit_candidate_rate":float((new_scores>novelty_q[q])[~incoming_task_new].mean()) if incoming_task_new is not None and np.any(~incoming_task_new) else None,
                "query_unit_novelty_threshold":novelty_q[q],
                "query_score_above_novelty_threshold_rate":float(np.mean(query_scores>novelty_q[q])) if query_scores is not None else None} for q in motion_q}
            transition_rows.append({"stage":stage_label,"new_suites":list(new_suites),"novelty_auroc":auroc,
                "task_new_motion_auroc":task_auroc,"segment_new_motion_auroc":segment_auroc,
                "task_new_motion_rate":None if incoming_task_new is None else float(np.mean(incoming_task_new)),
                "segment_new_motion_rate":float(np.mean(incoming_segment_new)),"unit_add_rate_by_motion":{
                    "new_motion":float(unit_event[incoming_segment_new].mean()) if np.any(incoming_segment_new) else None,
                    "known_motion":float(unit_event[~incoming_segment_new].mean()) if np.any(~incoming_segment_new) else None},
                "threshold_add_rates":add_rates,
                "known_false_add_rate":false_add,"stream_rows":len(incoming["step"]),"accepted_memory_items":accepted,
                "memory_item_add_rate":item_event_rate,"new_skill_units":unit_delta,"unit_add_rate":unit_event_rate,
                "pending_count_after_stage":len(getattr(_m6_unit_context(method),"pending",[])) if _m6_unit_context(method) is not None else None,
                "pending_fraction_of_stage_rows":(len(getattr(_m6_unit_context(method),"pending",[]))/max(len(incoming["step"]),1)) if _m6_unit_context(method) is not None else None,
                "accept_rate":unit_event_rate,"unit_count_before":units_before,
                "unit_count_after":_unit_count(method),"new_units":unit_delta,"unit_growth_per_input":unit_delta/max(len(incoming["step"]),1),
                "new_units_per_1000_windows":1000.0*unit_delta/max(len(incoming["step"]),1),"requires_new_motion_unit":new_short in new_suites,
                "new_unit_examples":unit_examples,"improvement_mse":improvement,
                "improvement_per_new_unit_or_item":improvement_per_unit,"new_unit_duplicate_fraction":float(np.mean(duplicates)) if duplicates else None,
                "new_units_reused_by_query_rate":float(np.mean(reuse)) if reuse else None,"merge":merge_report,"reasons":reasons})
            active.extend([s for s in new_suites if s not in active])
            if stage_i>0: protocol_data_seen+=len(incoming["step"])
            stage_result=_sequence_stage(method,sequence_data,fit_base,active,stage_label,order_name,seed_value,protocol_data_seen)
            transition_timing["all_seen_suite_stage_metrics"]=time.perf_counter()-tick
            transition_rows[-1]["timing_seconds"]=transition_timing
            for suite,item in stage_result["suites_evaluated"].items():
                current=item.get("reconstruction_mse",{}).get("memory")
                previous=previous_mse.get(suite)
                delta=None if previous is None or current is None else current-previous
                item["R1_forgetting_delta_from_previous_stage"]=delta
                if delta is not None: forgetting_deltas.append(delta)
                if current is not None: previous_mse[suite]=current
            stage_rows.append(stage_result)
        final=stage_rows[-1]["suites_evaluated"] if stage_rows else {}
        forget=max(forgetting_deltas,default=0.0)
        red=[v.get("R2",{}).get("redundant_unit_fraction") for v in final.values()]
        red=[x for x in red if x is not None]
        sep=[v.get("R2",{}).get("unit_action_separation",{}).get("separation_ratio") for v in final.values()]
        sep=[x for x in sep if x is not None]
        acc=[v.get("R2",{}).get("motion_to_unit_accuracy",{}).get("accuracy") for v in final.values()]
        acc=[x for x in acc if x is not None]
        chance=[v.get("R2",{}).get("motion_to_unit_accuracy",{}).get("chance") for v in final.values()]
        chance=[x for x in chance if x is not None]
        cover=[v.get("coverage",{}).get("outside_train_base_radius_rate") for v in final.values()]
        cover=[x for x in cover if x is not None]
        auc=[x.get("novelty_auroc") for x in transition_rows if x.get("novelty_auroc") is not None]
        task_auc=[x.get("task_new_motion_auroc") for x in transition_rows if x.get("task_new_motion_auroc") is not None]
        segment_auc=[x.get("segment_new_motion_auroc") for x in transition_rows if x.get("segment_new_motion_auroc") is not None]
        false=[x.get("known_false_add_rate") for x in transition_rows if x.get("known_false_add_rate") is not None]
        improves=[x.get("improvement_mse") for x in transition_rows if x.get("improvement_mse") is not None]
        reuse=[x.get("new_units_reused_by_query_rate") for x in transition_rows if x.get("new_units_reused_by_query_rate") is not None]
        dup=[x.get("new_unit_duplicate_fraction") for x in transition_rows if x.get("new_unit_duplicate_fraction") is not None]
        growth=[x.get("new_units_per_1000_windows") for x in transition_rows if x.get("new_units_per_1000_windows") is not None]
        novel_stages=[x for x in transition_rows if x.get("requires_new_motion_unit")]
        motion_growth=any((x.get("unit_add_rate_by_motion",{}).get("new_motion") or 0)>0 and
                          (x.get("unit_add_rate_by_motion",{}).get("known_motion") is None or
                           x["unit_add_rate_by_motion"]["known_motion"] < x["unit_add_rate_by_motion"]["new_motion"])
                          for x in novel_stages)
        checks={"R1":{"max_forgetting_delta":forget,"forgetting_threshold_mse":.002,
                       "unit_growth_per_1000_windows_by_stage":growth,"new_motion_growth_required":True,
                       "higher_unit_add_rate_for_new_than_known_motion":motion_growth,
                       "meets":bool(forget<=.002 and motion_growth)},
                "R2":{"mean_separation_ratio":float(np.mean(sep)) if sep else None,"mean_redundant_unit_pair_fraction":float(np.mean(red)) if red else None,"mean_motion_to_unit_accuracy":float(np.mean(acc)) if acc else None,
                      "mean_motion_chance":float(np.mean(chance)) if chance else None,
                      "meets":bool(sep and red and acc and chance and np.mean(sep)>1.0 and np.mean(red)<=.1 and np.mean(acc)>np.mean(chance))},
                "R3":{"mean_outside_coverage_rate":float(np.mean(cover)) if cover else None,"outside_coverage_threshold":.1,"meets":bool(cover and np.mean(cover)<=.1)},
                "R4":{"mean_novelty_auroc":float(np.mean(auc)) if auc else None,"task_new_motion_auroc":float(np.mean(task_auc)) if task_auc else None,
                      "segment_new_motion_auroc":float(np.mean(segment_auc)) if segment_auc else None,
                      "mean_known_false_add_rate":float(np.mean(false)) if false else None,
                      "mean_new_query_improvement":float(np.mean(improves)) if improves else None,"mean_reuse_rate":float(np.mean(reuse)) if reuse else None,
                      "mean_duplicate_fraction":float(np.mean(dup)) if dup else None,
                      "meets":bool(auc and false and improves and reuse and dup and np.mean(auc)>=.8 and np.mean(false)<=.05 and np.mean(improves)>0 and np.mean(reuse)>0 and np.mean(dup)<=.1 and (not task_auc or np.mean(task_auc)>=.8) and (not segment_auc or np.mean(segment_auc)>=.8))}}
        outputs[order_name]={"stages":stage_rows,"transitions":transition_rows,"requirements":checks,
                             "requirements_met_count":sum(bool(v["meets"]) for v in checks.values())}
        if _m6_unit_context(method) is not None:
            fit_state=getattr(method,"_fit_growth_state",{})
            eval_state=fit_state.get("_bench_m6_unitizer",fit_state) if method.__class__.__module__.endswith(("methods.m0","methods.m1")) else fit_state
            fit_count=len(eval_state.get("_centers",[])); fit_pending=len(eval_state.get("pending",[]))
            outputs[order_name]["unit_count_trajectory"]=[
                {"phase":"fit","data_seen":int(len(fit_base)),"unit_count":int(fit_count),
                 "units_per_1000_windows":float(1000*fit_count/max(len(fit_base),1)),
                 "pending_count":int(fit_pending),"pending_fraction":float(fit_pending/max(len(fit_base),1))}
            ]+[ {"phase":stage["stage"],"data_seen":stage.get("data_seen"),"unit_count":stage["unit_count"],
                  "units_per_1000_windows":stage.get("unit_count_per_1000_windows"),"pending_count":stage.get("pending_count"),
                  "pending_fraction":stage.get("pending_fraction")} for stage in stage_rows]
    method.memory=[]
    return {"available":True,"protocol":f"{'+'.join(train_short)} store/query -> {new_short} -> LIBERO-10; no gradient updates after fit",
            "timing_seconds_total":time.perf_counter()-protocol_started,
            "orders":outputs,"libero10_note":libero10_note,
            "decision_thresholds":{"R1_max_forgetting_mse":.002,"R2_max_redundant_fraction":.1,"R3_max_outside_coverage":.1,
                                   "R4_min_AUROC":.8,"R4_max_known_false_add":.05,"R4_min_improvement":0.0,"R4_min_reuse_rate":0.0,"R4_max_duplicates":.1}}


def _run(args):
    methods=discover_methods()
    if args.exact_methods and args.methods:
        selected={k:methods[k] for k in args.methods if k in methods}
    else:
        selected={k:v for k,v in methods.items() if not args.methods or any(k==m or k.startswith(m+"_") for m in args.methods)}
    if not selected: raise ValueError("no method variants selected")
    runid=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    root=args.output_root/f"skill-formation-{runid}";root.mkdir(parents=True,exist_ok=False)
    camera=_camera_features(args.camera_cache,load_z_run(PREVIOUS/"H25_beta0.01/eval_input",horizon=25))
    results=[]
    for split_name in getattr(args,"splits",["B"]):
      train_suites,new_suite=(
          (("libero_object","libero_spatial"),"libero_goal") if split_name=="A" else
          (("libero_object","libero_goal"),"libero_spatial"))
      for horizon in args.horizons:
        run=load_z_run(PREVIOUS/f"H{horizon}_beta0.01/eval_input",horizon=horizon)
        manifest=json.loads((PREVIOUS/f"H{horizon}_beta0.01/eval_input/manifest.json").read_text())
        split_seed=int(args.seeds[0])
        candidate=np.flatnonzero(run.usable & np.isin(run.suite,(*train_suites,new_suite)))
        split_rows,queries=_split_suite_query_groups(run,candidate,split_seed,suites=(*train_suites,new_suite))
        train_pool=np.asarray([r for r in split_rows if run.suite[r] in train_suites],dtype=np.int64)
        novel_store=np.asarray([r for r in split_rows if run.suite[r]==new_suite],dtype=np.int64)
        train_rows=_rng_sample(train_pool,run.episode_id,args.max_train_rows,split_seed)
        qrows=_rng_sample(queries[new_suite],run.episode_id,args.max_query_rows,split_seed+13)
        spatial_store=_rng_sample(novel_store,run.episode_id,args.max_spatial_rows,split_seed+27)
        if set(run.init_state_id[qrows]) & set(run.init_state_id[spatial_store]):
            raise AssertionError(f"Split {split_name}: held-out suite initial-state leakage")
        all_train=train_rows
        og_queries={s:queries[s] for s in train_suites}
        keymap,wrist_encoder,combined_encoder=_keys(run,train_rows,camera,horizon)
        l10=None; l10_note=None
        if args.sequential:
            l10_root=_find_libero10_root(args.libero10_root)
            l10,l10_note=_load_libero10_rows(l10_root,horizon,
                    combined_encoder,wrist_encoder,
                    max_rows_per_split=args.max_libero10_rows,seed_value=split_seed+47)
        for seed_value in args.seeds:
            for name,template in selected.items():
                active_keymap={k:v for k,v in keymap.items() if not args.key_names or k in args.key_names}
                if not active_keymap: raise ValueError(f"no matching key names: {args.key_names}")
                for key_name,key in active_keymap.items():
                    # Instantiate fresh method for each seed/key/horizon to keep memories isolated.
                    method=_new_method(name)
                    if callable(getattr(method,"to",None)): method.to(DEVICE)
                    if hasattr(method,"device"): method.device=torch.device(DEVICE)
                    if hasattr(method,"_device"): method._device=torch.device(DEVICE)
                    if args.epochs is not None and hasattr(method,"epochs"): method.epochs=args.epochs
                    if args.batch_size is not None and hasattr(method,"batch_size"): method.batch_size=args.batch_size
                    if hasattr(method,"seed_value"): method.seed_value=seed_value
                    if method.__class__.__module__.endswith("methods.m6"):
                        # Requested full-data pilot settings; κ, m and e remain the grid.
                        method.PROBE_SIZE=32; method.pending_batch_size=200
                    method._active_key_name=key_name
                    method._split_name=split_name; method._train_suites=train_suites; method._new_suite=new_suite
                    method._sequence_orders=args.sequential_orders
                    seq_queries={suite:_rng_sample(rows,run.episode_id,args.max_query_rows,seed_value+31+i)
                                 for i,(suite,rows) in enumerate(og_queries.items())} if args.sequential else None
                    # fit's retrieval key is selected outside the method and must match its threshold calibration.
                    output=_evaluate_one(method,name,{"epochs":args.epochs,"batch_size":args.batch_size,"early_stopping_patience":args.early_stopping_patience},run,train_rows,qrows,spatial_store,all_train,key,seed_value=seed_value,mi_permutations=args.mi_permutations,mi_bootstrap=args.mi_bootstrap,
                        sequence_queries=seq_queries,libero10=l10,libero10_note=l10_note)
                    output["key_name"]=key_name;output["split"]=split_name;results.append(output)
                    method_json=root/"runs"/f"split{split_name}"/f"H{horizon}"/f"seed{seed_value}"/name/key_name
                    method_json.mkdir(parents=True,exist_ok=False)
                    (method_json/"result.json").write_text(json.dumps(output,indent=2,default=float))
                    print(f"H={horizon} seed={seed_value} {name} key={key_name}: MSE={output['mse']['spatial_added_top5']:.4f}",flush=True)
    _report(root,results,args)
    print(f"report: {root/'report.md'}",flush=True)


def _report(root,results,args):
    lines=["# Skill formation benchmark","",
      "IMPORTANT: earlier reports used the cached suite names, which had Goal and Spatial reversed. Task-index mapping is now authoritative. Split A trains on object+spatial (the same task allocation called object+goal in the old manifests); its held-out Goal data is the same data previously reported as ‘spatial’. Split B trains on object+goal and evaluates held-out spatial. LIBERO-10 is streamed last. Cached features are reused without extraction. PCA, normalization and novelty calibration are fit on training suites only.","",
      f"Splits={getattr(args,'splits',['B'])}; seeds={args.seeds}; horizons={args.horizons}; exact method selection={args.exact_methods}; max train/query/new-suite-store rows={args.max_train_rows}/{args.max_query_rows}/{args.max_spatial_rows}; epochs override={args.epochs}; batch override={args.batch_size}; metrics permutations/bootstrap={args.mi_permutations}/{args.mi_bootstrap}. M6 uses 32 functional probes and pending batch 200 (expiry follows method default); backbone fits are reused across κ/m variants with identical train rows and encoder settings.","",
      "## Held-out suite reconstruction","","Each cell is mean masked action MSE. The retrieved-memory conditions use the training-suite memory before and after held-out-suite additions.","",
      "| Split | Method | H | seed | key | Prior | train memory top1 | train memory top5 | after-add top1 | after-add top5 | item add rate | unit add rate |",
      "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        m=r["mse"]; bi=r.get("independence_interventions",{}); sep=r.get("redundancy",{}).get("mean_pair_action_mse")
        lines.append(f"| {r.get('split','?')} | {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {m['prior']:.4f} | {m['object_goal_top1']:.4f} | {m['object_goal_top5']:.4f} | {m['spatial_added_top1']:.4f} | {m['spatial_added_top5']:.4f} | {r['spatial_addition']['memory_item_rate']:.1%} | {r['spatial_addition']['unit_add_rate']:.1%} |")
    lines += ["","## Compact summary by split","","Reconstruction and requirement values summarize the selected train-fitted key(s); detailed per-key transitions follow below.","","| Split | Method | held-out MSE after add | R1 (forget, new units/1k new-motion) | R2 separation / redundancy | R3 outside | R4 suite / task / segment AUROC | new/known unit-add rate | q90 new/known | q95 new/known | q99 new/known |","|---|---|---:|---|---|---:|---|---|---|---|---|"]
    methods_by_split=sorted({(str(x.get("split","?")),x["method"]) for x in results})
    def mean_defined(xs):
        vals=[float(x) for x in xs if x is not None and np.isfinite(float(x))]
        return float(np.mean(vals)) if vals else None
    for split,method_name in methods_by_split:
        group=[x for x in results if str(x.get("split","?"))==split and x["method"]==method_name]
        mse=mean_defined([x["mse"].get("spatial_added_top5") for x in group])
        reqs=[]; trs=[]
        for x in group:
            orders=x.get("sequential_protocol",{}).get("orders",{})
            if not orders: continue
            order=next(iter(orders.values())); reqs.append(order.get("requirements",{}))
            trs.extend(t for t in order.get("transitions",[]) if t.get("requires_new_motion_unit"))
        r1pass=bool(reqs) and all(q.get("R1",{}).get("meets",False) for q in reqs)
        forget=mean_defined([q.get("R1",{}).get("max_forgetting_delta") for q in reqs])
        sep=mean_defined([q.get("R2",{}).get("mean_separation_ratio") for q in reqs])
        red=mean_defined([q.get("R2",{}).get("mean_redundant_unit_pair_fraction") for q in reqs])
        cover=mean_defined([q.get("R3",{}).get("mean_outside_coverage_rate") for q in reqs])
        suite_auc=mean_defined([q.get("R4",{}).get("mean_novelty_auroc") for q in reqs])
        task_auc=mean_defined([q.get("R4",{}).get("task_new_motion_auroc") for q in reqs])
        seg_auc=mean_defined([q.get("R4",{}).get("segment_new_motion_auroc") for q in reqs])
        rates=mean_defined([t.get("unit_add_rate_by_motion",{}).get("new_motion") for t in trs]); known=mean_defined([t.get("unit_add_rate_by_motion",{}).get("known_motion") for t in trs])
        qrates={q:(mean_defined([t.get("threshold_add_rates",{}).get(str(q),{}).get("new_motion_unit_add_rate") for t in trs]),
                   mean_defined([t.get("threshold_add_rates",{}).get(str(q),{}).get("known_motion_unit_add_rate") for t in trs])) for q in (.9,.95,.99)}
        fmt=lambda v:"—" if v is None else f"{v:.4f}"
        new_per_1k=None if rates is None else 1000.0*rates
        lines.append(f"| {split} | {method_name} | {fmt(mse)} | {r1pass} (forget={fmt(forget)}; new={fmt(new_per_1k)}/1k, new>known) | {fmt(sep)} / {fmt(red)} | {fmt(cover)} | {fmt(suite_auc)} / {fmt(task_auc)} / {fmt(seg_auc)} | {fmt(rates)} / {fmt(known)} | {fmt(qrates[.9][0])}/{fmt(qrates[.9][1])} | {fmt(qrates[.95][0])}/{fmt(qrates[.95][1])} | {fmt(qrates[.99][0])}/{fmt(qrates[.99][1])} |")
    lines += ["","## Other study metrics","","Per-run JSON includes motion/task MI with shuffle correction, conditional task/base MI, base unit utilization and perplexity, decoder-based redundancy, HSIC(base,residual), residual motion-type stability, and an estimated raw bits/window plus spatial reconstruction distortion. These metrics use the row limits listed in the run conditions above.",""]
    for r in results:
        mi=r["base_motion_mi"]; lines.append(f"- {r['method']} H={r['horizon']} seed={r['seed']} key={r['key_name']}: env joint MI shuffle-adjusted={mi.get('env/joint',{}).get('shuffle_adjusted_bits',float('nan')):.4g}; task MI shuffle-adjusted={mi.get('task',{}).get('shuffle_adjusted_bits',float('nan')):.4g}; I(base;motion|task)={mi.get('env/joint_given_task',{}).get('shuffle_adjusted_bits',float('nan')):.4g}; units={r['base_usage']['n_units_used']}/{r['base_usage']['total_units']}, perplexity={r['base_usage']['perplexity']:.3g}; same-residual pair action MSE={r.get('redundancy',{}).get('mean_pair_action_mse')}; HSIC={r['hsic_base_residual']}; residual stability={r['residual_motion_type_stability']}; rate={r['rate_distortion']['estimated_bits_per_window']} bits/window, distortion={r['rate_distortion']['spatial_added_top5_mse']:.4f}.")
    lines += ["","## Rate-distortion and update properties","","Continuous rates are KL in bits (M0 uses its learned conditional q||p; deterministic continuous methods use the reported train-fitted Gaussian surrogate). Discrete base rate is log2(code count); residual rate uses a train-fitted Gaussian KL surrogate. These are estimates, not measured entropy-coded bytes.","",
      "| Method | H | seed | key | estimated bits/window | spatial MSE | neural weights fixed on add | representatives change | old IDs change |",
      "|---|---:|---:|---|---:|---:|---|---|---|"]
    for r in results:
        rp=r["rate_distortion"]; up=r["memory_update_properties"]
        lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {rp['estimated_bits_per_window']:.3f} | {rp['spatial_added_top5_mse']:.4f} | {up['neural_weights_fixed_on_memory_add']} | {up['memory_representatives_change_on_add']} | {up['old_ids_change_on_add']} |")
    sequential=[r for r in results if r.get("sequential_protocol",{}).get("available")]
    if sequential:
        lines += ["","## Sequential suite protocol: R1–R4","","Store/query groups are split by namespaced initial-state ID. The selected training suites are streamed first, then the held-out suite, then LIBERO-10. Fit, PCA, normalization, and novelty calibration use only the selected training suites. R4 task labels in Split A are evaluator-only; segment labels use motion-feature nearest-neighbor distance against training suites (new-motion if above training q95). Neither label is provided to fit or memory updates.","",
          "LIBERO-10 language-conditioned camera features are loaded as optional arrays with row-level `has_base_lang` / `has_wrist_lang` masks. Missing or nonfinite camera rows are unavailable and are not used as implicit zero-valued features. The current retrieval key remains language-free.",
          "","### LIBERO-10 camera-language availability","","| Method | H | seed | base-language rows available / total | wrist-language rows available / total |","|---|---:|---:|---:|---:|"]
        for r in sequential:
            availability=r.get("libero10_camera_language_availability",{})
            if availability:
                b=availability.get("base_lang",{}); w=availability.get("wrist_lang",{})
                lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {b.get('available_rows',0)} / {b.get('total_rows',0)} | {w.get('available_rows',0)} / {w.get('total_rows',0)} |")
        lines += ["","### Stage metrics","","| Method | H | seed | key | order | stage | suite | query MSE own/prior/memory | item blend | skill blend | memory units | outside coverage | train q95 radius | motion→unit accuracy/chance | between/within separation ratio | redundant unit-pair fraction |","|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sequential:
            protocol=r["sequential_protocol"]
            for order_name,order in protocol.get("orders",{}).items():
                for stage in order.get("stages",[]):
                    for suite,metric in stage.get("suites_evaluated",{}).items():
                        m=metric.get("reconstruction_mse",{}); motion=metric.get("R2",{}).get("motion_to_unit_accuracy") or {}
                        sep=metric.get('R2',{}).get('unit_action_separation',{}).get('separation_ratio')
                        lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {order_name} | {stage['stage']} | {suite} | {m.get('own',float('nan')):.4f}/{m.get('prior',float('nan')):.4f}/{m.get('memory',float('nan')):.4f} | {m.get('memory_item',m.get('memory'))} | {m.get('memory_skill', '—')} | {metric.get('units',{}).get('memory_units')} | {metric.get('coverage',{}).get('outside_train_base_radius_rate')} | {metric.get('coverage',{}).get('threshold_train_q95')} | {motion.get('accuracy')}/{motion.get('chance')} | {sep} | {metric.get('R2',{}).get('redundant_unit_fraction')} |")
        m6runs=[r for r in sequential if r["method"].startswith("M6_")]
        if m6runs:
            lines += ["","### M6 unit growth and pending rows","","Unit growth uses M6's functional distance, pending-component support count m, and minimum e=3 episodes. Item and skill retrieval are both softmax blends. Unit registration events are matched to source episode/init-state/segment IDs, including rows that remain pending or were rejected from item memory.","","| Split | Method | key | order | phase | data windows seen | units | units per 1k windows | pending rows | pending share | new-motion registration | known-motion registration |","|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in m6runs:
                for order_name,order in r.get("sequential_protocol",{}).get("orders",{}).items():
                    for item in order.get("unit_count_trajectory",[]):
                        transition=next((t for t in order.get("transitions",[]) if t.get("stage")==item.get("phase")),{})
                        rates=transition.get("unit_add_rate_by_motion",{})
                        lines.append(f"| {r.get('split')} | {r['method']} | {r['key_name']} | {order_name} | {item.get('phase')} | {item.get('data_seen')} | {item.get('unit_count')} | {item.get('units_per_1000_windows')} | {item.get('pending_count')} | {item.get('pending_fraction')} | {rates.get('new_motion','—')} | {rates.get('known_motion','—')} |")
        lines += ["","### R1 growth and R4 motion novelty","","R1 passes growth only when unit-add rate on segment-level new-motion rows is nonzero and exceeds the rate on known-motion rows; forgetting threshold remains 0.002 MSE. Split B's spatial suite is expected to mostly add memory items rather than new units.","","| Split | Method | stage | forget max | R1 | new/known unit-add rate | R4 suite AUROC | task AUROC | segment AUROC | task new-motion rate | segment new-motion rate |","|---|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
        for r in sequential:
            for order_name,order in r["sequential_protocol"].get("orders",{}).items():
                req=order.get("requirements",{}); r1=req.get("R1",{}); r4=req.get("R4",{})
                for tr in order.get("transitions",[]):
                    rates=tr.get("unit_add_rate_by_motion",{})
                    lines.append(f"| {r.get('split')} | {r['method']} | {tr.get('stage')} | {r1.get('max_forgetting_delta')} | {r1.get('meets')} | {rates.get('new_motion')}/{rates.get('known_motion')} | {tr.get('novelty_auroc')} | {tr.get('task_new_motion_auroc')} | {tr.get('segment_new_motion_auroc')} | {tr.get('task_new_motion_rate')} | {tr.get('segment_new_motion_rate')} |")
                    lines.append("| thresholds | " + r["method"] + " | " + tr.get("stage","") + " | " + " | ".join(f"q{int(float(q)*100)} candidate new/known={v.get('new_motion_unit_candidate_rate',v.get('new_motion_unit_add_rate'))}/{v.get('known_motion_unit_candidate_rate',v.get('known_motion_unit_add_rate'))} (base novelty τ={v.get('unit_novelty_threshold')}; motion-label threshold=q95; query score>{v.get('query_score_above_novelty_threshold_rate')})" for q,v in tr.get("threshold_add_rates",{}).items()) + " |")
        lines += ["","### R4 additions and order transitions","","| Method | H | seed | key | order | stage/new suite | AUROC | known false-add rate | item add rate | unit add / 1000 windows | MSE improvement | reused by new-suite query | duplicate fraction |","|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sequential:
            for order_name,order in r["sequential_protocol"].get("orders",{}).items():
                for tr in order.get("transitions",[]):
                    lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {order_name} | {tr['stage']}/{'+'.join(tr['new_suites'])} | {tr.get('novelty_auroc')} | {tr.get('known_false_add_rate')} | {tr.get('memory_item_add_rate')} | {tr.get('new_units_per_1000_windows')} | {tr.get('improvement_mse')} | {tr.get('new_units_reused_by_query_rate')} | {tr.get('new_unit_duplicate_fraction')} |")
        merge_rows=[]
        for r in sequential:
            if r.get("method")!="M4_merge": continue
            for order_name,order in r.get("sequential_protocol",{}).get("orders",{}).items():
                for tr in order.get("transitions",[]):
                    if tr.get("merge"):
                        merge_rows.append((r,order_name,tr))
        if merge_rows:
            lines += ["","### M4_merge additions before/after merge","","Unit additions are counted from row-level new-prototype events before the stage merge. Active units after merge and the number merged are reported separately; the fitted seed stage is not merged.","",
              "| Split | key | stage | new prototype events | active before add | pre-merge units | post-merge units | merged units |","|---|---|---|---:|---:|---:|---:|---:|"]
            for r,order_name,tr in merge_rows:
                lines.append(f"| {r.get('split')} | {r.get('key_name')} | {tr.get('stage')} | {tr.get('new_units_added_before_merge',tr.get('new_skill_units'))} | {tr.get('active_unit_count_before_merge',tr.get('unit_count_before'))} | {tr['merge'].get('pre_merge_count')} | {tr['merge'].get('post_merge_count')} | {tr.get('units_merged',tr['merge'].get('pre_merge_count',0)-tr['merge'].get('post_merge_count',0))} |")
        lines += ["","### 判定定義・閾値","",
          "R1 継続的な追加: 忘却は各 suite の連続する評価段階間での memory MSE 増分の最大正値で、閾値は 0.002。成長は new-motion 区間で単位追加率が 0 より大きく、かつ known-motion 区間より高いこと。suite 全体での追加ゼロだけでは不合格にしない。Split B の spatial は新しい場面・既知運動のため、単位より記憶項目が増えることを想定する。",
          "R2 区別: 分離比 = 同一 Prior residual・同一状態集合で復号した単位間行動 MSE / 同一単位に割り当てられた区間間行動 MSE。1 超で単位差が単位内ばらつきを上回る。機能的重複は単位ペア行動 MSE が単位内行動 MSE 以下のペア割合。閾値は 0.1。運動特徴 1-NN 単位分類精度も偶然率を超えることを要求。M0/M1 の連続表現も M6 の機能距離・保留・登録規則を共通評価器として使い、固定個数の k-means は使わない。",
          "R3 表現: own(base,residual) の masked MSE と、量子化前 base から割当単位代表までの距離が学習 suite の q95 を超える割合を報告する。連続 M0/M1 と M6 は M6 の機能距離による保留単位化を使う。離散方式は量子化前 base をコード/プロトタイプ代表（M3 は粗＋細コードの和、M5 は原子再構成）と比較する。被覆外の合格閾値は 10%。",
          "R4: suite 単位の AUROC は参考。主に単位新規性点数から task-level / segment-level new-motion を識別する AUROC を報告する。threshold 表の q90/q95/q99 は train の単位新規性点数分位をしきい値として、segment new/known は固定の train motion-NN q95 で分けた候補単位追加率を示す。タスク new/known の候補率も JSON に記録する。実際に追加された単位の率は別表の `unit_add_rate_by_motion` に示す。主判定の参考閾値は AUROC ≥0.8、forgetting ≤0.002。数値は比較用で統計的有意性を意味せず、境界値は丸め前 JSON を参照する。","",
          "| Method | H | seed | key | order | R1 忘却 / spatial 成長 | R2 分離 / 重複 / 精度 | R3 被覆外 | R4 AUROC / 誤追加 / 再利用 / 重複 | met / 4 |","|---|---:|---:|---|---|---|---|---|---|---:|"]
        for r in sequential:
            for order_name,order in r["sequential_protocol"].get("orders",{}).items():
                checks=order.get("requirements",{}); flags=["yes" if checks.get(f"R{i}",{}).get("meets") else "no" for i in range(1,5)]
                c1,c2,c3,c4=(checks.get(f"R{i}",{}) for i in range(1,5))
                growth=[(x.get("stage"),x.get("new_units_per_1000_windows"),x.get("unit_add_rate_by_motion")) for x in order.get("transitions",[]) if x.get("requires_new_motion_unit")]
                r1=f"{flags[0]} (forget={c1.get('max_forgetting_delta')} ≤ .002; new-suite growth={growth})"
                r2=f"{flags[1]} (sep={c2.get('mean_separation_ratio')}; pair overlap={c2.get('mean_redundant_unit_pair_fraction')}; acc={c2.get('mean_motion_to_unit_accuracy')} vs {c2.get('mean_motion_chance')})"
                r3=f"{flags[2]} ({c3.get('mean_outside_coverage_rate')})"
                r4=f"{flags[3]} ({c4.get('mean_novelty_auroc')}; {c4.get('mean_known_false_add_rate')}; {c4.get('mean_reuse_rate')}; {c4.get('mean_duplicate_fraction')})"
                lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {order_name} | {r1} | {r2} | {r3} | {r4} | {order.get('requirements_met_count')}/4 |")
        lines += ["","#### 新単位と判定された区間の例（各段階 最大5件）","","| Method | order | stage/suite | task | task-level motion label | segment-level motion label | direction/rotation/gripper label | Δxyz sum | gripper gap Δ | unit ID |","|---|---|---|---|---|---|---|---|---:|---|"]
        for r in sequential:
            for order_name,order in r["sequential_protocol"].get("orders",{}).items():
                for tr in order.get("transitions",[]):
                    examples=tr.get("new_unit_examples",[])[:5]
                    for ex in examples:
                        ml=ex.get("motion_labels",{}); mf=ex.get("motion_features",{})
                        lines.append(f"| {r['method']} | {order_name} | {tr['stage']}/{ex.get('suite')} | {ex.get('task')} | {ex.get('task_motion_label')} | {ex.get('segment_motion_label')} | {ml.get('direction')}/{ml.get('rotation')}/{ml.get('gripper')} | {mf.get('sum_delta_xyz')} | {mf.get('gripper_gap_delta')} | {ex.get('unit_id')} |")
                    if not examples: lines.append(f"| {r['method']} | {order_name} | {tr['stage']} | — | — | — | no unit added (or row-level unit ID unavailable) | — | — | — |")
        boundary=[]
        def close(value,threshold,tolerance):
            return value is not None and np.isfinite(value) and abs(float(value)-threshold)<=tolerance
        for r in sequential:
            for order_name,order in r["sequential_protocol"].get("orders",{}).items():
                checks=order.get("requirements",{})
                r1=checks.get("R1",{}); r2=checks.get("R2",{}); r3=checks.get("R3",{}); r4=checks.get("R4",{})
                if close(r1.get("max_forgetting_delta"),.002,.0002): boundary.append(f"{r['method']}/{order_name} R1 forgetting={r1.get('max_forgetting_delta')} vs .002")
                if close(r2.get("mean_separation_ratio"),1.,.1): boundary.append(f"{r['method']}/{order_name} R2 separation={r2.get('mean_separation_ratio')} vs 1.0")
                if close(r2.get("mean_redundant_unit_pair_fraction"),.1,.02): boundary.append(f"{r['method']}/{order_name} R2 pair overlap={r2.get('mean_redundant_unit_pair_fraction')} vs .1")
                if close(r3.get("mean_outside_coverage_rate"),.1,.02): boundary.append(f"{r['method']}/{order_name} R3 coverage={r3.get('mean_outside_coverage_rate')} vs .1")
                for tr in order.get("transitions",[]):
                    if close(tr.get("novelty_auroc"),.8,.02): boundary.append(f"{r['method']}/{order_name}/{tr['stage']} R4 AUROC={tr.get('novelty_auroc')} vs .8")
                    if close(tr.get("known_false_add_rate"),.05,.01): boundary.append(f"{r['method']}/{order_name}/{tr['stage']} R4 false-add={tr.get('known_false_add_rate')} vs .05")
                    if tr.get("requires_new_motion_unit") and tr.get("new_skill_units")==0: boundary.append(f"{r['method']}/{order_name}/{tr['stage']} R1 spatial unit growth=0 (fails required growth)")
        lines += ["","#### 閾値境界の注記","","判定値が以下の許容帯（R1 ±0.0002 MSE、R2 分離比 ±0.1・重複率 ±0.02、R3 被覆率 ±0.02、R4 AUROC ±0.02・誤追加率 ±0.01）に入った結果、および spatial 新運動段階での追加 0 件を列挙する。丸めで合否は決めず JSON の生値を使う。"]
        lines.extend(["- "+x for x in boundary] if boundary else ["- 該当なし"])
        notes=sorted({r["sequential_protocol"].get("libero10_note") for r in sequential if r["sequential_protocol"].get("libero10_note")})
        if notes: lines += ["", "LIBERO-10: " + "; ".join(notes) + ". The protocol still completes the Object→Goal→Spatial order when that cache is absent."]
    elif args.sequential:
        lines += ["","## Sequential suite protocol","","Sequential metrics were requested but no run produced the required Object, Goal and Spatial suite tables."]
    lines += ["","## Unit contribution and order sensitivity","","Functional redundancy is the fraction of sampled units whose removal increases MSE by a usage-weighted relative contribution of at most 0.01. Each unit is evaluated on queries that retrieved it. Per-unit deltas are in each result JSON. Order sensitivity replays spatial insertion in temporal, reverse, and seeded-random order; query subset is capped at 128 rows.",""]
    for r in results:
        fu=r["functional_unit_ablation"]; od=r["addition_order_sensitivity"]; iv=r["independence_interventions"]
        lines.append(f"- {r['method']} H={r['horizon']} seed={r['seed']} key={r['key_name']}: redundant units={fu.get('redundant_unit_fraction')}; ΔMSE threshold={fu.get('threshold_mse')}; order add-rate std={od.get('add_rate_std')}; order spatial-MSE std={od.get('spatial_mse_std')}; residual-swap type stability={iv.get('residual_shuffle_base_type_stability')}; residual-only zero-base MSE={iv.get('zero_base_residual_only_mse')}; residual with random-base MSE={iv.get('random_base_with_residual_mse')}.")
    timed=[r for r in results if r.get("timing_seconds")]
    if timed:
        lines += ["","## Measured wall time by method and stage","","Times are wall-clock seconds for one method/key evaluation. Object/Goal pre-add and post-add rechecks use a deterministic sample of at most 1,024 training rows; the primary spatial query set remains uncapped. Training time includes fit-time key-distance calibration.","",
          "| Method | H | seed | key | train + calibration | encode | OG seed | OG retrieval/decode | spatial add | spatial retrieval/decode | MI bootstrap | core indicators | unit ablation | order sensitivity | other | total |","|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in timed:
            t=r["timing_seconds"]
            def tsum(*names): return sum(float(t.get(n,0.0)) for n in names)
            other=tsum("sequence_table_preparation","object_goal_post_add_recheck","residual_interventions","sequential_protocol","unattributed_seconds")
            lines.append(f"| {r['method']} | {r['horizon']} | {r['seed']} | {r['key_name']} | {tsum('training'):.2f} | {tsum('encoding'):.2f} | {tsum('object_goal_memory_seed'):.2f} | {tsum('object_goal_retrieval_and_decode'):.2f} | {tsum('spatial_memory_add'):.2f} | {tsum('spatial_retrieval_and_decode'):.2f} | {tsum('mutual_information_bootstrap'):.2f} | {tsum('usage_hsic_rate_distortion_core_metrics'):.2f} | {tsum('functional_unit_ablation'):.2f} | {tsum('addition_order_sensitivity'):.2f} | {other:.2f} | {tsum('total_seconds'):.2f} |")
    lines += ["","## Method details","","Methods are discovered dynamically from `methods/*.py` via `build_variants()` or public `VARIANTS`; methods use duck typing, no shared base class. Each method JSON records its `describe()` output and memory addition reasons.",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path);p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUTS);p.add_argument("--camera-cache",type=Path,default=CAMERA_CACHE);p.add_argument("--methods",nargs="*",default=None);p.add_argument("--exact-methods",action="store_true",help="treat --methods as exact variant names rather than family prefixes");p.add_argument("--key-names",nargs="+",default=None);p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--horizons",nargs="+",type=int,default=[25,10]);p.add_argument("--splits",nargs="+",choices=("A","B"),default=["B"]);p.add_argument("--epochs",type=int);p.add_argument("--batch-size",type=int);p.add_argument("--early-stopping-patience",type=int,default=3);p.add_argument("--max-train-rows",type=int);p.add_argument("--max-query-rows",type=int);p.add_argument("--max-spatial-rows",type=int);p.add_argument("--max-libero10-rows",type=int);p.add_argument("--mi-permutations",type=int,default=20);p.add_argument("--mi-bootstrap",type=int,default=100);p.add_argument("--sequential",action="store_true");p.add_argument("--sequential-orders",nargs="+",default=None);p.add_argument("--libero10-root",type=Path)
    args=p.parse_args()
    if args.config:
        cfg=json.loads(args.config.read_text())
        cli_defaults={"mi_permutations":20,"mi_bootstrap":100,"early_stopping_patience":3}
        for key in ("methods","key_names","seeds","horizons","splits","epochs","batch_size","early_stopping_patience","output_root","camera_cache","sequential","sequential_orders","libero10_root","mi_permutations","mi_bootstrap","max_libero10_rows"):
            if (getattr(args,key,None) is None or (key=="seeds" and args.seeds==[0,1,2]) or (key=="horizons" and args.horizons==[25,10]) or (key=="splits" and args.splits==["B"]) or (key=="sequential" and args.sequential is False) or (key in cli_defaults and getattr(args,key)==cli_defaults[key])) and key in cfg:
                val=cfg[key]
                if key in ("output_root","camera_cache") and val is not None: val=Path(val)
                setattr(args,key,val)
    _run(args)


if __name__=="__main__": main()
