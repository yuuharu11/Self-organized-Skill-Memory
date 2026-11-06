#!/usr/bin/env python3
"""CPU study: label-free variable-duration skill shapes and representation grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from .data import (HELDOUT_TASKS, LONG_TASKS, choose_type_threshold,
                  choose_velocity_threshold, fixed10, grip_segments,
                  grip_velocity_segments, load_source, normalize_physical,
                  threshold_segments, type_change_segments, training_row_mask,
                  unnormalize)
    from .representations import (PHASE, apply_pose, canonicalize, decode_canonical,
        decode_original, dmp_encode, m6_fit_generic, medoid_generic, pca_decode,
        pca_encode, pca_fit, procrustes_rotation, raw_pca_input, resample, spline_encode)
except ImportError:
    from data import (HELDOUT_TASKS, LONG_TASKS, choose_type_threshold,
                      choose_velocity_threshold, fixed10, grip_segments,
                      grip_velocity_segments, load_source, normalize_physical,
                      threshold_segments, type_change_segments, training_row_mask,
                      unnormalize)
    from representations import (PHASE, apply_pose, canonicalize, decode_canonical,
        decode_original, dmp_encode, m6_fit_generic, medoid_generic, pca_decode,
        pca_encode, pca_fit, procrustes_rotation, raw_pca_input, resample, spline_encode)

ROOT = Path(os.environ.get("SKILL_MEMORY_ARTIFACTS", "/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104"))
DEFAULT_RUN = ROOT / "run-01"
DEFAULT_OUT = ROOT / "shape-skills-01"
CHECKPOINT = Path("/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero")
KAPPAS = (.025, .05, .1, .2, .4, .8)


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def append_progress(path, **event):
    with Path(path).open("a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}, ensure_ascii=False) + "\n")


def source_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def make_segment_sets(episodes, rows, progress_path):
    dev_keys = [k for k,v in episodes.items() if k[0] in HELDOUT_TASKS and v["meta"]["validation"]]
    if not dev_keys:
        raise AssertionError("no heldout validation episodes for threshold selection")
    vthr, vrec = choose_velocity_threshold(episodes, dev_keys)
    tthr, trec = choose_type_threshold(episodes, dev_keys)
    configs = [("S0_fixed10", "fixed10", None)]
    for kind in ("greedy", "dp"):
        for eps in (.005,.01,.02,.04):
            configs.append((f"S1_{kind}_e{str(eps).replace('.','p')}", kind, eps))
    configs += [("S2_grip", "grip", None), ("S2_grip_velocity", "gripvel", vthr),
                ("S3_type_change", "typechange", tthr)]
    rows_lookup = {}
    episode_rows = {}
    for rid,(t,e,f) in enumerate(zip(rows["task"], rows["episode"], rows["frame"], strict=True)):
        rows_lookup[(int(t),int(e),int(f))] = rid
        episode_rows.setdefault((int(t),int(e)), []).append((int(f),rid))
    for key in episode_rows: episode_rows[key].sort()
    result={}
    all_stats=[]
    for name,kind,param in configs:
        segs=[]; fallback_eps=[]; fallback_segments=0
        for key, seq in episodes.items():
            if kind == "fixed10": bounds=fixed10(seq)
            elif kind == "greedy": bounds=threshold_segments(seq,float(param),kind)
            elif kind == "dp":
                bounds, did_fallback = threshold_segments(seq,float(param),kind,return_fallback=True)
                if did_fallback:
                    fallback_eps.append(f"{key[0]}:{key[1]}"); fallback_segments += len(bounds)
            elif kind == "grip": bounds=grip_segments(seq)
            elif kind == "gripvel": bounds=grip_velocity_segments(seq,float(param))
            elif kind == "typechange": bounds=type_change_segments(seq,float(param))
            else: raise AssertionError(kind)
            for start,end in bounds:
                if end-start < 10: continue
                frame=int(seq["frame"][start])
                candidates=episode_rows.get(key,[])
                cand=candidates[:0]
                if candidates:
                    pos=int(np.searchsorted(np.asarray([x[0] for x in candidates]),start,"left"))
                    if pos<len(candidates) and candidates[pos][0]<end:cand=candidates[pos:pos+1]
                row_id=min(cand)[1] if cand else -1
                segs.append({"task":key[0],"episode":key[1],"start":frame,"length":int(end-start),
                             "row_id":int(row_id),"a":seq["actions_phys"][start:end],
                             "meta":seq["meta"]})
        result[name]=segs
        lens=np.asarray([s["length"] for s in segs],np.int32)
        def context_summary(selected):
            if not selected:return {"count":0}
            vals=[]
            for s in selected:
                a=np.asarray(s["a"],np.float32);g=a[:,6]
                vals.append({"task":int(s["task"]),"episode":int(s["episode"]),"start":int(s["start"]),
                    "length":int(s["length"]),"translation_path_norm":float(np.linalg.norm(a[:,:3].sum(0))),
                    "rotation_path_norm":float(np.linalg.norm(a[:,3:6].sum(0))),
                    "mean_translation_speed":float(np.linalg.norm(a[:,:3],axis=1).mean()),
                    "grip_switches":int(np.sum((g[1:]>=0)!=(g[:-1]>=0)))})
            return {"count":len(vals),"mean_translation_path_norm":float(np.mean([x["translation_path_norm"] for x in vals])),
                    "mean_rotation_path_norm":float(np.mean([x["rotation_path_norm"] for x in vals])),
                    "mean_translation_speed":float(np.mean([x["mean_translation_speed"] for x in vals])),
                    "mean_grip_switches":float(np.mean([x["grip_switches"] for x in vals])),
                    "examples":vals[:5]}
        q25,q75=np.quantile(lens,[.25,.75]) if len(lens) else (0,0)
        all_stats.append({"name":name,"kind":kind,"parameter":param,"segments":len(segs),
                          "length_q10_q25_q50_q75_q90":np.quantile(lens,[.1,.25,.5,.75,.9]).round(2).tolist(),
                          "short_context_length_le_q25":context_summary([s for s in segs if s["length"]<=q25]),
                          "long_context_length_ge_q75":context_summary([s for s in segs if s["length"]>=q75]),
                          "n_memory_no_eval":int(sum(s["meta"].get("present",False) and s["meta"]["memory"] and s["task"] not in LONG_TASKS+HELDOUT_TASKS for s in segs)),
                          "n_train_no_eval":int(sum(s["meta"].get("present",False) and (not s["meta"]["memory"]) and (not s["meta"]["validation"]) and s["task"] not in LONG_TASKS+HELDOUT_TASKS for s in segs)),
                          "dp_fallback_episode_count":len(fallback_eps),"dp_fallback_segment_count":fallback_segments,
                          "dp_fallback_episode_keys":fallback_eps})
        append_progress(progress_path,event="segmentation_done",name=name,segments=len(segs))
    return result, all_stats, {"velocity":vrec,"type_change":trec}


def row_groups(rows):
    groups={}
    for i,(t,e,f) in enumerate(zip(rows["task"],rows["episode"],rows["frame"],strict=True)):
        groups.setdefault((int(t),int(e)),[]).append((int(f),i))
    return {k:sorted(v) for k,v in groups.items()}


def fit_m1(rows, pooled, fit_row_ids, base_dim):
    # Reuse M1's exact codec losses/settings; fitting remains CPU-only.
    import torch
    torch.set_num_threads(8)
    import sys
    methods=Path(__file__).resolve().parents[3] / "emergent" / "light" / "methods"
    sys.path.insert(0,str(methods.parent.parent.parent))
    from emergent.light.methods.m1 import M1
    ids=np.asarray(fit_row_ids,np.int64)
    actions=np.asarray(rows["actions"][ids],np.float32)
    states=np.asarray(rows["state"][ids],np.float32)
    feat=np.asarray(pooled[ids],np.float32)
    tasks=np.asarray(rows["task"][ids]); eps=np.asarray(rows["episode"][ids])
    key=np.zeros((len(ids),64),np.float32);key[:,:min(64,feat.shape[1])]=feat[:,:64]
    model=M1(epochs=30,batch_size=128,seed_value=20261104,base_dim=base_dim,residual_dim=4,
             device="cpu",independence_weight=.01,beta_r=1.,base_use_weight=5.,base_use_margin=.05,
             width=128,decoder_obs=False)
    model.fit({"actions":actions,"state":states,"feature":feat,"key":key,
               "episode_id":np.asarray([f"{t}:{e}" for t,e in zip(tasks,eps,strict=True)]),
               "init_state_id":np.asarray([f"{t}:{e}" for t,e in zip(tasks,eps,strict=True)]),
               "seed":20261104})
    return model


def make_rep_models(train_segments, rows, pooled, fit_row_ids, *, segmentation_name=None, q01=None, q99=None):
    models=[]
    native_fixed10 = segmentation_name == "S0_fixed10"
    if q01 is None or q99 is None:
        raise ValueError("official q01/q99 statistics are required for raw-action PCA")
    if native_fixed10 and any(len(s["a"]) != 10 for s in train_segments):
        raise AssertionError("S0_fixed10 must contain native 10-step segments")
    # Generalized Procrustes reference: mean canonical xyz path, fixed using Train only.
    can_train=[]
    for s in train_segments:
        can,_=canonicalize(s["a"]);can_train.append(np.cumsum(resample(can,PHASE),axis=0)[:,:3])
    ref=np.mean(can_train,axis=0) if can_train else np.zeros((PHASE,3),np.float32)
    ref=ref-ref.mean(0,keepdims=True)
    # Preserve an explicit no-compression control for the canonical physical
    # shape. The scale/rotation/length and gripper remain in pose metadata.
    models.append({"kind":"canonical_raw","code_dim":PHASE*6})
    for family in ("pca_raw","pca_raw_white","pca_canonical","pca_gripsep","procrustes_pca"):
        for k in (8,12,16,24,32):
            xs=[]
            for s in train_segments:
                if family.startswith("pca_raw"):
                    x=raw_pca_input(s["a"],q01,q99,
                                    native_length=10 if native_fixed10 else None).reshape(-1)
                else:
                    c,p=canonicalize(s["a"],procrustes_ref=ref if family=="procrustes_pca" else None)
                    c=resample(c,PHASE)
                    if family=="pca_canonical":
                        x=np.concatenate([c,resample(s["a"][:,6:7],PHASE)],axis=1).reshape(-1)
                    else:x=c.reshape(-1)
                xs.append(x)
            whiten=family=="pca_raw_white"
            model={"kind":family,"pca":pca_fit(xs,k,whiten=whiten),"reference":ref if family=="procrustes_pca" else None,
                   "k":k,"code_dim":min(k,len(xs))}
            if family.startswith("pca_raw"):
                model.update(q01=np.asarray(q01,np.float32),q99=np.asarray(q99,np.float32),
                             pca_input_length=10 if native_fixed10 else None)
            models.append(model)
    for n in (5,10,20): models.append({"kind":"dmp","n_basis":n,"code_dim":n*6})
    # A fixed 16-control spline is projected to each requested code dimension.
    # Thus the advertised 8/16/24 are actual stored scalar values, not a
    # misleading count of 6-D control points.
    spline_controls=16
    spline_train=[]
    for s in train_segments:
        can,_=canonicalize(s["a"])
        spline_train.append(spline_encode(can,spline_controls).reshape(-1))
    spline_train=np.asarray(spline_train,np.float32)
    for dim in (8,16,24):
        models.append({"kind":"spline","n_control":spline_controls,
                       "pca":pca_fit(spline_train,dim),"code_dim":dim})
    return models


def make_vq_models(out_root, seg_name, q01, q99, expected_segments):
    """Read compatible impl-g2 VQ-VAE bundles; never trains or edits them."""
    reps=Path(out_root)/"representations"; found=[]
    expected={(int(s["task"]),int(s["episode"]),int(s["start"]),int(s["length"])) for s in expected_segments}
    for mode in ("raw","procrustes"):
        for rdim in (0,4,8):
            source=reps/f"vq_{seg_name}_{mode}_K256"/f"r{rdim}"
            if not (source/"manifest.json").exists():continue
            meta=json.loads((source/"manifest.json").read_text())
            if meta.get("kind")!="vqvae_ema" or meta.get("segment_name")!=seg_name:
                raise ValueError(f"foreign VQ bundle at {source}")
            if np.any(np.isin(meta["fit_tasks"],LONG_TASKS+HELDOUT_TASKS)):
                raise AssertionError("VQ fit manifest contains Long/heldout task")
            sc=np.load(source/"segments.npz",allow_pickle=False);pc=np.load(source/"pose.npz",allow_pickle=False)
            allcodes=np.load(source/"codes.npy",mmap_mode="r",allow_pickle=False);cmap={}
            tt,ee,ss,ll=(sc[k] for k in ("task","episode","start","length"))
            pose_arrays={k:pc[k] for k in pc.files}
            source_keys={(int(t),int(e),int(st),int(ln)) for t,e,st,ln in zip(tt,ee,ss,ll,strict=True)}
            if source_keys!=expected:
                continue
            for i,(t,e,st,ln) in enumerate(zip(tt,ee,ss,ll,strict=True)):
                cmap[(int(t),int(e),int(st),int(ln))]=i
            found.append({"kind":"vq","input_mode":mode,"residual_dim":rdim,
                          "code_dim":32+rdim,"bundle_src":source,"seg_name":seg_name,
                          "q01":np.asarray(meta["q01"],np.float32),"q99":np.asarray(meta["q99"],np.float32),
                          "codes_all":allcodes,"pose_all":pose_arrays,"codes_by_key":cmap})
    return found


def make_ae_models(out_root, seg_name, expected_segments):
    """Load corrected masked-AE bundles from this run into the common M6 grid."""
    reps=Path(out_root)/"representations";found=[]
    expected={(int(s["task"]),int(s["episode"]),int(s["start"]),int(s["length"])) for s in expected_segments}
    for variant in ("raw","procrustes_shape","raw_invariant"):
        for dim in (8,16,24,32):
            source=reps/f"ae_{seg_name}_{variant}_d{dim}"
            if not (source/"manifest.json").exists():continue
            meta=json.loads((source/"manifest.json").read_text())
            if meta.get("kind")!="masked_variable_length_ae" or meta.get("segmentation")!=seg_name:
                raise ValueError(f"foreign AE bundle at {source}")
            if np.any(np.isin(meta["fit_tasks"],LONG_TASKS+HELDOUT_TASKS)):
                raise AssertionError("AE fit manifest contains Long/heldout task")
            sc=np.load(source/"segments.npz",allow_pickle=False);pc=np.load(source/"pose.npz",allow_pickle=False)
            allcodes=np.load(source/"codes.npy",mmap_mode="r",allow_pickle=False);cmap={}
            tt,ee,ss,ll=(sc[k] for k in ("task","episode","start","length"))
            pose_arrays={k:pc[k] for k in pc.files}
            keys={(int(t),int(e),int(st),int(ln)) for t,e,st,ln in zip(tt,ee,ss,ll,strict=True)}
            if keys!=expected:
                raise AssertionError(f"AE keys differ for {seg_name}/{variant}/d{dim}: {len(keys)} vs {len(expected)}")
            for i,(t,e,st,ln) in enumerate(zip(tt,ee,ss,ll,strict=True)):
                cmap[(int(t),int(e),int(st),int(ln))]=(np.asarray(allcodes[i],np.float32),{k:v[i] for k,v in pose_arrays.items()})
            found.append({"kind":"ae","variant":variant,"code_dim":int(dim),"bundle_src":source,
                          "seg_name":seg_name,"codes_by_key":cmap})
    return found


def _load_ae_decoder(bundle):
    """Instantiate a CPU AE decoder once; called lazily for AE bundles."""
    if "ae_model" in bundle:return
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import ae as aemod
    torch,_,_,Model=aemod._torch_classes()
    f=np.load(Path(bundle["bundle_src"])/"model.npz",allow_pickle=False)
    cfg=json.loads(str(f["config_json"].item()));net=Model(7,7,int(cfg["latent_dim"]))
    net.load_state_dict({k:torch.as_tensor(f["weight__"+k]) for k in net.state_dict()});net.eval()
    bundle.update({"ae_model":net,"ae_cfg":cfg,
        "ae_stats":{k:np.asarray(f[k],np.float32) for k in ("xmean","xstd","ymean","ystd")},
        "q01":np.asarray(cfg["q01"],np.float32),"q99":np.asarray(cfg["q99"],np.float32),
        "lmax":int(cfg["lmax"]),
        "reference":(np.load(Path(bundle["bundle_src"])/"procrustes_reference.npy",allow_pickle=False)
                     if (Path(bundle["bundle_src"])/"procrustes_reference.npy").exists() else None)})


def encode_one(model, action):
    kind=model["kind"]
    if kind=="canonical_raw":
        can,pose=canonicalize(action)
        pose["canonical"]=True
        return resample(can,PHASE).reshape(-1),pose
    if kind.startswith("pca_raw"):
        x=raw_pca_input(action,model["q01"],model["q99"],
                        native_length=model.get("pca_input_length")).reshape(-1)
        return pca_encode(model["pca"],x),{}
    if kind in {"pca_canonical","pca_gripsep","procrustes_pca"}:
        can,pose=canonicalize(action,procrustes_ref=model.get("reference"))
        can=resample(can,PHASE)
        if kind=="pca_canonical":x=np.concatenate([can,resample(action[:,6:7],PHASE)],axis=1).reshape(-1)
        else:x=can.reshape(-1)
        pose["canonical"]=True
        return pca_encode(model["pca"],x),pose
    if kind=="dmp":
        can,pose=canonicalize(action);can=resample(can,PHASE);w,g=dmp_encode(can,model["n_basis"]);pose["goal"]=g;pose["tau"]=float(len(action)/10.);pose["canonical"]=True
        return w.reshape(-1),pose
    if kind=="spline":
        can,pose=canonicalize(action)
        pose["canonical"]=True
        control=spline_encode(can,model["n_control"]).reshape(-1)
        return pca_encode(model["pca"],control),pose
    if kind=="m1":
        import torch
        sample=resample(action[:,:7],10)[None]
        row=model["row"]
        state=np.asarray(model["states"][row],np.float32)[None]
        feat=np.asarray(model["features"][row],np.float32)[None]
        b,r=model["codec"].encode(sample,state)
        pose={"length":len(action),"scale":1.,"rotation":np.eye(3,dtype=np.float32),"r":r[0],"grip":resample(action[:,6:7],PHASE)[:,0],"state":state[0],"feature":feat[0]}
        return b[0],pose
    raise ValueError(kind)


def decoded_candidate(model,code,pose,length,mode,qpose,q01,q99):
    kind=model["kind"]
    if kind=="ae":
        _load_ae_decoder(model)
        import torch
        n=int(length);phase=torch.as_tensor(np.linspace(0,1,n,dtype=np.float32)[None])
        lf=torch.as_tensor([[np.log1p(n)/np.log1p(model["lmax"])]],dtype=torch.float32)
        z=torch.as_tensor(np.array(code,dtype=np.float32,copy=True)[None])
        with torch.no_grad(): y=model["ae_model"].decode(z,phase,lf).numpy()[0]
        stats=model["ae_stats"];y=y*stats["ystd"]+stats["ymean"]
        if model["variant"]!="raw":
            import sys
            sys.path.insert(0,str(Path(__file__).resolve().parent))
            import ae as aemod
            y=aemod.shape_repr.apply_pose(y,pose)
            y=normalize_physical(y,model["q01"],model["q99"])
        physical=unnormalize(y,model["q01"],model["q99"])
        if mode=="raw":return np.asarray(physical,np.float32)
        canonical,_=canonicalize(physical);motion=resample(canonical,length)
        grip=resample(physical[:,6:7],length)[:,0]
        if mode=="aligned":motion=apply_pose(motion,qpose)
        return np.column_stack([motion,grip]).astype(np.float32)
    if kind=="vq":
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import vq as vqmod
        if "vq_model" not in model:
            model["vq_model"],_,extras=vqmod._load_model_npz(Path(model["bundle_src"])/"model.npz",device="cpu")
            model["stats"]={"mean":np.asarray(extras["feature_mean"],np.float32),"std":np.asarray(extras["feature_std"],np.float32)}
        import torch
        c=np.array(code,dtype=np.float32,copy=True);q=torch.as_tensor(c[:32][None],dtype=torch.float32)
        residual=torch.as_tensor(c[32:][None],dtype=torch.float32) if model["residual_dim"] else None
        phase=torch.as_tensor(np.linspace(0,1,int(length),dtype=np.float32)[None])
        with torch.no_grad():y=model["vq_model"].decode(q,residual,phase,int(model["residual_dim"])).cpu().numpy()[0]
        y=y*model["stats"]["std"]+model["stats"]["mean"]
        norm=vqmod._restore_sequence(y,model["input_mode"],pose,q01[:7],q99[:7])
        physical=unnormalize(norm,q01[:7],q99[:7])
        if mode=="raw":return physical.astype(np.float32)
        canonical,_=canonicalize(physical)
        motion=resample(canonical,length);grip=resample(physical[:,6:7],length)[:,0]
        if mode=="aligned":motion=apply_pose(motion,qpose)
        return np.column_stack([motion,grip]).astype(np.float32)
    elif kind.startswith("pca_raw"):
        n=int(model.get("pca_input_length") or PHASE)
        raw_norm=pca_decode(model["pca"],code).reshape(n,7)
        raw=unnormalize(raw_norm,q01,q99)
        # Raw decoder is used unchanged for the no-separation comparator.
        if mode=="raw":return resample(raw,length)
        canonical,cp=canonicalize(raw)
        motion=resample(canonical,length)
        grip=resample(raw[:,6:7],length)[:,0]
    elif kind in {"canonical_raw","pca_canonical","pca_gripsep","procrustes_pca"}:
        if kind=="canonical_raw":
            motion=np.asarray(code,np.float32).reshape(PHASE,6)
            grip=resample(np.asarray(pose["grip"],np.float32)[:,None],length)[:,0]
        else:
            raw=pca_decode(model["pca"],code).reshape(PHASE,-1)
            motion=raw[:,:6]
            if kind=="pca_canonical":grip=resample(raw[:,6:7],length)[:,0]
            else:grip=resample(np.asarray(pose["grip"])[:,None],length)[:,0]
        motion=resample(motion,length)
        if mode=="raw": motion=apply_pose(motion,pose)
    elif kind=="dmp":
        weights=np.asarray(code).reshape(model["n_basis"],6)
        B=np.exp(-max(float(model["n_basis"])**1.5,1.)*(np.linspace(0,1,PHASE)[:,None]-np.linspace(0,1,model["n_basis"])[None])**2)
        B=B/np.maximum(B.sum(1,keepdims=True),1e-12)
        phase=np.linspace(0,1,PHASE)[:,None];path=phase*np.asarray(pose["goal"])[None]+B@weights
        motion=resample(np.diff(np.concatenate([np.zeros((1,6)),path],axis=0),axis=0),length)
        grip=resample(np.asarray(pose["grip"])[:,None],length)[:,0]
        if mode=="raw":motion=apply_pose(motion,pose)
    elif kind=="spline":
        control=pca_decode(model["pca"],code).reshape(model["n_control"],6);src=np.linspace(0,1,len(control));dest=np.linspace(0,1,PHASE)
        path=np.stack([np.interp(dest,src,control[:,j]) for j in range(6)],axis=1)
        motion=resample(np.diff(np.concatenate([np.zeros((1,6)),path],axis=0),axis=0),length)
        grip=resample(np.asarray(pose["grip"])[:,None],length)[:,0]
        if mode=="raw":motion=apply_pose(motion,pose)
    elif kind=="m1":
        y=model["codec"].decode(np.asarray(code)[None],np.asarray(pose["r"])[None],np.asarray(pose["feature"])[None],np.asarray(pose["state"])[None])[0]
        normed=np.asarray(y[:,:7],np.float32)
        physical=unnormalize(normed,q01,q99)
        if mode=="raw": return resample(physical,length)
        motion, _ = canonicalize(physical)
        motion=resample(motion,length);grip=resample(physical[:,6:7],length)[:,0]
    else: raise ValueError(kind)
    if mode=="aligned": motion=apply_pose(motion,qpose)
    # type-only intentionally leaves the canonical shape at unit scale and +X.
    return np.column_stack([motion,grip]).astype(np.float32)


def nearest_medoids(query_codes, medoid_codes):
    q=np.asarray(query_codes,np.float32);m=np.asarray(medoid_codes,np.float32)
    qn=np.sum(q*q,axis=1);mn=np.sum(m*m,axis=1)
    d=np.maximum(qn[:,None]+mn[None,:]-2*q@m.T,0.)/max(1,q.shape[1])
    return np.argmin(d,axis=1),d


def interp_window(actions,start,n=10):
    a=np.asarray(actions,np.float32);start=float(start)
    t=np.arange(n,dtype=np.float32)+start
    t=np.clip(t,0,max(len(a)-1,0));i=np.floor(t).astype(int);j=np.minimum(i+1,len(a)-1);u=t-i
    return a[i]*(1-u[:,None])+a[j]*u[:,None]


def make_features_for_segment(model, seg, row_for_key, rows, pooled, states):
    if model["kind"]!="m1":return encode_one(model,seg["a"])
    # M1's public API conditions its residual decoder on a query row. Use the
    # closest source row at this segment start; no label or action target is read.
    rid=row_for_key.get((seg["task"],seg["episode"],seg["start"]),seg["row_id"])
    if rid<0: rid=seg["row_id"]
    if rid<0: raise ValueError("M1 segment lacks a dataset row for feature/state")
    model["row"]=rid; model["states"]=rows["state"];model["features"]=pooled
    return encode_one(model,seg["a"])


def make_row_lookup(rows):
    return {(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows["task"],rows["episode"],rows["frame"],strict=True))}


def first_row_in_segment(seg, lookup):
    for f in range(seg["start"],seg["start"]+seg["length"]):
        rid=lookup.get((seg["task"],seg["episode"],f))
        if rid is not None:return int(rid)
    return -1


def conditioning_row_for_segment(seg, lookup, *, search_radius=50):
    """Use the first in-segment row, else the nearest preceding row in-episode."""
    rid=first_row_in_segment(seg,lookup)
    if rid>=0:return rid
    for delta in range(1,search_radius+1):
        rid=lookup.get((seg["task"],seg["episode"],seg["start"]-delta))
        if rid is not None:return int(rid)
    for delta in range(1,search_radius+1):
        rid=lookup.get((seg["task"],seg["episode"],seg["start"]+seg["length"]+delta-1))
        if rid is not None:return int(rid)
    return -1


def fit_m1_wrappers(rows, pooled, fit_ids, q01, q99, cache_dir=None):
    wrappers=[]
    import torch
    import sys
    methods=Path(__file__).resolve().parents[3]/"emergent"/"light"/"methods"
    sys.path.insert(0,str(methods.parent.parent.parent))
    from emergent.light.methods.m1 import M1  # noqa: F401 - needed for cached pickle import
    cache_dir=Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:cache_dir.mkdir(parents=True,exist_ok=True)
    for bdim in (12,96):
        cache=cache_dir/f"m1_b{bdim}.pt" if cache_dir is not None else None
        if cache is not None and cache.exists():
            codec=torch.load(cache,map_location="cpu",weights_only=False)
        else:
            codec=fit_m1(rows,pooled,fit_ids,bdim)
            if cache is not None:
                tmp=cache.with_suffix(".tmp")
                torch.save(codec,tmp);tmp.replace(cache)
        wrappers.append({"kind":"m1","codec":codec,"base_dim":bdim,
                         "states":rows["state"],"features":pooled,"q01":q01,"q99":q99,
                         "normalizer":lambda a,lo=q01,hi=q99: normalize_physical(a,lo,hi),
                         "code_dim":bdim})
    return wrappers


def pose_for(model, action):
    if model["kind"]=="procrustes_pca":
        _,p=canonicalize(action,procrustes_ref=model["reference"]);return p
    _,p=canonicalize(action);return p


def encode_segment(model, seg, row_lookup, rows, pooled):
    kind=model["kind"]
    if kind=="ae":
        key=(int(seg["task"]),int(seg["episode"]),int(seg["start"]),int(seg["length"]))
        if key in model.get("codes_by_key",{}):
            c,p=model["codes_by_key"][key]
            return np.asarray(c,np.float32),{k:np.asarray(v) for k,v in p.items()}
        _load_ae_decoder(model)
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import ae as aemod
        import torch
        raw=np.asarray(seg["a"],np.float32)[:,:7]
        if model["variant"]=="raw":
            x=normalize_physical(raw,model["q01"],model["q99"])
            pose={"scale":1.,"rotation":np.eye(3,dtype=np.float32),"length":len(raw)}
        elif model["variant"]=="procrustes_shape":
            if model.get("reference") is None:raise ValueError("Procrustes AE lacks its Train-only reference")
            x,pose=aemod.canonicalize_7(raw,model["reference"])
        elif model["variant"]=="raw_invariant":
            if model.get("reference") is None:raise ValueError("invariant AE lacks its Train-only reference")
            x=raw.copy();_,pose=aemod.canonicalize_7(raw,model["reference"])
        else:raise ValueError(model["variant"])
        st=model["ae_stats"];lmax=model["lmax"]
        xb,_,phase,mask,lf=aemod._pad_batch([x],[x],st["xmean"],st["xstd"],
            st["ymean"],st["ystd"],lmax,"cpu")
        with torch.no_grad():code=model["ae_model"].encode(xb,phase,mask,lf).cpu().numpy()[0]
        return np.asarray(code,np.float32),pose
    if kind=="vq":
        key=(int(seg["task"]),int(seg["episode"]),int(seg["start"]),int(seg["length"]))
        cached=model.get("codes_by_key",{}).get(key)
        if cached is not None:
            if isinstance(cached,tuple):return cached
            pose={k:v[cached] for k,v in model["pose_all"].items()}
            return np.asarray(model["codes_all"][cached],np.float32),pose
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import vq as vqmod
        x,pose=vqmod._seq_input(seg["a"],model["input_mode"],model["q01"],model["q99"],model.get("reference"))
        enc=vqmod._encode_sequences(model["vq_model"],[x],model["stats"],batch_size=1,device="cpu")
        code=np.concatenate([enc["z"][0],enc["res"][str(model["residual_dim"])][0]],axis=0) if model["residual_dim"] else enc["z"][0]
        return np.asarray(code,np.float32),pose
    if kind!="m1":return encode_one(model,seg["a"])
    import torch
    rid=conditioning_row_for_segment(seg,row_lookup)
    if rid<0:rid=int(seg["row_id"])
    if rid<0:raise ValueError("no source row for M1 conditioning")
    # M1 was trained on the saved fixed [10,32] run windows. For a variable
    # segment, its first overlapping global row is the defined M1 control.
    action=np.asarray(rows["actions"][rid],np.float32)[None]
    state=np.asarray(rows["state"][rid],np.float32)[None]
    with torch.no_grad():
        b,r=model["codec"].encode(action,state)
    as_numpy=lambda x: x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)
    pose={"length":len(seg["a"]),"scale":1.,"rotation":np.eye(3,dtype=np.float32),
          "r":as_numpy(r[0]),"grip":resample(seg["a"][:,6:7],PHASE)[:,0],
          "state":state[0],"feature":np.asarray(pooled[rid],np.float32)}
    return as_numpy(b[0]).astype(np.float32),pose


def save_model_bundle(out, model, q01, q99):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    kind=model["kind"]
    arrays={}
    metadata={"kind":kind,"q01":np.asarray(q01).tolist(),"q99":np.asarray(q99).tolist(),
              "phase_points":PHASE,"outputs_physical_action_shape":[None,7]}
    if kind=="ae":
        import shutil
        src=Path(model["bundle_src"])
        shutil.copy2(src/"model.npz",out/"model.npz")
        if (src/"procrustes_reference.npy").exists():
            shutil.copy2(src/"procrustes_reference.npy",out/"procrustes_reference.npy")
        metadata.update({"variant":model["variant"],"code_dim":int(model["code_dim"]),
                         "ae_source_relative":os.path.relpath(src,out)})
        (out/"reconstruct.py").write_text(
            'from pathlib import Path\nfrom ae import decode_bundle\nBUNDLE=Path(__file__).parent\n'
            'def decode(codes,pose,length): return decode_bundle(BUNDLE,codes,pose,length)\n')
        return metadata
    if kind=="vq":
        import shutil
        shutil.copy2(Path(model["bundle_src"])/"model.npz",out/"model.npz")
        metadata.update({"vq_source_relative":os.path.relpath(model["bundle_src"],out),
                         "input_mode":model["input_mode"],"residual_dim":int(model["residual_dim"]),
                         "code_dim":int(model["code_dim"])})
        (out/"reconstruct.py").write_text(
            'from pathlib import Path\nfrom vq import decode_bundle\nBUNDLE=Path(__file__).parent\n'
            'def decode(codes,pose,length): return decode_bundle(BUNDLE,codes,pose,length)\n')
        return metadata
    if "pca" in model:
        arrays.update({"mean":model["pca"]["mean"],"components":model["pca"]["components"],
                       "variance":model["pca"]["variance"],"whiten":np.asarray(model["pca"]["whiten"])})
        if model.get("reference") is not None:arrays["reference"]=model["reference"]
        metadata.update({"k":int(model.get("k",model.get("code_dim",0))),"whiten":bool(model["pca"]["whiten"])})
    if kind in {"pca_raw", "pca_raw_white"}:
        metadata.update({"pca_input_space":"official_q01_q99_normalized_active7",
                         "pca_input_length":model.get("pca_input_length")})
    if kind=="dmp":metadata["n_basis"]=int(model["n_basis"])
    if kind=="spline":metadata["n_control"]=int(model["n_control"])
    if kind=="m1":
        import torch
        codec=model["codec"]
        payload={"base_dim":model["base_dim"],"residual_dim":4,
                 "stats":{k:torch.tensor(getattr(codec,k)) for k in ["am","ass","sm","ss","fm","fs"]},
                 "nets":{k:getattr(codec,k).state_dict() for k in ["base_decoder","residual_decoder"]}}
        torch.save(payload,out/"codec.pt")
        metadata["codec"]="codec.pt (torch state_dict); codes store b only, reconstruction pose stores posterior residual and conditioning context for autoencoder control"
    np.savez_compressed(out/"model.npz",**arrays)
    (out/"reconstruct.py").write_text(
        'from pathlib import Path\nfrom shape_skills.representations import decode_from_bundle as _decode\n'
        'def decode(codes, pose, length):\n    return _decode(Path(__file__).parent, codes, pose, length)\n')
    return metadata


class _FrozenM1Decoder:
    """Minimal CPU decoder loaded from the M1 checkpoint for artifact readers."""
    def __init__(self, payload):
        import torch
        self.torch=torch
        self.stats={k:torch.as_tensor(v,dtype=torch.float32) for k,v in payload["stats"].items()}
        self.nets={}
        for name,sd in payload["nets"].items():
            dims=sd["0.weight"].shape[1]
            layers=[]
            for ix in (0,2,4):
                w=sd[f"{ix}.weight"];b=sd[f"{ix}.bias"]
                layer=torch.nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(b)
                layers.extend([layer,torch.nn.GELU()] if ix<4 else [layer])
            self.nets[name]=torch.nn.Sequential(*layers).eval()
    def decode(self,b,r,f,s):
        torch=self.torch
        with torch.no_grad():
            bt=torch.as_tensor(b,dtype=torch.float32);rt=torch.as_tensor(r,dtype=torch.float32)
            st=(torch.as_tensor(s,dtype=torch.float32)-self.stats["sm"])/self.stats["ss"]
            base=self.nets["base_decoder"](torch.cat([bt,st],dim=1))
            residual=self.nets["residual_decoder"](rt)
            y=(base+residual)*self.stats["ass"]+self.stats["am"]
            return y.reshape(len(y),10,32).cpu().numpy()


def load_eval_bundle(out_root, config, segments_by_name, episodes):
    """Reload only one selected configuration after Dev selection."""
    root=Path(out_root)/"representations"/config
    meta=json.loads((root/"manifest.json").read_text())
    codes=np.load(root/"codes.npy",mmap_mode="r",allow_pickle=False)
    poses=load_pose_rows(root/"pose.npz",len(codes))
    sm=np.load(root/"segments.npz",allow_pickle=False)
    mem=[]
    for t,e,start,length,rowid in zip(sm["task"],sm["episode"],sm["start"],sm["length"],sm["row_ids"],strict=True):
        seq=episodes[(int(t),int(e))]
        mem.append({"task":int(t),"episode":int(e),"start":int(start),"length":int(length),"row_id":int(rowid),
                    "a":seq["actions_phys"][int(start):int(start)+int(length)]})
    model_file=np.load(root/"model.npz",allow_pickle=False)
    model={k:model_file[k] for k in model_file.files};model["kind"]=meta["kind"]
    nested=meta["model"]
    if model["kind"]=="vq":
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import vq as vqmod
        src=(root/nested["vq_source_relative"]).resolve()
        vq_model,vqmeta,extras=vqmod._load_model_npz(root/"model.npz",device="cpu")
        model={"kind":"vq","input_mode":nested["input_mode"],"residual_dim":int(nested["residual_dim"]),
               "code_dim":int(nested["code_dim"]),"bundle_src":root,"vq_model":vq_model,
               "stats":{"mean":np.asarray(extras["feature_mean"],np.float32),"std":np.asarray(extras["feature_std"],np.float32)},
               "q01":np.asarray(vqmeta["q01"],np.float32),"q99":np.asarray(vqmeta["q99"],np.float32),"codes_by_key":{}}
        sc=np.load(src/"segments.npz",allow_pickle=False);pc=np.load(src/"pose.npz",allow_pickle=False)
        allcodes=np.load(src/"codes.npy",mmap_mode="r",allow_pickle=False)
        tt,ee,ss,ll=(sc[k] for k in ("task","episode","start","length"))
        p_arrays={k:pc[k] for k in pc.files}
        for i,(t,e,st,ln) in enumerate(zip(tt,ee,ss,ll,strict=True)):
            key=(int(t),int(e),int(st),int(ln))
            model["codes_by_key"][key]=(np.asarray(allcodes[i],np.float32),{k:v[i] for k,v in p_arrays.items()})
        expected={(int(s["task"]),int(s["episode"]),int(s["start"]),int(s["length"])) for s in segments_by_name}
        if set(model["codes_by_key"])!=expected:
            raise AssertionError("VQ code/pose source keys failed segment integrity check")
        model["reference"]=None
    elif model["kind"]=="ae":
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import ae as aemod
        import torch
        f=np.load(root/"model.npz",allow_pickle=False);cfg=json.loads(str(f["config_json"].item()))
        _,_,_,Model=aemod._torch_classes();net=Model(7,7,int(cfg["latent_dim"]))
        net.load_state_dict({k:torch.as_tensor(f["weight__"+k]) for k in net.state_dict()});net.eval()
        sc=np.load(root/"segments.npz",allow_pickle=False);pc=np.load(root/"pose.npz",allow_pickle=False)
        allcodes=np.load(root/"codes.npy",mmap_mode="r",allow_pickle=False);p_arrays={k:pc[k] for k in pc.files}
        cmap={}
        for i,(t,e,st,ln) in enumerate(zip(sc["task"],sc["episode"],sc["start"],sc["length"],strict=True)):
            cmap[(int(t),int(e),int(st),int(ln))]=(np.asarray(allcodes[i],np.float32),{k:v[i] for k,v in p_arrays.items()})
        model={"kind":"ae","variant":str(nested["variant"]),"code_dim":int(nested["code_dim"]),
               "bundle_src":root,"codes_by_key":cmap,"ae_model":net,
               "ae_cfg":cfg,"ae_stats":{k:np.asarray(f[k],np.float32) for k in ("xmean","xstd","ymean","ystd")},
               "q01":np.asarray(cfg["q01"],np.float32),"q99":np.asarray(cfg["q99"],np.float32),
               "lmax":int(cfg["lmax"])}
        src_rel=nested.get("ae_source_relative")
        if src_rel:
            src=(root/src_rel).resolve()
            asc=np.load(src/"segments.npz",allow_pickle=False);apc=np.load(src/"pose.npz",allow_pickle=False)
            acodes=np.load(src/"codes.npy",mmap_mode="r",allow_pickle=False);apose={k:apc[k] for k in apc.files}
            model["codes_by_key"]={
                (int(t),int(e),int(st),int(ln)):(np.asarray(acodes[i],np.float32),{k:v[i] for k,v in apose.items()})
                for i,(t,e,st,ln) in enumerate(zip(asc["task"],asc["episode"],asc["start"],asc["length"],strict=True))}
        model["reference"]=(np.load(root/"procrustes_reference.npy",allow_pickle=False)
                            if (root/"procrustes_reference.npy").exists() else None)
    else:
        src=None
    if model["kind"]!="vq" and "mean" in model:
        model["pca"]={"mean":model["mean"],"components":model["components"],"variance":model["variance"],
                      "whiten":bool(model["whiten"])}
        model["reference"]=model.get("reference")
        model["k"]=int(nested.get("k",meta["code_dim"]))
    if model["kind"] in {"pca_raw", "pca_raw_white"}:
        model["q01"]=np.asarray(meta["q01"],np.float32)
        model["q99"]=np.asarray(meta["q99"],np.float32)
        model["pca_input_length"]=nested.get("pca_input_length")
    if model["kind"]=="dmp":model["n_basis"]=int(nested["n_basis"])
    if model["kind"]=="spline":model["n_control"]=int(nested["n_control"])
    if model["kind"]=="m1":
        import torch
        import sys
        emergent_root=Path(__file__).resolve().parents[3]
        if str(emergent_root) not in sys.path:sys.path.insert(0,str(emergent_root))
        # Query segments must be encoded by the same frozen M1 encoder before
        # comparison with the memory codes. The compact representation bundle
        # stores only the decoder, so use the read-only full fit cache created
        # once during Dev evaluation. This does not refit any parameter.
        base_dim=int(nested.get("base_dim",meta["code_dim"]))
        full_codec=Path(out_root)/"fit-cache"/f"m1_b{base_dim}.pt"
        if not full_codec.exists():
            raise FileNotFoundError(f"M1 query encoding requires frozen fit cache: {full_codec}")
        model["codec"]=torch.load(full_codec,map_location="cpu",weights_only=False)
        model["codec"].encoder.eval()
        model["codec"].base_decoder.eval()
        model["codec"].residual_decoder.eval()
        model["q01"]=np.asarray(meta["q01"],np.float32);model["q99"]=np.asarray(meta["q99"],np.float32)
    mfile=np.load(root/"m6.npz",allow_pickle=False)
    labels=np.asarray(mfile["row_units"],np.int32);members=[np.flatnonzero(labels==u).astype(np.int64) for u in range(int(labels.max())+1)]
    m6={"row_units":labels,"members_local":members,"codes":codes,
        "support":mfile["support"],"epsilon":float(mfile["epsilon"]),"registered":len(members),
        "medoid_local":np.asarray(mfile["medoid_local"],np.int64)}
    return model,codes,poses,mem,m6


def write_representation(out_root, name, seg_name, model, mem_segments, mem_codes, mem_poses,
                         m6_rows, row_units, q01, q99, config_meta):
    root=Path(out_root)/"representations"/name
    root.mkdir(parents=True,exist_ok=True)
    mseg=mem_segments
    np.savez_compressed(root/"segments.npz",task=np.asarray([s["task"] for s in mseg],np.int32),
        episode=np.asarray([s["episode"] for s in mseg],np.int32),start=np.asarray([s["start"] for s in mseg],np.int32),
        length=np.asarray([s["length"] for s in mseg],np.int32),row_ids=np.asarray([s["row_id"] for s in mseg],np.int64))
    np.save(root/"codes.npy",np.asarray(mem_codes,np.float32),allow_pickle=False)
    arrays={"scale":np.asarray([p.get("scale",1.) for p in mem_poses],np.float32),
            "rotation":np.asarray([p.get("rotation",np.eye(3)) for p in mem_poses],np.float32),
            "length":np.asarray([p.get("length",s["length"]) for p,s in zip(mem_poses,mseg,strict=True)],np.int32),
            "grip":np.asarray([p.get("grip",np.zeros(PHASE)) for p in mem_poses],np.float32)}
    for field in ("goal","tau","r","state","feature"):
        if all(field in p for p in mem_poses):arrays[field]=np.asarray([p[field] for p in mem_poses],np.float32)
    np.savez_compressed(root/"pose.npz",**arrays)
    model_meta=save_model_bundle(root,model,q01,q99)
    pose_counts=np.asarray([sum(np.asarray(v).size for v in p.values() if np.asarray(v).dtype.kind in "fiu") for p in mem_poses],np.int64)
    code_per_step=np.asarray(mem_codes).shape[1]/np.maximum(np.asarray([s["length"] for s in mem_segments],np.float32),1)
    write_json(root/"manifest.json",{"version":1,"representation":name,"segmentation":seg_name,
        "kind":model["kind"],"fit_tasks_excluded":list(LONG_TASKS+HELDOUT_TASKS),
        "pool":"memory segments from non-Long, non-heldout tasks only","memory_segments":len(mseg),
        "code_dim":int(np.asarray(mem_codes).shape[1]),"m6_registered":int(m6_rows["registered"]),
        "code_values_per_step_mean":float(np.mean(code_per_step)),"code_values_per_step_median":float(np.median(code_per_step)),
        "pose_numeric_values_per_segment_mean":float(np.mean(pose_counts)),
        "pose_numeric_values_per_segment_median":float(np.median(pose_counts)),
        "m6_epsilon":float(m6_rows["epsilon"]),"m6_kappa":float(config_meta["kappa"]),
        "m6_min_support":10,"m6_min_episodes":3,"m6_seed":20261104,
        "m6_medoid_segment_indices":np.asarray([medoid_generic(m,m6_rows["codes"]) for m in m6_rows["members_local"]],np.int64).tolist(),
        "q01":np.asarray(q01).tolist(),"q99":np.asarray(q99).tolist(),"physical_transform":"q01/q99 unnormalize before SO(3) rotation and scale; same R on delta-position and delta-rotation; gripper untouched",
        "model":model_meta,"seed":20261104})
    np.savez_compressed(root/"m6.npz",row_units=row_units,medoid_local=np.asarray([medoid_generic(m,m6_rows["codes"]) for m in m6_rows["members_local"]],np.int64),
        support=m6_rows["support"],epsilon=np.asarray(m6_rows["epsilon"],np.float32),kappa=np.asarray(config_meta["kappa"],np.float32))


def segment_for_frame(seglist, frame):
    # Segments are sorted and non-overlapping within an episode.
    starts=np.asarray([s["start"] for s in seglist],np.int64)
    i=int(np.searchsorted(starts,int(frame),side="right")-1)
    if i<0 or int(frame)>=seglist[i]["start"]+seglist[i]["length"]:return None
    return i


def ensure_query_segment_coverage(q_segments_by_ep, query_row_ids, rows, episodes, *, window=10):
    """Add query-only fixed windows for evaluation rows outside a segment.

    Some variable-length segmentations omit a terminal remainder shorter than
    their minimum. A stored 10-step evaluation row can then fall outside every
    query segment. This helper creates an exact 10-step query segment only for
    those rows; it never adds anything to the memory pool or fit set.
    """
    result={k:list(v) for k,v in q_segments_by_ep.items()}
    tasks=np.asarray(rows["task"]);eps=np.asarray(rows["episode"]);frames=np.asarray(rows["frame"])
    fallback=[]
    for rid in np.asarray(query_row_ids,np.int64):
        key=(int(tasks[rid]),int(eps[rid]));frame=int(frames[rid])
        seq=episodes[key]
        seglist=result.setdefault(key,[])
        if any(int(s["start"])<=frame<int(s["start"])+int(s["length"]) for s in seglist):continue
        if frame+window>len(seq["frame"]):
            raise AssertionError(f"query row {rid} lacks a complete {window}-step action window")
        seglist.append({"task":key[0],"episode":key[1],"start":frame,"length":window,
                        "row_id":int(rid),"a":seq["actions_phys"][frame:frame+window],
                        "meta":seq["meta"]})
        fallback.append(int(rid))
    for key in result:result[key].sort(key=lambda s:s["start"])
    return result,np.asarray(fallback,np.int64)


def load_pose_rows(path, n_rows):
    """Materialize each compressed NPZ pose member once, then expose rows."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return [
        {k: (v[i] if v.ndim > 0 and v.shape[0] == n_rows else v) for k, v in arrays.items()}
        for i in range(n_rows)
    ]


def select_dev_rows(rows):
    tasks=np.asarray(rows["task"]);eps=np.asarray(rows["episode"]);frames=np.asarray(rows["frame"])
    valid=np.asarray(rows["validation"],bool)
    base=np.flatnonzero(valid & np.isin(tasks,HELDOUT_TASKS))
    chosen=[]
    for key in sorted(set(zip(tasks[base].tolist(),eps[base].tolist()))):
        ix=base[(tasks[base]==key[0])&(eps[base]==key[1])]
        ix=ix[np.argsort(frames[ix])]
        chosen.extend(ix[::4].tolist())
    return np.asarray(chosen,np.int64)


def eval_config(model, codes, poses, mem_segments, m6, q_segments_by_ep,
                query_row_ids, rows, episodes, row_lookup, pooled, q01, q99, *, seed=8401):
    if not len(query_row_ids) or not len(m6["members_local"]):return None
    med_local=np.asarray([medoid_generic(m,codes) for m in m6["members_local"]],np.int64)
    med_codes=codes[med_local]
    mem_task=np.asarray([s["task"] for s in mem_segments],np.int32)
    actual=np.asarray(rows["actions"])
    row_task=np.asarray(rows["task"]);row_ep=np.asarray(rows["episode"]);row_frame=np.asarray(rows["frame"])
    # Only encode segments that contain an evaluated row. In particular, the
    # Long source contains many episodes/windows that are not part of the fixed
    # 548-row Eval mask; encoding every one for a context-conditioned M1 codec
    # needlessly retains large feature/state sidecars. This filtering does not
    # change candidate or query values for any evaluated row.
    needed_query_segments=set()
    for rid in np.asarray(query_row_ids,np.int64):
        key=(int(row_task[rid]),int(row_ep[rid]))
        seglist=q_segments_by_ep.get(key,[])
        loc=segment_for_frame(seglist,int(row_frame[rid]))
        if loc is not None:
            needed_query_segments.add((key,loc))
    # One code and pose per query segment; a single medoid unit is reused for
    # all ten-frame windows in that query segment (no per-window reselection).
    qinfo={}
    for key,local in sorted(needed_query_segments):
        seg=q_segments_by_ep[key][local]
        code,pose=encode_segment(model,seg,row_lookup,rows,pooled)
        qinfo[(key,local)]=(np.asarray(code,np.float32),pose,seg)
    result={"row_ids":[],"task":[],"episode":[],"mse_reconstruction":[],"mse_type":[],"mse_aligned":[],"mse_raw":[],
            "mse_random_aligned":[],"mse_random_type":[],"mse_selected_code":[],"unit":[],"random_unit":[]}
    rng=np.random.default_rng(seed)
    qdecoded={}
    # Cache unit->medoid decode once per window length; variable lengths are small.
    for rid in np.asarray(query_row_ids,np.int64):
        key=(int(row_task[rid]),int(row_ep[rid]));seglist=q_segments_by_ep.get(key,[])
        loc=segment_for_frame(seglist,int(row_frame[rid]))
        if loc is None:continue
        qcode,qpose,qseg=qinfo[(key,loc)]
        qcache=(key,loc)
        if qcache not in qdecoded:
            qdecoded[qcache]=decoded_candidate(model,qcode,qpose,qseg["length"],"raw",
                                               pose_for(model,qseg["a"]),q01,q99)
        d=np.mean((med_codes-qcode[None])**2,axis=1)
        uid=int(np.argmin(d));r_uid=int(rng.integers(len(med_local)))
        sel=int(med_local[uid]);rnd=int(med_local[r_uid])
        target=unnormalize(actual[rid,:,:7],q01,q99)
        targetn=np.asarray(actual[rid,:,:7],np.float32)
        query_pose=pose_for(model,qseg["a"])
        rel=float(row_frame[rid]-qseg["start"])
        def make_window(ix,mode):
            seg=mem_segments[ix];p=poses[ix]
            pred=decoded_candidate(model,codes[ix],p,seg["length"],mode,query_pose,q01,q99)
            if mode in {"type","aligned"}:
                # decoded_candidate returns physical values; apply the fitted
                # query pose for aligned mode, then official q01/q99 normalize.
                pass
            scale=(seg["length"]/max(qseg["length"],1))
            offset=rel if mode=="raw" else rel*scale
            return interp_window(pred,offset,10)
        selected_type=make_window(sel,"type")
        selected_aligned=make_window(sel,"aligned")
        selected_raw=make_window(sel,"raw")
        random_type=make_window(rnd,"type")
        random_aligned=make_window(rnd,"aligned")
        # Keep all comparisons in actual normalized active7 space.
        type_n=normalize_physical(selected_type,q01,q99)
        aligned_n=normalize_physical(selected_aligned,q01,q99)
        raw_n=normalize_physical(selected_raw,q01,q99)
        rand_type_n=normalize_physical(random_type,q01,q99)
        rand_align_n=normalize_physical(random_aligned,q01,q99)
        recon_window=interp_window(qdecoded[qcache],rel,10)
        recon_n=normalize_physical(recon_window,q01,q99)
        def mse(x):return float(np.mean((np.asarray(x)[:,:7]-targetn)**2))
        result["row_ids"].append(int(rid));result["task"].append(key[0]);result["episode"].append(key[1])
        result["mse_reconstruction"].append(mse(recon_n))
        result["mse_type"].append(mse(type_n));result["mse_aligned"].append(mse(aligned_n));result["mse_raw"].append(mse(raw_n))
        result["mse_random_type"].append(mse(rand_type_n));result["mse_random_aligned"].append(mse(rand_align_n))
        result["mse_selected_code"].append(float(d[uid]));result["unit"].append(uid);result["random_unit"].append(r_uid)
    for k in result:
        result[k]=np.asarray(result[k])
    return result


def metrics_summary(res, hard=None):
    if res is None:return {}
    out={}
    for label,mask in (("all",np.ones(len(res["row_ids"]),bool)),("hard",np.asarray(hard,bool) if hard is not None else np.zeros(len(res["row_ids"]),bool))):
        if len(mask)!=len(res["row_ids"]):mask=np.ones(len(res["row_ids"]),bool)
        out[label]={}
        for k in ("mse_reconstruction","mse_type","mse_aligned","mse_raw","mse_random_type","mse_random_aligned"):
            v=res[k][mask]
            out[label][k]={"mean":float(np.mean(v)) if len(v) else None,"median":float(np.median(v)) if len(v) else None,
                           "le_0p01":float(np.mean(v<=.01)) if len(v) else None,
                           "le_0p02":float(np.mean(v<=.02)) if len(v) else None,
                           "le_0p04":float(np.mean(v<=.04)) if len(v) else None}
        out[label]["aligned_minus_random_mean"]=float(np.mean(res["mse_aligned"][mask]-res["mse_random_aligned"][mask])) if np.any(mask) else None
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,default=DEFAULT_RUN);ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--checkpoint",type=Path,default=CHECKPOINT);ap.add_argument("--resume",action="store_true")
    ap.add_argument("--dev-only",action="store_true",help="fit/save the grid and Dev curves but defer Long to an explicit final pass")
    args=ap.parse_args();out=args.out
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True,exist_ok=True);(out/"representations").mkdir(exist_ok=True)
    (out/"figures").mkdir(parents=True,exist_ok=True)
    progress=out/"progress.jsonl"
    episodes,rows,data_meta=load_source(args.run,args.checkpoint)
    q01=np.asarray(data_meta["q01"],np.float32);q99=np.asarray(data_meta["q99"],np.float32)
    pooled=np.load(args.run/"pooled.npy",mmap_mode="r",allow_pickle=False)
    row_lookup=make_row_lookup(rows);row_groups_map=row_groups(rows)
    heldout_dev_keys=[k for k,v in episodes.items() if k[0] in HELDOUT_TASKS and v["meta"]["validation"]]
    if not any(t in LONG_TASKS for t,e in episodes):raise AssertionError("source is missing the Long evaluation episodes")
    segment_sets,segment_stats,thresholds=make_segment_sets(episodes,rows,progress)
    hard_path=ROOT/"topk-long-01"/"production"/"long_hard_rows.npz"
    if not hard_path.exists():
        # Try the normalized fixed row/hard mask artifacts used by fair_expert.
        hard_path=ROOT/"fair-expert-01"/"evaluation-final"/"offline_predictions.npz"
    hardz=np.load(hard_path,allow_pickle=False)
    long_rows=np.asarray(hardz["row_ids"],np.int64)
    long_tasks=np.asarray(rows["task"])[long_rows]
    long_rows=long_rows[long_tasks<10]
    if len(long_rows)!=548:raise AssertionError(f"Long Eval row count must be 548, got {len(long_rows)}")
    if "hard_mask" in hardz:hard_mask=np.asarray(hardz["hard_mask"],bool)[np.asarray(hardz["task_ids"])<10]
    else:
        hard_file=ROOT/"fair-expert-01"/"evaluation-final"/"offline_predictions.npz"
        hz=np.load(hard_file,allow_pickle=False);hard_mask=np.asarray(hz["hard_mask"],bool)
    if int(hard_mask.sum())!=63:raise AssertionError(f"Long hard mask expected 63, got {int(hard_mask.sum())}")
    dev_rows=select_dev_rows(rows)
    # Fit rows are strictly non-Long, non-heldout, non-validation, non-memory.
    tasks=np.asarray(rows["task"]);fit_rows=np.flatnonzero(training_row_mask(rows))
    if np.any(np.isin(tasks[fit_rows],LONG_TASKS+HELDOUT_TASKS)):raise AssertionError("Long/heldout leaked into representation fit")
    protocol={"run":str(args.run),"out":str(out),"data":data_meta,"fit_rows":len(fit_rows),"dev_rows":len(dev_rows),
        "long_eval_rows":len(long_rows),"long_hard_rows":int(hard_mask.sum()),"heldout_tasks":list(HELDOUT_TASKS),
        "long_tasks":list(LONG_TASKS),"fit_exclusion_assert":True,"threshold_selection":thresholds,
        "segment_stats":segment_stats,"kappas":KAPPAS,"seed":20261104,
        "selection_rule":"mandatory S0-R1 raw PCA k24 and M1-b12; add <=4 by rank(MSE aligned asc)+rank(coverage<=.04 desc), at most one per family; Dev only"}
    if not (out/"protocol.json").exists():write_json(out/"protocol.json",protocol)
    if not (out/"thresholds.json").exists():write_json(out/"thresholds.json",thresholds)
    append_progress(progress,event="input_ready",episodes=len(episodes),fit_rows=len(fit_rows),dev_rows=len(dev_rows),long_rows=len(long_rows))
    # M1 b12/b96 are each fit once on the same fixed training-row split, then
    # queried with variable segments phase-normalized to ten steps.
    m1_models=fit_m1_wrappers(rows,pooled,fit_rows,q01,q99,out/"fit-cache")
    for m in m1_models:
        m["states"]=rows["state"];m["features"]=pooled
        m["normalizer"]=lambda a,lo=q01,hi=q99:normalize_physical(a,lo,hi)
    dev_results=[]; saved_curves={}; rep_stats=[]; config_cache={}
    for sname,segs in segment_sets.items():
        train=[s for s in segs if s["meta"].get("present",False) and s["task"] not in LONG_TASKS+HELDOUT_TASKS and not s["meta"]["memory"] and not s["meta"]["validation"]]
        mem=[s for s in segs if s["meta"].get("present",False) and s["task"] not in LONG_TASKS+HELDOUT_TASKS and s["meta"]["memory"]]
        devsegs=[s for s in segs if s["task"] in HELDOUT_TASKS and s["meta"]["validation"]]
        if not train or not mem:continue
        # Save raw as the execution-space baseline plus all requested shape codes.
        models=make_rep_models(train,rows,pooled,fit_rows,segmentation_name=sname,q01=q01,q99=q99)
        models += m1_models
        models += make_vq_models(out,sname,q01,q99,segs)
        models += make_ae_models(out,sname,segs)
        # Query segments indexed by episode for Dev and Long; full episodes are
        # used only to define phase and action history, never to fit transforms.
        all_query_segs={}
        for s in segs:
            if s["task"] in HELDOUT_TASKS or s["task"] in LONG_TASKS:
                all_query_segs.setdefault((s["task"],s["episode"]),[]).append(s)
        for key in all_query_segs:all_query_segs[key].sort(key=lambda s:s["start"])
        dev_episode_keys=set((int(tasks[r]),int(rows["episode"][r])) for r in dev_rows)
        devq=np.asarray([r for r in dev_rows if (int(tasks[r]),int(rows["episode"][r])) in dev_episode_keys],np.int64)
        seg_memory_ids=np.asarray([s["row_id"] for s in mem],np.int64)
        if any(s["task"] in LONG_TASKS+HELDOUT_TASKS for s in mem):raise AssertionError("Long/heldout memory leak")
        for model in models:
            family=model["kind"]
            suffix=(f"{model['input_mode']}K256r{model['residual_dim']}" if family=="vq" else
                    f"{model['variant']}_d{int(model['code_dim'])}" if family=="ae" else
                    int(model.get('code_dim',model.get('k',model.get('n_basis',model.get('n_control',model.get('base_dim',0)))))))
            cfg=(f"{sname}__{family}__{suffix}" if family=="ae" else f"{sname}__{family}__d{suffix}")
            repdir=out/"representations"/cfg
            if (repdir/"dev_curve.json").exists() and (repdir/"m6.npz").exists() and (repdir/"manifest.json").exists():
                curve=json.loads((repdir/"dev_curve.json").read_text())
                mfile=np.load(repdir/"m6.npz",allow_pickle=False)
                labels=np.asarray(mfile["row_units"],np.int32)
                for c in curve:
                    dev_results.append({"config":cfg,"segmentation":sname,"family":family,
                        "code_dim":int(json.loads((repdir/"manifest.json").read_text())["code_dim"]),
                        "kappa":float(c["kappa"]),"units":int(c["units"]),
                        "dev_aligned_mse":float(c["dev_aligned_mse"]),
                        "dev_aligned_le_0p04":float(c["dev_aligned_le_0p04"]),
                        "dev_random_aligned_mse":float(c.get("dev_aligned_mse",0.)-c.get("dev_aligned_minus_random",0.))})
                config_cache[cfg]={"seg_name":sname}
                rep_stats.append({"config":cfg,"memory_segments":len(labels),"train_segments":len(train),
                    "dev_segments":len(devsegs),"code_dim":int(json.loads((repdir/"manifest.json").read_text())["code_dim"]),
                    "unit_curve":curve,"resumed":True})
                append_progress(progress,event="representation_reused",config=cfg,curve_points=len(curve))
                continue
            mem_codes=[];mem_poses=[]
            for seg in mem:
                code,pose=encode_segment(model,seg,row_lookup,rows,pooled)
                mem_codes.append(np.asarray(code,np.float32));mem_poses.append(pose)
            mem_codes=np.asarray(mem_codes,np.float32)
            if not len(mem_codes):continue
            # For M1, row ids use dataset-global IDs, never pool offsets.
            if family=="m1":
                model["q01"]=q01;model["q99"]=q99
            curve=[]
            for kappa in KAPPAS:
                m6=m6_fit_generic(mem_codes,np.asarray([s["task"] for s in mem]),np.asarray([s["episode"] for s in mem]),kappa=kappa)
                if m6["registered"]==0:continue
                m6["codes"]=mem_codes
                dres=eval_config(model,mem_codes,mem_poses,mem,m6,all_query_segs,devq,rows,episodes,row_lookup,pooled,q01,q99,seed=8401)
                if dres is None:continue
                hard_dummy=np.zeros(len(dres["row_ids"]),bool)
                metr=metrics_summary(dres,hard_dummy)
                aligned=metr["all"]["mse_aligned"]["mean"];coverage=metr["all"]["mse_aligned"]["le_0p04"]
                curve.append({"kappa":kappa,"epsilon":m6["epsilon"],"units":m6["registered"],
                              "pending":len(m6["pending_local"]),"dev_aligned_mse":aligned,
                              "dev_aligned_le_0p04":coverage,
                              "dev_aligned_minus_random":metr["all"]["aligned_minus_random_mean"],
                              "dev_result":dres})
                append_progress(progress,event="dev_curve_point",config=cfg,kappa=kappa,
                                units=m6["registered"],dev_aligned_mse=aligned,
                                dev_aligned_le_0p04=coverage)
            if not curve:continue
            # pick one grain for this configuration by Dev MSE + threshold coverage;
            # grain itself is not selected from Long outcomes.
            best=min(curve,key=lambda c:(c["dev_aligned_mse"],-c["dev_aligned_le_0p04"],c["units"],c["kappa"]))
            kappa=best["kappa"]
            m6=m6_fit_generic(mem_codes,np.asarray([s["task"] for s in mem]),np.asarray([s["episode"] for s in mem]),kappa=kappa)
            m6["codes"]=mem_codes
            mem_units=m6["members_local"]
            # Persist every code family once; M6 representatives and curves are
            # compact sidecars so Stage B can select its own grain without refit.
            config_meta={"kappa":kappa}
            write_representation(out,cfg,sname,model,mem,mem_codes,mem_poses,m6,m6["row_units"],q01,q99,config_meta)
            # Save all threshold curves for future readers, not Long predictions.
            curve_out=[{k:v for k,v in c.items() if k!="dev_result"} for c in curve]
            repdir=out/"representations"/cfg
            write_json(repdir/"dev_curve.json",curve_out)
            config_cache[cfg]={"seg_name":sname}
            for ci,c in enumerate(curve):
                rr=c["dev_result"]
                dev_results.append({"config":cfg,"segmentation":sname,"family":family,"code_dim":mem_codes.shape[1],
                                    "kappa":c["kappa"],"units":c["units"],"dev_aligned_mse":c["dev_aligned_mse"],
                                    "dev_aligned_le_0p04":c["dev_aligned_le_0p04"],"dev_random_aligned_mse":float(np.mean(rr["mse_random_aligned"]))})
            rep_stats.append({"config":cfg,"memory_segments":len(mem),"train_segments":len(train),"dev_segments":len(devsegs),
                              "code_dim":int(mem_codes.shape[1]),"unit_curve":[{k:v for k,v in c.items() if k!="dev_result"} for c in curve]})
        append_progress(progress,event="segmentation_representations_done",segmentation=sname,
                        train_segments=len(train),memory_segments=len(mem),representations=len(models))
        del models,train,mem,devsegs,all_query_segs
    if not dev_results:raise RuntimeError("No representation configuration was evaluated on heldout Dev")
    # Mechanical Dev-only Stage-B list. Rank all (configuration, grain) records,
    # keep one grain per config and at most one per representation family.
    config_best={}
    for r in dev_results:
        k=r["config"]
        if k not in config_best or (r["dev_aligned_mse"],-r["dev_aligned_le_0p04"],r["units"],r["kappa"]) < (config_best[k]["dev_aligned_mse"],-config_best[k]["dev_aligned_le_0p04"],config_best[k]["units"],config_best[k]["kappa"]):config_best[k]=r
    candidates=list(config_best.values())
    mse_order={v["config"]:i for i,v in enumerate(sorted(candidates,key=lambda x:(x["dev_aligned_mse"],x["config"]))) }
    cov_order={v["config"]:i for i,v in enumerate(sorted(candidates,key=lambda x:(-x["dev_aligned_le_0p04"],x["config"]))) }
    for r in candidates:r["selection_rank_sum"]=mse_order[r["config"]]+cov_order[r["config"]]
    required=[]
    for target in ("S0_fixed10__pca_raw__d24","S0_fixed10__m1__d12"):
        matches=[r for r in candidates if r["config"]==target]
        if matches:required.append(matches[0])
    chosen=list(required);families={r["family"] for r in chosen}
    for r in sorted(candidates,key=lambda x:(x["selection_rank_sum"],x["dev_aligned_mse"],x["config"])):
        if r in chosen or r["family"] in families:continue
        chosen.append(r);families.add(r["family"])
        if len(chosen)>=6:break
    chosen=chosen[:6]
    write_json(out/"selected_configs.json",{"selection":"Dev-only, no Long scores; fixed raw-PCA24 + M1-b12 mandatory, then minimum rank-sum, <=one per family, max six",
        "selected":[{k:r[k] for k in ("config","segmentation","family","code_dim","kappa","units","dev_aligned_mse","dev_aligned_le_0p04","selection_rank_sum")} for r in chosen],
        "all_dev_records":len(dev_results)})
    write_json(out/"dev_scores.json",dev_results)
    append_progress(progress,event="dev_selection_fixed",selected=[r["config"] for r in chosen])
    if args.dev_only:
        write_json(out/"dev_only_complete.json",{"complete":True,"representations":len(config_best),
            "dev_rows":len(dev_rows),"long_rows_evaluated":False,"selected_interim":[r["config"] for r in chosen]})
        return
    # Final Long pass, executed once after the selected set was frozen. Evaluate
    # each selected representation at its preselected M6 grain.
    long_records=[]
    long_episode_keys=set((int(tasks[r]),int(rows["episode"][r])) for r in long_rows)
    longq=np.asarray([r for r in long_rows if (int(tasks[r]),int(rows["episode"][r])) in long_episode_keys],np.int64)
    rng=np.random.default_rng(4101)
    for selected in chosen:
        cfg=selected["config"];repdir=out/"representations"/cfg
        sname=config_cache[cfg]["seg_name"]
        model,codes,poses,mem,m6=load_eval_bundle(out,cfg,segment_sets[sname],episodes)
        longqsegs={}
        for s in segment_sets[sname]:
            if s["task"] in LONG_TASKS:longqsegs.setdefault((s["task"],s["episode"]),[]).append(s)
        for key in longqsegs:longqsegs[key].sort(key=lambda x:x["start"])
        longqsegs,fallback_rows=ensure_query_segment_coverage(longqsegs,longq,rows,episodes,window=10)
        res=eval_config(model,codes,poses,mem,m6,longqsegs,longq,rows,episodes,row_lookup,pooled,q01,q99,seed=4101)
        if res is None:continue
        got=np.asarray(res["row_ids"],np.int64)
        if len(got)!=548 or len(np.unique(got))!=548 or set(got.tolist())!=set(long_rows.tolist()):
            raise AssertionError(f"selected config {cfg}: Long query coverage mismatch")
        order=np.asarray([{int(r):i for i,r in enumerate(got)}[int(r)] for r in long_rows],np.int64)
        for key,value in list(res.items()):
            if isinstance(value,np.ndarray) and value.ndim and value.shape[0]==len(got):res[key]=value[order]
        # In the full fixed-order query list, hard mask is from the existing 63-row artifact.
        hard=np.asarray([int(r) in set(np.asarray(long_rows)[hard_mask].tolist()) for r in res["row_ids"]],bool)
        summary=metrics_summary(res,hard)
        unit_task_counts=np.asarray([len({int(mem[i]["task"]) for i in m}) for m in m6["members_local"]],np.int32)
        rep_manifest=json.loads((repdir/"manifest.json").read_text())
        raw_oracle=np.load(ROOT/"fair-expert-01"/"evaluation-final"/"offline_predictions.npz",allow_pickle=False)
        baseline_raw=np.mean((raw_oracle["target"]-raw_oracle["m2_draft"][:,:,:7])**2,axis=(1,2))
        long_records.append({"config":cfg,"segmentation":sname,"family":model["kind"],"code_dim":int(codes.shape[1]),
                             "kappa":float(selected["kappa"]),"units":int(m6["registered"]),"shared_units_fraction":float(np.mean([len(set(np.asarray([mem[i]["task"] for i in m]).tolist()))>1 for m in m6["members_local"]])) if m6["members_local"] else 0.,
                             "unit_task_count_q10_q50_q90":np.quantile(unit_task_counts,[.1,.5,.9]).round(2).tolist() if len(unit_task_counts) else [],
                             "units_shared_across_2plus_tasks":int(np.sum(unit_task_counts>=2)),
                             "code_values_per_step_mean":rep_manifest["code_values_per_step_mean"],
                             "pose_numeric_values_per_segment_mean":rep_manifest["pose_numeric_values_per_segment_mean"],
                             "metrics":summary,"query_row_ids":res["row_ids"].tolist(),"row_metrics":{k:v.tolist() for k,v in res.items()}})
        append_progress(progress,event="long_eval_done",config=cfg,units=m6["registered"],mse_aligned=summary["all"]["mse_aligned"]["mean"])
    write_json(out/"long_metrics.json",long_records)
    # Figures: M6 unit-count curve, segmentation lengths, and oracle pose vs random-type.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors={"pca_raw":"#1f77b4","pca_raw_white":"#1f77b4","pca_canonical":"#ff7f0e","pca_gripsep":"#ff7f0e","procrustes_pca":"#9467bd","dmp":"#2ca02c","spline":"#d62728","m1":"#8c564b"}
        markers={"S0":"s","S1":"o","S2":"^","S3":"D"}
        fig,ax=plt.subplots(figsize=(10,6))
        selected_cfgs={r["config"] for r in chosen}
        for cfg in sorted(selected_cfgs):
            family=next(r["family"] for r in dev_results if r["config"]==cfg)
            segname=next(r["segmentation"] for r in dev_results if r["config"]==cfg)
            points=sorted([r for r in dev_results if r["config"]==cfg],key=lambda x:x["units"])
            ax.plot([r["units"] for r in points],[r["dev_aligned_mse"] for r in points],
                    color=colors.get(family,"#777777"),marker=markers.get(segname[:2],"o"),label=cfg)
        ax.set_xscale("log");ax.set_xlabel("M6 registered units (log)");ax.set_ylabel("Heldout Dev oracle-aligned executed 10-step normalized active7 MSE");ax.grid(True,alpha=.25)
        handles,labels=ax.get_legend_handles_labels();ax.legend(handles,labels,fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/"figures"/"unit_count_distance.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,5))
        seg_colors={"S0":"#1f77b4","S1":"#ff7f0e","S2":"#2ca02c","S3":"#9467bd"}
        for pos,r in enumerate(segment_stats):
            # quantile summary encoded as error bars, preserving one y axis.
            q=r["length_q10_q25_q50_q75_q90"]
            key=r["name"][:2]
            ax.plot([pos]*5,q,marker=markers.get(key,"o"),color=seg_colors.get(key,"#777777"))
        ax.set_xticks(list(range(len(segment_stats))),[x["name"] for x in segment_stats],rotation=60,ha="right")
        ax.set_ylabel("Segment length (10 Hz frames)");ax.grid(True,axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/"figures"/"segment_lengths.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(long_records));w=.38
        ax.bar(x-w/2,[r["metrics"]["all"]["mse_aligned"]["mean"] for r in long_records],w,label="selected type + oracle pose")
        ax.bar(x+w/2,[r["metrics"]["all"]["mse_random_aligned"]["mean"] for r in long_records],w,label="random type + same pose")
        ax.set_xticks(x,[r["config"] for r in long_records],rotation=70,ha="right",fontsize=6);ax.set_ylabel("Long executed 10-step window normalized active7 MSE")
        ax.legend();ax.grid(True,axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/"figures"/"aligned_vs_random.png",dpi=160);plt.close(fig)
    except ImportError as exc:
        write_json(out/"figures_unavailable.json",{"reason":str(exc)})
    write_json(out/"long_report.json",{"selected_configs":json.loads((out/"selected_configs.json").read_text()),
        "long_eval_rows":548,"hard_rows":63,"metric":"mean squared error over executed 10×7 normalized action window",
        "long_used_only_after_dev_selection":True,"fixed_raw_memory_oracle_reference":{"all":.028991,"hard":.035122},
        "results":long_records})
    append_progress(progress,event="complete",representations=len(rep_stats),dev_configs=len(dev_results),long_selected=len(long_records))
    write_json(out/"DONE.json",{"complete":True,"selected_count":len(chosen),"long_eval_configs":len(long_records)})


if __name__=="__main__":main()
