"""Compare cross-suite memory keys while freezing the light model and z values."""
from __future__ import annotations

import argparse
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.memory import KeyEncoder
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import _pairwise_outcomes
from .diagnose_sharing import PREVIOUS, _decode_mse
from .extract import dataset_episodes
from .runtime import seed_experiment as _seed
from .model import BottleneckModel

DATA = Path("/work/.skillnet-state/data/lerobot/emergent/libero_40_unsegmented")
CKPT = Path("/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_base/params")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _camera_pooler(checkpoint: Path, batch_size: int):
    """Return separate no-language VLM pools for base and wrist images."""
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0_config
    from openpi.models.model import restore_params
    from openpi.shared import image_tools
    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                               action_horizon=25, max_token_len=200, dtype="bfloat16")
    model = cfg.load(restore_params(checkpoint, dtype=jnp.bfloat16))
    resize = jax.jit(lambda x: image_tools.resize_with_pad(x, 224, 224))

    def infer_one(image):
        tok, _ = model.PaliGemma.img(image, train=False)
        n = tok.shape[1]
        mask = jnp.ones((tok.shape[0], n), dtype=jnp.bool_)
        ar = jnp.zeros((n,), dtype=jnp.bool_)
        attention = (jnp.cumsum(ar)[None, :, None] <= jnp.cumsum(ar)[None, None, :])
        attention = attention & mask[:, None, :] & mask[:, :, None]
        positions = jnp.cumsum(mask, axis=1) - 1
        (out, _), _ = model.PaliGemma.llm([tok, None], mask=attention, positions=positions,
                                          adarms_cond=[None, None])
        return jnp.mean(out.astype(jnp.float32), axis=1)

    infer_jit = jax.jit(infer_one)

    def encode(base_bytes, wrist_bytes):
        images = []
        for a, b in zip(base_bytes, wrist_bytes):
            images.append(np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.uint8))
            images.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8))
        x = np.asarray(resize(jnp.asarray(np.stack(images))))
        x = jnp.asarray(x, dtype=jnp.float32) / 127.5 - 1.0
        out = np.asarray(infer_jit(x), dtype=np.float32)
        return out[0::2], out[1::2]
    return encode


def _extract_camera_cache(dataset: Path, checkpoint: Path, cache: Path, *, batch_size=2):
    cache.mkdir(parents=True, exist_ok=False)
    epdir = cache / "episodes"; epdir.mkdir()
    encode = _camera_pooler(checkpoint, batch_size)
    manifest = []
    for n, ep in enumerate(dataset_episodes(dataset)):
        idx = np.arange(0, len(ep["steps"]), 5, dtype=np.int32)
        base, wrist = [], []
        for lo in range(0, len(idx), batch_size):
            ix = idx[lo:lo+batch_size]; real = len(ix)
            if real < batch_size: ix = np.pad(ix, (0, batch_size-real), mode="edge")
            b, w = encode([ep["images"][i] for i in ix], [ep["wrist_images"][i] for i in ix])
            base.extend(b[:real]); wrist.extend(w[:real])
        path = epdir / f"{ep['episode_id']}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, step=ep["steps"][idx], base=np.asarray(base, dtype=np.float16),
                                wrist=np.asarray(wrist, dtype=np.float16))
        manifest.append({"episode_id": ep["episode_id"], "suite": ep["suite"], "task": ep["task"],
                         "task_index": ep["task_index"], "n_rows": len(idx), "cache": str(path)})
        if (n+1) % 50 == 0: print(f"camera feature extraction: {n+1} episodes", flush=True)
    (cache / "manifest.json").write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(checkpoint),
        "stride": 5, "features": "per-camera image-only PaliGemma, no language, pooled output tokens",
        "episodes": manifest}, indent=2))
    return cache


def _camera_arrays(cache: Path, run):
    # Infer the pooled width from the first cached episode; PaliGemma versions
    # may use a different hidden width than the older concatenated cache.
    first = next((cache / "episodes").glob("*.npz"))
    with np.load(first, allow_pickle=False) as d:
        width = d["base"].shape[1]
    base, wrist = np.zeros((len(run), width), np.float32), np.zeros((len(run), width), np.float32)
    seen = np.zeros(len(run), bool)
    for ep in run.episodes:
        p = cache / "episodes" / f"{ep.episode_id}.npz"
        with np.load(p, allow_pickle=False) as d:
            steps = d["step"]
            lookup = {int(s): j for j, s in enumerate(steps)}
            rows = np.flatnonzero(run.episode_id == ep.episode_id)
            for r in rows:
                j = lookup.get(int(run.step[r]))
                if j is not None:
                    base[r], wrist[r], seen[r] = d["base"][j], d["wrist"][j], True
    if not seen[run.usable].all():
        raise ValueError(f"camera feature cache misses {int((run.usable & ~seen).sum())} usable rows")
    return base, wrist


def _fit_pca(train, all_x, n=32):
    x=np.asarray(train,dtype=np.float64); mu=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1
    x=(x-mu)/sd; xm=x.mean(0)
    _,_,vt=np.linalg.svd(x-xm,full_matrices=False); comp=vt[:min(n,len(vt))]
    return ((np.asarray(all_x,dtype=np.float64)-mu)/sd-xm)@comp.T


def _fit_learned_key(train_rows, run, base, wrist, *, horizon, seed, epochs=15):
    """Siamese InfoNCE, with z-neighbor positives and same-episode adjacent masking."""
    rng=np.random.default_rng(seed+horizon)
    tr=np.asarray(train_rows); z=run.z["post"][tr].astype(np.float32)
    zm=z.mean(0); zs=z.std(0); zs[zs<1e-6]=1; z=(z-zm)/zs
    # Train-only feature normalization. No spatial rows enter these statistics.
    raw=np.concatenate([base[tr],wrist[tr],run.proprio[tr]],axis=1).astype(np.float32)
    mean=raw.mean(0); scale=raw.std(0); scale[scale<1e-6]=1
    x=((raw-mean)/scale).astype(np.float32)
    # For each anchor, select its closest z neighbor outside the adjacent-window exclusion.
    pos=np.empty(len(tr),np.int64)
    train_ep=run.episode_id[tr]; train_step=run.step[tr]
    z2=(z*z).sum(1)
    for lo in range(0,len(tr),512):
        hi=min(len(tr),lo+512); d2=np.maximum(z2[lo:hi,None]+z2[None]-2*z[lo:hi]@z.T,0)
        bad=(train_ep[lo:hi,None]==train_ep[None]) & (np.abs(train_step[lo:hi,None]-train_step[None])<horizon)
        d2[bad]=np.inf
        pos[lo:hi]=d2.argmin(1)
    if np.any(pos==np.arange(len(tr))): raise AssertionError("self-positive selected")
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,128),
                              torch.nn.GELU(),torch.nn.Linear(128,32)).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    xt=torch.as_tensor(x,device=DEVICE)
    for epoch in range(epochs):
        order=rng.permutation(len(tr)); losses=[]
        for st in range(0,len(order),256):
            anchors=order[st:st+256]
            if len(anchors)<2: continue
            partners=pos[anchors]
            cand=np.concatenate([anchors,partners])
            emb=torch.nn.functional.normalize(model(xt[torch.as_tensor(cand,device=DEVICE)]),dim=-1)
            q=emb[:len(anchors)]; candidates=emb
            logits=q@candidates.T/0.08
            # Candidate columns with same-episode adjacent windows are excluded;
            # positive partner is always valid by construction.
            ca=np.asarray(cand); aa=np.asarray(anchors)
            invalid=(train_ep[aa,None]==train_ep[None,ca]) & (np.abs(train_step[aa,None]-train_step[None,ca])<horizon)
            invalid[np.arange(len(anchors)),len(anchors)+np.arange(len(anchors))]=False
            logits[torch.as_tensor(invalid,device=DEVICE)]=-1e4
            labels=torch.arange(len(anchors),device=DEVICE)+len(anchors)
            loss=torch.nn.functional.cross_entropy(logits,labels)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        print(f"learned key H={horizon} epoch={epoch+1}/{epochs} InfoNCE={np.mean(losses):.4f}",flush=True)
    allraw=np.concatenate([base,wrist,run.proprio],axis=1).astype(np.float32)
    with torch.no_grad():
        emb=torch.nn.functional.normalize(model(torch.as_tensor(((allraw-mean)/scale).astype(np.float32),device=DEVICE)),dim=-1)
    return emb.cpu().numpy(),{"epochs":epochs,"train_rows":len(tr),"objective":"InfoNCE; nearest standardized z_post neighbor positive; adjacent same-episode rows masked from positives and negatives","temperature":0.08}


def _memory_for_key(key, train_rows, spatial_store_rows, run, *, horizon, threshold_q=.9, topk=5):
    train_key=key[train_rows]; tr_ep=run.episode_id[train_rows]; tr_init=run.init_state_id[train_rows]
    n=len(train_rows); nearest=np.full(n,np.inf)
    # Chunked train-memory calibration using the same episode/init-state exclusion as retrieval.
    for lo in range(0,n,256):
        hi=min(n,lo+256); d2=np.maximum((train_key[lo:hi]**2).sum(1)[:,None]+(train_key**2).sum(1)[None]-2*train_key[lo:hi]@train_key.T,0)
        bad=(tr_ep[lo:hi,None]==tr_ep[None]) | (tr_init[lo:hi,None]==tr_init[None]); d2[bad]=np.inf
        nearest[lo:hi]=np.sqrt(d2.min(1))
    finite=nearest[np.isfinite(nearest)]
    tau=float(np.quantile(finite,threshold_q)); dref=float(np.median(finite))
    rows=list(train_rows); added=[]; tau_history=[]
    order=spatial_store_rows[np.lexsort((run.step[spatial_store_rows],run.episode_id[spatial_store_rows]))]
    for r in order:
        rr=np.asarray(rows,dtype=np.int64)
        ok=(run.episode_id[rr]!=run.episode_id[r]) & (run.init_state_id[rr]!=run.init_state_id[r])
        d=np.linalg.norm(key[r]-key[rr[ok]],axis=1)
        md=float(d.min()) if len(d) else np.inf
        if md>tau:
            rows.append(int(r)); added.append(int(r))
    return {"rows":np.asarray(rows,dtype=np.int64),"tau":tau,"d_ref":dref,
            "added_rows":np.asarray(added,dtype=np.int64),"candidate_rows":len(order),"topk":topk}


def _retrieve(key, memory, query_rows, run, values):
    rows=memory["rows"]; dref=memory["d_ref"]
    top1=np.empty((len(query_rows),values.shape[1]),np.float32); blend=np.empty_like(top1)
    chosen=np.empty((len(query_rows),5),np.int64); weights=np.zeros((len(query_rows),5),np.float32)
    for qi,r in enumerate(query_rows):
        ok=(run.episode_id[rows]!=run.episode_id[r]) & (run.init_state_id[rows]!=run.init_state_id[r])
        ix=rows[ok]; dist=np.linalg.norm(key[r]-key[ix],axis=1); order=np.argsort(dist,kind="stable")[:5]
        take=ix[order]; ds=dist[order]; w=np.exp(-(ds-ds[0])/max(.5*dref,1e-12)); w/=w.sum()
        chosen[qi]=take; weights[qi]=w; top1[qi]=values[take[0]]; blend[qi]=w@values[take]
    return top1,blend,chosen,weights


def _motion_table(run, query_rows, train_rows):
    components=[]; best=[]; scale_sum=np.zeros(3); count=0
    stores=np.asarray(train_rows)
    for i,r in enumerate(query_rows):
        comp=_pairwise_outcomes(run.motion[r:r+1],run.motion[stores])[0]
        eligible=(run.episode_id[stores]!=run.episode_id[r]) & (run.init_state_id[stores]!=run.init_state_id[r])
        scale_sum+=comp[eligible].sum(0); count+=int(eligible.sum()); components.append(comp)
        if (i+1)%400==0: print(f"motion diagnostics {i+1}/{len(query_rows)}",flush=True)
    scale=scale_sum/max(count,1)
    best_score=np.empty(len(query_rows)); nearest=np.empty((len(query_rows),3))
    for i,r in enumerate(query_rows):
        eligible=(run.episode_id[stores]!=run.episode_id[r]) & (run.init_state_id[stores]!=run.init_state_id[r])
        score=(components[i]/np.maximum(scale,1e-12)).sum(1); score[~eligible]=np.inf
        j=int(score.argmin()); best_score[i]=score[j]; nearest[i]=components[i][j]
    return stores,np.asarray(components),scale,best_score,nearest


def _setting(root, horizon, beta, cam_cache, seed, key_epochs):
    tag=f"H{horizon}_beta{beta:g}"; prev=PREVIOUS/tag
    checkpoint=torch.load(prev/"model.pt",map_location="cpu",weights_only=False)
    run=load_z_run(prev/"eval_input",horizon=horizon)
    model=BottleneckModel(tuple(run.target_actions.shape[1:]),run.proprio.shape[1],run.keys["vlm_img_lang"].shape[1],32).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"]); norm=checkpoint["normalizers"]
    manifest=json.loads((prev/"eval_input/manifest.json").read_text()); train_ids=set(manifest["emergent"]["training_episodes"])
    qids={json.loads(l)["query_episode_id"] for l in (prev/"query_retrieval_log.jsonl").open() if l.strip()}
    spatial_ids=set(run.episode_id[run.suite=="libero_spatial"].tolist())
    qrows=np.flatnonzero(np.isin(run.episode_id,sorted(qids))&run.usable)
    train_rows=np.flatnonzero(np.isin(run.episode_id,sorted(train_ids))&run.usable)
    spatial_store=np.flatnonzero(np.isin(run.episode_id,sorted(spatial_ids-qids))&run.usable)
    if set(run.init_state_id[qrows]) & set(run.init_state_id[spatial_store]): raise AssertionError("query/store leakage")
    base,wrist=_camera_arrays(cam_cache,run)
    keys={}
    # Existing baseline is transformed through exactly the frozen train20 encoder from heldout.
    existing=KeyEncoder.load(prev/"key_encoder_train20.npz")
    keys["combined_base_wrist_proprio"] = existing.transform({"vlm_img_nolang":run.keys["vlm_img_nolang"],"proprio":run.proprio}).astype(np.float32)
    wrist_encoder=KeyEncoder.fit({"vlm_img_nolang":wrist[train_rows]},include_language=False,n_components=32)
    keys["wrist_only"]=wrist_encoder.transform({"vlm_img_nolang":wrist}).astype(np.float32)
    keys["proprio_only"]=_fit_pca(run.proprio[train_rows],run.proprio)
    wrist_state_encoder=KeyEncoder.fit({"vlm_img_nolang":wrist[train_rows],"proprio":run.proprio[train_rows]},
                                        include_language=False,n_components=32)
    keys["wrist_plus_proprio"]=wrist_state_encoder.transform({"vlm_img_nolang":wrist,"proprio":run.proprio}).astype(np.float32)
    learned,learned_meta=_fit_learned_key(train_rows,run,base,wrist,horizon=horizon,seed=seed,epochs=key_epochs)
    learned_meta["seed"]=int(seed)
    keys["learned_z_contrastive"]=learned
    # Prior z is standardized using only the 20 training tasks; no spatial fit.
    pm=run.z["prior"][train_rows].mean(0); ps=run.z["prior"][train_rows].std(0); ps[ps<1e-6]=1
    keys["prior_z"]=(run.z["prior"]-pm)/ps
    print(f"{tag}: fit all six keys; train rows={len(train_rows)}, query rows={len(qrows)}",flush=True)
    # Exact query-to-store motion distances, computed once for quantile placement and candidate closeness.
    motion_rows,motion_components,motion_scale,best_motion_score,best_motion_components=_motion_table(run,qrows,train_rows)
    qedges=np.quantile(best_motion_score,[.25,.5,.75]); qbin=np.digitize(best_motion_score,qedges,right=True)
    values=run.z["post"].astype(np.float32); rows_out={}; variants={}
    for name,key in keys.items():
        m0=_memory_for_key(key,train_rows,np.zeros(0,np.int64),run,horizon=horizon)
        m1=_memory_for_key(key,train_rows,spatial_store,run,horizon=horizon)
        variants[name]={"object_goal":m0,"spatial_added":m1}
        for mname,memory in (("object_goal",m0),("spatial_added",m1)):
            z1,z5,chosen,weights=_retrieve(key,memory,qrows,run,values)
            rows_out[f"{name}/{mname}/top1"]=_decode_mse(model,norm,run,qrows,z1)
            rows_out[f"{name}/{mname}/top5_blend"]=_decode_mse(model,norm,run,qrows,z5)
            # Selected items' future-motion distances; top-5 distance is weighted by the actual blend weights.
            lookup={int(r):j for j,r in enumerate(motion_rows)}
            qmotion=[]
            for qi,r in enumerate(qrows):
                comps=motion_components[qi]
                selected=chosen[qi]
                topcomp=np.empty((len(selected),3),dtype=np.float64)
                missing=[]; missing_pos=[]
                for j,row_id in enumerate(selected):
                    loc=lookup.get(int(row_id))
                    if loc is None:
                        missing.append(int(row_id)); missing_pos.append(j)
                    else:
                        topcomp[j]=comps[loc]
                if missing:
                    topcomp[missing_pos]=_pairwise_outcomes(run.motion[r:r+1],run.motion[np.asarray(missing)])[0]
                qmotion.append({"top1":topcomp[0],"top5_weighted":weights[qi]@topcomp})
            rows_out[f"__motion__{name}/{mname}"]=qmotion
        print(f"{tag}: key={name} add={m1['added_rows'].size}/{m1['candidate_rows']}",flush=True)
    # Include previous diagnostic action/MSE modes as references and whole motion oracle.
    rows_out["__refs__"]={"own":_decode_mse(model,norm,run,qrows,run.z["post"][qrows]),
                          "prior":_decode_mse(model,norm,run,qrows,run.z["prior"][qrows]),
                          "random":np.asarray(json.load(open(PREVIOUS.parent/f"emergent-light-diag-20261026T072358Z-c8d064/H{horizon}_beta0.01/results.json"))["paired_row_mse"]["random"])}

    # Aggregate tables and use per-query motion-proximity bins matching the previous report.
    mse={k:float(v.mean()) for k,v in rows_out.items() if isinstance(v,np.ndarray)}
    motion_by_key={}
    key_names=list(keys)
    for name in key_names:
        motion_by_key[name]={}
        for mname in ("object_goal","spatial_added"):
            perq=rows_out[f"__motion__{name}/{mname}"]
            bins=[]
            for b in range(4):
                mask=qbin==b
                bins.append({"quantile":f"Q{b+1}","n":int(mask.sum()),
                    "best_available_combined_normalized":float(best_motion_score[mask].mean()),
                    "top1_components":np.mean([x["top1"] for x,keep in zip(perq,mask) if keep],axis=0).tolist(),
                    "top5_weighted_components":np.mean([x["top5_weighted"] for x,keep in zip(perq,mask) if keep],axis=0).tolist()})
            motion_by_key[name][mname]=bins
    out={"horizon":horizon,"beta":beta,"train_rows":len(train_rows),"query_rows":len(qrows),
        "query_episodes":len(set(run.episode_id[qrows])),"spatial_store_rows":len(spatial_store),
        "spatial_store_episodes":len(spatial_ids-qids),"spatial_motion_quantile_edges":qedges.tolist(),
        "key_names":key_names,"mse_mean":mse,
        "spatial_add_rates":{n:float(v["spatial_added"]["added_rows"].size/v["spatial_added"]["candidate_rows"]) for n,v in variants.items()},
        "memory_sizes":{n:{m:len(v["rows"]) for m,v in variants[n].items()} for n in key_names},
        "motion_selected_by_best_available_quantile":motion_by_key,
        "learned_key_training":learned_meta,
        "references":{k:float(v.mean()) for k,v in rows_out["__refs__"].items()},
        "notes":{"PCA_stats_and_components":"fit on object/goal train rows only","learned_key":"trained on object/goal z_post neighborhood supervision only; no spatial rows or env motion labels","memory":"same executed posterior values and episode/init exclusions; only key vectors and key-dependent spatial additions vary","spatial_add_threshold":"fixed q=0.9 NN distance quantile calibrated on train20 memory; softmax temperature=0.5*d_ref"}}
    d=root/tag; d.mkdir(exist_ok=False)
    (d/"results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    return out


def _report(root, outputs, camera_cache):
    names=["combined_base_wrist_proprio","wrist_only","proprio_only","wrist_plus_proprio","learned_z_contrastive","prior_z"]
    previous_report=PREVIOUS.parent/"emergent-light-diag-20261026T072358Z-c8d064"/"report.md"
    lines=["# Cross-suite memory key comparison","",
      f"Previous value/key diagnosis (unchanged): [{previous_report}]({previous_report}). Reused its checkpoints, query episodes, executed posterior z values, and action targets. No Enc/Prior/Dec parameters were updated.","",
      f"Per-camera feature cache: `{camera_cache}`. Base and wrist pools were extracted independently with the frozen base PaliGemma, no language. Every PCA/scaler and the learned contrastive key were fit on object/goal training rows only. Spatial rows were used only for insertion/query evaluation; learned key training used z_post neighborhoods and did not use environment motion.","",
      f"Key variants: (1) existing combined base+wrist language-free pool plus proprio and train20 PCA; (2) independently pooled wrist-camera feature and train20 PCA; (3) proprio-only standardized PCA; (4) wrist feature plus proprio with train20 block standardization/PCA; (5) a 32-D normalized MLP key trained with InfoNCE on nearest standardized z_post neighbors (same-episode overlapping windows excluded from positive/negative pairs); (6) train20-standardized Prior z. The learned key ran {outputs[0]['learned_key_training']['epochs']} epochs with seed {outputs[0]['learned_key_training']['seed']}.","",
      "For each key, object/goal-only memory and spatial-added memory use the same executed z values. The spatial memory streams the same spatial store rows with key-specific thresholding (fixed q=0.9 NN distance threshold calibrated on train20). Both retrievals exclude query episode and init-state; blend uses top-5 softmax with temperature 0.5 × train-memory d_ref.","",
      "## Reconstruction MSE","","Lower is better; `OG` means object/goal-only memory and `+spatial` means threshold-accepted spatial entries were added.","",
      "| H | key | top-1 OG | top-5 blend OG | top-1 +spatial | top-5 blend +spatial | spatial add rate |","|---:|---|---:|---:|---:|---:|---:|"]
    for d in outputs:
        for name in names:
            m=d["mse_mean"]; lines.append(f"| {d['horizon']} | {name} | {m[name+'/object_goal/top1']:.4f} | {m[name+'/object_goal/top5_blend']:.4f} | {m[name+'/spatial_added/top1']:.4f} | {m[name+'/spatial_added/top5_blend']:.4f} | {d['spatial_add_rates'][name]:.1%} |")
    for d in outputs:
        lines += ["",f"## H={d['horizon']}, β={d['beta']:g}","",
                  f"Training rows={d['train_rows']}; spatial queries={d['query_rows']} rows / {d['query_episodes']} episodes; spatial store={d['spatial_store_rows']} rows / {d['spatial_store_episodes']} episodes.","",
                  f"Reference MSE: own={d['references']['own']:.4f}, Prior={d['references']['prior']:.4f}, random={d['references']['random']:.4f}. Future-motion top-5 reference from the prior diagnostic: {0.1624 if d['horizon']==25 else 0.1424:.4f}. This uses query future env motion and is an upper-bound diagnostic.","",
                  "Selected item motion distances by quantile of best achievable object/goal future-motion proximity (components are mean position distance, rotation radians, gripper distance):", ""]
        for name in names:
            lines += [f"### {name}","","| Memory | proximity quantile | n | best available normalized combined distance | key top-1 pos/rot/grip | key top-5 weighted pos/rot/grip |","|---|---|---:|---:|---|---|"]
            for mname in ("object_goal","spatial_added"):
                for x in d["motion_selected_by_best_available_quantile"][name][mname]:
                    t=x["top1_components"]; b=x["top5_weighted_components"]
                    lines.append(f"| {mname} | {x['quantile']} | {x['n']} | {x['best_available_combined_normalized']:.4f} | {t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f} | {b[0]:.4f}/{b[1]:.4f}/{b[2]:.4f} |")
            lines.append("")
    lines += ["## Interpretation", "",
      "Compare object/goal-only top-5 MSE with the motion-oracle reference. Improvement over the existing combined key shows key choice matters; proximity tables show whether retrieval is selecting motion-similar items. Prior-z retrieval is a learned observation-conditioned key, while `learned_z_contrastive` is trained to group items with nearby z_post. Neither is trained on spatial labels or motion features.",
      "A low all-value oracle from the previous diagnosis means these results remain limited by query-conditioned value compatibility as well as retrieval. Motion distance uses future environment trajectories and remains diagnostic only. Exact per-query MSE and candidate statistics are in each horizon's `results.json`.",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,default=Path("/work/.skillnet-state/outputs/memory-pilot")); p.add_argument("--data",type=Path,default=DATA); p.add_argument("--checkpoint",type=Path,default=CKPT); p.add_argument("--camera-cache",type=Path); p.add_argument("--seed",type=int,default=73); p.add_argument("--feature-batch",type=int,default=2); p.add_argument("--key-epochs",type=int,default=15)
    a=p.parse_args(); _seed(a.seed)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    root=a.output_root/f"emergent-light-keys-{run_id}"; root.mkdir(parents=True,exist_ok=False)
    cache=a.camera_cache or (root/"camera_features")
    if not cache.exists(): _extract_camera_cache(a.data,a.checkpoint,cache,batch_size=a.feature_batch)
    outputs=[]
    for h in (25,10):
        outputs.append(_setting(root,h,.01,cache,a.seed,a.key_epochs))
    _report(root,outputs,cache)
    print(f"report: {root/'report.md'}",flush=True)


if __name__=="__main__": main()
