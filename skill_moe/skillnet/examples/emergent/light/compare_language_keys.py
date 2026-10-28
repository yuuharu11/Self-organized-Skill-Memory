"""Add two language-aware keys to the frozen cross-suite key comparison."""
from __future__ import annotations

import argparse
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image

from openpi.models import pi0_config
from openpi.models.model import restore_params
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import image_tools
from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.memory import KeyEncoder
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import _pairwise_outcomes
from .compare_keys import DATA, CKPT, _memory_for_key, _retrieve, _motion_table
from .diagnose_sharing import PREVIOUS, _decode_mse
from .extract import dataset_episodes
from .model import BottleneckModel

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_DIAG=Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-diag-20261026T072358Z-c8d064")
PREVIOUS_KEYS=Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-keys-20261026T080437Z-eedfc4")


def _wrist_language_pooler(checkpoint: Path, batch_size: int):
    cfg=pi0_config.Pi0Config(pi05=True,paligemma_variant="gemma_2b",action_expert_variant="gemma_300m",
                             action_horizon=25,max_token_len=200,dtype="bfloat16")
    model=cfg.load(restore_params(checkpoint,dtype=jnp.bfloat16))
    resize=jax.jit(lambda x:image_tools.resize_with_pad(x,224,224))
    tokenizer=PaligemmaTokenizer(200)
    def infer(image,tokens,token_mask):
        image_tokens,_=model.PaliGemma.img(image,train=False)
        text_tokens=model.PaliGemma.llm(tokens,method="embed")
        prefix=jnp.concatenate([image_tokens,text_tokens],axis=1)
        image_mask=jnp.ones((prefix.shape[0],image_tokens.shape[1]),dtype=jnp.bool_)
        input_mask=jnp.concatenate([image_mask,token_mask],axis=1)
        ar_mask=jnp.zeros((prefix.shape[1],),dtype=jnp.bool_)
        attention=(jnp.cumsum(ar_mask)[None,:,None]<=jnp.cumsum(ar_mask)[None,None,:])
        attention=attention&input_mask[:,None,:]&input_mask[:,:,None]
        positions=jnp.cumsum(input_mask,axis=1)-1
        (out,_),_=model.PaliGemma.llm([prefix,None],mask=attention,positions=positions,adarms_cond=[None,None])
        return jnp.mean(out[:,:image_tokens.shape[1],:].astype(jnp.float32),axis=1)
    infer_jit=jax.jit(infer)
    def encode(images,tasks):
        raw=[np.asarray(Image.open(io.BytesIO(x)).convert("RGB"),dtype=np.uint8) for x in images]
        x=np.asarray(resize(jnp.asarray(np.stack(raw))))
        x=jnp.asarray(x,dtype=jnp.float32)/127.5-1.0
        toks,masks=zip(*(tokenizer.tokenize(task,None) for task in tasks))
        return np.asarray(infer_jit(x,jnp.asarray(np.stack(toks)),jnp.asarray(np.stack(masks))),dtype=np.float32)
    return encode


def _extract_wrist_lang(dataset: Path,checkpoint: Path,out: Path,*,batch_size=2):
    out.mkdir(parents=True,exist_ok=False); epdir=out/"episodes"; epdir.mkdir()
    encode=_wrist_language_pooler(checkpoint,batch_size); inventory=[]
    for n,ep in enumerate(dataset_episodes(dataset)):
        ix=np.arange(0,len(ep["steps"]),5,dtype=np.int32); feats=[]
        for lo in range(0,len(ix),batch_size):
            sel=ix[lo:lo+batch_size]; real=len(sel)
            if real<batch_size: sel=np.pad(sel,(0,batch_size-real),mode="edge")
            v=encode([ep["wrist_images"][i] for i in sel],[ep["task"]]*len(sel))
            feats.extend(v[:real])
        path=epdir/f"{ep['episode_id']}.npz"
        with path.open("xb") as f: np.savez_compressed(f,step=ep["steps"][ix],wrist_lang=np.asarray(feats,dtype=np.float16))
        inventory.append({"episode_id":ep["episode_id"],"suite":ep["suite"],"task":ep["task"],"n_rows":len(ix),"cache":str(path)})
        if (n+1)%50==0: print(f"wrist-language extraction: {n+1} episodes",flush=True)
    (out/"manifest.json").write_text(json.dumps({"dataset":str(dataset),"checkpoint":str(checkpoint),"stride":5,
        "input":"wrist RGB + original task instruction; language-conditioned PaliGemma image-token output mean","episodes":inventory},indent=2))
    return out


def _align_wrist_lang(cache: Path,run):
    first=next((cache/"episodes").glob("*.npz"))
    with np.load(first,allow_pickle=False) as d: width=d["wrist_lang"].shape[1]
    x=np.zeros((len(run),width),np.float32); seen=np.zeros(len(run),bool)
    for ep in run.episodes:
        with np.load(cache/"episodes"/f"{ep.episode_id}.npz",allow_pickle=False) as d:
            index={int(step):i for i,step in enumerate(d["step"])}
            rows=np.flatnonzero(run.episode_id==ep.episode_id)
            for r in rows:
                j=index.get(int(run.step[r]))
                if j is not None: x[r]=d["wrist_lang"][j]; seen[r]=True
    if not seen[run.usable].all(): raise ValueError("wrist-language cache misses usable eval rows")
    return x


def _eval_horizon(root,horizon,cache,seed):
    tag=f"H{horizon}_beta0.01"; old=PREVIOUS_KEYS/tag; old_diag=OLD_DIAG/tag
    ckpt=torch.load(PREVIOUS/tag/"model.pt",map_location="cpu",weights_only=False)
    run=load_z_run(PREVIOUS/tag/"eval_input",horizon=horizon)
    model=BottleneckModel(tuple(run.target_actions.shape[1:]),run.proprio.shape[1],run.keys["vlm_img_lang"].shape[1],32).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); norm=ckpt["normalizers"]
    manifest=json.loads((PREVIOUS/tag/"eval_input/manifest.json").read_text())
    train_ids=set(manifest["emergent"]["training_episodes"])
    qids={json.loads(line)["query_episode_id"] for line in (PREVIOUS/tag/"query_retrieval_log.jsonl").open() if line.strip()}
    spatial_ids=set(run.episode_id[run.suite=="libero_spatial"].tolist())
    train_rows=np.flatnonzero(np.isin(run.episode_id,sorted(train_ids))&run.usable)
    qrows=np.flatnonzero(np.isin(run.episode_id,sorted(qids))&run.usable)
    spatial_store=np.flatnonzero(np.isin(run.episode_id,sorted(spatial_ids-qids))&run.usable)
    if set(run.init_state_id[qrows])&set(run.init_state_id[spatial_store]): raise AssertionError("query/store leakage")
    wrist_lang=_align_wrist_lang(cache,run)

    # Feature 7: cached language-conditioned base+wrist output f_t plus proprio.
    lang_encoder=KeyEncoder.fit({"vlm_img_lang":run.keys["vlm_img_lang"][train_rows],"proprio":run.proprio[train_rows]},
                                include_language=True,n_components=32)
    key_lang_base_wrist=lang_encoder.transform({"vlm_img_lang":run.keys["vlm_img_lang"],"proprio":run.proprio}).astype(np.float32)
    # Feature 8: language-conditioned wrist-only visual feature; no proprio block.
    wrist_encoder=KeyEncoder.fit({"vlm_img_lang":wrist_lang[train_rows]},include_language=True,n_components=32)
    key_lang_wrist=wrist_encoder.transform({"vlm_img_lang":wrist_lang}).astype(np.float32)
    keys={"language_base_wrist_proprio":key_lang_base_wrist,"language_wrist_only":key_lang_wrist}
    values=run.z["post"].astype(np.float32)
    # Future-motion bins and per-selected-entry closeness use the same train20 candidate reference as before.
    motion_rows,motion_components,motion_scale,best_score,best_comp=_motion_table(run,qrows,train_rows)
    previous_edges=np.asarray(json.load(open(old_diag/"results.json"))["motion_search"]["best_motion_distance_quantile_edges"])
    bins=np.digitize(best_score,previous_edges,right=True)
    motion_lookup={int(r):i for i,r in enumerate(motion_rows)}
    mse={}; rates={}; by_motion={}; sizes={}
    for name,key in keys.items():
        mem0=_memory_for_key(key,train_rows,np.zeros(0,dtype=np.int64),run,horizon=horizon)
        mem1=_memory_for_key(key,train_rows,spatial_store,run,horizon=horizon)
        rates[name]=float(len(mem1["added_rows"])/mem1["candidate_rows"])
        sizes[name]={"object_goal":len(mem0["rows"]),"spatial_added":len(mem1["rows"])}
        by_motion[name]={}
        for mname,memory in (("object_goal",mem0),("spatial_added",mem1)):
            z1,z5,chosen,weights=_retrieve(key,memory,qrows,run,values)
            mse[f"{name}/{mname}/top1"]=_decode_mse(model,norm,run,qrows,z1)
            mse[f"{name}/{mname}/top5_blend"]=_decode_mse(model,norm,run,qrows,z5)
            comps_q=[]
            for qi,r in enumerate(qrows):
                selected=chosen[qi]; comp=np.empty((len(selected),3),np.float64); missing=[]; missing_pos=[]
                for j,row_id in enumerate(selected):
                    ix=motion_lookup.get(int(row_id))
                    if ix is None: missing.append(int(row_id)); missing_pos.append(j)
                    else: comp[j]=motion_components[qi,ix]
                if missing: comp[missing_pos]=_pairwise_outcomes(run.motion[r:r+1],run.motion[np.asarray(missing)])[0]
                comps_q.append({"top1":comp[0],"top5_weighted":weights[qi]@comp})
            rows=[]
            for q in range(4):
                choose=bins==q
                rows.append({"quantile":f"Q{q+1}","n":int(choose.sum()),
                    "top1_components":np.mean([v["top1"] for v,keep in zip(comps_q,choose) if keep],axis=0).tolist(),
                    "top5_weighted_components":np.mean([v["top5_weighted"] for v,keep in zip(comps_q,choose) if keep],axis=0).tolist(),
                    "best_available_normalized_motion":float(best_score[choose].mean())})
            by_motion[name][mname]=rows
        print(f"{tag}: {name} spatial_add={len(mem1['added_rows'])}/{mem1['candidate_rows']}",flush=True)

    prior_diag=json.load(open(old_diag/"results.json"))
    prior_keys=json.load(open(old/"results.json"))
    result={"horizon":horizon,"beta":.01,"train_rows":int(len(train_rows)),"query_rows":int(len(qrows)),
        "query_episodes":int(len(qids)),"spatial_store_rows":int(len(spatial_store)),"spatial_store_episodes":int(len(spatial_ids-qids)),
        "keys":list(keys),"mse_mean":{k:float(v.mean()) for k,v in mse.items()},
        "spatial_add_rates":rates,"memory_sizes":sizes,"motion_by_best_available_quantile":by_motion,
        "motion_quantile_edges_from_previous_run":previous_edges.tolist(),
        "references":{"own":float(prior_keys["references"]["own"]),"prior":float(prior_keys["references"]["prior"]),
                      "random":float(prior_keys["references"]["random"]),
                      "motion_top5_object_goal":float(prior_diag["mse_mean"]["motion_top5_blend_object_goal"])},
        "method":{"feature7":"existing f_t, language-conditioned base+wrist image-token pool + proprio; train20 PCA",
                  "feature8":"new wrist-only language-conditioned image-token pool; train20 PCA, no proprio",
                  "threshold":"fixed q=0.9 train-memory nearest-neighbor distance; spatial candidates streamed in episode/step order",
                  "blend":"top-5 softmax(-distance/(0.5*d_ref))","parameters":"Enc/Prior/Dec and stored z values frozen; no spatial rows used for fitting"}}
    out=root/tag; out.mkdir(exist_ok=False); (out/"results.json").write_text(json.dumps(result,indent=2))
    return result


def _write_report(root,results,wrist_cache):
    prev_report=PREVIOUS_KEYS/"report.md"
    lines=["# Language-aware key comparison","",
      f"Previous six-key experiment (existing numerical values are referenced without recomputation): [{prev_report}]({prev_report}). This run computes only the two language-aware keys.","",
      f"Model, executed posterior z values, query episodes, train20 rows, and spatial store/query split are from `{PREVIOUS}`. The current f_t key reuses the cached language-conditioned base+wrist features. Wrist-language features were extracted separately using wrist RGB plus the original instruction and the frozen base PaliGemma: `{wrist_cache}`. PCA/scaling fits use only object/goal train20. Enc/Prior/Dec and z values remain frozen.","",
      "Key 7: cached language-conditioned base+wrist image-token pool plus proprio, train20 PCA. Key 8: language-conditioned wrist-only image-token pool, train20 PCA without proprio. The same fixed q=0.9 threshold is recalibrated on the train20 memory for each key; top-5 blend temperature is 0.5 × its train-memory d_ref. Query episode and init-state exclusions match the previous comparison.","",
      "## Top-1 and top-5 blend MSE", "",
      "The six earlier rows below are copied from the previous report's saved results; they were not recomputed. New language-key rows are from this run.","",
      "| H | key | top-1 OG | top-5 OG | top-1 +spatial | top-5 +spatial | spatial add rate | source |","|---:|---|---:|---:|---:|---:|---:|---|"]
    prior_json={h:json.load(open(PREVIOUS_KEYS/f"H{h}_beta0.01/results.json")) for h in (25,10)}
    old_names=["combined_base_wrist_proprio","wrist_only","proprio_only","wrist_plus_proprio","learned_z_contrastive","prior_z"]
    for d in results:
        h=d["horizon"]; old=prior_json[h]; om=old["mse_mean"]
        for name in old_names:
            lines.append(f"| {h} | {name} | {om[name+'/object_goal/top1']:.4f} | {om[name+'/object_goal/top5_blend']:.4f} | {om[name+'/spatial_added/top1']:.4f} | {om[name+'/spatial_added/top5_blend']:.4f} | {old['spatial_add_rates'][name]:.1%} | previous run |")
        m=d["mse_mean"]
        for name in d["keys"]:
            lines.append(f"| {h} | {name} | {m[name+'/object_goal/top1']:.4f} | {m[name+'/object_goal/top5_blend']:.4f} | {m[name+'/spatial_added/top1']:.4f} | {m[name+'/spatial_added/top5_blend']:.4f} | {d['spatial_add_rates'][name]:.1%} | current run |")
    for d in results:
        lines += ["",f"## H={d['horizon']}, β=0.01","",
          f"Training rows={d['train_rows']}; spatial query={d['query_rows']} rows / {d['query_episodes']} episodes; spatial store={d['spatial_store_rows']} rows / {d['spatial_store_episodes']} episodes.","",
          f"References copied from previous diagnostic: own={d['references']['own']:.4f}, Prior={d['references']['prior']:.4f}, random={d['references']['random']:.4f}, future-motion top-5={d['references']['motion_top5_object_goal']:.4f} (future-conditioned upper-bound diagnostic).","",
          "Selected-item future-motion distances by the same best-available object/goal motion-proximity quartiles as the previous run; components are mean position distance, rotation radians, and gripper distance:",""]
        for name in d["keys"]:
            lines += [f"### {name}","","| Memory | quantile | n | best available normalized motion | selected top-1 pos/rot/grip | selected top-5 weighted pos/rot/grip |","|---|---|---:|---:|---|---|"]
            for memory,rows in d["motion_by_best_available_quantile"][name].items():
                for r in rows:
                    a=r["top1_components"]; b=r["top5_weighted_components"]
                    lines.append(f"| {memory} | {r['quantile']} | {r['n']} | {r['best_available_normalized_motion']:.4f} | {a[0]:.4f}/{a[1]:.4f}/{a[2]:.4f} | {b[0]:.4f}/{b[1]:.4f}/{b[2]:.4f} |")
            lines.append("")
    lines += ["## Interpretation","",
      "The object/goal-only comparison does not show a consistent gain from language: compare each new top-5 row with the corresponding language-free row above. Spatial-added results are also affected by key-specific threshold acceptance rates and candidate composition. The wrist-only language feature is computed with one camera as the image prefix, so its representation differs from simply slicing the two-camera f_t pool. These offline decoder errors do not establish closed-loop policy benefit.",""]
    for d in results:
        prior=prior_json[d["horizon"]]["mse_mean"]; cur=d["mse_mean"]; h=d["horizon"]
        lines.append(f"- H={h}: language base+wrist OG top-5 {cur['language_base_wrist_proprio/object_goal/top5_blend']:.4f} vs language-free combined {prior['combined_base_wrist_proprio/object_goal/top5_blend']:.4f}; language wrist-only {cur['language_wrist_only/object_goal/top5_blend']:.4f} vs language-free wrist-only {prior['wrist_only/object_goal/top5_blend']:.4f}. Spatial-added language-key add rates are {d['spatial_add_rates']['language_base_wrist_proprio']:.1%}/{d['spatial_add_rates']['language_wrist_only']:.1%}.")
    (root/"report.md").write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,default=Path("/work/.skillnet-state/outputs/memory-pilot")); p.add_argument("--data",type=Path,default=DATA); p.add_argument("--checkpoint",type=Path,default=CKPT); p.add_argument("--wrist-cache",type=Path); p.add_argument("--seed",type=int,default=73); p.add_argument("--batch-size",type=int,default=2)
    a=p.parse_args(); runid=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    root=a.output_root/f"emergent-light-langkey-{runid}"; root.mkdir(parents=True,exist_ok=False)
    wrist_cache=a.wrist_cache or _extract_wrist_lang(a.data,a.checkpoint,root/"wrist_lang_cache",batch_size=a.batch_size)
    results=[_eval_horizon(root,h,wrist_cache,a.seed) for h in (25,10)]
    _write_report(root,results,wrist_cache); print(f"report: {root/'report.md'}",flush=True)


if __name__=="__main__": main()
