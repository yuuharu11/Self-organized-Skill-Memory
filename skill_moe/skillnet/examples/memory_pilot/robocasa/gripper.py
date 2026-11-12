"""Offline comparison of gripper control rules with the P5 RoboCasa arm draft.

The held-out demonstration actions are used only for scoring.  The classifier
and its switch threshold are fitted/selected on the memory-side split.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex
import expert

OUT = Path("/work/.skillnet-state/outputs/robocasa-memory/gripper-01")
EXPERT_OUT = Path("/work/.skillnet-state/outputs/robocasa-memory/expert-01")
SOURCE = expert.SOURCE
CHECKPOINT = expert.CHECKPOINT
SEED = 20261112
THRESHOLDS = np.arange(.1, 1.0, .1, dtype=np.float32)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def _sign(x):
    return np.where(np.asarray(x) >= 0, 1.0, -1.0).astype(np.float32)


def _initial_grip(hist_storage, state_storage):
    valid = np.flatnonzero(np.linalg.norm(np.asarray(hist_storage), axis=1) > 1e-8)
    if len(valid):
        return float(_sign(hist_storage[valid[-1], 11]))
    # RoboCasa state storage ends in the two gripper joint positions.  Open is
    # approximately (+/- .0205); closed is approximately (+/- .04).
    return 1.0 if float(np.asarray(state_storage)[14]) < .03 else -1.0


def _feature(hist, state):
    """10 executed stored-order actions + state + 6D EEF action-speed summary."""
    h = np.asarray(hist, np.float32)
    s = np.asarray(state, np.float32)
    if h.shape != (10, 12) or s.shape != (16,):
        raise ValueError(f"expected history [10,12], state [16], got {h.shape}, {s.shape}")
    speed = np.mean(np.abs(h[:, 5:11]), axis=0)
    return np.concatenate([h.reshape(-1), s, speed]).astype(np.float32)


def _label_next5(hist, state, target):
    """11 classes: no switch, then offset 0..4 × new sign (-1,+1)."""
    prev = _initial_grip(hist, state)
    g = _sign(np.asarray(target)[:5, 11])
    sw = np.flatnonzero(g != prev)
    if not len(sw):
        return 0
    i = int(sw[0])
    return 1 + i * 2 + (1 if g[i] > 0 else 0)


def _switch_positions(actions, initial):
    g = _sign(actions[:, 11])
    before = np.concatenate([[float(initial)], g[:-1]])
    ix = np.flatnonzero(g != before)
    return [(int(i), float(g[i])) for i in ix]


def _fit_classifier(rows, out, *, epochs=40):
    import torch
    from torch import nn

    torch.set_num_threads(8)
    rng = np.random.default_rng(SEED)
    # The fixed RM-C validation episodes stay outside classifier fitting.
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    reserved = set(tuple(map(int, x)) for x in expert._mem_val_episodes(manifest))
    fit_ids = np.asarray([i for i, r in enumerate(rows)
                          if (int(r["task"]), int(r["episode"])) not in reserved], np.int64)
    episode_pairs = sorted({(int(rows[i]["task"]), int(rows[i]["episode"])) for i in fit_ids})
    # Episode-level five-fold cross-fitting.  A row's classifier target never
    # shares its episode with the model that scores its fold; final inference
    # averages the five fold models.
    perm = rng.permutation(len(episode_pairs))
    folds = [set(episode_pairs[i] for i in perm[j::5]) for j in range(5)]
    X = np.stack([_feature(r["hist"], r["state"]) for r in rows])
    y = np.asarray([_label_next5(r["hist"], r["state"], r["target"]) for r in rows], np.int64)
    x_mean = X[fit_ids].mean(0)
    x_std = np.maximum(X[fit_ids].std(0), 1e-4)
    X = (X - x_mean) / x_std
    models = []
    fold_summaries = []
    for fi, held in enumerate(folds):
        train_ids = np.asarray([i for i in fit_ids if (int(rows[i]["task"]), int(rows[i]["episode"])) not in held], np.int64)
        if len(train_ids) == 0:
            raise RuntimeError("empty classifier fold")
        torch.manual_seed(SEED + fi)
        model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 11))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.as_tensor(X[train_ids], dtype=torch.float32)
        yt = torch.as_tensor(y[train_ids], dtype=torch.long)
        # Inverse-frequency class weights avoid the no-switch majority collapse.
        counts = torch.bincount(yt, minlength=11).float()
        weights = (counts.sum() / (counts + 1)).sqrt()
        weights = weights / weights.mean()
        for ep in range(int(epochs)):
            model.train()
            order = rng.permutation(len(train_ids))
            for start in range(0, len(order), 256):
                ix = torch.as_tensor(order[start:start + 256], dtype=torch.long)
                logits = model(xt[ix])
                loss = nn.functional.cross_entropy(logits, yt[ix], weight=weights)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        models.append(model)
        fold_summaries.append({"fold": fi, "training_rows": len(train_ids), "held_episode_count": len(held),
                               "label_counts": counts.int().tolist()})
    payload = {"models": [m.state_dict() for m in models], "input_mean": x_mean, "input_std": x_std,
               "feature_dim": int(X.shape[1]), "reserved_episodes": sorted([list(x) for x in reserved]),
               "fit_episodes": len(episode_pairs), "folds": fold_summaries, "seed": SEED}
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(payload, tmp); tmp.replace(out)
    return payload


def _load_classifier(path):
    import torch
    from torch import nn
    torch.set_num_threads(8)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    models=[]
    for state in ck["models"]:
        m=nn.Sequential(nn.Linear(ck["feature_dim"],128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,11))
        m.load_state_dict(state);m.eval();models.append(m)
    return ck, models


def _probs(models, ck, hist, state):
    x = (_feature(hist, state) - ck["input_mean"]) / ck["input_std"]
    logits_all=[]
    for m in models:
        layers=[z for z in m if hasattr(z,"weight")]
        y=x
        for j,layer in enumerate(layers):
            w=layer.weight.detach().numpy();b=layer.bias.detach().numpy()
            y=y@w.T+b
            if j<len(layers)-1:y=np.maximum(y,0)
        logits_all.append(y)
    logits=np.mean(logits_all,axis=0)
    logits -= logits.max()
    p = np.exp(logits); return p / p.sum()


def _apply_g4(hist, state, arm, models, ck, threshold):
    """Roll a 50-step gripper trace, querying the classifier each five steps."""
    grip = np.empty((len(arm),), np.float32)
    h = np.asarray(hist, np.float32).copy()
    s = np.asarray(state, np.float32).copy()
    current = _initial_grip(h, s)
    for start in range(0, len(arm), 5):
        p = _probs(models, ck, h, s)
        switch_prob = float(1 - p[0])
        block = min(5, len(arm) - start)
        if switch_prob >= threshold:
            cls = int(np.argmax(p[1:]) + 1)
            offset = (cls - 1) // 2
            new = 1.0 if (cls - 1) % 2 else -1.0
            grip[start:start + min(offset, block)] = current
            if offset < block:
                grip[start + offset:start + block] = new
                current = new
        else:
            grip[start:start + block] = current
        # Advance only with the actions that would have been executed.
        executed = np.asarray(arm[start:start + block], np.float32).copy()
        executed[:, 11] = grip[start:start + block]
        h = np.concatenate([h, executed], axis=0)[-10:]
        # Keep estimated arm pose fixed; update gripper joints to canonical
        # open/closed state.  This is an offline approximation, not simulation.
        s[14:16] = np.asarray([.0205, -.0205] if current > 0 else [.04, -.04], np.float32)
    return grip


def _g3_trace(pool, selected, initial, horizon=50):
    """Switch only at an actually switch-ending selected source unit."""
    signs=[]; current=float(initial)
    for j in selected:
        j=int(j); start=int(pool["action_offsets"][j]); end=int(pool["action_offsets"][j+1])
        seg=np.asarray(pool["actions_flat"][start:end],np.float32)
        if not len(seg): continue
        n=min(len(seg), horizon-len(signs))
        signs.extend([current]*n)
        if len(signs)>=horizon: break
        source_sign=_sign(seg[:,11])
        # A unit is constant-gripper internally.  Its *end boundary* is a
        # switch if the next contiguous unit in that source episode starts at
        # the opposite command.
        ge=int(pool["global_episode"][j]); next_frame=int(pool["frame"][j])+len(seg)
        nxt=np.flatnonzero((pool["global_episode"]==ge)&(pool["frame"]==next_frame))
        if len(nxt):
            nj=int(nxt[0]); na=int(pool["action_offsets"][nj]); nb=int(pool["action_offsets"][nj+1])
            if nb>na:
                next_sign=float(_sign(pool["actions_flat"][na,11]))
                if next_sign != float(source_sign[0]): current=next_sign
    if len(signs)<horizon: signs.extend([current]*(horizon-len(signs)))
    return np.asarray(signs[:horizon],np.float32)


def _metrics(pred, target, initial):
    pred=np.asarray(pred,np.float32);target=np.asarray(target,np.float32)
    g=_sign(pred); truth=_sign(target)
    a=_switch_positions(np.column_stack([np.zeros((len(pred),11)),g]),initial)
    b=_switch_positions(np.column_stack([np.zeros((len(target),11)),truth]),initial)
    # Greedy one-to-one matching by same direction, nearest time.
    remaining=set(range(len(a))); errors=[]; missed=0
    for tb, direction in b:
        candidates=[i for i in remaining if a[i][1]==direction]
        if not candidates: missed+=1;continue
        i=min(candidates,key=lambda z:abs(a[z][0]-tb));remaining.remove(i);errors.append(abs(a[i][0]-tb))
    false=len(remaining)
    return {"mse5":float(np.mean((pred[:5]-target[:5])**2)),"mse50":float(np.mean((pred-target)**2)),
            "switch_count_true":len(b),"switch_count_pred":len(a),"switch_errors":errors,
            "missed":missed,"false":false,"timing_abs_errors":errors}


def _collect_demo_rows(out, split, max_rows=None):
    """Use existing cross-fitted P5 drafts where present; expand memory val to all windows."""
    z=np.load(EXPERT_OUT/"drafts/train_dev.npz",allow_pickle=False)
    cache={k:z[k] for k in z.files}
    want_val=split=="validation"
    records=[]
    if want_val:
        for i in range(len(cache["task"])):
            if not bool(cache["validation"][i]): continue
            r={k:np.asarray(cache[k][i]) for k in ("task","episode","frame","draft","target","hist","state")}
            records.append({k:(int(v) if k in ("task","episode","frame") else v) for k,v in r.items()})
    if split=="heldout":
        manifest=json.loads((SOURCE/"manifest.json").read_text())
        for info in manifest["demos"]:
            if info["split"]!="heldout":continue
            act,state,_=ex._load_cached_episode(SOURCE,int(info["task_id"]),int(info["episode"]))
            if len(act)<50:continue
            for start in range(0,len(act)-49,10):
                hist=np.zeros((10,12),np.float32);prev=act[max(0,start-10):start]
                if len(prev):hist[-len(prev):]=prev
                records.append({"task":int(info["task_id"]),"episode":int(info["episode"]),"frame":start,
                                "target":act[start:start+50].copy(),"hist":hist,"state":state[start].copy(),"draft":None,
                                "manifest":info})
    if max_rows is not None:
        records=records[:int(max_rows)]
    return records


def _p5_keys(models, pool):
    import torch
    from experiment import _load_make_net
    torch.set_num_threads(8)
    result=[]
    with torch.no_grad():
        for net, _ck in models:
            key=torch.as_tensor(pool["key"],dtype=torch.float32)
            result.append(torch.nn.functional.normalize(net.key(key),dim=-1).cpu().numpy())
    return result


def _chain_fast(net_ck_pairs, pool, action_hist, state, key_emb, *, episode=None, length=50):
    """Same autoregressive P5 chain with candidate key embeddings precomputed."""
    import torch
    from experiment import _features
    hist=np.asarray(action_hist,np.float32).copy();chunks=[];chosen=[]
    for _ in range(100):
        scores=[]
        for (net,ck),keys in zip(net_ck_pairs,key_emb):
            x=_features({"hist":hist[None],"state":np.asarray(state,np.float32)[None]},ck["norm"])
            with torch.no_grad():
                q=torch.nn.functional.normalize(net.query(torch.as_tensor(x,dtype=torch.float32)),dim=-1)[0].cpu().numpy()
                scale=float(net.log_scale.detach().exp().cpu())
            scores.append((keys@q)*scale)
        score=np.mean(scores,axis=0)
        if episode is not None:score[np.asarray(pool["global_episode"])==int(episode)]=-np.inf
        j=int(np.argmax(score));start=int(pool["action_offsets"][j]);end=int(pool["action_offsets"][j+1])
        raw=np.asarray(pool["actions_flat"][start:end],np.float32).copy()
        if np.linalg.norm(hist[:,5:11])>1e-7:raw=ex._scale_to_ref(raw,hist)
        emit=raw[:min(len(raw),length-sum(len(x) for x in chunks))]
        chunks.append(emit);chosen.append(j);hist=np.concatenate([hist,emit],axis=0)[-10:]
        if sum(len(x) for x in chunks)>=length:break
    if not chunks:return np.zeros((length,12),np.float32),[]
    return np.concatenate(chunks,axis=0)[:length],chosen


def _make_p5_draft(r, pool, models, key_emb):
    if r["draft"] is not None:
        # For validation, reconstruct selected ids only when G3 asks for them.
        return np.asarray(r["draft"],np.float32), None
    d,ids=_chain_fast(models,pool,r["hist"],r["state"],key_emb,episode=None,length=50)
    return d,ids


def _memory_validation_rows():
    z0=np.load(EXPERT_OUT/"drafts/train_dev.npz",allow_pickle=False)
    z={k:z0[k] for k in z0.files}
    rows=[]
    for i in np.flatnonzero(z["validation"]):
        row={k:z[k][i] for k in ("task","episode","frame","draft","target","hist","state")}
        rows.append({k:(int(v) if k in ("task","episode","frame") else np.asarray(v)) for k,v in row.items()})
    return rows


def select_threshold(out=OUT, *, max_rows=None):
    import torch
    torch.set_num_threads(8)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model_path=out/"g4_classifier.pt"
    z0=np.load(EXPERT_OUT/"drafts/train_dev.npz",allow_pickle=False)
    all_cache={k:z0[k] for k in z0.files}
    if model_path.exists(): ck,models=_load_classifier(model_path)
    else:
        fit_rows=[]
        for i in range(len(all_cache["task"])):
            fit_rows.append({k:(int(all_cache[k][i]) if k in ("task","episode","frame") else np.asarray(all_cache[k][i]))
                             for k in ("task","episode","frame","draft","target","hist","state")})
        ck=_fit_classifier(fit_rows,model_path)
        ck,models=_load_classifier(model_path)
    rows=_memory_validation_rows()
    # Fixed, episode-stratified calibration sample: retain switch windows first
    # and add non-switch windows with a deterministic per-episode sample.
    rng=np.random.default_rng(SEED+91); sample=[]
    grouped={}
    for r in rows:grouped.setdefault((r["task"],r["episode"]),[]).append(r)
    for pair,rr in sorted(grouped.items()):
        sw=[r for r in rr if _label_next5(r["hist"],r["state"],r["target"])!=0]
        ns=[r for r in rr if _label_next5(r["hist"],r["state"],r["target"])==0]
        take=list(sw)
        if len(take)>4:take=[take[i] for i in rng.choice(len(take),4,replace=False)]
        n_extra=max(0,5-len(take))
        if n_extra and ns:
            take.extend([ns[i] for i in rng.choice(len(ns),min(n_extra,len(ns)),replace=False)])
        sample.extend(take)
    rows=sample
    # All fixed Expert-validation episodes are considered; this includes switch
    # windows, unlike the earlier frame-zero-only selection.
    pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
    p5=[expert._load_net(SOURCE/f"models/P5_U_seed{s}.pt",pool,device="cpu") for s in (0,1,2)]
    key_emb=_p5_keys(p5,pool)
    if max_rows is not None: rows=rows[:int(max_rows)]
    scores={}; switch_rows=0
    for r in rows:
        draft=np.asarray(r["draft"],np.float32)
        target=np.asarray(r["target"],np.float32)
        if len(_switch_positions(target[:5],_initial_grip(r["hist"],r["state"]))):switch_rows+=1
        for th in THRESHOLDS:
            pred=_apply_g4(r["hist"],r["state"],draft,models,ck,float(th))
            scores.setdefault(float(th),[]).append(float(np.mean((pred[:5]-target[:5,11])**2)))
    means={str(th):float(np.mean(v)) for th,v in scores.items()}
    best=float(min(scores,key=lambda th:float(np.mean(scores[th]))))
    selection={"selected_threshold":best,"selection_metric":"memory-side reserved episodes; mean first-5 raw gripper command MSE",
               "selection_rows":len(rows),"switch_bearing_first5_rows":switch_rows,"threshold_mse":means,
               "sampling":"per reserved episode: retain up to 4 switch-bearing windows then sample non-switch windows to 5 total; fixed seed",
               "classifier":"5-fold episode-level cross-fit ensemble; fitted only on non-reserved memory episodes",
               "state_update":"during 50-step offline rollout, EEF pose held fixed and gripper joints set to canonical open/closed values"}
    _write_json(out/"selection.json",selection)
    return selection


def _normal_predictions(rows, out, *, limit=None):
    """Generate ordinary pi05 chunks. This stage needs the shared GPU/JAX model."""
    import jax
    from openpi.shared.nnx_utils import module_jit
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available() and os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        torch.cuda.set_per_process_memory_fraction(.10)
    dest=Path(out)/"normal_cache.npz"
    dest.parent.mkdir(parents=True,exist_ok=True)
    cache={}
    if dest.exists():
        z=np.load(dest,allow_pickle=False);cache={int(i):a for i,a in zip(z["row"],z["actions"])}
    if len(cache)>=len(rows) and all(i in cache for i in range(len(rows))):
        return cache
    base,_=expert._load_expert_model()
    normal_fn=module_jit(base.sample_actions,static_argnames=("num_steps",))
    stats=__import__("openpi.shared.normalize",fromlist=["load"]).load(CHECKPOINT/"assets")["actions"]
    done=set(cache)
    info_by_pair={(int(x["task_id"]),int(x["episode"])):x for x in json.loads((SOURCE/"manifest.json").read_text())["demos"]}
    batch_size=8
    pending=[i for i in range(len(rows)) if i not in done]
    for base_i in range(0,len(pending),batch_size):
        indices=pending[base_i:base_i+batch_size]
        obslist=[]; noises=[]
        for i in indices:
            r=rows[i];info=info_by_pair[(int(r["task"]),int(r["episode"]))]
            obs,_,_,_=expert._build_obs_and_target(info,int(r["frame"]),SOURCE,CHECKPOINT)
            obslist.append(obs)
            key=jax.random.fold_in(jax.random.key(4101),int(r["task"])*1_000_000+int(r["episode"])*1000+int(r["frame"]))
            noises.append(jax.random.normal(key,(1,50,32)))
        batched_obs=jax.tree_util.tree_map(lambda *xs: __import__("jax.numpy",fromlist=["concatenate"]).concatenate(xs,axis=0),*obslist)
        pred=np.asarray(normal_fn(jax.random.key(0),batched_obs,num_steps=10,noise=__import__("jax.numpy",fromlist=["concatenate"]).concatenate(noises,axis=0)))
        for i,pred_i in zip(indices,pred,strict=True):
            raw_policy=expert._denorm_action(pred_i[:,:12],stats)
            cache[i]=ex.action_policy_to_storage(raw_policy).astype(np.float32)
        ids=np.asarray(sorted(cache),np.int32);np.savez_compressed(dest,row=ids,actions=np.stack([cache[j] for j in ids]))
        print(json.dumps({"stage":"normal","rows":len(cache)}),flush=True)
    ids=np.asarray(sorted(cache),np.int32);np.savez_compressed(dest,row=ids,actions=np.stack([cache[j] for j in ids]))
    return cache


def evaluate(out=OUT, *, smoke=False, include_normal=True):
    import torch
    torch.set_num_threads(8)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selection=select_threshold(out)
    ck,models=_load_classifier(out/"g4_classifier.pt")
    pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
    p5=[expert._load_net(SOURCE/f"models/P5_U_seed{s}.pt",pool,device="cpu") for s in (0,1,2)]
    key_emb=_p5_keys(p5,pool)
    rows=_collect_demo_rows(EXPERT_OUT,"heldout",max_rows=(24 if smoke else None))
    if include_normal:
        try: normal=_normal_predictions(rows,out)
        except Exception as e:
            (out/"normal_unavailable.txt").write_text(f"{type(e).__name__}: {e}\n")
            normal={}
    else: normal={}
    result_path=out/"evaluation.jsonl"
    existing={}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            try:
                x=json.loads(line);existing[(x["task"],x["episode"],x["frame"])]=x
            except Exception:pass
    fout=result_path.open("a",buffering=1)
    val_metrics=[]
    if smoke: rows=rows[:24]
    added_normal=False
    for i,r in enumerate(rows):
        key=(r["task"],r["episode"],r["frame"])
        if key in existing:
            if i in normal and "G1_pi05" not in existing[key]["metrics"]:
                # Existing JSONL rows intentionally store aggregate metrics only;
                # target commands are available from the held-out cache here.
                initial=_initial_grip(r["hist"],r["state"])
                m=_metrics(normal[i][:50,11],np.asarray(r["target"])[:50,11],initial)
                m["switch_bearing_window"]=bool(m["switch_count_true"]>0)
                existing[key]["metrics"]["G1_pi05"]=m;added_normal=True
            continue
        draft,selected=_make_p5_draft(r,pool,p5,key_emb)
        initial=_initial_grip(r["hist"],r["state"])
        target=np.asarray(r["target"],np.float32)
        # All compared methods share precisely the P5 arm channels.
        pred={"G0_type":draft[:,11].copy()}
        pred["G2_hold"] = np.full(50,initial,np.float32)
        if selected is None:
            _,selected=_chain_fast(p5,pool,r["hist"],r["state"],key_emb,episode=None,length=50)
        pred["G3_unit_boundary"]=_g3_trace(pool,selected,initial)
        pred["G4_classifier"]=_apply_g4(r["hist"],r["state"],draft,models,ck,selection["selected_threshold"])
        if i in normal:
            pred["G1_pi05"] = normal[i][:50,11].copy()
        elif normal and str(i) in normal: pred["G1_pi05"]=normal[str(i)][:50,11].copy()
        per={}
        for name,g in pred.items():
            met=_metrics(g,target[:,11],initial)
            met["switch_bearing_window"]=bool(met["switch_count_true"]>0)
            per[name]=met
        rec={"task":int(r["task"]),"episode":int(r["episode"]),"frame":int(r["frame"]),
             "switch_bearing":bool(_metrics(target[:,11],target[:,11],initial)["switch_count_true"]>0),"metrics":per}
        fout.write(json.dumps(rec,separators=(",",":"))+"\n")
        if (i+1)%100==0:print(json.dumps({"stage":"evaluate","rows":i+1}),flush=True)
    fout.close()
    if added_normal:
        tmp=result_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for rec in existing.values():f.write(json.dumps(rec,separators=(",",":"))+"\n")
        tmp.replace(result_path)
    return summarize(out)


def summarize(out=OUT):
    out=Path(out);path=out/"evaluation.jsonl"
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    names=sorted({k for r in rows for k in r["metrics"]})
    summary={}
    for subset,label in ((rows,"all"),([r for r in rows if r["switch_bearing"]],"true_switch_in_50")):
        summary[label]={}
        for name in names:
            vals=[r["metrics"][name] for r in subset if name in r["metrics"]]
            if not vals:continue
            errs=[e for v in vals for e in v["timing_abs_errors"]]
            summary[label][name]={"n_windows":len(vals),"grip_mse_first5_mean":float(np.mean([v["mse5"] for v in vals])),
                "grip_mse_50_mean":float(np.mean([v["mse50"] for v in vals])),
                "switch_windows":sum(v["switch_count_true"]>0 for v in vals),
                "switch_events_true":sum(v["switch_count_true"] for v in vals),"switch_events_pred":sum(v["switch_count_pred"] for v in vals),
                "timing_abs_error_mean":float(np.mean(errs)) if errs else None,"timing_abs_error_median":float(np.median(errs)) if errs else None,
                "timing_abs_error_p90":float(np.quantile(errs,.9)) if errs else None,
                "missed_switches":sum(v["missed"] for v in vals),"false_switches":sum(v["false"] for v in vals),
                "miss_rate":float(sum(v["missed"] for v in vals)/max(1,sum(v["switch_count_true"] for v in vals))),
                "false_rate":float(sum(v["false"] for v in vals)/max(1,sum(v["switch_count_pred"] for v in vals)))}
    result={"rows":len(rows),"summary":summary,"notes":{"mse":"raw stored-order gripper command space, commands sign-normalized to +/-1","timing":"switch events matched one-to-one by direction; unmatched events counted as misses/false switches","holdout":"offline held-out demos; not closed-loop"}}
    _write_json(out/"summary.json",result)
    _report(out,result)
    return result


def _report(out,result):
    lines=["実装: 完了","","# RoboCasa gripper policy offline comparison","",
           "Arm channels are the same P5 draft for G0/G2/G3/G4; only gripper commands differ. π0.5 is the ordinary policy command stream. G4 trains on memory episodes outside the fixed validation episodes; its threshold is selected on reserved memory-side validation windows. Held-out actions are scoring-only. The first-five metric is raw ±1 gripper-command MSE. Timing events are matched one-to-one by direction; unmatched events are misses/false switches. This is offline demonstration scoring, not closed-loop success.","",
           "| subset | method | windows | grip MSE first5 | grip MSE 50 | switch timing median / mean / p90 (steps) | missed / false count (rate) |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for subset,methods in result["summary"].items():
        for name,m in methods.items():
            timing="—" if m["timing_abs_error_mean"] is None else f"{m['timing_abs_error_median']:.1f} / {m['timing_abs_error_mean']:.1f} / {m['timing_abs_error_p90']:.1f}"
            lines.append(f"| {subset} | {name} | {m['n_windows']} | {m['grip_mse_first5_mean']:.4f} | {m['grip_mse_50_mean']:.4f} | {timing} | {m['missed_switches']} ({m['miss_rate']:.3f}) / {m['false_switches']} ({m['false_rate']:.3f}) |")
    sel=json.loads((Path(out)/"selection.json").read_text())
    lines += ["",f"G4 threshold selected on {sel['selection_rows']} memory-side windows ({sel['switch_bearing_first5_rows']} with a true next-5 switch): {sel['selected_threshold']:.1f}; selection criterion was first-five raw grip MSE.","",
              "Limitations: G4 is a small action/state-only MLP (no wrist image). In the 50-step offline rollout the EEF state is held fixed while gripper joints are updated to canonical open/closed values; an environment would provide updated state every replan. G3 reads only unit boundaries and their stored switch direction.",""]
    (Path(out)/"report.md").write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("select","normal","evaluate","report"));p.add_argument("--out",type=Path,default=OUT);p.add_argument("--smoke",action="store_true");p.add_argument("--skip-normal",action="store_true")
    a=p.parse_args()
    if a.stage=="select":print(json.dumps(select_threshold(a.out),indent=2))
    elif a.stage=="normal":
        rows=_collect_demo_rows(EXPERT_OUT,"heldout",max_rows=(24 if a.smoke else None))
        got=_normal_predictions(rows,a.out);print(json.dumps({"normal_rows":len(got)}))
    elif a.stage=="evaluate":print(json.dumps(evaluate(a.out,smoke=a.smoke,include_normal=not a.skip_normal),indent=2))
    else:print(json.dumps(summarize(a.out),indent=2))

if __name__=="__main__":main()
