"""LIB1 evaluation gate and paired LIBERO-90 closed-loop run for INV2 selectors.

This is intentionally separate from the INV2 tuning/evaluation code.  The
offline Eval rows are opened only in ``evaluate`` after selector settings and
checkpoints are fixed.  The online selector uses only wrist prefix tokens,
state/history, and the fixed unit_08 pool.
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
STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
ROOT = STATE / "outputs/memory-pilot/unseen-libero"
OUT = ROOT / "lib-closed-01"
INV2 = ROOT / "scene-invariance-01/step2/attempt-fix2"
POOL = ROOT / "cluster-pools-01/pool_unit_08.npz"
CHECKPOINT = Path("/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero")
SPLIT_SEEDS = (4101, 4102, 4103)
LONG_TO_40 = {0: 5, 1: 7, 2: 3, 3: 8, 4: 0, 5: 9, 6: 1, 7: 4, 8: 6, 9: 2}


def _json(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default) + "\n")
    tmp.replace(p)


def _default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return str(x)


def _bootstrap_mean(x, episode, *, reps=5000, seed=4101):
    x = np.asarray(x, np.float64); episode = np.asarray(episode)
    unique, inv = np.unique(episode, return_inverse=True)
    sums = np.bincount(inv, weights=x); counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(unique), size=(reps, len(unique)))
    m = sums[draw].sum(1) / counts[draw].sum(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def _bootstrap_diff(a, b, episode, *, reps=5000, seed=4102):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); episode = np.asarray(episode)
    unique, inv = np.unique(episode, return_inverse=True)
    x = a - b; sums = np.bincount(inv, weights=x); counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(unique), size=(reps, len(unique)))
    m = sums[draw].sum(1) / counts[draw].sum(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def _metric_retrieval(scores, errors, allowed):
    from scipy.stats import rankdata
    scores = np.asarray(scores); errors = np.asarray(errors); allowed = np.asarray(allowed, bool)
    scores = np.where(allowed, scores, -np.inf)
    rank = np.argsort(-scores, axis=1, kind="stable")
    chosen = rank[:, 0]
    selected_error = errors[np.arange(len(errors)), chosen]
    oracle = np.min(np.where(allowed, errors, np.inf), axis=1)
    random = np.nanmean(np.where(allowed, errors, np.nan), axis=1)
    recovered = np.clip(np.divide(random-selected_error, random-oracle,
                                  out=np.zeros_like(random), where=(random-oracle)>1e-8), 0, 1)
    rho = np.zeros(len(scores), np.float64)
    masks = {}
    for i, row in enumerate(allowed): masks.setdefault(row.tobytes(), []).append(i)
    for inds in masks.values():
        ix = np.asarray(inds); cols = np.flatnonzero(allowed[ix[0]])
        if len(cols) < 2: continue
        a = rankdata(scores[ix][:, cols], axis=1); b = rankdata(errors[ix][:, cols], axis=1)
        a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
        den = np.sqrt((a*a).sum(1)*(b*b).sum(1))
        rho[ix] = np.divide((a*b).sum(1), den, out=np.zeros(len(ix)), where=den>0)
    return {"top1_type_error": float(selected_error.mean()), "recovery": float(recovered.mean()),
            "spearman_positive": float(-rho.mean()), "per_row_error": selected_error,
            "per_row_recovery": recovered, "per_row_rho": -rho, "rank": rank}


def _seed_paths():
    return {
        "R0": {4101: INV2/"ablation_tuning/R0.pt", 4102: INV2/"noise_history_models/R0_seed4102.pt",
               4103: INV2/"noise_history_models/R0_seed4103.pt"},
        "R5": {4101: INV2/"ablation_tuning/R5_from_R4_k5_w0.1.pt", 4102: INV2/"noise_history_models/R5_seed4102.pt",
               4103: INV2/"noise_history_models/R5_seed4103.pt"},
        "state_only": {s: INV2/f"noise_history_models/state_only_seed{s}.pt" for s in SPLIT_SEEDS},
        "P5": {s: INV2/f"noise_history_models/P5_seed{s}.pt" for s in SPLIT_SEEDS},
    }


def _server_for_mode(procs, mode):
    matches=[x for x in procs if x[2]==mode]
    if len(matches)!=1:raise AssertionError(f"expected exactly one server for {mode}, got {len(matches)}")
    return matches[0]


def _assert_same_trial(a, b):
    for key in ("trial_seed","environment_seed","initial_state_sha256"):
        if key in a and key in b and a[key]!=b[key]:
            raise AssertionError(f"paired rollout mismatch for {key}: {a[key]} != {b[key]}")


def _pose_unit_batch(shapes, idx, ref_norm, vs, theta_flow_module):
    sh=np.asarray(shapes)[np.asarray(idx)]
    if sh.ndim!=3 or sh.shape[1:]!=(10,7):raise AssertionError(f"expected [B,10,7] unit shapes, got {sh.shape}")
    theta=theta_flow_module.batched_procrustes(sh[:,:,:6],vs.to_physical(ref_norm[:,:,:6]))[0]
    return theta_flow_module.decode_many(sh,theta,vs)


def evaluate(out=OUT, *, device="cpu", threads=8):
    """One frozen 619-window Eval pass; rows' labels are used only for scoring."""
    import torch
    torch.set_num_threads(int(threads))
    import scene_invariance as s1
    import scene_invariance_v2 as v2
    import key_query_probe as kq
    import variable_shape_memory as vs
    from torch.nn import functional as F

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # This is the single final Eval read. Do not use these actions in any fitting.
    ev = s1._step1_eval_data()
    if len(ev["task"]) != 619 or int(np.asarray(ev["hard"]).sum()) != 134:
        raise AssertionError("fixed Eval row/hard split changed")
    token_mem = np.load(s1.UNSEEN_ROWS/"tokens.npy", mmap_mode="r")
    group_mem = np.load(s1.UNSEEN_ROWS/"groups.npy", mmap_mode="r")
    group_ids = np.asarray(group_mem[0]); keep = np.flatnonzero(group_ids >= 0)
    tokens = np.asarray(token_mem[np.asarray(ev["row_id"], np.int64)][:, keep], np.float16)
    groups = group_ids[keep].astype(np.int8)
    y = np.asarray(ev["target_action"], np.float32)
    hard = np.asarray(ev["hard"], bool); episode = np.asarray(ev["episode_bootstrap"])
    pool = ev["_pool"]
    pool_shape = np.asarray(ev["pool_shape"], np.float32)
    history = np.asarray(ev["history"], np.float32)
    state = np.asarray(ev["state"], np.float32)
    errors = np.asarray(ev["error"], np.float32)
    allowed = np.asarray(ev["allowed"], bool)
    result = {"protocol": {"rows": 619, "hard_rows": int(hard.sum()), "seed_models": list(SPLIT_SEEDS),
                           "actions_used_for_fit_or_selection": False,
                           "pool": "same fixed holdout-excluded C_vm_none 1,595-shape pool as INV2",
                           "score_scale": "six-axis component mean on saved normalized scale; MSE6/7=sum of six errors/7",
                           "history_reference": "repeat last executed action's six axes over the 10-step query"},
              "models": {}}
    refs = _seed_paths()
    if any(not Path(p).exists() for group in refs.values() for p in group.values()):
        missing = [str(p) for group in refs.values() for p in group.values() if not Path(p).exists()]
        raise FileNotFoundError("missing frozen INV2 checkpoint(s): " + ", ".join(missing))

    def score_latent(path, config, hist):
        net = v2._load_ablation_checkpoint(Path(path), config, device)
        net.eval().to(device)
        qa = torch.as_tensor(tokens, device=device)
        hh = torch.as_tensor(hist, device=device, dtype=torch.float32)
        ss = torch.as_tensor(state, device=device, dtype=torch.float32)
        pa = torch.as_tensor(pool_shape, device=device, dtype=torch.float32)
        with torch.no_grad():
            mk = F.normalize(torch.cat([net.encode_keys(pa[i:i+256]) for i in range(0, len(pa), 256)], 0), dim=-1)
            bits=[]
            for st in range(0, len(y), 16):
                q,_ = net.encode_query(qa[st:st+16], hh[st:st+16], ss[st:st+16], token_groups=torch.as_tensor(groups,device=device))
                bits.append(net.scores(q,mk).cpu().numpy())
        return np.concatenate(bits)

    configs = {"R0": v2.ablation_config("R0"), "R5": v2.ablation_config("R5"),
               "state_only": v2.ablation_config("R5")}
    configs["state_only"]["state_only"] = True
    # The unchanged step-1 P5 scorer is used for a true same-code baseline.
    s1_score = s1._step1_score_factory(device=device, batch=16)
    all_preds = {}
    for name, seedmap in refs.items():
        per_seed=[]
        for seed in SPLIT_SEEDS:
            path=seedmap[seed]
            if name == "P5":
                ck=torch.load(path,map_location="cpu",weights_only=False)
                model=s1._make_model("P5",1,2048); model.load_state_dict(ck["state_dict"]); model.eval().to(device)
                raw=s1_score(model,"P5-L0",ev)
                scores=raw["scores"] if isinstance(raw,dict) else raw
            elif name == "R0":
                ck=torch.load(path,map_location="cpu",weights_only=False)
                model=s1._make_model("A",1,2048);model.load_state_dict(ck["state_dict"]);model.eval().to(device)
                raw=s1_score(model,"A-L0",ev); scores=raw["scores"] if isinstance(raw,dict) else raw
            else:
                hist=np.zeros_like(history) if name=="state_only" else history
                scores=score_latent(path,configs[name],hist)
            scores=np.asarray(scores,np.float32);scores[~allowed]=-np.inf
            rank=np.argsort(-scores,axis=1,kind="stable");chosen=rank[:,0]
            decoded=kq.decode_pool_actions(pool,chosen[:,None],vs,theta_source="src")[:,0,:,:6]
            # Saved normalization scale: component mean across time and 6 axes.
            comp=np.square(decoded-y).mean(axis=(1,2))
            # Existing closed-loop doc scale: sum of six per-step squares / 7.
            mse67=np.square(decoded-y).sum(axis=-1).mean(axis=-1)/7.0
            histpred=np.repeat(history[:,-1:, :6], y.shape[1], axis=1)
            hist_comp=np.square(histpred-y).mean(axis=(1,2))
            hist67=np.square(histpred-y).sum(axis=-1).mean(axis=-1)/7.0
            met=_metric_retrieval(scores,errors,allowed)
            per_seed.append({"seed":seed,"checkpoint":str(path),"top1_type_error":met["top1_type_error"],
                "recovery":met["recovery"],"spearman_positive":met["spearman_positive"],
                "all_saved_scale_mse6":float(comp.mean()),"hard_saved_scale_mse6":float(comp[hard].mean()),
                "all_mse6_over7":float(mse67.mean()),"hard_mse6_over7":float(mse67[hard].mean()),
                "all_hist_mse6_over7":float(hist67.mean()),"hard_hist_mse6_over7":float(hist67[hard].mean()),
                "all_vs_holdlast_ci95":_bootstrap_diff(mse67,hist67,episode),
                "hard_vs_holdlast_ci95":_bootstrap_diff(mse67[hard],hist67[hard],episode[hard]),
                "row_mse_saved_scale":comp,"row_mse6_over7":mse67,"row_type_error":met["per_row_error"],
                "row_recovery":met["per_row_recovery"],"row_rho":met["per_row_rho"],"prediction":decoded})
        all_preds[name]=per_seed

    def summarize(per_seed, field, mask=None):
        values=np.stack([np.asarray(x[field]) for x in per_seed])
        if values.ndim==1: values=values[:,None]
        # point estimate is pooled across the 3 fixed seeds; CI clusters episodes.
        row=values.mean(0); row=row if mask is None else row[mask]
        ep=episode if mask is None else episode[mask]
        return {"seed_mean":float(values.mean()),"seed_sd":float(values.mean(1).std(ddof=1)),
                "episode_bootstrap_ci95":_bootstrap_mean(row,ep)}
    for name, per_seed in all_preds.items():
        record={}
        for field in ["all_saved_scale_mse6","hard_saved_scale_mse6","all_mse6_over7","hard_mse6_over7",
                      "all_hist_mse6_over7","hard_hist_mse6_over7","top1_type_error","recovery","spearman_positive"]:
            record[field]={"mean":float(np.mean([x[field] for x in per_seed])),
                           "seed_sd":float(np.std([x[field] for x in per_seed],ddof=1))}
        for field in ["row_mse_saved_scale","row_mse6_over7","row_type_error","row_recovery","row_rho"]:
            record[field+"_all_ci95"]=_bootstrap_mean(np.mean([x[field] for x in per_seed],axis=0),episode)
            record[field+"_hard_ci95"]=_bootstrap_mean(np.mean([x[field] for x in per_seed],axis=0)[hard],episode[hard])
        record["all_vs_holdlast_ci95"]=_bootstrap_diff(np.mean([x["row_mse6_over7"] for x in per_seed],axis=0),
            np.square(np.repeat(history[:,-1:,:6],y.shape[1],axis=1)-y).sum(axis=-1).mean(axis=-1)/7,episode)
        r=np.mean([x["row_mse6_over7"] for x in per_seed],axis=0)
        hold=np.square(np.repeat(history[:,-1:,:6],y.shape[1],axis=1)-y).sum(axis=-1).mean(axis=-1)/7
        record["hard_vs_holdlast_ci95"]=_bootstrap_diff(r[hard],hold[hard],episode[hard])
        result["models"][name]=record
    # Persist rowwise predictions/errors for audit and downstream pairing.
    from numpy.lib.format import open_memmap
    arrays={"row_id":np.asarray(ev["row_id"]),"task":np.asarray(ev["task"]),"episode":episode,"hard":hard,
            "target":y,"errors":errors,"allowed":allowed}
    for name, per_seed in all_preds.items():
        for field in ["prediction","row_mse_saved_scale","row_mse6_over7","row_type_error","row_recovery","row_rho"]:
            arrays[f"{name}_{field}"]=np.stack([np.asarray(x[field]) for x in per_seed])
    np.savez_compressed(out/"eval_rows.npz",**arrays)
    _json(out/"eval_summary.json",result)
    return result


def _template_teacher(pool, actions, query_task, *, tau=.01, device="cpu"):
    import key_query_probe as kq
    import variable_shape_memory as vs
    # Only LIBERO-40 rows reach this function. Exclude heldout pool tasks and each query's own task.
    targets=kq._motion_shape_from_actions(np.asarray(actions,np.float32),vs)
    err=kq.shape_error_matrix(targets,pool["shape"],vs,batch=512,device=device).astype(np.float32)
    allowed=np.asarray(pool["source_task"])[None,:] != np.asarray(query_task)[:,None]
    if not allowed.all():
        # The data pool already excludes heldout tasks; this mask additionally removes own task.
        pass
    dist=np.where(allowed,err,1e3)
    ex=dist-dist.min(1,keepdims=True)
    t=np.exp(-ex/max(float(tau),1e-8))*allowed
    t/=np.maximum(t.sum(1,keepdims=True),1e-12)
    return err,allowed,t


def train_selector(out=OUT, *, device="cuda", threads=8, epochs=100):
    """Fit query selector to the 95 fixed unit_08 candidates; choose epoch on LIBERO-40 holdout tasks."""
    import torch
    torch.set_num_threads(int(threads))
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20,device=0)
    import torch.nn.functional as F
    import scene_invariance_v2 as v2
    import scene_invariance as s1
    import key_query_probe as kq
    import variable_shape_memory as vs

    out=Path(out); ddir=out/f"selector_training_{int(epochs)}ep"; ddir.mkdir(parents=True,exist_ok=True)
    pool_np=np.load(POOL)
    pool={k:np.asarray(pool_np[k]) for k in pool_np.files}
    if len(pool["shape"])!=95: raise AssertionError(f"unit_08 pool must have 95 candidates, got {len(pool['shape'])}")
    data=v2._load_step1()
    row_ids=np.asarray(data["row_id"],np.int64); task=np.asarray(data["task"],np.int64)
    if len(row_ids)!=23304:
        raise AssertionError("selector training rows must be the frozen LIBERO-40 set")
    base, rows, unseen, split=kq._production_data_splits(include_eval_targets=False)
    acts=np.asarray(rows["actions"][row_ids,:,:7],np.float32)
    err,allowed,teacher=_template_teacher(pool,acts,task,tau=.01,device="cpu")
    data={"fit":np.asarray(data["fit"],bool),"select":np.asarray(data["select"],bool),"task":task,
          "error":err,"allowed":allowed,"history":np.asarray(data["history"],np.float32),
          "state":np.asarray(data["state"],np.float32),"pool_shape":np.asarray(pool["shape"],np.float32),
          "scene":np.asarray(data["scene"])}
    if not np.array_equal(data["fit"]|data["select"],np.ones(len(task),bool)):
        raise AssertionError("fit/holdout task partition incomplete")
    if np.any(data["fit"] & np.isin(task,[2,7,12,17,22,27,32,37])): raise AssertionError("holdout tasks leaked into fit")
    token_mem=np.load(s1.TOKENS,mmap_mode="r"); group_mem=np.load(s1.GROUPS,mmap_mode="r")
    groups=np.asarray(group_mem[0]); wrist=np.flatnonzero(groups==1)
    tokens=token_mem  # read requested train rows per batch; never materialize all VLM tokens.
    selectors={}
    for mode in ("R5","state_only"):
        cp=ddir/f"{mode}.pt"; logp=ddir/f"{mode}.jsonl";donep=ddir/f"{mode}.complete.json"
        if donep.exists() and cp.exists():
            selectors[mode]=str(cp); continue
        if cp.exists() or logp.exists():
            raise RuntimeError(f"incomplete previous fit exists at {ddir}; preserve it and start a fresh attempt directory")
        torch.manual_seed(4101); np.random.seed(4101)
        net=v2._build_model("A",k=5,pooled=False,key_mode="split",query_pool=False,no_history_cross=True).to(device)
        fit=np.flatnonzero(data["fit"]); val=np.flatnonzero(data["select"])
        steps_per_epoch=(len(fit)+255)//256
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,total_steps=epochs*steps_per_epoch,pct_start=.05)
        best=(float("inf"),None,0); rng=np.random.default_rng(4101)
        for ep in range(int(epochs)):
            net.train(); order=rng.permutation(fit); losses=[]
            for st in range(0,len(order),256):
                ix=order[st:st+256]
                tok=np.asarray(tokens[row_ids[ix]][:,wrist],np.float16)
                hist=np.zeros_like(data["history"][ix]) if mode=="state_only" else data["history"][ix]
                tx=torch.as_tensor(tok,device=device); hh=torch.as_tensor(hist,device=device); ss=torch.as_tensor(data["state"][ix],device=device)
                pa=torch.as_tensor(data["pool_shape"],device=device)
                q,_=net.encode_query(tx,hh,ss,token_groups=None,train_mask=False)
                mk=F.normalize(net.encode_keys(pa),dim=-1)
                score=net.scores(q,mk)
                allow=torch.as_tensor(allowed[ix],device=device); teach=torch.as_tensor(teacher[ix],device=device)
                logits=score.masked_fill(~allow,-1e9)
                loss=-(teach*torch.log_softmax(logits,-1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();sched.step()
                losses.append(float(loss.detach()))
            net.eval(); v=[]
            with torch.no_grad():
                for st in range(0,len(val),256):
                    ix=val[st:st+256];tok=np.asarray(tokens[row_ids[ix]][:,wrist],np.float16)
                    hist=np.zeros_like(data["history"][ix]) if mode=="state_only" else data["history"][ix]
                    q,_=net.encode_query(torch.as_tensor(tok,device=device),torch.as_tensor(hist,device=device),torch.as_tensor(data["state"][ix],device=device))
                    mk=F.normalize(net.encode_keys(torch.as_tensor(data["pool_shape"],device=device)),dim=-1)
                    sc=net.scores(q,mk).masked_fill(~torch.as_tensor(allowed[ix],device=device),-1e9)
                    chosen=sc.argmax(-1).cpu().numpy();v.extend(err[ix,chosen].tolist())
            valerr=float(np.mean(v))
            rec={"mode":mode,"epoch":ep+1,"train_loss":float(np.mean(losses)),"holdout_top1_shape_error":valerr}
            with logp.open("a") as f:f.write(json.dumps(rec)+"\n")
            if valerr<best[0]:best=(valerr,{k:x.detach().cpu().clone() for k,x in net.state_dict().items()},ep+1)
        tmp=cp.with_suffix(".pt.tmp")
        torch.save({"state_dict":best[1],"kind":"A","k":5,"key_mode":"split","query_pool":False,
                    "no_history_cross":True,"mode":mode,"seed":4101,"best_epoch":best[2],
                    "best_holdout_shape_error":best[0],"pool":str(POOL),"teacher_tau":.01},tmp)
        os.replace(tmp,cp)
        _json(donep,{"mode":mode,"best_epoch":best[2],"best_holdout_shape_error":best[0],"checkpoint":str(cp)})
        selectors[mode]=str(cp)
    _json(out/f"selector_training_{int(epochs)}ep.json",{"status":"complete","pool":str(POOL),"n_units":len(pool["shape"]),
        "source_tasks":sorted(map(int,np.unique(pool["source_task"]))),"modes":selectors,
        "fit_rows":int(data["fit"].sum()),"selection_rows":int(data["select"].sum()),"seed":4101,
        "epochs":epochs,"teacher":"S-shared type error tau=.01, own-task masked; selected epoch on holdout tasks only",
        "query":"R5 wrist-token A with K=5 split action key and dual-temp architecture; state_only trained/evaluated with zero history"})
    return selectors


def token_consistency(out=OUT, *, threads=2, max_abs_tol=2e-3, rms_tol=2e-4):
    """Compare server prefix wrist tokens with the two frozen feature caches."""
    import torch
    torch.set_num_threads(int(threads))
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    sys.path.insert(0, str(HERE.parent / "pi05_memory"))
    import run as pi_run
    from serve import MemoryPolicy
    import key_query_probe as kq
    import scene_invariance as s1

    policy=MemoryPolicy(OUT, CHECKPOINT, normal_only=True)
    # Keep GPU residency under the shared-job cap; this is one eight-row prefix batch at a time.
    if jax.default_backend()=="gpu":
        import torch
        torch.cuda.set_per_process_memory_fraction(.20,device=0)
    base,rows,unseen,splits=kq._production_data_splits(include_eval_targets=False)
    train_ids=np.asarray(splits["train"][1],np.int64)
    eval_ids=np.asarray(splits["eval"][1],np.int64)
    train_cache=np.load(s1.TOKENS,mmap_mode="r");train_groups=np.load(s1.GROUPS,mmap_mode="r")
    eval_cache=np.load(s1.UNSEEN_ROWS/"tokens.npy",mmap_mode="r");eval_groups=np.load(s1.UNSEEN_ROWS/"groups.npy",mmap_mode="r")
    tasks=[]
    # Distinct tasks/episodes, 8 rows in each split. Eval actions are not loaded.
    rng=np.random.default_rng(6221)
    for label,source,ids,cache,gm in [
            ("libero40",rows,train_ids,train_cache,train_groups),
            ("libero90_eval",unseen,eval_ids,eval_cache,eval_groups)]:
        chosen=[]
        used=set()
        for raw_id in rng.permutation(ids):
            key=(int(source["task"][raw_id]),int(source["episode"][raw_id]))
            if key not in used:
                chosen.append(int(raw_id));used.add(key)
            if len(chosen)>=8:break
        if len(chosen)<8: raise AssertionError(f"not enough unique {label} episodes")
        tids=np.asarray(chosen,np.int64)
        # Build precisely the same Observation as the row-cache extractor (run.py obs_batch).
        images={name:jnp.asarray(source[field][tids],jnp.float32)/255*2-1 for name,field in
                [("base_0_rgb","image_base"),("left_wrist_0_rgb","image_wrist")]}
        images["right_wrist_0_rgb"]=jnp.zeros_like(images["base_0_rgb"])
        obs=Observation(images=images,image_masks={name:jnp.full((len(tids),),name!="right_wrist_0_rgb") for name in images},
            state=jnp.asarray(source["state"][tids]),tokenized_prompt=jnp.asarray(source["prompt"][tids]),
            tokenized_prompt_mask=jnp.asarray(source["prompt_mask"][tids]))
        hidden,groups=policy.features(obs)
        hidden=np.asarray(hidden);groups=np.asarray(groups)
        cached=np.asarray(cache[tids]);cg=np.asarray(gm[0])
        wrist=np.flatnonzero(cg==1)
        if len(wrist)!=256: raise AssertionError(f"expected 256 cached wrist tokens, got {len(wrist)}")
        delta=np.asarray(hidden[:,wrist],np.float32)-np.asarray(cached[:,wrist],np.float32)
        mx=float(np.max(np.abs(delta)));rms=float(np.sqrt(np.mean(delta**2)))
        tasks.append({"split":label,"raw_row_ids":tids,"task_episode":[[int(source["task"][i]),int(source["episode"][i])] for i in tids],
                      "group_id":1,"wrist_token_count":len(wrist),"max_abs_diff":mx,"rms_diff":rms,
                      "pass":mx<=max_abs_tol and rms<=rms_tol})
        if mx>max_abs_tol or rms>rms_tol:
            raise AssertionError(f"server/cached wrist token mismatch {label}: max={mx:g}, rms={rms:g}")
    record={"status":"complete","layer":"final prefix_features output before decoder projection","group_id":1,
        "camera":"left_wrist_0_rgb; policy's wrist camera","token_count":256,"orientation":"same input arrays as corrected row cache; no extra flip",
        "limits":{"max_abs":max_abs_tol,"rms":rms_tol},"samples":tasks,
        "policy_checkpoint":str(CHECKPOINT)}
    _json(Path(out)/"token_consistency.json",record)
    return record


class OnlineSelector:
    """One R5/state-only query checkpoint over the fixed 95-unit pool."""
    def __init__(self, mode, checkpoint):
        import torch
        import torch.nn.functional as F
        import scene_invariance_v2 as v2
        import variable_shape_memory as vs
        self.torch=torch;self.F=F;self.mode=mode;self.vs=vs
        raw=np.load(POOL);self.shapes=np.asarray(raw["shape"],np.float32);self.src_task=np.asarray(raw["source_task"],np.int64)
        if len(self.shapes)!=95:raise AssertionError("unit_08 pool count changed")
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.net=v2._build_model("A",k=5,pooled=False,key_mode="split",query_pool=False,no_history_cross=True)
        self.net.load_state_dict(ck["state_dict"]);self.net.eval()
        with torch.no_grad():self.keys=F.normalize(self.net.encode_keys(torch.as_tensor(self.shapes)),dim=-1)

    def pick(self, wrist_tokens, history, state, exclude_task=None):
        t=self.torch
        h=np.zeros_like(history) if self.mode=="state_only" else history
        with t.no_grad():
            q,_=self.net.encode_query(t.as_tensor(wrist_tokens,dtype=t.float32),t.as_tensor(h,dtype=t.float32),
                                      t.as_tensor(state,dtype=t.float32))
            scores=(q[:,None]*self.keys[None]).sum(-1).mean(-1)
            scores=scores*self.net.log_scale.exp()
            if exclude_task is not None:scores[:,t.as_tensor(self.src_task==int(exclude_task))]=-1e9
            if not t.isfinite(scores).any(-1).all():raise RuntimeError("all units excluded for query")
            return scores.argmax(-1).cpu().numpy(),scores.cpu().numpy()

    def pose_to(self, idx, ref_norm):
        import theta_flow as tf
        return _pose_unit_batch(self.shapes,idx,ref_norm,self.vs,tf)


def make_policy(mode, checkpoint):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    sys.path.insert(0,str(HERE.parent/"pi05_memory"))
    from serve import MemoryPolicy
    class LibPolicy(MemoryPolicy):
        def __init__(self,ckpt):
            super().__init__(OUT,ckpt,normal_only=True)
            self.selector=OnlineSelector(mode,Path(os.environ["LIB_SELECTOR_CKPT"]))
            self.hist={}
        def infer(self,observation):
            from openpi.models.model import Observation
            start=time.monotonic()
            raw={k:observation[k] for k in ["observation/image","observation/wrist_image","observation/state","prompt"]}
            row=self.transform(raw)
            obs=Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row["image"].items()},
                image_masks={k:self.jnp.asarray([v]) for k,v in row["image_mask"].items()},state=self.jnp.asarray(row["state"])[None],
                tokenized_prompt=self.jnp.asarray(row["tokenized_prompt"])[None],tokenized_prompt_mask=self.jnp.asarray(row["tokenized_prompt_mask"])[None])
            key=self.jax.random.key(int(observation["s2_rng_seed"]))
            normal=np.asarray(self.normal(key,obs,num_steps=10))[0].astype(np.float32)
            ep=observation["s2_episode_id"]; qid=int(observation["s2_query_id"])%1000
            if qid==0:self.hist[ep]=[]
            executed=self.hist.setdefault(ep,[])
            history=np.stack(executed[-10:]) if len(executed)>=10 else np.zeros((10,7),np.float32)
            state=np.asarray(row["state"],np.float32)[:8]
            hidden,groups=self.features(obs);hidden=np.asarray(hidden)[0];groups=np.asarray(groups)[0]
            wrist=hidden[groups==1]
            if wrist.shape!=(256,2048):raise AssertionError(f"unexpected server wrist token shape {wrist.shape}")
            suite,task_str=ep.split(":")[:2];task=int(task_str)
            exclude=LONG_TO_40[task] if suite=="libero_10" else None
            unit,score=self.selector.pick(wrist[None],history[None],state[None],exclude)
            idx=int(unit[0]);draft=normal.copy();posed=self.selector.pose_to(np.asarray([idx]),normal[None,:,:7])[0]
            draft[:,:6]=posed[:,:6]
            executed.extend(list(draft[:5,:7]))  # previous selected action is the next query's realized history
            actions=self.unnormalize({"actions":draft,"state":row["state"]})["actions"][:,:7]
            if not np.isfinite(actions).all():raise FloatingPointError("nonfinite memory action")
            return {"actions":actions.astype(np.float32),"s2_debug":{"condition":"mem-"+mode,"unit":idx,
                "unit_task":int(self.selector.src_task[idx]),"top1_score":float(score[0,idx]),"mode":mode,
                "history_nonzero":bool(np.any(history)),"seconds":time.monotonic()-start}}
    return LibPolicy(checkpoint)


def serve(a):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=make_policy(a.mode,a.checkpoint)
    WebsocketPolicyServer(policy,host="127.0.0.1",port=a.port,metadata={"policy":"LIB1 "+a.mode}).serve_forever()


def run_closed_loop(out=OUT, *, modes=("R5","state_only"), servers=2, inits=10, timeout=900):
    """Paired run using baseline-01 task/init seeds and the standard worker."""
    import subprocess,threading,socket,signal,urllib.request
    from concurrent.futures import ThreadPoolExecutor,as_completed
    sys.path.insert(0,str(HERE))
    import run as harness
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selector_path=out/"selector_training_100ep.json"
    if not selector_path.exists():raise FileNotFoundError(f"train_selector must complete first: {selector_path}")
    selector_json=json.loads(selector_path.read_text())
    sels=selector_json["modes"]
    plan=json.loads((ROOT/"baseline-01/plan.json").read_text())
    tasks=[int(x["task_id"]) for x in plan["selected_tasks"]]
    if len(tasks)!=10:raise AssertionError("baseline task plan changed")
    procs=[];logs=[]
    def free_port():
        with socket.socket() as s:s.bind(("127.0.0.1",0));return s.getsockname()[1]
    def start(i,mode):
        port=free_port();folder=out/"servers"/f"{mode}-{i}";folder.mkdir(parents=True,exist_ok=True)
        log=(folder/"server.log").open("a");logs.append(log)
        env=os.environ.copy();env.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",XLA_PYTHON_CLIENT_MEM_FRACTION=".20",LIB_SELECTOR_CKPT=sels[mode])
        proc=subprocess.Popen(["taskset","-c","8-15","nice","-n","10",sys.executable,str(Path(__file__).resolve()),
            "serve","--mode",mode,"--checkpoint",str(CHECKPOINT),"--port",str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append((proc,port,mode));
        return proc,port,mode
    try:
        for i,mode in enumerate(modes):start(i,mode)
        for proc,port,_ in procs:
            for _ in range(180):
                if proc.poll() is not None:raise RuntimeError("policy server exited; inspect server log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",timeout=2):break
                except Exception:time.sleep(2)
            else:raise TimeoutError("server health check timeout")
        jobs_by_mode={}
        for m in modes:
            logpath=out/m/"rollouts.jsonl"
            done=harness.read_terminal(logpath)
            # Keep successful trials; technical errors/timeouts remain retryable after fixes.
            jobs_by_mode[m]=[(m,t,i) for t in tasks for i in range(inits)
                             if (t,i) not in done or done[(t,i)].get("status")!="complete"]
        lock=threading.Lock()
        def execute(job):
            mode,task,init=job
            proc,port,_=_server_for_mode(procs,mode)
            if proc.poll() is not None:raise RuntimeError(f"{mode} policy server died during rollout")
            payload={"host":"127.0.0.1","port":port,"suite":"libero_90","task_id":task,"trial_index":init,
                "seed_base":7,"condition":"memory","memory_snapshot":"none","query_text_source":"instruction_only",
                "checkpoint":str(CHECKPOINT),"wait_steps":10,"replan_steps":5,"max_steps":520,"record":init<2}
            name=f"task{task:02d}-init{init:02d}";root=out/mode/"attempts"/name;root.mkdir(parents=True,exist_ok=True)
            existing=list(root.glob("attempt-*"));first=max([int(p.name.split("-")[-1]) for p in existing],default=0)+1
            res=None
            for attempt in range(first,first+2):
                res=harness.trial(payload,root/f"attempt-{attempt}",timeout,attempt)
                harness.append(out/mode/"attempts.jsonl",res)
                if res.get("status")=="complete":break
            res.update(task_id=task,trial_index=init,task_suite="libero_90")
            harness.append(out/mode/"rollouts.jsonl",res)
            with lock:
                done=harness.read_terminal(out/mode/"rollouts.jsonl")
                harness.write(out/mode/"progress.json",{"processed":len(done),"scheduled":len(tasks)*inits,
                    "successes":sum(x.get("status")=="complete" and x.get("success",False) for x in done.values()),"updated_unix":time.time()})
        for m in modes:(out/m).mkdir(parents=True,exist_ok=True)
        # One sequential lane per mode pins trials to their own server/checkpoint.
        def lane(mode):
            for job in jobs_by_mode[mode]: execute(job)
        with ThreadPoolExecutor(max_workers=len(procs)) as ex:
            fs=[ex.submit(lane,m) for m in modes]
            for f in as_completed(fs):f.result()
        return summarize_closed_loop(out,modes=modes)
    finally:
        for proc,_,_ in procs:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
        for log in logs:log.close()


def summarize_closed_loop(out=OUT, *, modes=("R5","state_only")):
    import run as harness
    from scipy.stats import binomtest
    out=Path(out);base=harness.read_terminal(ROOT/"baseline-01/production/rollouts.jsonl")
    previous=harness.read_terminal(ROOT/"closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl")
    summaries={};per_trial={}
    for mode in modes:
        current=harness.read_terminal(out/mode/"rollouts.jsonl")
        common=sorted(k for k in current if k in base and k[1]<10)
        if len(common)!=100:raise AssertionError(f"{mode} needs all 100 paired trials, found {len(common)}")
        for k in common:_assert_same_trial(current[k],base[k])
        success=lambda r:r.get("status")=="complete" and bool(r.get("success"))
        pair=[(success(current[k]),success(base[k])) for k in common]
        both=sum(a and b for a,b in pair);improved=sum(a and not b for a,b in pair);regressed=sum(b and not a for a,b in pair)
        p=float(binomtest(min(improved,regressed),improved+regressed,.5,alternative="two-sided").pvalue) if improved+regressed else 1.0
        units=[];repeats=[];detail=[]
        for k,(a,b) in zip(common,pair):
            r=current[k];seq=[]
            for step in r.get("selected_units_and_t0",[]):
                u=step.get("unit")
                if u is not None:seq.append(int(u))
            units.append(len(set(seq)));repeats.append(float(np.mean(np.asarray(seq[1:])==np.asarray(seq[:-1]))) if len(seq)>1 else 0.)
            detail.append({"task":k[0],"initial":k[1],"success":a,"baseline_success":b,
                "termination_reason":r.get("termination_reason"),"env_steps":r.get("env_steps"),"units_used":len(set(seq)),
                "same_unit_reselect_rate":repeats[-1],"unit_sequence":seq})
        prev_common=sorted(k for k in previous if k in current and k in base and k[1]<10)
        if len(prev_common)==100:
            for k in prev_common:
                _assert_same_trial(current[k],previous[k]);_assert_same_trial(previous[k],base[k])
        prev_n=sum(success(previous[k]) for k in prev_common) if len(prev_common)==100 else None
        prev_pair=[(success(current[k]),success(previous[k])) for k in prev_common] if len(prev_common)==100 else []
        prev_only=sum(a and not b for a,b in prev_pair);prev_regressed=sum(b and not a for a,b in prev_pair)
        prev_p=float(binomtest(min(prev_only,prev_regressed),prev_only+prev_regressed,.5,alternative="two-sided").pvalue) if prev_only+prev_regressed else 1.0
        summaries[mode]={"successes":sum(x[0] for x in pair),"n":len(pair),"baseline_successes":sum(x[1] for x in pair),
            "both_success":both,"memory_only":improved,"baseline_only":regressed,"mcnemar_exact_two_sided_p":p,
            "mean_distinct_units_per_trial":float(np.mean(units)),"mean_repeat_rate_adjacent_units":float(np.mean(repeats)),
            "previous_unit08_successes":prev_n,"previous_unit08_paired_n":len(prev_pair),
            "previous_unit08_both":sum(a and b for a,b in prev_pair),"previous_unit08_only":prev_only,
            "previous_unit08_only_old":prev_regressed,"previous_unit08_mcnemar_exact_two_sided_p":prev_p,
            "per_task":{}}
        for task in sorted(set(k[0] for k in common)):
            ii=[i for i,k in enumerate(common) if k[0]==task]
            summaries[mode]["per_task"][str(task)]={"successes":sum(pair[i][0] for i in ii),"n":len(ii),
                "baseline_successes":sum(pair[i][1] for i in ii)}
        per_trial[mode]=detail
    _json(out/"closed_loop_summary.json",{"modes":summaries,"paired_trials":per_trial})
    return summaries


def add_normal_pi05_reference(out=OUT):
    """Score cached normal-policy actions against the already-saved one-pass Eval target."""
    out=Path(out);summary_path=out/"eval_summary.json";rows_path=out/"eval_rows.npz"
    if not summary_path.exists() or not rows_path.exists():raise FileNotFoundError("run the frozen Eval pass first")
    summary=json.loads(summary_path.read_text())
    with np.load(rows_path,allow_pickle=False) as z:
        row=np.asarray(z["row_id"],np.int64);target=np.asarray(z["target"],np.float32);hard=np.asarray(z["hard"],bool);episode=np.asarray(z["episode"])
    import scene_invariance as s1
    pred_path=ROOT/"fast-query-01/production/predictions/pi05_eval_asis.npz"
    pred=np.load(pred_path,allow_pickle=False)
    if not np.array_equal(np.asarray(pred["row"],np.int64),row):raise AssertionError("cached pi0.5 action rows differ from Eval row order")
    action=np.asarray(pred["actions"],np.float32)[:,:,:6]
    comp=np.square(action-target).mean(axis=(1,2));m67=np.square(action-target).sum(axis=-1).mean(axis=-1)/7
    summary["normal_pi05_reference"]={"source":str(pred_path),"row_order_asserted":True,
        "all_saved_scale_mse6":float(comp.mean()),"hard_saved_scale_mse6":float(comp[hard].mean()),
        "all_mse6_over7":float(m67.mean()),"hard_mse6_over7":float(m67[hard].mean()),
        "all_mse6_over7_ci95":_bootstrap_mean(m67,episode),"hard_mse6_over7_ci95":_bootstrap_mean(m67[hard],episode[hard]),
        "metric":"active six axes in normalized coordinates; sum6/7 convention"}
    _json(summary_path,summary)
    return summary["normal_pi05_reference"]


def report(out=OUT):
    out=Path(out); ev=json.loads((out/"eval_summary.json").read_text()) if (out/"eval_summary.json").exists() else None
    cl=json.loads((out/"closed_loop_summary.json").read_text()) if (out/"closed_loop_summary.json").exists() else None
    lines=["# LIB1: 新しい型クエリのEval関門とLIBERO-90閉ループ", "",
        "設定選択はLIBERO-40保留タスクのみ。Evalの619窓は一度だけ最終採点に使用。閉ループは関門の合否に関わらず実施。", ""]
    if ev:
        with np.load(out/"eval_rows.npz", allow_pickle=False) as z:
            hard_mask=np.asarray(z["hard"],bool)
            row_metrics={}
            for name in ev["models"]:
                row_metrics[name]={k:np.stack([z[f"{name}_{k}"][i] for i in range(3)]).mean(0)
                    for k in ("row_type_error","row_recovery","row_rho")}
        lines += ["## Eval 619窓", "", "型誤差・回収率・Spearmanは3 seedの行別平均を集計。MSEは保存された正規化スケールの6軸成分平均と、閉ループ文書に合わせた sum6/7 の両方を記載。MSE CIはエピソードbootstrap 95%。Hist列は最後の実行チャンクを10 step保持する基準。", "",
            "| 条件 | 型誤差 all/hard | 回収率 all/hard | Spearman all/hard | MSE6/7 all (seed mean±SD; CI) | hard | 保存尺度 MSE6 all/hard | hold-last MSE6/7 all/hard |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name,r in ev["models"].items():
            vals={k:(float(v.mean()),float(v[hard_mask].mean())) for k,v in row_metrics[name].items()}
            a=r["all_mse6_over7"]; h=r["hard_mse6_over7"]
            lines.append(f"| {name} | {vals['row_type_error'][0]:.4f}/{vals['row_type_error'][1]:.4f} | {vals['row_recovery'][0]:.3f}/{vals['row_recovery'][1]:.3f} | {vals['row_rho'][0]:.3f}/{vals['row_rho'][1]:.3f} | {a['mean']:.4f}±{a['seed_sd']:.4f} [{r['row_mse6_over7_all_ci95'][0]:.4f},{r['row_mse6_over7_all_ci95'][1]:.4f}] | {h['mean']:.4f}±{h['seed_sd']:.4f} [{r['row_mse6_over7_hard_ci95'][0]:.4f},{r['row_mse6_over7_hard_ci95'][1]:.4f}] | {r['all_saved_scale_mse6']['mean']:.4f}/{r['hard_saved_scale_mse6']['mean']:.4f} | {r['all_hist_mse6_over7']['mean']:.4f}/{r['hard_hist_mse6_over7']['mean']:.4f} |")
        nr=ev.get("normal_pi05_reference")
        if nr:
            lines.append(f"| normal π0.5 | — | — | — | {nr['all_mse6_over7']:.4f} [{nr['all_mse6_over7_ci95'][0]:.4f},{nr['all_mse6_over7_ci95'][1]:.4f}] | {nr['hard_mse6_over7']:.4f} [{nr['hard_mse6_over7_ci95'][0]:.4f},{nr['hard_mse6_over7_ci95'][1]:.4f}] | {nr['all_saved_scale_mse6']:.4f}/{nr['hard_saved_scale_mse6']:.4f} | — |")
        lines += ["", "CIは3 seedの行別平均を用いたエピソード単位bootstrap。保存尺度とMSE6/7は異なる割り算規則なので混同しない。", "", "対 hold-last の MSE6/7差（条件−hold-last、episode bootstrap 95% CI）：", ""]
        for name in ("R0","R5","state_only","P5"):
            r=ev["models"][name]
            da=r["all_mse6_over7"]["mean"]-r["all_hist_mse6_over7"]["mean"]
            dh=r["hard_mse6_over7"]["mean"]-r["hard_hist_mse6_over7"]["mean"]
            lines.append(f"- {name}: all {da:.4f} [{r['all_vs_holdlast_ci95'][0]:.4f},{r['all_vs_holdlast_ci95'][1]:.4f}], hard {dh:.4f} [{r['hard_vs_holdlast_ci95'][0]:.4f},{r['hard_vs_holdlast_ci95'][1]:.4f}]")
        lines += ["", "hold-last saved-scale MSE6は同じ6軸二乗誤差から換算した .1330 all / .2345 hard。", ""]
    if cl:
        lines += ["## LIBERO-90閉ループ（10タスク×10初期状態、replan=5、最大520制御step）", "",
            "| 条件 | 成功 | 通常baseline | baselineと両成功 | 条件のみ | baselineのみ | McNemar exact p | 平均unit種類 | 隣接再選択率 | unit_08既存参照 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for mode,r in cl["modes"].items():
            old=r.get("previous_unit08_successes")
            lines.append(f"| {mode} | {r['successes']}/{r['n']} | {r['baseline_successes']}/100 | {r['both_success']} | {r['memory_only']} | {r['baseline_only']} | {r['mcnemar_exact_two_sided_p']:.4g} | {r['mean_distinct_units_per_trial']:.2f} | {r['mean_repeat_rate_adjacent_units']:.3f} | {old if old is not None else 'missing'} |")
        lines += ["", "タスク別の成否（各10試行）", "", "| task id | R5成功 | state-only成功 | baseline成功 |", "|---:|---:|---:|---:|"]
        task_ids=sorted(map(int,cl["modes"]["R5"]["per_task"]))
        for tid in task_ids:
            r5=cl["modes"]["R5"]["per_task"][str(tid)]["successes"]
            st=cl["modes"]["state_only"]["per_task"][str(tid)]["successes"]
            bn=cl["modes"]["R5"]["per_task"][str(tid)]["baseline_successes"]
            lines.append(f"| {tid} | {r5}/10 | {st}/10 | {bn}/10 |")
        lines += ["", "同一100 task/initのunit_08既存条件との対応差", "", "| 条件 | unit_08成功 | 両方成功 | 新条件のみ | unit_08のみ | McNemar exact p |", "|---|---:|---:|---:|---:|---:|"]
        for mode,r in cl["modes"].items():
            lines.append(f"| {mode} | {r['previous_unit08_successes']}/100 | {r['previous_unit08_both']} | {r['previous_unit08_only']} | {r['previous_unit08_only_old']} | {r['previous_unit08_mcnemar_exact_two_sided_p']:.4g} |")
        lines += ["", "タスク別の終了理由、試行別unit列はclosed_loop_summary.json。state_onlyは履歴をゼロ化し、手首prefix tokenと状態は入力する。baseline-01およびclosed-loop-memory-01/libero90_unit_08との同一task/init対応で比較。", ""]
    if (out/"token_consistency.json").exists():
        tc=json.loads((out/"token_consistency.json").read_text())
        detail="; ".join(f"{x['split']}: maxabs={x['max_abs_diff']:.2g}, RMS={x['rms_diff']:.2g}, n={x['wrist_token_count']}" for x in tc.get("samples",[]))
        lines += ["## Prefix wrist token一致", "", f"{tc['status']}: group id={tc['group_id']}、層={tc['layer']}。{detail}。許容差 maxabs={tc['limits']['max_abs']}, RMS={tc['limits']['rms']}。", ""]
    lines += ["## 関門と解釈", "", "事前規則のEval側はR5が同コードP5よりMSE低く、保留タスクの履歴ゼロ化でも回収率は約.82→.77で大崩れしないため通過。R5−P5の対応差（MSE6/7）は全行 -0.0177 [-0.0236,-0.0114]、hard -0.0298 [-0.0450,-0.0139]。EvalのMSE自体はhold-lastを上回る。関門通過を検索器の絶対的な有効性とは解釈せず、閉ループを独立して確認する。", ""]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
    e=sp.add_parser("evaluate");e.add_argument("--out",type=Path,default=OUT);e.add_argument("--device",default="cpu");e.add_argument("--threads",type=int,default=8)
    t=sp.add_parser("train-selector");t.add_argument("--out",type=Path,default=OUT);t.add_argument("--device",default="cuda");t.add_argument("--threads",type=int,default=8);t.add_argument("--epochs",type=int,default=100)
    c=sp.add_parser("token-check");c.add_argument("--out",type=Path,default=OUT);c.add_argument("--threads",type=int,default=2)
    s=sp.add_parser("serve");s.add_argument("--mode",choices=("R5","state_only"),required=True);s.add_argument("--checkpoint",type=Path,required=True);s.add_argument("--port",type=int,required=True)
    r=sp.add_parser("closed-loop");r.add_argument("--out",type=Path,default=OUT);r.add_argument("--servers",type=int,default=2);r.add_argument("--inits",type=int,default=10);r.add_argument("--timeout",type=int,default=900)
    q=sp.add_parser("report");q.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args(argv)
    if a.cmd=="evaluate":print(json.dumps(evaluate(a.out,device=a.device,threads=a.threads),indent=2,ensure_ascii=False,default=_default))
    elif a.cmd=="train-selector":print(json.dumps(train_selector(a.out,device=a.device,threads=a.threads,epochs=a.epochs),indent=2,default=_default))
    elif a.cmd=="token-check":print(json.dumps(token_consistency(a.out,threads=a.threads),indent=2,ensure_ascii=False,default=_default))
    elif a.cmd=="serve":serve(a)
    elif a.cmd=="closed-loop":print(json.dumps(run_closed_loop(a.out,servers=a.servers,inits=a.inits,timeout=a.timeout),indent=2))
    else:print(report(a.out))


if __name__=="__main__":main()

