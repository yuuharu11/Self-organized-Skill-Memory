#!/usr/bin/env python3
"""MAG1-R: offline amplitude/duration ceilings on RoboCasa windows.

The six-task evaluation rows are exactly the frozen gt-def2 rows.  The only
selection rows are held-out episodes from the other twelve memory-18 tasks.
Oracle action values are used only by A1/T1 and scoring; candidate type is
always chosen from a task- and episode-disjoint pool.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex
from units import cut

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
ROOT = STATE / "outputs/robocasa-memory"
OUT = ROOT / "magnitude-01"
SRC = ROOT / "memory-01/policy-order-v2"
MEM18 = ROOT / "memory-18"
GT = ROOT / "gt-def2/per_query.npz"
GT3 = ROOT / "gt-def3/per_query.npz"
PI05 = ROOT / "gt-def3/pi05_predictions.npz"
CTX = ROOT / "ctx-01"
CKPT = STATE / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
EVAL_TASKS = tuple(range(6))
SEED = 20261113
MAX_LEN = 50


def _scale_fit(x, y):
    """Nonnegative shared least-squares scalar on arm xyz+rotation."""
    x = np.asarray(x, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    den = float(x @ x)
    return max(float(x @ y) / den, 0.0) if den > 1e-12 else 0.0


def _duration(candidate_len, remaining_len, candidate, recent_speed, method):
    if method == "T1":
        value = remaining_len
    elif method == "T2":
        value = candidate_len
    elif method == "T3":
        value = 10
    elif method == "T4":
        # Actions are per-step displacement/rotation increments, so total
        # traveled amount is the sum of command magnitudes, not the difference
        # between adjacent commands.
        path = float(np.linalg.norm(candidate[:, :6], axis=1).sum())
        value = int(round(path / max(float(recent_speed), 1e-6))) if recent_speed > 1e-6 else 10
    else:
        raise ValueError(method)
    return int(np.clip(value, 1, MAX_LEN))


def _amplitude(candidate, target, pi05, recent_speed, method):
    if method == "A1":
        return _scale_fit(candidate[:, :6], target[:, :6])
    if method == "A2":
        speed = float(np.linalg.norm(candidate[:, :6], axis=1).mean())
        return float(np.clip(recent_speed / max(speed, 1e-6), 0.0, 10.0))
    if method == "A3":
        return _scale_fit(candidate[:min(5, len(candidate)), :6], pi05[:min(5, len(candidate)), :6])
    if method == "A4":
        # The stored raw unit is the source amplitude; candidate is not
        # normalized before execution, so the multiplier is exactly one.
        return 1.0
    raise ValueError(f"{method} requires an expert/magnitude-head prediction")


def _row_prediction(unit, pool, off, flat, n, scale, grip):
    raw = np.asarray(flat[off:off + int(pool["length"][unit])], np.float32)
    draft = ex.resample_rows(raw, n)
    policy = ex.action_storage_to_policy(draft)
    policy[:, :6] *= float(scale)
    grip = np.asarray(grip, np.float32).reshape(-1)
    if len(grip) < len(policy):
        grip = np.concatenate([grip, np.repeat(grip[-1:], len(policy) - len(grip))])
    policy[:, 6] = grip[:len(policy)]
    if len(policy) < 10:
        policy = np.concatenate([policy, np.repeat(policy[-1:], 10 - len(policy), axis=0)])
    return policy[:10]


def _episode_units(actions):
    return cut(actions, min_length=10)


def _ood_task_ids(task_names, eval_names):
    return [i for i,name in enumerate(task_names) if name not in set(eval_names)]


def _norm_stats():
    p = CKPT / "assets/norm_stats.json"
    doc = json.loads(p.read_text())
    stats = doc["norm_stats"]["actions"]
    return np.asarray(stats["mean"], np.float32)[:7], np.asarray(stats["std"], np.float32)[:7]


def _load_gripper(prefix, state_policy):
    """Frozen Context classifier used by the existing RoboCasa protocol."""
    import torch
    ck = torch.load(CTX / "context_model.pt", map_location="cpu", weights_only=False)
    from torch import nn
    class ContextNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.query=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256))
            self.key=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,256))
            self.log_scale=nn.Parameter(torch.tensor(float(np.log(20.))))
            self.grip=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,6))
        def forward(self,q,keys):
            return nn.functional.normalize(self.query(q),dim=-1) @ nn.functional.normalize(self.key(keys),dim=-1).T * self.log_scale.exp()
    net = ContextNet(int(ck["input_dim"]))
    net.load_state_dict(ck["state"])
    net.eval()
    norm = ck["norm"]
    x = np.concatenate([
        (prefix - np.asarray(norm["feature_mean"], np.float32)) /
        np.asarray(norm["feature_std"], np.float32),
        (state_policy - np.asarray(norm["state_mean"], np.float32)) /
        np.asarray(norm["state_std"], np.float32),
    ], axis=1).astype(np.float32)
    with torch.no_grad():
        logits = net.grip(torch.as_tensor(x)).numpy()
    return np.where(logits[:, :5] >= 0, 1.0, -1.0).astype(np.float32)


def _magnitude_v2_scales(records, prefix, pool):
    """Run the frozen MAG-B v2 head on the oracle-selected candidate."""
    import torch
    from torch import nn
    cck=torch.load(CTX/"context_model.pt",map_location="cpu",weights_only=False)
    mck=torch.load(ROOT/"key-lite-01/magnitude_v2/magnitude_head.pt",map_location="cpu",weights_only=False)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(int(mck["input_dim"]),256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
        def forward(self,x):return self.net(x).squeeze(-1)
    net=Net();net.load_state_dict(mck["state"]);net.eval()
    norm=cck["norm"]
    states=np.stack([r["state_policy"] for r in records]).astype(np.float32)
    q=np.concatenate([
        (prefix-np.asarray(norm["feature_mean"],np.float32))/np.asarray(norm["feature_std"],np.float32),
        (states-np.asarray(norm["state_mean"],np.float32))/np.asarray(norm["state_std"],np.float32),
    ],axis=1).astype(np.float32)
    motion=np.asarray(pool["action10"],np.float32)[:,:,5:11]
    magnitude=np.sqrt(np.mean(np.sum(motion.astype(np.float64)**2,axis=2),axis=1)).astype(np.float32)
    meta=np.concatenate([np.asarray(pool["key"],np.float32),np.log(np.maximum(magnitude,1e-6))[:,None]],axis=1)
    units=np.asarray([r["unit_id"] for r in records],np.int64)
    x=np.concatenate([q,meta[units]],axis=1)
    if x.shape[1]!=int(mck["input_dim"]):
        raise ValueError(f"MAG-B feature contract mismatch: {x.shape[1]} vs {mck['input_dim']}")
    x=(x-np.asarray(mck["feature_mean"],np.float32))/np.asarray(mck["feature_std"],np.float32)
    with torch.no_grad():pred=net(torch.as_tensor(x)).numpy()
    return np.clip(np.exp(pred),.25,4.).astype(np.float32)


def _dev_nonoracle_metrics(records, grips, pool, mean, std):
    """Score OOD A2/A4 rules with the same fixed context-gripper head as Eval."""
    flat, offsets = pool["actions_flat"], pool["action_offsets"]
    metrics = {}
    for r, grip in zip(records, grips, strict=True):
        target = np.asarray(r["target"], np.float32)
        full = len(target) >= 10
        yn = (target[:10, :7] - mean) / (std + 1e-6)
        if len(yn) < 10:
            yn = np.concatenate([yn, np.repeat(yn[-1:], 10 - len(yn), axis=0)])
        for tm in ("T2", "T3", "T4"):
            n = _duration(r["candidate_length"], r["remaining"],
                          r["candidate_policy"], r["recent_speed"], tm)
            c = ex.action_storage_to_policy(ex.resample_rows(
                np.asarray(flat[int(offsets[r["unit_id"]]):int(offsets[r["unit_id"] + 1])], np.float32), n))
            a2 = _amplitude(c, target, np.zeros((5, 12), np.float32), r["recent_speed"], "A2")
            for am, scale in (("A2", a2), ("A4", 1.0)):
                pred = _row_prediction(r["unit_id"], pool, int(offsets[r["unit_id"]]),
                                       flat, n, scale, grip)
                pn = (pred[:, :7] - mean) / (std + 1e-6)
                sq = (pn - yn) ** 2
                sq6 = sq[:, :6]
                metrics.setdefault(f"{am}_{tm}", []).append({
                    "first5_mse7": float(sq[:5].mean()),
                    "cont5_mse7": float(sq[5:10].mean()) if full else None,
                    "active6_first5_mse": float(sq6[:5].mean()),
                    "active6_cont5_mse": float(sq6[5:10].mean()) if full else None,
                    "task": int(r["task"]), "episode": int(r["global_episode"]),
                })
    return metrics


def _bootstrap(values, tasks, episodes, *, draws=5000, seed=SEED):
    values = np.asarray(values, np.float64)
    keys = sorted(set(zip(np.asarray(tasks).tolist(), np.asarray(episodes).tolist())))
    means = np.asarray([values[(np.asarray(tasks) == t) & (np.asarray(episodes) == e)].mean()
                        for t, e in keys])
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(means), size=(draws, len(means)))
    return np.quantile(means[take].mean(1), [.025, .975]).tolist()


def _eval_records():
    z = np.load(GT, allow_pickle=False)
    z3 = np.load(GT3, allow_pickle=False)
    pi = np.load(PI05, allow_pickle=False)
    manifest = json.loads((SRC / "manifest.json").read_text())
    demos = {(int(d["task_id"]), int(d["episode"])): d for d in manifest["demos"]}
    global_to_demo = {int(d["global_episode"]): d for d in manifest["demos"]}
    # The stored gt-def2 row order is verified against gt-def3 and pi05 cache.
    if not (np.array_equal(z["task"], z3["task"]) and
            np.array_equal(z["episode"], z3["episode"]) and
            np.array_equal(z["frame"], z3["frame"]) and
            np.array_equal(z["task"], pi["task"]) and
            np.array_equal(z["episode"], pi["global_episode"]) and
            np.array_equal(z["frame"], pi["frame"])):
        raise ValueError("gt-def2, gt-def3 and π0.5 prediction rows are not aligned")
    return z, z3, pi, global_to_demo


def run(*, out=OUT, dev_per_task=50):
    import torch
    torch.set_num_threads(8)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    z, z3, pi, global_to_demo = _eval_records()
    # Compressed npz row indexing decompresses the full member each time.
    gt_type_errors = np.asarray(z["type_error"], np.float32)
    pool = dict(np.load(SRC / "pools/U_memory.npz", allow_pickle=False))
    flat, offsets = pool["actions_flat"], pool["action_offsets"]
    mean, std = _norm_stats()
    prefix = np.asarray(z["qquery_prefix"], np.float32)
    pi_actions = np.asarray(pi["actions_storage"][0], np.float32)
    records = []
    # Reconstruct exact target segments and query units from cached held-out demos.
    episode_cache = {}
    for i, (task, global_ep, frame) in enumerate(zip(z["task"], z["episode"], z["frame"], strict=True)):
        d = global_to_demo[int(global_ep)]
        key = (int(d["task_id"]), int(d["episode"]))
        if key not in episode_cache:
            a, s, _ = ex._load_cached_episode(SRC, *key)
            episode_cache[key] = (a, s, _episode_units(a), d)
        actions, states, units, d = episode_cache[key]
        f = int(frame)
        own = next((u for u in units if int(u["start"]) <= f < int(u["end"])), None)
        if own is None:
            raise ValueError(f"query {i} at {key}:{f} not assigned to a unit")
        target_storage = actions[f:min(f + 10, len(actions))]
        target = ex.action_storage_to_policy(target_storage)
        target_amplitude = ex.action_storage_to_policy(actions[f:min(f + MAX_LEN, len(actions))])
        if len(target) < 5:
            continue
        hist = actions[max(0, f - 10):f]
        recent = ex.action_storage_to_policy(hist)[:, :6] if len(hist) else np.zeros((0, 6), np.float32)
        speed = float(np.linalg.norm(recent, axis=1).mean()) if len(recent) else 0.0
        # Enforce both exclusions, regardless of the older gt-def2 mask.
        allow = (pool["task"] != int(task)) & (pool["global_episode"] != int(global_ep))
        err = gt_type_errors[i].copy()
        err[~allow] = np.inf
        if not np.isfinite(err).any():
            raise ValueError("no eligible candidate remains after task+episode exclusion")
        unit = int(np.argmin(err))
        off = int(offsets[unit])
        candidate_len = int(pool["length"][unit])
        raw_candidate = np.asarray(flat[off:off + candidate_len], np.float32)
        # For generated duration rules, fit using a fixed 10-step canonical
        # candidate/target; for each duration the candidate is resampled below.
        pi_store = pi_actions[i]
        pi_policy = ex.action_storage_to_policy(pi_store)
        own_remaining = int(own["end"]) - f
        grip = _load_gripper(prefix[i:i+1], ex.state_storage_to_policy(states[f:f+1]))[0]
        row = {"row": i, "task": int(task), "episode": int(global_ep), "frame": f,
               "unit_id": unit, "unit_task": int(pool["task"][unit]),
               "unit_global_episode": int(pool["global_episode"][unit]),
               "type_error": float(err[unit]), "remaining": own_remaining,
               "candidate_length": candidate_len, "gripper_context_head": grip.tolist(),
               "target": target, "target_amplitude":target_amplitude,"pi05": pi_policy, "recent_speed": speed,
               "candidate": raw_candidate,
               "state_policy": ex.state_storage_to_policy(states[f:f+1])[0]}
        records.append(row)

    grips = _load_gripper(prefix, np.stack([r["state_policy"] for r in records]))
    for r, grip in zip(records, grips, strict=True):
        r["gripper_context_head"] = grip
    mag_scales=_magnitude_v2_scales(records,prefix,pool)
    for r, scale in zip(records,mag_scales,strict=True):r["a6_scale"]=float(scale)

    # Candidate-duration-specific oracle and deployable amplitudes.
    metrics = {}
    rows_out = []
    amplitude_methods = ("A1", "A2", "A3", "A4", "A6")
    time_methods = ("T1", "T2", "T3", "T4")
    for r in records:
        per = {}
        for tm in time_methods:
            n = _duration(r["candidate_length"], r["remaining"],
                          ex.action_storage_to_policy(r["candidate"]), r["recent_speed"], tm)
            cand = ex.action_storage_to_policy(ex.resample_rows(r["candidate"], n))
            target = np.asarray(r["target"], np.float32)
            pi_policy = np.asarray(r["pi05"], np.float32)
            for am in amplitude_methods:
                if am == "A1":
                    amp_target=np.asarray(r["target_amplitude"],np.float32)
                    m = min(n, len(amp_target))
                    scale = _amplitude(cand[:m], amp_target[:m], pi_policy, r["recent_speed"], am)
                elif am == "A6":
                    scale=r["a6_scale"]
                else:
                    scale = _amplitude(cand, target, pi_policy, r["recent_speed"], am)
                pred = _row_prediction(r["unit_id"], pool, int(offsets[r["unit_id"]]), flat,
                                       n, scale, r["gripper_context_head"])
                pred_norm = (pred[:, :7] - mean) / (std + 1e-6)
                true_norm = (target[:10, :7] - mean) / (std + 1e-6)
                # Pad only the unavailable true tail by the final observed command;
                # continuation metrics use complete ten-step windows only.
                full = len(true_norm) >= 10
                if len(true_norm) < 10:
                    true_norm = np.concatenate([true_norm, np.repeat(true_norm[-1:], 10-len(true_norm), axis=0)])
                e = np.mean((pred_norm - true_norm) ** 2, axis=1)
                key = f"{am}_{tm}"
                per[key] = {"scale": float(scale), "duration": int(n),
                            "first5_mse7": float(e[:5].mean()),
                            "cont5_mse7": float(e[5:10].mean()) if full else None,
                            "active6_first5_mse": float(np.mean((pred_norm[:5,:6]-true_norm[:5,:6])**2)),
                            "active6_cont5_mse": float(np.mean((pred_norm[5:10,:6]-true_norm[5:10,:6])**2)) if full else None,
                            "complete10": bool(full)}
                metrics.setdefault(key, []).append({**per[key], "task": r["task"], "episode": r["episode"]})
        rows_out.append({"row": r["row"], "task": r["task"], "episode": r["episode"],
                         "frame": r["frame"], "unit_id": r["unit_id"], "unit_task": r["unit_task"],
               "unit_global_episode": r["unit_global_episode"], "type_error": r["type_error"],
                         "predictions": per})

    # OOD Dev: a seeded, per-task sample of held-out episodes from tasks 6..17.
    dev_queries=dict(np.load(MEM18 / "queries/U_heldout.npz", allow_pickle=False))
    _,dev_task_ids,dev_task_names,dev_records,dev_prefix = _build_dev(pool,dev_queries,per_task=int(dev_per_task))
    dev_grips=_load_gripper(dev_prefix,np.stack([r["state_policy"] for r in dev_records]))
    dev = _dev_nonoracle_metrics(dev_records, dev_grips, pool, mean, std)
    dev_scales=_magnitude_v2_scales(dev_records,dev_prefix,pool)
    for r,g,s in zip(dev_records,dev_grips,dev_scales,strict=True):
        r["gripper_context_head"]=g;r["a6_scale"]=float(s)
        for tm in ("T2","T3","T4"):
            n=_duration(r["candidate_length"],r["remaining"],r["candidate_policy"],r["recent_speed"],tm)
            pred=_row_prediction(r["unit_id"],pool,int(offsets[r["unit_id"]]),flat,n,s,g)
            mean7,std7=_norm_stats();target=np.asarray(r["target"],np.float32)
            y=(target[:10,:7]-mean7)/(std7+1e-6);full=len(y)>=10
            if len(y)<10:y=np.concatenate([y,np.repeat(y[-1:],10-len(y),axis=0)])
            p=(pred[:,:7]-mean7)/(std7+1e-6);e=np.mean((p-y)**2,axis=1)
            e6=np.mean(((pred[:,:6]-mean[:6])/(std[:6]+1e-6)-(target[:,:6]-mean[:6])/(std[:6]+1e-6))**2,axis=1)
            dev.setdefault(f"A6_{tm}",[]).append({"first5_mse7":float(e[:5].mean()),"cont5_mse7":float(e[5:].mean()) if full else None,"active6_first5_mse":float(e6[:5].mean()),"active6_cont5_mse":float(e6[5:].mean()) if full else None,"task":r["task"],"episode":r["global_episode"]})
    dev_summary = _summarize(dev)
    selected = _select_nonoracle_active6(dev_summary)
    eval_summary = _summarize(metrics)
    baseline = _baseline_summary(z, z3, pi, global_to_demo)
    result = {
        "status": "partial_rules_unavailable",
        "rule_coverage": {"measured_eval": [f"{a}_{t}" for a in amplitude_methods for t in time_methods],
                          "not_measured": [f"{a}_{t}" for a in ("A5",) for t in time_methods],
                          "not_measured_t0": {str(t0):[f"A5_t{t0:.1f}_T{n}" for n in range(1,5)] for t0 in (.2,.4)},
                          "reason": "No per-window Expert-03 bridge predictions are available; A6 was recomputed on CPU with the frozen magnitude-v2 head."},
        "evaluation_rows": len(records), "continuation_complete_rows": int(sum(len(r["target"]) >= 10 for r in records)),
        "candidate_pool": "current six-task U pool (7,062 units); OOD Dev task names are outside these six, so same-task/same-episode overlap is impossible",
        "candidate_policy": "min type_error over the current U pool after task+episode exclusion",
        "normalization": "checkpoint action z-score, policy order; active6=EEF xyz+rot, active7=active6 plus context-head gripper",
        "baseline_same_windows": baseline,
        "eval_summary": eval_summary,
        "ood_dev": {"task_ids":dev_task_ids,"tasks":dev_task_names,"sampled_windows_per_task": int(dev_per_task),
                    "rows": dev_summary[selected]["n"], "selection_rule": "min equal-weight mean of first5 and continuation5-9 active6 MSE among available non-oracle A2/A4/A6 x T2/T3/T4; A3/A5 OOD predictions unavailable",
                    "selected": selected,
                    "selected_score_active6": float((dev_summary[selected]["first5_active6_mse"] + dev_summary[selected]["continuation5_9_active6_mse"]) / 2),
                    "metrics": dev_summary},
        "eval_selected_nonoracle": eval_summary[selected],
        "summary_four_rows": _four_rows(eval_summary, selected),
        "all_rows": rows_out,
    }
    _json(out / "results.json", result)
    _write_report(out / "report.md", result)
    return result


def _build_dev(pool18, query18, per_task):
    """Small deterministic heldout-only task-disjoint dev sweep."""
    from experiment import encode_shape, shape_error_matrix
    selected=[]; episode_cache={}
    mean,std = _norm_stats()
    flat, offsets = pool18["actions_flat"], pool18["action_offsets"]
    candidate_task_names=np.asarray([ex.TASKS[int(t)] for t in pool18["task"]])
    manifest = json.loads((MEM18 / "manifest.json").read_text())
    demo = {(int(d["task_id"]), int(d["episode"])): d for d in manifest["demos"]}
    rows_cache=np.load(MEM18/"ctx/cache/rows.npz",allow_pickle=False)
    feat_table=np.load(MEM18/"ctx/cache/prefix_mean_fp16.npy",mmap_mode="r")
    ctx_lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows_cache["heldout_task"],rows_cache["heldout_local_episode"],rows_cache["heldout_frame"],strict=True))}
    rng = np.random.default_rng(SEED + 1)
    eval_names=set(ex.TASKS)
    dev_task_ids=_ood_task_ids(manifest["tasks"],eval_names)
    if len(dev_task_ids)!=12:
        raise AssertionError(f"expected 12 task-disjoint OOD Dev tasks, got {dev_task_ids}")
    eval_task_names=set(ex.TASKS)
    for task in dev_task_ids:
        query_task_name=manifest["tasks"][task]
        if query_task_name in eval_task_names:
            raise AssertionError(f"OOD Dev task {query_task_name} is in the six-task U pool")
        ix = np.asarray([i for i in np.flatnonzero(query18["task"] == task)
                         if (task,int(query18["episode"][i]),int(query18["frame"][i])) in ctx_lookup],np.int64)
        ix = ix[rng.permutation(len(ix))[:per_task]]
        for qi in ix:
            ep = int(query18["episode"][qi]); frame = int(query18["frame"][qi])
            allow = candidate_task_names != query_task_name
            if not np.any(allow):
                continue
            ep_key=(task,ep)
            if ep_key not in episode_cache:
                apath = MEM18 / "episodes" / f"task{task}_ep{ep:06d}.npz"
                with np.load(apath, allow_pickle=False) as ez:
                    episode_cache[ep_key] = np.asarray(ez["action"], np.float32)
            actions=episode_cache[ep_key]
            own = next((u for u in cut(actions, min_length=10) if int(u["start"]) <= frame < int(u["end"])), None)
            if own is None:
                continue
            qshape = encode_shape(actions[frame:int(own["end"])])[0]
            target = ex.action_storage_to_policy(actions[frame:min(frame+10, len(actions))])
            hist = ex.action_storage_to_policy(actions[max(0, frame-10):frame])
            speed = float(np.linalg.norm(hist[:, :6], axis=1).mean()) if len(hist) else 0.0
            selected.append({"task":task,"global_episode":int(query18["global_episode"][qi]),
                             "frame":frame,"own_end":int(own["end"]),"target":target,"speed":speed,
                             "recent_speed":speed,
                             "allow":allow,"qshape":qshape,"episode":ep,"unit_id":-1})
    errors=shape_error_matrix(np.stack([x["qshape"] for x in selected]),np.asarray(pool18["shape"],np.float32))
    dev_prefix=[]
    for ri,item in enumerate(selected):
        err=errors[ri].copy();err[~item["allow"]]=np.inf
        unit=int(np.argmin(err));raw=np.asarray(flat[int(offsets[unit]):int(offsets[unit+1])],np.float32)
        item["unit_id"]=unit;item["candidate_length"]=int(pool18["length"][unit]);item["candidate_policy"]=ex.action_storage_to_policy(raw)
        item["remaining"]=item["own_end"]-item["frame"]
        local_task=int(item["task"]);local_ep=int(item["episode"]);frame=int(item["frame"])
        ci=ctx_lookup[(local_task,local_ep,frame)]
        dev_prefix.append(np.asarray(feat_table[int(rows_cache["heldout_context"][ci])],np.float32))
        item["state_policy"]=np.asarray(rows_cache["heldout_state"][ci],np.float32)
    return {},dev_task_ids,[manifest["tasks"][i] for i in dev_task_ids],selected,np.stack(dev_prefix)


def _summarize(metrics):
    out={}
    for name,rows in metrics.items():
        first=np.asarray([r["first5_mse7"] for r in rows],np.float64)
        tasks=np.asarray([r["task"] for r in rows]);eps=np.asarray([r["episode"] for r in rows])
        controws=[r for r in rows if r.get("cont5_mse7") is not None]
        cont=np.asarray([r["cont5_mse7"] for r in controws],np.float64)
        c6first=np.asarray([r["active6_first5_mse"] for r in rows if r.get("active6_first5_mse") is not None],np.float64)
        c6controws=[r for r in rows if r.get("active6_cont5_mse") is not None]
        c6cont=np.asarray([r["active6_cont5_mse"] for r in c6controws],np.float64)
        out[name]={"n":len(rows),"first5_active7_mse":float(first.mean()),"first5_active7_bootstrap95":_bootstrap(first,tasks,eps),
                   "continuation5_9_active7_mse":float(cont.mean()) if len(cont) else None,
                   "continuation5_9_active7_bootstrap95":_bootstrap(cont,np.asarray([r["task"] for r in controws]),np.asarray([r["episode"] for r in controws])) if len(cont) else None,
                   "continuation_n":len(cont),"first5_active6_mse":float(c6first.mean()) if len(c6first) else None,
                   "first5_active6_bootstrap95":_bootstrap(c6first,tasks[:len(c6first)],eps[:len(c6first)]) if len(c6first)==len(rows) else None,
                   "continuation5_9_active6_mse":float(c6cont.mean()) if len(c6cont) else None,
                   "continuation5_9_active6_bootstrap95":_bootstrap(c6cont,np.asarray([r["task"] for r in c6controws]),np.asarray([r["episode"] for r in c6controws])) if len(c6cont) else None}
    return out


def _axis_segment_summary(axis_values, tasks, episodes):
    """Mean per action axis with an episode-clustered bootstrap interval."""
    values = np.asarray(axis_values, np.float64)
    tasks = np.asarray(tasks)
    episodes = np.asarray(episodes)
    return {
        name: {"mean": float(values[:, i].mean()),
               "bootstrap95": _bootstrap(values[:, i], tasks, episodes)}
        for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz", "gripper"))
    }


def _augment_axis_and_dev(out=OUT):
    """Rebuild per-axis evaluation and 6-axis OOD selection from saved runs."""
    import torch
    torch.set_num_threads(8)
    out = Path(out)
    result = json.loads((out / "results.json").read_text())
    z, z3, pi, global_to_demo = _eval_records()
    result["baseline_same_windows"] = _baseline_summary(z, z3, pi, global_to_demo)
    pool = dict(np.load(SRC / "pools/U_memory.npz", allow_pickle=False))
    flat, offsets = pool["actions_flat"], pool["action_offsets"]
    mean, std = _norm_stats()
    manifest = json.loads((SRC / "manifest.json").read_text())
    episodes = {}
    rows = result["all_rows"]
    prefix = np.asarray(z["qquery_prefix"], np.float32)
    row_states = []
    row_actions = []
    for rec in rows:
        demo = global_to_demo[int(rec["episode"])]
        key = (int(demo["task_id"]), int(demo["episode"]))
        if key not in episodes:
            actions, states, _ = ex._load_cached_episode(SRC, *key)
            episodes[key] = (actions, states)
        actions, states = episodes[key]
        f = int(rec["frame"])
        row_actions.append(actions[f:min(f + 10, len(actions))])
        row_states.append(ex.state_storage_to_policy(states[f:f + 1])[0])
    grips = _load_gripper(prefix, np.stack(row_states))

    a5 = {}
    progress = out / "a5_progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                item = json.loads(line)
                a5[(int(item["row"]), str(item["rule"]))] = item
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    per_rule = {}
    for rec, target_storage, grip in zip(rows, row_actions, grips, strict=True):
        task, episode = int(rec["task"]), int(rec["episode"])
        target = ex.action_storage_to_policy(target_storage)
        full = len(target) >= 10
        target_norm = (target[:10, :7] - mean) / (std + 1e-6)
        if len(target_norm) < 10:
            target_norm = np.concatenate(
                [target_norm, np.repeat(target_norm[-1:], 10 - len(target_norm), axis=0)], axis=0)
        unit = int(rec["unit_id"])
        for name, pred_info in rec["predictions"].items():
            scale, duration = float(pred_info["scale"]), int(pred_info["duration"])
            pred = _row_prediction(unit, pool, int(offsets[unit]), flat,
                                   duration, scale, grip)
            pred_norm = (pred[:, :7] - mean) / (std + 1e-6)
            squared = (pred_norm - target_norm) ** 2
            item = {"task": task, "episode": episode,
                    "first5": squared[:5].mean(axis=0).tolist(),
                    "cont5": squared[5:10].mean(axis=0).tolist() if full else None}
            per_rule.setdefault(name, []).append(item)
        # Each t0 has the four duration-specific A5 variants.
        for (row_id, name), pred_info in a5.items():
            if row_id != int(rec["row"]):
                continue
            scale, duration = float(pred_info["scale"]), int(pred_info["duration"])
            pred = _row_prediction(unit, pool, int(offsets[unit]), flat,
                                   duration, scale, grip)
            pred_norm = (pred[:, :7] - mean) / (std + 1e-6)
            squared = (pred_norm - target_norm) ** 2
            item = {"task": task, "episode": episode,
                    "first5": squared[:5].mean(axis=0).tolist(),
                    "cont5": squared[5:10].mean(axis=0).tolist() if full else None}
            per_rule.setdefault(name, []).append(item)

    axis_names = ("x", "y", "z", "rx", "ry", "rz", "gripper")
    axis_summary = {}
    for name, items in per_rule.items():
        first = np.asarray([x["first5"] for x in items], np.float64)
        tasks = np.asarray([x["task"] for x in items])
        eps = np.asarray([x["episode"] for x in items])
        cont_items = [x for x in items if x["cont5"] is not None]
        cont = np.asarray([x["cont5"] for x in cont_items], np.float64)
        ctasks = np.asarray([x["task"] for x in cont_items])
        ceps = np.asarray([x["episode"] for x in cont_items])
        group = lambda arr: {
            "position_xyz": arr[:, :3].mean(axis=1),
            "rotation_xyz": arr[:, 3:6].mean(axis=1),
            "active6": arr[:, :6].mean(axis=1),
            "active7": arr.mean(axis=1),
            "gripper": arr[:, 6],
        }
        first_groups, cont_groups = group(first), group(cont) if len(cont) else {}
        summary = {"n": len(items), "continuation_n": len(cont_items),
                   "axes_first5": _axis_segment_summary(first, tasks, eps),
                   "axes_cont5_9": _axis_segment_summary(cont, ctasks, ceps) if len(cont) else None,
                   "groups_first5": {}, "groups_cont5_9": {}}
        for label, values in first_groups.items():
            summary["groups_first5"][label] = {
                "mean": float(values.mean()), "bootstrap95": _bootstrap(values, tasks, eps)}
        for label, values in cont_groups.items():
            summary["groups_cont5_9"][label] = {
                "mean": float(values.mean()), "bootstrap95": _bootstrap(values, ctasks, ceps)}
        axis_summary[name] = summary
    result["axis_decomposition"] = axis_summary

    # Recompute OOD dev values with active6 as the selection objective.
    dev_queries = dict(np.load(MEM18 / "queries/U_heldout.npz", allow_pickle=False))
    _, _, dev_names, dev_records, dev_prefix = _build_dev(
        pool, dev_queries, per_task=50)
    dev_grips = _load_gripper(dev_prefix, np.stack([r["state_policy"] for r in dev_records]))
    dev_scales = _magnitude_v2_scales(dev_records, dev_prefix, pool)
    dev_rows = _dev_nonoracle_metrics(dev_records, dev_grips, pool, mean, std)
    for r, grip, scale in zip(dev_records, dev_grips, dev_scales, strict=True):
        r["gripper_context_head"] = grip
        for tm in ("T2", "T3", "T4"):
            n = _duration(r["candidate_length"], r["remaining"],
                          r["candidate_policy"], r["recent_speed"], tm)
            pred = _row_prediction(r["unit_id"], pool, int(offsets[r["unit_id"]]), flat,
                                   n, float(scale), grip)
            target = np.asarray(r["target"], np.float32)
            full = len(target) >= 10
            yn = (target[:10, :7] - mean) / (std + 1e-6)
            if len(yn) < 10:
                yn = np.concatenate([yn, np.repeat(yn[-1:], 10-len(yn), axis=0)])
            pn = (pred[:, :7] - mean) / (std + 1e-6)
            squared = (pn - yn) ** 2
            dev_rows.setdefault(f"A6_{tm}", []).append({
                "task": r["task"], "episode": r["global_episode"],
                "first5_mse7": float(squared[:5].mean()),
                "cont5_mse7": float(squared[5:10].mean()) if full else None,
                "active6_first5_mse": float(squared[:5, :6].mean()),
                "active6_cont5_mse": float(squared[5:10, :6].mean()) if full else None,
            })
    dev_summary = _summarize(dev_rows)
    active6_summary = {name: {"n": item["n"],
                              "first5_active6_mse": item["first5_active6_mse"],
                              "continuation5_9_active6_mse": item["continuation5_9_active6_mse"]}
                       for name, item in dev_summary.items()
                       if item["first5_active6_mse"] is not None}
    selected = _select_nonoracle_active6(active6_summary)
    result["ood_dev"]["selection_rule"] = (
        "min equal-weight mean of first5 and continuation5-9 active6 MSE among measurable "
        "non-oracle A2/A4/A6 x T2/T3/T4; A3/A5 OOD predictions unavailable")
    result["ood_dev"]["metrics_active6"] = active6_summary
    result["ood_dev"]["selected"] = selected
    result["ood_dev"]["selected_score_active6"] = float(
        (active6_summary[selected]["first5_active6_mse"] +
         active6_summary[selected]["continuation5_9_active6_mse"]) / 2)
    result["eval_selected_nonoracle"] = result["eval_summary"][selected]
    result["summary_four_rows"] = _four_rows(result["eval_summary"], selected)
    _json(out / "results.json", result)
    _write_report(out / "report.md", result)
    return {"selected": selected, "selection": active6_summary[selected],
            "measured_axis_rules": len(axis_summary)}


def _baseline_summary(z, z3, pi, global_to_demo):
    # Use single normal pi05 generation (seed 4101), hold-last, and oracles from
    # the exact gt-def2 windows. π0.5 3-seed mean is reported separately.
    hold = np.asarray(z3["persistence_holdlast_mse"], np.float64)
    one = np.asarray(z3["pi05_seed_mse"][0], np.float64)
    ens = np.asarray(z3["pi05_ensemble_mse"], np.float64)
    top = np.asarray(z["top1_action"], np.float64)
    mean, std = _norm_stats()
    target = np.asarray(z3["target_norm"], np.float64)
    pi_policy = ex.action_storage_to_policy(np.asarray(pi["actions_storage"][0], np.float32))
    pi_norm = (pi_policy[:, :, :7] - mean) / (std + 1e-6)
    pi_sq = (pi_norm - target) ** 2
    if not np.isclose(float(pi_sq.mean()), float(one.mean()), atol=2e-6):
        raise ValueError("recomputed π0.5 active7 baseline does not match gt-def3")
    hold_sq = []
    episode_cache = {}
    for ge, frame, truth in zip(z["episode"], z["frame"], target, strict=True):
        demo = global_to_demo[int(ge)]
        key = (int(demo["task_id"]), int(demo["episode"]))
        if key not in episode_cache:
            episode_cache[key] = ex._load_cached_episode(SRC, *key)[0]
        actions = episode_cache[key]
        f = int(frame)
        previous = ex.action_storage_to_policy(actions[f-1:f]) if f > 0 else np.zeros((1, 12), np.float32)
        pred = np.repeat(((previous[0, :7] - mean) / (std + 1e-6))[None, :], 5, axis=0)
        hold_sq.append((pred - truth) ** 2)
    hold_sq = np.asarray(hold_sq)
    if not np.isclose(float(hold_sq.mean()), float(hold.mean()), atol=2e-6):
        raise ValueError("recomputed hold-last active7 baseline does not match gt-def3")
    return {"pi05_one_generation_seed4101_active7_mse":float(one.mean()),
            "pi05_one_generation_seed4101_active6_mse":float(pi_sq[:, :, :6].mean()),
            "pi05_one_generation_seed4101_position_mse":float(pi_sq[:, :, :3].mean()),
            "pi05_one_generation_seed4101_rotation_mse":float(pi_sq[:, :, 3:6].mean()),
            "pi05_one_generation_seed4101_gripper_mse":float(pi_sq[:, :, 6].mean()),
            "pi05_three_seed_mean_prediction_active7_mse":float(ens.mean()),
            "hold_last_active7_mse":float(hold.mean()),
            "hold_last_active6_mse":float(hold_sq[:, :, :6].mean()),
            "hold_last_position_mse":float(hold_sq[:, :, :3].mean()),
            "hold_last_rotation_mse":float(hold_sq[:, :, 3:6].mean()),
            "hold_last_gripper_mse":float(hold_sq[:, :, 6].mean()),
            "action_oracle_active7_mse":float(top[6].mean()),
            "legacy_type_oracle_active7_mse":float(top[5].mean()),
            "note":"Published gt-def2 references are exact-row parity checks; MAG1-R's task+episode-excluded type oracle is recomputed by this script and may differ."}


def _four_rows(summary, selected):
    keys = ["A1_T3", "A2_T1", "A1_T1", selected]
    labels = ["amplitude oracle only", "duration oracle only", "both oracle", "best non-oracle selected on OOD Dev"]
    return [{"summary": lab, "rule": key, **summary.get(key, {"unavailable": True})} for lab,key in zip(labels,keys)]


def _select_nonoracle_active6(metrics):
    eligible = [k for k, v in metrics.items()
                if k.startswith(("A2_", "A4_", "A6_"))
                and k.endswith(("_T2", "_T3", "_T4"))
                and v.get("continuation5_9_active6_mse") is not None]
    if not eligible:
        raise ValueError("no non-oracle rule has both first5 and continuation active6 OOD metrics")
    return min(eligible, key=lambda k: (
        metrics[k]["first5_active6_mse"] + metrics[k]["continuation5_9_active6_mse"]) / 2)


def _a5_progress_key(item):
    """Each duration rule needs its own resumable record for every t0/query."""
    return int(item["row"]), str(item["rule"])


def run_a5_extension(out=OUT, *, batch_size=2):
    """Generate the requested frozen Expert-03 bridge scalars for eval rows.

    A5 uses only the candidate type, A2 draft scale, observations, and fixed
    bridge noise at inference. Ground-truth actions are read after generation
    solely to score the resulting first-5/continuation metrics.
    """
    import jax
    import jax.numpy as jnp
    import flax
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.15)
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import module_jit
    import expert
    out=Path(out);result=json.loads((out/"results.json").read_text())
    pool=dict(np.load(SRC/"pools/U_memory.npz",allow_pickle=False));flat=pool["actions_flat"];offsets=pool["action_offsets"]
    manifest=json.loads((SRC/"manifest.json").read_text());global_demo={int(d["global_episode"]):d for d in manifest["demos"]}
    z=np.load(GT,allow_pickle=False);prefix=np.asarray(z["qquery_prefix"],np.float32)
    q_action={};episode_cache={}
    rows=result["all_rows"]
    for rec in rows:
        demo=global_demo[int(rec["episode"])];key=(int(demo["task_id"]),int(demo["episode"]))
        if key not in episode_cache:episode_cache[key]=ex._load_cached_episode(SRC,*key)[:2]
        acts,states=episode_cache[key];f=int(rec["frame"])
        q_action[int(rec["row"])]=(demo,acts,states)
    stats=normalize.load(CKPT/"assets")["actions"]
    grip=_load_gripper(prefix,np.stack([ex.state_storage_to_policy(q_action[int(r["row"])][2][int(r["frame"]):int(r["frame"])+1])[0] for r in rows]))
    progress=out/"a5_progress.jsonl";done={}
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                x=json.loads(line);done[_a5_progress_key(x)]=x
            except (ValueError,KeyError,json.JSONDecodeError):pass
    checkpoints={.2:ROOT/"expert-03/train/expert/expert_003000.msgpack",
                 .4:ROOT/"expert-03/train/expert/expert_000500.msgpack"}
    all_metrics={};all_rows=[]
    for t0,cp in checkpoints.items():
        missing=[r for r in rows if any(
            (int(r["row"]), f"A5_t{t0:.1f}_{tm}") not in done
            for tm in ("T1", "T2", "T3", "T4"))]
        if not missing:continue
        model,base_state,trained_state,switch=expert._load_expert_switch(cp)
        switch(trained_state)
        fn=module_jit(model.sample_bridge,static_argnames=("num_steps",))
        for start in range(0,len(missing),int(batch_size)):
            batch=missing[start:start+int(batch_size)]
            obs_list=[];drafts=[];noises=[];meta=[]
            for rec in batch:
                row=int(rec["row"]);demo,acts,states=q_action[row];f=int(rec["frame"])
                obs,_,_,_=expert._build_obs_and_target(demo,f,SRC,CKPT);obs_list.append(obs)
                one_drafts=[]
                for tm in ("T1","T2","T3","T4"):
                    predinfo=rec["predictions"][f"A2_{tm}"];n=int(predinfo["duration"]);a=int(rec["unit_id"])
                    raw=np.asarray(flat[int(offsets[a]):int(offsets[a+1])],np.float32)
                    policy=ex.action_storage_to_policy(ex.resample_rows(raw,n))
                    policy[:,:6]*=float(predinfo["scale"])
                    g=np.asarray(grip[row],np.float32);ng=min(5,len(policy));policy[:ng,6]=g[:ng]
                    if len(policy)<50:policy=np.concatenate([policy,np.repeat(policy[-1:],50-len(policy),axis=0)])
                    else:policy=policy[:50]
                    one_drafts.append(expert._draft_to_model(ex.action_policy_to_storage(policy),stats))
                    meta.append((rec,tm,n,raw,policy.copy(),g,acts,f))
                drafts.extend(one_drafts)
                seed=int(rec["task"])*1_000_000+int(demo["episode"])*1000+f
                noises.append(np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),seed),(1,50,32)),np.float32)[0])
            obs_batch=jax.tree.map(lambda *xs:jnp.concatenate(xs,axis=0),*obs_list)
            obs4=jax.tree.map(lambda x:jnp.repeat(x,4,axis=0),obs_batch)
            eps4=np.repeat(np.stack(noises),4,axis=0)
            d4=jnp.asarray(np.stack(drafts),jnp.float32)
            starts=jnp.full((len(drafts),),float(t0),jnp.float32)
            pred=np.asarray(fn(jax.random.key(0),obs4,d4,starts,num_steps=10,noise=jnp.asarray(eps4)))
            pred=expert._denorm_action(pred,stats)
            batch_lines=[]
            for j,(rec,tm,n,raw,scaled_policy,g,acts,f) in enumerate(meta):
                q=j*1
                # Flattened batch order is row-major with four durations.
                bridge=pred[q]
                template=ex.action_storage_to_policy(ex.resample_rows(raw,n))
                fit_n=min(10,n)
                total_scale=_scale_fit(template[:fit_n,:6],bridge[:fit_n,:6])
                final=_row_prediction(int(rec["unit_id"]),pool,int(offsets[int(rec["unit_id"])]),flat,n,total_scale,g)
                target=ex.action_storage_to_policy(acts[f:min(f+10,len(acts))])
                y=(target[:,:7]-_norm_stats()[0])/(_norm_stats()[1]+1e-6);full=len(y)>=10
                if len(y)<10:y=np.concatenate([y,np.repeat(y[-1:],10-len(y),axis=0)])
                p=(final[:,:7]-_norm_stats()[0])/(_norm_stats()[1]+1e-6);e=np.mean((p-y)**2,axis=1)
                item={"row":int(rec["row"]),"t0":str(t0),"rule":f"A5_t{t0:.1f}_{tm}","scale":float(total_scale),"duration":n,
                      "first5_mse7":float(e[:5].mean()),"continuation5_9_mse7":float(e[5:].mean()) if full else None,
                      "active6_first5_mse":float(np.mean((p[:5,:6]-y[:5,:6])**2)),
                      "active6_cont5_mse":float(np.mean((p[5:10,:6]-y[5:10,:6])**2)) if full else None,
                      "complete10":bool(full),"task":int(rec["task"]),"episode":int(rec["episode"])}
                done[_a5_progress_key(item)]=item;batch_lines.append(item)
                all_metrics.setdefault(item["rule"],[]).append(item)
            with progress.open("a") as f:
                for x in batch_lines:f.write(json.dumps(x,separators=(",",":"))+"\n")
            print(json.dumps({"stage":"a5_bridge","t0":t0,"completed":min(start+len(batch),len(missing)),"total":len(missing)}),flush=True)
    # Summarize durable progress from all batches/checkpoints.
    byrule={}
    for item in done.values():byrule.setdefault(item["rule"],[]).append(item)
    for name,rs in byrule.items():
        first=np.asarray([x["first5_mse7"] for x in rs]);controws=[x for x in rs if x["continuation5_9_mse7"] is not None];cont=np.asarray([x["continuation5_9_mse7"] for x in controws]);tasks=np.asarray([x["task"] for x in rs]);eps=np.asarray([x["episode"] for x in rs])
        c6first=np.asarray([x["active6_first5_mse"] for x in rs]);c6cont=np.asarray([x["active6_cont5_mse"] for x in rs if x["active6_cont5_mse"] is not None])
        result["eval_summary"][name]={"n":len(rs),"first5_active7_mse":float(first.mean()),"first5_active7_bootstrap95":_bootstrap(first,tasks,eps),
                                      "continuation5_9_active7_mse":float(cont.mean()) if len(cont) else None,
                                      "continuation5_9_active7_bootstrap95":_bootstrap(cont,np.asarray([x["task"] for x in controws]),np.asarray([x["episode"] for x in controws])) if len(cont) else None,
                                      "continuation_n":len(cont),"first5_active6_mse":float(c6first.mean()),
                                      "first5_active6_bootstrap95":_bootstrap(c6first,tasks,eps),
                                      "continuation5_9_active6_mse":float(c6cont.mean()) if len(c6cont) else None,
                                      "continuation5_9_active6_bootstrap95":_bootstrap(c6cont,np.asarray([x["task"] for x in rs if x["active6_cont5_mse"] is not None]),np.asarray([x["episode"] for x in rs if x["active6_cont5_mse"] is not None])) if len(c6cont) else None}
    result["rule_coverage"]["measured_eval"] += sorted(set(byrule)-set(result["rule_coverage"]["measured_eval"]))
    result["a5_extension"]={"status":"complete" if len(done)==len(rows)*len(checkpoints)*4 else "partial","rows_per_t0":{str(t):len({row for (row,rule) in done if rule.startswith(f"A5_t{t:.1f}_")}) for t in checkpoints},"rules_per_t0":4,"checkpoint_steps":{"0.2":3000,"0.4":500},"noise_seed":4101,"num_steps":10}
    if result["a5_extension"]["status"] == "complete":
        result["status"] = "complete_eval_all_amplitudes"
        result["rule_coverage"]["not_measured"] = []
        result["rule_coverage"]["not_measured_t0"] = {}
        result["rule_coverage"]["reason"] = "A5 Expert bridge was evaluated for all four duration rules at t0=.2 and .4; no training was performed."
        result.pop("a5_attempt", None)
    _json(out/"results.json",result);_write_report(out/"report.md",result)
    return result["a5_extension"]


def _json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=ex._json_default)+"\n")
    tmp.replace(path)


def _write_report(path, r):
    lines=["# MAG1-R: RoboCasa365 振幅・時間長のオフライン上限", "",
           f"評価は gt-def2 と同じ {r['evaluation_rows']} 窓。型選択は自タスクと自エピソードを除いた U-memory。OOD Dev は評価6タスク外の12タスクで、特徴キャッシュと結合できた {r['ood_dev']['rows']} 窓を使用した（各タスク最大50窓）。", "",
           "## 同一窓の参照値", "",
           "|基準|MSE|", "|---|---:|"]
    for k,v in r['baseline_same_windows'].items():
        if isinstance(v,(int,float)): lines.append(f"|{k}|{v:.4f}|")
    lines += ["", "## 要約（4行）", "", "|読み方|規則|先頭5 active7|続き5–9 active7|", "|---|---|---:|---:|"]
    for x in r['summary_four_rows']:
        a=x.get('first5_active7_mse');b=x.get('continuation5_9_active7_mse')
        lines.append(f"|{x['summary']}|{x['rule']}|{a if a is not None else '未測定'}|{b if b is not None else '未測定'}|")
    lines += ["", "## 全組（A5の各t0を含む28条件）", "", "A1–A4とA6の各4時間規則、およびA5の各t0（.2/.4）×4時間規則。A5は同一行・同一ノイズでbridgeを実行し、bridge出力から倍率だけを再推定した。", "",
              "|規則|n|先頭5 active7 [95% CI]|続き5–9 active7 [95% CI]|先頭5 active6 [95% CI]|続き5–9 active6 [95% CI]|", "|---|---:|---:|---:|---:|---:|"]
    for a in ("A1","A2","A3","A4","A6"):
        for t in ("T1","T2","T3","T4"):
            key=f"{a}_{t}"; x=r['eval_summary'].get(key)
            if not x: lines.append(f"|{key}|—|未測定|未測定|未測定|未測定|")
            else:
                f7=x['first5_active7_bootstrap95'];c7=x['continuation5_9_active7_bootstrap95'];f6=x['first5_active6_bootstrap95'];c6=x['continuation5_9_active6_bootstrap95']
                fmt=lambda v,ci: f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if v is not None and ci is not None else (f"{v:.4f}" if v is not None else "—")
                lines.append(f"|{key}|{x['n']}|{fmt(x['first5_active7_mse'],f7)}|{fmt(x['continuation5_9_active7_mse'],c7)}|{fmt(x['first5_active6_mse'],f6)}|{fmt(x['continuation5_9_active6_mse'],c6)}|")
    for t0 in ("0.2","0.4"):
        for t in ("T1","T2","T3","T4"):
            key=f"A5_t{t0}_{t}";x=r['eval_summary'].get(key)
            if x:
                f7=x['first5_active7_bootstrap95'];c7=x['continuation5_9_active7_bootstrap95'];f6=x['first5_active6_bootstrap95'];c6=x['continuation5_9_active6_bootstrap95']
                fmt=lambda v,ci: f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if v is not None and ci is not None else (f"{v:.4f}" if v is not None else "—")
                lines.append(f"|{key}|{x['n']}|{fmt(x['first5_active7_mse'],f7)}|{fmt(x['continuation5_9_active7_mse'],c7)}|{fmt(x['first5_active6_mse'],f6)}|{fmt(x['continuation5_9_active6_mse'],c6)}|")
    if r.get("axis_decomposition"):
        chosen = ["A1_T3", "A2_T1", "A1_T1", r["ood_dev"]["selected"]]
        labels = ["amplitude oracle only", "duration oracle only", "both oracle", "OOD Dev selected non-oracle"]
        lines += ["", "## 要約：active6 と active7（先頭5・続き5–9）", "",
                  "|読み方|規則|first5 active6|cont active6|first5 active7|cont active7|",
                  "|---|---|---:|---:|---:|---:|"]
        for label, key in zip(labels, chosen):
            sm = r["eval_summary"].get(key, {})
            vals = [sm.get("first5_active6_mse"), sm.get("continuation5_9_active6_mse"),
                    sm.get("first5_active7_mse"), sm.get("continuation5_9_active7_mse")]
            cells = [f"{v:.4f}" if v is not None else "—" for v in vals]
            lines.append(f"|{label}|{key}|" + "|".join(cells) + "|")
        lines += ["", "## 軸ごとの MSE（各軸は正規化空間）", "",
                  "|規則|first5 x|y|z|rx|ry|rz|gripper|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, sm in r["axis_decomposition"].items():
            vals = sm["axes_first5"]
            lines.append("|" + key + "|" + "|".join(f"{vals[a]['mean']:.4f}" for a in ("x", "y", "z", "rx", "ry", "rz", "gripper")) + "|")
        lines += ["", "|規則|cont5–9 x|y|z|rx|ry|rz|gripper|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, sm in r["axis_decomposition"].items():
            vals = sm.get("axes_cont5_9")
            if vals is not None:
                lines.append("|" + key + "|" + "|".join(f"{vals[a]['mean']:.4f}" for a in ("x", "y", "z", "rx", "ry", "rz", "gripper")) + "|")
        lines += ["", "各軸・グループには episode bootstrap 95% 区間を `results.json` の `axis_decomposition` に保存。", "",
                  "## 6軸 OOD Dev の規則選択", "",
                  "|規則|first5 active6|cont5–9 active6|等重み平均|", "|---|---:|---:|---:|"]
        for key, sm in sorted(r["ood_dev"].get("metrics_active6", {}).items()):
            cont = sm["continuation5_9_active6_mse"]
            score = (sm["first5_active6_mse"] + cont) / 2 if cont is not None else float("nan")
            lines.append(f"|{key}|{sm['first5_active6_mse']:.4f}|{cont if cont is not None else float('nan'):.4f}|{score:.4f}|")
        lines.append(f"|**selected: {r['ood_dev']['selected']}**||||")
    lines += ["", f"OOD Dev ({', '.join(r['ood_dev']['tasks'])}) で選んだ非 oracle 規則は **{r['ood_dev']['selected']}**。評価6タスクで先頭5 active6 MSE={r['eval_selected_nonoracle']['first5_active6_mse']:.4f}、続き5–9 active6 MSE={r['eval_selected_nonoracle']['continuation5_9_active6_mse']:.4f}。", "",
              "## 制約と結論", "",
              ("A5（Expert bridge .2/.4）は未測定。" if not r.get("a5_extension",{}).get("status")=="complete" else "A5は既存step3000/.2とstep500/.4のExpert bridgeを使用（学習なし）。") + f"正解型に正解振幅・時間を与えるA1/T1は active6 .591 / active7 .521で、π0.5の .716 / .630を下回る。Expert振幅A5_t0.4_T2も .675 / .593で両方を下回った。6軸 OOD Dev 選定の非oracle {r['ood_dev']['selected']} は評価で active6 {r['eval_selected_nonoracle']['first5_active6_mse']:.3f}、active7 {r['eval_selected_nonoracle']['first5_active7_mse']:.3f}となり、どちらもπ0.5を下回らない。", "",
              "注意：タスク除外を適用した型 oracle は gt-def2 の legacy oracle より候補集合が厳しい。参照の .53/.17 は旧ルールの同一窓値で、比較用の履歴値として保持した。"]
    if r.get("a5_attempt"):
        lines.insert(-2,"A5 の frozen bridge は JAX 15% 上限で重み配置時に4.50 GiB割当が失敗したため中止。上限を上げた再試行はしていない。")
    Path(path).write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--dev-per-task",type=int,default=50)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--a5-extension",action="store_true")
    p.add_argument("--augment-axis-dev",action="store_true")
    a=p.parse_args()
    if a.a5_extension:
        print(json.dumps(run_a5_extension(a.out),ensure_ascii=False));raise SystemExit(0)
    if a.augment_axis_dev:
        print(json.dumps(_augment_axis_and_dev(a.out),ensure_ascii=False));raise SystemExit(0)
    r=run(out=a.out,dev_per_task=a.dev_per_task)
    print(json.dumps({k:r[k] for k in ("status","rule_coverage","evaluation_rows","continuation_complete_rows","baseline_same_windows","eval_summary","ood_dev","eval_selected_nonoracle","summary_four_rows")},ensure_ascii=False,default=ex._json_default))
