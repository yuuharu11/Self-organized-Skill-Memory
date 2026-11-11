"""Cross-fitted RoboCasa P5 drafts and an action-expert bridge.

The implementation is intentionally offline.  All policy-facing actions and
states use the fork order; only the cached demonstrations use LeRobot order.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
LIBERO_MEMORY = HERE.parent / "unseen_libero"
if str(LIBERO_MEMORY) not in sys.path:
    sys.path.insert(0, str(LIBERO_MEMORY))
import experiment as ex

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
SOURCE = STATE / "outputs/robocasa-memory/memory-01/policy-order-v2"
CHECKPOINT = Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT", STATE / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"))
OUT = Path(os.environ.get("ROBOCASA_EXPERT_OUT", "/work/.skillnet-state/outputs/robocasa-memory/expert-01"))
SEED = 20261110
T0S = (.2, .4, .6)
HORIZON = 50


def _json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def _mem_val_episodes(manifest):
    """Use a fixed 20% of the 80 memory episodes for Expert checkpoint selection."""
    vals = []
    for task in manifest["task_splits"]:
        eps = np.asarray(task["memory_episodes"], np.int64)
        chosen = np.random.default_rng(SEED + int(task["task_id"])).permutation(eps)[:16]
        vals.extend((int(task["task_id"]), int(e)) for e in chosen)
    return sorted(vals)


def _query_unit_records(out, split="memory"):
    """Expand the existing U query cache into aligned P5 query records."""
    q = dict(np.load(Path(out) / f"queries/U_{split}.npz", allow_pickle=False))
    p = dict(np.load(Path(out) / "pools/U_memory.npz", allow_pickle=False))
    return q, p


def _fold_ids(q, val_pairs, folds=5):
    eps = sorted({(int(t), int(e)) for t, e in zip(q["task"], q["episode"]) if (int(t), int(e)) not in set(val_pairs)})
    rng = np.random.default_rng(SEED)
    groups = [[] for _ in range(folds)]
    # Shuffle independently within task, then distribute round-robin so each
    # fold contains held-out episodes from every task.
    for t in range(len(ex.TASKS)):
        te = [ep for tt, ep in eps if tt == t]
        te = list(np.asarray(te)[rng.permutation(len(te))])
        for j, ep in enumerate(te): groups[j % folds].append((t, int(ep)))
    return groups


def _fit_p5_fold(q, pool, held_pairs, *, output, seed, epochs=20, device="cuda", excluded_pairs=()):
    """Fit the same soft listwise P5 objective on four episode folds."""
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.15)
    torch.set_num_threads(8)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.15)
    from experiment import _type_error_file, _allowed, query_features, _teacher_tau, _load_make_net
    excluded = set(held_pairs) | set(excluded_pairs)
    held = np.asarray([(int(t), int(e)) in excluded for t, e in zip(q["task"], q["episode"])])
    ids = np.flatnonzero(~held)
    if len(ids) < 8:
        raise ValueError("P5 fold has too few fit queries")
    x, norm = query_features(q, ids)
    # The existing training-set type-error matrix is read-only; it is aligned
    # with all memory queries and candidate pool rows.
    cache = SOURCE / "cache/errors_U_memory.npy"
    err = np.load(cache, mmap_mode="r")
    allowed = _allowed(q["global_episode"][ids], pool["global_episode"])
    tau = _teacher_tau(err[ids], allowed)
    keys = np.asarray(pool["key"], np.float32)
    torch.manual_seed(int(seed))
    net = _load_make_net()("U", "P5", x.shape[1], keys.shape[1], d=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    kt = torch.as_tensor(keys, dtype=torch.float32, device=device)
    at = torch.as_tensor(allowed, dtype=torch.bool, device=device)
    et = torch.as_tensor(np.asarray(err[ids], np.float32), device=device)
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    curve = []
    # Internal split is episode-based and disjoint from the held fold.
    fit_eps = np.unique(q["global_episode"][ids])
    val_eps = set(np.random.default_rng(seed + 77).permutation(fit_eps)[:max(1, len(fit_eps)//8)].tolist())
    valmask = np.asarray([int(e) in val_eps for e in q["global_episode"][ids]])
    fit_local = np.flatnonzero(~valmask); val_local = np.flatnonzero(valmask)
    for ep in range(int(epochs)):
        net.train(); order = rng.permutation(fit_local); losses=[]
        for st in range(0, len(order), 256):
            bi = torch.as_tensor(order[st:st+256], dtype=torch.long, device=device)
            logits = net(xt[torch.as_tensor(ids, device=device)[bi]], kt)
            ok = at[bi]; er = et[bi]
            teacher = torch.softmax((-er / tau).masked_fill(~ok, -1e9), dim=-1)
            pred = logits.masked_fill(~ok, -1e9)
            loss = -(teacher * torch.log_softmax(pred, dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        net.eval()
        with torch.no_grad():
            vi = torch.as_tensor(val_local, dtype=torch.long, device=device)
            scores = net(xt[torch.as_tensor(ids, device=device)[vi]], kt).masked_fill(~at[vi], -1e9).cpu().numpy()
        local_pick = scores.argmax(1)
        valerr = np.asarray(err[ids[val_local]], np.float32)
        metric = float(valerr[np.arange(len(vi)), local_pick].mean())
        curve.append({"epoch": ep+1, "train_loss": float(np.mean(losses)), "val_top1_type_error": metric})
        if metric < best[0]:
            best = (metric, {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}, ep+1)
    result = {"state": best[1], "summary": {"seed": seed, "best_epoch": best[2], "val_top1_type_error": best[0], "curve": curve}, "norm": norm,
              "input_dim": int(x.shape[1]), "key_dim": int(keys.shape[1])}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp"); torch.save(result, tmp); tmp.replace(output)
    return result


def _load_net(checkpoint, pool, *, device="cpu"):
    import torch
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from experiment import _load_make_net
    net = _load_make_net()("U", "P5", ck["input_dim"], ck["key_dim"], d=256)
    net.load_state_dict(ck["state"]); net.to(device).eval()
    return net, ck


def _p5_select(net, ck, pool, hist, state, *, device="cpu"):
    import torch
    from experiment import _features
    q = {"hist": np.asarray(hist, np.float32)[None], "state": np.asarray(state, np.float32)[None]}
    x = _features(q, ck["norm"])
    with torch.no_grad():
        sc = net(torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(pool["key"], dtype=torch.float32, device=device))[0].cpu().numpy()
    # Current episode exclusion is applied by the caller before this function.
    return int(np.argmax(sc)), sc


def _chain(net, ck, pool, action_hist, state, *, episode=None, device="cpu", length=50, return_debug=False, historyless=False):
    """P5 -> one stored unit at a time, updating only from generated commands."""
    hist = np.asarray(action_hist, np.float32).copy(); chunks=[]; chosen=[]; valid=np.linalg.norm(hist,axis=1)>1e-8
    selections=[]
    models = list(net) if isinstance(net, (list, tuple)) and len(net) and isinstance(net[0], (list, tuple)) else [(net, ck)]
    while sum(len(x) for x in chunks) < length:
        query_hist = np.zeros_like(hist) if historyless else hist
        scores = np.mean([_p5_select(m,c,pool,query_hist,state,device=device)[1] for m,c in models],axis=0)
        if episode is not None: scores[np.asarray(pool["global_episode"]) == int(episode)] = -np.inf
        order=np.argsort(-scores,kind="stable")
        j=int(order[0]); selections.append({"selected":j,"top3":[(int(i),float(scores[i])) for i in order[:3]]})
        raw=np.asarray(pool["actions_flat"][pool["action_offsets"][j]:pool["action_offsets"][j+1]],np.float32).copy()
        if not historyless and np.any(valid) and np.linalg.norm(hist[valid,5:11])>1e-7:
            raw=ex._scale_to_ref(raw,hist[valid])
        emit=raw[:max(1,length-sum(len(x) for x in chunks))]
        chunks.append(emit);chosen.append(j)
        hist=np.concatenate([hist,emit],axis=0)[-10:];valid=np.concatenate([valid,np.ones(len(emit),bool)])[-10:]
    result=(np.concatenate(chunks,axis=0)[:length], chosen)
    return (*result,selections) if return_debug else result


def make_runtime_draft(history_policy_order, state_policy_order, pool, p5_models, *, episode=None, device="cpu", return_debug=False, historyless=False):
    """Serving adapter: executed fork-order history/state -> 50-step LeRobot-order draft.

    `p5_models` is a sequence of (network, checkpoint metadata), usually the
    three RM-B U seeds. This function performs no oracle selection.
    """
    hist=np.asarray(history_policy_order,np.float32)
    if hist.shape != (10,12): raise ValueError("history must be [10,12] fork action order")
    # Experiment records and candidate units are LeRobot order; adapt the live
    # fork history once before feeding RM-B's exact query feature transform.
    hist_store=ex.action_policy_to_storage(hist)
    state_store=ex.action_policy_to_storage(np.pad(np.asarray(state_storage_order,np.float32)[None,:],((0,0),(0,0)))) if False else None
    state=np.asarray(state_policy_order,np.float32)
    if state.shape!=(16,): raise ValueError("state must be [16] in fork state order")
    state_store=state_policy_to_storage(state)
    # Build with each P5 seed and select a unit per seed, then vote on the
    # ensemble score at every step. For runtime simplicity, use seed 0 by default.
    result=_chain(p5_models,None,pool,hist_store,state_store,episode=episode,device=device,length=HORIZON,return_debug=return_debug,historyless=historyless)
    if return_debug:
        draft, chosen, selections=result
        return draft, {"chosen_indices":chosen,"selections":selections}
    return result[0]


def state_policy_to_storage(state):
    """Fork state [EEF pose7, base pose7, grip2] -> dataset state order."""
    s=np.asarray(state,np.float32)
    if s.shape[-1]!=16: raise ValueError("state must end in 16 channels")
    return np.concatenate([s[...,7:14],s[...,:7],s[...,14:16]],axis=-1)


def serve_bridge_action(model, observation, history_policy_order, state_policy_order, pool, p5_models,
                        norm_stats, *, t0=.4, seed=0, num_steps=10, device="cpu",
                        compiled_bridge_fn=None, prefix_context=None, scale_reference=None,
                        gripper_reference=None, reference_action=None, replace_draft_channels=(),
                        return_debug=False, debug_scale_kind="positive-dot", historyless=False):
    """Drop-in serving helper: executed policy-order history -> P5 draft -> bridge action.

    Returns one 50x12 action chunk in fork order, unnormalized, as expected by
    the RoboCasa serving policy. `observation` is already the model's
    `Observation` (normalized state and tokenized prompt); `norm_stats` is the
    checkpoint action `NormStats`.
    """
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    draft_storage,draft_info=make_runtime_draft(history_policy_order,state_policy_order,pool,p5_models,device=device,return_debug=True,historyless=historyless)
    draft_policy=ex.action_storage_to_policy(draft_storage)
    original_draft_policy=draft_policy.copy()
    if replace_draft_channels:
        if reference_action is None:
            raise ValueError("reference_action required for draft channel overrides")
        reference_action=np.asarray(reference_action,np.float32)
        if reference_action.shape != draft_policy.shape:
            raise ValueError(f"reference action must be {draft_policy.shape}, got {reference_action.shape}")
        draft_policy[:,list(replace_draft_channels)]=reference_action[:,list(replace_draft_channels)]
    dim=int(model.action_dim)
    padded=np.zeros((1,HORIZON,dim),np.float32);padded[0,:,:12]=draft_policy
    mean=np.asarray(norm_stats.mean,np.float32);std=np.asarray(norm_stats.std,np.float32)
    padded=(padded-mean[:dim][None,None,:])/(std[:dim][None,None,:]+1e-6)
    # Callers serving repeated fixed-shape queries should construct this once
    # at server startup and pass it here. The fallback retains the historical
    # API for offline scripts and parity checks.
    fn=compiled_bridge_fn or module_jit(model.sample_bridge,static_argnames=("num_steps",))
    noise=jax.random.normal(jax.random.key(int(seed)),(1,HORIZON,dim))
    kwargs={"num_steps":int(num_steps),"noise":noise}
    if prefix_context is not None: kwargs["prefix_context"]=prefix_context
    if scale_reference is not None: kwargs["scale_reference"]=scale_reference
    pred=np.asarray(fn(jax.random.key(int(seed)+1),observation,jnp.asarray(padded),jnp.asarray([float(t0)],jnp.float32),**kwargs))[0]
    pred=pred*np.pad(std[:dim],(0,dim-len(std[:dim])),constant_values=1)[:dim]+np.pad(mean[:dim],(0,dim-len(mean[:dim])),constant_values=0)[:dim]
    pred=pred[:,:12].astype(np.float32)
    if gripper_reference is not None:
        # RoboCasa fork action order is [EEF xyz/rot, gripper, base4, mode].
        from memory_server_utils import replace_gripper_only
        grip=np.zeros_like(pred)
        grip[:,6:7]=np.asarray(gripper_reference,np.float32)
        pred=replace_gripper_only(pred,grip)
    if not return_debug:
        return pred
    # Summarize exact fit factors for transparent diagnostics. This mirrors
    # _ReferenceScaleBridge's positive least-squares rule in physical action
    # space; the bridge itself remains the source of the actual prediction.
    scale_pos=scale_rot=None
    if scale_reference is not None:
        ref=np.asarray(scale_reference,np.float32).reshape(-1,6)
        def positive_fit(x,y):
            return max(float(np.sum(x*y))/max(float(np.sum(x*x)),1e-12),0.)
        def norm_fit(x,y):
            return float(np.linalg.norm(y))/max(float(np.linalg.norm(x)),1e-8)
        fit=norm_fit if debug_scale_kind == "norm" else positive_fit
        scale_pos=fit(draft_policy[:,:3],ref[:,:3])
        scale_rot=fit(draft_policy[:,3:6],ref[:,3:6])
    def unit_record(i):
        i=int(i)
        return {"index":i,"task_id":int(pool["task"][i]),"episode":int(pool["episode"][i]),
                "start":int(pool["frame"][i]),"end":int(pool["frame"][i]+pool["length"][i])}
    selections=[]
    for s in draft_info["selections"]:
        selections.append({"selected":unit_record(s["selected"]),"top3":[{"unit":unit_record(i),"score":score} for i,score in s["top3"]]})
    info={"selections":selections,"draft_before_overrides_first5":original_draft_policy[:5].tolist(),
          "draft_first5":draft_policy[:5].tolist(),"scale_position":scale_pos,"scale_rotation":scale_rot}
    return pred,info


def _build_obs_and_target(info, start, out, checkpoint):
    """Read one camera observation, then apply the exact fork dataset transforms."""
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.robocasa_policy import RobocasaInputs
    from openpi.shared import normalize
    from openpi.training import config as train_config
    from experiment import action_storage_to_policy, state_storage_to_policy, _load_cached_episode
    root=Path(json.loads((Path(out)/"manifest.json").read_text())["task_splits"][int(info["task_id"])]["root"])
    ep=int(info["episode"]); cams=[]
    for camera in ("robot0_agentview_left","robot0_agentview_right","robot0_eye_in_hand"):
        p=root/"videos/chunk-000"/f"observation.images.{camera}"/f"episode_{ep:06d}.mp4"
        cams.append(_video_frame(str(p),int(start)))
    action,state,frames=_load_cached_episode(out,int(info["task_id"]),ep)
    raw={"observation/image":cams[0],"observation/right_image":cams[1],"observation/wrist_image":cams[2],
         "observation/state":state_storage_to_policy(state[start]),"prompt":info["prompt"],
         "actions":action_storage_to_policy(action[start:min(start+50,len(action))])}
    cfg=train_config.get_config("pi05_robocasa_multitask")
    cfg=__import__("dataclasses").replace(cfg,model=__import__("dataclasses").replace(cfg.model,discrete_state_input=True,action_horizon=50))
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    item=raw
    for tr in [*dc.data_transforms.inputs, transforms.Normalize(normalize.load(Path(checkpoint)/"assets"),use_quantiles=False), *dc.model_transforms.inputs]: item=tr(item)
    # model inputs are exactly Observation.from_dict fields; actions are normalized target.
    model_inputs={}
    for k,v in item.items():
        if k in ("actions","prompt"): continue
        if isinstance(v,dict): model_inputs[k]={kk:jnp.asarray(vv)[None] for kk,vv in v.items()}
        else: model_inputs[k]=jnp.asarray(v)[None]
    obs=model_api.Observation.from_dict(model_inputs)
    target=np.asarray(item["actions"],np.float32)
    if target.shape[0] < 50: target=np.pad(target,((0,50-target.shape[0]),(0,0)))
    return obs,target[:50],action[start:min(start+50,len(action))],state[start]


_VIDEO_READERS = OrderedDict()


def _video_reader(path):
    import av
    path = str(path)
    cached = _VIDEO_READERS.pop(path, None)
    if cached is not None:
        _VIDEO_READERS[path] = cached
        return cached
    container = av.open(path)
    # Thousands of random frame reads otherwise leave hundreds of decoders
    # open, each with an auto-sized worker pool. Bound both dimensions so a
    # long training run does not exhaust FFmpeg/scaler resources.
    for stream in container.streams.video:
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
    _VIDEO_READERS[path] = container
    while len(_VIDEO_READERS) > 48:
        _, old = _VIDEO_READERS.popitem(last=False)
        old.close()
    return container


def _video_frame(path, index):
    """Randomly read one frame without forking an ffmpeg subprocess per query."""
    container=_video_reader(str(path));stream=container.streams.video[0]
    fps=float(stream.average_rate or 20.0);target_seconds=float(index)/fps
    offset=int(target_seconds/float(stream.time_base))
    container.seek(offset,stream=stream,backward=True)
    last=None
    for frame in container.decode(stream):
        last=frame
        if frame.pts is None or frame.pts*float(stream.time_base)+1e-6>=target_seconds:
            return frame.to_ndarray(format="rgb24")
    if last is not None:return last.to_ndarray(format="rgb24")
    raise IndexError(f"frame {index} unavailable in {path}")


def _load_expert_model(checkpoint=CHECKPOINT, *, params_dir=None):
    import jax
    import flax.nnx as nnx
    from openpi.models.model import restore_params
    from openpi.models.pi0_config import Pi0Config
    # `core` is the already-reviewed bridge module in the shared worktree.
    core_dir=Path(__file__).resolve().parents[1]/"pi05_memory"
    if str(core_dir) not in sys.path: sys.path.insert(0,str(core_dir))
    import core
    model,cfg=core.make_model(action_horizon=50,discrete_state_input=True,robocasa=True)
    params=restore_params(params_dir or (Path(checkpoint)/"params"),restore_type=np.ndarray)
    state=nnx.state(model); expected=state.to_pure_dict()
    a,ta=jax.tree.flatten(expected);b,tb=jax.tree.flatten(params)
    if ta!=tb or any(x.shape!=y.shape for x,y in zip(a,b,strict=True)): raise ValueError("RoboCasa pi05 checkpoint tree mismatch")
    state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray,params));nnx.update(model,state)
    return model,cfg


def _bridge_step(filt,tx):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    @nnx.jit
    def step(model,opt,obs,target,draft,key):
        key,kt,kn=jax.random.split(key,3); b=target.shape[0]
        t0=jax.random.uniform(kt,(b,),minval=.2,maxval=.6)
        progress=jax.random.uniform(kn,(b,),minval=.01,maxval=1.)
        eps=jax.random.normal(jax.random.fold_in(key,1),target.shape)
        xt,t,vel=_bridge_training_sample(target,draft,eps,t0,progress)
        weight=jnp.asarray([1.,1.,1.,1.,1.,1.,1.,.1,.1,.1,.1,.1]+[0.]*20,jnp.float32)
        def loss_fn(m):
            pred=m.velocity(obs,xt,t)
            return jnp.sum((pred-vel)**2*weight[None,None,:])/(jnp.sum(weight)*target.shape[1]*b)
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(model)
        selected=nnx.state(model,filt); updates,opt=tx.update(grads,opt,selected)
        nnx.update(model,optax.apply_updates(selected,updates))
        return loss,opt
    return step


def _bridge_training_sample(target,draft,noise,t0,progress):
    """Sample the same bridge path and model-time used by sample_bridge."""
    import jax.numpy as jnp
    t0=jnp.asarray(t0).reshape(-1)
    progress=jnp.asarray(progress).reshape(-1)
    source=(1-t0[:,None,None])*draft+t0[:,None,None]*noise
    x=(1-progress[:,None,None])*target+progress[:,None,None]*source
    model_time=t0*progress
    velocity=(source-target)/t0[:,None,None]
    return x,model_time,velocity


def _prepare_data(out, *, smoke=False):
    """Prepare cross-fit P5 drafts and raw observation row inventory.

    P5 folds are fitted only on memory episodes outside the target episode and
    the fixed Expert-validation episode set. The cache is immutable once made.
    """
    import torch
    from experiment import _load_cached_episode
    out=Path(out);src=Path(SOURCE);m=json.loads((src/"manifest.json").read_text())
    cache_dir=out/"drafts";cache_dir.mkdir(parents=True,exist_ok=True)
    manifest={"source":str(src),"tasks":m["tasks"],"checkpoint":str(CHECKPOINT),"folds":5,"validation_memory_episodes":_mem_val_episodes(m)}
    _json(out/"manifest.json",manifest)
    q=dict(np.load(src/"queries/U_memory.npz",allow_pickle=False));pool=dict(np.load(src/"pools/U_memory.npz",allow_pickle=False))
    valpairs=_mem_val_episodes(m); fold_groups=_fold_ids(q,valpairs,folds=5)
    folds=np.full(len(q["task"]),-1,np.int8)
    for fi,g in enumerate(fold_groups):
        for i,(t,e) in enumerate(zip(q["task"],q["episode"])):
            if (int(t),int(e)) in set(g):folds[i]=fi
    if np.any(folds==-1):
        # -1 is reserved for the fixed Expert-validation episodes; they are
        # never used to fit a fold model and are scored with fold 0.
        missing=[(int(t),int(e)) for t,e,f in zip(q["task"],q["episode"],folds) if f==-1]
        if not set(missing).issubset(set(valpairs)): raise AssertionError("unassigned non-validation P5 episodes")
    # Train fold model files separately and reuse completed ones.
    for fi in np.unique(folds[folds>=0]):
        path=out/"p5_folds"/f"fold{int(fi)}.pt"
        if path.exists(): continue
        held=set((int(t),int(e)) for t,e in zip(q["task"][folds==fi],q["episode"][folds==fi]))
        _fit_p5_fold(q,pool,held,output=path,seed=SEED+int(fi),epochs=2 if smoke else 20,excluded_pairs=valpairs)
    # Generate 50-step draft at every non-overlapping memory-query start.
    records=[]; episode_infos=[x for x in m["demos"] if x["split"]=="memory"]
    valset=set(tuple(x) for x in valpairs)
    selected=np.arange(len(q["task"]))
    for info in episode_infos:
        t,e=int(info["task_id"]),int(info["episode"])
        epcache=cache_dir/f"drafts_task{t}_ep{e:06d}.npz"
        if epcache.exists():
            z=np.load(epcache,allow_pickle=False)
            for i in range(len(z["frame"])):
                records.append({"task":t,"episode":e,"frame":int(z["frame"][i]),"validation":bool(z["validation"][i]),
                    "draft":z["draft"][i],"target":z["target"][i],"hist":z["hist"][i],"state":z["state"][i]})
            continue
        act,st,fr=_load_cached_episode(src,t,e)
        if len(act)<50: continue
        isval=(t,e) in valset
        # Validation episodes use P5 ensemble checkpoints from the fitting
        # memory P5; they are excluded from Expert updates, not from P5 itself.
        qmatch=np.flatnonzero((q["task"]==t)&(q["episode"]==e))
        fi=int(folds[qmatch[0]]) if len(qmatch) and int(folds[qmatch[0]])>=0 else 0
        p5,ck=_load_net(out/"p5_folds"/f"fold{fi}.pt",pool,device="cuda" if torch.cuda.is_available() else "cpu")
        hist=np.zeros((10,12),np.float32)
        episode_rows=[]
        for start in range(0,len(act)-49,10):
            # Each offline query begins with the preceding executed demo
            # actions, then generated actions advance the autoregressive history.
            prev=act[max(0,start-10):start]
            hist=np.zeros((10,12),np.float32)
            if len(prev): hist[-len(prev):]=prev
            d,sel=_chain(p5,ck,pool,hist,st[start],episode=t*1000+e,device="cuda" if torch.cuda.is_available() else "cpu")
            rec={"task":t,"episode":e,"frame":start,"validation":bool(isval),"draft":d,"target":act[start:start+50].copy(),"hist":hist.copy(),"state":st[start].copy()}
            records.append(rec);episode_rows.append(rec)
        _npz(epcache,frame=np.asarray([r["frame"] for r in episode_rows],np.int32),
             validation=np.asarray([r["validation"] for r in episode_rows],bool),draft=np.stack([r["draft"] for r in episode_rows]),
             target=np.stack([r["target"] for r in episode_rows]),hist=np.stack([r["hist"] for r in episode_rows]),state=np.stack([r["state"] for r in episode_rows]))
    if not records: raise RuntimeError("no cross-fitted action windows prepared")
    final_npz=out/"drafts/train_dev.npz"
    if not final_npz.exists():
        np.savez_compressed(final_npz,task=np.asarray([r["task"] for r in records]),episode=np.asarray([r["episode"] for r in records]),frame=np.asarray([r["frame"] for r in records]),validation=np.asarray([r["validation"] for r in records]),draft=np.stack([r["draft"] for r in records]),target=np.stack([r["target"] for r in records]),hist=np.stack([r["hist"] for r in records]),state=np.stack([r["state"] for r in records]))
    _json(out/"prepare.json",{"rows":len(records),"memory_validation_episodes":len(valpairs),"crossfit_folds":5,"draft_rule":"P5-generated U units autoregressively chained; candidate from same episode masked; held-out split not used."})
    return records


def _npz(path, **a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**a)
    tmp.replace(path)


def train_expert(out=OUT, *, steps=4000, batch=4, threads=8):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex,module_jit
    import torch;torch.set_num_threads(threads)
    torch.cuda.set_per_process_memory_fraction(.35)
    out=Path(out); cache=np.load(out/"drafts/train_dev.npz",allow_pickle=False)
    from openpi.shared import normalize
    action_stats=normalize.load(CHECKPOINT/"assets")["actions"]
    draft_model=np.stack([_draft_to_model(d,action_stats) for d in cache["draft"]]).astype(np.float32)
    train_ids=np.flatnonzero(~cache["validation"]);dev_ids=np.flatnonzero(cache["validation"])
    if not len(dev_ids): raise ValueError("memory-side validation cache empty")
    model,_=_load_expert_model()
    filt=nnx.All(nnx.Param,PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    selected=nnx.state(model,filt)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5));opt=tx.init(selected)
    stepfn=_bridge_step(filt,tx)
    # Transform/cache observations on demand, one batch at a time.
    m=json.loads((SOURCE/"manifest.json").read_text()); info={(int(x["task_id"]),int(x["episode"])):x for x in m["demos"]}
    rng=np.random.default_rng(SEED); order_cache=np.stack([rng.choice(train_ids,batch,replace=True) for _ in range(steps)])
    expdir=out/"expert"; expdir.mkdir(parents=True,exist_ok=True); progress=expdir/"progress.json"
    best=float("inf"); best_t=.4; best_step=0; logs=[]; start_step=0
    if progress.exists():
        previous=json.loads(progress.read_text());latest=int(previous.get("latest_step",0));resume_path=expdir/"resume"/f"{latest:06d}"
        if latest and resume_path.exists():
            # Restore against the live Optax template so nested namedtuple
            # state (e.g. ScaleByAdamState) is not flattened to plain dicts.
            template={"model":nnx.state(model,filt).to_pure_dict(),"opt":opt}
            restored=ocp.PyTreeCheckpointer().restore(
                resume_path,args=ocp.args.PyTreeRestore(item=template))
            model_state=nnx.state(model,filt);model_state.replace_by_pure_dict(jax.tree.map(jnp.asarray,restored["model"]));nnx.update(model,model_state)
            opt=jax.tree.map(jnp.asarray,restored["opt"]);start_step=latest;logs=list(previous.get("logs",[]))
            old=previous.get("best",{});best=float(old.get("dev_first5_mse",float("inf")));best_t=float(old.get("t0",.4));best_step=int(old.get("step",0))
    from openpi.shared.nnx_utils import module_jit
    for step in range(start_step,steps):
        ids=order_cache[step]; observations=[]; targets=[]
        for ix in ids:
            obs,targ,_,_=_build_obs_and_target(info[(int(cache["task"][ix]),int(cache["episode"][ix]))],int(cache["frame"][ix]),SOURCE,CHECKPOINT)
            observations.append(obs);targets.append(targ)
        obs=jax.tree.map(lambda *xs:jnp.concatenate(xs,axis=0),*observations); target=jnp.asarray(np.stack(targets));draft=jnp.asarray(draft_model[ids])
        key=jax.random.fold_in(jax.random.key(SEED),step)
        loss,opt=stepfn(model,opt,obs,target,draft,key); value=float(loss)
        rec={"step":step+1,"train_loss":value}; logs.append(rec)
        if (step+1)%500==0 or step+1==steps:
            # Use a fixed memory-side validation episode subset; same P5 draft
            # generator, no held-out 120 demos enter this choice.
            # One fixed query at frame zero from every reserved memory-side
            # validation episode (rather than a few rows from one episode).
            devbatch=np.asarray([int(i) for i in dev_ids if int(cache["frame"][i])==0],np.int64)
            if len(devbatch)<len(ex.TASKS)*16:
                raise AssertionError(f"validation must include one start row for all 96 reserved episodes; got {len(devbatch)}")
            dmetrics={}
            # Save candidate params and select with the same bridge sampler.
            cp=expdir/f"expert_{step+1:06d}.msgpack"
            from flax import serialization
            if not cp.exists(): cp.write_bytes(serialization.to_bytes(nnx.state(model,filt).to_pure_dict()))
            fn=module_jit(model.sample_bridge,static_argnames=("num_steps",))
            for t0 in T0S:
                vals=[]
                # Keep checkpoint selection memory-bounded; doing the whole
                # 96-episode validation in batch 8 left no headroom for the
                # next training update on shared GPUs.
                for st in range(0,len(devbatch),1):
                    local=devbatch[st:st+1]; obss=[]; targets=[]; noises=[]
                    for ix in local:
                        obs0,target0,_,_=_build_obs_and_target(info[(int(cache["task"][ix]),int(cache["episode"][ix]))],int(cache["frame"][ix]),SOURCE,CHECKPOINT)
                        obss.append(obs0);targets.append(target0)
                        noises.append(np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(ix)),(1,50,32))))
                    obs_batch=jax.tree.map(lambda *xs:jnp.concatenate(xs,axis=0),*obss)
                    pred=np.asarray(fn(jax.random.key(0),obs_batch,jnp.asarray(draft_model[local]),jnp.full((len(local),),float(t0)),num_steps=10,noise=jnp.asarray(np.concatenate(noises,axis=0))))
                    vals.extend(np.mean((pred[:,:5,:12]-np.stack(targets)[:,:5,:12])**2,axis=(1,2)).tolist())
                dmetrics[str(t0)]=float(np.mean(vals))
            chosen=min(T0S,key=lambda t:dmetrics[str(t)]); metric=dmetrics[str(chosen)]
            rec.update({"dev_first5_by_t0":dmetrics,"dev_selected_t0":chosen,"dev_first5_mse":metric,"batch_size":batch})
            if metric<best: best=metric;best_t=chosen;best_step=step+1
            _json(progress,{"status":"running","latest_step":step+1,"batch_size":batch,"best":{"step":best_step,"t0":best_t,"dev_first5_mse":best},"logs":logs})
            resume_dir=expdir/"resume"/f"{step+1:06d}"
            if not resume_dir.exists():
                ocp.PyTreeCheckpointer().save(resume_dir,{"model":nnx.state(model,filt).to_pure_dict(),"opt":opt})
            print(json.dumps(rec),flush=True)
            # Drop the short-lived validation JIT wrapper and its device
            # buffers before returning to the large training graph. Keep the
            # training step's compiled executable cached.
            del fn, obs_batch, pred, obss, targets, noises, vals
            import gc; gc.collect()
    _json(expdir/"best.json",{"checkpoint":f"expert_{best_step:06d}.msgpack","steps":best_step,"t0":best_t,"dev_first5_mse":best,"selection_split":"memory-side fixed validation episodes only"})
    _json(expdir/"trained.json",{"steps":steps,"batch_size":batch,"best_step":best_step,"best_t0":best_t,"trainable_filter":"llm.*_1|action_in_proj|time_mlp|action_out_proj","optimizer":"AdamW 2.5e-5 clip1","loss":"12-D bridge velocity, EEF+gripper weight 1, base/mode .1; padding dims zero"})


def _overlay_expert(model, path):
    import flax.nnx as nnx
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex
    filt=nnx.All(nnx.Param,PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    selected=nnx.state(model,filt)
    loaded=serialization.from_bytes(selected.to_pure_dict(),Path(path).read_bytes())
    selected.replace_by_pure_dict(loaded);nnx.update(model,selected)
    return model


def _norm_action(a, stats):
    a=np.asarray(a,np.float32);dim=a.shape[-1]
    mean=np.asarray(stats.mean,np.float32)[:dim];std=np.asarray(stats.std,np.float32)[:dim]
    return (a-mean)/(std+1e-6)


def _denorm_action(a, stats):
    a=np.asarray(a,np.float32);dim=a.shape[-1]
    mean=np.asarray(stats.mean,np.float32)[:dim];std=np.asarray(stats.std,np.float32)[:dim]
    return a*(std+1e-6)+mean


def _draft_to_model(raw_storage, stats):
    """Convert a stored LeRobot draft [T,12] into normalized fork [T,32]."""
    raw=np.asarray(raw_storage,np.float32)
    policy=ex.action_storage_to_policy(raw)
    padded=np.zeros((len(policy),32),np.float32);padded[:,:12]=policy
    return _norm_action(padded,stats)


def _oracle_first_unit(pool, target10, hist, state, *, device="cpu"):
    qshape,_=ex.encode_shape(target10)
    err=ex.shape_error_matrix(qshape[None],pool["shape"])[0]
    err[np.asarray(pool["global_episode"])<0]=np.inf
    j=int(np.argmin(err)); raw=np.asarray(pool["actions_flat"][pool["action_offsets"][j]:pool["action_offsets"][j+1]],np.float32).copy()
    valid=np.linalg.norm(hist,axis=1)>1e-8
    if np.any(valid) and np.linalg.norm(hist[valid,5:11])>1e-7:raw=ex._scale_to_ref(raw,hist[valid])
    first=raw[:min(len(raw),HORIZON)]
    rem=HORIZON-len(first)
    if rem<=0:return first[:HORIZON]
    hist2=np.concatenate([hist,first],axis=0)[-10:]
    tail,_= _chain(_oracle_first_unit.net,_oracle_first_unit.ck,_oracle_first_unit.pool,hist2,state,device=device,length=rem)
    return np.concatenate([first,tail],axis=0)[:HORIZON]


def evaluate(out=OUT, *, device="cuda", include_pi05=True):
    """Offline comparison on the 120 held-out demonstrations, at each 10-frame window."""
    import jax
    import jax.numpy as jnp
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import module_jit
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.15)
    out=Path(out); dest=out/"evaluation.json"
    if dest.exists(): return json.loads(dest.read_text())
    source_manifest=json.loads((SOURCE/"manifest.json").read_text())
    info={(int(r["task_id"]),int(r["episode"])):r for r in source_manifest["demos"]}
    manifest=json.loads((SOURCE/"manifest.json").read_text())
    pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
    p5_models=[_load_net(SOURCE/f"models/P5_U_seed{seed}.pt",pool,device=device) for seed in (0,1,2)]
    base,_=_load_expert_model()
    best=json.loads((out/"expert/best.json").read_text())
    trained,_=_load_expert_model()
    _overlay_expert(trained,out/"expert"/best["checkpoint"])
    stats=normalize.load(CHECKPOINT/"assets")["actions"]
    normal_fn=module_jit(base.sample_actions,static_argnames=("num_steps",))
    base_fn=module_jit(base.sample_bridge,static_argnames=("num_steps",));trained_fn=module_jit(trained.sample_bridge,static_argnames=("num_steps",))
    rows_path=out/"eval"/"rows.jsonl";rows_path.parent.mkdir(parents=True,exist_ok=True)
    rows=[];seen=set();accum={}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try: rec=json.loads(line)
            except json.JSONDecodeError: continue
            key=(int(rec["task"]),int(rec["episode"]),int(rec["frame"]))
            if key not in seen: rows.append(rec);seen.add(key)
    # Resume aggregates from already durable rows as well as newly evaluated
    # rows, so a restarted evaluation never silently drops its completed work.
    for rec in rows:
        for name,metric in rec.items():
            if name not in ("task","episode","frame"):
                accum.setdefault(name,[]).append(metric)
    append=rows_path.open("a",buffering=1)
    eval_infos=[r for r in manifest["demos"] if r["split"]=="heldout"]
    for info_row in eval_infos:
        tid,ep=int(info_row["task_id"]),int(info_row["episode"])
        action,state,_=ex._load_cached_episode(SOURCE,tid,ep)
        if len(action)<50:continue
        for start in range(0,len(action)-49,10):
            key=(tid,ep,int(start))
            if key in seen: continue
            prev=action[max(0,start-10):start]; hist=np.zeros((10,12),np.float32)
            if len(prev):hist[-len(prev):]=prev
            draft_raw,_=_chain(p5_models,None,pool,hist,state[start],episode=None,device=device,length=50)
            # Oracle type is deliberately only a diagnostic ceiling: exact
            # current 10-step action chooses the first memory unit.
            q10=action[start:start+10]
            _oracle_first_unit.net=p5_models[0][0];_oracle_first_unit.ck=p5_models[0][1];_oracle_first_unit.pool=pool
            oracle_raw=_oracle_first_unit(pool,q10,hist,state[start],device=device)
            obs,target,_,_= _build_obs_and_target(info_row,start,SOURCE,CHECKPOINT)
            target_policy=np.asarray(target,np.float32)
            draft_policy=ex.action_storage_to_policy(draft_raw)
            oracle_policy=ex.action_storage_to_policy(oracle_raw)
            def padnorm(v):
                pp=np.zeros((1,50,32),np.float32);pp[0,:,:12]=v
                return _norm_action(pp,stats)
            dn=padnorm(draft_policy); on=padnorm(oracle_policy); zeros=np.zeros_like(dn)
            hold=np.zeros((1,50,32),np.float32)
            if len(prev): hold[0,:,:12]=np.repeat(ex.action_storage_to_policy(prev[-1:]),50,axis=0)
            hold=_norm_action(hold,stats)
            eps=np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(tid*1000000+ep*1000+start)),(1,50,32)))
            records={}
            # (a) Normal flow from noise; bridge with t0=1 is the same action ODE.
            normal=np.asarray(normal_fn(jax.random.key(0),obs,num_steps=10,noise=eps))[0]
            records["a_normal"]=(normal,None)
            for t0 in T0S:
                pred_b=np.asarray(base_fn(jax.random.key(0),obs,dn,jnp.asarray([t0]),num_steps=10,noise=eps))[0]
                pred_c=np.asarray(trained_fn(jax.random.key(0),obs,dn,jnp.asarray([t0]),num_steps=10,noise=eps))[0]
                pred_d=np.asarray(trained_fn(jax.random.key(0),obs,on,jnp.asarray([t0]),num_steps=10,noise=eps))[0]
                records[f"b_pretrained_t{t0:.1f}"]=(pred_b,None)
                records[f"c_trained_t{t0:.1f}"]=(pred_c,None)
                records[f"d_oracle_t{t0:.1f}"]=(pred_d,None)
            records["e_hold_last"]=(hold[0],None);records["f_draft"]=(dn[0],None)
            group={"task":tid,"episode":ep,"frame":start}
            target7=target_policy
            for name,(pred,_) in records.items():
                err=(np.asarray(pred)[:50,:12]-target7[:50,:12])**2
                metric={"mse12_first5":float(err[:5].mean()),"mse12_50":float(err.mean()),
                    "mse_eef6_first5":float(err[:5,:6].mean()),"mse_eef6_50":float(err[:,:6].mean()),
                    "mse_grip_first5":float(err[:5,6].mean()),"mse_grip_50":float(err[:,6].mean()),
                    "mse_base_mode_first5":float(err[:5,7:12].mean()),"mse_base_mode_50":float(err[:,7:12].mean())}
                accum.setdefault(name,[]).append(metric); group[name]=metric
            rows.append(group);seen.add(key);append.write(json.dumps(group,separators=(",",":"))+"\n")
            if len(rows)%200==0: print(json.dumps({"stage":"evaluate","rows":len(rows)}),flush=True)
    append.close()
    summary={}
    for name,vals in accum.items(): summary[name]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    by_task={}
    for tid,task in enumerate(ex.TASKS):
        rr=[r for r in rows if r["task"]==tid]
        by_task[task]={name:{k:float(np.mean([r[name][k] for r in rr])) for k in summary[name]} for name in summary}
    _json(dest,{"rows":len(rows),"t0":list(T0S),"checkpoint_step":best["steps"],"selected_t0":best["t0"],"metrics":summary,"by_task":by_task,"per_row":rows,
        "action_metric":"checkpoint-normalized fork-order action dimensions; active EEF xyz+rotation 0:6, gripper 6, base/mode 7:12","normal":"same pretrained Pi0.5 weights, t0=1, ten Euler steps","oracle":"current window action selects first P5 memory unit only; diagnostic upper bound, not deployable","holdout_caveat":"The 120 held-out demonstrations may overlap the pretrained human300 training set; π0.5 offline errors may be optimistic."})
    return json.loads(dest.read_text())


def _fit_action_scale(shape, reference, groups):
    """Closed-form nonnegative scale(s) fitting a type to a policy-order chunk."""
    shape=np.asarray(shape,np.float64);reference=np.asarray(reference,np.float64)
    scales=[]
    for sl in groups:
        x=shape[...,sl].reshape(-1);y=reference[...,sl].reshape(-1);den=float(x@x)
        scales.append(max(0.,float(x@y)/den) if den>1e-12 else 0.)
    return scales


def _p5_scale_draft(hist_storage,state_storage,normal_policy,pool,p5_models,*,episode,stats,device="cpu"):
    """Chain P5-selected type units; fit only their magnitude to pi0.5 output.

    Returns normalized fork-order drafts for shared/separate scale ×
    type/pi0.5 gripper choices, plus scale diagnostics. The reference chunk is
    used only to fit magnitude; none of its arm actions are copied into drafts.
    """
    import torch
    from experiment import _features
    initial_hist=np.asarray(hist_storage,np.float32).copy();state=np.asarray(state_storage,np.float32)
    norm_policy=_denorm_action(np.asarray(normal_policy,np.float32)[None],stats)[0,:,:12]
    configs=("shared_type","shared_pi05","separate_type","separate_pi05")
    out={key:np.zeros((HORIZON,12),np.float32) for key in configs}
    scales={key:[] for key in configs}
    models=list(p5_models)
    def rms_scale(src,ref):
        ns=float(np.sqrt(np.mean(np.sum(src**2,axis=1))))
        nr=float(np.sqrt(np.mean(np.sum(ref**2,axis=1))))
        return nr/ns if ns>=1e-8 and nr>=1e-8 else 1.
    # Every condition has its own autoregressive history. This matters because
    # scale and gripper choices alter the next P5 query, just as the executed
    # generated command does in the closed-loop adapter.
    for key in configs:
        hist=initial_hist.copy();cursor=0
        while cursor<HORIZON:
            scores=np.mean([_p5_select(net,ck,pool,hist,state,device=device)[1] for net,ck in models],axis=0)
            scores[np.asarray(pool["global_episode"])==int(episode)]=-np.inf
            j=int(np.argmax(scores));a0=int(pool["action_offsets"][j]);a1=int(pool["action_offsets"][j+1])
            action_storage=np.asarray(pool["actions_flat"][a0:a1],np.float32)
            action_policy=ex.action_storage_to_policy(action_storage)
            n=min(len(action_policy),HORIZON-cursor)
            if n<=0: raise RuntimeError("P5 selected an empty memory unit")
            ref=norm_policy[cursor:cursor+n,:6]
            shared=_fit_action_scale(action_policy[:n],ref,(slice(0,6),))[0]
            separate=_fit_action_scale(action_policy[:n],ref,(slice(0,3),slice(3,6)))
            valid=np.linalg.norm(hist,axis=1)>1e-8
            if np.any(valid):
                ref_hist=hist[valid,5:11]
                hist_sc=rms_scale(action_policy[:n,:6],ref_hist)
                hist_sep=[rms_scale(action_policy[:n,:3],ref_hist[:,:3]),rms_scale(action_policy[:n,3:6],ref_hist[:,3:6])]
            else: hist_sc=1.;hist_sep=[1.,1.]
            scales[key].append({"row_unit":int(j),"offset":int(cursor),"length":int(n),"stored":1.,"shared":shared,"separate_pos":separate[0],"separate_rot":separate[1],"history":hist_sc,"history_pos":hist_sep[0],"history_rot":hist_sep[1]})
            if key.startswith("shared_"): scale=[shared,shared]
            else: scale=separate
            emitted=action_policy[:n].copy()
            emitted[:,:3]*=scale[0];emitted[:,3:6]*=scale[1]
            if key.endswith("_pi05"):
                emitted[:,6]=norm_policy[cursor:cursor+n,6]
            out[key][cursor:cursor+n]=emitted
            generated_storage=ex.action_policy_to_storage(emitted)
            hist=np.concatenate([hist,generated_storage],axis=0)[-10:]
            cursor+=n
    # Convert fork-order actions to the bridge's normalized 32-D input.
    return {k:_draft_to_model(ex.action_policy_to_storage(v),stats) for k,v in out.items()},scales


def _fixed_holdout_sample(per_row,seed=20261111,per_episode=5):
    """Select a fixed task×episode-stratified sample, five windows per episode."""
    by={}
    for r in per_row: by.setdefault((int(r["task"]),int(r["episode"])),[]).append(r)
    chosen=[]
    for key,rows in sorted(by.items()):
        rows=sorted(rows,key=lambda r:int(r["frame"]))
        rng=np.random.default_rng(seed+key[0]*100000+key[1])
        n=min(int(per_episode),len(rows));ids=np.sort(rng.choice(len(rows),n,replace=False))
        chosen.extend(rows[int(i)] for i in ids)
    return chosen


def _load_expert_switch(checkpoint_path):
    """Load one model and host-side snapshots for pretrained/trained switching."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex
    model,_=_load_expert_model()
    filt=nnx.All(nnx.Param,PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    template=nnx.state(model,filt).to_pure_dict()
    base_state=jax.tree.map(lambda x:np.asarray(x),template)
    trained_state=serialization.from_bytes(template,Path(checkpoint_path).read_bytes())
    trained_state=jax.tree.map(lambda x:np.asarray(x),trained_state)
    def switch(state):
        selected=nnx.state(model,filt)
        selected.replace_by_pure_dict(jax.tree.map(jnp.asarray,state))
        nnx.update(model,selected)
    del template
    return model,base_state,trained_state,switch


def evaluate_magnitude(out=OUT,*,device="cuda"):
    """Memory-validation-first selection, followed by stratified 600-window eval."""
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import module_jit
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.35)
    out=Path(out);sel_path=out/"selection.json"
    if sel_path.exists() and json.loads(sel_path.read_text()).get("validation_version") not in (None,3):
        raise FileExistsError("magnitude-fit implementation changed; use a fresh --out directory to avoid mixing old and new results")
    if sel_path.exists() and json.loads(sel_path.read_text()).get("validation_version")==3: selection=json.loads(sel_path.read_text())
    else:
        source_manifest=json.loads((SOURCE/"manifest.json").read_text())
        info={(int(r["task_id"]),int(r["episode"])):r for r in source_manifest["demos"]}
        pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
        p5=[_load_net(SOURCE/f"models/P5_U_seed{s}.pt",pool,device="cpu") for s in (0,1,2)]
        best=json.loads((out/"expert/best.json").read_text())
        model,base_state,trained_state,switch=_load_expert_switch(out/"expert"/best["checkpoint"])
        stats=normalize.load(CHECKPOINT/"assets")["actions"]
        switch(base_state)
        normal_fn=module_jit(model.sample_actions,static_argnames=("num_steps",))
        switch(trained_state)
        bridge_fn=module_jit(model.sample_bridge,static_argnames=("num_steps",))
        cache=np.load(out/"drafts/train_dev.npz",allow_pickle=False)
        val_ids=np.flatnonzero(cache["validation"]&(cache["frame"]==0))
        if len(val_ids)<96: raise AssertionError(f"expected 96 memory validation starts, got {len(val_ids)}")
        candidates={}
        for k in ("shared_type","shared_pi05","separate_type","separate_pi05"):
            for t0 in (.1,.2,.3): candidates[f"h_{k}_t{t0:.1f}"]=[]
        for ix in val_ids:
            switch(base_state)
            tid,ep,frame=int(cache["task"][ix]),int(cache["episode"][ix]),int(cache["frame"][ix]);row=info[(tid,ep)]
            obs,target_norm,_,state=_build_obs_and_target(row,frame,SOURCE,CHECKPOINT)
            eps=np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(tid*1000000+ep*1000+frame)),(1,50,32)))
            normal=np.asarray(normal_fn(jax.random.key(0),obs,num_steps=10,noise=eps))[0]
            drafts,scales=_p5_scale_draft(cache["hist"][ix],state,normal,pool,p5,episode=tid*1000+ep,stats=stats,device="cpu")
            target=cache["target"][ix];target_norm=_draft_to_model(target,stats)[None]
            obs4=jax.tree.map(lambda x:jnp.concatenate([x]*4,axis=0),obs)
            noise4=np.repeat(eps,4,axis=0);draft4=jnp.asarray(np.stack([drafts[k] for k in ("shared_type","shared_pi05","separate_type","separate_pi05")],axis=0))
            switch(trained_state)
            for t0 in (.1,.2,.3):
                pred=np.asarray(bridge_fn(jax.random.key(1),obs4,jnp.asarray(draft4),jnp.full((4,),t0),num_steps=10,noise=jnp.asarray(noise4)))
                for i,k in enumerate(("shared_type","shared_pi05","separate_type","separate_pi05")):
                    candidates[f"h_{k}_t{t0:.1f}"].append(float(np.mean((pred[i:i+1,:5,:12]-target_norm[:,:5,:12])**2)))
        # Compare c at its previously fixed memory-validation selection.
        # c uses the cross-fit draft cache; no held-out row is involved.
        cvals=[]
        for ix in val_ids:
            row=info[(int(cache["task"][ix]),int(cache["episode"][ix]))]
            obs,target_norm,_,_=_build_obs_and_target(row,int(cache["frame"][ix]),SOURCE,CHECKPOINT)
            draft=_draft_to_model(cache["draft"][ix],stats)[None]
            eps=np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(cache["task"][ix]*1000000+cache["episode"][ix]*1000+cache["frame"][ix])),(1,50,32)))
            pred=np.asarray(bridge_fn(jax.random.key(1),obs,jnp.asarray(draft),jnp.asarray([.2]),num_steps=10,noise=jnp.asarray(eps)))
            cvals.append(float(np.mean((pred[:,:5,:12]-target_norm[:5,:12])**2)))
        vals={k:float(np.mean(v)) for k,v in candidates.items()};vals["c_trained_p5_t0.2"]=float(np.mean(cvals))
        winner=min(vals,key=vals.get)
        selection={"validation_version":3,"selection_source":"96 frame-0 windows from memory-side Expert validation episodes only","metric":"mean first-5 action MSE over normalized fork-order 12 dimensions","t0_for_c":.2,"t0_grid_for_h":[.1,.2,.3],"validation_mse":vals,"selected_condition":winner,"selected_value":vals[winner],"configuration_order":["shared_type","shared_pi05","separate_type","separate_pi05"],"note":"Each scale/gripper condition autoregressively chains from its own generated history. Pretrained and trained bridge wrappers were separately frozen after loading their respective weights. Selection uses memory-side validation only and is written before held-out evaluation."}
        _json(sel_path,selection)

    # The held-out comparison is fixed to five randomly selected windows per
    # task×episode (seeded independently by task and episode), approx. 600 rows.
    epath=out/"evaluation.json";e=json.loads(epath.read_text())
    saved=e.get("magnitude_extension",{});ext=saved if saved.get("complete") and saved.get("state_switch_version")==3 else None
    if ext is None:
        if saved and saved.get("state_switch_version") not in (None,3):
            raise FileExistsError("magnitude evaluation implementation changed; use a fresh --out directory to avoid mixing old and new results")
        source_manifest=json.loads((SOURCE/"manifest.json").read_text())
        info={(int(r["task_id"]),int(r["episode"])):r for r in source_manifest["demos"]}
        pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
        p5=[_load_net(SOURCE/f"models/P5_U_seed{s}.pt",pool,device="cpu") for s in (0,1,2)]
        best=json.loads((out/"expert/best.json").read_text())
        model,base_state,trained_state,switch=_load_expert_switch(out/"expert"/best["checkpoint"])
        stats=normalize.load(CHECKPOINT/"assets")["actions"]
        switch(base_state)
        normal_fn=module_jit(model.sample_actions,static_argnames=("num_steps",))
        bridge_base=module_jit(model.sample_bridge,static_argnames=("num_steps",))
        switch(trained_state)
        bridge_trained=module_jit(model.sample_bridge,static_argnames=("num_steps",))
        subset=_fixed_holdout_sample(e["per_row"]); rows=list(saved.get("rows",[])) if saved.get("state_switch_version")==3 else [];done={(int(r["task"]),int(r["episode"]),int(r["frame"])) for r in rows};cfgs=("shared_type","shared_pi05","separate_type","separate_pi05")
        # Preserve exact a/c values from the original paired evaluation.
        bykey={(int(r["task"]),int(r["episode"]),int(r["frame"])):r for r in e["per_row"]}
        for ri,old in enumerate(subset):
            tid,ep,frame=int(old["task"]),int(old["episode"]),int(old["frame"]);item=info[(tid,ep)]
            if (tid,ep,frame) in done: continue
            switch(base_state)
            obs, target_norm, target_raw, state=_build_obs_and_target(item,frame,SOURCE,CHECKPOINT)
            eps=np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(tid*1000000+ep*1000+frame)),(1,50,32)))
            normal=np.asarray(normal_fn(jax.random.key(0),obs,num_steps=10,noise=eps))[0]
            drafts,unit_scales=_p5_scale_draft(_prev_hist(SOURCE,tid,ep,frame),state,normal,pool,p5,episode=tid*1000+ep,stats=stats,device="cpu")
            target_model=_draft_to_model(target_raw,stats)[None]
            draft_mse={k:{"first5_mse12":float(np.mean((v[None,:5,:12]-target_model[:,:5,:12])**2)),"first5_mse6":float(np.mean((v[None,:5,:6]-target_model[:,:5,:6])**2)),"first5_grip":float(np.mean((v[None,:5,6:7]-target_model[:,:5,6:7])**2)),"mse12_50":float(np.mean((v[None,:,:12]-target_model[:,:,:12])**2)),"mse6_50":float(np.mean((v[None,:,:6]-target_model[:,:,:6])**2)),"grip_50":float(np.mean((v[None,:,6:7]-target_model[:,:,6:7])**2))} for k,v in drafts.items()}
            draft4=np.stack([drafts[k] for k in cfgs],axis=0)
            obs4=jax.tree.map(lambda x:jnp.concatenate([x]*4,axis=0),obs);noise4=np.repeat(eps,4,axis=0)
            hres={}
            for t0 in (.1,.2,.3):
                switch(base_state)
                preds0=np.asarray(bridge_base(jax.random.key(1),obs4,jnp.asarray(draft4),jnp.full((4,),t0),num_steps=10,noise=jnp.asarray(noise4)))
                switch(trained_state)
                preds1=np.asarray(bridge_trained(jax.random.key(1),obs4,jnp.asarray(draft4),jnp.full((4,),t0),num_steps=10,noise=jnp.asarray(noise4)))
                for modelname,preds in (("pretrained",preds0),("trained",preds1)):
                    for i,k in enumerate(cfgs):
                        err=(preds[i:i+1,:,:12]-target_model[:,:,:12])**2
                        hres[f"{modelname}_{k}_t{t0:.1f}"]={"first5_mse12":float(err[:,:5].mean()),"first5_mse6":float(err[:,:5,:6].mean()),"first5_grip":float(err[:,:5,6].mean()),"mse12_50":float(err.mean()),"mse6_50":float(err[:,:,:6].mean()),"grip_50":float(err[:,:,6].mean())}
            # Ground-truth scale is diagnostic only, never used for draft construction or selection.
            ratios={k:[] for k in cfgs}
            target_policy=ex.action_storage_to_policy(target_raw)
            for cfg in cfgs:
                for sm in unit_scales[cfg]:
                    st=int(sm["offset"]);en=st+int(sm["length"]);shape=np.asarray(pool["actions_flat"][pool["action_offsets"][sm["row_unit"]]:pool["action_offsets"][sm["row_unit"]+1]],np.float32)
                    shape=ex.action_storage_to_policy(shape[:en-st]);ref=target_policy[st:en,:6]
                    sstar=_fit_action_scale(shape,ref,(slice(0,6),))[0];sstar_sep=_fit_action_scale(shape,ref,(slice(0,3),slice(3,6)))
                    ratios[cfg].append({**sm,"sstar":sstar,"sstar_pos":sstar_sep[0],"sstar_rot":sstar_sep[1],
                        "fit_over_sstar_shared":float(sm["shared"]/max(sstar,1e-8)),"fit_over_sstar_sep_pos":float(sm["separate_pos"]/max(sstar_sep[0],1e-8)),"fit_over_sstar_sep_rot":float(sm["separate_rot"]/max(sstar_sep[1],1e-8))})
            rows.append({"task":tid,"episode":ep,"frame":frame,"baseline_a":bykey[(tid,ep,frame)]["a_normal"],"baseline_c_selected":bykey[(tid,ep,frame)][f"c_trained_t{best['t0']:.1f}"],"drafts":draft_mse,"bridges":hres,"unit_scales":ratios})
            if len(rows)%50==0:
                e=json.loads(epath.read_text());e["magnitude_extension"]={"complete":False,"state_switch_version":3,"subset_seed":20261111,"subset_per_task_episode":5,"rows":rows,"checkpoint_step":best["steps"],"best_t0_c":best["t0"]};_json(epath,e)
                print(json.dumps({"stage":"magnitude_eval","rows":len(rows)}),flush=True)
        ext={"complete":True,"state_switch_version":3,"subset_seed":20261111,"subset_per_task_episode":5,"rows":rows,"checkpoint_step":best["steps"],"best_t0_c":best["t0"]}
        e=json.loads(epath.read_text());e["magnitude_extension"]=ext;_json(epath,e)
    report(out)
    return {"selection":selection,"evaluation_rows":len(ext["rows"])}


def _prev_hist(source,task,episode,frame):
    action,_,_ = ex._load_cached_episode(source,task,episode)
    hist=np.zeros((10,12),np.float32);prev=action[max(0,int(frame)-10):int(frame)]
    if len(prev):hist[-len(prev):]=prev
    return hist


def report(out=OUT):
    out=Path(out); e=json.loads((out/"evaluation.json").read_text()); p=json.loads((out/"prepare.json").read_text()); t=json.loads((out/"expert/trained.json").read_text())
    batch_note=""
    schedule=out/"expert/batch_schedule.json"
    if schedule.exists():
        s=json.loads(schedule.read_text())
        batch_note=(f" Training used batch {s['initial_batch']} through step {s['resumed_at_step']}, "
                    f"batch {s['resumed_batch']} through step {s['final_resume_at_step']}, then batch {s['final_batch']} "
                    "with validation batch 1 and explicit release of its temporary JIT/arrays to stay within the shared-GPU limit; "
                    "each change resumed model and optimizer state from the saved checkpoint.")
    lines=["# RoboCasa P5 draft action-expert bridge","", "Offline held-out demonstrations are used only for scoring. The checkpoint / t0 choice uses memory-side validation episodes. The held-out demos may overlap the pretrained human300 data, so π0.5 open-loop MSE can be optimistic; this is not a closed-loop success result.","",
        f"Cross-fit P5 generated {p['rows']} 50-step drafts from memory demonstrations; its selected unit candidates exclude the query episode. Expert best update: {t['best_step']} (t0={t['best_t0']}). Action channels are fork order [EEF6,grip,base4,mode].{batch_note}", "",
        "| condition | first-5 MSE12 | first-5 EEF6 | first-5 grip | 50-step MSE12 | 50-step EEF6 | 50-step grip |", "|---|---:|---:|---:|---:|---:|---:|"]
    order=["a_normal"]+[f"b_pretrained_t{t:.1f}" for t in T0S]+[f"c_trained_t{t:.1f}" for t in T0S]+[f"d_oracle_t{t:.1f}" for t in T0S]+["e_hold_last","f_draft"]
    for name in order:
        v=e["metrics"][name];lines.append(f"| {name} | {v['mse12_first5']:.5f} | {v['mse_eef6_first5']:.5f} | {v['mse_grip_first5']:.5f} | {v['mse12_50']:.5f} | {v['mse_eef6_50']:.5f} | {v['mse_grip_50']:.5f} |")
    lines += ["", "## Per task", "", "| task | normal first5 | pretrained P5 first5 | trained P5 first5 | trained P5 50-step | oracle first5 | hold-last first5 | P5 draft first5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task,d in e["by_task"].items():
        t=e["selected_t0"];b=f"b_pretrained_t{t:.1f}";c=f"c_trained_t{t:.1f}";o=f"d_oracle_t{t:.1f}"
        lines.append(f"| {task} | {d['a_normal']['mse12_first5']:.5f} | {d[b]['mse12_first5']:.5f} | {d[c]['mse12_first5']:.5f} | {d[c]['mse12_50']:.5f} | {d[o]['mse12_first5']:.5f} | {d['e_hold_last']['mse12_first5']:.5f} | {d['f_draft']['mse12_first5']:.5f} |")
    lines += ["", "## Serving adapter", "", "`serve_bridge_action(...)` in expert.py accepts a model Observation, the last 10 executed fork-order actions, fork-order state, memory pool and P5 model, then returns a 50×12 unnormalized fork-order chunk. It is intended to be called from the serving policy after observation transforms. A real memory observation passed through the trained step-3500 Expert and returned a finite 50×12 chunk (`smoke/trained_serving.npz`).", ""]
    ext=e.get("magnitude_extension")
    if ext and ext.get("complete") and ext.get("state_switch_version")!=3:
        raise RuntimeError("stored magnitude results predate the per-condition autoregressive chain; rerun in a fresh --out directory")
    if ext and ext.get("complete"):
        sr=json.loads((out/"selection.json").read_text())
        lines += ["## π0.5 magnitude fit (speed-priority subset)", "", f"Magnitude-fit scales were selected on {sr['selection_source']}; the recorded lowest first-five MSE condition was **{sr['selected_condition']}** ({sr['selected_value']:.5f}). Held-out evaluation used {len(ext['rows'])} windows: at most five starts selected with a fixed seed per task×episode (seed 20261111), stratified across held-out episodes. The same sampled rows report (a) normal and (c) trained P5 bridge from the original paired evaluation. For (g), π0.5 contributes only fitted scale and optionally its gripper; its arm actions are not copied into the draft.", "", "| condition | first-5 MSE12 | first-5 EEF6 | first-5 grip | 50-step MSE12 | 50-step EEF6 | 50-step grip |", "|---|---:|---:|---:|---:|---:|---:|"]
        picked=ext["rows"]
        for name,metric in (("a_normal",("baseline_a",None)),("c_trained_P5_selected",("baseline_c_selected",None))):
            vals=[r[metric[0]] for r in picked];m={k:float(np.mean([v[k] for v in vals])) for k in ("mse12_first5","mse_eef6_first5","mse_grip_first5","mse12_50","mse_eef6_50","mse_grip_50")}
            lines.append(f"| {name} | {m['mse12_first5']:.5f} | {m['mse_eef6_first5']:.5f} | {m['mse_grip_first5']:.5f} | {m['mse12_50']:.5f} | {m['mse_eef6_50']:.5f} | {m['mse_grip_50']:.5f} |")
        for cfg in ("shared_type","shared_pi05","separate_type","separate_pi05"):
            for name in ("g_draft",):
                vals=[r["drafts"][cfg] for r in picked];m={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
                lines.append(f"| {name}_{cfg} | {m['first5_mse12']:.5f} | {m['first5_mse6']:.5f} | {m['first5_grip']:.5f} | {m['mse12_50']:.5f} | {m['mse6_50']:.5f} | {m['grip_50']:.5f} |")
            for modelname in ("pretrained","trained"):
                for t0 in (.1,.2,.3):
                    vals=[r["bridges"][f"{modelname}_{cfg}_t{t0:.1f}"] for r in picked];m={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
                    lines.append(f"| h_{modelname}_{cfg}_t{t0:.1f} | {m['first5_mse12']:.5f} | {m['first5_mse6']:.5f} | {m['first5_grip']:.5f} | {m['mse12_50']:.5f} | {m['mse6_50']:.5f} | {m['grip_50']:.5f} |")
        for cfg in ("shared_type","shared_pi05","separate_type","separate_pi05"):
          for scale_key,title in (("fit_over_sstar_shared","shared fitted scale / s*"),("fit_over_sstar_sep_pos","separate position scale / s*"),("fit_over_sstar_sep_rot","separate rotation scale / s*"),("stored","stored scale"),("history","history-derived shared scale")):
            vals=[unit[scale_key] for row in picked for unit in row["unit_scales"][cfg] if np.isfinite(unit[scale_key])]
            if vals:
                q=np.quantile(vals,[.25,.5,.75]);lines.append(f"- {cfg} {title}: median {q[1]:.3f}, IQR [{q[0]:.3f}, {q[2]:.3f}] (per selected segment).")
        lines += ["", "Memory-side validation chooses among h conditions only; all holdout values above are descriptive. π0.5 estimates scale only. Holdout actions enter the scale oracle ratio and MSE scoring, not the fitted scale, P5 unit choice, or bridge selection.", ""]
    lines += ["## Review notes", "", "Inspect the fold-level P5 validation, whether first-five Dev MSE is the preferred checkpoint criterion for the intended control use, and the oracle-ceiling interpretation. No closed-loop rollouts were run.", ""]
    (out/"report.md").write_text("\n".join(lines))
    return str(out/"report.md")


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","train","evaluate","magnitude","report","all"));p.add_argument("--out",type=Path,default=OUT);p.add_argument("--steps",type=int,default=4000);p.add_argument("--batch",type=int,default=4);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    if a.stage in ("prepare","all"):_prepare_data(a.out,smoke=a.smoke)
    if a.stage in ("train","all"):train_expert(a.out,steps=2 if a.smoke else a.steps,batch=a.batch)
    if a.stage in ("evaluate","all"):evaluate(a.out)
    if a.stage=="magnitude":evaluate_magnitude(a.out)
    if a.stage in ("report","all"):print(report(a.out))

if __name__=="__main__":main()
