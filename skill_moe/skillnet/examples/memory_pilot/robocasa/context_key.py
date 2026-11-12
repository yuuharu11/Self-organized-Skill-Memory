"""RM-CTX offline context-key experiment (no closed-loop code here).

The query uses only a frozen pi05 prefix representation and robot state.  The
retrieved memory value is one variable-length U motion unit.  Held-out actions
are read only by the scoring/evaluation stage.
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
ROBOCASA_EXAMPLES = HERE.parents[1] / "robocasa"
if str(ROBOCASA_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_EXAMPLES))
import experiment as ex
import expert

OUT = Path(os.environ.get("ROBOCASA_CTX_OUT", "/work/.skillnet-state/outputs/robocasa-memory/ctx-01"))
SOURCE = expert.SOURCE
CHECKPOINT = expert.CHECKPOINT
SEED = 4111
PI05_NOISE_SEED = 4101
PI05_NOISE_DESCRIPTION = f"per-query Gaussian, seed={PI05_NOISE_SEED}, same key rule as gripper-01, 10-step pi05 sampler"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def _write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def _log(out, stage, **payload):
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **payload}
    with (Path(out) / "progress.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _inventory():
    """Return stride-10 memory/eval windows and all U candidate start contexts."""
    m = json.loads((SOURCE / "manifest.json").read_text())
    pool = dict(np.load(SOURCE / "pools/U_memory.npz", allow_pickle=False))
    records = {"memory": [], "heldout": []}
    for info in m["demos"]:
        split = "memory" if info["split"] == "memory" else "heldout"
        actions, states, _frames = ex._load_cached_episode(SOURCE, info["task_id"], info["episode"])
        for frame in range(0, max(0, len(actions) - 9), 10):
            if frame + 10 > len(actions):
                continue
            hist = np.zeros((10, 12), np.float32)
            past = actions[max(0, frame - 10):frame]
            if len(past): hist[-len(past):] = past
            records[split].append({"task": int(info["task_id"]), "episode": int(info["episode"]),
                "global_episode": int(info["global_episode"]), "frame": int(frame), "state": states[frame].copy(),
                "hist": hist, "target": actions[frame:frame+10].copy(), "prompt": str(info["prompt"]),
                "root": str(m["task_splits"][int(info["task_id"])] ["root"])})
    # Every candidate's visual key is the observation at the unit's start.
    prompt_by_episode={(int(x["task_id"]),int(x["episode"])):str(x["prompt"]) for x in m["demos"]}
    candidates = []; episode_state_cache={}
    for i in range(len(pool["length"])):
        ek=(int(pool["task"][i]),int(pool["episode"][i]))
        if ek not in episode_state_cache:episode_state_cache[ek]=ex._load_cached_episode(SOURCE,*ek)[1]
        candidates.append({"task": int(pool["task"][i]), "episode": int(pool["episode"][i]),
            "global_episode": int(pool["global_episode"][i]), "frame": int(pool["frame"][i]),
            "state": episode_state_cache[ek][int(pool["frame"][i])].copy(),
            "pool_index": i, "prompt": prompt_by_episode[ek],
            "root": str(m["task_splits"][int(pool["task"][i])]["root"])})
    return m, pool, records, candidates


def _ctx_id(row):
    return f"{int(row['task'])}:{int(row['episode'])}:{int(row['frame'])}"


def _pi05_eval_noise(row, *, action_dim=32):
    """Use the same per-query Gaussian noise as gripper-01's G1 baseline."""
    import jax
    return jax.random.normal(
        jax.random.fold_in(jax.random.key(PI05_NOISE_SEED),
                           int(row["task"]) * 1_000_000 + int(row["episode"]) * 1000 + int(row["frame"])),
        (1, 50, int(action_dim)),
    )


def _require_current_pi05_noise_manifest(out):
    manifest_path = Path(out) / "cache/feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing feature manifest: {manifest_path}; run extract into a fresh output directory")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("evaluation_actions_noise") != PI05_NOISE_DESCRIPTION:
        raise ValueError("cached π0.5 scale references use a different noise rule; extract and evaluate in a fresh output directory")
    return manifest


def _make_transforms(checkpoint):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config as train_config
    cfg = train_config.get_config("pi05_robocasa_multitask")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    return transforms.compose([transforms.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
        transforms.Normalize(normalize.load(Path(checkpoint) / "assets"), use_quantiles=False),
        *dc.model_transforms.inputs])


def _prefix_extractor(model):
    """Compile the final transformer-prefix mean; image+text masks are honored."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0
    from openpi.shared.nnx_utils import module_jit

    class Extract(nnx.Module):
        def __init__(self, base): self.base = base
        def __call__(self, obs):
            x, mask, ar = self.base.embed_prefix(obs)
            input_weights = mask.astype(x.dtype)[..., None]
            input_pooled = jnp.sum(x * input_weights, axis=1) / jnp.maximum(jnp.sum(input_weights, axis=1), 1)
            attention = pi0.make_attn_mask(mask, ar)
            positions = jnp.cumsum(mask, axis=1) - 1
            (hidden, _), _ = self.base.PaliGemma.llm([x, None], mask=attention, positions=positions,
                adarms_cond=[None, None])
            weights = mask.astype(hidden.dtype)[..., None]
            pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
            return pooled.astype(jnp.float16), input_pooled.astype(jnp.float16)
    extractor = Extract(model)
    return module_jit(extractor.__call__)


def _read_camera(root, camera, episode, frame):
    # Reuse bounded decoder cache from the existing server, without starting it.
    from expert import _video_frame
    path = Path(root) / "videos/chunk-000" / f"observation.images.{camera}" / f"episode_{int(episode):06d}.mp4"
    return _video_frame(str(path), int(frame))


def _feature_rows(rows, transform, extractor, *, batch_size=4, predict_actions=False, model=None):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.shared import normalize
    norm = normalize.load(CHECKPOINT / "assets")["actions"]
    result = np.empty((len(rows), 2048), np.float16)
    input_result = np.empty_like(result)
    action_result = np.empty((len(rows), 50, 12), np.float32) if predict_actions else None
    base_ref_fn = None
    if predict_actions:
        from openpi.shared.nnx_utils import module_jit
        base_ref_fn = module_jit(model.sample_actions, static_argnames=("num_steps",))
    for start in range(0, len(rows), int(batch_size)):
        chunk = rows[start:start + int(batch_size)]
        items = []
        for r in chunk:
            images = [_read_camera(r["root"], cam, r["episode"], r["frame"]) for cam in CAMERAS]
            raw = {"observation/image": images[0], "observation/right_image": images[1],
                   "observation/wrist_image": images[2], "observation/state": ex.state_storage_to_policy(r["state"]),
                   "prompt": r["prompt"]}
            items.append(transform(raw))
        # Transforms are single-example; batch their nested output after resize/tokenization.
        batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])), *items)
        obs = model_api.Observation.from_dict(batched)
        pooled, input_pooled = extractor(obs)
        pooled = np.asarray(pooled, np.float16); input_pooled = np.asarray(input_pooled, np.float16)
        if pooled.shape[1] != 2048 or input_pooled.shape[1] != 2048:
            raise ValueError(f"expected two 2048-wide PaliGemma prefix views, got {pooled.shape}, {input_pooled.shape}")
        result[start:start + len(chunk)] = pooled
        input_result[start:start + len(chunk)] = input_pooled
        if predict_actions:
            noise = jnp.concatenate([_pi05_eval_noise(r, action_dim=model.action_dim) for r in chunk], axis=0)
            pred = base_ref_fn(jax.random.key(0), obs, num_steps=10, noise=noise)
            pred = np.asarray(pred, np.float32) * (np.asarray(norm.std, np.float32)[None, None, :] + 1e-6) + np.asarray(norm.mean, np.float32)[None, None, :]
            action_result[start:start + len(chunk)] = np.stack([ex.action_policy_to_storage(a[:, :12]) for a in pred])
        if start % max(int(batch_size) * 100, int(batch_size)) == 0:
            print(f"features {min(start+len(chunk),len(rows))}/{len(rows)}", flush=True)
    return result, input_result, action_result


def extract(out=OUT, *, batch_size=4):
    import jax
    from openpi import transforms
    out = Path(out); cache_dir = out / "cache"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite cached features under {cache_dir}; use a fresh output directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest, pool, rows, candidates = _inventory()
    # Keep only context locations needed by training queries, held-out queries, and candidates.
    contexts = {}; references = {}
    for split in ("memory", "heldout"):
        for i, r in enumerate(rows[split]):
            key = _ctx_id(r); contexts.setdefault(key, r); references[f"{split}:{i}"] = key
    for i, r in enumerate(candidates):
        key = _ctx_id(r); contexts.setdefault(key, r); references[f"candidate:{i}"] = key
    ordered = list(contexts.values()); keys = list(contexts)
    transform = _make_transforms(CHECKPOINT)
    model, _cfg = expert._load_expert_model(CHECKPOINT)
    extract_fn = _prefix_extractor(model)
    # For held-out evaluation, the same prefix pass supplies the frozen pi05 action reference.
    ev_rows = rows["heldout"]
    ev_features, ev_input_features, ev_actions = _feature_rows(ev_rows, transform, extract_fn, batch_size=batch_size,
        predict_actions=True, model=model)
    ev_feature_by_id = {_ctx_id(r): ev_features[i] for i, r in enumerate(ev_rows)}
    ev_input_by_id = {_ctx_id(r): ev_input_features[i] for i, r in enumerate(ev_rows)}
    ev_action_by_id = {_ctx_id(r): ev_actions[i] for i, r in enumerate(ev_rows)}
    # Run non-evaluation locations once. Avoid duplicate extraction for all heldout query contexts.
    rest = [r for r in ordered if _ctx_id(r) not in ev_feature_by_id]
    rest_features, rest_input_features, _ = _feature_rows(rest, transform, extract_fn, batch_size=batch_size, model=model)
    lookup = dict(ev_feature_by_id); lookup.update({_ctx_id(r): rest_features[i] for i, r in enumerate(rest)})
    input_lookup = dict(ev_input_by_id); input_lookup.update({_ctx_id(r): rest_input_features[i] for i, r in enumerate(rest)})
    all_features = np.stack([lookup[k] for k in keys]).astype(np.float16)
    np.save(out / "cache/prefix_mean_fp16.npy", all_features)
    np.save(out / "cache/prefix_input_mean_fp16.npy", np.stack([input_lookup[k] for k in keys]).astype(np.float16))
    # Dense location table links named rows/candidates to the one cached prefix feature.
    key_to_ix = {k: i for i, k in enumerate(keys)}
    tables = {"candidate_context": np.asarray([key_to_ix[references[f"candidate:{i}"]] for i in range(len(candidates))], np.int32)}
    for split in ("memory", "heldout"):
        rec = rows[split]
        tables[f"{split}_context"] = np.asarray([key_to_ix[references[f"{split}:{i}"]] for i in range(len(rec))], np.int32)
        tables[f"{split}_task"] = np.asarray([r["task"] for r in rec], np.int16)
        tables[f"{split}_episode"] = np.asarray([r["global_episode"] for r in rec], np.int32)
        tables[f"{split}_local_episode"] = np.asarray([r["episode"] for r in rec], np.int32)
        tables[f"{split}_frame"] = np.asarray([r["frame"] for r in rec], np.int32)
        tables[f"{split}_state"] = np.stack([ex.state_storage_to_policy(r["state"]) for r in rec]).astype(np.float32)
        tables[f"{split}_storage_state"] = np.stack([r["state"] for r in rec]).astype(np.float32)
        tables[f"{split}_hist"] = np.stack([r["hist"] for r in rec]).astype(np.float32)
        tables[f"{split}_target"] = np.stack([r["target"] for r in rec]).astype(np.float32)
    tables["candidate_task"] = np.asarray([r["task"] for r in candidates], np.int16)
    tables["candidate_episode"] = np.asarray([r["global_episode"] for r in candidates], np.int32)
    tables["candidate_frame"] = np.asarray([r["frame"] for r in candidates], np.int32)
    candidate_state=[]
    episode_cache={}
    for r in candidates:
        ek=(int(r["task"]),int(r["episode"]))
        if ek not in episode_cache:
            episode_cache[ek]=ex._load_cached_episode(SOURCE,*ek)[1]
        candidate_state.append(episode_cache[ek][int(r["frame"])])
    tables["candidate_state"]=ex.state_storage_to_policy(np.stack(candidate_state)).astype(np.float32)
    tables["candidate_storage_state"]=np.stack(candidate_state).astype(np.float32)
    if np.intersect1d(tables["candidate_episode"],tables["heldout_episode"]).size:
        raise AssertionError("held-out evaluation episodes leaked into Context memory candidates")
    tables["eval_pi05_actions"] = np.stack([ev_action_by_id[_ctx_id(r)] for r in ev_rows]).astype(np.float32)
    tables["heldout_grip_sign"] = np.asarray([1 if r["target"][0, 11] >= 0 else -1 for r in ev_rows], np.int8)
    tables["memory_grip_sign"] = np.asarray([1 if r["target"][0, 11] >= 0 else -1 for r in rows["memory"]], np.int8)
    # Prefix features are float16 on disk (same rounding convention as M2); state is stored separately.
    np.savez_compressed(out / "cache/rows.npz", **tables)
    np.savez_compressed(out / "cache/pool.npz", **pool)
    _write_json(out / "cache/feature_manifest.json", {"checkpoint": str(CHECKPOINT), "source": str(SOURCE),
        "feature": "mean of valid final PaliGemma prefix hidden tokens, plus separate mean of valid pre-Transformer prefix embeddings; images left/right/wrist plus task text",
        "storage": "both feature tensors fp16; cast to float32 only at network input", "contexts": len(keys),
        "memory_stride10_windows": len(rows["memory"]), "heldout_stride10_windows": len(rows["heldout"]),
        "candidates": len(candidates), "batch_size": int(batch_size),
        "evaluation_actions_noise": PI05_NOISE_DESCRIPTION})
    _log(out, "extract", contexts=len(keys), memory_windows=len(rows["memory"]), heldout_windows=len(rows["heldout"]), candidates=len(candidates))


def _context_x(out, *, fit_ids=None, feature="final"):
    z = np.load(Path(out) / "cache/rows.npz", allow_pickle=False)
    feature_file = {"final": "prefix_mean_fp16.npy", "input": "prefix_input_mean_fp16.npy"}.get(feature)
    if feature_file is None:
        raise ValueError(f"unknown feature view: {feature}")
    f = np.asarray(np.load(Path(out) / "cache" / feature_file, mmap_mode="r"), np.float32)
    mem_ix = np.asarray(z["memory_context"], np.int64)
    if fit_ids is None: fit_ids = np.arange(len(mem_ix))
    state = np.asarray(z["memory_state"], np.float32)
    mean = f[mem_ix[fit_ids]].mean(0); std = np.maximum(f[mem_ix[fit_ids]].std(0), .01)
    sm = state[fit_ids].mean(0); ss = np.maximum(state[fit_ids].std(0), 1e-4)
    cand_ix = np.asarray(z["candidate_context"], np.int64)
    x = np.concatenate([(f[mem_ix] - mean) / std, (state - sm) / ss], axis=1).astype(np.float32)
    k = np.concatenate([(f[cand_ix] - mean) / std,
                        (np.asarray(z["candidate_state"], np.float32) - sm) / ss], axis=1).astype(np.float32) if "candidate_state" in z else None
    # Candidate state follows each memory-unit start; lookup it from the episodes' state cache below.
    return z, x, k, {"feature_mean": mean, "feature_std": std, "state_mean": sm, "state_std": ss}


def _candidate_context_state(out):
    z=np.load(Path(out)/"cache/rows.npz",allow_pickle=False)
    return np.asarray(z["candidate_state"],np.float32)


def _ensure_policy_state_cache(out):
    """Keep raw LeRobot state for grip bookkeeping and policy-order state for the key."""
    path=Path(out)/"cache/rows.npz"
    z=np.load(path,allow_pickle=False)
    if "memory_storage_state" in z and "heldout_local_episode" in z:
        return
    tables={k:z[k] for k in z.files}
    for split in ("memory","heldout"):
        if f"{split}_storage_state" not in tables:
            raw=np.asarray(tables[f"{split}_state"],np.float32)
            tables[f"{split}_storage_state"]=raw
            tables[f"{split}_state"]=ex.state_storage_to_policy(raw).astype(np.float32)
    if "heldout_local_episode" not in tables:
        manifest=json.loads((SOURCE/"manifest.json").read_text())
        local={int(d["global_episode"]):int(d["episode"]) for d in manifest["demos"]}
        tables["heldout_local_episode"]=np.asarray([local[int(e)] for e in tables["heldout_episode"]],np.int32)
    if "candidate_storage_state" not in tables:
        if "candidate_state" not in tables:
            pool=dict(np.load(Path(out)/"cache/pool.npz",allow_pickle=False));raw_rows=[];state_cache={}
            for task,episode,frame in zip(pool["task"],pool["episode"],pool["frame"],strict=True):
                key=(int(task),int(episode))
                if key not in state_cache:state_cache[key]=ex._load_cached_episode(SOURCE,*key)[1]
                raw_rows.append(state_cache[key][int(frame)])
            raw=np.stack(raw_rows).astype(np.float32)
        else:
            raw=np.asarray(tables["candidate_state"],np.float32)
        tables["candidate_storage_state"]=raw
        tables["candidate_state"]=ex.state_storage_to_policy(raw).astype(np.float32)
    tmp=path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**tables)
    tmp.replace(path)


def _teacher(out, *, block=128):
    """Compute scale-shared DCT-unit teacher for stride-10 query action windows."""
    z=np.load(Path(out)/"cache/rows.npz",allow_pickle=False);pool=dict(np.load(Path(out)/"cache/pool.npz",allow_pickle=False))
    queries=np.asarray(z["memory_target"],np.float32)[:,:,5:11]
    qshape=np.stack([ex.encode_shape(np.concatenate([np.zeros((len(a),5),np.float32),a,np.zeros((len(a),1),np.float32)],axis=1))[0] for a in queries])
    err=ex.shape_error_matrix(qshape,pool["shape"],block=block)
    allowed=np.asarray(z["memory_episode"])[:,None]!=np.asarray(pool["global_episode"])[None,:]
    if not allowed.any(1).all():raise AssertionError("a memory query has no candidate after self-episode exclusion")
    if np.any(allowed & (np.asarray(z["memory_episode"])[:,None] == np.asarray(pool["global_episode"])[None,:])):
        raise AssertionError("cross-fit self-episode candidate exclusion failed")
    return err,allowed


def train(out=OUT, *, epochs=20, batch=128, device="cuda", feature="final"):
    import torch
    from torch import nn
    torch.set_num_threads(8)
    out=Path(out);_ensure_policy_state_cache(out);z0=np.load(out/"cache/rows.npz",allow_pickle=False)
    if np.intersect1d(np.asarray(z0["candidate_episode"]),np.asarray(z0["heldout_episode"])).size:
        raise AssertionError("held-out evaluation episodes leaked into Context memory candidates")
    if np.intersect1d(np.asarray(z0["candidate_episode"]),np.asarray(z0["heldout_episode"])).size:
        raise AssertionError("held-out evaluation episodes leaked into Context memory candidates")
    if not torch.cuda.is_available() and str(device).startswith("cuda"): device="cpu"
    if str(device).startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.20)
    candidate_state=_candidate_context_state(out)
    # New table is copied into the existing cache atomically; candidate state is small.
    tables={k:z0[k] for k in z0.files};tables["candidate_state"]=candidate_state
    np.savez_compressed(out/"cache/rows.npz",**tables)
    z0=np.load(out/"cache/rows.npz",allow_pickle=False)
    episodes=np.unique(z0["memory_episode"]);val_eps=set(np.random.default_rng(SEED).permutation(episodes)[:max(1,len(episodes)//5)].tolist())
    val=np.asarray([int(e) in val_eps for e in z0["memory_episode"]]);fit=np.flatnonzero(~val);vid=np.flatnonzero(val)
    # Feature normalization is fit on train episodes only; Dev episodes are
    # held out for early stopping and cannot influence its input scaling.
    z,x,k,norm=_context_x(out,fit_ids=fit,feature=feature)
    feature_file = "prefix_mean_fp16.npy" if feature == "final" else "prefix_input_mean_fp16.npy"
    cix=np.asarray(z["candidate_context"],np.int64); f=np.asarray(np.load(out/"cache"/feature_file,mmap_mode="r"),np.float32)
    k=np.concatenate([(f[cix]-norm["feature_mean"])/norm["feature_std"],(candidate_state-norm["state_mean"])/norm["state_std"]],axis=1).astype(np.float32)
    err,allowed=_teacher(out)
    gap=np.mean(np.where(allowed[fit],err[fit],0),axis=1)-np.min(np.where(allowed[fit],err[fit],np.inf),axis=1)
    tau=max(.1*float(np.median(gap)),1e-8)

    class ContextNet(nn.Module):
        def __init__(self,dim):
            super().__init__();self.query=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256));self.key=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,256));self.log_scale=nn.Parameter(torch.tensor(float(np.log(20.))));self.grip=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,6))
        def forward(self,q,keys):return nn.functional.normalize(self.query(q),dim=-1)@nn.functional.normalize(self.key(keys),dim=-1).T*self.log_scale.exp()
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED+1);net=ContextNet(x.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    xt=torch.as_tensor(x,device=device);kt=torch.as_tensor(k,device=device);et=torch.as_tensor(err,device=device);at=torch.as_tensor(allowed,device=device)
    # Gripper targets: next-5 sign and any switch relative to the previous command/state.
    target=np.asarray(z["memory_target"],np.float32)[:,:,11];state=np.asarray(z["memory_storage_state"],np.float32)
    # Current gripper opening is part of observed state; do not infer it from action history.
    ysign=(target[:,:5]>=0).astype(np.float32);prev=np.where(state[:,14]<.03,1.,-1.).astype(np.float32);yswitch=np.any((target[:,:5]>=0)!=(prev[:,None]>=0),axis=1).astype(np.float32)
    gy=torch.as_tensor(np.concatenate([ysign,yswitch[:,None]],axis=1),device=device)
    best=(float("inf"),None,0);curve=[];stale=0
    for epoch in range(1,int(epochs)+1):
        net.train();order=rng.permutation(fit);losses=[]
        for st in range(0,len(order),int(batch)):
            ix=order[st:st+int(batch)];bi=torch.as_tensor(ix,device=device)
            logits=net(xt[bi],kt).masked_fill(~at[bi],-1e9);teachers=torch.softmax((-et[bi]/tau).masked_fill(~at[bi],-1e9),dim=-1)
            loss=-(teachers*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
            grip_logits=net.grip(xt[bi]);loss=loss+.1*nn.functional.binary_cross_entropy_with_logits(grip_logits,gy[bi])
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        net.eval()
        with torch.no_grad():
            vl=torch.cat([net(xt[vid[s:s+batch]],kt).masked_fill(~at[vid[s:s+batch]],-1e9) for s in range(0,len(vid),batch)],0).cpu().numpy()
        pick=vl.argmax(1);vmetric=float(np.mean(err[vid,pick]))
        curve.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_top1_type_error":vmetric})
        _log(out,"train",epoch=epoch,train_loss=curve[-1]["train_loss"],val_top1_type_error=vmetric)
        if vmetric<best[0]:
            best=(vmetric,{kk:v.detach().cpu().clone() for kk,v in net.state_dict().items()},epoch);stale=0
        else:
            stale+=1
            if stale>=5:
                _log(out,"early_stop",epoch=epoch,best_epoch=best[2],best_val_top1_type_error=best[0],patience=5)
                break
    checkpoint={"state":best[1],"norm":norm,"input_dim":int(x.shape[1]),"key_dim":int(k.shape[1]),"tau":tau,"seed":SEED,"best_epoch":best[2],"val_top1_type_error":best[0],"val_episodes":sorted(val_eps),"curve":curve}
    model_file = "context_model.pt" if feature == "final" else "context_model_input.pt"
    torch.save(checkpoint,out/model_file)
    np.savez_compressed(out/f"cache/train_teacher_{feature}.npz",error=err,allowed=allowed)


def _p5_scores(z, qids, pool, qhist=None, *, device="cuda"):
    import torch
    from experiment import _features
    torch.set_num_threads(8)
    if not torch.cuda.is_available() and str(device).startswith("cuda"):device="cpu"
    hist=np.asarray(z["heldout_hist"])[qids] if qhist is None else np.asarray(qhist,np.float32)
    state=np.asarray(z["heldout_state"])[qids]
    models=[expert._load_net(SOURCE/"models"/f"P5_U_seed{i}.pt",pool,device=device) for i in (0,1,2)]
    scores=[]
    q={"hist":hist,"state":state}
    for net,ck in models:
        x=_features(q,ck["norm"])
        with torch.no_grad():scores.append(net(torch.as_tensor(x,device=device),torch.as_tensor(pool["key"],device=device)).cpu().numpy())
    return np.mean(scores,axis=0)


def evaluate(out=OUT, *, device="cuda", feature="final"):
    import torch
    out=Path(out);_require_current_pi05_noise_manifest(out);_ensure_policy_state_cache(out);z=np.load(out/"cache/rows.npz",allow_pickle=False);pool=dict(np.load(out/"cache/pool.npz",allow_pickle=False))
    feature_file = "prefix_mean_fp16.npy" if feature == "final" else "prefix_input_mean_fp16.npy"
    features=np.asarray(np.load(out/"cache"/feature_file,mmap_mode="r"),np.float32)
    model_file = "context_model.pt" if feature == "final" else "context_model_input.pt"
    with open(out/model_file,"rb") as f: ck=torch.load(f,map_location="cpu",weights_only=False)
    from torch import nn
    class ContextNet(nn.Module):
        def __init__(self,dim):super().__init__();self.query=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256));self.key=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,256));self.log_scale=nn.Parameter(torch.tensor(float(np.log(20.))));self.grip=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,6))
        def forward(self,q,keys):return nn.functional.normalize(self.query(q),dim=-1)@nn.functional.normalize(self.key(keys),dim=-1).T*self.log_scale.exp()
    device=device if torch.cuda.is_available() else "cpu";torch.set_num_threads(8)
    if device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.20)
    net=ContextNet(ck["input_dim"]);net.load_state_dict(ck["state"]);net.to(device).eval()
    mean=ck["norm"]["feature_mean"];std=ck["norm"]["feature_std"];sm=ck["norm"]["state_mean"];ss=ck["norm"]["state_std"]
    eid=np.asarray(z["heldout_context"],np.int64);state=np.asarray(z["heldout_state"],np.float32);qx=np.concatenate([(features[eid]-mean)/std,(state-sm)/ss],axis=1).astype(np.float32)
    cix=np.asarray(z["candidate_context"],np.int64);cs=np.asarray(z["candidate_state"],np.float32);kx=np.concatenate([(features[cix]-mean)/std,(cs-sm)/ss],axis=1).astype(np.float32)
    with torch.no_grad():
        context_score=torch.cat([net(torch.as_tensor(qx[s:s+128],device=device),torch.as_tensor(kx,device=device)) for s in range(0,len(qx),128)],0).cpu().numpy()
        grip_logits=torch.cat([net.grip(torch.as_tensor(qx[s:s+512],device=device)) for s in range(0,len(qx),512)],0).cpu().numpy()
    errors=[];targets=np.asarray(z["heldout_target"],np.float32);qshape=np.stack([ex.encode_shape(np.concatenate([np.zeros((10,5),np.float32),a[:,5:11],np.zeros((10,1),np.float32)],axis=1))[0] for a in targets]);err=ex.shape_error_matrix(qshape,pool["shape"])
    allowed=np.asarray(z["heldout_episode"])[:,None]!=np.asarray(pool["global_episode"])[None,:];masked=np.where(allowed,context_score,-np.inf);pick=masked.argmax(1)
    # P5 was evaluated by the parallel baseline job; reuse the persisted per-seed
    # picks and require exact row alignment instead of rerunning its GPU inference.
    pb=np.load(Path(out)/"p5_baselines/masked-input/p5_baseline_rows.npz",allow_pickle=False)
    for key,local in (("task","heldout_task"),("episode","heldout_local_episode"),("frame","heldout_frame")):
        if not np.array_equal(np.asarray(pb[key]),np.asarray(z[local])):
            raise AssertionError(f"P5 baseline row alignment differs at {key}")
    p5_pick=np.asarray(pb["pick_correct"],np.int64);zero_pick=np.asarray(pb["pick_none"],np.int64)
    positive=np.zeros_like(err,bool)
    for i in range(len(err)):
        ids=np.flatnonzero(allowed[i]);best=ids[np.argsort(err[i,ids],kind="stable")[:max(1,int(np.ceil(.01*len(ids))))]];positive[i,best]=True
    ref=np.asarray(z["eval_pi05_actions"],np.float32)[:,:,:12]
    cand_actions=[]
    offsets=np.asarray(pool["action_offsets"]);flat=np.asarray(pool["actions_flat"])
    def make_drafts(sel):
        if np.asarray(sel).ndim==1:
            picks=np.asarray(sel)[None,:]
        else:picks=np.asarray(sel)
        out_rows=[]
        for seed_picks in picks:
            rows=[]
            for i,k in enumerate(seed_picks):
                a=flat[offsets[k]:offsets[k+1]].copy()[:50];a=ex._scale_to_ref(a,ref[i,:len(a)])
                if len(a)==0:raise AssertionError("empty memory unit cannot be used as a draft")
                if len(a)<50:a=np.concatenate([a,np.repeat(a[-1:],50-len(a),axis=0)],axis=0)
                rows.append(a)
            out_rows.append(rows)
        return np.asarray(out_rows,np.float32)
    def make_dct_drafts(sel):
        picks=np.asarray(sel);picks=picks[None,:] if picks.ndim==1 else picks
        out_rows=[]
        for seed_picks in picks:
            rows_dct=[]
            for i,k in enumerate(seed_picks):
                raw=flat[offsets[k]:offsets[k+1]].astype(np.float32)
                length=len(raw)
                # pool['shape'] is the inverse-DCT-3 canonical shape; stretch
                # it back to the stored segment length, then take its first 10 commands.
                decoded=ex.resample_rows(pool["shape"][k],length)
                source=ex.resample_rows(raw,length)
                if length<10:
                    pad=10-length;decoded=np.concatenate([decoded,np.repeat(decoded[-1:],pad,axis=0)],axis=0)
                    source=np.concatenate([source,np.repeat(source[-1:],pad,axis=0)],axis=0)
                draft=source[:10].copy();draft[:,5:11]=decoded[:10]
                draft=ex._scale_to_ref(draft,ref[i,:10])
                rows_dct.append(draft)
            out_rows.append(rows_dct)
        return np.asarray(out_rows,np.float32)
    cand_actions=[make_drafts(pick),make_drafts(p5_pick),make_drafts(zero_pick)]
    cand_dct_actions=[make_dct_drafts(pick),make_dct_drafts(p5_pick),make_dct_drafts(zero_pick)]
    # Convert retrieved-unit grip majority into a five-step prediction. Context head predicts sign independently.
    unit_grip=[]
    for k in pick:
        a=flat[offsets[k]:offsets[k+1]];unit_grip.append(1 if np.mean(a[:,11]>=0)>=.5 else -1)
    ctx_sign=np.where(grip_logits[:,:5]>=0,1,-1);truth=np.where(targets[:,:5,11]>=0,1,-1)
    def type_metrics(sel, draft_rows):
        picks=np.asarray(sel);picks=picks[None,:] if picks.ndim==1 else picks
        chosen=np.stack([err[np.arange(len(err)),row] for row in picks])
        task_match=np.stack([np.asarray(z["heldout_task"])==np.asarray(pool["task"])[row] for row in picks])
        mse_by_seed=[np.mean([np.mean((draft_rows[s][i,:5,5:11]-targets[i,:5,5:11])**2) for i in range(len(targets))]) for s in range(len(picks))]
        return {"top1_type_error_mean":float(chosen.mean()),"top1_type_error_median":float(np.median(chosen)),"seed_mean":chosen.mean(axis=1).tolist(),"task_match":float(task_match.mean()),"first5_eef_mse":float(np.mean(mse_by_seed)),"first5_eef_mse_by_seed":mse_by_seed}
    results={"rows":int(len(targets)),"episodes":int(len(np.unique(z["heldout_episode"]))),"context":type_metrics(pick,cand_actions[0]),"P5_correct_history":type_metrics(p5_pick,cand_actions[1]),"P5_zero_history":type_metrics(zero_pick,cand_actions[2]),"random":float(np.mean([np.mean(err[i,np.random.default_rng(SEED+i).choice(np.flatnonzero(allowed[i]))]) for i in range(len(err))])),"oracle_type_error":float(np.mean(np.min(np.where(allowed,err,np.inf),axis=1))),"R_at_5_oracle_best_1pct":float(np.mean(np.any(positive[np.arange(len(err))[:,None],np.argsort(-masked,axis=1)[:,:5]],axis=1))),"frame0":{}}
    tau_refs={}
    for tau_name in ("0.04","0.08"):
        tp=dict(np.load(out/f"p5_baselines/pools/U_tau_{tau_name}.npz",allow_pickle=False))
        if len(tp["shape"])==1:
            err_tau=ex.shape_error_matrix(qshape,tp["shape"])
            representative=int(tp["representative_unit"][0])
            unit_episode=int(pool["global_episode"][representative])
            if unit_episode in set(map(int,z["heldout_episode"])):raise AssertionError("tau representative leaked from held-out episode")
            tau_draft=make_drafts(np.full(len(targets),representative,np.int32))[0]
            tau_refs[tau_name]={"units":1,"top1_type_error_mean":float(err_tau[:,0].mean()),"frame0_type_error_mean":float(err_tau[np.asarray(z["heldout_frame"])==0,0].mean()),"first5_eef_mse":float(np.mean((tau_draft[:,:5,5:11]-targets[:,:5,5:11])**2)),"representative_unit":representative}
    results["tau_singleton_references"]=tau_refs
    results["draft_reconstruction"]={
        name:{"raw_first5_eef_mse":float(val["first5_eef_mse"]),"dct_first5_eef_mse":type_metrics(sel,dctdraft)["first5_eef_mse"]}
        for name,val,sel,dctdraft in (("context",results["context"],pick,cand_dct_actions[0]),("P5_correct_history",results["P5_correct_history"],p5_pick,cand_dct_actions[1]),("P5_zero_history",results["P5_zero_history"],zero_pick,cand_dct_actions[2]))
    }
    frame0=np.asarray(z["heldout_frame"])==0
    for name,sel in (("context",pick),("P5_correct_history",p5_pick),("P5_zero_history",zero_pick)):
        picks=np.asarray(sel);picks=picks[None,:] if picks.ndim==1 else picks
        results["frame0"][name]={"rows":int(frame0.sum()),"top1_type_error_mean":float(np.mean([err[frame0,row[frame0]].mean() for row in picks])),"seed_mean":[float(err[frame0,row[frame0]].mean()) for row in picks]}
    # Paired episode-cluster bootstrap for the two predeclared gate comparisons.
    ctx_error=err[np.arange(len(err)),pick]
    pc_errors=np.stack([err[np.arange(len(err)),row] for row in p5_pick])
    pz_errors=np.stack([err[np.arange(len(err)),row] for row in zero_pick])
    episode_ids=np.asarray(z["heldout_episode"]);unique_eps=np.unique(episode_ids);ep_rows={int(e):np.flatnonzero(episode_ids==e) for e in unique_eps}
    rng=np.random.default_rng(SEED+700);draws=5000;boot={"context_minus_P5_correct_history":[],"context_minus_P5_zero_history":[]}
    for _ in range(draws):
        sampled=rng.choice(unique_eps,size=len(unique_eps),replace=True);ix=np.concatenate([ep_rows[int(e)] for e in sampled])
        boot["context_minus_P5_correct_history"].append(float(np.mean(ctx_error[ix]-pc_errors[:,ix].mean(0))))
        boot["context_minus_P5_zero_history"].append(float(np.mean(ctx_error[ix]-pz_errors[:,ix].mean(0))))
    results["paired_episode_bootstrap_95"]={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in boot.items()}
    # Gripper first-five MSE against the three task-independent outputs.
    def grip_mse(pred):
        pred=np.asarray(pred);pred=np.repeat(pred[:,None],5,axis=1) if pred.ndim==1 else pred
        return float(np.mean((pred-targets[:,:5,11])**2))
    switch_truth=[]
    for i in range(len(targets)):
        prev=1. if float(np.asarray(z["heldout_storage_state"])[i,14])<.03 else -1.
        switch_truth.append(bool(np.any((targets[i,:5,11]>=0)!=(prev>=0))))
    g1_sign=np.where(ref[:,:5,11]>=0,1,-1)
    results["gripper"]={"retrieved_unit_majority_mse5":grip_mse(np.asarray(unit_grip)),"context_head_mse5":grip_mse(ctx_sign),"G1_pi05_first5_mse":float(np.mean((ref[:,:5,11]-targets[:,:5,11])**2)),"G1_pi05_sign_mse5":float(np.mean((g1_sign-truth)**2)),"retrieved_unit_sign_error":float(np.mean(np.asarray(unit_grip)[:,None]!=truth)),"context_head_sign_error":float(np.mean(ctx_sign!=truth)),"G1_pi05_sign_error":float(np.mean(g1_sign!=truth)),"context_switch_bce_accuracy":float(np.mean((grip_logits[:,5]>=0)==np.asarray(switch_truth)))}
    # Full gripper comparison reuses the existing memory-side G4 classifier.
    from gripper import _apply_g4, _load_classifier, _metrics as grip_metrics
    g4_root=Path("/work/.skillnet-state/outputs/robocasa-memory/gripper-01/production")
    g4_ck,g4_models=_load_classifier(g4_root/"g4_classifier.pt")
    g4_threshold=json.loads((g4_root/"selection.json").read_text())["selected_threshold"]
    g4_rows=[]
    for i in range(len(targets)):
        g4_rows.append(_apply_g4(np.asarray(z["heldout_hist"])[i],np.asarray(z["heldout_storage_state"])[i],cand_actions[0][0][i],g4_models,g4_ck,g4_threshold))
        if (i+1)%300==0:_log(out,"evaluate_gripper",completed=i+1,total=len(targets))
    g4_actions=np.stack(g4_rows)
    candidates={"unit_majority":np.repeat(np.asarray(unit_grip)[:,None],5,axis=1),"Context_head":ctx_sign,"G1_pi05":g1_sign,"G4":np.where(g4_actions[:,:5]>=0,1,-1)}
    detailed={}
    for name,commands in candidates.items():
        metr=[]
        for i in range(len(targets)):
            initial=1. if float(np.asarray(z["heldout_storage_state"])[i,14])<.03 else -1.
            metr.append(grip_metrics(commands[i],targets[i,:5,11],initial))
        bearing=np.asarray([m["switch_count_true"]>0 for m in metr],bool)
        detail={"windows_with_true_switch":int(bearing.sum()),"first5_mse":float(np.mean([m["mse5"] for m in metr])),
                "first5_mse_switch_bearing":float(np.mean([metr[i]["mse5"] for i in np.flatnonzero(bearing)])) if bearing.any() else None,
                "missed_switches":int(sum(m["missed"] for m in metr)),"false_switches":int(sum(m["false"] for m in metr)),
                "mean_timing_error_steps":float(np.mean([x for m in metr for x in m["timing_abs_errors"]])) if any(m["timing_abs_errors"] for m in metr) else None}
        detailed[name]=detail
    results["gripper_detailed_5step"]=detailed
    results["context_selection_rows"]={"index":pick.astype(np.int32),"errors":err[np.arange(len(err)),pick],"scores_top5":np.argsort(-masked,axis=1)[:,:5].astype(np.int32),"task":np.asarray(z["heldout_task"]),"episode":np.asarray(z["heldout_episode"]),"frame":np.asarray(z["heldout_frame"])}
    eval_file = "evaluation.json" if feature == "final" else "evaluation_input.json"
    (out/eval_file).write_text(json.dumps(results,indent=2,ensure_ascii=False,default=ex._json_default)+"\n")
    _log(out,"evaluate",rows=len(targets),context_top1=results["context"]["top1_type_error_mean"],p5_good=results["P5_correct_history"]["top1_type_error_mean"],p5_zero=results["P5_zero_history"]["top1_type_error_mean"])


def report(out=OUT):
    out=Path(out);_require_current_pi05_noise_manifest(out);r=json.loads((out/"evaluation.json").read_text());
    with (out/"report.md").open("w") as f:
        f.write("実装: 完了\n\n# RM-CTX offline report\n\n")
        f.write("一次判定はstride-10保留窓で、行動履歴を入力しないContext Keyが既存P5を置き換えられるかを比較した。閉ループは実行していない。保留120デモの行動は採点のみ。\n\n")
        p5=json.loads((out/"p5_baselines/masked-input/results.json").read_text())
        f.write("## 比較条件（既存P5ベースラインを再利用）\n\n|条件|履歴の処理|top-1 S-shared型誤差|タスク一致率|\n|---|---|---:|---:|\n")
        labels={"correct":"正しい履歴","none":"履歴なし（0埋め）","shift_-20":"同一デモ −20 frame","shift_-10":"同一デモ −10 frame","shift_+10":"同一デモ +10 frame","shift_+20":"同一デモ +20 frame","replace":"別エピソード履歴"}
        for key,label in labels.items():
            d=p5["conditions"][key];f.write(f"|P5|{label}|{d['top1_type_error_mean']:.6f}|{d['task_match_rate']:.3f}|\n")
        f.write("|Context|画像・指示・state16、履歴なし|—|—|\n\n")
        ps=json.loads((out/"p5_baselines/pool_summary.json").read_text())
        f.write(f"Uメモリは{ps['base_units']}区間。τ=.04/.08 の統合プールはいずれも {ps['tau']['0.04']['cluster_count']}単位となり、全体を一群に併合するため検索比較として退化する。Uのグリッパーは+側割合 {ps['gripper']['grip_plus_fraction']:.3f}、終端切替{ps['gripper']['terminal_switch_units']}単位、内部切替{ps['gripper']['units_with_internal_switch']}単位（総件数{ps['gripper']['internal_switch_count_total']}）。\n\n")
        f.write("## 第1段：U・最終層平均・seed 1・生行動下書き\n\n|条件|窓数|top-1 S-shared型誤差|frame-0誤差|タスク一致|先頭5 EEF MSE|\n|---|---:|---:|---:|---:|---:|\n")
        for name in ("context","P5_correct_history","P5_zero_history"):
            d=r[name];f0=r["frame0"][name];f.write(f"|{name}|{r['rows']}|{d['top1_type_error_mean']:.6f}|{f0['top1_type_error_mean']:.6f}|{d['task_match']:.3f}|{d['first5_eef_mse']:.5f}|\n")
        f.write(f"|random|{r['rows']}|{r['random']:.6f}|—|—|—|\n|oracle ceiling|{r['rows']}|{r['oracle_type_error']:.6f}|—|—|—|\n\n")
        f.write(f"Context − P5 correct-history episode bootstrap 95% CI (5,000 draws): `{r['paired_episode_bootstrap_95']['context_minus_P5_correct_history']}`; Context − P5 no-history: `{r['paired_episode_bootstrap_95']['context_minus_P5_zero_history']}`. Context R@5 against oracle top-1%: {r['R_at_5_oracle_best_1pct']:.3f}.\n\n")
        f.write("## τ統合プールの退化参照\n\n|τ|単位数|型誤差全体|frame-0|raw代表の先頭5 EEF MSE|\n|---:|---:|---:|---:|---:|\n")
        for tau_name,d in r["tau_singleton_references"].items():
            f.write(f"|{tau_name}|{d['units']}|{d['top1_type_error_mean']:.6f}|{d['frame0_type_error_mean']:.6f}|{d['first5_eef_mse']:.6f}|\n")
        f.write("\n単位が1個なので、これは候補選択の比較ではなく統合後の固定代表参照。\n\n")
        import torch
        torch.set_num_threads(8)
        ck=torch.load(out/"context_model.pt",map_location="cpu",weights_only=False)
        f.write(f"Memory-side episode validation: {len(ck['val_episodes'])} episodes; best epoch {ck['best_epoch']}/{len(ck['curve'])}; top-1 type error {ck['val_top1_type_error']:.6f}; listwise temperature {ck['tau']:.6g}.\n\n")
        f.write("## 下書き復元（同じ選択候補）\n\n|選択器|生行動列の先頭5 EEF MSE|逆DCT-3標準形復元の先頭5 EEF MSE|\n|---|---:|---:|\n")
        for name in ("context","P5_correct_history","P5_zero_history"):
            d=r["draft_reconstruction"][name];f.write(f"|{name}|{d['raw_first5_eef_mse']:.6f}|{d['dct_first5_eef_mse']:.6f}|\n")
        f.write("\n")
        f.write(f"R@5 (oracle top 1%): {r['R_at_5_oracle_best_1pct']:.3f}.\n\n")
        f.write("### Gripper (先頭5ステップの連続指令MSE)\n\n|取得単位の多数決|Context分類頭(±1出力)|G1 π0.5連続出力|符号誤り率: 単位 / Context / G1|切替判定正解率|\n|---:|---:|---:|---:|---:|\n")
        g=r['gripper'];f.write(f"|{g['retrieved_unit_majority_mse5']:.4f}|{g['context_head_mse5']:.4f}|{g['G1_pi05_first5_mse']:.4f}|{g['retrieved_unit_sign_error']:.3f} / {g['context_head_sign_error']:.3f} / {g['G1_pi05_sign_error']:.3f}|{g['context_switch_bce_accuracy']:.3f}|\n\n")
        f.write("|手法|5-stepで切替を含む窓|符号出力MSE 全体 / 切替窓|見逃し|誤切替|検出切替の平均時刻誤差(step)|\n|---|---:|---:|---:|---:|---:|\n")
        for name,d in r["gripper_detailed_5step"].items():
            switched="—" if d["first5_mse_switch_bearing"] is None else f"{d['first5_mse_switch_bearing']:.4f}"
            timing="—" if d["mean_timing_error_steps"] is None else f"{d['mean_timing_error_steps']:.3f}"
            f.write(f"|{name}|{d['windows_with_true_switch']}|{d['first5_mse']:.4f} / {switched}|{d['missed_switches']}|{d['false_switches']}|{timing}|\n")
        f.write("\n")
        f.write("## Interpretation\n\n")
        p5seeds=p5["conditions"]["correct"]["seed_mean"];ctxv=r["context"]["top1_type_error_mean"]
        same_level=ctxv <= max(p5seeds)+.1*(max(p5seeds)-min(p5seeds))
        beats_empty=r["paired_episode_bootstrap_95"]["context_minus_P5_zero_history"][1] < 0
        gate="一次基準を満たす" if same_level and beats_empty else "一次基準を満たさない"
        f.write(f"事前判定: {gate}（Context が正しい履歴 P5 の seed 上端＋seed幅10%以内: {same_level}; 履歴なし P5 との差CI上端<0: {beats_empty}）。P5 の既存3 seed幅を使用した。タスク一致だけの改善は通過と見なさない。これはオフライン検索の結果であり、閉ループの成功率を示さない。\n")
        if (out/"evaluation_input.json").exists():
            ri=json.loads((out/"evaluation_input.json").read_text())
            f.write("\n## 追加表現：transformer前のprefix embedding平均（第2段の中間特徴要求に対する近似対照）\n\n|条件|top-1 S-shared型誤差|frame-0誤差|タスク一致|先頭5 EEF MSE|R@5 oracle top-1%|\n|---|---:|---:|---:|---:|---:|\n")
            d=ri["context"];f0=ri["frame0"]["context"]
            f.write(f"|Context input-embedding|{d['top1_type_error_mean']:.6f}|{f0['top1_type_error_mean']:.6f}|{d['task_match']:.3f}|{d['first5_eef_mse']:.6f}|{ri['R_at_5_oracle_best_1pct']:.3f}|\n\n")
            cki=torch.load(out/"context_model_input.pt",map_location="cpu",weights_only=False)
            f.write(f"Memory-side episode validation for this view: {len(cki['val_episodes'])} episodes; best epoch {cki['best_epoch']}/{len(cki['curve'])}; top-1 type error {cki['val_top1_type_error']:.6f}; listwise temperature {cki['tau']:.6g}.\n\n")
            f.write("この追加表現は transformer の中間層ではなく、PaliGemma に入力する prefix embedding の平均である。指定された中間transformer層の比較は未実施として区別する。\n")
    _log(out,"report",path=str(out/"report.md"))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("extract","train","evaluate","report","all"));p.add_argument("--out",type=Path,default=OUT);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--epochs",type=int,default=20);p.add_argument("--device",default="cuda");p.add_argument("--feature",choices=("final","input"),default="final");a=p.parse_args()
    if a.stage in ("extract","all"):extract(a.out,batch_size=a.batch_size)
    if a.stage in ("train","all"):train(a.out,epochs=a.epochs,device=a.device,feature=a.feature)
    if a.stage in ("evaluate","all"):evaluate(a.out,device=a.device,feature=a.feature)
    if a.stage in ("report","all"):report(a.out)

if __name__=="__main__":main()
