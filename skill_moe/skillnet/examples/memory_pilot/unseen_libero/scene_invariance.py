#!/usr/bin/env python3
"""Scene-invariance retrieval probes for unseen LIBERO.

All image features come from the orientation-corrected, frozen pi0.5 caches.
The only learned output is a query embedding; memory keys are scale-only motion
types (W pool), and all action-derived labels are confined to LIBERO-40.

Stages: prepare, tune, train, cv, evaluate, report, all.  Artifacts are
resumable and each stage refuses to silently replace a completed artifact.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROWS

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
ROOT = STATE / "outputs/memory-pilot/unseen-libero"
SOURCE = ROOT / "scale-only-02/production"
RUN = Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01")
TOKENS = RUN / "tokens.npy"
GROUPS = RUN / "groups.npy"
MID = ROOT / "key-query-01/policy_features"
OUT = ROOT / "scene-invariance-01"
HOLDOUT = (2, 7, 12, 17, 22, 27, 32, 37)
SEEDS = (4101, 4102, 4103)
LAYERS = ("final", "mid16", "mid17")
KS = (1, 4, 8)
TAUS = (.01, .02, .05)
CONDITIONS = ("P5-L0", "A-L0", "B-L0", "A-L1", "B-L1")
BASE_REV = "2fc3e6e11fad0473d9bfe4e8d4e377b69b32686c"


def _json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    tmp.replace(p)


def _json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return str(x)


def _atomic_npz(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    tmp.replace(p)


def _log(out, **rec):
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with (Path(out) / "progress.jsonl").open("a") as f: f.write(line + "\n")


def _run_manifest(out):
    p=Path(out)/"run_manifest.json"
    if p.exists():return json.loads(p.read_text())
    try: rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE.parents[5],text=True).strip()
    except Exception: rev=BASE_REV
    item={"status":"prepared-for-run","base_revision":BASE_REV,"worktree_revision_at_start":rev,
        "branch":"feat/query-scene-invariance","seeds":list(SEEDS),"conditions":list(CONDITIONS),
        "candidate_set":"scale-only W 1,595; LIBERO-40 holdout tasks excluded from candidate memory and self-task candidates excluded per query",
        "selection":"designated LIBERO-40 tasks [2,7,12,17,22,27,32,37] only; min mean top-1 S-shared type error",
        "teacher_temperature_grid":list(TAUS),"B_latent_counts":list(KS),"B_layers":list(LAYERS),
        "contrastive_weight":.20,"contrastive_temperature":.05,"contrastive_positive_threshold":"holdout cross-scene/cross-task S-shared type-error q25",
        "training_epochs":{"tuning":3,"final_fit":8,"scene_cv":2},"gpu_fraction_cap":.40,
        "gpu_wall_clock_cap_hours":30,"orientation":"orientation-corrected token cache; no re-extraction",
        "pair_error":"directional scale-only projection; ||q_i||^2 - <q_i,u_j>^2/||u_j||^2",
        "unseen_eval_targets_read_during_prepare_tune_train_cv":False}
    _json(p,item);return item


def _modules():
    import key_query_probe as kq
    import theta_regression as tr
    import variable_shape_memory as vs
    import scale_only_query as sq
    return kq, tr, vs, sq


def _task_names():
    p = RUN / "data_manifest.json"
    return {int(k): str(v) for k, v in json.loads(p.read_text())["tasks"].items()}


def _libero10_scene_map():
    """Read the vendored public task map without importing simulator code."""
    p=HERE.parents[2]/"third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
    tree=ast.parse(p.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="libero_task_map" for t in n.targets))
    mapping=ast.literal_eval(node.value)["libero_10"]
    out={}
    for item in mapping:
        m=re.match(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_(.*)",item)
        if not m: raise ValueError(f"invalid LIBERO-10 task map entry: {item}")
        instruction=m.group(2).replace("_"," ").lower()
        out[instruction]=m.group(1)
    return out


def scene_ids(task_ids, task_names=None):
    """Scene identity: explicit LIBERO-10 scene XML; legacy suites by suite scene.

    In the 30 legacy task slots, each named suite uses its benchmark's shared
    scene family; LIBERO-10's ten XML scenes remain distinct. This mapping is
    stable and deliberately independent of actions or outcomes.
    """
    names = task_names or _task_names(); scene_map=_libero10_scene_map()
    out = []
    for t in np.asarray(task_ids, np.int64):
        if t < 10:
            name = names[int(t)].strip().lower()
            scene=scene_map.get(name)
            if scene is None: raise ValueError(f"no LIBERO-10 scene mapping for task {t}: {name}")
            out.append("libero10:"+scene)
        elif t < 20: out.append("libero_goal")
        elif t < 30: out.append("libero_object")
        elif t < 40: out.append("libero_spatial")
        else: raise ValueError(f"task outside LIBERO-40: {t}")
    return np.asarray(out, dtype="U32")


def scene_group_ids(task_ids, task_names=None):
    """Four preregistered folds: the three legacy suites and all LIBERO-10 scenes."""
    s=scene_ids(task_ids,task_names)
    return np.asarray(["libero10_all" if x.startswith("libero10:") else str(x) for x in s],dtype="U32")


def unseen_scene_ids(task_ids):
    """Resolve LIBERO-90 held-out tasks from the fixed unseen-row manifest."""
    manifest=UNSEEN_ROWS/"manifest.json"
    records=json.loads(manifest.read_text())["source_records"]
    names={int(r["task_id"]):str(r["task_name"]) for r in records}
    out=[]
    for t in np.asarray(task_ids,np.int64):
        name=names.get(int(t))
        if name is None:raise ValueError(f"unseen task {int(t)} missing from row manifest")
        m=re.match(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_",name)
        if m is None:raise ValueError(f"no LIBERO-90 scene prefix for task {int(t)}: {name}")
        out.append("libero90:"+m.group(1))
    return np.asarray(out,dtype="U40")


def _load_source():
    kq, tr, vs, sq = _modules()
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    train_rows = np.asarray(splits["train"][1], np.int64)
    cache = np.load(SOURCE / "prep/W_inputs_train.npz", allow_pickle=False)
    errors = np.load(SOURCE / "prep/W_err_S-shared_train.npz", allow_pickle=False)["err"]
    pool0 = tr.candidate_shape_pool(rep="C_vm_none")
    keep = ~np.isin(np.asarray(pool0["source_task"], np.int64), HOLDOUT)
    pool = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v)
            for k, v in pool0.items()}
    if len(errors) != len(train_rows) or errors.shape[1] != len(pool["shape"]):
        raise ValueError(f"fixed W teacher mismatch: {errors.shape}, rows={len(train_rows)}, pool={len(pool['shape'])}")
    for key in ("task", "episode", "frame"):
        if not np.array_equal(np.asarray(cache[key]), np.asarray(rows[key][train_rows])):
            raise ValueError(f"W cache row order mismatch: {key}")
    if not TOKENS.exists() or not GROUPS.exists(): raise FileNotFoundError("full final-token cache is required")
    return kq, tr, vs, sq, base, rows, unseen, train_rows, cache, errors, pool


def prepare(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "prepared.json"
    if meta_path.exists(): return json.loads(meta_path.read_text())
    kq, tr, vs, sq, base, rows, unseen, row_ids, d, err, pool = _load_source()
    task = np.asarray(d["task"], np.int64); ep = np.asarray(d["episode"], np.int64)
    scene = scene_ids(task)
    # LIBERO-40 fit and calibration subsets are separated by task, never by Eval.
    fit = ~np.isin(task, HOLDOUT)
    select = np.isin(task, HOLDOUT)
    if not fit.any() or not select.any(): raise ValueError("empty LIBERO-40 fit/holdout split")
    if np.any(np.isin(np.asarray(pool["source_task"]), HOLDOUT)):
        raise AssertionError("holdout-task memory leaked into W pool")
    if np.asarray(rows["actions"]).shape[0] <= np.max(row_ids): raise ValueError("row id mismatch")
    # Scene-level positive threshold from holdout-task rows only.  Sample at most
    # 48 rows/task; these labels tune the auxiliary objective, never Eval.
    rng = np.random.default_rng(4101); ix = []
    for t in HOLDOUT:
        q = np.flatnonzero(task == t)
        if len(q): ix.extend(rng.choice(q, min(48, len(q)), replace=False).tolist())
    target_phys = np.asarray(vs.to_physical(np.asarray(rows["actions"][row_ids, :, :6], np.float32)), np.float32)
    # Contrastive positive scale comes only from LIBERO-40 holdout query pairs.
    if len(ix) >= 2:
        sx=np.asarray(ix,np.int64); pair=_pair_err(target_phys[sx]); it=task[sx]; isc=scene[sx]
        eligible=(it[:,None]!=it[None,:])&(isc[:,None]!=isc[None,:])
        vals=pair[eligible]; tau_pos=float(np.quantile(vals,.25)) if len(vals) else .02
    else: tau_pos=.02
    # Save row-local metadata; target actions remain in LIBERO-40 and are used
    # for training only. Eval actions are not read here.
    _atomic_npz(out / "fit_rows.npz", row_id=row_ids, task=task, episode=ep, frame=np.asarray(d["frame"]),
        scene=scene, fit=fit, select=select, target=target_phys, error=err.astype(np.float32),
        tau_pos=np.asarray(tau_pos,np.float32),
        allowed=(np.asarray(pool["source_task"])[None, :] != task[:, None]),
        pool_shape=np.asarray(pool["shape"], np.float32), pool_task=np.asarray(pool["source_task"], np.int64),
        history=np.asarray(d["hist"], np.float32), state=np.asarray(d["state"], np.float32))
    meta = {"status":"prepared","source_run":str(RUN),"source_scale_run":str(SOURCE),
        "train_rows":len(row_ids),"fit_rows":int(fit.sum()),"selection_rows":int(select.sum()),
        "holdout_tasks":list(HOLDOUT),"candidate_count":len(pool["shape"]),
        "scenes":{s:int(np.sum(scene==s)) for s in sorted(set(scene.tolist()))},
        "scene_definition":"LIBERO-10 explicit XML scene; each legacy LIBERO goal/object/spatial suite as its shared scene family",
        "token_cache":str(TOKENS),"token_shape":list(np.load(TOKENS,mmap_mode="r").shape),
        "orientation":"unseen-rows-02 and run-01; corrected 180 degree flip; no new extraction",
        "target_physical_axes":"scale-only-query S-shared physical conversion, action first six axes",
        "target_label_used_for":"LIBERO-40 training and holdout selection only",
        "contrastive_tau_pos_holdout_q25":tau_pos,
        "eval_actions_loaded":False,"base_revision":BASE_REV}
    _json(meta_path, meta); _log(out, stage="prepare", **meta)
    return meta


def _make_model(kind, k, token_dim, d=256):
    import torch
    from torch import nn
    import torch.nn.functional as F
    class Cross(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
            self.norm1 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d,d))
            self.norm2 = nn.LayerNorm(d)
        def forward(self, q, tokens, pad):
            a,w = self.attn(q, tokens, tokens, key_padding_mask=pad, need_weights=True, average_attn_weights=False)
            q = self.norm1(q+a); q = self.norm2(q+self.ff(q))
            return q,w
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.kind=kind; self.k=int(k)
            self.key=nn.Sequential(nn.Linear(70,512),nn.GELU(),nn.Linear(512,d))
            self.context=nn.Sequential(nn.Linear(78,512),nn.GELU(),nn.Linear(512,d))
            self.log_scale=nn.Parameter(torch.tensor(np.log(10.),dtype=torch.float32))
            if kind in ("A","B"):
                self.token_in=nn.Sequential(nn.Linear(token_dim,d),nn.LayerNorm(d))
                self.cross=nn.ModuleList([Cross(),Cross()])
                n=1 if kind=="A" else int(k)
                self.slot=nn.Parameter(torch.randn(1,n,d)*.02) if kind=="B" else None
                self.qproj=nn.Linear(d,n*d) if kind=="A" else nn.Identity()
                self.out=nn.Sequential(nn.Linear(d*2,512),nn.GELU(),nn.Linear(512,d))
        def forward(self, hist, state, tokens=None, token_mask=None):
            ctx=self.context(torch.cat([hist.flatten(1),state],-1))
            attn=None
            if self.kind=="P5": q=ctx
            else:
                kv=self.token_in(tokens)
                if self.kind=="A": queries=self.qproj(ctx).view(len(ctx),1,-1)
                else: queries=ctx[:,None,:]+self.slot.expand(len(ctx),-1,-1)
                for layer in self.cross: queries,attn=layer(queries,kv,~token_mask.bool())
                q=self.out(torch.cat([ctx,queries.mean(1)],-1))
            keys=F.normalize(self.key(torch.cat([torch.zeros((1,10,1),device=hist.device),torch.zeros((1,10,6),device=hist.device)],-1).flatten(1)),dim=-1) if False else None
            return q,attn
    return Net()


def _tokens_for_rows(row_ids, source, layer="final"):
    """Get final token cache or an existing intermediate V cache; return compact token mask."""
    if layer == "final":
        vals=np.load(TOKENS,mmap_mode="r"); groups=np.load(GROUPS,mmap_mode="r")
        ii=np.flatnonzero(np.asarray(groups[0])>=0)
        return np.asarray(vals[np.asarray(row_ids,np.int64)][:,ii],np.float32), np.ones((len(row_ids),len(ii)),bool), np.asarray(groups[0,ii])
    layer_id=int(layer[3:]); layer_ids=(6,10,14,15,16,17); li=layer_ids.index(layer_id)
    prefix=source
    vals=np.load(MID/f'{prefix}_mid_v.npy',mmap_mode='r'); mask=np.load(MID/f'{prefix}_mid_mask.npy',mmap_mode='r')
    tok=np.asarray(vals[:,li],np.float32); m=np.asarray(mask,bool)
    groups=np.full((tok.shape[1],),2,np.int8);groups[:64]=0;groups[64:128]=1
    # Rows 128 onward are text; dynamic padding is masked by the stored mask.
    return tok,m,groups


def _allowed(task, pool_task):
    a=np.asarray(pool_task)[None,:] != np.asarray(task)[:,None]
    if np.any(a.sum(1)==0): raise ValueError("query has no cross-task memory candidates")
    return a


def _soft_teacher(err, allowed, tau):
    z=np.where(allowed,-np.asarray(err,np.float64)/float(tau),-np.inf)
    z-=np.max(z,axis=1,keepdims=True); p=np.exp(z);p[~allowed]=0
    return (p/np.maximum(p.sum(1,keepdims=True),1e-30)).astype(np.float32)


def _pair_err(x):
    x=np.asarray(x,np.float32).reshape(len(x),-1); den=np.maximum((x*x).sum(1),1e-10)
    dot=np.maximum(x@x.T,0)
    # S-shared error is directional: query i is projected onto candidate j.
    # The residual norm is ||q_i||² - <q_i,u_j>²/||u_j||².
    return np.maximum((den[:,None]-dot*dot/den[None,:])/x.shape[1],0)


def _tau_pos_for_rows(data, row_ids, *, seed=4101, per_task=48):
    """Choose L1's positive threshold from the supplied split only."""
    ids=np.asarray(row_ids,np.int64);tasks=np.asarray(data["task"]);scenes=np.asarray(data["scene"])
    rng=np.random.default_rng(seed);picked=[]
    for task in sorted(set(tasks[ids].tolist())):
        q=ids[tasks[ids]==task]
        picked.extend(rng.choice(q,min(int(per_task),len(q)),replace=False).tolist())
    picked=np.asarray(picked,np.int64)
    if len(picked)<2:return .02
    err=_pair_err(np.asarray(data["target"])[picked]);t=tasks[picked];s=scenes[picked]
    eligible=(t[:,None]!=t[None,:])&(s[:,None]!=s[None,:])
    vals=err[eligible]
    return float(np.quantile(vals,.25)) if len(vals) else .02


def _contrastive_loss(z, target, scene, task, tau_pos):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    # Weakly labelled positives: small scale-only type error, different scene
    # and task. Same-scene, different-type examples stay in the denominator.
    with torch.no_grad():
        x=target.flatten(1); den=x.square().sum(-1).clamp_min(1e-10)
        dot=(x@x.T).clamp_min(0); pair=((den[:,None]-dot.square()/den[None,:])/x.shape[1]).clamp_min(0)
        s0=np.asarray(scene); _,si=np.unique(s0,return_inverse=True)
        s=torch.as_tensor(si,device=z.device); t=torch.as_tensor(task,device=z.device)
        off=~torch.eye(len(z),device=z.device,dtype=torch.bool)
        pos=(pair<=tau_pos)&(s[:,None]!=s[None,:])&(t[:,None]!=t[None,:])&off
        w=torch.exp(-pair/max(float(tau_pos),1e-6))*pos
    norm=F.normalize(z,dim=-1); logits=norm@norm.T/.05
    logits=logits.masked_fill(torch.eye(len(z),device=z.device,dtype=torch.bool),-1e4)
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    roww=w.sum(1,keepdim=True); valid=roww[:,0]>0
    if not valid.any(): return z.sum()*0.,0
    soft=w/roww.clamp_min(1e-12)
    return -(soft*logp).sum(1)[valid].mean(), int(pos.sum().item())


def _load_fit(out):
    z=np.load(Path(out)/"fit_rows.npz",allow_pickle=False)
    data={k:z[k] for k in z.files}
    if "tau_pos" not in data:
        ids=np.flatnonzero(np.asarray(data["select"],bool)); pair=_pair_err(np.asarray(data["target"])[ids])
        t=np.asarray(data["task"])[ids];s=np.asarray(data["scene"])[ids]
        eligible=(t[:,None]!=t[None,:])&(s[:,None]!=s[None,:])
        vals=pair[eligible];data["tau_pos"]=np.asarray(np.quantile(vals,.25) if len(vals) else .02,np.float32)
    return data


def _dataset_tokens(data, layer):
    if layer=="final":
        row_ids=np.asarray(data["row_id"],np.int64)
        return _tokens_for_rows(row_ids,"train","final")
    # Mid cache is stored in train split order, the same order as fit_rows.
    return _tokens_for_rows(np.arange(len(data["task"])),"train",layer)


def _preload_final_tokens(data, kind, device="cuda"):
    """Load this split's required final-layer tokens once, compact fp16, to device."""
    import torch
    if "_token_gpu" in data and data.get("_token_kind")==kind: return data
    _release_cached_tokens(data)
    row_ids=np.asarray(data["row_id"],np.int64)
    vals=np.load(TOKENS,mmap_mode="r"); groups=np.load(GROUPS,mmap_mode="r")
    g=np.asarray(groups[0]); keep=np.flatnonzero(g==1) if kind=="A" else np.flatnonzero(g>=0)
    # Gather once before optimization. Transfer compact half precision; batch
    # conversion happens after GPU indexing to avoid repeated memmap traffic.
    x=np.asarray(vals[row_ids][:,keep],dtype=np.float16)
    data["_token_gpu"]=torch.as_tensor(x,device=device)
    data["_token_mask_gpu"]=torch.ones((len(row_ids),len(keep)),device=device,dtype=torch.bool)
    data["_token_source_group"]=g[keep]
    data["_token_kind"]=kind
    return data


def _preload_eval_tokens(data, token_path, group_path, *, kind, device="cuda"):
    """Preload corrected unseen Eval tokens once; labels are not used here."""
    import torch
    vals=np.load(token_path,mmap_mode="r");groups=np.load(group_path,mmap_mode="r");g=np.asarray(groups[0])
    keep=np.flatnonzero(g==1) if kind=="A" else np.flatnonzero(g>=0)
    row_ids=np.asarray(data.get("row_id",np.arange(len(data["task"]))),np.int64)
    x=np.asarray(vals[row_ids][:,keep],dtype=np.float16)
    _release_cached_tokens(data)
    data["_token_gpu"]=torch.as_tensor(x,device=device)
    data["_token_mask_gpu"]=torch.ones((len(x),len(keep)),device=device,dtype=torch.bool)
    data["_token_source_group"]=g[keep]
    data["_token_kind"]=kind
    return data


def _release_cached_tokens(data):
    for k in ("_token_gpu","_token_mask_gpu","_token_source_group","_token_kind"):
        data.pop(k,None)


def _token_batch(data, ids, layer, kind):
    """Read only requested rows from the large read-only VLM memmaps."""
    ids=np.asarray(ids,np.int64)
    if layer=="final" and "_token_gpu" in data:
        import torch
        ix=torch.as_tensor(ids,device=data["_token_gpu"].device)
        return data["_token_gpu"][ix].float(),data["_token_mask_gpu"][ix]
    if layer=="final":
        vals=np.load(TOKENS,mmap_mode="r");groups=np.load(GROUPS,mmap_mode="r")
        row_ids=np.asarray(data["row_id"],np.int64)[ids]
        tok=np.asarray(vals[row_ids],np.float32);g=np.asarray(groups[0])
        keep=np.flatnonzero(g==1) if kind=="A" else np.flatnonzero(g>=0)
        return tok[:,keep],np.ones((len(ids),len(keep)),bool)
    lid=int(layer[3:]);li=(6,10,14,15,16,17).index(lid)
    vals=np.load(MID/"train_mid_v.npy",mmap_mode="r");masks=np.load(MID/"train_mid_mask.npy",mmap_mode="r")
    tok=np.asarray(vals[ids,li],np.float32);mask=np.asarray(masks[ids],bool)
    if kind=="A":
        keep=np.arange(64,128);tok=tok[:,keep];mask=mask[:,keep]
    return tok,mask


def _train_once(out, condition, *, seed, tau=.02, k=4, layer="final", epochs=8,
                fit_mask=None, save_path=None, device="cuda", batch_size=64, l1_weight=.20,
                tau_pos=None, report_each=False, data_override=None):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.40,device=0)
    torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed)) if torch.cuda.is_available() else None
    data=_load_fit(out) if data_override is None else data_override; device=torch.device(device)
    tasks=np.asarray(data["task"],np.int64); scenes=np.asarray(data["scene"])
    fit=np.asarray(data["fit"],bool) if fit_mask is None else np.asarray(fit_mask,bool)
    fit_ids=np.flatnonzero(fit)
    if not len(fit_ids): raise ValueError("empty training fold")
    label,goal=condition.split("-"); kind={"P5":"P5","A":"A","B":"B"}[label]
    if kind in ("A","B"):
        token_dim=2048 if layer=="final" else 256
    else: tok=mask=None;token_dim=2048
    model=_make_model(kind,k,token_dim).to(device)
    # Key encoder shares the exact W scale-only representation used in K1.
    shapes=np.asarray(data["pool_shape"],np.float32).reshape(len(data["pool_shape"]),-1)
    shapes_t=torch.as_tensor(shapes,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    err=np.asarray(data["error"],np.float32);allowed=np.asarray(data["allowed"],bool)
    teacher=_soft_teacher(err,allowed,tau)
    hist=torch.as_tensor(data["history"],device=device);state=torch.as_tensor(data["state"],device=device)
    targ=torch.as_tensor(data["target"],device=device);task_t=torch.as_tensor(tasks,device=device)
    scene_t=scenes
    rng=np.random.default_rng(seed); records=[]
    tau_pos=float(np.asarray(data["tau_pos"]).item()) if tau_pos is None else float(tau_pos)
    # Guarantee one same-scene, different-type negative in every L1 minibatch.
    hard_pairs={}
    if goal=="L1":
        for sc in sorted(set(scenes[fit_ids].tolist())):
            local=fit_ids[scenes[fit_ids]==sc]
            if len(local)<2: continue
            sample=rng.choice(local,min(96,len(local)),replace=False)
            pe=_pair_err(np.asarray(data["target"])[sample]);eligible=(~np.eye(len(sample),dtype=bool))&(pe>tau_pos)
            if not eligible.any(): continue
            ij=np.unravel_index(np.where(eligible,pe,-1).argmax(),pe.shape)
            hard_pairs[sc]=(int(sample[ij[0]]),int(sample[ij[1]]))
        if not hard_pairs: raise ValueError("L1 could not form a same-scene different-type negative pair")
    if kind in ("A","B") and layer=="final": data=_preload_final_tokens(data,kind,device)
    for epoch in range(int(epochs)):
        order=rng.permutation(fit_ids);model.train();loss_sum=0.;cnt=0;pos_count=0
        chunk_size=int(batch_size)-2 if goal=="L1" else int(batch_size)
        for st in range(0,len(order),chunk_size):
            ix=order[st:st+chunk_size]
            if goal=="L1":
                sc=rng.choice(list(hard_pairs));ix=np.concatenate([ix,np.asarray(hard_pairs[sc],np.int64)])
            if len(ix)<2: continue
            bi=torch.as_tensor(ix,device=device)
            token_arg=mask_arg=None
            if kind in ("A","B"):
                tb,mb=_token_batch(data,ix,layer,kind)
                token_arg=torch.as_tensor(tb,device=device);mask_arg=torch.as_tensor(mb,device=device)
            q,_=model(hist[bi],state[bi],token_arg,mask_arg)
            keys=F.normalize(model.key(shapes_t),dim=-1); qn=F.normalize(q,dim=-1)
            scores=qn@keys.T*model.log_scale.exp()
            al=torch.as_tensor(allowed[ix],device=device); teach=torch.as_tensor(teacher[ix],device=device)
            scores=scores.masked_fill(~al,-1e4)
            llist=-(teach*F.log_softmax(scores,dim=-1)).sum(-1).mean()
            if goal=="L1":
                lc,npos=_contrastive_loss(q,targ[bi],scene_t[ix],tasks[ix],tau_pos)
                loss=llist+float(l1_weight)*lc;pos_count+=npos
            else: loss=llist
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            loss_sum+=float(loss.detach());cnt+=1
        rec={"epoch":epoch+1,"train_loss":loss_sum/max(cnt,1),"batches":cnt,"positive_pairs":pos_count}
        records.append(rec)
        if report_each: print(json.dumps({"condition":condition,"seed":seed,**rec}),flush=True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        if Path(save_path).exists(): raise FileExistsError(save_path)
        tmp=Path(str(save_path)+".tmp")
        torch.save({"state_dict":model.cpu().state_dict(),"condition":condition,"seed":seed,"tau":tau,
            "k":k,"layer":layer,"epochs":epochs,"tau_pos":tau_pos,"l1_weight":l1_weight,
            "model_kind":kind,"token_dim":token_dim},tmp)
        os.replace(tmp,save_path)
    return model,records


def _score_model(model,condition,data,ids,*,layer="final",device="cuda",batch=32):
    import torch
    import torch.nn.functional as F
    if str(device).startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.40,device=0)
    model.eval().to(device); d={k:np.asarray(v) for k,v in data.items() if not str(k).startswith("_")}
    kind=condition.split("-")[0]
    if kind in ("A","B"):tok=mask=True
    else:tok=mask=None
    keys0=np.asarray(d["pool_shape"],np.float32).reshape(len(d["pool_shape"]),-1)
    with torch.no_grad(): keys=F.normalize(model.key(torch.as_tensor(keys0,device=device)),dim=-1)
    scores=[]; attn_rows=[]
    for st in range(0,len(ids),batch):
        ix=np.asarray(ids[st:st+batch],np.int64)
        th=torch.as_tensor(d["history"][ix],device=device);ss=torch.as_tensor(d["state"][ix],device=device)
        tt=mm=None
        if kind in ("A","B"):
            tb,mb=_token_batch(data,ix,layer,kind);tt=torch.as_tensor(tb,device=device);mm=torch.as_tensor(mb,device=device)
        with torch.no_grad(): q,aw=model(th,ss,tt,mm);sc=F.normalize(q,dim=-1)@keys.T*model.log_scale.exp()
        scores.append(sc.cpu().numpy())
        if aw is not None: attn_rows.append(aw.mean(1).cpu().numpy())
    scores=np.concatenate(scores) if scores else np.empty((0,len(keys0)),np.float32)
    allow=np.asarray(d["allowed"])[ids]; scores[~allow]=-np.inf
    rank=np.argsort(-scores,axis=1,kind="stable")
    attn=np.concatenate(attn_rows) if attn_rows else None
    return scores,rank,attn


def _top1_metrics(rank,err,allowed):
    ix=np.asarray(rank)[:,0];e=np.asarray(err);return float(np.mean(e[np.arange(len(ix)),ix]))


def _metric_retrieval(scores,err,allowed,scene):
    from scipy.stats import rankdata
    s=np.asarray(scores);e=np.asarray(err);a=np.asarray(allowed,bool);scene=np.asarray(scene)
    ranks=np.argsort(-s,axis=1,kind="stable");top=e[np.arange(len(e)),ranks[:,0]]
    oracle=np.min(np.where(a,e,np.inf),axis=1);rand=np.nanmean(np.where(a,e,np.nan),axis=1)
    gap=rand-oracle
    recovered=np.divide(rand-top,gap,out=np.zeros_like(rand),where=gap>1e-8)
    recovered=np.clip(recovered,0.,1.)
    ratio=float(np.mean(recovered))
    rho=np.zeros(len(s),np.float64); mask_groups={}
    for i,row in enumerate(a):mask_groups.setdefault(row.tobytes(),[]).append(i)
    for idx in mask_groups.values():
        ids=np.asarray(idx,np.int64);cols=np.flatnonzero(a[ids[0]])
        rs=rankdata(s[ids][:,cols],axis=1);re=rankdata(e[ids][:,cols],axis=1)
        rs-=rs.mean(1,keepdims=True);re-=re.mean(1,keepdims=True)
        num=(rs*re).sum(1);den=np.sqrt((rs*rs).sum(1)*(re*re).sum(1))
        rho[ids]=np.divide(num,den,out=np.zeros_like(num),where=den>0)
    per_scene={}
    for sc in sorted(set(scene.tolist())):
        ix=np.flatnonzero(scene==sc);per_scene[str(sc)]={"n":len(ix),"top1_type_error":float(top[ix].mean()),
            "oracle_type_error":float(oracle[ix].mean()),"recovery":float(np.mean(recovered[ix])),
            "spearman":float(np.mean(rho[ix]))}
    return {"n":len(s),"top1_type_error":float(top.mean()),"oracle_type_error":float(oracle.mean()),
        "random_type_error":float(np.nanmean(rand)),"recovery":ratio,"spearman_mean":float(np.mean(rho)),
        "scene":per_scene,"per_query_rho":rho.tolist()}


def _step1_metrics(scores, data, ids):
    """Metric adapter shared by step 1 and impl-g2's alternate models."""
    ids=np.asarray(ids,np.int64); allowed=np.asarray(data["allowed"])[ids]
    met=_metric_retrieval(np.asarray(scores),np.asarray(data["error"])[ids],allowed,
        np.asarray(data["group"])[ids])
    # _metric_retrieval returns corr(score,error), negative is good. Public
    # reports use positive-is-good (corr(score,-error)).
    rho=np.asarray(met.pop("per_query_rho"),np.float64)
    met["spearman_mean"]=-float(np.mean(rho));met["spearman_median"]=-float(np.median(rho))
    return met


def _slice_rows(data, ids):
    ids=np.asarray(ids,np.int64); out={}
    for k,v in data.items():
        if str(k).startswith("_"): continue
        a=np.asarray(v)
        out[k]=a[ids] if a.ndim and len(a)==len(data["task"]) else a
    return out


def run_step1_evaluations(out, train_fn, score_fn, *, data, eval_data,
                          conditions=CONDITIONS, groups=None, seed=4101, epochs=8):
    """Run four scene-group folds then one frozen Eval pass via injected models.

    `train_fn(condition, train_rows, seed=..., epochs=..., save_path=..., fold=...)`
    returns a model. `score_fn(model, condition, query_rows)` returns a [Q,M]
    higher-is-better score matrix, or {'scores': ..., 'extras': {...}}. The
    caller owns model construction/scoring; this function owns split/leakage
    checks, retrieval metrics, stable artifacts and the single Eval pass.
    Data mappings require row-aligned `task, group, fit, select, error, allowed`.
    `eval_data` additionally requires `group,error,allowed`; optional target
    arrays may be consumed by a score_fn but are never touched here.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups=np.asarray(data["group"] if groups is None else groups).astype(str)
    fit=np.asarray(data["fit"],bool);select=np.asarray(data["select"],bool)
    if len(groups)!=len(fit) or set(np.unique(groups))!={"libero10_all","libero_goal","libero_object","libero_spatial"}:
        raise ValueError("step1 requires the four fixed scene groups")
    cvdir=out/"group_cv";cvdir.mkdir(exist_ok=True);folds=[]
    for gi,held in enumerate(("libero_goal","libero_object","libero_spatial","libero10_all")):
        train_ids=np.flatnonzero(fit&(groups!=held));held_ids=np.flatnonzero(fit&(groups==held))
        seen_ids=np.flatnonzero(select&(groups!=held))
        if not len(train_ids) or not len(held_ids) or not len(seen_ids):raise ValueError(f"empty scene fold {held}")
        train_rows=_slice_rows(data,train_ids)
        # Leave the held scene out of the candidate/key cache as well as query
        # rows and teacher labels. Evaluation still scores the complete fixed W
        # pool, so it measures transfer to unseen candidate shapes.
        if "pool_task" in train_rows and "pool_shape" in train_rows:
            pg=np.asarray(train_rows.get("pool_group",scene_group_ids(train_rows["pool_task"]))).astype(str)
            keep=pg!=held
            if not keep.any():raise ValueError(f"fold {held} has no remaining memory candidates")
            for key in ("pool_task","pool_group","pool_shape"):
                if key in train_rows:train_rows[key]=np.asarray(train_rows[key])[keep]
            for key,value in list(train_rows.items()):
                if key.startswith("pool_") and key not in ("pool_task","pool_group","pool_shape"):
                    arr=np.asarray(value)
                    if arr.ndim and arr.shape[0]==len(keep):train_rows[key]=arr[keep]
            for key in ("error","allowed"):
                if key in train_rows:train_rows[key]=np.asarray(train_rows[key])[:,keep]
        if np.any(np.asarray(train_rows["group"]).astype(str)==held):raise AssertionError("held scene leaked into fit cache")
        if "pool_group" in train_rows and np.any(np.asarray(train_rows["pool_group"]).astype(str)==held):
            raise AssertionError("held scene candidate leaked into training key cache")
        test_sets=(("held_group",_slice_rows(data,held_ids)),("unseen_holdout_tasks_in_seen_groups",_slice_rows(data,seen_ids)))
        for cond in conditions:
            ckpt=cvdir/f"{held}_{cond.replace('-','_')}_s{seed}.pt";result=cvdir/f"{held}_{cond.replace('-','_')}_s{seed}.json"
            if result.exists():
                item=json.loads(result.read_text());folds.append(item);continue
            model=train_fn(cond,train_rows,seed=seed,epochs=epochs,save_path=ckpt,fold=held)
            item={"fold":held,"condition":cond,"seed":seed,"epochs":epochs,"n_train":len(train_ids),"excluded_group_in_train":False,"metrics":{}}
            for name,qd in test_sets:
                raw=score_fn(model,cond,qd)
                extras={}
                if isinstance(raw,dict):extras=raw.get("extras",{});raw=raw["scores"]
                _atomic_npz(cvdir/f"{held}_{cond.replace('-','_')}_s{seed}_{name}_scores.npz",scores=np.asarray(raw,np.float32))
                met=_step1_metrics(raw,qd,np.arange(len(qd["task"])))
                item["metrics"][name]={**met,"extras":extras,"n":len(qd["task"])}
            _release_cached_tokens(train_rows)
            for _,qd in test_sets:_release_cached_tokens(qd)
            if hasattr(model,"cpu"): model.cpu()
            _json(result,item);folds.append(item)
            _log(out,stage="step1_cv_done",fold=held,condition=cond,seed=seed,
                held_top1=item["metrics"]["held_group"]["top1_type_error"],
                held_recovery=item["metrics"]["held_group"]["recovery"])
    _json(cvdir/"summary.json",{"status":"complete","fold_count":4,"conditions":list(conditions),"seed":seed,"folds":folds})
    # Final evaluation is deliberately one call per condition after all choices freeze.
    if callable(eval_data):eval_data=eval_data()
    evdir=out/"eval";evdir.mkdir(exist_ok=True);eval_results={}
    for cond in conditions:
        ckpt=evdir/f"{cond.replace('-','_')}_s{seed}.pt";rp=evdir/f"{cond.replace('-','_')}_s{seed}.json"
        if rp.exists():eval_results[cond]=json.loads(rp.read_text());continue
        train_ids=np.flatnonzero(fit);train_rows=_slice_rows(data,train_ids)
        model=train_fn(cond,train_rows,seed=seed,epochs=epochs,save_path=ckpt,fold="full_fit")
        raw=score_fn(model,cond,eval_data);extras={}
        if isinstance(raw,dict):extras=raw.get("extras",{});raw=raw["scores"]
        _atomic_npz(evdir/f"{cond.replace('-','_')}_s{seed}_scores.npz",scores=np.asarray(raw,np.float32))
        met=_step1_metrics(raw,eval_data,np.arange(len(eval_data["task"])))
        item={"condition":cond,"seed":seed,"epochs":epochs,"n_train":len(train_ids),"metrics":met,"extras":extras,"eval_calls":1}
        if hasattr(model,"cpu"):model.cpu()
        _release_cached_tokens(train_rows);_release_cached_tokens(eval_data)
        _json(rp,item);eval_results[cond]=item;_log(out,stage="step1_eval_done",condition=cond,seed=seed,top1=met["top1_type_error"],recovery=met["recovery"])
    _json(evdir/"summary.json",{"status":"complete","rows":len(eval_data["task"]),"seed":seed,"selection_frozen":True,"eval_calls_per_condition":1,"conditions":eval_results})
    return {"cv":folds,"eval":eval_results}


def _step1_prepare_data(out):
    """Create the fixed 4-way group label; no target from unseen Eval read."""
    prepare(out)
    d=_load_fit(out)
    d["group"]=scene_group_ids(d["task"])
    d["pool_group"]=scene_group_ids(d["pool_task"])
    return d


def _step1_eval_data():
    """Read Eval targets only at this final stage, exactly once."""
    kq,tr,vs,sq=_modules()
    base,rows,unseen,splits=kq._production_data_splits(include_eval_targets=True)
    ids=np.asarray(splits["eval"][1],np.int64);ue=np.flatnonzero(~np.asarray(unseen["is_dev"],bool))
    if len(ids)!=619 or not np.array_equal(ids,ue):raise ValueError("Eval row order/count changed")
    evnp=np.load(SOURCE/"prep/W_inputs_eval.npz",allow_pickle=False)
    for k in ("task","episode","frame"):
        if not np.array_equal(np.asarray(evnp[k]),np.asarray(unseen[k][ue])):raise ValueError(f"Eval row order mismatch: {k}")
    pool0=tr.candidate_shape_pool(rep="C_vm_none");keep=~np.isin(np.asarray(pool0["source_task"]),HOLDOUT)
    pool={k:(np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v) for k,v in pool0.items()}
    actions=np.asarray(unseen["actions"][ue,:,:6],np.float32)
    target=np.asarray(vs.to_physical(actions),np.float32)
    errors=kq.shape_error_matrix(np.concatenate([target,np.zeros((*target.shape[:2],1),np.float32)],-1),pool["shape"],vs,batch=8,device="cpu")
    task=np.asarray(unseen["task"][ue],np.int64);episode=task*100000+np.asarray(unseen["episode"][ue],np.int64)
    out={"row_id":ue,"task":task,"episode":episode,"frame":np.asarray(unseen["frame"][ue]),
        "group":unseen_scene_ids(task),"error":errors,"allowed":np.asarray(pool["source_task"])[None,:]!=task[:,None],
        "pool_shape":np.asarray(pool["shape"],np.float32),"history":np.asarray(evnp["hist"],np.float32),
        "state":np.asarray(evnp["state"],np.float32),"target_action":np.asarray(unseen["actions"][ue,:,:6],np.float32),
        "target_physical":target,"hard":np.asarray(evnp["hard"],bool),"episode_bootstrap":episode,
        "_pool":pool}
    return out


def _step1_train_factory(out, device="cuda", batch=64):
    def train_fn(condition, train_rows, *, seed, epochs, save_path, fold):
        label=condition.split("-")[0]
        if label in ("A","B"):_preload_final_tokens(train_rows,label,device)
        tau_pos=.02
        if condition.endswith("-L1"):
            tau_pos=_tau_pos_for_rows(train_rows,np.arange(len(train_rows["task"])),seed=seed)
        if Path(save_path).exists():model,_=_torch_load(save_path,device)
        else:
            model,_=_train_once(out,condition,seed=seed,tau=.01,k=1,layer="final",epochs=epochs,
                save_path=save_path,device=device,batch_size=batch,tau_pos=tau_pos,data_override=train_rows,report_each=True)
        return model
    return train_fn


def _step1_score_factory(device="cuda", batch=16):
    def score_fn(model, condition, query_rows):
        label=condition.split("-")[0]
        if label in ("A","B") and "_token_gpu" not in query_rows:
            if "target_action" in query_rows:
                _preload_eval_tokens(query_rows,UNSEEN_ROWS/"tokens.npy",UNSEEN_ROWS/"groups.npy",kind=label,device=device)
            else:_preload_final_tokens(query_rows,label,device)
        # _score_model's row ids are local after slicing. The query token cache
        # is already resident, so this path does no per-batch memmap gather.
        scores,rank,_=_score_model(model,condition,query_rows,np.arange(len(query_rows["task"])),
            layer="final",device=device,batch=batch)
        extras={}
        if "target_action" in query_rows:
            import key_query_probe as kq
            import variable_shape_memory as vs
            pool=query_rows["_pool"];selected=rank[:,0]
            decoded=kq.decode_pool_actions(pool,selected[:,None],vs,theta_source="src")[:,0,:,:6]
            truth=np.asarray(query_rows["target_action"],np.float32)
            mse=np.mean((decoded-truth)**2,axis=(1,2))
            hist=np.asarray(query_rows["history"],np.float32);hold=np.repeat(hist[:,-1:,:6],truth.shape[1],axis=1)
            hmse=np.mean((hold-truth)**2,axis=(1,2));hard=np.asarray(query_rows["hard"],bool);ep=np.asarray(query_rows["episode_bootstrap"])
            extras["action_mse"]={"all_mean":float(mse.mean()),"hard_mean":float(mse[hard].mean()),
                "vs_hold_last":_bootstrap_diff(mse,hmse,ep),
                "vs_hold_last_hard":_bootstrap_diff(mse[hard],hmse[hard],ep[hard]),
                "hold_last_all":float(hmse.mean()),"hold_last_hard":float(hmse[hard].mean()),
                "hard_rows":int(hard.sum()),"rows":len(mse)}
        return {"scores":scores,"extras":extras}
    return score_fn


def step1(out=OUT/"step1", *, data_root=OUT, device="cuda", batch=64, epochs=8):
    """Fixed-config Step 1: four scene-group folds and one LIBERO-90 Eval pass."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=_step1_prepare_data(data_root)
    groups=np.asarray(data["group"])
    selection={"status":"frozen","conditions":list(CONDITIONS),"selected":{
        c:{"k":1,"layer":"final","teacher_tau":.01} for c in CONDITIONS},
        "seed":4101,"epochs":int(epochs),"basis":"task-holdout metrics supplied in scratchpad/inv/score5.json; no additional search",
        "holdout_score5":{"P5-L0":{"top1_type_error":.0282,"recovery":.60,"spearman_positive":.42},
          "A-L0":{"top1_type_error":.0183,"recovery":.76,"spearman_positive":.47},
          "A-L1":{"top1_type_error":.0220,"recovery":.70,"spearman_positive":.45},
          "B-L0":{"top1_type_error":.0194,"recovery":.74,"spearman_positive":.48},
          "B-L1":{"top1_type_error":.0210,"recovery":.71,"spearman_positive":.45}},
        "spearman_convention":"reported as corr(score,-type_error), so positive is better"}
    if (out/"selection.json").exists():
        previous=json.loads((out/"selection.json").read_text())
        if previous.get("seed")!=4101 or previous.get("selected")!=selection["selected"]:
            raise ValueError("existing Step1 selection differs; refusing to overwrite")
        selection=previous
    else:_json(out/"selection.json",selection)
    # Defer even loading Eval action labels until all CV models finish.
    result=run_step1_evaluations(out,_step1_train_factory(out.parent,device,batch),_step1_score_factory(device),
        data=data,eval_data=_step1_eval_data,conditions=CONDITIONS,groups=groups,seed=4101,epochs=epochs)
    _json(out/"run_manifest.json",{"status":"complete","seed":4101,"epochs":epochs,"conditions":list(CONDITIONS),
        "groups":["libero_goal","libero_object","libero_spatial","libero10_all"],"eval_rows":619,
        "eval_targets_used_for_selection":False,"model_checkpoint_selection":"fixed 8-epoch final checkpoint; no Eval selection"})
    return result


def step1_report(out=OUT/"step1"):
    out=Path(out);sel=json.loads((out/"selection.json").read_text())
    cv=json.loads((out/"group_cv/summary.json").read_text());ev=json.loads((out/"eval/summary.json").read_text())
    lines=["# INV1 Step 1: 場面グループ抜きと未学習 Eval","",
      "固定条件は k=1、final 層、教師 τ=.01、seed 4101、8 epoch。Eval は各条件1回だけ採点し、選択には使用していない。","",
      "## 場面グループ抜き","",
      "外した場面の fit 行と、残り場面にある学習未使用の LIBERO-40 holdout-task 行を比較した。値は全て `corr(score, -type_error)` の Spearman で、正が良い。差は seen-group holdout − held-group。","",
      "| 条件 | 外した群 | n held / n seen | held top1 | seen top1 | seen−held | held 回収率 | seen 回収率 | seen−held 回収率 | held Spearman mean / median | seen Spearman mean / median | seen−held Spearman mean |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        for f in cv["folds"]:
            if f["condition"]!=c:continue
            h=f["metrics"]["held_group"];s=f["metrics"]["unseen_holdout_tasks_in_seen_groups"]
            lines.append(f"| {c} | {f['fold']} | {h['n']} / {s['n']} | {h['top1_type_error']:.6f} | {s['top1_type_error']:.6f} | {s['top1_type_error']-h['top1_type_error']:+.6f} | {h['recovery']:.4f} | {s['recovery']:.4f} | {s['recovery']-h['recovery']:+.4f} | {h['spearman_mean']:.4f} / {h['spearman_median']:.4f} | {s['spearman_mean']:.4f} / {s['spearman_median']:.4f} | {s['spearman_mean']-h['spearman_mean']:+.4f} |")
    lines += ["","## Eval 619 窓","",
      "6軸 MSE は選択した候補を保存スケールで復元した値。差の CI は episode bootstrap 5,000 回で、負なら hold-last より小さい。Spearman は符号を反転し正を良い向きにした。","",
      "| 条件 | top1 型誤差 | 回収率 | Spearman mean / median | MSE6 all | MSE6 hard | 差 vs hold-last all (95% CI) | 差 vs hold-last hard (95% CI) |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    methods=ev["conditions"]
    for c in CONDITIONS:
        r=methods[c];m=r["extras"]["action_mse"]
        a=m["vs_hold_last"];h=m["vs_hold_last_hard"]
        lines.append(f"| {c} | {r['metrics']['top1_type_error']:.6f} | {r['metrics']['recovery']:.4f} | {r['metrics']['spearman_mean']:.4f} / {r['metrics']['spearman_median']:.4f} | {m['all_mean']:.6f} | {m['hard_mean']:.6f} | {a['mean']:+.6f} [{a['ci95'][0]:+.6f}, {a['ci95'][1]:+.6f}] | {h['mean']:+.6f} [{h['ci95'][0]:+.6f}, {h['ci95'][1]:+.6f}] |")
    ref=methods[CONDITIONS[0]]["extras"]["action_mse"]
    lines += ["",f"再計算 hold-last は {ref['hold_last_all']:.6f}/{ref['hold_last_hard']:.6f} (all/hard)。従来参照値 .114/.201 と併記し、評価分割差を保持する。", "",
      "## 選択設定と符号","", "保留タスクの既存 score5 を根拠に k=1 / final / τ=.01 を固定した。追加の探索は行っていない。既存指標の raw rho は score と error の相関なので、ここでは符号反転し `corr(score,-error)` を報告した。", "",
      "## 読み方と制約","", "場面グループ間の差は記述的。小さい差を同等性とは解釈しない。LIBERO-90 の正解は最終の型誤差・行動 MSE 採点だけに使った。3 seed、履歴破損、デノイズはステップ1の後続作業。", ""]
    p=out/"report.md";p.parent.mkdir(parents=True,exist_ok=True);p.write_text("\n".join(lines));return p


def _torch_load(path,device):
    import torch
    ck=torch.load(path,map_location=device,weights_only=False)
    model=_make_model(ck["model_kind"],ck["k"],ck["token_dim"])
    model.load_state_dict(ck["state_dict"]); return model,ck


def _config_grid():
    # P5 and A use fixed architecture. B selection is performed only on the
    # LIBERO-40 task-holdout rows, across K, available layer cache, and target τ.
    return [("P5-L0",1,"final",t) for t in TAUS]


def tune(out=OUT,*,epochs=3,device="cuda",batch=64):
    """Tune temperature, B latent count and cached layer on LIBERO-40 holdout tasks."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True); result_path=out/"selection.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    data=_load_fit(out); task=np.asarray(data["task"]); sel=np.flatnonzero(np.asarray(data["select"],bool))
    configs=[]
    # Compare teacher temperatures for P5 and A; B additionally searches latent
    # count and all cached token taps (final plus intermediate V=16/17). We use
    # the two intermediate layers with full multimodal coverage in the existing
    # cache; final features retain the unpooled 712-token representation.
    tau_pos=float(np.asarray(data["tau_pos"]).item())
    for tau in TAUS:
        configs += [("P5-L0",1,"final",tau),("A-L0",1,"final",tau),("A-L1",1,"final",tau)]
        for layer in LAYERS:
            for k in KS:
                configs += [("B-L0",k,layer,tau),("B-L1",k,layer,tau)]
    report=[]; tstart=time.time()
    for ci,(cond,k,layer,tau) in enumerate(configs):
        ckpt=out/f"tuning/{cond.replace('-','_')}_k{k}_{layer}_tau{tau:g}.pt"
        if ckpt.exists(): model,meta=_torch_load(ckpt,device);losses=[{"train_loss":None}]
        else:model,losses=_train_once(out,cond,seed=4101,tau=tau,k=k,layer=layer,epochs=epochs,tau_pos=tau_pos,
            save_path=ckpt,device=device,batch_size=batch)
        sc,rank,aw=_score_model(model,cond,data,sel,layer=layer,device=device)
        e=np.asarray(data["error"])[sel];a=np.asarray(data["allowed"])[sel];ix=rank[:,0]
        top=float(e[np.arange(len(ix)),ix].mean())
        row={"condition":cond,"k":int(k),"layer":layer,"teacher_tau":float(tau),"selection_mean_top1_type_error":top,
             "train_loss":losses[-1]["train_loss"]}
        report.append(row);_log(out,stage="tune",done=ci+1,total=len(configs),**row)
        del model
    # Each L0/L1 condition selects by holdout mean top-1 S-shared type error.
    chosen={}
    for cond in ("P5-L0","A-L0","B-L0","A-L1","B-L1"):
        rs=[r for r in report if r["condition"]==cond]
        chosen[cond]=min(rs,key=lambda r:(r["selection_mean_top1_type_error"],r["teacher_tau"],r["k"],r["layer"]))
    result={"status":"selected","selection_split":"LIBERO-40 designated holdout tasks only",
        "eval_labels_used":False,"primary_selection":"minimum holdout mean top-1 S-shared type error",
        "grid":report,"selected":chosen,"selection_seconds":time.time()-tstart,
        "contrastive_weight":.20,"contrastive_temperature":.05,
        "contrastive_positive":"per-query normalized cross-scene AND cross-task query weights exp(-e/tau_pos); tau_pos is LIBERO-40 holdout pair q25",
        "tau_pos":tau_pos}
    _json(result_path,result);return result


def train(out=OUT,*,epochs=8,device="cuda",batch=64):
    """Train selected five conditions with three independent seeds on fit tasks."""
    out=Path(out); sel=json.loads((out/"selection.json").read_text())
    data=_load_fit(out); records=[]
    for cond in CONDITIONS:
        cfg=sel["selected"][cond]
        for seed in SEEDS:
            p=out/f"models/{cond.replace('-','_')}_s{seed}.pt"
            if not p.exists():
                model,losses=_train_once(out,cond,seed=seed,tau=cfg["teacher_tau"],k=cfg["k"],layer=cfg["layer"],
                    epochs=epochs,save_path=p,device=device,batch_size=batch,report_each=True)
                rec={"condition":cond,"seed":seed,"checkpoint":str(p),"final_train_loss":losses[-1]["train_loss"],"epochs":epochs}
                _log(out,stage="train_done",**rec);del model
            else:
                import torch
                ck=torch.load(p,map_location="cpu",weights_only=False)
                if ck.get("condition")!=cond or int(ck.get("seed"))!=seed:raise ValueError(f"checkpoint metadata mismatch: {p}")
            records.append({"condition":cond,"seed":seed,"checkpoint":str(p),"epochs":epochs})
    _json(out/"train_manifest.json",{"status":"trained","records":records,"selected":sel["selected"],"seeds":SEEDS})
    return records


def _scene_fold_mask(data, held_scene):
    fit=np.asarray(data["fit"],bool)
    return fit & (np.asarray(data["scene"])!=held_scene)


def cv(out=OUT,*,epochs=5,device="cuda",batch=64,max_scenes=None):
    """Leave-one-fit-scene-out. One model per condition/seed/fold; artifacts resume."""
    out=Path(out);data=_load_fit(out);selection=json.loads((out/"selection.json").read_text())
    fit_scenes=sorted(set(np.asarray(data["scene"])[np.asarray(data["fit"],bool)].tolist()))
    if max_scenes: fit_scenes=fit_scenes[:int(max_scenes)]
    fold_dir=out/"cv";fold_dir.mkdir(parents=True,exist_ok=True);all_res=[];fit=np.asarray(data["fit"],bool)
    for fs in fit_scenes:
        eval_ids=np.flatnonzero(fit & (np.asarray(data["scene"])==fs))
        rng=np.random.default_rng(4101+len(all_res)+sum(bytearray(str(fs).encode())))
        held_ids=rng.choice(eval_ids,min(128,len(eval_ids)),replace=False) if len(eval_ids)>128 else eval_ids
        trained_ids=[]
        for sc0 in fit_scenes:
            if sc0==fs: continue
            cand=np.flatnonzero(fit & (np.asarray(data["scene"])==sc0))
            if len(cand):trained_ids.extend(rng.choice(cand,min(128,len(cand)),replace=False).tolist())
        trained_ids=np.asarray(trained_ids,np.int64)
        train_mask=_scene_fold_mask(data,fs)
        fold_tau_pos=_tau_pos_for_rows(data,np.flatnonzero(train_mask),seed=4101)
        train_scene_set=sorted(set(np.asarray(data["scene"])[train_mask].tolist()))
        for cond in CONDITIONS:
            cfg=selection["selected"][cond]
            for seed in SEEDS:
                ckpt=fold_dir/f"{str(fs).replace(':','_')}_{cond.replace('-','_')}_s{seed}.pt"
                p=fold_dir/f"{str(fs).replace(':','_')}_{cond.replace('-','_')}_s{seed}.json"
                if p.exists():
                    all_res.append(json.loads(p.read_text()));continue
                if not ckpt.exists():
                    _train_once(out,cond,seed=seed,tau=cfg["teacher_tau"],k=cfg["k"],layer=cfg["layer"],
                        epochs=epochs,fit_mask=train_mask,save_path=ckpt,device=device,batch_size=batch,
                        tau_pos=fold_tau_pos)
                model,meta=_torch_load(ckpt,device)
                score_ids=np.concatenate([held_ids,trained_ids]);sc,rank,attn=_score_model(model,cond,data,score_ids,layer=cfg["layer"],device=device,batch=16)
                ee=np.asarray(data["error"])[score_ids];aa=np.asarray(data["allowed"])[score_ids];ss=np.asarray(data["scene"])[score_ids]
                met=_metric_retrieval(sc,ee,aa,ss)
                held_met=_metric_retrieval(sc[:len(held_ids)],ee[:len(held_ids)],aa[:len(held_ids)],ss[:len(held_ids)])
                train_met=_metric_retrieval(sc[len(held_ids):],ee[len(held_ids):],aa[len(held_ids):],ss[len(held_ids):])
                item={"held_scene":fs,"condition":cond,"seed":seed,"n_held":len(held_ids),"n_trained":len(trained_ids),"metrics":met,
                      "held_scene_metrics":held_met,"trained_scene_metrics":train_met,
                      "trained_scenes":train_scene_set,"fold_tau_pos_from_training_scenes_only":fold_tau_pos,
                      "checkpoint":str(ckpt)}
                _json(p,item);all_res.append(item)
                _log(out,stage="cv_fold",held_scene=fs,condition=cond,seed=seed,n=len(held_ids),top1=held_met["top1_type_error"],recovery=held_met["recovery"])
                del model
    _json(fold_dir/"summary.json",{"status":"complete" if not max_scenes else "partial","folds":all_res,"fit_scene_count":len(fit_scenes),"max_scenes":max_scenes})
    return all_res


def _bootstrap_diff(values, baseline, episode, *, reps=5000,seed=4101):
    v=np.asarray(values,float)-np.asarray(baseline,float); ep=np.asarray(episode)
    unique,inv=np.unique(ep,return_inverse=True); sums=np.bincount(inv,weights=v); counts=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(len(unique),size=(reps,len(unique)))
    draws=sums[ix].sum(1)/counts[ix].sum(1)
    return {"mean":float(v.mean()),"ci95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],"episodes":len(unique)}


def _history_perturbations(eval_history, fit_history, allowed, pool_shape, *, seed=4101):
    """Deterministic Eval history corruption; never consumes Eval labels."""
    hist=np.asarray(eval_history,np.float32);fit=np.asarray(fit_history,np.float32)
    allowed=np.asarray(allowed,bool);pool=np.asarray(pool_shape,np.float32)
    if hist.ndim!=3 or fit.ndim!=3 or hist.shape[1:]!=fit.shape[1:] or allowed.shape!=(len(hist),len(pool)):
        raise ValueError("history corruption shape mismatch")
    sd=np.maximum(fit.std(axis=(0,1),keepdims=True),1e-6)
    rng=np.random.default_rng(seed)
    noisy=hist+rng.normal(size=hist.shape).astype(np.float32)*(.10*sd)
    replacement=np.empty(len(hist),np.int64)
    for i in range(len(hist)):
        cand=np.flatnonzero(allowed[i])
        if not len(cand):raise ValueError("history replacement has no allowed candidate")
        replacement[i]=rng.choice(cand)
    replaced=pool[replacement,:,:hist.shape[-1]].copy()
    return noisy,replaced,replacement


def smoke(out=OUT,*,device="cuda"):
    """One-epoch real-cache path check for all five methods; no Eval labels."""
    out=Path(out);p=out/"smoke.json"
    if p.exists():return json.loads(p.read_text())
    data=_load_fit(out);rng=np.random.default_rng(20261112);fit=np.flatnonzero(data["fit"])
    # Ensure a compact but task-balanced fit subset for CPU/GPU path coverage.
    sample=[]
    for t in sorted(set(np.asarray(data["task"])[fit].tolist())):
        ids=fit[np.asarray(data["task"])[fit]==t]
        sample.extend(rng.choice(ids,min(32,len(ids)),replace=False).tolist())
    sample=np.asarray(sample,np.int64);mask=np.zeros(len(data["fit"]),bool);mask[sample]=True
    cases=[("P5-L0",1,"final"),("A-L0",1,"final"),("B-L0",4,"final"),
           ("A-L1",1,"final"),("B-L1",4,"mid16")]
    results=[];sel=np.flatnonzero(data["select"])[:8]
    for cond,k,layer in cases:
        model,losses=_train_once(out,cond,seed=4101,tau=.02,k=k,layer=layer,epochs=1,
            fit_mask=mask,device=device,batch_size=32,tau_pos=float(np.asarray(data["tau_pos"]).item()))
        scores,rank,attn=_score_model(model,cond,data,sel,layer=layer,device=device,batch=4)
        results.append({"condition":cond,"k":k,"layer":layer,"fit_rows":len(sample),
            "score_shape":list(scores.shape),"rank_shape":list(rank.shape),"attention_shape":None if attn is None else list(attn.shape),
            "one_epoch_loss":losses[-1]["train_loss"]});del model
    result={"status":"smoke_complete","eval_targets_read":False,"fit_rows":len(sample),"cases":results,
        "feature_sources":{"final":"run-01 full 712-token output, group masked","mid16":"key-query-01 policy_features train_mid_v, 328 pooled prefix tokens"}}
    _json(p,result);return result


def evaluate(out=OUT,*,device="cuda",batch=16):
    """One final 619-row Eval scoring pass after all selection is frozen."""
    out=Path(out);dest=out/"eval";dest.mkdir(parents=True,exist_ok=True)
    if (dest/"summary.json").exists():return json.loads((dest/"summary.json").read_text())
    kq,tr,vs,sq,base,rows,unseen,splits=_modules()[0]._production_data_splits(include_eval_targets=True)
    ids=np.asarray(splits["eval"][1],np.int64);ue=np.flatnonzero(~np.asarray(unseen["is_dev"],bool))
    # K1 eval split order is sorted unseen row order; assert the target rows match.
    if len(ids)!=619 or len(ue)!=619 or not np.array_equal(ids,ue):raise ValueError("Eval row order/count changed")
    # Compute action shape errors only at the final scoring stage.
    pool0=tr.candidate_shape_pool(rep="C_vm_none");keep=~np.isin(np.asarray(pool0["source_task"]),HOLDOUT)
    pool={k:(np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v) for k,v in pool0.items()}
    actions=np.asarray(unseen["actions"][ue,:,:6],np.float32); qshape=np.asarray(vs.to_physical(actions),np.float32)
    errors=kq.shape_error_matrix(np.concatenate([qshape,np.zeros((*qshape.shape[:2],1),np.float32)],-1),pool["shape"],vs,batch=8,device="cpu")
    task=np.asarray(unseen["task"][ue],np.int64);episode_local=np.asarray(unseen["episode"][ue],np.int64)
    episode=task*100000+episode_local  # episode bootstrap clusters are task-unique
    evfile=SOURCE/"prep/W_inputs_eval.npz";evnp=np.load(evfile,allow_pickle=False)
    for k in ("task","episode","frame"):
        if not np.array_equal(np.asarray(evnp[k]),np.asarray(unseen[k][ue])):raise ValueError(f"Eval row order mismatch for {k}")
    hard=np.asarray(evnp["hard"],bool)
    evdata={"task":task,"episode":episode,
        "scene":unseen_scene_ids(task),"error":errors,"allowed":np.asarray(pool["source_task"])[None,:]!=task[:,None],
        "pool_shape":np.asarray(pool["shape"],np.float32)}
    # Eval cache's histories/states are the fixed corrected inputs; labels only
    # entered the metric arrays above. Per-condition attention is summarized by
    # camera/language token group.
    evdata["history"]=np.asarray(evnp["hist"],np.float32);evdata["state"]=np.asarray(evnp["state"],np.float32)
    # Token lookup over corrected unseen rows uses the full final VLM cache; its
    # row ordering is the UNSEEN_ROWS order. Middle V arrays use the same eval split.
    # _dataset_tokens has a split name, so build direct wrappers for this scoring pass.
    def score_eval(model,cond,cfg,history_values=None):
        import torch, torch.nn.functional as F
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
        if str(device).startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.40,device=0)
        model.eval().to(device);kind=cond.split("-")[0]
        if kind in ("A","B"):
            if cfg["layer"]=="final":
                vals=np.load(UNSEEN_ROWS/"tokens.npy",mmap_mode="r");groups=np.load(UNSEEN_ROWS/"groups.npy",mmap_mode="r")
                ix=np.flatnonzero(groups[0]>=0);tok=np.asarray(vals[ue][:,ix],np.float32);mask=np.ones((len(ue),len(ix)),bool);grp=groups[0,ix]
            else:
                lid=int(cfg["layer"][3:]);li=(6,10,14,15,16,17).index(lid)
                midval=np.load(MID/"eval_mid_v.npy",mmap_mode="r");midmask=np.load(MID/"eval_mid_mask.npy",mmap_mode="r")
                tok=np.asarray(midval[:,li],np.float32);mask=np.asarray(midmask,bool);grp=np.full(tok.shape[1],2,np.int8);grp[:64]=0;grp[64:128]=1
            use=np.flatnonzero(grp==1) if kind=="A" else np.flatnonzero(grp>=0)
            tok=tok[:,use];mask=mask[:,use]
        else:tok=mask=None
        hist_all=evdata["history"] if history_values is None else np.asarray(history_values,np.float32)
        with torch.no_grad():
            keys=F.normalize(model.key(torch.as_tensor(np.asarray(evdata["pool_shape"]).reshape(len(evdata["pool_shape"]),-1),device=device)),dim=-1)
            arr=[];aw=[]
            for st in range(0,len(ue),batch):
                ix=np.arange(st,min(st+batch,len(ue)));h=torch.as_tensor(hist_all[ix],device=device);s=torch.as_tensor(evdata["state"][ix],device=device)
                t=None if tok is None else torch.as_tensor(tok[ix],device=device);m=None if mask is None else torch.as_tensor(mask[ix],device=device)
                z,a=model(h,s,t,m);arr.append((F.normalize(z,dim=-1)@keys.T*model.log_scale.exp()).cpu().numpy())
                if a is not None:aw.append((a.mean(1).cpu().numpy(),m.cpu().numpy()))
        sc=np.concatenate(arr);allow=evdata["allowed"];sc[~allow]=-np.inf
        attn=(np.concatenate([x[0] for x in aw]),np.concatenate([x[1] for x in aw])) if aw else None
        return sc,np.argsort(-sc,axis=1,kind="stable"),attn
    selection=json.loads((out/"selection.json").read_text());results={};preds={}
    for cond in CONDITIONS:
        cfg=selection["selected"][cond];seed_rows=[];scores_by_seed=[];attn_by_seed=[];seed_mses=[]
        for seed in SEEDS:
            cp=out/f"models/{cond.replace('-','_')}_s{seed}.pt"
            model,_=_torch_load(cp,device);sc,rank,att=score_eval(model,cond,cfg)
            met=_metric_retrieval(sc,errors,evdata["allowed"],evdata["scene"])
            selected=errors[np.arange(len(ue)),rank[:,0]]
            seed_rows.append({"seed":seed,**{k:v for k,v in met.items() if k!="per_query_rho"},
                "hard_top1_type_error":float(selected[hard].mean()) if hard.any() else None})
            decoded=kq.decode_pool_actions(pool,rank[:,0,None],vs,theta_source="src")[:,0,:,:6]
            seed_mses.append(np.mean((decoded-target)**2,axis=(1,2)))
            scores_by_seed.append(sc);attn_by_seed.append(att)
            del model
        results[cond]={"selected_config":cfg,"seed_results":seed_rows,
            "mean_top1_type_error":float(np.mean([r["top1_type_error"] for r in seed_rows])),
            "mean_recovery":float(np.mean([r["recovery"] for r in seed_rows])),
            "mean_spearman":float(np.mean([r["spearman_mean"] for r in seed_rows])),
            "hard_top1_type_error":float(np.mean([r["hard_top1_type_error"] for r in seed_rows])),
            "action_mse_seed_arrays":seed_mses}
        preds[f"scores_{cond}"]=np.mean(scores_by_seed,axis=0).astype(np.float32)
        if cond.startswith("B-"):
            a=np.mean([x[0] for x in attn_by_seed],axis=0); valid=attn_by_seed[0][1]
            groups=np.load(UNSEEN_ROWS/"groups.npy",mmap_mode="r")[0]
            if cfg["layer"]=="final": labels=groups[np.flatnonzero(groups>=0)]
            else:
                g=np.full(328,2,np.int8);g[:64]=0;g[64:128]=1;labels=g
            results[cond]["attention_mass_by_latent_token"]={
                f"latent_{slot}":{str(int(g)):float((a[:,slot,:]*valid[:,None,:]*(labels[None,None,:]==g)).sum(-1).mean())
                    for g in (0,1,2) if np.any((labels[None,:]==g)&valid)}
                for slot in range(a.shape[1])}
    # Baselines: P5 model above is the learned control; fixed reported values
    # are retained as historical references, not blended into this split.
    baseline_hist=np.asarray(evnp["hist"],np.float32)
    target=np.asarray(unseen["actions"][ue,:,:6],np.float32)
    hold_last=np.repeat(baseline_hist[:,-1:, :6],10,axis=1)
    baseline_mse=np.mean((hold_last-target)**2,axis=(1,2))
    oracle_err=np.min(np.where(evdata["allowed"],errors,np.inf),axis=1)
    results["reference"]={"hold_last_reported":{"all":.114,"hard":.201},"P5_reported":{"all":.111,"hard":.226},
        "oracle_type_reported":{"all":.079,"hard":.164},"rows":len(ue),"hard_rows":int(hard.sum()),
        "previous_action_hold_last_recomputed_mse":{"all":float(baseline_mse.mean()),"hard":float(baseline_mse[hard].mean()) if hard.any() else None},
        "oracle_type_error_mean":float(oracle_err.mean()),
        "eval_ground_truth_read_only_at_final_scoring":True}
    # Decoding selected templates with their saved theta preserves the K1
    # action-space scoring convention. This is a fixed stored-scale baseline;
    # the experiment's primary selection still uses physical S-shared type error.
    action_mse={}
    for cond in CONDITIONS:
        seed_mses=results[cond].pop("action_mse_seed_arrays")
        action_mse[cond]={"all_mean":float(np.mean([x.mean() for x in seed_mses])),
            "hard_mean":float(np.mean([x[hard].mean() for x in seed_mses])) if hard.any() else None,
            "seed_means":[float(x.mean()) for x in seed_mses],
            "episode_bootstrap_vs_hold_last":_bootstrap_diff(np.mean(seed_mses,axis=0),baseline_mse,episode),
            "episode_bootstrap_vs_hold_last_hard":_bootstrap_diff(np.mean(seed_mses,axis=0)[hard],baseline_mse[hard],episode[hard]) if hard.any() else None}
    results["action_mse_stored_scale"]=action_mse
    # Robustness is assessed only for the method chosen on the LIBERO-40
    # selection tasks. Perturbations are deterministic and never use Eval
    # actions: Gaussian noise is 10% of each history feature's fit-set SD;
    # replacement uses a fixed random, cross-task W memory movement.
    best_by_selection=min(CONDITIONS,key=lambda c:(selection["selected"][c]["selection_mean_top1_type_error"],c))
    fit_data=_load_fit(out);fit_hist=np.asarray(fit_data["history"],np.float32)[np.asarray(fit_data["fit"],bool)]
    noisy_hist,replaced_hist,replace_ix=_history_perturbations(evdata["history"],fit_hist,
        evdata["allowed"],pool["shape"],seed=4101)
    best_cfg=selection["selected"][best_by_selection];robust={"selected_on":"LIBERO-40 holdout-task mean top-1 type error",
        "condition":best_by_selection,"history_noise":"Gaussian, per-feature SD from fit histories × 0.10",
        "replacement":"fixed-seed random allowed W candidate movement; cross-task candidate mask",
        "clean":{},"noise":{},"other_unit":{}}
    for variant,hist_override in (("clean",evdata["history"]),("noise",noisy_hist),("other_unit",replaced_hist)):
        ss=[]
        for seed in SEEDS:
            model,_=_torch_load(out/f"models/{best_by_selection.replace('-','_')}_s{seed}.pt",device)
            sc,_,_=score_eval(model,best_by_selection,best_cfg,history_values=hist_override);ss.append(sc);del model
        sc=np.mean(ss,axis=0);robust[variant]=_metric_retrieval(sc,errors,evdata["allowed"],evdata["scene"])
    results["history_robustness"]=robust
    # Paired bootstrap on type-error primary metric as an auxiliary effect size.
    best=min(CONDITIONS,key=lambda c:results[c]["mean_top1_type_error"])
    best_rank=np.argsort(-preds[f"scores_{best}"],axis=1,kind="stable")[:,0]
    best_mse=errors[np.arange(len(ue)),best_rank]
    results["best_type_error_vs_hold_last_action_mse_note"]={"condition":best,"type_error_mean":float(best_mse.mean()),
        "note":"These metrics are on different scales and are not subtracted."}
    top1={c:errors[np.arange(len(ue)),np.argsort(-preds[f"scores_{c}"],axis=1)[:,0]] for c in CONDITIONS}
    _atomic_npz(dest/"per_query.npz",task=task,episode=episode,episode_local=episode_local,scene=evdata["scene"],hard=hard,error=errors,allowed=evdata["allowed"],
        hold_last_mse=baseline_mse,**top1,**preds)
    summary={"status":"complete","rows":len(ue),"hard_rows":int(hard.sum()),"methods":results,"best_by_eval_type_error":best,
        "selection_frozen_before_eval":True,"eval_used_for_selection":False,"final_eval_passes":1,
        "note":"P5/oracle historical MSE values are reference-only; this experiment reports top-1 type errors and final action MSE baseline recomputation."}
    _json(dest/"summary.json",summary);return summary


def report(out=OUT):
    out=Path(out);sel=json.loads((out/"selection.json").read_text()) if (out/"selection.json").exists() else {}
    ev=json.loads((out/"eval/summary.json").read_text()) if (out/"eval/summary.json").exists() else {}
    lines=["# Query scene invariance", "", "## 5条件の結果", "",
      "| 条件 | 選択設定 | 選択 top-1型誤差 | Eval top-1型誤差 | Eval回収率 | 場面抜き差 (held−fit) | Eval MSE6 all (差CI) | Eval MSE6 hard (差CI) |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
    cvp=out/"cv/summary.json"; cv={}
    if cvp.exists(): cv=json.loads(cvp.read_text())
    folds=cv.get("folds",[])
    for c in CONDITIONS:
        r=ev.get("methods",{}).get(c,{}); cfg=sel.get("selected",{}).get(c,{})
        am=ev.get("methods",{}).get("action_mse_stored_scale",{}).get(c,{})
        ff=[f for f in folds if f["condition"]==c]
        gap=float(np.mean([f["held_scene_metrics"]["top1_type_error"]-f["trained_scene_metrics"]["top1_type_error"] for f in ff])) if ff else None
        selected_err=cfg.get("selection_mean_top1_type_error")
        def fmt_diff(v):
            if not v:return "未実行"
            return f"{v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
        allv=am.get("all_mean"); hardv=am.get("hard_mean")
        lines.append(f"| {c} | K={cfg.get('k')} / {cfg.get('layer')} / τ={cfg.get('teacher_tau')} | {selected_err if selected_err is not None else '未実行'} | {r.get('mean_top1_type_error','未実行')} | {r.get('mean_recovery','未実行')} | {gap if gap is not None else '未実行'} | {allv if allv is not None else '未実行'} ({fmt_diff(am.get('episode_bootstrap_vs_hold_last'))}) | {hardv if hardv is not None else '未実行'} ({fmt_diff(am.get('episode_bootstrap_vs_hold_last_hard'))}) |")
    ref=ev.get("methods",{}).get("reference",{})
    if ref:
        hist=ref.get("previous_action_hold_last_recomputed_mse",{})
        lines += ["", f"参照値（履歴報告値）：hold-last .114/.201、P5 .111/.226、oracle shape .079/.164（all/hard）。今回の再計算 hold-last は {hist.get('all')} / {hist.get('hard')}。履歴値と再計算値は別表記。"]
    lines += ["", "## 場面抜き交差検証", ""]
    if cvp.exists():
        lines += ["各 fit-scene fold はその scene の query を評価し、比較群として残りの fit scenes から scene ごとに固定乱数で最大128窓を抽出した。各 fold のモデルは同じ選択設定で2 epoch学習し、L1 の τ_pos もその fold の訓練場面だけで計算した。", "",
            "| 条件 | 訓練場面 top-1誤差 | 外した場面 top-1誤差 | 外した−訓練 | 訓練回収率 | 外した回収率 |",
            "|---|---:|---:|---:|---:|---:|"]
        for c in CONDITIONS:
            ff=[f for f in folds if f["condition"]==c]
            tr=[f["trained_scene_metrics"]["top1_type_error"] for f in ff]
            ho=[f["held_scene_metrics"]["top1_type_error"] for f in ff]
            trr=[f["trained_scene_metrics"]["recovery"] for f in ff];hor=[f["held_scene_metrics"]["recovery"] for f in ff]
            if ff: lines.append(f"| {c} | {np.mean(tr):.6f} | {np.mean(ho):.6f} | {np.mean(np.asarray(ho)-np.asarray(tr)):.6f} | {np.mean(trr):.4f} | {np.mean(hor):.4f} |")
    else: lines += ["CVは未完了。途中記録は `cv/` にある。"]
    lines += ["", "## 読み方", ""]
    if sel.get("selected") and ev.get("methods"):
        m=ev["methods"]
        l1a=m.get("A-L1",{}).get("mean_top1_type_error");l0a=m.get("A-L0",{}).get("mean_top1_type_error")
        l1b=m.get("B-L1",{}).get("mean_top1_type_error");l0b=m.get("B-L0",{}).get("mean_top1_type_error")
        if l1a is not None and l0a is not None and l1b is not None and l0b is not None:
            lines.append(f"- L1 の Eval top-1型誤差差（L1−L0）：A {l1a-l0a:+.6f}、B {l1b-l0b:+.6f}。負なら改善。")
        if folds:
            by={c:float(np.mean([f["held_scene_metrics"]["top1_type_error"]-f["trained_scene_metrics"]["top1_type_error"] for f in folds if f["condition"]==c])) for c in CONDITIONS}
            lines.append("- 場面をまたぐ効果は held−fit の差を条件間で読む。差が小さいこと自体を同等性とは解釈しない。場面別の分布は下表の平均に加え `cv/summary.json` に保存。")
            lines.append(f"- held−fit 差（型誤差）の P5/A/B: {by.get('P5-L0')} / {by.get('A-L0')} / {by.get('B-L0')}。")
        lines.append("- 各条件の選択設定は指定 holdout tasks の平均 top-1型誤差で固定した。最終 Eval は設定選択に使っていない。")
    rb=ev.get("methods",{}).get("history_robustness")
    if rb:
        lines += ["", f"最良条件の履歴頑健性（選択条件: {rb['condition']}）", "",
            "| 履歴 | top-1型誤差 | 回収率 | Spearman |", "|---|---:|---:|---:|"]
        for v in ("clean","noise","other_unit"):
            x=rb.get(v,{})
            lines.append(f"| {v} | {x.get('top1_type_error')} | {x.get('recovery')} | {x.get('spearman_mean')} |")
        lines.append(f"履歴ノイズ: {rb['history_noise']}。置換: {rb['replacement']}。")
    lines += ["", "## 制約と解釈", "", "- 学習・設定選択は LIBERO-40 の指定 holdout tasks のみを基準にし、LIBERO-90 Eval の正解は最後の採点以外に読まない。", "- Eval の P5 / oracle / hold-last 比較は、歴史値と本実験内で再計算した値を区別する。", "- 利用した中間特徴は既存の層16/17 prefix V キャッシュ（2×2画像token pooling＋有効言語token）。全層の完全token出力ではない。最終層は全712 token cache。", "- scene mapping は LIBERO-10 の名前にある XML scene と、legacy goal/object/spatial の各 shared suite scene を使用。", "- CV は fit scene 群のみの leave-one-scene-out で、各 fold に対して最大128 query/scene を固定抽出した。", "- `/work/docs/memory_pilot_gates.md` は環境に存在しなかった。", ""]
    p=out/"report.md";p.write_text("\n".join(lines));return p


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("prepare","smoke","tune","train","cv","evaluate","report","step1","report-step1","all"))
    ap.add_argument("--out",type=Path,default=OUT);ap.add_argument("--epochs",type=int,default=None);ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--device",default="cuda");ap.add_argument("--max-scenes",type=int,default=None)
    a=ap.parse_args(argv);out=a.out;Path(out).mkdir(parents=True,exist_ok=True);_run_manifest(out)
    if a.stage in ("prepare","smoke","all"):prepare(out)
    if a.stage=="smoke":smoke(out,device=a.device)
    if a.stage in ("tune","all"):tune(out,epochs=a.epochs or 3,device=a.device,batch=a.batch)
    if a.stage in ("train","all"):train(out,epochs=a.epochs or 8,device=a.device,batch=a.batch)
    if a.stage in ("cv","all"):cv(out,epochs=a.epochs or 2,device=a.device,batch=a.batch,max_scenes=a.max_scenes)
    if a.stage in ("evaluate","all"):evaluate(out,device=a.device)
    if a.stage=="step1":step1(out/"step1",data_root=out,device=a.device,batch=a.batch,epochs=a.epochs or 8)
    if a.stage=="report-step1":print(step1_report(out/"step1"))
    if a.stage in ("report","all"):print(report(out))


if __name__ == "__main__": main()
