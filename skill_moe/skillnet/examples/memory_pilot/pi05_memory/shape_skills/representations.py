"""Physical-space canonicalization, shape codes, decoding, and variable-D M6."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE = 32


def _unnormalize_physical(actions, q01, q99):
    return (np.asarray(actions, np.float32) + 1.) * .5 * (
        np.asarray(q99, np.float32) - np.asarray(q01, np.float32) + 1e-6
    ) + np.asarray(q01, np.float32)


def resample(x, n):
    x = np.asarray(x, np.float32)
    if len(x) == n:
        return x.copy()
    if not len(x):
        return np.zeros((n,) + x.shape[1:], np.float32)
    src = np.linspace(0., 1., len(x)); dst = np.linspace(0., 1., n)
    flat = x.reshape(len(x), -1)
    y = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return y.reshape((n,) + x.shape[1:]).astype(np.float32)


def rotation_to_x(v):
    """Proper minimal rotation mapping a physical direction to +X."""
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return np.eye(3, dtype=np.float32)
    a = v / n; b = np.asarray([1., 0., 0.])
    c = float(np.clip(a @ b, -1., 1.))
    if c > 1. - 1e-12:
        return np.eye(3, dtype=np.float32)
    if c < -1. + 1e-12:
        return np.diag([-1., -1., 1.]).astype(np.float32)
    cross = np.cross(a, b)
    s = float(np.linalg.norm(cross))
    axis = cross / s
    K = np.asarray([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    # Rodrigues for a unit axis: R = I + sin(theta)[u]x +
    # (1-cos(theta))[u]x^2. Omitting sin(theta) is invisible at 90 degrees
    # but creates a non-orthogonal matrix at general directions.
    R = np.eye(3) + s * K + (1. - c) * (K @ K)
    return R.astype(np.float32)


def procrustes_rotation(x, reference):
    """Proper Kabsch rotation of the xyz path into a fixed reference path."""
    x = np.asarray(x, np.float64); y = np.asarray(reference, np.float64)
    x = x - x.mean(0, keepdims=True); y = y - y.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(x.T @ y)
    sign = np.linalg.det(u @ vt)
    return (u @ np.diag([1., 1., sign]) @ vt).astype(np.float32)


def canonicalize(actions_phys, *, procrustes_ref=None):
    a = np.asarray(actions_phys, np.float32)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError("actions must be [time,>=7]")
    p = a[:, :6].copy()
    total = p[:, :3].sum(0)
    scale = float(np.linalg.norm(total))
    if scale < 1e-5:
        scale = float(np.sqrt(np.mean(p*p)))
    if scale < 1e-5:
        scale = 1.
    R = rotation_to_x(total)
    if procrustes_ref is not None:
        path = resample(np.cumsum(p[:, :3], axis=0), PHASE)
        R = procrustes_rotation(path, procrustes_ref).T
    canon = p.copy()
    canon[:, :3] = p[:, :3] @ R.T / scale
    canon[:, 3:6] = p[:, 3:6] @ R.T / scale
    grip = resample(a[:, 6:7], PHASE)[:, 0]
    pose = {"scale": scale, "rotation": R, "length": len(a), "grip": grip}
    return canon, pose


def apply_pose(canonical, pose):
    x = np.asarray(canonical, np.float32).copy()
    R = np.asarray(pose["rotation"], np.float32); scale = float(pose["scale"])
    x[:, :3] = (x[:, :3] * scale) @ R
    x[:, 3:6] = (x[:, 3:6] * scale) @ R
    return x


def _phase_raw(action):
    return resample(np.asarray(action, np.float32)[:, :7], PHASE)


def _phase_canon(action, *, procrustes_ref=None):
    can, pose = canonicalize(action, procrustes_ref=procrustes_ref)
    return resample(can, PHASE), pose


def pca_fit(x, k, whiten=False):
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(int(k), vt.shape[0])
    comp = vt[:k]
    var = np.maximum(singular[:k] ** 2 / max(1, len(x)-1), 1e-12)
    return {"kind": "pca", "mean": mean.astype(np.float32), "components": comp.astype(np.float32),
            "variance": var.astype(np.float32), "whiten": bool(whiten), "code_dim": k}


def pca_encode(model, x):
    z = (np.asarray(x, np.float32) - model["mean"]) @ model["components"].T
    return z / np.sqrt(model["variance"]) if model["whiten"] else z


def pca_decode(model, code):
    z = np.asarray(code, np.float32)
    if model["whiten"]: z = z * np.sqrt(model["variance"])
    return z @ model["components"] + model["mean"]


def raw_pca_input(actions_phys, q01, q99, *, native_length=None):
    """Officially normalize raw actions, preserving native S0 windows.

    Variable-length R1 inputs are phase-resampled after normalization. The
    fixed 10-step control is the original 10x7 vector (70 values), as in the
    established PCA24 reconstruction baseline.
    """
    try:
        from .data import normalize_physical
    except ImportError:
        from data import normalize_physical
    a = np.asarray(actions_phys, np.float32)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError("raw PCA actions must be [time,>=7]")
    x = normalize_physical(a[:, :7], np.asarray(q01, np.float32), np.asarray(q99, np.float32))
    if native_length is not None:
        if len(x) != int(native_length):
            raise ValueError(f"expected a native {native_length}-step PCA window, got {len(x)}")
        return x
    return resample(x, PHASE)


def dmp_basis(n_basis, n=PHASE):
    phase = np.linspace(0., 1., n, dtype=np.float64)
    centers = np.linspace(0., 1., n_basis)
    width = max(float(n_basis) ** 1.5, 1.)
    psi = np.exp(-width * (phase[:, None] - centers[None]) ** 2)
    return (psi / np.maximum(psi.sum(1, keepdims=True), 1e-12)).astype(np.float32)


def dmp_encode(canonical, n_basis):
    # Fit forcing weights to the trajectory residual around a linear goal path.
    v = resample(canonical, PHASE)
    y = np.cumsum(v, axis=0)
    y -= y[:1]
    goal = y[-1].copy()
    phase = np.linspace(0., 1., PHASE, dtype=np.float32)[:, None]
    forcing = y - phase * goal[None]
    B = dmp_basis(n_basis)
    w = np.linalg.lstsq(B.astype(np.float64), forcing.astype(np.float64), rcond=1e-6)[0]
    return w.astype(np.float32), goal.astype(np.float32)


def dmp_decode(weights, goal, length):
    n_basis = len(weights); B = dmp_basis(n_basis, PHASE)
    phase = np.linspace(0., 1., PHASE, dtype=np.float32)[:, None]
    path = phase * np.asarray(goal, np.float32)[None] + B @ np.asarray(weights, np.float32)
    vel = np.diff(np.concatenate([np.zeros((1, 6), np.float32), path], axis=0), axis=0)
    return resample(vel, int(length))


def spline_encode(canonical, n_control):
    trajectory = np.cumsum(resample(canonical, PHASE), axis=0)
    trajectory -= trajectory[:1]
    ix = np.rint(np.linspace(0, PHASE-1, n_control)).astype(int)
    return trajectory[ix].astype(np.float32)


def spline_decode(control, length):
    c = np.asarray(control, np.float32)
    src = np.linspace(0., 1., len(c)); mid = np.linspace(0., 1., PHASE)
    path = np.stack([np.interp(mid, src, c[:, j]) for j in range(6)], axis=1)
    vel = np.diff(np.concatenate([np.zeros((1, 6), np.float32), path], axis=0), axis=0)
    return resample(vel, int(length))


def model_code(model, actions_phys):
    """Encode one segment; returns code, pose and family-specific model extras."""
    kind = model["kind"]
    if kind == "canonical_raw":
        can, pose = _phase_canon(actions_phys)
        return can.reshape(-1), pose, None
    if kind in {"pca_raw", "pca_raw_white"}:
        x = raw_pca_input(actions_phys, model["q01"], model["q99"],
                          native_length=model.get("pca_input_length")).reshape(-1)
        return pca_encode(model["pca"], x), {}, None
    if kind in {"pca_canonical", "pca_gripsep", "procrustes_pca"}:
        can, pose = _phase_canon(actions_phys, procrustes_ref=model.get("reference"))
        z = pca_encode(model["pca"], can.reshape(-1))
        return z, pose, None
    if kind == "dmp":
        can, pose = _phase_canon(actions_phys)
        weights, goal = dmp_encode(can, model["n_basis"])
        pose["goal"] = goal
        return weights.reshape(-1), pose, None
    if kind == "spline":
        can, pose = _phase_canon(actions_phys)
        control = spline_encode(can, model["n_control"])
        return control.reshape(-1), pose, None
    if kind == "m1":
        import torch
        sample = model["normalizer"](_phase_raw(actions_phys))[None]
        state = np.asarray(model["state"], np.float32)[None]
        feat = np.asarray(model["feature"], np.float32)[None]
        b, r = model["codec"].encode(sample, state)
        pose = {"length": len(actions_phys), "scale": 1., "rotation": np.eye(3, dtype=np.float32),
                "grip": resample(actions_phys[:, 6:7], PHASE)[:, 0], "r": r[0],
                "state": state[0], "feature": feat[0]}
        return b[0].astype(np.float32), pose, None
    raise ValueError(kind)


def decode_canonical(model, code, pose, length):
    kind = model["kind"]
    if kind == "canonical_raw":
        can = np.asarray(code, np.float32).reshape(PHASE, 6)
        return resample(can, length), resample(np.asarray(pose["grip"])[:, None], length)[:, 0]
    if kind in {"pca_raw", "pca_raw_white"}:
        n = int(model.get("pca_input_length") or PHASE)
        raw = pca_decode(model["pca"], code).reshape(n, 7)
        raw = _unnormalize_physical(raw, model["q01"], model["q99"])
        # For type-only/aligned diagnostics, remove the decoded movement pose.
        can, _ = canonicalize(raw)
        return resample(can, length), resample(raw[:, 6:7], length)[:, 0]
    if kind in {"pca_canonical", "pca_gripsep", "procrustes_pca"}:
        can = pca_decode(model["pca"], code).reshape(PHASE, 6)
        return resample(can, length), resample(np.asarray(pose["grip"])[:, None], length)[:, 0]
    if kind == "dmp":
        can = dmp_decode(np.asarray(code).reshape(model["n_basis"], 6), pose["goal"], length)
        return can, resample(np.asarray(pose["grip"])[:, None], length)[:, 0]
    if kind == "spline":
        control=pca_decode(model["pca"],code).reshape(model["n_control"],6)
        can = spline_decode(control, length)
        return can, resample(np.asarray(pose["grip"])[:, None], length)[:, 0]
    if kind == "m1":
        codec = model["codec"]
        b = np.asarray(code, np.float32)[None]
        action = codec.decode(b, np.asarray(pose["r"], np.float32)[None],
                              np.asarray(pose["feature"], np.float32)[None],
                              np.asarray(pose["state"], np.float32)[None])[0]
        physical = model["unnormalizer"](action[:, :7])
        can, _ = canonicalize(physical)
        return resample(can, length), resample(physical[:, 6:7], length)[:, 0]
    raise ValueError(kind)


def decode_original(model, code, pose, length, *, q01, q99):
    kind = model["kind"]
    if kind in {"pca_raw", "pca_raw_white"}:
        n = int(model.get("pca_input_length") or PHASE)
        x = pca_decode(model["pca"], code).reshape(n, 7)
        return resample(_unnormalize_physical(x, q01, q99), length)
    can, grip = decode_canonical(model, code, pose, length)
    if kind == "m1":
        codec = model["codec"]
        b = np.asarray(code, np.float32)[None]
        y = codec.decode(b, np.asarray(pose["r"], np.float32)[None],
                         np.asarray(pose["feature"], np.float32)[None],
                         np.asarray(pose["state"], np.float32)[None])[0]
        return np.concatenate([model["unnormalize"](y[:, :6]), model["unnormalize"](y[:, 6:7])], axis=1)
    restored = apply_pose(can, pose)
    return np.concatenate([restored, resample(grip[:, None], length)], axis=1)


def decode_from_bundle(bundle_dir, codes, pose, length):
    d = Path(bundle_dir)
    meta = json.loads((d / "manifest.json").read_text())
    model_file = np.load(d / "model.npz", allow_pickle=False)
    model = {k: model_file[k] for k in model_file.files}
    model["kind"] = meta["kind"]
    p = {k: pose[k] for k in pose.files} if hasattr(pose, "files") else pose
    if meta["kind"] in {"pca_raw", "pca_raw_white", "pca_canonical", "pca_gripsep", "procrustes_pca", "spline"}:
        model["pca"]={"mean":model_file["mean"],"components":model_file["components"],
                      "variance":model_file["variance"],"whiten":bool(model_file["whiten"])}
        model["reference"]=model_file["reference"] if "reference" in model_file.files else None
    if meta["kind"] in {"pca_raw", "pca_raw_white"}:
        model["q01"] = np.asarray(meta["q01"], np.float32)
        model["q99"] = np.asarray(meta["q99"], np.float32)
        model["pca_input_length"] = meta.get("pca_input_length")
    if meta["kind"]=="dmp":model["n_basis"]=int(meta["n_basis"])
    if meta["kind"]=="spline":model["n_control"]=int(meta["n_control"])
    lengths=np.asarray(length if np.ndim(length)>0 else [length],np.int64).reshape(-1)
    if len(lengths)==1 and np.asarray(codes).ndim>1 and len(np.asarray(codes))>1:
        lengths=np.repeat(lengths,len(np.asarray(codes)))
    if meta["kind"]=="m1":
        import torch
        payload=torch.load(d/"codec.pt",map_location="cpu",weights_only=False)
        stats={k:np.asarray(v,dtype=np.float32) for k,v in payload["stats"].items()}
        def linear_stack(name,x):
            z=torch.as_tensor(x,dtype=torch.float32)
            sd=payload["nets"][name]
            for i in (0,2,4):
                z=torch.nn.functional.linear(z,sd[f"{i}.weight"],sd[f"{i}.bias"])
                if i<4:z=torch.nn.functional.gelu(z)
            return z.numpy()
        b=np.asarray(codes,np.float32).reshape(-1,int(meta["code_dim"]))
        r=np.asarray(p["r"],np.float32).reshape(len(b),-1)
        s=np.asarray(p["state"],np.float32).reshape(len(b),-1)
        sn=(s-stats["sm"])/stats["ss"]
        base=linear_stack("base_decoder",np.concatenate([b,sn],axis=1))
        residual=linear_stack("residual_decoder",r)
        y=(base+residual)*stats["ass"]+stats["am"]
        y=y.reshape(len(b),10,32)[:,:,:7]
        phys=unnormalize(y,np.asarray(meta["q01"],np.float32),np.asarray(meta["q99"],np.float32))
        physical=[resample(a,int(lengths[i])) for i,a in enumerate(phys)]
    else:
        c=np.asarray(codes,np.float32)
        if c.ndim==1:c=c[None]
        physical=[]
        for i,code in enumerate(c):
            pi={k:(np.asarray(v)[i] if np.asarray(v).ndim>0 and np.asarray(v).shape[0]==len(c) else v)
                for k,v in p.items()}
            physical.append(decode_original(model,code,pi,int(lengths[i]),
                q01=np.asarray(meta["q01"]),q99=np.asarray(meta["q99"])))
    try:
        from .data import normalize_physical
    except ImportError:
        from data import normalize_physical
    normalized=[normalize_physical(x,np.asarray(meta["q01"],np.float32),np.asarray(meta["q99"],np.float32)) for x in physical]
    if len(set(int(x) for x in lengths))==1:
        return np.stack(normalized).astype(np.float32)
    return normalized


def m6_fit_generic(values, tasks, episodes, *, kappa, seed=20261104, min_support=10, min_episodes=3):
    """M6 assign/pending/register/merge for arbitrary descriptor dimension."""
    x = np.asarray(values, np.float32)
    if x.ndim != 2 or not len(x): raise ValueError("M6 needs a nonempty [N,D] matrix")
    d = x.shape[1]; rng = np.random.default_rng(seed + 6046)
    pair = rng.integers(0, len(x), size=(64, 2))
    random = np.mean(np.square(x[pair[:,0]] - x[pair[:,1]]), axis=1)
    eps = max(float(kappa) * float(np.median(random)), 1e-12)
    def dist(a, b): return float(np.mean(np.square(np.asarray(a)-np.asarray(b))))
    def nearest(v, centers):
        if not centers: return None, float("inf")
        c=np.asarray(centers,np.float32); l2=np.sum((c-v[None])**2,axis=1)
        # Euclidean and MSE rankings are identical at fixed D. np.argmin is
        # stable at the first tied center and avoids sorting every unit.
        j=int(np.argmin(l2))
        return j,float(l2[j]/d)
    centers=[]; support=[]; epsets=[]; members=[]; labels=np.full(len(x),-1,np.int32)
    pending=[]; ages={}; pending_since=0; nid=0; expired=0
    def flush(force=False):
        nonlocal pending,pending_since,expired
        if not pending or (not force and pending_since<200):return
        pending_since=0; clusters=[]
        for item in pending:
            j,d0=nearest(item["value"],[c["center"] for c in clusters])
            if j is not None and d0<=eps:
                c=clusters[j];c["values"].append(item);c["eps"].add(item["episode_key"])
                c["center"]=np.mean([q["value"] for q in c["values"]],axis=0).astype(np.float32)
            else:clusters.append({"center":item["value"].copy(),"values":[item],"eps":{item["episode_key"]}})
        registered={}
        for c in clusters:
            vals=c["values"]
            if len(vals)<min_support or len(c["eps"])<min_episodes:continue
            j,d0=nearest(c["center"],centers)
            if j is not None and d0<=eps:
                total=support[j]+len(vals);centers[j]=((centers[j]*support[j]+c["center"]*len(vals))/total).astype(np.float32)
                support[j]=total;epsets[j].update(c["eps"]);members[j].extend(v["local"] for v in vals);uid=j
            else:
                uid=len(centers);centers.append(c["center"].copy());support.append(len(vals));epsets.append(set(c["eps"]));members.append([v["local"] for v in vals])
            for item in vals:registered[item["id"]]=uid
        keep=[]
        for item in pending:
            if item["id"] in registered:
                labels[item["local"]]=registered[item["id"]];ages.pop(item["id"],None)
            else:
                pid=item["id"];ages[pid]=ages.get(pid,0)+200
                if ages[pid]>=1000:ages.pop(pid,None);expired+=1
                else:keep.append(item)
        pending=keep
    for local in rng.permutation(len(x)):
        val=x[local];j,d0=nearest(val,centers);epkey=(int(tasks[local]),int(episodes[local]))
        if j is not None and d0<=eps:
            total=support[j]+1;centers[j]=((centers[j]*support[j]+val)/total).astype(np.float32)
            support[j]=total;epsets[j].add(epkey);members[j].append(int(local));labels[local]=j
        else:
            pending.append({"id":nid,"local":int(local),"value":val.copy(),"episode_key":epkey});ages[nid]=0;nid+=1;pending_since+=1
        flush()
    flush(True)
    return {"row_units":labels,"centers":np.asarray(centers,np.float32).reshape(-1,d),
            "members_local":[np.asarray(m,np.int64) for m in members],"support":np.asarray(support,np.int32),
            "episode_sets":epsets,"epsilon":eps,"median_random_distance":float(np.median(random)),
            "registered":len(centers),"pending_local":np.asarray([v["local"] for v in pending],np.int64),"expired":expired}


def medoid_generic(members, values):
    ix=np.asarray(members,np.int64);x=np.asarray(values[ix],np.float64)
    # The mean pairwise squared distance for point x_i is
    #   mean_j ||x_i - x_j||^2 = ||x_i - mean(x)||^2 + constant.
    # Thus nearest-to-centroid selects the exact same observed medoid while
    # avoiding a quadratic [members,members,D] distance tensor.
    center=x.mean(axis=0,keepdims=True)
    d=np.sum((x-center)**2,axis=1)
    return int(ix[np.argmin(d)])
