"""Offline RoboCasa skill-memory and P5 retrieval experiment.

Only atomic pretrain-human demonstrations are read.  The policy-memory path
does not use task text or success labels; text is retained solely for the
optional pi05 baseline.  Data are 20 Hz and action channels follow LeRobot's
12-D order documented in docs/robocasa365.md.
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
import units

TASKS = (
    "PickPlaceCounterToCabinet", "PickPlaceSinkToCounter", "OpenDrawer",
    "CloseFridge", "TurnOnSinkFaucet", "TurnOffStove",
)
STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
DATA_ROOT = STATE / "data/robocasa/v1.0/pretrain/atomic"
OUT_DEFAULT = STATE / "outputs/robocasa-memory/memory-01"
N_DEMOS = 100
N_MEMORY = 80
SPLIT_SEED = 20261110
SEEDS = (0, 1, 2)
HORIZON = 50
WINDOW = 10
TYPE_DIMS = slice(5, 11)


def action_storage_to_policy(action):
    """LeRobot [base4,mode,EEF6,grip] -> fork [EEF6,grip,base4,mode]."""
    a = np.asarray(action)
    if a.shape[-1] != 12:
        raise ValueError(f"expected 12 action channels, got {a.shape}")
    return np.concatenate([a[..., 5:11], a[..., 11:12], a[..., 0:5]], axis=-1)


def action_policy_to_storage(action):
    """Fork [EEF6,grip,base4,mode] -> LeRobot [base4,mode,EEF6,grip]."""
    a = np.asarray(action)
    if a.shape[-1] != 12:
        raise ValueError(f"expected 12 action channels, got {a.shape}")
    return np.concatenate([a[..., 7:12], a[..., 0:6], a[..., 6:7]], axis=-1)


def state_storage_to_policy(state):
    """LeRobot [base pose7,EEF pose7,grip2] -> fork [EEF pose7,base pose7,grip2]."""
    s = np.asarray(state)
    if s.shape[-1] != 16:
        raise ValueError(f"expected 16 state channels, got {s.shape}")
    return np.concatenate([s[..., 7:14], s[..., 0:7], s[..., 14:16]], axis=-1)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    tmp.replace(path)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def _log(out: Path, **record):
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with (out / "progress.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def split_episode_ids(ids, *, seed=SPLIT_SEED):
    """Fixed per-task episode split: 80 memory/train and 20 held-out."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 2:
        raise ValueError("at least two episode IDs are needed for an episode-level split")
    p = np.random.default_rng(int(seed)).permutation(ids)
    n_memory = min(N_MEMORY, max(1, int(round(.8 * len(ids)))))
    return np.sort(p[:n_memory]), np.sort(p[n_memory:])


def resample_rows(x, n=WINDOW):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not len(x):
        raise ValueError("resample_rows expects a non-empty [T,D] array")
    if len(x) == int(n):
        return x.copy()
    pos = np.linspace(0, len(x) - 1, int(n), dtype=np.float64)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    frac = (pos - lo)[:, None].astype(np.float32)
    return x[lo] * (1 - frac) + x[hi] * frac


def encode_shape(action):
    """Scale-only DCT-3 type from RoboCasa EEF position+rotation commands."""
    from scipy.fft import dct, idct
    a = np.asarray(action, np.float32)
    if a.ndim != 2 or a.shape[1] != 12 or len(a) == 0:
        raise ValueError("action must have shape [T,12]")
    motion = resample_rows(a[:, TYPE_DIMS], WINDOW)
    coeff = dct(motion.astype(np.float64), type=2, norm="ortho", axis=0)[:3]
    norm = float(np.linalg.norm(coeff))
    normalized = coeff / max(norm, 1e-9)
    full = np.zeros((WINDOW, 6), np.float64)
    full[:3] = normalized
    shape = idct(full, type=2, norm="ortho", axis=0).astype(np.float32)
    key = np.concatenate([normalized.reshape(-1), np.asarray([np.log(max(len(a), 1)) / 3.0])]).astype(np.float32)
    return shape, key


def shape_error_matrix(query_shape, candidate_shape, *, block=256):
    """S-shared error min_{s>=0} ||s*c-q||²/(10*6), with no rotation."""
    q = np.asarray(query_shape, np.float32).reshape(len(query_shape), -1)
    c = np.asarray(candidate_shape, np.float32).reshape(len(candidate_shape), -1)
    if q.shape[1] != WINDOW * 6 or c.shape[1] != WINDOW * 6:
        raise ValueError("shape representation must be [N,10,6]")
    cn = np.sum(c.astype(np.float64) ** 2, axis=1)
    out = np.empty((len(q), len(c)), np.float32)
    cf = c.astype(np.float64).T
    for start in range(0, len(q), int(block)):
        qb = q[start:start + int(block)].astype(np.float64)
        dot = np.maximum(qb @ cf, 0.0)
        err = np.sum(qb * qb, axis=1, keepdims=True) - dot * dot / np.maximum(cn[None], 1e-12)
        out[start:start + len(qb)] = (np.maximum(err, 0.0) / (WINDOW * 6)).astype(np.float32)
    return out


def _dataset_path(task):
    root = DATA_ROOT / task / "20250819" / "lerobot"
    if not (root / "meta/info.json").exists():
        candidates = sorted((DATA_ROOT / task).glob("*/lerobot/meta/info.json"))
        if not candidates:
            raise FileNotFoundError(f"LeRobot info.json not found for {task}")
        root = candidates[-1].parent.parent
    return root


def _episode_files(root):
    files = sorted((root / "data").glob("**/episode_*.parquet"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return files[:N_DEMOS]


def _read_episode(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["action", "observation.state", "frame_index"])
    action = np.asarray(table["action"].to_pylist(), np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), np.float32)
    frame = np.asarray(table["frame_index"].to_numpy(), np.int64)
    if action.shape != (len(action), 12) or state.shape != (len(state), 16):
        raise ValueError(f"bad action/state shape in {path}: {action.shape}, {state.shape}")
    if len(action) != len(state) or len(frame) != len(action):
        raise ValueError(f"row alignment failure in {path}")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError(f"non-finite observation in {path}")
    return action, state, frame


def _read_prompt(root, episode):
    p = root / "meta/episodes.jsonl"
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            if int(d["episode_index"]) == int(episode):
                return str(d["tasks"][0])
    return ""


def prepare(out: Path, *, tasks=TASKS, demo_limit=N_DEMOS, smoke=False):
    out = Path(out)
    data_dir = out / "episodes"
    data_dir.mkdir(parents=True, exist_ok=True)
    chosen_tasks = tuple(tasks)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if tuple(old["tasks"]) != chosen_tasks or int(old["demo_limit"]) != int(demo_limit):
            raise ValueError("existing prepared output has a different task/demo configuration")
        # The manifest is committed before derived pools are written. If a run
        # stopped during pool construction, re-enter that resumable stage.
        build_pools(out)
        return old
    task_manifest = []
    episode_info = []
    for task_index, task in enumerate(chosen_tasks):
        root = _dataset_path(task)
        files = _episode_files(root)[:int(demo_limit)]
        if len(files) < (4 if smoke else N_DEMOS):
            raise FileNotFoundError(f"{task} has only {len(files)} demos; expected {N_DEMOS}")
        selected = files[:int(demo_limit)]
        ids = np.asarray([int(p.stem.split("_")[-1]) for p in selected], np.int64)
        train_ids, eval_ids = split_episode_ids(ids, seed=SPLIT_SEED + task_index)
        for p in selected:
            ep = int(p.stem.split("_")[-1])
            target = data_dir / f"task{task_index}_ep{ep:06d}.npz"
            if not target.exists():
                action, state, frame = _read_episode(p)
                tmp = target.with_name(target.stem + ".tmp.npz")
                np.savez_compressed(tmp, action=action, state=state, frame=frame)
                tmp.replace(target)
            with np.load(target, allow_pickle=False) as z:
                if z["action"].shape[1] != 12 or z["state"].shape[1] != 16:
                    raise ValueError(f"cached rows mismatch for {task}/{ep}")
                n = len(z["action"])
            episode_info.append({"task_id": task_index, "task": task, "episode": ep,
                "global_episode": task_index * 1000 + ep, "split": "memory" if ep in set(train_ids) else "heldout",
                "frames": n, "source": str(p), "prompt": _read_prompt(root, ep)})
        task_manifest.append({"task_id": task_index, "task": task, "root": str(root),
            "available_episodes": len(_episode_files(root)), "selected_episodes": ids.tolist(),
            "memory_episodes": train_ids.tolist(), "heldout_episodes": eval_ids.tolist()})
    manifest = {"tasks": list(chosen_tasks), "demo_limit": int(demo_limit), "seed": SPLIT_SEED,
        "memory_per_task": N_MEMORY if int(demo_limit) == N_DEMOS else int(round(.8 * int(demo_limit))),
        "heldout_per_task": int(demo_limit) - (N_MEMORY if int(demo_limit) == N_DEMOS else int(round(.8 * int(demo_limit)))),
        "fps": 20, "action_order": ["base0", "base1", "base2", "base3", "control_mode", "eef_x", "eef_y", "eef_z", "rot_x", "rot_y", "rot_z", "gripper"],
        "state_dim": 16, "demos": episode_info, "task_splits": task_manifest,
        "split_rule": "First 100 episode files by numeric ID; per-task NumPy permutation seeded with 20261110+task_id; 80 memory/train and 20 held-out. Smoke uses its explicitly recorded reduced count.",
        "labels_used_for": "Only prompt strings for optional pi05 policy baseline; no task/subtask labels enter unit cuts, P5 inputs, targets, or retrieval.",
        "heldout_actions_policy": "heldout trajectories are read for scoring and oracle selection only; never used to fit memory or P5 and never used for setting selection."}
    _json(manifest_path, manifest)
    _log(out, stage="prepare", tasks=len(chosen_tasks), episodes=len(episode_info), rows=sum(x["frames"] for x in episode_info))
    build_pools(out)
    return manifest


def _load_cached_episode(out, task_id, episode):
    p = Path(out) / "episodes" / f"task{int(task_id)}_ep{int(episode):06d}.npz"
    z = np.load(p, allow_pickle=False)
    return np.asarray(z["action"], np.float32), np.asarray(z["state"], np.float32), np.asarray(z["frame"], np.int64)


def _unit_records(action, state, frame, *, task_id, episode, split):
    units_rec = units.cut(action, min_length=10)
    out = []
    for u in units_rec:
        start, end = int(u["start"]), int(u["end"])
        if end <= start:
            continue
        hist = np.zeros((WINDOW, 12), np.float32)
        past = action[max(0, start - WINDOW):start]
        if len(past):
            hist[-len(past):] = past
        arm10 = resample_rows(action[start:end], WINDOW)
        shape, key = encode_shape(action[start:end])
        out.append({"action": action[start:end].copy(), "action10": arm10, "shape": shape, "key": key,
            "length": end - start, "task": int(task_id), "episode": int(episode),
            "global_episode": int(task_id) * 1000 + int(episode), "frame": int(frame[start]),
            "hist": hist, "state": state[start].copy(), "split": split})
    return out


def _window_records(action, state, frame, *, task_id, episode, split):
    out = []
    for start in range(0, len(action) - WINDOW + 1, WINDOW):
        end = start + WINDOW
        hist = np.zeros((WINDOW, 12), np.float32)
        past = action[max(0, start - WINDOW):start]
        if len(past):
            hist[-len(past):] = past
        chunk = action[start:end].copy()
        shape, key = encode_shape(chunk)
        out.append({"action": chunk, "action10": chunk.copy(), "shape": shape, "key": key,
            "length": WINDOW, "task": int(task_id), "episode": int(episode),
            "global_episode": int(task_id) * 1000 + int(episode), "frame": int(frame[start]),
            "hist": hist, "state": state[start].copy(), "split": split})
    return out


def _records_to_pool(records):
    flat = np.concatenate([r["action"] for r in records], axis=0).astype(np.float32)
    offsets = np.cumsum([0] + [len(r["action"]) for r in records], dtype=np.int64)
    return {"actions_flat": flat, "action_offsets": offsets,
        "action10": np.stack([r["action10"] for r in records]).astype(np.float32),
        "shape": np.stack([r["shape"] for r in records]).astype(np.float32),
        "key": np.stack([r["key"] for r in records]).astype(np.float32),
        "length": np.asarray([r["length"] for r in records], np.int32),
        "task": np.asarray([r["task"] for r in records], np.int16),
        "episode": np.asarray([r["episode"] for r in records], np.int32),
        "global_episode": np.asarray([r["global_episode"] for r in records], np.int32),
        "frame": np.asarray([r["frame"] for r in records], np.int32)}


def _records_to_queries(records):
    return {"hist": np.stack([r["hist"] for r in records]).astype(np.float32),
        "state": np.stack([r["state"] for r in records]).astype(np.float32),
        "action10": np.stack([r["action10"] for r in records]).astype(np.float32),
        "shape": np.stack([r["shape"] for r in records]).astype(np.float32),
        "length": np.asarray([r["length"] for r in records], np.int32),
        "task": np.asarray([r["task"] for r in records], np.int16),
        "episode": np.asarray([r["episode"] for r in records], np.int32),
        "global_episode": np.asarray([r["global_episode"] for r in records], np.int32),
        "frame": np.asarray([r["frame"] for r in records], np.int32)}


def _save_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def build_pools(out):
    out = Path(out); m = json.loads((out / "manifest.json").read_text())
    if (out / "pools/complete.json").exists():
        return json.loads((out / "pools/complete.json").read_text())
    data = {name: {s: [] for s in ("memory", "heldout")} for name in ("U", "W")}
    memory_ids = {(int(t["task_id"]), int(ep)) for t in m["task_splits"] for ep in t["memory_episodes"]}
    for info in m["demos"]:
        split = "memory" if (int(info["task_id"]), int(info["episode"])) in memory_ids else "heldout"
        act, st, fr = _load_cached_episode(out, info["task_id"], info["episode"])
        data["U"][split].extend(_unit_records(act, st, fr, task_id=info["task_id"], episode=info["episode"], split=split))
        data["W"][split].extend(_window_records(act, st, fr, task_id=info["task_id"], episode=info["episode"], split=split))
    stats = {}
    for family in ("U", "W"):
        pool = _records_to_pool(data[family]["memory"])
        _save_npz(out / f"pools/{family}_memory.npz", **pool)
        for split in ("memory", "heldout"):
            q = _records_to_queries(data[family][split])
            _save_npz(out / f"queries/{family}_{split}.npz", **q)
        # Training-fit episodes are chosen once, task-stratified by a seeded global permutation.
        train_ep = np.unique(np.asarray(pool["global_episode"], np.int32))
        rng = np.random.default_rng(SPLIT_SEED + (0 if family == "U" else 1))
        val_ep = np.sort(rng.choice(train_ep, size=max(1, int(round(.2 * len(train_ep)))), replace=False))
        stats[family] = {"memory_units": int(len(pool["key"])), "val_episodes": val_ep.tolist(),
            "length_frames": {k: float(v) for k, v in zip(("min", "p10", "median", "p90", "max"),
                (np.min(pool["length"]), np.quantile(pool["length"], .1), np.median(pool["length"]), np.quantile(pool["length"], .9), np.max(pool["length"])))}}
    # Pool action summaries are also used by the exact self-episode exclusion and baseline reports.
    task_rows = {}
    for family in ("U", "W"):
        p = dict(np.load(out / f"pools/{family}_memory.npz"))
        lens = p["length"]
        by_task = {}
        for tid, task in enumerate(m["tasks"]):
            ix = p["task"] == tid
            by_task[task] = {"units": int(ix.sum()), "episodes": int(len(np.unique(p["episode"][ix]))),
                "length_median_frames": float(np.median(lens[ix])) if np.any(ix) else None}
        task_rows[family] = by_task
    for info in m["demos"]:
        act, _, _ = _load_cached_episode(out, info["task_id"], info["episode"])
        info["base_nonzero_fraction"] = float(np.mean(np.abs(act[:, :4]) > 1e-8))
        info["mode_nonzero_fraction"] = float(np.mean(np.abs(act[:, 4]) > 1e-8))
    _json(out / "manifest.json", m)
    result = {"pools": stats, "by_task": task_rows,
        "base_nonzero_fraction_memory": float(np.mean([x["base_nonzero_fraction"] for x in m["demos"] if x["split"] == "memory"])),
        "mode_nonzero_fraction_memory": float(np.mean([x["mode_nonzero_fraction"] for x in m["demos"] if x["split"] == "memory"])),
        "segmentation": "10-frame minimum, local minima of 3-frame-smoothed EEF command speed plus every gripper sign transition; task annotations are unused."}
    _json(out / "pools/complete.json", result)
    _log(out, stage="build_pools", U_units=stats["U"]["memory_units"], W_windows=stats["W"]["memory_units"])
    return result


def query_features(q, fit_ids=None):
    # Demonstrations are stored in LeRobot order. Convert the history and state
    # into the same channel order that the fork policy sees and emits.
    hist = np.asarray(action_storage_to_policy(q["hist"]), np.float32)
    state = np.asarray(state_storage_to_policy(q["state"]), np.float32)
    if fit_ids is None:
        fit_ids = np.arange(len(hist))
    fit_hist = hist[np.asarray(fit_ids, np.int64)]
    fit_state = state[np.asarray(fit_ids, np.int64)]
    # Standardize per action/state coordinate using memory-fit query contexts only.
    am = fit_hist.reshape(-1, 12).mean(0); ast = fit_hist.reshape(-1, 12).std(0)
    sm = fit_state.mean(0); sst = fit_state.std(0)
    ast = np.maximum(ast, 1e-4); sst = np.maximum(sst, 1e-4)
    x = np.concatenate([((hist - am) / ast).reshape(len(hist), -1), (state - sm) / sst], axis=1).astype(np.float32)
    return x, {"action_mean": am, "action_std": ast, "state_mean": sm, "state_std": sst}


def _type_error_file(out, family, split):
    out = Path(out); q = dict(np.load(out / f"queries/{family}_{split}.npz")); p = dict(np.load(out / f"pools/{family}_memory.npz"))
    path = out / f"cache/errors_{family}_{split}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        expected = (len(q["shape"]), len(p["shape"]))
        if cached.shape != expected:
            raise ValueError(f"stale/corrupt type-error cache {path}: {cached.shape} != {expected}")
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after the complete matrix is flushed; an interrupted write
    # must not be mistaken for a reusable cache on the next invocation.
    tmp = path.with_name(path.stem + ".tmp.npy")
    err = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(q["shape"]), len(p["shape"])))
    c = p["shape"].reshape(len(p["shape"]), -1).astype(np.float64)
    cn = np.sum(c * c, axis=1)
    cf = c.T
    for st in range(0, len(q["shape"]), 128):
        qb = q["shape"][st:st + 128].reshape(-1, WINDOW * 6).astype(np.float64)
        dot = np.maximum(qb @ cf, 0.)
        val = np.maximum(np.sum(qb * qb, axis=1, keepdims=True) - dot * dot / np.maximum(cn[None], 1e-12), 0.) / (WINDOW * 6)
        err[st:st + len(qb)] = val.astype(np.float32)
    err.flush()
    del err
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


def _teacher_tau(err, allowed, frac=.1):
    minima = np.min(np.where(allowed, err, np.inf), axis=1)
    means = np.sum(np.where(allowed, err, 0), axis=1) / np.maximum(allowed.sum(1), 1)
    return max(float(frac * np.median(np.maximum(means - minima, 0))), 1e-8)


def _load_make_net():
    source_dir = Path("/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import scale_only_query as so
    return so.make_net


def _allowed(query_ep, pool_ep):
    a = np.asarray(query_ep)[:, None] != np.asarray(pool_ep)[None, :]
    if not np.all(a.any(1)):
        raise ValueError("a P5 query has no candidates after self-episode exclusion")
    return a


def train(out: Path, *, epochs=20, batch=256, device="cuda"):
    import torch
    torch.set_num_threads(8)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.25, device=0 if str(device) == "cuda" else device)
    out = Path(out); result_path = out / "training.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    make_net = _load_make_net()
    for family in ("U", "W"):
        pool = dict(np.load(out / f"pools/{family}_memory.npz")); keys = np.asarray(pool["key"], np.float32)
        trainq = dict(np.load(out / f"queries/{family}_memory.npz")); val_eps = np.asarray(json.loads((out / "pools/complete.json").read_text())["pools"][family]["val_episodes"], np.int32)
        valmask = np.isin(trainq["global_episode"], val_eps)
        fit_ids = np.flatnonzero(~valmask); val_ids = np.flatnonzero(valmask)
        x, norm = query_features(trainq, fit_ids)
        fit_err = _type_error_file(out, family, "memory")
        tau = _teacher_tau(fit_err[fit_ids], _allowed(trainq["global_episode"][fit_ids], pool["global_episode"]))
        val_err = np.asarray(fit_err[val_ids], np.float32)
        fit_allowed_np = _allowed(trainq["global_episode"][fit_ids], pool["global_episode"])
        val_allowed_np = _allowed(trainq["global_episode"][val_ids], pool["global_episode"])
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=device)
        fit_at = torch.as_tensor(fit_allowed_np, dtype=torch.bool, device=device)
        fit_et = torch.as_tensor(np.asarray(fit_err[fit_ids], np.float32), device=device)
        val_at = torch.as_tensor(val_allowed_np, dtype=torch.bool, device=device)
        val_et = torch.as_tensor(val_err, dtype=torch.float32, device=device)
        model_dim = int(keys.shape[1])
        for seed in SEEDS:
            ckpt = out / "models" / f"P5_{family}_seed{seed}.pt"
            if ckpt.exists():
                results[f"{family}|{seed}"] = torch.load(ckpt, map_location="cpu", weights_only=False)["summary"]
                continue
            torch.manual_seed(int(seed)); rng = np.random.default_rng(int(seed) + SPLIT_SEED)
            net = make_net("U", "P5", int(x.shape[1]), model_dim, d=256).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best = (float("inf"), None, 0); curve = []
            for ep in range(int(epochs)):
                net.train(); order = rng.permutation(fit_ids)
                losses = []
                for start in range(0, len(order), int(batch)):
                    local = order[start:start + int(batch)]
                    # `fit_ids` are full query indices; retain them for indexed tensors.
                    bi = torch.as_tensor(local, dtype=torch.long, device=device)
                    logits = net(xt[bi], kt)
                    allowed = fit_at[torch.searchsorted(torch.as_tensor(fit_ids, device=device), bi)]
                    errb = fit_et[torch.searchsorted(torch.as_tensor(fit_ids, device=device), bi)]
                    teach_logits = (-errb / float(tau)).masked_fill(~allowed, -1e9)
                    teacher = torch.softmax(teach_logits, dim=-1)
                    pred_logits = logits.masked_fill(~allowed, -1e9)
                    loss = -(teacher * torch.log_softmax(pred_logits, dim=-1)).sum(-1).mean()
                    if not torch.isfinite(loss): raise FloatingPointError(f"non-finite P5 loss for {family}/{seed}")
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    losses.append(float(loss.detach().cpu()))
                net.eval()
                with torch.no_grad():
                    scores = []
                    for start in range(0, len(val_ids), int(batch)):
                        vi = torch.as_tensor(val_ids[start:start + int(batch)], device=device)
                        scores.append(net(xt[vi], kt).masked_fill(~val_at[start:start + len(vi)], -1e9).cpu().numpy())
                    scores = np.concatenate(scores, axis=0)
                top = scores.argmax(1); metric = float(np.mean(val_err[np.arange(len(top)), top]))
                curve.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_top1_type_error": metric})
                if metric < best[0]:
                    best = (metric, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep + 1)
                _log(out, stage="train", family=family, seed=seed, epoch=ep + 1, val_top1_type_error=round(metric, 7))
            summary = {"family": family, "seed": int(seed), "best_epoch": int(best[2]), "val_top1_type_error": float(best[0]),
                "tau": float(tau), "fit_queries": int(len(fit_ids)), "val_queries": int(len(val_ids)),
                "fit_episodes": sorted(np.unique(trainq["global_episode"][fit_ids]).tolist()),
                "val_episodes": val_eps.tolist(), "curve": curve,
                "selection_uses": "only the internal 20% of memory episodes; held-out demos never used"}
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            temp = ckpt.with_suffix(".tmp"); torch.save({"state": best[1], "summary": summary, "norm": norm, "input_dim": int(x.shape[1]), "key_dim": model_dim}, temp); temp.replace(ckpt)
            results[f"{family}|{seed}"] = summary
            _json(result_path, results)
            del net, opt
    _json(result_path, results)
    return results


def _load_model(out, family, seed, *, device="cpu"):
    import torch
    ckpt = torch.load(Path(out) / "models" / f"P5_{family}_seed{seed}.pt", map_location="cpu", weights_only=False)
    net = _load_make_net()("U", "P5", ckpt["input_dim"], ckpt["key_dim"], d=256)
    net.load_state_dict(ckpt["state"]); net.to(device).eval()
    return net, ckpt


def _features(q, norm):
    hm = np.asarray(norm["action_mean"], np.float32); hs = np.asarray(norm["action_std"], np.float32)
    sm = np.asarray(norm["state_mean"], np.float32); ss = np.asarray(norm["state_std"], np.float32)
    hist = np.asarray(action_storage_to_policy(q["hist"]), np.float32)
    state = np.asarray(state_storage_to_policy(q["state"]), np.float32)
    return np.concatenate([((hist - hm) / hs).reshape(len(hist), -1),
                           (state - sm) / ss], axis=1).astype(np.float32)


def _score(net, x, key, *, batch=256, device="cpu"):
    import torch
    xt = torch.as_tensor(x, dtype=torch.float32, device=device); kt = torch.as_tensor(key, dtype=torch.float32, device=device)
    values = []
    with torch.no_grad():
        for st in range(0, len(x), int(batch)):
            values.append(net(xt[st:st + int(batch)], kt).cpu().numpy())
    return np.concatenate(values, axis=0)


def _scale_to_ref(action, ref):
    a = np.asarray(action, np.float32).copy(); r = np.asarray(ref, np.float32)
    na = float(np.sqrt(np.mean(np.sum(a[:, TYPE_DIMS] ** 2, axis=1))))
    nr = float(np.sqrt(np.mean(np.sum(r[:, TYPE_DIMS] ** 2, axis=1))))
    if na < 1e-8 or nr < 1e-8:
        return a
    a[:, TYPE_DIMS] *= nr / na
    return a


def _mse6(pred, target):
    return float(np.mean((np.asarray(pred, np.float64)[:, TYPE_DIMS] - np.asarray(target, np.float64)[:, TYPE_DIMS]) ** 2))


def _candidate_actions(pool):
    actions = []
    for i in range(len(pool["length"])):
        a = pool["actions_flat"][pool["action_offsets"][i]:pool["action_offsets"][i + 1]]
        actions.append(resample_rows(a, WINDOW))
    return np.stack(actions).astype(np.float32)


def _metrics_for_scores(scores, q, pool, err, *, random_reps=5, seed=791):
    n, k = scores.shape; rows = np.arange(n); picked = np.argmax(scores, axis=1)
    allowed = _allowed(q["global_episode"], pool["global_episode"])
    oracle_err = np.min(np.where(allowed, err, np.inf), axis=1)
    avg_err = np.sum(np.where(allowed, err, 0), axis=1) / np.maximum(allowed.sum(1), 1)
    positive = err <= (oracle_err + .1 * np.maximum(avg_err - oracle_err, 1e-9))[:, None]
    top5 = np.argsort(-scores, axis=1)[:, :5]
    recalls = [float(np.mean(np.any(positive[rows[:, None], top5], axis=1)))]
    from scipy.stats import rankdata
    sp = []; pe = []; psp = []; ppe = []
    for i in range(n):
        ids = np.flatnonzero(allowed[i]);best = ids[np.argsort(err[i, ids], kind="stable")[:max(2, int(np.ceil(.1 * len(ids))))]]
        if np.ptp(scores[i, best]) < 1e-12 or np.ptp(err[i, best]) < 1e-12:
            psp.append(0.); ppe.append(0.)
        else:
            psp.append(float(np.corrcoef(rankdata(scores[i, best]), rankdata(-err[i, best]))[0, 1]))
            ppe.append(float(np.corrcoef(scores[i, best], -np.log(err[i, best] + 1e-6))[0, 1]))
        if np.ptp(scores[i, ids]) < 1e-12 or np.ptp(err[i, ids]) < 1e-12:
            sp.append(0.); pe.append(0.)
        else:
            sp.append(float(np.corrcoef(rankdata(scores[i, ids]), rankdata(-err[i, ids]))[0, 1]))
            pe.append(float(np.corrcoef(scores[i, ids], -np.log(err[i, ids] + 1e-6))[0, 1]))
    cand = _candidate_actions(pool); target = q["action10"]
    mse = {k: np.empty(n, np.float32) for k in ("stored", "hist", "s_star")}
    for i, j in enumerate(picked):
        raw = cand[j]; mse["stored"][i] = _mse6(raw, target[i])
        h = q["hist"][i, -WINDOW:]
        h = _scale_to_ref(raw, h); mse["hist"][i] = _mse6(h, target[i])
        oracle = _scale_to_ref(raw, target[i]); mse["s_star"][i] = _mse6(oracle, target[i])
    random_ix = np.random.default_rng(seed).integers(0, k, size=(n, int(random_reps)))
    random_error = np.take_along_axis(err, random_ix, axis=1).mean(1)
    random_mse = []
    for rep in range(int(random_reps)):
        random_mse.append(np.asarray([_mse6(cand[random_ix[i, rep]], target[i]) for i in range(n)]))
    oracle_pick = np.argmin(np.where(allowed, err, np.inf), axis=1)
    oracle_mse_stored = np.asarray([_mse6(cand[j], target[i]) for i, j in enumerate(oracle_pick)], np.float32)
    top = err[rows, picked]
    return {"rows": int(n), "top1_type_error": float(np.mean(top)), "top1_type_error_rows": top,
        "top1_type_error_median": float(np.median(top)), "recall@5": recalls[0],
        "spearman_top10": float(np.nanmean(psp)), "pearson_top10": float(np.nanmean(ppe)),
        "spearman_all": float(np.nanmean(sp)), "pearson_all": float(np.nanmean(pe)),
        "mse6_stored": float(mse["stored"].mean()), "mse6_hist": float(mse["hist"].mean()),
        "mse6_s_star": float(mse["s_star"].mean()), "mse6_stored_rows": mse["stored"],
        "mse6_hist_rows": mse["hist"], "mse6_s_star_rows": mse["s_star"],
        "oracle_type_error": float(oracle_err.mean()), "oracle_type_error_rows": oracle_err.astype(np.float32),
        "oracle_mse6_stored": float(oracle_mse_stored.mean()), "oracle_mse6_stored_rows": oracle_mse_stored,
        "oracle_mse6_s_star": float(np.mean([_mse6(_scale_to_ref(cand[j], target[i]), target[i]) for i, j in enumerate(oracle_pick)])),
        "random_type_error": float(random_error.mean()), "random_mse6_stored": float(np.mean(random_mse)),
        "hold_last_mse6": float(np.mean([_mse6(np.repeat(q["hist"][i, -1:, :], WINDOW, axis=0), target[i]) for i in range(n)])),
        "task": q["task"].astype(np.int16), "episode": q["global_episode"].astype(np.int32),
        "picked": picked.astype(np.int32)}


def _policy_pi05(checkpoint_dir, manifest, out):
    """Fixed-zero-noise 50-action baseline from the configured Robocasa policy."""
    import av
    import jax
    import numpy as np
    from openpi.training import config as train_config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    cfg = train_config.get_config("pi05_robocasa_multitask")
    if not bool(cfg.model.discrete_state_input):
        raise RuntimeError("pi05_robocasa_multitask is not yet on the required discrete_state_input=True config; π0.5 comparison intentionally postponed")
    # RoboCasa's released checkpoint stores norm_stats.json directly under
    # assets/; its config factory has no asset_id, so pass these stats explicitly.
    norm_stats = normalize.load(Path(checkpoint_dir) / "assets")
    policy = policy_config.create_trained_policy(
        cfg, Path(checkpoint_dir), norm_stats=norm_stats
    )
    rows = []; cache = {}
    for info in manifest["demos"]:
        if info["split"] != "heldout":
            continue
        task_id = int(info["task_id"]); ep = int(info["episode"]); root = Path(manifest["task_splits"][task_id]["root"])
        def frames(camera):
            key = (str(root), ep, camera)
            if key in cache: return cache[key]
            p = root / "videos/chunk-000" / camera / f"episode_{ep:06d}.mp4"
            c = av.open(str(p)); wanted = {0}; got = {}
            for i, frame in enumerate(c.decode(video=0)):
                if i in wanted: got[i] = frame.to_ndarray(format="rgb24")
                if i > max(wanted): break
            c.close()
            cache[key] = got
            return got
        act, state, frame_ids = _load_cached_episode(out, task_id, ep)
        # Match the same demo-start queries used by the 50-step memory-chain
        # comparison. Do not pool overlapping starts into a different sample.
        cams = ("observation.images.robot0_agentview_left", "observation.images.robot0_agentview_right", "observation.images.robot0_eye_in_hand")
        images = [frames(c) for c in cams]
        for start in (0,):
            if start + WINDOW > len(act): continue
            obs = {"observation/image": images[0][start], "observation/right_image": images[1][start],
                "observation/wrist_image": images[2][start], "observation/state": state_storage_to_policy(state[start]), "prompt": info["prompt"]}
            noise = np.zeros((HORIZON, 32), np.float32)
            pred = action_policy_to_storage(np.asarray(policy.infer(obs, noise=noise)["actions"], np.float32))
            if pred.ndim != 2 or pred.shape[0] < HORIZON or pred.shape[1] < 12:
                raise ValueError(f"pi05 checkpoint returned unexpected action shape {pred.shape}")
            end = min(start + HORIZON, len(act))
            rows.append({"task": task_id, "episode": ep, "frame": start,
                "pred50": pred[:HORIZON, :12], "target50": act[start:min(start + HORIZON, len(act))],
                "mse6_5": _mse6(pred[:5, :12], act[start:start + 5]),
                "mse6_50": _mse6(pred[:end-start, :12], act[start:end])})
    if not rows: raise ValueError("pi05 baseline made no predictions")
    _save_npz(out / "eval/pi05_holdout.npz", task=np.asarray([r["task"] for r in rows]),
        episode=np.asarray([r["episode"] for r in rows]), frame=np.asarray([r["frame"] for r in rows]),
        pred50=np.stack([r["pred50"] for r in rows]), mse6_5=np.asarray([r["mse6_5"] for r in rows]),
        mse6_50=np.asarray([r["mse6_50"] for r in rows]))
    return {"available": True, "checkpoint": str(checkpoint_dir), "config": "pi05_robocasa_multitask",
        "rows": len(rows), "prediction_noise": "all-zero 50x32",
        "mse6_5": float(np.mean([r["mse6_5"] for r in rows])),
        "mse6_50": float(np.mean([r["mse6_50"] for r in rows])), "window_starts": [0]}


def _generate_50(net, ck, pool, initial_hist, initial_state, *, device="cpu"):
    """Autoregressively append retrieved variable units, updating history only with own draft."""
    import torch
    hist = np.asarray(initial_hist, np.float32).copy(); state = np.asarray(initial_state, np.float32).copy(); chunks=[]; selected=[]
    # The initial history is unavailable at the demo start (the caller supplies
    # zeros); those padding slots must not dilute the magnitude of the first
    # generated chunk when it becomes the reference for the next retrieval.
    hist_valid = np.linalg.norm(hist, axis=1) > 1e-8
    # Variable units need their original lengths/actions, not 10-frame key summaries.
    remaining = HORIZON
    while remaining > 0:
        q = {"hist": hist[None], "state": state[None]}
        x = _features(q, ck["norm"])
        with torch.no_grad():
            sc = net(torch.as_tensor(x, dtype=torch.float32, device=device),
                     torch.as_tensor(pool["key"], dtype=torch.float32, device=device)).cpu().numpy()[0]
        j = int(np.argmax(sc)); raw = pool["actions_flat"][pool["action_offsets"][j]:pool["action_offsets"][j + 1]].copy()
        # Set magnitude from executed/generated history when available; otherwise use stored size.
        if np.any(hist_valid) and np.linalg.norm(hist[hist_valid, TYPE_DIMS]) > 1e-6:
            raw = _scale_to_ref(raw, hist[hist_valid])
        emit = raw[:remaining]
        chunks.append(emit); selected.append(j)
        remaining -= len(emit)
        hist = np.concatenate([hist, emit], axis=0)[-WINDOW:]
        hist_valid = np.concatenate([hist_valid, np.ones(len(emit), dtype=bool)])[-WINDOW:]
        # We do not query future recorded states. State remains at its start observation;
        # this isolates the requested autoregressive history update.
    out = np.concatenate(chunks, axis=0)[:HORIZON]
    return out, selected


def evaluate(out: Path, *, device="cuda", include_pi05=True):
    import torch
    torch.set_num_threads(8)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.25, device=0 if str(device) == "cuda" else device)
    out = Path(out); res_path = out / "evaluation.json"
    if res_path.exists(): return json.loads(res_path.read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    metrics = {}; per_row = {}; drafts50 = {}
    for family in ("U", "W"):
        pool = dict(np.load(out / f"pools/{family}_memory.npz")); q = dict(np.load(out / f"queries/{family}_heldout.npz"))
        err = _type_error_file(out, family, "heldout")
        models=[]
        for seed in SEEDS:
            net, ck = _load_model(out, family, seed, device=device); models.append((net, ck))
        x = _features(q, models[0][1]["norm"]); all_scores=[]; seed_metrics=[]
        for seed,(net,ck) in zip(SEEDS,models):
            sc = _score(net, x, pool["key"], device=device)
            met = _metrics_for_scores(sc,q,pool,err,seed=seed+4100)
            seed_metrics.append(met); per_row[f"{family}|seed{seed}"]={k:v for k,v in met.items() if isinstance(v,np.ndarray)}
            all_scores.append(sc)
        ens=np.mean(all_scores,axis=0); ensemble=_metrics_for_scores(ens,q,pool,err,seed=991)
        # Overall summaries average the three independent seeds and retain the spread.
        scalar_keys=[k for k,v in seed_metrics[0].items() if np.isscalar(v) and k not in ("rows",)]
        metrics[family]={"queries": int(len(q["hist"])), "memory_items": int(len(pool["key"])),
            "per_seed": [{k:float(m[k]) for k in scalar_keys if isinstance(m[k],(int,float,np.number))} for m in seed_metrics],
            "seed_mean": {k:float(np.mean([m[k] for m in seed_metrics])) for k in scalar_keys if isinstance(seed_metrics[0][k],(int,float,np.number))},
            "seed_min": {k:float(np.min([m[k] for m in seed_metrics])) for k in scalar_keys if isinstance(seed_metrics[0][k],(int,float,np.number))},
            "seed_max": {k:float(np.max([m[k] for m in seed_metrics])) for k in scalar_keys if isinstance(seed_metrics[0][k],(int,float,np.number))},
            "ensemble_top1_type_error": float(ensemble["top1_type_error"]),
            "by_task": {manifest["tasks"][tid]: {"queries": int(np.sum(q["task"]==tid)),
                "P5_top1_type_error": float(np.mean(ensemble["top1_type_error_rows"][q["task"]==tid])),
                "oracle_type_error": float(np.mean(ensemble["oracle_type_error_rows"][q["task"]==tid])),
                "P5_mse6_stored": float(np.mean(ensemble["mse6_stored_rows"][q["task"]==tid]))}
                for tid in range(len(manifest["tasks"]))}}
        per_row[f"{family}|ensemble"]={k:v for k,v in ensemble.items() if isinstance(v,np.ndarray)}
        if family == "U":
            # A single initial query per held-out episode, chained to 50 steps.
            qstart = {}
            for i,(tid,ep,fr) in enumerate(zip(q["task"],q["episode"],q["frame"])):
                if int(fr)==0: qstart[(int(tid),int(ep))]=i
            for seed,(net,ck) in zip(SEEDS,models):
                rows=[]
                for (tid,ep),i in sorted(qstart.items()):
                    target, state, _ = _load_cached_episode(out,tid,ep)
                    hist = np.zeros((WINDOW,12),np.float32)
                    pred, picked = _generate_50(net,ck,pool,hist,state[0],device=device)
                    true = target[:HORIZON]
                    if len(true)<HORIZON: continue
                    rows.append({"task":tid,"episode":ep,"pred":pred,"true":true,
                        "mse6_50":_mse6(pred,true),"mse6_5":_mse6(pred[:5],true[:5]),"units":len(picked)})
                drafts50[str(seed)]={"rows":rows,"mse6_50":float(np.mean([r["mse6_50"] for r in rows])),
                    "mse6_5":float(np.mean([r["mse6_5"] for r in rows])),"mean_units":float(np.mean([r["units"] for r in rows]))}
        _log(out,stage="evaluate_family",family=family,queries=len(q["hist"]),memory=len(pool["key"]))
    if include_pi05:
        ckpt=Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT",STATE/"checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"))
        if ckpt.exists():
            try:
                metrics["pi05"]=_policy_pi05(ckpt,manifest,out)
            except Exception as exc:
                metrics["pi05"]={"available":False,"checkpoint":str(ckpt),"error":f"{type(exc).__name__}: {exc}"}
        else:
            metrics["pi05"]={"available":False,"reason":"configured checkpoint missing"}
    _save_npz(out/"eval/per_row_metrics.npz", **{f"{k.replace('|','_')}__{a}":v for k,d in per_row.items() for a,v in d.items()})
    # Save compact 50-step drafts for post hoc inspection.
    if drafts50:
        arr=[]
        for seed, d in drafts50.items():
            arr.extend(d["rows"])
        _save_npz(out/"eval/drafts50.npz",seed=np.concatenate([np.full(len(drafts50[s]["rows"]),int(s),np.int8) for s in drafts50]),
            task=np.concatenate([np.asarray([r["task"] for r in drafts50[s]["rows"]],np.int16) for s in drafts50]),
            episode=np.concatenate([np.asarray([r["episode"] for r in drafts50[s]["rows"]],np.int32) for s in drafts50]),
            pred=np.concatenate([np.stack([r["pred"] for r in drafts50[s]["rows"]]) for s in drafts50]),
            target=np.concatenate([np.stack([r["true"] for r in drafts50[s]["rows"]]) for s in drafts50]))
    eval_result={"metrics":metrics,"drafts50":{s:{k:v for k,v in d.items() if k!="rows"} for s,d in drafts50.items()},
        "selection_rule":"P5 checkpoint epoch selected by internal memory-episode validation top1 type error; no heldout query/action used.",
        "oracle_rule":"Heldout true shape selects nearest stored type only for labelled ceiling; not used in retrieval or P5 training.",
        "action_mse_space":"raw LeRobot action command values, active EEF 6 channels [5:11]; type removes only a common scale and never rotates.",
        "p5_input_order":"Recorded LeRobot history [base4,mode,EEF6,grip] and state [base pose7,EEF pose7,grip2] are converted to fork order [EEF6,grip,base4,mode] and [EEF pose7,base pose7,grip2] before P5 normalization/inference.",
        "pi05_control":"zero 50x32 noise per query; fork-order outputs are converted back to LeRobot storage order before comparison."}
    _json(res_path,eval_result);_log(out,stage="evaluate",families=list(metrics),done=True)
    return eval_result


def _report_pi05_metrics(out, evaluation):
    """Prefer an already-integrated baseline; otherwise load the resumable pi05 stage."""
    pi = evaluation.get("metrics", {}).get("pi05", {"available": False})
    if pi.get("available"):
        return pi
    separate = Path(out) / "pi05_evaluation.json"
    if separate.exists():
        return json.loads(separate.read_text())
    return pi


def report(out):
    out=Path(out);m=json.loads((out/"manifest.json").read_text());p=json.loads((out/"pools/complete.json").read_text());t=json.loads((out/"training.json").read_text());ep=out/"evaluation_ext.json";e=json.loads((ep if ep.exists() else out/"evaluation.json").read_text())
    lines=["# RM-B offline skill-memory/P5 results","",
        "This is an offline demonstration-memory diagnostic, not a closed-loop result. Held-out demos are never in the memory or P5 training; held-out actions are used only for scoring and the explicitly labelled oracle-type ceiling. The units are chosen with ground-truth type error only for the ceiling.","",
        "## Memory construction","",
        "| pool | memory items | held-out queries | length frames p10 / median / p90 |","|---|---:|---:|---:|"]
    for fam in ("U","W"):
        q=dict(np.load(out/f"queries/{fam}_heldout.npz"));u=p["pools"][fam]["length_frames"]
        lines.append(f"| {fam} | {p['pools'][fam]['memory_units']} | {len(q['hist'])} | {u['p10']:.1f} / {u['median']:.1f} / {u['p90']:.1f} |")
    lines += ["",f"Memory per task: {p['by_task']}. Base-channel nonzero fraction={p['base_nonzero_fraction_memory']:.5f}; control-mode nonzero fraction={p['mode_nonzero_fraction_memory']:.5f}.","",
        "## P5 and oracle ceiling","","| pool | metric | mean across seeds | seed min–max |","|---|---|---:|---:|"]
    keys=("top1_type_error","mse6_stored","mse6_hist","mse6_s_star","recall@5","spearman_top10")
    for fam in ("U","W"):
        f=e["metrics"][fam]
        for k in keys:
            lines.append(f"| {fam} | P5 {k} | {f['seed_mean'][k]:.5f} | {f['seed_min'][k]:.5f}–{f['seed_max'][k]:.5f} |")
        lines.append(f"| {fam} | oracle type error | {f['seed_mean']['oracle_type_error']:.5f} | {f['seed_min']['oracle_type_error']:.5f}–{f['seed_max']['oracle_type_error']:.5f} |")
        if "oracle_mse6_stored" in f["seed_mean"]:
            lines.append(f"| {fam} | oracle selected, stored-scale MSE6 | {f['seed_mean']['oracle_mse6_stored']:.5f} | {f['seed_min']['oracle_mse6_stored']:.5f}–{f['seed_max']['oracle_mse6_stored']:.5f} |")
        lines.append(f"| {fam} | oracle selected, s* MSE6 | {f['seed_mean']['oracle_mse6_s_star']:.5f} | {f['seed_min']['oracle_mse6_s_star']:.5f}–{f['seed_max']['oracle_mse6_s_star']:.5f} |")
        lines.append(f"| {fam} | random type MSE6 stored | {f['seed_mean']['random_mse6_stored']:.5f} | {f['seed_min']['random_mse6_stored']:.5f}–{f['seed_max']['random_mse6_stored']:.5f} |")
        lines.append(f"| {fam} | hold-last MSE6 | {f['seed_mean']['hold_last_mse6']:.5f} | {f['seed_min']['hold_last_mse6']:.5f}–{f['seed_max']['hold_last_mse6']:.5f} |")
    lines += ["","### Per-task P5 (ensemble logits)","","| pool / task | queries | P5 top1 type error | oracle type error | P5 stored-scale MSE6 |","|---|---:|---:|---:|---:|"]
    for fam in ("U","W"):
        for task, v in e["metrics"][fam]["by_task"].items():
            lines.append(f"| {fam} / {task} | {v['queries']} | {v['P5_top1_type_error']:.5f} | {v['oracle_type_error']:.5f} | {v['P5_mse6_stored']:.5f} |")
    lines += ["","## Autoregressive 50-step U draft","","| seed | held-out demos | MSE6 / 50 steps | MSE6 / first 5 | mean retrieved units |","|---:|---:|---:|---:|---:|"]
    for seed,d in e["drafts50"].items():
        lines.append(f"| {seed} | {len(m['tasks'])*m['heldout_per_task']} | {d['mse6_50']:.5f} | {d['mse6_5']:.5f} | {d['mean_units']:.2f} |")
    pi=_report_pi05_metrics(out,e)
    if pi.get("available"):
        lines.append(f"| pi05_robocasa_multitask | {pi['rows']} query starts | {pi['mse6_50']:.5f} | {pi['mse6_5']:.5f} (first 5) | — |")
        lines += ["",f"π0.5 uses checkpoint `{pi['checkpoint']}` and zero noise. It is compared on the same held-out demo starts as the P5 50-step chains."]
    else:
        reason = pi.get("error", pi.get("reason", "no π0.5 result is recorded"))
        lines += ["",f"π0.5 comparison unavailable: {reason}. The memory and oracle results are not π0.5 comparisons."]
    lines += ["","## Notes / review","",
        "P5 is the LIBERO P5 recipe adapted to 12-D history + 16-D state and a scale-only DCT-3 key; it is trained separately for U and W, with self-episode candidates masked during both training and internal validation. Three seeds are reported as mean and range. Evaluation uses all held-out 20 demos per task.",
        "The 50-step rollout chains only retrieved U units and updates the history from its own generated commands; it holds the initial observed state fixed because this offline experiment has no learned state transition model. A first query with an all-zero history retains the retrieved unit's stored magnitude; later queries match magnitude to recent generated history. P5 does not receive task labels or language.",
        "The ceiling uses held-out action shape to choose the nearest matching memory type and is intentionally optimistic, not a retrieval result. π0.5 is run only when the configured RoboCasa checkpoint is present. No closed-loop inference is run.",""]
    (out/"report.md").write_text("\n".join(lines))
    return str(out/"report.md")


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","train","evaluate","pi05","report","all"));p.add_argument("--out",type=Path,default=OUT_DEFAULT)
    p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch",type=int,default=256);p.add_argument("--device",default="cuda")
    p.add_argument("--no-pi05",action="store_true");p.add_argument("--checkpoint",type=Path)
    a=p.parse_args();out=a.out
    if a.stage=="prepare":prepare(out)
    elif a.stage=="train":train(out,epochs=a.epochs,batch=a.batch,device=a.device)
    elif a.stage=="evaluate":evaluate(out,device=a.device,include_pi05=not a.no_pi05)
    elif a.stage=="pi05":
        result_path=out/"pi05_evaluation.json"
        if result_path.exists():
            print(f"preserving existing {result_path}")
        else:
            manifest=json.loads((out/"manifest.json").read_text())
            ckpt=a.checkpoint or Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT",STATE/"checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"))
            if not ckpt.exists(): raise FileNotFoundError(f"π0.5 checkpoint not found: {ckpt}")
            value=_policy_pi05(ckpt,manifest,out)
            _json(result_path,value)
            _log(out,stage="pi05",done=True,rows=value["rows"])
    elif a.stage=="report":print(report(out))
    else:
        prepare(out);train(out,epochs=a.epochs,batch=a.batch,device=a.device)
        evaluate(out,device=a.device,include_pi05=not a.no_pi05);print(report(out))
if __name__=="__main__":main()
