#!/usr/bin/env python3
"""RoboCasa skill memory rebuilt in the format fixed on LIBERO (11 / experiment 6).

Stages (each writes under ``--out`` and refuses to overwrite a finished stage):

  prepare   read the first 100 pretrain human demos of the six development tasks,
            keep the 80/20 episode split of ``memory-01`` and split the held-out
            20 into Dev 5 / Eval 15 per task.
  units     variable-length motion units from the memory side: cut at gripper
            sign switches and minima of the 3-frame-smoothed EEF speed
            (minimum 5 steps, the LIBERO CMP3 rule); arm six axes are stored as
            DCT-3 coefficients at natural length, the gripper separately.
  rows      query rows: memory-side training rows (stride 10) and held-out
            Dev/Eval rows (contiguous stride 10). Each row has causal history,
            the current state and the next 10 actions (for scoring only).
  oracle    upper bound (nearest stored unit), hold-last, zero and random unit
            on the Dev/Eval rows, plus the fixed ``hard`` mask.

All actions used for scoring are in fork/policy order and z-scored with the
fine-tuned checkpoint's own ``norm_stats.json`` (the space the action expert
works in). Arm MSE is the sum of six squared arm errors over (10 x 7), the
LIBERO "arm6 over 7" convention; the gripper is reported separately.

No task/subtask labels enter cuts, keys or values. Task and episode IDs are
used only for the split, self-episode exclusion and bootstrap grouping.
Held-out actions are used only for scoring, oracle rows and setting selection
on Dev.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402  (split rule, parquet readers, order conversions)
import units as unit_cut  # noqa: E402

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
OUT_DEFAULT = STATE / "outputs/robocasa-memory/latent-skill-01"
CHECKPOINT = Path(os.environ.get(
    "SKILLNET_ROBOCASA_CHECKPOINT",
    STATE / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"))
TASKS = ex.TASKS
N_DEMOS = ex.N_DEMOS
SPLIT_SEED = ex.SPLIT_SEED          # identical memory/held-out split to memory-01
DEV_PER_TASK = 5                    # of the 20 held-out episodes per task
DEV_SEED = 20261116
MIN_LENGTH = 5                      # LIBERO CMP3 cut rule
DCT_K = 3
WINDOW = 10
HISTORY = 10
ROW_STRIDE = 10
ARM = slice(0, 6)                   # policy order: EEF pos3 + rot3
GRIP = 6
HARD_QUANTILE = .8
RANDOM_SEEDS = (4101, 4102, 4103, 4104, 4105)


# ----------------------------------------------------------------------------- io

def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def _log(out, stage, **payload):
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **payload}
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out) / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=ex._json_default) + "\n")
    print(json.dumps(rec, ensure_ascii=False, default=ex._json_default), flush=True)


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _guard_output(out):
    """Refuse to write experiment outputs into the read-only dataset mount."""
    resolved = Path(out).resolve()
    for forbidden in (Path("/work/data").resolve(),):
        if resolved == forbidden or forbidden in resolved.parents:
            raise PermissionError(f"refusing to write under the read-only data mount: {resolved}")


# ----------------------------------------------------------------- normalization

def load_action_stats(checkpoint=CHECKPOINT):
    """Policy-order action mean/std from the checkpoint (z-score, as served)."""
    d = json.loads((Path(checkpoint) / "assets/norm_stats.json").read_text())["norm_stats"]
    mean = np.asarray(d["actions"]["mean"], np.float32)[:12]
    std = np.asarray(d["actions"]["std"], np.float32)[:12]
    return mean, std


def normalize_policy_actions(policy_actions, stats):
    mean, std = stats
    a = np.asarray(policy_actions, np.float32)
    return (a - mean[: a.shape[-1]]) / (std[: a.shape[-1]] + 1e-6)


# ------------------------------------------------------------------- DCT values

def dct3_natural(arm, k=DCT_K):
    """Encode [T,6] per axis with k orthonormal DCT-II coefficients; decode at T."""
    a = np.asarray(arm, np.float64)
    if a.ndim != 2 or not len(a):
        raise ValueError("expected non-empty [T,D]")
    n = min(int(k), len(a))
    coeff = dct(a, type=2, norm="ortho", axis=0)[:n]
    full = np.zeros_like(a)
    full[:n] = coeff
    padded = np.zeros((int(k), a.shape[1]), np.float64)
    padded[:n] = coeff
    return padded.astype(np.float32), idct(full, type=2, norm="ortho", axis=0).astype(np.float32)


def window_from_unit(unit, length, n=WINDOW):
    """First n stored steps of a unit, holding the last step (KEY2 convention)."""
    length = int(length)
    if length < 1:
        raise ValueError("unit length must be positive")
    u = np.asarray(unit)[:length]
    m = min(length, n)
    out = np.empty((n, u.shape[-1]), np.float32)
    out[:m] = u[:m]
    if m < n:
        out[m:] = u[m - 1]
    return out


def arm6_mse_matrix(query10, cand10, *, block=256):
    """Sum of six squared arm errors / (10*7) for every query x candidate."""
    q = np.asarray(query10, np.float64)[..., ARM].reshape(len(query10), -1)
    c = np.asarray(cand10, np.float64)[..., ARM].reshape(len(cand10), -1)
    denom = WINDOW * 7.0
    cn = np.einsum("nd,nd->n", c, c)
    out = np.empty((len(q), len(c)), np.float32)
    for s in range(0, len(q), block):
        x = q[s:s + block]
        out[s:s + len(x)] = np.maximum(np.einsum("nd,nd->n", x, x)[:, None] + cn[None] - 2 * x @ c.T, 0) / denom
    return out


def arm6_mse(pred10, target10):
    p = np.asarray(pred10, np.float64)[..., ARM]
    t = np.asarray(target10, np.float64)[..., ARM]
    return (np.square(p - t).sum(axis=(-1, -2)) / (WINDOW * 7.0)).astype(np.float32)


def grip_mse(pred10, target10):
    p = np.asarray(pred10, np.float64)[..., GRIP]
    t = np.asarray(target10, np.float64)[..., GRIP]
    return (np.square(p - t).sum(-1) / (WINDOW * 7.0)).astype(np.float32)


# ------------------------------------------------------------------- stage: prepare

def split_heldout(heldout_ids, task_index):
    ids = np.sort(np.asarray(heldout_ids, np.int64))
    p = np.random.default_rng(DEV_SEED + int(task_index)).permutation(ids)
    return np.sort(p[:DEV_PER_TASK]), np.sort(p[DEV_PER_TASK:])


def prepare(out, *, tasks=TASKS, demo_limit=N_DEMOS, smoke=False):
    out = Path(out)
    _guard_output(out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    demos, splits = [], []
    for task_index, task in enumerate(tasks):
        root = ex._dataset_path(task)
        files = ex._episode_files(root)[: int(demo_limit)]
        if len(files) < (4 if smoke else N_DEMOS):
            raise FileNotFoundError(f"{task}: {len(files)} demos found under {root}")
        ids = np.asarray([int(p.stem.split("_")[-1]) for p in files], np.int64)
        memory_ids, heldout_ids = ex.split_episode_ids(ids, seed=SPLIT_SEED + task_index)
        dev_ids, eval_ids = split_heldout(heldout_ids, task_index) if len(heldout_ids) > DEV_PER_TASK else \
            (heldout_ids[:1], heldout_ids[1:])
        role = {**{int(e): "memory" for e in memory_ids}, **{int(e): "dev" for e in dev_ids},
                **{int(e): "eval" for e in eval_ids}}
        for p in files:
            ep = int(p.stem.split("_")[-1])
            target = episodes_dir / f"task{task_index}_ep{ep:06d}.npz"
            if not target.exists():
                action, state, frame = ex._read_episode(p)
                tmp = target.with_name(target.stem + ".tmp.npz")
                np.savez_compressed(tmp, action=action, state=state, frame=frame)
                tmp.replace(target)
            with np.load(target) as z:
                n = int(len(z["action"]))
            demos.append({"task_id": task_index, "task": task, "episode": ep, "split": role[ep],
                          "frames": n, "source": str(p), "prompt": ex._read_prompt(root, ep)})
        splits.append({"task_id": task_index, "task": task, "root": str(root),
                       "memory_episodes": memory_ids.tolist(), "dev_episodes": dev_ids.tolist(),
                       "eval_episodes": eval_ids.tolist()})
    manifest = {
        "tasks": list(tasks), "demo_limit": int(demo_limit), "split_seed": SPLIT_SEED, "dev_seed": DEV_SEED,
        "demos": demos, "task_splits": splits, "checkpoint": str(CHECKPOINT),
        "split_rule": ("memory/held-out: experiment.split_episode_ids with seed 20261110+task_id "
                       "(identical to memory-01); held-out 20 -> Dev 5 / Eval 15 with seed 20261116+task_id."),
    }
    _json(manifest_path, manifest)
    _log(out, "prepare", episodes=len(demos), frames=int(sum(d["frames"] for d in demos)))
    return manifest


def _episode(out, task_id, episode):
    with np.load(Path(out) / "episodes" / f"task{int(task_id)}_ep{int(episode):06d}.npz") as z:
        return np.asarray(z["action"], np.float32), np.asarray(z["state"], np.float32)


# --------------------------------------------------------------------- stage: units

def build_units(out, *, min_length=MIN_LENGTH):
    out = Path(out)
    target = out / "units/memory_units.npz"
    if target.exists():
        return json.loads((out / "units/summary.json").read_text())
    m = json.loads((out / "manifest.json").read_text())
    stats = load_action_stats(m["checkpoint"])
    recs = []
    for d in m["demos"]:
        if d["split"] != "memory":
            continue
        storage, state = _episode(out, d["task_id"], d["episode"])
        policy = normalize_policy_actions(ex.action_storage_to_policy(storage), stats)
        for u in unit_cut.cut(storage, min_length=int(min_length)):
            s, e = int(u["start"]), int(u["end"])
            coeff, arm_dec = dct3_natural(policy[s:e, ARM])
            recs.append({"task": d["task_id"], "episode": d["episode"], "start": s, "end": e,
                         "coeff": coeff, "arm_dct3": arm_dec, "raw": policy[s:e, :7],
                         "grip": policy[s:e, GRIP]})
    lmax = max(r["end"] - r["start"] for r in recs)
    n = len(recs)
    natural = np.zeros((n, lmax, 7), np.float32)   # DCT-3 arm + stored raw gripper channel
    raw = np.zeros((n, lmax, 7), np.float32)
    for i, r in enumerate(recs):
        L = r["end"] - r["start"]
        natural[i, :L, :6] = r["arm_dct3"]
        natural[i, :L, 6] = r["grip"]
        raw[i, :L] = r["raw"]
    length = np.asarray([r["end"] - r["start"] for r in recs], np.int32)
    window10 = np.stack([window_from_unit(natural[i], length[i]) for i in range(n)])
    _save_npz(target,
              task=np.asarray([r["task"] for r in recs], np.int16),
              episode=np.asarray([r["episode"] for r in recs], np.int32),
              start=np.asarray([r["start"] for r in recs], np.int32),
              end=np.asarray([r["end"] for r in recs], np.int32),
              length=length, coeff=np.stack([r["coeff"] for r in recs]).astype(np.float32),
              natural_actions=natural, raw_actions=raw, window10=window10)
    q = np.quantile(length, [0, .1, .5, .9, 1])
    summary = {"units": int(n), "min_length": int(min_length), "dct_k": DCT_K,
               "length": dict(zip(("min", "p10", "median", "p90", "max"), map(float, q))),
               "by_task": {t: int(np.sum([r["task"] == i for r in recs])) for i, t in enumerate(m["tasks"])},
               "value": "arm6 DCT-3 per axis at natural length (normalized policy order); gripper stored raw, scored separately",
               "cut": "units.cut: gripper sign switch + local minima of 3-frame-smoothed EEF translation command speed"}
    _json(out / "units/summary.json", summary)
    _log(out, "units", **{k: summary[k] for k in ("units", "length")})
    return summary


# ---------------------------------------------------------------------- stage: rows

def episode_rows(policy_norm, state_policy, *, stride=ROW_STRIDE, offset=0):
    """Causal rows: history (10x7 + mask), state, and next 10 actions (target only)."""
    T = len(policy_norm)
    rows = []
    for f in range(int(offset), T - WINDOW + 1, int(stride)):
        hist = np.zeros((HISTORY, 7), np.float32)
        mask = np.zeros(HISTORY, bool)
        past = policy_norm[max(0, f - HISTORY):f, :7]
        if len(past):
            hist[-len(past):] = past
            mask[-len(past):] = True
        rows.append((f, hist, mask, state_policy[f], policy_norm[f:f + WINDOW, :7]))
    return rows


def build_rows(out):
    out = Path(out)
    done = out / "rows/summary.json"
    if done.exists():
        return json.loads(done.read_text())
    m = json.loads((out / "manifest.json").read_text())
    stats = load_action_stats(m["checkpoint"])
    acc = {s: {k: [] for k in ("task", "episode", "frame", "history", "history_mask", "state", "target")}
           for s in ("train", "dev", "eval")}
    for d in m["demos"]:
        storage, state = _episode(out, d["task_id"], d["episode"])
        policy = normalize_policy_actions(ex.action_storage_to_policy(storage), stats)
        sp = ex.state_storage_to_policy(state).astype(np.float32)
        split = {"memory": "train", "dev": "dev", "eval": "eval"}[d["split"]]
        for f, h, mk, st, tg in episode_rows(policy, sp):
            a = acc[split]
            a["task"].append(d["task_id"]); a["episode"].append(d["episode"]); a["frame"].append(f)
            a["history"].append(h); a["history_mask"].append(mk); a["state"].append(st); a["target"].append(tg)
    summary = {}
    for split, a in acc.items():
        arrays = {k: np.asarray(v) for k, v in a.items()}
        arrays["task"] = arrays["task"].astype(np.int16)
        arrays["episode"] = arrays["episode"].astype(np.int32)
        arrays["frame"] = arrays["frame"].astype(np.int32)
        _save_npz(out / f"rows/{split}.npz", **arrays)
        summary[split] = {"rows": int(len(arrays["frame"])),
                          "episodes": int(len(set(zip(a["task"], a["episode"]))))}
    summary["stride"] = ROW_STRIDE
    _json(done, summary)
    _log(out, "rows", **summary)
    return summary


# -------------------------------------------------------------------- stage: oracle

def episode_bootstrap(values, task, episode, *, reps=5000, seed=4101):
    v = np.asarray(values, np.float64)
    groups = {}
    for i, key in enumerate(zip(np.asarray(task).tolist(), np.asarray(episode).tolist())):
        groups.setdefault(key, []).append(i)
    idx = [np.asarray(x) for x in groups.values()]
    sums = np.asarray([v[i].sum() for i in idx]); cnt = np.asarray([len(i) for i in idx])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(idx), size=(int(reps), len(idx)))
    boot = sums[draw].sum(1) / cnt[draw].sum(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(boot, .025)), "hi": float(np.quantile(boot, .975))}


def hold_last(rows):
    """Repeat the last executed action (zeros when there is no history)."""
    last = rows["history"][:, -1, :7] * rows["history_mask"][:, -1, None]
    return np.repeat(last[:, None], WINDOW, axis=1)


def oracle(out):
    out = Path(out)
    done = out / "oracle/results.json"
    if done.exists():
        return json.loads(done.read_text())
    bank = dict(np.load(out / "units/memory_units.npz"))
    cand = bank["window10"]
    result = {}
    for split in ("dev", "eval"):
        rows = dict(np.load(out / f"rows/{split}.npz"))
        tgt = rows["target"]
        err = arm6_mse_matrix(tgt, cand)                    # held-out episodes are never in memory
        best = err.argmin(1)
        oracle_err = err[np.arange(len(err)), best]
        hard = oracle_err >= np.quantile(oracle_err, HARD_QUANTILE)
        preds = {"oracle_top1": cand[best], "hold_last": hold_last(rows),
                 "zero": np.zeros_like(tgt)}
        block = {}
        for name, p in preds.items():
            e = arm6_mse(p, tgt)
            block[name] = {"arm6_all": episode_bootstrap(e, rows["task"], rows["episode"]),
                           "arm6_hard": float(e[hard].mean()),
                           "grip_all": float(grip_mse(p, tgt).mean())}
        rnd = []
        for s in RANDOM_SEEDS:
            pick = np.random.default_rng(s).integers(0, len(cand), len(tgt))
            rnd.append(err[np.arange(len(err)), pick])
        rnd = np.mean(rnd, 0)
        block["random_unit"] = {"arm6_all": episode_bootstrap(rnd, rows["task"], rows["episode"]),
                                "arm6_hard": float(rnd[hard].mean())}
        block["per_task_oracle_arm6"] = {t: float(oracle_err[rows["task"] == i].mean())
                                         for i, t in enumerate(TASKS) if np.any(rows["task"] == i)}
        _save_npz(out / f"oracle/{split}.npz", hard=hard, oracle_index=best.astype(np.int32),
                  oracle_arm6=oracle_err)
        block["rows"] = int(len(tgt)); block["hard_rows"] = int(hard.sum())
        result[split] = block
    result["definition"] = {"metric": "arm6 sum of squares / (10*7) in checkpoint z-score policy space",
                            "hard": f"oracle top-1 arm6 >= {HARD_QUANTILE:.0%} quantile within the split",
                            "candidate_window": "first 10 steps of the natural-length DCT-3 unit, last step held"}
    _json(done, result)
    _log(out, "oracle", dev=result["dev"]["oracle_top1"]["arm6_all"], eval=result["eval"]["oracle_top1"]["arm6_all"])
    return result


# -------------------------------------------------------------------- stage: tokens
#
# Final-layer PaliGemma prefix tokens of the fine-tuned RoboCasa π0.5 (the
# hidden states the action expert attends to), for every frame that is a query
# row or a memory-unit start. Image tokens are kept per camera (no pooling
# before E); text/state-prompt tokens are trimmed to ``text_tokens``.

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
IMAGE_TOKENS = 256


def token_index(out):
    """Unique (task, episode, frame) contexts needed by rows and unit starts."""
    out = Path(out)
    keys = set()
    for split in ("train", "dev", "eval"):
        r = np.load(out / f"rows/{split}.npz")
        keys.update(zip(r["task"].tolist(), r["episode"].tolist(), r["frame"].tolist()))
    u = np.load(out / "units/memory_units.npz")
    keys.update(zip(u["task"].tolist(), u["episode"].tolist(), u["start"].tolist()))
    k = np.asarray(sorted(keys), np.int64)
    return k


def shard_of_episodes(keys, nshards):
    """Assign whole episodes to shards, balancing context counts (deterministic)."""
    eps, counts = np.unique(keys[:, :2], axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    load = np.zeros(int(nshards), np.int64)
    owner = {}
    for i in order:
        s = int(np.argmin(load)); owner[tuple(eps[i].tolist())] = s; load[s] += counts[i]
    return np.asarray([owner[(int(t), int(e))] for t, e, _ in keys], np.int16)


def group_ids(text_tokens):
    return np.concatenate([np.repeat(np.arange(len(CAMERAS), dtype=np.int8), IMAGE_TOKENS),
                           np.full(int(text_tokens), len(CAMERAS), np.int8)])


def _episode_frames(root, camera, episode, wanted):
    """Decode one episode video once, returning the requested frame indices."""
    import av
    path = Path(root) / "videos/chunk-000" / f"observation.images.{camera}" / f"episode_{int(episode):06d}.mp4"
    want = set(int(w) for w in wanted)
    got = {}
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                got[i] = fr.to_ndarray(format="rgb24")
            if len(got) == len(want):
                break
    missing = want - set(got)
    if missing:
        raise IndexError(f"{path}: frames {sorted(missing)[:5]} unavailable")
    return got


def extract_tokens(out, *, shard=0, nshards=1, batch_size=8, text_tokens=200):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    import context_key as ctx
    import expert
    out = Path(out)
    tok_dir = out / "tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    idx_path = tok_dir / "index.npz"
    if not idx_path.exists():
        raise FileNotFoundError("run the 'token-index' stage once before starting token shards")
    idx = np.load(idx_path)
    if int(idx["nshards"]) != int(nshards):
        raise ValueError(f"token index was built for {int(idx['nshards'])} shards")
    keys = idx["keys"][idx["shard"] == int(shard)]
    m = json.loads((out / "manifest.json").read_text())
    demo = {(d["task_id"], d["episode"]): d for d in m["demos"]}
    roots = {s["task_id"]: s["root"] for s in m["task_splits"]}
    seq = len(CAMERAS) * IMAGE_TOKENS + int(text_tokens)
    tpath, vpath, ppath = (tok_dir / f"shard{shard:02d}_{n}" for n in ("tokens_fp16.npy", "valid.npy", "done.json"))
    mode = "r+" if tpath.exists() else "w+"
    tokens = np.lib.format.open_memmap(tpath, mode=mode, dtype=np.float16, shape=(len(keys), seq, 2048))
    valid = np.lib.format.open_memmap(vpath, mode=mode, dtype=np.bool_, shape=(len(keys), seq))
    done_eps = set(map(tuple, json.loads(ppath.read_text())["episodes"])) if ppath.exists() else set()
    transform = ctx._make_transforms(Path(m["checkpoint"]))
    model, _ = expert._load_expert_model(Path(m["checkpoint"]))
    extractor = _prefix_token_extractor(model)
    eps = list(dict.fromkeys(map(tuple, keys[:, :2].tolist())))
    t0 = time.time(); text_max = 0
    obs_leaves = None
    for ei, (task, ep) in enumerate(eps):
        if (task, ep) in done_eps:
            continue
        rows = np.flatnonzero((keys[:, 0] == task) & (keys[:, 1] == ep))
        frames = keys[rows, 2]
        _, state = _episode(out, task, ep)
        imgs = {cam: _episode_frames(roots[task], cam, ep, frames) for cam in CAMERAS}
        prompt = demo[(task, ep)]["prompt"]
        for s in range(0, len(rows), int(batch_size)):
            sub = rows[s:s + int(batch_size)]
            items = [transform({"observation/image": imgs[CAMERAS[0]][int(keys[r, 2])],
                                "observation/right_image": imgs[CAMERAS[1]][int(keys[r, 2])],
                                "observation/wrist_image": imgs[CAMERAS[2]][int(keys[r, 2])],
                                "observation/state": ex.state_storage_to_policy(state[int(keys[r, 2])]),
                                "prompt": prompt}) for r in sub]
            n_real = len(items)
            items += [items[-1]] * (int(batch_size) - n_real)   # fixed batch shape: one compile
            stacked = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)
            leaves, treedef = jax.tree.flatten(stacked)
            if obs_leaves is None:
                obs_leaves = _open_obs_leaves(tok_dir, shard, leaves, treedef, len(keys), mode)
            for mm, leaf in zip(obs_leaves, leaves, strict=True):
                mm[sub] = leaf[:n_real]
            batch = jax.tree.map(jnp.asarray, stacked)
            hidden, mask = extractor(model_api.Observation.from_dict(batch))
            hidden = np.asarray(hidden)[:n_real]; mask = np.asarray(mask)[:n_real]
            if hidden.shape[1] < seq - int(text_tokens) or hidden.shape[-1] != 2048:
                raise ValueError(f"unexpected prefix shape {hidden.shape}")
            text_max = max(text_max, int(mask[:, len(CAMERAS) * IMAGE_TOKENS:].sum(1).max()))
            w = min(seq, hidden.shape[1])
            tokens[sub, :w] = hidden[:, :w]; valid[sub, :w] = mask[:, :w]
            if mask[:, w:].any():
                raise ValueError(f"valid text tokens beyond text_tokens={text_tokens}; raise it")
        done_eps.add((task, ep))
        if ei % 10 == 0 or ei + 1 == len(eps):
            tokens.flush(); valid.flush()
            for mm in obs_leaves or ():
                mm.flush()
            _json(ppath, {"episodes": sorted(done_eps), "text_valid_max": text_max})
            _log(out, "tokens", shard=shard, episodes_done=len(done_eps), episodes=len(eps),
                 contexts=int(len(keys)), seconds=round(time.time() - t0, 1))
    tokens.flush(); valid.flush()
    for mm in obs_leaves or ():
        mm.flush()
    _json(ppath, {"episodes": sorted(done_eps), "text_valid_max": text_max, "complete": True})
    return len(keys)


def _open_obs_leaves(tok_dir, shard, leaves, treedef, n, mode):
    """Memmaps for the transformed model inputs (images, state, tokenized prompt)."""
    import pickle
    meta = tok_dir / f"shard{shard:02d}_obs_meta.pkl"
    spec = [(tuple(x.shape[1:]), str(x.dtype)) for x in leaves]
    if meta.exists():
        old = pickle.loads(meta.read_bytes())
        if old["spec"] != spec or str(old["treedef"]) != str(treedef):
            raise ValueError("observation leaf layout differs from the existing shard")
    else:
        meta.write_bytes(pickle.dumps({"treedef": treedef, "spec": spec}))
    return [np.lib.format.open_memmap(tok_dir / f"shard{shard:02d}_obs_leaf{j:02d}.npy", mode=mode,
                                      dtype=np.dtype(dt), shape=(n, *shape)) for j, (shape, dt) in enumerate(spec)]


def _prefix_token_extractor(model):
    """Final-layer prefix hidden states (fp16) and validity mask, as served."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared.nnx_utils import module_jit

    class PrefixTokens(nnx.Module):
        def __init__(self, base):
            self.base = base

        def __call__(self, observation):
            obs = model_api.preprocess_observation(None, observation, train=False)
            tok, val, ar = self.base.embed_prefix(obs)
            positions = jnp.cumsum(val, axis=1) - 1
            (hidden, _), _ = self.base.PaliGemma.llm([tok, None], mask=make_attn_mask(val, ar), positions=positions)
            return hidden.astype(jnp.float16), val

    return module_jit(PrefixTokens(model).__call__)


class TokenStore:
    """Exact (task, episode, frame) -> token row lookup across shards (read-only mmap)."""

    def __init__(self, out):
        tok_dir = Path(out) / "tokens"
        self.tok_dir = tok_dir
        idx = np.load(tok_dir / "index.npz")
        keys, shard = idx["keys"], idx["shard"]
        self.tokens, self.valid, self.map = [], [], {}
        for s in range(int(idx["nshards"])):
            done = json.loads((tok_dir / f"shard{s:02d}_done.json").read_text())
            if not done.get("complete"):
                raise RuntimeError(f"token shard {s} is incomplete")
            # Optional byte-identical local copy (NFS reads were the training bottleneck).
            local = os.environ.get("SKILLNET_TOKEN_LOCAL")
            src = tok_dir / f"shard{s:02d}_tokens_fp16.npy"
            lp = Path(local) / src.name if local else None
            if lp is not None and lp.exists():
                if lp.stat().st_size != src.stat().st_size:
                    raise ValueError(f"local token copy differs in size: {lp}")
                src = lp
            self.tokens.append(np.load(src, mmap_mode="r"))
            self.valid.append(np.load(tok_dir / f"shard{s:02d}_valid.npy", mmap_mode="r"))
            for j, k in enumerate(map(tuple, keys[shard == s].tolist())):
                self.map[k] = (s, j)
        self.seq = self.tokens[0].shape[1]
        self.keys0 = tuple(int(x) for x in keys[0])

    def observation(self, task, episode, frame):
        """Transformed model-input PyTree (numpy, batched) for exact contexts."""
        import pickle
        import jax
        tok_dir = self.tok_dir
        if not hasattr(self, "_obs"):
            self._obs = []
            for s in range(len(self.tokens)):
                meta = pickle.loads((tok_dir / f"shard{s:02d}_obs_meta.pkl").read_bytes())
                leaves = [np.load(tok_dir / f"shard{s:02d}_obs_leaf{j:02d}.npy", mmap_mode="r")
                          for j in range(len(meta["spec"]))]
                self._obs.append((meta["treedef"], leaves))
        picks = [self.map[k] for k in zip(*(np.asarray(x).astype(np.int64).tolist() for x in (task, episode, frame)))]
        treedef = self._obs[0][0]
        nleaf = len(self._obs[0][1])
        stacked = [np.stack([self._obs[s][1][j][i] for s, i in picks]) for j in range(nleaf)]
        return jax.tree.unflatten(treedef, stacked)

    def pooled(self, task, episode, frame):
        """Masked mean of valid prefix tokens (P0/P5 input), computed once and cached."""
        path = self.tok_dir / "pooled_fp32.npy"
        if not hasattr(self, "_pooled"):
            if not path.exists():
                order = sorted(self.map.items(), key=lambda kv: kv[1])
                table = np.zeros((len(order), 2048), np.float32)
                self._row = {}
                for i, (k, (s, j)) in enumerate(order):
                    v = self.valid[s][j]
                    table[i] = self.tokens[s][j][v].astype(np.float32).mean(0)
                tmp = path.with_name(f"pooled_{os.getpid()}.tmp.npy")
                np.save(tmp, table); tmp.replace(path)
            self._pooled = np.load(path, mmap_mode="r")
            order = sorted(self.map.items(), key=lambda kv: kv[1])
            self._row = {k: i for i, (k, _) in enumerate(order)}
        ix = [self._row[k] for k in zip(*(np.asarray(x).astype(np.int64).tolist() for x in (task, episode, frame)))]
        return np.asarray(self._pooled[ix], np.float32)

    def get(self, task, episode, frame):
        task, episode, frame = (np.asarray(x).astype(np.int64) for x in (task, episode, frame))
        out = np.zeros((len(task), self.seq, 2048), np.float16)
        grp = np.full((len(task), self.seq), -1, np.int8)
        base = group_ids(self.seq - len(CAMERAS) * IMAGE_TOKENS)
        for i, k in enumerate(zip(task.tolist(), episode.tolist(), frame.tolist())):
            if k not in self.map:
                raise KeyError(f"no token row for {k}")
            s, j = self.map[k]
            out[i] = self.tokens[s][j]
            grp[i] = np.where(self.valid[s][j], base, -1)
        return out, grp


# ---------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "units", "rows", "oracle", "r1", "token-index", "tokens"))
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--smoke", action="store_true", help="allow fewer demos (wiring checks only)")
    p.add_argument("--demo-limit", type=int, default=N_DEMOS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--text-tokens", type=int, default=200)
    a = p.parse_args()
    _guard_output(a.out)
    if a.stage == "token-index":
        path = a.out / "tokens/index.npz"
        if path.exists():
            raise FileExistsError(path)
        keys = token_index(a.out)
        _save_npz(path, keys=keys, shard=shard_of_episodes(keys, a.nshards), nshards=np.int64(a.nshards))
        _log(a.out, "token-index", contexts=int(len(keys)), nshards=a.nshards)
        return
    if a.stage == "tokens":
        extract_tokens(a.out, shard=a.shard, nshards=a.nshards, batch_size=a.batch_size,
                       text_tokens=a.text_tokens)
        return
    if a.stage in ("prepare", "r1"):
        prepare(a.out, demo_limit=a.demo_limit, smoke=a.smoke)
    if a.stage in ("units", "r1"):
        build_units(a.out)
    if a.stage in ("rows", "r1"):
        build_rows(a.out)
    if a.stage in ("oracle", "r1"):
        oracle(a.out)


if __name__ == "__main__":
    main()
