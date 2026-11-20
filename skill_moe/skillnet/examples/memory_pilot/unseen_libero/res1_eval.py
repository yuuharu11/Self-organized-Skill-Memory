"""RES1 offline evaluation on the fixed KQ2-CL 20 Hz splits.

The selected checkpoint is read only from RES1's Dev-v2 selection.json.  Eval
labels are loaded for scoring after checkpoint selection and never affect the
draft, query, or checkpoint choice.
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
sys.path.insert(0, str(HERE))
import kq2cl_common as C
import kq2_run as KR
import kq2_data as KD
import res1_train as RT
import res1_experiment as RE
import kq2cl_obs as OBS

OUT = Path(os.environ.get("RES1_RUN_ROOT", C.ROOT / "res1-01" / "production"))
SPLIT_OFFSETS = {"learned": 0, "unseen20": 50_000_000, "dev2": 60_000_000}


def _log(event: str, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}
    with (OUT / "progress.log").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(row, ensure_ascii=False, default=float), flush=True)


def build_context_rows(split: str, query: str, *, device="cpu", batch=128) -> Path:
    """Build current KQ2 context on exact rows, reusing data-split key order."""
    if split not in ("learned", "unseen20", "dev2") or query not in ("i", "iii"):
        raise ValueError((split, query))
    dest = OUT / "contexts" / f"{split}_{query}.npz"
    if dest.exists():
        return dest
    # KQ2 row arrays live under kq2-01; KQ2-CL keeps only observation / retrieval artifacts.
    ctx = KR.Ctx(device, splits=(split,), vlm=(query == "iii"), out=KD.OUT)
    sp = ctx.splits[split]
    gs = KR.GPUSplit(sp, ctx.device, query == "iii")
    model, _ = C.load_query(query, ctx.device)
    out = {"key": np.asarray(sp["key"]), "h": [], "state": []}
    if query == "iii": out["vlm"] = []
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    with torch.inference_mode():
        for lo in range(0, len(sp["key"]), batch):
            hi = min(lo + batch, len(sp["key"]))
            ix = torch.arange(lo, hi, device=ctx.device)
            st = torch.as_tensor(sp["state"][lo:hi], device=ctx.device)
            q = RT.RC.query_context(model, gs, ix, state=st, visual=(query == "iii"))
            out["h"].append(q["h"].cpu().numpy().astype(np.float32))
            out["state"].append(q["state"].cpu().numpy().astype(np.float32))
            if query == "iii": out["vlm"].append(q["vlm"].cpu().numpy().astype(np.float16))
    arrays = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}
    if not np.array_equal(arrays["key"], np.load(C.KQ2 / "data" / split / "key.npy")):
        raise AssertionError("context row key/order differs from KQ2 split")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".npz.tmp")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    tmp.replace(dest)
    _log("offline_context_ready", split=split, query=query, rows=len(arrays["key"]))
    return dest


def _contexts(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def _load_selected():
    meta = C.read_json(OUT / "selection.json")
    if not meta or meta.get("smoke"):
        raise RuntimeError("full Dev-v2 checkpoint selection is required")
    result = {}
    spec = {"flow_iii": ("iii", "flow", False), "flow_iii_x": ("iii", "flow", True),
            "flow_i": ("i", "flow", False), "reg_iii": ("iii", "regression", False)}
    for name, (query, objective, delta) in spec.items():
        sel = meta["selected"][name]
        run = sel["run"]
        train = C.read_json(OUT / "train" / run / "train.json")
        step = int(train["selected"]["checkpoint_step"])
        result[name] = {"query": query, "objective": objective, "use_delta": delta,
                        "run": run, "checkpoint_step": step,
                        "checkpoint": OUT / "train" / run / "checkpoints" / f"{step:06d}",
                        "dev2_arm6_mse": float(sel["dev2_arm6_mse"])}
    return result


def _episode_groups(split):
    keys = np.load(C.KQ2 / "data" / split / "key.npy")
    return np.asarray(keys[:, 0], np.int64) * 10_000_000 + np.asarray(keys[:, 1], np.int64)


def evaluate_split(split: str, name: str, cfg: dict, *, batch=8) -> dict:
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import module_jit
    import kq2cl_obs as O

    query, objective, delta = cfg["query"], cfg["objective"], cfg["use_delta"]
    data = np.load(C.KQ2 / "data" / split / "actions.npy").astype(np.float32)
    retrieval = np.load(C.OUT / "retrieval" / f"{query}__{split}.npz")
    unit = retrieval["ids"][:, 0].astype(np.int64)
    if len(unit) != len(data): raise AssertionError("retrieval and target row count mismatch")
    bank = np.load(KD.KF_DATA / "bank_actions.npy", mmap_mode="r").astype(np.float32)
    bt = np.load(KD.KF_DATA / "bank__task.npy"); be = np.load(KD.KF_DATA / "bank__episode.npy")
    qkey = np.load(C.KQ2 / "data" / split / "key.npy")
    if np.any((bt[unit] == qkey[:, 0]) & (be[unit] == qkey[:, 1])):
        raise AssertionError("retrieval top1 leaked query episode")
    hist = np.load(C.KQ2 / "data" / split / "history.npy").astype(np.float32)
    hist_mask = np.load(C.KQ2 / "data" / split / "history_mask.npy").astype(bool)
    draft = np.zeros((len(unit), 10, 32), np.float32)
    draft[..., :6] = bank[unit, :, :6]
    draft[..., 6] = C.hold_last_gripper(hist[:, -1, 6], hist_mask.any(1))
    obs_rows = O.load_obs(split)
    base = EX3_BASE = __import__("expert_ex3")._base()[1]
    # All observations are indexed in the same KQ2 split order.
    obs = base.obs_batch(obs_rows, np.arange(len(unit)))
    current = _contexts(build_context_rows(split, query))
    saved = RT._load_saved_context(query)
    exp3, base, model, filt = RT._build_expert(query, delta, seed=RT.SEED)
    RT._restore_state(model, filt, cfg["checkpoint"])
    if not np.isfinite(np.asarray(data)).all(): raise ValueError("nonfinite targets")
    saved_batch = current_batch = valid = None
    if delta:
        saved_batch, current_batch, valid = RT._context_batch(saved, current, unit,
                                                              np.arange(len(unit)), query)
        saved_batch = jax.tree.map(jnp.asarray, {k:v for k,v in saved_batch.items() if k != "vlm_valid"})
        current_batch = jax.tree.map(jnp.asarray, {k:v for k,v in current_batch.items() if k != "vlm_valid"})
        valid = None if valid is None else jnp.asarray(valid)
    noise_seed_offset = SPLIT_OFFSETS[split]
    rng = np.random.default_rng(4101 + noise_seed_offset)
    noise_all = rng.standard_normal((len(unit), 10, 6)).astype(np.float32)
    if delta:
        def run(m, ob, dr, eps, sc, cc, vv):
            if objective == "regression":
                correction = RE.res1_velocity(m, ob, dr, jnp.zeros_like(eps), jnp.zeros((len(dr),), jnp.float32),
                                               saved_context=sc, current_context=cc, context_valid=vv, use_delta=True)
                return dr[..., :6] + correction
            def vf(o, d, r, t, saved_context=None, current_context=None, context_valid=None, prefix_context=None):
                return RE.res1_velocity(m, o, d, r, t, saved_context=saved_context,
                    current_context=current_context, context_valid=context_valid, use_delta=True)
            return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10, saved_context=sc,
                current_context=cc, context_valid=vv)
        fn = nnx.jit(run)
    else:
        def run(m, ob, dr, eps):
            if objective == "regression":
                correction = RE.res1_velocity(m, ob, dr, jnp.zeros_like(eps), jnp.zeros((len(dr),), jnp.float32))
                return dr[..., :6] + correction
            def vf(o, d, r, t, saved_context=None, current_context=None, context_valid=None, prefix_context=None):
                return RE.res1_velocity(m, o, d, r, t)
            return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10)
        fn = nnx.jit(run)
    predictions = []
    for lo in range(0, len(unit), batch):
        hi = min(lo + batch, len(unit)); sl = slice(lo, hi)
        ob = jax.tree.map(lambda x: x[sl], obs)
        args = [model, ob, jnp.asarray(draft[sl]), jnp.asarray(noise_all[sl])]
        if delta:
            sc = jax.tree.map(lambda x: x[sl], saved_batch); cc = jax.tree.map(lambda x: x[sl], current_batch)
            vv = None if valid is None else valid[sl]
            pred = fn(*args, sc, cc, vv)
        else: pred = fn(*args)
        predictions.append(np.asarray(pred, np.float32))
    pred_arm = np.concatenate(predictions)
    final = draft.copy(); final[..., :6] = pred_arm
    target7 = data[..., :7]
    arm_per = C.arm6(final, target7)
    active_per = np.square(final[..., :7] - target7).sum((-1, -2)) / 70.
    draft_per = C.arm6(draft, target7)
    diff = pred_arm - draft[..., :6]
    draft_norm = np.linalg.norm(draft[..., :6], axis=(-1, -2))
    residual_norm = np.linalg.norm(diff, axis=(-1, -2))
    ratio = residual_norm / np.maximum(draft_norm, 1e-8)
    # Gate is scalar; include it even for models without RES1-X.
    gate = float(np.tanh(np.asarray(model.res1_delta.gate.value))) if delta else 0.
    # Shared KQ2 canonical subsets; first_unit and post_switch are the decision-point rows.
    sp = {k: np.load(C.KQ2 / "data" / split / f"{k}.npy", mmap_mode="r")
          for k in ("task", "episode", "start", "key", "actions")}
    subsets = KR.subsets_for(split, sp)
    groups = _episode_groups(split)
    metrics = {"rows": len(unit), "arm6_mse": float(arm_per.mean()),
               "arm6_ci95": C.bootstrap_ci(arm_per, groups, reps=5000),
               "active7_mse_hold_grip": float(active_per.mean()),
               "draft_arm6_mse": float(draft_per.mean()),
               "draft_to_output_arm6_mse": float(np.square(diff).sum((-1,-2)).mean()/60.),
               "residual_ratio": C.quantiles(ratio), "delta_gate_tanh": gate,
               "subsets": {k: {"rows": int(v.sum()), "arm6_mse": float(arm_per[v].mean()) if v.any() else None,
                               "draft_arm6_mse": float(draft_per[v].mean()) if v.any() else None,
                               "residual_ratio_mean": float(ratio[v].mean()) if v.any() else None}
                           for k,v in subsets.items()}}
    dest = OUT / "offline" / f"{split}_{name}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, prediction=pred_arm, unit_id=unit, arm6_mse_per_row=arm_per,
                        draft_arm6_mse_per_row=draft_per, residual_ratio_per_row=ratio,
                        task=qkey[:,0], episode=qkey[:,1], frame=qkey[:,2])
    return metrics


def run(splits=("learned", "unseen20", "dev2"), *, batch=8):
    selected = _load_selected()
    results = {"selection": selected, "splits": {}}
    for split in splits:
        # Query caches are built on exact key order; normal Eval label access is scoring only.
        OBS.build_obs(split)
        results["splits"][split] = {}
        for name, cfg in selected.items():
            m = evaluate_split(split, name, cfg, batch=batch)
            results["splits"][split][name] = m
            _log("offline_condition_complete", split=split, condition=name, rows=m["rows"],
                 arm6_mse=m["arm6_mse"], residual_ratio_mean=m["residual_ratio"]["mean"],
                 delta_gate=m["delta_gate_tanh"])
    path = OUT / "offline" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="learned,unseen20,dev2")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(run(tuple(a.splits.split(",")), batch=a.batch), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
