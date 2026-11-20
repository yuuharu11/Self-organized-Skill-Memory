"""RES1 production training and Dev-v2 checkpoint selection.

Stages are restartable and write beneath res1-01/production by default (or
RES1_RUN_ROOT when set). Training
uses KQ2's measured retrieval-rank law and the exact saved context cache made by
res1_exact_context.py. Eval splits are not loaded by this module.
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
import expert_ex3 as EX3
import kq2_data as KD
import kq2_run as KR
import kq2cl_common as C
import res1_context as RC
from res1_core import residual_ratio
import res1_experiment as RE

OUT = Path(os.environ.get("RES1_RUN_ROOT", C.ROOT / "res1-01" / "production"))
CTX_DIR = C.ROOT / "res1-01" / "context-exact-01"
K1_BASE = C.OUT / "expert" / "train-K1" / "checkpoints" / "000500"
TRAIN_SPLIT = "eh"
SAVE_EVERY = 500
DEFAULT_UPDATES = 5000
SEED = 20261120


def log(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **payload}
    with (OUT / "progress.log").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(rec, ensure_ascii=False, default=float), flush=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float) + "\n")
    tmp.replace(path)


def exact_row_intersection(query_keys: np.ndarray, row_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map KQ2 rows to exact LeRobot rows; never substitute a nearby frame."""
    lookup = {tuple(map(int, k)): i for i, k in enumerate(np.asarray(row_keys))}
    qids = np.asarray([i for i, k in enumerate(np.asarray(query_keys))
                       if tuple(map(int, k)) in lookup], np.int64)
    rids = np.asarray([lookup[tuple(map(int, query_keys[i]))] for i in qids], np.int64)
    return qids, rids


def assert_action_alignment(query_actions: np.ndarray, row_actions: np.ndarray) -> None:
    """Fail closed if exact-key windows disagree in their seven action axes."""
    q = np.asarray(query_actions, np.float32)
    r = np.asarray(row_actions, np.float32)[..., :7]
    if q.shape != r.shape:
        raise ValueError(f"action-window shape mismatch: {q.shape} vs {r.shape}")
    if not np.allclose(q, r, rtol=0., atol=1e-6):
        raise AssertionError(f"exact-key action windows differ: max_abs={np.max(np.abs(q-r))}")


def build_context_rows(split: str, query: str, *, device: str = "cuda", batch: int = 128) -> Path:
    """Cache the current query context for KQ2 train rows or Dev v2 rows."""
    if split not in ("eh", "dev2") or query not in ("i", "iii"):
        raise ValueError("only eh/dev2 and i/iii are supported")
    dest = OUT / "contexts" / f"{split}_{query}.npz"
    if dest.exists():
        return dest
    ctx = KR.Ctx(device, splits=(split,), vlm=(query == "iii"), out=C.KQ2)
    sp = ctx.splits[split]
    gs = KR.GPUSplit(sp, ctx.device, query == "iii")
    model, _cfg = C.load_query(query, ctx.device)
    h, state, vlm = [], [], []
    with __import__("torch").inference_mode():
        for lo in range(0, len(sp["key"]), batch):
            ix = __import__("torch").arange(lo, min(lo + batch, len(sp["key"])), device=ctx.device)
            st = __import__("torch").as_tensor(sp["state"][lo:lo + len(ix)], device=ctx.device)
            q = RC.query_context(model, gs, ix, state=st, visual=(query == "iii"))
            h.append(q["h"].cpu().numpy().astype(np.float32))
            state.append(q["state"].cpu().numpy().astype(np.float32))
            if query == "iii":
                vlm.append(q["vlm"].cpu().numpy().astype(np.float16))
    arrays = {"key": np.asarray(sp["key"]), "h": np.concatenate(h), "state": np.concatenate(state)}
    if query == "iii":
        arrays["vlm"] = np.concatenate(vlm)
        arrays["vlm_valid"] = np.ones(arrays["vlm"].shape[:2], bool)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(dest)
    log("current_context_ready", split=split, query=query, rows=len(arrays["key"]), shape=list(arrays["h"].shape))
    return dest


def prepare_order(*, batch: int = 128) -> Path:
    """Cache per-row candidate ranks on the exact shared KQ2/LeRobot frames.

    KQ2 EH is denser than the LeRobot row cache used by the Expert runner. We
    keep only the 6,583 exact key intersections instead of borrowing a nearby
    image for the other 6,183 rows.
    """
    d = OUT / "drafts"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / "order_all.npy"
    complete = (dest.exists() and (d / "order_excluded.npy").exists()
                and (d / "training_rows.npz").exists())
    if complete:
        return dest
    if dest.exists() or (d / "order_excluded.npy").exists() or (d / "training_rows.npz").exists():
        raise RuntimeError(f"partial draft-order cache exists at {d}; use a new RES1_RUN_ROOT")
    eh = C.KQ2 / "data" / "eh"
    key = np.load(eh / "key.npy").astype(np.int64)
    run_root = C.RUN40
    rows_dir = run_root / "rows"
    row_key = np.stack([np.load(rows_dir / f"{k}.npy", mmap_mode="r") for k in ("task", "episode", "frame")], axis=1).astype(np.int64)
    kq_ids, run_ids = exact_row_intersection(key, row_key)
    if len(kq_ids) != 6583:
        raise AssertionError(f"expected exact input frames for 6,583 KQ2 eh rows, got {len(kq_ids)}")
    target_full = np.load(eh / "actions.npy", mmap_mode="r")
    target = np.asarray(target_full[kq_ids], np.float32)[..., :6].reshape(len(kq_ids), -1)
    run_actions = np.load(rows_dir / "actions.npy", mmap_mode="r")
    assert_action_alignment(np.asarray(target_full[kq_ids]), np.asarray(run_actions[run_ids]))
    task = np.load(eh / "task.npy")[kq_ids].astype(np.int64)
    episode = np.load(eh / "episode.npy")[kq_ids].astype(np.int64)
    bt = np.load(KD.KF_DATA / "bank__task.npy").astype(np.int64)
    be = np.load(KD.KF_DATA / "bank__episode.npy").astype(np.int64)
    units = np.load(KD.KF_DATA / "bank_actions.npy", mmap_mode="r").astype(np.float32)[..., :6].reshape(4516, -1)
    mm = np.lib.format.open_memmap(dest, mode="w+", dtype=np.int16, shape=(len(kq_ids), len(units)))
    order_exc = np.lib.format.open_memmap(d / "order_excluded.npy", mode="w+", dtype=np.int16, shape=(len(kq_ids), len(units)))
    for lo in range(0, len(kq_ids), batch):
        hi = min(lo + batch, len(kq_ids))
        y = target[lo:hi].astype(np.float64); x = units.astype(np.float64)
        dist = np.square(y).sum(1, keepdims=True) + np.square(x).sum(1)[None] - 2 * y @ x.T
        dist = np.maximum(dist, 0)
        own_ep = ((task[lo:hi, None] == bt[None]) & (episode[lo:hi, None] == be[None]))
        dist_all = dist.copy(); dist_all[own_ep] = np.inf
        mm[lo:hi] = np.argsort(dist_all, axis=1, kind="stable").astype(np.int16)
        dist_exc = dist.copy(); dist_exc[task[lo:hi, None] == bt[None]] = np.inf
        order_exc[lo:hi] = np.argsort(dist_exc, axis=1, kind="stable").astype(np.int16)
    mm.flush(); order_exc.flush()
    np.savez_compressed(d / "training_rows.npz", kq_row_ids=kq_ids, run_row_ids=run_ids,
                        target=np.asarray(target_full[kq_ids], np.float32), task=task, episode=episode,
                        key=key[kq_ids])
    write_json(d / "training_rows.json", {
        "query_split": "KQ2-CL eh", "run40": str(run_root), "rows": int(len(kq_ids)),
        "query_rows": int(len(key)), "exact_key_intersection_only": True,
        "unmatched_query_rows_skipped": int(len(key) - len(kq_ids)), "candidate_units": int(len(units)),
        "own_episode_masked_order": str(dest), "own_task_masked_order": str(d / "order_excluded.npy"),
        "action_alignment": "exact key, normalized seven active dimensions, asserted before caching",
        "no_nearest_frame_substitution": True,
    })
    log("draft_order_ready", rows=len(kq_ids), exact_frames=6583, skipped_no_exact_frame=6183,
        candidates=len(units), exclusions="own episode / own task")
    return dest


def _ctx_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_saved_context(which: str) -> dict:
    p = CTX_DIR / f"bank_{which}.npz"
    meta = p.with_suffix(".json")
    if not p.exists():
        raise FileNotFoundError(f"run res1_exact_context.py first: {p}")
    if not meta.exists() or not json.loads(meta.read_text()).get("unit_start_exact", False):
        raise ValueError("saved context metadata is not exact-start")
    context = _ctx_arrays(p)
    source = np.load(KD.KF_DATA / 'bank__source.npy', allow_pickle=False)
    if not np.array_equal(context.get('bank_source'), source):
        raise ValueError('saved context bank source/order mismatch')
    return context


def select_ranked_units(order_all, order_excluded, rows, ranks, excluded,
                        query_task, query_episode, bank_task, bank_episode):
    """Apply the sampled rank only within the row's eligible candidate prefix."""
    qt = np.asarray(query_task)[rows]
    qe = np.asarray(query_episode)[rows]
    own_task = qt[:, None] == np.asarray(bank_task)[None]
    own_episode = own_task & (qe[:, None] == np.asarray(bank_episode)[None])
    forbidden = np.where(np.asarray(excluded)[:, None], own_task, own_episode)
    eligible = (~forbidden).sum(1)
    if np.any(eligible == 0):
        raise ValueError('no eligible unit for sampled draft')
    positions = np.minimum(np.maximum(np.asarray(ranks) - 1, 0), eligible - 1)
    chosen = np.asarray([int((order_excluded if exc else order_all)[row, pos])
                         for row, pos, exc in zip(rows, positions, excluded)], np.int64)
    if np.any(forbidden[np.arange(len(chosen)), chosen]):
        raise AssertionError('rank-order cache selected an excluded task/episode')
    return chosen


def _load_current(split: str, query: str) -> dict:
    return _ctx_arrays(build_context_rows(split, query))


def _row_context(ctx: dict, idx: np.ndarray, *, query: str) -> dict:
    out = {"h": ctx["h"][idx], "state": ctx["state"][idx]}
    if query == "iii":
        out["vlm"] = ctx["vlm"][idx]
        out["vlm_valid"] = ctx["vlm_valid"][idx]
    return out


def _context_batch(saved: dict, current: dict, units: np.ndarray, rows: np.ndarray, query: str):
    s = _row_context(saved, units, query=query)
    c = _row_context(current, rows, query=query)
    valid = c.get("vlm_valid")
    return s, c, valid


def _build_expert(query: str, use_delta: bool, *, seed: int):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    exp3, base = EX3._base()
    model, _ = base.load_policy(C.PI05)
    filt0 = EX3._filter()
    with ocp.PyTreeCheckpointer() as cp:
        restore = cp.restore(K1_BASE)
    st = nnx.state(model, filt0); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, restore)); nnx.update(model, st)
    if use_delta:
        RE.attach_delta_attention(model, rng_seed=seed)
    return exp3, base, model, RE.res1_trainable_filter()


def _save_state(model, filt, path: Path):
    import flax.nnx as nnx
    import jax
    import orbax.checkpoint as ocp
    path.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as cp:
        cp.save(path, jax.tree.map(lambda x: x, nnx.state(model, filt).to_pure_dict()), force=True)


def _restore_state(model, filt, path: Path):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as cp:
        state = cp.restore(path)
    cur = nnx.state(model, filt); cur.replace_by_pure_dict(jax.tree.map(jnp.asarray, state)); nnx.update(model, cur)


def _dev_score(model, base, query: str, use_delta: bool, *, checkpoint_step: int,
               saved: dict, current: dict, batch_size: int = 8, max_rows: int | None = None,
               objective: str = 'flow') -> dict:
    """Dev-v2-only score: top-1 draft, fixed noise, end residual output."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import module_jit
    import kq2cl_obs as O
    d = O.load_obs("dev2")
    targets = np.load(C.KQ2 / "data" / "dev2" / "actions.npy").astype(np.float32)
    retrieval = np.load(C.OUT / "retrieval" / f"{query}__dev2.npz")
    unit_id = retrieval["ids"][:, 0].astype(np.int64)
    if max_rows is not None:
        unit_id = unit_id[:max_rows]
        targets = targets[:max_rows]
        d = {k: v[:max_rows] if isinstance(v, np.ndarray) and len(v) == len(retrieval["ids"]) else v
             for k, v in d.items()}
    units = np.load(KD.KF_DATA / "bank_actions.npy", mmap_mode="r").astype(np.float32)
    drafts = np.zeros((len(unit_id), 10, 32), np.float32)
    drafts[..., :6] = units[unit_id, :, :6]
    hist = np.load(C.KQ2 / "data" / "dev2" / "history.npy")[:len(unit_id)]
    mask = np.load(C.KQ2 / "data" / "dev2" / "history_mask.npy")[:len(unit_id)]
    drafts[..., 6] = C.hold_last_gripper(hist[:, -1, 6], mask.any(1))
    obs = base.obs_batch(d, np.arange(len(unit_id)))
    if objective not in ('flow', 'regression'):
        raise ValueError('objective must be flow or regression')
    noise = np.random.default_rng(SEED).standard_normal((len(unit_id), 10, 6)).astype(np.float32)
    if use_delta:
        def run(m, ob, dr, eps, sc, cc, valid):
            if objective == 'regression':
                correction = RE.res1_velocity(m, ob, dr, jnp.zeros_like(eps),
                    jnp.zeros((len(dr),), jnp.float32), saved_context=sc,
                    current_context=cc, context_valid=valid, use_delta=True)
                return dr[..., :6] + correction
            def vf(o, draft, r, u, saved_context=None, current_context=None, context_valid=None,
                   prefix_context=None):
                return RE.res1_velocity(m, o, draft, r, u, saved_context=saved_context,
                                        current_context=current_context, context_valid=context_valid,
                                        use_delta=True)
            return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10,
                                           saved_context=sc, current_context=cc, context_valid=valid)
        fn = nnx.jit(run)
    else:
        def run(m, ob, dr, eps):
            if objective == 'regression':
                correction = RE.res1_velocity(m, ob, dr, jnp.zeros_like(eps),
                                              jnp.zeros((len(dr),), jnp.float32))
                return dr[..., :6] + correction
            def vf(o, draft, r, u, saved_context=None, current_context=None, context_valid=None,
                   prefix_context=None):
                return RE.res1_velocity(m, o, draft, r, u)
            return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10)
        fn = nnx.jit(run)
    preds = []
    with jax.disable_jit(False):
        for lo in range(0, len(unit_id), batch_size):
            hi = min(lo + batch_size, len(unit_id)); sl = slice(lo, hi)
            ob = jax.tree.map(lambda x: x[sl], obs)
            if use_delta:
                sc, cc, valid = _context_batch(saved, current, unit_id[sl], np.arange(lo, hi), query)
                sc = jax.tree.map(jnp.asarray, {k: v for k, v in sc.items() if k != "vlm_valid"})
                cc = jax.tree.map(jnp.asarray, {k: v for k, v in cc.items() if k != "vlm_valid"})
                preds.append(np.asarray(fn(model, ob, jnp.asarray(drafts[sl]), jnp.asarray(noise[sl]),
                                          sc, cc, None if valid is None else jnp.asarray(valid))))
            else:
                preds.append(np.asarray(fn(model, ob, jnp.asarray(drafts[sl]), jnp.asarray(noise[sl]))))
    pred = np.concatenate(preds)
    return {"split": "dev2", "rows": len(pred), "arm6_mse": float(C.arm6(pred, targets).mean()),
            "residual_ratio": residual_ratio(pred[..., :6] - drafts[..., :6], drafts[..., :6]),
            "checkpoint_step": checkpoint_step}


def train_one(name: str, query: str, objective: str = "flow", use_delta: bool = False,
              *, own_task_fraction: float = .5, updates: int = DEFAULT_UPDATES, batch: int = 8,
              save_every: int = SAVE_EVERY, seed: int = SEED, resume: bool = True,
              smoke: bool = False) -> dict:
    """Train one RES1 variant; resume from its own checkpoints; select by Dev-v2 arm6."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    if query not in ("i", "iii") or objective not in ("flow", "regression"):
        raise ValueError("query must be i/iii, objective flow/regression")
    if use_delta and query != "iii":
        raise ValueError("RES1-X is defined only for the iii visual-context query")
    if not (0 <= own_task_fraction <= 1):
        raise ValueError("own-task exclusion fraction outside [0,1]")
    if not C.OUT.joinpath("DONE").exists():
        raise RuntimeError("KQ2-CL must be complete before RES1 training")
    if not (CTX_DIR / "bank_i.npz").exists() or not (CTX_DIR / "bank_iii.npz").exists():
        raise FileNotFoundError("exact-start saved contexts must be complete first")

    tdir = OUT / "train" / name
    ckdir = tdir / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    state_path = tdir / "train.json"
    exp3, base, model, filt = _build_expert(query, use_delta, seed=seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    train_rows = _ctx_arrays(OUT / "drafts" / "training_rows.npz")
    order_exc = np.load(OUT / "drafts" / "order_excluded.npy", mmap_mode="r")
    order_all = np.load(OUT / "drafts" / "order_all.npy", mmap_mode="r")
    units = np.load(KD.KF_DATA / "bank_actions.npy", mmap_mode="r").astype(np.float32)
    bank_task = np.load(KD.KF_DATA / 'bank__task.npy', allow_pickle=False)
    bank_episode = np.load(KD.KF_DATA / 'bank__episode.npy', allow_pickle=False)
    target = train_rows["target"].astype(np.float32)
    run_row_ids = train_rows["run_row_ids"].astype(np.int64)
    kq_row_ids = train_rows["kq_row_ids"].astype(np.int64)
    last_hist = np.load(C.KQ2 / "data" / "eh" / "history.npy", mmap_mode="r")[kq_row_ids]
    valid_hist = np.load(C.KQ2 / "data" / "eh" / "history_mask.npy", mmap_mode="r")[kq_row_ids]
    n = len(run_row_ids)
    # Rank laws correspond to the matched KQ2 retrievers; top-rank distribution is
    # transferred to action-error-ordered candidates, exactly as KQ2-CL Expert prep.
    law_exc = np.load(C.OUT / "retrieval" / f"{query}__eh_oofT.npz")["rank"].astype(np.int64)[kq_row_ids]
    law_all = np.load(C.OUT / "retrieval" / f"{query}__eh_insE.npz")["rank"].astype(np.int64)[kq_row_ids]
    current = _load_current("eh", query)
    saved = _load_saved_context(query)
    rows40 = base.load_rows(C.RUN40)
    filt_base = EX3._filter()
    start = 0
    opt_state = optimizer.init(nnx.state(model, filt))
    present = sorted(int(q.name) for q in ckdir.iterdir() if q.name.isdigit() and (ckdir / f"{q.name}.opt").exists())
    if resume and present:
        start = present[-1]
        _restore_state(model, filt, ckdir / f"{start:06d}")
        with ocp.PyTreeCheckpointer() as cp:
            opt_state = cp.restore(ckdir / f"{start:06d}.opt", item=opt_state)
    else:
        # K1 seed is the exact selected action-expert init; the added RES1-X gate is zero.
        if use_delta:
            if abs(float(np.asarray(model.res1_delta.gate.value))) > 1e-8:
                raise AssertionError("RES1-X must start with a zero tanh gate")
    if opt_state is None:
        opt_state = optimizer.init(nnx.state(model, filt))
    meta = C.read_json(state_path) or {"name": name, "query": query, "objective": objective,
        "use_delta": use_delta, "own_task_exclusion_fraction": own_task_fraction,
        "rank_law": {"excluded": f"{query}__eh_oofT", "included": f"{query}__eh_insE"},
        "init_checkpoint": str(K1_BASE), "updates": updates, "batch": batch, "seed": seed,
        "optimizer": "AdamW 2.5e-5, clip 1, b1=.9 b2=.95 wd=1e-5", "history": [], "status": "running"}
    grad_step = RE.make_nnx_train_step(use_delta=use_delta, objective=objective)
    rng = np.random.default_rng(seed + start)
    jax_rng = jax.random.key(seed + start)
    for step in range(start + 1, updates + 1):
        pos = rng.integers(0, n, size=batch)
        excl = rng.random(batch) < own_task_fraction
        rank = np.empty(batch, np.int64)
        if excl.any(): rank[excl] = law_exc[rng.integers(0, len(law_exc), size=int(excl.sum()))]
        if (~excl).any(): rank[~excl] = law_all[rng.integers(0, len(law_all), size=int((~excl).sum()))]
        rank = np.maximum(1, rank)
        unit = select_ranked_units(order_all, order_exc, pos, rank, excl,
                                   train_rows['task'], train_rows['episode'], bank_task, bank_episode)
        rows = run_row_ids[pos]
        context_rows = kq_row_ids[pos]
        draft = np.zeros((batch, 10, 32), np.float32)
        draft[..., :6] = units[unit, :, :6]
        draft[..., 6] = C.hold_last_gripper(last_hist[pos, -1, 6], valid_hist[pos].any(1))
        act = np.zeros_like(draft); act[..., :7] = target[pos, :, :7]
        obs = base.obs_batch(rows40, rows)
        sc = cc = valid = None
        if use_delta:
            sc, cc, valid = _context_batch(saved, current, unit, context_rows, query)
            sc = jax.tree.map(jnp.asarray, {k: v for k, v in sc.items() if k != "vlm_valid"})
            cc = jax.tree.map(jnp.asarray, {k: v for k, v in cc.items() if k != "vlm_valid"})
            valid = None if valid is None else jnp.asarray(valid)
        jax_rng, rk1, rk2 = jax.random.split(jax_rng, 3)
        noise = jax.random.normal(rk1, (batch, 10, 6), dtype=jnp.float32)
        progress = jax.random.uniform(rk2, (batch,), minval=0., maxval=1.)
        value, grads = grad_step(model, obs, jnp.asarray(act), jnp.asarray(draft), noise, progress,
                                 sc, cc, valid)
        state = nnx.state(model, filt)
        updates_tree, opt_state = optimizer.update(grads, opt_state, state)
        nnx.update(model, optax.apply_updates(state, updates_tree))
        if step % 50 == 0 or step == 1:
            log("train_step", name=name, step=step, loss=float(value))
        if step % save_every == 0 or step == updates:
            checkpoint = ckdir / f"{step:06d}"
            _save_state(model, filt, checkpoint)
            with ocp.PyTreeCheckpointer() as cp:
                cp.save(ckdir / f"{step:06d}.opt", opt_state, force=True)
            devctx = _load_current("dev2", query)
            result = _dev_score(model, base, query, use_delta, checkpoint_step=step,
                                saved=saved, current=devctx, max_rows=8 if smoke else None,
                                objective=objective)
            result.update({"train_loss": float(value), "own_task_exclusion_fraction": own_task_fraction})
            meta["history"].append(result)
            meta["selected"] = min(meta["history"], key=lambda z: z["arm6_mse"])
            write_json(state_path, meta)
            log("dev_checkpoint", name=name, step=step, dev_arm6=result["arm6_mse"])
    meta["status"] = "complete"
    write_json(state_path, meta)
    return meta


def train_conditions(conditions: list[str], *, updates=DEFAULT_UPDATES, batch=8, resume=True,
                    own_task_fractions: tuple[float, ...] = (0., .5, 1.), smoke: bool = False):
    spec = {
        "flow_iii": ("iii", "flow", False),
        "flow_iii_x": ("iii", "flow", True),
        "flow_i": ("i", "flow", False),
        "reg_iii": ("iii", "regression", False),
    }
    out = {}
    for name in conditions:
        if name not in spec:
            raise ValueError(f"unknown condition {name}")
        query, objective, delta = spec[name]
        # Exclusion-rate candidates are trained independently; one command can be
        # resumed per named run. Explicit suffix keeps checkpoints non-overwriting.
        for frac in own_task_fractions:
            label = f"{name}_exclude{int(frac*100):02d}"
            out[label] = train_one(label, query, objective, delta, own_task_fraction=frac,
                                  updates=updates, batch=batch, resume=resume, smoke=smoke)
    selected = {}
    for name in conditions:
        cands = [v for k, v in out.items() if k.startswith(name + "_exclude") and v.get("selected")]
        if cands:
            best = min(cands, key=lambda r: r["selected"]["arm6_mse"])
            selected[name] = {"run": best["name"], "own_task_exclusion_fraction": best["own_task_exclusion_fraction"],
                              "checkpoint_step": best["selected"]["checkpoint_step"],
                              "dev2_arm6_mse": best["selected"]["arm6_mse"]}
    write_json(OUT / "selection.json", {"selection_split": "dev2 only", "metric": "top-1 residual output arm6 MSE",
                                         "smoke": bool(smoke), "selected": selected})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("contexts", "order", "train"))
    ap.add_argument("--splits", default="eh,dev2")
    ap.add_argument("--queries", default="i,iii")
    ap.add_argument("--conditions", default="flow_iii,flow_iii_x,flow_i,reg_iii")
    ap.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--own-task-fractions", default="0,.5,1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2-update actual K1 smoke, Dev scoring limited to first 8 Dev-v2 rows")
    a = ap.parse_args()
    if a.stage == "contexts":
        for s in a.splits.split(","):
            for q in a.queries.split(","):
                build_context_rows(s, q)
    elif a.stage == "order":
        print(prepare_order())
    else:
        print(json.dumps(train_conditions(a.conditions.split(","), updates=a.updates,
                                          batch=a.batch, resume=not a.no_resume,
                                          own_task_fractions=tuple(float(x) for x in a.own_task_fractions.split(",")),
                                          smoke=a.smoke), indent=2))


if __name__ == "__main__":
    main()
