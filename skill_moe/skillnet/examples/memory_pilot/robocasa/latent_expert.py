#!/usr/bin/env python3
"""R3: one key-agnostic skill Expert that restores actions from stored DCT-3 units.

Design (decided with the user): the Expert learns, once, how to recover the true
actions from a draft made of stored motion units. It never sees which key chose
the draft, so any key's retrieval can be plugged in at inference. The denoising
strength t0 grows with the draft's error during training; at inference t0 is
set from an estimate of that error (a confidence), with a fixed t0 and the true
error as references.

Stages (under ``--out``, after ``latent_skill r1/tokens`` and ``latent_key``):

  drafts  50-step drafts.
            train rows: key-free "spread" drafts. Each chain step takes the unit of
                        rank r by arm6 error to the true next 10 actions (r drawn
                        log-uniformly in [0, 500) per row, kept for the whole
                        chain), so draft quality covers what retrieval produces.
            dev/eval:   oracle (r=0) and the top-1 chains of trained keys, which
                        re-query with the own draft appended to the history and
                        the same observation tokens/state.
          Same-episode units are always excluded. Units play at natural length.
  train   bridge-train the action expert of the fine-tuned RoboCasa π0.5 from
          (1-t0)*draft + t0*noise to the true 50 actions. VLM frozen; trainable
          subset/optimizer as expert-03. t0 = map(draft error) + U(-.05,.05),
          where map is the training-draft error CDF scaled to [T0_MIN, T0_MAX].
  eval    every Dev/Eval row: normal π0.5, hold-last, drafts, and bridges of the
          untrained π0.5 and the Expert at every t0 in T0_EVAL; per-row errors
          and confidence signals are saved.
  report  t0 policies, chosen/fitted on Dev, scored on Eval:
            fixed     one t0 per (weights, draft) with the lowest Dev error
            conf      t0 = map(predicted error), predicted from deployable
                      signals (draft-vs-π0.5 distance, key score margin) by a
                      log-linear fit on Dev
            true-err  t0 = map(true draft error)       (upper bound)
            row-best  best grid t0 per row             (upper bound)
          The gripper after the bridge is replaced by normal π0.5 (EX1).

Metric: arm6 over the first 10 steps / (10*7) in checkpoint z-score space
(LIBERO convention), plus first-5 arm6 and gripper.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import latent_skill as ls  # noqa: E402

HORIZON = 50
SEED = 20261116
ACTION_DIM = 32
MAX_RANK = 500
T0_MIN, T0_MAX, T0_JITTER = .1, .9, .05
T0_EVAL = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
LOSS_WEIGHT = np.asarray([1.] * 6 + [.1] + [.1] * 5 + [0.] * 20, np.float32)  # arm, grip, base/mode, pad


# ------------------------------------------------------------------ episodes

class Episodes:
    """Normalized policy-order actions per episode (12 dims), cached lazily."""

    def __init__(self, out):
        self.out = Path(out)
        m = json.loads((self.out / "manifest.json").read_text())
        self.stats = ls.load_action_stats(m["checkpoint"])
        self._c = {}

    def actions(self, task, ep):
        k = (int(task), int(ep))
        if k not in self._c:
            a, _ = ls._episode(self.out, *k)
            self._c[k] = ls.normalize_policy_actions(ls.ex.action_storage_to_policy(a), self.stats)
        return self._c[k]

    def future(self, task, ep, frame, n):
        """Next n actions (12 dims) and a validity mask; tail padded with the last action."""
        a = self.actions(task, ep)
        w = a[int(frame):int(frame) + n]
        if not len(w):                      # past the episode end: hold the last action
            out = np.repeat(a[-1:], n, 0).astype(np.float32)
            return out, np.zeros(n, bool)
        out = np.repeat(w[-1:], n, 0).astype(np.float32)
        out[:len(w)] = w
        mask = np.zeros(n, bool); mask[:len(w)] = True
        return out, mask


def draft_error(draft7, target12, mask, steps=HORIZON):
    """Arm6 squared error per valid step / 7, averaged over valid steps (first ``steps``)."""
    d = np.square(draft7[:steps, :6] - target12[:steps, :6]).sum(-1) / 7.
    m = mask[:steps]
    return float(d[m].mean()) if m.any() else 0.


# ------------------------------------------------------------------- drafts

def chain_rank(eps, bank, task, ep, frame, rank):
    """Chain where every step takes the unit of the given error rank (0 = oracle)."""
    allowed = ~((bank["task"] == task) & (bank["episode"] == ep))
    cand = bank["window10"][allowed]
    ids = np.flatnonzero(allowed)
    rank = int(min(rank, len(ids) - 1))
    draft, pos = [], 0
    while pos < HORIZON:
        tgt, _ = eps.future(task, ep, frame + pos, ls.WINDOW)
        err = ls.arm6_mse_matrix(tgt[None, :, :7], cand)[0]
        k = np.argpartition(err, rank)[:rank + 1]
        j = int(ids[k[np.argsort(err[k], kind="stable")][rank]])
        L = int(bank["length"][j])
        draft.append(bank["natural_actions"][j, :L]); pos += L
    return np.concatenate(draft)[:HORIZON]


def chain_key(scorer, bank, rows, *, batch=64):
    """Key-top-1 chain for many rows at once; history is updated with the own draft only.

    Also returns the first-step top-1 minus top-2 score (a deployable confidence).
    """
    n = len(rows["task"])
    hist = rows["history"].copy(); hmask = rows["history_mask"].copy()
    drafts = [[] for _ in range(n)]; filled = np.zeros(n, np.int64)
    first = np.full(n, -1, np.int64); margin = np.zeros(n, np.float32)
    while (filled < HORIZON).any():
        todo = np.flatnonzero(filled < HORIZON)
        for s in range(0, len(todo), batch):
            ix = todo[s:s + batch]
            sc = scorer(ix, hist[ix], hmask[ix])
            same = (bank["task"][None] == rows["task"][ix, None]) & (bank["episode"][None] == rows["episode"][ix, None])
            sc[same] = -np.inf
            top2 = np.sort(np.partition(sc, -2, axis=1)[:, -2:], axis=1)
            pick = sc.argmax(1)
            for r, (i, j) in enumerate(zip(ix, pick)):
                L = int(bank["length"][j])
                u = bank["natural_actions"][j, :L]
                drafts[i].append(u); filled[i] += L
                if first[i] < 0:
                    first[i] = j; margin[i] = top2[r, 1] - top2[r, 0]
                hist[i] = np.concatenate([hist[i], u[:, :7]])[-ls.HISTORY:]; hmask[i] = True
    return np.stack([np.concatenate(d)[:HORIZON] for d in drafts]), first, margin


def make_key_scorer(out, mode, run, device):
    import torch
    import latent_key as lk
    inp = lk.Inputs(out, mode, bank_device=device)
    model = lk.new_model(mode).to(device)
    model.load_state_dict(torch.load(Path(out) / "key/runs" / run / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        keys = torch.cat([lk.encode(model, inp.batch(inp.bank_in, np.arange(s, min(s + 64, len(inp.bank_in["task"]))),
                                                     device), key=True)
                          for s in range(0, len(inp.bank_in["task"]), 64)])

    def scorer_for(rows):
        def scorer(ix, hist, hmask):
            part = {"task": rows["task"][ix], "episode": rows["episode"][ix], "frame": rows["frame"][ix],
                    "history": hist, "history_mask": hmask, "state": rows["state"][ix]}
            with torch.no_grad():
                q = lk.encode(model, inp.batch(part, np.arange(len(ix)), device), key=False)
                return lk.token_scores(q, keys).float().cpu().numpy()
        return scorer
    return scorer_for


def spread_ranks(n, seed=SEED):
    """Log-uniform ranks in [0, MAX_RANK): most drafts good, a long tail of poor ones."""
    u = np.random.default_rng(seed + 3).uniform(0, np.log(MAX_RANK + 1), int(n))
    return np.minimum(np.floor(np.exp(u)).astype(np.int64) - 1, MAX_RANK - 1)


def build_drafts(out, *, sources, device="cuda", splits=("train", "dev", "eval")):
    out = Path(out)
    bank = dict(np.load(out / "units/memory_units.npz"))
    eps = Episodes(out)
    ddir = out / "expert/drafts"; ddir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = dict(np.load(out / f"rows/{split}.npz"))
        keys = list(zip(rows["task"], rows["episode"], rows["frame"]))
        tg = [eps.future(t, e, f, HORIZON) for t, e, f in keys]
        if split == "train":
            path = ddir / "train_spread.npz"
            if not path.exists():
                ranks = spread_ranks(len(keys))
                d = np.stack([chain_rank(eps, bank, t, e, f, r) for (t, e, f), r in zip(keys, ranks)])
                err = np.asarray([draft_error(x, a, m) for x, (a, m) in zip(d, tg)], np.float32)
                ls._save_npz(path, draft=d.astype(np.float32), rank=ranks.astype(np.int32), err50=err)
                ls._log(out, "drafts", split=split, source="spread", rows=len(d),
                        err50_quantiles=np.quantile(err, [.1, .5, .9]).round(4).tolist())
            continue
        path = ddir / f"{split}_oracle.npz"
        if not path.exists():
            d = np.stack([chain_rank(eps, bank, t, e, f, 0) for t, e, f in keys])
            err = np.asarray([draft_error(x, a, m) for x, (a, m) in zip(d, tg)], np.float32)
            ls._save_npz(path, draft=d.astype(np.float32), err50=err, margin=np.zeros(len(d), np.float32))
            ls._log(out, "drafts", split=split, source="oracle", rows=len(d))
        if sources:
            sel = json.loads((out / "key/selected_configs.json").read_text())["selected"]
        for mode in sources:
            path = ddir / f"{split}_{mode}.npz"
            if path.exists():
                continue
            scorer = make_key_scorer(out, mode, sel[mode]["run"], device)(rows)
            d, first, margin = chain_key(scorer, bank, rows)
            err = np.asarray([draft_error(x, a, m) for x, (a, m) in zip(d, tg)], np.float32)
            ls._save_npz(path, draft=d.astype(np.float32), first=first.astype(np.int32), margin=margin, err50=err)
            ls._log(out, "drafts", split=split, source=mode, rows=len(d))


class T0Map:
    """t0 = T0_MIN + (T0_MAX-T0_MIN) * CDF_train(error); the same map at inference."""

    def __init__(self, train_errors):
        self.q = np.quantile(np.asarray(train_errors, np.float64), np.linspace(0, 1, 101))

    def __call__(self, err):
        cdf = np.interp(np.asarray(err, np.float64), self.q, np.linspace(0, 1, 101))
        return (T0_MIN + (T0_MAX - T0_MIN) * cdf).astype(np.float32)


def to_model_draft(draft7):
    """[T,7] normalized policy draft -> [T,32] model draft (base/mode/pad zero)."""
    d = np.zeros((len(draft7), ACTION_DIM), np.float32)
    d[:, :7] = draft7[:, :7]
    return d


# --------------------------------------------------------------------- train

def train(out, *, steps=5000, effective_batch=32, microbatch=8, lr=2.5e-5, name="key-agnostic-01"):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    from flax import serialization
    from openpi.models import model as model_api
    from openpi.shared.nnx_utils import PathRegex
    import expert as rc_expert

    out = Path(out)
    run = out / "expert/train" / name
    if (run / "summary.json").exists():
        raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True)
    rows = dict(np.load(out / "rows/train.npz"))
    sp = np.load(out / "expert/drafts/train_spread.npz")
    drafts, err = sp["draft"], sp["err50"]
    t0map = T0Map(err)
    t0_base = t0map(err)
    eps = Episodes(out)
    targets, tmask = zip(*(eps.future(t, e, f, HORIZON) for t, e, f in zip(rows["task"], rows["episode"], rows["frame"])))
    targets = np.stack(targets); tmask = np.stack(tmask)
    store = ls.TokenStore(out)
    m = json.loads((out / "manifest.json").read_text())
    model, _ = rc_expert._load_expert_model(Path(m["checkpoint"]))
    filt = nnx.All(nnx.Param, PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    state = nnx.state(model, filt)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(lr, b1=.9, b2=.95, weight_decay=1e-5))
    opt = tx.init(state)
    weight = jnp.asarray(LOSS_WEIGHT)

    @nnx.jit
    def grad_fn(model, obs, target, mask, draft, t0, key):
        kp, ke = jax.random.split(key)
        b = target.shape[0]
        progress = jax.random.uniform(kp, (b,), minval=.01, maxval=1.)
        noise = jax.random.normal(ke, target.shape)
        xt, mt, vel = rc_expert._bridge_training_sample(target, draft, noise, t0, progress)
        w = weight[None, None, :] * mask[:, :, None]

        def loss_fn(m_):
            pred = m_.velocity(obs, xt, mt)
            return jnp.sum((pred - vel) ** 2 * w) / jnp.maximum(jnp.sum(w), 1.)
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(model)

    rng = np.random.default_rng(SEED)
    order = rng.integers(0, len(rows["task"]), size=(steps, effective_batch))
    jitter = rng.uniform(-T0_JITTER, T0_JITTER, size=(steps, effective_batch)).astype(np.float32)
    ls._json(run / "config.json", {"steps": steps, "effective_batch": effective_batch, "microbatch": microbatch,
                                   "lr": lr, "t0_map_quantiles": t0map.q.tolist(), "t0_range": [T0_MIN, T0_MAX],
                                   "t0_jitter": T0_JITTER, "max_rank": MAX_RANK,
                                   "loss_weight": "arm6 1, grip .1, base/mode .1", "key_agnostic": True})
    log = (run / "train.jsonl").open("a")
    t_start = time.time()
    for step in range(steps):
        ids = order[step]; accum = None; losses = []
        for s in range(0, effective_batch, microbatch):
            sub = ids[s:s + microbatch]
            obs = model_api.Observation.from_dict(jax.tree.map(jnp.asarray, store.observation(
                rows["task"][sub], rows["episode"][sub], rows["frame"][sub])))
            tgt = np.zeros((len(sub), HORIZON, ACTION_DIM), np.float32); tgt[..., :12] = targets[sub]
            dr = np.stack([to_model_draft(d) for d in drafts[sub]])
            t0 = np.clip(t0_base[sub] + jitter[step, s:s + microbatch], .02, .98)
            loss, g = grad_fn(model, obs, jnp.asarray(tgt), jnp.asarray(tmask[sub], jnp.float32), jnp.asarray(dr),
                              jnp.asarray(t0), jax.random.fold_in(jax.random.key(SEED), step * 97 + s))
            accum = g if accum is None else jax.tree.map(lambda a, b: a + b, accum, g)
            losses.append(float(loss))
        accum = jax.tree.map(lambda x: x / (effective_batch // microbatch), accum)
        st = nnx.state(model, filt)
        upd, opt = tx.update(accum, opt, st)
        nnx.update(model, optax.apply_updates(st, upd))
        log.write(json.dumps({"step": step + 1, "loss": float(np.mean(losses))}) + "\n")
        if (step + 1) % 100 == 0:
            log.flush()
            ls._log(out, "expert-train", step=step + 1, loss=float(np.mean(losses)), sec=round(time.time() - t_start))
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            (run / f"expert_{step + 1:06d}.msgpack").write_bytes(
                serialization.to_bytes(nnx.state(model, filt).to_pure_dict()))
    ls._json(run / "summary.json", {"steps": steps, "seconds": time.time() - t_start})
    return run


# ---------------------------------------------------------------------- eval

def evaluate(out, *, expert, sources, split, batch=8, seed=SEED):
    """All conditions on one split; saves per-row first-10 arm errors for every t0."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.shared.nnx_utils import module_jit
    import expert as rc_expert

    out = Path(out)
    edir = out / "expert/eval" / split; edir.mkdir(parents=True, exist_ok=True)
    if (edir / "per_row.npz").exists():
        raise FileExistsError(edir / "per_row.npz")
    rows = dict(np.load(out / f"rows/{split}.npz"))
    n = len(rows["task"]); target = rows["target"]
    store = ls.TokenStore(out)
    m = json.loads((out / "manifest.json").read_text())
    dsrc = {"oracle": dict(np.load(out / f"expert/drafts/{split}_oracle.npz"))}
    for s in sources:
        dsrc[s] = dict(np.load(out / f"expert/drafts/{split}_{s}.npz"))
    weights = {"untrained": None, "expert": expert}
    first10 = {}   # name -> [n,10,7]
    for wname, wpath in weights.items():
        model, _ = rc_expert._load_expert_model(Path(m["checkpoint"]))
        if wpath is not None:
            rc_expert._overlay_expert(model, wpath)
        enc = module_jit(model.encode_prefix_context)
        normal = module_jit(model.sample_actions_with_context, static_argnames=("num_steps",))
        bridge = module_jit(model.sample_bridge, static_argnames=("num_steps",))
        for s in range(0, n, batch):
            sub = np.arange(s, min(s + batch, n)); k = len(sub)
            pad = np.r_[sub, np.repeat(sub[-1:], batch - k)]
            obs = model_api.Observation.from_dict(jax.tree.map(jnp.asarray, store.observation(
                rows["task"][pad], rows["episode"][pad], rows["frame"][pad])))
            ctx = enc(obs)
            noise = jax.random.normal(jax.random.key(seed + s), (batch, HORIZON, ACTION_DIM))
            if wname == "untrained":
                a = np.asarray(normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise))
                first10.setdefault("pi05", []).append(a[:k, :ls.WINDOW, :7])
            for dname, d in dsrc.items():
                dr = jnp.asarray(np.stack([to_model_draft(x) for x in d["draft"][pad]]))
                for t0 in T0_EVAL:
                    a = np.asarray(bridge(jax.random.key(0), obs, dr, jnp.full((batch,), t0), num_steps=10,
                                          noise=noise, prefix_context=ctx))
                    first10.setdefault(f"{wname}|{dname}|{t0:g}", []).append(a[:k, :ls.WINDOW, :7])
            if s % (batch * 25) == 0:
                ls._log(out, "expert-eval", split=split, weights=wname, rows_done=int(s + k), rows=n)
    first10 = {k: np.concatenate(v) for k, v in first10.items()}
    pi = first10["pi05"]
    save = {"pi05_arm10": ls.arm6_mse(pi, target), "pi05_grip10": ls.grip_mse(pi, target),
            "hold_last_arm10": ls.arm6_mse(ls.hold_last(rows), target)}
    for dname, d in dsrc.items():
        dr10 = d["draft"][:, :ls.WINDOW, :7]
        save[f"draft__{dname}__arm10"] = ls.arm6_mse(dr10, target)
        save[f"draft__{dname}__err50"] = d["err50"]
        save[f"draft__{dname}__margin"] = d["margin"]
        save[f"draft__{dname}__dist_pi05"] = ls.arm6_mse(dr10, pi)   # deployable confidence signal
        for wname in weights:
            save[f"{wname}__{dname}__arm10"] = np.stack(
                [ls.arm6_mse(first10[f"{wname}|{dname}|{t0:g}"], target) for t0 in T0_EVAL], 1)
            save[f"{wname}__{dname}__arm5"] = np.stack(
                [np.square(first10[f"{wname}|{dname}|{t0:g}"][:, :5, :6] - target[:, :5, :6]).sum((1, 2)) / 35.
                 for t0 in T0_EVAL], 1)
    np.savez(edir / "per_row.npz", t0_grid=np.asarray(T0_EVAL), task=rows["task"], episode=rows["episode"],
             hard=np.load(out / f"oracle/{split}.npz")["hard"], **save)
    np.savez_compressed(edir / "first10.npz", **{k.replace("|", "__"): v.astype(np.float16) for k, v in first10.items()})
    return edir


def _grid_index(t0):
    g = np.asarray(T0_EVAL)
    return np.abs(np.asarray(t0)[:, None] - g[None]).argmin(1)


def _fit_conf(dev, dname):
    """log err50 ~ a + b log(dist_pi05) + c margin, least squares on Dev rows."""
    y = np.log(np.maximum(dev[f"draft__{dname}__err50"], 1e-6))
    x = np.c_[np.ones_like(y), np.log(np.maximum(dev[f"draft__{dname}__dist_pi05"], 1e-6)), dev[f"draft__{dname}__margin"]]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    return coef, float(np.corrcoef(pred, y)[0, 1])


def _predict_err(split, dname, coef):
    x = np.c_[np.ones(len(split[f"draft__{dname}__err50"])),
              np.log(np.maximum(split[f"draft__{dname}__dist_pi05"], 1e-6)), split[f"draft__{dname}__margin"]]
    return np.exp(x @ coef)


def report(out, *, expert_run="key-agnostic-01"):
    out = Path(out)
    dev = dict(np.load(out / "expert/eval/dev/per_row.npz"))
    ev = dict(np.load(out / "expert/eval/eval/per_row.npz"))
    cfg = json.loads((out / "expert/train" / expert_run / "config.json").read_text())
    t0map = T0Map([0, 1]); t0map.q = np.asarray(cfg["t0_map_quantiles"])
    dnames = sorted({k.split("__")[1] for k in ev if k.startswith("draft__")})
    rows, per_row = [], {}

    def add(name, v_ev, policy=""):
        per_row[name] = v_ev
        rows.append({"condition": name, "policy": policy,
                     "eval": ls.episode_bootstrap(v_ev, ev["task"], ev["episode"]),
                     "eval_hard": float(v_ev[ev["hard"]].mean())})

    add("π0.5 normal", ev["pi05_arm10"]); add("hold-last", ev["hold_last_arm10"])
    conf_fit = {}
    for dn in dnames:
        add(f"draft only [{dn}]", ev[f"draft__{dn}__arm10"])
        coef, r = _fit_conf(dev, dn); conf_fit[dn] = {"coef": coef.tolist(), "dev_corr_log": r}
        for w in ("untrained", "expert"):
            kd, ke = dev[f"{w}__{dn}__arm10"], ev[f"{w}__{dn}__arm10"]
            j = int(kd.mean(0).argmin())
            add(f"{w} [{dn}] fixed", ke[:, j], f"t0={T0_EVAL[j]:g} (Dev)")
            add(f"{w} [{dn}] conf", ke[np.arange(len(ke)), _grid_index(t0map(_predict_err(ev, dn, coef)))],
                "t0=map(pred err)")
            add(f"{w} [{dn}] true-err", ke[np.arange(len(ke)), _grid_index(t0map(ev[f"draft__{dn}__err50"]))],
                "upper bound")
            add(f"{w} [{dn}] row-best", ke.min(1), "upper bound")
    paired = {}
    for name, v in per_row.items():
        d = v - per_row["π0.5 normal"]
        paired[name] = ls.episode_bootstrap(d, ev["task"], ev["episode"])
    for r in rows:
        r["minus_pi05"] = paired[r["condition"]]
    ls._json(out / "expert/report.json", {"rows": rows, "confidence_fit": conf_fit, "t0_grid": list(T0_EVAL)})
    lines = ["| condition | t0 | Eval arm6 (10) [95% CI] | hard | − π0.5 [95% CI] |", "|---|---|---:|---:|---:|"]
    for r in rows:
        e, d = r["eval"], r["minus_pi05"]
        lines.append(f"| {r['condition']} | {r['policy']} | {e['mean']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}] | "
                     f"{r['eval_hard']:.4f} | {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |")
    (out / "expert/report.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("drafts", "train", "eval", "report"))
    p.add_argument("--out", type=Path, default=ls.OUT_DEFAULT)
    p.add_argument("--sources", nargs="*", default=["e_vlm_h", "p5"])
    p.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--effective-batch", type=int, default=32)
    p.add_argument("--expert", type=Path)
    p.add_argument("--split", choices=("dev", "eval"), default="dev")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ls._guard_output(a.out)
    if a.stage == "drafts":
        build_drafts(a.out, sources=a.sources, device=a.device, splits=a.splits)
    elif a.stage == "train":
        train(a.out, steps=a.steps, microbatch=a.microbatch, effective_batch=a.effective_batch)
    elif a.stage == "eval":
        evaluate(a.out, expert=a.expert, sources=a.sources, split=a.split)
    else:
        report(a.out)
        print((a.out / "expert/report.md").read_text())


if __name__ == "__main__":
    main()
