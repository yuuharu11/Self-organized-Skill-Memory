#!/usr/bin/env python3
"""R2: shared latent-token key E for the RoboCasa DCT-3 unit memory (KEY2 port).

The encoder, objective and candidate rule follow LIBERO KEY2-F
(``unseen_libero/key_shared_latent.py`` on ``exp/key2-expert-plan``); only the
input sizes change for RoboCasa (three cameras + text/state prompt tokens,
16-d state, 7-d policy-order history).

  E(o) = 5 learned latent tokens -> cross-attention over
         [VLM prefix tokens | 10-step executed history + mask | current state]
         -> self-attention -> L2-normalised [5, 256]
  score(q, k) = mean_k <E(q)_k, E(k)_k> * exp(logit_scale)   (token-wise product)

One E encodes both the query observation and each memory unit's start
observation (shared). Training queries are memory-side unit starts; the
teacher is softmax(-arm6 error / tau) of the query's next 10 actions against
each candidate's stored 10-step window. Candidates per step are the other unit
keys in the 8-query minibatch plus 256 uniform memory units; labels shape the
teacher only. Same-episode candidates are masked. History is corrupted
(25% zero / 25% other row / 15% xyz reversed / 35% true) as in KEY2.

Baselines: P0 (pooled VLM -> MLP) and P5 (pooled VLM + history + state -> MLP),
with an independently encoded key, as in KEY2.

Settings are selected on Dev by top-1 arm6 MSE; Eval is scored once.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import latent_skill as ls  # noqa: E402

LOGIT_SCALE_INITIAL = 1.0 / .07
LOGIT_SCALE_MAX = 100.0
MODES = ("e_vlm", "e_vlm_h", "e_h", "p0", "p5")
INPUTS = {"e_vlm": (True, False, False), "e_vlm_h": (True, True, True), "e_h": (False, True, True)}
STATE_DIM = 16
HIST_DIM = 7
TOKEN_COUNT = 5
WIDTH = 256
GRID = {"lr": (1e-4, 3e-4), "epochs": (4, 8, 16), "teacher_tau": (.002, .005, .01, .02)}
RANK_TAU = .01


# ------------------------------------------------------------------- models

class _Cross(nn.Module):
    def __init__(self, dim, heads=8, ff=4):
        super().__init__()
        self.qn, self.kn, self.fn = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff), nn.GELU(), nn.Linear(dim * ff, dim))

    def forward(self, q, mem, pad):
        kv = self.kn(mem)
        q = q + self.attn(self.qn(q), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ff(self.fn(q))


class SharedLatentEncoder(nn.Module):
    def __init__(self, mode="e_vlm_h", token_dim=2048, width=WIDTH, tokens=TOKEN_COUNT, groups=4,
                 history=ls.HISTORY, hist_dim=HIST_DIM, state_dim=STATE_DIM, heads=8):
        super().__init__()
        self.use_vlm, self.use_hist, self.use_state = INPUTS[mode]
        self.vlm = nn.Linear(token_dim, width)
        self.group = nn.Embedding(groups, width)
        self.hist = nn.Linear(hist_dim + 1, width)
        self.hist_pos = nn.Parameter(torch.zeros(1, history, width))
        self.state = nn.Sequential(nn.Linear(state_dim, width), nn.GELU(), nn.Linear(width, width))
        self.state_type = nn.Parameter(torch.zeros(1, 1, width))
        self.latent = nn.Parameter(torch.randn(1, tokens, width) * .02)
        self.cross = _Cross(width, heads)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_norm, self.out_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(LOGIT_SCALE_INITIAL))))

    def forward(self, b):
        mem, pad = [], []
        n = len(b["state"])
        if self.use_vlm:
            g = b["groups"]
            mem.append(self.vlm(b["tokens"].float()) + self.group(g.clamp(min=0)))
            pad.append(g < 0)
        if self.use_hist:
            h = torch.cat([b["history"], b["history_mask"].float()[..., None]], -1)
            mem.append(self.hist(h) + self.hist_pos); pad.append(~b["history_mask"])
        if self.use_state:
            mem.append(self.state(b["state"])[:, None] + self.state_type)
            pad.append(torch.zeros(n, 1, dtype=torch.bool, device=b["state"].device))
        mem, pad = torch.cat(mem, 1), torch.cat(pad, 1)
        q = self.cross(self.latent.expand(n, -1, -1), mem, pad)
        x = self.self_norm(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        return F.normalize(self.out_norm(q), dim=-1)


class VectorRanker(nn.Module):
    """P0/P5: pooled-VLM query MLP and independently encoded key MLP."""

    def __init__(self, use_history, token_dim=2048, width=WIDTH, tokens=TOKEN_COUNT):
        super().__init__()
        self.use_history = bool(use_history)
        extra = ls.HISTORY * HIST_DIM + ls.HISTORY + STATE_DIM if use_history else 0
        self.q = nn.Sequential(nn.Linear(token_dim + extra, 512), nn.GELU(), nn.Linear(512, width))
        self.k = nn.Sequential(nn.Linear(token_dim + extra, 512), nn.GELU(), nn.Linear(512, width * tokens))
        self.tokens, self.width = tokens, width
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(LOGIT_SCALE_INITIAL))))

    def _x(self, b):
        x = [b["pooled"]]
        if self.use_history:
            x += [b["history"].flatten(1), b["history_mask"].float(), b["state"]]
        return torch.cat(x, -1)

    def query(self, b):
        return F.normalize(self.q(self._x(b)), dim=-1)[:, None].expand(-1, self.tokens, -1)

    def key(self, b):
        return F.normalize(self.k(self._x(b)).reshape(-1, self.tokens, self.width), dim=-1)


def new_model(mode):
    if mode in INPUTS:
        return SharedLatentEncoder(mode)
    if mode in ("p0", "p5"):
        return VectorRanker(use_history=(mode == "p5"))
    raise ValueError(mode)


def encode(model, b, *, key):
    if isinstance(model, SharedLatentEncoder):
        return model(b)
    return model.key(b) if key else model.query(b)


def token_scores(q, k):
    return torch.einsum("bkd,nkd->bnk", q, k).mean(-1)


def scaled(model, s):
    return s * model.logit_scale.exp().clamp(max=LOGIT_SCALE_MAX)


def soft_listwise_loss(scores, errors, allowed, tau):
    valid = allowed.any(-1)
    target = torch.softmax((-errors.float() / float(tau)).masked_fill(~allowed, -1e9), -1)
    logp = torch.log_softmax(scores.float().masked_fill(~allowed, -1e9), -1)
    return -(target * logp).sum(-1)[valid].mean() if valid.any() else scores.sum() * 0


def history_corrupt(h, mask, rng):
    """KEY2 schedule: 25% zero, 25% other row, 15% reversed xyz, 35% true."""
    h = np.asarray(h, np.float32).copy(); mask = np.asarray(mask, bool).copy(); n = len(h)
    modes = rng.choice(4, size=n, p=[.25, .25, .15, .35])
    z = modes == 0; h[z] = 0; mask[z] = False
    for i in np.flatnonzero(modes == 1):
        if n > 1:
            j = int(rng.integers(n - 1)); j += int(j >= i); h[i] = h[j]; mask[i] = mask[j]
    h[modes == 2, :, :3] *= -1
    return h, mask


# ----------------------------------------------------------------------- data

class Inputs:
    """Causal inputs for bank keys (unit starts) and query rows; never targets."""

    def __init__(self, out, mode, bank_device=None):
        out = Path(out)
        self.mode = mode
        self.store = ls.TokenStore(out) if mode != "e_h" else None
        self.bank = dict(np.load(out / "units/memory_units.npz"))
        m = json.loads((out / "manifest.json").read_text())
        stats = ls.load_action_stats(m["checkpoint"])
        # Bank key inputs: history before the unit start and the state at the start.
        hist, hmask, state = [], [], []
        cache = {}
        for t, e, s in zip(self.bank["task"], self.bank["episode"], self.bank["start"]):
            if (t, e) not in cache:
                a, st = ls._episode(out, t, e)
                cache.clear()
                cache[(t, e)] = (ls.normalize_policy_actions(ls.ex.action_storage_to_policy(a), stats),
                                 ls.ex.state_storage_to_policy(st).astype(np.float32))
            pol, sp = cache[(t, e)]
            h = np.zeros((ls.HISTORY, HIST_DIM), np.float32); mk = np.zeros(ls.HISTORY, bool)
            past = pol[max(0, s - ls.HISTORY):s, :HIST_DIM]
            if len(past):
                h[-len(past):] = past; mk[-len(past):] = True
            hist.append(h); hmask.append(mk); state.append(sp[s])
        self.bank_in = {"task": self.bank["task"], "episode": self.bank["episode"], "frame": self.bank["start"],
                        "history": np.stack(hist), "history_mask": np.stack(hmask), "state": np.stack(state)}
        # Training queries are the unit starts themselves (KEY2); targets are the next 10 actions.
        self.train_target = self._future(out, stats)
        self.state_mean = self.bank_in["state"].mean(0)
        self.state_std = self.bank_in["state"].std(0) + 1e-3
        self.rows = {s: dict(np.load(out / f"rows/{s}.npz")) for s in ("dev", "eval")}
        self.cand10 = self.bank["window10"]
        self._pooled_cache = {}
        # Bank-start tokens are read every step; hold them in RAM once (~40 GB fp16)
        # instead of copying scattered memmap rows (that copy dominated step time).
        self.bank_tokens = self.bank_groups = None
        if self.store is not None and mode in INPUTS and bank_device is not None:
            tok, grp = self.store.get(self.bank_in["task"], self.bank_in["episode"], self.bank_in["frame"])
            self.bank_tokens = torch.as_tensor(tok, device=bank_device)
            self.bank_groups = torch.as_tensor(grp.astype(np.int64), device=bank_device)
            del tok, grp

    def _future(self, out, stats):
        tg = np.zeros((len(self.bank["start"]), ls.WINDOW, 7), np.float32)
        cache = {}
        for i, (t, e, s) in enumerate(zip(self.bank["task"], self.bank["episode"], self.bank["start"])):
            if (t, e) not in cache:
                a, _ = ls._episode(out, t, e); cache.clear()
                cache[(t, e)] = ls.normalize_policy_actions(ls.ex.action_storage_to_policy(a), stats)
            pol = cache[(t, e)]
            w = pol[s:s + ls.WINDOW, :7]
            tg[i, :len(w)] = w
            if len(w) < ls.WINDOW:
                tg[i, len(w):] = w[-1]
        return tg

    def batch(self, part, idx, device, *, corrupt_rng=None):
        idx = np.asarray(idx, np.int64)
        h, mk = part["history"][idx], part["history_mask"][idx]
        if corrupt_rng is not None:
            h, mk = history_corrupt(h, mk, corrupt_rng)
        b = {"history": torch.as_tensor(h, device=device),
             "history_mask": torch.as_tensor(mk, device=device),
             "state": torch.as_tensor((part["state"][idx] - self.state_mean) / self.state_std, device=device)}
        if self.store is not None:
            if self.mode in ("p0", "p5"):
                b["pooled"] = torch.as_tensor(self.store.pooled(part["task"][idx], part["episode"][idx],
                                                                part["frame"][idx]), device=device)
            else:
                if part is self.bank_in and self.bank_tokens is not None:
                    ti = torch.as_tensor(idx, device=self.bank_tokens.device)
                    b["tokens"] = self.bank_tokens[ti].to(device)
                    b["groups"] = self.bank_groups[ti].to(device)
                else:
                    tok, grp = self.store.get(part["task"][idx], part["episode"][idx], part["frame"][idx])
                    b["tokens"] = torch.as_tensor(tok, device=device)
                    b["groups"] = torch.as_tensor(grp.astype(np.int64), device=device)
        return b


# ---------------------------------------------------------------------- train

def train(inp, *, mode, lr, epochs, teacher_tau, seed, device, batch=8, random_candidates=256, log=None,
          on_epoch=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 101)
    model = new_model(mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bank = inp.bank_in
    n = len(bank["task"])
    cand10 = torch.as_tensor(inp.cand10, device=device)
    tgt = torch.as_tensor(inp.train_target, device=device)
    task_t = torch.as_tensor(bank["task"].astype(np.int64), device=device)
    ep_t = torch.as_tensor(bank["episode"].astype(np.int64), device=device)
    history = []
    for epoch in range(int(epochs)):
        model.train(); order = rng.permutation(n); losses = []
        for s in range(0, n, batch):
            qi = order[s:s + batch]
            rnd = rng.choice(n, size=min(random_candidates, n), replace=False)
            cids = np.unique(np.r_[qi, rnd])
            q = encode(model, inp.batch(bank, qi, device, corrupt_rng=rng), key=False)
            k = encode(model, inp.batch(bank, cids, device), key=True)
            score = scaled(model, token_scores(q, k))
            ct = torch.as_tensor(cids, device=device)
            err = ((tgt[qi][:, None, :, :6] - cand10[ct][None, :, :, :6]) ** 2).sum((-1, -2)) / (ls.WINDOW * 7.)
            qit = torch.as_tensor(qi, device=device)
            allowed = ~((task_t[qit][:, None] == task_t[ct][None]) & (ep_t[qit][:, None] == ep_t[ct][None]))
            loss = soft_listwise_loss(score, err, allowed, teacher_tau)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=float(np.log(LOGIT_SCALE_MAX)))
            losses.append(float(loss.detach()))
        rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "steps": len(losses)}
        history.append(rec)
        if log:
            log(rec)
        if on_epoch is not None:
            on_epoch(epoch + 1, model)
            model.train()
    return model, history


@torch.no_grad()
def score_split(model, inp, split, device, chunk=64):
    model.eval()
    bank = inp.bank_in
    keys = torch.cat([encode(model, inp.batch(bank, np.arange(s, min(s + chunk, len(bank["task"]))), device), key=True)
                      for s in range(0, len(bank["task"]), chunk)])
    rows = inp.rows[split]
    out = []
    for s in range(0, len(rows["task"]), chunk):
        q = encode(model, inp.batch(rows, np.arange(s, min(s + chunk, len(rows["task"]))), device), key=False)
        out.append(token_scores(q, keys).float().cpu().numpy())
    return np.concatenate(out)


def metrics(scores, inp, split, hard):
    from scipy.stats import spearmanr
    rows = inp.rows[split]
    err = ls.arm6_mse_matrix(rows["target"], inp.cand10)
    top = scores.argmax(1)
    top_err = err[np.arange(len(err)), top]
    best = err.min(1); rnd = err.mean(1)
    rho = [spearmanr(scores[i], -err[i]).statistic for i in range(len(err))]
    best_ix = err.argmin(1)
    order = np.argsort(-scores, 1)
    rank = np.argmax(order == best_ix[:, None], 1) + 1
    return {"top1_arm6": ls.episode_bootstrap(top_err, rows["task"], rows["episode"]),
            "top1_arm6_hard": float(top_err[hard].mean()),
            "recovery": float(np.mean((rnd - top_err) / np.maximum(rnd - best, 1e-12))),
            "spearman_positive": float(np.nanmean(rho)), "recall@1": float(np.mean(top == best_ix)),
            "mrr": float(np.mean(1. / rank)), "top1_grip": float(ls.grip_mse(inp.cand10[top], rows["target"]).mean()),
            "per_task_top1_arm6": {t: float(top_err[rows["task"] == i].mean())
                                   for i, t in enumerate(ls.TASKS) if np.any(rows["task"] == i)}}, top


def config_name(mode, lr, epochs, teacher_tau, seed):
    return f"{mode}_lr{lr:g}_ep{epochs}_tau{teacher_tau:g}_s{seed}"


def run_one(out, *, mode, lr, epochs, teacher_tau, seed, device, eval_epochs=None):
    """Train once to ``epochs``; snapshot and score Dev at every epoch in ``eval_epochs``.

    Each snapshot is its own run directory (``..._ep{k}_...``), identical to a
    separate run stopped at epoch k (same seed, same data order up to k).
    """
    out = Path(out)
    eval_epochs = sorted(set(eval_epochs or [epochs]) | {epochs})
    dirs = {k: out / "key/runs" / config_name(mode, lr, k, teacher_tau, seed) for k in eval_epochs}
    if all((d / "dev.json").exists() for d in dirs.values()):
        return json.loads((dirs[epochs] / "dev.json").read_text())
    t0 = time.time()
    inp = Inputs(out, mode, bank_device=device)
    hard = np.load(out / "oracle/dev.npz")["hard"]
    hist_all = []
    results = {}

    def log(rec):
        hist_all.append(rec)

    def on_epoch(k, model):
        if k not in dirs:
            return
        d = dirs[k]; d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "model.pt")
        dev, dev_top = metrics(score_split(model, inp, "dev", device), inp, "dev", hard)
        results[k] = {"config": {"mode": mode, "lr": lr, "epochs": k, "teacher_tau": teacher_tau, "seed": seed},
                      "dev": dev, "train": list(hist_all), "seconds": time.time() - t0}
        np.save(d / "dev_top1.npy", dev_top.astype(np.int32))
        ls._json(d / "dev.json", results[k])
        print(json.dumps({"run": d.name, "dev_top1_arm6": dev["top1_arm6"]["mean"]}), flush=True)

    train(inp, mode=mode, lr=lr, epochs=epochs, teacher_tau=teacher_tau, seed=seed, device=device,
          log=log, on_epoch=on_epoch)
    return results[epochs]


# ------------------------------------------------------------- grid / select / eval

def grid_configs(modes, seed=1):
    """One training run per (mode, lr, tau) to max(epochs); smaller epochs are snapshots."""
    return [dict(mode=m, lr=lr, epochs=max(GRID["epochs"]), teacher_tau=tau, seed=seed)
            for m in modes for lr, tau in itertools.product(GRID["lr"], GRID["teacher_tau"])]


def launch_grid(out, modes, gpus, per_gpu=2):
    """Dispatch grid configs as subprocesses, ``per_gpu`` at a time on each GPU."""
    cfgs = [c for c in grid_configs(modes)
            if not all((Path(out) / "key/runs" / config_name(**{**c, "epochs": k}) / "dev.json").exists()
                       for k in GRID["epochs"])]
    # Cheap pooled baselines and the token-free E-H first, so early results arrive soon.
    cfgs.sort(key=lambda c: {"p0": 0, "p5": 0, "e_h": 1}.get(c["mode"], 2))
    if any(c["mode"] in ("p0", "p5") for c in cfgs):
        s = ls.TokenStore(out); s.pooled([s.keys0[0]], [s.keys0[1]], [s.keys0[2]])
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    logdir = Path(out) / "key/logs"; logdir.mkdir(parents=True, exist_ok=True)
    while cfgs or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not cfgs:
                break
            c = cfgs.pop(0)
            cmd = [sys.executable, __file__, "run", "--out", str(out), "--mode", c["mode"], "--lr", str(c["lr"]),
                   "--epochs", str(c["epochs"]), "--teacher-tau", str(c["teacher_tau"]), "--seed", str(c["seed"])]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slots[slot])}
            f = (logdir / f"{config_name(**c)}.log").open("w")
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), f, c)
        time.sleep(10)
        for slot, (p, f, c) in list(running.items()):
            if p.poll() is not None:
                f.close(); del running[slot]
                print(json.dumps({"finished": config_name(**c), "returncode": p.returncode}), flush=True)


def select(out, modes):
    out = Path(out)
    chosen = {}
    for m in modes:
        best = None
        for d in sorted((out / "key/runs").glob(f"{m}_lr*")):
            if not (d / "dev.json").exists():
                continue
            r = json.loads((d / "dev.json").read_text())
            v = r["dev"]["top1_arm6"]["mean"]
            if best is None or v < best[0]:
                best = (v, d.name, r["config"])
        if best:
            chosen[m] = {"run": best[1], "dev_top1_arm6": best[0], "config": best[2]}
    ls._json(out / "key/selected_configs.json", {"rule": "lowest Dev top-1 arm6 MSE", "selected": chosen})
    return chosen


def evaluate(out, device):
    out = Path(out)
    sel = json.loads((out / "key/selected_configs.json").read_text())["selected"]
    if (out / "key/eval/results.json").exists():
        raise FileExistsError("Eval has already been scored once")
    hard = {s: np.load(out / f"oracle/{s}.npz")["hard"] for s in ("dev", "eval")}
    ev = out / "key/eval"
    ev.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode, s in sel.items():
        inp = Inputs(out, mode, bank_device=device)
        model = new_model(mode).to(device)
        model.load_state_dict(torch.load(out / "key/runs" / s["run"] / "model.pt", map_location=device))
        for split in ("dev", "eval"):
            sc = score_split(model, inp, split, device)
            r, top = metrics(sc, inp, split, hard[split])
            res.setdefault(mode, {})[split] = r
            np.save(ev / f"{mode}_{split}_top1.npy", top.astype(np.int32))
            np.save(ev / f"{mode}_{split}_scores_fp16.npy", sc.astype(np.float16))
    res["references"] = json.loads((out / "oracle/results.json").read_text())
    ls._json(out / "key/eval/results.json", res)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("run", "grid", "select", "eval"))
    p.add_argument("--out", type=Path, default=ls.OUT_DEFAULT)
    p.add_argument("--mode", choices=MODES, default="e_vlm_h")
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--teacher-tau", type=float, default=.005)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ls._guard_output(a.out)
    if a.stage == "run":
        r = run_one(a.out, mode=a.mode, lr=a.lr, epochs=a.epochs, teacher_tau=a.teacher_tau, seed=a.seed,
                    device=a.device, eval_epochs=[k for k in GRID["epochs"] if k <= a.epochs])
        print(json.dumps({"dev_top1_arm6": r["dev"]["top1_arm6"]}))
    elif a.stage == "grid":
        launch_grid(a.out, a.modes, a.gpus, a.per_gpu)
    elif a.stage == "select":
        print(json.dumps(select(a.out, a.modes), indent=2))
    else:
        print(json.dumps(evaluate(a.out, a.device), indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
