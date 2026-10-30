#!/usr/bin/env python3
"""Train the small retrieval projection and save train/heldout retrieval-loss curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (str(HERE), str(HERE.parents[1])):
    if path not in sys.path:
        sys.path.insert(0, path)

from key_check import DEFAULT_HELDOUT, DEFAULT_ROWS, _read_npz
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from memory import load_memory
from train_retrieval_key import _behavior_neighbors, _normalize, _sample_task_queries

MEMORY_PATH = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/memory/initial_libero90.pkl')
OUT = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/key-projection-20261029T174550Z/training_validation.json'))
SEED_ALL, SEED_TASK = 20261030, 20261031
STEPS, BATCH = 600, 256
LOG_STEPS = {1, 100, 200, 300, 400, 500, 600}


def _positive_ids(query_bases, query_eps, query_init, memory_bases, memory_eps, memory_init, k=8):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(memory_bases, np.float32))
    _, nearest = tree.query(np.asarray(query_bases, np.float32),
                            k=min(48, len(memory_bases)), workers=8)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    out = np.empty((len(query_bases), k), np.int64)
    for i, b in enumerate(query_bases):
        chosen = [int(j) for j in nearest[i]
                  if memory_eps[int(j)] != query_eps[i] and memory_init[int(j)] != query_init[i]]
        if not chosen:
            raise ValueError("validation query has no eligible memory positive")
        chosen = (chosen * ((k + len(chosen) - 1) // len(chosen)))[:k]
        out[i] = chosen
    return out


def _retrieval_loss(q, mem, positives, q_eps, q_init, mem_eps, mem_init):
    logits = (q @ mem.T).astype(np.float64) / 0.07
    losses = []
    for i, row in enumerate(logits):
        eligible = (mem_eps != q_eps[i]) & (mem_init != q_init[i])
        den_values = row[eligible]
        den_max = den_values.max()
        log_den = den_max + np.log(np.exp(den_values - den_max).sum())
        pos = row[positives[i]]
        pos_max = pos.max()
        log_pos = pos_max + np.log(np.exp(pos - pos_max).sum())
        losses.append(log_den - log_pos)
    return float(np.mean(losses))


def _fit_curve(name, keys, states, bases, episodes, initials, active_ids,
               val_keys, val_states, val_bases, val_eps, val_init, *, seed):
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lk, ls = keys[active_ids], states[active_ids]
    lb, le, li = bases[active_ids], episodes[active_ids], initials[active_ids]
    key_mean = lk.mean(axis=0, keepdims=True)
    train_np = np.concatenate([_normalize(lk - key_mean), _normalize(ls)], axis=1).astype(np.float32)
    val_np = np.concatenate([_normalize(val_keys - key_mean), _normalize(val_states)], axis=1).astype(np.float32)

    class Projector(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 128))
        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    model = Projector(train_np.shape[1]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = torch.from_numpy(train_np)
    pos = _behavior_neighbors(lb, le, li, k=8, workers=8)
    val_pos = _positive_ids(val_bases, val_eps, val_init, lb, le, li)
    probe_rng = np.random.default_rng(seed + 9000)
    probe = np.sort(probe_rng.choice(len(active_ids), min(256, len(active_ids)), replace=False))
    curve = []
    for step in range(1, STEPS + 1):
        model.train()
        ids = rng.integers(0, len(active_ids), size=BATCH)
        pos_ids = np.asarray([rng.choice(pos[i]) for i in ids], np.int64)
        anchor = model(train_x[torch.from_numpy(ids)])
        candidate = model(train_x[torch.from_numpy(pos_ids)])
        logits = (anchor @ candidate.T) / 0.07
        mask_np = np.any(pos[ids, :, None] == pos_ids[None, None, :], axis=1)
        mask = torch.from_numpy(mask_np)
        mask[torch.arange(BATCH), torch.arange(BATCH)] = True
        loss = (torch.logsumexp(logits, 1)
                - torch.logsumexp(logits.masked_fill(~mask, -1e9), 1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in LOG_STEPS:
            model.eval()
            with torch.no_grad():
                mem_out = model(train_x).numpy().astype(np.float32)
                val_out = model(torch.from_numpy(val_np)).numpy().astype(np.float32)
            row = {
                "step": step,
                "minibatch_infonce": float(loss.detach()),
                "train_retrieval_infonce": _retrieval_loss(mem_out[probe], mem_out, pos[probe],
                    le[probe], li[probe], le, li),
                "validation_retrieval_infonce": _retrieval_loss(val_out, mem_out, val_pos,
                    val_eps, val_init, le, li),
            }
            curve.append(row)
            print(name, row, flush=True)
    return {"steps": STEPS, "batch_size": BATCH, "seed": seed, "active_memory_rows": len(active_ids),
            "training_probe_rows": len(probe), "validation_rows": len(val_eps),
            "validation_episodes": len(set(val_eps)), "curve": curve}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", default=MEMORY_PATH)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--query-keys", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    memory = load_memory(args.memory)
    held = _read_npz(args.heldout, ("actions", "state", "episode_id", "init_state_id", "source_file"))
    q_keys = np.load(args.query_keys or Path(args.heldout).with_name("query-keys.npy")).astype(np.float32)
    with np.load(args.rows, allow_pickle=True) as z:
        states = np.asarray(z["state"], np.float32)
        episodes = np.asarray(z["episode_id"]).astype(str)
        initials = np.asarray(z["init_state_id"]).astype(str)
        sources = np.asarray(z["source_file"]).astype(str)
    keys = _normalize(np.stack([np.asarray(x["key"], np.float32).reshape(-1) for x in memory.memory]))
    bases = np.stack([np.asarray(x["base"], np.float32).reshape(-1) for x in memory.memory])
    if len(keys) != len(states):
        raise ValueError("memory and row cache are not aligned")
    _, heldout_tasks, _ = _sample_task_queries(sources, episodes, 20261030,
                                               episodes_per_task=2, rows_per_episode=8)
    task_ids = np.flatnonzero(~np.isin(sources, heldout_tasks))
    val_bases = np.asarray(memory.encode(held["actions"], held["state"])[0], np.float32)
    val_eps = np.asarray(held["episode_id"]).astype(str)
    val_init = np.asarray(held["init_state_id"]).astype(str)
    val_sources = np.asarray(held["source_file"]).astype(str)
    known = ~np.isin(val_sources, heldout_tasks)
    curves = {
        "all90": _fit_curve("all90", keys, states, bases, episodes, initials, np.arange(len(keys)),
            q_keys, np.asarray(held["state"], np.float32), val_bases, val_eps, val_init, seed=SEED_ALL),
        "task_memory": _fit_curve("task_memory", keys, states, bases, episodes, initials, task_ids,
            q_keys[known], np.asarray(held["state"], np.float32)[known], val_bases[known],
            val_eps[known], val_init[known], seed=SEED_TASK),
    }
    result = {
        "validation_split": "heldout-observations.npz, episode-disjoint from memory; task-memory validation additionally excludes the 10 task-heldout files",
        "validation_task_count": int(len(set(val_sources[known]))),
        "heldout_task_files": heldout_tasks,
        "positive_rule": "nearest 8 cached M1 bases excluding same episode/init",
        "retrieval_loss": "all eligible active-memory projected keys in denominator; nearest-8 M1 positives in numerator; temperature .07",
        "elapsed_seconds": time.perf_counter() - started,
        "curves": curves,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
