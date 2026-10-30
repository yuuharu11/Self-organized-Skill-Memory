#!/usr/bin/env python3
"""Re-evaluate retrieval unit hits under the M6 kappa/epsilon sweep.

Uses the frozen M1 bases/decoder already stored in the LIBERO-90 memory. It
does not retrain the policy, M1 codec, or learned retrieval projection.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
for path in (str(HERE), str(HERE.parents[1])):
    if path not in sys.path:
        sys.path.insert(0, path)

from key_check import DEFAULT_HELDOUT, DEFAULT_ROWS, _read_npz
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from memory import load_memory
from train_retrieval_key import _macro_recall, _normalize, _sample_task_queries, _topk_cosine

KAPPAS = (0.05, 0.1, 0.2)
SEED = 20261030
MEMORY_PATH = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/memory/initial_libero90.pkl')
QUERY_KEYS_PATH = str(Path(DEFAULT_HELDOUT).with_name("query-keys.npy"))


def _recluster(template, bases, episodes, *, kappa, rng_seed):
    """Replay M6 online unit formation from cached bases at one epsilon."""
    model = copy.deepcopy(template)
    n = len(bases)
    model._centers = []
    model.support_counts = []
    model.episode_sets = []
    model.pending = []
    model.pending_age = {}
    model._pending_since_cluster = 0
    model._next_pending_id = 0
    model._last_pending_id = None
    model._data_seen = 0
    model.registered_count = model.merged_count = model.pending_expired_count = 0
    model.unit_count_history = []
    model.unit_registration_events = []
    model._fit_in_progress = True
    model.kappa = float(kappa)

    rng = np.random.default_rng(rng_seed + 6046)
    rng.choice(n, size=model.PROBE_SIZE, replace=n < model.PROBE_SIZE)
    pair_indices = rng.integers(0, n, size=(model.RANDOM_DISTANCE_PAIRS, 2))
    distances = [model._functional_distance(bases[i], bases[j]) for i, j in pair_indices]
    model.functional_median = float(np.median(distances))
    model.epsilon = max(model.kappa * model.functional_median, 1e-12)
    order = rng.permutation(n)
    for i in order:
        model._data_seen += 1
        model._assign(bases[i], str(episodes[i]), training=True)
        model._cluster_pending()
    model._cluster_pending(force=True)
    model._fit_in_progress = False

    labels = np.full(n, -1, np.int64)
    for i, base in enumerate(bases):
        # Seed memory rows retain their nearest unit id even when their
        # insertion audit says "seed_unassigned" (M6.memory_add semantics).
        unit_id, _ = model._nearest(base)
        if unit_id is not None:
            labels[i] = int(unit_id)
    return model, labels


def _score(retrieval_ids, candidate_units, target_units):
    majority = int(np.bincount(candidate_units[candidate_units >= 0]).argmax())
    result = _macro_recall(target_units, candidate_units[retrieval_ids], majority)
    result["majority_macro_recall"] = 0.0 if len(set(target_units[target_units >= 0])) > 1 else float(np.mean(target_units == majority))
    result["majority_hit1"] = float(np.mean(target_units == majority))
    result["candidate_units"] = int(len(set(candidate_units[candidate_units >= 0])))
    result["query_units"] = int(len(set(target_units[target_units >= 0])))
    return result


def main(output_dir: str, *, memory_path=MEMORY_PATH, heldout_path=DEFAULT_HELDOUT,
         rows_path=DEFAULT_ROWS, query_keys_path=QUERY_KEYS_PATH):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "epsilon_sweep_v2.json"
    if path.exists():
        raise FileExistsError(path)

    memory = load_memory(memory_path)
    held = _read_npz(heldout_path, ("actions", "state", "episode_id", "init_state_id", "partition"))
    q_keys = np.load(query_keys_path).astype(np.float32)
    with np.load(rows_path, allow_pickle=True) as z:
        rows = {k: z[k] for k in ("state", "episode_id", "init_state_id", "source_file")}
    bases = np.stack([np.asarray(item["base"], np.float32).reshape(-1) for item in memory.memory])
    keys = _normalize(np.stack([np.asarray(item["key"], np.float32).reshape(-1) for item in memory.memory]))
    episodes = np.asarray(rows["episode_id"]).astype(str)
    initials = np.asarray(rows["init_state_id"]).astype(str)
    source_files = np.asarray(rows["source_file"]).astype(str)
    if len(memory.memory) != len(bases) or len(bases) != len(episodes):
        raise ValueError("memory/cache rows are not aligned")

    task_qids, heldout_tasks, task_plan = _sample_task_queries(source_files, episodes, SEED,
                                                               episodes_per_task=2, rows_per_episode=8)
    task_mem_ids = np.flatnonzero(~np.isin(source_files, heldout_tasks))
    held_bases = np.asarray(memory.encode(held["actions"], held["state"])[0], np.float32)
    task_bases = bases[task_qids]
    conditions = {
        "heldout_known_tasks": {"mem_ids": np.arange(len(bases)), "q_bases": held_bases,
            "q_eps": np.asarray(held["episode_id"]).astype(str), "q_init": np.asarray(held["init_state_id"]).astype(str),
            "q_keys": q_keys, "q_state": np.asarray(held["state"], np.float32), "seed": memory.seed},
        "heldout_10_tasks": {"mem_ids": task_mem_ids, "q_bases": task_bases,
            "q_eps": episodes[task_qids], "q_init": initials[task_qids], "q_keys": keys[task_qids],
            "q_state": np.asarray(rows["state"][task_qids], np.float32), "seed": memory.seed},
    }
    cache_path = out / "projection_cache.npz"
    with np.load(cache_path) as cache:
        projection = {k: cache[k] for k in cache.files}
    report = {"kappas": list(KAPPAS), "epsilon_sweep": {},
              "task_split": {"available_tasks": int(len(set(source_files))), "heldout_tasks": heldout_tasks,
                             "memory_tasks": int(len(set(source_files[task_mem_ids]))), "query_rows": len(task_qids),
                             "episode_plan": task_plan},
              "protocol": "M6 region formation replayed from frozen cached M1 bases; candidate keys and P remain fixed. Ap/R and flow denoising are invariant to unit-label epsilon because retrieval weights/decoder do not use unit ids."}

    for condition, data in conditions.items():
        ids = data["mem_ids"]
        mem_b = bases[ids]
        mem_ep, mem_init = episodes[ids], initials[ids]
        if condition == "heldout_known_tasks":
            learned_mem, learned_q = projection["full_mem_out"], projection["held_out"]
        else:
            learned_mem, learned_q = projection["task_mem_out"], projection["task_query_out"]
        centered_mem = _normalize(keys[ids] - keys[ids].mean(axis=0, keepdims=True))
        centered_q = _normalize(data["q_keys"] - keys[ids].mean(axis=0, keepdims=True))
        retrievals = {}
        for name, km, kq in (("centered_no_training", centered_mem, centered_q),
                             ("learned_projection", learned_mem, learned_q)):
            retrievals[name], _ = _topk_cosine(km, kq, mem_ep, mem_init, data["q_eps"], data["q_init"])

        cond_result = {"rows": len(data["q_bases"]), "episodes": int(len(set(data["q_eps"]))), "keys": {}}
        for kappa in KAPPAS:
            model, candidate_units = _recluster(memory, mem_b, mem_ep, kappa=kappa, rng_seed=data["seed"])
            target_units = np.full(len(data["q_bases"]), -1, np.int64)
            for i, b in enumerate(data["q_bases"]):
                unit_id, _ = model._nearest(b)
                if unit_id is not None:
                    target_units[i] = int(unit_id)
            for name, retrieval_ids in retrievals.items():
                metric = _score(retrieval_ids, candidate_units, target_units)
                metric.update({"epsilon": float(model.epsilon), "functional_median": float(model.functional_median),
                               "unit_count": len(model._centers)})
                cond_result["keys"].setdefault(name, {})[str(kappa)] = metric
        report["epsilon_sweep"][condition] = cond_result
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory", default=MEMORY_PATH)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--query-keys", default=QUERY_KEYS_PATH)
    args = parser.parse_args()
    main(args.output_dir, memory_path=args.memory, heldout_path=args.heldout,
         rows_path=args.rows, query_keys_path=args.query_keys)
