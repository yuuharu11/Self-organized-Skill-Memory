"""Centered-key retrieval for the offline anchor experiments.

The caller L2-normalizes raw keys first, as in the online memory builder.
All candidates sharing either the query's episode or initial state are excluded.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from .offline_common import normalize_keys
else:
    from offline_common import normalize_keys


def center_normalize(keys, mean=None):
    keys = np.asarray(keys, np.float32)
    if keys.ndim != 2 or not len(keys):
        raise ValueError("keys must be a nonempty matrix")
    if mean is None:
        mean = keys.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean = np.asarray(mean, np.float32)
    if mean.shape != (keys.shape[1],):
        raise ValueError("key mean has the wrong width")
    centered = keys - mean
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8), mean


def topk_centered(queries, keys, memory_episodes, memory_initials, query_episodes,
                  query_initials, *, k=5, block_size=32):
    queries, keys = np.asarray(queries, np.float32), np.asarray(keys, np.float32)
    if queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError("query and memory keys must be matrices of equal width")
    if k < 1 or block_size < 1 or k > len(keys):
        raise ValueError("k and block_size must be positive; memory must contain k rows")
    memory_episodes, memory_initials = map(np.asarray, (memory_episodes, memory_initials))
    query_episodes, query_initials = map(np.asarray, (query_episodes, query_initials))
    for values, size in ((memory_episodes, len(keys)), (memory_initials, len(keys)),
                         (query_episodes, len(queries)), (query_initials, len(queries))):
        if values.shape != (size,):
            raise ValueError("episode and initial-state IDs must align with key rows")
    indices = np.empty((len(queries), k), np.int64)
    scores = np.empty((len(queries), k), np.float32)
    for start in range(0, len(queries), block_size):
        stop = min(start + block_size, len(queries))
        similarities = queries[start:stop] @ keys.T
        invalid = ((query_episodes[start:stop, None] == memory_episodes[None])
                   | (query_initials[start:stop, None] == memory_initials[None]))
        similarities[invalid] = -np.inf
        if np.any(np.isfinite(similarities).sum(axis=1) < k):
            raise ValueError("fewer than k eligible memory entries after episode/init exclusion")
        # Stable order matches the online index's row-ID tie break.
        chosen = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        indices[start:stop] = chosen
        scores[start:stop] = np.take_along_axis(similarities, chosen, axis=1)
    return indices, scores


def softmax_weights(similarities, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(similarities, np.float64) / float(temperature)
    if logits.ndim != 2 or not logits.shape[1] or not np.isfinite(logits).any(axis=1).all():
        raise ValueError("each query must have at least one finite similarity")
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def sample_task_queries(source_files, episode_ids, seed, n_tasks=10, episodes_per_task=2, rows_per_episode=8):
    rng = np.random.default_rng(seed)
    tasks = sorted(set(map(str, source_files)))
    selected_tasks = sorted(rng.choice(tasks, size=min(n_tasks, len(tasks)), replace=False).tolist())
    selected_rows, selected_episode_ids = [], []
    task_reports = {}
    source_files = np.asarray(source_files).astype(str)
    episode_ids = np.asarray(episode_ids).astype(str)
    for task in selected_tasks:
        task_rows = np.flatnonzero(source_files == task)
        eps = sorted(set(episode_ids[task_rows]))
        chosen_eps = sorted(rng.choice(eps, size=min(episodes_per_task, len(eps)), replace=False).tolist())
        task_reports[task] = chosen_eps
        for episode in chosen_eps:
            rows = task_rows[episode_ids[task_rows] == episode]
            if len(rows) > rows_per_episode:
                take = np.linspace(0, len(rows) - 1, rows_per_episode).round().astype(int)
                rows = rows[take]
            selected_rows.extend(map(int, rows))
            selected_episode_ids.extend([episode] * len(rows))
    return np.asarray(selected_rows, np.int64), selected_tasks, task_reports


def behavior_neighbors(bases, episodes, initials, *, k=8, query_k=48, workers=8):
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(bases, np.float32))
    distances, indices = tree.query(bases, k=min(query_k, len(bases)), workers=workers)
    if indices.ndim == 1:
        indices, distances = indices[:, None], distances[:, None]
    positives = np.empty((len(bases), k), np.int64)
    for row in range(len(bases)):
        candidates = [int(j) for j in indices[row]
                      if int(j) != row and episodes[int(j)] != episodes[row]
                      and initials[int(j)] != initials[row]]
        if len(candidates) < k:
            # Keep the closest available different-demo examples; self/init are always excluded.
            candidates = [int(j) for j in np.argsort(np.linalg.norm(bases - bases[row], axis=1))
                          if int(j) != row and episodes[int(j)] != episodes[row]
                          and initials[int(j)] != initials[row]]
        if not candidates:
            raise ValueError(f"no different-demo positive for row {row}")
        fill = (candidates * ((k + len(candidates) - 1) // len(candidates)))[:k]
        positives[row] = fill
    return positives


def topk_cosine(memory_keys, query_keys, mem_eps, mem_initials, q_eps, q_initials, *, k=5):
    memory_keys = normalize_keys(memory_keys)
    query_keys = normalize_keys(query_keys)
    ids = np.empty((len(query_keys), k), np.int64)
    sims = np.empty((len(query_keys), k), np.float32)
    for i, query in enumerate(query_keys):
        scores = memory_keys @ query
        eligible = (mem_eps != q_eps[i]) & (mem_initials != q_initials[i])
        scores[~eligible] = -np.inf
        selected = np.argsort(-scores, kind="stable")[:k]
        ids[i], sims[i] = selected, scores[selected]
    return ids, sims


def macro_recall(true_unit, candidate_units, majority_unit):
    true_unit = np.asarray(true_unit, np.int64)
    candidate_units = np.asarray(candidate_units, np.int64)
    classes = np.unique(true_unit[true_unit >= 0])
    hit1 = candidate_units[:, 0] == true_unit
    hit5 = np.any(candidate_units == true_unit[:, None], axis=1)
    per1, per5 = {}, {}
    for unit in classes:
        mask = true_unit == unit
        per1[str(int(unit))] = float(hit1[mask].mean())
        per5[str(int(unit))] = float(hit5[mask].mean())
    majority_macro = float(np.mean(classes == int(majority_unit))) if len(classes) else 0.0
    return {"hit1": float(hit1.mean()), "hit5": float(hit5.mean()),
            "macro_recall1": float(np.mean(list(per1.values()))) if per1 else 0.0,
            "macro_recall5": float(np.mean(list(per5.values()))) if per5 else 0.0,
            "per_unit_recall1": per1, "per_unit_recall5": per5,
            "majority_macro_recall1_and_5": majority_macro}
