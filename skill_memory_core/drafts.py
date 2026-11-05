"""Safe row-ID candidate sampling and cross-task shuffled controls."""
from __future__ import annotations

import numpy as np


def validate_candidates(candidate_row_ids, query_row_ids, query_tasks,
                        memory_row_ids, memory_tasks, *, require_different_task=True):
    cand = np.asarray(candidate_row_ids, np.int64)
    qid = np.asarray(query_row_ids, np.int64)
    qt, mids, mt = np.asarray(query_tasks), np.asarray(memory_row_ids, np.int64), np.asarray(memory_tasks)
    if cand.ndim != 2 or cand.shape[0] != len(qid) or len(qt) != len(qid):
        raise ValueError("candidate table must be [queries,K] and align with row IDs")
    if len(mids) != len(mt):
        raise ValueError("memory row IDs and task IDs do not align")
    loc = {int(r): i for i, r in enumerate(mids)}
    if any(int(r) not in loc for r in cand.flat):
        raise AssertionError("candidate is not a dataset row ID in the supplied memory")
    candidate_tasks = mt[np.asarray([[loc[int(r)] for r in c] for c in cand])]
    if require_different_task and np.any(candidate_tasks == qt[:, None]):
        raise AssertionError("candidate from same task")
    return candidate_tasks


def sample_rank_mixture(candidate_row_ids, rng, *, probabilities=(.70, .075, .075, .075, .075)):
    """Pick one actual candidate per query; no averaging of action windows."""
    cand = np.asarray(candidate_row_ids, np.int64)
    p = np.asarray(probabilities, np.float64)
    if cand.ndim != 2 or len(p) != cand.shape[1] or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("rank probabilities must match the candidate-list width and sum to one")
    ranks = rng.choice(cand.shape[1], size=len(cand), p=p).astype(np.int64)
    return cand[np.arange(len(cand)), ranks], ranks


def shuffle_cross_task(source_row_ids, query_row_ids, query_tasks, source_tasks, *, seed: int,
                       max_attempts: int = 100):
    """Permute selected dataset row IDs, requiring every shuffled source task differ."""
    src, qid = np.asarray(source_row_ids, np.int64), np.asarray(query_row_ids, np.int64)
    qt, st = np.asarray(query_tasks), np.asarray(source_tasks)
    if src.shape != qid.shape or qt.shape != qid.shape or st.shape != src.shape:
        raise ValueError("source/query IDs and task vectors must align")
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        perm = rng.permutation(len(src))
        out = src[perm]
        if np.all(st[perm] != qt):
            return out, perm
    # Sampling with replacement can alter the selected-action distribution;
    # preserve the exact source multiset or fail explicitly.
    raise RuntimeError("could not find a cross-task permutation preserving the source multiset")


def drafts_by_row_id(dataset_actions, candidate_row_ids):
    """Get raw action draft(s) by global dataset IDs."""
    ids = np.asarray(candidate_row_ids, np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= len(dataset_actions)):
        raise IndexError("candidate ID is not a valid dataset row ID")
    return np.asarray(dataset_actions[ids], np.float32)
