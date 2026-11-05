"""Small pure helpers for the privileged-situation retrieval audit."""
from __future__ import annotations

import numpy as np


def compact_positions_to_row_ids(positions, pool_row_ids):
    """Convert positions in a compact candidate array back to dataset row IDs."""
    p = np.asarray(positions, dtype=np.int64)
    rows = np.asarray(pool_row_ids, dtype=np.int64)
    if np.any(p < 0) or np.any(p >= len(rows)):
        raise IndexError("compact candidate position is out of range")
    return rows[p]


def topk_row_ids(scores, pool_row_ids, k):
    """Stable descending rank, returning dataset IDs rather than pool offsets."""
    s = np.asarray(scores, dtype=np.float32)
    r = np.asarray(pool_row_ids, dtype=np.int64)
    if s.ndim != 1 or len(s) != len(r):
        raise ValueError("scores and pool_row_ids must be aligned vectors")
    pos = np.lexsort((r, -s))[: min(int(k), len(r))]
    return compact_positions_to_row_ids(pos, r)


def assert_nonlong_roles(task_by_row, named_row_ids):
    """Fail if any fitting/selection population contains a Long task row."""
    tasks = np.asarray(task_by_row)
    for name, ids in named_row_ids.items():
        rows = np.asarray(ids, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= len(tasks)):
            raise IndexError(f'{name} contains an invalid dataset row ID')
        if np.any(tasks[rows] < 10):
            raise AssertionError(f'{name} contains Long rows')


def align_subsequence(source_actions, demo_actions, atol=1e-6):
    """Map filtered RLDS actions to a monotone subsequence of an HDF5 demo.

    The LIBERO-40 RLDS source omits no-op frames. Matching is exact to the
    stored float32 action precision (with the explicit tolerance below), and
    skips only HDF5 frames. The returned indices are source-frame -> HDF5-frame.
    """
    src = np.asarray(source_actions, dtype=np.float32)
    demo = np.asarray(demo_actions, dtype=np.float32)
    if src.ndim != 2 or demo.ndim != 2 or src.shape[1] != demo.shape[1]:
        raise ValueError(f"expected [T,D] arrays, got {src.shape}, {demo.shape}")
    indices = np.full(len(src), -1, dtype=np.int64)
    cursor = 0
    # Exact action values are quantized in these demonstrations; rounding is
    # only an index accelerator, and every hit is checked with allclose.
    buckets: dict[tuple[float, ...], list[int]] = {}
    for j, a in enumerate(demo):
        buckets.setdefault(tuple(np.round(a, 7)), []).append(j)
    for i, a in enumerate(src):
        candidates = buckets.get(tuple(np.round(a, 7)), ())
        found = None
        for j in candidates:
            if j >= cursor and np.allclose(demo[j], a, atol=atol, rtol=0):
                found = j
                break
        if found is None:
            return indices, i
        indices[i] = found
        cursor = found + 1
    return indices, len(src)


def choose_action_aligned_demo(source_actions, demo_actions_by_id, atol=1e-6):
    """Return the unique complete demo mapping; fail closed on ambiguity."""
    matches = []
    for name in sorted(demo_actions_by_id):
        idx, matched = align_subsequence(source_actions, demo_actions_by_id[name], atol)
        if matched == len(source_actions):
            matches.append((name, idx))
    if not matches:
        raise ValueError("no HDF5 demo contains the complete RLDS action sequence")
    if len(matches) != 1:
        raise ValueError(f"ambiguous HDF5 action mapping: {[m[0] for m in matches]}")
    return matches[0]


def action_mse(query, candidates):
    q = np.asarray(query, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    return np.mean(np.square(c - q[None, ...]), axis=(-1, -2), dtype=np.float32)


def normalize_block(train, other):
    """Memory-only z-score and row normalization for one feature block."""
    tr = np.asarray(train, dtype=np.float32)
    x = np.asarray(other, dtype=np.float32)
    mu = tr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = tr.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-5] = 1.0
    tr = (tr - mu) / sd
    x = (x - mu) / sd
    tr /= np.maximum(np.linalg.norm(tr, axis=1, keepdims=True), 1e-8)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return tr.astype(np.float32), x.astype(np.float32), {"mean": mu, "std": sd}


def combine_blocks(normalized, names, weights):
    names = tuple(names)
    w = np.asarray(weights, dtype=np.float32)
    if len(names) != len(w) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("invalid block weights")
    x = np.concatenate([np.asarray(normalized[n], np.float32) * np.sqrt(wi)
                        for n, wi in zip(names, w, strict=True)], axis=1)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def retrieval_metrics(query_actions, candidate_actions, candidate_row_ids,
                      all_candidate_actions, all_candidate_row_ids, tau=0.01):
    """Top-1, best-of-5 (oracle diagnostic), and Recall@8 on full pool."""
    q = np.asarray(query_actions, np.float32)
    ca = np.asarray(candidate_actions, np.float32)
    rows = np.asarray(candidate_row_ids, np.int64)
    all_a = np.asarray(all_candidate_actions, np.float32)
    all_rows = np.asarray(all_candidate_row_ids, np.int64)
    row_to_pos = {int(r): i for i, r in enumerate(all_rows)}
    qn = q.reshape(len(q), -1)
    alln = all_a.reshape(len(all_a), -1)
    out = {"top1_mse": [], "top5_best_mse": [], "recall8": [],
           "xyz_cosine": [], "gripper_sign_agreement": []}
    for i in range(len(q)):
        local = action_mse(q[i], ca[i])
        full = np.mean(np.square(alln - qn[i]), axis=1)
        relevant = full <= float(tau)
        denom = int(relevant.sum())
        retrieved_pos = np.asarray([row_to_pos[int(r)] for r in rows[i, :min(8, len(rows[i]))]], np.int64)
        out["top1_mse"].append(float(local[0]))
        out["top5_best_mse"].append(float(local[:min(5, len(local))].min()))
        out["recall8"].append(float("nan") if denom == 0 else float(relevant[retrieved_pos].sum() / denom))
        qxyz = q[i, :, :3].sum(axis=0)
        pxyz = ca[i, 0, :, :3].sum(axis=0)
        out["xyz_cosine"].append(float(np.dot(qxyz, pxyz) /
                                        max(np.linalg.norm(qxyz) * np.linalg.norm(pxyz), 1e-12)))
        out["gripper_sign_agreement"].append(float(np.mean(np.sign(q[i, :, 6]) == np.sign(ca[i, 0, :, 6]))))
    return out

