"""Centered-key retrieval for the offline anchor experiments.

The caller L2-normalizes raw keys first, as in the online memory builder.
All candidates sharing either the query's episode or initial state are excluded.
"""
from __future__ import annotations

import numpy as np


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
