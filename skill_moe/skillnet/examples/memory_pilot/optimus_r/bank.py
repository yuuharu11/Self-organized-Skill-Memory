"""Fixed-size Optimus-R-style prototype bank and top-k retrieval helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InitialBank:
    query: np.ndarray
    skill: np.ndarray
    support: np.ndarray
    exemplars: np.ndarray
    exemplar_mask: np.ndarray
    metadata: dict


def _normalize_rows(values):
    values = np.asarray(values, np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def _lloyd(data: np.ndarray, k: int, *, seed: int, iterations: int = 24):
    """Small deterministic cosine k-means used only for initial bank discovery."""
    x = _normalize_rows(data)
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError(f"cannot initialize {k} clusters from only {len(x)} rows")
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros(len(x), np.int32)
    for _ in range(iterations):
        labels = np.argmax(x @ centers.T, axis=1).astype(np.int32)
        counts = np.bincount(labels, minlength=k)
        updated = np.zeros_like(centers)
        np.add.at(updated, labels, x)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            best_fit = np.max(x @ centers.T, axis=1)
            replacements = np.argsort(best_fit)[:len(empty)]
            updated[empty] = x[replacements]
            counts[empty] = 1
        centers = _normalize_rows(updated / counts[:, None])
    labels = np.argmax(x @ centers.T, axis=1).astype(np.int32)
    return labels


def _split_high_skill_variance(indices: np.ndarray, query: np.ndarray, skill: np.ndarray):
    """Bisect a query cluster along the dominant skill-variance axis."""
    if len(indices) < 2:
        raise ValueError("cannot split a singleton query cluster")
    values = skill[indices].astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    axis = np.ones(centered.shape[1], np.float64) / np.sqrt(centered.shape[1])
    for _ in range(10):
        next_axis = centered.T @ (centered @ axis)
        norm = np.linalg.norm(next_axis)
        if norm < 1e-10:
            break
        axis = next_axis / norm
    if np.linalg.norm(axis) < 1e-8:
        q = query[indices].astype(np.float64)
        _, _, right = np.linalg.svd(q - q.mean(axis=0, keepdims=True), full_matrices=False)
        axis = right[0]
        projection = (q - q.mean(axis=0, keepdims=True)) @ axis
    else:
        projection = centered @ axis
    order = np.argsort(projection, kind="stable")
    midpoint = len(order) // 2
    left, right = indices[order[:midpoint]], indices[order[midpoint:]]
    if not len(left) or not len(right):
        raise ValueError("variance split produced an empty child")
    return left, right


def build_initial_bank(query, skill, *, bank_size: int = 128, query_seed_clusters: int = 64,
                       exemplars_per_entry: int = 3, seed: int = 7) -> InitialBank:
    """Build 128 fixed prototypes: query clustering then skill-variance splits.

    Unspecified paper clustering details are completed by a deterministic
    two-stage policy. First, query-space cosine k-means makes 64 coarse groups.
    The groups with greatest within-group skill variance are bisected along
    their dominant skill axis until exactly 128 entries are present.
    """
    query = _normalize_rows(query)
    skill = np.asarray(skill, np.float32)
    if query.ndim != 2 or skill.ndim != 2 or query.shape[0] != skill.shape[0]:
        raise ValueError("query and skill must be aligned rank-2 arrays")
    if bank_size < 2 or query_seed_clusters < 1 or query_seed_clusters > bank_size:
        raise ValueError("invalid initial bank cluster counts")
    if bank_size > len(query):
        raise ValueError("fixed bank cannot exceed the number of source samples")
    coarse_count = min(query_seed_clusters, len(query))
    labels = _lloyd(query, coarse_count, seed=seed)
    clusters = [np.flatnonzero(labels == label) for label in range(coarse_count)]
    clusters = [cluster for cluster in clusters if len(cluster)]

    while len(clusters) < bank_size:
        candidates = [i for i, cluster in enumerate(clusters) if len(cluster) >= 2]
        if not candidates:
            raise ValueError("not enough splittable source rows to reach fixed bank size")
        # Prefer high skill variance, then large clusters, then stable low index.
        variances = [float(np.mean(np.sum((skill[clusters[i]] - skill[clusters[i]].mean(axis=0)) ** 2, axis=1)))
                    for i in candidates]
        chosen = candidates[max(range(len(candidates)), key=lambda j: (variances[j], len(clusters[candidates[j]]), -candidates[j]))]
        left, right = _split_high_skill_variance(clusters[chosen], query, skill)
        clusters[chosen:chosen + 1] = [left, right]

    proto_q, proto_s, support = [], [], []
    exemplars = np.full((bank_size, exemplars_per_entry), -1, np.int64)
    exemplar_mask = np.zeros((bank_size, exemplars_per_entry), np.bool_)
    rng = np.random.default_rng(seed + 1)
    for bank_index, members in enumerate(clusters):
        proto_q.append(_normalize_rows(query[members].mean(axis=0, keepdims=True))[0])
        proto_s.append(skill[members].mean(axis=0))
        support.append(len(members))
        selected = members if len(members) <= exemplars_per_entry else rng.choice(
            members, size=exemplars_per_entry, replace=False
        )
        exemplars[bank_index, :len(selected)] = selected
        exemplar_mask[bank_index, :len(selected)] = True
    bank = InitialBank(
        query=np.asarray(proto_q, np.float32), skill=np.asarray(proto_s, np.float32),
        support=np.asarray(support, np.int32), exemplars=exemplars, exemplar_mask=exemplar_mask,
        metadata={"bank_size": bank_size, "query_seed_clusters": coarse_count,
                  "split_rule": "highest skill variance; dominant-axis median split",
                  "similarity": "cosine", "source_rows": len(query), "seed": seed},
    )
    if bank.query.shape[0] != bank_size or int(bank.support.sum()) != len(query):
        raise AssertionError("bank construction did not preserve the fixed capacity or all rows")
    return bank


def retrieve_numpy(query, bank_query, bank_skill, *, top_k: int = 2, temperature: float = 0.1):
    """Numpy equivalent for runner-side retrieval and deterministic tests."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = np.asarray(query, np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    keys = _normalize_rows(bank_query)
    scores = q @ keys.T
    indices = np.argsort(-scores, axis=-1, kind="stable")[..., :top_k]
    selected_scores = np.take_along_axis(scores, indices, axis=-1)
    logits = selected_scores / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    values = np.asarray(bank_skill, np.float32)[indices]
    merged = np.sum(values * weights[..., None], axis=-2)
    return merged, indices, weights, selected_scores
