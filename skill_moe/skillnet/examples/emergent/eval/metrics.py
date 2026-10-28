"""Go/no-go item 1: information between clustered z and label-free motion features.

z is discretized by k-means fitted on the store split only and MI is measured on the
held-out query split (whole initial-state groups, G2 split). Motion labels come from

* ``env``: the end-effector trajectory that actually happened (env_*), and
* ``action``: the encoder's own input actions (triviality reference),

so that posterior-z MI explained by the encoder input can be told apart from MI of
the observation-only prior z. Point estimates and label-shuffle nulls reuse the
primitives experiment helpers; differences against baselines (untrained model, proprio)
use a whole-initial-state-group bootstrap with Bonferroni correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.primitives.experiment import (
    _conditional_routing_information,
    _entropy_bits,
    _mi_with_shuffle_null,
    _motion_categories,
)

from .data import ZRun

LABELS = ("direction", "rotation", "gripper", "joint")


@dataclass(frozen=True)
class MIConfig:
    cluster_counts: tuple[int, ...] = (8, 16, 32)
    primary_clusters: int = 16
    n_permutations: int = 200
    bootstrap_samples: int = 2000
    kmeans_restarts: int = 4
    kmeans_iterations: int = 100
    seed: int = 0
    minimum_groups: int = 10


# --------------------------------------------------------------------------
# Discretization
# --------------------------------------------------------------------------

def standardize(fit: np.ndarray, *values: np.ndarray) -> list[np.ndarray]:
    mean, scale = fit.mean(axis=0), fit.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return [(v - mean) / scale for v in values]


def _sq_dist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None], 0.0)


def kmeans(x: np.ndarray, k: int, *, seed: int = 0, restarts: int = 4, iterations: int = 100) -> np.ndarray:
    """k-means++ / Lloyd, best of `restarts`; returns centers (k', D), k' = min(k, unique rows)."""
    x = np.asarray(x, dtype=np.float64)
    k = int(min(k, len(np.unique(x, axis=0))))
    if k < 1:
        raise ValueError("kmeans needs at least one row")
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d = _sq_dist(x, np.asarray(centers)).min(axis=1)
            p = d / d.sum() if d.sum() > 0 else None
            centers.append(x[rng.choice(len(x), p=p)])
        c = np.asarray(centers)
        for _ in range(iterations):
            labels = _sq_dist(x, c).argmin(axis=1)
            new = np.stack([x[labels == j].mean(axis=0) if np.any(labels == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = _sq_dist(x, c).min(axis=1).sum()
        if inertia < best_inertia:
            best, best_inertia = c, inertia
    return best


def assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return _sq_dist(np.asarray(x, dtype=np.float64), centers).argmin(axis=1)


# --------------------------------------------------------------------------
# Motion labels (evaluation only)
# --------------------------------------------------------------------------

def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(n, np.clip(q[..., 3:4], -1, 1))
    return v * np.divide(angle, n, out=np.zeros_like(n), where=n > 1e-12)


def _joint(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels = dict(labels)
    labels["joint"] = np.char.add(np.char.add(labels["direction"].astype(str), "|"),
                                  np.char.add(np.char.add(labels["rotation"].astype(str), "|"),
                                              labels["gripper"].astype(str)))
    return labels


def env_motion_labels(motion: np.ndarray) -> dict[str, np.ndarray]:
    """Observed eef outcome over the z interval -> primitives' coarse categories.

    `motion` is the G2 target (N,H,7+Dg): dpos, xyzw dquat, dgripper. Gripper uses the
    two-finger gap change q_left - q_right (primitives rule) when Dg == 2.
    """
    rot = _quat_to_rotvec(motion[..., 3:7])
    converted = np.concatenate([motion[..., :3], rot, motion[..., 7:]], axis=-1)
    return _joint(_motion_categories(converted))


def action_motion_labels(actions: np.ndarray, mask: np.ndarray, *, gripper_index: int = 6) -> dict[str, np.ndarray]:
    """Same three categories from the encoder-input action chunk (triviality reference).

    Direction/rotation: dominant signed axis of the summed delta commands. Gripper:
    LIBERO command +1 close / -1 open; closing gives -1 (gap shrinks), opening +1.
    """
    m = mask[..., None].astype(float)
    summed = (actions * m).sum(axis=1)
    out = {}
    for name, sl, thr in (("direction", slice(0, 3), 1e-3), ("rotation", slice(3, 6), 1e-3)):
        v = summed[:, sl]
        cls = np.zeros(len(v), dtype=np.int64)
        moving = np.linalg.norm(v, axis=1) >= thr
        axis = np.argmax(np.abs(v), axis=1)
        sign = v[np.arange(len(v)), axis] >= 0
        cls[moving] = (1 + 2 * axis + (~sign).astype(np.int64))[moving]
        out[name] = cls
    first = np.array([a[mk][0, gripper_index] if mk.any() else 0. for a, mk in zip(actions, mask)])
    last = np.array([a[mk][-1, gripper_index] if mk.any() else 0. for a, mk in zip(actions, mask)])
    change = last - first
    out["gripper"] = np.where(change > 0.5, -1, np.where(change < -0.5, 1, 0))
    return _joint(out)


# --------------------------------------------------------------------------
# Fast integer MI for the bootstrap
# --------------------------------------------------------------------------

def _codes(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(labels).astype(str), return_inverse=True)[1]


def _mi_codes(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb) / len(a)
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])))


def _group_bootstrap_diff(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], groups: np.ndarray,
                          config: MIConfig) -> list[dict[str, Any]]:
    """Each pair: (assign_a, assign_b, labels). Diff = MI(a) - MI(b), whole-group resampling, Bonferroni."""
    labels_g, gi = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(gi == g) for g in range(len(labels_g))]
    coded = []
    for a, b, y in pairs:
        ac, bc, yc = _codes(a), _codes(b), _codes(y)
        coded.append((ac, bc, yc, ac.max() + 1, bc.max() + 1, yc.max() + 1))
    estimates = [(_mi_codes(ac, yc, na, ny) - _mi_codes(bc, yc, nb, ny)) for ac, bc, yc, na, nb, ny in coded]
    out = [{"estimate": e, "ci95_familywise": None} for e in estimates]
    if len(labels_g) < config.minimum_groups or not pairs:
        for o in out:
            o["unavailable_reason"] = "too_few_query_init_states"
        return out
    tail = 0.05 / (2 * len(pairs))
    n_res = max(config.bootstrap_samples, int(np.ceil(10 / tail)))
    rng = np.random.default_rng(config.seed + 5)
    boot = np.empty((n_res, len(pairs)))
    for r in range(n_res):
        idx = np.concatenate([members[g] for g in rng.integers(len(labels_g), size=len(labels_g))])
        for j, (ac, bc, yc, na, nb, ny) in enumerate(coded):
            boot[r, j] = _mi_codes(ac[idx], yc[idx], na, ny) - _mi_codes(bc[idx], yc[idx], nb, ny)
    lo, hi = np.quantile(boot, [tail, 1 - tail], axis=0)
    for j, o in enumerate(out):
        o["ci95_familywise"] = [float(lo[j]), float(hi[j])]
        o["bootstrap_samples"] = n_res
    return out


# --------------------------------------------------------------------------
# Posterior diagnostics
# --------------------------------------------------------------------------

def posterior_diagnostics(run: ZRun, rows: np.ndarray) -> dict[str, Any]:
    mu, lv = run.z["post"][rows], run.z_logvar["post"][rows]
    kl_dim = 0.5 * (np.exp(lv) + mu ** 2 - 1 - lv)  # KL(q || N(0, I)) per dimension, nats
    out = {"kl_to_standard_normal_nats_per_row": float(kl_dim.sum(1).mean()),
           "kl_per_dim_nats": kl_dim.mean(0).tolist(),
           "active_dims_kl_gt_0.01": int((kl_dim.mean(0) > 0.01).sum()), "z_dim": int(mu.shape[1]),
           "logvar_recorded": "z_post_logvar" not in run.derived}
    if "prior" in run.z_logvar and np.all(np.isfinite(run.z_logvar["prior"][rows])):
        pm, plv = run.z["prior"][rows], run.z_logvar["prior"][rows]
        kl = 0.5 * (plv - lv + (np.exp(lv) + (mu - pm) ** 2) / np.exp(plv) - 1)
        out["kl_post_to_prior_nats_per_row"] = float(kl.sum(1).mean())
    if "prior" in run.z and run.z["prior"].shape == run.z["post"].shape:
        pm = run.z["prior"][rows]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(mu))
        d_pair = np.linalg.norm(mu - pm, axis=1).mean()
        d_rand = np.linalg.norm(mu - pm[perm], axis=1).mean()
        out["prior_to_post_distance_ratio_vs_shuffled"] = float(d_pair / max(d_rand, 1e-12))
    return out


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def evaluate_mi(run: ZRun, query: np.ndarray, store: np.ndarray, config: MIConfig = MIConfig(),
                sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """MI tables for every z source (plus proprio as a state baseline)."""
    q_rows = np.flatnonzero(query & run.usable)
    s_rows = np.flatnonzero(store & run.usable)
    if not len(q_rows) or not len(s_rows):
        raise ValueError("no usable query or store rows")
    label_sets = {"env": env_motion_labels(run.motion[q_rows])}
    if run.target_actions is not None:
        label_sets["action"] = action_motion_labels(run.target_actions[q_rows], run.target_mask[q_rows])
    tasks = run.task[q_rows]
    groups = run.init_state_id[q_rows]
    sources = tuple(sources or sorted(run.z))
    matrices = {s: run.z[s] for s in sources}
    matrices["proprio"] = run.proprio
    assignments: dict[str, dict[int, np.ndarray]] = {}
    table: dict[str, Any] = {}
    for name, mat in matrices.items():
        fit, values = standardize(mat[s_rows], mat[s_rows], mat[q_rows])
        assignments[name] = {}
        table[name] = {}
        for K in config.cluster_counts:
            centers = kmeans(fit, K, seed=config.seed, restarts=config.kmeans_restarts,
                             iterations=config.kmeans_iterations)
            a = assign(values, centers)
            assignments[name][K] = a
            entry: dict[str, Any] = {"n_clusters_fitted": int(len(centers)),
                                     "n_clusters_used_on_query": int(len(np.unique(a))),
                                     "cluster_entropy_bits": _entropy_bits(a)}
            for set_name, labels in label_sets.items():
                for lab in LABELS:
                    entry[f"{set_name}/{lab}"] = _mi_with_shuffle_null(a, labels[lab], seed=config.seed,
                                                                       n_permutations=config.n_permutations)
            entry["task"] = _mi_with_shuffle_null(a, tasks, seed=config.seed, n_permutations=config.n_permutations)
            joint = label_sets["env"]["joint"]
            entry["env/joint_given_task"] = _conditional_routing_information(
                a, joint, tasks, seed=config.seed, n_permutations=config.n_permutations)
            entry["task_given_env/joint"] = _conditional_routing_information(
                a, tasks, joint, seed=config.seed, n_permutations=config.n_permutations)
            table[name][str(K)] = entry
    # Paired differences at the primary K, env labels: trained source minus each baseline.
    K = config.primary_clusters if config.primary_clusters in config.cluster_counts else config.cluster_counts[0]
    pairs, labels_out = [], []
    for s in sources:
        if s.startswith("untrained_"):
            continue
        baselines = ["proprio"] + ([f"untrained_{s}"] if f"untrained_{s}" in matrices else [])
        for b in baselines:
            for lab in LABELS:
                pairs.append((assignments[s][K], assignments[b][K], label_sets["env"][lab]))
                labels_out.append({"source": s, "baseline": b, "label": f"env/{lab}"})
    diffs = _group_bootstrap_diff(pairs, groups, config)
    comparisons = [dict(meta, **d) for meta, d in zip(labels_out, diffs)]
    verdict = {}
    for s in sources:
        if s.startswith("untrained_"):
            continue
        rows = [c for c in comparisons if c["source"] == s]
        def positive(b):
            sel = [c for c in rows if c["baseline"] == b]
            if not sel:
                return "baseline_missing"
            if any(c["ci95_familywise"] is None for c in sel):
                return "no_interval"
            return "above" if any(c["ci95_familywise"][0] > 0 for c in sel) else "not_above"
        shuffle = table[s][str(K)]["env/joint"]
        verdict[s] = {"vs_shuffle_null_p95": "above" if shuffle["raw_bits"] > shuffle["shuffle_null_p95_bits"] else "not_above",
                      "vs_untrained": positive(f"untrained_{s}"), "vs_proprio": positive("proprio")}
    return {
        "config": asdict(config), "primary_clusters": K,
        "n_query_rows": int(len(q_rows)), "n_store_rows": int(len(s_rows)),
        "n_query_init_states": int(len(np.unique(groups))), "n_query_tasks": int(len(np.unique(tasks))),
        "label_sets": {k: {lab: {str(u): int(c) for u, c in zip(*np.unique(v[lab].astype(str), return_counts=True))}
                           for lab in ("direction", "rotation", "gripper")} for k, v in label_sets.items()},
        "table": table, "comparisons_primary_k": comparisons, "verdict": verdict,
        "posterior": posterior_diagnostics(run, q_rows),
        "notes": ("posterior z saw the action chunk of the same interval: its MI with `action/*` (and largely "
                  "`env/*`) measures what passed the bottleneck, not emergence. Prior z sees only the "
                  "observation; `prior` vs `env/*` is the non-trivial number. Shuffle null = cardinality "
                  "diagnostic for correlated windows, not a significance test."),
    }
