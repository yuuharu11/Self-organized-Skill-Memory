"""G2 follow-up: does retrieval still find similar motion across tasks and subtasks?

CPU only. Reuses the G2 reader, initial-state split, leakage checks, motion
target, kNN standardization and the cluster bootstrap with Bonferroni
correction from ``evaluate.py``. See ``SCHEMA.md`` (section "Cross-task and
cross-subtask retrieval") for the subtask-labelling rule and the metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import (
    EvaluationConfig,
    _compare_to_proprio,
    _episode_entries,
    _feature_matrix,
    _rotation_angle,
    _stack_rows,
    load_run,
    split_by_init_state,
)

METRICS = ("eef_relative_position", "eef_relative_rotation_rad", "gripper_relative")
# Annotation skill IDs of data_process/libero/instruct2plan_40.json (all_classes).
SKILL_NAMES = {0: "pick", 1: "place", 2: "push", 3: "open/close", 4: "turn"}
PICK, PLACE = 0, 1
AMBIGUOUS = -1
CONDITIONS = {
    "a_unconstrained": "no constraint",
    "b_other_task": "store rows from a different task only",
    "c_other_subtask": "store rows with a different (known) subtask skill ID only",
    "d_other_task_same_subtask": "supplementary: different task but the same subtask skill ID",
}
SOURCE_CATEGORIES = ("same_task_same_subtask", "same_task_other_subtask",
                     "other_task_same_subtask", "other_task_other_subtask", "subtask_unknown")


@dataclass(frozen=True)
class CrossTaskConfig:
    base: EvaluationConfig = field(default_factory=EvaluationConfig)
    recall_k_values: tuple[int, ...] = (1, 5, 20)
    gripper_debounce_steps: int = 5
    gripper_action_index: int = 6
    # Recorded `skills` are annotation IDs + 1 (0 is padding; LiberoInputs adds 1).
    skill_id_offset: int = 1
    near_oracle_k: int = 5
    near_quantiles: tuple[float, ...] = (0.5, 0.9)


# --------------------------------------------------------------------------
# Current-subtask labels
# --------------------------------------------------------------------------

def episode_plan(arrays: dict[str, np.ndarray], offset: int) -> list[int]:
    """Annotation skill-ID plan passed to the policy; must be constant in an episode."""
    skills, mask = np.asarray(arrays["skills"]), np.asarray(arrays["skill_mask"]).astype(bool)
    if skills.ndim != 2 or skills.shape != mask.shape or len(skills) == 0:
        raise ValueError("skills and skill_mask must have identical shape (N, L)")
    if np.any(skills != skills[0]) or np.any(mask != mask[0]):
        raise ValueError("skills/skill_mask change within an episode; current position is then recorded "
                         "explicitly and this plan-alignment rule does not apply")
    plan = [int(x) - offset for x in skills[0][mask[0]]]
    if not plan or any(x < 0 for x in plan):
        raise ValueError(f"invalid recorded plan {skills[0].tolist()} with mask {mask[0].tolist()}")
    return plan


def executed_gripper_closed(arrays: dict[str, np.ndarray], gripper_index: int) -> np.ndarray:
    """Per environment step: was the executed gripper command 'close' (>0)?

    Index t is the action taken after observation step t. Steps before the first
    inference use LIBERO's dummy (open) action.
    """
    steps = np.asarray(arrays["step"])
    n_exec = np.asarray(arrays["n_executed"])
    chunk = np.asarray(arrays["pred_chunk"])
    if chunk.ndim != 3 or len(chunk) != len(steps) or len(n_exec) != len(steps):
        raise ValueError("pred_chunk (N,H,Da), n_executed (N) and step (N) must align")
    total = int(steps[-1] + n_exec[-1])
    closed = np.zeros(total, dtype=bool)
    for s, n, actions in zip(steps, n_exec, chunk):
        if n < 0 or n > len(actions):
            raise ValueError("n_executed out of range")
        closed[int(s):int(s) + int(n)] = actions[:int(n), gripper_index] > 0
    return closed


def gripper_events(closed: np.ndarray, debounce: int) -> list[tuple[int, str]]:
    """Debounced transitions (time, 'close'|'open'), starting from open.

    A change counts only if the new command persists for ``debounce`` steps.
    """
    events, state, t = [], False, 0
    while t < len(closed):
        if closed[t] != state and t + debounce <= len(closed) and np.all(closed[t:t + debounce] == closed[t]):
            state = bool(closed[t])
            events.append((t, "close" if state else "open"))
            t += debounce
            continue
        t += 1
    return events


def subtask_labels(plan: list[int], closed: np.ndarray, debounce: int) -> np.ndarray:
    """Current annotation skill ID for every environment step (AMBIGUOUS if undetermined).

    The plan position advances on gripper events: pick ends when a grasp is
    completed (close event + debounce), place when a release is completed. A
    non-grasp skill (push, open/close, turn) followed by more plan items ends at
    the completion of its own grasp-release cycle if the trajectory contains
    one more grasp than the remaining picks need; otherwise its boundary with
    the following pick is not observable and those steps are AMBIGUOUS. After
    the plan is consumed, steps keep the last skill until an unexpected extra
    grasp, after which they are AMBIGUOUS.
    """
    total = len(closed)
    labels = np.full(total, AMBIGUOUS, dtype=np.int64)
    events = gripper_events(closed, debounce)

    def next_event(kind: str, t0: int) -> int | None:
        for t, k in events:
            if k == kind and t >= t0:
                return min(total, t + debounce)
        return None

    if len(plan) == 1:
        labels[:] = plan[0]
        return labels
    cursor, j = 0, 0
    while j < len(plan) and cursor < total:
        skill = plan[j]
        if skill in (PICK, PLACE):
            # Place's release must follow the grasp that ended the pick (cursor is after it).
            end = next_event("close" if skill == PICK else "open", cursor)
            if end is None:
                labels[cursor:] = skill
                return labels
            labels[cursor:end] = skill
            cursor, j = end, j + 1
            continue
        if j == len(plan) - 1:
            labels[cursor:] = skill
            return labels
        needed = sum(1 for x in plan[j + 1:] if x == PICK)
        closes = [t for t, k in events if k == "close" and t >= cursor]
        own_release = next_event("open", closes[0]) if closes else None
        if len(closes) > needed and own_release is not None:
            labels[cursor:own_release] = skill
            cursor, j = own_release, j + 1
            continue
        if plan[j + 1] != PICK:
            return labels  # rest stays AMBIGUOUS
        end = next_event("close", cursor)
        if end is None:
            return labels
        labels[cursor:end] = AMBIGUOUS  # this skill or the following pick
        cursor, j = end, j + 2
    if cursor < total:
        labels[cursor:] = plan[-1]
        extra = next_event("close", cursor) if plan[-1] == PLACE else None
        if extra is not None:
            start = max(cursor, extra - debounce)
            labels[start:] = AMBIGUOUS
    return labels


# --------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------

def _knn_weights(distances: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Weights of the k nearest, integrating uniform tie breaking at the boundary (as in evaluate.py)."""
    effective_k = min(k, len(distances))
    order = np.argsort(distances, kind="stable")
    boundary = distances[order[effective_k - 1]]
    closer = distances < boundary
    tied = distances == boundary
    weights = np.zeros(len(distances))
    weights[closer] = 1.0
    weights[tied] = (effective_k - closer.sum()) / tied.sum()
    return weights, effective_k


def _pairwise_outcomes(q_targets: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
    """(nq, ns, 3) motion distances, identical to evaluate._outcome_distance."""
    out = np.empty((len(q_targets), len(s_targets), 3))
    for i, q in enumerate(q_targets):
        diff = s_targets - q[None]
        out[i, :, 0] = np.linalg.norm(diff[:, :, :3], axis=2).mean(axis=1)
        out[i, :, 1] = _rotation_angle(np.broadcast_to(q[None, :, 3:7], s_targets[:, :, 3:7].shape),
                                       s_targets[:, :, 3:7]).mean(axis=1)
        out[i, :, 2] = np.abs(diff[:, :, 7:]).mean(axis=(1, 2))
    return out


def _standardized_distances(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Euclidean distance after store-only per-dimension standardization (as evaluate._distances)."""
    mean, scale = s.mean(axis=0), s.std(axis=0)
    scale[scale < 1e-8] = 1.0
    zs, zq = (s - mean) / scale, (q - mean) / scale
    out = np.stack([np.linalg.norm(zq[i][None] - zs, axis=1) for i in range(len(zq))])
    if not np.all(np.isfinite(out)):
        raise ValueError("retrieval distances are not finite")
    return out


def _cluster_bootstrap(values: np.ndarray, groups: np.ndarray, seed: int, min_samples: int) -> dict[str, Any]:
    """Whole-group percentile bootstrap of row-weighted means, Bonferroni over all columns.

    Same procedure and thresholds as evaluate._compare_to_proprio.
    """
    values = values.reshape(len(values), -1)
    labels, index = np.unique(groups, return_inverse=True)
    counts = np.bincount(index)
    sums = np.zeros((len(labels), values.shape[1]))
    np.add.at(sums, index, values)
    tail = 0.05 / (2 * values.shape[1])
    n_resamples = max(min_samples, int(np.ceil(10 / tail)))
    intervals: list[Any] = [None] * values.shape[1]
    if len(labels) >= 10:
        rng = np.random.default_rng(seed)
        boot = np.empty((n_resamples, values.shape[1]))
        for start in range(0, n_resamples, 256):
            idx = rng.integers(len(labels), size=(min(256, n_resamples - start), len(labels)))
            boot[start:start + len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
        bounds = np.quantile(boot, [tail, 1 - tail], axis=0)
        for j in range(values.shape[1]):
            if np.ptp(sums[:, j] / counts) > 1e-12 or np.all(values[:, j] == 0):
                intervals[j] = [float(bounds[0, j]), float(bounds[1, j])]
    return {"estimates": values.mean(axis=0).tolist(), "intervals": intervals,
            "n_groups": int(len(labels)), "n_comparisons": int(values.shape[1]),
            "bootstrap_samples": n_resamples if len(labels) >= 10 else 0}


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def _row_table(episodes, ids, key, horizon, meta):
    rows = _stack_rows(episodes, ids, key, horizon)
    table = {
        "episode_id": np.array([r["episode_id"] for r in rows]),
        "init_state_id": np.array([r["init_state_id"] for r in rows]),
        "step": np.array([r["step"] for r in rows]),
        "feature": np.stack([r["feature"] for r in rows]) if rows else np.zeros((0, 0)),
        "target": np.stack([r["target"] for r in rows]) if rows else np.zeros((0, horizon, 8)),
    }
    table["task"] = np.array([meta[e]["task"] for e in table["episode_id"]])
    table["subtask"] = np.array([meta[e]["labels"][s] if s < len(meta[e]["labels"]) else AMBIGUOUS
                                 for e, s in zip(table["episode_id"], table["step"])], dtype=np.int64)
    return table


def condition_masks(q: dict[str, np.ndarray], s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Eligible store rows per query (nq, ns) for each condition, with leakage guards."""
    leak_free = ((q["episode_id"][:, None] != s["episode_id"][None])
                 & (q["init_state_id"][:, None] != s["init_state_id"][None]))
    other_task = q["task"][:, None] != s["task"][None]
    known = (q["subtask"][:, None] != AMBIGUOUS) & (s["subtask"][None] != AMBIGUOUS)
    same_sub = q["subtask"][:, None] == s["subtask"][None]
    return {
        "a_unconstrained": leak_free,
        "b_other_task": leak_free & other_task,
        "c_other_subtask": leak_free & known & ~same_sub,
        "d_other_task_same_subtask": leak_free & known & other_task & same_sub,
    }


def _source_category(q, s) -> np.ndarray:
    same_task = q["task"][:, None] == s["task"][None]
    unknown = (s["subtask"][None] == AMBIGUOUS) | (q["subtask"][:, None] == AMBIGUOUS)
    same_sub = q["subtask"][:, None] == s["subtask"][None]
    cat = np.where(same_task, np.where(same_sub, 0, 1), np.where(same_sub, 2, 3))
    return np.where(unknown, 4, cat)


def evaluate_crosstask(run_dir: str | Path, config: CrossTaskConfig = CrossTaskConfig(),
                       plan_annotations: str | Path | None = None) -> dict[str, Any]:
    base = config.base
    manifest, episodes = load_run(run_dir)
    entries = {str(e.get("episode_id", e.get("id"))): e for e in _episode_entries(manifest)}
    query_ids, store_ids = split_by_init_state(episodes, base.query_fraction, base.seed)
    if query_ids & store_ids:
        raise AssertionError("episode leakage detected")
    if ({e.init_state_id for e in episodes if e.episode_id in query_ids}
            & {e.init_state_id for e in episodes if e.episode_id in store_ids}):
        raise AssertionError("initial-state leakage detected")

    annotations = None
    if plan_annotations is not None:
        annotations = json.loads(Path(plan_annotations).read_text(encoding="utf-8"))
    meta, plan_check = {}, {"checked": annotations is not None, "mismatches": []}
    for ep in episodes:
        task = entries[ep.episode_id].get("task")
        if not task:
            raise ValueError(f"manifest episode {ep.episode_id} is missing task")
        plan = episode_plan(ep.arrays, config.skill_id_offset)
        if annotations is not None:
            expected = annotations.get(str(task).strip(), {}).get("all_classes")
            if expected != plan:
                plan_check["mismatches"].append({"episode_id": ep.episode_id, "recorded": plan,
                                                 "annotation": expected})
        closed = executed_gripper_closed(ep.arrays, config.gripper_action_index)
        labels = subtask_labels(plan, closed, config.gripper_debounce_steps)
        meta[ep.episode_id] = {"task": str(task).strip(), "plan": plan, "labels": labels,
                               "n_grasp_events": sum(1 for _, k in gripper_events(
                                   closed, config.gripper_debounce_steps) if k == "close")}

    feature_keys = sorted({key for ep in episodes for key in ep.arrays if key.startswith("feat__")})
    if not feature_keys:
        raise ValueError("no feat__* arrays found")
    methods = feature_keys + ["proprio"]
    q_tab, s_tab, q_feat, s_feat = None, None, {}, {}
    for key in methods:
        if any(key not in ep.arrays for ep in episodes):
            raise ValueError(f"{key} missing from one or more episodes; identical coverage is required")
        if len({_feature_matrix(ep, key).shape[1] for ep in episodes}) != 1:
            raise ValueError(f"{key} dimensions differ between episodes")
        qt = _row_table(episodes, query_ids, key, base.horizon, meta)
        st = _row_table(episodes, store_ids, key, base.horizon, meta)
        if q_tab is not None and (not np.array_equal(qt["episode_id"], q_tab["episode_id"])
                                  or not np.array_equal(qt["step"], q_tab["step"])
                                  or not np.array_equal(st["episode_id"], s_tab["episode_id"])
                                  or not np.array_equal(st["step"], s_tab["step"])):
            raise AssertionError("feature and proprio query/store rows differ")
        if q_tab is None:
            q_tab, s_tab = qt, st
        q_feat[key], s_feat[key] = qt["feature"], st["feature"]
    if not len(q_tab["step"]) or not len(s_tab["step"]):
        raise ValueError("query or store side has no valid future rows")
    if set(q_tab["episode_id"]) & set(s_tab["episode_id"]):
        raise AssertionError("episode leakage detected at row level")
    if set(q_tab["init_state_id"]) & set(s_tab["init_state_id"]):
        raise AssertionError("initial-state leakage detected at row level")

    masks_all = condition_masks(q_tab, s_tab)
    determined = q_tab["subtask"] != AMBIGUOUS
    nonempty = np.all(np.stack([m.any(axis=1) for m in masks_all.values()]), axis=0)
    keep = determined & nonempty
    if not keep.any():
        raise ValueError("no query rows with a determined subtask and candidates in every condition")
    q = {k: v[keep] for k, v in q_tab.items()}
    masks = {c: m[keep] for c, m in masks_all.items()}
    groups = q["init_state_id"]
    outcomes = _pairwise_outcomes(q["target"], s_tab["target"])
    retrieval = {key: _standardized_distances(q_feat[key][keep], s_feat[key]) for key in methods}

    k_values = base.k_values
    scores = {c: {m: np.empty((len(groups), len(k_values), 3)) for m in methods} for c in masks}
    random = {c: np.empty((len(groups), 3)) for c in masks}
    oracle = {c: np.empty((len(groups), len(k_values), 3)) for c in masks}
    candidates = {c: masks[c].sum(axis=1) for c in masks}
    for c, mask in masks.items():
        for i in range(len(groups)):
            idx = np.flatnonzero(mask[i])
            out = outcomes[i, idx]
            random[c][i] = out.mean(axis=0)
            for kk, k in enumerate(k_values):
                ke = min(k, len(idx))
                oracle[c][i, kk] = np.sort(out, axis=0)[:ke].mean(axis=0)
                for m in methods:
                    w, ke = _knn_weights(retrieval[m][i, idx], k)
                    scores[c][m][i, kk] = w @ out / ke

    # Item 1 uncertainty: one Bonferroni family over conditions x features x k x components,
    # reusing evaluate._compare_to_proprio by stacking conditions along its k axis.
    labels = tuple(f"{c}|k={k}" for c in masks for k in k_values)
    pseudo = replace(base, k_values=labels)
    query_rows = [{"init_state_id": g} for g in groups]
    stacked = {m: np.concatenate([scores[c][m] for c in masks], axis=1) for m in methods}
    vs_proprio = _compare_to_proprio(query_rows, stacked, pseudo)
    random_stacked = np.concatenate([np.repeat(random[c][:, None], len(k_values), axis=1) for c in masks], axis=1)
    vs_random = _compare_to_proprio(query_rows, dict({("proprio_knn" if m == "proprio" else m): v for m, v in stacked.items()}, proprio=random_stacked),
                                    replace(pseudo, seed=base.seed + 10))

    def unstack(comparison):
        out = {}
        for name, value in comparison["comparisons"].items():
            for label, by_metric in value["by_k"].items():
                c, k = label.split("|k=")
                out.setdefault(c, {}).setdefault(name, {})[k] = by_metric
        for c, by_name in out.items():
            for name, by_k in by_name.items():
                clear = True
                for by_metric in by_k.values():
                    lower = [v["ci95_familywise"][0] if v["ci95_familywise"] else None for v in by_metric.values()]
                    clear &= (all(x is not None and x >= 0 for x in lower)
                              and any(x is not None and x > 0 for x in lower))
                by_k["decision"] = "clear_improvement" if clear else "inconclusive"
        return out

    item1 = {}
    for c in masks:
        item1[c] = {
            "description": CONDITIONS[c],
            "candidates_per_query": {"mean": float(candidates[c].mean()), "min": int(candidates[c].min())},
            "methods": {m: {str(k): dict(zip(METRICS, scores[c][m][:, kk].mean(axis=0).tolist()))
                            for kk, k in enumerate(k_values)} for m in methods},
            "random": dict(zip(METRICS, random[c].mean(axis=0).tolist())),
            "oracle_upper_bound": {str(k): dict(zip(METRICS, oracle[c][:, kk].mean(axis=0).tolist()))
                                   for kk, k in enumerate(k_values)},
        }
        # Share of the random-to-oracle gap closed (descriptive).
        item1[c]["gap_closed"] = {
            m: {str(k): {metric: float((random[c][:, j].mean() - scores[c][m][:, kk, j].mean())
                                       / max(random[c][:, j].mean() - oracle[c][:, kk, j].mean(), 1e-12))
                         for j, metric in enumerate(METRICS)} for kk, k in enumerate(k_values)}
            for m in methods}

    # Combined motion distance: each component scaled by its mean over all eligible pairs.
    scale = np.array([outcomes[..., j][masks["a_unconstrained"]].mean() for j in range(3)])
    combined = (outcomes / np.maximum(scale, 1e-12)).sum(axis=2)

    # Item 2: where unconstrained neighbours come from.
    category = _source_category(q, s_tab)
    rank_sets = {m: retrieval[m] for m in methods}
    rank_sets["oracle_motion_combined"] = combined
    item2 = {"k": list(k_values), "categories": list(SOURCE_CATEGORIES), "by_method": {}}
    frac, task_frac = {}, {}
    mask_a = masks["a_unconstrained"]
    # Same-task share is taken from task identity directly: store rows with an undetermined
    # subtask fall in `subtask_unknown` but still count as same task when the task matches.
    same_task_rows = q["task"][:, None] == s_tab["task"][None]
    for name, dist in rank_sets.items():
        frac[name] = np.zeros((len(groups), len(k_values), len(SOURCE_CATEGORIES)))
        task_frac[name] = np.zeros((len(groups), len(k_values)))
        for i in range(len(groups)):
            idx = np.flatnonzero(mask_a[i])
            for kk, k in enumerate(k_values):
                w, ke = _knn_weights(dist[i, idx], k)
                frac[name][i, kk] = np.bincount(category[i, idx], weights=w, minlength=5) / ke
                task_frac[name][i, kk] = float(w @ same_task_rows[i, idx]) / ke
    chance = np.zeros((len(groups), len(SOURCE_CATEGORIES)))
    task_chance = np.zeros(len(groups))
    for i in range(len(groups)):
        idx = np.flatnonzero(mask_a[i])
        chance[i] = np.bincount(category[i, idx], minlength=5) / len(idx)
        task_chance[i] = same_task_rows[i, idx].mean()
    for name in rank_sets:
        item2["by_method"][name] = {str(k): dict(zip(SOURCE_CATEGORIES, frac[name][:, kk].mean(axis=0).tolist()))
                                    for kk, k in enumerate(k_values)}
    item2["by_method"]["chance (candidate share)"] = dict(zip(SOURCE_CATEGORIES, chance.mean(axis=0).tolist()))
    item2["same_task_share"] = {**{name: {str(k): float(task_frac[name][:, kk].mean()) for kk, k in enumerate(k_values)}
                                   for name in rank_sets},
                                "chance (candidate share)": float(task_chance.mean())}
    # Same-task / same-subtask share minus chance, with one Bonferroni family.
    names2 = methods + ["oracle_motion_combined"]
    cols, col_labels = [], []
    for name in names2:
        for kk, k in enumerate(k_values):
            same_task = task_frac[name][:, kk] - task_chance
            same_sub = frac[name][:, kk, 0] + frac[name][:, kk, 2] - chance[:, 0] - chance[:, 2]
            cols += [same_task, same_sub]
            col_labels += [(name, str(k), "same_task_minus_chance"), (name, str(k), "same_subtask_minus_chance")]
            if name != "oracle_motion_combined":
                # Task clustering beyond what motion similarity itself implies.
                cols.append(task_frac[name][:, kk] - task_frac["oracle_motion_combined"][:, kk])
                col_labels.append((name, str(k), "same_task_minus_motion_oracle"))
    boot2 = _cluster_bootstrap(np.stack(cols, axis=1), groups, base.seed + 3, base.bootstrap_samples)
    item2["excess_over_chance"] = [
        {"method": n, "k": k, "quantity": qn, "estimate": e, "ci95_familywise": ci}
        for (n, k, qn), e, ci in zip(col_labels, boot2["estimates"], boot2["intervals"])]
    item2["bootstrap"] = {k: boot2[k] for k in ("n_groups", "n_comparisons", "bootstrap_samples")}

    # Item 3: recall of cross-task motion-near store rows.
    near_k = config.near_oracle_k
    pooled = np.concatenate([np.sort(combined[i, mask_a[i]])[:near_k] for i in range(len(groups))])
    item3 = {"definition": (f"motion-near = combined motion distance <= tau, tau = quantile of the pooled "
                            f"distances to each query's {near_k} oracle (combined-motion) nearest "
                            "unconstrained store rows; relevant set = motion-near rows of other tasks"),
             "component_scale": dict(zip(METRICS, scale.tolist())), "thresholds": {}}
    rk = config.recall_k_values
    for quantile in config.near_quantiles:
        tau = float(np.quantile(pooled, quantile))
        relevant = masks["b_other_task"] & (combined <= tau)
        n_rel = relevant.sum(axis=1)
        has = n_rel > 0
        entry = {"tau": tau, "queries_with_relevant": int(has.sum()), "queries_total": int(len(groups)),
                 "relevant_per_query_mean": float(n_rel[has].mean()) if has.any() else 0.0, "pools": {}}
        if not has.any():
            item3["thresholds"][str(quantile)] = entry
            continue
        boot_cols, boot_labels = [], []
        for pool_name, pool_mask in (("other_task_pool", masks["b_other_task"]),
                                     ("unconstrained_pool", mask_a)):
            res = {}
            per = {name: np.zeros((has.sum(), len(rk), 2)) for name in list(rank_sets) + ["random (chance)"]}
            for qi, i in enumerate(np.flatnonzero(has)):
                idx = np.flatnonzero(pool_mask[i])
                rel = relevant[i, idx].astype(float)
                for kk, k in enumerate(rk):
                    for name, dist in rank_sets.items():
                        w, ke = _knn_weights(dist[i, idx], k)
                        hits = float(w @ rel)
                        per[name][qi, kk] = (hits / n_rel[i], hits / ke)
                    ke = min(k, len(idx))
                    per["random (chance)"][qi, kk] = (ke / len(idx), rel.sum() / len(idx))
            for name, arr in per.items():
                res[name] = {f"k={k}": {"recall": float(arr[:, kk, 0].mean()), "precision": float(arr[:, kk, 1].mean())}
                             for kk, k in enumerate(rk)}
            entry["pools"][pool_name] = res
            for name in feature_keys:
                for kk, k in enumerate(rk):
                    for jj, measure in enumerate(("recall", "precision")):
                        boot_cols.append(per[name][:, kk, jj] - per["proprio"][:, kk, jj])
                        boot_labels.append((pool_name, name, f"k={k}", f"{measure}_minus_proprio"))
        boot3 = _cluster_bootstrap(np.stack(boot_cols, axis=1), groups[has], base.seed + 4 + int(quantile * 100),
                                   base.bootstrap_samples)
        entry["feature_minus_proprio"] = [
            {"pool": p, "method": n, "k": k, "quantity": qn, "estimate": e, "ci95_familywise": ci}
            for (p, n, k, qn), e, ci in zip(boot_labels, boot3["estimates"], boot3["intervals"])]
        entry["bootstrap"] = {k: boot3[k] for k in ("n_groups", "n_comparisons", "bootstrap_samples")}
        item3["thresholds"][str(quantile)] = entry

    # Data description.
    sub_counts = {SKILL_NAMES.get(int(k), str(k)) if k != AMBIGUOUS else "ambiguous": int(v)
                  for k, v in zip(*np.unique(q_tab["subtask"], return_counts=True))}
    store_sub = {SKILL_NAMES.get(int(k), str(k)) if k != AMBIGUOUS else "ambiguous": int(v)
                 for k, v in zip(*np.unique(s_tab["subtask"], return_counts=True))}
    grasp_mismatch = [e for e, v in meta.items()
                      if v["n_grasp_events"] != sum(1 for x in v["plan"] if x == PICK)]
    return {
        "config": asdict(config),
        "split": {"query_episode_ids": sorted(query_ids), "store_episode_ids": sorted(store_ids),
                  "query_init_state_ids": sorted({e.init_state_id for e in episodes if e.episode_id in query_ids}),
                  "store_init_state_ids": sorted({e.init_state_id for e in episodes if e.episode_id in store_ids})},
        "data_counts": {
            "query_windows_total": int(len(q_tab["step"])), "query_windows_used": int(keep.sum()),
            "query_windows_dropped_ambiguous_subtask": int((~determined).sum()),
            "query_windows_dropped_empty_condition": int((determined & ~nonempty).sum()),
            "store_windows": int(len(s_tab["step"])),
            "query_init_state_groups": int(len(np.unique(groups))),
            "query_tasks": int(len(np.unique(q["task"]))),
            "query_subtask_counts_all_windows": sub_counts, "store_subtask_counts": store_sub,
            "episodes_with_grasp_count_not_equal_picks": len(grasp_mismatch),
        },
        "subtask_labelling": {
            "plan_check_against_annotations": plan_check,
            "episodes": {e: {"task": v["task"], "plan": v["plan"], "n_grasp_events": v["n_grasp_events"],
                             "ambiguous_steps": int((v["labels"] == AMBIGUOUS).sum()),
                             "steps": int(len(v["labels"]))} for e, v in meta.items()},
        },
        "item1_constrained_retrieval": item1,
        "item1_uncertainty_vs_proprio": {"by_condition": unstack(vs_proprio),
                                         **{k: vs_proprio[k] for k in ("method", "n_query_init_states",
                                                                       "bootstrap_samples", "n_comparisons", "scope")}},
        "item1_uncertainty_vs_random": {"by_condition": unstack(vs_random),
                                        **{k: vs_random[k] for k in ("n_query_init_states", "bootstrap_samples",
                                                                     "n_comparisons")}},
        "item2_neighbor_sources": item2,
        "item3_cross_task_motion_recall": item3,
        "manifest_summary": {k: manifest.get(k) for k in ("suite", "checkpoint", "hidden_features") if k in manifest},
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _ci(ci) -> str:
    return f"[{ci[0]:.3g}, {ci[1]:.3g}]" if ci else "n/a"


def write_report(results: dict[str, Any], path: str | Path, run_dir: str | Path | None = None,
                 interpretation: str | None = None) -> None:
    counts = results["data_counts"]
    cfg = results["config"]["base"]
    k_values = [str(k) for k in cfg["k_values"]]
    lines = ["# G2 cross-task / cross-subtask retrieval", ""]
    if run_dir:
        lines += [f"Recording: `{run_dir}`", ""]
    lines += [f"H={cfg['horizon']}, query_fraction={cfg['query_fraction']}, seed={cfg['seed']}. "
              "Split, leakage guards, motion target, kNN standardization and bootstrap are those of `evaluate.py`.", "",
              f"Query windows used: {counts['query_windows_used']} of {counts['query_windows_total']} "
              f"({counts['query_windows_dropped_ambiguous_subtask']} dropped: subtask undetermined; "
              f"{counts['query_windows_dropped_empty_condition']} dropped: an empty condition). "
              f"Store windows: {counts['store_windows']}. Query initial-state groups: {counts['query_init_state_groups']}; "
              f"query tasks: {counts['query_tasks']}.", "",
              f"Query subtask counts (all windows): {counts['query_subtask_counts_all_windows']}. "
              f"Store: {counts['store_subtask_counts']}. "
              f"Episodes whose debounced grasp count differs from the number of picks: "
              f"{counts['episodes_with_grasp_count_not_equal_picks']}.", ""]
    if interpretation:
        lines += ["## Summary", "", interpretation, ""]
    lines += ["## 1. Motion error of retrieved windows by constraint", "",
              "Lower is better. Units: m, rad, gripper units. `gap closed` = (random - method) / (random - oracle), "
              "position component.", ""]
    item1 = results["item1_constrained_retrieval"]
    for c, value in item1.items():
        lines += [f"### {c}: {value['description']}", "",
                  f"Candidates per query: mean {value['candidates_per_query']['mean']:.0f}, "
                  f"min {value['candidates_per_query']['min']}.", "",
                  "| Retrieval | k | Position | Rotation | Gripper | gap closed (pos) |", "|---|---:|---:|---:|---:|---:|"]
        for m, by_k in value["methods"].items():
            for k in k_values:
                v = by_k[k]
                lines.append(f"| {m} | {k} | {_fmt(v[METRICS[0]])} | {_fmt(v[METRICS[1]])} | {_fmt(v[METRICS[2]])} | "
                             f"{value['gap_closed'][m][k][METRICS[0]]:.2f} |")
        r = value["random"]
        lines.append(f"| random | any | {_fmt(r[METRICS[0]])} | {_fmt(r[METRICS[1]])} | {_fmt(r[METRICS[2]])} | 0 |")
        for k in k_values:
            o = value["oracle_upper_bound"][k]
            lines.append(f"| oracle (per metric) | {k} | {_fmt(o[METRICS[0]])} | {_fmt(o[METRICS[1]])} | "
                         f"{_fmt(o[METRICS[2]])} | 1 |")
        lines.append("")
    for title, key in (("Improvement over proprio", "item1_uncertainty_vs_proprio"),
                       ("Improvement over random", "item1_uncertainty_vs_random")):
        unc = results[key]
        lines += [f"### {title}", "",
                  f"Positive = smaller error than the baseline. 95% familywise Bonferroni intervals over "
                  f"{unc['n_comparisons']} comparisons (all conditions, features, k, components); whole query "
                  f"initial-state groups resampled ({unc['n_query_init_states']} groups, "
                  f"{unc['bootstrap_samples']} resamples).", "",
                  "| Condition | Feature | k | Position | Rotation | Gripper | Decision |", "|---|---|---:|---|---|---|---|"]
        for c, by_name in unc["by_condition"].items():
            for name, by_k in by_name.items():
                for k in k_values:
                    cells = [f"{by_k[k][m]['improvement']:.3g} {_ci(by_k[k][m]['ci95_familywise'])}" for m in METRICS]
                    lines.append(f"| {c} | {name} | {k} | " + " | ".join(cells) + f" | {by_k['decision']} |")
        lines.append("")
    item2 = results["item2_neighbor_sources"]
    lines += ["## 2. Where unconstrained neighbours come from (condition a)", "",
              "Share of the k retrieved windows by category. `chance` is the candidate share. "
              "`oracle_motion_combined` ranks by the combined motion distance (section 3).", "",
              "| Method | k | " + " | ".join(SOURCE_CATEGORIES) + " |", "|---|---:|" + "---:|" * len(SOURCE_CATEGORIES)]
    for name, by_k in item2["by_method"].items():
        if name.startswith("chance"):
            lines.append(f"| chance | - | " + " | ".join(f"{by_k[c]:.3f}" for c in SOURCE_CATEGORIES) + " |")
            continue
        for k, v in by_k.items():
            lines.append(f"| {name} | {k} | " + " | ".join(f"{v[c]:.3f}" for c in SOURCE_CATEGORIES) + " |")
    if "same_task_share" in item2:
        shares = item2["same_task_share"]
        lines += ["", "Same-task share (task identity, including store rows with an undetermined subtask): "
                  + "; ".join(f"{n} " + ", ".join(f"k={k} {v:.3f}" for k, v in by_k.items())
                              for n, by_k in shares.items() if isinstance(by_k, dict))
                  + f"; chance {shares['chance (candidate share)']:.3f}."]
    lines += ["", f"Excess over chance (95% familywise intervals, {item2['bootstrap']['n_comparisons']} comparisons):", "",
              "| Method | k | Quantity | Estimate | Interval |", "|---|---:|---|---:|---|"]
    for e in item2["excess_over_chance"]:
        lines.append(f"| {e['method']} | {e['k']} | {e['quantity']} | {e['estimate']:.3f} | {_ci(e['ci95_familywise'])} |")
    item3 = results["item3_cross_task_motion_recall"]
    lines += ["", "## 3. Recall of motion-near windows from other tasks", "", item3["definition"] + ".",
              f" Component scales (mean over eligible pairs): {', '.join(f'{k}={v:.4g}' for k, v in item3['component_scale'].items())}.", ""]
    for quantile, entry in item3["thresholds"].items():
        lines += [f"### tau = quantile {quantile} = {entry['tau']:.3f}", "",
                  f"Queries with at least one relevant window: {entry['queries_with_relevant']} / "
                  f"{entry['queries_total']}; relevant windows per such query: {entry['relevant_per_query_mean']:.1f}.", ""]
        for pool, res in entry["pools"].items():
            ks = list(next(iter(res.values())).keys())
            lines += [f"Ranking within `{pool}` (recall / precision):", "",
                      "| Method | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
            for name, by_k in res.items():
                lines.append(f"| {name} | " + " | ".join(f"{by_k[k]['recall']:.3f} / {by_k[k]['precision']:.3f}"
                                                        for k in ks) + " |")
            lines.append("")
        if "feature_minus_proprio" in entry:
            lines += [f"Feature minus proprio (95% familywise intervals, {entry['bootstrap']['n_comparisons']} comparisons):", "",
                      "| Pool | Feature | k | Quantity | Estimate | Interval |", "|---|---|---:|---|---:|---|"]
            for e in entry["feature_minus_proprio"]:
                lines.append(f"| {e['pool']} | {e['method']} | {e['k']} | {e['quantity']} | {e['estimate']:.3f} | "
                             f"{_ci(e['ci95_familywise'])} |")
            lines.append("")
    lines += ["## Caveats", "",
              "- The policy receives the whole plan (constant `skills` in every inference); the current subtask is "
              "inferred from executed gripper commands aligned to the plan (rule in `SCHEMA.md`). It is a proxy for a "
              "high-level subtask signal, not a recorded label.",
              "- Subtask skill IDs are coarse (pick, place, push, open/close, turn); most windows are pick or place, so "
              "condition c mostly pairs pick with place.",
              "- Many of the 40 tasks share the same pick-place structure and scenes (for example all libero_spatial "
              "tasks), so `different task` does not mean different motion repertoire.",
              "- Intervals are conditional on this store and split; overlapping windows are not independent; "
              "families are Bonferroni-corrected separately per table.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--plan-annotations", type=Path,
                        help="instruct2plan JSON to check recorded plans against (optional)")
    parser.add_argument("--output-dir", type=Path, help="new, non-existing output directory")
    args = parser.parse_args()
    config = CrossTaskConfig(base=EvaluationConfig(args.horizon, args.query_fraction, args.seed,
                                                   bootstrap_samples=args.bootstrap_samples))
    result = evaluate_crosstask(args.run_dir, config, args.plan_annotations)
    if args.output_dir:
        out = args.output_dir
    else:
        from datetime import datetime, timezone

        out = Path("/work/.skillnet-state/outputs/memory-pilot") / (
            "g2-crosstask-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, out / "report.md", run_dir=args.run_dir)


if __name__ == "__main__":
    main()
