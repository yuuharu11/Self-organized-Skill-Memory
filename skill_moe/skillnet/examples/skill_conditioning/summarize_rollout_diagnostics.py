"""Summarize a frozen, canonical JSONL result log without running a policy.

Only task/initial-state pairs present in every requested condition are compared.
An administrative interruption is missing data, never a policy failure. Technical
errors explicitly recorded in the canonical log count as failures. Stage events
are descriptive observations, not causal failure labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

CONDITIONS = ("L0", "L1", "L2", "L3", "L4", "L3-shuffled")
PAIR_FIELDS = ("initial_state_sha256", "environment_seed", "trial_seed",
               "max_steps", "wait_steps", "horizon", "replan_steps")


def paired_summary(pairs, *, resamples=5000, seed=20261102):
    """pairs: (task_id, success_A, success_B); hierarchical paired bootstrap."""
    by_task = defaultdict(list)
    counts = Counter()
    for task, a, b in pairs:
        by_task[task].append(int(a) - int(b))
        counts[f"{int(a)}{int(b)}"] += 1
    tasks = sorted(by_task)
    if not tasks:
        return {"pairs": 0}
    task_deltas = {str(t): sum(by_task[t]) / len(by_task[t]) for t in tasks}
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        values = []
        for t in rng.choices(tasks, k=len(tasks)):
            samples = rng.choices(by_task[t], k=len(by_task[t]))
            values.append(sum(samples) / len(samples))
        draws.append(sum(values) / len(values))
    draws.sort()
    discordant = counts["10"] + counts["01"]
    p = min(1.0, 2 * sum(math.comb(discordant, k)
                        for k in range(min(counts["10"], counts["01"]) + 1)) / 2**discordant)
    return {
        "pairs": len(pairs), "both_success": counts["11"],
        "a_only": counts["10"], "b_only": counts["01"],
        "both_failure": counts["00"],
        "pair_weighted_delta": (counts["10"] - counts["01"]) / len(pairs),
        "task_equal_delta": sum(task_deltas.values()) / len(tasks),
        "task_equal_hierarchical_bootstrap_95": [
            draws[int(.025 * (resamples - 1))], draws[int(.975 * (resamples - 1))]],
        "exact_mcnemar_p_exploratory": p, "task_deltas": task_deltas,
    }


def stage_summary(rows):
    """Use official goal events; never require a particular action order."""
    counts = Counter()
    for row in rows:
        if row["status"] != "complete":
            counts["technical_errors"] += 1
            continue
        if row["success"]:
            counts["successes"] += 1
            continue
        counts["failures"] += 1
        goals = [s for s in row.get("stages", []) if s["kind"] == "goal"]
        if not goals:
            counts["stage_data_missing"] += 1
        elif all(s["ever"] for s in goals):
            counts["all_goals_ever_achieved_but_final_failure"] += 1
        elif any(s["ever"] for s in goals):
            counts["some_goals_ever_achieved"] += 1
        else:
            counts["no_goal_ever_achieved"] += 1
        # A grasp proxy can cease after a successful release; do not count that
        # as goal regression. Only official goals enter the preceding groups.
        counts["failed_with_goal_lost_at_end"] += any(s["ever"] and not s["final"] for s in goals)
    return dict(sorted(counts.items()))


def summarize(rows, *, conditions=CONDITIONS, excluded_tasks=(), resamples=5000, seed=20261102):
    if resamples < 1 or len(set(conditions)) != len(conditions) or not conditions:
        raise ValueError("Require positive resamples and distinct conditions")
    by_condition = {c: {} for c in conditions}
    ignored = Counter()
    for row in rows:
        c = row["condition"]
        if c not in by_condition or row["task_id"] in excluded_tasks:
            continue
        if row["status"] not in ("complete", "error"):
            ignored[row["status"]] += 1
            continue
        if not isinstance(row["success"], bool) or (row["status"] == "error" and row["success"]):
            raise ValueError("Inconsistent success/status")
        key = (row["task_id"], row["trial_index"])
        if key in by_condition[c]:
            raise ValueError(f"Duplicate canonical result: {c} {key}")
        by_condition[c][key] = row
    cohort = sorted(set.intersection(*(set(v) for v in by_condition.values())))
    for key in cohort:
        for field in PAIR_FIELDS:
            values = {json.dumps(by_condition[c][key].get(field), sort_keys=True) for c in conditions}
            if len(values) != 1:
                raise ValueError(f"Mismatched paired {field}: {key}")
    protocols = {r.get("protocol") for v in by_condition.values() for r in v.values()}
    if len(protocols) > 1:
        raise ValueError("Mixed evaluation protocols")
    result = {
        "common_pairs": len(cohort), "ignored_statuses": dict(ignored),
        "pairs_per_task": dict(sorted(Counter(str(t) for t, _ in cohort).items())),
        "conditions": {}, "comparisons": {},
        "uncertainty": {"method": "resample tasks, then paired initial states within each sampled task",
                        "resamples": resamples, "seed": seed,
                        "scope": "exploratory; few tasks; no training-seed uncertainty or final-test claim"},
    }
    for c, records in by_condition.items():
        common_rows = [records[k] for k in cohort]
        result["conditions"][c] = {
            "available": len(records), "successes": sum(r["success"] for r in common_rows),
            "stage_observations": stage_summary(common_rows),
            "by_task": {str(t): {"trials": sum(k[0] == t for k in cohort),
                                  "successes": sum(records[k]["success"] for k in cohort if k[0] == t)}
                        for t in sorted({k[0] for k in cohort})},
        }
    for a, b in itertools.combinations(conditions, 2):
        pairs = [(t, by_condition[a][t, i]["success"], by_condition[b][t, i]["success"]) for t, i in cohort]
        result["comparisons"][f"{a} - {b}"] = paired_summary(pairs, resamples=resamples, seed=seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Frozen canonical JSONL; one final result per task/init/condition")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--exclude-tasks", type=int, nargs="*", default=[], help="Zero-based task IDs; sensitivity only")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--output", type=Path, help="New JSON output; existing files are never overwritten")
    args = parser.parse_args()
    data = args.results.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    result = summarize(rows, conditions=args.conditions, excluded_tasks=args.exclude_tasks, resamples=args.resamples)
    result["source"] = {"file": args.results.name, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
