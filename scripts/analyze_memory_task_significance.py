"""Exploratory, task-level paired sign tests for the memory presentation table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

COHORTS = ("B_D1_all", "B_D1_goal_only", "B_D3_all")
METHODS = ("N:noise:n10", "N:unit:cal_sim", "A:unit:cal_sim")


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    result = np.empty_like(p)
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    rows = list(csv.DictReader(source.decode().splitlines()))
    results = []
    for cohort in COHORTS:
        group = [r for r in rows if r["cohort"] == cohort]
        tasks = sorted({int(r["task"]) for r in group})
        if len(tasks) != (20 if cohort == "B_D1_all" else 10):
            raise ValueError("unexpected task count")
        values = {}
        for method in METHODS:
            means = []
            for task in tasks:
                selected = [r for r in group if r["method"] == method and int(r["task"]) == task]
                if sorted(int(r["seed"]) for r in selected) != [0, 1, 2]:
                    raise ValueError("missing or duplicate seed/task/method")
                means.append(np.mean([float(r["mse"]) for r in selected]))
            values[method] = np.asarray(means)
        for baseline, candidate in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
            delta = values[candidate] - values[baseline]
            if not np.isfinite(delta).all():
                raise ValueError("nonfinite task differences")
            wins, losses = int((delta < 0).sum()), int((delta > 0).sum())
            result = {"cohort": cohort, "baseline": baseline, "candidate": candidate,
                      "tasks": len(tasks), "improved_tasks": wins, "worsened_tasks": losses,
                      "tied_tasks": int((delta == 0).sum()), "task_equal_mean_difference": float(delta.mean()),
                      "p_two_sided": float(binomtest(wins, wins+losses, .5).pvalue) if wins+losses else 1.,
                      "per_task_difference": dict(zip(map(str, tasks), map(float, delta)))}
            results.append(result)
    for result, corrected in zip(results, holm([r["p_two_sided"] for r in results])):
        result["p_holm_9"] = float(corrected)
    output = {"input": str(args.input), "input_sha256": hashlib.sha256(source).hexdigest(),
              "test": "two-sided exact paired sign test on task-level MSE differences",
              "null": "independent task differences have probability 0.5 of being negative, conditional on nonzero difference",
              "aggregation": "within each task, window-average MSE per model seed; average 3 seeds; one observation per task",
              "multiplicity": "Holm correction over these 9 comparisons; goal subset overlaps D1_all",
              "scope": "post-hoc exploratory analysis; conditional on existing trained models; does not test all training-seed uncertainty or closed-loop success",
              "results": results}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps([{k:r[k] for k in ("cohort","candidate","baseline","tasks","improved_tasks","p_two_sided","p_holm_9")} for r in results], indent=2))


if __name__ == "__main__":
    main()
