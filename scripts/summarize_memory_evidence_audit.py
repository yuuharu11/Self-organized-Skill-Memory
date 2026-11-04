"""Aggregate restored-model diagnostics with episode/task-aware uncertainty."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_memory_slide_evidence as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.output / "summary.json").read_text())
    if not data["source_unchanged"]:
        raise ValueError("source changed during diagnostic")
    arrays = np.load(args.output / "diagnostic_arrays.npz", allow_pickle=False)
    aggregate = {"source": str(args.output), "source_unchanged": True,
                 "elapsed_seconds": data["elapsed_seconds"], "protocol": data["protocol"], "splits": {}}
    comparisons = {
        "memory_minus_normal": ("memory_adaptive", "normal10"),
        "memory_pair_minus_two_normal_pair": ("oracle_normal_memory", "oracle_two_normal_matched_n"),
        "unit_minus_equal_entry_raw": ("unit_fixed4", "raw_equal_entries_fixed4"),
    }
    for split in ("A", "B"):
        runs = [r for r in data["runs"] if r["split"] == split]
        first = f"{split}_s0"
        meta = {k: arrays[f"{first}|sample_{k}"] for k in ("task", "episode", "frame")}
        mean_arrays = {}
        for method in runs[0]["mean_errors"]:
            values = []
            for seed in (0, 1, 2):
                stem = f"{split}_s{seed}"
                for k in meta:
                    if not np.array_equal(meta[k], arrays[f"{stem}|sample_{k}"]):
                        raise ValueError("sampled observations differ across model seeds")
                value = arrays[f"{stem}|{method}|per_noise_mse"]
                if value.shape != (8, len(meta["task"])) or not np.isfinite(value).all():
                    raise ValueError("invalid per-noise errors")
                values.append(value.mean(0))
            mean_arrays[method] = np.mean(values, axis=0)
        result = {
            "sampled_episodes": len(np.unique(meta["episode"])), "sampled_windows": len(meta["task"]),
            "sampled_tasks": len(np.unique(meta["task"])),
            "sample_means": {k: float(v.mean()) for k, v in mean_arrays.items()},
            "paired_differences": {name: audit.hierarchical_interval(mean_arrays[a]-mean_arrays[b], meta, 1000, 7183)
                                   for name, (a,b) in comparisons.items()},
            "full_retrieval_hits": {k: float(np.mean([r["retrieval"]["hits"][k] for r in runs])) for k in ("1","5","10","20")},
            "full_functional_coverage": float(np.mean([r["retrieval"]["functional_coverage"] for r in runs])),
            "full_functional_recall_on_covered": {k: float(np.mean([r["retrieval"]["functional_recall_on_covered"][k] for r in runs])) for k in ("1","5","10","20")},
            "unit_entries_by_seed": [r["unit_entries"] for r in runs],
            "random_retrieval_expected_hits": {k: float(np.mean([min(int(k),r["unit_entries"])/r["unit_entries"] for r in runs])) for k in ("1","5","10","20")},
            "minimum_retrieval_replay": min(r["retrieval"]["top1_replay_agreement"] for r in runs),
            "maximum_fp_replay_mse_difference": max(max(r["fp_replay_max_abs_error_difference"].values()) for r in runs),
            "pair_mean_nfe": float(np.mean([r["candidate_nfe_per_row"] for r in runs])),
            "grid": {str(t): {str(n): float(np.mean([r["mse_by_n"][str(n)] for r in data["grid"] if r["split"]==split and r["t0"]==t])) for n in (1,2,3,5,10)} for t in (.2,.4,.6)},
            "validation_selected_fixed": {gen: {"t0_by_seed": [r["selected_t0"] for r in data["fixed_validation"] if r["split"]==split and r["generator"]==gen],
                **{k: float(np.mean([r[k] for r in data["fixed_validation"] if r["split"]==split and r["generator"]==gen])) for k in ("fixed_mse","adaptive_mse")}} for gen in ("A","N")},
            "dimensions": {model: {k: float(np.mean([r[model][k] for r in data["dimensions"] if r["split"]==split])) for k in data["dimensions"][0][model]} for model in ("normal","memory")},
            "per_task_sample_means": {str(t): {k: float(v[meta["task"]==t].mean()) for k,v in mean_arrays.items()} for t in np.unique(meta["task"])},
            "ci_scope": "task then episode bootstrap; 3 model seeds and 8 noises averaged first; conditional on these models, one window per episode",
        }
        aggregate["splits"][split] = result
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({s: r["paired_differences"] for s,r in aggregate["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
