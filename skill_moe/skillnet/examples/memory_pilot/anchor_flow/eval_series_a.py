#!/usr/bin/env python3
"""Create/run the S2 series-A rollout matrix through an existing LIBERO runner.

An S2-aware rollout command is required to provide sequential memory snapshots
and the requested inference conditions. The command is invoked once per task and
condition with those values in environment variables; by default this script
only writes the protocol matrix and never starts a simulator or policy server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time


RETENTION_CONDITIONS = ("anchor_entropy_current_memory", "anchor_disabled_t0_1")
S2_GATE_NEW_TASK_CONDITIONS = (
    "anchor_entropy_current_memory", "anchor_disabled_t0_1",
    "anchor_memory_shuffled", "anchor_before_task_add",
)
FULL_NEW_TASK_CONDITIONS = S2_GATE_NEW_TASK_CONDITIONS + (
    "rts_raw_chunk_entropy", "optimus_gaussian_entropy", "anchor_absolute_similarity",
    "anchor_fixed_t0", "anchor_t0_shuffled",
)


def make_cases(trials: int, *, minimal: bool = False, checkpoint: str | None = None, seed: int = 7,
               query_text_source: str = "instruction_only"):
    cases = []
    for task_id in range(10):
        for condition in RETENTION_CONDITIONS:
            cases.append({"suite": "libero_90", "task_id": task_id,
                          "task_key": f"libero90_retention_{task_id:02d}", "stage": -1,
                          "condition": condition, "trials": trials,
                          "minimum_trials": 20 if trials >= 20 else trials,
                          "memory_snapshot": "initial_libero90", "checkpoint": checkpoint, "seed": seed,
                          "query_text_source": query_text_source})
    sequence = [{"suite": "libero_10", "task_id": i, "task_key": f"libero10_{i:02d}"} for i in range(10)]
    sequence += [{"suite": "libero_skill_obj", "task_id": i, "task_key": f"skill_obj_{i:02d}"}
                 for i in range(1, 9)]
    conditions = S2_GATE_NEW_TASK_CONDITIONS if minimal else FULL_NEW_TASK_CONDITIONS
    for stage, task in enumerate(sequence):
        for condition in conditions:
            before_snapshot = "initial_libero90" if stage == 0 else f"after_{stage - 1:02d}_adds"
            cases.append({**task, "stage": stage, "condition": condition,
                          "trials": trials, "minimum_trials": 20 if trials >= 20 else trials,
                          "checkpoint": checkpoint, "seed": seed, "query_text_source": query_text_source,
                          "memory_snapshot": f"after_{stage:02d}_adds" if condition != "anchor_before_task_add"
                          else before_snapshot})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--runner", help="S2-aware command; supports env vars listed in DESIGN.md")
    parser.add_argument("--minimal", action="store_true",
                        help="only emit the four S2-a/b controls (92 cases at the default task set)")
    parser.add_argument("--checkpoint", help="the one shared fine-tuned S2 checkpoint used by all cases")
    parser.add_argument("--seed", type=int, default=7, help="paired policy-noise and initial-state seed base")
    parser.add_argument("--query-text-source", choices=("instruction_only", "pi05_generated_subtask"),
                        default="instruction_only")
    parser.add_argument("--resume", action="store_true", help="resume existing case JSONL outputs")
    parser.add_argument("--execute", action="store_true", help="actually run all task/condition cases")
    args = parser.parse_args()
    if args.trials < 20:
        raise ValueError("series A requires at least 20 trials per task and condition")
    if args.execute and not args.runner:
        raise ValueError("--execute requires --runner")
    out = Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError(f"{out} exists; pass --resume to continue")
    out.mkdir(parents=True, exist_ok=True)
    if args.execute and not args.checkpoint:
        raise ValueError("--execute requires --checkpoint so every control uses the same fine-tuned model")
    cases = make_cases(args.trials, minimal=args.minimal, checkpoint=args.checkpoint, seed=args.seed,
                       query_text_source=args.query_text_source)
    cases_path = out / "series_a_cases.json"
    if cases_path.exists() and json.loads(cases_path.read_text()) != cases:
        raise ValueError("existing case plan differs; refusing to resume with changed settings")
    if not cases_path.exists():
        cases_path.write_text(json.dumps(cases, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"cases": len(cases), "episodes": len(cases) * args.trials,
                          "series_a_tasks": 18, "retention_tasks": 10,
                          "series_a_conditions": len(S2_GATE_NEW_TASK_CONDITIONS if args.minimal else FULL_NEW_TASK_CONDITIONS),
                          "minimal_gate_matrix": args.minimal, "paired_seed": args.seed,
                          "checkpoint": args.checkpoint,
                          "trials_per_task_condition": args.trials,
                          "plan": str(out / "series_a_cases.json"), "executed": False}, indent=2))
        return
    base_env = os.environ.copy()
    runner = shlex.split(args.runner)
    results_path = out / "results.json"
    results_by_key = {}
    if results_path.exists():
        for row in json.loads(results_path.read_text()):
            results_by_key[(row["task_key"], row["condition"])] = row
    results = [results_by_key[key] for key in results_by_key]
    for index, case in enumerate(cases):
        case_key = (case["task_key"], case["condition"])
        previous = results_by_key.get(case_key)
        if previous and previous.get("returncode") == 0:
            continue
        env = base_env | {"S2_TASK_SUITE": case["suite"], "S2_TASK_ID": str(case["task_id"]),
                          "S2_TASK_KEY": case["task_key"], "S2_CONDITION": case["condition"],
                          "S2_MEMORY_SNAPSHOT": case["memory_snapshot"],
                          "S2_NUM_TRIALS": str(args.trials), "S2_SEED": str(args.seed),
                          "S2_TRIAL_SEED_BASE": str(args.seed),
                          "S2_CHECKPOINT": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
                          "S2_QUERY_TEXT_SOURCE": case["query_text_source"],
                          "S2_OUTPUT_DIR": str(out / case["task_key"] / case["condition"])}
        Path(env["S2_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=args.resume)
        started = time.perf_counter()
        completed = subprocess.run(runner, env=env, check=False)
        record = {**case, "returncode": completed.returncode,
                  "elapsed_seconds": time.perf_counter() - started}
        results_by_key[case_key] = record
        results = list(results_by_key.values())
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out, delete=False) as stream:
            stream.write(json.dumps(results, indent=2) + "\n")
            temp_path = Path(stream.name)
        temp_path.replace(results_path)
        if completed.returncode:
            raise SystemExit(f"S2 rollout runner failed on case {index}: {case}")
    print(json.dumps({"cases": len(results), "executed": True, "results": str(out / "results.json")}, indent=2))


if __name__ == "__main__":
    main()
