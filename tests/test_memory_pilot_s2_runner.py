from __future__ import annotations

import sys
from pathlib import Path
import json

EXAMPLES = Path(__file__).resolve().parents[1] / "skill_moe" / "skillnet" / "examples"
ANCHOR_FLOW = EXAMPLES / "memory_pilot" / "anchor_flow"
for path in (str(EXAMPLES), str(ANCHOR_FLOW)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_series_a import make_cases
from run_case import completed_trial_indices
from run_series_a import free_port


def test_minimal_gate_runner_matrix_is_92_cases_and_paired():
    cases = make_cases(20, minimal=True, checkpoint="s2-snapshot", seed=19)
    assert len(cases) == 92
    assert len(cases) * 20 == 1840
    assert all(case["query_text_source"] == "instruction_only" for case in cases)
    assert all(case["checkpoint"] == "s2-snapshot" and case["seed"] == 19 for case in cases)
    by_task = {}
    for case in cases:
        by_task.setdefault(case["task_key"], set()).add(case["condition"])
    assert all(len(conditions) == 2 for task, conditions in by_task.items() if task.startswith("libero90_"))
    assert all(len(conditions) == 4 for task, conditions in by_task.items() if not task.startswith("libero90_"))


def test_full_matrix_includes_rts_optimus_and_t0_controls():
    cases = make_cases(20, minimal=False, query_text_source="pi05_generated_subtask")
    assert len(cases) == 182
    conditions = {case["condition"] for case in cases if case["suite"] == "libero_10"}
    assert {"rts_raw_chunk_entropy", "optimus_gaussian_entropy", "anchor_absolute_similarity",
            "anchor_fixed_t0", "anchor_t0_shuffled"}.issubset(conditions)
    assert all(case["query_text_source"] == "pi05_generated_subtask" for case in cases)


def test_free_port_is_bound_to_loopback_and_nonzero():
    port = free_port()
    assert 0 < port < 65536


def test_resume_retries_errors_and_skips_latest_complete_trials(tmp_path):
    rows = tmp_path / "trials.jsonl"
    rows.write_text("\n".join(json.dumps(row) for row in (
        {"trial_index": 0, "status": "error"},
        {"trial_index": 1, "status": "complete"},
        {"trial_index": 1, "status": "error"},
        {"trial_index": 0, "status": "complete"},
    )) + "\n")
    assert completed_trial_indices(rows) == {0}
