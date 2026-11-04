"""Protect paired diagnostics against missing-result and stage-label mistakes."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/skill_conditioning/summarize_rollout_diagnostics.py"
spec = importlib.util.spec_from_file_location("rollout_diagnostics", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def row(c, task=0, init=0, success=False, **extra):
    return dict(condition=c, task_id=task, trial_index=init, status="complete", success=success, **extra)


def test_missing_and_administratively_stopped_trials_are_not_failures():
    rows = [row("A", success=True), row("B"), row("A", init=1),
            {**row("B", init=1), "status": "stopped_by_user"}]
    result = diagnostics.summarize(rows, conditions=["A", "B"], resamples=100)
    assert result["common_pairs"] == 1
    assert result["comparisons"]["A - B"]["a_only"] == 1
    assert result["conditions"]["A"]["available"] == 2


def test_duplicate_results_and_mismatched_initial_states_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics.summarize([row("A"), row("A")], conditions=["A"])
    with pytest.raises(ValueError, match="initial_state"):
        diagnostics.summarize([row("A", initial_state_sha256="a"), row("B", initial_state_sha256="b")], conditions=["A", "B"])


def test_paired_counts_and_task_equal_weighting():
    pairs = [(0, True, False), (0, True, False), (0, True, False), (1, False, True)]
    result = diagnostics.paired_summary(pairs, resamples=100)
    assert result["a_only"] == 3 and result["b_only"] == 1
    assert result["pair_weighted_delta"] == .5
    assert result["task_equal_delta"] == 0
    assert result["exact_mcnemar_p_exploratory"] == .625
    assert diagnostics.paired_summary([(0, False, False)], resamples=100)["exact_mcnemar_p_exploratory"] == 1


def test_grasp_proxy_loss_and_goal_regression_are_distinct():
    stages = [dict(kind="pick", ever=True, final=False), dict(kind="goal", ever=True, final=True),
              dict(kind="goal", ever=False, final=False)]
    result = diagnostics.stage_summary([row("A", stages=stages)])
    assert result["some_goals_ever_achieved"] == 1
    assert result["failed_with_goal_lost_at_end"] == 0
    stages[-1].update(ever=True)
    result = diagnostics.stage_summary([row("A", stages=stages)])
    assert result["all_goals_ever_achieved_but_final_failure"] == 1
    assert result["failed_with_goal_lost_at_end"] == 1


def test_technical_error_is_a_failure_but_missing_stage_data_is_not_fabricated():
    result = diagnostics.summarize([row("A", success=True), {**row("B"), "status": "error"}], conditions=["A", "B"], resamples=100)
    assert result["comparisons"]["A - B"]["a_only"] == 1
    assert result["conditions"]["B"]["stage_observations"] == {"technical_errors": 1}
