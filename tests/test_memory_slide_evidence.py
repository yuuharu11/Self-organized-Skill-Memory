"""Checks for cached-evidence alignment, clustered summaries, and routing splits."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "memory_slide_evidence", Path(__file__).parents[1] / "scripts/analyze_memory_slide_evidence.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_alignment_rejects_reordered_tasks_and_duplicate_windows():
    meta = {"task": np.array([1, 2]), "episode": np.array(["a", "b"]), "frame": np.array([0, 0])}
    audit.check_alignment(meta, np.array([1, 2]))
    with pytest.raises(ValueError, match="align"):
        audit.check_alignment(meta, np.array([2, 1]))
    meta.update(task=np.array([1, 1]), episode=np.array(["a", "a"]))
    with pytest.raises(ValueError, match="duplicate"):
        audit.check_alignment(meta, np.array([1, 1]))


def test_equal_episode_weight_does_not_count_overlapping_windows_as_independent():
    tasks = np.array([1, 1, 1, 2])
    episodes = np.array(["a", "a", "b", "c"])
    weights = audit.balanced_weights(tasks, episodes)
    assert np.allclose(weights, [.125, .125, .25, .5])
    meta = {"task": tasks, "episode": episodes}
    result = audit.hierarchical_interval(np.array([0., 0., 2., 2.]), meta, 20, 3)
    assert result["task_episode_equal_mean"] == 1.5


def test_heldout_task_labels_cannot_change_that_tasks_gate():
    tasks = np.repeat([1, 2, 3], 8)
    episodes = np.array([f"t{t}-e{i//2}" for t in [1, 2, 3] for i in range(8)])
    x = np.column_stack([np.tile(np.arange(8), 3), np.tile(np.arange(8)[::-1], 3)]) / 8
    normal = np.ones(24)
    memory = np.tile(np.r_[np.full(4, 2.), np.full(4, .2)], 3)
    choices, errors, folds = audit.evaluate_gate(x, tasks, episodes, normal, memory)
    altered = memory.copy(); altered[tasks == 1] = 100.
    choices2, _, _ = audit.evaluate_gate(x, tasks, episodes, normal, altered)
    for name in choices:
        assert np.array_equal(choices[name][tasks == 1], choices2[name][tasks == 1])
    for fold in folds:
        assert fold["test_task"] not in fold["train_tasks"]
    assert np.all(errors["oracle"] <= normal)
    assert np.all(errors["oracle"] <= memory)


def test_shared_episode_across_tasks_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        audit.evaluate_gate(np.ones((4, 2)), np.array([1, 1, 2, 2]),
                            np.array(["same"]*4), np.ones(4), np.ones(4)*2)


def test_seed_summary_does_not_inflate_shared_episode_count():
    meta = {"task": np.array([1, 2]), "episode": np.array(["a", "b"]), "frame": np.array([0, 0])}
    result = audit.summarize_seed_arrays([np.array([1., 2.])]*3, [meta]*3, 20, 7)
    assert result["episodes_per_seed"] == [2, 2, 2]
    assert result["row_mean_seed_average"] == 1.5
    assert "averaging 3 seeds" in result["ci_scope"]
