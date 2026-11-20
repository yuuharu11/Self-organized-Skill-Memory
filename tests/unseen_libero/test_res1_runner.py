from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero"
sys.path.insert(0, str(ROOT))
import res1_train
import res1_closed


def test_exact_row_intersection_does_not_use_neighbor_frames():
    q = np.asarray([[0, 1, 0], [0, 1, 5], [2, 3, 10]], np.int64)
    rows = np.asarray([[0, 1, 0], [0, 1, 10], [2, 3, 10]], np.int64)
    qids, rids = res1_train.exact_row_intersection(q, rows)
    assert qids.tolist() == [0, 2]
    assert rids.tolist() == [0, 2]


def test_exact_row_intersection_keeps_query_order():
    q = np.asarray([[2, 0, 4], [1, 2, 7]], np.int64)
    rows = np.asarray([[1, 2, 7], [2, 0, 4]], np.int64)
    qids, rids = res1_train.exact_row_intersection(q, rows)
    assert qids.tolist() == [0, 1]
    assert rids.tolist() == [1, 0]


def test_training_actions_must_match_exact_key_windows():
    q = np.zeros((2, 10, 7), np.float32)
    r = np.zeros((2, 10, 32), np.float32)
    res1_train.assert_action_alignment(q, r)
    r[0, 0, 0] = 1
    with pytest.raises(AssertionError, match="action windows differ"):
        res1_train.assert_action_alignment(q, r)


def test_closed_loop_checkpoint_uses_dev_selected_run(monkeypatch, tmp_path):
    selected = {"run": "flow_iii_exclude100", "checkpoint_step": 2}
    selected_all = {k: {"run": f"{k}_exclude100", "checkpoint_step": 2}
                    for k in ("flow_iii", "flow_iii_x", "flow_i")}
    monkeypatch.setattr(res1_closed, "OUT", tmp_path)
    monkeypatch.setattr(res1_closed.C, "read_json", lambda path, default=None: {
        "selected": selected_all}
        if Path(path) == tmp_path / "selection.json" else {
            "selected": {"checkpoint_step": 2}})
    cp = res1_closed._checkpoints()
    assert cp["flow_iii"]["checkpoint"].endswith(
        "flow_iii_exclude100/checkpoints/000002")
