import numpy as np

from skill_moe.skillnet.examples.memory_pilot.anchor_flow.overnight_prep import _segments_from_episode


def test_gripper_sign_changes_form_intervals_and_relative_motion_features():
    actions = np.zeros((6, 7), np.float32)
    actions[:, 6] = [-1, -1, 1, 1, -1, -1]
    state = np.zeros((6, 8), np.float32)
    state[:, 0] = np.arange(6, dtype=np.float32)
    segments = _segments_from_episode(actions, state, "pick the cup", "task.hdf5", "ep", 20)
    assert [s["steps"] for s in segments] == [2, 2, 2]
    assert [s["dx"] for s in segments] == [1.0, 1.0, 1.0]
    assert all(s["feature"].shape == (25,) for s in segments)
    assert all(s["verb"] == "pick" for s in segments)
