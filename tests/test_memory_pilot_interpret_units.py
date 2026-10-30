from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.anchor_flow import interpret_units as iu


def test_motion_signature_reports_xyz_rotation_and_gripper_commands():
    action = np.zeros((10, 7), np.float32)
    action[:, 0] = 1.0
    action[:5, 6] = -1.0
    action[5:, 6] = 1.0
    action[:, 5] = 1.0

    result = iu.motion_signature(action, translation_matrix_m=np.eye(3) * 0.001,
                                 rotation_matrix_rad=np.diag([0., 0., np.deg2rad(9.)]))

    assert np.allclose(result["endpoint_displacement_mm"], [10.0, 0.0, 0.0])
    assert result["direction"] == "+x"
    assert result["rotation"] == "+z"
    assert result["cumulative_rotation_deg"][-1] == pytest.approx(90.0, abs=1e-3)
    assert result["gripper_mode"] == "mixed_commands"
    assert result["gripper_events"] == [{"step": 5, "command": "close"}]


def test_action_motion_calibration_recovers_per_command_physical_scales():
    n, stride = 120, 10
    rng = np.random.default_rng(7)
    actions = np.zeros((n, 10, 32), np.float32)
    actions[:, :, :3] = rng.normal(0, 0.3, (n, 10, 3))
    actions[:, :, 3] = rng.normal(0, 0.3, (n, 10))
    states = np.zeros((n, 8), np.float32)
    q01, q99 = [-1.] * 8, [1.] * 8
    raw_actions = actions[:, :, :7]
    scale_pos = np.array([.011, .012, .010])
    scale_rot = np.array([.10, .09, .11])
    for i in range(n - 1):
        pos_delta = raw_actions[i, :, :3].sum(axis=0) * scale_pos
        rot_delta = raw_actions[i, :, 3:6].sum(axis=0) * scale_rot
        states[i + 1, :3] = states[i, :3] + pos_delta
        states[i + 1, 3:6] = states[i, 3:6] + rot_delta
    # The stored arrays are quantile-normalized. With symmetric [-1, 1] bounds,
    # these test values invert to the original action/state values.
    stats = {"norm_stats": {name: {"q01": q01, "q99": q99}
                            for name in ("state", "actions")}}
    episodes = np.array([f"ep-{i // 120}" for i in range(n)])
    frames = np.arange(n) * stride
    fit = iu.calibrate_action_motion_scales(actions, states, episodes, frames, stats, stride=stride)
    assert np.allclose(np.diag(fit["translation_action_to_meters_matrix"]), scale_pos, atol=1e-5)
    assert fit["rotation_action_to_radians_matrix"][0][0] == pytest.approx(scale_rot[0], abs=1e-5)


def test_plan_parser_handles_verb_object_and_destination():
    parsed = iu.parse_plan_phrase("place the white mug on the left plate")
    assert parsed == {
        "text": "place the white mug on the left plate",
        "verb": "place",
        "object": "white mug",
        "destination": "left plate",
    }


def test_episode_progress_uses_three_coarse_time_bins():
    assert iu.episode_progress(0, 90) == (0.0, "early")
    assert iu.episode_progress(40, 90)[1] == "middle"
    assert iu.episode_progress(80, 90)[1] == "late"


def test_transition_summary_counts_adjacent_windows_and_collapsed_sequences():
    episodes = {
        "e1": [
            {"row_index": 0, "frame": 0, "unit_id": 2},
            {"row_index": 1, "frame": 10, "unit_id": 2},
            {"row_index": 2, "frame": 20, "unit_id": 5},
        ],
        "e2": [{"row_index": 3, "frame": 0, "unit_id": 2},
               {"row_index": 4, "frame": 10, "unit_id": 5}],
    }
    result = iu.build_transition_summary(episodes)
    assert result["adjacent_window_pairs"] == 3
    assert result["unit_switches"] == 2
    assert result["switch_fraction_of_adjacent_pairs"] == pytest.approx(2 / 3)
    assert result["typical_collapsed_sequences"][0] == {
        "unit_sequence": [2, 5], "episode_count": 2,
    }


def test_temporal_summary_reports_stride_duration_and_task_sequences():
    episodes = {"e": [
        {"row_index": 0, "frame": 0, "unit_id": 2, "task": "taskA"},
        {"row_index": 1, "frame": 10, "unit_id": 2, "task": "taskA"},
        {"row_index": 2, "frame": 20, "unit_id": 5, "task": "taskA"},
    ]}
    result = iu.build_temporal_summary(episodes)
    assert result["window_overlap_steps"] == 0
    assert result["run_length_windows"]["histogram"] == {"1": 1, "2": 1}
    assert result["run_length_seconds"]["median"] == pytest.approx(.75)
    assert result["typical_collapsed_sequences_by_task"][0]["sequences"][0]["unit_sequence"] == [2, 5]


def test_temporal_summary_uses_export_stride_for_duration_and_overlap():
    episodes = {"e": [
        {"row_index": 0, "frame": 0, "unit_id": 2, "task": "taskA"},
        {"row_index": 1, "frame": 5, "unit_id": 2, "task": "taskA"},
        {"row_index": 2, "frame": 10, "unit_id": 5, "task": "taskA"},
    ]}
    result = iu.build_temporal_summary(episodes, stride=5)
    assert result["window_overlap_steps"] == 5
    assert result["run_length_seconds"]["median"] == pytest.approx(.375)
    assert "overlap by 5 steps" in result["interpretation"]


def test_duplicate_motion_description_groups_units_by_direction_grip_rotation():
    units = [{"unit_id": 0, "center_motion_signature": {"direction": "+x", "gripper_mode": "close_command", "rotation": "none"}},
             {"unit_id": 1, "center_motion_signature": {"direction": "+x", "gripper_mode": "close_command", "rotation": "none"}},
             {"unit_id": 2, "center_motion_signature": {"direction": "-z", "gripper_mode": "open_command", "rotation": "+y"}}]
    result = iu.motion_description_overlap(units)
    assert result["units_in_duplicated_groups"] == 2
    assert result["matching_unit_pairs"] == 1


def test_context_agreement_and_interpretability_threshold():
    signatures = [{"direction": "+z", "rotation": "none", "gripper_mode": "close_command"}]
    signatures *= 4
    stable = iu.signature_agreement(signatures)
    assert stable["direction"] == stable["rotation"] == stable["gripper"] == 1.0
    assert iu.interpretability_label(stable)
    unstable = iu.signature_agreement(signatures[:2] + [
        {"direction": "-z", "rotation": "+x", "gripper_mode": "open_command"},
        {"direction": "-z", "rotation": "+x", "gripper_mode": "open_command"},
    ])
    assert not iu.interpretability_label(unstable)


@pytest.mark.parametrize("sampling_stride", [10, 5])
def test_synthetic_catalog_writes_machine_readable_units_and_figures(tmp_path: Path, sampling_stride: int):
    class FakeBackbone:
        def prior(self, key, state):
            return np.zeros((len(key), 2), np.float32), np.zeros((len(key), 1), np.float32)

        def decode(self, base, residual, feature, state):
            out = np.zeros((len(base), 10, 32), np.float32)
            out[:, :, 0] = base[:, 0, None] * 0.001
            out[:, :, 6] = np.where(base[:, 0, None] > 0, 1.0, -1.0)
            return out

    class FakeMemory:
        memory = [
            {"key": np.array([1., 0.], np.float32), "base": np.array([1., 0.], np.float32),
             "unit_id": 0, "meta": {"episode_id": "libero90/task_a/rlds-000001", "init_state_id": "a"}},
            {"key": np.array([0.9, 0.1], np.float32), "base": np.array([0.9, 0.1], np.float32),
             "unit_id": 0, "meta": {"episode_id": "libero90/task_a/rlds-000001", "init_state_id": "a"}},
            {"key": np.array([0., 1.], np.float32), "base": np.array([-1., 0.], np.float32),
             "unit_id": 1, "meta": {"episode_id": "libero90/task_b/rlds-000002", "init_state_id": "b"}},
            {"key": np.array([0.1, 0.9], np.float32), "base": np.array([-0.9, 0.1], np.float32),
             "unit_id": 1, "meta": {"episode_id": "libero90/task_b/rlds-000002", "init_state_id": "b"}},
        ]
        _centers = [np.array([0.95, 0.05], np.float32), np.array([-0.95, 0.05], np.float32)]
        backbone = FakeBackbone()
        key_text_source = "instruction_only"
        representation_version = "test-v1"

        @staticmethod
        def prior(key, state):
            return FakeBackbone.prior(FakeBackbone(), key, state)

        @staticmethod
        def decode(base, residual, feature, state):
            return FakeBackbone.decode(FakeBackbone(), base, residual, feature, state)

    memory = FakeMemory()
    rows = {
        "actions": np.zeros((4, 10, 32), np.float32),
        "state": np.zeros((4, 3), np.float32),
        "episode_id": np.array(["libero90/task_a/rlds-000001"] * 2 +
                                ["libero90/task_b/rlds-000002"] * 2),
        "init_state_id": np.array(["a", "a", "b", "b"]),
        "instruction": np.array(["pick the cup"] * 2 + ["open the drawer"] * 2),
        "source_file": np.array(["KITCHEN_SCENE1_pick_the_cup_demo.hdf5"] * 2 +
                                 ["STUDY_SCENE2_open_the_drawer_demo.hdf5"] * 2),
        "frame": np.array([0, 10, 0, 10]),
    }
    keys = np.stack([item["key"] for item in memory.memory])
    annotations = {
        "pick the cup": {"plan": ["pick the cup"], "all_classes": [0]},
        "open the drawer": {"plan": ["open the drawer"], "all_classes": [3]},
    }
    norm_stats = {"norm_stats": {"actions": {"q01": [-1.] * 7, "q99": [1.] * 7}}}
    norm_stats["norm_stats"]["state"] = {"q01": [-1.] * 8, "q99": [1.] * 8}

    def fake_frames(requests):
        return {(episode, frame): np.zeros((16, 16, 3), np.uint8)
                for episode, frames in requests.items() for frame in frames}

    output = tmp_path / "catalog"
    calibration = {"translation_action_to_meters_matrix": (np.eye(3) * .001).tolist(),
                   "rotation_action_to_radians_matrix": (np.eye(3) * .01).tolist(),
                   "paired_windows": 4}
    result = iu.analyze_arrays(memory, rows, keys, annotations, norm_stats, output,
                               frame_loader=fake_frames, stability_samples=4,
                               physical_calibration=calibration, sampling_stride=sampling_stride)

    assert result["metadata"]["labels_used_for_unit_creation"] is False
    assert result["metadata"]["plan_boundary_alignment"] is None
    assert len(result["units"]) == 2
    assert result["units"][0]["member_composition"]["n_tasks"] == 1
    assert (output / "units.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "figures/unit_000.jpg").is_file()
    assert (output / "figures/pca_traversal.jpg").is_file()
    windowing = result["metadata"]["windowing"]
    temporal = result["summary"]["temporal_run_statistics"]
    assert windowing["stride_steps"] == temporal["window_stride_steps"] == sampling_stride
    assert windowing["overlap_steps"] == temporal["window_overlap_steps"] == 10 - sampling_stride

