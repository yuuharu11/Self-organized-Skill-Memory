import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/varlen/run.py'
SPEC = importlib.util.spec_from_file_location('pi05_varlen_test_module', PATH)
varlen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(varlen)


def test_gripper_switches_are_boundaries_and_short_segments_merge():
    actions = np.zeros((50, 7), np.float32)
    actions[:, 6] = -1
    actions[21:, 6] = 1
    bounds = varlen.segment_boundaries(actions, 'grip', min_len=10)
    assert bounds[0] == 0 and bounds[-1] == 50
    assert 21 in bounds
    assert np.all(np.diff(bounds) >= 10)


def test_velocity_mode_requires_threshold_and_uses_local_minima():
    actions = np.zeros((80, 7), np.float32)
    actions[:, 0] = 1
    actions[20:30, 0] = .01
    actions[50:60, 0] = .01
    actions[:, 6] = -1
    try:
        varlen.segment_boundaries(actions, 'grip_vel')
    except ValueError:
        pass
    else:
        raise AssertionError('grip_vel must require a threshold')
    step = np.ones(80, np.float32)
    step[20:30] = .01
    step[50:60] = .01
    position = np.zeros((80, 3), np.float32)
    position[:, 0] = np.cumsum(step)
    bounds = varlen.segment_boundaries(actions, 'grip_vel', velocity_threshold=.1,
                                       eef_position=position)
    assert len(bounds) >= 3
    assert np.all(np.diff(bounds) >= 10)


def test_elapsed_phase_alignment_and_end_padding():
    x = np.arange(20 * 32, dtype=np.float32).reshape(20, 32)
    elapsed = varlen.draft_from_segment(x, 5, 20, 'elapsed')
    phase = varlen.draft_from_segment(x, 10, 20, 'phase')
    assert np.array_equal(elapsed, x[5:15])
    assert np.array_equal(phase, x[10:20])
    padded = varlen.draft_from_segment(x, 18, 20, 'elapsed')
    assert np.array_equal(padded[:2], x[18:20])
    assert np.array_equal(padded[2:], np.repeat(x[-1:], 8, axis=0))


def test_fixed10_boundaries_have_minimum_tail_segment():
    a = np.zeros((25, 7), np.float32)
    b = varlen.segment_boundaries(a, 'fixed10')
    assert np.array_equal(b, [0, 10, 20, 25])


def test_segment_key_uses_first_cached_observation_at_or_after_boundary():
    frames = np.asarray([0, 10, 20, 30], np.int32)
    assert varlen.key_observation_index(frames, 10) == 1
    assert varlen.key_observation_index(frames, 11) == 2
    assert varlen.key_observation_index(frames, 31) == 3
