import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'skill_moe' / 'skillnet' / 'examples' / 'memory_pilot' / 'unseen_libero'))
import compression_seen_cuts as cs  # noqa: E402


def test_gripper_switches_are_mandatory_and_cover_trace():
    x = np.zeros((24, 7), np.float32)
    x[8:, 6] = 1
    x[16:, 6] = -1
    parts = cs.cut(x, mode='grip', min_len=5)
    assert parts[0][0] == 0 and parts[-1][1] == 24
    assert any(b == 8 for _, b in parts[:-1])
    assert any(b == 16 for _, b in parts[:-1])


def test_g1_cut_matches_motion_core_anchor_spans_on_synthetic_trace():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(100, 7)).astype(np.float32) * .03, axis=0)
    x[:, 6] = np.where(np.arange(100) < 50, -1, 1)
    ours = cs.cut(x, mode='all', min_len=5, width=3)
    reference = cs.motion.anchor_spans(x, min_length=5)
    assert ours == reference


def test_velocity_thresholds_are_calculated_from_memory_only():
    rng = np.random.default_rng(2)
    traces = {(0, i): rng.normal(size=(30, 7)).astype(np.float32) for i in range(3)}
    cfg, meta = cs.make_configs(traces)
    assert cfg['G2_prom_q25']['prom'] <= cfg['G2_prom_q50']['prom'] <= cfg['G2_prom_q75']['prom']
    assert cfg['G3_slow_q10']['speed_max'] <= cfg['G3_slow_q20']['speed_max']
    assert meta['memory_frames'] == 90


def test_seen_split_uses_validation_nonmemory_demos_only():
    # Two validation and one memory demo for every task; selected task set is balanced.
    actions, task, episode, frame, memory, validation = [], [], [], [], [], []
    for t in range(40):
        for ep, is_mem, is_val in ((t * 3, True, False), (t * 3 + 1, False, True), (t * 3 + 2, False, True)):
            for f in (0, 10):
                actions.append(np.zeros((10, 32), np.float32))
                task.append(t); episode.append(ep); frame.append(f)
                memory.append(is_mem); validation.append(is_val)
    rows = {'actions': np.stack(actions), 'task': np.array(task), 'episode': np.array(episode),
            'frame': np.array(frame), 'memory': np.array(memory), 'validation': np.array(validation)}
    out = cs.select_seen(rows, seed=13)
    assert len(out['task_ids']) == 20
    assert len(out['demos']) == 40
    assert len(out['targets']) == 80
    for t, ep in out['demos']:
        mask = (rows['task'] == t) & (rows['episode'] == ep)
        assert rows['validation'][mask].all()
        assert not rows['memory'][mask].any()


def test_phase_and_dct_helpers_preserve_segment_lengths():
    rng = np.random.default_rng(3)
    units = [rng.normal(size=(n, 7)).astype(np.float32) for n in (5, 11, 19)]
    coded = cs.dct3(units)
    assert [len(x) for x in coded] == [5, 11, 19]
    assert all(x.shape[1] == 7 for x in coded)


def test_fixed10_pool_drops_incomplete_episode_tail():
    trace = np.arange(19 * 7, dtype=np.float32).reshape(19, 7)
    units, lengths = cs.spans({(0, 0): trace}, None)
    assert lengths.tolist() == [10]
    np.testing.assert_array_equal(units[0], trace[:10])
