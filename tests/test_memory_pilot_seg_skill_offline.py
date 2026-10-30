import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[1]
          / "skill_moe/skillnet/examples/memory_pilot/anchor_flow/seg_skill_offline.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("seg_skill_offline", SCRIPT)
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)


def test_gripper_segmentation_and_phase_chunk():
    action = np.zeros((2, 10, 7), np.float32)
    action[:, :, 6] = -1
    action[1, 3:, 6] = 1
    assert seg.gripper_segments(action) == [(0, 13), (13, 20)]
    seq = np.stack([np.arange(20), np.zeros(20)], axis=-1).astype(np.float32)
    out = seg.phase_chunk(seq, .5, horizon=4)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 0], [10, 11, 12, 13], atol=1e-5)


def test_centered_keys_and_cluster_templates_are_deterministic():
    m = np.asarray([[1., 0.], [0., 1.], [-1., 0.]], np.float32)
    cm, cq = seg.centered_keys(m, np.asarray([[1., 0.]], np.float32))
    assert cm.shape == (3, 2) and cq.shape == (1, 2)
    assert np.all(np.isfinite(cm)) and np.all(np.isfinite(cq))
    samples = []
    for i in range(24):
        a = np.zeros((12 + i % 3, 32), np.float32)
        a[:, 0] = (i % 4) * .1
        a[:, 6] = -1 if i % 2 else 1
        samples.append({"actions": a})
    labels, templates, info = seg.cluster_segments(samples, seed=4)
    assert len(labels) == len(samples)
    assert info["selected_k"] in range(4, 17)
    assert all(v.shape == (64, 32) for v in templates.values())
