"""CPU-only checks for RM-D1 history, scale modes, and resume identities."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/robocasa"))
from memory_server_utils import (ClientHistory, completed_trials_from_shard, fit_scale,
                                 merge_worker_shards, replace_gripper_only)


class MemoryServerHelpersTest(unittest.TestCase):
    def test_left_padding_append_and_episode_reset(self):
        history = ClientHistory(10)
        rows, mask, call = history.begin_request("task-a:0", True)
        self.assertEqual(rows.shape, (10, 12))
        self.assertEqual(mask.sum(), 0)
        self.assertEqual(call, 0)
        action = np.ones((5, 12), np.float32)
        history.set_pending(action)
        rows, mask, call = history.begin_request("task-a:0", False)
        np.testing.assert_array_equal(rows[-5:], action)
        np.testing.assert_array_equal(mask, np.array([False] * 5 + [True] * 5))
        self.assertEqual(call, 1)
        rows, mask, _ = history.begin_request("task-b:1", True)
        self.assertEqual(mask.sum(), 0)
        self.assertEqual(float(np.abs(rows).sum()), 0.0)

    def test_scale_modes_only_change_eef_magnitudes(self):
        cand = np.ones((10, 12), np.float32)
        ref = np.zeros((10, 12), np.float32)
        ref[:, :3] = 2.0
        ref[:, 3:6] = 3.0
        ref[:, 6:] = 7.0
        np.testing.assert_array_equal(fit_scale(cand, ref, mode="saved"), cand)
        shared = fit_scale(cand, ref, mode="s-shared")
        sep = fit_scale(cand, ref, mode="s-sep")
        np.testing.assert_allclose(shared[:, :6], 2.5)
        np.testing.assert_allclose(sep[:, :3], 2.0)
        np.testing.assert_allclose(sep[:, 3:6], 3.0)
        np.testing.assert_array_equal(shared[:, 6:], cand[:, 6:])
        np.testing.assert_array_equal(sep[:, 6:], cand[:, 6:])

    def test_resume_uses_task_trial_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "worker.json"
            path.write_text(json.dumps([
                {"task": "OpenDrawer", "trial": 3, "status": "complete"},
                {"task": "OpenDrawer", "trial": 4, "status": "failed"},
            ]))
            done, rows = completed_trials_from_shard(path)
            self.assertEqual(done, {("OpenDrawer", 3)})
            self.assertEqual(len(rows), 2)

    def test_resume_merge_keeps_records_from_old_worker_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "worker-00.json").write_text(json.dumps([
                {"task": "OpenDrawer", "trial": 0, "status": "complete"},
            ]))
            (root / "worker-03.json").write_text(json.dumps([
                {"task": "CloseFridge", "trial": 5, "status": "complete"},
            ]))
            merged = merge_worker_shards(root)
            self.assertEqual({(r["task"], r["trial"]) for r in merged},
                             {("OpenDrawer", 0), ("CloseFridge", 5)})

    def test_gripper_replacement_changes_only_channel_six(self):
        bridge = np.arange(60, dtype=np.float32).reshape(5, 12)
        pi05 = np.full((5, 12), -3, np.float32)
        out = replace_gripper_only(bridge, pi05)
        np.testing.assert_array_equal(out[:, 6], -3)
        np.testing.assert_array_equal(out[:, :6], bridge[:, :6])
        np.testing.assert_array_equal(out[:, 7:], bridge[:, 7:])


if __name__ == "__main__":
    unittest.main()
