"""KC1 contract tests: saved DCT codes, episode masking, causal query inputs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'
sys.path.insert(0, str(HERE))
import kc1_run as kc


class KC1ContractTests(unittest.TestCase):
    def test_saved_coefficients_reencode_from_original_full_units(self):
        bank = kc.memory_bank()
        small = {k: np.asarray(v[:64]) for k, v in bank.items()}
        got = kc.verify_saved_coefficients(small, atol=2e-5)
        self.assertEqual(got['units'], 64)
        self.assertLessEqual(got['max_coefficient_abs_diff'], 2e-5)

    def test_mask_excludes_only_same_episode(self):
        mask = kc.candidate_mask([3, 3], [7, 8], [3, 3, 4], [7, 7, 7])
        np.testing.assert_array_equal(mask, [[False, False, True], [True, True, True]])

    def test_short_remainder_extends_through_next_anchor(self):
        actions = np.zeros((20, 7), np.float32)
        actions[:, 0] = np.linspace(-.5, .5, len(actions))
        _, span = kc.target_from_spans(actions, 6, [(0, 8), (8, 10), (10, 20)], min_tail=5)
        self.assertEqual(span, (6, 20))
        _, span = kc.target_from_spans(actions, 8, [(0, 8), (8, 10), (10, 20)], min_tail=5)
        self.assertEqual(span, (8, 20))

    def test_query_forward_accepts_observations_only(self):
        from kq2_run import QueryNet
        torch.set_num_threads(1)
        net = QueryNet('gru', 1, 0, False, history='w10', dct=True).eval()
        # Only the observed token window is exposed to the model. Ground-truth
        # coefficients, task IDs, and pi0.5 outputs are deliberately absent.
        observed = SimpleNamespace()
        observed.window = lambda ix: (torch.zeros((len(ix), 10, 16)), torch.ones((len(ix), 10), dtype=torch.bool))
        observed.seq_a = observed.seq_s = observed.off = observed.slot = observed.start = None
        with torch.no_grad():
            out = net(observed, torch.tensor([0, 1]))
        self.assertEqual(tuple(out.shape), (2, 256))
        self.assertTrue(torch.isfinite(out).all())

    def test_retrained_gripper_head_features_have_fixed_widths(self):
        from kc1_head import head_input
        n=3; feats={'scalar':np.zeros((n,12),np.float32),'q':np.zeros((n,256),np.float32),
                    'k1':np.zeros((n,18),np.float32)}
        gx=np.zeros((n,4),np.float32)
        self.assertEqual(head_input(feats,gx,'scalar').shape,(n,16))
        self.assertEqual(head_input(feats,gx,'full').shape,(n,290))


if __name__ == '__main__':
    unittest.main()
