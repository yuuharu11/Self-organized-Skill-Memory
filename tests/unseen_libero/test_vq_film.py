import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import dev_v2  # noqa: E402
import vq_film  # noqa: E402


def test_dev_tasks_exclude_eval_l40_and_old_dev():
    t = dev_v2.choose_tasks()
    banned = set(dev_v2.EVAL_TASKS) | set(dev_v2.OLD_DEV) | set(dev_v2.L40_SAME_INSTRUCTION) | set(dev_v2.MISSING)
    assert len(t) == 12 and not (set(t) & banned)
    assert t == dev_v2.choose_tasks()  # deterministic


def test_subset_masks():
    h = np.zeros((3, 10, 7), np.float32); m = np.ones((3, 10), bool); a = np.zeros((3, 10, 7), np.float32)
    h[..., 6] = -1; a[..., 6] = -1
    a[1, 1:, 6] = 1                       # switch in first future steps
    h[2, -5:, 0] = 1; a[2, :5, 0] = -1    # direction reversal
    m[0] = False
    s = dev_v2.subset_masks(np.array([0, 20, 30]), h, m, a)
    assert s['first_unit'].tolist() == [True, False, False]
    assert s['post_switch'].tolist() == [False, True, False]
    assert s['dir_change'].tolist() == [False, False, True]


def test_film_identity_at_init():
    torch.manual_seed(0)
    q = vq_film.FiLMQuery('resampler', k=4, d=8)
    base = vq_film.FiLMQuery('none')
    base.tower.load_state_dict(q.tower.state_dict())
    hs = torch.randn(5, 88); tok = torch.randn(5, 20, 2048); g = torch.zeros(5, 20, dtype=torch.long); g[:, -3:] = -1
    q.eval(); base.eval()
    assert torch.allclose(q(hs=hs, tokens=tok, groups=g), base(hs=hs), atol=1e-6)


def test_vib_kl_and_pooled_shapes():
    q = vq_film.FiLMQuery('pooled', d=16, vib=True); q.train()
    out = q(hs=torch.randn(4, 88), tokens=torch.randn(4, 7, 2048), groups=torch.zeros(4, 7, dtype=torch.long))
    assert out.shape == (4, 256) and float(q.kl) >= 0
