import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import x_attn_query as xq


def test_predecessor_is_within_same_episode_only():
    task = np.array([1, 1, 1, 2])
    episode = np.array([4, 4, 4, 4])
    frame = np.array([0, 10, 20, 10])
    np.testing.assert_array_equal(xq.predecessor_indices(task, episode, frame), [-1, 0, 1, -1])


def test_temporal_visual_difference_norm_history_ratio_and_flag():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(8)
    task = np.array([1, 1, 1, 2])
    episode = np.array([4, 4, 4, 4])
    frame = np.array([0, 10, 20, 10])
    prev = xq.predecessor_indices(task, episode, frame)
    image = np.stack([np.full((128, 2), v, np.float32) for v in (2, 5, 9, 20)])
    ds = {'task': task, 'episode': episode, 'frame': frame, 'prev_index': prev,
          'hist': np.ones((4, 10, 7), np.float32), 'state': np.zeros((4, 8), np.float32),
          'image': image}
    feat = xq._feature(ds, np.array([1, 2, 3]), 'final', arch='P5-imgdelta', device='cpu')
    np.testing.assert_allclose(feat['visual_delta'].numpy()[:, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(feat['context'].numpy()[:, 3], [1, 1, 0])
    # Context ratio is visual-change norm divided by recent-motion norm.
    delta_norm=np.linalg.norm(feat['visual_delta'].numpy(),axis=(1,2))
    hist_norm=np.linalg.norm(np.ones((3,10,6),np.float32),axis=(1,2))
    np.testing.assert_allclose(feat['context'].numpy()[:, 2], np.log1p(delta_norm/np.maximum(hist_norm,1e-6)), rtol=1e-6)
    assert feat['context'].shape == (3, 4)
    assert feat['context'][0, 1] < feat['context'][1, 1]


def test_temporal_model_uses_context_but_not_phase_metadata():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(8)
    torch.manual_seed(9)
    model = xq.make_model('P5-imgdelta', 78, 60, 64).eval()
    feat = {'hist': torch.randn(2, 10, 7), 'state': torch.randn(2, 8),
            'visual': torch.randn(2, 32, 64), 'visual_delta': torch.randn(2, 32, 64),
            'context': torch.randn(2, 4), 'query_phi': torch.tensor([.1, .7])}
    candidates = torch.randn(9, 60)
    torch.testing.assert_close(model(feat, candidates), model(feat, candidates))
    changed = dict(feat); changed['query_phi'] = torch.tensor([.9, .01])
    torch.testing.assert_close(model(feat, candidates), model(changed, candidates))
    changed['context'] = feat['context'].clone(); changed['context'][:, 1] += 10
    assert not torch.allclose(model(feat, candidates), model(changed, candidates))

def test_old_checkpoint_obsolete_phase_scalar_is_accepted():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(8)
    model = xq.make_model('P5', 78, 60, 64)
    state = dict(model.state_dict()); state['phase_scale'] = torch.zeros(())
    xq._load_state_compat(model, state)


def test_fusion_gate_is_query_conditioned():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(8)
    torch.manual_seed(3)
    gate = xq.make_fusion_gate(14)
    with torch.no_grad():
        gate[0].weight.fill_(.03); gate[0].bias.zero_()
        gate[2].weight.fill_(.02); gate[2].bias.zero_()
    p5 = torch.tensor([[3., 1., 0.], [.1, 0., -.1]])
    nn = torch.tensor([[0., 2., 1.], [3., 0., -1.]])
    feat = {'hist': torch.ones(2, 10, 7), 'state': torch.ones(2, 8),
            'context': torch.tensor([[0., 0., 0., 0.], [2., 1., 3., 1.]])}
    score, alpha = xq.combine_with_gate(gate, p5, nn, feat)
    assert score.shape == (2, 3)
    assert alpha.shape == (2,) and torch.all((alpha > 0) & (alpha < 1))
    assert not torch.isclose(alpha[0], alpha[1])


def test_smoke_selection_has_contiguous_windows():
    task = np.repeat([1, 2, 3, 4], 8)
    episode = np.tile(np.repeat([0, 1], 4), 4)
    frame = np.tile(np.tile(np.arange(4) * 10, 2), 4)
    take = xq.smoke_positions(task, episode, frame, 16)
    assert len(take) == 16
    assert any(frame[j + 1] == frame[j] + 10 and episode[j + 1] == episode[j]
               for j in take if j + 1 in set(take.tolist()))
