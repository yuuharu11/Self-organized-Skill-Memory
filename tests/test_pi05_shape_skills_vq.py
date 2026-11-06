import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/shape_skills/vq.py'
spec = importlib.util.spec_from_file_location('shape_skills_vq_test', SCRIPT)
vq = importlib.util.module_from_spec(spec); sys.modules[spec.name] = vq; spec.loader.exec_module(vq)


def test_padding_is_masked_and_does_not_change_valid_frame_reconstruction_loss():
    torch.set_num_threads(1)
    x = torch.tensor([[[1., 2.], [3., 4.]]])
    y = torch.tensor([[[1., 2.], [3., 4.]]])
    mask = torch.tensor([[True, True]])
    base = vq._masked_reconstruction_loss(y, x, mask)
    xp = torch.cat([x, torch.zeros((1, 3, 2))], dim=1)
    yp = torch.cat([y, torch.full((1, 3, 2), 100.)], dim=1)
    mp = torch.tensor([[True, True, False, False, False]])
    padded = vq._masked_reconstruction_loss(yp, xp, mp)
    torch.testing.assert_close(base, padded)


def test_pad_sequences_tracks_valid_lengths_and_mask_only():
    x, phase, mask, lengths = vq.pad_sequences([np.ones((2, 7)), np.ones((5, 7))*3])
    assert x.shape == (2, 5, 7)
    np.testing.assert_array_equal(lengths.numpy(), [2, 5])
    np.testing.assert_array_equal(mask.numpy().sum(1), [2, 5])
    np.testing.assert_array_equal(x.numpy()[0, 2:], 0.)
    np.testing.assert_allclose(phase.numpy()[0, :2], [0., 1.])


def test_vqvae_forward_handles_masked_variable_lengths_and_all_residual_branches():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    _, _, _, _, Model = vq._torch_classes()
    model = Model(input_dim=7, latent_dim=8, k=16, residual_dims=(4, 8))
    x, phase, mask, _ = vq.pad_sequences([np.random.randn(6, 7).astype(np.float32),
                                         np.random.randn(11, 7).astype(np.float32)])
    pred, ids, qloss, reg = model(x, phase, mask, update=True)
    assert pred['0'].shape == (2, 11, 7)
    assert pred['4'].shape == pred['8'].shape
    assert ids.shape == (2,) and torch.isfinite(qloss) and torch.isfinite(reg)
    losses = vq._masked_reconstruction_loss(pred['0'], x, mask)
    assert torch.isfinite(losses).all()


def test_masked_padding_cannot_change_the_segment_latent():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    _, _, _, _, Model = vq._torch_classes()
    model = Model(input_dim=7, latent_dim=8, k=16, residual_dims=(4, 8)).eval()
    x, phase, mask, _ = vq.pad_sequences([np.random.default_rng(7).normal(size=(5, 7)).astype(np.float32)])
    x_changed = x.clone()
    x_changed[:, 5:] = 123.0  # invalid padding is deliberately nonzero
    with torch.no_grad():
        z0, *_ = model.encode_quantized(x, phase, mask, update=False)
        z1, *_ = model.encode_quantized(x_changed, phase, mask, update=False)
    torch.testing.assert_close(z0, z1)


def test_fit_split_excludes_long_and_heldout_tasks():
    segments = [{'task': 0}, {'task': 16}, {'task': 17}, {'task': 39}]
    ids = vq._training_indices(segments)
    np.testing.assert_array_equal(ids, [1, 3])


def test_procrustes_transform_accepts_variable_length_with_train_reference():
    rng=np.random.default_rng(3)
    seq=rng.normal(0,.01,(17,7)).astype(np.float32)
    seq[:,6]=np.where(np.arange(len(seq))<8,-1.,1.)
    ref=np.cumsum(rng.normal(0,.01,(32,3)).astype(np.float32),axis=0)
    x,pose=vq._seq_input(seq,'procrustes',np.full(7,-1,np.float32),np.full(7,1,np.float32),ref)
    assert x.shape==(17,7) and pose['rotation'].shape==(3,3)
    np.testing.assert_allclose(pose['rotation'].T@pose['rotation'],np.eye(3),atol=1e-5)

