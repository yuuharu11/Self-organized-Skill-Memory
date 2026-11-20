import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'
sys.path.insert(0, str(ROOT))
import res1_core as core
import res1_context as ctx
import res1_experiment as exp
import kq2_run as kr


def test_residual_flow_has_correct_noise_to_residual_direction():
    draft = torch.zeros(2, 4, 7)
    target = torch.ones_like(draft); target[..., 6] = -1
    noise = torch.full((2, 4, 6), 3.)
    x0, v = core.sample_residual_flow(target, draft, noise, torch.zeros(2))
    x1, _ = core.sample_residual_flow(target, draft, noise, torch.ones(2))
    assert torch.allclose(x0, torch.ones_like(x0))
    assert torch.allclose(x1, noise)
    assert torch.allclose(v, torch.full_like(v, 2.))
    # Descending-time Euler integration subtracts noise-target velocity.
    assert torch.allclose(noise - v, x0)


def test_residual_integrator_returns_draft_plus_target_residual():
    draft=torch.zeros(1,3,7);draft[...,6]=-1
    target=torch.full((1,3,7),1.);target[...,6]=-1
    noise=torch.full((1,3,6),3.)
    # Flow vector is d(noise->target)/du = noise - target.
    def vf(_obs, _point, _u, _saved, _current): return torch.full_like(noise,2.)
    out=core.integrate_residual(vf,None,draft,noise,steps=10)
    assert torch.allclose(out,torch.ones_like(out),atol=1e-6)


def test_delta_attention_is_identity_at_initialization_and_gate_learns():
    torch.manual_seed(2)
    layer = core.DeltaContextCrossAttention(dim=16, heads=4)
    q, s, c = torch.randn(3, 5, 16), torch.randn(3, 4, 16), torch.randn(3, 4, 16)
    out, weights = layer(q, s, c, return_attention=True)
    assert torch.equal(out, q)
    assert weights.shape == (3, 4, 5, 12)
    (out.square().mean()).backward()
    assert layer.gate.grad is not None and torch.isfinite(layer.gate.grad)


def test_delta_memory_is_saved_current_difference_and_masks_align():
    s = torch.tensor([[[1., 2.], [3., 4.]]])
    c = torch.tensor([[[2., 4.], [6., 8.]]])
    mem, pad = core.DeltaContextCrossAttention.make_memory(s, c, torch.tensor([[True, False]]))
    assert torch.equal(mem, torch.cat((s, c, c - s), 1))
    assert pad.tolist() == [[False, True, False, True, False, True]]


def test_saved_context_validation_and_task_free_context_assembly():
    saved = core.SavedContext(torch.zeros(4, 1, 8), torch.zeros(4, 8))
    saved.validate()
    projector = ctx.StateContextProjection(8, 8)
    memory, valid = ctx.kq_context_tokens({'h': saved.h, 'state': saved.state},
                                          {'h': saved.h + 1, 'state': saved.state + 1},
                                          projector, visual=False)
    assert memory.shape == (4, 6, 8) and valid.shape == (4, 6)
    with pytest.raises(ValueError):
        core.SavedContext(torch.zeros(4, 2, 8), torch.zeros(4, 8)).validate()


def test_rank_sampler_respects_mixture_and_positive_ranks():
    ranks, excl = core.draw_rank_samples(np.random.default_rng(1), np.arange(1, 8), 1000,
                                         own_task_fraction=.6, uniform_fraction=.1)
    assert ranks.min() >= 1 and ranks.max() <= 4516
    assert .56 < excl.mean() < .64


@pytest.mark.parametrize('visual', [False, True])
def test_saved_query_context_is_exact_pre_l2_kq2_query_output(visual):
    torch.manual_seed(7)
    n=3
    sp={'seq_actions':np.random.default_rng(1).normal(size=(24,7)).astype(np.float32),
        'seq_states':np.random.default_rng(2).normal(size=(24,8)).astype(np.float32),
        'ep_offset':np.array([0]),'ep_slot':np.zeros(n,np.int64),'start':np.array([0,5,11]),
        'key':np.array([[1,2,0],[1,2,5],[1,2,11]]),'errors':np.ones((n,4),np.float32),
        'allowed':np.ones((n,4),bool),'task':np.ones(n,np.int64),'episode':np.full(n,2,np.int64)}
    if visual:
        sp['vlm']=np.random.default_rng(3).normal(size=(n,34,2048)).astype(np.float16)
    gs=kr.GPUSplit(sp,torch.device('cpu'),visual)
    model=kr.QueryNet(enc='gru',gru_layers=1,xattn=1 if visual else 0,ffn=True,history='w10').eval()
    ix=torch.arange(n)
    state=torch.randn(n,8)
    c=ctx.query_context(model,gs,ix,state=state,visual=visual)
    q=model(gs,ix)
    assert c['h'].shape==(n,256)
    assert torch.allclose(torch.nn.functional.normalize(c['h'],dim=-1),q,atol=2e-6,rtol=2e-6)
    assert c['state'].shape==(n,8)
    if visual: assert c['vlm'].shape==(n,34,256)


def test_smoke_updates_residual_flow_regression_and_xattn():
    out = exp.smoke(steps=24, seed=5)
    assert out['status'] == 'smoke_complete'
    assert all(np.isfinite(list(out['final_loss'].values())))
    assert out['delta_gate'] != 0

