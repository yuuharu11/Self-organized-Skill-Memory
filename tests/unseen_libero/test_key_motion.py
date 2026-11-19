import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero"))

import key_motion as km
import key_shared_latent as key2


def test_motion_key_shapes_and_padding_invariance():
    torch.manual_seed(3)
    x = torch.randn(2, 9, 6)
    mask = torch.tensor([[1] * 7 + [0] * 2, [1] * 9], dtype=torch.bool)
    padded = x.clone(); padded[0, 7:] = 1e4
    for mode, shape in (("m1", (2, 1, 256)), ("m5", (2, 1, 256))):
        model = km.MotionKeyEncoder(mode).eval()
        a = model(x, mask); b = model(padded, mask)
        assert a.shape == shape
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        assert torch.allclose(a.norm(dim=-1), torch.ones(2, 1), atol=1e-6)


def test_motion_input_has_no_observation_or_task_argument():
    import inspect
    parameters = inspect.signature(km.MotionKeyEncoder.forward).parameters
    assert set(parameters) == {"self", "actions", "valid", "return_details"}


@pytest.mark.parametrize("query_mode,expected", [("q1", (2, 1, 256)), ("q5", (2, 1, 256))])
def test_query_shapes(query_mode, expected):
    torch.manual_seed(4)
    q = km.MotionQueryEncoder("e_h", query_mode)
    out = q(history=torch.randn(2, 10, 7), history_mask=torch.ones(2, 10, dtype=torch.bool),
            state=torch.randn(2, 8))
    assert out.shape == expected


def test_eval_episodes_are_rejected_from_motion_training():
    task = [1, 1, 2]; episode = [10, 11, 20]
    km.assert_no_eval_motion_rows([0, 2], {(1, 11)}, task, episode)
    with pytest.raises(AssertionError, match="Eval episodes leaked"):
        km.assert_no_eval_motion_rows([0, 1], {(1, 11)}, task, episode)


@pytest.mark.parametrize("arch", ["x1", "x1s", "x5"])
@pytest.mark.parametrize("position", ["absolute", "phase"])
def test_arch_encoder_shapes_and_padding(arch, position):
    torch.manual_seed(8)
    x=torch.randn(2,8,6);mask=torch.tensor([[1,1,1,1,1,0,0,0],[1]*8],dtype=torch.bool)
    x2=x.clone();x2[0,5:]=999
    model=km.ArchMotionEncoder(arch,position).eval()
    a=model(x,mask);b=model(x2,mask)
    assert a.shape==(2,1,256)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    if arch=="x5":assert model(x,mask,return_details=True)["work"].shape==(2,5,256)


def test_dct_mlp_encoder_shape():
    model=km.DCTMLPKeyEncoder()
    assert model(coefficients=torch.randn(3,18)).shape==(3,1,256)


def test_retrieval_query_encoder_q5_and_dct_projection():
    torch.manual_seed(9)
    model=km.RetrievalQueryEncoder("e_vlm_h","q5",key_mode="dct",token_dim=8,dim=16,heads=4)
    out=model(history=torch.randn(2,10,7),history_mask=torch.ones(2,10,dtype=torch.bool),
        state=torch.randn(2,8),tokens=torch.randn(2,6,8),groups=torch.tensor([[0,0,1,1,2,2]]*2))
    assert out.shape==(2,18)
    details=model(history=torch.randn(2,10,7),history_mask=torch.ones(2,10,dtype=torch.bool),
        state=torch.randn(2,8),tokens=torch.randn(2,6,8),groups=torch.tensor([[0,0,1,1,2,2]]*2),return_details=True)
    assert details["work"].shape==(2,5,16)
    eh=km.RetrievalQueryEncoder("e_h","q1",token_dim=8,dim=16,heads=4)
    with_motion=eh(history=torch.randn(2,10,7),history_mask=torch.ones(2,10,dtype=torch.bool),
        state=torch.randn(2,8),motion_history=torch.randn(2,16),motion_history_valid=torch.tensor([True,False]))
    assert with_motion.shape==(2,1,16)
    import inspect
    params=inspect.signature(km.RetrievalQueryEncoder.forward).parameters
    assert "task" not in params and "target_actions" not in params and "future" not in params


def test_import_does_not_change_existing_shared_latent_output():
    torch.manual_seed(7)
    m = key2.SharedLatentEncoder(token_dim=8, context_dim=16, token_count=5,
                                 history_steps=10, history_action_dim=7, state_dim=8,
                                 group_count=6, heads=4, input_mode="e_h").eval()
    args = dict(history=torch.randn(2, 10, 7), history_mask=torch.ones(2, 10, dtype=torch.bool),
                state=torch.randn(2, 8))
    first = m(**args)
    # Merely creating MK1 modules does not alter E's class, parameters, or output.
    km.MotionKeyEncoder("m1")
    second = m(**args)
    torch.testing.assert_close(first, second)
