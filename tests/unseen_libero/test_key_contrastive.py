import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_contrastive as ct


def test_full_fixed_key_cache_contract():
    x=np.ones((4516,256),np.float32)
    ct.validate_fixed_key_cache(x)
    with pytest.raises(AssertionError):ct.validate_fixed_key_cache(x[:10])
    with pytest.raises(AssertionError):ct.validate_fixed_key_cache(np.full((4516,256),np.nan))


def test_eval_identity_must_not_enter_training():
    ct.assert_training_eval_disjoint(np.array([[1,2,3],[1,4,5]]),np.array([[9,9,9]]))
    with pytest.raises(AssertionError):
        ct.assert_training_eval_disjoint(np.array([[1,2,3]]),np.array([[1,2,3]]))


def test_symmetric_soft_loss_two_directions_and_gradients():
    s=torch.tensor([[2.,0.,-1.],[0.,2.,-1.]],requires_grad=True)
    e=torch.tensor([[0.,1.,3.],[1.,0.,3.]])
    a=torch.ones_like(e,dtype=torch.bool)
    forward=ct.symmetric_soft_loss(s,e,a,.2)
    reverse=ct.symmetric_soft_loss(s,e,a,.2,reverse_scores=s.T,reverse_errors=e.T,
                                   reverse_allowed=a.T)
    assert forward.item()>0 and reverse.item()>0
    reverse.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert not torch.allclose(forward,reverse)


def test_key_motion_q1_import_shape_is_single_query_token():
    model=ct._make_model('e_h','x1','cpu')
    out=model(history=torch.zeros(2,10,7),history_mask=torch.ones(2,10,dtype=torch.bool),
              state=torch.zeros(2,8))
    assert out.shape==(2,1,256)

