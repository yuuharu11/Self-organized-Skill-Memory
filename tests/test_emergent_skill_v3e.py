import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.intermediate.skill_v3e import (
    _cluster_membership_metrics,
    _kl_diag,
    sinkhorn_assign,
)


def test_sinkhorn_assignments_are_valid_and_nearly_balanced():
    logits = torch.tensor([[0.08, 0.00, -0.01], [0.07, 0.00, -0.01],
                           [0.06, 0.01, 0.00], [0.00, 0.08, -0.01],
                           [0.00, 0.07, -0.01], [0.00, 0.00, 0.08]])
    q = sinkhorn_assign(logits)
    assert q.shape == logits.shape
    assert torch.allclose(q.sum(-1), torch.ones(6), atol=1e-5)
    assert q.sum(0).max() - q.sum(0).min() < 0.7


def test_kl_balancing_has_finite_gradients_for_both_alpha_values():
    for alpha in (0.5, 0.8):
        qm = torch.randn(5, 4, requires_grad=True)
        ql = torch.zeros(5, 4, requires_grad=True)
        pm = torch.randn(5, 4, requires_grad=True)
        pl = torch.zeros(5, 4, requires_grad=True)
        prior_following = _kl_diag(qm.detach(), ql.detach(), pm, pl).mean()
        posterior_following = _kl_diag(qm, ql, pm.detach(), pl.detach()).mean()
        loss = alpha * prior_following + (1 - alpha) * posterior_following
        loss.backward()
        assert torch.isfinite(loss)
        assert qm.grad is not None and qm.grad.abs().sum() > 0
        assert pm.grad is not None and pm.grad.abs().sum() > 0


def test_functional_accuracy_uses_per_skill_epsilon_and_reports_no_match():
    train = np.asarray([[0.0], [0.01], [0.02], [2.0], [2.01], [2.02]], np.float32)
    assignments = np.asarray([0, 0, 0, 1, 1, 1])
    centers = np.asarray([[0.01], [2.01]], np.float32)
    query = np.asarray([[0.015], [2.015], [9.0]], np.float32)
    selected = np.asarray([0, 1, 0])
    metrics = _cluster_membership_metrics(train, assignments, query, selected, centers)
    assert metrics["strict_accuracy"] == 2 / 3
    assert metrics["functional_coverage"] == 2 / 3
    assert metrics["functional_accuracy_all"] == 2 / 3
    assert metrics["no_match_rate"] == 1 / 3
    assert metrics["functional_accuracy_covered"] == 1.0
