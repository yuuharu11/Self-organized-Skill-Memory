import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_latent_predict as jp1


def test_ema_target_is_stop_gradient_and_tracks_online():
    online = torch.nn.Linear(3, 2)
    target = jp1.make_ema_target(online)
    assert all(not p.requires_grad for p in target.parameters())
    before = target.weight.detach().clone()
    with torch.no_grad():
        online.weight.add_(1.)
    jp1.update_ema(target, online, decay=.5)
    assert torch.allclose(target.weight, before + .5)
    x = torch.randn(4, 3, requires_grad=True)
    with torch.no_grad():
        target(x).sum()
    assert x.grad is None  # target forward is explicitly wrapped by caller no_grad
    assert all(p.grad is None for p in target.parameters())


def test_next_boundary_uses_unit_end_as_next_unit_start_and_excludes_eval_episode():
    source = np.asarray([[1, 2, 0, 7], [1, 2, 7, 15], [1, 2, 15, 20],
                         [1, 3, 0, 5], [1, 3, 5, 9]], np.int32)
    q, frames, nxt = jp1.next_boundary_pairs(source, eval_pairs={(1, 3)})
    assert q.tolist() == [0, 1]
    assert frames.tolist() == [7, 15]
    assert nxt.tolist() == [1, 2]


def test_tplus10_key_definition():
    keys = np.asarray([[1, 4, 0], [2, 9, 13]], np.int32)
    assert jp1.tplus10_keys(keys).tolist() == [[1, 4, 10], [2, 9, 23]]
    assert keys.tolist() == [[1, 4, 0], [2, 9, 13]]


def test_tplus10_pairs_match_exact_cache_identities():
    current=np.asarray([[1,2,0],[1,2,10],[3,4,0]])
    available=np.asarray([[1,2,10],[1,2,20],[3,4,5]])
    src,dst=jp1.match_tplus10_pairs(current,available)
    assert src.tolist()==[0,1]
    assert dst.tolist()==[0,1]


def test_l90x_pairs_exclude_selection_rows_without_reading_actions():
    rows = {'task': np.asarray([8, 8, 8, 9, 9]),
            'episode': np.asarray([1, 1, 1, 2, 2]),
            'frame': np.asarray([0, 10, 20, 0, 10]),
            'is_select': np.asarray([False, False, False, True, True])}
    src, dst = jp1.l90x_transition_indices(rows, step=10)
    assert src.tolist() == [0, 1]
    assert dst.tolist() == [1, 2]


def test_l90x_manifest_asserts_dev_and_eval_tasks_excluded():
    from latent_action_units_la2 import assert_l90x_manifest
    assert_l90x_manifest({'tasks': [1, 2, 5], 'episodes': [{'task_id': 5}]})
    for task in (6, 7, 0, 10, 88):
        try:
            assert_l90x_manifest({'tasks': [1, task]})
        except AssertionError:
            pass
        else:
            raise AssertionError(f'forbidden task {task} was accepted')


def test_dev_selection_uses_top1_metric_only():
    candidates = [
        {'name': 'blend-best', 'dev_top1_arm6_mse': .04, 'dev_blend_arm6_mse': .001},
        {'name': 'top1-best', 'dev_top1_arm6_mse': .02, 'dev_blend_arm6_mse': .03},
    ]
    assert jp1.choose_top1_dev(candidates)['name'] == 'top1-best'


def test_declared_ablations_cover_required_training_stages():
    recipes=jp1.condition_recipes()
    names={x['name'] for x in recipes}
    for mode in ('e_vlm','e_vlm_h'):
        assert f'a_{mode}_lret' in names
        assert f'b_{mode}_pred_then_ret' in names
        assert f'b_{mode}_pred_then_ret_unconditioned' in names
        assert f'c_{mode}_joint' in names
        assert f'd_{mode}_pred_only' in names
        assert f'e_{mode}_l90x_pred_then_ret' in names
    assert all(next(x for x in recipes if x['name']==f'c_{mode}_joint')['lambda_pred_grid']==[.1,.3,1.]
               for mode in ('e_vlm','e_vlm_h'))


def test_predictor_and_vicreg_are_finite_and_differentiable():
    z = torch.nn.functional.normalize(torch.randn(8, 5, 256), dim=-1).requires_grad_()
    c = torch.randn(8, 30)
    predictor = jp1.NextLatentPredictor(condition_dim=30)
    pred = predictor(z, c)
    target = torch.nn.functional.normalize(torch.randn_like(pred), dim=-1)
    loss, parts = jp1.vicreg_prediction_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert set(parts) == {'prediction', 'variance', 'covariance'}


def test_vicreg_detaches_target():
    pred = torch.randn(3, 5, 8, requires_grad=True)
    target = torch.randn(3, 5, 8, requires_grad=True)
    loss, _ = jp1.vicreg_prediction_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None


def test_vicreg_variance_penalty_uses_unormalized_predictor_scale():
    collapsed = torch.ones(8, 5, 256)
    varied = torch.randn(8, 5, 256) * 2.0
    _, collapsed_parts = jp1.vicreg_prediction_loss(collapsed, torch.randn_like(collapsed))
    _, varied_parts = jp1.vicreg_prediction_loss(varied, torch.randn_like(varied))
    assert collapsed_parts['variance'].item() > .98
    assert varied_parts['variance'].item() < .01
