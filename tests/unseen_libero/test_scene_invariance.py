from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/scene_invariance.py"
spec = importlib.util.spec_from_file_location("scene_invariance_test", SCRIPT)
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def test_scene_ids_use_scene_map_and_legacy_suite_groups():
    names = si._task_names()
    ids = si.scene_ids(np.arange(40), names)
    assert len(ids) == 40
    assert len(set(ids[:10])) > 1
    assert len(set(ids[10:20])) == 1 and ids[10] == "libero_goal"
    assert len(set(ids[20:30])) == 1 and ids[20] == "libero_object"
    assert len(set(ids[30:40])) == 1 and ids[30] == "libero_spatial"


def test_scene_ids_fail_on_unmapped_task():
    names = si._task_names()
    names[0] = "unknown task"
    try:
        si.scene_ids([0], names)
    except ValueError as e:
        assert "no LIBERO-10 scene mapping" in str(e)
    else:
        raise AssertionError("unknown scene mapping must fail closed")


def test_bootstrap_mean_returns_ordered_interval_covering_mean():
    import numpy as np

    rng = np.random.default_rng(0)
    values = rng.normal(loc=2.0, scale=1.0, size=60)
    episode = np.repeat(np.arange(6), 10)
    lo, hi = si.bootstrap_mean(values, episode)
    assert lo <= float(values.mean()) <= hi
    assert lo < hi
    lo2, hi2 = si.bootstrap_mean(values, episode)
    assert (lo, hi) == (lo2, hi2)


def test_unseen_eval_scene_ids_resolve_from_manifest():
    got = si.unseen_scene_ids(np.array([0, 10, 84, 88]))
    assert got.shape == (4,)
    assert all(x.startswith("libero90:") for x in got)
    assert got[0].startswith("libero90:KITCHEN_SCENE")


def test_scene_fold_excludes_held_scene_and_nonfit_rows():
    data={"fit":np.array([True,True,True,False]),
          "scene":np.array(["s0","s1","s0","s1"])}
    mask=si._scene_fold_mask(data,"s0")
    assert mask.tolist()==[False,True,False,False]


def test_soft_teacher_respects_candidate_mask():
    err = np.array([[0., .1, .2], [.2, .1, 0.]], np.float32)
    allowed = np.array([[False, True, True], [True, True, False]])
    p = si._soft_teacher(err, allowed, .05)
    assert np.all(p[~allowed] == 0)
    assert np.allclose(p.sum(1), 1)
    assert p[0, 1] > p[0, 2] and p[1, 1] > p[1, 0]


def test_pair_error_and_contrastive_are_finite():
    x = np.zeros((4, 10, 6), np.float32)
    x[0, :, 0] = 1; x[1, :, 0] = 1.01
    x[2, :, 1] = 1; x[3, :, 2] = 1
    e = si._pair_err(x)
    assert e.shape == (4, 4)
    assert e[0, 1] < e[0, 2]
    den=(x.reshape(len(x),-1)**2).sum(1)
    dot=max(float((x[0]*x[1]).sum()),0.)
    expected=max(float(den[0]-dot*dot/den[1]),0.)/x[0].size
    assert np.isclose(e[0,1],expected)
    import torch
    z = torch.randn(4, 8, requires_grad=True)
    loss, npos = si._contrastive_loss(z, torch.tensor(x), np.array([0, 1, 1, 0]),
                                      np.array([0, 1, 2, 3]), .01)
    assert torch.isfinite(loss) and npos > 0
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_fold_positive_threshold_ignores_rows_outside_fold():
    data={"target":np.arange(4*10*6,dtype=np.float32).reshape(4,10,6)+1,
          "task":np.arange(4),"scene":np.array(["a","b","a","b"])}
    first=si._tau_pos_for_rows(data,[0,1,2],seed=1,per_task=48)
    data["target"][3]*=1000
    second=si._tau_pos_for_rows(data,[0,1,2],seed=1,per_task=48)
    assert first == second and first > 0


def test_eval_history_perturbations_are_fixed_seed_and_cross_candidate_masked():
    hist = np.zeros((3, 10, 7), np.float32)
    fit_hist = np.ones((4, 10, 7), np.float32)
    pool = np.arange(5*10*7, dtype=np.float32).reshape(5, 10, 7)
    allowed = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,1]], bool)
    a = si._history_perturbations(hist, fit_hist, allowed, pool, seed=99)
    b = si._history_perturbations(hist, fit_hist, allowed, pool, seed=99)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.all(allowed[np.arange(3), a[2]])
    assert a[0].shape == hist.shape and a[1].shape == hist.shape


def test_model_shapes_for_p5_wrist_and_latent_queries():
    import torch
    for kind, k, dim, n in [("P5", 1, 2048, 0), ("A", 1, 2048, 8), ("B", 4, 256, 16)]:
        m = si._make_model(kind, k, dim)
        hist = torch.randn(2, 10, 7); state = torch.randn(2, 8)
        tokens = None if kind == "P5" else torch.randn(2, n, dim)
        mask = None if kind == "P5" else torch.ones(2, n, dtype=torch.bool)
        q, attn = m(hist, state, tokens, mask)
        assert q.shape == (2, 256)
        assert (attn is None) == (kind == "P5")


def test_step1_scene_groups_collapse_all_libero10_scenes():
    g=si.scene_group_ids(np.arange(40))
    assert set(g)=={"libero10_all","libero_goal","libero_object","libero_spatial"}
    assert np.all(g[:10]=="libero10_all")


def test_step1_api_excludes_held_group_and_flips_spearman_sign(tmp_path):
    group=np.array(["libero_goal","libero_object","libero_spatial","libero10_all"]*2)
    task=np.arange(8)%4
    error=np.tile(np.array([[.0,.1,.2,.3],[.1,.0,.2,.3]],np.float32),(4,1))
    d={"task":task,"group":group,"fit":np.array([1,1,1,1,0,0,0,0],bool),
       "select":np.array([0,0,0,0,1,1,1,1],bool),"error":error,
       "allowed":np.ones_like(error,bool),"pool_task":np.arange(4),
       "pool_group":np.array(["libero_goal","libero_object","libero_spatial","libero10_all"]),
       "pool_shape":np.arange(4,dtype=np.float32)[:,None]}
    eval_d={"task":np.arange(4),"group":group[:4],"error":error[:4],
            "allowed":np.ones_like(error[:4],bool)}
    seen=[]
    def train_fn(condition, rows, *, seed, epochs, save_path, fold):
        seen.append((fold,set(rows["group"].tolist()),set(rows["pool_group"].tolist())))
        return object()
    def score_fn(model, condition, rows):
        return -np.asarray(rows["error"])
    result=si.run_step1_evaluations(tmp_path,train_fn,score_fn,data=d,eval_data=eval_d,
                                    conditions=("P5-L0",),seed=4101,epochs=8)
    assert len(result["cv"])==4
    for fold,trained,candidates in seen[:4]:
        assert fold not in trained
        assert fold not in candidates
    assert result["eval"]["P5-L0"]["metrics"]["spearman_mean"] > .99


def test_step1_six_axis_mse_uses_sum_over_seven_scale():
    pred=np.zeros((2,10,7),np.float32)
    target=np.ones_like(pred)
    got=si._six_axis_mse_over7(pred,target)
    # Six active axes each have unit error; the standard metric divides by 7.
    assert np.allclose(got,6/7)
    scalar_mean=np.mean((pred[...,:6]-target[...,:6])**2,axis=(1,2))
    assert np.allclose(got,scalar_mean*6/7)

