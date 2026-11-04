"""Checks for episode isolation and action-space distance diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location("geometry",Path(__file__).parents[1]/"scripts/analyze_vlm_action_geometry.py")
geometry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def test_queries_have_one_window_per_episode_and_reference_is_disjoint():
    task=np.repeat(["a","b"],30)
    episode=np.array([f"{t}-{i//3}" for t in ("a","b") for i in range(30)])
    query,ref=geometry.select_queries(task,episode,2,13)
    assert len(query)==4
    assert len(np.unique(episode[query]))==4
    assert not set(episode[query]) & set(episode[ref])
    assert all(np.sum(task[query]==t)==2 for t in ("a","b"))
    assert np.array_equal(query,geometry.select_queries(task,episode,2,13)[0])


def test_pairwise_mse_matches_explicit_action_differences():
    rng=np.random.default_rng(44)
    query=rng.normal(size=(3,70)).astype(np.float32)
    reference=rng.normal(size=(7,70)).astype(np.float32)
    measured=geometry.squared_mse(query,reference)
    explicit=np.mean((query[:,None].astype(float)-reference[None].astype(float))**2,axis=2)
    assert np.allclose(measured,explicit,atol=1e-12)
    assert np.allclose(geometry.squared_mse(query,query).diagonal(),0,atol=1e-12)


def test_task_bootstrap_does_not_weight_task_with_more_rows_more_heavily():
    task=np.array(["a","a","a","b"])
    stats=geometry.bootstrap_tasks(np.array([0.,0.,0.,2.]),task,100)
    assert stats["task_equal_mean"]==1.
    assert stats["task_bootstrap_ci95"][0]<=1<=stats["task_bootstrap_ci95"][1]


def test_quantiles_accept_boolean_events_and_missing_conditional_values():
    assert geometry.quantiles([True,False])["mean"]==.5
    assert geometry.quantiles([np.nan,2.])["median"]==2.
    assert geometry.quantiles([np.nan])["n"]==0
