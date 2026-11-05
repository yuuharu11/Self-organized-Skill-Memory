from __future__ import annotations

import subprocess
import sys
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from skill_memory_core.bridge import bridge_batch, integrate_euler
from skill_memory_core.drafts import sample_rank_mixture, shuffle_cross_task, validate_candidates
from skill_memory_core.key import SalientTokenKey, build_context, maxsim_rerank
from skill_memory_core.memory import MotionMemory, cut_fixed_windows
from skill_memory_core.metrics import (active_mse, exact_mcnemar, nearest_oracle,
    output_sensitivity, paired_episode_bootstrap, wilson_interval)
from skill_memory_core.units import RawFunctional, m6_fit, medoid


def test_cut_windows_and_memory_row_id_lookup(tmp_path):
    windows, starts = cut_fixed_windows(np.arange(26 * 2).reshape(26, 2), horizon=10, stride=10)
    assert windows.shape == (2, 10, 2)
    np.testing.assert_array_equal(starts, [0, 10])
    source = np.arange(12 * 10 * 32, dtype=np.float32).reshape(12, 10, 32)
    ids = np.asarray([9, 2, 7])
    mem = MotionMemory.from_dataset_rows(ids, source, np.arange(12), np.arange(12))
    np.testing.assert_array_equal(mem.drafts(source, np.asarray([2, 9])), source[[2, 9]])
    path = tmp_path / "memory.npz"
    mem.save(path)
    loaded = MotionMemory.load(path)
    np.testing.assert_array_equal(loaded.row_ids, ids)
    np.testing.assert_array_equal(loaded.exclude_tasks([2]).row_ids, [9, 7])
    with np.testing.assert_raises(KeyError):
        loaded.drafts(source, np.asarray([1]))


def test_context_layout_preserves_trained_bug_and_fixed_layout():
    state = np.arange(16, dtype=np.float32).reshape(2, 8)
    prev = np.arange(140, dtype=np.float32).reshape(2, 70)
    has = np.asarray([1, 0], np.float32)
    trained = build_context(state, prev, has, layout="as_trained")
    fixed = build_context(state, prev, has, layout="fixed")
    np.testing.assert_array_equal(trained[:, :8], state)
    np.testing.assert_array_equal(trained[:, 8:70], prev[:, 8:70])
    assert np.all(trained[:, 70:78] == 0)
    np.testing.assert_array_equal(fixed[:, 8:78], prev)
    np.testing.assert_array_equal(trained[:, 78], has)


def test_salient_token_key_shapes_and_expected_m2_state_dict_names():
    model = SalientTokenKey()
    sd = model.state_dict()
    expected = {"position", "token_proj.weight", "context.0.weight", "context.0.bias",
                "context.2.weight", "context.2.bias", "salience.0.weight", "salience.0.bias",
                "salience.2.weight", "salience.2.bias", "out.weight", "out.bias"}
    assert set(sd) == expected
    assert sd["token_proj.weight"].shape == (96, 2048)
    assert model.position.shape == (512, 96)
    key, h, w = model.encode_tokens(torch.randn(2, 512, 2048), torch.randn(2, 79),
                                    torch.ones(2, 512, dtype=torch.bool))
    assert key.shape == (2, 96) and h.shape == (2, 512, 96) and w.shape == (2, 512)
    np.testing.assert_allclose(w.sum(1).detach().numpy(), 1., atol=1e-6)


def test_maxsim_rank_matches_reference_equation_and_stable_sort():
    qh = np.asarray([[[1., 0.], [0., 1.]]], np.float16)
    qw = np.asarray([[.75, .25]], np.float16)
    mh = np.asarray([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]], np.float16)
    mw = np.asarray([[.8, .2], [.8, .2]], np.float16)
    c, s = maxsim_rerank(qh, qw, np.ones((1, 2), bool), mh, mw,
                         np.ones((2, 2), bool), [100], [[11, 12]], [11, 12])
    ref = []
    for m, w in zip(mh.astype(np.float32), mw.astype(np.float32)):
        sim = qh[0].astype(np.float32) @ m.T
        ref.append(.5 * ((sim.max(1) * qw[0].astype(np.float32)).sum() +
                         (sim.max(0) * w).sum()))
    np.testing.assert_allclose(s[0], ref, atol=1e-7)
    np.testing.assert_array_equal(c, [[11, 12]])


def test_rank_mixture_is_per_query_and_shuffle_is_cross_task():
    c = np.tile(np.arange(5, dtype=np.int64), (10000, 1))
    picked, ranks = sample_rank_mixture(c, np.random.default_rng(7))
    np.testing.assert_array_equal(picked, ranks)
    assert .68 < np.mean(ranks == 0) < .72
    row_ids = np.arange(100, 110)
    source = np.arange(10, dtype=np.int64)
    qtask = np.repeat([0, 1], 5)
    stask = np.repeat([2, 3], 5)
    shuffled, perm = shuffle_cross_task(source, row_ids, qtask, stask, seed=99)
    assert np.all(stask[shuffled] != qtask)
    assert sorted(perm.tolist()) == list(range(10))
    candidate_tasks = validate_candidates(np.tile(source[:2], (10, 1)), row_ids, qtask,
                                          source, stask)
    assert candidate_tasks.shape == (10, 2)


def test_shuffle_fails_instead_of_changing_the_source_multiset():
    # Every possible permutation assigns one source task to a matching query.
    with np.testing.assert_raises(RuntimeError):
        shuffle_cross_task([10, 11], [100, 101], [0, 0], [0, 1],
                           seed=4, max_attempts=3)


def test_bridge_training_equations_and_euler_identity():
    rng = np.random.default_rng(3)
    a, d, eps = (rng.normal(size=(3, 10, 32)).astype(np.float32) for _ in range(3))
    t0, u = np.asarray([.2, .4, .6], np.float32), np.asarray([.1, .5, 1.], np.float32)
    x, target, t = bridge_batch(a, d, eps, t0, u)
    source = (1 - t0[:, None, None]) * d + t0[:, None, None] * eps
    np.testing.assert_allclose(x, (1 - u[:, None, None]) * a + u[:, None, None] * source)
    np.testing.assert_allclose(target, (source - a) / t0[:, None, None])
    np.testing.assert_allclose(t, t0 * u)
    pred = integrate_euler(lambda _x, _t: np.broadcast_to(2., _x.shape),
                           np.ones((3, 2, 7), np.float32), t0, steps=4)
    expected = np.broadcast_to(1 - 2 * t0[:, None, None], pred.shape)
    np.testing.assert_allclose(pred, expected, atol=1e-6)


def test_raw_m6_is_data_dependent_and_medoid_is_member():
    x = np.zeros((64, 70), np.float32)
    tasks = np.repeat(np.arange(4), 16)
    eps = np.arange(64)
    rows = np.arange(200, 264)
    fit = m6_fit(x, rows, tasks, eps, kappa=.1, seed=5, min_support=10, min_episodes=3)
    assert fit["registered"] == 1
    assert np.all(fit["row_units"] == 0)
    assert fit["members_local"][0].shape == (64,)
    assert medoid(fit["members_local"][0], x) in fit["members_local"][0]
    assert RawFunctional().distance(x[0], x[1]) == 0.


def test_metrics_helpers():
    a = np.zeros((3, 10, 32), np.float32)
    b = np.ones_like(a)
    np.testing.assert_array_equal(active_mse(a, b), np.ones(3))
    idx, dist = nearest_oracle(b[:2], a, k=2)
    assert idx.shape == dist.shape == (2, 2) and np.all(dist == 1.)
    boot = paired_episode_bootstrap(np.arange(4.), np.zeros(4), [0, 0, 1, 1], reps=500, seed=2)
    assert boot["mean"] == 1.5 and boot["episodes"] == 2
    p, ci = wilson_interval(8, 10)
    assert p == .8 and ci[0] < p < ci[1]
    assert exact_mcnemar(0, 0) == 1 and exact_mcnemar(10, 0) < .01
    sens = output_sensitivity(a, b, a, b)
    assert sens["mean_ratio"] == 1.


def test_core_python_sources_are_portable():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("skill" + "_moe", "emerg" + "ent", ".skillnet" + "-state", "/" + "work")
    for path in root.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{token!r} leaked into {path.relative_to(root)}"


def test_minimal_example_runs_cpu_only():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root.parent))
    proc = subprocess.run([sys.executable, str(root / "examples" / "minimal.py")],
                          cwd=root.parent, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "'key_updates': 2" in proc.stdout


def test_package_copy_is_self_contained():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="skill-memory-core-") as temp:
        parent = Path(temp)
        target = parent / "skill_memory_core"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        bootstrap = parent / "bootstrap"
        bootstrap.mkdir()
        # The workspace venv contains editable-install .pth files. Remove those
        # workspace roots at interpreter startup while retaining site-packages.
        (bootstrap / "sitecustomize.py").write_text(
            "import sys; p=chr(47)+'work'; sys.path[:]=[x for x in sys.path if not x.startswith(p)]\n"
        )
        env = dict(os.environ)
        env.pop("SKILL_MEMORY_ARTIFACTS", None)
        env["PYTHONPATH"] = os.pathsep.join((str(bootstrap), str(parent)))
        probe = "import sys,skill_memory_core; p=chr(47)+'work'; assert not any(x.startswith(p) for x in sys.path)"
        imported = subprocess.run([sys.executable, "-c", probe], cwd=parent, env=env,
                                  capture_output=True, text=True, timeout=30)
        assert imported.returncode == 0, imported.stderr
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_core.py",
                                "-k", "not package_copy_is_self_contained"],
                               cwd=target, env=env, capture_output=True, text=True, timeout=120)
        assert tests.returncode == 0, tests.stdout + tests.stderr
        minimal = subprocess.run([sys.executable, "examples/minimal.py"], cwd=target,
                                 env=env, capture_output=True, text=True, timeout=120)
        assert minimal.returncode == 0, minimal.stderr
