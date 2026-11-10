import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.fft import dct

PATH = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/low_freq_denoise.py'
spec = importlib.util.spec_from_file_location('low_freq_denoise_test', PATH)
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)


def test_basis_matches_scipy_dct():
    x = np.random.default_rng(0).normal(size=(10, 4))
    assert np.allclose(lf.dct_basis().T @ x, dct(x, type=2, norm='ortho', axis=0))


def test_projection_keeps_only_low_frequencies():
    noise = np.random.default_rng(1).normal(size=(64, 10, 32))
    for k in (3, 5):
        low = lf.project(noise, k)
        coef = dct(low, type=2, norm='ortho', axis=1)
        assert np.abs(coef[:, k:]).max() < 1e-5
        assert np.abs(coef[:, :k]).max() > .1


def test_mean_element_variance_is_one():
    for k in (1, 3, 5, 10):
        p = lf.low_projector(k)
        assert np.isclose(np.trace(p @ p.T) / 10, 1.0)


def test_k10_is_white_noise():
    noise = np.random.default_rng(2).normal(size=(3, 10, 7))
    assert np.allclose(lf.project(noise, 10), noise, atol=1e-5)


def test_conditions_reuse_white_drafts():
    assert set(c[0] for c in lf.CONDS.values()) <= {'A-raw', 'A-vm-none', 'A-vm-08'}
    assert all(0 < lo < hi <= 1 for _, _, (lo, hi) in lf.CONDS.values())


def test_installed_eval_hook_projects_the_shared_white_noise(monkeypatch):
    captured = {}

    def original(model, obs, drafts, noise, t0, *, batch):
        captured['noise'] = noise
        captured['t0'] = t0
        captured['batch'] = batch
        return 'pred'

    cd = SimpleNamespace(BATCH=8, CONDS=('A-raw',), T0_EVAL=(1.,),
                         _predict_action=original, _t0_range=lambda _: (0., 1.))
    monkeypatch.setattr(lf, '_link_drafts', lambda _name: None)
    installed = lf.install(cd, 'L-vm-none-k3')
    white = np.random.default_rng(9).normal(size=(4, 10, 32)).astype(np.float32)
    result = installed._predict_action('m', 'o', 'd', white, .2)
    assert result == 'pred'
    np.testing.assert_allclose(captured['noise'], lf.project(white, 3), atol=1e-6)
    assert captured['t0'] == .2 and captured['batch'] == 8


def test_comparison_rejects_different_draft_or_target(tmp_path, monkeypatch):
    out = tmp_path / 'low'
    white = tmp_path / 'white'
    condition, white_condition = 'L-vm-none-k3', 'A-vm-none'
    low_dir = out / 'predictions' / condition
    white_dir = white / 'predictions' / white_condition
    low_dir.mkdir(parents=True); white_dir.mkdir(parents=True)
    (out / 'experts' / condition).mkdir(parents=True)
    (white / 'experts' / white_condition).mkdir(parents=True)
    (out / 'experts' / condition / 'trained.json').write_text(json.dumps({'best': {'t0': .2}}))
    (white / 'experts' / white_condition / 'trained.json').write_text(json.dumps({'best': {'t0': .2}}))
    n = 3
    rows = {'row_ids': np.arange(n), 'target': np.zeros((n, 10, 7), np.float32),
            'draft': np.ones((n, 10, 7), np.float32), 'hard': np.array([False, True, False]),
            'task': np.zeros(n, np.int32), 'episode': np.arange(n, dtype=np.int32),
            'err_t0.2': np.ones(n, np.float32)}
    np.savez(low_dir / 'eval_rows.npz', **rows)
    paired = dict(rows); paired['draft'] = rows['draft'].copy(); paired['target'] = rows['target'].copy()
    paired['draft'][0, 0, 0] += 1
    np.savez(white_dir / 'eval_rows.npz', **paired)
    monkeypatch.setattr(lf, 'OUT', out); monkeypatch.setattr(lf, 'WHITE', white)
    monkeypatch.setattr(lf, '_load_cd', lambda: SimpleNamespace(
        _episode_keys=lambda task, ep: task * 100000 + ep,
        _bootstrap_all_and_hard=lambda *args: {'all': {}, 'hard': {}}))
    try:
        lf.compare(condition)
    except AssertionError as exc:
        assert 'draft differs' in str(exc)
    else:
        raise AssertionError('paired comparison accepted a different draft')
