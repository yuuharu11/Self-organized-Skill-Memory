"""Contract tests against the published S1-v3c dictionary artifact."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102')
S1 = ROOT / 's1-v3'
V3C = ROOT / 's1-v3c'
PRIV = Path('/work/.skillnet-state/outputs/memory-pilot/privstate-20261101T003957Z')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifacts():
    with np.load(V3C / 'library.npz', allow_pickle=True) as z:
        lib = {k: np.asarray(z[k]) for k in z.files}
    with np.load(S1 / 'frames.npz', allow_pickle=True) as z:
        frames = {k: np.asarray(z[k]) for k in z.files}
    return lib, frames


def test_contract_library_is_explicit_and_scales_round_trip():
    lib, _ = _artifacts()
    assert str(lib['contract_version']) == 'dmp-v3c-raw-action-coef16-v1'
    assert lib['coef'].shape == (98, 6, 16)
    assert lib['action_scale'].shape == (6,)
    assert lib['shape_samples'].shape == (98, 32, 6)
    assert 'shapes' not in lib  # runtime cannot silently treat samples as weights
    dmp = _load_module('dmp_v3c_contract_test', V3C / 'dmp.py')
    residual = np.arange(60, dtype=np.float64).reshape(10, 6) / 37.
    normalized = dmp.encode_residual(residual, lib['action_scale'])
    np.testing.assert_allclose(dmp.decode_residual(normalized, lib['action_scale']), residual, atol=1e-12)


def test_actual_train_interval_fit_nearest_primitive_matches_build_reconstruction():
    lib, fr = _artifacts()
    dmp = _load_module('dmp_v3c_train_interval_test', V3C / 'dmp.py')
    chosen = None
    for i in range(len(fr['t'])):
        if str(fr['split'][i]) != 'train':
            continue
        st, en = int(fr['seg_start'][i]), int(fr['seg_end'][i])
        if en - st >= 10 and int(fr['t'][i]) == st:
            chosen = i
            break
    assert chosen is not None
    ep = str(fr['episode_id'][chosen])
    row = next(r for r in csv.DictReader(open(PRIV / 'memory_episode_map.csv')) if r['episode_id'] == ep)
    action_path = PRIV / 'episodes' / (Path(row['source_file']).stem + '__' + row['demo_key'] + '.npz')
    with np.load(action_path, allow_pickle=True) as z:
        actions = np.asarray(z['action'], dtype=np.float32)
    st, en = int(fr['seg_start'][chosen]), int(fr['seg_end'][chosen])
    interval = actions[st:en, :6]
    coef, goal, tau = dmp.fit(interval, lib['action_scale'])
    sample_phase = np.linspace(0., 1., lib['shape_samples'].shape[1])
    sample_curve = dmp.residual_basis(sample_phase, coef.shape[-1]) @ coef.T
    grip = int(np.sign(np.median(actions[st:en, 6])) or 1)
    candidates = np.flatnonzero(lib['grip'] == grip)
    distances = np.mean((lib['shape_samples'][candidates] - sample_curve[None]) ** 2, axis=(1, 2))
    nearest = int(candidates[np.argmin(distances)])
    assert nearest == int(fr['prim_id'][chosen])

    phase = np.linspace(0., 1., len(interval))
    sampled = lib['shape_samples'][nearest]
    residual = np.stack([np.interp(phase, sample_phase, sampled[:, d]) for d in range(6)], axis=1)
    baseline = interval[0] + dmp.smoothstep(phase)[:, None] * goal[None]
    build_reconstruction = baseline + residual * lib['action_scale'][None]
    rollout = dmp.rollout(nearest, interval[0], goal, tau, len(interval), library=lib)
    np.testing.assert_allclose(rollout, build_reconstruction, rtol=5e-5, atol=5e-5)


def test_real_dictionary_rollout_prefix_and_tau_endpoint_contract():
    lib, _ = _artifacts()
    dmp = _load_module('dmp_v3c_schedule_test', V3C / 'dmp.py')
    y0 = np.asarray([.03, -.02, .1, .2, -.1, .04], np.float32)
    g = np.asarray([.1, .03, -.04, .2, .05, -.1], np.float32)
    tau = 2.0
    short = dmp.rollout(4, y0, g, tau, 20, library=lib)
    longer = dmp.rollout(4, y0, g, tau, 41, library=lib)
    np.testing.assert_array_equal(short, longer[:20])
    np.testing.assert_allclose(longer[-1], y0 + g, atol=1e-7)


def test_real_dictionary_midphase_reaches_endpoint_at_remaining_tau():
    lib, _ = _artifacts()
    dmp = _load_module('dmp_v3c_midphase_test', V3C / 'dmp.py')
    y0 = np.asarray([-.2, .1, .03, -.1, .07, .2], np.float32)
    g = np.asarray([.08, -.04, .05, .1, -.03, .06], np.float32)
    tau, s0 = 2.5, .4
    remaining = (1. - s0) * tau
    n_steps = int(round(remaining / dmp.DT)) + 1
    out = dmp.rollout(13, y0, g, tau, n_steps, s0=s0, library=lib)
    np.testing.assert_allclose(out[-1], y0 + g, atol=1e-7)
