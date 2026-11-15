import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/probes/noro_closed_loop.py'
spec = importlib.util.spec_from_file_location('noro_closed_loop', MODULE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class _ToyVs:
    @staticmethod
    def to_physical(x): return np.asarray(x, np.float64)
    @staticmethod
    def to_normalized(x): return np.asarray(x, np.float64)


class _ToySelector:
    def __init__(self, shape):
        self.canonical = np.asarray(shape, np.float32)[None]
        self.pool = {'phys': np.asarray(shape, np.float32)[None], 'source_row': np.asarray([41]),
                     'source_unit': np.asarray([3])}
        self.vs = _ToyVs()

    def score_and_pick(self, history, state, allowed=None):
        return 0, np.asarray([.7], np.float32)


def test_selected_type_is_only_scaled_not_rotated():
    rng = np.random.default_rng(5)
    shape = rng.normal(size=(10, 6)).astype(np.float32)
    selector = _ToySelector(shape)
    pi = rng.normal(size=(10, 7)).astype(np.float32)
    pi[:, :6] = .5 * shape + .01 * pi[:, :6]  # aligned with the type, so the fit is positive
    history = np.zeros((10, 7), np.float32)
    state = np.zeros(8, np.float32)
    output, dbg = policy.compose_normalized_action(pi, history, state, selector)
    assert not dbg['fallback_pi05']
    # The toy codec is identity; all six physical axes must share one scalar.
    valid = np.abs(shape) > 1e-7
    ratios = output[:, :6][valid] / shape[valid]
    assert np.allclose(ratios, ratios[0], atol=1e-5)
    assert np.array_equal(output[:, 6:], pi[:, 6:])
    assert dbg['candidate_index'] == 0 and dbg['source_row'] == 41


def test_raw_live_camera_inputs_are_flipped_180_degrees():
    base = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = base + 1
    got = policy.rotate_observation_images_180({'observation/image': base,
                                                'observation/wrist_image': wrist,
                                                'observation/state': np.zeros(8)})
    assert np.array_equal(got['observation/image'], base[::-1, ::-1])
    assert np.array_equal(got['observation/wrist_image'], wrist[::-1, ::-1])
    assert got['observation/image'].flags.c_contiguous


def test_dev_server_composer_matches_lpg_t_p5_s_pi05_on_ten_rows(tmp_path):
    required = [policy.DEV_INPUTS, policy.DEV_PI05,
                policy.SCALE/'prep/W_pool.npz', policy.SCALE/'selection.json',
                policy.ROOT/'latent-probe-01/gate-full/summary.json',
                policy.ROOT/'compressed-denoise-01/production/drafts/C_vm_none_train.npz']
    if not all(p.exists() for p in required):
        pytest.skip('LPG actual-cache artifacts unavailable; online/offline parity smoke skipped')
    result = policy.dev_parity(tmp_path/'parity', n=10)
    assert result['rows_checked'] == 10
    assert result['max_abs_server_offline_action_delta'] <= 1e-5
    assert result['pi05_orientation_check']['rot180_mse6_over7'] == pytest.approx(.1144, abs=.01)
    assert result['pi05_orientation_check']['asis_mse6_over7'] == pytest.approx(.1917, abs=.02)
    assert result['rotation_applied_to_selected_type'] is False
    assert len(result['selected_pool_indices']) == len(result['multipliers']) == 10



def test_fallback_to_pi05_without_history_or_with_zero_scale():
    rng = np.random.default_rng(6)
    shape = rng.normal(size=(10, 6)).astype(np.float32)
    selector = _ToySelector(shape)
    pi = rng.normal(size=(10, 7)).astype(np.float32)
    out, dbg = policy.compose_normalized_action(pi, np.zeros((10, 7), np.float32), np.zeros(8, np.float32), selector, has_history=False)
    assert dbg['fallback_pi05'] and dbg['fallback_reason'] == 'no_history' and np.array_equal(out, pi)
    anti = pi.copy(); anti[:, :6] = -shape
    out, dbg = policy.compose_normalized_action(anti, np.zeros((10, 7), np.float32), np.zeros(8, np.float32), selector)
    assert dbg['fallback_pi05'] and dbg['fallback_reason'] == 'zero_scale' and np.array_equal(out, anti)
