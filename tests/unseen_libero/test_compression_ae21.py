import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/compression_ae21.py'
spec = importlib.util.spec_from_file_location('compression_ae21_tested', MOD_PATH)
ae21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae21)


def test_fit_ids_must_be_memory_only():
    rows = {'memory': np.array([1, 1, 0, 0], bool),
            'validation': np.array([0, 0, 1, 0], bool)}
    assert ae21.assert_fit_is_memory_only(rows, [0, 1])
    with pytest.raises(AssertionError):
        ae21.assert_fit_is_memory_only(rows, [0, 2])


def test_ae_accepts_only_action_windows_and_has_21_latents():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(8)
    model = ae21.ActionAE(width=8, depth=1, latent=21, device='cpu')
    x = np.random.default_rng(3).normal(size=(4, 10, 7)).astype(np.float32)
    z = model.encode(x)
    y = model.decode(z)
    assert tuple(z.shape) == (4, 21)
    assert tuple(y.shape) == (4, 10, 7)
    with pytest.raises(ValueError):
        model.encode(np.zeros((4, 10, 8), np.float32))


def test_axis_contributions_sum_to_active7_mse():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(5, 10, 7))
    pred = target + rng.normal(size=target.shape) * .1
    by_axis = ae21.per_row_axis_mse(pred, target)
    active7 = np.mean((pred - target) ** 2, axis=(1, 2))
    np.testing.assert_allclose(by_axis.sum(axis=1), active7, rtol=1e-12, atol=1e-12)
    summary = ae21.summarize_errors(pred, target, np.array([1, 0, 0, 1, 0], bool))
    assert summary['all']['axis_sum'] == pytest.approx(summary['all']['mse'])


def test_variable_g1_reference_is_fixed_and_declared():
    assert ae21.c3.G1_EXPECT['units'] == 4516
    assert ae21.c3.G1_EXPECT['all_mse'] == pytest.approx(.034447897620114326)
    assert ae21.c3.G1_EXPECT['hard_mse'] == pytest.approx(.06845424480130637)


def test_g1_dct_roundtrip_helper_matches_cmp3_encoder_decoder():
    x = np.random.default_rng(8).normal(size=(6, 10, 7)).astype(np.float32)
    ours = ae21.c2.dct_decode(ae21.c2.dct_encode(x, 3), 10)
    direct = ae21.c2.compressed_variable([v for v in x], 'dct', 3)
    np.testing.assert_allclose(ours, np.stack(direct), atol=1e-7, rtol=1e-7)

