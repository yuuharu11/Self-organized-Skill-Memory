"""History-free magnitude adapter: route legacy inputs to the MAG-v2 layout.

The legacy retrieval helper feeds [context | 130 history dims | unit metadata]
while the MAG-v2 checkpoint was trained on [context 2064 | unit metadata 20].
The adapter must drop exactly the middle history block; taking the tail by a
hard-coded width once fed history values as unit metadata instead.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MP = str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa")
if MP not in sys.path:
    sys.path.insert(0, MP)

import expert3_prep as e3p
import expert2_prep as prep


def _install(fake_ckpt, monkeypatch, pool_key_width=19):
    mag_net = torch.nn.Linear(2084, 1)
    with torch.no_grad():
        mag_net.weight.copy_(torch.arange(2084, dtype=torch.float32).reshape(1, -1) * 1e-4)
        mag_net.bias.zero_()
    pool = {"key": np.zeros((4, pool_key_width), np.float32)}
    mag_ck = {"history_free": True}
    old = (None, None, pool, None, None, None, mag_ck, mag_net)
    monkeypatch.setattr(prep, "_load_models", lambda device="cpu": old)
    monkeypatch.setattr(torch, "load", lambda *a, **k: fake_ckpt)
    e3p._install_history_free_magnitude_adapter()
    return prep._load_models


def _fake_ckpt():
    n = 2084
    return {"input_dim": n, "history_free": True,
            "feature_mean": (np.arange(n, dtype=np.float32) * 0.01),
            "feature_std": (1.0 + np.arange(n, dtype=np.float32) * 0.001)}


def test_adapter_routes_context_and_unit_metadata_around_history(monkeypatch):
    load_models = _install(_fake_ckpt(), monkeypatch)
    _, _, _, _, _, _, mag_ck, adapter = load_models("cpu")
    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.normal(size=(2, 2084 + 130)).astype(np.float32))
    # The retrieval helper normalizes with the adapted stats before the
    # adapter only reroutes, mirroring _make_dynamic_drafts.
    mean = torch.as_tensor(np.asarray(mag_ck["feature_mean"], np.float32))
    std = torch.as_tensor(np.asarray(mag_ck["feature_std"], np.float32))
    normed = (x - mean) / std
    compact = torch.cat([normed[..., :2064], normed[..., 2064 + 130:]], dim=-1)
    with torch.no_grad():
        want = adapter.inner(compact)
        got = adapter(normed)
    torch.testing.assert_close(got, want)
    # The unit slice must be the trailing unit metadata, not history tail.
    assert compact.shape[-1] == 2084
    # feature stats seen by the retrieval helper match the checkpoint layout.
    np.testing.assert_array_equal(np.asarray(mag_ck["feature_mean"][:2064]),
                                  _fake_ckpt()["feature_mean"][:2064])
    np.testing.assert_array_equal(np.asarray(mag_ck["feature_mean"][-20:]),
                                  _fake_ckpt()["feature_mean"][-20:])


def test_adapter_rejects_changed_unit_metadata_width(monkeypatch):
    load_models = _install(_fake_ckpt(), monkeypatch, pool_key_width=32)
    with pytest.raises(AssertionError):
        load_models("cpu")
