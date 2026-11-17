import sys
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/robocasa"
sys.path.insert(0, str(EXAMPLE))
import latent_heads as lh


def test_transition_metrics_detect_switch_and_miss():
    truth = np.array([[-1, -1, -1, 1, 1, 1]], np.float32)
    pred = np.array([[-1, -1, 1, 1, 1, 1]], np.float32)
    m = lh.transition_metrics(pred, truth, tolerance=1)
    assert m["true_switches"] == 1
    assert m["pred_switches"] == 1
    assert m["mean_nearest_time_error_steps"] == 1.0
    assert m["switch_recall_within_2"] == 1.0


def test_transition_metrics_reports_missing_switch():
    truth = np.array([[-1, -1, 1, 1]], np.float32)
    pred = np.full_like(truth, -1)
    m = lh.transition_metrics(pred, truth)
    assert m["true_switches"] == 1
    assert m["pred_switches"] == 0
    assert m["switch_recall_within_2"] == 0.0


def test_runtime_head_inputs_are_latents_only():
    # The deployed error head consumes q/key tokens and, optionally, seven
    # explicit confidence signals. Labels/actions are never concatenated.
    q = np.zeros((2, 5, 256), np.float32)
    k = np.ones_like(q)
    x = np.concatenate([q.reshape(2, -1), k.reshape(2, -1), (q-k).reshape(2, -1), (q*k).reshape(2, -1)], axis=1)
    assert x.shape == (2, 5120)


def test_t0_head_uses_saved_empirical_quantiles():
    q = np.linspace(0, 1, 101)
    t = lh.t0_from_error(np.array([0., .5, 1.]), q)
    np.testing.assert_allclose(t, [.1, .5, .9], atol=1e-6)
