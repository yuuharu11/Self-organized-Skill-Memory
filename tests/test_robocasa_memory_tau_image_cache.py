from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/robocasa"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

import tau_image_cache as tic


def test_greedy_tau_cover_has_valid_representatives():
    d = np.asarray([[0., .01, .20], [.01, 0., .19], [.20, .19, 0.]], np.float32)
    reps, assign = tic._greedy_cover(d, .02)
    assert reps.tolist() == [0, 2]
    assert assign.tolist() == [0, 0, 1]
    assert all(d[int(reps[c]), i] <= .02 for i, c in enumerate(assign))


def test_cached_resized_frames_match_random_frame_reader():
    cache = tic.FRAME_ROOT / "PickPlaceCounterToCabinet" / "episode_000006.npz"
    if not cache.exists():
        pytest.skip("full robocasa frame cache has not been generated yet")
    checks = tic.verify_cache_sample()
    assert len(checks) >= 9
    assert all(x["equal"] for x in checks)

