import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/robocasa"))

from norm_utils import norm_ratio_scale


def test_norm_ratio_is_independent_of_direction_cosine():
    draft = np.zeros((10, 3), np.float32)
    reference = np.zeros_like(draft)
    draft[:, 0] = 1.0
    reference[:, 1] = 1.0  # cosine is zero, but magnitudes are equal
    assert norm_ratio_scale(draft, reference, 10) == 1.0


def test_norm_ratio_uses_selected_unit_length_not_padded_tail():
    draft = np.zeros((50, 3), np.float32)
    reference = np.zeros_like(draft)
    draft[:7, 0] = 2.0
    reference[:7, 1] = 4.0  # valid-interval ratio is exactly two
    draft[7:, 0] = 1000.0
    reference[7:, 1] = 0.0
    assert norm_ratio_scale(draft, reference, 7) == 2.0


def test_norm_ratio_clips_to_configured_bounds():
    draft = np.ones((3, 3), np.float32)
    reference = np.ones((3, 3), np.float32) * 100
    assert norm_ratio_scale(draft, reference, 3) == 4.0
