"""Checkout-relative paths shared by the research experiment entry points."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path(os.environ.get("SKILLNET_STATE_DIR", REPO_ROOT / ".skillnet-state")).expanduser().resolve()


# Historical S2 assets shared by the offline experiment entry points.
S2_DEFAULT_HELDOUT = str(
    STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/calibration/fit-v2/heldout-observations.npz'
)
S2_DEFAULT_ROWS = str(
    STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz'
)
S2_DEFAULT_MEMORY = str(
    STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/memory/initial_libero90.pkl'
)
S2_DEFAULT_CHECKPOINT = str(
    STATE_DIR / 'checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params'
)
