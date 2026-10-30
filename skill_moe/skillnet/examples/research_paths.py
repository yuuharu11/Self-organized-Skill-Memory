"""Checkout-relative paths shared by the research experiment entry points."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path(os.environ.get("SKILLNET_STATE_DIR", REPO_ROOT / ".skillnet-state")).expanduser().resolve()
