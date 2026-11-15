"""Single source of the offline unseen LIBERO-90 rows (Dev 113 + Eval 619 windows) and their image orientation.

LIBERO's HDF5 demonstrations store the camera frames as rendered by the simulator, i.e. rotated 180 degrees
relative to the images pi0.5 was trained on.  The closed loop (``worker.py``, which runs
``anchor_flow/run_case.py``) therefore flips every frame with ``[::-1, ::-1]`` and pads it to 224 with
``image_tools.resize_with_pad`` before the policy sees it.  The first build of the offline rows
(``oracle-01/unseen_rows``) did not, so its ``image_base`` / ``image_wrist`` were upside down.  It is kept
unchanged for audit as ``LEGACY_UNSEEN_ROWS``; every consumer reads ``UNSEEN_ROWS``, built by
``prepare_oracle.py prepare`` with :func:`orient_libero_frame`.  See
docs/research/09_unseen_libero/image_orientation_fix.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
UNSEEN_ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
UNSEEN_ROWS = Path(os.environ.get('SKILLNET_UNSEEN_ROWS', UNSEEN_ROOT / 'unseen-rows-02')).expanduser()
LEGACY_UNSEEN_ROWS = UNSEEN_ROOT / 'oracle-01' / 'unseen_rows'      # images rotated 180 degrees; audit only
IMAGE_ORIENTATION = 'libero-hdf5 frame[::-1, ::-1] (as worker.py)'


def orient_libero_frame(frame):
    """Raw LIBERO frame (HDF5 ``agentview_rgb`` / ``eye_in_hand_rgb`` or env obs) -> the policy's orientation.

    Identical to the closed loop (anchor_flow/run_case.py, used by worker.py):
    ``np.ascontiguousarray(obs[...][::-1, ::-1])``.
    """
    return np.ascontiguousarray(np.asarray(frame)[::-1, ::-1])
