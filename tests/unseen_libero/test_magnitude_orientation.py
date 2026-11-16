"""MAG3 geometry and server/offline draft parity checks."""
from __future__ import annotations

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import magnitude_orientation as mag3


def test_minimal_row_rotation_maps_leading_direction():
    rotation, ok = mag3._minimal_row_rotation(np.array([1., 0., 0.]), np.array([0., 1., 0.]))
    assert ok
    np.testing.assert_allclose(np.array([1., 0., 0.]) @ rotation, [0., 1., 0.], atol=1e-10)


def test_server_pose_hook_matches_offline_on_ten_dev_windows(tmp_path):
    pack = mag3._pool_w()
    selected = {'pool': 'scale_only_W', 'orientation': 'O1_source', 'amplitude': 'A2_history'}
    result = mag3.verify_server_parity(pack, selected, tmp_path, n_windows=10)
    assert result['pass']
    assert result['windows'] >= 10
    assert result['max_abs_6axis'] <= 1e-5
