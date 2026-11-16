import numpy as np
import json
from types import SimpleNamespace

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.eef_frame_units import (
    axis_angle_matrix,
    d1_summary,
    from_eef_normalized,
    from_eef_physical,
    low_frequency_fraction,
    normalized_state_rotvec,
    to_eef_normalized,
    to_eef_normalized_at_rotvec,
    to_eef_physical,
    _u_source_rotations,
    _unseen_segment_rotvecs,
)


def test_to_from_eef_round_trip_physical():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(9, 10, 6)).astype(np.float32)
    r0 = rng.normal(size=(9, 3)).astype(np.float32)
    got = from_eef_physical(to_eef_physical(x, r0), r0)
    np.testing.assert_allclose(got, x, atol=2e-6, rtol=2e-6)


def test_source_and_query_pose_equal_returns_original_delta():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(10, 6)).astype(np.float32)
    r0 = rng.normal(size=3).astype(np.float32)
    np.testing.assert_allclose(from_eef_physical(to_eef_physical(x, r0), r0), x, atol=2e-6)


def test_rotation_direction_matches_column_vector_eef_definition():
    # A +90 degree world-from-EEF pose maps a world +x delta to EEF -y
    # under R^T, then maps it back to world +x under R.
    r0 = np.array([0., 0., np.pi / 2], np.float32)
    x = np.array([[1., 0., 0., 1., 0., 0.]], np.float32)
    to = to_eef_physical(x, r0)
    np.testing.assert_allclose(to[:, :3], [[0., -1., 0.]], atol=2e-6)
    np.testing.assert_allclose(to[:, 3:6], [[0., -1., 0.]], atol=2e-6)
    back = from_eef_physical(to, r0)
    np.testing.assert_allclose(back, x, atol=2e-6)


def test_normalized_round_trip_keeps_gripper_and_uses_query_pose():
    rng = np.random.default_rng(16)
    action_stats = {'q01': np.array([-1, -1, -1, -.2, -.2, -.2, -1], np.float32),
                    'q99': np.array([1, 1, 1, .2, .2, .2, 1], np.float32)}
    state_stats = {'q01': np.array([-1, -1, -1, -np.pi, -np.pi, -np.pi], np.float32),
                   'q99': np.array([1, 1, 1, np.pi, np.pi, np.pi], np.float32)}
    x = rng.uniform(-.5, .5, (4, 10, 7)).astype(np.float32)
    states = rng.uniform(-.3, .3, (4, 8)).astype(np.float32)
    got = from_eef_normalized(to_eef_normalized(x, states, action_stats, state_stats), states,
                              action_stats, state_stats)
    np.testing.assert_allclose(got, x, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(got[..., 6], x[..., 6])


def test_low_frequency_fraction_is_invariant_to_per_unit_rotation():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(20, 10, 6)).astype(np.float32)
    rotations = rng.normal(size=(20, 3)).astype(np.float32)
    y = np.stack([to_eef_physical(a, r) for a, r in zip(x, rotations, strict=True)])
    np.testing.assert_allclose(low_frequency_fraction(x), low_frequency_fraction(y), atol=2e-7, rtol=2e-7)


def test_axis_angle_matrix_is_orthogonal():
    r = axis_angle_matrix(np.array([[.2, -.3, .4], [0., 0., 0.]], np.float32))
    np.testing.assert_allclose(r @ np.swapaxes(r, -1, -2), np.broadcast_to(np.eye(3), (2, 3, 3)), atol=2e-6)


def test_d1_counts_are_cluster_counts_on_equal_full_pool():
    rng = np.random.default_rng(18)
    actions = rng.normal(size=(12, 10, 6)).astype(np.float32)
    tasks = np.repeat(np.arange(3), 4)
    result = d1_summary(actions, tasks, counts=(2, 4), seed=19)
    assert result['unit_count'] == len(actions)
    assert result['cluster_separation']['2']['k_requested'] == 2
    assert result['cluster_separation']['2']['k_used'] == 2
    assert result['cluster_separation']['4']['k_used'] == 4
    assert 0 <= result['nearest_cross_task_fraction'] <= 1


def test_exact_source_u_pose_uses_arbitrary_frame_not_floor_stride(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = tmp_path / 'run'; dataset = tmp_path / 'dataset'; run.mkdir()
    (dataset / 'data' / 'chunk-000').mkdir(parents=True)
    (run / 'data_manifest.json').write_text(json.dumps({'source': str(dataset)}))
    states = np.zeros((15, 8), np.float32)
    states[:, 3] = np.arange(15, dtype=np.float32) * .1
    table = pa.table({'state': states.tolist(), 'frame_index': np.arange(15, dtype=np.int64),
                      'task_index': np.full(15, 2, np.int64)})
    pq.write_table(table, dataset / 'data/chunk-000/episode_000007.parquet')
    pool = {'task': np.array([2]), 'episode': np.array([7]), 'start': np.array([3])}
    rotvec, _ = _u_source_rotations(pool, tmp_path, SimpleNamespace(RUN=run), {})
    np.testing.assert_allclose(rotvec[0], [.3, 0, 0], atol=1e-7)


def test_unseen_query_pose_reads_exact_segment_start_frame(tmp_path):
    import h5py

    h5 = tmp_path / 'demo.h5'
    with h5py.File(h5, 'w') as f:
        ori = np.zeros((12, 3), np.float32)
        ori[:, 1] = np.arange(12, dtype=np.float32) * .1
        demo = f.create_group('data').create_group('demo_0').create_group('obs')
        demo.create_dataset('ee_ori', data=ori)
    manifest = {'source_records': [{'task_id': 6, 'episode': 1, 'source_hdf5': str(h5), 'source_demo': 'demo_0'}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    inp = {'task': np.array([6]), 'episode': np.array([1]), 'start': np.array([3])}
    rotvec = _unseen_segment_rotvecs(inp, tmp_path)
    np.testing.assert_allclose(rotvec[0], [0, .6, 0], atol=1e-7)


def test_normalized_transform_exact_rotvec_matches_state_based_transform():
    rng = np.random.default_rng(20)
    action_stats = {'q01': np.full(7, -.2, np.float32), 'q99': np.full(7, .2, np.float32)}
    state_stats = {'q01': np.full(6, -np.pi, np.float32), 'q99': np.full(6, np.pi, np.float32)}
    states = rng.uniform(-.3, .3, (3, 8)).astype(np.float32)
    actions = rng.normal(size=(3, 10, 7)).astype(np.float32)
    rotvec = normalized_state_rotvec(states, state_stats)
    np.testing.assert_allclose(to_eef_normalized_at_rotvec(actions, rotvec, action_stats),
                               to_eef_normalized(actions, states, action_stats, state_stats), atol=1e-6)
