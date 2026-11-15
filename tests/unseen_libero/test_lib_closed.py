import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.probes import lib_closed as lc


def test_metric_uses_only_allowed_candidates_and_returns_recovery():
    score = np.array([[.1, .9, .8], [.8, .7, .6]], np.float32)
    err = np.array([[.01, .9, .2], [.5, .1, .4]], np.float32)
    allowed = np.array([[True, False, True], [True, True, False]])
    m = lc._metric_retrieval(score, err, allowed)
    # Row 1's invalid high score must not be selected; row 2's best allowed is index 1.
    assert np.array_equal(m["rank"][:, 0], [2, 0])
    assert np.allclose(m["per_row_error"], [.2, .5])
    assert np.isfinite(m["recovery"]).all()


def test_episode_bootstrap_is_deterministic_and_paired():
    a = np.array([2., 4., 6., 8.]); b = np.array([1., 2., 3., 4.])
    ep = np.array([1, 1, 2, 2])
    x = lc._bootstrap_diff(a, b, ep, reps=100, seed=7)
    y = lc._bootstrap_diff(a, b, ep, reps=100, seed=7)
    assert x == y
    assert np.allclose(x, [1.5, 3.5])


def test_libero10_mapping_is_fixed_and_complete():
    assert sorted(lc.LONG_TO_40) == list(range(10))
    assert len(set(lc.LONG_TO_40.values())) == 10


def test_each_condition_is_pinned_to_its_own_policy_server():
    servers=[("proc-r5",41001,"R5"),("proc-state",41002,"state_only")]
    assert lc._server_for_mode(servers,"R5")==servers[0]
    assert lc._server_for_mode(servers,"state_only")==servers[1]


def test_pose_unit_batch_indexes_pool_rows_as_three_dimensional_chunks():
    class VS:
        @staticmethod
        def to_physical(x): return x + 10
    class TF:
        @staticmethod
        def batched_procrustes(source, target):
            assert source.shape == (1, 10, 6)
            assert target.shape == (1, 10, 6)
            return (target[:, 0, 0], None, None)
        @staticmethod
        def decode_many(shape, theta, vs):
            assert shape.shape == (1, 10, 7)
            return shape + theta[:, None, None]
    shapes=np.arange(3*10*7,dtype=np.float32).reshape(3,10,7)
    ref=np.zeros((1,10,7),np.float32)
    got=lc._pose_unit_batch(shapes,[2],ref,VS,TF)
    assert got.shape==(1,10,7)
    assert np.allclose(got[0],shapes[2]+10)


def test_same_trial_pair_requires_identical_seed_and_initial_state():
    a={"trial_seed":7,"environment_seed":7,"initial_state_sha256":"abc"}
    lc._assert_same_trial(a,dict(a))
    bad=dict(a,trial_seed=8)
    try:
        lc._assert_same_trial(a,bad)
    except AssertionError as e:
        assert "trial_seed" in str(e)
    else:
        raise AssertionError("seed mismatch should fail paired comparison")

