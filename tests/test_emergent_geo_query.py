import numpy as np

from skill_moe.skillnet.examples.emergent.light.intermediate.geo_query import (
    _action_errors, _paired_episode_interval, _target_extent_metrics,
    fit_geometry_predictor, geometric_query_variants, oracle_geometric_targets,
    predict_geometry, topk_retrieve,
)


def test_geometric_queries_have_expected_components_and_train_statistics():
    kind = np.array([[1], [2], [1]], np.float32)
    delta = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 0]], np.float32)
    grip = np.array([[0], [1], [0]], np.float32)
    queries, means, scales = geometric_query_variants(kind, delta, grip)
    assert queries["Q1"].shape == (3, 5)
    assert queries["Q2"].shape == (3, 4)
    assert queries["Q3"].shape == (3, 3)
    assert np.allclose(queries["Q3"].mean(axis=0), 0, atol=1e-6)
    assert len(means["Q1"]) == len(scales["Q1"]) == 3


def test_retrieval_excludes_same_episode_and_softmax_blends():
    memory_q = np.array([[0.0], [1.0], [2.0]], np.float32)
    memory_b = np.array([[0, 0], [2, 0], [0, 2]], np.float32)
    one, blend, indices = topk_retrieve(memory_q, memory_b, np.array([[0.1]], np.float32),
                                        ["same", "other", "third"], ["same"], topk=2, temperature=10)
    assert indices[0, 0] == 1
    assert not np.array_equal(one[0], blend[0])
    assert np.allclose(blend[0].sum(), 2, atol=1e-4)


def test_softmax_weights_are_invariant_to_query_variant_distance_scale():
    memory_q = np.array([[0.0], [0.5], [1.0], [2.0]], np.float32)
    memory_b = np.arange(8, dtype=np.float32).reshape(4, 2)
    episodes = ["a", "b", "c", "d"]
    query_episode = ["query"]
    one, blend, indices = topk_retrieve(
        memory_q, memory_b, np.array([[0.2]], np.float32), episodes, query_episode, topk=3
    )
    one_scaled, blend_scaled, indices_scaled = topk_retrieve(
        memory_q * 100.0, memory_b, np.array([[20.0]], np.float32), episodes, query_episode, topk=3
    )
    np.testing.assert_array_equal(indices, indices_scaled)
    np.testing.assert_array_equal(one, one_scaled)
    np.testing.assert_allclose(blend, blend_scaled, atol=1e-6)


def test_target_extent_metrics_separate_switches_inside_and_outside_horizon():
    gold = np.array([[0, 0], [1, 1], [2, 2]], np.float32)
    retrieved = np.array([[0, 0], [0, 1], [4, 2]], np.float32)
    metrics = _target_extent_metrics(np.array([25, 26, 40]), retrieved, gold)
    assert metrics["within_horizon"] == {
        "n": 1, "mean_b_l2": 0.0, "median_offset_frames": 25.0,
    }
    assert metrics["beyond_horizon"]["n"] == 2
    assert np.isclose(metrics["beyond_horizon"]["mean_b_l2"], 1.5)


def test_oracle_searches_past_horizon_to_segment_end_and_censors_without_fallback(tmp_path):
    n = 60
    state = np.zeros((n, 8), np.float32)
    state[:, 0] = np.arange(n)
    state[:, 6] = 0.02
    state[:, 7] = -0.02
    action = np.zeros((n, 7), np.float32)
    action[:, 6] = -1
    action[12:45, 6] = 1
    ep = tmp_path / "episode.npz"
    np.savez(ep, step=np.array([0, 30, 50]), env_step=np.arange(n),
             proprio=state[[0, 30, 50]], env_state=state, action=action)
    (tmp_path / "manifest.json").write_text('{"episodes":[{"episode_id":"e","cache":"' + str(ep) + '"}]}')
    rows = {"episode_id": np.array(["e", "e", "e"]), "frame": np.array([0, 30, 50]),
            "skill_segment": np.array([0, 1, 1]),
            "s": np.stack([state[x] for x in (0, 30, 50)])}
    delta, grip, censored, offsets = oracle_geometric_targets(rows, cache_dir=tmp_path, horizon=10)
    assert np.allclose(delta[:2], [[12, 0, 0], [15, 0, 0]])
    assert np.allclose(offsets[:2], [12, 15])  # both switches occur beyond the action horizon
    assert grip[0, 0] == np.float32(0.04)  # mirrored finger qpos -> width is qpos[6] - qpos[7]
    assert censored[2]
    assert np.isnan(delta[2]).all()  # no endpoint-position substitute


def test_target_search_uses_full_rate_boundaries_even_when_later_rows_are_filtered(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 60
    state = np.zeros((n, 8), np.float32)
    state[:, 0] = np.arange(n)
    action = np.zeros((n, 7), np.float32)
    action[:, 6] = -1
    action[29:45, 6] = 1
    ep = tmp_path / "cache.npz"
    np.savez(ep, step=np.array([0, 30, 50]), env_step=np.arange(n),
             proprio=state[[0, 30, 50]], env_state=state, action=action)
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes":[{"episode_id":"e-0","cache":str(ep)}]}))
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table({"frame_index":np.arange(n),
                            "eval_skill_segment":np.repeat([0,1,2],[28,12,20])}),
                   tmp_path / "data" / "episode_000000.parquet")
    rows = {"episode_id":np.array(["e-0","e-0"]),"frame":np.array([0,30]),
            "skill_segment":np.array([0,1]),"s":state[[0,30]]}
    # Switch 29 is after the exact boundary 28, and switch 45 is after 40.
    # Stride-5 rows / filtering the final short chunk must not extend segments.
    delta, _, censored, _ = oracle_geometric_targets(rows,cache_dir=tmp_path,dataset_dir=tmp_path)
    assert censored.all()
    assert np.isnan(delta).all()


def test_geometry_fit_has_disjoint_episode_validation_train_only_scales_and_seed():
    import torch
    from skill_moe.skillnet.examples.emergent.light.intermediate.experiment import _episode_train_validation_indices
    torch.set_num_threads(1)
    episodes = np.repeat([f"e{i}" for i in range(6)], 3)
    x = np.arange(36,dtype=np.float32).reshape(18,2)
    y = np.column_stack([x, x[:,0]])
    fit, valid = _episode_train_validation_indices({"episode_id":episodes}, 14)
    x[valid] = 10000
    y[valid] = 20000
    args = dict(epochs=3, seed=3, episode_ids=episodes,batch_size=4)
    model,norm,classes = fit_geometry_predictor(x,y,np.arange(18)%2,**args)
    np.testing.assert_allclose(norm[0],x[fit].mean(0))
    np.testing.assert_allclose(norm[2],y[fit].mean(0))
    assert not set(model.fit_info['train_episodes']) & set(model.fit_info['validation_episodes'])
    # Initialization, not just minibatch order, must depend on the run seed.
    torch.manual_seed(999)
    model2,norm2,classes2 = fit_geometry_predictor(x,y,np.arange(18)%2,**args)
    np.testing.assert_array_equal(predict_geometry(model,norm,classes,x)[0],
                                  predict_geometry(model2,norm2,classes2,x)[0])


def test_action_error_is_per_query_mean_of_squared_errors_with_paired_noise(monkeypatch):
    from skill_moe.skillnet.examples.emergent.light.intermediate import geo_query as geo
    seeds = []
    def predict(m1,head,bases,f,s,*,t0,seed,draws):
        seeds.append(seed)
        # Averaging predictions first would falsely produce zero MSE.
        return np.stack([np.ones((len(bases),25,7)), -np.ones((len(bases),25,7))])
    monkeypatch.setattr(geo,"_predict_abstracted",predict)
    rows = {"f":np.zeros((3,1)),"s":np.zeros((3,8))}
    errors = _action_errors(None,None,np.zeros((3,12)),rows,np.zeros((3,25,7)),seed=43)
    assert set(errors) == {'0.0','0.2','0.4'}
    assert all(np.array_equal(e,np.ones(3)) for e in errors.values())
    assert seeds == [43,43,43]


def test_bootstrap_resamples_episodes_and_does_not_count_windows_as_independent():
    episodes = np.repeat([f"e{i}" for i in range(10)], 2)
    ref = np.repeat(np.arange(10,dtype=float), 2)
    result = _paired_episode_interval(ref,np.zeros(20),episodes)
    duplicate = _paired_episode_interval(np.repeat(ref,5),np.zeros(100),np.repeat(episodes,5))
    assert result == duplicate
    assert result['ci95'][0] < result['mean'] < result['ci95'][1]
    assert _paired_episode_interval(ref[:6],np.zeros(6),episodes[:6])['ci95'] is None


def test_normal_flow_starts_from_noise_without_gold_action_source(monkeypatch):
    from skill_moe.skillnet.examples.emergent.light.intermediate import geo_query as geo
    def integrate(flow,norm,rows,noises,sources,start_t,*,device,return_per_row):
        assert sources == {"normal":None}
        assert start_t == 1.0 and return_per_row
        assert len(noises) == 2 and noises[0].shape == (3,175)
        assert not np.array_equal(noises[0],rows['a'].reshape(3,-1))
        return {}, {'normal':np.arange(3.)}
    monkeypatch.setattr(geo,'integrate_flow_sources',integrate)
    errors = geo._normal_action_errors(None,{}, {'a':np.zeros((3,25,7))},seed=51,device='cpu')
    np.testing.assert_array_equal(errors,np.arange(3.))


def test_unit_exact_does_not_assign_outside_epsilon_to_nearest_center():
    from skill_moe.skillnet.examples.emergent.light.intermediate.geo_query import _m6_unit_metrics
    class Units:
        epsilon = 0.1
        def _nearest(self,b): return 0,float(b[0])
        def _functional_distance(self,a,b): return float(np.square(a-b).sum())
    result = _m6_unit_metrics(Units(), np.array([[.2],[.01]]),np.array([[.3],[.02]]))
    assert result['unit_exact'] == .5
