from __future__ import annotations

import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.light.intermediate import review_tests as rt


def test_motion_labels_and_cross_state_transfer_are_posthoc():
    actions = np.zeros((3, 10, 7), np.float32)
    actions[:, :, 2] = 0.2
    actions[:, :, 6] = 1
    bases = np.ones((3, 2), np.float32)
    states_a = np.zeros((3, 4), np.float32)
    states_b = np.ones((3, 4), np.float32)

    def decode(b, s, r):
        out = np.zeros((len(b), 10, 7), np.float32)
        out[:, :, 2] = b[:, 0, None]
        out[:, :, 6] = 1
        return out

    result = rt.r1_transfer_diagnostics(bases, states_a, states_b, actions, decode)
    assert result["agreement"] == {"direction": 1.0, "gripper": 1.0, "rotation": 1.0}
    assert result["cross_state_agreement"]["direction"] == 1.0
    assert result["labels_are_posthoc_only"]


def test_residual_ablation_compares_zero_and_paired_normal_draw():
    b = np.ones((4, 1), np.float32)
    r = np.ones((4, 1), np.float32)
    s = np.zeros((4, 2), np.float32)
    target = np.full((4, 1), 2., np.float32)
    decode = lambda bb, rr, ss: bb + rr
    result = rt.residual_ablation_metrics(decode, b, r, s, target, seed=4)
    assert result["r_encoded_mse"] == 0
    assert result["r_zero_mse"] == 1
    assert result["r_standard_normal_mse"] >= 0


def test_r3_functional_state_sets_and_mean_medoid():
    bases = np.array([[0.0], [1.0], [2.0]], np.float32)
    states = [np.zeros((1, 1), np.float32)] * 3
    decode = lambda b, s, r: np.repeat(b, 2, axis=1) + np.repeat(s, 2, axis=1)
    result = rt.functional_distance_grid(bases, states, decode,
                                        common_states=np.zeros((2, 1), np.float32))
    assert set(result) == {"mean/common", "mean/members", "medoid/common", "medoid/members"}


def test_multi_positive_infonce_and_topk_candidates():
    logits = np.array([[3., 2., 0.], [0., 1., 2.]])
    positive = np.array([[1, 1, 0], [0, 1, 1]], bool)
    loss = rt.multipositive_infonce(logits, positive)
    assert loss > 0
    ids, weights = rt.topk_candidates(logits, 2)
    assert ids.shape == weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 1)


def test_confidence_risk_coverage_and_t0_binning():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = x[:, 0] * .1 + x[:, 1] * .2
    model = rt.fit_confidence_model(x[:16], y[:16])
    pred = rt.predict_confidence_model(model, x[16:])
    curve = rt.risk_coverage(y[16:], pred)
    bins = rt.t0_confidence_bins(pred)
    assert len(curve) == 6 and curve[0]["coverage"] == .1
    assert sum(bins["counts"]) == 4
    assert model["fit_scope"] == "training-suite held-out episodes only"


def test_r6_grid_and_r7_bridge_equations():
    grid = rt.independent_t0_n_grid()
    assert len(grid) == 30
    assert {x["n"] for x in grid} == {1, 2, 3, 5, 10}
    a = np.array([[1., 2.]], np.float32)
    anchor = np.array([[3., 4.]], np.float32)
    eps = np.array([[2., -2.]], np.float32)
    xt, velocity = rt.bridge_training_targets(a, anchor, eps, np.array([1.]), .1)
    assert np.allclose(xt, anchor + .1 * eps)
    assert np.allclose(velocity, xt - a)


def test_r8_raw_memories_have_identical_budget_and_are_deterministic():
    keys = np.random.default_rng(3).normal(size=(100, 8)).astype(np.float32)
    selected = rt.select_equal_count_raw_memory(keys, 12, seed=9)
    again = rt.select_equal_count_raw_memory(keys, 12, seed=9)
    assert len(selected["random_ids"]) == len(selected["kmedoids_ids"]) == 12
    assert len(np.unique(selected["kmedoids_ids"])) == 12
    assert np.array_equal(selected["random_ids"], again["random_ids"])
    assert np.array_equal(selected["kmedoids_ids"], again["kmedoids_ids"])


def test_r8_d2_raw_memory_is_episode_block_prequential():
    class FakeFairCompare:
        HORIZON = 1
        @staticmethod
        def _nrm(x):
            return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        @staticmethod
        def cosine_stats(query, keys, tau, *, device):
            sims = FakeFairCompare._nrm(query) @ FakeFairCompare._nrm(keys).T
            ids = np.argmax(sims, axis=1)
            return ids, sims.max(axis=1), np.zeros(len(query))
        @staticmethod
        def rollout(model, norm, f, s, anchors, eps, t0, n, *, device):
            model.seen_anchor_blocks.append(np.asarray(anchors).copy())
            return np.broadcast_to(np.asarray(anchors)[None], (eps.shape[0], *anchors.shape)).copy()
        @staticmethod
        def row_metrics(pred, target, normal, scales, *, device):
            mse = np.mean((pred - target[None]) ** 2, axis=(0, 2, 3))
            return {"mse": mse}
        @staticmethod
        def steps_for(t0):
            return np.ones(len(t0), dtype=np.int64)

    class Model:
        seen_anchor_blocks = []

    rows = {"episode_id": np.array(["a", "a", "b", "b"]),
            "frame": np.arange(4), "a": np.arange(4, dtype=np.float32).reshape(4, 1, 1),
            "f": np.zeros((4, 1), np.float32), "s": np.zeros((4, 1), np.float32)}
    keys = np.eye(4, dtype=np.float32)
    result, prediction = rt._eval_raw_d2_prequential(
        FakeFairCompare, Model(), {}, rows, keys, np.zeros((2, 4, 1), np.float32), {},
        np.zeros(4, np.float32), np.eye(3, 4, dtype=np.float32),
        np.arange(3, dtype=np.float32).reshape(3, 1, 1), np.array([0, 1, 2, 3]),
        budget=2, seed=0, device="cpu")
    assert result["random_ids"]["episode_block_prequential"]
    assert prediction["random_ids"].shape == (4, 1, 1)
    # Both rows in an episode are answered in one call before the next block.
    assert [len(x) for x in Model.seen_anchor_blocks] == [2, 2, 2, 2]


def test_report_run_refuses_overwrite_and_records_reference(tmp_path):
    # The production baseline is audited separately by the manual run.
    import json
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    cell = {method: {metric: {"mean": .1, "sd": .01} for metric in
                     ("mse", "best8", "pos_term_mm", "grip_agree")}
            for method in ("N:noise:n10", "A:noise:n10", "A:unit:cal_sim", "N:raw:cal_sim",
                           "A:shuf:cal_sim", "A:gold_unit:cal_sim")}
    aggregate = {f"{split}|instruction|D3|all": cell for split in ("A", "B")}
    (baseline / "results.json").write_text(json.dumps({"runs": [{}, {}, {}, {}, {}, {}],
        "aggregate": aggregate, "duration_minutes": 1}))
    np.savez(baseline / "per_row.npz", dummy=np.zeros(1))
    out = tmp_path / "new"
    rt.run(out, baseline)
    assert (out / "results.json").is_file()
    assert (out / "summary.md").is_file()
    assert (out / "figures/fair_compare_d3_baseline.png").is_file()
    with pytest.raises(FileExistsError):
        rt.run(out, baseline)
