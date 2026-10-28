import json

import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.eval import gonogo
from skill_moe.skillnet.examples.emergent.eval.crossrecon import cross_reconstruction
from skill_moe.skillnet.examples.emergent.eval.data import load_z_run, split_rows
from skill_moe.skillnet.examples.emergent.eval.metrics import (
    MIConfig,
    action_motion_labels,
    assign,
    env_motion_labels,
    evaluate_mi,
    kmeans,
)
from skill_moe.skillnet.examples.emergent.eval.synthetic import make_synthetic_z_run

FAST = MIConfig(cluster_counts=(4, 8), primary_clusters=8, n_permutations=30, bootstrap_samples=200)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("z")
    good = make_synthetic_z_run(root / "good", n_tasks=8, episodes_per_task=6)
    blind = make_synthetic_z_run(root / "blind", n_tasks=8, episodes_per_task=6, prior_informative=False)
    return good, blind


def test_loader_marks_unexecuted_and_truncated_rows(runs):
    run = load_z_run(runs[0])
    assert run.horizon == 10
    assert not run.usable[~run.executed].any()
    assert not run.usable[run.interval_len < 10].any()
    assert np.all(np.isfinite(run.z["post"][run.usable]))
    assert np.all(np.isnan(run.motion[~run.has_future]))
    assert {"post", "prior", "untrained_post", "untrained_prior"} <= set(run.z)
    assert set(run.keys) == {"vlm_img_nolang", "vlm_img_lang"}
    q, s = split_rows(run)
    assert not set(run.init_state_id[q]) & set(run.init_state_id[s])
    # Derived target actions only use executed steps.
    assert run.target_mask[run.usable].all()


def test_aliases_map_existing_feature_keys(tmp_path, runs):
    import shutil
    dst = tmp_path / "alias"
    shutil.copytree(runs[0], dst)
    for f in (dst / "episodes").glob("*.npz"):
        d = dict(np.load(f))
        d["feat__X"] = d.pop("z_post_mean")
        d.pop("z_post_logvar")
        d.pop("z_interval_len")
        d.pop("z_executed")
        np.savez(f, **d)
    run = load_z_run(dst, aliases={"z_post_mean": "feat__X"})
    ref = load_z_run(runs[0])
    np.testing.assert_array_equal(run.usable, ref.usable)
    assert "z_post_logvar" in run.derived and "z_interval_len" in run.derived


def test_kmeans_and_motion_labels():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]], float)
    x = np.concatenate([c + rng.normal(0, .1, (30, 2)) for c in centers])
    a = assign(x, kmeans(x, 3, seed=0))
    assert len(np.unique(a)) == 3 and all(len(np.unique(a[i * 30:(i + 1) * 30])) == 1 for i in range(3))
    H = 4
    motion = np.zeros((2, H, 9))
    motion[..., 6] = 1.0  # identity quaternion
    motion[0, -1, 0] = 0.05     # +x
    motion[1, -1, 2] = -0.05    # -z
    motion[1, -1, 7:] = [-0.02, 0.02]  # finger gap shrinks by 0.04 -> closing
    lab = env_motion_labels(motion)
    assert lab["direction"].tolist() == [1, 6] and lab["gripper"].tolist() == [0, -1]
    actions = np.zeros((2, H, 7))
    actions[0, :, 1] = 1.0
    actions[1, :, 6] = [-1, -1, 1, 1]
    lab_a = action_motion_labels(actions, np.ones((2, H), bool))
    assert lab_a["direction"].tolist() == [3, 0] and lab_a["gripper"].tolist() == [0, -1]


def test_mi_separates_informative_prior_from_blind_prior(runs):
    good, blind = runs
    res = {}
    for name, path in (("good", good), ("blind", blind)):
        run = load_z_run(path)
        q, s = split_rows(run)
        res[name] = evaluate_mi(run, q, s, FAST)
    g, b = res["good"]["table"], res["blind"]["table"]
    assert g["prior"]["8"]["env/joint"]["shuffle_adjusted_bits"] > 0.5
    assert b["prior"]["8"]["env/joint"]["shuffle_adjusted_bits"] < g["prior"]["8"]["env/joint"]["shuffle_adjusted_bits"] - 0.4
    # Posterior is informative in both: it is the triviality reference, not the criterion.
    assert b["post"]["8"]["action/joint"]["shuffle_adjusted_bits"] > 0.5
    assert res["good"]["verdict"]["prior"]["vs_untrained"] == "above"
    assert res["blind"]["verdict"]["prior"]["vs_untrained"] != "above"
    json.dumps(res["good"], default=float)
    d_good = gonogo.decide(res["good"], None, None)
    d_blind = gonogo.decide(res["blind"], None, None)
    assert d_good["criteria"]["i_prior_mi"]["status"] == "pass"
    assert d_blind["criteria"]["i_prior_mi"]["status"] == "fail" and d_blind["overall"] == "no_go"


def test_cross_reconstruction_with_dummy_model(runs):
    run = load_z_run(runs[0])
    q, s = split_rows(run)
    calls = []
    # Dummy model: decodes actions from the nearest primitive code; ignores everything else.
    from skill_moe.skillnet.examples.emergent.eval.synthetic import PRIMITIVES
    codes = np.stack([np.nanmean(run.z["post"][run.usable][np.argmax(
        run.target_actions[run.usable][:, :, :3].sum(1) @ (PRIMITIVES[:, :3] * 100).T, axis=1) == p], axis=0)
        for p in range(len(PRIMITIVES))])

    def model(z, context):
        calls.append(len(z))
        assert set(context) >= {"row_index", "episode_id", "step", "proprio"}
        p = np.argmin(((z[:, None] - codes[None]) ** 2).sum(-1), axis=1)
        act = np.concatenate([PRIMITIVES[p, :3] * 100, np.zeros((len(z), 3)), PRIMITIVES[p, 3:4]], axis=1)
        return np.repeat(act[:, None], run.horizon, axis=1)

    res = cross_reconstruction(run, model, q, s, extra_z={"memory_blend": run.z["prior"]}, random_draws=3,
                               bootstrap_samples=200)
    m = res["mse"]
    assert m["own"] < m["random"] and m["oracle_motion"] < m["random"]
    assert {x["mode"] for x in res["improvement_over_prior"]} >= {"memory_blend", "oracle_motion"}
    assert all(c <= 256 for c in calls)
    assert not set(run.episode_id[res["query_rows"]]) & set(run.episode_id[np.concatenate(list(res["donor_rows"].values()))])
    with pytest.raises(ValueError):
        cross_reconstruction(run, lambda z, c: np.zeros((len(z), 1, 7)), q, s, random_draws=1)


def test_gonogo_rules_on_fabricated_results():
    def mi(minus_ci, untrained=True, motion_adj=1.0, task_adj=0.2):
        comps = ([{"source": "prior", "baseline": "untrained_prior", "label": "env/joint",
                   "estimate": 0.5, "ci95_familywise": minus_ci}] if untrained else [])
        return {"primary_clusters": 16, "comparisons_primary_k": comps,
                "table": {"prior": {"16": {"env/joint": {"raw_bits": 1.5, "shuffle_null_p95_bits": 0.4,
                                                         "shuffle_adjusted_bits": motion_adj},
                                           "task": {"shuffle_adjusted_bits": task_adj}}}}}
    ct = {"item1_constrained_retrieval": {"b_other_task": {"gap_closed": {
        "feat__z_prior": {"1": {"eef_relative_position": 0.3}},
        "feat__z_memory": {"1": {"eef_relative_position": 0.6}}}}}}
    cr = {"mse": {"prior": 1.0, "memory_blend": 0.8},
          "improvement_over_prior": [{"mode": "memory_blend", "estimate": 0.2, "ci95_familywise": [0.05, 0.3]}]}
    d = gonogo.decide(mi([0.1, 0.9]), ct, cr)
    assert d["overall"] == "go"
    assert gonogo.decide(mi(None, untrained=False), ct, cr)["overall"] == "undetermined"
    assert gonogo.decide(mi([0.1, 0.9], task_adj=2.0), ct, cr)["overall"] == "no_go"
    strict = gonogo.GoNoGoCriteria(crosstask_require="all")
    assert gonogo.decide(mi([0.1, 0.9]), ct, cr, strict)["criteria"]["ii_crosstask_gap"]["status"] == "fail"
    cr_bad = {"mse": {}, "improvement_over_prior": [{"mode": "memory_blend", "estimate": -0.1,
                                                     "ci95_familywise": [-0.2, 0.0]}]}
    assert gonogo.decide(mi([0.1, 0.9]), ct, cr_bad)["criteria"]["iii_cross_reconstruction"]["status"] == "fail"
    cr_wide = {"mse": {}, "improvement_over_prior": [{"mode": "memory_blend", "estimate": 0.1,
                                                      "ci95_familywise": [-0.2, 0.3]}]}
    assert gonogo.decide(mi([0.1, 0.9]), ct, cr_wide)["criteria"]["iii_cross_reconstruction"]["status"] == "undetermined"
    assert gonogo.decide(mi([0.1, 0.9]), {"skipped": "x"}, cr)["overall"] == "undetermined"
