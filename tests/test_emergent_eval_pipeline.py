import json

import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.metrics import MIConfig
from skill_moe.skillnet.examples.emergent.eval.pipeline import run_all
from skill_moe.skillnet.examples.emergent.eval.retrieval import write_crosstask_view
from skill_moe.skillnet.examples.emergent.eval.synthetic import make_synthetic_z_run


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    run = make_synthetic_z_run(root / "run", n_tasks=8, episodes_per_task=6)
    res = run_all(run, root / "out", bootstrap_samples=200,
                  mi_config=MIConfig(cluster_counts=(8,), primary_clusters=8, n_permutations=20, bootstrap_samples=200))
    return root, res


def test_pipeline_writes_outputs_and_gonogo(result):
    root, res = result
    out = root / "out"
    for name in ("results.json", "report.md", "crosstask_report.md", "memory_nolang/store.npz",
                 "memory_lang/key_encoder.npz", "memory_log_nolang.jsonl"):
        assert (out / name).exists(), name
    saved = json.loads((out / "results.json").read_text())
    assert saved["gonogo"]["overall"] in ("go", "no_go", "undetermined")
    assert set(saved["gonogo"]["criteria"]) == {"i_prior_mi", "ii_crosstask_gap", "iii_cross_reconstruction"}
    assert res["gonogo"]["criteria"]["i_prior_mi"]["status"] == "pass"
    assert res["gonogo"]["criteria"]["ii_crosstask_gap"]["status"] == "pass"
    report = (out / "report.md").read_text()
    assert "Go/no-go" in report and "reference (saw actions)" in report
    assert "memory" in res["mi"]["table"] and "memory_lang" in res["mi"]["table"]


def test_replay_never_registers_unexecuted_rows(result):
    root, res = result
    for tag in ("nolang", "lang"):
        rq = res["memory"][tag]["replay_query_split"]
        assert rq["unexecuted_or_truncated_added"] == 0
        assert rq["reasons"].get("not_executed", 0) > 0
        assert 0.0 < rq["add_rate_eligible"] < 0.35
        h = res["memory"][tag]["replay_held_out_suite"]
        assert h["add_rate_unseen_suite"] > h["add_rate_seen_suite"]
    log = [json.loads(x) for x in (root / "out" / "memory_log_nolang.jsonl").read_text().splitlines()]
    assert all(not e["added"] for e in log if e["reason"] in ("not_executed", "interval_truncated"))


def test_crosstask_view_rejects_future_rows_that_are_not_usable(tmp_path):
    path = make_synthetic_z_run(tmp_path / "run", n_tasks=4, episodes_per_task=3)
    run = load_z_run(path)
    i = int(np.flatnonzero(run.usable)[0])
    run.usable[i] = False
    with pytest.raises(ValueError, match="full env future"):
        write_crosstask_view(run, tmp_path / "view")
