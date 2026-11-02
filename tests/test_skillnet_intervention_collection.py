from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/skill_conditioning"))
import collect_interventions as collection
from collect_interventions import counterfactual_inputs
from analyze_interventions import routing_summary, repeat_summary


def test_fixed_observation_interventions_change_only_the_designated_input():
    raw = {"skills": "[0, 1, 3]", "objects": "[1, 1, 0]", "prompt": "full instruction",
           "s2_rng_seed": 700001, "observation/state": np.ones(8), "observation/image": np.zeros((2, 2, 3))}
    variants = counterfactual_inputs(raw, "close drawer")
    expected = {"reverse_skills": "skills", "wrong_skill_ids": "skills", "no_skills": "diagnostic_no_skills",
                "remaining_instruction": "prompt", "noise_only": "s2_rng_seed"}
    for name, v in variants.items():
        changed = {k for k in set(raw) | set(v) if k not in raw or not np.array_equal(raw[k], v[k])}
        assert changed == {expected[name]}
        np.testing.assert_array_equal(v["observation/state"], raw["observation/state"])
        assert v["objects"] == raw["objects"]
    assert raw["skills"] == "[0, 1, 3]"


def record(placement, routes):
    return {"stage_truth": [False, placement, False], "response": {"diagnostics": {
        "router_top1": routes, "selected_contribution_norm": np.ones((18, 10)), "shared_output_norm": np.ones((18, 10))}}}


def test_routing_uses_only_executed_tokens_and_separates_placement_truth():
    routes = np.zeros((18, 10), dtype=int);routes[:, 5:] = 3
    counts, norms, exposures = routing_summary([record(False, routes), record(True, np.ones((18, 10), int))])
    assert exposures.tolist() == [5, 5]
    assert np.all(counts[0, :, 0] == 5) and np.all(counts[0, :, 3] == 0)
    assert np.all(counts[1, :, 1] == 5)
    with pytest.raises(ValueError):routing_summary([record(False, np.zeros((17, 10)))])


def test_repeat_control_reports_changed_inputs_and_unmatched_lengths():
    a = [{"step": 100, "image_sha256": "a", "response": {"actions": [[1., 2.]]}}]
    b = [{"step": 100, "image_sha256": "b", "response": {"actions": [[1., 3.]]}},
         {"step": 105, "image_sha256": "c", "response": {"actions": [[1., 3.]]}}]
    r = repeat_summary(a, b)
    assert r["common_queries"] == 1 and r["queries_repeat"] == 2
    assert not r["all_policy_images_identical"] and r["max_action_difference"] == 1


@pytest.mark.parametrize("shape,valid", [((7, 32), True), ((10, 32), True), ((4, 32), False), ((7, 6), False)])
def test_rpc_respects_public_five_step_execution_contract(monkeypatch, shape, valid):
    monkeypatch.setattr(collection.msgpack_numpy, "unpackb", lambda _: {"actions": np.zeros(shape)})
    client = SimpleNamespace(_packer=SimpleNamespace(pack=lambda _: b""),
                             _ws=SimpleNamespace(send=lambda _: None, recv=lambda timeout: b""))
    if valid:
        assert collection.rpc(client, {})["actions"].shape == shape
    else:
        with pytest.raises(ValueError, match="invalid action"):
            collection.rpc(client, {})


def test_rpc_rejects_nonfinite_actions(monkeypatch):
    monkeypatch.setattr(collection.msgpack_numpy, "unpackb", lambda _: {"actions": np.full((7, 32), np.nan)})
    client = SimpleNamespace(_packer=SimpleNamespace(pack=lambda _: b""),
                             _ws=SimpleNamespace(send=lambda _: None, recv=lambda timeout: b""))
    with pytest.raises(ValueError, match="invalid action"):
        collection.rpc(client, {})
