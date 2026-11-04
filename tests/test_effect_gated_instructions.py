from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/skill_conditioning"))
from effect_controller import Belief, InstructionController, causal_vector


def test_temporal_verification_uncertainty_and_regression_are_distinct():
    belief = Belief()
    assert [belief.update(p) for p in [.1, .1, .9, .9, .5, .1, .1]] == [
        "unknown", "unmet", "unknown", "achieved", "unknown", "unknown", "regressed"]
    assert belief.update(.9) == "unknown"
    assert belief.update(.9) == "achieved"


def test_effect_gate_rejects_uncertain_and_harmful_updates_then_rolls_back():
    controller = InstructionController("effect")
    assert not controller.decide([.95, .1], 100, (.4, -.1))[0]
    assert not controller.decide([.95, .1], 110, (.4, -.1))[0]
    assert not controller.decide([.95, .1], 120, (.01, .005))[0]
    assert controller.decide([.95, .1], 130, (.3, .1))[0]
    assert controller.decide([.1, .1], 140, (.3, .1))[0]
    assert not controller.decide([.1, .1], 150, (.3, .1))[0]
    controller.decide([.95, .1], 160, (.3, .1))
    assert not controller.decide([.95, .1], 170, (.3, .1))[0]
    assert controller.switch_count == controller.rollback_count == 1


def test_simple_baseline_does_not_consult_effect_predictor():
    controller = InstructionController("achievement")
    assert not controller.decide([.95, .1], 100, (-1., -1.))[0]
    assert controller.decide([.95, .1], 110, (-1., -1.))[0]
    original = InstructionController("original")
    assert not original.decide([1., 1.], 800, (1., 1.))[0]


def test_runtime_feature_contract_contains_only_causal_inputs_and_budget():
    vector = causal_vector(np.ones(2048), np.arange(8), np.arange(8)-1, np.ones(7)*2, 400)
    assert vector.shape == (2072,)
    np.testing.assert_array_equal(vector[2056:2064], np.ones(8))
    np.testing.assert_array_equal(vector[2064:2071], np.ones(7)*2)
    assert vector[-1] == .5


def test_incomplete_matching_arm_is_not_a_policy_failure_or_an_extra_pair():
    from summarize_effect_experiment import paired_statistics
    rows = {(0,20): {"effect": {"success": True}, "original": {"success": False}},
            (1,20): {"original": {"success": False}}}
    result = paired_statistics(rows, "effect", "original")
    assert result["n"] == result["rescue"] == 1
    assert result["harm"] == 0
    assert result["difference"] == 1.
