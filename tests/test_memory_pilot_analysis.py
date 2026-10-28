import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot import analyze_offline as analysis


def test_skill_shuffle_changes_content_not_just_an_unordered_set():
    for seed in range(10):
        shuffled = analysis._shuffle_skill_ids([0, 1], [0, 1], np.random.default_rng(seed))
        assert set(shuffled) <= {0, 1}
        assert sorted(shuffled) != [0, 1]


def test_g1_distinguishes_off_segment_outputs_and_identical_endpoints():
    alphas = np.linspace(0, 1, 5)
    path = np.array([[-1, 0], [-.5, 1], [0, 2], [.5, 1], [1, 0]])[:, None, :]
    result = analysis._g1_path_metrics(path, alphas)
    assert result["g1_endpoint_l2"] == 2
    np.testing.assert_allclose(result["g1_endpoint_projection_by_alpha"], alphas)
    np.testing.assert_allclose(result["g1_distance_to_endpoint_segment_by_alpha"], [0, 1, 2, 1, 0])
    loop = np.array([0, 1, 2, 1, 0])[:, None, None]
    result = analysis._g1_path_metrics(loop, alphas)
    assert result["g1_endpoint_projection_by_alpha"] is None
    assert result["g1_distance_to_endpoint_segment_by_alpha"][2] == 2


def test_limited_replay_covers_the_episode_instead_of_only_its_start():
    np.testing.assert_array_equal(analysis._replay_indices(101, 3), [0, 50, 100])
    np.testing.assert_array_equal(analysis._replay_indices(2, 3), [0, 1])
    np.testing.assert_array_equal(analysis._replay_indices(3, 0), [0, 1, 2])


def _recording(tmp_path):
    run = tmp_path / "input"
    (run / "episodes").mkdir(parents=True)
    checkpoint = tmp_path / "recorded-checkpoint"
    (run / "manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "suite_skill_ids": [0, 1],
        "episodes": [{"episode_id": "ep", "init_state_id": 0, "task": "test"}]}))
    np.savez(run / "episodes" / "ep.npz", step=np.array([10]),
             skills=np.array([[1, 2, 0, 0]]), skill_mask=np.array([[True, True, False, False]]),
             img_base=np.zeros((1, 2, 2, 3), dtype=np.uint8), img_wrist=np.zeros((1, 2, 2, 3), dtype=np.uint8),
             proprio=np.arange(8, dtype=np.float32)[None, :], prompt=np.array(["test"]))
    return run, checkpoint


def test_replay_conditions_share_observation_noise_and_only_intended_skills_change(tmp_path, monkeypatch):
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType

    run, checkpoint = _recording(tmp_path)

    @dataclass
    class Config:
        model: object = None
        exp_name: str = "test"

    class Policy:
        def __init__(self):
            self._sample_kwargs = {"capture_auxiliary": True}
            self.calls = []
            self.table = np.arange(30, dtype=np.float32).reshape(6, 5)
            self._input_transform = LiberoInputs(ModelType.PI0)
            self._model = SimpleNamespace(PaliGemma=SimpleNamespace(llm=self.embed))
        def embed(self, tokens, method):
            assert method == "skill_embed"
            return self.table[np.asarray(tokens)]
        def infer(self, raw, noise):
            self.calls.append((raw, np.array(noise, copy=True), self._sample_kwargs["capture_auxiliary"]))
            result = {"actions": np.array(noise, copy=True)}
            if self._sample_kwargs["capture_auxiliary"]:
                result.update(memory_pilot_selected_expert_contribution_norm=np.ones((2, 10)),
                              memory_pilot_shared_expert_norm=np.full((2, 10), 4.),
                              memory_pilot_router_top1=np.zeros((2, 10), dtype=np.int32))
            return result

    policy = Policy()
    observed_checkpoints = []
    def factory(config, selected_checkpoint, **kwargs):
        observed_checkpoints.append(selected_checkpoint)
        return policy
    monkeypatch.setattr(analysis.config_moe_skill, "get_config",
                        lambda _: Config(model=SimpleNamespace(action_dim=7, skill_num=6)))
    monkeypatch.setattr(analysis, "create_trained_policy", factory)
    monkeypatch.setattr(analysis, "STATE", tmp_path)
    monkeypatch.setattr("sys.argv", ["analysis", str(run), "--g0", "--g1"])
    analysis.main()
    assert observed_checkpoints == [checkpoint]
    assert len(policy.calls) == 12
    for i, (raw, noise, capture) in enumerate(policy.calls):
        for key in ("observation/state", "observation/image", "observation/wrist_image", "prompt"):
            np.testing.assert_array_equal(raw[key], policy.calls[0][0][key])
        if i == 5:
            assert not np.array_equal(noise, policy.calls[0][1])
        else:
            np.testing.assert_array_equal(noise, policy.calls[0][1])
        assert capture == (i != 1)
    assert sorted(json.loads(policy.calls[3][0]["skills"])) != [0, 1]
    assert not np.any(policy.calls[4][0]["skill_mask_override"])
    for alpha, (raw, _, _) in zip([0, .25, .5, .75, 1], policy.calls[6:11]):
        np.testing.assert_array_equal(raw["skill_mask_override"], [True, False, False, False])
        np.testing.assert_allclose(raw["skill_embed_override"][0], (1 - alpha) * policy.table[1] + alpha * policy.table[2])
        assert not np.any(raw["skill_embed_override"][1:])
    set_raw = policy.calls[11][0]
    np.testing.assert_array_equal(set_raw["skill_mask_override"], [True, True, False, False])
    np.testing.assert_array_equal(set_raw["skill_embed_override"][:2], policy.table[[1, 2]])
    assert not np.any(set_raw["skill_embed_override"][2:])
    result = next((tmp_path / "outputs" / "memory-pilot").glob("analysis-*/metrics.json"))
    metric = json.loads(result.read_text())[0]
    assert metric["g1_skill_ids"] == [0, 1]
    assert metric["capture_disabled_l2_all"] == 0
    assert "g1_distance_to_endpoint_segment_by_alpha" in metric


def test_mismatched_replay_checkpoint_fails_before_loading_model(tmp_path, monkeypatch):
    run, _ = _recording(tmp_path)
    def unexpected_load(*args, **kwargs):
        pytest.fail("mismatched checkpoint must not be loaded")
    monkeypatch.setattr(analysis, "create_trained_policy", unexpected_load)
    monkeypatch.setattr("sys.argv", ["analysis", str(run), "--checkpoint", str(tmp_path / "different")])
    with pytest.raises(ValueError, match="checkpoint differs"):
        analysis.main()
