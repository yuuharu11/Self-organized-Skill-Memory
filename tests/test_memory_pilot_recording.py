import jax.numpy as jnp
import numpy as np
import json
from dataclasses import dataclass
from types import SimpleNamespace

from openpi.models.gemma_moe_skill import _masked_skill_softmax
from skill_moe.skillnet.examples.memory_pilot import record_rollouts


def test_fully_masked_skill_tokens_contribute_zero_attention():
    scores = jnp.asarray([[[[0.2, -0.4, 1.0]]]], dtype=jnp.float32)
    mask = jnp.asarray([[False, False, False]])
    weights = np.asarray(_masked_skill_softmax(scores, mask))
    np.testing.assert_array_equal(weights, np.zeros_like(weights))


def test_single_valid_skill_gets_all_attention_weight():
    scores = jnp.asarray([[[[0.2, -0.4, 1.0]]]], dtype=jnp.float32)
    mask = jnp.asarray([[False, True, False]])
    weights = np.asarray(_masked_skill_softmax(scores, mask))
    np.testing.assert_allclose(weights, [[[[0.0, 1.0, 0.0]]]])


def test_disabled_capture_path_keeps_skill_embedding_hook_identity():
    from openpi.models.pi0_moe_skill import _apply_skill_embed_override

    original = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    assert _apply_skill_embed_override(original, None) is original


def test_recording_round_trips_environment_steps_features_and_executed_prefix(tmp_path, monkeypatch):
    from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import load_run, _stack_rows
    from skill_moe.skillnet.examples.memory_pilot.memory import MemoryStore

    @dataclass
    class Config:
        name: str = "synthetic"
        exp_name: str = "test"

    class Suite:
        n_tasks = 2
        def get_task(self, task_id):
            return SimpleNamespace(language=f"task {task_id}", problem_folder="", bddl_file="dummy")
        def get_task_init_states(self, task_id):
            return [0, 1]

    class Environment:
        closed = False
        def __init__(self, **kwargs):
            pass
        def seed(self, seed):
            pass
        def observation(self):
            return {"robot0_eef_pos": np.array([self.time ** 2 + self.init * 100, 0., 0.]),
                    "robot0_eef_quat": np.array([0., 0., 0., 1.]), "robot0_gripper_qpos": np.zeros(2),
                    "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((224, 224, 3), dtype=np.uint8)}
        def reset(self):
            assert not self.closed
            self.time, self.init = 0, 0
            return self.observation()
        def set_init_state(self, init):
            self.init = init
            return self.observation()
        def step(self, action):
            self.time += 1
            return self.observation(), 0., False, {}
        def close(self):
            self.closed = True

    class Policy:
        def infer(self, raw):
            return {"actions": np.array([[1.] * 7, [2.] * 7, [999.] * 7]),
                    "memory_pilot_features": np.broadcast_to(np.arange(18)[:, None, None], (18, 3, 4)),
                    "memory_pilot_router_top1": np.zeros((18, 3), dtype=np.int32)}

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "instruct2plan_40.json").write_text(json.dumps({
        "task 0": {"all_classes": [0, 1]}, "task 1": {"all_classes": [1]},
        "outside suite": {"all_classes": [4]}}))
    monkeypatch.setenv("SKILLNET_MEMORY_PILOT_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(record_rollouts, "ANNOTATION_DIR", annotations)
    monkeypatch.setattr(record_rollouts.config_moe_skill, "get_config", lambda _: Config())
    monkeypatch.setattr(record_rollouts, "create_trained_policy", lambda *args, **kwargs: Policy())
    monkeypatch.setattr(record_rollouts.benchmark, "get_benchmark_dict", lambda: {"synthetic": Suite})
    monkeypatch.setattr(record_rollouts, "OffScreenRenderEnv", Environment)
    monkeypatch.setattr(record_rollouts, "get_libero_path", lambda _: str(tmp_path))
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["record", "--suite", "synthetic", "--trials", "2", "--max-steps", "14",
                                     "--replan-steps", "2", "--run-dir", str(out)])
    record_rollouts.main()
    manifest, episodes = load_run(out)
    assert manifest["suite_skill_ids"] == [0, 1]
    assert manifest["hidden_features"]["layers"] == [2, 9, 17]
    assert manifest["env_step_offset"] == 1
    assert [ep["init_state_id"] for ep in manifest["episodes"]] == [0, 1]
    for ep in episodes:
        arrays = ep.arrays
        np.testing.assert_array_equal(arrays["step"], [10, 12])
        np.testing.assert_array_equal(arrays["n_executed"], [2, 2])
        np.testing.assert_array_equal(arrays["eef_pos"], arrays["env_eef_pos"][arrays["step"] - ep.env_step_offset])
        np.testing.assert_array_equal(arrays["proprio"][:, :3], arrays["eef_pos"])
        np.testing.assert_array_equal(arrays["skills"], [[1, 2, 0, 0], [1, 2, 0, 0]])
        for layer in manifest["hidden_features"]["layers"]:
            np.testing.assert_array_equal(arrays[f"feat__L{layer:02d}_mean_t_last"], np.full((2, 4), layer))
        rows = _stack_rows(episodes, {ep.episode_id}, "proprio", 2)
        np.testing.assert_allclose(rows[0]["target"][:, 0], [21, 44])
    memory = MemoryStore.from_recording(out, key_feature="feat__L02_mean_t_last", value_source="env_trajectory",
                                       representation_version="test")
    np.testing.assert_allclose(memory.entries[0].value[:, 0], [21, 44])
    commands = MemoryStore.from_recording(out, key_feature="feat__L02_mean_t_last", value_source="pred_chunk_executed",
                                         representation_version="test")
    np.testing.assert_array_equal(commands.entries[0].value[:, 0], [1, 2])


