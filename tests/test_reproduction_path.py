"""Regressions found by following docs/reproduction.md from a fresh clone."""
import ast
import importlib.util
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from openpi.models import pi0 as plain_pi0
from openpi.models import pi0_moe_skill
from openpi.models.model import Observation
from openpi.models.pi0_optimus_r_config import Pi0OptimusRConfig
from openpi.policies.policy import Policy

from tests.support import (
    EXAMPLES,
    PI05,
    REPO,
)


def _load_helper():
    path = EXAMPLES / "libero/install_libero_skill_assets.py"
    spec = importlib.util.spec_from_file_location("install_libero_skill_assets_testmod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM_BENCHMARK = """\
libero_suites = ["libero_90", "libero_10"]
task_maps = {}
def register_benchmark(cls): return cls
class Benchmark:
    def _make_benchmark(self):
        if self.name == "libero_90":
            pass
"""


def test_libero_patch_registers_suite_in_task_maps_on_a_clean_checkout():
    patched = _load_helper().benchmark_registration_patch(UPSTREAM_BENCHMARK)
    namespace = {}
    exec(patched, namespace)
    assert "libero_skill_obj" in namespace["libero_suites"]
    assert "LIBERO_SKILL_OBJ" in namespace


class _Model:
    def generate_subtask(self, *_args, **_kwargs):
        raise AssertionError("subtask generation must stay off")

    def build_full_observation(self, *_args, **_kwargs):
        raise AssertionError("subtask generation must stay off")


class _MoeModel(_Model):
    subtask_generation_default = False


def _policy(model):
    policy = Policy.__new__(Policy)  # The constructor would JIT a real model.
    policy._model = model
    policy._is_pytorch_model = False
    return policy


def test_released_moe_model_does_not_decode_subtasks_by_default(monkeypatch):
    monkeypatch.delenv("SKILLNET_SUBTASK_GENERATE", raising=False)
    monkeypatch.delenv("OPENPI_SUBTASK_GENERATE", raising=False)
    assert pi0_moe_skill.Pi0.subtask_generation_default is False
    assert not hasattr(plain_pi0.Pi0, "subtask_generation_default")
    observation = Observation(
        images={}, image_masks={}, state=jnp.zeros((1, 8)),
        tokenized_prompt=jnp.ones((1, 4), jnp.int32), tokenized_prompt_mask=jnp.ones((1, 4), bool),
    )
    assert _policy(_MoeModel())._maybe_generate_subtask(observation, prompt="x") is observation


@pytest.mark.parametrize(("value", "model", "expected"), [
    (None, _MoeModel, False), ("1", _MoeModel, True), ("0", _MoeModel, False),
    (None, _Model, True), ("false", _Model, False),
])
def test_subtask_generation_switch(monkeypatch, value, model, expected):
    monkeypatch.delenv("OPENPI_SUBTASK_GENERATE", raising=False)
    if value is None:
        monkeypatch.delenv("SKILLNET_SUBTASK_GENERATE", raising=False)
    else:
        monkeypatch.setenv("SKILLNET_SUBTASK_GENERATE", value)
    assert _policy(model())._subtask_generation_enabled() is expected


def test_moe_subtask_decoding_runs_past_the_first_token():
    config = Pi0OptimusRConfig(
        pi05=True, action_dim=8, action_horizon=2, max_token_len=32, discrete_state_input=False,
        paligemma_variant="dummy", action_expert_variant="dummy_moe_4", vision_variant="mu/32",
        memory_token_count=2, query_dim=8, skill_dim=8, skill_token_count=2, interface_hidden_dim=16,
    )
    model = config.create(jax.random.key(3))
    images = {name: jnp.zeros((1, 224, 224, 3), jnp.float32) for name in (
        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    masks = {name: jnp.ones((1,), bool) for name in images}
    prompt = jnp.zeros((1, 32), jnp.int32).at[:, :4].set(jnp.asarray([2, 10, 11, 12]))
    observation = Observation(
        images=images, image_masks=masks, state=jnp.zeros((1, 8), jnp.float32),
        tokenized_prompt=prompt, tokenized_prompt_mask=jnp.arange(32)[None] < 4,
    )
    assert isinstance(model, pi0_moe_skill.Pi0)
    assert model.generate_subtask(observation, max_tokens=3).shape == (1, 3)


def _experiment_sources():
    return sorted(p for p in EXAMPLES.rglob("*.py") if "third_party" not in p.parts)


def test_no_path_method_is_called_on_a_string_literal():
    """`root / 'dir'.rglob(...)` binds to the string and fails only at run time."""
    path_methods = {"rglob", "glob", "iterdir", "read_text", "read_bytes", "exists", "is_dir", "is_file", "mkdir"}
    offenders = []
    for path in _experiment_sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in path_methods
                    and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize("stage", [
    "topk_long/run.py", "skill_memories/build.py", "local_key/run.py",
    "key_mechanism/run.py", "fair_expert/run.py",
])
def test_core_path_stages_do_not_pin_the_saved_hard_threshold(stage):
    assert ".01549410422449" not in (PI05 / stage).read_text(encoding="utf-8")


def test_score_audit_checkpoint_follows_the_state_dir(tmp_path):
    code = ("import json,sys;sys.argv=['x'];"
            f"sys.path.insert(0,{str(PI05 / 'score_audit')!r});import keys;print(json.dumps(str(keys.CHECKPOINT)))")
    env = {"SKILLNET_STATE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"}
    output = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(output.stdout.splitlines()[-1]).startswith(str(tmp_path))
