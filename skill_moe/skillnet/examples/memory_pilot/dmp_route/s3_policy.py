"""Inference-only LIBERO policy for the S3 DMP-routed SkillNet model."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils

from command_dmp import command_draft
from openpi.models.pi0_dmp_route import shuffle_route_primitives


class DmpRoutePolicy(_policy.Policy):
    """Online route prediction only; no annotations or privileged state labels."""

    def __init__(
        self,
        model,
        *,
        input_transforms,
        output_transforms,
        dmp_module,
        library,
        action_q01,
        action_q99,
        prompt_map,
        prompt_level="L0",
        bridge_t0=0.4,
        num_steps=2,
        injection_mode="both",
        position_m_per_command=0.05,
        rotation_rad_per_command=0.5,
        seed=0,
        shape_mode="learned",
        primitive_mode="learned",
    ):
        # Policy's base class expects an optional legacy-compatible key and
        # uses `rng or key(0)` internally. JAX typed keys cannot be truth-tested,
        # so leave rng unset; the policy's deterministic seed is managed below.
        super().__init__(model, transforms=input_transforms, output_transforms=output_transforms)
        self._rng = jax.random.key(seed)
        if prompt_level not in ("L0", "L3"):
            raise ValueError("prompt_level must be L0 or L3")
        if injection_mode not in ("both", "router_only", "adarms_only", "none"):
            raise ValueError("unsupported primitive injection path")
        self._model = model
        self._dmp_module = dmp_module
        self._library = library
        self._last_raw_commands = {}
        self._q01 = np.asarray(action_q01, np.float32)[:7]
        self._q99 = np.asarray(action_q99, np.float32)[:7]
        self._prompt_map = prompt_map
        self._default_prompt_level = prompt_level
        self._bridge_t0 = float(bridge_t0)
        self._num_steps = int(num_steps)
        self._injection_mode = injection_mode
        self._position_scale = float(position_m_per_command)
        self._rotation_scale = float(rotation_rad_per_command)
        if shape_mode not in ("learned", "straight") or primitive_mode not in ("learned", "shuffle"):
            raise ValueError("shape_mode must be learned/straight; primitive_mode learned/shuffle")
        self._shape_mode = shape_mode
        self._primitive_mode = primitive_mode
        self._history_len = int(model.dmp_config.state_history_len)
        self._histories = defaultdict(lambda: deque(maxlen=self._history_len))
        self._predict_route = nnx_utils.module_jit(model.predict_route)
        self._encode_shapes = nnx_utils.module_jit(model.encode_primitive_shapes)
        self._sample_s3 = nnx_utils.module_jit(
            model.sample_actions_s3,
            static_argnames=("num_steps", "injection_mode", "t0"),
        )

    def _select_prompt(self, raw):
        level = str(raw.pop("dmp_prompt_level", self._default_prompt_level))
        if level not in ("L0", "L3"):
            raise ValueError(f"unsupported prompt level {level!r}")
        instruction = str(raw.get("prompt", ""))
        if level == "L0":
            return instruction, level
        translations = self._prompt_map.get(instruction)
        if not isinstance(translations, dict) or "L3" not in translations:
            raise KeyError(f"no L3 prompt translation for instruction: {instruction!r}")
        raw["prompt"] = str(translations["L3"])
        return raw["prompt"], level

    def infer(self, obs: dict, *, noise=None):
        started = time.monotonic()
        raw = dict(obs)
        _, level = self._select_prompt(raw)
        history_key = raw.pop("dmp_history_id", None)
        if history_key is None:
            history_key = raw.pop("episode_id", None)
        else:
            raw.pop("episode_id", None)
        episode_key = None if history_key is None else str(history_key)
        reset_history = bool(raw.pop("dmp_reset_history", False))
        supplied_history = raw.pop("dmp_state_history", None)
        shape_mode = str(raw.pop("dmp_shape_mode", self._shape_mode))
        primitive_mode = str(raw.pop("dmp_primitive_mode", self._primitive_mode))
        transformed = self._input_transform(raw)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        observation = _model.Observation.from_dict(inputs)
        if reset_history and episode_key is not None:
            self._histories[episode_key].clear()
            self._last_raw_commands.pop(episode_key, None)
        if supplied_history is not None:
            state_history = np.asarray(supplied_history, np.float32)
            if state_history.shape != (self._history_len, inputs["state"].shape[-1]):
                raise ValueError("dmp_state_history must match [history_len, transformed_state_width]")
        else:
            current_state = np.asarray(inputs["state"][0], np.float32)
            if episode_key is None:
                # Do not share temporal state across concurrent websocket
                # clients if the caller supplied no stream identity.
                history_rows = [current_state] * self._history_len
            else:
                history = self._histories[episode_key]
                history.append(current_state)
                history_rows = list(history)
                if len(history_rows) < self._history_len:
                    history_rows = [history_rows[0]] * (self._history_len - len(history_rows)) + history_rows
            state_history = np.stack(history_rows)
        state_history = jnp.asarray(state_history[None], jnp.float32)
        self._rng, route_rng, action_rng = jax.random.split(self._rng, 3)
        if shape_mode not in ("learned", "straight") or primitive_mode not in ("learned", "shuffle"):
            raise ValueError("unsupported shape/primitive ablation mode")
        if episode_key is None:
            previous_command = np.zeros((1, 6), np.float32)
        else:
            previous_command = np.asarray(self._last_raw_commands.get(episode_key, np.zeros(6)), np.float32)[None]
        if self._injection_mode == "none":
            from openpi.models.pi0_dmp_route import DmpRouteState

            route = DmpRouteState(
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1, self._model.dmp_config.skill_embed_dim + (4 if self._model.positional_encoding else 0)), jnp.float32),
                jnp.zeros((1, 6), jnp.float32), jnp.ones((1,), jnp.float32), jnp.zeros((1,), jnp.float32),
            )
            draft = np.zeros((1, self._model.action_horizon, 7), np.float32)
            route_np = jax.tree.map(np.asarray, route)
        else:
            shapes = jnp.asarray(self._library["shapes"], jnp.float32)
            route = self._predict_route(observation, shapes, state_history)
            route_np = jax.tree.map(np.asarray, route)
            if primitive_mode == "shuffle":
                embeddings = self._encode_shapes(shapes)
                route = shuffle_route_primitives(route, route_rng, embeddings)
                route_np = jax.tree.map(np.asarray, route)
            # S1-v3 is a direct-command residual basis. The straight ablation
            # removes only that residual, keeping predicted g/tau/phase.
            draft = command_draft(
                self._dmp_module, self._library, route_np, previous_command,
                q01=self._q01, q99=self._q99,
                horizon=self._model.action_horizon,
                dt=float(self._library.get("dt", 0.05)), shape_mode=shape_mode,
            )
        noise_array = None
        if noise is not None:
            noise_array = jnp.asarray(noise, jnp.float32)
            if noise_array.ndim == 2:
                noise_array = noise_array[None]
        actions = self._sample_s3(
            action_rng,
            observation,
            draft=jnp.asarray(draft),
            noise=noise_array,
            route=route,
            t0=self._bridge_t0,
            num_steps=self._num_steps,
            injection_mode=self._injection_mode,
        )
        outputs = self._output_transform({"state": inputs["state"], "actions": actions})
        result = jax.tree.map(lambda x: np.asarray(x[0]), outputs)
        if episode_key is not None and "actions" in result:
            # `output_transform` has already restored public raw action-command
            # units. The next replan starts from the last command of this chunk.
            self._last_raw_commands[episode_key] = np.asarray(result["actions"][-1, :6], np.float32)
        result["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
        result["dmp_route"] = {
            "prim_id": int(np.asarray(route.prim_id)[0]),
            "goal": np.asarray(route.goal[0]).tolist(),
            "tau": float(np.asarray(route.tau[0])),
            "phase_s": float(np.asarray(route.phase_s[0])),
            "prompt_level": level,
            "shape_mode": shape_mode,
            "primitive_mode": primitive_mode,
        }
        return result


def load_prompt_map(path: str | Path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("prompts.json must be a JSON object")
    return data
