"""Serve the RM-D1 P5 -> chained-motion-draft -> pi0.5 bridge policy.

Run in the SkillNet Python environment. Simulator actions and the online
observation state use RoboCasa fork order; P5's stored demonstrations remain
in LeRobot order and are adapted by expert.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

# RoboCasa evaluations must match the released checkpoint's task-only prompt.
os.environ["SKILLNET_SUBTASK_GENERATE"] = "0"

ROOT = Path(__file__).resolve().parents[4]
SKILLNET_SRC = ROOT / "skill_moe/skillnet/src"
EXPERT_DIR = ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"
PI05_MEMORY_DIR = ROOT / "skill_moe/skillnet/examples/memory_pilot/pi05_memory"
for _path in (SKILLNET_SRC, EXPERT_DIR, PI05_MEMORY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import expert as expert_api  # noqa: E402
from norm_utils import norm_ratio_scale  # noqa: E402


class _ContextPrefixCache:
    """Jitted prefix pass returning both the bridge KV cache and Context key."""

    @staticmethod
    def build(model):
        import flax.nnx as nnx
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from openpi.models.pi0 import make_attn_mask
        from openpi.shared.nnx_utils import module_jit

        class Prefix(nnx.Module):
            def __init__(self, base):
                self.base = base

            def __call__(self, observation):
                obs = model_api.preprocess_observation(None, observation, train=False)
                tokens, valid, ar = self.base.embed_prefix(obs)
                positions = jnp.cumsum(valid, axis=1) - 1
                (hidden, _), cache = self.base.PaliGemma.llm(
                    [tokens, None], mask=make_attn_mask(valid, ar), positions=positions
                )
                weights = valid.astype(hidden.dtype)[..., None]
                pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
                return (valid, cache), pooled.astype(jnp.float16)

        return module_jit(Prefix(model).__call__)


def _context_net_class():
    import torch
    from torch import nn

    class ContextNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
            self.key = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 256))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))
            self.grip = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 6))
        def forward(self, query, keys):
            return nn.functional.normalize(self.query(query), dim=-1) @ nn.functional.normalize(self.key(keys), dim=-1).T * self.log_scale.exp()

    return ContextNet


class _ReferenceScaleBridge:
    """Build an NNX proxy class lazily so module_jit sees the wrapped model."""

    @staticmethod
    def build(model, action_stats, mode):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        class ScaledBridge(nnx.Module):
            def __init__(self, base):
                self.base = base
                self.action_dim = base.action_dim
                self.mean = nnx.Variable(jnp.asarray(action_stats.mean, jnp.float32))
                self.std = nnx.Variable(jnp.asarray(action_stats.std, jnp.float32))
                self.scale_mode = mode

            def sample_bridge(self, rng, observation, drafts, starts, *, num_steps=10, noise=None,
                              prefix_context=None, scale_reference=None):
                mean = self.mean.value[: drafts.shape[-1]]
                std = self.std.value[: drafts.shape[-1]]
                physical = drafts * (std[None, None, :] + 1e-6) + mean[None, None, :]
                if scale_reference is None:
                    raise ValueError("scale_reference is required for the fitted-scale bridge")
                ref = jnp.asarray(scale_reference, jnp.float32)

                def fit_scale(x, y):
                    den = jnp.sum(jnp.square(x), axis=(1, 2), keepdims=True)
                    dot = jnp.sum(x * y, axis=(1, 2), keepdims=True)
                    return jnp.maximum(dot / jnp.maximum(den, 1e-12), 0.0)

                def fit_norm(x, y):
                    den = jnp.sqrt(jnp.sum(jnp.square(x), axis=(1, 2), keepdims=True))
                    num = jnp.sqrt(jnp.sum(jnp.square(y), axis=(1, 2), keepdims=True))
                    return num / jnp.maximum(den, 1e-8)

                if self.scale_mode == "norm-sep":
                    pos = physical[:, :, :3]
                    rot = physical[:, :, 3:6]
                    physical = physical.at[:, :, :3].set(pos * fit_norm(pos, ref[:, :, :3]))
                    physical = physical.at[:, :, 3:6].set(rot * fit_norm(rot, ref[:, :, 3:6]))
                elif self.scale_mode == "s-shared":
                    eef = physical[:, :, :6]
                    physical = physical.at[:, :, :6].set(eef * fit_scale(eef, ref))
                else:
                    pos = physical[:, :, :3]
                    rot = physical[:, :, 3:6]
                    pos_scale = fit_scale(pos, ref[:, :, :3])
                    rot_scale = fit_scale(rot, ref[:, :, 3:6])
                    physical = physical.at[:, :, :3].set(pos * pos_scale)
                    physical = physical.at[:, :, 3:6].set(rot * rot_scale)
                normalized = (physical - mean[None, None, :]) / (std[None, None, :] + 1e-6)
                return self.base.sample_bridge(
                    rng, observation, normalized, starts, num_steps=num_steps, noise=noise,
                    prefix_context=prefix_context
                )

        proxy = ScaledBridge(model)

        def set_reference(actions):
            return jnp.asarray(np.asarray(actions, np.float32)[None, :, :6])

        return proxy, set_reference


class MemoryBridgePolicy:
    """Websocket policy with episode-local, actually executed action history."""

    def __init__(self, args):
        import torch
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from openpi.shared import normalize
        from openpi.shared.nnx_utils import module_jit
        from openpi import transforms
        from openpi.training import config as train_config

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.set_num_interop_threads(1)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TORCH_MEMORY_FRACTION", ".04")))

        self.args = args
        self.lock = threading.RLock()
        self.histories = {}
        self.log_lock = threading.Lock()
        self.log_path = Path(args.query_log) if args.query_log else None
        self.checkpoint = Path(args.checkpoint)
        self.stats = normalize.load(self.checkpoint / "assets")
        self.action_stats = self.stats["actions"]
        # Snapshot the pretrained flow sampler before overlaying the Expert.
        # Both paths can then consume one shared image/language prefix cache.
        self.bridge_model, cfg = expert_api._load_expert_model(self.checkpoint)
        self.base_model = self.bridge_model
        self.prefix_context_fn = module_jit(self.bridge_model.encode_prefix_context)
        self.context_prefix_fn = _ContextPrefixCache.build(self.bridge_model) if args.search_mode in ("context", "r2") else None
        self.base_reference_fn = module_jit(
            self.bridge_model.sample_actions_with_context, static_argnames=("num_steps",)
        )
        expert_path = None
        if args.weights == "expert":
            best_path = Path(args.expert_dir) / "best.json"
            best = json.loads(best_path.read_text())
            expert_path = best_path.parent / best["checkpoint"]
            self.expert_path = str(expert_path)
        else:
            self.expert_path = None

        from openpi.training import config as train_config
        policy_cfg = train_config.get_config("pi05_robocasa_multitask")
        data_cfg = policy_cfg.data.create(policy_cfg.assets_dirs, policy_cfg.model)
        input_transforms = [
            transforms.InjectDefaultPrompt(None),
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(self.stats, use_quantiles=False),
            *data_cfg.model_transforms.inputs,
        ]
        output_transforms = [
            *data_cfg.model_transforms.outputs,
            transforms.Unnormalize(self.stats, use_quantiles=False),
            *data_cfg.data_transforms.outputs,
        ]
        self.input_transform = transforms.compose(input_transforms)
        if expert_path is not None:
            expert_api._overlay_expert(self.bridge_model, expert_path)
        self.set_reference = None
        if args.scale_mode != "saved":
            self.bridge_model, self.set_reference = _ReferenceScaleBridge.build(
                self.bridge_model, self.action_stats, args.scale_mode
            )
        self.bridge_fn = module_jit(self.bridge_model.sample_bridge, static_argnames=("num_steps",))

        pool_path = Path(args.memory_dir) / "pools/U_memory.npz"
        self.pool = dict(np.load(pool_path, allow_pickle=False))
        manifest_path = Path(args.memory_dir) / "manifest.json"
        self.task_names = json.loads(manifest_path.read_text()).get("tasks", []) if manifest_path.exists() else []
        p5_dir = Path(args.memory_dir) / "models"
        self.p5_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.p5_device == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TORCH_MEMORY_FRACTION", ".03")))
        self.p5_models = []
        self.context_net = None
        self.context_keys = None
        self.context_key_embeddings = None
        self.context_norm = None
        self.context_pool = None
        self.context_rows = None
        if args.search_mode in ("context", "r2"):
            context_dir = Path(args.context_dir)
            ck = torch.load(context_dir / "context_model.pt", map_location="cpu", weights_only=False)
            self.context_norm = {k: np.asarray(v, np.float32) for k, v in ck["norm"].items()}
            self.context_net = _context_net_class()(int(ck["input_dim"]))
            self.context_net.load_state_dict(ck["state"]); self.context_net.eval().to(self.p5_device)
            rows = np.load(context_dir / "cache/rows.npz", allow_pickle=False)
            feats = np.asarray(np.load(context_dir / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
            cix = np.asarray(rows["candidate_context"], np.int64)
            state = np.asarray(rows["candidate_state"], np.float32)
            n = self.context_norm
            cand = np.concatenate([(feats[cix] - n["feature_mean"]) / n["feature_std"],
                                   (state - n["state_mean"]) / n["state_std"]], axis=1).astype(np.float32)
            self.context_keys = torch.as_tensor(cand, dtype=torch.float32, device=self.p5_device)
            with torch.no_grad():
                self.context_key_embeddings = torch.nn.functional.normalize(
                    self.context_net.key(self.context_keys), dim=-1
                )
            self.context_pool = dict(np.load(context_dir / "cache/pool.npz", allow_pickle=False))
            self.context_rows = rows
            source_manifest = json.loads((context_dir / "cache/feature_manifest.json").read_text())
            self.prompt_task = {}
            try:
                source_data = json.loads((Path(source_manifest["source"]) / "manifest.json").read_text())
                self.prompt_task = {str(d["prompt"]).strip().lower(): int(d["task_id"]) for d in source_data["demos"]}
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                self.prompt_task = {}
            if len(cix) != len(self.context_pool["length"]):
                raise ValueError("Context candidate embedding rows do not align with U memory pool")
            for name in ("task", "frame", "global_episode"):
                key = "candidate_episode" if name == "global_episode" else f"candidate_{name}"
                if not np.array_equal(rows[key], self.context_pool[name]):
                    raise ValueError(f"Context candidate order mismatch in {name}")
        else:
            if self.p5_device == "cuda":
                self.pool["key"] = torch.as_tensor(self.pool["key"], dtype=torch.float32, device="cuda")
            self.p5_models = [
                expert_api._load_net(p5_dir / f"P5_U_seed{i}.pt", self.pool, device=self.p5_device)
                for i in (0, 1, 2)
            ]
        print(json.dumps({
            "serving": "rm-d2-memory-bridge",
            "port": args.port,
            "weights": args.weights,
            "expert": self.expert_path,
            "scale_mode": args.scale_mode,
            "t0": args.t0,
            "scale_reference": "frozen_pretrained_pi05_only_scale_fit" if args.scale_mode != "saved" else "stored_action_scale",
            "memory_units": len(self.pool["length"]),
            "history": "client_id + episode_id; left-zero-padded 10x12 with explicit mask",
            "subtask_generation": os.environ["SKILLNET_SUBTASK_GENERATE"],
            "search_mode": args.search_mode,
            "context_gripper": args.context_gripper,
        }), flush=True)

    def _context_select(self, prefix_feature, state):
        import torch
        n = self.context_norm
        q = np.concatenate([(np.asarray(prefix_feature, np.float32).reshape(-1) - n["feature_mean"]) / n["feature_std"],
                            (np.asarray(state, np.float32).reshape(-1) - n["state_mean"]) / n["state_std"]]).astype(np.float32)
        with torch.no_grad():
            qt = torch.as_tensor(q[None], dtype=torch.float32, device=self.p5_device)
            qemb = torch.nn.functional.normalize(self.context_net.query(qt), dim=-1)
            scores = (qemb @ self.context_key_embeddings.T)[0] * self.context_net.log_scale.exp()
            selected = int(torch.argmax(scores).item())
            grip_logits = self.context_net.grip(qt)[0]
        return selected, float(scores[selected].item()), (grip_logits >= 0).to(torch.float32).cpu().numpy()[:5] * 2 - 1

    def _r2_select(self, reference, prefix_feature, state):
        """Select the unit whose scale-normalized DCT motion type matches π0.5.

        For each candidate's stored duration L, the comparison uses the first L
        reference actions, resampled to the same ten phase points as the memory
        unit. Position and rotation share one nonnegative scale (S-shared); a
        negative type dot product is inadmissible rather than flipped.
        """
        import experiment as memory_experiment

        lengths = np.clip(np.asarray(self.context_pool["length"], np.int64), 1, 50)
        candidates = np.asarray(self.context_pool["shape"], np.float32).reshape(len(lengths), -1)
        best_error = np.full(len(lengths), np.inf, np.float64)
        best_dot = np.full(len(lengths), -np.inf, np.float64)
        ref = np.asarray(reference, np.float32)
        for length in np.unique(lengths):
            ids = np.flatnonzero(lengths == length)
            # The stored unit shapes were encoded from LeRobot storage order
            # (EEF channels 5:11). Online reference actions are in fork order,
            # so convert before applying the same DCT feature definition.
            query_storage = memory_experiment.action_policy_to_storage(ref[:int(length)])
            query_shape, _ = memory_experiment.encode_shape(query_storage)
            q = np.asarray(query_shape, np.float64).reshape(-1)
            c = candidates[ids].astype(np.float64)
            dot = c @ q
            norm = np.sum(c * c, axis=1)
            valid = dot > 0.0
            err = np.full(len(ids), np.inf, np.float64)
            err[valid] = np.maximum(np.dot(q, q) - dot[valid] ** 2 / np.maximum(norm[valid], 1e-12), 0.0)
            best_error[ids] = err
            best_dot[ids] = dot
        if not np.isfinite(best_error).any():
            raise RuntimeError("R2 found no candidate with positive S-shared type alignment")
        selected = int(np.argmin(best_error))
        # Context grip head is retained exactly as in ctx-cl-01; only unit
        # selection changes in R2.
        import torch
        n = self.context_norm
        qctx = np.concatenate([
            (np.asarray(prefix_feature, np.float32).reshape(-1) - n["feature_mean"]) / n["feature_std"],
            (np.asarray(state, np.float32).reshape(-1) - n["state_mean"]) / n["state_std"],
        ]).astype(np.float32)
        with torch.no_grad():
            qt = torch.as_tensor(qctx[None], dtype=torch.float32, device=self.p5_device)
            grip_logits = self.context_net.grip(qt)[0]
        signs = (grip_logits >= 0).to(torch.float32).cpu().numpy()[:5] * 2 - 1
        return selected, float(-best_error[selected]), signs, float(best_dot[selected]), float(best_error[selected])

    def _context_draft(self, selected, reference, grip_signs):
        """One retrieved raw unit, unit-interval norm fit, then 50-step pad."""
        pool = self.context_pool
        a, b = int(pool["action_offsets"][selected]), int(pool["action_offsets"][selected + 1])
        raw_storage = np.asarray(pool["actions_flat"][a:b], np.float32)
        if len(raw_storage) == 0:
            raise ValueError(f"empty U unit {selected}")
        raw = expert_api.ex.action_storage_to_policy(raw_storage)
        unit_len = len(raw)
        draft = np.empty((50, 12), np.float32)
        n = min(unit_len, 50)
        draft[:n] = raw[:n]
        if n < 50:
            draft[n:] = raw[n - 1]
        # Fit scale on the selected unit's valid duration, never the padded
        # 50-step tail. The historical projection mode remains available; the
        # norm-ratio mode is angle independent and shares position scale with
        # rotation (rot=pos), whose tiny commands make a separate fit noisy.
        ref = np.asarray(reference, np.float32)
        scale_pos_norm = norm_ratio_scale(draft[:n, :3], ref[:n, :3], n)
        scale_rot_norm = norm_ratio_scale(draft[:n, 3:6], ref[:n, 3:6], n)
        if self.args.context_scale_mode == "norm-ratio":
            scale_pos = scale_pos_norm
            scale_rot = scale_pos
        else:
            def projection(x, y):
                value = float(np.sum(x * y) / max(float(np.sum(x * x)), 1e-12))
                return float(np.clip(value, .25, 4.0))
            scale_pos = projection(draft[:n, :3], ref[:n, :3])
            scale_rot = projection(draft[:n, 3:6], ref[:n, 3:6])
        draft[:, :3] *= scale_pos
        draft[:, 3:6] *= scale_rot
        scales = [scale_pos, scale_rot, scale_rot_norm, scale_pos_norm]
        draft[:, 7:12] = np.asarray(reference, np.float32)[:, 7:12]
        if self.args.context_gripper == "unit":
            g = 1.0 if float(np.mean(raw_storage[:, 11])) >= 0 else -1.0
            draft[:, 6] = g
        elif self.args.context_gripper == "context":
            signs = np.asarray(grip_signs, np.float32).reshape(-1)
            draft[:5, 6] = signs
            draft[5:, 6] = signs[-1]
        else:
            draft[:, 6] = np.asarray(reference, np.float32)[:, 6]
        return draft, unit_len, scales

    def _log(self, row):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def infer(self, raw_observation):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api

        obs_in = dict(raw_observation)
        client_id = str(obs_in.pop("client_id", "default"))
        episode_id = str(obs_in.pop("episode_id", "0"))
        episode_start = bool(obs_in.pop("episode_start", False))
        state = np.asarray(obs_in["observation/state"], np.float32)
        if state.shape != (16,):
            raise ValueError(f"RoboCasa fork state must be [16], got {state.shape}")
        with self.lock:
            hist = self.histories.get(client_id)
            if hist is None:
                # Import the lightweight helper from this directory, not the
                # impl-3-owned memory adapter.
                from memory_server_utils import ClientHistory
                hist = ClientHistory(10)
                self.histories[client_id] = hist
            history, mask, request_index = hist.begin_request(episode_id, episode_start)

            started = time.perf_counter()
            transformed = self.input_transform(obs_in)
            batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
            observation = model_api.Observation.from_dict(batched)
            reference = None
            ref_physical = None
            prefix_context = None
            context_debug = None
            if self.args.search_mode in ("context", "r2"):
                # One compiled prefix pass supplies both the Context feature
                # and the exact KV cache consumed by reference/bridge sampling.
                prefix_context, prefix_feature = self.context_prefix_fn(observation)
                dim = int(self.base_model.action_dim)
                ref_noise = jax.random.normal(
                    jax.random.key(int(self.args.seed) + request_index + 2),
                    (1, expert_api.HORIZON, dim),
                )
                ref_norm = self.base_reference_fn(
                    jax.random.key(int(self.args.seed) + request_index + 3), observation,
                    prefix_context, num_steps=10, noise=ref_noise,
                )
                mean = np.asarray(self.action_stats.mean, np.float32)
                std = np.asarray(self.action_stats.std, np.float32)
                ref_physical = np.asarray(ref_norm, np.float32)[0] * (std[None, :] + 1e-6) + mean[None, :]
                if self.args.search_mode == "r2":
                    selected, score, grip_signs, type_dot, type_error = self._r2_select(
                        ref_physical[:, :12], np.asarray(prefix_feature)[0], state
                    )
                else:
                    selected, score, grip_signs = self._context_select(np.asarray(prefix_feature)[0], state)
                    type_dot, type_error = None, None
                draft, unit_len, scales = self._context_draft(selected, ref_physical[:, :12], grip_signs)
                if self.args.context_output == "direct":
                    action = draft
                else:
                    dim = int(self.bridge_model.action_dim)
                    physical = np.zeros((1, 50, dim), np.float32)
                    physical[0, :, :12] = draft
                    draft_norm = (physical - mean[None, None, :dim]) / (std[None, None, :dim] + 1e-6)
                    bridge_noise = jax.random.normal(
                        jax.random.key(int(self.args.seed) + request_index + 4), draft_norm.shape
                    )
                    bridge_norm = self.bridge_fn(
                        jax.random.key(int(self.args.seed) + request_index + 5), observation,
                        jnp.asarray(draft_norm), jnp.asarray([float(self.args.t0)], jnp.float32),
                        num_steps=int(self.args.integral_steps), noise=bridge_noise,
                        prefix_context=prefix_context,
                    )
                    action = np.asarray(bridge_norm, np.float32)[0] * (std[None, :dim] + 1e-6) + mean[None, :dim]
                    action = action[:, :12]
                task_id = int(self.context_pool["task"][selected])
                expected_task = self.prompt_task.get(str(obs_in.get("prompt", "")).strip().lower())
                context_debug = {
                    "selected_index": int(selected), "selected_task_id": task_id,
                    "selected_task": self.task_names[task_id] if 0 <= task_id < len(self.task_names) else str(task_id),
                    "selected_episode": int(self.context_pool["episode"][selected]),
                    "selected_frame": int(self.context_pool["frame"][selected]),
                    "unit_length": int(unit_len), "cosine_score": float(score),
                    "prompt_task_id": expected_task,
                    "task_match": None if expected_task is None else bool(task_id == expected_task),
                    "scale_method": self.args.context_scale_mode + ("_unit_length_rot_pos" if self.args.context_scale_mode == "norm-ratio" else "_unit_length_separate"),
                    "scale_window_length": int(min(unit_len, 50)),
                    "scale_position": float(scales[0]), "scale_rotation": float(scales[1]),
                    "scale_rotation_independent_diagnostic": float(scales[2]),
                    "scale_position_norm_ratio_diagnostic": float(scales[3]),
                    "final_position_norm_ratio": float(
                        np.linalg.norm(action[:min(unit_len, 50), :3]) /
                        max(float(np.linalg.norm(ref_physical[:min(unit_len, 50), :3])), 1e-8)
                    ),
                    "position_cos": float(np.sum(draft[:, :3] * ref_physical[:, :3]) / max(
                        float(np.linalg.norm(draft[:, :3]) * np.linalg.norm(ref_physical[:, :3])), 1e-12)),
                    "rotation_cos": float(np.sum(draft[:, 3:6] * ref_physical[:, 3:6]) / max(
                        float(np.linalg.norm(draft[:, 3:6]) * np.linalg.norm(ref_physical[:, 3:6])), 1e-12)),
                    "type_dot": type_dot, "type_error": type_error,
                    "selector": self.args.search_mode,
                    "context_gripper": self.args.context_gripper,
                }
            elif self.set_reference is not None:
                # The frozen pretrained π0.5 and bridge share this VLM prefix
                # KV cache. Its action chunk sets scale only; it is not mixed
                # into the final output.
                prefix_context = self.prefix_context_fn(observation)
                dim = int(self.base_model.action_dim)
                ref_noise = jax.random.normal(
                    jax.random.key(int(self.args.seed) + request_index + 2),
                    (1, expert_api.HORIZON, dim),
                )
                ref_norm = self.base_reference_fn(
                    jax.random.key(int(self.args.seed) + request_index + 3), observation,
                    prefix_context, num_steps=10, noise=ref_noise,
                )
                mean = np.asarray(self.action_stats.mean, np.float32)
                std = np.asarray(self.action_stats.std, np.float32)
                ref_physical = np.asarray(ref_norm, np.float32)[0] * (std[None, :] + 1e-6) + mean[None, :]
                reference = self.set_reference(ref_physical[:, :6])
                grip_reference = ref_physical[:, 6:7] if getattr(self.args, "replace_gripper", False) else None
            else:
                grip_reference = None

            if self.args.search_mode in ("context", "r2"):
                draft_info = None
            else:
                # The shared helper normally rescales each recalled unit to the
            # preceding executed motion. RM-D1's `saved` mode is the requested
            # stored-magnitude condition, so suppress only that helper step;
            # P5 still receives the true executed history above.
                effective_history = np.zeros_like(history) if self.args.diagnostic_mode == "no-history" else history
                old_scale = expert_api.ex._scale_to_ref
                if self.args.scale_mode == "saved":
                    expert_api.ex._scale_to_ref = lambda actions, _ref: np.asarray(actions, np.float32).copy()
                try:
                    action = expert_api.serve_bridge_action(
                        self.bridge_model,
                        observation,
                        effective_history,
                        state,
                        self.pool,
                        self.p5_models,
                        self.action_stats,
                        t0=float(self.args.t0),
                        seed=int(self.args.seed) + request_index,
                        num_steps=int(self.args.integral_steps),
                        device=self.p5_device,
                        compiled_bridge_fn=self.bridge_fn,
                        prefix_context=prefix_context,
                        scale_reference=reference,
                        gripper_reference=grip_reference,
                        reference_action=ref_physical[:, :12] if ref_physical is not None else None,
                        replace_draft_channels=({"replace-base-mode": tuple(range(7, 12)),
                                                 "replace-base-mode-grip": tuple(range(6, 12))}.get(self.args.diagnostic_mode, ())),
                        return_debug=bool(self.args.debug_log),
                        debug_scale_kind="norm" if self.args.diagnostic_mode == "norm-scale" else "positive-dot",
                        historyless=self.args.diagnostic_mode == "no-history",
                    )
                finally:
                    expert_api.ex._scale_to_ref = old_scale
            draft_info = None
            if self.args.debug_log and self.args.search_mode not in ("context", "r2"):
                action, draft_info = action
                for selection in draft_info["selections"]:
                    for unit in [selection["selected"], *(x["unit"] for x in selection["top3"])]:
                        task_id = unit["task_id"]
                        unit["task"] = self.task_names[task_id] if 0 <= task_id < len(self.task_names) else str(task_id)
            action = np.asarray(action, np.float32)
            if action.shape != (50, 12):
                raise ValueError(f"bridge returned {action.shape}; expected [50,12]")
            hist.set_pending(action[: int(self.args.replan_steps)])
            elapsed = time.perf_counter() - started
            self._log({
                "client_id": client_id,
                "episode_id": episode_id,
                "query_index": request_index,
                "episode_start": episode_start,
                "history_valid_steps": int(mask.sum()),
                "scale_mode": self.args.scale_mode,
                "search_mode": self.args.search_mode,
                "context_output": self.args.context_output if self.args.search_mode in ("context", "r2") else None,
                "scale_reference": "frozen_pretrained_pi05_only_scale_fit" if self.args.scale_mode != "saved" else "stored_action_scale",
                "weights": self.args.weights,
                "replace_gripper": bool(getattr(self.args, "replace_gripper", False)),
                "t0": float(self.args.t0),
                "inference_seconds": elapsed,
            })
            if self.args.search_mode in ("context", "r2") and self.args.debug_log:
                self._log({"record_type": "context_selection", "client_id": client_id,
                           "episode_id": episode_id, "query_index": request_index,
                           "selected": context_debug, "reference_first5": ref_physical[:5, :12].tolist(),
                           "draft_first5": draft[:5].tolist(),
                           "final_first5": action[:5].tolist(), "t0": float(self.args.t0) if self.args.context_output == "pretrained-bridge" else None})
            elif self.args.debug_log:
                self._log({
                    "record_type": "draft_diagnostic", "client_id": client_id,
                    "episode_id": episode_id, "query_index": request_index,
                    "variant": self.args.diagnostic_mode, "history_norm": float(np.linalg.norm(history)),
                    "effective_history_norm": float(np.linalg.norm(effective_history)),
                    "history_eef_norm": float(np.linalg.norm(history[:, :6])),
                    "effective_history_eef_norm": float(np.linalg.norm(effective_history[:, :6])),
                    "draft_selections": draft_info["selections"],
                    "draft_first5_physical_fork_order": draft_info["draft_first5"],
                    "draft_original_first5_physical_fork_order": draft_info["draft_before_overrides_first5"],
                    "scale_position": draft_info["scale_position"], "scale_rotation": draft_info["scale_rotation"],
                    "pi05_reference_first5_physical_fork_order": ref_physical[:5].tolist() if ref_physical is not None else None,
                    "final_first5_physical_fork_order": action[:5].tolist(), "t0": float(self.args.t0),
                })
            return {"actions": action}


def main():
    parser = argparse.ArgumentParser()
    state_root = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get(
        "SKILLNET_ROBOCASA_CHECKPOINT", state_root / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000")))
    parser.add_argument("--memory-dir", type=Path, default=state_root / "outputs/robocasa-memory/memory-01/policy-order-v2")
    parser.add_argument("--expert-dir", type=Path, default=state_root / "outputs/robocasa-memory/expert-01/expert")
    parser.add_argument("--weights", choices=("expert", "pretrained"), default="expert")
    parser.add_argument("--search-mode", choices=("p5", "context", "r2"), default="p5")
    parser.add_argument("--context-dir", type=Path, default=state_root / "outputs/robocasa-memory/ctx-01")
    parser.add_argument("--context-output", choices=("direct", "pretrained-bridge"), default="direct")
    parser.add_argument("--context-gripper", choices=("unit", "context", "pi05"), default="context")
    parser.add_argument("--context-scale-mode", choices=("projection", "norm-ratio"), default="projection",
                        help="fit EEF magnitude on the selected unit's valid interval")
    parser.add_argument("--replace-gripper", action="store_true",
                        help="replace only action channel 6 after bridge with same-observation pretrained pi0.5 gripper")
    parser.add_argument("--scale-mode", choices=("saved", "s-shared", "s-sep", "norm-sep"), default="saved")
    parser.add_argument("--diagnostic-mode", choices=("normal", "replace-base-mode", "replace-base-mode-grip", "norm-scale", "no-history"), default="normal")
    parser.add_argument("--debug-log", action="store_true", help="append per-query draft/reference/output diagnostics to --query-log")
    parser.add_argument("--t0", type=float, default=.4)
    parser.add_argument("--integral-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--query-log", type=Path)
    args = parser.parse_args()
    if args.search_mode in ("context", "r2") and args.weights != "pretrained":
        parser.error("context and r2 modes use the pretrained pi0.5 reference; set --weights pretrained")
    if args.diagnostic_mode == "norm-scale":
        args.scale_mode = "norm-sep"
    if args.diagnostic_mode in ("replace-base-mode", "replace-base-mode-grip") and args.scale_mode == "saved":
        parser.error("draft channel replacement requires an s-sep scale reference")
    if not 0 < args.t0 <= 1:
        parser.error("--t0 must be in (0,1]")
    if args.replace_gripper and args.scale_mode == "saved":
        parser.error("--replace-gripper requires a fitted-scale mode so the shared π0.5 reference is computed")
    policy = MemoryBridgePolicy(args)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"policy": "rm-d2-memory-bridge", "action_horizon": 50,
                  "scale_mode": args.scale_mode, "t0": args.t0,
                  "replace_gripper": args.replace_gripper, "subtask_generation": False},
        ping_interval=None,
        ping_timeout=None,
        infer_in_thread=True,
    ).serve_forever()


if __name__ == "__main__":
    main()
